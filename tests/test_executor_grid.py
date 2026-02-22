"""Tests pour l'Executor Grid DCA (Sprint 12)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.execution.executor import (
    Executor,
    GridLivePosition,
    GridLiveState,
    LivePosition,
    TradeEvent,
    TradeEventType,
    to_futures_symbol,
)
from backend.execution.risk_manager import LiveRiskManager, LiveTradeResult


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_config() -> MagicMock:
    config = MagicMock()
    config.secrets.live_trading = True
    config.secrets.bitget_api_key = "test_key"
    config.secrets.bitget_secret = "test_secret"
    config.secrets.bitget_passphrase = "test_pass"
    config.risk.position.max_concurrent_positions = 3
    config.risk.position.default_leverage = 15
    config.risk.margin.min_free_margin_percent = 20
    config.risk.margin.mode = "cross"
    config.risk.fees.taker_percent = 0.06
    config.risk.fees.maker_percent = 0.02
    config.risk.kill_switch.max_session_loss_percent = 5.0
    config.risk.kill_switch.grid_max_session_loss_percent = 25.0
    config.assets = [
        MagicMock(
            symbol="BTC/USDT", min_order_size=0.001,
            tick_size=0.1, correlation_group=None,
        ),
    ]
    config.correlation_groups = {}
    # Config envelope_dca
    config.strategies.envelope_dca.sl_percent = 20.0
    config.strategies.envelope_dca.leverage = 6
    config.strategies.envelope_dca.live_eligible = True
    return config


def _make_mock_exchange() -> AsyncMock:
    exchange = AsyncMock()
    exchange.load_markets = AsyncMock(return_value={
        "BTC/USDT:USDT": {
            "limits": {"amount": {"min": 0.001}},
            "precision": {"amount": 3, "price": 1},
        },
    })
    exchange.fetch_balance = AsyncMock(return_value={
        "free": {"USDT": 5_000.0},
        "total": {"USDT": 10_000.0},
    })
    exchange.fetch_positions = AsyncMock(return_value=[])
    exchange.set_leverage = AsyncMock()
    exchange.set_margin_mode = AsyncMock()
    exchange.create_order = AsyncMock(return_value={
        "id": "order_grid_1",
        "status": "closed",
        "filled": 0.001,
        "average": 50_000.0,
    })
    exchange.cancel_order = AsyncMock()
    exchange.fetch_order = AsyncMock(return_value={"status": "open"})
    exchange.fetch_my_trades = AsyncMock(return_value=[
        {"price": 50_000.0, "amount": 0.001},
    ])
    exchange.watch_orders = AsyncMock(return_value=[])
    exchange.close = AsyncMock()
    exchange.fetch_open_orders = AsyncMock(return_value=[])
    exchange.amount_to_precision = MagicMock(
        side_effect=lambda sym, qty: f"{int(qty * 1000) / 1000:.3f}",
    )
    exchange.price_to_precision = MagicMock(
        side_effect=lambda sym, price: f"{int(price * 10) / 10:.1f}",
    )
    return exchange


def _make_notifier() -> AsyncMock:
    notifier = AsyncMock()
    return notifier


def _make_selector(allowed: bool = True) -> MagicMock:
    selector = MagicMock()
    selector.is_allowed = MagicMock(return_value=allowed)
    selector.set_active_symbols = MagicMock()
    selector.get_status = MagicMock(return_value={})
    return selector


def _make_executor(
    config=None, risk_manager=None, notifier=None, exchange=None,
    selector=None,
) -> Executor:
    if config is None:
        config = _make_config()
    if risk_manager is None:
        risk_manager = LiveRiskManager(config)
        risk_manager.set_initial_capital(10_000.0)
    if notifier is None:
        notifier = _make_notifier()

    executor = Executor(config, risk_manager, notifier, selector=selector)
    if exchange is None:
        exchange = _make_mock_exchange()
    executor._exchange = exchange
    executor._markets = exchange.load_markets.return_value
    executor._running = True
    executor._connected = True
    return executor


def _make_grid_open_event(
    symbol="BTC/USDT",
    strategy="envelope_dca",
    direction="LONG",
    entry_price=50_000.0,
    quantity=0.001,
) -> TradeEvent:
    return TradeEvent(
        event_type=TradeEventType.OPEN,
        strategy_name=strategy,
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        quantity=quantity,
        tp_price=0.0,
        sl_price=0.0,
        score=0.0,
        timestamp=datetime.now(tz=timezone.utc),
        market_regime="RANGING",
    )


def _make_grid_close_event(
    symbol="BTC/USDT",
    strategy="envelope_dca",
    direction="LONG",
    exit_price=51_000.0,
    exit_reason="tp_global",
) -> TradeEvent:
    return TradeEvent(
        event_type=TradeEventType.CLOSE,
        strategy_name=strategy,
        symbol=symbol,
        direction=direction,
        entry_price=50_000.0,
        quantity=0.002,
        tp_price=0.0,
        sl_price=0.0,
        score=0.0,
        timestamp=datetime.now(tz=timezone.utc),
        market_regime="RANGING",
        exit_reason=exit_reason,
        exit_price=exit_price,
    )


# ─── Ouverture grid ──────────────────────────────────────────────────────


class TestGridOpen:
    """Tests ouverture de niveaux grid."""

    @pytest.mark.asyncio
    async def test_avg_entry_price_correct(self):
        """Vérifie le calcul du prix moyen pondéré."""
        state = GridLiveState(
            symbol="BTC/USDT:USDT", direction="LONG",
            strategy_name="envelope_dca", leverage=6,
            positions=[
                GridLivePosition(
                    level=0, entry_price=50_000.0, quantity=0.001,
                    entry_order_id="e0",
                ),
                GridLivePosition(
                    level=1, entry_price=48_000.0, quantity=0.001,
                    entry_order_id="e1",
                ),
            ],
        )
        # avg = (50000*0.001 + 48000*0.001) / 0.002 = 49000
        assert state.avg_entry_price == pytest.approx(49_000.0)
        assert state.total_quantity == pytest.approx(0.002)


# ─── Surveillance ─────────────────────────────────────────────────────────


class TestGridSurveillance:
    """Tests surveillance watchOrders + polling pour grid."""

    @pytest.mark.asyncio
    async def test_watch_orders_runs_with_grid_only(self):
        """Bug 4 fix : watchOrders tourne même sans positions mono."""
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT", direction="LONG",
            strategy_name="envelope_dca", leverage=6,
            sl_order_id="sl_1",
        )

        # Pas de positions mono
        assert not executor._positions
        assert executor._grid_states

        # La condition devrait passer (pas de sleep)
        # On vérifie que watch_orders est appelé
        watch_called = False
        original_watch = executor._exchange.watch_orders

        async def mock_watch(*a, **kw):
            nonlocal watch_called
            watch_called = True
            executor._running = False  # Stop loop
            return []

        executor._exchange.watch_orders = AsyncMock(side_effect=mock_watch)

        # Lance la loop brièvement
        task = asyncio.create_task(executor._watch_orders_loop())
        await asyncio.sleep(0.1)
        executor._running = False
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

        assert watch_called

    @pytest.mark.asyncio
    async def test_watch_orders_detects_grid_sl(self):
        """watchOrders détecte le SL grid exécuté."""
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT", direction="LONG",
            strategy_name="envelope_dca", leverage=6,
            positions=[GridLivePosition(
                level=0, entry_price=50_000.0, quantity=0.001,
                entry_order_id="e0",
            )],
            sl_order_id="sl_grid_1",
            sl_price=40_000.0,
        )
        executor._risk_manager.register_position({
            "symbol": "BTC/USDT:USDT", "direction": "LONG",
            "entry_price": 50_000.0, "quantity": 0.001,
        })

        order = {"id": "sl_grid_1", "status": "closed", "average": 40_000.0}
        await executor._process_watched_order(order)

        # Grid nettoyé
        assert "BTC/USDT:USDT" not in executor._grid_states
        executor._notifier.notify_grid_cycle_closed.assert_called_once()

    @pytest.mark.asyncio
    async def test_polling_detects_grid_closed(self):
        """Bug 5 fix : polling détecte grid fermée côté exchange."""
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT", direction="LONG",
            strategy_name="envelope_dca", leverage=6,
            positions=[GridLivePosition(
                level=0, entry_price=50_000.0, quantity=0.001,
                entry_order_id="e0",
            )],
            sl_order_id="sl_1",
            sl_price=40_000.0,
        )
        executor._risk_manager.register_position({
            "symbol": "BTC/USDT:USDT", "direction": "LONG",
            "entry_price": 50_000.0, "quantity": 0.001,
        })

        # fetch_positions retourne vide (position fermée)
        executor._exchange.fetch_positions = AsyncMock(return_value=[])

        await executor._check_grid_still_open("BTC/USDT:USDT")

        assert "BTC/USDT:USDT" not in executor._grid_states

    @pytest.mark.asyncio
    async def test_orphan_orders_includes_grid_sl(self):
        """Bug 6 fix : cancel_orphan_orders n'annule pas le SL grid."""
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT", direction="LONG",
            strategy_name="envelope_dca", leverage=6,
            positions=[GridLivePosition(
                level=0, entry_price=50_000.0, quantity=0.001,
                entry_order_id="grid_entry_1",
            )],
            sl_order_id="grid_sl_1",
        )

        # Exchange a un ordre ouvert qui est le SL grid
        executor._exchange.fetch_open_orders = AsyncMock(return_value=[
            {"id": "grid_sl_1", "symbol": "BTC/USDT:USDT"},
        ])

        await executor._cancel_orphan_orders()

        # Le SL grid NE doit PAS être annulé
        executor._exchange.cancel_order.assert_not_called()


# ─── State persistence ────────────────────────────────────────────────────


class TestGridState:
    """Tests sérialisation/désérialisation grid states."""

    def test_get_state_includes_grid(self):
        """get_state_for_persistence inclut grid_states."""
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT", direction="LONG",
            strategy_name="envelope_dca", leverage=6,
            positions=[GridLivePosition(
                level=0, entry_price=50_000.0, quantity=0.001,
                entry_order_id="e0",
            )],
            sl_order_id="sl_1",
            sl_price=40_000.0,
        )

        state = executor.get_state_for_persistence()
        assert "grid_states" in state
        gs = state["grid_states"]["BTC/USDT:USDT"]
        assert gs["direction"] == "LONG"
        assert gs["leverage"] == 6
        assert len(gs["positions"]) == 1
        assert gs["positions"][0]["entry_price"] == 50_000.0

    def test_restore_grid_states(self):
        """restore_positions restaure grid_states."""
        executor = _make_executor()
        state = {
            "positions": {},
            "grid_states": {
                "BTC/USDT:USDT": {
                    "symbol": "BTC/USDT:USDT",
                    "direction": "LONG",
                    "strategy_name": "envelope_dca",
                    "leverage": 6,
                    "sl_order_id": "sl_1",
                    "sl_price": 40_000.0,
                    "opened_at": "2025-01-01T00:00:00+00:00",
                    "positions": [
                        {
                            "level": 0,
                            "entry_price": 50_000.0,
                            "quantity": 0.001,
                            "entry_order_id": "e0",
                            "entry_time": "2025-01-01T00:00:00+00:00",
                        },
                    ],
                },
            },
        }

        executor.restore_positions(state)

        assert "BTC/USDT:USDT" in executor._grid_states
        gs = executor._grid_states["BTC/USDT:USDT"]
        assert gs.direction == "LONG"
        assert gs.leverage == 6
        assert len(gs.positions) == 1

    def test_state_round_trip(self):
        """Sérialisation → désérialisation = identique."""
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT", direction="LONG",
            strategy_name="envelope_dca", leverage=6,
            positions=[
                GridLivePosition(
                    level=0, entry_price=50_000.0, quantity=0.001,
                    entry_order_id="e0",
                ),
                GridLivePosition(
                    level=1, entry_price=48_000.0, quantity=0.001,
                    entry_order_id="e1",
                ),
            ],
            sl_order_id="sl_1",
            sl_price=39_200.0,
        )

        # Sérialiser
        saved = executor.get_state_for_persistence()

        # Restaurer dans un nouvel executor
        executor2 = _make_executor()
        executor2.restore_positions(saved)

        gs = executor2._grid_states["BTC/USDT:USDT"]
        assert gs.direction == "LONG"
        assert gs.leverage == 6
        assert len(gs.positions) == 2
        assert gs.avg_entry_price == pytest.approx(49_000.0)
        assert gs.total_quantity == pytest.approx(0.002)
        assert gs.sl_order_id == "sl_1"

    def test_get_status_includes_grid(self):
        """get_status inclut les grid positions avec type et levels."""
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT", direction="LONG",
            strategy_name="envelope_dca", leverage=6,
            positions=[
                GridLivePosition(
                    level=0, entry_price=50_000.0, quantity=0.001,
                    entry_order_id="e0",
                ),
            ],
            sl_order_id="sl_1",
            sl_price=40_000.0,
        )

        status = executor.get_status()
        assert len(status["positions"]) == 1
        pos = status["positions"][0]
        assert pos["type"] == "grid"
        assert pos["levels"] == 1
        assert pos["strategy_name"] == "envelope_dca"

    @pytest.mark.asyncio
    async def test_reconcile_grid_position_still_open(self):
        """Réconciliation : grid + position exchange → reprise."""
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT", direction="LONG",
            strategy_name="envelope_dca", leverage=6,
            positions=[GridLivePosition(
                level=0, entry_price=50_000.0, quantity=0.001,
                entry_order_id="e0",
            )],
            sl_order_id="sl_1",
            sl_price=40_000.0,
        )

        # Position toujours ouverte sur exchange
        executor._exchange.fetch_positions = AsyncMock(return_value=[
            {"contracts": 0.001, "symbol": "BTC/USDT:USDT"},
        ])
        # SL toujours actif
        executor._exchange.fetch_order = AsyncMock(
            return_value={"status": "open"},
        )

        await executor._reconcile_grid_symbol("BTC/USDT:USDT")

        # Grid state conservé
        assert "BTC/USDT:USDT" in executor._grid_states


# ─── Helpers grid ─────────────────────────────────────────────────────────


class TestGridHelpers:
    """Tests pour les helpers grid."""

    def test_is_grid_strategy(self):
        assert Executor._is_grid_strategy("envelope_dca") is True
        assert Executor._is_grid_strategy("vwap_rsi") is False
        assert Executor._is_grid_strategy("momentum") is False

    def test_get_grid_sl_percent(self):
        executor = _make_executor()
        assert executor._get_grid_sl_percent("envelope_dca") == 20.0

    def test_get_grid_leverage(self):
        executor = _make_executor()
        assert executor._get_grid_leverage("envelope_dca") == 6
