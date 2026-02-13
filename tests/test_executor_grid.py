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
    config.secrets.bitget_sandbox = False
    config.risk.position.max_concurrent_positions = 3
    config.risk.position.default_leverage = 15
    config.risk.margin.min_free_margin_percent = 20
    config.risk.margin.mode = "cross"
    config.risk.fees.taker_percent = 0.06
    config.risk.fees.maker_percent = 0.02
    config.risk.kill_switch.max_session_loss_percent = 5.0
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
    async def test_first_level_opens_with_pre_trade_check(self):
        """1er niveau : pre_trade_check + leverage + market + SL + register."""
        executor = _make_executor()

        # SL order distinct de l'entry
        call_count = 0
        async def create_order_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # entry
                return {"id": "entry_1", "filled": 0.001, "average": 50_000.0}
            return {"id": "sl_1", "filled": 0.001}  # SL

        executor._exchange.create_order = AsyncMock(side_effect=create_order_side_effect)

        event = _make_grid_open_event()
        await executor.handle_event(event)

        # Leverage setup
        executor._exchange.set_leverage.assert_called_with(
            6, "BTC/USDT:USDT", params={},
        )
        # Position enregistrée
        assert "BTC/USDT:USDT" in executor._grid_states
        state = executor._grid_states["BTC/USDT:USDT"]
        assert len(state.positions) == 1
        assert state.sl_order_id == "sl_1"
        assert state.direction == "LONG"
        assert state.leverage == 6
        # RiskManager registré
        assert executor._risk_manager.open_positions_count == 1
        # Telegram notifié
        executor._notifier.notify_grid_level_opened.assert_called_once()

    @pytest.mark.asyncio
    async def test_second_level_skips_pre_trade_check(self):
        """2ème niveau : pas de pre_trade_check, SL recalculé."""
        executor = _make_executor()

        # Pré-remplir un grid state
        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT",
            direction="LONG",
            strategy_name="envelope_dca",
            leverage=6,
            positions=[GridLivePosition(
                level=0, entry_price=50_000.0, quantity=0.001,
                entry_order_id="entry_0",
            )],
            sl_order_id="old_sl",
            sl_price=40_000.0,
        )
        # Registrer dans RiskManager aussi
        executor._risk_manager.register_position({
            "symbol": "BTC/USDT:USDT", "direction": "LONG",
            "entry_price": 50_000.0, "quantity": 0.001,
        })

        call_count = 0
        async def create_order_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"id": "entry_1", "filled": 0.001, "average": 48_000.0}
            return {"id": "new_sl", "filled": 0.002}

        executor._exchange.create_order = AsyncMock(side_effect=create_order_side_effect)

        event = _make_grid_open_event(entry_price=48_000.0)
        await executor.handle_event(event)

        state = executor._grid_states["BTC/USDT:USDT"]
        assert len(state.positions) == 2
        # Ancien SL annulé
        executor._exchange.cancel_order.assert_called_with(
            "old_sl", "BTC/USDT:USDT", params={},
        )
        # Nouveau SL placé
        assert state.sl_order_id == "new_sl"
        # RiskManager NON ré-enregistré (toujours 1 position)
        assert executor._risk_manager.open_positions_count == 1
        # Pas de fetch_balance (pas de pre_trade_check)
        executor._exchange.fetch_balance.assert_not_called()

    @pytest.mark.asyncio
    async def test_reject_grid_if_mono_active(self):
        """Exclusion mutuelle : reject grid si mono active sur même symbol."""
        executor = _make_executor()
        executor._positions["BTC/USDT:USDT"] = LivePosition(
            symbol="BTC/USDT:USDT", direction="LONG",
            entry_price=50_000.0, quantity=0.001,
            entry_order_id="mono_1",
        )

        event = _make_grid_open_event()
        await executor.handle_event(event)

        assert "BTC/USDT:USDT" not in executor._grid_states

    @pytest.mark.asyncio
    async def test_reject_mono_if_grid_active(self):
        """Exclusion mutuelle : reject mono si grid active sur même symbol."""
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT", direction="LONG",
            strategy_name="envelope_dca", leverage=6,
        )

        # Événement mono
        event = TradeEvent(
            event_type=TradeEventType.OPEN,
            strategy_name="vwap_rsi",
            symbol="BTC/USDT",
            direction="LONG",
            entry_price=50_000.0, quantity=0.001,
            tp_price=50_800.0, sl_price=49_700.0,
            score=0.75,
            timestamp=datetime.now(tz=timezone.utc),
        )
        await executor.handle_event(event)

        # Pas de position mono créée
        assert "BTC/USDT:USDT" not in executor._positions

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

    @pytest.mark.asyncio
    async def test_emergency_close_if_sl_fails(self):
        """Règle #1 : close urgence si SL impossible après retries."""
        executor = _make_executor()

        call_count = 0
        async def create_order_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # entry OK
                return {"id": "entry_1", "filled": 0.001, "average": 50_000.0}
            # SL et emergency close
            if call_count <= 4:  # 3 retries SL
                raise Exception("SL placement failed")
            return {"id": "emergency_close"}  # emergency close

        executor._exchange.create_order = AsyncMock(side_effect=create_order_side_effect)

        event = _make_grid_open_event()
        await executor.handle_event(event)

        # Grid state nettoyé (emergency close)
        assert "BTC/USDT:USDT" not in executor._grid_states
        executor._notifier.notify_live_sl_failed.assert_called_once()

    @pytest.mark.asyncio
    async def test_selector_blocks_grid(self):
        """AdaptiveSelector bloque les trades grid non autorisés."""
        selector = _make_selector(allowed=False)
        executor = _make_executor(selector=selector)

        event = _make_grid_open_event()
        await executor.handle_event(event)

        assert "BTC/USDT:USDT" not in executor._grid_states


# ─── Fermeture grid ──────────────────────────────────────────────────────


class TestGridClose:
    """Tests fermeture de cycles grid."""

    def _setup_grid_state(self, executor: Executor) -> None:
        """Crée un grid state avec 2 niveaux."""
        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT",
            direction="LONG",
            strategy_name="envelope_dca",
            leverage=6,
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
            sl_order_id="sl_grid",
            sl_price=39_200.0,
        )
        executor._risk_manager.register_position({
            "symbol": "BTC/USDT:USDT", "direction": "LONG",
            "entry_price": 49_000.0, "quantity": 0.002,
        })

    @pytest.mark.asyncio
    async def test_close_tp_global(self):
        """TP global : cancel SL + market close + P&L + cleanup."""
        executor = _make_executor()
        self._setup_grid_state(executor)

        executor._exchange.create_order = AsyncMock(return_value={
            "id": "close_1", "average": 51_000.0, "filled": 0.002,
        })

        event = _make_grid_close_event(exit_price=51_000.0, exit_reason="tp_global")
        await executor.handle_event(event)

        # SL annulé
        executor._exchange.cancel_order.assert_called_once()
        # Market close
        executor._exchange.create_order.assert_called_once()
        # Grid nettoyé
        assert "BTC/USDT:USDT" not in executor._grid_states
        # RiskManager
        assert executor._risk_manager.open_positions_count == 0
        # Telegram
        executor._notifier.notify_grid_cycle_closed.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_sl_global_no_market_order(self):
        """SL global : pas de market close (déjà exécuté par Bitget)."""
        executor = _make_executor()
        self._setup_grid_state(executor)

        event = _make_grid_close_event(
            exit_price=39_200.0, exit_reason="sl_global",
        )
        await executor.handle_event(event)

        # Pas de cancel_order (SL déjà exécuté)
        executor._exchange.cancel_order.assert_not_called()
        # Pas de create_order (Bitget a déjà fermé)
        executor._exchange.create_order.assert_not_called()
        # Grid nettoyé
        assert "BTC/USDT:USDT" not in executor._grid_states

    @pytest.mark.asyncio
    async def test_close_single_level(self):
        """Fermeture avec 1 seul niveau."""
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

        executor._exchange.create_order = AsyncMock(return_value={
            "id": "close_1", "average": 51_000.0, "filled": 0.001,
        })

        event = _make_grid_close_event(exit_reason="tp_global")
        await executor.handle_event(event)

        assert "BTC/USDT:USDT" not in executor._grid_states

    @pytest.mark.asyncio
    async def test_close_returns_silently_if_no_state(self):
        """Close sur un symbol sans grid state → return silencieux."""
        executor = _make_executor()
        event = _make_grid_close_event()
        await executor.handle_event(event)
        # Pas d'erreur, pas de call
        executor._exchange.create_order.assert_not_called()


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
