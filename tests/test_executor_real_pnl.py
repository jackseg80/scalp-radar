"""Tests Hotfix 34 — P&L Executor basé sur fills réels Bitget.

Vérifie que :
- _fetch_fill_price récupère le vrai prix via fetch_order / fetch_my_trades
- _calculate_real_pnl utilise les fees absolues (pas estimées)
- Les entries/exits utilisent les prix réels quand average=None
- La persistence round-trip de entry_fee fonctionne
- Les watched orders extraient les fees pour les handlers
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

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
from backend.execution.risk_manager import LiveRiskManager


# ─── Helpers (réutilisés depuis test_executor_grid.py) ────────────────────


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
        "id": "order_1", "status": "closed",
        "filled": 0.001, "average": 50_000.0,
    })
    exchange.cancel_order = AsyncMock()
    exchange.fetch_order = AsyncMock(return_value={"status": "open"})
    exchange.fetch_my_trades = AsyncMock(return_value=[])
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


def _make_executor(config=None, exchange=None) -> Executor:
    if config is None:
        config = _make_config()
    risk_manager = LiveRiskManager(config)
    risk_manager.set_initial_capital(10_000.0)
    notifier = AsyncMock()

    executor = Executor(config, risk_manager, notifier)
    if exchange is None:
        exchange = _make_mock_exchange()
    executor._exchange = exchange
    executor._markets = exchange.load_markets.return_value
    executor._running = True
    executor._connected = True
    return executor


def _make_grid_open_event(**kwargs) -> TradeEvent:
    defaults = dict(
        event_type=TradeEventType.OPEN,
        strategy_name="envelope_dca",
        symbol="BTC/USDT",
        direction="LONG",
        entry_price=50_000.0,
        quantity=0.001,
        tp_price=0.0,
        sl_price=0.0,
        score=0.0,
        timestamp=datetime.now(tz=timezone.utc),
        market_regime="RANGING",
    )
    defaults.update(kwargs)
    return TradeEvent(**defaults)


def _make_grid_close_event(**kwargs) -> TradeEvent:
    defaults = dict(
        event_type=TradeEventType.CLOSE,
        strategy_name="envelope_dca",
        symbol="BTC/USDT",
        direction="LONG",
        entry_price=50_000.0,
        quantity=0.002,
        tp_price=0.0,
        sl_price=0.0,
        score=0.0,
        timestamp=datetime.now(tz=timezone.utc),
        market_regime="RANGING",
        exit_reason="tp_global",
        exit_price=51_000.0,
    )
    defaults.update(kwargs)
    return TradeEvent(**defaults)


def _setup_grid_state(executor: Executor, entry_fee: float = 0.05) -> None:
    """Pré-remplit un grid state avec 1 position ouverte."""
    executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
        symbol="BTC/USDT:USDT",
        direction="LONG",
        strategy_name="envelope_dca",
        leverage=6,
        positions=[GridLivePosition(
            level=0, entry_price=50_000.0, quantity=0.001,
            entry_order_id="entry_0", entry_fee=entry_fee,
        )],
        sl_order_id="sl_1",
        sl_price=40_000.0,
    )
    executor._risk_manager.register_position({
        "symbol": "BTC/USDT:USDT", "direction": "LONG",
        "entry_price": 50_000.0, "quantity": 0.001,
    })


# ─── TestFetchFillPrice ──────────────────────────────────────────────────


class TestFetchFillPrice:
    """Tests pour _fetch_fill_price."""

    @pytest.mark.asyncio
    async def test_from_fetch_order(self):
        """fetch_order retourne average + fee → tuple correct."""
        executor = _make_executor()
        executor._exchange.fetch_order = AsyncMock(return_value={
            "id": "ord_1", "average": 51_200.0, "status": "closed",
            "fee": {"cost": 0.12, "currency": "USDT"},
        })

        price, fee = await executor._fetch_fill_price(
            "ord_1", "BTC/USDT:USDT", 51_000.0,
        )
        assert price == 51_200.0
        assert fee == 0.12
        executor._exchange.fetch_order.assert_called_once_with("ord_1", "BTC/USDT:USDT")

    @pytest.mark.asyncio
    async def test_fallback_fetch_my_trades(self):
        """fetch_order échoue → fetch_my_trades → prix moyen pondéré + somme fees."""
        executor = _make_executor()
        executor._exchange.fetch_order = AsyncMock(side_effect=Exception("network"))
        executor._exchange.fetch_my_trades = AsyncMock(return_value=[
            {"order": "ord_1", "price": 51_000.0, "amount": 0.0005,
             "fee": {"cost": 0.06}},
            {"order": "ord_1", "price": 51_100.0, "amount": 0.0005,
             "fee": {"cost": 0.07}},
            {"order": "other", "price": 49_000.0, "amount": 0.001,
             "fee": {"cost": 0.10}},
        ])

        price, fee = await executor._fetch_fill_price(
            "ord_1", "BTC/USDT:USDT", 50_000.0,
        )
        # Prix moyen : (51000*0.0005 + 51100*0.0005) / 0.001 = 51050
        assert price == pytest.approx(51_050.0)
        # Somme fees des trades matchés seulement
        assert fee == pytest.approx(0.13)

    @pytest.mark.asyncio
    async def test_fallback_returns_none_fee(self):
        """Tout échoue → (fallback_price, None)."""
        executor = _make_executor()
        executor._exchange.fetch_order = AsyncMock(side_effect=Exception("err"))
        executor._exchange.fetch_my_trades = AsyncMock(side_effect=Exception("err"))

        price, fee = await executor._fetch_fill_price(
            "ord_1", "BTC/USDT:USDT", 50_000.0,
        )
        assert price == 50_000.0
        assert fee is None


# ─── TestCalculateRealPnl ────────────────────────────────────────────────


class TestCalculateRealPnl:
    """Tests pour _calculate_real_pnl."""

    def test_long(self):
        executor = _make_executor()
        pnl = executor._calculate_real_pnl("LONG", 100.0, 102.0, 10, 0.06, 0.07)
        # gross = (102 - 100) * 10 = 20, fees = 0.13, net = 19.87
        assert pnl == pytest.approx(19.87)

    def test_short(self):
        executor = _make_executor()
        pnl = executor._calculate_real_pnl("SHORT", 102.0, 100.0, 10, 0.06, 0.07)
        assert pnl == pytest.approx(19.87)

    def test_zero_fees(self):
        """Fees à 0.0 (VIP Bitget) → P&L = gross."""
        executor = _make_executor()
        pnl = executor._calculate_real_pnl("LONG", 100.0, 102.0, 10, 0.0, 0.0)
        assert pnl == pytest.approx(20.0)


# ─── TestEntryFillPrice ──────────────────────────────────────────────────


class TestEntryFillPrice:
    """Tests pour l'entry avec prix réel."""

    @pytest.mark.asyncio
    async def test_open_grid_average_none(self):
        """create_order average=None → fetch_fill_price appelé, entry_fee peuplé."""
        executor = _make_executor()

        call_count = 0

        async def create_order_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # entry
                return {
                    "id": "entry_1", "status": "closed",
                    "filled": 0.001, "average": None, "fee": None,
                }
            return {"id": "sl_1", "filled": 0.001}  # SL

        executor._exchange.create_order = AsyncMock(
            side_effect=create_order_side_effect,
        )
        executor._exchange.fetch_order = AsyncMock(return_value={
            "average": 49_800.0,
            "fee": {"cost": 0.05, "currency": "USDT"},
        })

        event = _make_grid_open_event(entry_price=50_000.0)
        await executor.handle_event(event)

        state = executor._grid_states.get("BTC/USDT:USDT")
        assert state is not None
        # Entry price doit être 49800 (réel), pas 50000 (paper)
        assert state.positions[0].entry_price == 49_800.0
        assert state.positions[0].entry_fee == 0.05

    @pytest.mark.asyncio
    async def test_open_grid_average_present(self):
        """create_order average=50000 → pas de fetch, entry_fee depuis order."""
        executor = _make_executor()

        call_count = 0

        async def create_order_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # entry
                return {
                    "id": "entry_1", "status": "closed",
                    "filled": 0.001, "average": 50_100.0,
                    "fee": {"cost": 0.03, "currency": "USDT"},
                }
            return {"id": "sl_1", "filled": 0.001}  # SL

        executor._exchange.create_order = AsyncMock(
            side_effect=create_order_side_effect,
        )

        event = _make_grid_open_event(entry_price=50_000.0)
        await executor.handle_event(event)

        state = executor._grid_states.get("BTC/USDT:USDT")
        assert state is not None
        assert state.positions[0].entry_price == 50_100.0
        assert state.positions[0].entry_fee == 0.03
        # fetch_order NE doit PAS être appelé (average était présent)
        executor._exchange.fetch_order.assert_not_called()


# ─── TestExitFillPrice ───────────────────────────────────────────────────


class TestExitFillPrice:
    """Tests pour l'exit avec prix réel et P&L réel."""

    @pytest.mark.asyncio
    async def test_close_grid_real_pnl(self):
        """Close avec fees réelles → _calculate_real_pnl utilisé."""
        executor = _make_executor()
        _setup_grid_state(executor, entry_fee=0.05)

        # close_order retourne average + fee
        executor._exchange.create_order = AsyncMock(return_value={
            "id": "close_1", "status": "closed",
            "filled": 0.001, "average": 51_000.0,
            "fee": {"cost": 0.06, "currency": "USDT"},
        })

        event = _make_grid_close_event(exit_price=52_000.0)
        await executor.handle_event(event)

        # Position fermée
        assert "BTC/USDT:USDT" not in executor._grid_states

        # P&L calculé avec fees réelles (0.05 entry + 0.06 exit)
        # gross = (51000 - 50000) * 0.001 = 1.0
        # net = 1.0 - 0.05 - 0.06 = 0.89
        last_result = executor._risk_manager._trade_history[-1]
        assert last_result.net_pnl == pytest.approx(0.89)

    @pytest.mark.asyncio
    async def test_close_grid_none_fee_fallback(self):
        """exit_fee=None → fallback sur _calculate_pnl estimé."""
        executor = _make_executor()
        _setup_grid_state(executor, entry_fee=0.0)  # pas de fee entry

        # close_order average=None, fetch échoue aussi
        executor._exchange.create_order = AsyncMock(return_value={
            "id": "close_1", "status": "closed",
            "filled": 0.001, "average": None, "fee": None,
        })
        executor._exchange.fetch_order = AsyncMock(side_effect=Exception("err"))
        executor._exchange.fetch_my_trades = AsyncMock(side_effect=Exception("err"))

        event = _make_grid_close_event(exit_price=51_000.0)
        await executor.handle_event(event)

        # Position fermée malgré l'échec de fetch
        assert "BTC/USDT:USDT" not in executor._grid_states

        # P&L estimé (fallback _calculate_pnl car fee=None)
        last_result = executor._risk_manager._trade_history[-1]
        # gross = (51000 - 50000) * 0.001 = 1.0
        # fees estimées = 0.001 * 50000 * 0.0006 + 0.001 * 51000 * 0.0006 = 0.03 + 0.0306 = 0.0606
        assert last_result.net_pnl == pytest.approx(1.0 - 0.0606, abs=0.01)


# ─── TestWatchedOrderFees ────────────────────────────────────────────────


class TestWatchedOrderFees:
    """Tests pour l'extraction des fees dans _process_watched_order."""

    @pytest.mark.asyncio
    async def test_watched_sl_fetches_fees(self):
        """SL watched avec fee=null → _fetch_fill_price appelé."""
        executor = _make_executor()
        _setup_grid_state(executor, entry_fee=0.05)

        # fetch_order retourne le fill réel du SL
        executor._exchange.fetch_order = AsyncMock(return_value={
            "average": 40_500.0,
            "fee": {"cost": 0.08, "currency": "USDT"},
        })

        # Simuler un ordre watched (SL exécuté, fee absent du WS)
        watched_order = {
            "id": "sl_1",
            "status": "closed",
            "average": 40_500.0,
            "price": 40_000.0,
            "fee": None,  # Bitget ne retourne pas la fee dans le WS push
        }
        await executor._process_watched_order(watched_order)

        # Position doit être fermée
        assert "BTC/USDT:USDT" not in executor._grid_states

        # _fetch_fill_price doit avoir été appelé (car fee=None)
        executor._exchange.fetch_order.assert_called()

        # P&L avec fees réelles
        last_result = executor._risk_manager._trade_history[-1]
        # gross = (40500 - 50000) * 0.001 = -9.5
        # net = -9.5 - 0.05 - 0.08 = -9.63
        assert last_result.net_pnl == pytest.approx(-9.63)


# ─── TestPersistence ─────────────────────────────────────────────────────


class TestPersistence:
    """Tests persistence entry_fee."""

    def test_entry_fee_round_trip(self):
        """save → restore → entry_fee préservée."""
        executor = _make_executor()

        # Positions avec entry_fee
        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT",
            direction="LONG",
            strategy_name="envelope_dca",
            leverage=6,
            positions=[
                GridLivePosition(
                    level=0, entry_price=50_000.0, quantity=0.001,
                    entry_order_id="e0", entry_fee=0.05,
                ),
                GridLivePosition(
                    level=1, entry_price=49_000.0, quantity=0.001,
                    entry_order_id="e1", entry_fee=0.06,
                ),
            ],
            sl_order_id="sl_1",
        )

        # Save
        state = executor.get_state_for_persistence()

        # Restore dans un nouvel executor
        executor2 = _make_executor()
        executor2.restore_positions(state)

        restored = executor2._grid_states["BTC/USDT:USDT"]
        assert restored.positions[0].entry_fee == 0.05
        assert restored.positions[1].entry_fee == 0.06
        assert restored.total_entry_fees == pytest.approx(0.11)

    def test_backward_compat(self):
        """State JSON sans entry_fee → restaure avec 0.0."""
        executor = _make_executor()

        # Ancien format sans entry_fee
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
                    "opened_at": "2026-02-18T10:00:00+00:00",
                    "positions": [{
                        "level": 0,
                        "entry_price": 50_000.0,
                        "quantity": 0.001,
                        "entry_order_id": "e0",
                        "entry_time": "2026-02-18T10:00:00+00:00",
                        # PAS de entry_fee ici
                    }],
                },
            },
        }
        executor.restore_positions(state)

        restored = executor._grid_states["BTC/USDT:USDT"]
        assert restored.positions[0].entry_fee == 0.0

    def test_live_position_entry_fee_round_trip(self):
        """LivePosition entry_fee save → restore."""
        executor = _make_executor()

        executor._positions["BTC/USDT:USDT"] = LivePosition(
            symbol="BTC/USDT:USDT",
            direction="LONG",
            entry_price=50_000.0,
            quantity=0.001,
            entry_order_id="e0",
            strategy_name="momentum",
            entry_fee=0.12,
        )

        state = executor.get_state_for_persistence()
        executor2 = _make_executor()
        executor2.restore_positions(state)

        restored = executor2._positions["BTC/USDT:USDT"]
        assert restored.entry_fee == 0.12


# ─── TestDataclasses ─────────────────────────────────────────────────────


class TestDataclasses:
    """Tests pour les nouvelles properties."""

    def test_grid_state_total_entry_fees(self):
        """total_entry_fees somme les fees de tous les niveaux."""
        state = GridLiveState(
            symbol="BTC/USDT:USDT",
            direction="LONG",
            strategy_name="grid_atr",
            leverage=6,
        )
        state.positions = [
            GridLivePosition(
                level=0, entry_price=100, quantity=1,
                entry_order_id="e0", entry_fee=0.05,
            ),
            GridLivePosition(
                level=1, entry_price=99, quantity=1,
                entry_order_id="e1", entry_fee=0.06,
            ),
            GridLivePosition(
                level=2, entry_price=98, quantity=1,
                entry_order_id="e2", entry_fee=0.07,
            ),
        ]
        assert state.total_entry_fees == pytest.approx(0.18)

    def test_grid_live_position_entry_fee_default(self):
        """entry_fee a un défaut à 0.0."""
        pos = GridLivePosition(
            level=0, entry_price=100, quantity=1, entry_order_id="e1",
        )
        assert pos.entry_fee == 0.0

    def test_live_position_entry_fee_default(self):
        """LivePosition.entry_fee a un défaut à 0.0."""
        pos = LivePosition(
            symbol="BTC/USDT:USDT", direction="LONG",
            entry_price=100, quantity=1, entry_order_id="e1",
        )
        assert pos.entry_fee == 0.0
