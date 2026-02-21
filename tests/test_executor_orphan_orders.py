"""Tests pour le nettoyage des ordres trigger orphelins (Hotfix Orphan Orders).

4 scénarios :
1. Fermeture grid annule TOUS les ordres ouverts, pas juste sl_order_id
2. Au boot, les ordres trigger sur des symbols sans grid active sont annulés
3. Quand un nouveau SL est placé, l'ancien est correctement annulé (avec log)
4. Si cancel d'un SL spécifique échoue, fallback sur cancel_all
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from backend.execution.executor import (
    Executor,
    GridLivePosition,
    GridLiveState,
    TradeEvent,
    TradeEventType,
    to_futures_symbol,
)
from backend.execution.risk_manager import LiveRiskManager


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
    config.strategies.grid_atr.sl_percent = 20.0
    config.strategies.grid_atr.leverage = 6
    config.strategies.grid_atr.live_eligible = True
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
        "id": "order_1", "status": "closed", "filled": 0.001, "average": 50_000.0,
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


def _make_grid_state(
    symbol: str = "BTC/USDT:USDT",
    direction: str = "LONG",
    strategy: str = "grid_atr",
    sl_order_id: str | None = "sl_old_1",
) -> GridLiveState:
    return GridLiveState(
        symbol=symbol.split(":")[0] if ":" in symbol else symbol,
        direction=direction,
        strategy_name=strategy,
        leverage=6,
        sl_order_id=sl_order_id,
        sl_price=45_000.0,
        opened_at=datetime.now(tz=timezone.utc),
        positions=[
            GridLivePosition(
                level=0,
                entry_price=50_000.0,
                quantity=0.001,
                entry_order_id="entry_1",
                entry_time=datetime.now(tz=timezone.utc),
            ),
        ],
    )


def _make_close_event(
    symbol: str = "BTC/USDT",
    exit_reason: str = "tp_global",
    exit_price: float = 51_000.0,
) -> TradeEvent:
    return TradeEvent(
        event_type=TradeEventType.CLOSE,
        strategy_name="grid_atr",
        symbol=symbol,
        direction="LONG",
        entry_price=50_000.0,
        quantity=0.001,
        tp_price=0.0,
        sl_price=0.0,
        score=0.0,
        timestamp=datetime.now(tz=timezone.utc),
        market_regime="RANGING",
        exit_reason=exit_reason,
        exit_price=exit_price,
    )


# ─── Tests ─────────────────────────────────────────────────────────────────


class TestCloseGridCancelsAllOrders:
    """Fermeture grid annule TOUS les ordres ouverts, pas juste sl_order_id."""

    @pytest.mark.asyncio
    async def test_close_grid_cancels_all_open_orders(self):
        """Lors d'un tp_global, fetch_open_orders + cancel de chaque ordre."""
        executor = _make_executor()
        futures_sym = "BTC/USDT:USDT"

        # Installer une grid active avec un SL tracké
        state = _make_grid_state()
        executor._grid_states[futures_sym] = state
        executor._risk_manager.register_position({
            "symbol": futures_sym, "direction": "LONG",
            "quantity": 0.001, "entry_price": 50_000.0, "sl_price": 45_000.0,
        })

        # Simuler 3 ordres ouverts sur Bitget (SL courant + 2 orphelins)
        executor._exchange.fetch_open_orders = AsyncMock(return_value=[
            {"id": "sl_old_1", "symbol": futures_sym},
            {"id": "sl_orphan_2", "symbol": futures_sym},
            {"id": "sl_orphan_3", "symbol": futures_sym},
        ])

        # Close order retourne un fill
        executor._exchange.create_order = AsyncMock(return_value={
            "id": "close_1", "filled": 0.001, "average": 51_000.0,
            "fee": {"cost": 0.03},
        })

        event = _make_close_event(exit_reason="tp_global")
        await executor._close_grid_cycle(event)

        # Vérifier que fetch_open_orders a été appelé
        executor._exchange.fetch_open_orders.assert_called_once_with(
            futures_sym, params={"type": "swap"},
        )

        # Vérifier que les 3 ordres ont été annulés
        cancel_calls = executor._exchange.cancel_order.call_args_list
        cancelled_ids = {c.args[0] for c in cancel_calls}
        assert cancelled_ids == {"sl_old_1", "sl_orphan_2", "sl_orphan_3"}

    @pytest.mark.asyncio
    async def test_sl_global_does_not_cancel(self):
        """Quand c'est le SL qui déclenche, pas de cancel_all (déjà exécuté)."""
        executor = _make_executor()
        futures_sym = "BTC/USDT:USDT"

        state = _make_grid_state()
        executor._grid_states[futures_sym] = state
        executor._risk_manager.register_position({
            "symbol": futures_sym, "direction": "LONG",
            "quantity": 0.001, "entry_price": 50_000.0, "sl_price": 45_000.0,
        })

        # SL exécuté → _close_grid_cycle avec sl_global
        event = _make_close_event(exit_reason="sl_global", exit_price=45_000.0)
        await executor._close_grid_cycle(event)

        # fetch_open_orders NE devrait PAS avoir été appelé (pas de cancel_all)
        # Note: _handle_grid_sl_executed fait le cleanup, pas _close_grid_cycle
        executor._exchange.fetch_open_orders.assert_not_called()


class TestBootCleansOrphanedOrders:
    """Au boot, les ordres trigger orphelins sont annulés."""

    @pytest.mark.asyncio
    async def test_boot_cleans_orphaned_orders(self):
        """Ordres trigger sur des symbols sans grid active sont annulés au boot."""
        executor = _make_executor()

        # Pas de grid active — état vierge
        assert len(executor._grid_states) == 0

        # Simuler 2 ordres orphelins sur Bitget pour un symbol sans grid
        executor._exchange.fetch_open_orders = AsyncMock(return_value=[
            {"id": "orphan_1", "symbol": "ETH/USDT:USDT"},
            {"id": "orphan_2", "symbol": "ETH/USDT:USDT"},
        ])

        await executor._cancel_orphan_orders()

        # Les 2 ordres orphelins doivent être annulés
        cancel_calls = executor._exchange.cancel_order.call_args_list
        cancelled_ids = {c.args[0] for c in cancel_calls}
        assert cancelled_ids == {"orphan_1", "orphan_2"}

    @pytest.mark.asyncio
    async def test_boot_preserves_tracked_orders(self):
        """Ordres trackés par une grid active ne sont PAS annulés."""
        executor = _make_executor()
        futures_sym = "BTC/USDT:USDT"

        # Grid active avec un SL tracké
        state = _make_grid_state(sl_order_id="sl_tracked_1")
        executor._grid_states[futures_sym] = state

        # Bitget a le SL tracké + 1 orphelin
        executor._exchange.fetch_open_orders = AsyncMock(return_value=[
            {"id": "sl_tracked_1", "symbol": futures_sym},
            {"id": "orphan_99", "symbol": "SOL/USDT:USDT"},
        ])

        await executor._cancel_orphan_orders()

        # Seul l'orphelin doit être annulé
        cancel_calls = executor._exchange.cancel_order.call_args_list
        assert len(cancel_calls) == 1
        assert cancel_calls[0].args[0] == "orphan_99"


class TestUpdateSlCancelsPrevious:
    """Quand un nouveau SL est placé, l'ancien est correctement annulé."""

    @pytest.mark.asyncio
    async def test_update_sl_cancels_and_logs(self):
        """Cancel ancien SL + log info quand ça marche."""
        executor = _make_executor()
        futures_sym = "BTC/USDT:USDT"

        state = _make_grid_state(sl_order_id="sl_old_1")
        executor._grid_states[futures_sym] = state

        # create_order pour le nouveau SL
        executor._exchange.create_order = AsyncMock(return_value={
            "id": "sl_new_2", "filled": 0.001,
        })

        await executor._update_grid_sl(futures_sym, state)

        # L'ancien SL doit avoir été cancel
        executor._exchange.cancel_order.assert_called_once_with(
            "sl_old_1", futures_sym,
        )

        # Le nouveau SL doit être stocké
        assert state.sl_order_id == "sl_new_2"

    @pytest.mark.asyncio
    async def test_cancel_failure_falls_back_to_cancel_all(self):
        """Si cancel SL spécifique échoue, fallback sur cancel_all."""
        executor = _make_executor()
        futures_sym = "BTC/USDT:USDT"

        state = _make_grid_state(sl_order_id="sl_stuck_1")
        executor._grid_states[futures_sym] = state

        # cancel_order échoue pour l'ancien SL
        executor._exchange.cancel_order = AsyncMock(
            side_effect=Exception("order not found"),
        )

        # fetch_open_orders retourne l'ancien SL (orphelin)
        executor._exchange.fetch_open_orders = AsyncMock(return_value=[
            {"id": "sl_stuck_1", "symbol": futures_sym},
        ])

        # create_order pour le nouveau SL — doit réussir après le fallback
        # On reset cancel_order pour que le fallback + nouveau placement marchent
        call_count = 0
        async def cancel_side_effect(order_id, symbol):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Premier appel : cancel direct échoue
                raise Exception("order not found")
            # Appels suivants (fallback cancel_all) : succès
            return {}

        executor._exchange.cancel_order = AsyncMock(side_effect=cancel_side_effect)
        executor._exchange.create_order = AsyncMock(return_value={
            "id": "sl_new_2", "filled": 0.001,
        })

        await executor._update_grid_sl(futures_sym, state)

        # fetch_open_orders appelé en fallback
        executor._exchange.fetch_open_orders.assert_called_once_with(
            futures_sym, params={"type": "swap"},
        )

        # Le nouveau SL doit être placé malgré l'échec initial
        assert state.sl_order_id == "sl_new_2"


class TestHandleSlCleansOrphans:
    """Après exécution SL, les ordres orphelins restants sont nettoyés."""

    @pytest.mark.asyncio
    async def test_sl_executed_cleans_remaining_orders(self):
        """Quand SL s'exécute, les anciens SL orphelins sont annulés."""
        executor = _make_executor()
        futures_sym = "BTC/USDT:USDT"

        state = _make_grid_state(sl_order_id="sl_executed")
        executor._grid_states[futures_sym] = state
        executor._risk_manager.register_position({
            "symbol": futures_sym, "direction": "LONG",
            "quantity": 0.001, "entry_price": 50_000.0, "sl_price": 45_000.0,
        })

        # 1 orphelin restant après exécution du SL courant
        executor._exchange.fetch_open_orders = AsyncMock(return_value=[
            {"id": "sl_orphan_old", "symbol": futures_sym},
        ])

        await executor._handle_grid_sl_executed(
            futures_sym, state, exit_price=45_000.0, exit_fee=0.03,
        )

        # L'orphelin doit avoir été cancel
        executor._exchange.cancel_order.assert_called_once_with(
            "sl_orphan_old", futures_sym,
        )

        # Grid state nettoyé
        assert futures_sym not in executor._grid_states
