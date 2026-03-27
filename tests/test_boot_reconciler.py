"""Tests de réconciliation au boot — persistence PnL positions fermées pendant downtime.

Sprint 66 item A : _persist_live_trade doit être appelé pour les Cas 3 et 4
(positions fermées pendant que le bot était éteint).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.execution.boot_reconciler import _reconcile_symbol, _reconcile_grid_symbol
from backend.execution.executor import GridLiveState, GridLivePosition, LivePosition


def _make_ex() -> MagicMock:
    """Executor minimal duck-typed pour le reconciler."""
    ex = MagicMock()
    ex._running = False  # Boot réel
    ex._positions = {}
    ex._grid_states = {}
    ex._reconciliation_pnl = 0.0
    ex._reconciliation_count = 0

    # Config
    ex._config.risk.position.default_leverage = 6

    # Méthodes async
    ex._fetch_positions_safe = AsyncMock()
    ex._fetch_exit_price = AsyncMock(return_value=95.0)
    ex._update_grid_sl = AsyncMock()
    ex._cancel_all_open_orders = AsyncMock()
    ex._persist_live_trade = AsyncMock()
    ex._record_grid_close = MagicMock()
    ex._calculate_pnl = MagicMock(return_value=-5.0)

    # RiskManager
    ex._risk_manager = MagicMock()
    ex._risk_manager.record_trade_result = MagicMock()
    ex._risk_manager.unregister_position = MagicMock()

    # Notifier
    ex._notifier = AsyncMock()

    return ex


# ── Cas 3 : position mono fermée pendant downtime ─────────────────────────────

@pytest.mark.asyncio
async def test_downtime_close_persisted_mono():
    """Cas 3 : position mono locale sans équivalent exchange → _persist_live_trade appelé."""
    ex = _make_ex()

    futures_sym = "BTC/USDT:USDT"
    pos = LivePosition(
        symbol=futures_sym,
        direction="LONG",
        entry_price=100.0,
        quantity=0.01,
        entry_order_id="entry_1",
        strategy_name="grid_atr",
    )
    ex._positions = {futures_sym: pos}
    ex._fetch_positions_safe.return_value = []  # Pas de position sur l'exchange

    await _reconcile_symbol(ex, futures_sym)

    # _persist_live_trade doit être appelé avec les bons arguments
    ex._persist_live_trade.assert_called_once()
    call_kwargs = ex._persist_live_trade.call_args
    args, kwargs = call_kwargs

    assert args[0] == "force_close"
    assert args[1] == futures_sym
    assert args[2] == "sell"    # LONG → close = sell
    assert args[3] == "LONG"
    assert kwargs.get("context") == "closed_during_downtime"
    assert kwargs.get("strategy_name") == "grid_atr"
    assert kwargs.get("pnl") is not None


@pytest.mark.asyncio
async def test_downtime_close_persisted_mono_short():
    """Cas 3 SHORT : close_side doit être 'buy'."""
    ex = _make_ex()

    futures_sym = "ETH/USDT:USDT"
    pos = LivePosition(
        symbol=futures_sym,
        direction="SHORT",
        entry_price=200.0,
        quantity=0.05,
        entry_order_id="entry_2",
        strategy_name="grid_atr",
    )
    ex._positions = {futures_sym: pos}
    ex._fetch_positions_safe.return_value = []

    await _reconcile_symbol(ex, futures_sym)

    ex._persist_live_trade.assert_called_once()
    args, kwargs = ex._persist_live_trade.call_args
    assert args[2] == "buy"     # SHORT → close = buy
    assert args[3] == "SHORT"
    assert kwargs.get("context") == "closed_during_downtime"


# ── Cas 4 : grid fermée pendant downtime ──────────────────────────────────────

@pytest.mark.asyncio
async def test_downtime_close_persisted_grid():
    """Cas 4 : cycle grid local sans position exchange → _persist_live_trade appelé."""
    ex = _make_ex()

    futures_sym = "SOL/USDT:USDT"
    state = GridLiveState(
        symbol=futures_sym,
        direction="LONG",
        strategy_name="grid_atr",
        leverage=6,
        sl_order_id="sl_123",
        positions=[
            GridLivePosition(level=0, entry_price=100.0, quantity=0.5, entry_order_id="e1"),
            GridLivePosition(level=1, entry_price=95.0, quantity=0.5, entry_order_id="e2"),
        ],
    )
    ex._grid_states = {futures_sym: state}
    ex._fetch_positions_safe.return_value = []  # Pas de position sur l'exchange
    ex._exchange = AsyncMock()
    ex._exchange.fetch_order = AsyncMock(side_effect=Exception("not found"))

    await _reconcile_grid_symbol(ex, futures_sym)

    ex._persist_live_trade.assert_called_once()
    args, kwargs = ex._persist_live_trade.call_args
    assert args[0] == "force_close"
    assert args[1] == futures_sym
    assert args[2] == "sell"    # LONG → sell
    assert args[3] == "LONG"
    assert kwargs.get("context") == "closed_during_downtime"
    assert kwargs.get("strategy_name") == "grid_atr"


@pytest.mark.asyncio
async def test_downtime_no_persist_when_position_still_open():
    """Cas 1 : position toujours ouverte → _persist_live_trade NON appelé."""
    ex = _make_ex()

    futures_sym = "ADA/USDT:USDT"
    state = GridLiveState(
        symbol=futures_sym,
        direction="LONG",
        strategy_name="grid_atr",
        leverage=6,
        sl_order_id="sl_ok",
        positions=[
            GridLivePosition(level=0, entry_price=0.50, quantity=100.0, entry_order_id="e1"),
        ],
    )
    ex._grid_states = {futures_sym: state}
    # Position toujours ouverte côté exchange
    ex._fetch_positions_safe.return_value = [{"contracts": 100.0}]
    ex._exchange = AsyncMock()
    ex._exchange.fetch_order = AsyncMock(return_value={"status": "open"})

    await _reconcile_grid_symbol(ex, futures_sym)

    ex._persist_live_trade.assert_not_called()
