"""Tests pour valider la correction des SL orphelins (Mission 2026-03-06)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ccxt import OrderNotFound

from backend.execution.executor import Executor, GridLiveState, GridLivePosition
from backend.execution.boot_reconciler import _reconcile_grid_symbol


@pytest.mark.asyncio
async def test_reconcile_grid_symbol_handles_40109_order_not_found():
    """Vérifie que l'erreur 40109 reset sl_order_id et purge les orphelins."""
    ex = MagicMock()
    ex._log_prefix = "Executor"
    ex._running = False # Simule boot
    
    futures_sym = "BTC/USDT:USDT"
    state = GridLiveState(
        symbol=futures_sym,
        direction="LONG",
        strategy_name="grid_atr",
        leverage=6,
        sl_order_id="old_sl_id",
        positions=[GridLivePosition(level=0, entry_price=100.0, quantity=1.0, entry_order_id="entry_1")]
    )
    ex._grid_states = {futures_sym: state}
    
    # Simuler l'erreur Bitget 40109 (Order not found)
    # has_position doit être True pour entrer dans le bloc sl_order_id
    ex._fetch_positions_safe = AsyncMock(return_value=[{"contracts": 1.0, "symbol": futures_sym}])
    ex._exchange.fetch_order = AsyncMock(side_effect=Exception("bitget 40109: Order not found"))
    ex._cancel_all_open_orders = AsyncMock()
    ex._notifier.notify_reconciliation = AsyncMock()
    ex._exchange.set_leverage = AsyncMock()
    ex._update_grid_sl = AsyncMock()
    
    # Exécuter la réconciliation
    await _reconcile_grid_symbol(ex, futures_sym)
    
    # Vérifications
    assert state.sl_order_id is None
    ex._cancel_all_open_orders.assert_called_once_with(futures_sym)


@pytest.mark.asyncio
async def test_update_grid_sl_idempotence():
    """Vérifie que _update_grid_sl réutilise un SL existant sur l'exchange."""
    config = MagicMock()
    config.secrets.live_trading = True
    # Mock strategies dict
    strat_cfg = MagicMock()
    strat_cfg.sl_percent = 10.0
    config.strategies.grid_atr = strat_cfg
    
    ex = Executor(config, MagicMock(), MagicMock(), strategy_name="grid_atr")
    ex._exchange = AsyncMock()
    ex._data_engine = MagicMock()
    ex._data_engine.get_last_update.return_value = datetime.now(tz=timezone.utc)
    
    futures_sym = "BTC/USDT:USDT"
    state = GridLiveState(
        symbol=futures_sym,
        direction="LONG",
        strategy_name="grid_atr",
        leverage=6,
        positions=[GridLivePosition(level=0, entry_price=100.0, quantity=1.0, entry_order_id="entry_1")]
    )
    
    # Simuler un SL identique déjà présent sur l'exchange (trigger=90, qty=1.0, side=sell)
    mock_order = {
        "id": "existing_sl_999",
        "triggerPrice": 90.0,
        "amount": 1.0,
        "side": "sell"
    }
    ex._exchange.fetch_open_orders = AsyncMock(return_value=[mock_order])
    ex._place_sl_with_retry = AsyncMock()
    
    # Mock precision
    ex._exchange.price_to_precision = lambda s, p: str(p)
    
    await ex._update_grid_sl(futures_sym, state)
    
    # Doit avoir trouvé et utilisé l'ID existant
    assert state.sl_order_id == "existing_sl_999"
    ex._place_sl_with_retry.assert_not_called()


@pytest.mark.asyncio
async def test_update_grid_sl_blocks_if_stale():
    """Vérifie que _update_grid_sl refuse de modifier le SL si les prix sont stale."""
    config = MagicMock()
    ex = Executor(config, MagicMock(), MagicMock(), strategy_name="grid_atr")
    ex._data_engine = MagicMock()
    
    # Simuler des données vieilles de 10 minutes
    old_ts = datetime.now(tz=timezone.utc).timestamp() - 600
    ex._data_engine.get_last_update.return_value = datetime.fromtimestamp(old_ts, tz=timezone.utc)
    
    futures_sym = "BTC/USDT:USDT"
    state = GridLiveState(symbol=futures_sym, direction="LONG", strategy_name="grid_atr", leverage=6)
    state.sl_order_id = "should_not_change"
    
    await ex._update_grid_sl(futures_sym, state)
    
    # Ne doit pas avoir bougé car stale
    assert state.sl_order_id == "should_not_change"
