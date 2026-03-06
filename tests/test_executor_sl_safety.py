"""Tests pour la sécurisation critique du replacement de SL (Anti-Position Nue)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.execution.executor import Executor, GridLiveState, GridLivePosition


@pytest.mark.asyncio
async def test_proactive_missing_sl_replacement():
    """Vérifie que _check_missing_sl détecte et replace un SL manquant."""
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
    
    # Simuler une position active SANS sl_order_id
    futures_sym = "BTC/USDT:USDT"
    state = GridLiveState(
        symbol=futures_sym,
        direction="LONG",
        strategy_name="grid_atr",
        leverage=6,
        sl_order_id=None, # MANQUANT
        positions=[GridLivePosition(level=0, entry_price=100.0, quantity=1.0, entry_order_id="entry_1")]
    )
    ex._grid_states = {futures_sym: state}
    
    # Mock des méthodes de placement
    ex._exchange.fetch_open_orders = AsyncMock(return_value=[])
    ex._exchange.create_order = AsyncMock(return_value={"id": "new_sl_id"})
    ex._state_save_callback = AsyncMock()
    
    # 1. Lancer le check proactif
    await ex._check_missing_sl()
    
    # 2. Vérifier que le SL a été remplacé
    assert state.sl_order_id == "new_sl_id"
    ex._exchange.create_order.assert_called()
    ex._state_save_callback.assert_called()


@pytest.mark.asyncio
async def test_sl_replacement_handles_stale_price_via_rest():
    """Vérifie que _update_grid_sl utilise fetch_ticker si le flux WS est stale."""
    config = MagicMock()
    config.secrets.live_trading = True
    strat_cfg = MagicMock()
    strat_cfg.sl_percent = 10.0
    config.strategies.grid_atr = strat_cfg
    
    ex = Executor(config, MagicMock(), MagicMock(), strategy_name="grid_atr")
    ex._exchange = AsyncMock()
    ex._data_engine = MagicMock()
    
    # Simuler un flux STALE (> 5 min)
    stale_ts = datetime.now(tz=timezone.utc) - timedelta(minutes=10)
    ex._data_engine.get_last_update.return_value = stale_ts
    
    futures_sym = "BTC/USDT:USDT"
    state = GridLiveState(
        symbol=futures_sym,
        direction="LONG",
        strategy_name="grid_atr",
        leverage=6,
        sl_order_id=None,
        positions=[GridLivePosition(level=0, entry_price=100.0, quantity=1.0, entry_order_id="entry_1")]
    )
    
    # Mocks
    ex._exchange.fetch_ticker = AsyncMock(return_value={"last": 105.0})
    ex._exchange.fetch_open_orders = AsyncMock(return_value=[])
    ex._exchange.create_order = AsyncMock(return_value={"id": "emergency_sl_id"})
    
    # Exécuter
    await ex._update_grid_sl(futures_sym, state)
    
    # Vérifier que fetch_ticker a été appelé pour sécuriser le prix
    ex._exchange.fetch_ticker.assert_called_once_with(futures_sym)
    assert state.sl_order_id == "emergency_sl_id"


@pytest.mark.asyncio
async def test_boot_reconciler_forces_sl_replacement_post_purge():
    """Vérifie que le boot_reconciler force le replacement du SL après une purge 40109."""
    from backend.execution.boot_reconciler import _reconcile_grid_symbol
    
    ex = MagicMock()
    ex._log_prefix = "Executor"
    ex._running = True # On est au runtime (via Watchdog)
    
    futures_sym = "BTC/USDT:USDT"
    # Utiliser de vrais objets dataclass pour éviter TypeError sur les comparaisons
    state = GridLiveState(
        symbol=futures_sym,
        direction="LONG",
        strategy_name="grid_atr",
        leverage=6,
        sl_order_id="dead_id",
        positions=[GridLivePosition(level=0, entry_price=100.0, quantity=1.0, entry_order_id="entry_1")]
    )
    ex._grid_states = {futures_sym: state}
    
    # Simuler erreur 40109 -> purge
    ex._exchange.fetch_order = AsyncMock(side_effect=Exception("bitget 40109: Order not found"))
    ex._fetch_positions_safe = AsyncMock(return_value=[{"contracts": 1.0, "symbol": futures_sym}])
    ex._cancel_all_open_orders = AsyncMock()
    ex._update_grid_sl = AsyncMock()
    ex._exchange.set_leverage = AsyncMock()
    
    await _reconcile_grid_symbol(ex, futures_sym)
    
    # Vérifier que sl_order_id a été reset ET que _update_grid_sl a été forcé
    assert state.sl_order_id is None
    ex._cancel_all_open_orders.assert_called_once()
    # C'est ICI le point critique : _update_grid_sl doit être appelé même si _running=True
    ex._update_grid_sl.assert_called_once_with(futures_sym, state)
