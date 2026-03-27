"""Tests pour la sécurisation critique du replacement de SL (Anti-Position Nue)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from backend.execution.executor import Executor, GridLiveState, GridLivePosition
from backend.execution.order_monitor import watch_orders_loop


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
async def test_sl_replacement_ignores_stale_ws_uses_avg_entry():
    """Sprint 65b : _update_grid_sl place le SL sur avg_entry_price sans vérification réseau.

    Le stale-check WS a été supprimé (Bug 1 Sprint 65b) — le SL se base sur
    avg_entry_price (donnée locale fiable), même si le flux WS est stale.
    """
    config = MagicMock()
    config.secrets.live_trading = True
    strat_cfg = MagicMock()
    strat_cfg.sl_percent = 10.0
    config.strategies.grid_atr = strat_cfg

    ex = Executor(config, MagicMock(), MagicMock(), strategy_name="grid_atr")
    ex._exchange = AsyncMock()

    futures_sym = "BTC/USDT:USDT"
    state = GridLiveState(
        symbol=futures_sym,
        direction="LONG",
        strategy_name="grid_atr",
        leverage=6,
        sl_order_id=None,
        positions=[GridLivePosition(level=0, entry_price=100.0, quantity=1.0, entry_order_id="entry_1")],
    )

    ex._exchange.fetch_open_orders = AsyncMock(return_value=[])
    ex._exchange.create_order = AsyncMock(return_value={"id": "sl_placed"})
    ex._state_save_callback = AsyncMock()

    await ex._update_grid_sl(futures_sym, state)

    # SL placé via avg_entry_price — sans fetch_ticker
    ex._exchange.fetch_ticker.assert_not_called()
    assert state.sl_order_id == "sl_placed"


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


# ── Sprint 66 : Tests B (WS reconnect) + D (alerte Telegram) ─────────────────

def _make_grid_state_for_sl(symbol: str = "BTC/USDT:USDT", sl_order_id=None) -> GridLiveState:
    return GridLiveState(
        symbol=symbol,
        direction="LONG",
        strategy_name="grid_atr",
        leverage=6,
        sl_order_id=sl_order_id,
        positions=[GridLivePosition(level=0, entry_price=100.0, quantity=1.0, entry_order_id="e1")],
    )


@pytest.mark.asyncio
async def test_check_missing_sl_triggered_after_ws_error():
    """B — Après une erreur watchOrders, _check_missing_sl est appelé immédiatement."""
    ex = MagicMock()
    ex._running = True
    ex._positions = {}
    ex._grid_states = {"BTC/USDT:USDT": _make_grid_state_for_sl()}
    ex._pending_entry_orders = {}

    call_count = 0

    async def fake_watch_orders():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("WS disconnect 1006")
        # Arrêter après 2 itérations
        ex._running = False
        return []

    ex._exchange = AsyncMock()
    ex._exchange.watch_orders = fake_watch_orders
    ex._check_missing_sl = AsyncMock()

    await watch_orders_loop(ex)

    # _check_missing_sl doit avoir été appelé au moins une fois (post-reconnect)
    ex._check_missing_sl.assert_called()


@pytest.mark.asyncio
async def test_sl_missing_telegram_alert_after_30s():
    """D — Alerte Telegram envoyée si SL manquant depuis >30s."""
    config = MagicMock()
    config.secrets.live_trading = True
    strat_cfg = MagicMock()
    strat_cfg.sl_percent = 10.0
    config.strategies.grid_atr = strat_cfg

    notifier = AsyncMock()
    ex = Executor(config, MagicMock(), notifier, strategy_name="grid_atr")
    ex._exchange = AsyncMock()
    ex._exchange.fetch_open_orders = AsyncMock(return_value=[])
    ex._exchange.create_order = AsyncMock(return_value={"id": "new_sl"})
    ex._state_save_callback = AsyncMock()

    futures_sym = "BTC/USDT:USDT"
    state = _make_grid_state_for_sl(futures_sym, sl_order_id=None)
    ex._grid_states = {futures_sym: state}

    # Simuler SL manquant depuis 35 secondes
    ex._sl_missing_since[futures_sym] = datetime.now(tz=timezone.utc) - timedelta(seconds=35)

    await ex._check_missing_sl()

    # notify_anomaly doit avoir été appelé
    notifier.notify_anomaly.assert_called_once()
    args = notifier.notify_anomaly.call_args[0]
    assert "SL_PLACEMENT_FAILED" in str(args[0]) or hasattr(args[0], "name")


@pytest.mark.asyncio
async def test_sl_missing_no_alert_before_30s():
    """D — Pas d'alerte Telegram si SL manquant depuis <30s."""
    config = MagicMock()
    config.secrets.live_trading = True
    strat_cfg = MagicMock()
    strat_cfg.sl_percent = 10.0
    config.strategies.grid_atr = strat_cfg

    notifier = AsyncMock()
    ex = Executor(config, MagicMock(), notifier, strategy_name="grid_atr")
    ex._exchange = AsyncMock()
    ex._exchange.fetch_open_orders = AsyncMock(return_value=[])
    ex._exchange.create_order = AsyncMock(return_value={"id": "new_sl"})
    ex._state_save_callback = AsyncMock()

    futures_sym = "ETH/USDT:USDT"
    state = _make_grid_state_for_sl(futures_sym, sl_order_id=None)
    ex._grid_states = {futures_sym: state}

    # SL manquant depuis seulement 10 secondes → pas d'alerte
    ex._sl_missing_since[futures_sym] = datetime.now(tz=timezone.utc) - timedelta(seconds=10)

    await ex._check_missing_sl()

    notifier.notify_anomaly.assert_not_called()


@pytest.mark.asyncio
async def test_sl_missing_since_reset_when_sl_present():
    """D — _sl_missing_since est nettoyé quand sl_order_id est valide."""
    config = MagicMock()
    config.secrets.live_trading = True
    strat_cfg = MagicMock()
    strat_cfg.sl_percent = 10.0
    config.strategies.grid_atr = strat_cfg

    notifier = AsyncMock()
    ex = Executor(config, MagicMock(), notifier, strategy_name="grid_atr")
    ex._exchange = AsyncMock()

    futures_sym = "SOL/USDT:USDT"
    state = _make_grid_state_for_sl(futures_sym, sl_order_id="existing_sl")
    ex._grid_states = {futures_sym: state}

    # Pré-remplir _sl_missing_since (résidu d'un ancien check)
    ex._sl_missing_since[futures_sym] = datetime.now(tz=timezone.utc) - timedelta(seconds=60)

    await ex._check_missing_sl()

    # Le tracking doit être nettoyé
    assert futures_sym not in ex._sl_missing_since
