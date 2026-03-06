"""Tests pour le Parity Watchdog (réconciliation à chaud)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.monitoring.watchdog import Watchdog


@pytest.mark.asyncio
async def test_watchdog_triggers_parity_check():
    """Vérifie que le Watchdog appelle reconcile_on_boot après le bon nombre de ticks."""
    data_engine = MagicMock()
    data_engine.is_connected = True
    data_engine.last_update = None
    
    simulator = MagicMock()
    simulator.runners = []
    simulator.is_kill_switch_triggered.return_value = False
    
    notifier = MagicMock()
    notifier.notify_anomaly = AsyncMock()
    
    executor = MagicMock()
    executor.is_enabled = True
    executor.is_connected = True
    # mock risk manager
    executor._risk_manager.is_kill_switch_triggered = False
    
    # On mocke reconcile_on_boot car c'est lui qu'on veut voir appelé
    with patch("backend.monitoring.watchdog.reconcile_on_boot", new_callable=AsyncMock) as mock_reconcile:
        # On règle l'intervalle sur 1s pour que 900 // 1 = 900 ticks
        # Pour le test on va forcer ticks_15min à une petite valeur en manipulant _check_interval
        watchdog = Watchdog(
            data_engine=data_engine,
            simulator=simulator,
            notifier=notifier,
            check_interval=1,  # 1 tick = 1s
            executor=executor
        )
        
        # On veut que ticks_15min soit petit pour le test
        # 15 min = 900s. Si interval=1s -> 900 ticks.
        # On va tricher et mettre le tick à 899 manuellement.
        watchdog._heartbeat_tick = 899
        
        # Exécuter un check
        await watchdog._check()

        # Le tick doit être reset à 0 après avoir atteint ticks_15min (900)
        assert watchdog._heartbeat_tick == 0
        mock_reconcile.assert_called_once_with(executor)

@pytest.mark.asyncio
async def test_watchdog_parity_skips_disabled_executor():
    """Vérifie que le Watchdog ignore les executors désactivés."""
    data_engine = MagicMock()
    data_engine.is_connected = True
    data_engine.last_update = None
    simulator = MagicMock()
    simulator.runners = []
    simulator.is_kill_switch_triggered.return_value = False
    notifier = MagicMock()
    
    executor = MagicMock()
    executor.is_enabled = False # DÉSACTIVÉ
    
    with patch("backend.monitoring.watchdog.reconcile_on_boot", new_callable=AsyncMock) as mock_reconcile:
        watchdog = Watchdog(
            data_engine=data_engine,
            simulator=simulator,
            notifier=notifier,
            check_interval=900, # 1 tick = 15 min
            executor=executor
        )
        
        await watchdog._check()
        
        # ticks_15min = 900 // 900 = 1. Donc tick devient 1 puis reset à 0.
        assert watchdog._heartbeat_tick == 0
        # Ne doit PAS être appelé car executor.is_enabled est False
        mock_reconcile.assert_not_called()
