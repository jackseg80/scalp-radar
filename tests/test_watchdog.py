"""Tests pour le Watchdog."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from backend.alerts.notifier import AnomalyType, Notifier
from backend.monitoring.watchdog import Watchdog


def _make_engine(connected: bool = True, last_update_age_seconds: float = 10.0):
    """Crée un mock DataEngine."""
    engine = MagicMock()
    engine.is_connected = connected
    if last_update_age_seconds is not None:
        engine.last_update = datetime.now(tz=timezone.utc) - timedelta(
            seconds=last_update_age_seconds
        )
    else:
        engine.last_update = None
    return engine


def _make_simulator(runners_kill_switched: list[bool] | None = None):
    """Crée un mock Simulator."""
    simulator = MagicMock()
    if runners_kill_switched is None:
        runners_kill_switched = [False]

    runners = []
    for ks in runners_kill_switched:
        runner = MagicMock()
        runner.is_kill_switch_triggered = ks
        runners.append(runner)

    simulator.runners = runners
    simulator.is_kill_switch_triggered.return_value = any(runners_kill_switched)
    return simulator


class TestWatchdogCheck:
    """Tests pour _check()."""

    @pytest.mark.asyncio
    async def test_all_ok_no_alert(self):
        """Pas d'alerte quand tout va bien."""
        engine = _make_engine(connected=True, last_update_age_seconds=30)
        simulator = _make_simulator([False])
        notifier = MagicMock(spec=Notifier)
        notifier.notify_anomaly = AsyncMock()

        wd = Watchdog(engine, simulator, notifier, check_interval=30)
        await wd._check()

        notifier.notify_anomaly.assert_not_called()
        assert wd._current_issues == []

    @pytest.mark.asyncio
    async def test_ws_disconnected_alert(self):
        """Alerte quand le WS est déconnecté."""
        engine = _make_engine(connected=False, last_update_age_seconds=30)
        simulator = _make_simulator([False])
        notifier = MagicMock(spec=Notifier)
        notifier.notify_anomaly = AsyncMock()

        wd = Watchdog(engine, simulator, notifier, check_interval=30)
        await wd._check()

        notifier.notify_anomaly.assert_called()
        call_args = notifier.notify_anomaly.call_args_list[0]
        assert call_args[0][0] == AnomalyType.WS_DISCONNECTED
        assert "WebSocket déconnecté" in wd._current_issues

    @pytest.mark.asyncio
    async def test_data_stale_alert(self):
        """Alerte quand les données sont obsolètes (> 5 min)."""
        engine = _make_engine(connected=True, last_update_age_seconds=600)
        simulator = _make_simulator([False])
        notifier = MagicMock(spec=Notifier)
        notifier.notify_anomaly = AsyncMock()

        wd = Watchdog(engine, simulator, notifier, check_interval=30)
        await wd._check()

        # Trouver l'appel DATA_STALE
        calls = [c for c in notifier.notify_anomaly.call_args_list
                 if c[0][0] == AnomalyType.DATA_STALE]
        assert len(calls) == 1
        assert any("obsolètes" in issue for issue in wd._current_issues)

    @pytest.mark.asyncio
    async def test_all_strategies_stopped_alert(self):
        """Alerte quand toutes les stratégies sont arrêtées."""
        engine = _make_engine(connected=True, last_update_age_seconds=30)
        simulator = _make_simulator([True, True])
        notifier = MagicMock(spec=Notifier)
        notifier.notify_anomaly = AsyncMock()

        wd = Watchdog(engine, simulator, notifier, check_interval=30)
        await wd._check()

        calls = [c for c in notifier.notify_anomaly.call_args_list
                 if c[0][0] == AnomalyType.ALL_STRATEGIES_STOPPED]
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_cooldown_anti_spam(self):
        """L'alerte n'est pas renvoyée avant le cooldown."""
        engine = _make_engine(connected=False, last_update_age_seconds=30)
        simulator = _make_simulator([False])
        notifier = MagicMock(spec=Notifier)
        notifier.notify_anomaly = AsyncMock()

        wd = Watchdog(engine, simulator, notifier, check_interval=30)

        # Premier check → alerte envoyée
        await wd._check()
        assert wd._alerts_sent == 1

        # Deuxième check → cooldown, pas d'alerte
        await wd._check()
        assert wd._alerts_sent == 1  # Toujours 1


class TestWatchdogStatus:
    """Tests pour get_status()."""

    @pytest.mark.asyncio
    async def test_get_status_initial(self):
        engine = _make_engine()
        simulator = _make_simulator()
        notifier = MagicMock(spec=Notifier)

        wd = Watchdog(engine, simulator, notifier)
        status = wd.get_status()

        assert status["last_check"] is None
        assert status["issues"] == []
        assert status["alerts_sent"] == 0

    @pytest.mark.asyncio
    async def test_get_status_after_check(self):
        engine = _make_engine(connected=False)
        simulator = _make_simulator([False])
        notifier = MagicMock(spec=Notifier)
        notifier.notify_anomaly = AsyncMock()

        wd = Watchdog(engine, simulator, notifier)
        await wd._check()

        status = wd.get_status()
        assert status["last_check"] is not None
        assert len(status["issues"]) > 0
        assert status["alerts_sent"] > 0


class TestWatchdogLifecycle:
    """Tests pour start/stop."""

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        engine = _make_engine()
        simulator = _make_simulator()
        notifier = MagicMock(spec=Notifier)

        wd = Watchdog(engine, simulator, notifier, check_interval=3600)
        await wd.start()

        assert wd._task is not None
        assert not wd._task.done()

        await wd.stop()
        assert wd._running is False
