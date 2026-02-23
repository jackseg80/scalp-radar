"""Tests pour les fonctionnalités de maintenance :
- WAL checkpoint (Database)
- Monitoring disque (Watchdog)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from backend.alerts.notifier import AnomalyType, Notifier
from backend.core.database import Database
from backend.monitoring.watchdog import Watchdog, _DISK_ALERT_THRESHOLD_PCT


# ─── WAL CHECKPOINT ─────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def db():
    """DB en mémoire pour chaque test."""
    database = Database(db_path=":memory:")
    await database.init()
    yield database
    await database.close()


@pytest.mark.asyncio
class TestWalCheckpoint:
    async def test_wal_checkpoint_returns_dict(self, db):
        """wal_checkpoint() retourne les métriques busy/log/checkpointed."""
        result = await db.wal_checkpoint()
        assert isinstance(result, dict)
        assert "busy" in result
        assert "log" in result
        assert "checkpointed" in result

    async def test_wal_checkpoint_values_are_ints(self, db):
        """Les valeurs du checkpoint sont des entiers (PRAGMA retourne des ints)."""
        result = await db.wal_checkpoint()
        assert isinstance(result["busy"], int)
        assert isinstance(result["log"], int)
        assert isinstance(result["checkpointed"], int)

    async def test_wal_checkpoint_on_in_memory_db(self, db):
        """WAL checkpoint fonctionne sur une DB en mémoire (pas de WAL réel, log=0)."""
        result = await db.wal_checkpoint()
        # DB en mémoire = pas de WAL, log et checkpointed sont à 0 ou -1
        assert result["log"] >= -1
        assert result["checkpointed"] >= -1

    async def test_start_maintenance_loop_creates_task(self, db):
        """start_maintenance_loop() crée une tâche asyncio."""
        db.start_maintenance_loop()
        assert db._maintenance_task is not None
        assert not db._maintenance_task.done()
        # Nettoyage
        db._maintenance_task.cancel()
        try:
            await db._maintenance_task
        except asyncio.CancelledError:
            pass

    async def test_start_maintenance_loop_idempotent(self, db):
        """Appeler start_maintenance_loop() deux fois ne crée pas deux tâches."""
        db.start_maintenance_loop()
        task_first = db._maintenance_task
        db.start_maintenance_loop()
        # La seconde invocation ne remplace pas la tâche déjà active
        assert db._maintenance_task is task_first
        task_first.cancel()
        try:
            await task_first
        except asyncio.CancelledError:
            pass

    async def test_close_cancels_maintenance_task(self, db):
        """close() annule la tâche de maintenance."""
        db.start_maintenance_loop()
        task = db._maintenance_task
        assert not task.done()
        await db.close()
        # Après close(), la tâche doit être annulée
        assert task.done()


# ─── MONITORING DISQUE (WATCHDOG) ───────────────────────────────────────────


def _make_engine(connected: bool = True, last_update_age_seconds: float = 10.0):
    from datetime import timedelta
    engine = MagicMock()
    engine.is_connected = connected
    engine.last_update = datetime.now(tz=timezone.utc) - timedelta(seconds=last_update_age_seconds)
    return engine


def _make_simulator():
    simulator = MagicMock()
    runner = MagicMock()
    runner.is_kill_switch_triggered = False
    simulator.runners = [runner]
    simulator.is_kill_switch_triggered.return_value = False
    return simulator


@pytest.mark.asyncio
class TestDiskMonitoring:
    async def test_disk_ok_no_alert(self):
        """Pas d'alerte quand le disque est en dessous du seuil."""
        engine = _make_engine()
        simulator = _make_simulator()
        notifier = MagicMock(spec=Notifier)
        notifier.notify_anomaly = AsyncMock()

        wd = Watchdog(engine, simulator, notifier)

        # Simuler un disque à 50% (en dessous du seuil 85%)
        with patch("backend.monitoring.watchdog.shutil.disk_usage") as mock_du:
            mock_du.return_value = MagicMock(
                total=100 * 1024 ** 3,
                used=50 * 1024 ** 3,
                free=50 * 1024 ** 3,
            )
            await wd._check()

        disk_calls = [
            c for c in notifier.notify_anomaly.call_args_list
            if c[0][0] == AnomalyType.DISK_FULL
        ]
        assert len(disk_calls) == 0

    async def test_disk_full_alert_triggered(self):
        """Alerte DISK_FULL quand le disque dépasse le seuil."""
        engine = _make_engine()
        simulator = _make_simulator()
        notifier = MagicMock(spec=Notifier)
        notifier.notify_anomaly = AsyncMock()

        wd = Watchdog(engine, simulator, notifier)

        # Simuler un disque à 90% (au-dessus du seuil 85%)
        with patch("backend.monitoring.watchdog.shutil.disk_usage") as mock_du:
            mock_du.return_value = MagicMock(
                total=100 * 1024 ** 3,
                used=90 * 1024 ** 3,
                free=10 * 1024 ** 3,
            )
            await wd._check()

        disk_calls = [
            c for c in notifier.notify_anomaly.call_args_list
            if c[0][0] == AnomalyType.DISK_FULL
        ]
        assert len(disk_calls) == 1

    async def test_disk_full_issue_in_current_issues(self):
        """Le problème disque apparaît dans _current_issues."""
        engine = _make_engine()
        simulator = _make_simulator()
        notifier = MagicMock(spec=Notifier)
        notifier.notify_anomaly = AsyncMock()

        wd = Watchdog(engine, simulator, notifier)

        with patch("backend.monitoring.watchdog.shutil.disk_usage") as mock_du:
            mock_du.return_value = MagicMock(
                total=100 * 1024 ** 3,
                used=92 * 1024 ** 3,
                free=8 * 1024 ** 3,
            )
            await wd._check()

        assert any("Disque" in issue for issue in wd._current_issues)

    async def test_disk_error_does_not_raise(self):
        """Si disk_usage() échoue, le watchdog continue sans planter."""
        engine = _make_engine()
        simulator = _make_simulator()
        notifier = MagicMock(spec=Notifier)
        notifier.notify_anomaly = AsyncMock()

        wd = Watchdog(engine, simulator, notifier)

        with patch("backend.monitoring.watchdog.shutil.disk_usage", side_effect=OSError("Permission denied")):
            # Ne doit pas lever d'exception
            await wd._check()

    async def test_disk_full_cooldown(self):
        """L'alerte DISK_FULL n'est pas envoyée deux fois de suite (cooldown watchdog)."""
        engine = _make_engine()
        simulator = _make_simulator()
        notifier = MagicMock(spec=Notifier)
        notifier.notify_anomaly = AsyncMock()

        wd = Watchdog(engine, simulator, notifier)

        with patch("backend.monitoring.watchdog.shutil.disk_usage") as mock_du:
            mock_du.return_value = MagicMock(
                total=100 * 1024 ** 3,
                used=90 * 1024 ** 3,
                free=10 * 1024 ** 3,
            )
            await wd._check()
            first_count = wd._alerts_sent
            await wd._check()
            # Cooldown watchdog : pas de deuxième alerte
            assert wd._alerts_sent == first_count


# ─── DISK_FULL dans AnomalyType ─────────────────────────────────────────────


class TestDiskFullAnomalyType:
    def test_disk_full_in_anomaly_type(self):
        """DISK_FULL existe bien dans l'enum AnomalyType."""
        assert AnomalyType.DISK_FULL == "disk_full"

    def test_disk_full_has_cooldown_in_notifier(self):
        """DISK_FULL a un cooldown défini dans _ANOMALY_COOLDOWNS."""
        from backend.alerts.notifier import _ANOMALY_COOLDOWNS
        assert AnomalyType.DISK_FULL in _ANOMALY_COOLDOWNS
        assert _ANOMALY_COOLDOWNS[AnomalyType.DISK_FULL] == 3600

    def test_disk_full_has_message_in_notifier(self):
        """DISK_FULL a un message défini dans _ANOMALY_MESSAGES."""
        from backend.alerts.notifier import _ANOMALY_MESSAGES
        assert AnomalyType.DISK_FULL in _ANOMALY_MESSAGES
        assert "85" in _ANOMALY_MESSAGES[AnomalyType.DISK_FULL]
