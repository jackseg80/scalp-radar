"""Watchdog : surveillance du système avec alertes sur anomalies.

Vérifie périodiquement : WS connecté, data freshness, stratégies actives.
Alertes via Notifier avec cooldown anti-spam (5 min par type).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

from backend.alerts.notifier import AnomalyType

if TYPE_CHECKING:
    from backend.alerts.notifier import Notifier
    from backend.backtesting.simulator import Simulator
    from backend.core.data_engine import DataEngine

# Cooldown anti-spam en secondes par type d'anomalie
_ALERT_COOLDOWN_SECONDS = 300  # 5 minutes


class Watchdog:
    """Surveillance du système : data freshness, WS, stratégies.

    Dépendances explicites (pas app_state) pour rester testable.
    """

    def __init__(
        self,
        data_engine: DataEngine,
        simulator: Simulator,
        notifier: Notifier,
        check_interval: int = 30,
    ) -> None:
        self._data_engine = data_engine
        self._simulator = simulator
        self._notifier = notifier
        self._check_interval = check_interval

        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._alerts_sent: int = 0
        self._last_check: datetime | None = None
        self._current_issues: list[str] = []

        # Cooldown : dernière alerte par type
        self._last_alert_time: dict[AnomalyType, datetime] = {}

    async def start(self) -> None:
        """Lance la boucle de surveillance."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Watchdog: activé (intervalle {}s)", self._check_interval)

    async def _loop(self) -> None:
        """Boucle principale du watchdog."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval)
                if self._running:
                    await self._check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Watchdog: erreur check: {}", e)

    async def _check(self) -> None:
        """Exécute tous les checks."""
        self._last_check = datetime.now(tz=timezone.utc)
        self._current_issues = []

        # 1. WebSocket connecté ?
        if not self._data_engine.is_connected:
            self._current_issues.append("WebSocket déconnecté")
            await self._alert(AnomalyType.WS_DISCONNECTED)

        # 2. Data freshness (< 5 min ?)
        last_update = self._data_engine.last_update
        if last_update is not None:
            age = (datetime.now(tz=timezone.utc) - last_update).total_seconds()
            if age > 300:  # 5 minutes
                self._current_issues.append(
                    f"Données obsolètes ({age:.0f}s)"
                )
                await self._alert(
                    AnomalyType.DATA_STALE,
                    f"dernière mise à jour il y a {age:.0f}s",
                )

        # 3. Stratégies actives ?
        if self._simulator.runners:
            all_stopped = all(
                r.is_kill_switch_triggered for r in self._simulator.runners
            )
            if all_stopped:
                self._current_issues.append("Toutes les stratégies arrêtées")
                await self._alert(AnomalyType.ALL_STRATEGIES_STOPPED)

        # 4. Kill switch global ?
        if self._simulator.is_kill_switch_triggered():
            if AnomalyType.KILL_SWITCH_GLOBAL not in self._last_alert_time:
                self._current_issues.append("Kill switch global")
                await self._alert(AnomalyType.KILL_SWITCH_GLOBAL)

    async def _alert(self, anomaly_type: AnomalyType, details: str = "") -> None:
        """Envoie une alerte avec cooldown anti-spam."""
        now = datetime.now(tz=timezone.utc)

        last = self._last_alert_time.get(anomaly_type)
        if last is not None:
            elapsed = (now - last).total_seconds()
            if elapsed < _ALERT_COOLDOWN_SECONDS:
                return  # Cooldown pas expiré

        self._last_alert_time[anomaly_type] = now
        self._alerts_sent += 1
        await self._notifier.notify_anomaly(anomaly_type, details)

    def get_status(self) -> dict:
        """Retourne le statut du watchdog."""
        return {
            "last_check": (
                self._last_check.isoformat() if self._last_check else None
            ),
            "issues": list(self._current_issues),
            "alerts_sent": self._alerts_sent,
        }

    async def stop(self) -> None:
        """Arrête le watchdog."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Watchdog: arrêté")
