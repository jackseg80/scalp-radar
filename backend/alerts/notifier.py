"""Notifier : point d'entrée unique pour toutes les alertes.

Dispatche vers Telegram (+ futurs canaux).
Si Telegram est None (token absent), les notifications sont juste loguées.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from backend.alerts.telegram import TelegramClient


class AnomalyType(str, Enum):
    """Types d'anomalies détectées par le Watchdog."""

    WS_DISCONNECTED = "ws_disconnected"
    DATA_STALE = "data_stale"
    ALL_STRATEGIES_STOPPED = "all_strategies_stopped"
    KILL_SWITCH_GLOBAL = "kill_switch_global"


# Messages formatés par type d'anomalie
_ANOMALY_MESSAGES = {
    AnomalyType.WS_DISCONNECTED: "WebSocket déconnecté",
    AnomalyType.DATA_STALE: "Données obsolètes (>5 min)",
    AnomalyType.ALL_STRATEGIES_STOPPED: "Toutes les stratégies arrêtées",
    AnomalyType.KILL_SWITCH_GLOBAL: "Kill switch global déclenché",
}


class Notifier:
    """Centralise les notifications vers Telegram (+ futurs canaux).

    Si telegram est None, les notifications sont loguées mais pas envoyées.
    """

    def __init__(self, telegram: TelegramClient | None = None) -> None:
        self._telegram = telegram

    async def notify_trade(self, trade: dict, strategy: str) -> None:
        """Notifie un trade clôturé."""
        pnl = trade.get("net_pnl", 0)
        logger.info(
            "Notifier: trade {} net={:+.2f}$", strategy, pnl
        )
        if self._telegram:
            await self._telegram.send_trade_alert(trade, strategy)

    async def notify_kill_switch(self, strategy: str, loss_pct: float) -> None:
        """Notifie l'activation du kill switch."""
        logger.warning(
            "Notifier: KILL SWITCH {} (perte {:.1f}%)", strategy, loss_pct
        )
        if self._telegram:
            await self._telegram.send_kill_switch_alert(strategy, loss_pct)

    async def notify_anomaly(
        self, anomaly_type: AnomalyType, details: str = ""
    ) -> None:
        """Notifie une anomalie détectée par le Watchdog."""
        message = _ANOMALY_MESSAGES.get(anomaly_type, str(anomaly_type))
        if details:
            message = f"{message} — {details}"

        logger.warning("Notifier: anomalie {}: {}", anomaly_type.value, message)
        if self._telegram:
            text = f"<b>Anomalie</b>\n{message}"
            await self._telegram.send_message(text)

    async def notify_startup(self, strategies: list[str]) -> None:
        """Notifie le démarrage."""
        logger.info("Notifier: startup (stratégies: {})", strategies)
        if self._telegram:
            await self._telegram.send_startup_message(strategies)

    async def notify_shutdown(self) -> None:
        """Notifie l'arrêt."""
        logger.info("Notifier: shutdown")
        if self._telegram:
            await self._telegram.send_shutdown_message()
