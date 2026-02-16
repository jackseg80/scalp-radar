"""Notifier : point d'entrée unique pour toutes les alertes.

Dispatche vers Telegram (+ futurs canaux).
Si Telegram est None (token absent), les notifications sont juste loguées.
"""

from __future__ import annotations

import time
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
    EXECUTOR_DISCONNECTED = "executor_disconnected"
    KILL_SWITCH_LIVE = "kill_switch_live"
    SL_PLACEMENT_FAILED = "sl_placement_failed"


# Messages formatés par type d'anomalie
_ANOMALY_MESSAGES = {
    AnomalyType.WS_DISCONNECTED: "WebSocket déconnecté",
    AnomalyType.DATA_STALE: "Données obsolètes (>5 min)",
    AnomalyType.ALL_STRATEGIES_STOPPED: "Toutes les stratégies arrêtées",
    AnomalyType.KILL_SWITCH_GLOBAL: "Kill switch global déclenché",
    AnomalyType.EXECUTOR_DISCONNECTED: "Executor live déconnecté",
    AnomalyType.KILL_SWITCH_LIVE: "Kill switch LIVE déclenché",
    AnomalyType.SL_PLACEMENT_FAILED: "Placement SL échoué — close market déclenché",
}


# Cooldowns par type d'anomalie (secondes) — seul l'envoi Telegram est throttlé
_ANOMALY_COOLDOWNS: dict[AnomalyType, int] = {
    AnomalyType.SL_PLACEMENT_FAILED: 300,       # 5 min — critique, garder court
    AnomalyType.WS_DISCONNECTED: 1800,           # 30 min
    AnomalyType.DATA_STALE: 1800,                # 30 min
    AnomalyType.EXECUTOR_DISCONNECTED: 1800,     # 30 min
    AnomalyType.ALL_STRATEGIES_STOPPED: 3600,    # 1h — état persistant
    AnomalyType.KILL_SWITCH_GLOBAL: 3600,        # 1h
    AnomalyType.KILL_SWITCH_LIVE: 3600,          # 1h
}
_DEFAULT_COOLDOWN = 600  # 10 min pour tout type non listé


class Notifier:
    """Centralise les notifications vers Telegram (+ futurs canaux).

    Si telegram est None, les notifications sont loguées mais pas envoyées.
    """

    def __init__(self, telegram: TelegramClient | None = None) -> None:
        self._telegram = telegram
        self._last_anomaly_sent: dict[AnomalyType, float] = {}

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
        """Notifie une anomalie détectée par le Watchdog.

        Le log WARNING est systématique. L'envoi Telegram est throttlé
        par un cooldown par type d'anomalie pour éviter le spam.
        """
        message = _ANOMALY_MESSAGES.get(anomaly_type, str(anomaly_type))
        if details:
            message = f"{message} — {details}"

        logger.warning("Notifier: anomalie {}: {}", anomaly_type.value, message)
        if self._telegram:
            now = time.monotonic()
            cooldown = _ANOMALY_COOLDOWNS.get(anomaly_type, _DEFAULT_COOLDOWN)
            last_sent = self._last_anomaly_sent.get(anomaly_type, 0)

            if now - last_sent >= cooldown:
                text = f"<b>Anomalie</b>\n{message}"
                await self._telegram.send_message(text)
                self._last_anomaly_sent[anomaly_type] = now

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

    # ─── Sprint 5a : alertes ordres live ───────────────────────────────

    async def notify_live_order_opened(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        strategy: str,
        order_id: str,
    ) -> None:
        """Notifie un ordre live ouvert."""
        logger.info(
            "Notifier: LIVE ORDER {} {} {} @ {:.2f} (SL={:.2f}, TP={:.2f})",
            direction, quantity, symbol, entry_price, sl_price, tp_price,
        )
        if self._telegram:
            await self._telegram.send_live_order_opened(
                symbol, direction, quantity, entry_price,
                sl_price, tp_price, strategy, order_id,
            )

    async def notify_live_order_closed(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        net_pnl: float,
        exit_reason: str,
        strategy: str,
    ) -> None:
        """Notifie un ordre live fermé."""
        logger.info(
            "Notifier: LIVE CLOSE {} {} {:.2f} → {:.2f} net={:+.2f} ({})",
            direction, symbol, entry_price, exit_price, net_pnl, exit_reason,
        )
        if self._telegram:
            await self._telegram.send_live_order_closed(
                symbol, direction, entry_price, exit_price,
                net_pnl, exit_reason, strategy,
            )

    async def notify_live_sl_failed(self, symbol: str, strategy: str) -> None:
        """Notifie un échec critique de placement SL."""
        logger.critical(
            "Notifier: SL ÉCHOUÉ {} ({}) — close market déclenché",
            symbol, strategy,
        )
        if self._telegram:
            await self._telegram.send_live_sl_failed(symbol, strategy)

    async def notify_grid_level_opened(
        self,
        symbol: str,
        direction: str,
        level_num: int,
        quantity: float,
        entry_price: float,
        avg_price: float,
        sl_price: float,
        strategy: str,
    ) -> None:
        """Notifie l'ouverture d'un niveau grid."""
        logger.info(
            "Notifier: GRID ENTRY #{} {} {} qty={:.6f} @ {:.2f} (avg={:.2f}, SL={:.2f})",
            level_num, direction, symbol, quantity, entry_price, avg_price, sl_price,
        )
        if self._telegram:
            await self._telegram.send_grid_level_opened(
                symbol, direction, level_num, quantity, entry_price,
                avg_price, sl_price, strategy,
            )

    async def notify_grid_cycle_closed(
        self,
        symbol: str,
        direction: str,
        num_positions: int,
        avg_entry: float,
        exit_price: float,
        net_pnl: float,
        exit_reason: str,
        strategy: str,
    ) -> None:
        """Notifie la fermeture d'un cycle grid."""
        logger.info(
            "Notifier: GRID CLOSE {} {} — {} niveaux, net={:+.2f} ({})",
            direction, symbol, num_positions, net_pnl, exit_reason,
        )
        if self._telegram:
            await self._telegram.send_grid_cycle_closed(
                symbol, direction, num_positions, avg_entry, exit_price,
                net_pnl, exit_reason, strategy,
            )

    async def notify_reconciliation(self, result: str) -> None:
        """Notifie le résultat de la réconciliation au boot."""
        logger.info("Notifier: réconciliation: {}", result)
        if self._telegram:
            await self._telegram.send_message(
                f"<b>Réconciliation</b>\n{result}"
            )
