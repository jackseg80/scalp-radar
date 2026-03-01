"""Heartbeat Telegram à intervalle configurable.

Envoie un message périodique avec le statut du Simulator.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from backend.alerts.telegram import TelegramClient
    from backend.backtesting.simulator import Simulator


class Heartbeat:
    """Heartbeat Telegram à intervalle configurable."""

    def __init__(
        self,
        telegram: TelegramClient,
        simulator: Simulator,
        interval_seconds: int = 3600,
    ) -> None:
        self._telegram = telegram
        self._simulator = simulator
        self._interval = interval_seconds
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Lance la boucle de heartbeat."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Heartbeat: activé (intervalle {}s)", self._interval)

    async def _loop(self) -> None:
        """Boucle de heartbeat."""
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                if self._running:
                    message = self._build_message()
                    await self._telegram.send_message(message, alert_type="heartbeat")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat: erreur: {}", e)

    def _build_message(self) -> str:
        """Construit le message heartbeat."""
        status = self._simulator.get_all_status()
        total_pnl = 0.0
        total_trades = 0
        total_wins = 0
        active_strategies: list[str] = []

        for name, s in status.items():
            total_pnl += s.get("net_pnl", 0)
            total_trades += s.get("total_trades", 0)
            total_wins += s.get("wins", 0)
            if s.get("is_active", False):
                active_strategies.append(name)

        pnl_sign = "+" if total_pnl >= 0 else ""
        win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0

        strats = ", ".join(active_strategies) if active_strategies else "aucune"
        return (
            f"<b>Heartbeat Scalp Radar</b>\n"
            f"PnL session: <b>{pnl_sign}{total_pnl:.2f}$</b>\n"
            f"Trades: {total_trades} (win rate: {win_rate:.0f}%)\n"
            f"Stratégies actives: {strats}"
        )

    async def stop(self) -> None:
        """Arrête le heartbeat."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Heartbeat: arrêté")
