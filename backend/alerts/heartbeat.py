"""Heartbeat Telegram à intervalle configurable.

Envoie un message périodique avec le statut du Simulator.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from backend.alerts.telegram import TelegramClient
    from backend.backtesting.simulator import Simulator
    from backend.execution.executor_manager import ExecutorManager


class Heartbeat:
    """Heartbeat Telegram à intervalle configurable."""

    def __init__(
        self,
        telegram: TelegramClient,
        simulator: Simulator,
        interval_seconds: int = 3600,
        executor_mgr: ExecutorManager | None = None,
    ) -> None:
        self._telegram = telegram
        self._simulator = simulator
        self._executor_mgr = executor_mgr
        self._interval = interval_seconds
        self._task: asyncio.Task[None] | None = None
        self._running = False
        
        # Tracking pour les trades incrémentaux (si besoin futur, 
        # mais ici on affiche le cumul session comme demandé)
        self._last_trade_count = 0

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
        """Construit le message heartbeat (données Live prioritaires)."""
        config = self._simulator.config
        is_live_trading = config.secrets.live_trading

        # 1. Collecte données LIVE (Executor)
        live_pnl = 0.0
        latent_pnl = 0.0
        live_trades = 0
        open_positions_details: list[str] = []
        active_live_strats: set[str] = set()

        if self._executor_mgr and is_live_trading:
            for strat_name, ex in self._executor_mgr.executors.items():
                if not ex.is_enabled:
                    continue
                
                status = ex.get_status()
                # PnL réalisé session (depuis RiskManager)
                rm_status = status.get("risk_manager", {})
                live_pnl += rm_status.get("session_pnl", 0.0)
                
                # Trades session (nombre de clôtures enregistrées)
                # On utilise le nombre de résultats de trades dans le RiskManager
                rm = self._executor_mgr.risk_managers.get(strat_name)
                if rm:
                    live_trades += len(getattr(rm, "_trade_history", []))

                # Positions ouvertes et PnL latent
                positions = status.get("positions", [])
                for pos in positions:
                    latent_pnl += pos.get("unrealized_pnl") or 0.0
                    sym_short = pos["symbol"].split(":")[0].split("/")[0]
                    open_positions_details.append(f"{sym_short} {pos['direction']}")
                
                if ex.is_connected:
                    active_live_strats.add(strat_name)

        # 2. Fallback / Complément SIMULATOR (si pas de live ou pour le win rate)
        sim_status = self._simulator.get_all_status()
        active_sim_strats: list[str] = []
        sim_trades = 0
        sim_wins = 0

        for name, s in sim_status.items():
            if is_live_trading:
                strat_cfg = getattr(config.strategies, name, None)
                if not strat_cfg or not getattr(strat_cfg, "live_eligible", False):
                    continue
            
            sim_trades += s.get("total_trades", 0)
            sim_wins += s.get("wins", 0)
            if s.get("is_active", False):
                active_sim_strats.append(name)

        # 3. Formatage du message
        pnl_sign = "+" if live_pnl >= 0 else ""
        latent_sign = "+" if latent_pnl >= 0 else ""
        
        pos_count = len(open_positions_details)
        pos_text = f"{pos_count} ouverte" + ("s" if pos_count > 1 else "")
        if open_positions_details:
            pos_text += f" ({', '.join(open_positions_details)})"

        strats_list = sorted(list(active_live_strats if active_live_strats else active_sim_strats))
        strats_text = ", ".join(strats_list) if strats_list else "aucune"

        return (
            f"<b>Heartbeat Scalp Radar</b>\n"
            f"PnL session: <b>{pnl_sign}{live_pnl:.2f}$</b> | Latent: <b>{latent_sign}{latent_pnl:.2f}$</b>\n"
            f"Positions: {pos_text}\n"
            f"Trades session: {live_trades if is_live_trading else sim_trades} | Stratégies: {strats_text}"
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
