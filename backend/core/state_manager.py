"""StateManager : sauvegarde et restauration de l'état du Simulator.

Gère la persistance de l'état des runners (capital, stats, positions)
dans un fichier JSON pour permettre le crash recovery.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from backend.core.database import Database

if TYPE_CHECKING:
    from backend.backtesting.simulator import LiveStrategyRunner, Simulator


class StateManager:
    """Sauvegarde et restauration de l'état du Simulator.

    - Écriture atomique (tmp + os.replace)
    - Lecture robuste (fichier absent/corrompu → None)
    - Sauvegarde périodique via boucle asyncio
    """

    def __init__(
        self,
        db: Database,
        state_file: str = "data/simulator_state.json",
        executor_state_file: str = "data/executor_state.json",
    ) -> None:
        self._db = db
        self._state_file = state_file
        self._executor_state_file = executor_state_file
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def save_runner_state(self, runners: list[LiveStrategyRunner]) -> None:
        """Sérialise l'état de tous les runners dans un fichier JSON.

        Écriture atomique : écrit dans un .tmp puis os.replace().
        """
        state: dict[str, Any] = {
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
            "runners": {},
        }

        for runner in runners:
            position_data = None
            if runner._position is not None:
                pos = runner._position
                position_data = {
                    "direction": pos.direction.value,
                    "entry_price": pos.entry_price,
                    "quantity": pos.quantity,
                    "entry_time": pos.entry_time.isoformat(),
                    "tp_price": pos.tp_price,
                    "sl_price": pos.sl_price,
                    "entry_fee": pos.entry_fee,
                }

            state["runners"][runner.name] = {
                "capital": runner._capital,
                "net_pnl": runner._stats.net_pnl,
                "total_trades": runner._stats.total_trades,
                "wins": runner._stats.wins,
                "losses": runner._stats.losses,
                "kill_switch": runner._kill_switch_triggered,
                "is_active": runner._stats.is_active,
                "position": position_data,
                "position_symbol": runner._position_symbol,
            }

        # Écriture atomique
        Path(self._state_file).parent.mkdir(parents=True, exist_ok=True)
        tmp_file = self._state_file + ".tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            os.replace(tmp_file, self._state_file)
            logger.debug(
                "StateManager: état sauvegardé ({} runners)",
                len(state["runners"]),
            )
        except OSError as e:
            logger.error("StateManager: erreur sauvegarde: {}", e)
            # Nettoyer le tmp si échec
            try:
                os.unlink(tmp_file)
            except OSError:
                pass

    async def load_runner_state(self) -> dict[str, Any] | None:
        """Charge l'état sauvegardé depuis le fichier JSON.

        Retourne None si le fichier est absent, vide ou corrompu.
        """
        if not Path(self._state_file).exists():
            logger.info("StateManager: pas de fichier d'état, démarrage fresh")
            return None

        try:
            with open(self._state_file, encoding="utf-8") as f:
                data = json.load(f)

            # Validation minimale
            if not isinstance(data, dict) or "runners" not in data:
                logger.warning(
                    "StateManager: fichier d'état invalide (clé 'runners' absente), démarrage fresh"
                )
                return None

            if not isinstance(data["runners"], dict) or not data["runners"]:
                logger.warning(
                    "StateManager: fichier d'état vide (aucun runner), démarrage fresh"
                )
                return None

            saved_at = data.get("saved_at", "inconnu")
            logger.info(
                "StateManager: état chargé ({} runners, sauvegardé à {})",
                len(data["runners"]),
                saved_at,
            )
            return data

        except json.JSONDecodeError as e:
            logger.warning(
                "StateManager: fichier d'état corrompu ({}), démarrage fresh", e
            )
            return None
        except OSError as e:
            logger.warning(
                "StateManager: erreur lecture fichier d'état ({}), démarrage fresh", e
            )
            return None

    async def start_periodic_save(
        self,
        simulator: Simulator,
        interval: int = 60,
    ) -> None:
        """Lance la boucle de sauvegarde périodique."""
        self._running = True
        self._task = asyncio.create_task(
            self._periodic_save_loop(simulator, interval)
        )
        logger.info("StateManager: sauvegarde périodique activée ({}s)", interval)

    async def _periodic_save_loop(
        self,
        simulator: Simulator,
        interval: int,
    ) -> None:
        """Boucle de sauvegarde périodique."""
        while self._running:
            try:
                await asyncio.sleep(interval)
                if self._running and simulator.runners:
                    await self.save_runner_state(simulator.runners)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("StateManager: erreur sauvegarde périodique: {}", e)

    # ─── Sprint 5a : état Executor ────────────────────────────────────

    async def save_executor_state(self, executor: Any, risk_manager: Any) -> None:
        """Sauvegarde l'état de l'Executor et du RiskManager."""
        state: dict[str, Any] = {
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
            "executor": executor.get_state_for_persistence(),
        }

        Path(self._executor_state_file).parent.mkdir(parents=True, exist_ok=True)
        tmp_file = self._executor_state_file + ".tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            os.replace(tmp_file, self._executor_state_file)
            logger.debug("StateManager: état executor sauvegardé")
        except OSError as e:
            logger.error("StateManager: erreur sauvegarde executor: {}", e)
            try:
                os.unlink(tmp_file)
            except OSError:
                pass

    async def load_executor_state(self) -> dict[str, Any] | None:
        """Charge l'état de l'Executor sauvegardé."""
        if not Path(self._executor_state_file).exists():
            logger.info("StateManager: pas de fichier état executor, démarrage fresh")
            return None

        try:
            with open(self._executor_state_file, encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict) or "executor" not in data:
                logger.warning(
                    "StateManager: fichier état executor invalide, démarrage fresh"
                )
                return None

            saved_at = data.get("saved_at", "inconnu")
            logger.info(
                "StateManager: état executor chargé (sauvegardé à {})", saved_at,
            )
            return data.get("executor")

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "StateManager: erreur lecture état executor ({}), démarrage fresh", e,
            )
            return None

    async def stop(self) -> None:
        """Arrête la boucle de sauvegarde."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("StateManager: arrêté")
