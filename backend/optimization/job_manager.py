"""Gestionnaire de jobs WFO — Sprint 14.

File d'attente FIFO, exécution séquentielle en background,
callbacks de progression via WebSocket.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

import aiosqlite
from loguru import logger
from pathlib import Path

from backend.optimization import STRATEGY_REGISTRY

# Limite globale : max pending jobs dans la queue
MAX_PENDING_JOBS = 5

# Max jobs WFO en parallèle (chacun dans son propre subprocess)
MAX_CONCURRENT_JOBS = 2

# Timeout global par job (secondes) — protège le worker loop si un WFO bloque
JOB_TIMEOUT_SECONDS = 3600  # 1h max par job


@dataclass
class OptimizationJob:
    """Représentation en mémoire d'un job d'optimisation WFO."""

    id: str
    strategy_name: str
    asset: str
    timeframe: str
    status: str  # pending | running | completed | failed | cancelled
    progress_pct: float
    current_phase: str
    params_override: dict | None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    result_id: int | None = None
    error_message: str | None = None


class JobManager:
    """Gestionnaire de jobs WFO avec file d'attente FIFO.

    - Un seul job running à la fois (CPU-bound)
    - Max 5 pending jobs
    - Progress callback → update DB (sync) + broadcast WS (async via main loop)
    - Annulation via threading.Event
    """

    def __init__(
        self,
        db_path: str,
        ws_broadcast: Callable[[dict], Coroutine] | None = None,
    ) -> None:
        self._db_path = db_path
        self._ws_broadcast = ws_broadcast
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._job_tasks: set[asyncio.Task] = set()  # tasks running jobs
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
        self._cancel_events: dict[str, threading.Event] = {}
        self._running_procs: dict[str, Any] = {}  # subprocess.Popen instances
        self._running = False
        self._main_loop: asyncio.AbstractEventLoop | None = None

    # ─── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Démarre le worker loop et récupère les jobs orphelins."""
        self._main_loop = asyncio.get_event_loop()
        self._running = True
        await self._recover_orphaned_jobs()
        await self._enqueue_pending_jobs()
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("JobManager démarré")

    async def stop(self) -> None:
        """Arrête le worker loop. Les jobs running passent en failed."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        # Annuler tous les jobs en cours
        for event in self._cancel_events.values():
            event.set()
        self._cancel_events.clear()
        # Tuer les subprocesses WFO encore actifs
        for proc in list(self._running_procs.values()):
            try:
                proc.terminate()
            except Exception:
                pass
        self._running_procs.clear()
        logger.info("JobManager arrêté")

    # ─── Public API ────────────────────────────────────────────────────────

    async def submit_job(
        self,
        strategy_name: str,
        asset: str,
        params_override: dict | None = None,
    ) -> str:
        """Crée un job pending et l'ajoute à la queue.

        Raises:
            ValueError: stratégie inconnue
            RuntimeError: doublon (strategy, asset) pending/running
            RuntimeError: queue pleine (> MAX_PENDING_JOBS)
        """
        # Validation stratégie
        if strategy_name not in STRATEGY_REGISTRY:
            raise ValueError(
                f"Stratégie '{strategy_name}' non optimisable. "
                f"Disponibles : {list(STRATEGY_REGISTRY.keys())}"
            )

        # Résoudre le timeframe depuis le registre
        config_cls, _ = STRATEGY_REGISTRY[strategy_name]
        timeframe = config_cls().timeframe

        # Anti-doublon (strategy, asset)
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                """SELECT COUNT(*) as cnt FROM optimization_jobs
                   WHERE strategy_name = ? AND asset = ?
                   AND status IN ('pending', 'running')""",
                (strategy_name, asset),
            )
            row = await cursor.fetchone()
            if row["cnt"] > 0:
                raise RuntimeError(
                    f"Un job {strategy_name} × {asset} est déjà en cours ou en attente"
                )

            # Limite de queue
            cursor = await conn.execute(
                "SELECT COUNT(*) as cnt FROM optimization_jobs WHERE status = 'pending'"
            )
            row = await cursor.fetchone()
            if row["cnt"] >= MAX_PENDING_JOBS:
                raise RuntimeError(
                    f"Queue pleine ({MAX_PENDING_JOBS} jobs en attente). "
                    "Attendez qu'un job se termine."
                )

        # Créer le job
        job_id = str(uuid.uuid4())
        now = datetime.now(tz=timezone.utc)
        params_json = json.dumps(params_override) if params_override else None

        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute(
                """INSERT INTO optimization_jobs
                   (id, strategy_name, asset, timeframe, status,
                    progress_pct, current_phase, params_override, created_at)
                   VALUES (?, ?, ?, ?, 'pending', 0, '', ?, ?)""",
                (job_id, strategy_name, asset, timeframe, params_json,
                 now.isoformat()),
            )
            await conn.commit()

        # Ajouter à la queue interne
        await self._queue.put(job_id)
        logger.info("Job soumis : {} ({} × {})", job_id[:8], strategy_name, asset)
        return job_id

    async def cancel_job(self, job_id: str) -> bool:
        """Annule un job pending ou running.

        Returns:
            True si annulé, False si le job n'est pas annulable.
        """
        job = await self.get_job(job_id)
        if job is None:
            return False

        if job.status == "pending":
            await self._update_job_status(job_id, "cancelled")
            logger.info("Job {} annulé (pending)", job_id[:8])
            return True

        if job.status == "running":
            # Signaler l'annulation et tuer le subprocess WFO
            event = self._cancel_events.get(job_id)
            if event:
                event.set()
            proc = self._running_procs.get(job_id)
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
            if event or proc:
                logger.info("Job {} : annulation demandée (running)", job_id[:8])
                return True

        return False

    async def get_job(self, job_id: str) -> OptimizationJob | None:
        """Retourne un job par ID."""
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM optimization_jobs WHERE id = ?", (job_id,)
            )
            row = await cursor.fetchone()
            if not row:
                return None
            return self._row_to_job(dict(row))

    async def list_jobs(
        self, status: str | None = None, limit: int = 50
    ) -> list[OptimizationJob]:
        """Liste les jobs, optionnellement filtrés par status."""
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = aiosqlite.Row
            if status:
                cursor = await conn.execute(
                    """SELECT * FROM optimization_jobs
                       WHERE status = ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (status, limit),
                )
            else:
                cursor = await conn.execute(
                    """SELECT * FROM optimization_jobs
                       ORDER BY created_at DESC LIMIT ?""",
                    (limit,),
                )
            rows = await cursor.fetchall()
            return [self._row_to_job(dict(r)) for r in rows]

    async def pending_count(self) -> int:
        """Nombre de jobs en attente."""
        async with aiosqlite.connect(self._db_path) as conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) as cnt FROM optimization_jobs WHERE status = 'pending'"
            )
            row = await cursor.fetchone()
            return row[0]

    # ─── Worker loop (Bloc C — stub pour l'instant) ───────────────────────

    async def _worker_loop(self) -> None:
        """Boucle FIFO : dépile les jobs et les lance en parallèle.

        Max MAX_CONCURRENT_JOBS simultanés via un sémaphore.
        Chaque job tourne dans son propre subprocess isolé.
        """
        logger.info("Worker loop démarré (max {} jobs parallèles)", MAX_CONCURRENT_JOBS)
        try:
            while self._running:
                try:
                    job_id = await asyncio.wait_for(
                        self._queue.get(), timeout=2.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Vérifier que le job est toujours pending (pas annulé entre-temps)
                job = await self.get_job(job_id)
                if job is None or job.status != "pending":
                    logger.info(
                        "Job {} ignoré (status={})",
                        job_id[:8], job.status if job else "supprimé",
                    )
                    continue

                # Lancer le job dans une task parallèle (limité par sémaphore)
                task = asyncio.create_task(self._run_job_limited(job))
                self._job_tasks.add(task)
                task.add_done_callback(self._job_tasks.discard)
        except asyncio.CancelledError:
            # Attendre que les jobs en cours se terminent proprement
            if self._job_tasks:
                logger.info("Worker loop arrêté, {} job(s) en cours d'annulation", len(self._job_tasks))
                for t in self._job_tasks:
                    t.cancel()
                await asyncio.gather(*self._job_tasks, return_exceptions=True)
            logger.info("Worker loop arrêté")

    async def _run_job_limited(self, job: OptimizationJob) -> None:
        """Wrapper qui acquiert le sémaphore avant de lancer le job."""
        async with self._semaphore:
            await self._run_job(job)

    async def _run_job(self, job: OptimizationJob) -> None:
        """Exécute un job WFO complet.

        Implémentation complète dans le Bloc C.
        Pour l'instant : passe en running puis completed immédiatement (stub).
        """
        # Re-vérifier le status (race entre queue.get() et cancel_job)
        fresh = await self.get_job(job.id)
        if fresh is None or fresh.status != "pending":
            logger.info(
                "Job {} ignoré dans _run_job (status={})",
                job.id[:8], fresh.status if fresh else "supprimé",
            )
            return

        t_start = time.monotonic()
        cancel_event = threading.Event()
        self._cancel_events[job.id] = cancel_event

        try:
            # Passer en running
            now = datetime.now(tz=timezone.utc)
            await self._update_job_fields(job.id, {
                "status": "running",
                "started_at": now.isoformat(),
            })
            await self._broadcast_progress(job, "running", 0, "Démarrage...")

            # Lancer le WFO dans un subprocess isolé (segfault-safe)
            # Si numpy/numba crashe, seul le subprocess meurt — le serveur survit
            try:
                result_id = await asyncio.wait_for(
                    self._run_job_subprocess(job, cancel_event),
                    timeout=JOB_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                # Tuer le subprocess s'il tourne encore
                proc = self._running_procs.get(job.id)
                if proc and proc.returncode is None:
                    proc.terminate()
                raise TimeoutError(
                    f"Job timeout après {JOB_TIMEOUT_SECONDS}s — "
                    "le WFO a été interrompu"
                )

            # Marquer le job completed
            duration = time.monotonic() - t_start
            now = datetime.now(tz=timezone.utc)
            await self._update_job_fields(job.id, {
                "status": "completed",
                "completed_at": now.isoformat(),
                "duration_seconds": round(duration, 1),
                "progress_pct": 100,
                "current_phase": "Terminé",
                "result_id": result_id,
            })
            await self._broadcast_progress(job, "completed", 100, "Terminé")
            logger.info("Job {} terminé en {:.1f}s", job.id[:8], duration)

        except asyncio.CancelledError:
            duration = time.monotonic() - t_start
            now = datetime.now(tz=timezone.utc)
            await self._update_job_fields(job.id, {
                "status": "cancelled",
                "completed_at": now.isoformat(),
                "duration_seconds": round(duration, 1),
                "error_message": "Annulé par l'utilisateur",
            })
            await self._broadcast_progress(job, "cancelled", 0, "Annulé")

        except Exception as exc:
            duration = time.monotonic() - t_start
            now = datetime.now(tz=timezone.utc)
            error_msg = str(exc)[:500]
            await self._update_job_fields(job.id, {
                "status": "failed",
                "completed_at": now.isoformat(),
                "duration_seconds": round(duration, 1),
                "error_message": error_msg,
            })
            await self._broadcast_progress(job, "failed", 0, f"Erreur : {error_msg}")
            logger.error("Job {} échoué : {}", job.id[:8], exc)

        finally:
            self._cancel_events.pop(job.id, None)

    def _run_job_wfo_thread(
        self, job: OptimizationJob, cancel_event: threading.Event
    ) -> int | None:
        """Exécute le WFO complet dans un thread dédié avec son propre event loop.

        Returns:
            result_id (int) si succès, None si échec
        """
        # Progress callback thread-safe : sync DB + async WS via main loop
        def progress_callback(pct: float, phase: str) -> None:
            # Update DB (sync sqlite3, pas aiosqlite)
            self._update_job_progress_sync(job.id, pct, phase)

            # Broadcast WS via l'event loop principal (thread-safe)
            if self._main_loop and self._ws_broadcast:
                asyncio.run_coroutine_threadsafe(
                    self._ws_broadcast({
                        "type": "optimization_progress",
                        "job_id": job.id,
                        "status": "running",
                        "progress_pct": pct,
                        "current_phase": phase,
                        "strategy_name": job.strategy_name,
                        "asset": job.asset,
                    }),
                    self._main_loop,
                )

        # Lancer run_optimization dans un event loop dédié à ce thread
        from scripts.optimize import run_optimization

        # Normaliser params_override : convertir valeurs scalaires en listes
        normalized_params = None
        if job.params_override:
            normalized_params = {}
            for key, value in job.params_override.items():
                if isinstance(value, list):
                    normalized_params[key] = value
                else:
                    normalized_params[key] = [value]

        try:
            report, result_id = asyncio.run(run_optimization(
                strategy_name=job.strategy_name,
                symbol=job.asset,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
                params_override=normalized_params,
            ))
            # result_id retourné directement par save_report() → plus fiable que re-query
            return result_id

        except asyncio.CancelledError:
            # Propagé depuis walk_forward.optimize()
            raise
        except Exception as exc:
            logger.error("WFO thread échoué : {}", exc)
            raise

    async def _run_job_subprocess(
        self, job: OptimizationJob, cancel_event: threading.Event
    ) -> int | None:
        """Lance le WFO dans un subprocess isolé (segfault-safe).

        Utilise subprocess.Popen (compatible SelectorEventLoop Windows)
        au lieu de asyncio.create_subprocess_exec (nécessite ProactorEventLoop).

        Si numpy/numba provoque un segfault, seul ce subprocess meurt.
        Le serveur FastAPI (processus parent) survit et marque le job failed.

        Returns:
            result_id (int) si succès, None si échec
        """
        import subprocess
        import sys

        worker_script = Path(__file__).parent.parent.parent / "scripts" / "wfo_worker.py"

        proc = subprocess.Popen(
            [sys.executable, str(worker_script),
             "--job-id", job.id,
             "--db-path", str(self._db_path)],
            stdout=subprocess.PIPE,
            stderr=None,  # Hérite du terminal parent → logs visibles
        )
        self._running_procs[job.id] = proc
        logger.debug("WFO subprocess lancé (PID {})", proc.pid)

        def _monitor_stdout() -> tuple[int | None, str | None, bool]:
            """Lit stdout, met à jour DB + WS en temps réel depuis le thread.

            Utilise readline() au lieu de l'itérateur fichier pour éviter
            le buffering interne de Python (8KB) qui retarde les updates.
            """
            _result_id: int | None = None
            _error_msg: str | None = None
            _cancelled = False
            assert proc.stdout is not None
            while True:
                raw_line = proc.stdout.readline()
                if not raw_line:
                    break  # EOF = subprocess terminé
                line = raw_line.decode(errors="replace").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")
                if msg_type == "progress":
                    pct = float(data.get("pct", 0))
                    phase = str(data.get("phase", ""))
                    # Update DB (sync, OK dans ce thread)
                    self._update_job_progress_sync(job.id, pct, phase)
                    # Broadcast WS via main event loop (thread-safe)
                    if self._main_loop and self._ws_broadcast:
                        asyncio.run_coroutine_threadsafe(
                            self._broadcast_progress(job, "running", pct, phase),
                            self._main_loop,
                        )
                elif msg_type == "done":
                    _result_id = data.get("result_id")
                elif msg_type == "error":
                    _error_msg = data.get("message", "Erreur inconnue")

                # Check annulation entre chaque ligne
                if cancel_event.is_set():
                    proc.terminate()
                    _cancelled = True
                    break

            return _result_id, _error_msg, _cancelled

        try:
            # Monitoring stdout dans un thread (blocking I/O)
            result_id, error_msg, cancelled = await asyncio.to_thread(_monitor_stdout)

            if cancelled:
                raise asyncio.CancelledError("Annulé par l'utilisateur")

        finally:
            self._running_procs.pop(job.id, None)
            # Attendre que le subprocess se termine (max 15s)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.warning("WFO subprocess (PID {}) ne se termine pas, kill forcé", proc.pid)
                proc.kill()
                proc.wait()

        returncode = proc.returncode
        if returncode not in (0, 2) and result_id is None:
            msg = error_msg or f"WFO subprocess crash (exit {returncode})"
            raise RuntimeError(msg)

        return result_id

    # ─── DB helpers ────────────────────────────────────────────────────────

    async def _update_job_status(self, job_id: str, status: str) -> None:
        """Met à jour le status d'un job."""
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute(
                "UPDATE optimization_jobs SET status = ? WHERE id = ?",
                (status, job_id),
            )
            await conn.commit()

    async def _update_job_fields(self, job_id: str, fields: dict[str, Any]) -> None:
        """Met à jour plusieurs champs d'un job."""
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [job_id]
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute(
                f"UPDATE optimization_jobs SET {set_clause} WHERE id = ?",
                values,
            )
            await conn.commit()

    def _update_job_progress_sync(
        self, job_id: str, progress_pct: float, current_phase: str
    ) -> None:
        """Met à jour la progression d'un job (sync, pour le thread WFO)."""
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                """UPDATE optimization_jobs
                   SET progress_pct = ?, current_phase = ?
                   WHERE id = ?""",
                (progress_pct, current_phase, job_id),
            )
            conn.commit()
        finally:
            conn.close()

    async def _recover_orphaned_jobs(self) -> None:
        """Au boot, passe les jobs 'running' en 'failed' (serveur redémarré)."""
        async with aiosqlite.connect(self._db_path) as conn:
            cursor = await conn.execute(
                """UPDATE optimization_jobs SET status = 'failed',
                   error_message = 'Serveur redémarré pendant l''exécution'
                   WHERE status = 'running'"""
            )
            if cursor.rowcount > 0:
                logger.warning(
                    "{} job(s) orphelin(s) passé(s) en failed (redémarrage)",
                    cursor.rowcount,
                )
            await conn.commit()

    async def _enqueue_pending_jobs(self) -> None:
        """Au boot, réenqueue les jobs 'pending' restants."""
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                """SELECT id FROM optimization_jobs
                   WHERE status = 'pending'
                   ORDER BY created_at ASC"""
            )
            rows = await cursor.fetchall()
            for row in rows:
                await self._queue.put(row["id"])
            if rows:
                logger.info("{} job(s) pending réenqueueé(s)", len(rows))

    # ─── WS broadcast helper ──────────────────────────────────────────────

    async def _broadcast_progress(
        self,
        job: OptimizationJob,
        status: str,
        progress_pct: float,
        current_phase: str,
    ) -> None:
        """Broadcast une mise à jour de progression via WebSocket."""
        if self._ws_broadcast is None:
            return
        try:
            await self._ws_broadcast({
                "type": "optimization_progress",
                "job_id": job.id,
                "status": status,
                "progress_pct": progress_pct,
                "current_phase": current_phase,
                "strategy_name": job.strategy_name,
                "asset": job.asset,
            })
        except Exception as exc:
            logger.warning("Broadcast WS échoué : {}", exc)

    # ─── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_job(row: dict) -> OptimizationJob:
        """Convertit une row DB en OptimizationJob."""
        params_override = None
        if row.get("params_override"):
            try:
                params_override = json.loads(row["params_override"])
            except (json.JSONDecodeError, TypeError):
                pass

        return OptimizationJob(
            id=row["id"],
            strategy_name=row["strategy_name"],
            asset=row["asset"],
            timeframe=row["timeframe"],
            status=row["status"],
            progress_pct=row.get("progress_pct", 0) or 0,
            current_phase=row.get("current_phase", "") or "",
            params_override=params_override,
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=(
                datetime.fromisoformat(row["started_at"])
                if row.get("started_at")
                else None
            ),
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row.get("completed_at")
                else None
            ),
            duration_seconds=row.get("duration_seconds"),
            result_id=row.get("result_id"),
            error_message=row.get("error_message"),
        )
