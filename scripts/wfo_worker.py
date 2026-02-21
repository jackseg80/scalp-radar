"""Worker subprocess pour les jobs WFO — exécution isolée (segfault-safe).

Lancé par backend/optimization/job_manager.py._run_job_subprocess().
Les progress updates sont envoyés sur stdout en JSON (une ligne par message).

Isolation : si numpy/numba provoque un segfault, seul ce subprocess meurt.
Le serveur FastAPI (processus parent) survit.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

# Désactiver JIT Python 3.13 (segfaults numpy) — doit être AVANT tout import
os.environ.setdefault("PYTHON_JIT", "0")

# Ajouter la racine du projet au path (pour les imports backend.*)
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))


def _read_job(db_path: str, job_id: str) -> dict[str, Any] | None:
    """Lit un job depuis la DB SQLite (sync)."""
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT strategy_name, asset, timeframe, params_override "
            "FROM optimization_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return None

    strategy_name, asset, timeframe, params_override_json = row
    params_override = json.loads(params_override_json) if params_override_json else None

    # Normaliser params_override : scalaires → listes (attendu par run_optimization)
    if params_override:
        params_override = {
            k: (v if isinstance(v, list) else [v])
            for k, v in params_override.items()
        }

    return {
        "strategy_name": strategy_name,
        "asset": asset,
        "timeframe": timeframe,
        "params_override": params_override,
    }


def _emit(data: dict) -> None:
    """Émet un message JSON sur stdout (flush immédiat pour streaming)."""
    print(json.dumps(data), flush=True)


async def _run(job_id: str, db_path: str, config_dir: str) -> int:
    """Exécute le WFO et envoie les progress sur stdout en JSON."""
    job = _read_job(db_path, job_id)
    if not job:
        _emit({"type": "error", "message": f"Job {job_id} introuvable en DB"})
        return 1

    # Progress callback → stdout JSON (lu par le processus parent)
    def progress_callback(pct: float, phase: str) -> None:
        _emit({"type": "progress", "pct": round(float(pct), 1), "phase": phase})

    from scripts.optimize import run_optimization  # noqa: PLC0415

    try:
        _report, result_id = await run_optimization(
            strategy_name=job["strategy_name"],
            symbol=job["asset"],
            config_dir=config_dir,
            progress_callback=progress_callback,
            params_override=job["params_override"],
        )
        _emit({"type": "done", "result_id": result_id})
        return 0

    except (asyncio.CancelledError, KeyboardInterrupt):
        _emit({"type": "error", "message": "Annulé"})
        return 2

    except Exception as exc:
        _emit({"type": "error", "message": str(exc)})
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WFO worker subprocess")
    parser.add_argument("--job-id", required=True, help="UUID du job à exécuter")
    parser.add_argument("--db-path", required=True, help="Chemin vers la DB SQLite")
    parser.add_argument("--config-dir", default="config", help="Dossier de config")
    args = parser.parse_args()

    rc = asyncio.run(_run(args.job_id, args.db_path, args.config_dir))
    sys.exit(rc)
