"""Tests d'intégration Bloc C — Worker loop réel avec WFO."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest
import pytest_asyncio

from backend.optimization.job_manager import JobManager


def _create_tables(db_path: str) -> None:
    """Crée les tables nécessaires dans une DB temporaire."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS optimization_jobs (
            id TEXT PRIMARY KEY,
            strategy_name TEXT NOT NULL,
            asset TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            progress_pct REAL DEFAULT 0,
            current_phase TEXT DEFAULT '',
            params_override TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            duration_seconds REAL,
            result_id INTEGER,
            error_message TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON optimization_jobs(status);
    """)
    conn.close()


@pytest.fixture
def temp_db(tmp_path):
    """DB temporaire avec la table optimization_jobs."""
    db_path = str(tmp_path / "test_jobs.db")
    _create_tables(db_path)
    return db_path


@pytest_asyncio.fixture
async def manager_with_wfo(temp_db):
    """JobManager avec worker loop démarré (WFO réel)."""
    mgr = JobManager(db_path=temp_db, ws_broadcast=None)
    await mgr.start()
    yield mgr
    await mgr.stop()


@pytest.mark.asyncio
@pytest.mark.slow  # Marquer comme slow pour skip en CI rapide
async def test_worker_real_wfo_short_window(manager_with_wfo):
    """Le worker loop exécute un vrai WFO (fenêtre ultra-courte pour test rapide).

    Ce test nécessite des candles en DB. Si pas de candles, skip.
    """
    # Params override pour réduire drastiquement la grille (test rapide)
    override = {
        "ma_period": [7],         # 1 seule valeur
        "num_levels": [2],        # 1 seule valeur
        "envelope_start": [0.05], # 1 seule valeur
        "envelope_step": [0.02],  # 1 seule valeur
        "sl_percent": [20.0],     # 1 seule valeur
    }

    try:
        job_id = await manager_with_wfo.submit_job(
            "envelope_dca", "BTC/USDT", params_override=override
        )
    except ValueError as exc:
        if "non optimisable" in str(exc):
            pytest.skip("Stratégie non disponible")
        raise

    # Attendre que le worker traite le job (timeout 5 min — un WFO court ~ 2-3 min)
    for _ in range(300):  # 300 × 1s = 5 min max
        await asyncio.sleep(1)
        job = await manager_with_wfo.get_job(job_id)
        if job.status in ("completed", "failed", "cancelled"):
            break

    job = await manager_with_wfo.get_job(job_id)

    # Vérifier que le job a abouti (completed ou failed si pas de candles)
    if job.status == "failed" and "Pas de candles" in (job.error_message or ""):
        pytest.skip("Pas de candles en DB pour ce test")

    # Si completed : vérifier les champs
    if job.status == "completed":
        assert job.progress_pct == 100
        assert job.duration_seconds is not None
        assert job.duration_seconds > 0
        assert job.started_at is not None
        assert job.completed_at is not None
        # result_id peut être None si save_report échoue, mais le job est completed
        # (on ne teste pas la DB optimization_results ici, juste le flow job)

    else:
        # Si failed pour une autre raison, logger et fail
        pytest.fail(f"Job {job_id[:8]} échoué : {job.error_message}")


@pytest.mark.asyncio
async def test_progress_updates_via_callback(temp_db):
    """Le progress callback met à jour le job en DB pendant le WFO."""
    broadcasts = []

    async def mock_broadcast(data):
        broadcasts.append(data)

    mgr = JobManager(db_path=temp_db, ws_broadcast=mock_broadcast)
    await mgr.start()

    # Grille ultra-réduite pour test rapide
    override = {
        "ma_period": [7],
        "num_levels": [2],
        "envelope_start": [0.05],
        "envelope_step": [0.02],
        "sl_percent": [20.0],
    }

    try:
        job_id = await mgr.submit_job(
            "envelope_dca", "BTC/USDT", params_override=override
        )
    except ValueError:
        pytest.skip("Stratégie non disponible ou pas de candles")

    # Attendre la complétion (timeout 5 min)
    for _ in range(300):
        await asyncio.sleep(1)
        job = await mgr.get_job(job_id)
        if job.status in ("completed", "failed", "cancelled"):
            break

    await mgr.stop()

    job = await mgr.get_job(job_id)

    if job.status == "failed" and "Pas de candles" in (job.error_message or ""):
        pytest.skip("Pas de candles en DB")

    # Vérifier que le broadcast a été appelé plusieurs fois (running + progress + completed)
    assert len(broadcasts) >= 3, f"Expected ≥3 broadcasts, got {len(broadcasts)}"
    assert broadcasts[0]["status"] == "running"
    assert broadcasts[-1]["status"] in ("completed", "failed")

    # Vérifier que progress_pct a augmenté au fil du temps
    progress_values = [b["progress_pct"] for b in broadcasts if "progress_pct" in b]
    assert len(progress_values) >= 2
    assert progress_values[-1] >= progress_values[0], "Progress devrait augmenter"
