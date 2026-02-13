"""Tests pour job_manager.py — Sprint 14 Bloc A.

Tests du CRUD DB, submit/cancel, anti-doublon, limite queue.
Le worker loop réel (Bloc C) est testé séparément.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from backend.optimization.job_manager import (
    MAX_PENDING_JOBS,
    JobManager,
    OptimizationJob,
)

# ─── Fixtures ──────────────────────────────────────────────────────────────


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
        CREATE INDEX IF NOT EXISTS idx_jobs_status
            ON optimization_jobs(status);
    """)
    conn.close()


@pytest.fixture
def temp_db(tmp_path):
    """DB temporaire avec la table optimization_jobs."""
    db_path = str(tmp_path / "test_jobs.db")
    _create_tables(db_path)
    return db_path


@pytest_asyncio.fixture
async def manager(temp_db):
    """JobManager avec DB temporaire, sans broadcast WS."""
    mgr = JobManager(db_path=temp_db, ws_broadcast=None)
    # Ne pas démarrer le worker loop pour les tests CRUD
    return mgr


@pytest_asyncio.fixture
async def manager_with_loop(temp_db):
    """JobManager avec worker loop démarré."""
    mgr = JobManager(db_path=temp_db, ws_broadcast=None)
    await mgr.start()
    yield mgr
    await mgr.stop()


# ─── Tests CRUD ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_submit_job_creates_pending(manager):
    """submit_job crée un job en status pending dans la DB."""
    job_id = await manager.submit_job("envelope_dca", "BTC/USDT")

    assert job_id is not None
    assert len(job_id) == 36  # UUID format

    job = await manager.get_job(job_id)
    assert job is not None
    assert job.strategy_name == "envelope_dca"
    assert job.asset == "BTC/USDT"
    assert job.timeframe == "1h"
    assert job.status == "pending"
    assert job.progress_pct == 0
    assert job.params_override is None


@pytest.mark.asyncio
async def test_submit_job_with_params_override(manager):
    """submit_job avec params_override stocke le JSON en DB."""
    override = {"ma_period": [5, 7], "num_levels": [3, 4]}
    job_id = await manager.submit_job("envelope_dca", "BTC/USDT", params_override=override)

    job = await manager.get_job(job_id)
    assert job.params_override == override


@pytest.mark.asyncio
async def test_submit_job_invalid_strategy(manager):
    """submit_job avec une stratégie inconnue lève ValueError."""
    with pytest.raises(ValueError, match="non optimisable"):
        await manager.submit_job("unknown_strategy", "BTC/USDT")


@pytest.mark.asyncio
async def test_get_job_not_found(manager):
    """get_job retourne None pour un ID inexistant."""
    job = await manager.get_job("nonexistent-uuid")
    assert job is None


@pytest.mark.asyncio
async def test_list_jobs_empty(manager):
    """list_jobs retourne une liste vide sur DB vierge."""
    jobs = await manager.list_jobs()
    assert jobs == []


@pytest.mark.asyncio
async def test_list_jobs_with_filter(manager):
    """list_jobs filtre par status."""
    await manager.submit_job("envelope_dca", "BTC/USDT")
    await manager.submit_job("vwap_rsi", "ETH/USDT")

    all_jobs = await manager.list_jobs()
    assert len(all_jobs) == 2

    pending = await manager.list_jobs(status="pending")
    assert len(pending) == 2

    running = await manager.list_jobs(status="running")
    assert len(running) == 0


@pytest.mark.asyncio
async def test_list_jobs_ordered_by_created_at(manager):
    """list_jobs retourne les jobs les plus récents en premier."""
    id1 = await manager.submit_job("envelope_dca", "BTC/USDT")
    id2 = await manager.submit_job("vwap_rsi", "ETH/USDT")

    jobs = await manager.list_jobs()
    assert jobs[0].id == id2  # Plus récent en premier
    assert jobs[1].id == id1


# ─── Tests anti-doublon ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_submit_duplicate_raises(manager):
    """submit_job refuse un doublon (strategy, asset) pending/running."""
    await manager.submit_job("envelope_dca", "BTC/USDT")

    with pytest.raises(RuntimeError, match="déjà en cours"):
        await manager.submit_job("envelope_dca", "BTC/USDT")


@pytest.mark.asyncio
async def test_submit_same_strategy_different_asset(manager):
    """submit_job accepte la même stratégie sur un asset différent."""
    id1 = await manager.submit_job("envelope_dca", "BTC/USDT")
    id2 = await manager.submit_job("envelope_dca", "ETH/USDT")
    assert id1 != id2


@pytest.mark.asyncio
async def test_submit_different_strategy_same_asset(manager):
    """submit_job accepte une stratégie différente sur le même asset."""
    id1 = await manager.submit_job("envelope_dca", "BTC/USDT")
    id2 = await manager.submit_job("vwap_rsi", "BTC/USDT")
    assert id1 != id2


# ─── Tests limite queue ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_queue_limit(manager):
    """submit_job refuse au-delà de MAX_PENDING_JOBS."""
    # Utiliser des combos (strategy, asset) distinctes pour remplir la queue
    assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "LINK/USDT"]
    for asset in assets:
        await manager.submit_job("envelope_dca", asset)

    # Le 6ème doit échouer
    with pytest.raises(RuntimeError, match="Queue pleine"):
        await manager.submit_job("vwap_rsi", "BTC/USDT")


# ─── Tests cancel ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_pending_job(manager):
    """cancel_job passe un job pending en cancelled."""
    job_id = await manager.submit_job("envelope_dca", "BTC/USDT")

    result = await manager.cancel_job(job_id)
    assert result is True

    job = await manager.get_job(job_id)
    assert job.status == "cancelled"


@pytest.mark.asyncio
async def test_cancel_nonexistent_job(manager):
    """cancel_job retourne False pour un job inexistant."""
    result = await manager.cancel_job("nonexistent-uuid")
    assert result is False


@pytest.mark.asyncio
async def test_cancel_completed_job(manager):
    """cancel_job retourne False pour un job déjà terminé."""
    job_id = await manager.submit_job("envelope_dca", "BTC/USDT")
    # Simuler un job completed
    await manager._update_job_status(job_id, "completed")

    result = await manager.cancel_job(job_id)
    assert result is False


# ─── Tests update fields ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_job_fields(manager):
    """_update_job_fields met à jour plusieurs champs en une seule requête."""
    job_id = await manager.submit_job("envelope_dca", "BTC/USDT")

    await manager._update_job_fields(job_id, {
        "status": "running",
        "progress_pct": 42.5,
        "current_phase": "WFO Fenêtre 3/12",
    })

    job = await manager.get_job(job_id)
    assert job.status == "running"
    assert job.progress_pct == 42.5
    assert job.current_phase == "WFO Fenêtre 3/12"


@pytest.mark.asyncio
async def test_update_job_progress_sync(manager):
    """_update_job_progress_sync fonctionne en mode synchrone (pour le thread)."""
    job_id = await manager.submit_job("envelope_dca", "BTC/USDT")

    # Appel sync (comme depuis un thread WFO)
    manager._update_job_progress_sync(job_id, 75.0, "Monte Carlo")

    job = await manager.get_job(job_id)
    assert job.progress_pct == 75.0
    assert job.current_phase == "Monte Carlo"


# ─── Tests recovery ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recover_orphaned_jobs(temp_db):
    """Les jobs 'running' au boot passent en 'failed'."""
    # Insérer un job running directement
    conn = sqlite3.connect(temp_db)
    conn.execute(
        """INSERT INTO optimization_jobs
           (id, strategy_name, asset, timeframe, status, created_at)
           VALUES ('orphan-1', 'envelope_dca', 'BTC/USDT', '1h', 'running', '2025-01-01T00:00:00')"""
    )
    conn.commit()
    conn.close()

    mgr = JobManager(db_path=temp_db, ws_broadcast=None)
    await mgr._recover_orphaned_jobs()

    job = await mgr.get_job("orphan-1")
    assert job.status == "failed"
    assert "redémarré" in job.error_message


@pytest.mark.asyncio
async def test_enqueue_pending_jobs(temp_db):
    """Les jobs 'pending' au boot sont réenqueués."""
    conn = sqlite3.connect(temp_db)
    conn.execute(
        """INSERT INTO optimization_jobs
           (id, strategy_name, asset, timeframe, status, created_at)
           VALUES ('pending-1', 'envelope_dca', 'BTC/USDT', '1h', 'pending', '2025-01-01T00:00:00')"""
    )
    conn.commit()
    conn.close()

    mgr = JobManager(db_path=temp_db, ws_broadcast=None)
    await mgr._enqueue_pending_jobs()

    assert mgr._queue.qsize() == 1


# ─── Tests pending_count ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pending_count(manager):
    """pending_count retourne le nombre de jobs pending."""
    assert await manager.pending_count() == 0

    await manager.submit_job("envelope_dca", "BTC/USDT")
    assert await manager.pending_count() == 1

    await manager.submit_job("vwap_rsi", "ETH/USDT")
    assert await manager.pending_count() == 2


# ─── Tests worker loop (stub) ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_worker_loop_processes_job(manager_with_loop):
    """Le worker loop traite un job soumis (mock WFO rapide)."""
    from unittest.mock import patch

    # Mock _run_job_wfo_thread pour éviter un vrai WFO (test unitaire rapide)
    def mock_wfo_thread(job, cancel_event):
        # Simuler un WFO instantané qui réussit
        return 12345  # Fake result_id

    with patch.object(manager_with_loop, '_run_job_wfo_thread', side_effect=mock_wfo_thread):
        job_id = await manager_with_loop.submit_job("envelope_dca", "BTC/USDT")

        # Attendre que le worker traite le job (mock = quasi instantané)
        for _ in range(50):  # max 5 secondes
            await asyncio.sleep(0.1)
            job = await manager_with_loop.get_job(job_id)
            if job.status in ("completed", "failed"):
                break

        job = await manager_with_loop.get_job(job_id)
        assert job.status == "completed"
        assert job.progress_pct == 100
        assert job.duration_seconds is not None
        assert job.started_at is not None
        assert job.completed_at is not None
        assert job.result_id == 12345  # Fake result_id du mock


@pytest.mark.asyncio
async def test_run_job_skips_cancelled(manager):
    """_run_job ignore un job annulé (re-check DB avant exécution)."""
    job_id = await manager.submit_job("envelope_dca", "BTC/USDT")

    # Annuler le job
    await manager.cancel_job(job_id)

    # Simuler le worker qui dépile un job stale (lu avant le cancel)
    stale_job = OptimizationJob(
        id=job_id,
        strategy_name="envelope_dca",
        asset="BTC/USDT",
        timeframe="1h",
        status="pending",  # Stale — le worker l'avait lu avant le cancel
        progress_pct=0,
        current_phase="",
        params_override=None,
        created_at=datetime.now(timezone.utc),
    )
    await manager._run_job(stale_job)

    # Le job reste cancelled (pas écrasé par _run_job)
    job = await manager.get_job(job_id)
    assert job.status == "cancelled"


# ─── Test broadcast WS ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_broadcast_called_on_progress(temp_db):
    """Le broadcast WS est appelé lors de la progression."""
    from unittest.mock import patch

    broadcasts = []

    async def mock_broadcast(data):
        broadcasts.append(data)

    mgr = JobManager(db_path=temp_db, ws_broadcast=mock_broadcast)

    # Mock _run_job_wfo_thread pour un WFO instantané
    def mock_wfo_thread(job, cancel_event):
        return 12345

    with patch.object(mgr, '_run_job_wfo_thread', side_effect=mock_wfo_thread):
        await mgr.start()

        job_id = await mgr.submit_job("envelope_dca", "BTC/USDT")

        # Attendre le traitement
        for _ in range(50):
            await asyncio.sleep(0.1)
            job = await mgr.get_job(job_id)
            if job.status in ("completed", "failed"):
                break

        await mgr.stop()

    # Le mock émet au moins 2 broadcasts (running + completed)
    assert len(broadcasts) >= 2
    assert broadcasts[0]["type"] == "optimization_progress"
    assert broadcasts[0]["status"] == "running"
    assert broadcasts[-1]["status"] == "completed"
