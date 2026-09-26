"""Real worker subprocess/protocol with a synthetic optimization dependency."""

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
from pathlib import Path

import pytest
import pytest_asyncio

from backend.optimization.job_manager import JobManager


@pytest.fixture(autouse=True)
def isolated_worker(monkeypatch):
    original = subprocess.Popen
    fixture = Path(__file__).parent / "fixtures" / "protocol_worker.py"

    def launch(command, **kwargs):
        assert Path(command[1]).name == "wfo_worker.py"
        return original([command[0], str(fixture), *command[2:]], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch)


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
    """JobManager with a real process and synthetic optimization dependency."""
    mgr = JobManager(db_path=temp_db, ws_broadcast=None)
    await mgr.start()
    yield mgr
    await mgr.stop()


@pytest.mark.asyncio
async def test_worker_process_protocol(manager_with_wfo):
    """Actual worker framing, progress and completion; no candles required."""
    # Params override pour réduire drastiquement la grille (test rapide)
    override = {
        "ma_period": [7],         # 1 seule valeur
        "num_levels": [2],        # 1 seule valeur
        "envelope_start": [0.05], # 1 seule valeur
        "envelope_step": [0.02],  # 1 seule valeur
        "sl_percent": [20.0],     # 1 seule valeur
    }

    job_id = await manager_with_wfo.submit_job(
        "envelope_dca", "BTC/USDT", params_override=override
    )

    # Real subprocess startup, but no market data or real optimization.
    for _ in range(100):
        await asyncio.sleep(0.1)
        job = await manager_with_wfo.get_job(job_id)
        if job.status in ("completed", "failed", "cancelled"):
            break

    job = await manager_with_wfo.get_job(job_id)

    # Si completed : vérifier les champs
    if job.status == "completed":
        assert job.progress_pct == 100
        assert job.duration_seconds is not None
        assert job.duration_seconds > 0
        assert job.started_at is not None
        assert job.completed_at is not None
        assert job.result_id == 12345

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

    job_id = await mgr.submit_job(
        "envelope_dca", "BTC/USDT", params_override=override
    )

    for _ in range(100):
        await asyncio.sleep(0.1)
        job = await mgr.get_job(job_id)
        if job.status in ("completed", "failed", "cancelled"):
            break

    await mgr.stop()

    job = await mgr.get_job(job_id)

    assert job.status == "completed", job.error_message
    assert job.result_id == 12345

    # Vérifier que le broadcast a été appelé plusieurs fois (running + progress + completed)
    assert len(broadcasts) >= 3, f"Expected ≥3 broadcasts, got {len(broadcasts)}"
    assert broadcasts[0]["status"] == "running"
    assert broadcasts[-1]["status"] in ("completed", "failed")

    # Vérifier que progress_pct a augmenté au fil du temps
    progress_values = [b["progress_pct"] for b in broadcasts if "progress_pct" in b]
    assert len(progress_values) >= 2
    assert progress_values[-1] >= progress_values[0], "Progress devrait augmenter"
