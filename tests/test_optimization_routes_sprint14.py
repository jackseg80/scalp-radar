"""Tests API endpoints Sprint 14 — Explorateur de Paramètres."""

from __future__ import annotations

import sqlite3

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient


def _create_tables(db_path: str) -> None:
    """Crée les tables nécessaires."""
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

        CREATE TABLE IF NOT EXISTS optimization_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            asset TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            created_at TEXT NOT NULL,
            duration_seconds REAL,
            grade TEXT NOT NULL,
            total_score REAL NOT NULL,
            oos_sharpe REAL,
            consistency REAL,
            oos_is_ratio REAL,
            dsr REAL,
            param_stability REAL,
            monte_carlo_pvalue REAL,
            mc_underpowered INTEGER DEFAULT 0,
            n_windows INTEGER NOT NULL,
            n_distinct_combos INTEGER,
            best_params TEXT NOT NULL,
            wfo_windows TEXT,
            monte_carlo_summary TEXT,
            validation_summary TEXT,
            warnings TEXT,
            is_latest INTEGER DEFAULT 1,
            source TEXT DEFAULT 'local',
            regime_analysis TEXT,
            UNIQUE(strategy_name, asset, timeframe, created_at)
        );

        CREATE TABLE IF NOT EXISTS wfo_combo_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            optimization_result_id INTEGER NOT NULL,
            params TEXT NOT NULL,
            oos_sharpe REAL,
            oos_return_pct REAL,
            oos_trades INTEGER,
            oos_win_rate REAL,
            is_sharpe REAL,
            is_return_pct REAL,
            is_trades INTEGER,
            consistency REAL,
            oos_is_ratio REAL,
            is_best INTEGER DEFAULT 0,
            n_windows_evaluated INTEGER,
            per_window_sharpes TEXT,
            FOREIGN KEY (optimization_result_id) REFERENCES optimization_results(id)
        );
        CREATE INDEX IF NOT EXISTS idx_combo_opt_id ON wfo_combo_results(optimization_result_id);
        CREATE INDEX IF NOT EXISTS idx_combo_best ON wfo_combo_results(is_best) WHERE is_best = 1;
    """)
    conn.close()


@pytest.fixture
def temp_db(tmp_path):
    """DB temporaire avec tables."""
    db_path = str(tmp_path / "test_api.db")
    _create_tables(db_path)
    return db_path


@pytest_asyncio.fixture
async def client(temp_db, monkeypatch):
    """TestClient FastAPI avec JobManager mocké (sans lifespan complet)."""
    from backend.optimization.job_manager import JobManager
    from fastapi import FastAPI
    from backend.api.optimization_routes import router

    # Monkeypatch _get_db_path pour utiliser temp_db
    import backend.api.optimization_routes as opt_routes
    monkeypatch.setattr(opt_routes, "_get_db_path", lambda: temp_db)

    # App test minimaliste (pas de lifespan complet)
    test_app = FastAPI()
    test_app.include_router(router)

    # Créer et démarrer un JobManager
    mgr = JobManager(db_path=temp_db, ws_broadcast=None)
    await mgr.start()
    test_app.state.job_manager = mgr

    # TestClient synchrone
    with TestClient(test_app) as test_client:
        yield test_client

    # Cleanup
    await mgr.stop()


# ─── Tests POST /api/optimization/run ─────────────────────────────────────


def test_submit_job(client):
    """POST /api/optimization/run crée un job pending."""
    response = client.post(
        "/api/optimization/run",
        json={"strategy_name": "envelope_dca", "asset": "BTC/USDT"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"


def test_submit_job_invalid_strategy(client):
    """POST avec stratégie inconnue → 400."""
    response = client.post(
        "/api/optimization/run",
        json={"strategy_name": "unknown_strat", "asset": "BTC/USDT"},
    )
    assert response.status_code == 400
    assert "non optimisable" in response.json()["detail"]


def test_submit_job_duplicate(client):
    """POST doublon (strategy, asset) → 409."""
    client.post(
        "/api/optimization/run",
        json={"strategy_name": "envelope_dca", "asset": "BTC/USDT"},
    )
    # 2ème soumission
    response = client.post(
        "/api/optimization/run",
        json={"strategy_name": "envelope_dca", "asset": "BTC/USDT"},
    )
    assert response.status_code == 409


def test_submit_job_queue_full(client):
    """POST quand queue pleine → 429."""
    # Remplir la queue (5 pending max)
    for i in range(5):
        client.post(
            "/api/optimization/run",
            json={"strategy_name": "envelope_dca", "asset": f"ASSET{i}/USDT"},
        )
    # 6ème → queue pleine
    response = client.post(
        "/api/optimization/run",
        json={"strategy_name": "vwap_rsi", "asset": "BTC/USDT"},
    )
    assert response.status_code == 429


def test_submit_job_with_params_override(client):
    """POST avec params_override stocke le JSON."""
    override = {"ma_period": [7], "num_levels": [2]}
    response = client.post(
        "/api/optimization/run",
        json={
            "strategy_name": "envelope_dca",
            "asset": "BTC/USDT",
            "params_override": override,
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    # Vérifier via GET
    detail_resp = client.get(f"/api/optimization/jobs/{job_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["params_override"] == override


# ─── Tests GET /api/optimization/jobs ─────────────────────────────────────


def test_list_jobs_empty(client):
    """GET /api/optimization/jobs sur DB vierge → liste vide."""
    response = client.get("/api/optimization/jobs")
    assert response.status_code == 200
    data = response.json()
    assert data["jobs"] == []


def test_list_jobs_with_filter(client):
    """GET /api/optimization/jobs?status=pending filtre."""
    client.post(
        "/api/optimization/run",
        json={"strategy_name": "envelope_dca", "asset": "BTC/USDT"},
    )
    client.post(
        "/api/optimization/run",
        json={"strategy_name": "vwap_rsi", "asset": "ETH/USDT"},
    )

    # Tous
    resp_all = client.get("/api/optimization/jobs")
    assert len(resp_all.json()["jobs"]) == 2

    # Pending seulement
    resp_pending = client.get("/api/optimization/jobs?status=pending")
    assert len(resp_pending.json()["jobs"]) == 2


# ─── Tests GET /api/optimization/jobs/{job_id} ────────────────────────────


def test_get_job_detail(client):
    """GET /api/optimization/jobs/{id} retourne le détail."""
    resp = client.post(
        "/api/optimization/run",
        json={"strategy_name": "envelope_dca", "asset": "BTC/USDT"},
    )
    job_id = resp.json()["job_id"]

    detail_resp = client.get(f"/api/optimization/jobs/{job_id}")
    assert detail_resp.status_code == 200
    data = detail_resp.json()
    assert data["id"] == job_id
    assert data["strategy_name"] == "envelope_dca"
    assert data["asset"] == "BTC/USDT"
    assert data["status"] == "pending"


def test_get_job_not_found(client):
    """GET job inexistant → 404."""
    response = client.get("/api/optimization/jobs/nonexistent-uuid")
    assert response.status_code == 404


# ─── Tests DELETE /api/optimization/jobs/{job_id} ─────────────────────────


def test_cancel_job(client):
    """DELETE /api/optimization/jobs/{id} annule un job pending."""
    resp = client.post(
        "/api/optimization/run",
        json={"strategy_name": "envelope_dca", "asset": "BTC/USDT"},
    )
    job_id = resp.json()["job_id"]

    cancel_resp = client.delete(f"/api/optimization/jobs/{job_id}")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"

    # Vérifier status
    detail_resp = client.get(f"/api/optimization/jobs/{job_id}")
    assert detail_resp.json()["status"] == "cancelled"


def test_cancel_nonexistent_job(client):
    """DELETE job inexistant → 404."""
    response = client.delete("/api/optimization/jobs/nonexistent-uuid")
    assert response.status_code == 404


# ─── Tests GET /api/optimization/param-grid/{strategy} ────────────────────


def test_get_param_grid(client):
    """GET /api/optimization/param-grid/envelope_dca retourne les params."""
    response = client.get("/api/optimization/param-grid/envelope_dca")
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "envelope_dca"
    assert "params" in data
    # Vérifier que ma_period existe
    assert "ma_period" in data["params"]
    assert "values" in data["params"]["ma_period"]
    assert "default" in data["params"]["ma_period"]


def test_get_param_grid_unknown_strategy(client):
    """GET param-grid pour stratégie inconnue → 404."""
    response = client.get("/api/optimization/param-grid/unknown_strat")
    assert response.status_code == 404


# ─── Tests GET /api/optimization/heatmap ──────────────────────────────────


def test_get_heatmap_empty(client, temp_db):
    """GET /api/optimization/heatmap sans résultats → matrice vide."""
    response = client.get(
        "/api/optimization/heatmap",
        params={
            "strategy": "envelope_dca",
            "asset": "BTC/USDT",
            "param_x": "envelope_start",
            "param_y": "envelope_step",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["x_values"] == []
    assert data["y_values"] == []
    assert data["cells"] == []


def test_get_heatmap_with_data(client, temp_db):
    """GET /api/optimization/heatmap avec résultats → matrice remplie."""
    import json

    # Insérer 2 résultats manuellement
    conn = sqlite3.connect(temp_db)
    conn.execute(
        """INSERT INTO optimization_results
           (strategy_name, asset, timeframe, created_at, grade, total_score, n_windows, best_params)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "envelope_dca", "BTC/USDT", "1h", "2025-01-01T00:00:00",
            "B", 50.0, 10,
            json.dumps({"ma_period": 7, "envelope_start": 0.05, "envelope_step": 0.02}),
        ),
    )
    conn.execute(
        """INSERT INTO optimization_results
           (strategy_name, asset, timeframe, created_at, grade, total_score, n_windows, best_params)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "envelope_dca", "BTC/USDT", "1h", "2025-01-02T00:00:00",
            "C", 45.0, 10,
            json.dumps({"ma_period": 7, "envelope_start": 0.07, "envelope_step": 0.02}),
        ),
    )
    conn.commit()
    conn.close()

    response = client.get(
        "/api/optimization/heatmap",
        params={
            "strategy": "envelope_dca",
            "asset": "BTC/USDT",
            "param_x": "envelope_start",
            "param_y": "envelope_step",
            "metric": "total_score",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["x_values"] == [0.05, 0.07]
    assert data["y_values"] == [0.02]
    assert len(data["cells"]) == 1  # 1 row (y=0.02)
    assert len(data["cells"][0]) == 2  # 2 cols (x=0.05, x=0.07)
    # Premier point
    assert data["cells"][0][0]["value"] == 50.0
    assert data["cells"][0][0]["grade"] == "B"
    # Deuxième point
    assert data["cells"][0][1]["value"] == 45.0
    assert data["cells"][0][1]["grade"] == "C"
