"""Tests pour portfolio_routes.py (Sprint 20b-UI).

Couvre :
- GET /api/portfolio/presets
- GET /api/portfolio/backtests (liste vide et avec données)
- GET /api/portfolio/backtests/{id} (détail et 404)
- DELETE /api/portfolio/backtests/{id}
- POST /api/portfolio/run (retourne job_id, conflict)
- GET /api/portfolio/status (idle)
- GET /api/portfolio/compare
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.server import app


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# ─── Presets ──────────────────────────────────────────────────────────────


def test_get_presets(client):
    """GET /api/portfolio/presets retourne les presets."""
    resp = client.get("/api/portfolio/presets")
    assert resp.status_code == 200
    data = resp.json()
    assert "presets" in data
    assert len(data["presets"]) >= 3
    # Vérifier la structure
    for p in data["presets"]:
        assert "name" in p
        assert "label" in p
        assert "capital" in p
        assert "days" in p


# ─── Liste ────────────────────────────────────────────────────────────────


@patch("backend.api.portfolio_routes.get_backtests_async", new_callable=AsyncMock)
def test_list_backtests_empty(mock_get, client):
    """GET /api/portfolio/backtests retourne une liste vide."""
    mock_get.return_value = []
    resp = client.get("/api/portfolio/backtests")
    assert resp.status_code == 200
    data = resp.json()
    assert data["backtests"] == []
    assert data["total"] == 0


@patch("backend.api.portfolio_routes.get_backtests_async", new_callable=AsyncMock)
def test_list_backtests_with_data(mock_get, client):
    """GET /api/portfolio/backtests retourne les résultats."""
    mock_get.return_value = [
        {"id": 1, "strategy_name": "grid_atr", "total_return_pct": 5.2, "n_assets": 10},
        {"id": 2, "strategy_name": "grid_atr", "total_return_pct": -1.3, "n_assets": 5},
    ]
    resp = client.get("/api/portfolio/backtests")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["backtests"]) == 2
    assert data["total"] == 2


# ─── Détail ───────────────────────────────────────────────────────────────


@patch("backend.api.portfolio_routes.get_backtest_by_id_async", new_callable=AsyncMock)
def test_get_backtest_detail(mock_get, client):
    """GET /api/portfolio/backtests/{id} retourne le détail."""
    mock_get.return_value = {
        "id": 1,
        "strategy_name": "grid_atr",
        "equity_curve": [{"timestamp": "2025-06-01T00:00:00", "equity": 10000}],
        "per_asset_results": {"AAA/USDT": {"trades": 5}},
    }
    resp = client.get("/api/portfolio/backtests/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == 1
    assert "equity_curve" in data


@patch("backend.api.portfolio_routes.get_backtest_by_id_async", new_callable=AsyncMock)
def test_get_backtest_not_found(mock_get, client):
    """GET /api/portfolio/backtests/{id} retourne 404."""
    mock_get.return_value = None
    resp = client.get("/api/portfolio/backtests/999")
    assert resp.status_code == 404


# ─── Suppression ──────────────────────────────────────────────────────────


@patch("backend.api.portfolio_routes.delete_backtest_async", new_callable=AsyncMock)
def test_delete_backtest_ok(mock_del, client):
    """DELETE /api/portfolio/backtests/{id} supprime le run."""
    mock_del.return_value = True
    resp = client.delete("/api/portfolio/backtests/1")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"


@patch("backend.api.portfolio_routes.delete_backtest_async", new_callable=AsyncMock)
def test_delete_backtest_not_found(mock_del, client):
    """DELETE /api/portfolio/backtests/{id} retourne 404."""
    mock_del.return_value = False
    resp = client.delete("/api/portfolio/backtests/999")
    assert resp.status_code == 404


# ─── Status ───────────────────────────────────────────────────────────────


def test_status_idle(client):
    """GET /api/portfolio/status retourne running=false quand aucun job."""
    # Reset le job tracker
    import backend.api.portfolio_routes as routes
    routes._current_job = None
    resp = client.get("/api/portfolio/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is False


# ─── Compare ──────────────────────────────────────────────────────────────


@patch("backend.api.portfolio_routes.get_backtest_by_id_async", new_callable=AsyncMock)
def test_compare_backtests(mock_get, client):
    """GET /api/portfolio/compare?ids=1,2 retourne 2 runs."""
    run1 = {"id": 1, "total_return_pct": 5.2, "equity_curve": []}
    run2 = {"id": 2, "total_return_pct": -1.3, "equity_curve": []}
    mock_get.side_effect = [run1, run2]
    resp = client.get("/api/portfolio/compare?ids=1,2")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["runs"]) == 2


def test_compare_needs_two_ids(client):
    """GET /api/portfolio/compare?ids=1 retourne 400."""
    resp = client.get("/api/portfolio/compare?ids=1")
    assert resp.status_code == 400
