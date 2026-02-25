"""Tests pour GET /api/portfolio/backtests/{id}/robustness (Sprint 48).

Couvre :
- Données robustesse présentes → 200 + JSON parsé
- DB sans table portfolio_robustness → 200 + robustness=null
- Filtre par backtest_id (retourne le bon résultat)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.server import app


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# ─── Données présentes ────────────────────────────────────────────


@patch(
    "backend.api.portfolio_routes.get_robustness_by_backtest_id_async",
    new_callable=AsyncMock,
)
def test_get_robustness_with_data(mock_get, client):
    """GET /api/portfolio/backtests/{id}/robustness retourne les données."""
    mock_get.return_value = {
        "id": 1,
        "backtest_id": 42,
        "label": "grid_atr_v2_13assets_7x",
        "created_at": "2026-02-25T10:00:00+00:00",
        "verdict": "VIABLE",
        "verdict_details": {
            "verdict": "VIABLE",
            "criteria": [
                {"name": "CI95 return borne basse > 0%", "value": "+3.2%", "pass": True},
                {"name": "Probabilité de perte < 10%", "value": "2.1%", "pass": True},
                {"name": "CVaR 5% 30j < kill_switch (45%)", "value": "12.3%", "pass": True},
                {"name": "Survit crashes historiques (DD < -40%)", "value": "5/5", "pass": True},
            ],
            "n_pass": 4,
            "n_fail": 0,
        },
        "bootstrap_n_sims": 5000,
        "bootstrap_block_size": 7,
        "bootstrap_median_return": 0.05,
        "bootstrap_ci95_return_low": 0.01,
        "bootstrap_ci95_return_high": 0.12,
        "bootstrap_median_dd": -0.08,
        "bootstrap_ci95_dd_low": -0.15,
        "bootstrap_ci95_dd_high": -0.03,
        "bootstrap_prob_loss": 0.02,
        "bootstrap_prob_dd_30": 0.0,
        "bootstrap_prob_dd_ks": 0.0,
        "var_5_daily": -0.015,
        "cvar_5_daily": -0.022,
        "cvar_30d": -0.35,
        "cvar_5_annualized": -0.99,
        "cvar_by_regime": {"RANGE": -0.01, "BULL": -0.005, "BEAR": -0.03, "CRASH": -0.08},
        "regime_stress_results": {
            "Bear prolongé 6m": {
                "median_return": -0.15,
                "median_dd": -0.30,
                "prob_loss": 0.85,
            },
        },
        "historical_stress_results": {
            "COVID crash": {
                "status": "OK",
                "period": "2020-03-09 → 2020-03-23",
                "portfolio_dd": -0.18,
                "btc_dd": -0.45,
                "recovery_days": 14,
            },
        },
    }
    resp = client.get("/api/portfolio/backtests/42/robustness")
    assert resp.status_code == 200
    data = resp.json()
    assert data["robustness"] is not None
    assert data["robustness"]["verdict"] == "VIABLE"
    assert data["robustness"]["bootstrap_n_sims"] == 5000
    assert "regime_stress_results" in data["robustness"]
    assert "historical_stress_results" in data["robustness"]
    assert data["robustness"]["verdict_details"]["n_pass"] == 4


# ─── DB vide (table absente ou pas de résultat) ──────────────────


@patch(
    "backend.api.portfolio_routes.get_robustness_by_backtest_id_async",
    new_callable=AsyncMock,
)
def test_get_robustness_empty_db(mock_get, client):
    """GET /api/portfolio/backtests/{id}/robustness retourne null si pas de données."""
    mock_get.return_value = None
    resp = client.get("/api/portfolio/backtests/99/robustness")
    assert resp.status_code == 200
    data = resp.json()
    assert data["robustness"] is None


# ─── Filtre par backtest_id ───────────────────────────────────────


@patch(
    "backend.api.portfolio_routes.get_robustness_by_backtest_id_async",
    new_callable=AsyncMock,
)
def test_get_robustness_correct_backtest_id(mock_get, client):
    """GET /api/portfolio/backtests/{id}/robustness passe le bon backtest_id."""
    mock_get.return_value = {
        "id": 5,
        "backtest_id": 77,
        "label": "specific_label",
        "created_at": "2026-02-25T12:00:00+00:00",
        "verdict": "CAUTION",
        "verdict_details": {
            "verdict": "CAUTION",
            "criteria": [
                {"name": "CVaR 5% 30j < kill_switch (45%)", "value": "48.2%", "pass": False},
            ],
            "n_pass": 3,
            "n_fail": 1,
        },
        "bootstrap_n_sims": None,
        "bootstrap_block_size": None,
        "bootstrap_median_return": None,
        "bootstrap_ci95_return_low": None,
        "bootstrap_ci95_return_high": None,
        "bootstrap_median_dd": None,
        "bootstrap_ci95_dd_low": None,
        "bootstrap_ci95_dd_high": None,
        "bootstrap_prob_loss": None,
        "bootstrap_prob_dd_30": None,
        "bootstrap_prob_dd_ks": None,
        "var_5_daily": -0.02,
        "cvar_5_daily": -0.03,
        "cvar_30d": -0.50,
        "cvar_5_annualized": -0.99,
        "cvar_by_regime": {},
        "regime_stress_results": None,
        "historical_stress_results": None,
    }
    resp = client.get("/api/portfolio/backtests/77/robustness")
    assert resp.status_code == 200
    data = resp.json()
    assert data["robustness"]["backtest_id"] == 77
    assert data["robustness"]["verdict"] == "CAUTION"
    # Vérifier que le mock a été appelé avec le bon backtest_id
    mock_get.assert_awaited_once()
    call_args = mock_get.call_args
    assert call_args[0][1] == 77
