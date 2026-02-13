"""Tests pour les routes API d'optimisation WFO — Sprint 13 + sync."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.server import app


# ─── Helper ──────────────────────────────────────────────────────────────

def _make_test_config(monkeypatch, sync_api_key: str = ""):
    """Configure un AppConfig de test avec monkeypatch."""
    from backend.core.config import AppConfig
    test_config = AppConfig()
    test_config.secrets.database_url = "sqlite:///data/test.db"
    test_config.secrets.sync_api_key = sync_api_key
    monkeypatch.setattr("backend.api.optimization_routes.get_config", lambda: test_config)
    return test_config


def test_get_results_ok(monkeypatch):
    """Test GET /api/optimization/results : 200 + pagination."""
    from backend.core.config import AppConfig

    test_config = AppConfig()
    test_config.secrets.database_url = "sqlite:///data/test.db"
    monkeypatch.setattr("backend.api.optimization_routes.get_config", lambda: test_config)

    async def mock_get_results(*args, **kwargs):
        return {
            "results": [
                {
                    "id": 1,
                    "strategy_name": "vwap_rsi",
                    "asset": "BTC/USDT",
                    "timeframe": "5m",
                    "grade": "A",
                    "total_score": 87.0,
                    "oos_sharpe": 1.8,
                    "consistency": 0.85,
                    "oos_is_ratio": 0.92,
                    "dsr": 0.78,
                    "param_stability": 0.88,
                    "n_windows": 20,
                    "is_latest": 1,
                    "created_at": "2026-02-13T10:00:00",
                },
                {
                    "id": 2,
                    "strategy_name": "envelope_dca",
                    "asset": "ETH/USDT",
                    "timeframe": "1h",
                    "grade": "B",
                    "total_score": 72.0,
                    "oos_sharpe": 1.2,
                    "consistency": 0.70,
                    "oos_is_ratio": 0.80,
                    "dsr": 0.65,
                    "param_stability": 0.75,
                    "n_windows": 15,
                    "is_latest": 1,
                    "created_at": "2026-02-13T11:00:00",
                },
            ],
            "total": 2,
        }

    monkeypatch.setattr("backend.api.optimization_routes.get_results_async", mock_get_results)

    client = TestClient(app)
    response = client.get("/api/optimization/results")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "total" in data
    assert data["total"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["strategy_name"] == "vwap_rsi"
    assert data["results"][0]["grade"] == "A"


def test_get_results_filtered(monkeypatch):
    """Test GET /api/optimization/results avec filtres strategy + asset."""
    from backend.core.config import AppConfig

    test_config = AppConfig()
    test_config.secrets.database_url = "sqlite:///data/test.db"
    monkeypatch.setattr("backend.api.optimization_routes.get_config", lambda: test_config)

    async def mock_get_results(*args, **kwargs):
        # Simuler le filtrage côté DB
        if kwargs.get("strategy") == "vwap_rsi" and kwargs.get("asset") == "BTC/USDT":
            return {
                "results": [
                    {
                        "id": 1,
                        "strategy_name": "vwap_rsi",
                        "asset": "BTC/USDT",
                        "timeframe": "5m",
                        "grade": "A",
                        "total_score": 87.0,
                        "oos_sharpe": 1.8,
                        "consistency": 0.85,
                        "n_windows": 20,
                        "is_latest": 1,
                        "created_at": "2026-02-13T10:00:00",
                    }
                ],
                "total": 1,
            }
        return {"results": [], "total": 0}

    monkeypatch.setattr("backend.api.optimization_routes.get_results_async", mock_get_results)

    client = TestClient(app)
    response = client.get("/api/optimization/results?strategy=vwap_rsi&asset=BTC/USDT")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["results"][0]["strategy_name"] == "vwap_rsi"
    assert data["results"][0]["asset"] == "BTC/USDT"


def test_get_results_pagination(monkeypatch):
    """Test GET /api/optimization/results : pagination offset + limit."""
    from backend.core.config import AppConfig

    test_config = AppConfig()
    test_config.secrets.database_url = "sqlite:///data/test.db"
    monkeypatch.setattr("backend.api.optimization_routes.get_config", lambda: test_config)

    async def mock_get_results(*args, **kwargs):
        # Vérifier les params de pagination
        assert kwargs.get("offset") == 10
        assert kwargs.get("limit") == 20
        return {"results": [], "total": 50}

    monkeypatch.setattr("backend.api.optimization_routes.get_results_async", mock_get_results)

    client = TestClient(app)
    response = client.get("/api/optimization/results?offset=10&limit=20")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 50


def test_get_result_detail(monkeypatch):
    """Test GET /api/optimization/{id} : 200 avec JSON complet."""
    from backend.core.config import AppConfig

    test_config = AppConfig()
    test_config.secrets.database_url = "sqlite:///data/test.db"
    monkeypatch.setattr("backend.api.optimization_routes.get_config", lambda: test_config)

    async def mock_get_result_by_id(*args, **kwargs):
        return {
            "id": 1,
            "strategy_name": "vwap_rsi",
            "asset": "BTC/USDT",
            "timeframe": "5m",
            "created_at": "2026-02-13T10:00:00",
            "grade": "A",
            "total_score": 87.0,
            "best_params": {"rsi_period": 14, "tp_percent": 0.8},
            "wfo_windows": [{"window_index": 0, "is_sharpe": 2.1, "oos_sharpe": 1.9}],
            "monte_carlo_summary": {"p_value": 0.03, "significant": True},
            "validation_summary": {"bitget_sharpe": 1.5, "transfer_ratio": 0.85},
            "warnings": ["Test warning"],
        }

    monkeypatch.setattr("backend.api.optimization_routes.get_result_by_id_async", mock_get_result_by_id)

    client = TestClient(app)
    response = client.get("/api/optimization/1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 1
    assert data["strategy_name"] == "vwap_rsi"
    assert "best_params" in data
    assert "wfo_windows" in data
    assert "monte_carlo_summary" in data
    assert "validation_summary" in data
    assert isinstance(data["best_params"], dict)
    assert isinstance(data["wfo_windows"], list)
    assert data["best_params"]["rsi_period"] == 14


def test_get_result_not_found(monkeypatch):
    """Test GET /api/optimization/{id} : 404 si inexistant."""
    from backend.core.config import AppConfig

    test_config = AppConfig()
    test_config.secrets.database_url = "sqlite:///data/test.db"
    monkeypatch.setattr("backend.api.optimization_routes.get_config", lambda: test_config)

    async def mock_get_result_by_id(*args, **kwargs):
        return None

    monkeypatch.setattr("backend.api.optimization_routes.get_result_by_id_async", mock_get_result_by_id)

    client = TestClient(app)
    response = client.get("/api/optimization/999")
    assert response.status_code == 404
    assert "non trouvé" in response.json()["detail"]


def test_get_comparison(monkeypatch):
    """Test GET /api/optimization/comparison : matrice strategies × assets."""
    from backend.core.config import AppConfig

    test_config = AppConfig()
    test_config.secrets.database_url = "sqlite:///data/test.db"
    monkeypatch.setattr("backend.api.optimization_routes.get_config", lambda: test_config)

    async def mock_get_comparison(*args, **kwargs):
        return {
            "strategies": ["vwap_rsi", "envelope_dca"],
            "assets": ["BTC/USDT", "ETH/USDT"],
            "matrix": {
                "vwap_rsi": {
                    "BTC/USDT": {
                        "grade": "A",
                        "total_score": 87.0,
                        "oos_sharpe": 1.8,
                        "consistency": 0.85,
                        "oos_is_ratio": 0.92,
                        "dsr": 0.78,
                        "param_stability": 0.88,
                        "n_windows": 20,
                    }
                },
                "envelope_dca": {
                    "ETH/USDT": {
                        "grade": "B",
                        "total_score": 72.0,
                        "oos_sharpe": 1.2,
                        "consistency": 0.70,
                        "oos_is_ratio": 0.80,
                        "dsr": 0.65,
                        "param_stability": 0.75,
                        "n_windows": 15,
                    }
                },
            },
        }

    monkeypatch.setattr("backend.api.optimization_routes.get_comparison_async", mock_get_comparison)

    client = TestClient(app)
    response = client.get("/api/optimization/comparison")
    assert response.status_code == 200
    data = response.json()
    assert "strategies" in data
    assert "assets" in data
    assert "matrix" in data
    assert len(data["strategies"]) == 2
    assert len(data["assets"]) == 2
    assert "vwap_rsi" in data["matrix"]
    assert "BTC/USDT" in data["matrix"]["vwap_rsi"]
    assert data["matrix"]["vwap_rsi"]["BTC/USDT"]["grade"] == "A"
    assert data["matrix"]["envelope_dca"]["ETH/USDT"]["grade"] == "B"


# ─── Tests POST /api/optimization/results ─────────────────────────────────


def _make_post_payload(**overrides) -> dict:
    """Payload de test pour POST."""
    base = {
        "strategy_name": "vwap_rsi",
        "asset": "BTC/USDT",
        "timeframe": "5m",
        "created_at": "2026-02-13T12:00:00",
        "grade": "A",
        "total_score": 87.0,
        "n_windows": 20,
        "best_params": '{"rsi_period": 14}',
        "source": "local",
    }
    base.update(overrides)
    return base


def test_post_result_created(monkeypatch):
    """Test POST /api/optimization/results : 201 + created."""
    _make_test_config(monkeypatch, sync_api_key="test-secret-key")

    def mock_save(*args, **kwargs):
        return "created"

    monkeypatch.setattr(
        "backend.api.optimization_routes.save_result_from_payload_sync", mock_save
    )

    client = TestClient(app)
    response = client.post(
        "/api/optimization/results",
        json=_make_post_payload(),
        headers={"X-API-Key": "test-secret-key"},
    )
    assert response.status_code == 201
    assert response.json()["status"] == "created"


def test_post_result_duplicate(monkeypatch):
    """Test POST doublon → 200 + already_exists."""
    _make_test_config(monkeypatch, sync_api_key="test-secret-key")

    def mock_save(*args, **kwargs):
        return "exists"

    monkeypatch.setattr(
        "backend.api.optimization_routes.save_result_from_payload_sync", mock_save
    )

    client = TestClient(app)
    response = client.post(
        "/api/optimization/results",
        json=_make_post_payload(),
        headers={"X-API-Key": "test-secret-key"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "already_exists"


def test_post_result_no_api_key(monkeypatch):
    """Test POST sans header X-API-Key → 401."""
    _make_test_config(monkeypatch, sync_api_key="test-secret-key")

    client = TestClient(app)
    response = client.post(
        "/api/optimization/results",
        json=_make_post_payload(),
    )
    assert response.status_code == 401


def test_post_result_wrong_api_key(monkeypatch):
    """Test POST avec mauvaise clé → 401."""
    _make_test_config(monkeypatch, sync_api_key="test-secret-key")

    client = TestClient(app)
    response = client.post(
        "/api/optimization/results",
        json=_make_post_payload(),
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401
    assert "invalide" in response.json()["detail"]


def test_post_result_no_server_key(monkeypatch):
    """Test POST quand sync_api_key vide côté serveur → 401."""
    _make_test_config(monkeypatch, sync_api_key="")

    client = TestClient(app)
    response = client.post(
        "/api/optimization/results",
        json=_make_post_payload(),
        headers={"X-API-Key": "any-key"},
    )
    assert response.status_code == 401
    assert "non configuré" in response.json()["detail"]


def test_post_result_invalid_payload(monkeypatch):
    """Test POST avec champs obligatoires manquants → 422."""
    _make_test_config(monkeypatch, sync_api_key="test-secret-key")

    client = TestClient(app)
    # Payload incomplet : manque strategy_name, grade, etc.
    response = client.post(
        "/api/optimization/results",
        json={"asset": "BTC/USDT"},
        headers={"X-API-Key": "test-secret-key"},
    )
    assert response.status_code == 422
    assert "manquants" in response.json()["detail"]
