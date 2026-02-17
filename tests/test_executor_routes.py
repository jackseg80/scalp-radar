"""Tests pour l'authentification des routes executor — Micro-Sprint Audit."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.server import app


# ─── Helpers ──────────────────────────────────────────────────────────────


def _patch_config(monkeypatch, sync_api_key: str = ""):
    """Configure un AppConfig de test pour les routes executor."""
    from backend.core.config import AppConfig

    test_config = AppConfig()
    test_config.secrets.sync_api_key = sync_api_key
    monkeypatch.setattr("backend.api.executor_routes.get_config", lambda: test_config)
    return test_config


# ─── Tests Auth 401 ───────────────────────────────────────────────────────


class TestExecutorAuth:
    """Vérifie que les 3 routes executor exigent une API key valide."""

    def test_status_no_api_key(self, monkeypatch):
        """GET /status sans header → 401."""
        _patch_config(monkeypatch, sync_api_key="secret123")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/executor/status")
        assert resp.status_code == 401
        assert "invalide" in resp.json()["detail"]

    def test_trade_no_api_key(self, monkeypatch):
        """POST /test-trade sans header → 401."""
        _patch_config(monkeypatch, sync_api_key="secret123")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/executor/test-trade")
        assert resp.status_code == 401

    def test_close_no_api_key(self, monkeypatch):
        """POST /test-close sans header → 401."""
        _patch_config(monkeypatch, sync_api_key="secret123")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/executor/test-close")
        assert resp.status_code == 401

    def test_trade_wrong_api_key(self, monkeypatch):
        """POST /test-trade avec mauvaise clé → 401."""
        _patch_config(monkeypatch, sync_api_key="secret123")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/executor/test-trade",
            headers={"X-API-Key": "wrong_key"},
        )
        assert resp.status_code == 401
        assert "invalide" in resp.json()["detail"]

    def test_trade_no_server_key_configured(self, monkeypatch):
        """POST /test-trade quand sync_api_key est vide côté serveur → 401."""
        _patch_config(monkeypatch, sync_api_key="")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/executor/test-trade",
            headers={"X-API-Key": "any_key"},
        )
        assert resp.status_code == 401
        assert "non configurée" in resp.json()["detail"]

    def test_status_valid_key_returns_executor_state(self, monkeypatch):
        """GET /status avec clé valide → passe l'auth (executor absent = réponse par défaut)."""
        _patch_config(monkeypatch, sync_api_key="secret123")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/api/executor/status",
            headers={"X-API-Key": "secret123"},
        )
        # L'auth passe, on obtient la réponse par défaut (executor non actif)
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False

    def test_refresh_balance_no_api_key(self, monkeypatch):
        """POST /refresh-balance sans header → 401."""
        _patch_config(monkeypatch, sync_api_key="secret123")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/executor/refresh-balance")
        assert resp.status_code == 401

    def test_refresh_balance_no_executor(self, monkeypatch):
        """POST /refresh-balance sans executor actif → 400."""
        _patch_config(monkeypatch, sync_api_key="secret123")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/executor/refresh-balance",
            headers={"X-API-Key": "secret123"},
        )
        assert resp.status_code == 400
