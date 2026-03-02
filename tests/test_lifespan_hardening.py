"""Tests Sprint Audit-A : hardening du lifespan server.py.

Vérifie que le lifespan gère correctement les échecs de composants
et que le shutdown est protégé par timeout.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.server import _safe_stop, lifespan


# ─── Helpers ──────────────────────────────────────────────────────────────


def _make_app():
    """Crée un faux objet app avec state."""
    app = MagicMock()
    app.state = MagicMock()
    return app


def _mock_config(live_trading: bool = False, enable_ws: bool = False):
    """Crée un AppConfig mocké minimal."""
    config = MagicMock()
    config.secrets.log_level = "WARNING"
    config.secrets.telegram_bot_token = ""
    config.secrets.telegram_chat_id = ""
    config.secrets.enable_websocket = enable_ws
    config.secrets.live_trading = live_trading
    config.secrets.heartbeat_interval = 3600
    config.risk.initial_capital = 1000.0
    return config


# ─── Tests _safe_stop ─────────────────────────────────────────────────────


class TestSafeStop:
    """Vérifie le helper de shutdown protégé."""

    @pytest.mark.asyncio
    async def test_safe_stop_success(self):
        """Un composant qui s'arrête normalement."""
        mock_coro = AsyncMock()
        await _safe_stop("TestComponent", mock_coro())
        mock_coro.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_safe_stop_exception(self):
        """Un composant qui raise ne bloque pas le shutdown."""
        async def failing_stop():
            raise RuntimeError("boom")

        # Ne doit PAS raise
        await _safe_stop("FailComponent", failing_stop())

    @pytest.mark.asyncio
    async def test_safe_stop_timeout(self):
        """Un composant qui hang est coupé par timeout."""
        async def hanging_stop():
            await asyncio.sleep(999)

        # Timeout court pour le test (0.1s)
        await _safe_stop("HangComponent", hanging_stop(), timeout=0.1)


# ─── Tests lifespan — échec Database ──────────────────────────────────────


class TestLifespanDatabaseFailure:
    """Vérifie que le lifespan survit à un échec Database."""

    @pytest.mark.asyncio
    async def test_database_failure_yields_degraded(self):
        """Si Database.init() raise, le lifespan yield quand même."""
        app = _make_app()

        with (
            patch("backend.api.server.get_config", return_value=_mock_config()),
            patch("backend.api.server.setup_logging"),
            patch("backend.api.server.Database") as MockDB,
        ):
            mock_db = MockDB.return_value
            mock_db.init = AsyncMock(side_effect=RuntimeError("db connection refused"))

            async with lifespan(app):
                # L'app doit être disponible même sans DB
                components = app.state.startup_components
                assert isinstance(components, dict)
                assert components["database"].startswith("error")
                # Pas de simulator ni engine sans DB
                assert app.state.engine is None


# ─── Tests lifespan — startup partiel ─────────────────────────────────────


class TestLifespanPartialStartup:
    """Vérifie qu'un échec non-critique ne bloque pas le reste."""

    @pytest.mark.asyncio
    async def test_data_engine_failure_continues(self):
        """Si DataEngine.start() raise, le lifespan continue."""
        app = _make_app()
        config = _mock_config(enable_ws=True)

        mock_db = AsyncMock()
        mock_db.db_path = "data/test.db"
        mock_db.init = AsyncMock()
        mock_db.start_maintenance_loop = MagicMock()
        mock_db.close = AsyncMock()

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock(side_effect=ConnectionError("ws refused"))
        mock_engine.stop = AsyncMock()

        mock_jm = AsyncMock()
        mock_jm.start = AsyncMock()
        mock_jm.stop = AsyncMock()

        with (
            patch("backend.api.server.get_config", return_value=config),
            patch("backend.api.server.setup_logging"),
            patch("backend.api.server.Database", return_value=mock_db),
            patch("backend.api.server.DataEngine", return_value=mock_engine),
            patch(
                "backend.api.server._init_job_manager",
                new_callable=AsyncMock,
                return_value=mock_jm,
            ),
        ):
            async with lifespan(app):
                components = app.state.startup_components
                assert components["database"] == "ok"
                assert components["data_engine"].startswith("error")
                # Simulator skipped car engine a échoué
                assert components["simulator"] == "skipped"


# ─── Tests lifespan — shutdown protégé ────────────────────────────────────


class TestLifespanShutdown:
    """Vérifie que le shutdown est résilient aux erreurs."""

    @pytest.mark.asyncio
    async def test_shutdown_continues_after_component_failure(self):
        """Si un composant hang au shutdown, les suivants s'arrêtent quand même."""
        app = _make_app()

        mock_db = AsyncMock()
        mock_db.db_path = "data/test.db"
        mock_db.init = AsyncMock()
        mock_db.start_maintenance_loop = MagicMock()
        mock_db.close = AsyncMock()

        mock_jm = AsyncMock()
        mock_jm.start = AsyncMock()
        mock_jm.stop = AsyncMock()

        with (
            patch("backend.api.server.get_config", return_value=_mock_config()),
            patch("backend.api.server.setup_logging"),
            patch("backend.api.server.Database", return_value=mock_db),
            patch(
                "backend.api.server._init_job_manager",
                new_callable=AsyncMock,
                return_value=mock_jm,
            ),
        ):
            async with lifespan(app):
                pass
            # Si on arrive ici, le shutdown s'est terminé sans exception
            # Database.close() doit avoir été appelé
            mock_db.close.assert_awaited_once()


# ─── Tests health endpoint — composants ───────────────────────────────────


class TestHealthComponents:
    """Vérifie que /health expose le statut des composants."""

    @pytest.mark.asyncio
    async def test_health_shows_components(self):
        """Le endpoint /health inclut la clé 'components'."""
        from fastapi.testclient import TestClient
        from backend.api.server import app as real_app

        # Injecter un startup_components de test
        real_app.state.startup_components = {
            "database": "ok",
            "data_engine": "error: connection refused",
            "simulator": "skipped",
        }
        real_app.state.db = MagicMock()
        real_app.state.db._conn = None  # DB "disconnected"
        real_app.state.engine = None
        real_app.state.start_time = MagicMock()
        real_app.state.start_time.__rsub__ = MagicMock(
            return_value=MagicMock(total_seconds=MagicMock(return_value=42)),
        )
        real_app.state.watchdog = None
        real_app.state.executor = None

        client = TestClient(real_app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "components" in data
        assert data["components"]["database"] == "ok"
        assert data["components"]["data_engine"].startswith("error")
        # Dégradé car composant en erreur
        assert data["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_health_all_ok(self):
        """/health retourne 'ok' si tous les composants sont sains."""
        from fastapi.testclient import TestClient
        from backend.api.server import app as real_app

        real_app.state.startup_components = {
            "database": "ok",
            "data_engine": "ok",
            "simulator": "ok",
        }
        real_app.state.db = MagicMock()
        real_app.state.db._conn = MagicMock()  # DB connectée
        real_app.state.engine = None
        real_app.state.start_time = MagicMock()
        real_app.state.start_time.__rsub__ = MagicMock(
            return_value=MagicMock(total_seconds=MagicMock(return_value=100)),
        )
        real_app.state.watchdog = None
        real_app.state.executor = None

        client = TestClient(real_app, raise_server_exceptions=False)
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
