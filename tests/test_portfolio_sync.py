"""Tests pour la sync portfolio backtests local → serveur.

Couvre :
- build_portfolio_payload_from_row()
- save_portfolio_from_payload_sync() (insert + dedup)
- push_portfolio_to_server() (best-effort, mock httpx)
- POST /api/portfolio/results (auth + insert + dedup)
- sync_to_server.py --only portfolio
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.server import app
from backend.backtesting.portfolio_db import (
    build_portfolio_payload_from_row,
    push_portfolio_to_server,
    save_portfolio_from_payload_sync,
    save_result_sync,
)
from backend.backtesting.portfolio_engine import PortfolioResult, PortfolioSnapshot
from backend.core.database import Database


# ─── Fixtures ────────────────────────────────────────────────────────────


def _make_snapshot(ts_offset_hours: int, equity: float) -> PortfolioSnapshot:
    ts = datetime(2025, 6, 1, tzinfo=timezone.utc) + __import__("datetime").timedelta(hours=ts_offset_hours)
    return PortfolioSnapshot(
        timestamp=ts,
        total_equity=equity,
        total_capital=equity * 0.8,
        total_realized_pnl=equity - 10000,
        total_unrealized_pnl=equity * 0.2 - 2000,
        total_margin_used=equity * 0.4,
        margin_ratio=0.4,
        n_open_positions=5,
        n_assets_with_positions=3,
    )


def _make_result(n_snapshots: int = 10) -> PortfolioResult:
    snapshots = [_make_snapshot(i, 10000 + i * 10) for i in range(n_snapshots)]
    return PortfolioResult(
        initial_capital=10000.0,
        n_assets=5,
        period_days=90,
        assets=["AAA/USDT", "BBB/USDT", "CCC/USDT", "DDD/USDT", "EEE/USDT"],
        final_equity=10100.0,
        total_return_pct=1.0,
        total_trades=42,
        win_rate=55.5,
        realized_pnl=80.0,
        force_closed_pnl=20.0,
        max_drawdown_pct=-5.2,
        max_drawdown_date=datetime(2025, 7, 15, tzinfo=timezone.utc),
        max_drawdown_duration_hours=48.0,
        peak_margin_ratio=0.65,
        peak_open_positions=12,
        peak_concurrent_assets=5,
        kill_switch_triggers=1,
        kill_switch_events=[{"timestamp": "2025-07-15T12:00:00+00:00", "drawdown_pct": 31.2}],
        snapshots=snapshots,
        per_asset_results={
            "AAA/USDT": {"trades": 10, "wins": 6, "win_rate": 60.0, "net_pnl": 25.5},
            "BBB/USDT": {"trades": 8, "wins": 3, "win_rate": 37.5, "net_pnl": -10.0},
        },
    )


@pytest.fixture
def db_path(tmp_path):
    """Crée une DB temporaire avec les tables portfolio."""
    path = str(tmp_path / "test.db")

    async def _init():
        db = Database(path)
        await db.init()
        await db.close()

    asyncio.run(_init())
    return path


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# ─── build_portfolio_payload_from_row ────────────────────────────────────


def test_build_payload_from_row(db_path):
    """Construit un payload correct depuis une row DB."""
    result = _make_result()
    result_id = save_result_sync(db_path, result, strategy_name="grid_atr")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM portfolio_backtests WHERE id = ?", (result_id,)
    ).fetchone()
    conn.close()

    payload = build_portfolio_payload_from_row(dict(row))
    assert payload["strategy_name"] == "grid_atr"
    assert payload["initial_capital"] == 10000.0
    assert payload["n_assets"] == 5
    assert payload["total_return_pct"] == 1.0
    assert payload["created_at"] is not None
    assert payload["source"] == "local"
    # Les JSON blobs restent en string
    assert isinstance(payload["equity_curve"], str)
    assert isinstance(payload["per_asset_results"], str)


# ─── save_portfolio_from_payload_sync ────────────────────────────────────


def test_save_from_payload_creates(db_path):
    """Insert un portfolio backtest depuis un payload JSON."""
    payload = {
        "strategy_name": "grid_atr",
        "initial_capital": 10000.0,
        "n_assets": 5,
        "period_days": 90,
        "assets": json.dumps(["AAA/USDT", "BBB/USDT"]),
        "exchange": "binance",
        "kill_switch_pct": 30.0,
        "kill_switch_window_hours": 24,
        "final_equity": 10500.0,
        "total_return_pct": 5.0,
        "total_trades": 50,
        "win_rate": 60.0,
        "realized_pnl": 400.0,
        "force_closed_pnl": 100.0,
        "max_drawdown_pct": -8.0,
        "max_drawdown_date": "2025-07-15T12:00:00+00:00",
        "max_drawdown_duration_hours": 36.0,
        "peak_margin_ratio": 0.55,
        "peak_open_positions": 8,
        "peak_concurrent_assets": 4,
        "kill_switch_triggers": 0,
        "kill_switch_events": json.dumps([]),
        "equity_curve": json.dumps([{"t": 0, "e": 10000}]),
        "per_asset_results": json.dumps({"AAA/USDT": {"trades": 25}}),
        "created_at": "2026-02-18T10:00:00+00:00",
        "duration_seconds": 42.5,
        "label": "test sync",
    }

    status = save_portfolio_from_payload_sync(db_path, payload)
    assert status == "created"

    # Vérifier en DB
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM portfolio_backtests WHERE label = 'test sync'").fetchone()
    conn.close()
    assert row is not None
    assert row["strategy_name"] == "grid_atr"
    assert row["total_return_pct"] == 5.0


def test_save_from_payload_dedup(db_path):
    """Doublon détecté par created_at → retourne 'exists'."""
    payload = {
        "strategy_name": "grid_atr",
        "initial_capital": 10000.0,
        "n_assets": 3,
        "period_days": 90,
        "assets": json.dumps(["A/USDT"]),
        "final_equity": 10200.0,
        "total_return_pct": 2.0,
        "total_trades": 20,
        "win_rate": 55.0,
        "realized_pnl": 150.0,
        "force_closed_pnl": 50.0,
        "max_drawdown_pct": -3.0,
        "max_drawdown_duration_hours": 12.0,
        "peak_margin_ratio": 0.4,
        "peak_open_positions": 6,
        "peak_concurrent_assets": 3,
        "equity_curve": json.dumps([]),
        "per_asset_results": json.dumps({}),
        "created_at": "2026-02-18T12:00:00+00:00",
    }

    status1 = save_portfolio_from_payload_sync(db_path, payload)
    assert status1 == "created"

    status2 = save_portfolio_from_payload_sync(db_path, payload)
    assert status2 == "exists"


# ─── push_portfolio_to_server ────────────────────────────────────────────


def test_push_disabled_when_sync_off():
    """Ne pousse rien si sync_enabled=false."""
    mock_config = MagicMock()
    mock_config.secrets.sync_enabled = False

    with patch("backend.core.config.get_config", return_value=mock_config):
        with patch("httpx.Client") as mock_httpx:
            push_portfolio_to_server({"strategy_name": "test"})
            mock_httpx.assert_not_called()


def test_push_sends_post():
    """Envoie bien un POST avec le bon URL et headers."""
    mock_config = MagicMock()
    mock_config.secrets.sync_enabled = True
    mock_config.secrets.sync_server_url = "http://192.168.1.200:8000"
    mock_config.secrets.sync_api_key = "secret123"

    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {"status": "created"}

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_resp

    row = {
        "strategy_name": "grid_atr",
        "initial_capital": 10000,
        "n_assets": 5,
        "period_days": 90,
        "assets": json.dumps(["A/USDT"]),
        "final_equity": 10500,
        "total_return_pct": 5.0,
        "total_trades": 30,
        "win_rate": 55.0,
        "realized_pnl": 400,
        "force_closed_pnl": 100,
        "max_drawdown_pct": -8.0,
        "max_drawdown_duration_hours": 24.0,
        "peak_margin_ratio": 0.6,
        "peak_open_positions": 10,
        "peak_concurrent_assets": 5,
        "equity_curve": json.dumps([]),
        "per_asset_results": json.dumps({}),
        "created_at": "2026-02-18T10:00:00",
    }

    with patch("backend.core.config.get_config", return_value=mock_config):
        with patch("httpx.Client", return_value=mock_client):
            push_portfolio_to_server(row)

    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert "/api/portfolio/results" in call_args[0][0]
    assert call_args[1]["headers"]["X-API-Key"] == "secret123"


def test_push_never_crashes():
    """Erreur réseau ne remonte pas — best-effort."""
    mock_config = MagicMock()
    mock_config.secrets.sync_enabled = True
    mock_config.secrets.sync_server_url = "http://192.168.1.200:8000"
    mock_config.secrets.sync_api_key = "secret"

    with patch("backend.core.config.get_config", return_value=mock_config):
        with patch("httpx.Client", side_effect=Exception("Connection refused")):
            # Ne doit PAS lever d'exception
            push_portfolio_to_server({"strategy_name": "test", "created_at": "2026-01-01"})


# ─── POST /api/portfolio/results ─────────────────────────────────────────


@patch("backend.api.portfolio_routes.save_portfolio_from_payload_sync")
@patch("backend.api.portfolio_routes.get_config")
def test_post_portfolio_result_created(mock_config, mock_save, client):
    """POST /api/portfolio/results → 201 si inséré."""
    mock_cfg = MagicMock()
    mock_cfg.secrets.sync_api_key = "secret"
    mock_cfg.secrets.database_url = "sqlite:///data/test.db"
    mock_config.return_value = mock_cfg
    mock_save.return_value = "created"

    payload = {
        "strategy_name": "grid_atr",
        "initial_capital": 10000,
        "n_assets": 5,
        "period_days": 90,
        "assets": json.dumps(["A/USDT"]),
        "final_equity": 10500,
        "total_return_pct": 5.0,
        "total_trades": 30,
        "win_rate": 55.0,
        "equity_curve": json.dumps([]),
        "per_asset_results": json.dumps({}),
        "created_at": "2026-02-18T10:00:00",
    }

    resp = client.post(
        "/api/portfolio/results",
        json=payload,
        headers={"X-API-Key": "secret"},
    )
    assert resp.status_code == 201
    assert resp.json()["status"] == "created"


@patch("backend.api.portfolio_routes.save_portfolio_from_payload_sync")
@patch("backend.api.portfolio_routes.get_config")
def test_post_portfolio_result_exists(mock_config, mock_save, client):
    """POST /api/portfolio/results → 200 si doublon."""
    mock_cfg = MagicMock()
    mock_cfg.secrets.sync_api_key = "secret"
    mock_cfg.secrets.database_url = "sqlite:///data/test.db"
    mock_config.return_value = mock_cfg
    mock_save.return_value = "exists"

    payload = {
        "strategy_name": "grid_atr",
        "initial_capital": 10000,
        "n_assets": 5,
        "period_days": 90,
        "assets": json.dumps(["A/USDT"]),
        "final_equity": 10500,
        "total_return_pct": 5.0,
        "total_trades": 30,
        "win_rate": 55.0,
        "equity_curve": json.dumps([]),
        "per_asset_results": json.dumps({}),
        "created_at": "2026-02-18T10:00:00",
    }

    resp = client.post(
        "/api/portfolio/results",
        json=payload,
        headers={"X-API-Key": "secret"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "already_exists"


@patch("backend.api.portfolio_routes.get_config")
def test_post_portfolio_result_no_auth(mock_config, client):
    """POST /api/portfolio/results → 401 sans clé."""
    mock_cfg = MagicMock()
    mock_cfg.secrets.sync_api_key = "secret"
    mock_config.return_value = mock_cfg

    resp = client.post("/api/portfolio/results", json={"strategy_name": "x"})
    assert resp.status_code == 401


@patch("backend.api.portfolio_routes.get_config")
def test_post_portfolio_result_bad_key(mock_config, client):
    """POST /api/portfolio/results → 401 mauvaise clé."""
    mock_cfg = MagicMock()
    mock_cfg.secrets.sync_api_key = "secret"
    mock_config.return_value = mock_cfg

    resp = client.post(
        "/api/portfolio/results",
        json={"strategy_name": "x"},
        headers={"X-API-Key": "wrong"},
    )
    assert resp.status_code == 401


@patch("backend.api.portfolio_routes.get_config")
def test_post_portfolio_result_missing_fields(mock_config, client):
    """POST /api/portfolio/results → 422 si champs manquants."""
    mock_cfg = MagicMock()
    mock_cfg.secrets.sync_api_key = "secret"
    mock_config.return_value = mock_cfg

    resp = client.post(
        "/api/portfolio/results",
        json={"strategy_name": "grid_atr"},
        headers={"X-API-Key": "secret"},
    )
    assert resp.status_code == 422
    assert "manquants" in resp.json()["detail"]


# ─── sync_to_server.py --only portfolio ──────────────────────────────────


def test_sync_portfolio_dry_run(monkeypatch, db_path, capsys):
    """Mode --dry-run portfolio affiche sans envoyer."""
    # Insérer un backtest en DB
    result = _make_result()
    save_result_sync(db_path, result, strategy_name="grid_atr", label="sync test")

    mock_config = MagicMock()
    mock_config.secrets.sync_server_url = "http://192.168.1.200:8000"
    mock_config.secrets.sync_api_key = "secret"
    mock_config.secrets.database_url = f"sqlite:///{db_path}"

    monkeypatch.setattr("scripts.sync_to_server.get_config", lambda: mock_config)
    monkeypatch.setattr("sys.argv", ["sync_to_server", "--dry-run", "--only", "portfolio"])

    from scripts.sync_to_server import main

    mock_client_instance = MagicMock()
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("httpx.Client", return_value=mock_client_instance):
        main()

    # En dry-run, aucun POST n'est envoyé
    mock_client_instance.post.assert_not_called()

    captured = capsys.readouterr()
    assert "dry-run" in captured.out.lower()
    assert "grid_atr" in captured.out


def test_sync_portfolio_sends_post(monkeypatch, db_path):
    """Sync portfolio envoie un POST par backtest."""
    result = _make_result()
    save_result_sync(db_path, result, strategy_name="grid_atr")

    mock_config = MagicMock()
    mock_config.secrets.sync_server_url = "http://192.168.1.200:8000"
    mock_config.secrets.sync_api_key = "secret"
    mock_config.secrets.database_url = f"sqlite:///{db_path}"

    monkeypatch.setattr("scripts.sync_to_server.get_config", lambda: mock_config)
    monkeypatch.setattr("sys.argv", ["sync_to_server", "--only", "portfolio"])

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"status": "created"}

    mock_client_instance = MagicMock()
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)
    mock_client_instance.post.return_value = mock_response

    from scripts.sync_to_server import main

    with patch("httpx.Client", return_value=mock_client_instance):
        main()

    assert mock_client_instance.post.call_count == 1
    call_url = mock_client_instance.post.call_args[0][0]
    assert "/api/portfolio/results" in call_url
