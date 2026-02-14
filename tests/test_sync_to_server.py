"""Tests pour scripts/sync_to_server.py."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from scripts.sync_to_server import _load_all_results, main


@pytest.fixture
def populated_db(tmp_path):
    """DB avec 2 résultats pour tester le sync."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE optimization_results (
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
    """)
    for i, (strat, asset) in enumerate([("vwap_rsi", "BTC/USDT"), ("envelope_dca", "ETH/USDT")]):
        conn.execute("""INSERT INTO optimization_results (
            strategy_name, asset, timeframe, created_at,
            grade, total_score, n_windows, best_params, is_latest
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            strat, asset, "5m", f"2026-02-13T{10 + i}:00:00",
            "A", 80 + i, 20, '{"p": 1}', 1,
        ))
    conn.commit()
    conn.close()
    return db_path


def test_load_all_results(populated_db):
    """Charge tous les résultats de la DB."""
    results = _load_all_results(populated_db)
    assert len(results) == 2
    assert results[0]["strategy_name"] == "vwap_rsi"
    assert results[1]["strategy_name"] == "envelope_dca"


def test_sync_dry_run(monkeypatch, populated_db, capsys):
    """Mode --dry-run n'envoie rien."""
    mock_config = MagicMock()
    mock_config.secrets.sync_server_url = "http://192.168.1.200:8000"
    mock_config.secrets.sync_api_key = "secret"
    mock_config.secrets.database_url = f"sqlite:///{populated_db}"

    monkeypatch.setattr("scripts.sync_to_server.get_config", lambda: mock_config)
    monkeypatch.setattr("sys.argv", ["sync_to_server", "--dry-run"])

    with patch("httpx.Client") as mock_client:
        main()
        mock_client.assert_not_called()

    captured = capsys.readouterr()
    assert "dry-run" in captured.out.lower()
    assert "2" in captured.out  # 2 résultats affichés


def test_sync_sends_all_results(monkeypatch, populated_db):
    """Sync envoie N POST = N rows en DB."""
    mock_config = MagicMock()
    mock_config.secrets.sync_server_url = "http://192.168.1.200:8000"
    mock_config.secrets.sync_api_key = "secret"
    mock_config.secrets.database_url = f"sqlite:///{populated_db}"

    monkeypatch.setattr("scripts.sync_to_server.get_config", lambda: mock_config)
    monkeypatch.setattr("sys.argv", ["sync_to_server"])

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"status": "created"}

    mock_client_instance = MagicMock()
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)
    mock_client_instance.post.return_value = mock_response

    with patch("httpx.Client", return_value=mock_client_instance):
        main()

    assert mock_client_instance.post.call_count == 2


def test_sync_handles_errors(monkeypatch, populated_db, capsys):
    """Une erreur sur 1 résultat ne bloque pas les suivants."""
    import httpx

    mock_config = MagicMock()
    mock_config.secrets.sync_server_url = "http://192.168.1.200:8000"
    mock_config.secrets.sync_api_key = "secret"
    mock_config.secrets.database_url = f"sqlite:///{populated_db}"

    monkeypatch.setattr("scripts.sync_to_server.get_config", lambda: mock_config)
    monkeypatch.setattr("sys.argv", ["sync_to_server"])

    mock_response_ok = MagicMock()
    mock_response_ok.status_code = 201
    mock_response_ok.json.return_value = {"status": "created"}

    mock_client_instance = MagicMock()
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)
    # 1er appel échoue, 2ème réussit
    mock_client_instance.post.side_effect = [
        httpx.ConnectError("Connection refused"),
        mock_response_ok,
    ]

    with patch("httpx.Client", return_value=mock_client_instance):
        main()

    # Les 2 POST ont été tentés (pas d'arrêt sur erreur)
    assert mock_client_instance.post.call_count == 2

    captured = capsys.readouterr()
    assert "1" in captured.out  # Au moins 1 créé dans le récap
