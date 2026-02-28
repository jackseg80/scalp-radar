"""Tests pour --resume dans scripts/optimize.py (_get_done_assets)."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from scripts.optimize import _get_done_assets


@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    """Crée une DB SQLite minimale avec la table optimization_results."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE optimization_results (
            id INTEGER PRIMARY KEY,
            strategy_name TEXT,
            asset TEXT,
            is_latest INTEGER DEFAULT 0,
            win_rate_oos REAL,
            tail_risk_ratio REAL
        )
    """)
    conn.commit()
    conn.close()
    return db_path


def test_get_done_assets_empty_db(tmp_db: str) -> None:
    """DB sans résultats → set vide."""
    result = _get_done_assets("grid_range_atr", tmp_db)
    assert result == set()


def test_get_done_assets_with_results(tmp_db: str) -> None:
    """DB avec 3 résultats is_latest=1 → 3 assets retournés."""
    conn = sqlite3.connect(tmp_db)
    conn.executemany(
        "INSERT INTO optimization_results (strategy_name, asset, is_latest) VALUES (?, ?, ?)",
        [
            ("grid_range_atr", "BTC/USDT", 1),
            ("grid_range_atr", "ETH/USDT", 1),
            ("grid_range_atr", "SOL/USDT", 1),
        ],
    )
    conn.commit()
    conn.close()

    result = _get_done_assets("grid_range_atr", tmp_db)
    assert result == {"BTC/USDT", "ETH/USDT", "SOL/USDT"}


def test_get_done_assets_filters_by_strategy(tmp_db: str) -> None:
    """DB avec grid_atr et grid_range_atr → filtre correct par stratégie."""
    conn = sqlite3.connect(tmp_db)
    conn.executemany(
        "INSERT INTO optimization_results (strategy_name, asset, is_latest) VALUES (?, ?, ?)",
        [
            ("grid_atr", "BTC/USDT", 1),
            ("grid_atr", "ETH/USDT", 1),
            ("grid_range_atr", "DOGE/USDT", 1),
        ],
    )
    conn.commit()
    conn.close()

    atr_done = _get_done_assets("grid_atr", tmp_db)
    range_done = _get_done_assets("grid_range_atr", tmp_db)

    assert atr_done == {"BTC/USDT", "ETH/USDT"}
    assert range_done == {"DOGE/USDT"}
    assert atr_done.isdisjoint(range_done)


def test_get_done_assets_only_is_latest(tmp_db: str) -> None:
    """is_latest=0 ne doit pas apparaître dans le résultat."""
    conn = sqlite3.connect(tmp_db)
    conn.executemany(
        "INSERT INTO optimization_results (strategy_name, asset, is_latest) VALUES (?, ?, ?)",
        [
            ("grid_range_atr", "BTC/USDT", 1),   # is_latest=1 → inclus
            ("grid_range_atr", "ETH/USDT", 0),   # is_latest=0 → exclu
            ("grid_range_atr", "SOL/USDT", 0),   # is_latest=0 → exclu
        ],
    )
    conn.commit()
    conn.close()

    result = _get_done_assets("grid_range_atr", tmp_db)
    assert result == {"BTC/USDT"}
    assert "ETH/USDT" not in result
    assert "SOL/USDT" not in result


def test_get_done_assets_missing_db() -> None:
    """Chemin DB inexistant → set vide (pas d'exception)."""
    result = _get_done_assets("grid_range_atr", "/nonexistent/path.db")
    assert result == set()
