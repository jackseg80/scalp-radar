"""Tests pour migrate_optimization.py — Sprint 13."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.migrate_optimization import migrate_json_files


@pytest.fixture
def temp_env(tmp_path):
    """Environnement de test : DB + répertoire JSON."""
    db_path = str(tmp_path / "test.db")
    data_dir = tmp_path / "optimization"
    data_dir.mkdir()

    # Créer la table optimization_results
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
            win_rate_oos REAL,
            tail_risk_ratio REAL,
            UNIQUE(strategy_name, asset, timeframe, created_at)
        );
    """)
    conn.close()

    return {"db_path": db_path, "data_dir": data_dir}


@pytest.mark.asyncio
async def test_migrate_final_report(temp_env):
    """Test migration d'un JSON final complet."""
    data_dir = temp_env["data_dir"]
    db_path = temp_env["db_path"]

    # Créer un JSON de test
    report_data = {
        "strategy_name": "vwap_rsi",
        "symbol": "BTC/USDT",
        "timestamp": "2026-02-13T10:00:00",
        "grade": "A",
        "total_score": 87,
        "recommended_params": {"rsi_period": 14, "tp_percent": 0.8},
        "wfo": {
            "avg_is_sharpe": 2.0,
            "avg_oos_sharpe": 1.8,
            "consistency_rate": 0.85,
            "n_windows": 20,
            "oos_is_ratio": 0.90,
        },
        "overfitting": {
            "mc_p_value": 0.02,
            "mc_significant": True,
            "mc_underpowered": False,
            "dsr": 0.95,
            "dsr_max_expected_sharpe": 3.2,
            "stability": 0.88,
            "cliff_params": [],
            "convergence": 0.80,
            "divergent_params": [],
            "n_distinct_combos": 600,
        },
        "validation": {
            "bitget_sharpe": 1.5,
            "bitget_net_return_pct": 8.0,
            "bitget_trades": 25,
            "bitget_sharpe_ci_low": 0.8,
            "bitget_sharpe_ci_high": 2.1,
            "binance_oos_avg_sharpe": 1.7,
            "transfer_ratio": 0.88,
            "transfer_significant": True,
            "volume_warning": False,
            "volume_warning_detail": "",
        },
        "warnings": ["Test warning"],
    }

    json_path = data_dir / "vwap_rsi_BTC_USDT_20260213_100000.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f)

    # Migrer
    await migrate_json_files(data_dir=str(data_dir), dry_run=False, db_path=db_path)

    # Vérifier DB
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT strategy_name, asset, grade, total_score, is_latest FROM optimization_results")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "vwap_rsi"
    assert row[1] == "BTC/USDT"
    assert row[2] == "A"
    assert row[3] == 87
    assert row[4] == 1  # is_latest


@pytest.mark.asyncio
async def test_migrate_with_intermediate(temp_env):
    """Test migration avec fichier intermediate (wfo_windows)."""
    data_dir = temp_env["data_dir"]
    db_path = temp_env["db_path"]

    # Final report
    report_data = {
        "strategy_name": "vwap_rsi",
        "symbol": "BTC/USDT",
        "timestamp": "2026-02-13T10:00:00",
        "grade": "A",
        "total_score": 87,
        "recommended_params": {"rsi_period": 14},
        "wfo": {"avg_oos_sharpe": 1.8, "consistency_rate": 0.85, "n_windows": 2, "oos_is_ratio": 0.90},
        "overfitting": {"mc_p_value": 0.02, "dsr": 0.95, "stability": 0.88, "n_distinct_combos": 600},
        "validation": {"bitget_sharpe": 1.5, "transfer_ratio": 0.88},
        "warnings": [],
    }

    # Intermediate avec windows
    intermediate_data = {
        "windows": [
            {"window_index": 0, "is_sharpe": 2.0, "oos_sharpe": 1.9},
            {"window_index": 1, "is_sharpe": 2.1, "oos_sharpe": 1.7},
        ]
    }

    json_path = data_dir / "vwap_rsi_BTC_USDT_20260213_100000.json"
    inter_path = data_dir / "wfo_vwap_rsi_BTC_USDT_intermediate.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f)
    with open(inter_path, "w") as f:
        json.dump(intermediate_data, f)

    # Migrer
    await migrate_json_files(data_dir=str(data_dir), dry_run=False, db_path=db_path)

    # Vérifier que wfo_windows est présent
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT wfo_windows FROM optimization_results")
    row = cursor.fetchone()
    conn.close()

    assert row[0] is not None
    windows = json.loads(row[0])
    assert "windows" in windows
    assert len(windows["windows"]) == 2


@pytest.mark.asyncio
async def test_migrate_idempotent(temp_env):
    """Test que la migration est idempotente (2 runs → même résultat)."""
    data_dir = temp_env["data_dir"]
    db_path = temp_env["db_path"]

    report_data = {
        "strategy_name": "vwap_rsi",
        "symbol": "BTC/USDT",
        "timestamp": "2026-02-13T10:00:00",
        "grade": "A",
        "total_score": 87,
        "recommended_params": {},
        "wfo": {"avg_oos_sharpe": 1.8, "consistency_rate": 0.85, "n_windows": 20, "oos_is_ratio": 0.90},
        "overfitting": {"mc_p_value": 0.02, "dsr": 0.95, "stability": 0.88, "n_distinct_combos": 600},
        "validation": {"bitget_sharpe": 1.5, "transfer_ratio": 0.88},
        "warnings": [],
    }

    json_path = data_dir / "vwap_rsi_BTC_USDT_20260213_100000.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f)

    # 1ère migration
    await migrate_json_files(data_dir=str(data_dir), dry_run=False, db_path=db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT COUNT(*) FROM optimization_results")
    count1 = cursor.fetchone()[0]
    conn.close()

    # 2ème migration (même fichier)
    await migrate_json_files(data_dir=str(data_dir), dry_run=False, db_path=db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT COUNT(*) FROM optimization_results")
    count2 = cursor.fetchone()[0]
    conn.close()

    # Le count doit être identique (INSERT OR IGNORE)
    assert count1 == count2 == 1


@pytest.mark.asyncio
async def test_migrate_handles_missing_fields(temp_env):
    """Test que la migration gère les champs manquants avec .get() défensif."""
    data_dir = temp_env["data_dir"]
    db_path = temp_env["db_path"]

    # JSON incomplet (champs manquants)
    report_data = {
        "strategy_name": "vwap_rsi",
        "symbol": "BTC/USDT",
        "timestamp": "2026-02-13T10:00:00",
        "grade": "F",
        # total_score manquant → sera recalculé
        "recommended_params": {},
        "wfo": {"avg_oos_sharpe": -0.5, "n_windows": 10},
        "overfitting": {"mc_p_value": 0.95, "dsr": 0.10, "n_distinct_combos": 100},
        "validation": {},
        "warnings": [],
    }

    json_path = data_dir / "vwap_rsi_BTC_USDT_20260213_100000.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f)

    # La migration ne doit PAS crasher
    await migrate_json_files(data_dir=str(data_dir), dry_run=False, db_path=db_path)

    # Vérifier que total_score a été calculé
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT total_score FROM optimization_results")
    row = cursor.fetchone()
    conn.close()

    assert row[0] is not None
    assert row[0] >= 0  # Score recalculé


@pytest.mark.asyncio
async def test_migrate_dry_run(temp_env):
    """Test que --dry-run ne modifie pas la DB."""
    data_dir = temp_env["data_dir"]
    db_path = temp_env["db_path"]

    report_data = {
        "strategy_name": "vwap_rsi",
        "symbol": "BTC/USDT",
        "timestamp": "2026-02-13T10:00:00",
        "grade": "A",
        "total_score": 87,
        "recommended_params": {},
        "wfo": {"avg_oos_sharpe": 1.8, "n_windows": 20, "oos_is_ratio": 0.90, "consistency_rate": 0.85},
        "overfitting": {"mc_p_value": 0.02, "dsr": 0.95, "stability": 0.88, "n_distinct_combos": 600},
        "validation": {"bitget_sharpe": 1.5, "transfer_ratio": 0.88},
        "warnings": [],
    }

    json_path = data_dir / "vwap_rsi_BTC_USDT_20260213_100000.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f)

    # Dry-run
    await migrate_json_files(data_dir=str(data_dir), dry_run=True, db_path=db_path)

    # La DB doit être vide
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT COUNT(*) FROM optimization_results")
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 0
