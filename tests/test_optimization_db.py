"""Tests pour optimization_db.py — Sprint 13."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from backend.optimization.optimization_db import (
    get_comparison_async,
    get_result_by_id_async,
    get_results_async,
    save_result_sync,
)
from backend.optimization.report import FinalReport, ValidationResult


@pytest.fixture
def temp_db(tmp_path):
    """DB temporaire avec la table optimization_results."""
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
            UNIQUE(strategy_name, asset, timeframe, created_at)
        );
    """)
    conn.close()
    return db_path


def test_save_result_sync_insert(temp_db):
    """Test insertion d'un résultat avec save_result_sync."""
    validation = ValidationResult(
        bitget_sharpe=1.5, bitget_net_return_pct=8.0, bitget_trades=25,
        bitget_sharpe_ci_low=0.8, bitget_sharpe_ci_high=2.1,
        binance_oos_avg_sharpe=1.3, transfer_ratio=0.85,
        transfer_significant=True, volume_warning=False, volume_warning_detail="",
    )
    report = FinalReport(
        strategy_name="vwap_rsi", symbol="BTC/USDT", timestamp=datetime(2026, 2, 13, 12, 0),
        grade="A", total_score=87, wfo_avg_is_sharpe=2.0, wfo_avg_oos_sharpe=1.7,
        wfo_consistency_rate=0.80, wfo_n_windows=20, recommended_params={"rsi_period": 14},
        mc_p_value=0.02, mc_significant=True, mc_underpowered=False, dsr=0.95,
        dsr_max_expected_sharpe=3.2, stability=0.88, cliff_params=[], convergence=0.75,
        divergent_params=[], validation=validation, oos_is_ratio=0.85, bitget_transfer=0.85,
        live_eligible=True, warnings=[], n_distinct_combos=600,
    )

    save_result_sync(temp_db, report, wfo_windows=None, duration=120.5, timeframe="5m")

    # Vérifier insertion
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("SELECT * FROM optimization_results")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[1] == "vwap_rsi"  # strategy_name
    assert row[2] == "BTC/USDT"  # asset
    assert row[3] == "5m"  # timeframe
    assert row[6] == "A"  # grade
    assert row[7] == 87  # total_score
    assert row[22] == 1  # is_latest (colonne 22, pas 21)


def test_save_result_sync_updates_is_latest(temp_db):
    """Test que save_result_sync met à jour is_latest correctement."""
    validation = ValidationResult(
        bitget_sharpe=1.0, bitget_net_return_pct=5.0, bitget_trades=20,
        bitget_sharpe_ci_low=0.5, bitget_sharpe_ci_high=1.5,
        binance_oos_avg_sharpe=1.0, transfer_ratio=0.70,
        transfer_significant=False, volume_warning=False, volume_warning_detail="",
    )

    # Premier résultat
    report1 = FinalReport(
        strategy_name="vwap_rsi", symbol="BTC/USDT", timestamp=datetime(2026, 2, 12, 10, 0),
        grade="B", total_score=72, wfo_avg_is_sharpe=1.5, wfo_avg_oos_sharpe=1.2,
        wfo_consistency_rate=0.70, wfo_n_windows=15, recommended_params={"rsi_period": 12},
        mc_p_value=0.05, mc_significant=True, mc_underpowered=False, dsr=0.80,
        dsr_max_expected_sharpe=3.0, stability=0.75, cliff_params=[], convergence=None,
        divergent_params=[], validation=validation, oos_is_ratio=0.75, bitget_transfer=0.70,
        live_eligible=True, warnings=[], n_distinct_combos=400,
    )
    save_result_sync(temp_db, report1, wfo_windows=None, duration=100.0, timeframe="5m")

    # Deuxième résultat (même stratégie/asset/timeframe)
    report2 = FinalReport(
        strategy_name="vwap_rsi", symbol="BTC/USDT", timestamp=datetime(2026, 2, 13, 10, 0),
        grade="A", total_score=87, wfo_avg_is_sharpe=2.0, wfo_avg_oos_sharpe=1.7,
        wfo_consistency_rate=0.85, wfo_n_windows=20, recommended_params={"rsi_period": 14},
        mc_p_value=0.02, mc_significant=True, mc_underpowered=False, dsr=0.95,
        dsr_max_expected_sharpe=3.2, stability=0.88, cliff_params=[], convergence=0.80,
        divergent_params=[], validation=validation, oos_is_ratio=0.90, bitget_transfer=0.85,
        live_eligible=True, warnings=[], n_distinct_combos=600,
    )
    save_result_sync(temp_db, report2, wfo_windows=None, duration=120.0, timeframe="5m")

    # Vérifier : seul le dernier a is_latest=1
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("SELECT grade, is_latest FROM optimization_results ORDER BY created_at")
    rows = cursor.fetchall()
    conn.close()

    assert len(rows) == 2
    assert rows[0][0] == "B"
    assert rows[0][1] == 0  # is_latest=0 pour l'ancien
    assert rows[1][0] == "A"
    assert rows[1][1] == 1  # is_latest=1 pour le nouveau


@pytest.mark.asyncio
async def test_get_results_async_all(temp_db):
    """Test get_results_async sans filtres."""
    # Insérer 2 résultats
    conn = sqlite3.connect(temp_db)
    conn.execute("""INSERT INTO optimization_results (
        strategy_name, asset, timeframe, created_at, duration_seconds,
        grade, total_score, oos_sharpe, consistency, oos_is_ratio, dsr,
        param_stability, monte_carlo_pvalue, mc_underpowered, n_windows, n_distinct_combos,
        best_params, wfo_windows, monte_carlo_summary, validation_summary, warnings, is_latest
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
        "vwap_rsi", "BTC/USDT", "5m", "2026-02-13T10:00:00", 120.0,
        "A", 87.0, 1.8, 0.85, 0.92, 0.95, 0.88, 0.02, 0, 20, 600,
        '{"rsi_period": 14}', None, '{}', '{}', '[]', 1
    ))
    conn.execute("""INSERT INTO optimization_results (
        strategy_name, asset, timeframe, created_at, duration_seconds,
        grade, total_score, oos_sharpe, consistency, oos_is_ratio, dsr,
        param_stability, monte_carlo_pvalue, mc_underpowered, n_windows, n_distinct_combos,
        best_params, wfo_windows, monte_carlo_summary, validation_summary, warnings, is_latest
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
        "envelope_dca", "ETH/USDT", "1h", "2026-02-13T11:00:00", 200.0,
        "B", 72.0, 1.2, 0.70, 0.80, 0.85, 0.75, 0.08, 0, 15, 400,
        '{"ma_period": 7}', None, '{}', '{}', '[]', 1
    ))
    conn.commit()
    conn.close()

    result = await get_results_async(temp_db, latest_only=True, offset=0, limit=50)

    assert result["total"] == 2
    assert len(result["results"]) == 2
    assert result["results"][0]["strategy_name"] in ["vwap_rsi", "envelope_dca"]


@pytest.mark.asyncio
async def test_get_results_async_filter_strategy(temp_db):
    """Test get_results_async avec filtre strategy."""
    conn = sqlite3.connect(temp_db)
    conn.execute("""INSERT INTO optimization_results (
        strategy_name, asset, timeframe, created_at, duration_seconds,
        grade, total_score, oos_sharpe, consistency, oos_is_ratio, dsr,
        param_stability, monte_carlo_pvalue, mc_underpowered, n_windows, n_distinct_combos,
        best_params, wfo_windows, monte_carlo_summary, validation_summary, warnings, is_latest
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
        "vwap_rsi", "BTC/USDT", "5m", "2026-02-13T10:00:00", 120.0,
        "A", 87.0, 1.8, 0.85, 0.92, 0.95, 0.88, 0.02, 0, 20, 600,
        '{}', None, '{}', '{}', '[]', 1
    ))
    conn.execute("""INSERT INTO optimization_results (
        strategy_name, asset, timeframe, created_at, duration_seconds,
        grade, total_score, oos_sharpe, consistency, oos_is_ratio, dsr,
        param_stability, monte_carlo_pvalue, mc_underpowered, n_windows, n_distinct_combos,
        best_params, wfo_windows, monte_carlo_summary, validation_summary, warnings, is_latest
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
        "envelope_dca", "ETH/USDT", "1h", "2026-02-13T11:00:00", 200.0,
        "B", 72.0, 1.2, 0.70, 0.80, 0.85, 0.75, 0.08, 0, 15, 400,
        '{}', None, '{}', '{}', '[]', 1
    ))
    conn.commit()
    conn.close()

    result = await get_results_async(temp_db, strategy="vwap_rsi", latest_only=True)

    assert result["total"] == 1
    assert result["results"][0]["strategy_name"] == "vwap_rsi"


@pytest.mark.asyncio
async def test_get_results_async_pagination(temp_db):
    """Test get_results_async avec pagination."""
    conn = sqlite3.connect(temp_db)
    for i in range(5):
        conn.execute("""INSERT INTO optimization_results (
            strategy_name, asset, timeframe, created_at, duration_seconds,
            grade, total_score, oos_sharpe, consistency, oos_is_ratio, dsr,
            param_stability, monte_carlo_pvalue, mc_underpowered, n_windows, n_distinct_combos,
            best_params, wfo_windows, monte_carlo_summary, validation_summary, warnings, is_latest
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            f"strategy_{i}", "BTC/USDT", "5m", f"2026-02-13T{i:02d}:00:00", 100.0,
            "A", 80 + i, 1.5, 0.75, 0.80, 0.85, 0.80, 0.05, 0, 20, 500,
            '{}', None, '{}', '{}', '[]', 1
        ))
    conn.commit()
    conn.close()

    result = await get_results_async(temp_db, offset=2, limit=2, latest_only=False)

    assert result["total"] == 5
    assert len(result["results"]) == 2


@pytest.mark.asyncio
async def test_get_result_by_id_async(temp_db):
    """Test get_result_by_id_async avec JSON parsés."""
    conn = sqlite3.connect(temp_db)
    conn.execute("""INSERT INTO optimization_results (
        strategy_name, asset, timeframe, created_at, duration_seconds,
        grade, total_score, oos_sharpe, consistency, oos_is_ratio, dsr,
        param_stability, monte_carlo_pvalue, mc_underpowered, n_windows, n_distinct_combos,
        best_params, wfo_windows, monte_carlo_summary, validation_summary, warnings, is_latest
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
        "vwap_rsi", "BTC/USDT", "5m", "2026-02-13T10:00:00", 120.0,
        "A", 87.0, 1.8, 0.85, 0.92, 0.95, 0.88, 0.02, 0, 20, 600,
        '{"rsi_period": 14}',
        '{"windows": [{"window_index": 0, "is_sharpe": 2.0}]}',
        '{"p_value": 0.02, "significant": true}',
        '{"bitget_sharpe": 1.5}',
        '["Test warning"]',
        1
    ))
    conn.commit()
    conn.close()

    result = await get_result_by_id_async(temp_db, 1)

    assert result is not None
    assert result["strategy_name"] == "vwap_rsi"
    assert isinstance(result["best_params"], dict)
    assert result["best_params"]["rsi_period"] == 14
    assert isinstance(result["wfo_windows"], list)
    assert len(result["wfo_windows"]) == 1
    assert isinstance(result["monte_carlo_summary"], dict)
    assert result["monte_carlo_summary"]["p_value"] == 0.02
    assert isinstance(result["warnings"], list)
    assert result["warnings"][0] == "Test warning"


@pytest.mark.asyncio
async def test_get_result_by_id_not_found(temp_db):
    """Test get_result_by_id_async retourne None si inexistant."""
    result = await get_result_by_id_async(temp_db, 999)
    assert result is None


@pytest.mark.asyncio
async def test_get_comparison_async(temp_db):
    """Test get_comparison_async : matrice strategies × assets."""
    conn = sqlite3.connect(temp_db)
    conn.execute("""INSERT INTO optimization_results (
        strategy_name, asset, timeframe, created_at, duration_seconds,
        grade, total_score, oos_sharpe, consistency, oos_is_ratio, dsr,
        param_stability, monte_carlo_pvalue, mc_underpowered, n_windows, n_distinct_combos,
        best_params, wfo_windows, monte_carlo_summary, validation_summary, warnings, is_latest
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
        "vwap_rsi", "BTC/USDT", "5m", "2026-02-13T10:00:00", 120.0,
        "A", 87.0, 1.8, 0.85, 0.92, 0.95, 0.88, 0.02, 0, 20, 600,
        '{}', None, '{}', '{}', '[]', 1
    ))
    conn.execute("""INSERT INTO optimization_results (
        strategy_name, asset, timeframe, created_at, duration_seconds,
        grade, total_score, oos_sharpe, consistency, oos_is_ratio, dsr,
        param_stability, monte_carlo_pvalue, mc_underpowered, n_windows, n_distinct_combos,
        best_params, wfo_windows, monte_carlo_summary, validation_summary, warnings, is_latest
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
        "envelope_dca", "ETH/USDT", "1h", "2026-02-13T11:00:00", 200.0,
        "B", 72.0, 1.2, 0.70, 0.80, 0.85, 0.75, 0.08, 0, 15, 400,
        '{}', None, '{}', '{}', '[]', 1
    ))
    conn.commit()
    conn.close()

    result = await get_comparison_async(temp_db)

    assert "strategies" in result
    assert "assets" in result
    assert "matrix" in result
    assert len(result["strategies"]) == 2
    assert len(result["assets"]) == 2
    assert "vwap_rsi" in result["strategies"]
    assert "envelope_dca" in result["strategies"]
    assert "BTC/USDT" in result["assets"]
    assert "ETH/USDT" in result["assets"]
    assert result["matrix"]["vwap_rsi"]["BTC/USDT"]["grade"] == "A"
    assert result["matrix"]["envelope_dca"]["ETH/USDT"]["grade"] == "B"


def test_save_result_sync_special_values(temp_db):
    """Test que NaN/Infinity sont sanitizés en None."""
    validation = ValidationResult(
        bitget_sharpe=float('nan'), bitget_net_return_pct=float('inf'), bitget_trades=10,
        bitget_sharpe_ci_low=0.0, bitget_sharpe_ci_high=0.0,
        binance_oos_avg_sharpe=1.0, transfer_ratio=0.5,
        transfer_significant=False, volume_warning=False, volume_warning_detail="",
    )
    report = FinalReport(
        strategy_name="test", symbol="BTC/USDT", timestamp=datetime(2026, 2, 13, 12, 0),
        grade="F", total_score=15, wfo_avg_is_sharpe=float('nan'), wfo_avg_oos_sharpe=float('-inf'),
        wfo_consistency_rate=0.20, wfo_n_windows=10, recommended_params={},
        mc_p_value=1.0, mc_significant=False, mc_underpowered=True, dsr=0.0,
        dsr_max_expected_sharpe=3.0, stability=0.30, cliff_params=[], convergence=None,
        divergent_params=[], validation=validation, oos_is_ratio=0.10, bitget_transfer=0.20,
        live_eligible=False, warnings=["Test"], n_distinct_combos=100,
    )

    # Ne doit pas crasher avec NaN/Infinity
    save_result_sync(temp_db, report, wfo_windows=None, duration=None, timeframe="5m")

    # Vérifier que les valeurs ont été sanitizées
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("SELECT oos_sharpe FROM optimization_results")
    row = cursor.fetchone()
    conn.close()

    assert row[0] is None  # NaN → None
