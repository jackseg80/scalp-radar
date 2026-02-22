"""Tests pour optimization_db.py — Sprint 13 + sync."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from backend.optimization.optimization_db import (
    build_payload_from_db_row,
    build_push_payload,
    get_comparison_async,
    get_result_by_id_async,
    get_results_async,
    save_result_from_payload_sync,
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
            source TEXT DEFAULT 'local',
            regime_analysis TEXT,
            UNIQUE(strategy_name, asset, timeframe, created_at)
        );
        CREATE TABLE IF NOT EXISTS wfo_combo_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            optimization_result_id INTEGER NOT NULL,
            params TEXT NOT NULL,
            oos_sharpe REAL,
            oos_return_pct REAL,
            oos_trades INTEGER,
            oos_win_rate REAL,
            is_sharpe REAL,
            is_return_pct REAL,
            is_trades INTEGER,
            consistency REAL,
            oos_is_ratio REAL,
            is_best INTEGER DEFAULT 0,
            n_windows_evaluated INTEGER,
            per_window_sharpes TEXT,
            FOREIGN KEY (optimization_result_id) REFERENCES optimization_results(id)
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
    assert row[22] == 1  # is_latest
    assert row[23] == "local"  # source


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


def test_save_result_sync_small_combos_no_steal_is_latest(temp_db):
    """Un run avec < 10 combos ne doit PAS voler is_latest d'un run complet."""
    validation = ValidationResult(
        bitget_sharpe=1.0, bitget_net_return_pct=5.0, bitget_trades=20,
        bitget_sharpe_ci_low=0.5, bitget_sharpe_ci_high=1.5,
        binance_oos_avg_sharpe=1.0, transfer_ratio=0.70,
        transfer_significant=False, volume_warning=False, volume_warning_detail="",
    )

    # Run complet (300+ combos)
    report_full = FinalReport(
        strategy_name="envelope_dca", symbol="BTC/USDT", timestamp=datetime(2026, 2, 15, 1, 0),
        grade="C", total_score=60, wfo_avg_is_sharpe=3.6, wfo_avg_oos_sharpe=3.2,
        wfo_consistency_rate=0.40, wfo_n_windows=30, recommended_params={"ma_period": 8},
        mc_p_value=0.10, mc_significant=False, mc_underpowered=False, dsr=0.52,
        dsr_max_expected_sharpe=3.0, stability=0.78, cliff_params=[], convergence=None,
        divergent_params=[], validation=validation, oos_is_ratio=0.89, bitget_transfer=0.70,
        live_eligible=False, warnings=[], n_distinct_combos=328,
    )
    save_result_sync(temp_db, report_full, wfo_windows=None, duration=600.0, timeframe="1h")

    # Run Explorer (2 combos) — ne doit PAS voler is_latest
    report_small = FinalReport(
        strategy_name="envelope_dca", symbol="BTC/USDT", timestamp=datetime(2026, 2, 15, 2, 0),
        grade="A", total_score=85, wfo_avg_is_sharpe=3.6, wfo_avg_oos_sharpe=3.19,
        wfo_consistency_rate=0.37, wfo_n_windows=30, recommended_params={"ma_period": 7},
        mc_p_value=0.02, mc_significant=True, mc_underpowered=False, dsr=1.0,
        dsr_max_expected_sharpe=3.0, stability=0.83, cliff_params=[], convergence=None,
        divergent_params=[], validation=validation, oos_is_ratio=0.75, bitget_transfer=0.70,
        live_eligible=True, warnings=[], n_distinct_combos=2,
    )
    save_result_sync(temp_db, report_small, wfo_windows=None, duration=30.0, timeframe="1h")

    # Vérifier : le run complet garde is_latest=1, le petit run a is_latest=0
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute(
        "SELECT grade, total_score, n_distinct_combos, is_latest "
        "FROM optimization_results ORDER BY created_at"
    )
    rows = cursor.fetchall()
    conn.close()

    assert len(rows) == 2
    assert rows[0] == ("C", 60, 328, 1)  # run complet garde is_latest=1
    assert rows[1] == ("A", 85, 2, 0)    # petit run a is_latest=0


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


# ─── Tests sync local → serveur ──────────────────────────────────────────


def _make_sample_payload(**overrides) -> dict:
    """Construit un payload POST de test."""
    base = {
        "strategy_name": "vwap_rsi",
        "asset": "BTC/USDT",
        "timeframe": "5m",
        "created_at": "2026-02-13T12:00:00",
        "duration_seconds": 120.0,
        "grade": "A",
        "total_score": 87.0,
        "oos_sharpe": 1.8,
        "consistency": 0.85,
        "oos_is_ratio": 0.92,
        "dsr": 0.95,
        "param_stability": 0.88,
        "monte_carlo_pvalue": 0.02,
        "mc_underpowered": 0,
        "n_windows": 20,
        "n_distinct_combos": 600,
        "best_params": '{"rsi_period": 14}',
        "wfo_windows": None,
        "monte_carlo_summary": '{"p_value": 0.02}',
        "validation_summary": '{"bitget_sharpe": 1.5}',
        "warnings": '[]',
        "source": "local",
    }
    base.update(overrides)
    return base


def test_save_result_from_payload_sync_created(temp_db):
    """Test insertion via payload dict → retourne 'created'."""
    payload = _make_sample_payload()
    status = save_result_from_payload_sync(temp_db, payload)
    assert status == "created"

    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT * FROM optimization_results")
    row = dict(cursor.fetchone())
    conn.close()

    assert row["strategy_name"] == "vwap_rsi"
    assert row["grade"] == "A"
    assert row["is_latest"] == 1
    assert row["source"] == "local"


def test_save_result_from_payload_sync_duplicate(temp_db):
    """Test doublon (même UNIQUE key) → retourne 'exists', is_latest intact."""
    payload = _make_sample_payload()

    status1 = save_result_from_payload_sync(temp_db, payload)
    assert status1 == "created"

    status2 = save_result_from_payload_sync(temp_db, payload)
    assert status2 == "exists"

    # Vérifier qu'il n'y a qu'une seule row et que is_latest est toujours 1
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("SELECT COUNT(*) FROM optimization_results")
    count = cursor.fetchone()[0]
    cursor = conn.execute("SELECT is_latest FROM optimization_results")
    row = cursor.fetchone()
    conn.close()

    assert count == 1
    assert row[0] == 1  # is_latest préservé


def test_save_result_from_payload_sync_updates_is_latest(temp_db):
    """Test que l'insertion d'un nouveau run met à jour is_latest des anciens."""
    payload1 = _make_sample_payload(created_at="2026-02-12T10:00:00", grade="B", total_score=72)
    payload2 = _make_sample_payload(created_at="2026-02-13T10:00:00", grade="A", total_score=87)

    save_result_from_payload_sync(temp_db, payload1)
    save_result_from_payload_sync(temp_db, payload2)

    conn = sqlite3.connect(temp_db)
    cursor = conn.execute(
        "SELECT grade, is_latest FROM optimization_results ORDER BY created_at"
    )
    rows = cursor.fetchall()
    conn.close()

    assert len(rows) == 2
    assert rows[0] == ("B", 0)  # ancien → is_latest=0
    assert rows[1] == ("A", 1)  # nouveau → is_latest=1


def test_save_result_from_payload_sync_json_as_dict(temp_db):
    """Test que le payload peut contenir des JSON blobs comme dicts (pas seulement strings)."""
    payload = _make_sample_payload(
        best_params={"rsi_period": 14, "tp_percent": 0.8},
        monte_carlo_summary={"p_value": 0.02, "significant": True},
        validation_summary={"bitget_sharpe": 1.5},
        warnings=["Test warning"],
    )
    status = save_result_from_payload_sync(temp_db, payload)
    assert status == "created"

    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("SELECT best_params, warnings FROM optimization_results")
    row = cursor.fetchone()
    conn.close()

    # Les dicts/lists sont sérialisés en JSON strings en DB
    assert json.loads(row[0]) == {"rsi_period": 14, "tp_percent": 0.8}
    assert json.loads(row[1]) == ["Test warning"]


def test_save_result_sync_with_source(temp_db):
    """Test que save_result_sync respecte le paramètre source."""
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

    save_result_sync(temp_db, report, wfo_windows=None, duration=120.5, timeframe="5m", source="server")

    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT source FROM optimization_results")
    row = cursor.fetchone()
    conn.close()

    assert row["source"] == "server"


def test_build_push_payload_structure():
    """Test que build_push_payload retourne tous les champs requis sans NaN/Infinity."""
    validation = ValidationResult(
        bitget_sharpe=float("nan"), bitget_net_return_pct=float("inf"), bitget_trades=10,
        bitget_sharpe_ci_low=0.0, bitget_sharpe_ci_high=0.0,
        binance_oos_avg_sharpe=1.0, transfer_ratio=0.5,
        transfer_significant=False, volume_warning=False, volume_warning_detail="",
    )
    report = FinalReport(
        strategy_name="test", symbol="BTC/USDT", timestamp=datetime(2026, 2, 13, 12, 0),
        grade="F", total_score=15, wfo_avg_is_sharpe=1.0, wfo_avg_oos_sharpe=float("nan"),
        wfo_consistency_rate=0.20, wfo_n_windows=10, recommended_params={"rsi_period": 14},
        mc_p_value=1.0, mc_significant=False, mc_underpowered=True, dsr=0.0,
        dsr_max_expected_sharpe=3.0, stability=0.30, cliff_params=[], convergence=None,
        divergent_params=[], validation=validation, oos_is_ratio=0.10, bitget_transfer=0.20,
        live_eligible=False, warnings=["Test"], n_distinct_combos=100,
    )

    payload = build_push_payload(report, wfo_windows=None, duration=60.0, timeframe="5m")

    # Champs requis
    assert payload["strategy_name"] == "test"
    assert payload["asset"] == "BTC/USDT"
    assert payload["timeframe"] == "5m"
    assert payload["grade"] == "F"
    assert payload["n_windows"] == 10
    assert payload["source"] == "local"

    # NaN sanitizé
    assert payload["oos_sharpe"] is None

    # JSON blobs sont des strings
    assert isinstance(payload["best_params"], str)
    assert json.loads(payload["best_params"]) == {"rsi_period": 14}

    # Pas de NaN/Infinity dans le payload (json.dumps ne crashe pas)
    json.dumps(payload)  # Ne doit pas lever d'exception


def test_build_payload_from_db_row():
    """Test build_payload_from_db_row depuis une row DB simulée."""
    row = {
        "id": 1,
        "strategy_name": "vwap_rsi",
        "asset": "BTC/USDT",
        "timeframe": "5m",
        "created_at": "2026-02-13T12:00:00",
        "duration_seconds": 120.0,
        "grade": "A",
        "total_score": 87.0,
        "oos_sharpe": 1.8,
        "consistency": 0.85,
        "oos_is_ratio": 0.92,
        "dsr": 0.95,
        "param_stability": 0.88,
        "monte_carlo_pvalue": 0.02,
        "mc_underpowered": 0,
        "n_windows": 20,
        "n_distinct_combos": 600,
        "best_params": '{"rsi_period": 14}',
        "wfo_windows": None,
        "monte_carlo_summary": '{"p_value": 0.02}',
        "validation_summary": '{"bitget_sharpe": 1.5}',
        "warnings": '[]',
        "is_latest": 1,
        "source": "local",
    }

    payload = build_payload_from_db_row(row)

    assert payload["strategy_name"] == "vwap_rsi"
    assert payload["asset"] == "BTC/USDT"
    assert payload["source"] == "local"
    assert payload["best_params"] == '{"rsi_period": 14}'
    # id et is_latest ne sont PAS dans le payload (gérés côté serveur)
    assert "id" not in payload
    assert "is_latest" not in payload
