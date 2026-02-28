"""Tests is_latest : le run le plus récent prime TOUJOURS, sans comparaison de score."""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from backend.optimization.optimization_db import save_result_sync
from backend.optimization.report import FinalReport, ValidationResult


@pytest.fixture
def temp_db(tmp_path):
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
            win_rate_oos REAL,
            tail_risk_ratio REAL,
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


def _make_report(grade: str, score: float, ts: str) -> FinalReport:
    val = ValidationResult(
        bitget_sharpe=1.0, bitget_net_return_pct=5.0, bitget_trades=20,
        bitget_sharpe_ci_low=0.5, bitget_sharpe_ci_high=1.5,
        binance_oos_avg_sharpe=1.0, transfer_ratio=0.70,
        transfer_significant=False, volume_warning=False, volume_warning_detail="",
    )
    return FinalReport(
        strategy_name="grid_atr", symbol="BTC/USDT", timestamp=datetime.fromisoformat(ts),
        grade=grade, total_score=score, wfo_avg_is_sharpe=1.5, wfo_avg_oos_sharpe=1.2,
        wfo_consistency_rate=0.70, wfo_n_windows=15, recommended_params={"p": 1},
        mc_p_value=0.05, mc_significant=True, mc_underpowered=False, dsr=0.80,
        dsr_max_expected_sharpe=3.0, stability=0.75, cliff_params=[], convergence=None,
        divergent_params=[], validation=val, oos_is_ratio=0.75, bitget_transfer=0.70,
        live_eligible=True, warnings=[], n_distinct_combos=200,
    )


def test_grade_d_replaces_grade_a(temp_db):
    """Grade A (score 95) puis Grade D (score 40) → Grade D a is_latest=1."""
    r_a = _make_report("A", 95.0, "2026-02-10T10:00:00")
    save_result_sync(temp_db, r_a, wfo_windows=None, duration=60.0, timeframe="1h")

    r_d = _make_report("D", 40.0, "2026-02-15T10:00:00")
    save_result_sync(temp_db, r_d, wfo_windows=None, duration=60.0, timeframe="1h")

    conn = sqlite3.connect(temp_db)
    rows = conn.execute(
        "SELECT grade, total_score, is_latest FROM optimization_results ORDER BY created_at"
    ).fetchall()
    conn.close()

    assert rows[0] == ("A", 95.0, 0), "Grade A doit perdre is_latest"
    assert rows[1] == ("D", 40.0, 1), "Grade D (plus récent) doit avoir is_latest=1"


def test_unique_is_latest_per_strategy_asset(temp_db):
    """Un seul is_latest=1 par (strategy_name, asset) après insertions multiples."""
    for i, (grade, score) in enumerate([("A", 95), ("B", 70), ("C", 55), ("D", 40)]):
        r = _make_report(grade, float(score), f"2026-02-{10+i}T10:00:00")
        save_result_sync(temp_db, r, wfo_windows=None, duration=60.0, timeframe="1h")

    conn = sqlite3.connect(temp_db)
    count = conn.execute(
        "SELECT COUNT(*) FROM optimization_results "
        "WHERE strategy_name='grid_atr' AND asset='BTC/USDT' AND is_latest=1"
    ).fetchone()[0]
    latest = conn.execute(
        "SELECT grade FROM optimization_results WHERE is_latest=1"
    ).fetchone()
    conn.close()

    assert count == 1, f"Attendu 1 is_latest=1, trouvé {count}"
    assert latest[0] == "D", "Le dernier run (Grade D) doit être is_latest=1"
