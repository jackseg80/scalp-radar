"""Tests pour get_strategy_summary_async — Sprint 36."""

from __future__ import annotations

import json
import sqlite3
import tempfile

import pytest

from backend.optimization.optimization_db import get_strategy_summary_async


# ─── Helpers ──────────────────────────────────────────────────────────────


def _create_tables(conn: sqlite3.Connection) -> None:
    """Crée les tables nécessaires pour les tests."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS optimization_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            asset TEXT NOT NULL,
            timeframe TEXT NOT NULL DEFAULT '1h',
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
            is_latest INTEGER NOT NULL DEFAULT 1,
            source TEXT DEFAULT 'local',
            regime_analysis TEXT,
            win_rate_oos REAL,
            tail_risk_ratio REAL
        );

        CREATE TABLE IF NOT EXISTS portfolio_backtests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            initial_capital REAL NOT NULL,
            n_assets INTEGER NOT NULL,
            period_days INTEGER NOT NULL,
            assets TEXT NOT NULL,
            exchange TEXT NOT NULL DEFAULT 'binance',
            kill_switch_pct REAL NOT NULL DEFAULT 45.0,
            kill_switch_window_hours INTEGER NOT NULL DEFAULT 24,
            final_equity REAL NOT NULL,
            total_return_pct REAL NOT NULL,
            total_trades INTEGER NOT NULL,
            win_rate REAL NOT NULL,
            realized_pnl REAL NOT NULL,
            force_closed_pnl REAL NOT NULL,
            max_drawdown_pct REAL NOT NULL,
            max_drawdown_date TEXT,
            max_drawdown_duration_hours REAL NOT NULL,
            peak_margin_ratio REAL NOT NULL,
            peak_open_positions INTEGER NOT NULL,
            peak_concurrent_assets INTEGER NOT NULL,
            kill_switch_triggers INTEGER NOT NULL DEFAULT 0,
            kill_switch_events TEXT,
            equity_curve TEXT NOT NULL,
            per_asset_results TEXT NOT NULL,
            created_at TEXT NOT NULL,
            duration_seconds REAL,
            label TEXT
        );
    """)


def _insert_result(
    conn: sqlite3.Connection,
    strategy: str,
    asset: str,
    grade: str,
    total_score: float,
    oos_sharpe: float = 2.0,
    consistency: float = 0.8,
    oos_is_ratio: float = 0.9,
    mc_underpowered: int = 0,
    param_stability: float = 0.7,
    best_params: dict | None = None,
    created_at: str = "2026-02-20T10:00:00",
) -> int:
    """Insère un résultat WFO de test."""
    params = best_params or {"num_levels": 3, "atr_period": 10}
    cursor = conn.execute(
        """INSERT INTO optimization_results
           (strategy_name, asset, timeframe, created_at, grade, total_score,
            oos_sharpe, consistency, oos_is_ratio, mc_underpowered,
            param_stability, n_windows, best_params, is_latest)
           VALUES (?, ?, '1h', ?, ?, ?, ?, ?, ?, ?, ?, 12, ?, 1)""",
        (
            strategy, asset, created_at, grade, total_score,
            oos_sharpe, consistency, oos_is_ratio, mc_underpowered,
            param_stability, json.dumps(params),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_portfolio(
    conn: sqlite3.Connection,
    strategy: str,
    label: str,
    return_pct: float,
    days: int,
    created_at: str = "2026-02-20T12:00:00",
) -> int:
    """Insère un portfolio backtest de test."""
    cursor = conn.execute(
        """INSERT INTO portfolio_backtests
           (strategy_name, initial_capital, n_assets, period_days, assets,
            final_equity, total_return_pct, total_trades, win_rate,
            realized_pnl, force_closed_pnl, max_drawdown_pct,
            max_drawdown_duration_hours, peak_margin_ratio,
            peak_open_positions, peak_concurrent_assets,
            equity_curve, per_asset_results, created_at, label)
           VALUES (?, 5000, 10, ?, '[]', 6000, ?, 100, 55.0,
                   1000, 0, -15.0, 48.0, 0.35, 8, 10,
                   '[]', '{}', ?, ?)""",
        (strategy, days, return_pct, created_at, label),
    )
    conn.commit()
    return cursor.lastrowid


@pytest.fixture
def temp_db():
    """DB temporaire avec tables créées."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    conn = sqlite3.connect(db_path)
    _create_tables(conn)
    yield db_path, conn
    conn.close()


# ─── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_summary_empty_strategy(temp_db):
    """Stratégie sans résultats → total_assets: 0."""
    db_path, _conn = temp_db

    result = await get_strategy_summary_async(db_path, "inexistante")

    assert result["strategy_name"] == "inexistante"
    assert result["total_assets"] == 0


@pytest.mark.asyncio
async def test_summary_grade_distribution(temp_db):
    """Grades correctement comptés et agrégés."""
    db_path, conn = temp_db

    _insert_result(conn, "grid_atr", "BTC/USDT", "A", 90)
    _insert_result(conn, "grid_atr", "ETH/USDT", "A", 88)
    _insert_result(conn, "grid_atr", "DOGE/USDT", "B", 75)
    _insert_result(conn, "grid_atr", "SOL/USDT", "C", 60)
    # Autre stratégie — ne doit pas être comptée
    _insert_result(conn, "grid_boltrend", "BTC/USDT", "A", 92)

    result = await get_strategy_summary_async(db_path, "grid_atr")

    assert result["total_assets"] == 4
    assert result["grades"]["A"] == 2
    assert result["grades"]["B"] == 1
    assert result["grades"]["C"] == 1
    assert result["grades"]["D"] == 0
    assert result["grades"]["F"] == 0
    assert result["ab_count"] == 3
    assert result["ab_pct"] == 75.0
    assert result["avg_oos_sharpe"] == 2.0  # tous à 2.0


@pytest.mark.asyncio
async def test_summary_red_flags(temp_db):
    """Red flags détectés selon les seuils."""
    db_path, conn = temp_db

    # Normal
    _insert_result(conn, "grid_atr", "BTC/USDT", "A", 90)
    # OOS/IS suspect (> 1.5)
    _insert_result(conn, "grid_atr", "ETH/USDT", "B", 75, oos_is_ratio=2.1)
    # Underpowered
    _insert_result(conn, "grid_atr", "DOGE/USDT", "B", 72, mc_underpowered=1)
    # Low consistency (< 0.5)
    _insert_result(conn, "grid_atr", "SOL/USDT", "C", 55, consistency=0.3)
    # Low stability (< 0.3)
    _insert_result(conn, "grid_atr", "AVAX/USDT", "C", 52, param_stability=0.2)

    result = await get_strategy_summary_async(db_path, "grid_atr")

    assert result["red_flags"]["oos_is_ratio_suspect"] == 1
    assert result["red_flags"]["underpowered"] == 1
    assert result["red_flags"]["low_consistency"] == 1
    assert result["red_flags"]["low_stability"] == 1
    assert result["red_flags"]["sharpe_anomalous"] == 0
    assert result["red_flags_total"] == 4
    assert result["underpowered_count"] == 1
    assert result["underpowered_pct"] == 20.0


@pytest.mark.asyncio
async def test_summary_param_convergence(temp_db):
    """Convergence des paramètres calculée correctement."""
    db_path, conn = temp_db

    _insert_result(conn, "grid_atr", "BTC/USDT", "A", 90,
                   best_params={"num_levels": 3, "atr_period": 10})
    _insert_result(conn, "grid_atr", "ETH/USDT", "A", 88,
                   best_params={"num_levels": 3, "atr_period": 14})
    _insert_result(conn, "grid_atr", "DOGE/USDT", "B", 75,
                   best_params={"num_levels": 3, "atr_period": 10})

    result = await get_strategy_summary_async(db_path, "grid_atr")

    conv = {c["param"]: c for c in result["param_convergence"]}

    # num_levels : 3 valeurs identiques → mode 3, 100%
    assert conv["num_levels"]["mode"] == "3"
    assert conv["num_levels"]["mode_pct"] == 100
    assert conv["num_levels"]["n_unique"] == 1

    # atr_period : 2x10 + 1x14 → mode 10, 67%
    assert conv["atr_period"]["mode"] == "10"
    assert conv["atr_period"]["mode_pct"] == 67
    assert conv["atr_period"]["n_unique"] == 2


@pytest.mark.asyncio
async def test_summary_param_convergence_double_json(temp_db):
    """Gère le double-encoding JSON dans best_params."""
    db_path, conn = temp_db

    # Double-encoded : JSON string dans un JSON string
    double_encoded = json.dumps(json.dumps({"num_levels": 3, "sl_percent": 20}))
    conn.execute(
        """INSERT INTO optimization_results
           (strategy_name, asset, timeframe, created_at, grade, total_score,
            oos_sharpe, consistency, oos_is_ratio, mc_underpowered,
            param_stability, n_windows, best_params, is_latest)
           VALUES ('grid_atr', 'BTC/USDT', '1h', '2026-02-20', 'A', 90,
                   2.0, 0.8, 0.9, 0, 0.7, 12, ?, 1)""",
        (double_encoded,),
    )
    conn.commit()

    result = await get_strategy_summary_async(db_path, "grid_atr")

    conv = {c["param"]: c for c in result["param_convergence"]}
    assert "num_levels" in conv
    assert conv["num_levels"]["mode"] == "3"


@pytest.mark.asyncio
async def test_summary_portfolio_runs(temp_db):
    """Portfolio runs inclus dans le résumé."""
    db_path, conn = temp_db

    _insert_result(conn, "grid_atr", "BTC/USDT", "A", 90)
    _insert_portfolio(conn, "grid_atr", "backtest_730j", 221.0, 730)
    _insert_portfolio(conn, "grid_atr", "forward_365j", 82.4, 365,
                      created_at="2026-02-20T13:00:00")
    # Autre stratégie — ne doit pas apparaître
    _insert_portfolio(conn, "grid_boltrend", "bolt_test", 50.0, 365)

    result = await get_strategy_summary_async(db_path, "grid_atr")

    assert len(result["portfolio_runs"]) == 2
    # Trié par created_at DESC → forward_365j en premier
    assert result["portfolio_runs"][0]["label"] == "forward_365j"
    assert result["portfolio_runs"][0]["return_pct"] == 82.4
    assert result["portfolio_runs"][0]["days"] == 365
    assert result["portfolio_runs"][1]["label"] == "backtest_730j"
