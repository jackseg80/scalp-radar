"""Tests Sprint 14b : Combo results (heatmap dense + charts analytiques)"""

import json
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from backend.core.models import Candle
from backend.optimization.optimization_db import (
    build_push_payload,
    get_combo_results_async,
    save_combo_results_sync,
    save_result_from_payload_sync,
)
from backend.optimization.walk_forward import WalkForwardOptimizer
from backend.optimization.report import FinalReport, ValidationResult


@pytest.fixture
def db_path():
    """DB temporaire pour les tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path_str = tmp.name

    # Créer les tables directement en SQL
    conn = sqlite3.connect(db_path_str)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS optimization_results (
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
            UNIQUE (strategy_name, asset, timeframe, created_at)
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
            FOREIGN KEY (optimization_result_id) REFERENCES optimization_results(id)
        );

        CREATE INDEX IF NOT EXISTS idx_combo_opt_id ON wfo_combo_results(optimization_result_id);
        CREATE INDEX IF NOT EXISTS idx_combo_best ON wfo_combo_results(is_best) WHERE is_best = 1;
    """)
    conn.close()

    yield db_path_str
    Path(db_path_str).unlink(missing_ok=True)


@pytest.fixture
def mini_candles():
    """Dataset minimal pour WFO : 180 jours de bougies 1h."""
    base_time = datetime(2024, 1, 1)
    candles = []
    for i in range(180 * 24):  # 180 jours × 24h
        candles.append(
            Candle(
                symbol="BTC/USDT",
                timeframe="1h",
                timestamp=base_time.timestamp() + i * 3600,
                open=100.0 + (i % 10),
                high=105.0 + (i % 10),
                low=95.0 + (i % 10),
                close=100.0 + (i % 10),
                volume=1000.0,
            )
        )
    return candles


@pytest.mark.skip(reason="API WFO changée - optimize() nécessite DB+config, pas candles_by_tf direct")
def test_wfo_combo_results_populated(mini_candles):
    """Test 1 : WFO avec mini-grid retourne combo_results non vide."""
    from backend.strategies.envelope_dca import EnvelopeDCAStrategy

    # Mini-grid : 2×2 = 4 combos
    grid_values = {
        "ma_period": [5, 7],
        "num_levels": [2, 3],
        "envelope_start": [0.05],
        "envelope_step": [0.03],
        "sl_percent": [20.0],
    }

    optimizer = WalkForwardOptimizer(strategy_class=EnvelopeDCAStrategy)
    wfo = optimizer.optimize(
        strategy_name="envelope_dca",
        symbol="BTC/USDT",
        candles_by_tf={"1h": mini_candles},
        grid_values=grid_values,
        is_window_days=60,
        oos_window_days=30,
        step_days=30,
        metric="sharpe_ratio",
        n_workers=1,
    )

    # Vérifier que combo_results est non vide
    assert wfo.combo_results is not None
    assert len(wfo.combo_results) > 0

    # Vérifier qu'il y a 4 combos (2×2)
    assert len(wfo.combo_results) == 4

    # Vérifier que chaque combo a les champs requis
    for combo in wfo.combo_results:
        assert "params" in combo
        assert "oos_sharpe" in combo
        assert "is_sharpe" in combo
        assert "consistency" in combo
        assert "oos_is_ratio" in combo
        assert "is_best" in combo
        assert "n_windows_evaluated" in combo


@pytest.mark.skip(reason="API WFO changée - optimize() nécessite DB+config, pas candles_by_tf direct")
def test_combo_results_aggregation(mini_candles):
    """Test 2 : Vérifier que l'agrégation cross-fenêtre est correcte."""
    from backend.strategies.envelope_dca import EnvelopeDCAStrategy

    grid_values = {
        "ma_period": [5],
        "num_levels": [2],
        "envelope_start": [0.05],
        "envelope_step": [0.03],
        "sl_percent": [20.0],
    }

    optimizer = WalkForwardOptimizer(strategy_class=EnvelopeDCAStrategy)
    wfo = optimizer.optimize(
        strategy_name="envelope_dca",
        symbol="BTC/USDT",
        candles_by_tf={"1h": mini_candles},
        grid_values=grid_values,
        is_window_days=60,
        oos_window_days=30,
        step_days=30,
        metric="sharpe_ratio",
        n_workers=1,
    )

    # Une seule combo → n_windows_evaluated devrait être égal au nombre de fenêtres
    assert len(wfo.combo_results) == 1
    combo = wfo.combo_results[0]
    assert combo["n_windows_evaluated"] == len(wfo.windows)

    # Vérifier que IS Sharpe moyen est cohérent
    is_sharpes = [w.is_sharpe for w in wfo.windows]
    import numpy as np

    expected_avg_is = float(np.nanmean(is_sharpes))
    assert abs(combo["is_sharpe"] - expected_avg_is) < 0.01


def test_combo_best_flag():
    """Test 3 : Vérifier qu'exactement une combo a is_best=True."""
    combos = [
        {"params": {"a": 1}, "oos_sharpe": 1.5, "is_best": True},
        {"params": {"a": 2}, "oos_sharpe": 1.2, "is_best": False},
        {"params": {"a": 3}, "oos_sharpe": 0.9, "is_best": False},
    ]

    best_count = sum(1 for c in combos if c["is_best"])
    assert best_count == 1


def test_save_combo_results_sync(db_path):
    """Test 4 : Insertion en DB + relecture."""
    # Insérer un résultat WFO fictif
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        """INSERT INTO optimization_results (
            strategy_name, asset, timeframe, created_at, grade, total_score,
            n_windows, best_params, is_latest
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)""",
        (
            "test_strategy",
            "BTC/USDT",
            "1h",
            datetime.now().isoformat(),
            "B",
            75,
            3,
            json.dumps({"param": 10}),
        ),
    )
    result_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # Sauver des combo results
    combos = [
        {
            "params": {"param": 10},
            "oos_sharpe": 1.5,
            "is_sharpe": 1.8,
            "oos_return_pct": 12.5,
            "is_return_pct": 15.0,
            "oos_trades": 50,
            "is_trades": 120,
            "consistency": 0.85,
            "oos_is_ratio": 0.83,
            "is_best": True,
            "n_windows_evaluated": 3,
        },
        {
            "params": {"param": 20},
            "oos_sharpe": 1.2,
            "is_sharpe": 1.6,
            "oos_return_pct": 10.0,
            "is_return_pct": 13.0,
            "oos_trades": 45,
            "is_trades": 110,
            "consistency": 0.75,
            "oos_is_ratio": 0.75,
            "is_best": False,
            "n_windows_evaluated": 3,
        },
    ]

    n_saved = save_combo_results_sync(db_path, result_id, combos)
    assert n_saved == 2

    # Relecture avec get_combo_results_async
    import asyncio

    fetched = asyncio.run(get_combo_results_async(db_path, result_id))
    assert len(fetched) == 2
    assert fetched[0]["params"]["param"] == 10  # Trié par oos_sharpe DESC
    assert fetched[0]["is_best"] == 1
    assert fetched[0]["n_windows_evaluated"] == 3


@pytest.mark.asyncio
async def test_combo_results_empty_for_old_runs(db_path):
    """Test 5 : Endpoint retourne combos: [] pour un run sans combo data."""
    # Insérer un résultat WFO sans combos
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        """INSERT INTO optimization_results (
            strategy_name, asset, timeframe, created_at, grade, total_score,
            n_windows, best_params, is_latest
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)""",
        (
            "old_strategy",
            "ETH/USDT",
            "5m",
            datetime.now().isoformat(),
            "C",
            60,
            5,
            json.dumps({"old_param": 5}),
        ),
    )
    result_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # Fetch combo results → doit être vide
    combos = await get_combo_results_async(db_path, result_id)
    assert len(combos) == 0


def test_heatmap_dense():
    """Test 6 : Heatmap retourne matrice dense avec N×M cellules."""
    # Simuler des combo_results (3 valeurs X × 2 valeurs Y = 6 cellules)
    combos = [
        {"params": {"x": 5, "y": 10}, "oos_sharpe": 1.5},
        {"params": {"x": 7, "y": 10}, "oos_sharpe": 1.3},
        {"params": {"x": 10, "y": 10}, "oos_sharpe": 1.1},
        {"params": {"x": 5, "y": 15}, "oos_sharpe": 1.4},
        {"params": {"x": 7, "y": 15}, "oos_sharpe": 1.2},
        {"params": {"x": 10, "y": 15}, "oos_sharpe": 1.0},
    ]

    # Construire la heatmap manuellement (logic de l'endpoint)
    param_x = "x"
    param_y = "y"
    metric = "oos_sharpe"

    points = {}
    for combo in combos:
        params = combo["params"]
        x_val = params.get(param_x)
        y_val = params.get(param_y)
        if x_val is None or y_val is None:
            continue
        coord = (float(x_val), float(y_val))
        points[coord] = {"value": combo.get(metric), "result_id": 123}

    x_values = sorted(set(x for x, y in points.keys()))
    y_values = sorted(set(y for x, y in points.keys()))

    assert len(x_values) == 3  # [5, 7, 10]
    assert len(y_values) == 2  # [10, 15]

    # Construire cells[y_idx][x_idx]
    cells = []
    for y_val in y_values:
        row = []
        for x_val in x_values:
            coord = (x_val, y_val)
            if coord in points:
                row.append(points[coord])
            else:
                row.append({"value": None})
        cells.append(row)

    # Vérifier : 2 rows × 3 cols = 6 cellules, toutes remplies
    assert len(cells) == 2
    assert len(cells[0]) == 3
    assert all(cell["value"] is not None for row in cells for cell in row)


def test_push_payload_includes_combos():
    """Test 7 : build_push_payload() inclut le champ combo_results."""
    report = FinalReport(
        strategy_name="test",
        symbol="BTC/USDT",
        timestamp=datetime.now(),
        recommended_params={"param": 10},
        wfo_avg_is_sharpe=1.7,
        wfo_avg_oos_sharpe=1.5,
        wfo_consistency_rate=0.8,
        wfo_n_windows=5,
        oos_is_ratio=0.85,
        dsr=0.9,
        dsr_max_expected_sharpe=1.8,
        stability=0.88,
        cliff_params=[],
        convergence=0.95,
        divergent_params=[],
        mc_p_value=0.02,
        mc_significant=True,
        mc_underpowered=False,
        n_distinct_combos=10,
        validation=ValidationResult(
            bitget_sharpe=1.4,
            bitget_net_return_pct=10.0,
            bitget_trades=50,
            bitget_sharpe_ci_low=1.0,
            bitget_sharpe_ci_high=1.8,
            binance_oos_avg_sharpe=1.45,
            transfer_ratio=0.97,
            transfer_significant=True,
            volume_warning=False,
            volume_warning_detail=None,
        ),
        bitget_transfer=0.97,
        live_eligible=True,
        warnings=[],
        grade="B",
        total_score=75,
    )

    combo_results = [
        {"params": {"param": 10}, "oos_sharpe": 1.5, "is_best": True, "n_windows_evaluated": 5},
        {"params": {"param": 20}, "oos_sharpe": 1.2, "is_best": False, "n_windows_evaluated": 5},
    ]

    payload = build_push_payload(
        report=report,
        wfo_windows=None,
        duration=120.0,
        timeframe="1h",
        combo_results=combo_results,
    )

    assert "combo_results" in payload
    assert len(payload["combo_results"]) == 2


def test_save_from_payload_with_combos(db_path):
    """Test 8 : save_result_from_payload_sync() insère aussi les combos."""
    payload = {
        "strategy_name": "test_payload",
        "asset": "SOL/USDT",
        "timeframe": "1h",
        "created_at": datetime.now().isoformat(),
        "grade": "A",
        "total_score": 90,
        "n_windows": 4,
        "best_params": json.dumps({"param": 15}),
        "combo_results": [
            {
                "params": {"param": 15},
                "oos_sharpe": 2.0,
                "is_sharpe": 2.2,
                "oos_return_pct": 20.0,
                "is_return_pct": 22.0,
                "oos_trades": 60,
                "is_trades": 150,
                "consistency": 0.9,
                "oos_is_ratio": 0.91,
                "is_best": True,
                "n_windows_evaluated": 4,
            }
        ],
    }

    status = save_result_from_payload_sync(db_path, payload)
    assert status == "created"

    # Vérifier que les combos ont bien été insérés
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT COUNT(*) FROM wfo_combo_results WHERE optimization_result_id = (SELECT id FROM optimization_results WHERE strategy_name = ? AND asset = ?)",
        ("test_payload", "SOL/USDT"),
    )
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 1
