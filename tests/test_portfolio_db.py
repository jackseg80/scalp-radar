"""Tests pour portfolio_db.py (Sprint 20b-UI).

Couvre :
- Sauvegarde et récupération d'un PortfolioResult
- Liste sans equity_curve
- Parsing JSON des blobs
- Suppression
- Sous-échantillonnage equity_curve (max 500 points)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from backend.backtesting.portfolio_db import (
    delete_backtest_async,
    get_backtest_by_id_async,
    get_backtests_async,
    save_result_sync,
    _result_to_row,
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
        kill_switch_events=[{"timestamp": "2025-07-15T12:00:00+00:00", "drawdown_pct": 31.2, "equity": 6880.0, "window_start_equity": 10000.0}],
        snapshots=snapshots,
        per_asset_results={
            "AAA/USDT": {"trades": 10, "wins": 6, "win_rate": 60.0, "net_pnl": 25.5, "realized_trades": 8, "force_closed_trades": 2, "realized_pnl": 20.0, "force_closed_pnl": 5.5},
            "BBB/USDT": {"trades": 8, "wins": 3, "win_rate": 37.5, "net_pnl": -10.0, "realized_trades": 7, "force_closed_trades": 1, "realized_pnl": -8.0, "force_closed_pnl": -2.0},
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


# ─── Tests ────────────────────────────────────────────────────────────────


def test_save_and_get_backtest(db_path):
    """Round-trip : sauvegarde sync + lecture async."""
    result = _make_result()
    result_id = save_result_sync(db_path, result, strategy_name="grid_atr", exchange="binance")
    assert isinstance(result_id, int)
    assert result_id > 0

    # Lecture async
    detail = asyncio.run(
        get_backtest_by_id_async(db_path, result_id)
    )
    assert detail is not None
    assert detail["id"] == result_id
    assert detail["strategy_name"] == "grid_atr"
    assert detail["initial_capital"] == 10000.0
    assert detail["total_return_pct"] == 1.0
    assert detail["total_trades"] == 42
    assert detail["win_rate"] == 55.5
    assert detail["max_drawdown_pct"] == -5.2
    assert detail["peak_open_positions"] == 12


def test_list_backtests_no_equity_curve(db_path):
    """La liste exclut equity_curve (trop gros)."""
    result = _make_result()
    save_result_sync(db_path, result)

    backtests = asyncio.run(
        get_backtests_async(db_path)
    )
    assert len(backtests) == 1
    assert "equity_curve" not in backtests[0]
    assert "per_asset_results" not in backtests[0]
    # Mais les champs résumés sont présents
    assert backtests[0]["total_return_pct"] == 1.0
    assert backtests[0]["n_assets"] == 5


def test_get_backtest_parses_json(db_path):
    """Les JSON blobs sont bien parsés en objets Python."""
    result = _make_result()
    result_id = save_result_sync(db_path, result)

    detail = asyncio.run(
        get_backtest_by_id_async(db_path, result_id)
    )
    # assets = list (pas une string JSON)
    assert isinstance(detail["assets"], list)
    assert "AAA/USDT" in detail["assets"]
    # equity_curve = list de dicts
    assert isinstance(detail["equity_curve"], list)
    assert isinstance(detail["equity_curve"][0], dict)
    assert "equity" in detail["equity_curve"][0]
    # per_asset_results = dict
    assert isinstance(detail["per_asset_results"], dict)
    assert "AAA/USDT" in detail["per_asset_results"]
    # kill_switch_events = list
    assert isinstance(detail["kill_switch_events"], list)


def test_delete_backtest(db_path):
    """Suppression d'un run."""
    result = _make_result()
    result_id = save_result_sync(db_path, result)

    deleted = asyncio.run(
        delete_backtest_async(db_path, result_id)
    )
    assert deleted is True

    detail = asyncio.run(
        get_backtest_by_id_async(db_path, result_id)
    )
    assert detail is None

    # Supprimer un id inexistant
    deleted2 = asyncio.run(
        delete_backtest_async(db_path, 9999)
    )
    assert deleted2 is False


def test_subsample_equity_curve():
    """L'equity_curve est limitée à ~500 points même avec 2000 snapshots."""
    result = _make_result(n_snapshots=2000)
    row = _result_to_row(
        result, "grid_atr", "binance", 30.0, 24, None, None,
        "2025-01-01T00:00:00+00:00",
    )
    import json
    curve = json.loads(row["equity_curve"])
    # step = max(1, 2000//500) = 4, donc ~500 points
    assert len(curve) <= 501
    assert len(curve) >= 400


def test_save_and_get_btc_benchmark(db_path):
    """Round-trip : benchmark BTC sauvegardé et relu correctement."""
    btc_curve = [{"timestamp": "2025-06-01T00:00:00+00:00", "equity": 10000.0}]
    result = _make_result()
    result.btc_benchmark = {
        "return_pct": 12.5,
        "max_drawdown_pct": -18.3,
        "sharpe_ratio": 0.92,
        "final_equity": 11_250.0,
        "entry_price": 40_000.0,
        "exit_price": 45_000.0,
        "equity_curve": btc_curve,
    }
    result.alpha_vs_btc = 1.0 - 12.5  # total_return_pct=1.0 - btc=12.5

    result_id = save_result_sync(
        db_path, result, "grid_atr", "binance", 45.0, 24, 120.0, "test-btc"
    )

    detail = asyncio.run(get_backtest_by_id_async(db_path, result_id))
    assert detail is not None
    assert detail["btc_benchmark_return_pct"] == pytest.approx(12.5)
    assert detail["btc_benchmark_max_dd_pct"] == pytest.approx(-18.3)
    assert detail["btc_benchmark_sharpe"] == pytest.approx(0.92)
    assert detail["alpha_vs_btc"] == pytest.approx(1.0 - 12.5)
    # equity_curve parsée comme liste
    assert isinstance(detail["btc_equity_curve"], list)
    assert len(detail["btc_equity_curve"]) == 1
    assert detail["btc_equity_curve"][0]["equity"] == 10_000.0


def test_btc_benchmark_null_when_absent(db_path):
    """Un run sans benchmark BTC stocke NULL en DB et retourne [] pour btc_equity_curve."""
    result = _make_result()
    # btc_benchmark par défaut = None

    result_id = save_result_sync(
        db_path, result, "grid_atr", "binance", 45.0, 24, None, None
    )

    detail = asyncio.run(get_backtest_by_id_async(db_path, result_id))
    assert detail is not None
    assert detail["btc_benchmark_return_pct"] is None
    assert detail["btc_equity_curve"] == []
    assert detail["alpha_vs_btc"] is None


def test_list_includes_btc_columns(db_path):
    """La liste des backtests inclut btc_benchmark_return_pct et alpha_vs_btc."""
    result = _make_result()
    result.btc_benchmark = {
        "return_pct": 8.0,
        "max_drawdown_pct": -12.0,
        "sharpe_ratio": 0.7,
        "final_equity": 10_800.0,
        "entry_price": 40_000.0,
        "exit_price": 43_200.0,
        "equity_curve": [],
    }
    result.alpha_vs_btc = 1.0 - 8.0

    save_result_sync(db_path, result, "grid_atr", "binance", 45.0, 24, None, None)

    runs = asyncio.run(get_backtests_async(db_path))
    assert len(runs) == 1
    assert "btc_benchmark_return_pct" in runs[0]
    assert "alpha_vs_btc" in runs[0]
    assert runs[0]["btc_benchmark_return_pct"] == pytest.approx(8.0)
    # btc_equity_curve NON inclus dans la liste (trop volumineux)
    assert "btc_equity_curve" not in runs[0]
