"""Reporting must reconcile persisted history, never post-restart memory alone."""

from types import SimpleNamespace
from datetime import datetime
from unittest.mock import AsyncMock
import sqlite3

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest
import pytest_asyncio

from backend.api.arena_routes import router as arena_router
from backend.api.simulator_routes import router as simulator_router
from backend.backtesting.arena import StrategyArena
from backend.backtesting.simulator import RunnerStats
from backend.core.database import Database


def row(i, pnl, strategy="grid_boltrend", date=None):
    return dict(id=i, strategy=strategy, symbol="BTC/USDT", direction="LONG",
                entry_price=100., exit_price=101., quantity=1., gross_pnl=pnl+1,
                fee_cost=.6, slippage_cost=.4, net_pnl=pnl, exit_reason="signal_exit",
                market_regime="ranging", entry_time="2026-07-20T16:00:00+00:00",
                exit_time=date or f"2026-07-{20+i:02}T17:00:00+00:00")


def setup_reporting(pnls=(100., -50., 20.), funding=5., name="grid_boltrend"):
    stats = RunnerStats(capital=1000+sum(pnls)-funding, initial_capital=1000,
                        net_pnl=sum(pnls)-funding, total_trades=len(pnls),
                        wins=sum(p>0 for p in pnls), losses=sum(p<=0 for p in pnls))
    runner = SimpleNamespace(name=name, _total_funding_cost=funding,
                             get_stats=lambda: stats,
                             get_status=lambda: dict(name=name, net_pnl=stats.net_pnl),
                             get_trades=lambda: [])
    arena = StrategyArena(SimpleNamespace(runners=[runner]))
    rows = [row(i+1,p,name) for i,p in enumerate(pnls)]
    db = SimpleNamespace(get_simulation_trades=AsyncMock(return_value=rows[::-1]))
    return arena, db, stats, rows


@pytest.mark.asyncio
@pytest.mark.parametrize("funding", [5., -5., 0.])
async def test_restart_reconciles_full_history_and_funding(funding):
    arena, db, stats, rows = setup_reporting(funding=funding)
    detail = await arena.get_reporting_detail("grid_boltrend", db)
    p = detail["performance"]
    assert p["history_status"] == "reconciled"
    assert p["total_trades"] == p["history_trade_count"] == 3
    assert p["net_pnl"] == 70-funding
    assert p["profit_factor"] == pytest.approx(2.4)
    assert p["max_drawdown_pct"] == pytest.approx(50/1100*100)
    assert p["funding_cost"] == funding
    assert p["history_first_entry"] == rows[0]["entry_time"]
    assert p["history_first_exit"] == rows[0]["exit_time"]
    assert p["history_last_exit"] == rows[-1]["exit_time"]
    assert detail["trades"] == rows
    assert arena._simulator.runners[0].get_trades() == []
    db.get_simulation_trades.assert_awaited_once_with(strategy_name="grid_boltrend", limit=3)
    # Selection remains synchronous and unchanged; reporting doesn't hydrate it.
    assert arena.get_ranking()[0].profit_factor == 0.


@pytest.mark.asyncio
@pytest.mark.parametrize("fault,reason", [
    ("count", "trade_count_mismatch"), ("pnl", "pnl_funding_mismatch"),
    ("wins", "win_loss_mismatch"), ("strategy", "strategy_mismatch"),
    ("nan", "non_finite_pnl"), ("read", "database_read_failed"),
])
async def test_unreconciled_history_is_not_reported_as_zero(fault, reason):
    arena, db, stats, rows = setup_reporting()
    if fault == "count": db.get_simulation_trades.return_value = rows[:1]
    elif fault == "pnl": stats.net_pnl += 1
    elif fault == "wins": stats.wins += 1
    elif fault == "strategy": rows[0]["strategy"] = "other"
    elif fault == "nan": rows[0]["net_pnl"] = float("nan")
    elif fault == "read": db.get_simulation_trades.side_effect = sqlite3.OperationalError("locked")
    detail = await arena.get_reporting_detail("grid_boltrend", db)
    p = detail["performance"]
    assert p["history_status"] == "unavailable"
    assert p["history_reason"] == reason
    assert p["profit_factor"] is p["max_drawdown_pct"] is None
    assert detail["trades"] == []


@pytest.mark.asyncio
async def test_absent_database_and_unknown_strategy():
    arena, _, _, _ = setup_reporting()
    p = (await arena.get_reporting_ranking(None))[0]
    assert p["history_reason"] == "database_unavailable"
    assert p["profit_factor"] is None
    assert await arena.get_reporting_detail("unknown", None) is None


@pytest.mark.asyncio
async def test_empty_session_does_not_read_pre_reset_trades():
    arena, db, _, _ = setup_reporting(pnls=(), funding=0.)
    p = (await arena.get_reporting_ranking(db))[0]
    assert p["history_status"] == "reconciled"
    assert p["profit_factor"] == p["max_drawdown_pct"] == 0.
    assert p["history_first_exit"] is None
    db.get_simulation_trades.assert_not_awaited()


@pytest.mark.asyncio
async def test_zero_pnl_is_a_loss_like_runner_record_trade():
    arena, db, _, _ = setup_reporting(pnls=(0.,), funding=0.)
    assert (await arena.get_reporting_ranking(db))[0]["history_status"] == "reconciled"


@pytest.mark.asyncio
async def test_snapshot_is_not_changed_by_new_runner_stats_during_read():
    arena, db, stats, rows = setup_reporting()
    async def read(**kwargs):
        stats.total_trades += 1
        stats.net_pnl += 100
        return rows[::-1]
    db.get_simulation_trades.side_effect = read
    p = (await arena.get_reporting_ranking(db))[0]
    assert p["total_trades"] == 3
    assert p["net_pnl"] == 65.
    assert p["history_status"] == "unavailable"
    assert p["history_reason"] == "runner_changed_during_read"


@pytest.mark.asyncio
async def test_new_database_trade_during_query_fails_closed():
    arena, db, _, rows = setup_reporting()
    db.get_simulation_trades.return_value = [row(4,110), rows[2], rows[1]]
    p = (await arena.get_reporting_ranking(db))[0]
    assert p["history_status"] == "unavailable"
    assert p["profit_factor"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", [False, True])
async def test_in_memory_trades_are_anchors_not_appended_again(mismatch):
    arena, db, _, rows = setup_reporting()
    r = rows[-1]
    trade = SimpleNamespace(entry_time=datetime.fromisoformat(r["entry_time"]),
                            exit_time=datetime.fromisoformat(r["exit_time"]),
                            net_pnl=r["net_pnl"])
    arena._simulator.runners[0].get_trades = lambda: [("BTC/USDT", trade)]
    if mismatch:
        # Same count, wins and PnL, but different trade identity.
        rows[-1]["exit_time"] = "2026-09-26T13:00:00+00:00"
    detail = await arena.get_reporting_detail("grid_boltrend", db)
    if mismatch:
        assert detail["performance"]["history_reason"] == "in_memory_trades_mismatch"
    else:
        assert len(detail["trades"]) == 3
        assert detail["performance"]["history_status"] == "reconciled"


@pytest_asyncio.fixture
async def db():
    database = Database(db_path=":memory:")
    await database.init()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_database_window_excludes_pre_reset_and_orders_ties(db):
    arena, _, _, _ = setup_reporting(pnls=(100., -50.), funding=0.)
    # Same timestamp: database ID is the deterministic tie breaker, including
    # across an old session and unrelated strategies.
    for name, pnl in [("grid_boltrend", -999.), ("other", 10000.),
                      ("grid_boltrend", 100.), ("grid_boltrend", -50.)]:
        r = row(1,pnl,name)
        columns = [k for k in r if k != "id"]
        await db._conn.execute(
            "INSERT INTO simulation_trades (" + ",".join(
                "strategy_name" if k == "strategy" else k for k in columns
            ) + ") VALUES (" + ",".join("?" for _ in columns) + ")",
            [r[k] for k in columns],
        )
    await db._conn.commit()
    detail = await arena.get_reporting_detail("grid_boltrend", db)
    assert [r["id"] for r in detail["trades"]] == [3,4]
    assert detail["performance"]["profit_factor"] == 2.
    assert detail["performance"]["max_drawdown_pct"] == pytest.approx(50/1100*100)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/arena/ranking", "/api/simulator/performance",
                                      "/api/arena/strategy/grid_boltrend"])
@pytest.mark.parametrize("mode", ["valid", "incomplete", "all_wins"])
async def test_reporting_http_paths_share_reconciled_json(path, mode):
    arena, db, _, _ = setup_reporting(pnls=(100.,20.) if mode=="all_wins" else (100.,-50.,20.))
    if mode == "incomplete": db.get_simulation_trades.return_value = []
    app = FastAPI()
    app.include_router(arena_router)
    app.include_router(simulator_router)
    app.state.arena = arena
    app.state.db = db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(path)
    assert response.status_code == 200
    body = response.json()
    p = body["performance"] if "performance" in body else body["ranking"][0]
    assert p["metrics_basis"].endswith("excluding_funding_and_unrealized")
    if mode == "incomplete":
        assert p["profit_factor"] is None
        assert p["history_reason"] == "trade_count_mismatch"
    elif mode == "all_wins":
        assert p["profit_factor"] is None
        assert p["profit_factor_unbounded"] is True
    else:
        assert p["profit_factor"] == pytest.approx(2.4)
        assert p["history_status"] == "reconciled"
