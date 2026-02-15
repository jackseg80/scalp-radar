"""Tests pour les endpoints API simulator, arena, signals."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from backend.api.server import app
from backend.backtesting.arena import StrategyArena, StrategyPerformance
from backend.backtesting.simulator import Simulator


@pytest.fixture
def mock_app():
    """Configure l'app avec des mocks pour simulator et arena."""
    # Simulator mock
    sim = MagicMock(spec=Simulator)
    sim._running = True
    sim.runners = []
    sim.get_all_status.return_value = {
        "vwap_rsi": {
            "name": "vwap_rsi",
            "capital": 10_200.0,
            "net_pnl": 200.0,
            "total_trades": 5,
            "wins": 3,
            "losses": 2,
            "win_rate": 60.0,
            "is_active": True,
            "kill_switch": False,
            "has_position": False,
        }
    }
    sim.get_all_trades.return_value = [
        {
            "strategy": "vwap_rsi",
            "direction": "LONG",
            "entry_price": 100_000.0,
            "exit_price": 100_800.0,
            "quantity": 0.01,
            "entry_time": "2024-01-15T12:00:00+00:00",
            "exit_time": "2024-01-15T12:05:00+00:00",
            "gross_pnl": 8.0,
            "fee_cost": 1.2,
            "slippage_cost": 0.5,
            "net_pnl": 6.3,
            "exit_reason": "tp",
            "market_regime": "RANGING",
        }
    ]
    sim.is_kill_switch_triggered.return_value = False

    # Arena mock
    arena = MagicMock(spec=StrategyArena)
    arena.get_ranking.return_value = [
        StrategyPerformance(
            name="vwap_rsi",
            capital=10_200.0,
            net_pnl=200.0,
            net_return_pct=2.0,
            total_trades=5,
            win_rate=60.0,
            profit_factor=2.5,
            max_drawdown_pct=1.2,
            is_active=True,
        )
    ]
    arena.get_strategy_detail.return_value = {
        "status": {"name": "vwap_rsi"},
        "trades": [],
        "performance": {"name": "vwap_rsi", "net_pnl": 200.0},
    }

    # DB mock avec get_simulation_trades async
    db = MagicMock()
    db._conn = True
    db.get_simulation_trades = AsyncMock(return_value=[
        {
            "id": 1,
            "strategy": "vwap_rsi",
            "symbol": "BTC/USDT",
            "direction": "LONG",
            "entry_price": 100_000.0,
            "exit_price": 100_800.0,
            "quantity": 0.01,
            "gross_pnl": 8.0,
            "fee_cost": 1.2,
            "slippage_cost": 0.5,
            "net_pnl": 6.3,
            "exit_reason": "tp",
            "market_regime": "RANGING",
            "entry_time": "2024-01-15T12:00:00+00:00",
            "exit_time": "2024-01-15T12:05:00+00:00",
        }
    ])

    app.state.simulator = sim
    app.state.arena = arena
    app.state.db = db
    app.state.engine = None
    app.state.config = MagicMock()
    app.state.start_time = MagicMock()
    app.state.start_time.isoformat.return_value = "2024-01-15T12:00:00"

    return app


@pytest.mark.asyncio
async def test_simulator_status(mock_app):
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/simulator/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is True
    assert "vwap_rsi" in data["strategies"]


@pytest.mark.asyncio
async def test_simulator_positions(mock_app):
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/simulator/positions")
    assert resp.status_code == 200
    assert "positions" in resp.json()


@pytest.mark.asyncio
async def test_simulator_trades(mock_app):
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/simulator/trades?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["trades"]) == 1


@pytest.mark.asyncio
async def test_simulator_performance(mock_app):
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/simulator/performance")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["ranking"]) == 1
    assert data["ranking"][0]["name"] == "vwap_rsi"


@pytest.mark.asyncio
async def test_arena_ranking(mock_app):
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/arena/ranking")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["ranking"]) == 1


@pytest.mark.asyncio
async def test_arena_strategy_detail(mock_app):
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/arena/strategy/vwap_rsi")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "performance" in data


@pytest.mark.asyncio
async def test_signals_recent(mock_app):
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/signals/recent")
    assert resp.status_code == 200
    data = resp.json()
    assert "signals" in data


# ─── Tests GET /api/simulator/grid-state ───────────────────────────────


@pytest.mark.asyncio
async def test_grid_state_no_simulator(mock_app):
    """GET /api/simulator/grid-state sans simulator → JSON vide."""
    mock_app.state.simulator = None
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/simulator/grid-state")
    assert resp.status_code == 200
    data = resp.json()
    assert data["grid_positions"] == {}
    assert data["summary"]["total_positions"] == 0


@pytest.mark.asyncio
async def test_grid_state_empty(mock_app):
    """GET /api/simulator/grid-state sans grilles actives → grids vide."""
    mock_app.state.simulator.get_grid_state.return_value = {
        "grid_positions": {},
        "summary": {
            "total_positions": 0,
            "total_assets": 0,
            "total_margin_used": 0,
            "total_unrealized_pnl": 0,
            "capital_available": 10000,
        },
    }
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/simulator/grid-state")
    assert resp.status_code == 200
    data = resp.json()
    assert data["grid_positions"] == {}
    assert data["summary"]["total_assets"] == 0


@pytest.mark.asyncio
async def test_grid_state_with_data(mock_app):
    """GET /api/simulator/grid-state avec une grille active → JSON complet."""
    mock_app.state.simulator.get_grid_state.return_value = {
        "grid_positions": {
            "BTC/USDT": {
                "symbol": "BTC/USDT",
                "strategy": "envelope_dca",
                "direction": "LONG",
                "levels_open": 2,
                "levels_max": 4,
                "avg_entry": 95000.0,
                "current_price": 96000.0,
                "unrealized_pnl": 20.0,
                "unrealized_pnl_pct": 6.32,
                "tp_price": 97000.0,
                "sl_price": 92000.0,
                "tp_distance_pct": 1.04,
                "sl_distance_pct": -4.17,
                "margin_used": 316.67,
                "leverage": 6,
                "positions": [
                    {"level": 0, "entry_price": 96000.0, "quantity": 0.01,
                     "entry_time": "2024-01-15T12:00:00+00:00", "direction": "LONG"},
                    {"level": 1, "entry_price": 94000.0, "quantity": 0.01,
                     "entry_time": "2024-01-15T13:00:00+00:00", "direction": "LONG"},
                ],
            },
        },
        "summary": {
            "total_positions": 2,
            "total_assets": 1,
            "total_margin_used": 316.67,
            "total_unrealized_pnl": 20.0,
            "capital_available": 9683.33,
        },
    }
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/simulator/grid-state")
    assert resp.status_code == 200
    data = resp.json()
    assert "BTC/USDT" in data["grid_positions"]
    grid = data["grid_positions"]["BTC/USDT"]
    assert grid["levels_open"] == 2
    assert grid["levels_max"] == 4
    assert grid["unrealized_pnl"] == 20.0
    assert grid["tp_distance_pct"] == 1.04
    assert len(grid["positions"]) == 2
    assert data["summary"]["total_positions"] == 2
    assert data["summary"]["total_assets"] == 1
