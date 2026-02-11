"""Tests pour les endpoints API simulator, arena, signals."""

from __future__ import annotations

from unittest.mock import MagicMock

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

    app.state.simulator = sim
    app.state.arena = arena
    app.state.db = MagicMock()
    app.state.db._conn = True
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
