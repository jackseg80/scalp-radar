"""Tests pour les endpoints Journal V2 — slippage + per-asset."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from fastapi import FastAPI

from backend.api.journal_routes import router


def _make_order(
    order_id: str = "ord_1",
    avg_price: float = 100.5,
    paper_price: float = 100.0,
    symbol: str = "BTC/USDT:USDT",
    strategy: str = "grid_atr",
    order_type: str = "entry",
    side: str = "buy",
    quantity: float = 0.001,
) -> dict:
    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "order_type": order_type,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "filled": quantity,
        "average_price": avg_price,
        "order_id": order_id,
        "status": "closed",
        "strategy_name": strategy,
        "context": "mono",
        "paper_price": paper_price,
    }


class MockExecutor:
    def __init__(self, orders: list[dict] | None = None):
        self._order_history: deque[dict] = deque(maxlen=200)
        for o in (orders or []):
            self._order_history.appendleft(o)


# ─── SLIPPAGE ENDPOINT ────────────────────────────────────────────────────────


class TestSlippageEndpoint:
    """Tests pour GET /api/journal/slippage."""

    @pytest_asyncio.fixture
    def app_no_executor(self) -> FastAPI:
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest_asyncio.fixture
    def app_with_orders(self) -> FastAPI:
        app = FastAPI()
        app.include_router(router)
        app.state.executor = MockExecutor([
            _make_order("o1", avg_price=100.5, paper_price=100.0, symbol="BTC/USDT:USDT", strategy="grid_atr"),
            _make_order("o2", avg_price=50.1, paper_price=50.0, symbol="ETH/USDT:USDT", strategy="grid_atr"),
            _make_order("o3", avg_price=200.0, paper_price=200.0, symbol="BTC/USDT:USDT", strategy="grid_boltrend"),
        ])
        return app

    @pytest.mark.asyncio
    async def test_slippage_endpoint_no_executor(self, app_no_executor):
        """Retourne slippage=None si executor absent."""
        transport = ASGITransport(app=app_no_executor)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/journal/slippage")

        assert resp.status_code == 200
        assert resp.json()["slippage"] is None

    @pytest.mark.asyncio
    async def test_slippage_endpoint_with_data(self, app_with_orders):
        """Calculs slippage corrects + champ note present."""
        transport = ASGITransport(app=app_with_orders)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/journal/slippage")

        assert resp.status_code == 200
        data = resp.json()["slippage"]
        assert data["orders_analyzed"] == 3
        assert "note" in data
        # BTC: (100.5-100)/100*100 = 0.5%, ETH: (50.1-50)/50*100 = 0.2%, BTC2: 0%
        # Avg = (0.5 + 0.2 + 0.0) / 3 = 0.2333...
        assert 0.23 < data["avg_slippage_pct"] < 0.24

    @pytest.mark.asyncio
    async def test_slippage_by_asset_grouping(self, app_with_orders):
        """Groupement par symbol correct."""
        transport = ASGITransport(app=app_with_orders)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/journal/slippage")

        by_asset = resp.json()["slippage"]["by_asset"]
        assert "BTC/USDT:USDT" in by_asset
        assert "ETH/USDT:USDT" in by_asset
        assert by_asset["BTC/USDT:USDT"]["count"] == 2
        assert by_asset["ETH/USDT:USDT"]["count"] == 1

    @pytest.mark.asyncio
    async def test_slippage_filters_invalid_orders(self):
        """Ordres sans paper_price ou sans average_price exclus."""
        app = FastAPI()
        app.include_router(router)
        app.state.executor = MockExecutor([
            _make_order("o1", avg_price=100.5, paper_price=100.0),  # valide
            _make_order("o2", avg_price=0.0, paper_price=100.0),    # avg=0 → exclu
            _make_order("o3", avg_price=100.5, paper_price=0.0),    # paper=0 → exclu
        ])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/journal/slippage")

        data = resp.json()["slippage"]
        assert data["orders_analyzed"] == 1


# ─── PER-ASSET ENDPOINT ──────────────────────────────────────────────────────


class TestPerAssetEndpoint:
    """Tests pour GET /api/journal/per-asset."""

    @pytest_asyncio.fixture
    def app_no_db(self) -> FastAPI:
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.mark.asyncio
    async def test_per_asset_stats_empty(self, app_no_db):
        """Retourne liste vide si pas de DB."""
        transport = ASGITransport(app=app_no_db)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/journal/per-asset")

        assert resp.status_code == 200
        assert resp.json()["per_asset"] == []

    @pytest.mark.asyncio
    async def test_per_asset_stats_aggregation(self, tmp_path):
        """2 assets, verifie wins/losses/pnl apres aggregation."""
        import aiosqlite

        db_path = str(tmp_path / "test.db")
        conn = await aiosqlite.connect(db_path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("""
            CREATE TABLE simulation_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                gross_pnl REAL NOT NULL,
                fee_cost REAL NOT NULL,
                slippage_cost REAL NOT NULL,
                net_pnl REAL NOT NULL,
                exit_reason TEXT NOT NULL,
                market_regime TEXT,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL
            )
        """)

        # Inserer des trades
        trades = [
            ("grid_atr", "BTC/USDT", "LONG", 50000, 51000, 0.01, 10.0, 0.5, 0.1, 9.4, "tp", None, "2026-01-01T00:00:00", "2026-01-01T01:00:00"),
            ("grid_atr", "BTC/USDT", "LONG", 50000, 49000, 0.01, -10.0, 0.5, 0.1, -10.6, "sl", None, "2026-01-01T02:00:00", "2026-01-01T03:00:00"),
            ("grid_atr", "ETH/USDT", "LONG", 3000, 3100, 0.1, 10.0, 0.3, 0.1, 9.6, "tp", None, "2026-01-01T00:00:00", "2026-01-01T01:00:00"),
            ("grid_atr", "ETH/USDT", "LONG", 3000, 3200, 0.1, 20.0, 0.3, 0.1, 19.6, "tp", None, "2026-01-01T02:00:00", "2026-01-01T03:00:00"),
            ("grid_atr", "ETH/USDT", "LONG", 3000, 2900, 0.1, -10.0, 0.3, 0.1, -10.4, "sl", None, "2026-01-01T04:00:00", "2026-01-01T05:00:00"),
        ]
        for t in trades:
            await conn.execute(
                "INSERT INTO simulation_trades (strategy_name, symbol, direction, entry_price, exit_price, quantity, gross_pnl, fee_cost, slippage_cost, net_pnl, exit_reason, market_regime, entry_time, exit_time) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                t,
            )
        await conn.commit()

        # Creer un mock db avec la methode
        from backend.core.database import Database
        db = Database.__new__(Database)
        db._conn = conn

        # Tester
        result = await db.get_journal_per_asset_stats(period="all")

        assert len(result) == 2
        # ETH a un meilleur total_pnl → en premier (ORDER BY total_pnl DESC)
        eth = next(r for r in result if r["symbol"] == "ETH/USDT")
        btc = next(r for r in result if r["symbol"] == "BTC/USDT")

        assert eth["total_trades"] == 3
        assert eth["wins"] == 2
        assert eth["losses"] == 1
        assert eth["win_rate"] == 66.7
        assert eth["total_pnl"] == 18.8  # 9.6 + 19.6 - 10.4

        assert btc["total_trades"] == 2
        assert btc["wins"] == 1
        assert btc["losses"] == 1
        assert btc["total_pnl"] == -1.2  # 9.4 - 10.6

        await conn.close()
