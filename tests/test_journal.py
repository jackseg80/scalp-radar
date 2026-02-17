"""Tests Sprint 25 — Activity Journal (snapshots + events)."""

from __future__ import annotations

import json

import pytest
import pytest_asyncio

from backend.core.database import Database


@pytest_asyncio.fixture
async def db():
    """DB en mémoire initialisée avec toutes les tables."""
    database = Database(db_path=":memory:")
    await database.init()
    yield database
    await database.close()


# ─── Portfolio Snapshots ──────────────────────────────────────────────


class TestPortfolioSnapshots:
    @pytest.mark.asyncio
    async def test_insert_and_get_snapshot(self, db):
        """Round-trip : insert + get."""
        snapshot = {
            "timestamp": "2026-02-17T12:00:00+00:00",
            "equity": 10500.0,
            "capital": 9800.0,
            "margin_used": 1200.0,
            "margin_ratio": 0.12,
            "realized_pnl": 300.0,
            "unrealized_pnl": 400.0,
            "n_positions": 5,
            "n_assets": 3,
            "breakdown": {"ICP/USDT": {"positions": 2, "unrealized": 150.0}},
        }
        await db.insert_portfolio_snapshot(snapshot)

        rows = await db.get_portfolio_snapshots()
        assert len(rows) == 1
        assert rows[0]["equity"] == 10500.0
        assert rows[0]["n_positions"] == 5
        assert rows[0]["capital"] == 9800.0

    @pytest.mark.asyncio
    async def test_filter_since_until(self, db):
        """Filtrage par timestamp."""
        for hour in range(5):
            await db.insert_portfolio_snapshot({
                "timestamp": f"2026-02-17T{10+hour:02d}:00:00+00:00",
                "equity": 10000 + hour * 100,
                "capital": 10000.0,
                "margin_used": 0.0,
                "margin_ratio": 0.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": hour * 100,
                "n_positions": 0,
                "n_assets": 0,
            })

        # Filtre since
        rows = await db.get_portfolio_snapshots(since="2026-02-17T12:00:00+00:00")
        assert len(rows) == 3  # 12h, 13h, 14h

        # Filtre until
        rows = await db.get_portfolio_snapshots(until="2026-02-17T11:00:00+00:00")
        assert len(rows) == 2  # 10h, 11h

        # Filtre combiné
        rows = await db.get_portfolio_snapshots(
            since="2026-02-17T11:00:00+00:00",
            until="2026-02-17T13:00:00+00:00",
        )
        assert len(rows) == 3  # 11h, 12h, 13h

    @pytest.mark.asyncio
    async def test_breakdown_json_roundtrip(self, db):
        """Le breakdown JSON est bien sérialisé/désérialisé."""
        breakdown = {
            "BTC/USDT": {"positions": 2, "unrealized": -50.0, "margin": 500.0},
            "ICP/USDT": {"positions": 3, "unrealized": 120.0, "margin": 300.0},
        }
        await db.insert_portfolio_snapshot({
            "timestamp": "2026-02-17T12:00:00+00:00",
            "equity": 10000.0,
            "capital": 10000.0,
            "margin_used": 800.0,
            "margin_ratio": 0.08,
            "realized_pnl": 0.0,
            "unrealized_pnl": 70.0,
            "n_positions": 5,
            "n_assets": 2,
            "breakdown": breakdown,
        })

        rows = await db.get_portfolio_snapshots()
        stored = json.loads(rows[0]["breakdown_json"])
        assert stored["BTC/USDT"]["positions"] == 2
        assert stored["ICP/USDT"]["unrealized"] == 120.0

    @pytest.mark.asyncio
    async def test_snapshot_without_breakdown(self, db):
        """Snapshot sans breakdown (aucune position)."""
        await db.insert_portfolio_snapshot({
            "timestamp": "2026-02-17T12:00:00+00:00",
            "equity": 10000.0,
            "capital": 10000.0,
            "margin_used": 0.0,
            "margin_ratio": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "n_positions": 0,
            "n_assets": 0,
        })

        rows = await db.get_portfolio_snapshots()
        assert len(rows) == 1
        assert rows[0]["breakdown_json"] is None

    @pytest.mark.asyncio
    async def test_get_latest_snapshot(self, db):
        """get_latest_snapshot retourne le plus récent (ORDER BY DESC)."""
        for hour in [10, 12, 14]:
            await db.insert_portfolio_snapshot({
                "timestamp": f"2026-02-17T{hour:02d}:00:00+00:00",
                "equity": 10000 + hour * 10,
                "capital": 10000.0,
                "margin_used": 0.0,
                "margin_ratio": 0.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "n_positions": 0,
                "n_assets": 0,
            })

        latest = await db.get_latest_snapshot()
        assert latest is not None
        assert "T14:" in latest["timestamp"]
        assert latest["equity"] == 10140.0

    @pytest.mark.asyncio
    async def test_get_latest_snapshot_empty(self, db):
        """get_latest_snapshot retourne None si aucun snapshot."""
        latest = await db.get_latest_snapshot()
        assert latest is None

    @pytest.mark.asyncio
    async def test_snapshots_limit(self, db):
        """Le paramètre limit est respecté."""
        for i in range(10):
            await db.insert_portfolio_snapshot({
                "timestamp": f"2026-02-17T{10+i:02d}:00:00+00:00",
                "equity": 10000.0,
                "capital": 10000.0,
                "margin_used": 0.0,
                "margin_ratio": 0.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "n_positions": 0,
                "n_assets": 0,
            })

        rows = await db.get_portfolio_snapshots(limit=3)
        assert len(rows) == 3


# ─── Position Events ──────────────────────────────────────────────────


class TestPositionEvents:
    @pytest.mark.asyncio
    async def test_insert_and_get_event(self, db):
        """Round-trip : insert + get."""
        event = {
            "timestamp": "2026-02-17T14:30:00+00:00",
            "strategy_name": "grid_atr",
            "symbol": "ICP/USDT",
            "event_type": "OPEN",
            "level": 0,
            "direction": "LONG",
            "price": 5.42,
            "quantity": 12.0,
            "margin_used": 10.84,
            "metadata": {"levels_open": 1, "levels_max": 3},
        }
        await db.insert_position_event(event)

        rows = await db.get_position_events()
        assert len(rows) == 1
        assert rows[0]["event_type"] == "OPEN"
        assert rows[0]["price"] == 5.42
        assert rows[0]["level"] == 0

    @pytest.mark.asyncio
    async def test_close_event_with_pnl(self, db):
        """Événement CLOSE avec P&L."""
        await db.insert_position_event({
            "timestamp": "2026-02-17T15:00:00+00:00",
            "strategy_name": "grid_atr",
            "symbol": "NEAR/USDT",
            "event_type": "CLOSE",
            "level": None,
            "direction": "LONG",
            "price": 4.10,
            "quantity": 8.0,
            "unrealized_pnl": 12.40,
            "metadata": {"exit_reason": "tp_global", "net_pnl": 12.40},
        })

        rows = await db.get_position_events()
        assert rows[0]["unrealized_pnl"] == 12.40
        meta = json.loads(rows[0]["metadata_json"])
        assert meta["exit_reason"] == "tp_global"

    @pytest.mark.asyncio
    async def test_filter_by_strategy_and_symbol(self, db):
        """Filtrage par stratégie et symbol."""
        for sym in ["BTC/USDT", "ICP/USDT", "BTC/USDT"]:
            await db.insert_position_event({
                "timestamp": "2026-02-17T14:00:00+00:00",
                "strategy_name": "grid_atr",
                "symbol": sym,
                "event_type": "OPEN",
                "direction": "LONG",
                "price": 100.0,
                "quantity": 1.0,
            })

        btc = await db.get_position_events(symbol="BTC/USDT")
        assert len(btc) == 2

        icp = await db.get_position_events(symbol="ICP/USDT")
        assert len(icp) == 1

    @pytest.mark.asyncio
    async def test_events_ordered_desc(self, db):
        """Les events sont triés par timestamp DESC (plus récent en premier)."""
        for i in range(3):
            await db.insert_position_event({
                "timestamp": f"2026-02-17T{10+i:02d}:00:00+00:00",
                "strategy_name": "grid_atr",
                "symbol": "BTC/USDT",
                "event_type": "OPEN",
                "direction": "LONG",
                "price": 100.0,
                "quantity": 1.0,
            })

        rows = await db.get_position_events()
        # 12h, 11h, 10h (DESC)
        assert "T12:" in rows[0]["timestamp"]
        assert "T10:" in rows[2]["timestamp"]

    @pytest.mark.asyncio
    async def test_event_without_optional_fields(self, db):
        """Événement minimal sans level, unrealized, margin, metadata."""
        await db.insert_position_event({
            "timestamp": "2026-02-17T14:00:00+00:00",
            "strategy_name": "grid_atr",
            "symbol": "BTC/USDT",
            "event_type": "OPEN",
            "direction": "LONG",
            "price": 68000.0,
            "quantity": 0.01,
        })

        rows = await db.get_position_events()
        assert len(rows) == 1
        assert rows[0]["level"] is None
        assert rows[0]["unrealized_pnl"] is None
        assert rows[0]["metadata_json"] is None


# ─── Simulator Snapshot ───────────────────────────────────────────────


class TestSimulatorSnapshot:
    @pytest.mark.asyncio
    async def test_take_journal_snapshot_no_runners(self):
        """Snapshot retourne None sans runners."""
        from unittest.mock import MagicMock
        from backend.backtesting.simulator import Simulator

        sim = Simulator(data_engine=MagicMock(), config=MagicMock())
        result = await sim.take_journal_snapshot()
        assert result is None

    @pytest.mark.asyncio
    async def test_take_journal_snapshot_with_runners(self):
        """Snapshot avec runners actifs retourne la structure attendue."""
        from unittest.mock import MagicMock
        from backend.backtesting.simulator import Simulator

        # Mock un runner avec get_status()
        runner = MagicMock()
        runner._initial_capital = 10000.0
        runner.get_status.return_value = {
            "capital": 9800.0,
            "net_pnl": -200.0,
            "unrealized_pnl": 150.0,
            "margin_used": 500.0,
            "open_positions": 3,
        }

        sim = Simulator(data_engine=MagicMock(), config=MagicMock())
        sim._runners = [runner]

        result = await sim.take_journal_snapshot()
        assert result is not None
        assert result["capital"] == 9800.0
        assert result["unrealized_pnl"] == 150.0
        assert result["margin_used"] == 500.0
        assert result["n_positions"] == 3
        assert result["equity"] == 9950.0  # capital + unrealized
        assert result["realized_pnl"] == -200.0
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_take_journal_snapshot_multiple_runners(self):
        """Snapshot agrège correctement plusieurs runners."""
        from unittest.mock import MagicMock
        from backend.backtesting.simulator import Simulator

        runner1 = MagicMock()
        runner1._initial_capital = 5000.0
        runner1.get_status.return_value = {
            "capital": 5100.0,
            "net_pnl": 100.0,
            "unrealized_pnl": 50.0,
            "margin_used": 200.0,
            "open_positions": 2,
        }

        runner2 = MagicMock()
        runner2._initial_capital = 5000.0
        runner2.get_status.return_value = {
            "capital": 4900.0,
            "net_pnl": -100.0,
            "unrealized_pnl": -30.0,
            "margin_used": 300.0,
            "open_positions": 1,
        }

        sim = Simulator(data_engine=MagicMock(), config=MagicMock())
        sim._runners = [runner1, runner2]

        result = await sim.take_journal_snapshot()
        assert result["capital"] == 10000.0  # 5100 + 4900
        assert result["unrealized_pnl"] == 20.0  # 50 + (-30)
        assert result["margin_used"] == 500.0  # 200 + 300
        assert result["n_positions"] == 3  # 2 + 1
        assert result["equity"] == 10020.0  # 10000 + 20
        assert result["realized_pnl"] == 0.0  # 100 + (-100)


# ─── API Routes ───────────────────────────────────────────────────────


class TestJournalAPI:
    @pytest.mark.asyncio
    async def test_snapshots_endpoint_empty(self):
        """GET /api/journal/snapshots retourne une liste vide sans DB."""
        from httpx import AsyncClient, ASGITransport
        from fastapi import FastAPI
        from backend.api.journal_routes import router

        app = FastAPI()
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/journal/snapshots")
        assert resp.status_code == 200
        data = resp.json()
        assert data["snapshots"] == []
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_events_endpoint_empty(self):
        """GET /api/journal/events retourne une liste vide sans DB."""
        from httpx import AsyncClient, ASGITransport
        from fastapi import FastAPI
        from backend.api.journal_routes import router

        app = FastAPI()
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/journal/events")
        assert resp.status_code == 200
        data = resp.json()
        assert data["events"] == []
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_summary_endpoint_empty(self):
        """GET /api/journal/summary retourne la structure attendue sans DB."""
        from httpx import AsyncClient, ASGITransport
        from fastapi import FastAPI
        from backend.api.journal_routes import router

        app = FastAPI()
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/journal/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["latest_snapshot"] is None
        assert data["recent_events"] == []

    @pytest.mark.asyncio
    async def test_snapshots_endpoint_with_data(self, db):
        """GET /api/journal/snapshots retourne les données insérées."""
        from httpx import AsyncClient, ASGITransport
        from fastapi import FastAPI
        from backend.api.journal_routes import router

        await db.insert_portfolio_snapshot({
            "timestamp": "2026-02-17T12:00:00+00:00",
            "equity": 10500.0,
            "capital": 10000.0,
            "margin_used": 500.0,
            "margin_ratio": 0.05,
            "realized_pnl": 200.0,
            "unrealized_pnl": 300.0,
            "n_positions": 3,
            "n_assets": 2,
            "breakdown": {"BTC/USDT": {"positions": 2}},
        })

        app = FastAPI()
        app.state.db = db
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/journal/snapshots")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        s = data["snapshots"][0]
        assert s["equity"] == 10500.0
        assert s["breakdown"]["BTC/USDT"]["positions"] == 2
        assert "breakdown_json" not in s  # Champ raw supprimé

    @pytest.mark.asyncio
    async def test_events_endpoint_with_data(self, db):
        """GET /api/journal/events retourne les données insérées."""
        from httpx import AsyncClient, ASGITransport
        from fastapi import FastAPI
        from backend.api.journal_routes import router

        await db.insert_position_event({
            "timestamp": "2026-02-17T14:30:00+00:00",
            "strategy_name": "grid_atr",
            "symbol": "ICP/USDT",
            "event_type": "OPEN",
            "direction": "LONG",
            "price": 5.42,
            "quantity": 12.0,
            "metadata": {"levels_open": 1},
        })

        app = FastAPI()
        app.state.db = db
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/journal/events")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        e = data["events"][0]
        assert e["event_type"] == "OPEN"
        assert e["metadata"]["levels_open"] == 1
        assert "metadata_json" not in e  # Champ raw supprimé

    @pytest.mark.asyncio
    async def test_summary_endpoint_with_data(self, db):
        """GET /api/journal/summary retourne le dernier snapshot et events."""
        from httpx import AsyncClient, ASGITransport
        from fastapi import FastAPI
        from backend.api.journal_routes import router

        # Insérer 2 snapshots
        for hour in [10, 14]:
            await db.insert_portfolio_snapshot({
                "timestamp": f"2026-02-17T{hour:02d}:00:00+00:00",
                "equity": 10000 + hour * 10,
                "capital": 10000.0,
                "margin_used": 0.0,
                "margin_ratio": 0.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "n_positions": 0,
                "n_assets": 0,
            })

        # Insérer 1 event
        await db.insert_position_event({
            "timestamp": "2026-02-17T14:30:00+00:00",
            "strategy_name": "grid_atr",
            "symbol": "BTC/USDT",
            "event_type": "OPEN",
            "direction": "LONG",
            "price": 68000.0,
            "quantity": 0.01,
        })

        app = FastAPI()
        app.state.db = db
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/journal/summary")
        assert resp.status_code == 200
        data = resp.json()
        # Le dernier snapshot est celui de 14h (pas 10h)
        assert data["latest_snapshot"] is not None
        assert "T14:" in data["latest_snapshot"]["timestamp"]
        assert data["total_events"] == 1
        assert data["recent_events"][0]["event_type"] == "OPEN"
