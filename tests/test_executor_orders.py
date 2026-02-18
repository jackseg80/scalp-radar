"""Tests pour l'historique d'ordres Executor — Sprint 32."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from fastapi import FastAPI

from backend.api.executor_routes import router


def _make_order_result(order_id: str = "ord_123", filled: float = 0.5,
                       average: float = 100.0, status: str = "closed") -> dict:
    """Fabrique un resultat d'ordre factice."""
    return {
        "id": order_id,
        "filled": filled,
        "average": average,
        "status": status,
    }


class TestOrderHistoryUnit:
    """Tests unitaires deque _order_history."""

    def test_order_history_initialized(self):
        """deque maxlen=200 a l'init."""
        d = deque(maxlen=200)
        assert d.maxlen == 200
        assert len(d) == 0

    def test_record_order_format(self):
        """Un appel appendleft produit le bon format."""
        history: deque[dict] = deque(maxlen=200)
        order_result = _make_order_result()

        # Simuler _record_order
        record = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "order_type": "entry",
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "quantity": 0.001,
            "filled": float(order_result.get("filled") or 0),
            "average_price": float(order_result.get("average") or 0),
            "order_id": order_result.get("id", ""),
            "status": order_result.get("status", ""),
            "strategy_name": "grid_atr",
            "context": "mono",
        }
        history.appendleft(record)

        assert len(history) == 1
        entry = history[0]
        assert entry["order_type"] == "entry"
        assert entry["symbol"] == "BTC/USDT:USDT"
        assert entry["filled"] == 0.5
        assert entry["average_price"] == 100.0
        assert entry["order_id"] == "ord_123"
        assert entry["strategy_name"] == "grid_atr"

    def test_order_history_maxlen(self):
        """250 inserts → seuls 200 gardes (FIFO)."""
        history: deque[dict] = deque(maxlen=200)
        for i in range(250):
            history.appendleft({"order_id": f"ord_{i}"})

        assert len(history) == 200
        # Le plus recent est en premier
        assert history[0]["order_id"] == "ord_249"
        # Le plus ancien garde est ord_50 (249-199=50)
        assert history[-1]["order_id"] == "ord_50"

    def test_order_history_persistence_roundtrip(self):
        """Serialisation list() et restore deque()."""
        history: deque[dict] = deque(maxlen=200)
        for i in range(5):
            history.appendleft({
                "order_type": "entry",
                "order_id": f"ord_{i}",
                "symbol": "BTC/USDT:USDT",
            })

        # Serialiser (comme get_state_for_persistence)
        serialized = list(history)
        assert isinstance(serialized, list)
        assert len(serialized) == 5

        # Restaurer (comme restore_positions)
        restored = deque(serialized, maxlen=200)
        assert len(restored) == 5
        assert restored.maxlen == 200
        assert restored[0]["order_id"] == "ord_4"
        assert restored[-1]["order_id"] == "ord_0"


class TestOrdersEndpoint:
    """Tests endpoint GET /api/executor/orders."""

    @pytest_asyncio.fixture
    def app_no_executor(self) -> FastAPI:
        """App sans executor."""
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest_asyncio.fixture
    def app_with_orders(self) -> FastAPI:
        """App avec un executor mock ayant des ordres."""
        app = FastAPI()
        app.include_router(router)

        class MockExecutor:
            def __init__(self):
                self._order_history: deque[dict] = deque(maxlen=200)
                for i in range(10):
                    self._order_history.appendleft({
                        "timestamp": "2026-02-18T12:00:00+00:00",
                        "order_type": "entry" if i % 2 == 0 else "close",
                        "symbol": "BTC/USDT:USDT",
                        "side": "buy",
                        "quantity": 0.001,
                        "filled": 0.001,
                        "average_price": 50000.0 + i * 100,
                        "order_id": f"ord_{i}",
                        "status": "closed",
                        "strategy_name": "grid_atr",
                        "context": "mono",
                    })

        app.state.executor = MockExecutor()
        return app

    @pytest.mark.asyncio
    async def test_orders_endpoint_no_executor(self, app_no_executor):
        """Retourne vide si executor absent."""
        transport = ASGITransport(app=app_no_executor)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/executor/orders")

        assert resp.status_code == 200
        data = resp.json()
        assert data["orders"] == []
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_orders_endpoint_with_data(self, app_with_orders):
        """Retourne les ordres avec limit par defaut."""
        transport = ASGITransport(app=app_with_orders)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/executor/orders")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 10
        assert len(data["orders"]) == 10
        # Le plus recent en premier
        assert data["orders"][0]["order_id"] == "ord_9"

    @pytest.mark.asyncio
    async def test_orders_endpoint_with_limit(self, app_with_orders):
        """Limit respecte le parametre."""
        transport = ASGITransport(app=app_with_orders)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/executor/orders?limit=3")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3
        assert len(data["orders"]) == 3
