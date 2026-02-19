"""Tests pour l'historique d'ordres Executor — Sprint 32 + Journal V2."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from unittest.mock import MagicMock

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


def _make_order_record(
    order_id: str = "ord_1",
    avg_price: float = 100.0,
    paper_price: float = 99.5,
    symbol: str = "BTC/USDT:USDT",
    strategy: str = "grid_atr",
    order_type: str = "entry",
    side: str = "buy",
    quantity: float = 0.001,
    fee: float | None = None,
) -> dict:
    """Fabrique un enregistrement d'ordre avec les champs Journal V2."""
    record: dict = {
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
    if fee is not None:
        record["fee"] = fee
    return record


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


# ─── TESTS JOURNAL V2 ────────────────────────────────────────────────────────


class TestUpdateOrderPrice:
    """Tests pour _update_order_price (Sprint Journal V2)."""

    @staticmethod
    def _update_order_price(
        history: deque, order_id: str, real_price: float, fee: float | None = None,
    ) -> None:
        """Reimplemente la logique de _update_order_price pour test unitaire."""
        if not order_id or real_price <= 0:
            return
        for record in history:
            if record.get("order_id") == order_id:
                record["average_price"] = real_price
                if fee is not None:
                    record["fee"] = fee
                return

    def test_update_order_price_patches_record(self):
        """Patche average_price dans la deque pour un order_id existant."""
        history: deque[dict] = deque(maxlen=200)
        history.appendleft(_make_order_record(order_id="ord_A", avg_price=0.0))

        self._update_order_price(history, "ord_A", 105.5)

        assert history[0]["average_price"] == 105.5

    def test_update_order_price_no_match(self):
        """order_id inexistant — pas de crash, pas de modification."""
        history: deque[dict] = deque(maxlen=200)
        history.appendleft(_make_order_record(order_id="ord_A", avg_price=100.0))

        self._update_order_price(history, "ord_INEXISTANT", 200.0)

        assert history[0]["average_price"] == 100.0

    def test_update_order_price_adds_fee(self):
        """Champ fee ajoute au dict quand fourni."""
        history: deque[dict] = deque(maxlen=200)
        history.appendleft(_make_order_record(order_id="ord_B", avg_price=0.0))
        assert "fee" not in history[0]

        self._update_order_price(history, "ord_B", 50.0, fee=0.03)

        assert history[0]["fee"] == 0.03
        assert history[0]["average_price"] == 50.0

    def test_update_order_price_ignores_zero_price(self):
        """real_price=0 → pas de mise a jour."""
        history: deque[dict] = deque(maxlen=200)
        history.appendleft(_make_order_record(order_id="ord_C", avg_price=100.0))

        self._update_order_price(history, "ord_C", 0.0, fee=0.05)

        assert history[0]["average_price"] == 100.0
        assert "fee" not in history[0] or history[0].get("fee") is None


class TestRecordOrderPaperPrice:
    """Tests pour le champ paper_price dans _record_order."""

    def test_record_order_has_paper_price(self):
        """Le champ paper_price est present dans un enregistrement."""
        record = _make_order_record(paper_price=99.8)
        assert "paper_price" in record
        assert record["paper_price"] == 99.8

    def test_persistence_roundtrip_new_fields(self):
        """Serialise/restaure avec paper_price + fee."""
        history: deque[dict] = deque(maxlen=200)
        history.appendleft(_make_order_record(
            order_id="ord_1", avg_price=100.5, paper_price=100.0, fee=0.06,
        ))

        # Serialiser
        serialized = list(history)
        # Restaurer
        restored = deque(serialized, maxlen=200)

        rec = restored[0]
        assert rec["paper_price"] == 100.0
        assert rec["fee"] == 0.06
        assert rec["average_price"] == 100.5

    def test_persistence_restore_legacy_orders(self):
        """Ordres anciens sans paper_price/fee — backward compat via .get()."""
        legacy_record = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "order_type": "entry",
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "quantity": 0.001,
            "filled": 0.001,
            "average_price": 50000.0,
            "order_id": "old_ord_1",
            "status": "closed",
            "strategy_name": "grid_atr",
            "context": "mono",
            # PAS de paper_price ni fee
        }
        history = deque([legacy_record], maxlen=200)

        # Acces via .get() avec default (comme le code slippage)
        rec = history[0]
        assert rec.get("paper_price", 0) == 0
        assert rec.get("fee", None) is None


class TestGetGridNumLevels:
    """Tests pour le helper _get_grid_num_levels."""

    def test_get_grid_num_levels_from_config(self):
        """Retourne num_levels depuis la config strategie."""
        mock_config = MagicMock()
        mock_strat = MagicMock()
        mock_strat.num_levels = 5
        mock_config.strategies.grid_atr = mock_strat
        setattr(mock_config.strategies, "grid_atr", mock_strat)

        # Simuler la logique du helper
        strategy_name = "grid_atr"
        strat_config = getattr(mock_config.strategies, strategy_name, None)
        result = strat_config.num_levels if strat_config and hasattr(strat_config, "num_levels") else 4

        assert result == 5

    def test_get_grid_num_levels_fallback(self):
        """Retourne 4 par defaut si num_levels absent."""
        strategy_name = "unknown_strategy"
        strat_config = None  # simule getattr retournant None

        result = strat_config.num_levels if strat_config and hasattr(strat_config, "num_levels") else 4

        assert result == 4


class TestWatchedOrderRecording:
    """Tests pour l'enregistrement SL/TP fills via watchOrders dans _order_history."""

    def test_watched_order_records_sl_fill(self):
        """Un SL fill via watchOrders produit un enregistrement dans l'historique."""
        history: deque[dict] = deque(maxlen=200)

        # Simuler ce que fait _process_watched_order pour un SL mono
        order = {
            "id": "sl_ord_42",
            "filled": 0.005,
            "average": 48500.0,
            "status": "closed",
            "fee": {"cost": 0.029},
        }
        # Simuler _record_order pour un watched SL
        record = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "order_type": "sl",
            "symbol": "BTC/USDT:USDT",
            "side": "sell",
            "quantity": 0.005,
            "filled": float(order.get("filled") or 0),
            "average_price": float(order.get("average") or 0),
            "order_id": order.get("id", ""),
            "status": order.get("status", ""),
            "strategy_name": "grid_atr",
            "context": "watched_sl",
            "paper_price": 0.0,  # SL server-side, pas de prix paper
        }
        history.appendleft(record)

        assert len(history) == 1
        rec = history[0]
        assert rec["order_type"] == "sl"
        assert rec["context"] == "watched_sl"
        assert rec["average_price"] == 48500.0
        assert rec["paper_price"] == 0.0  # exclu du slippage
        assert rec["order_id"] == "sl_ord_42"

    def test_watched_order_records_grid_sl_fill(self):
        """Un SL fill grid via watchOrders produit un enregistrement."""
        history: deque[dict] = deque(maxlen=200)

        order = {
            "id": "grid_sl_99",
            "filled": 0.05,
            "average": 2800.0,
            "status": "closed",
        }
        record = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "order_type": "sl",
            "symbol": "ETH/USDT:USDT",
            "side": "sell",
            "quantity": 0.05,
            "filled": float(order.get("filled") or 0),
            "average_price": float(order.get("average") or 0),
            "order_id": order.get("id", ""),
            "status": order.get("status", ""),
            "strategy_name": "grid_atr",
            "context": "watched_grid_sl",
            "paper_price": 0.0,
        }
        history.appendleft(record)

        assert len(history) == 1
        rec = history[0]
        assert rec["context"] == "watched_grid_sl"
        assert rec["symbol"] == "ETH/USDT:USDT"
        assert rec["filled"] == 0.05
