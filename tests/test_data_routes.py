"""Tests pour GET /api/data/status — monitoring per-symbol DataEngine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from backend.api.data_routes import data_status


# ─── Helpers ──────────────────────────────────────────────────────────────


def _make_request(engine=None) -> MagicMock:
    req = MagicMock()
    req.app.state.engine = engine
    return req


def _make_engine(symbols: list[str], per_symbol_ages: dict[str, float | None]) -> MagicMock:
    """engine mock avec des ages per-symbol en secondes (None = jamais reçu)."""
    engine = MagicMock()
    engine.is_connected = True
    engine.get_all_symbols.return_value = symbols

    now = datetime.now(tz=timezone.utc)
    engine._last_update_per_symbol = {
        sym: (now - timedelta(seconds=age)) if age is not None else None
        for sym, age in per_symbol_ages.items()
        if age is not None
    }
    return engine


# ─── Tests ────────────────────────────────────────────────────────────────


class TestDataStatusEndpoint:

    @pytest.mark.asyncio
    async def test_no_engine_returns_empty(self):
        """Sans DataEngine, retourne connected=False et listes vides."""
        req = _make_request(engine=None)
        result = await data_status(req)

        assert result["connected"] is False
        assert result["total_symbols"] == 0
        assert result["active"] == 0
        assert result["stale"] == 0
        assert result["symbols"] == {}

    @pytest.mark.asyncio
    async def test_active_symbol_under_120s(self):
        """Symbol avec candle il y a 60s → status ok."""
        engine = _make_engine(["BTC/USDT"], {"BTC/USDT": 60.0})
        result = await data_status(_make_request(engine))

        assert result["symbols"]["BTC/USDT"]["status"] == "ok"
        assert result["symbols"]["BTC/USDT"]["last_update_ago_s"] is not None
        assert result["symbols"]["BTC/USDT"]["last_update_ago_s"] < 120
        assert result["active"] == 1
        assert result["stale"] == 0

    @pytest.mark.asyncio
    async def test_stale_symbol_over_120s(self):
        """Symbol avec candle il y a 200s → status stale."""
        engine = _make_engine(["ETH/USDT"], {"ETH/USDT": 200.0})
        result = await data_status(_make_request(engine))

        assert result["symbols"]["ETH/USDT"]["status"] == "stale"
        assert result["active"] == 0
        assert result["stale"] == 1

    @pytest.mark.asyncio
    async def test_never_received_symbol(self):
        """Symbol jamais reçu → status stale, last_update_ago_s=None."""
        engine = _make_engine(["DOGE/USDT"], {})  # pas dans per_symbol
        result = await data_status(_make_request(engine))

        assert result["symbols"]["DOGE/USDT"]["status"] == "stale"
        assert result["symbols"]["DOGE/USDT"]["last_update_ago_s"] is None

    @pytest.mark.asyncio
    async def test_mixed_symbols(self):
        """Mix actif/stale → compteurs corrects."""
        syms = ["BTC/USDT", "ETH/USDT", "DOGE/USDT"]
        engine = _make_engine(syms, {
            "BTC/USDT": 30.0,    # ok
            "ETH/USDT": 300.0,   # stale
            # DOGE/USDT absent  → stale
        })
        result = await data_status(_make_request(engine))

        assert result["total_symbols"] == 3
        assert result["active"] == 1
        assert result["stale"] == 2
        assert result["symbols"]["BTC/USDT"]["status"] == "ok"
        assert result["symbols"]["ETH/USDT"]["status"] == "stale"
        assert result["symbols"]["DOGE/USDT"]["status"] == "stale"

    @pytest.mark.asyncio
    async def test_connected_flag_forwarded(self):
        """Le champ connected reflète engine.is_connected."""
        engine = _make_engine(["BTC/USDT"], {"BTC/USDT": 10.0})
        engine.is_connected = False
        result = await data_status(_make_request(engine))
        assert result["connected"] is False
