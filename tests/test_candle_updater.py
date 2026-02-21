"""Tests pour CandleUpdater + endpoints API backfill/candle-status."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from backend.core.candle_updater import CandleUpdater
from backend.core.database import Database


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def db(tmp_path):
    """Base de données temporaire initialisée."""
    db = Database(db_path=str(tmp_path / "test.db"))
    await db.init()
    yield db
    await db.close()


@pytest.fixture
def mock_config():
    """Config avec 2 assets."""
    config = MagicMock()
    asset1 = MagicMock()
    asset1.symbol = "BTC/USDT"
    asset2 = MagicMock()
    asset2.symbol = "ETH/USDT"
    config.assets = [asset1, asset2]
    return config


# ─── Tests get_candle_stats ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_candle_stats_empty(db):
    """DB vide retourne candle_count=0 et is_stale=True."""
    stats = await db.get_candle_stats("BTC/USDT", "1h", exchange="binance")
    assert stats["candle_count"] == 0
    assert stats["days_available"] == 0
    assert stats["is_stale"] is True
    assert stats["first_candle"] is None
    assert stats["last_candle"] is None


@pytest.mark.asyncio
async def test_get_candle_stats_with_data(db):
    """Avec des candles insérées, retourne les bonnes stats."""
    from backend.core.models import Candle, TimeFrame

    now = datetime.now(tz=timezone.utc)
    first_ts = now - timedelta(days=30)
    candles = []
    for i in range(10):
        ts = first_ts + timedelta(hours=i)
        candles.append(
            Candle(
                timestamp=ts,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000.0,
                symbol="BTC/USDT",
                timeframe=TimeFrame.H1,
                exchange="binance",
            )
        )
    # Ajouter une candle récente pour tester is_stale=False
    candles.append(
        Candle(
            timestamp=now - timedelta(minutes=30),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000.0,
            symbol="BTC/USDT",
            timeframe=TimeFrame.H1,
            exchange="binance",
        )
    )
    await db.insert_candles_batch(candles)

    stats = await db.get_candle_stats("BTC/USDT", "1h", exchange="binance")
    assert stats["candle_count"] == 11
    assert stats["days_available"] >= 29
    assert stats["is_stale"] is False
    assert stats["first_candle"] is not None
    assert stats["last_candle"] is not None


# ─── Tests CandleUpdater ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_backfill_already_running(mock_config):
    """Si _running=True, run_backfill() raise RuntimeError."""
    db_mock = MagicMock()
    updater = CandleUpdater(mock_config, db_mock)
    updater._running = True

    with pytest.raises(RuntimeError, match="déjà en cours"):
        await updater.run_backfill()


@pytest.mark.asyncio
async def test_broadcast_progress(mock_config):
    """Vérifie que le broadcast WS est appelé avec le bon format."""
    db_mock = MagicMock()
    ws_broadcast = AsyncMock()
    updater = CandleUpdater(mock_config, db_mock, ws_broadcast=ws_broadcast)

    await updater._broadcast_progress(5, 10, "BTC/USDT", "binance")

    ws_broadcast.assert_called_once()
    call_data = ws_broadcast.call_args[0][0]
    assert call_data["type"] == "backfill_progress"
    assert call_data["data"]["progress_pct"] == 50.0
    assert call_data["data"]["done"] == 5
    assert call_data["data"]["total"] == 10
    assert call_data["data"]["running"] is True
    assert "BTC/USDT" in call_data["data"]["current"]


@pytest.mark.asyncio
async def test_daily_loop_calculates_next_run(mock_config):
    """Vérifie le calcul du prochain 03:00 UTC."""
    db_mock = MagicMock()
    updater = CandleUpdater(mock_config, db_mock)

    # Monkey-patch asyncio.sleep pour capturer le délai sans attendre
    captured_delay = None

    async def fake_sleep(seconds):
        nonlocal captured_delay
        captured_delay = seconds
        raise asyncio.CancelledError()  # Sortir de la boucle immédiatement

    with patch("backend.core.candle_updater.asyncio.sleep", side_effect=fake_sleep):
        with pytest.raises(asyncio.CancelledError):
            await updater._daily_loop()

    assert captured_delay is not None
    # Le délai doit être entre 0 et 24h
    assert 0 < captured_delay <= 86400


# ─── Tests endpoints API ────────────────────────────────────────────────


@pytest.fixture
def mock_app_with_updater(mock_config):
    """Configure l'app avec un CandleUpdater mock."""
    from backend.api.server import app

    updater = MagicMock(spec=CandleUpdater)
    updater.is_running = False
    updater.get_status = AsyncMock(return_value={
        "running": False,
        "assets": {
            "BTC/USDT": {
                "binance": {
                    "first_candle": "2020-01-01T00:00:00+00:00",
                    "last_candle": "2024-01-15T12:00:00+00:00",
                    "candle_count": 35000,
                    "days_available": 1475,
                    "is_stale": False,
                },
                "bitget": {
                    "first_candle": None,
                    "last_candle": None,
                    "candle_count": 0,
                    "days_available": 0,
                    "is_stale": True,
                },
            }
        },
    })
    updater.run_backfill = AsyncMock(return_value={})

    app.state.candle_updater = updater
    app.state.engine = None
    app.state.config = mock_config
    app.state.db = MagicMock()
    app.state.simulator = None
    app.state.arena = None
    app.state.start_time = MagicMock()
    app.state.start_time.isoformat.return_value = "2024-01-15T12:00:00"

    return app, updater


@pytest.mark.asyncio
async def test_status_endpoint(mock_app_with_updater):
    """GET /api/data/candle-status retourne les stats."""
    app, updater = mock_app_with_updater
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/data/candle-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is False
    assert "BTC/USDT" in data["assets"]
    assert data["assets"]["BTC/USDT"]["binance"]["candle_count"] == 35000


@pytest.mark.asyncio
async def test_backfill_endpoint(mock_app_with_updater):
    """POST /api/data/backfill démarre un backfill."""
    app, updater = mock_app_with_updater
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/data/backfill")
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "started"


@pytest.mark.asyncio
async def test_backfill_endpoint_already_running(mock_app_with_updater):
    """POST quand déjà en cours retourne 409."""
    app, updater = mock_app_with_updater
    updater.is_running = True
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/data/backfill")
    assert resp.status_code == 409
    data = resp.json()
    assert data["status"] == "already_running"
