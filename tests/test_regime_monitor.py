"""Tests du Regime Monitor — Sprint 61."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.regime.regime_monitor import (
    RegimeMonitor,
    RegimeSnapshot,
    _DailyBar,
    _classify_volatility,
    _resample_1h_to_daily,
    compute_regime_snapshot,
)


# ─── HELPERS ─────────────────────────────────────────────────────────────────


def _make_candle(ts: datetime, price: float, *, vol: float = 1000.0):
    """Candle-like object compatible avec db.get_candles() et _classify_regime."""
    return SimpleNamespace(
        timestamp=ts,
        open=price,
        high=price * 1.005,
        low=price * 0.995,
        close=price,
        volume=vol,
    )


def _make_1h_candles(
    n_days: int = 45,
    base_price: float = 50_000.0,
    daily_change: float = 0.0,
) -> list:
    """Génère n_days × 24 candles 1h avec un trend linéaire."""
    candles = []
    now = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=n_days)
    for h in range(n_days * 24):
        ts = start + timedelta(hours=h)
        day_idx = h // 24
        price = base_price * (1 + daily_change * day_idx)
        candles.append(_make_candle(ts, price))
    return candles


def _make_db(candles=None):
    db = MagicMock()
    if candles is None:
        candles = _make_1h_candles()
    db.get_candles = AsyncMock(return_value=candles)
    return db


# ─── TESTS CLASSIFY VOLATILITY ──────────────────────────────────────────────


def test_classify_volatility_low():
    """ATR < 2% → LOW, leverage 3."""
    level, lev = _classify_volatility(1.5)
    assert level == "LOW"
    assert lev == 3


def test_classify_volatility_medium():
    """ATR 2-3% → MEDIUM, leverage 4."""
    level, lev = _classify_volatility(2.5)
    assert level == "MEDIUM"
    assert lev == 4


def test_classify_volatility_high():
    """ATR > 4% → HIGH, leverage 6."""
    level, lev = _classify_volatility(5.0)
    assert level == "HIGH"
    assert lev == 6


# ─── TESTS RESAMPLE ─────────────────────────────────────────────────────────


def test_resample_with_gaps():
    """Les jours avec < 20 candles sont ignorés."""
    now = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    candles = []
    # Jour 1 : 24 candles (complet)
    day1 = now - timedelta(days=2)
    for h in range(24):
        candles.append(_make_candle(day1 + timedelta(hours=h), 50_000.0))
    # Jour 2 : seulement 10 candles (incomplet → ignoré)
    day2 = now - timedelta(days=1)
    for h in range(10):
        candles.append(_make_candle(day2 + timedelta(hours=h), 51_000.0))
    # Jour 3 : 22 candles (suffisant)
    day3 = now
    for h in range(22):
        candles.append(_make_candle(day3 + timedelta(hours=h), 52_000.0))

    daily = _resample_1h_to_daily(candles)
    assert len(daily) == 2  # Jour 2 ignoré
    assert daily[0].close == 50_000.0
    assert daily[1].close == 52_000.0


# ─── TESTS COMPUTE SNAPSHOT ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compute_snapshot_returns_valid():
    """compute_regime_snapshot retourne un RegimeSnapshot valide."""
    db = _make_db()
    snapshot = await compute_regime_snapshot(db)
    assert isinstance(snapshot, RegimeSnapshot)
    assert snapshot.regime in ("BULL", "BEAR", "RANGE", "CRASH")
    assert snapshot.suggested_leverage in (3, 4, 5, 6)
    assert snapshot.regime_days >= 1
    assert isinstance(snapshot.btc_atr_14d_pct, float)
    assert isinstance(snapshot.btc_change_30d_pct, float)


@pytest.mark.asyncio
async def test_compute_snapshot_no_candles():
    """Pas de candles → ValueError."""
    db = _make_db(candles=[])
    with pytest.raises(ValueError, match="candles"):
        await compute_regime_snapshot(db)


# ─── TESTS MONITOR ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_monitor_start_stop():
    """Start et stop fonctionnent sans erreur."""
    db = _make_db()
    monitor = RegimeMonitor(telegram=None, db=db)
    await monitor.start()
    assert monitor._running is True
    assert monitor.latest is not None
    await monitor.stop()
    assert monitor._running is False


@pytest.mark.asyncio
async def test_monitor_telegram_format():
    """Le message Telegram contient les infos clés."""
    db = _make_db()
    monitor = RegimeMonitor(telegram=None, db=db)
    await monitor.start()
    snapshot = monitor.latest
    assert snapshot is not None
    msg = monitor._format_telegram(snapshot)
    assert "REGIME MONITOR" in msg
    assert snapshot.regime in msg
    assert str(snapshot.suggested_leverage) in msg
    await monitor.stop()


def test_seconds_until_0005_positive():
    """Le calcul du prochain 00:05 UTC est toujours dans le futur, ≤ 24h."""
    seconds = RegimeMonitor._seconds_until_next_0005_utc()
    assert 0 < seconds <= 24 * 3600


# ─── TESTS API ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_api_snapshot_200():
    """GET /api/regime/snapshot retourne 200 avec les bons champs."""
    from fastapi.testclient import TestClient

    from backend.api.regime_routes import router

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    # Simuler un RegimeMonitor avec un snapshot
    db = _make_db()
    monitor = RegimeMonitor(telegram=None, db=db)
    await monitor.start()
    app.state.regime_monitor = monitor

    client = TestClient(app)
    resp = client.get("/api/regime/snapshot")
    assert resp.status_code == 200
    body = resp.json()
    assert "snapshot" in body
    snap = body["snapshot"]
    assert snap["regime"] in ("BULL", "BEAR", "RANGE", "CRASH")
    assert "btc_atr_14d_pct" in snap
    assert "suggested_leverage" in snap

    await monitor.stop()


@pytest.mark.asyncio
async def test_api_history_200():
    """GET /api/regime/history retourne 200 avec une liste."""
    from fastapi.testclient import TestClient

    from backend.api.regime_routes import router

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    db = _make_db()
    monitor = RegimeMonitor(telegram=None, db=db)
    await monitor.start()
    app.state.regime_monitor = monitor

    client = TestClient(app)
    resp = client.get("/api/regime/history?days=30")
    assert resp.status_code == 200
    body = resp.json()
    assert "history" in body
    assert isinstance(body["history"], list)
    assert body["count"] >= 1

    await monitor.stop()
