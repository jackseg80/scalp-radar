"""Tests pour l'auto-guérison per-symbol DataEngine — restart_stale_symbol + escalade."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.data_engine import DataEngine


# ─── Helpers ──────────────────────────────────────────────────────────────


def _make_config(symbols: list[str] | None = None) -> MagicMock:
    config = MagicMock()
    if symbols is None:
        symbols = []
    config.assets = [
        MagicMock(symbol=s, timeframes=["1h"]) for s in symbols
    ]
    config.exchange.websocket.reconnect_delay = 1
    config.exchange.websocket.max_reconnect_attempts = 3
    return config


def _make_engine(
    symbols: list[str] | None = None, notifier=None
) -> DataEngine:
    db = MagicMock()
    db.insert_candles_batch = AsyncMock()
    engine = DataEngine(
        config=_make_config(symbols), database=db, notifier=notifier
    )
    engine._running = True
    return engine


def _make_alive_task(name: str) -> MagicMock:
    """Crée un mock de task asyncio vivante (not done)."""
    task = MagicMock()
    task.get_name.return_value = name
    task.done.return_value = False
    task.cancelled.return_value = False
    task.cancel = MagicMock()
    return task


def _make_dead_task(name: str) -> MagicMock:
    """Crée un mock de task asyncio terminée (done)."""
    task = MagicMock()
    task.get_name.return_value = name
    task.done.return_value = True
    task.cancelled.return_value = False
    return task


# ─── Tests restart_stale_symbol ───────────────────────────────────────────


class TestRestartStaleSymbol:

    @pytest.mark.asyncio
    async def test_cancels_and_relaunches(self):
        """Symbol stale, task vivante → cancel + new task."""
        engine = _make_engine(["BTC/USDT"])

        old_task = _make_alive_task("watch_BTC/USDT")
        engine._tasks = [old_task]

        with patch("asyncio.create_task") as mock_create, \
             patch("asyncio.wait_for", new_callable=AsyncMock):
            new_task = MagicMock()
            new_task.get_name.return_value = "watch_BTC/USDT"
            mock_create.return_value = new_task

            result = await engine.restart_stale_symbol("BTC/USDT")

        assert result is True
        old_task.cancel.assert_called_once()
        assert engine._tasks[0] is new_task

    @pytest.mark.asyncio
    async def test_unknown_symbol_returns_false(self):
        """Symbol pas dans config → return False, pas de crash."""
        engine = _make_engine(["BTC/USDT"])

        task = _make_alive_task("watch_XYZ/USDT")
        engine._tasks = [task]

        # XYZ/USDT a une task mais pas dans config.assets
        result = await engine.restart_stale_symbol("XYZ/USDT")

        # La task existe mais le symbol n'est pas dans config → False
        assert result is False

    @pytest.mark.asyncio
    async def test_task_not_found_returns_false(self):
        """Pas de task matching → return False."""
        engine = _make_engine(["BTC/USDT"])
        engine._tasks = [_make_alive_task("watch_ETH/USDT")]

        result = await engine.restart_stale_symbol("BTC/USDT")
        assert result is False

    @pytest.mark.asyncio
    async def test_already_done_task_relaunched(self):
        """Task déjà done → relancée quand même (pas de cancel nécessaire)."""
        engine = _make_engine(["BTC/USDT"])

        dead_task = _make_dead_task("watch_BTC/USDT")
        engine._tasks = [dead_task]

        with patch("asyncio.create_task") as mock_create:
            new_task = MagicMock()
            new_task.get_name.return_value = "watch_BTC/USDT"
            mock_create.return_value = new_task

            result = await engine.restart_stale_symbol("BTC/USDT")

        assert result is True
        # Task déjà done → cancel pas appelé
        dead_task.cancel.assert_not_called()
        assert engine._tasks[0] is new_task


# ─── Tests heartbeat auto-heal ────────────────────────────────────────────


class TestHeartbeatAutoHeal:

    @pytest.mark.asyncio
    async def test_heals_stale_symbol_after_10min(self):
        """Symbol stale 650s → restart_stale_symbol appelé."""
        engine = _make_engine(["BTC/USDT", "XTZ/USDT"])
        engine._heartbeat_interval = 9999
        engine._last_candle_received = time.time() - 10
        engine.full_reconnect = AsyncMock()

        now = datetime.now(tz=timezone.utc)
        engine._last_update_per_symbol = {
            "BTC/USDT": now - timedelta(seconds=30),    # actif
            "XTZ/USDT": now - timedelta(seconds=650),   # stale > 10 min
        }

        engine.restart_stale_symbol = AsyncMock(return_value=True)
        engine.restart_dead_tasks = AsyncMock(return_value=0)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            engine._heartbeat_tick = 4  # → 5 après incrément
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            try:
                await engine._heartbeat_loop()
            except asyncio.CancelledError:
                pass

        engine.restart_stale_symbol.assert_awaited_once_with("XTZ/USDT")

    @pytest.mark.asyncio
    async def test_no_heal_under_10min(self):
        """Symbol stale 400s → PAS de restart (seulement log)."""
        engine = _make_engine(["BTC/USDT", "ETH/USDT"])
        engine._heartbeat_interval = 9999
        engine._last_candle_received = time.time() - 10
        engine.full_reconnect = AsyncMock()

        now = datetime.now(tz=timezone.utc)
        engine._last_update_per_symbol = {
            "BTC/USDT": now - timedelta(seconds=30),    # actif
            "ETH/USDT": now - timedelta(seconds=400),   # stale 6.6min < 10min
        }

        engine.restart_stale_symbol = AsyncMock(return_value=True)
        engine.restart_dead_tasks = AsyncMock(return_value=0)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            engine._heartbeat_tick = 4
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            try:
                await engine._heartbeat_loop()
            except asyncio.CancelledError:
                pass

        engine.restart_stale_symbol.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_full_reconnect_on_mass_stale(self):
        """>50% symbols stale >15min → full_reconnect."""
        symbols = [f"SYM{i}/USDT" for i in range(12)]
        engine = _make_engine(symbols)
        engine._heartbeat_interval = 9999
        engine._last_candle_received = time.time() - 10

        now = datetime.now(tz=timezone.utc)
        # 2 actifs, 10 stale > 15 min → >50% et > 5 long_stale
        engine._last_update_per_symbol = {
            "SYM0/USDT": now - timedelta(seconds=30),
            "SYM1/USDT": now - timedelta(seconds=30),
        }
        for i in range(2, 12):
            engine._last_update_per_symbol[f"SYM{i}/USDT"] = (
                now - timedelta(seconds=1000)  # > 15 min
            )

        engine.full_reconnect = AsyncMock()
        engine.restart_stale_symbol = AsyncMock(return_value=True)
        engine.restart_dead_tasks = AsyncMock(return_value=0)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            engine._heartbeat_tick = 4
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            try:
                await engine._heartbeat_loop()
            except asyncio.CancelledError:
                pass

        engine.full_reconnect.assert_awaited_once()


# ─── Test route prefix ────────────────────────────────────────────────────


class TestDataRoutePrefix:

    def test_api_data_status_path(self):
        """GET /api/data/status est la bonne route (pas /data/status)."""
        from backend.api.data_routes import router

        paths = [route.path for route in router.routes]
        assert "/api/data/status" in paths
        assert router.prefix == "/api/data"
