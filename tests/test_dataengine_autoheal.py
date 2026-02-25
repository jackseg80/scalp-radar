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
        engine = _make_engine(["BTC/USDT", "STALE/USDT"])
        engine._heartbeat_interval = 9999
        engine._last_candle_received = time.time() - 10
        engine.full_reconnect = AsyncMock()

        now = datetime.now(tz=timezone.utc)
        engine._last_update_per_symbol = {
            "BTC/USDT": now - timedelta(seconds=30),    # actif
            "STALE/USDT": now - timedelta(seconds=650),   # stale > 10 min
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

        engine.restart_stale_symbol.assert_awaited_once_with("STALE/USDT")

    @pytest.mark.asyncio
    async def test_no_heal_below_threshold(self):
        """Symbol stale 200s (< 5 min = seuil 300s) → PAS de restart."""
        engine = _make_engine(["BTC/USDT", "ETH/USDT"])
        engine._heartbeat_interval = 9999
        engine._last_candle_received = time.time() - 10
        engine.full_reconnect = AsyncMock()

        now = datetime.now(tz=timezone.utc)
        engine._last_update_per_symbol = {
            "BTC/USDT": now - timedelta(seconds=30),    # actif
            "ETH/USDT": now - timedelta(seconds=200),   # stale 3.3min < seuil 5min
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
    async def test_restart_stale_symbol_age_none(self):
        """Symbol avec age=None (jamais reçu) → restart + compteur incrémenté."""
        engine = _make_engine(["BTC/USDT", "DEAD/USDT"])
        engine._heartbeat_interval = 9999
        engine._last_candle_received = time.time() - 10
        engine.full_reconnect = AsyncMock()

        now = datetime.now(tz=timezone.utc)
        engine._last_update_per_symbol = {
            "BTC/USDT": now - timedelta(seconds=30),  # actif
            # DEAD/USDT absent → age=None dans le heartbeat
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

        engine.restart_stale_symbol.assert_awaited_once_with("DEAD/USDT")
        # Compteur incrémenté à 1
        assert engine._stale_restart_count["DEAD/USDT"] == 1

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


# ─── Tests backoff restart stale ─────────────────────────────────────────


class TestStaleBackoff:

    @pytest.mark.asyncio
    async def test_stale_backoff_stops_after_3_retries(self):
        """Symbol stale relancé 3 fois, puis abandonné au 4e tick."""
        engine = _make_engine(["BTC/USDT", "DEAD/USDT"])
        engine._heartbeat_interval = 9999
        engine._last_candle_received = time.time() - 10
        engine.full_reconnect = AsyncMock()
        engine.restart_dead_tasks = AsyncMock(return_value=0)
        engine.restart_stale_symbol = AsyncMock(return_value=True)

        # Run 4 heartbeat ticks (chacun déclenche le check stale)
        for _ in range(4):
            now = datetime.now(tz=timezone.utc)
            engine._last_update_per_symbol = {
                "BTC/USDT": now - timedelta(seconds=30),
                # DEAD/USDT absent → age=None
            }
            engine._heartbeat_tick = 4  # → 5 après incrément

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                mock_sleep.side_effect = [None, asyncio.CancelledError()]
                try:
                    await engine._heartbeat_loop()
                except asyncio.CancelledError:
                    pass

        # restart appelé 3 fois (pas 4)
        assert engine.restart_stale_symbol.await_count == 3
        # Symbol dans _stale_abandoned
        assert "DEAD/USDT" in engine._stale_abandoned
        # Compteur à 3
        assert engine._stale_restart_count.get("DEAD/USDT") == 3

    @pytest.mark.asyncio
    async def test_stale_backoff_resets_on_candle_received(self):
        """Symbol en compteur de restart → reset quand il reçoit une candle."""
        engine = _make_engine(["ETH/USDT"])

        # Simuler 2 tentatives précédentes
        engine._stale_restart_count["ETH/USDT"] = 2
        engine._stale_abandoned.add("ETH/USDT")

        # Simuler la réception d'une candle
        now_ts = time.time() * 1000
        ohlcv = [now_ts, 100.0, 105.0, 95.0, 102.0, 1000.0]

        await engine._on_candle_received("ETH/USDT", "1h", ohlcv)

        # Compteur reset
        assert "ETH/USDT" not in engine._stale_restart_count
        # Retiré de la liste abandonnée
        assert "ETH/USDT" not in engine._stale_abandoned

    @pytest.mark.asyncio
    async def test_stale_abandoned_skipped_in_heartbeat(self):
        """Symbol abandonné → restart_stale_symbol n'est PAS appelé."""
        engine = _make_engine(["BTC/USDT", "DEAD/USDT"])
        engine._heartbeat_interval = 9999
        engine._last_candle_received = time.time() - 10
        engine.full_reconnect = AsyncMock()
        engine.restart_dead_tasks = AsyncMock(return_value=0)
        engine.restart_stale_symbol = AsyncMock(return_value=True)

        # Marquer comme abandonné
        engine._stale_abandoned.add("DEAD/USDT")
        engine._stale_restart_count["DEAD/USDT"] = 3

        now = datetime.now(tz=timezone.utc)
        engine._last_update_per_symbol = {
            "BTC/USDT": now - timedelta(seconds=30),
            # DEAD/USDT absent → age=None, mais abandonné
        }

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            engine._heartbeat_tick = 4
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            try:
                await engine._heartbeat_loop()
            except asyncio.CancelledError:
                pass

        engine.restart_stale_symbol.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stale_abandon_notifies_telegram(self):
        """Abandon après 3 tentatives → alerte Telegram envoyée."""
        notifier = AsyncMock()
        notifier.notify_anomaly = AsyncMock()
        engine = _make_engine(["BTC/USDT", "DEAD/USDT"], notifier=notifier)
        engine._heartbeat_interval = 9999
        engine._last_candle_received = time.time() - 10
        engine.full_reconnect = AsyncMock()
        engine.restart_dead_tasks = AsyncMock(return_value=0)
        engine.restart_stale_symbol = AsyncMock(return_value=True)

        # Déjà 3 tentatives
        engine._stale_restart_count["DEAD/USDT"] = 3

        now = datetime.now(tz=timezone.utc)
        engine._last_update_per_symbol = {
            "BTC/USDT": now - timedelta(seconds=30),
        }

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            engine._heartbeat_tick = 4
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            try:
                await engine._heartbeat_loop()
            except asyncio.CancelledError:
                pass

        # Alerte envoyée
        notifier.notify_anomaly.assert_awaited()
        call_args = notifier.notify_anomaly.call_args
        assert "abandonné" in call_args[0][1]
        assert "DEAD/USDT" in call_args[0][1]


# ─── Test config assets ─────────────────────────────────────────────────


class TestConfigAssets:

    def test_config_no_xtz_jup(self):
        """XTZ/USDT et JUP/USDT ne sont plus dans assets.yaml (21 assets dont SUI/USDT re-ajouté)."""
        import yaml

        with open("config/assets.yaml") as f:
            data = yaml.safe_load(f)

        symbols = [a["symbol"] for a in data["assets"]]
        assert "XTZ/USDT" not in symbols, "XTZ/USDT encore présent"
        assert "JUP/USDT" not in symbols, "JUP/USDT encore présent"
        assert len(symbols) == 21, f"Attendu 21 assets, trouvé {len(symbols)}"


# ─── Test route prefix ────────────────────────────────────────────────────


class TestDataRoutePrefix:

    def test_api_data_status_path(self):
        """GET /api/data/status est la bonne route (pas /data/status)."""
        from backend.api.data_routes import router

        paths = [route.path for route in router.routes]
        assert "/api/data/status" in paths
        assert router.prefix == "/api/data"
