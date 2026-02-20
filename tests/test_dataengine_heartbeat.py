"""Tests pour le heartbeat DataEngine — détection silence WS + auto-reconnect."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.data_engine import DataEngine


# ─── Helpers ──────────────────────────────────────────────────────────────


def _make_config() -> MagicMock:
    config = MagicMock()
    config.assets = []
    config.exchange.websocket.reconnect_delay = 1
    config.exchange.websocket.max_reconnect_attempts = 3
    return config


def _make_engine(notifier=None) -> DataEngine:
    db = MagicMock()
    db.insert_candles_batch = AsyncMock()
    engine = DataEngine(config=_make_config(), database=db, notifier=notifier)
    engine._running = True
    return engine


# ─── Tests ────────────────────────────────────────────────────────────────


class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_heartbeat_triggers_reconnect_on_silence(self):
        """Si aucune candle depuis >5 min, full_reconnect est appelé."""
        engine = _make_engine()
        engine._heartbeat_interval = 300
        # Simuler qu'aucune candle n'est reçue depuis 310s
        engine._last_candle_received = time.time() - 310

        engine.full_reconnect = AsyncMock()

        # Lancer la boucle heartbeat et l'interrompre après 1 itération
        async def run_one_tick():
            task = asyncio.create_task(engine._heartbeat_loop())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Patcher asyncio.sleep pour ne pas attendre 60s
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            try:
                await engine._heartbeat_loop()
            except asyncio.CancelledError:
                pass

        engine.full_reconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_heartbeat_sends_telegram_alert_on_silence(self):
        """Si aucune candle depuis >5 min, notify_anomaly est appelé."""
        notifier = MagicMock()
        notifier.notify_anomaly = AsyncMock()
        engine = _make_engine(notifier=notifier)
        engine._heartbeat_interval = 300
        engine._last_candle_received = time.time() - 310

        engine.full_reconnect = AsyncMock()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            try:
                await engine._heartbeat_loop()
            except asyncio.CancelledError:
                pass

        notifier.notify_anomaly.assert_awaited_once()
        call_args = notifier.notify_anomaly.call_args
        from backend.alerts.notifier import AnomalyType
        assert call_args[0][0] == AnomalyType.DATA_STALE

    @pytest.mark.asyncio
    async def test_heartbeat_no_reconnect_when_candles_fresh(self):
        """Si candle reçue récemment, full_reconnect n'est PAS appelé."""
        engine = _make_engine()
        engine._heartbeat_interval = 300
        engine._last_candle_received = time.time() - 10  # 10s seulement

        engine.full_reconnect = AsyncMock()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            try:
                await engine._heartbeat_loop()
            except asyncio.CancelledError:
                pass

        engine.full_reconnect.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_heartbeat_reconnect_failure_doesnt_crash_loop(self):
        """Si full_reconnect échoue, la boucle heartbeat continue (pas de crash)."""
        engine = _make_engine()
        engine._heartbeat_interval = 300
        engine._last_candle_received = time.time() - 310

        engine.full_reconnect = AsyncMock(side_effect=RuntimeError("exchange down"))

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            try:
                await engine._heartbeat_loop()
            except asyncio.CancelledError:
                pass

        # La boucle ne doit pas avoir propagé l'erreur RuntimeError
        engine.full_reconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_heartbeat_resets_timer_after_reconnect(self):
        """Après full_reconnect réussi, _last_candle_received est reset."""
        engine = _make_engine()
        engine._heartbeat_interval = 300
        engine._last_candle_received = time.time() - 310

        engine.full_reconnect = AsyncMock()

        before = time.time()
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            try:
                await engine._heartbeat_loop()
            except asyncio.CancelledError:
                pass
        after = time.time()

        assert engine._last_candle_received >= before
        assert engine._last_candle_received <= after + 1

    @pytest.mark.asyncio
    async def test_heartbeat_no_telegram_when_notifier_none(self):
        """Sans notifier, le heartbeat ne plante pas sur absence de Telegram."""
        engine = _make_engine(notifier=None)
        engine._heartbeat_interval = 300
        engine._last_candle_received = time.time() - 310
        engine.full_reconnect = AsyncMock()

        # Ne doit pas lever d'AttributeError sur None._notifier.notify_anomaly
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            try:
                await engine._heartbeat_loop()
            except asyncio.CancelledError:
                pass

        engine.full_reconnect.assert_awaited_once()

    def test_notifier_stored_in_init(self):
        """Le notifier passé à __init__ est bien stocké."""
        notifier = MagicMock()
        engine = _make_engine(notifier=notifier)
        assert engine._notifier is notifier

    def test_no_notifier_by_default(self):
        """Sans notifier, _notifier est None."""
        engine = _make_engine()
        assert engine._notifier is None

    def test_last_candle_received_initialized(self):
        """_last_candle_received est initialisé au démarrage."""
        before = time.time()
        engine = _make_engine()
        after = time.time()
        assert before <= engine._last_candle_received <= after + 1
