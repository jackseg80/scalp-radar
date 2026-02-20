"""Tests pour le buffer d'écriture candles du DataEngine — Micro-Sprint Audit."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.core.data_engine import DataEngine
from backend.core.models import Candle, TimeFrame


# ─── Helpers ──────────────────────────────────────────────────────────────


def _make_config() -> MagicMock:
    """Crée un AppConfig minimal pour DataEngine."""
    config = MagicMock()
    config.assets = []
    config.exchange.websocket.reconnect_delay = 1
    config.exchange.websocket.max_reconnect_attempts = 3
    return config


def _make_candle(symbol: str = "BTC/USDT", ts_offset: int = 0) -> Candle:
    """Crée une candle de test avec un timestamp unique."""
    return Candle(
        timestamp=datetime(2025, 1, 15, 10, ts_offset % 60, tzinfo=timezone.utc),
        open=100_000.0 + ts_offset,
        high=100_100.0 + ts_offset,
        low=99_900.0 + ts_offset,
        close=100_050.0 + ts_offset,
        volume=42.0,
        symbol=symbol,
        timeframe=TimeFrame.M1,
    )


# ─── Tests ────────────────────────────────────────────────────────────────


class TestCandleBuffer:
    """Tests pour le buffer d'écriture candles."""

    @pytest.mark.asyncio
    async def test_candle_buffer_flushes_batch(self):
        """Vérifie que le flush écrit un batch > 1 candle."""
        db = MagicMock()
        db.insert_candles_batch = AsyncMock(return_value=3)

        engine = DataEngine(config=_make_config(), database=db)
        engine._running = True

        # Simuler 3 candles reçues (ajoutées au buffer)
        for i in range(3):
            candle = _make_candle(ts_offset=i)
            engine._write_buffer.append(candle)

        assert len(engine._write_buffer) == 3

        # Flush manuellement
        batch = engine._write_buffer.copy()
        engine._write_buffer.clear()
        await engine.db.insert_candles_batch(batch)

        # Vérifier que le batch contient 3 candles
        db.insert_candles_batch.assert_called_once()
        call_args = db.insert_candles_batch.call_args[0][0]
        assert len(call_args) == 3

    @pytest.mark.asyncio
    async def test_candle_buffer_flush_on_stop(self):
        """Vérifie que stop() flush le buffer restant avant fermeture."""
        db = MagicMock()
        db.insert_candles_batch = AsyncMock(return_value=2)

        engine = DataEngine(config=_make_config(), database=db)
        engine._running = True
        engine._exchange = AsyncMock()

        # Ajouter des candles au buffer sans flush
        for i in range(2):
            engine._write_buffer.append(_make_candle(ts_offset=i))

        assert len(engine._write_buffer) == 2

        # Arrêter le DataEngine
        await engine.stop()

        # Le buffer doit avoir été flushé
        assert len(engine._write_buffer) == 0
        db.insert_candles_batch.assert_called_once()
        batch = db.insert_candles_batch.call_args[0][0]
        assert len(batch) == 2

    @pytest.mark.asyncio
    async def test_callbacks_still_immediate(self):
        """Vérifie que les callbacks sont appelés immédiatement, pas au flush."""
        db = MagicMock()
        db.insert_candles_batch = AsyncMock(return_value=1)

        engine = DataEngine(config=_make_config(), database=db)
        engine._running = True

        # Enregistrer un callback qui enregistre les appels
        callback_calls: list[str] = []

        def on_candle(symbol, tf, candle):
            callback_calls.append(f"{symbol}:{tf}")

        engine.on_candle(on_candle)

        # Simuler la réception d'une candle via _on_candle_received
        ohlcv = [
            int(datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc).timestamp()) * 1000,
            100_000.0,  # open
            100_100.0,  # high
            99_900.0,   # low
            100_050.0,  # close
            42.0,       # volume
        ]
        await engine._on_candle_received("BTC/USDT", "1m", ohlcv)

        # Le callback doit avoir été appelé IMMÉDIATEMENT
        assert len(callback_calls) == 1
        assert callback_calls[0] == "BTC/USDT:1m"

        # Le buffer doit contenir la candle (pas encore flushé)
        assert len(engine._write_buffer) == 1

        # insert_candles_batch ne doit PAS avoir été appelé (pas de flush encore)
        db.insert_candles_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_task_runs_periodically(self):
        """Vérifie que _flush_candle_buffer flush effectivement le buffer."""
        db = MagicMock()
        db.insert_candles_batch = AsyncMock(return_value=1)

        engine = DataEngine(config=_make_config(), database=db)
        engine._running = True
        # Réduire l'intervalle pour le test
        engine._FLUSH_INTERVAL = 0.1

        # Ajouter une candle
        engine._write_buffer.append(_make_candle())

        # Lancer le flush en tâche de fond
        task = asyncio.create_task(engine._flush_candle_buffer())

        # Attendre un peu plus que l'intervalle
        await asyncio.sleep(0.3)

        # Arrêter
        engine._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Le buffer doit avoir été vidé
        assert len(engine._write_buffer) == 0
        db.insert_candles_batch.assert_called()

    # ── Tests mise à jour intra-candle ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_candle_update_in_place(self):
        """Une candle avec le même timestamp met à jour le buffer au lieu d'être rejetée."""
        db = MagicMock()
        db.insert_candles_batch = AsyncMock()
        engine = DataEngine(config=_make_config(), database=db)

        ts = datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)
        ts_ms = int(ts.timestamp()) * 1000

        # Première réception : close=50
        await engine._on_candle_received("BTC/USDT", "1m", [ts_ms, 49.0, 51.0, 48.0, 50.0, 1.0])
        assert len(engine._buffers["BTC/USDT"]["1m"]) == 1
        assert engine._buffers["BTC/USDT"]["1m"][-1].close == 50.0

        # Même timestamp, close=55 (mise à jour intra-candle)
        await engine._on_candle_received("BTC/USDT", "1m", [ts_ms, 49.0, 56.0, 48.0, 55.0, 2.0])
        assert len(engine._buffers["BTC/USDT"]["1m"]) == 1  # pas de doublon ajouté
        assert engine._buffers["BTC/USDT"]["1m"][-1].close == 55.0  # prix mis à jour

    @pytest.mark.asyncio
    async def test_candle_update_does_not_trigger_callback(self):
        """La mise à jour intra-candle ne déclenche pas on_candle callback."""
        db = MagicMock()
        db.insert_candles_batch = AsyncMock()
        engine = DataEngine(config=_make_config(), database=db)

        callback_calls: list[str] = []

        def on_candle(symbol, tf, candle):
            callback_calls.append(f"{symbol}:{tf}:{candle.close}")

        engine.on_candle(on_candle)

        ts_ms = int(datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc).timestamp()) * 1000

        # Première candle → callback appelé
        await engine._on_candle_received("BTC/USDT", "1m", [ts_ms, 49.0, 51.0, 48.0, 50.0, 1.0])
        assert len(callback_calls) == 1

        # Même timestamp → mise à jour in-place, callback PAS rappelé
        await engine._on_candle_received("BTC/USDT", "1m", [ts_ms, 49.0, 56.0, 48.0, 55.0, 2.0])
        assert len(callback_calls) == 1  # toujours 1

    @pytest.mark.asyncio
    async def test_old_duplicate_still_rejected(self):
        """Un doublon plus ancien (pas la dernière candle) est toujours rejeté."""
        db = MagicMock()
        db.insert_candles_batch = AsyncMock()
        engine = DataEngine(config=_make_config(), database=db)

        base_ts = int(datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc).timestamp()) * 1000
        min_ms = 60_000

        # 3 candles successives (high = close + 1 pour respecter la validation Pydantic)
        for i in range(3):
            close = 50.0 + i
            await engine._on_candle_received(
                "BTC/USDT", "1m",
                [base_ts + i * min_ms, 49.0, close + 1.0, 48.0, close, 1.0],
            )
        assert len(engine._buffers["BTC/USDT"]["1m"]) == 3

        # Re-envoyer la candle ts=0 (ancienne, pas la dernière) → rejetée
        await engine._on_candle_received(
            "BTC/USDT", "1m",
            [base_ts, 49.0, 100.0, 48.0, 99.0, 1.0],
        )
        assert len(engine._buffers["BTC/USDT"]["1m"]) == 3  # pas de 4e

    @pytest.mark.asyncio
    async def test_empty_buffer_no_flush(self):
        """Vérifie qu'un buffer vide ne déclenche pas d'insert."""
        db = MagicMock()
        db.insert_candles_batch = AsyncMock(return_value=0)

        engine = DataEngine(config=_make_config(), database=db)
        engine._running = True
        engine._FLUSH_INTERVAL = 0.05

        # Lancer le flush avec buffer vide
        task = asyncio.create_task(engine._flush_candle_buffer())
        await asyncio.sleep(0.15)

        engine._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Aucun appel à insert car buffer vide
        db.insert_candles_batch.assert_not_called()
