"""Tests pour le Self-Healing (auto-guérison des gaps) du DataEngine."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.core.data_engine import DataEngine, MAX_BUFFER_SIZE
from backend.core.models import TimeFrame


def _make_config() -> MagicMock:
    config = MagicMock()
    config.assets = []
    config.exchange.websocket.reconnect_delay = 1
    return config


@pytest.mark.asyncio
async def test_data_engine_auto_heal_gap():
    """Vérifie que le DataEngine détecte un gap et récupère les bougies manquantes."""
    db = MagicMock()
    db.insert_candles_batch = AsyncMock()
    
    engine = DataEngine(config=_make_config(), database=db)
    engine._exchange = AsyncMock()
    
    symbol = "BTC/USDT"
    tf_str = "1h"
    tf = TimeFrame.H1
    
    # 1. Ajouter une première bougie à 10h00
    ts1 = datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)
    ohlcv1 = [int(ts1.timestamp() * 1000), 100.0, 105.0, 95.0, 102.0, 1.0]
    await engine._on_candle_received(symbol, tf_str, ohlcv1)
    
    assert len(engine._buffers[symbol][tf_str]) == 1
    
    # 2. Simuler un gap : la prochaine bougie reçue est à 13h00 (manque 11h et 12h)
    ts3 = datetime(2025, 1, 15, 13, 0, tzinfo=timezone.utc)
    ohlcv3 = [int(ts3.timestamp() * 1000), 110.0, 115.0, 105.0, 112.0, 1.0]
    
    # Configurer le mock pour retourner les bougies de 11h et 12h
    ts11 = ts1 + timedelta(hours=1)
    ts12 = ts1 + timedelta(hours=2)
    
    mock_healed_ohlcv = [
        [int(ts11.timestamp() * 1000), 102.0, 106.0, 101.0, 104.0, 1.0],
        [int(ts12.timestamp() * 1000), 104.0, 111.0, 103.0, 110.0, 1.0],
    ]
    engine._exchange.fetch_ohlcv = AsyncMock(return_value=mock_healed_ohlcv)
    
    # 3. Recevoir la bougie de 13h00
    await engine._on_candle_received(symbol, tf_str, ohlcv3)
    
    # 4. Vérifications
    # Buffer doit contenir : 10h, 11h, 12h, 13h
    buffer = engine._buffers[symbol][tf_str]
    assert len(buffer) == 4
    assert buffer[0].timestamp == ts1
    assert buffer[1].timestamp == ts11
    assert buffer[2].timestamp == ts12
    assert buffer[3].timestamp == ts3
    
    # Vérifier que les bougies guéries sont dans le write_buffer pour la DB
    # (OHLCV3 y est aussi)
    assert len(engine._write_buffer) >= 3 
    
    # Vérifier l'appel à fetch_ohlcv
    engine._exchange.fetch_ohlcv.assert_called_once()
    args, kwargs = engine._exchange.fetch_ohlcv.call_args
    assert args[0] == symbol
    assert args[1] == tf_str
    assert kwargs["since"] == int(ts1.timestamp() * 1000) + 1
