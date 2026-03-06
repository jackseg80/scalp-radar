"""Tests pour l'agrégation native du DataEngine (Mode Smallest TF)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.core.data_engine import DataEngine
from backend.core.models import TimeFrame


def _make_config() -> MagicMock:
    config = MagicMock()
    # On simule un actif avec 5m et 1h
    asset = MagicMock()
    asset.symbol = "BTC/USDT"
    asset.timeframes = ["5m", "1h"]
    config.assets = [asset]
    config.exchange.websocket.reconnect_delay = 1
    return config


@pytest.mark.asyncio
async def test_data_engine_native_aggregation_5m_to_1h():
    """Vérifie que 12 bougies de 5m produisent une bougie 1h correcte."""
    db = MagicMock()
    db.insert_candles_batch = AsyncMock()
    
    engine = DataEngine(config=_make_config(), database=db)
    engine._exchange = AsyncMock()
    
    symbol = "BTC/USDT"
    
    # Simuler l'initialisation des TFs (normalement fait dans _watch_symbol)
    engine._source_tfs[symbol] = "5m"
    engine._target_tfs[symbol] = ["1h"]
    
    # Générer 12 bougies de 5m pour l'heure de 10h00
    base_ts = datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)
    
    for i in range(12):
        ts = base_ts + timedelta(minutes=i * 5)
        # Prix : monte de 100 à 112, avec des mèches
        ohlcv = [
            int(ts.timestamp() * 1000),
            100.0 + i,      # open
            105.0 + i,      # high
            95.0 + i,       # low
            101.0 + i,      # close
            10.0            # volume constant
        ]
        # On simule la réception du WS (qui ne reçoit que du 5m)
        await engine._on_candle_received(symbol, "5m", ohlcv)
    
    # Vérifications sur le buffer 1h
    h1_buffer = engine._buffers[symbol]["1h"]
    assert len(h1_buffer) == 1
    
    h1 = h1_buffer[0]
    assert h1.timeframe == TimeFrame.H1
    assert h1.timestamp == base_ts
    
    # Open de la 1ère (i=0)
    assert h1.open == 100.0
    # Close de la dernière (i=11) -> 101 + 11 = 112
    assert h1.close == 112.0
    # High max (i=11) -> 105 + 11 = 116
    assert h1.high == 116.0
    # Low min (i=0) -> 95
    assert h1.low == 95.0
    # Volume total -> 12 * 10 = 120
    assert h1.volume == 120.0
    
    # Vérifier que les callbacks ont été notifiés pour le 1h
    # (On peut ajouter un callback de test pour être sûr)
    callback_mock = MagicMock()
    engine.on_candle(callback_mock)
    
    # Envoyer une mise à jour de la dernière 5m (10:55)
    ts_last = base_ts + timedelta(minutes=55)
    ohlcv_update = [int(ts_last.timestamp() * 1000), 111.0, 120.0, 110.0, 115.0, 20.0]
    await engine._on_candle_received(symbol, "5m", ohlcv_update)
    
    # La bougie 1h doit être mise à jour
    h1 = h1_buffer[0]
    assert h1.close == 115.0  # Nouveau close
    assert h1.high == 120.0   # Nouveau high max
    assert h1.volume == 130.0 # 110 (précédent) + 20 (nouvelle bougie 10:55) - 10 (ancienne bougie 10:55)
    # Wait, ma logique de somme simple dans _aggregate_to_target_tf 
    # refait la somme de tout le buffer constitutif à chaque fois, donc c'est robuste.
    
    assert callback_mock.call_count >= 1
