"""Tests pour le mode Fallback Polling du DataEngine."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.data_engine import DataEngine
from backend.core.models import TimeFrame


def _make_config() -> MagicMock:
    config = MagicMock()
    asset = MagicMock()
    asset.symbol = "GALA/USDT"
    asset.timeframes = ["1h"]
    config.assets = [asset]
    config.exchange.websocket.reconnect_delay = 1
    return config


@pytest.mark.asyncio
async def test_data_engine_fallback_to_polling():
    """Vérifie que le DataEngine bascule en polling après 3 échecs WS."""
    db = MagicMock()
    engine = DataEngine(config=_make_config(), database=db)
    engine._exchange = AsyncMock()
    
    symbol = "GALA/USDT"
    engine._source_tfs[symbol] = "1h"
    engine._target_tfs[symbol] = []
    
    # Simuler 3 échecs de relance WS
    engine._stale_restart_count[symbol] = 3
    
    # Simuler un état "stale"
    engine._last_update_per_symbol[symbol] = datetime.now(tz=timezone.utc).replace(year=2020)
    
    # On mocke _start_polling
    with patch.object(engine, "_start_polling", new_callable=AsyncMock) as mock_start:
        with patch("asyncio.sleep", return_value=None):
            engine._heartbeat_tick = 4
            engine._running = True
            
            # On fait en sorte que la boucle s'arrête après avoir fait son travail
            # On utilise mock_start pour arrêter la boucle
            mock_start.side_effect = lambda *args, **kwargs: setattr(engine, "_running", False)
            
            # Lancer le heartbeat
            await engine._heartbeat_loop()
            
            # Doit avoir appelé _start_polling
            mock_start.assert_called_with(symbol)


@pytest.mark.asyncio
async def test_poll_symbol_rest_injects_data():
    """Vérifie que le polling REST injecte correctement les données."""
    db = MagicMock()
    engine = DataEngine(config=_make_config(), database=db)
    engine._exchange = AsyncMock()
    
    symbol = "GALA/USDT"
    tf = "1h"
    engine._polling_modes.add(symbol)
    
    # Mock fetch_ohlcv
    ts = datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)
    mock_ohlcv = [[int(ts.timestamp() * 1000), 1.0, 1.1, 0.9, 1.05, 1000.0]]
    engine._exchange.fetch_ohlcv = AsyncMock(return_value=mock_ohlcv)
    
    # Mock _process_ohlcv_item pour vérifier l'injection
    with patch.object(engine, "_process_ohlcv_item", new_callable=AsyncMock) as mock_process:
        with patch("asyncio.sleep", return_value=None):
            engine._running = True
            # On arrête après un appel
            mock_process.side_effect = lambda *args, **kwargs: setattr(engine, "_running", False)
            
            await engine._poll_symbol_rest(symbol, tf)
            
            mock_process.assert_called_with(symbol, tf, mock_ohlcv[0], is_ws=False)
