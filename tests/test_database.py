"""Tests de la couche database (async avec pytest-asyncio).

Chaque test utilise une base SQLite en mémoire pour l'isolation.
"""

from datetime import datetime, timezone

import pytest
import pytest_asyncio

from backend.core.database import Database
from backend.core.models import (
    Candle,
    Direction,
    SessionState,
    Signal,
    SignalStrength,
    TimeFrame,
    Trade,
)


@pytest_asyncio.fixture
async def db():
    """Crée une database en mémoire pour chaque test."""
    database = Database(db_path=":memory:")
    await database.init()
    yield database
    await database.close()


def _make_candle(
    symbol: str = "BTC/USDT",
    tf: TimeFrame = TimeFrame.M5,
    ts_offset: int = 0,
    price: float = 97500.0,
) -> Candle:
    """Helper pour créer des candles de test."""
    ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    if ts_offset:
        from datetime import timedelta
        ts = ts + timedelta(minutes=ts_offset)
    return Candle(
        timestamp=ts,
        open=price,
        high=price + 50,
        low=price - 50,
        close=price + 10,
        volume=1000.0,
        symbol=symbol,
        timeframe=tf,
    )


@pytest.mark.asyncio
class TestCandlesCRUD:
    async def test_insert_and_get_candles(self, db):
        candle = _make_candle()
        inserted = await db.insert_candles_batch([candle])
        assert inserted == 1

        result = await db.get_candles("BTC/USDT", "5m")
        assert len(result) == 1
        assert result[0].symbol == "BTC/USDT"
        assert result[0].open == 97500.0

    async def test_insert_batch(self, db):
        candles = [_make_candle(ts_offset=i * 5) for i in range(100)]
        inserted = await db.insert_candles_batch(candles)
        assert inserted == 100

        result = await db.get_candles("BTC/USDT", "5m", limit=200)
        assert len(result) == 100

    async def test_no_duplicates(self, db):
        candle = _make_candle()
        await db.insert_candles_batch([candle])
        inserted = await db.insert_candles_batch([candle])
        assert inserted == 0

        result = await db.get_candles("BTC/USDT", "5m")
        assert len(result) == 1

    async def test_large_batch(self, db):
        candles = [_make_candle(ts_offset=i) for i in range(1000)]
        inserted = await db.insert_candles_batch(candles)
        assert inserted == 1000

    async def test_get_candles_with_time_filter(self, db):
        candles = [_make_candle(ts_offset=i * 5) for i in range(10)]
        await db.insert_candles_batch(candles)

        start = candles[3].timestamp
        end = candles[7].timestamp
        result = await db.get_candles("BTC/USDT", "5m", start=start, end=end)
        assert len(result) == 5  # indices 3,4,5,6,7

    async def test_get_latest_timestamp(self, db):
        candles = [_make_candle(ts_offset=i * 5) for i in range(5)]
        await db.insert_candles_batch(candles)

        latest = await db.get_latest_candle_timestamp("BTC/USDT", "5m")
        assert latest == candles[-1].timestamp

    async def test_get_latest_timestamp_empty(self, db):
        latest = await db.get_latest_candle_timestamp("BTC/USDT", "5m")
        assert latest is None


@pytest.mark.asyncio
class TestSessionState:
    async def test_save_and_load(self, db):
        state = SessionState(
            start_time=datetime.now(tz=timezone.utc),
            total_pnl=150.0,
            total_trades=10,
            wins=7,
            losses=3,
            max_drawdown=2.5,
            available_margin=5000.0,
            kill_switch_triggered=False,
        )
        await db.save_session_state(state)

        loaded = await db.load_session_state()
        assert loaded is not None
        assert loaded.total_pnl == 150.0
        assert loaded.wins == 7
        assert loaded.kill_switch_triggered is False

    async def test_load_empty(self, db):
        loaded = await db.load_session_state()
        assert loaded is None

    async def test_overwrite_state(self, db):
        state1 = SessionState(
            start_time=datetime.now(tz=timezone.utc),
            total_pnl=100.0,
        )
        await db.save_session_state(state1)

        state2 = SessionState(
            start_time=datetime.now(tz=timezone.utc),
            total_pnl=200.0,
        )
        await db.save_session_state(state2)

        loaded = await db.load_session_state()
        assert loaded is not None
        assert loaded.total_pnl == 200.0


@pytest.mark.asyncio
class TestSignals:
    async def test_insert_signal(self, db):
        signal = Signal(
            timestamp=datetime.now(tz=timezone.utc),
            strategy_name="vwap_rsi",
            symbol="BTC/USDT",
            direction=Direction.LONG,
            strength=SignalStrength.STRONG,
            score=0.85,
            entry_price=97500.0,
        )
        await db.insert_signal(signal)
        # Pas de get_signals pour l'instant, on vérifie juste que ça ne crash pas


@pytest.mark.asyncio
class TestTrades:
    async def test_insert_trade(self, db):
        trade = Trade(
            id="test-001",
            symbol="BTC/USDT",
            direction=Direction.LONG,
            entry_price=97500.0,
            exit_price=97987.5,
            quantity=0.01,
            leverage=20,
            gross_pnl=9.75,
            fee_cost=1.17,
            slippage_cost=0.49,
            net_pnl=8.09,
            entry_time=datetime.now(tz=timezone.utc),
            exit_time=datetime.now(tz=timezone.utc),
            strategy_name="vwap_rsi",
        )
        await db.insert_trade(trade)
