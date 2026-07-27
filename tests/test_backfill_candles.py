from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.core.models import Candle, TimeFrame
from scripts.backfill_candles import _internal_gap_windows, repair_internal_gaps


def _ts(hour: int) -> datetime:
    return datetime(2025, 1, 1, hour, tzinfo=timezone.utc)


def test_internal_gap_windows_returns_only_missing_internal_range():
    assert _internal_gap_windows(
        [_ts(0), _ts(1), _ts(4), _ts(5)],
        start=_ts(0), end=_ts(5), interval_ms=3_600_000,
    ) == [(_ts(2), _ts(4))]


def test_internal_gap_windows_ignores_data_outside_requested_range():
    assert _internal_gap_windows(
        [_ts(0), _ts(3), _ts(4), _ts(5)],
        start=_ts(3), end=_ts(5), interval_ms=3_600_000,
    ) == []


@pytest.mark.asyncio
async def test_gap_repair_does_not_invent_a_missing_source_candle(monkeypatch):
    db = MagicMock()
    db.get_candles = AsyncMock(return_value=[
        Candle(
            timestamp=_ts(0), open=1, high=1, low=1, close=1, volume=1,
            symbol="BTC/USDT", timeframe=TimeFrame.H1, exchange="binance",
        ),
        Candle(
            timestamp=_ts(2), open=1, high=1, low=1, close=1, volume=1,
            symbol="BTC/USDT", timeframe=TimeFrame.H1, exchange="binance",
        ),
    ])
    db.insert_candles_batch = AsyncMock(return_value=1)

    async def returns_next_candle(*_args, **_kwargs):
        return [[int(_ts(2).timestamp() * 1000), "1", "1", "1", "1", "1"]]

    monkeypatch.setattr("scripts.backfill_candles.fetch_klines", returns_next_candle)

    repaired = await repair_internal_gaps(
        db, MagicMock(), "BTC/USDT", "1h", _ts(0), _ts(2),
    )

    assert repaired == 0
    db.insert_candles_batch.assert_not_awaited()
