from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backend.core.database import Database
from backend.core.models import Candle, TimeFrame
from scripts.fetch_history import (
    fetch_bitget_uta_history_batch,
    fetch_bitget_uta_history_page,
    find_missing_candle_ranges,
)
from scripts.fetch_funding import fetch_bitget_uta_funding_history


@pytest.mark.asyncio
async def test_bitget_uta_history_page_normalizes_and_bounds_rows():
    class FakeExchange:
        def publicUtaGetV3MarketHistoryCandles(self, params):
            assert params == {
                "category": "USDT-FUTURES",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "startTime": "1000",
                "endTime": "2000",
                "limit": "100",
                "type": "market",
            }
            return {
                "code": "00000",
                "data": [
                    ["999", "1", "1", "1", "1", "1", "1"],
                    ["1000", "1", "1", "1", "1", "1", "1"],
                    ["1999", "1", "1", "1", "1", "1", "1"],
                    ["2000", "1", "1", "1", "1", "1", "1"],
                ],
            }

    rows = await fetch_bitget_uta_history_page(
        FakeExchange(), "BTC/USDT", "1m", 1000, 2000,
    )

    assert [row[0] for row in rows] == [1000, 1999]
    assert all(isinstance(value, float) for row in rows for value in row[1:])


@pytest.mark.asyncio
async def test_bitget_uta_history_batch_orders_concurrent_pages():
    class FakeExchange:
        def publicUtaGetV3MarketHistoryCandles(self, params):
            start = int(params["startTime"])
            return {
                "code": "00000",
                "data": [[str(start), "1", "2", "0.5", "1.5", "3", "4"]],
            }

    rows = await fetch_bitget_uta_history_batch(
        FakeExchange(), "BTC/USDT", "1m", 0, 18_000_000, concurrency=3,
    )

    assert [row[0] for row in rows] == [0, 6_000_000, 12_000_000]


@pytest.mark.asyncio
async def test_missing_ranges_detect_prefix_internal_gap_and_suffix(tmp_path):
    db = Database(str(tmp_path / "candles.db"))
    await db.init()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    await db.insert_candles_batch([
        Candle(
            timestamp=base + timedelta(minutes=index),
            open=1, high=1, low=1, close=1, volume=1,
            symbol="BTC/USDT", timeframe=TimeFrame.M1, exchange="bitget",
        )
        for index in (1, 2, 4)
    ])

    ranges = await find_missing_candle_ranges(
        db,
        exchange="bitget",
        symbol="BTC/USDT",
        timeframe="1m",
        start_date=base,
        end_date=base + timedelta(minutes=6),
    )
    await db.close()

    assert ranges == [
        (base, base + timedelta(minutes=1)),
        (base + timedelta(minutes=3), base + timedelta(minutes=4)),
        (base + timedelta(minutes=5), base + timedelta(minutes=6)),
    ]


@pytest.mark.asyncio
async def test_bitget_uta_funding_history_pages_to_requested_range():
    class FakeExchange:
        def __init__(self):
            self.cursors: list[str] = []

        def publicUtaGetV3MarketHistoryFundRate(self, params):
            self.cursors.append(params["cursor"])
            pages = {
                "1": [
                    {"fundingRateTimestamp": "2000", "fundingRate": "0.001"},
                    {"fundingRateTimestamp": "1500", "fundingRate": "0.002"},
                ],
                "2": [
                    {"fundingRateTimestamp": "999", "fundingRate": "0.003"},
                ],
            }
            return {"code": "00000", "data": {"resultList": pages[params["cursor"]]}}

    exchange = FakeExchange()
    rates = await fetch_bitget_uta_funding_history(
        exchange, "BTC/USDT", 1000, 2000,
    )

    assert exchange.cursors == ["1"]
    assert rates == [{"fundingRateTimestamp": "1500", "fundingRate": "0.002"}]
