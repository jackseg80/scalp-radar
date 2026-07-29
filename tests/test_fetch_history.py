from __future__ import annotations

import pytest

from scripts.fetch_history import (
    fetch_bitget_uta_history_batch,
    fetch_bitget_uta_history_page,
)


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
