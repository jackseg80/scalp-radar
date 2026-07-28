from __future__ import annotations

import pytest

from scripts.fetch_history import fetch_bitget_uta_history_page


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
