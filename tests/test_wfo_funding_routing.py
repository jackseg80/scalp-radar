"""Funding-source routing contracts, without database reads or backtests."""
from types import SimpleNamespace
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from backend.optimization.walk_forward import WalkForwardOptimizer


@pytest.mark.parametrize("funding_exchange,signal_exchange,expected", [
    ("bitget", "binance", "bitget"),
    (None, "binance", "binance"),
    (None, None, None),
])
def test_fast_cache_receives_funding_source(monkeypatch, funding_exchange, signal_exchange, expected):
    build = MagicMock(return_value=SimpleNamespace(n_candles=0))
    monkeypatch.setattr("backend.optimization.indicator_cache.build_cache", build)
    monkeypatch.setattr("backend.optimization.fast_multi_backtest.run_multi_backtest_from_cache",
                        lambda strategy, params, cache, config: (params, 1., 2., 3., 4))
    result = WalkForwardOptimizer._run_fast(
        [{"bol_window": 50}], {"1h": []}, "grid_boltrend",
        {"symbol": "BTC/USDT", "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
         "end_date": datetime(2024, 1, 2, tzinfo=timezone.utc)}, "1h",
        db_path="synthetic.db", symbol="BTC/USDT", exchange=signal_exchange,
        funding_exchange=funding_exchange,
    )
    assert len(result) == 1
    assert build.call_args.kwargs == dict(db_path="synthetic.db", symbol="BTC/USDT", exchange=expected)


def test_dispatcher_forwards_explicit_funding_without_fallback():
    optimizer = WalkForwardOptimizer.__new__(WalkForwardOptimizer)
    optimizer._run_fast = MagicMock(return_value=[({}, 1., 2., 3., 4)])
    optimizer._run_sequential = MagicMock(side_effect=AssertionError("Unexpected fallback"))
    optimizer._parallel_backtest(
        [{}], {"1h": []}, "grid_boltrend", "BTC/USDT", {}, "1h", 1, "sharpe_ratio",
        db_path="synthetic.db", exchange="binance", funding_exchange="bitget",
    )
    assert optimizer._run_fast.call_args.kwargs["funding_exchange"] == "bitget"
    assert optimizer._run_fast.call_args.kwargs["exchange"] == "binance"
    optimizer._run_sequential.assert_not_called()
