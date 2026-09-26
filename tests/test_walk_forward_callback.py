"""WFO orchestration contracts on synthetic candles, never research data."""
from datetime import datetime, timedelta, timezone
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from backend.backtesting.metrics import BacktestMetrics
from backend.core.config import AppConfig
from backend.core.models import Candle, TimeFrame
from backend.optimization.walk_forward import WalkForwardOptimizer


@pytest.fixture
def synthetic_optimizer(config_dir, monkeypatch):
    grids = {"optimization": {"is_window_days": 1, "oos_window_days": 1,
                               "step_days": 1, "max_workers": 1},
             "envelope_dca": {"default": {"ma_period": [5, 7], "num_levels": [1, 2]}}}
    (config_dir / "param_grids.yaml").write_text(yaml.safe_dump(grids), encoding="utf-8")
    optimizer = WalkForwardOptimizer(str(config_dir),
                                     config=AppConfig(config_dir, env_file=None))
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = [Candle(symbol="BTC/USDT", timeframe=TimeFrame.H1,
                      timestamp=start + timedelta(hours=i), open=100, high=101,
                      low=99, close=100, volume=10) for i in range(12 * 24)]
    db = SimpleNamespace(init=AsyncMock(), close=AsyncMock(),
                         get_candles=AsyncMock(return_value=candles),
                         get_funding_rates=AsyncMock(return_value=[]),
                         get_open_interest=AsyncMock(return_value=[]),
                         db_path=":memory:")
    monkeypatch.setattr("backend.optimization.walk_forward.Database", lambda: db)
    # Exercise selection/callback plumbing without executing backtests or pools.
    optimizer._parallel_backtest = MagicMock(side_effect=lambda grid, *a, **k: [
        (params, 1., 2., 3., 4) for params in grid
    ])
    monkeypatch.setattr("backend.backtesting.multi_engine.run_multi_backtest_single",
                        lambda *a, **k: SimpleNamespace(trades=[]))
    monkeypatch.setattr("backend.optimization.walk_forward.calculate_metrics",
                        lambda result: BacktestMetrics(total_trades=4, sharpe_ratio=1,
                                                      net_return_pct=2, profit_factor=3))
    return optimizer


@pytest.mark.asyncio
async def test_progress_callback_called(synthetic_optimizer):
    callback = MagicMock()
    result = await synthetic_optimizer.optimize("envelope_dca", "BTC/USDT",
                                                progress_callback=callback)
    assert len(result.windows) > 1
    assert callback.call_count == len(result.windows)
    values = [call.args[0] for call in callback.call_args_list]
    assert values == sorted(values)
    assert values[-1] == 80.
    assert all("WFO Fenêtre" in call.args[1] for call in callback.call_args_list)


@pytest.mark.asyncio
async def test_cancel_event_interrupts(synthetic_optimizer):
    import asyncio
    event = threading.Event()
    event.set()
    with pytest.raises(asyncio.CancelledError, match="annul"):
        await synthetic_optimizer.optimize("envelope_dca", "BTC/USDT", cancel_event=event)
    synthetic_optimizer._parallel_backtest.assert_not_called()


@pytest.mark.asyncio
async def test_params_override_merged(synthetic_optimizer):
    result = await synthetic_optimizer.optimize("envelope_dca", "BTC/USDT",
                                                params_override={"ma_period": [7], "num_levels": [2]})
    assert result.recommended_params["ma_period"] == 7
    assert result.recommended_params["num_levels"] == 2
    assert all(params["ma_period"] == 7 and params["num_levels"] == 2
               for call in synthetic_optimizer._parallel_backtest.call_args_list
               for params in call.args[0])


@pytest.mark.asyncio
async def test_funding_source_reaches_coarse_fine_and_oos(synthetic_optimizer, monkeypatch):
    # Force a distinct fine pass, so missing forwarding at any call site is seen.
    monkeypatch.setattr("backend.optimization.walk_forward._fine_grid_around_top",
                        lambda *a, **k: [{"ma_period": 7, "num_levels": 2}])
    result = await synthetic_optimizer.optimize(
        "envelope_dca", "BTC/USDT", exchange="binance", funding_exchange="bitget",
    )
    calls = synthetic_optimizer._parallel_backtest.call_args_list
    assert len(calls) == 3 * len(result.windows)
    assert all(call.kwargs["funding_exchange"] == "bitget" for call in calls)
    assert all(call.kwargs["exchange"] == "binance" for call in calls)
