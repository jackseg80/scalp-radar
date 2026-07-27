"""Canonical 1h signal / 1m broker chronology regressions."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.backtesting.portfolio_engine import PortfolioBacktester
from backend.core.models import (
    Candle,
    Direction,
    ExecutionSpec,
    OrderStatus,
    TimeFrame,
)
from backend.strategies.base_grid import GridLevel
from tests.test_paper_execution_realism import BASE_TS, _candle, _runner


def _minute(
    timestamp: datetime,
    *,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.0,
) -> Candle:
    return Candle(
        timestamp=timestamp,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100.0,
        symbol="BTC/USDT",
        timeframe=TimeFrame.M1,
        exchange="bitget",
    )


@pytest.mark.asyncio
async def test_signal_close_precedes_exact_boundary_minute_without_lookahead():
    runner = _runner()
    runner._intrabar_execution = True

    await runner.on_execution_candle(
        "BTC/USDT", "1m",
        _minute(BASE_TS + timedelta(minutes=59), low=90.0),
    )
    await runner.on_candle("BTC/USDT", "1h", _candle(BASE_TS, low=90.0))

    assert runner._positions.get("BTC/USDT", []) == []
    assert runner._pending_grid_orders["BTC/USDT"][0].created_at == (
        BASE_TS + timedelta(hours=1)
    )

    await runner.on_execution_candle(
        "BTC/USDT", "1m",
        _minute(BASE_TS + timedelta(hours=1), low=94.0),
    )
    assert runner._positions["BTC/USDT"][0].entry_time == (
        BASE_TS + timedelta(hours=1)
    )


@pytest.mark.asyncio
async def test_new_fill_cannot_use_earlier_extreme_and_gets_immediate_protection():
    runner = _runner()
    runner._intrabar_execution = True
    runner._strategy.get_tp_price.return_value = 110.0
    runner._strategy.get_sl_price.return_value = 90.0

    await runner.on_candle("BTC/USDT", "1h", _candle(BASE_TS))
    # The minute both fills the limit and trades below the future SL. The
    # intraminute order is unknowable, so the old low cannot close the new fill.
    await runner.on_execution_candle(
        "BTC/USDT", "1m",
        _minute(BASE_TS + timedelta(hours=1), low=80.0, close=94.0),
    )
    assert len(runner._positions["BTC/USDT"]) == 1
    assert runner._planned_grid_exits["BTC/USDT"].sl_price == 90.0

    await runner.on_execution_candle(
        "BTC/USDT", "1m",
        _minute(
            BASE_TS + timedelta(hours=1, minutes=1),
            open_=94.0,
            low=89.0,
            close=90.0,
        ),
    )
    assert runner._positions["BTC/USDT"] == []
    assert runner._trades[-1][1].exit_reason == "sl_global"


@pytest.mark.asyncio
async def test_intrabar_partial_fill_keeps_original_intent_and_remainder():
    runner = _runner()
    runner._intrabar_execution = True
    runner._execution_spec = ExecutionSpec(
        partial_fill_probability=1.0,
        grid_order_expiry_minutes=180,
    )
    await runner.on_candle("BTC/USDT", "1h", _candle(BASE_TS))
    intent_id = runner._pending_grid_orders["BTC/USDT"][0].intent_id

    await runner.on_execution_candle(
        "BTC/USDT", "1m",
        _minute(BASE_TS + timedelta(hours=1), low=94.0),
    )
    pending = runner._pending_grid_orders["BTC/USDT"][0]
    assert pending.intent_id == intent_id
    assert pending.remaining_fraction == pytest.approx(0.5)
    assert runner._fill_events[-1].status == OrderStatus.PARTIALLY_FILLED


@pytest.mark.asyncio
async def test_intrabar_expiry_and_direction_replacement_are_audited():
    runner = _runner()
    runner._intrabar_execution = True
    runner._execution_spec = ExecutionSpec(grid_order_expiry_minutes=1)
    await runner.on_candle("BTC/USDT", "1h", _candle(BASE_TS))
    first_intent = runner._pending_grid_orders["BTC/USDT"][0].intent_id
    await runner.on_execution_candle(
        "BTC/USDT", "1m",
        _minute(BASE_TS + timedelta(hours=1, minutes=1), low=90.0),
    )
    assert runner._positions.get("BTC/USDT", []) == []
    assert any(
        event.order_intent_id == first_intent
        and event.status == OrderStatus.EXPIRED
        for event in runner._fill_events
    )

    runner._execution_spec = ExecutionSpec(grid_order_expiry_minutes=180)
    await runner.on_candle(
        "BTC/USDT", "1h", _candle(BASE_TS + timedelta(hours=1)),
    )
    long_intent = runner._pending_grid_orders["BTC/USDT"][0].intent_id
    runner._strategy.compute_grid.return_value = [GridLevel(
        index=0,
        entry_price=105.0,
        direction=Direction.SHORT,
        size_fraction=0.5,
    )]
    await runner.on_candle(
        "BTC/USDT", "1h", _candle(BASE_TS + timedelta(hours=2)),
    )
    assert any(
        event.order_intent_id == long_intent
        and event.status == OrderStatus.CANCELLED
        and event.reason == "direction_changed"
        for event in runner._fill_events
    )


@pytest.mark.asyncio
async def test_duplicate_intrabar_candle_fails_closed():
    runner = _runner()
    runner._intrabar_execution = True
    candle = _minute(BASE_TS)
    await runner.on_execution_candle("BTC/USDT", "1m", candle)
    with pytest.raises(ValueError, match="Non-monotonic"):
        await runner.on_execution_candle("BTC/USDT", "1m", candle)


@pytest.mark.asyncio
async def test_signal_flip_becomes_market_exit_on_next_minute():
    runner = _runner()
    runner._intrabar_execution = True
    runner._strategy.get_tp_price.return_value = 150.0
    runner._strategy.get_sl_price.return_value = 50.0

    await runner.on_candle("BTC/USDT", "1h", _candle(BASE_TS))
    await runner.on_execution_candle(
        "BTC/USDT", "1m",
        _minute(BASE_TS + timedelta(hours=1), low=94.0),
    )
    runner._strategy.should_close_all.return_value = "direction_flip"
    await runner.on_candle(
        "BTC/USDT", "1h", _candle(BASE_TS + timedelta(hours=1)),
    )
    assert runner._positions["BTC/USDT"]
    assert runner._pending_grid_exits["BTC/USDT"].created_at == (
        BASE_TS + timedelta(hours=2)
    )

    await runner.on_execution_candle(
        "BTC/USDT", "1m",
        _minute(
            BASE_TS + timedelta(hours=2),
            open_=97.0,
            high=98.0,
            low=96.0,
            close=97.0,
        ),
    )
    assert runner._positions["BTC/USDT"] == []
    assert runner._trades[-1][1].exit_reason == "direction_flip"
    assert runner._trades[-1][1].exit_price == 97.0
    assert any(
        event.status == OrderStatus.FILLED
        and event.reason == "direction_flip"
        for event in runner._fill_events
    )


@pytest.mark.asyncio
async def test_existing_server_stop_precedes_new_market_exit_at_same_boundary():
    runner = _runner()
    runner._intrabar_execution = True
    runner._strategy.get_tp_price.return_value = 150.0
    runner._strategy.get_sl_price.return_value = 90.0

    await runner.on_candle("BTC/USDT", "1h", _candle(BASE_TS))
    await runner.on_execution_candle(
        "BTC/USDT", "1m",
        _minute(BASE_TS + timedelta(hours=1), low=94.0),
    )
    runner._strategy.should_close_all.return_value = "direction_flip"
    await runner.on_candle(
        "BTC/USDT", "1h", _candle(BASE_TS + timedelta(hours=1)),
    )

    await runner.on_execution_candle(
        "BTC/USDT", "1m",
        _minute(
            BASE_TS + timedelta(hours=2),
            open_=85.0,
            high=86.0,
            low=84.0,
            close=85.0,
        ),
    )
    assert runner._trades[-1][1].exit_reason == "sl_global"


@pytest.mark.asyncio
async def test_portfolio_clock_consumes_minute_only_after_signal_close():
    runner = _runner()
    backtester = PortfolioBacktester.__new__(PortfolioBacktester)
    backtester._initial_capital = 10_000.0
    backtester._regime_signal = None
    backtester._execution_spec = ExecutionSpec()
    backtester._execution_candles_processed = 0
    backtester._execution_timeframe_used = "1h"

    indicator_engine = MagicMock()
    signal = _candle(BASE_TS, low=90.0)
    minutes = [
        _minute(BASE_TS + timedelta(minutes=59), low=90.0),
        _minute(BASE_TS + timedelta(hours=1), low=94.0),
    ]
    await backtester._simulate(
        {"grid_multi_tf:BTC/USDT": runner},
        indicator_engine,
        [signal],
        {"BTC/USDT": 0},
        execution_candles_by_symbol={"BTC/USDT": minutes},
    )

    assert runner._positions["BTC/USDT"][0].entry_time == (
        BASE_TS + timedelta(hours=1)
    )
    assert backtester._execution_timeframe_used == "1m"
    assert backtester._execution_candles_processed == 2


@pytest.mark.asyncio
async def test_runtime_rejects_missing_or_multi_bar_gap_execution_data():
    backtester = PortfolioBacktester.__new__(PortfolioBacktester)
    backtester._assets = ["BTC/USDT"]
    backtester._execution_spec = ExecutionSpec()
    db = MagicMock()
    db.get_candles = AsyncMock(return_value=[])
    start = BASE_TS
    end = BASE_TS + timedelta(minutes=2)

    with pytest.raises(ValueError, match="missing bitget 1m"):
        await backtester._load_execution_candles(db, start, end)

    db.get_candles = AsyncMock(return_value=[
        _minute(start),
        _minute(start + timedelta(minutes=3)),
    ])
    with pytest.raises(ValueError, match="max 1m gap 2 bars"):
        await backtester._load_execution_candles(
            db, start, start + timedelta(minutes=3),
        )
