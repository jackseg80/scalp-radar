"""Tests du modèle paper chronologique (signal close → ordre → fill futur)."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from backend.backtesting.simulator import (
    GridStrategyRunner,
    PendingGridOrder,
    PlannedGridExit,
    Simulator,
)
from backend.core.grid_position_manager import GridPositionManager
from backend.core.incremental_indicators import IncrementalIndicatorEngine
from backend.core.models import Candle, Direction, ExecutionSpec, OrderStatus, TimeFrame
from backend.core.position_manager import PositionManagerConfig
from backend.core.state_manager import StateManager
from backend.strategies.base_grid import GridLevel, GridPosition


BASE_TS = datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)


def _candle(
    timestamp: datetime,
    *,
    close: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
) -> Candle:
    return Candle(
        timestamp=timestamp,
        open=100.0,
        high=high,
        low=low,
        close=close,
        volume=1000.0,
        symbol="BTC/USDT",
        timeframe=TimeFrame.H1,
    )


def _runner() -> GridStrategyRunner:
    strategy = MagicMock()
    strategy.name = "grid_multi_tf"
    strategy.min_candles = {"1h": 50}
    strategy.max_positions = 2
    strategy._config.timeframe = "1h"
    strategy._config.ma_period = 3
    strategy._config.leverage = 5
    strategy._config.num_levels = 2
    strategy._config.per_asset = {"BTC/USDT": {}}
    strategy._config.min_grid_spacing_pct = 0.0
    strategy._config.min_atr_pct = 0.0
    strategy._config.min_profit_pct = 0.0
    strategy._config.cooldown_candles = 0
    strategy.compute_grid.return_value = [
        GridLevel(
            index=0,
            entry_price=95.0,
            direction=Direction.LONG,
            size_fraction=0.5,
        ),
    ]
    strategy.compute_live_indicators.return_value = {}
    strategy.should_close_all.return_value = None
    strategy.get_tp_price.return_value = float("nan")
    strategy.get_sl_price.return_value = float("nan")

    config = MagicMock()
    config.assets = [MagicMock()]
    config.risk.initial_capital = 10_000.0
    config.risk.max_margin_ratio = 0.70
    config.risk.regime_filter_enabled = False

    indicator_engine = MagicMock(spec=IncrementalIndicatorEngine)
    indicator_engine.get_indicators.return_value = {
        "1h": {
            "sma": 100.0,
            "close": 100.0,
            "atr": 2.0,
        },
    }

    gpm = GridPositionManager(PositionManagerConfig(
        leverage=5,
        maker_fee=0.0002,
        taker_fee=0.0006,
        slippage_pct=0.0005,
        high_vol_slippage_mult=2.0,
        max_risk_per_trade=0.02,
    ))
    data_engine = MagicMock()
    data_engine.get_funding_rate.return_value = None

    runner = GridStrategyRunner(
        strategy=strategy,
        config=config,
        indicator_engine=indicator_engine,
        grid_position_manager=gpm,
        data_engine=data_engine,
        chronological_execution=True,
    )
    runner._is_warming_up = False
    runner._close_buffer["BTC/USDT"] = deque(
        [100.0, 100.0, 100.0],
        maxlen=50,
    )
    return runner


@pytest.mark.asyncio
async def test_signal_candle_only_creates_order_and_next_candle_can_fill():
    runner = _runner()

    # Le niveau 95 est touché, mais il n'existait pas avant cette clôture.
    await runner.on_candle(
        "BTC/USDT", "1h",
        _candle(BASE_TS, low=94.0),
    )

    assert runner._positions.get("BTC/USDT", []) == []
    assert len(runner._pending_grid_orders["BTC/USDT"]) == 1
    assert runner._pending_grid_orders["BTC/USDT"][0].created_at == (
        BASE_TS + timedelta(hours=1)
    )

    # Sur la bougie suivante, l'ordre préexiste réellement et peut être rempli.
    await runner.on_candle(
        "BTC/USDT", "1h",
        _candle(BASE_TS + timedelta(hours=1), low=94.0),
    )

    positions = runner._positions["BTC/USDT"]
    assert len(positions) == 1
    assert positions[0].entry_time == BASE_TS + timedelta(hours=1)


@pytest.mark.asyncio
async def test_duplicate_and_out_of_order_candles_never_execute_orders():
    runner = _runner()
    await runner.on_candle("BTC/USDT", "1h", _candle(BASE_TS, low=99.0))
    assert len(runner._pending_grid_orders["BTC/USDT"]) == 1

    # Même timestamp et timestamp plus ancien : aucun fill, même si low=90.
    await runner.on_candle("BTC/USDT", "1h", _candle(BASE_TS, low=90.0))
    await runner.on_candle(
        "BTC/USDT", "1h",
        _candle(BASE_TS - timedelta(hours=1), low=90.0),
    )
    assert runner._positions.get("BTC/USDT", []) == []

    await runner.on_candle(
        "BTC/USDT", "1h",
        _candle(BASE_TS + timedelta(hours=1), low=90.0),
    )
    assert len(runner._positions["BTC/USDT"]) == 1


@pytest.mark.asyncio
async def test_first_live_candle_after_warmup_cannot_be_replayed():
    runner = _runner()
    runner._is_warming_up = True
    live_ts = datetime.now(tz=timezone.utc) - timedelta(minutes=1)

    await runner.on_candle("BTC/USDT", "1h", _candle(live_ts, low=99.0))
    await runner.on_candle("BTC/USDT", "1h", _candle(live_ts, low=90.0))

    assert runner._last_processed_candle["BTC/USDT"] == live_ts
    assert runner._positions.get("BTC/USDT", []) == []


@pytest.mark.asyncio
async def test_ohlc_exit_uses_threshold_known_before_candle():
    runner = _runner()
    runner._positions["BTC/USDT"] = [
        GridPosition(
            level=0,
            direction=Direction.LONG,
            entry_price=95.0,
            quantity=1.0,
            entry_time=BASE_TS - timedelta(hours=2),
            entry_fee=0.057,
        ),
    ]
    runner._planned_grid_exits["BTC/USDT"] = PlannedGridExit(
        tp_price=105.0,
        sl_price=80.0,
        created_at=BASE_TS,
    )
    # Le TP recalculé à la clôture serait 99, mais il n'était pas connu
    # pendant la bougie : il ne doit pas être appliqué rétroactivement.
    runner._strategy.get_tp_price.return_value = 99.0

    await runner.on_candle(
        "BTC/USDT", "1h",
        _candle(BASE_TS, high=101.0, low=98.0),
    )

    assert len(runner._positions["BTC/USDT"]) == 1
    assert runner._stats.total_trades == 0
    assert runner._planned_grid_exits["BTC/USDT"].tp_price == 99.0


@pytest.mark.asyncio
async def test_chronological_trade_always_has_exit_after_entry():
    runner = _runner()
    runner._strategy.get_tp_price.return_value = 105.0
    runner._strategy.get_sl_price.return_value = 80.0

    await runner.on_candle("BTC/USDT", "1h", _candle(BASE_TS, low=99.0))
    await runner.on_candle(
        "BTC/USDT", "1h",
        _candle(BASE_TS + timedelta(hours=1), low=94.0),
    )
    await runner.on_candle(
        "BTC/USDT", "1h",
        _candle(BASE_TS + timedelta(hours=2), high=106.0, low=100.0),
    )

    assert runner._stats.total_trades == 1
    trade = runner._trades[0][1]
    assert trade.exit_time > trade.entry_time


def test_realtime_update_changes_price_but_never_trades():
    runner = _runner()
    indicator_engine = MagicMock(spec=IncrementalIndicatorEngine)
    sim = Simulator(data_engine=MagicMock(), config=MagicMock())
    sim._indicator_engine = indicator_engine
    sim._runners = [runner]
    sim._running = True

    sim._update_realtime_candle(
        "BTC/USDT", "1h",
        _candle(BASE_TS, close=97.0, high=101.0, low=94.0),
    )

    assert runner._last_prices["BTC/USDT"] == 97.0
    assert runner._positions.get("BTC/USDT", []) == []
    assert runner._pending_grid_orders == {}
    indicator_engine.update.assert_called_once()


@pytest.mark.asyncio
async def test_state_roundtrip_preserves_execution_clock_and_funding(tmp_path):
    runner = _runner()
    runner._pending_grid_orders["BTC/USDT"] = [
        PendingGridOrder(
            level=GridLevel(
                index=1,
                entry_price=92.0,
                direction=Direction.LONG,
                size_fraction=0.5,
            ),
            created_at=BASE_TS,
        ),
    ]
    runner._planned_grid_exits["BTC/USDT"] = PlannedGridExit(
        tp_price=101.0,
        sl_price=80.0,
        created_at=BASE_TS,
    )
    runner._last_processed_candle["BTC/USDT"] = BASE_TS - timedelta(hours=1)
    runner._total_funding_cost = 12.34

    state_file = str(tmp_path / "simulator_state.json")
    manager = StateManager(db=MagicMock(), state_file=state_file)
    await manager.save_runner_state([runner])
    state = await manager.load_runner_state()

    restored = _runner()
    restored._is_warming_up = True
    restored.restore_state(state["runners"]["grid_multi_tf"])
    restored._end_warmup()

    order = restored._pending_grid_orders["BTC/USDT"][0]
    assert order.level.index == 1
    assert order.level.entry_price == 92.0
    assert order.created_at == BASE_TS
    assert restored._planned_grid_exits["BTC/USDT"].tp_price == 101.0
    assert restored._last_processed_candle["BTC/USDT"] == (
        BASE_TS - timedelta(hours=1)
    )
    assert restored._total_funding_cost == pytest.approx(12.34)


def test_legacy_intrabar_state_is_not_restored_into_chronological_runner():
    runner = _runner()
    runner._is_warming_up = True

    runner.restore_state({
        "capital": 31_703.0,
        "realized_pnl": 30_137.80,
        "total_trades": 2553,
    })
    runner._end_warmup()

    assert runner._capital == 10_000.0
    assert runner._realized_pnl == 0.0
    assert runner._stats.total_trades == 0


@pytest.mark.asyncio
async def test_funding_is_included_in_realized_pnl():
    runner = _runner()
    runner._data_engine.get_funding_rate.return_value = 0.01
    runner._positions["BTC/USDT"] = [
        GridPosition(
            level=0,
            direction=Direction.LONG,
            entry_price=100.0,
            quantity=10.0,
            entry_time=BASE_TS - timedelta(hours=1),
            entry_fee=0.6,
        ),
    ]
    runner._planned_grid_exits["BTC/USDT"] = PlannedGridExit(
        tp_price=150.0,
        sl_price=50.0,
        created_at=BASE_TS,
    )

    settlement = BASE_TS.replace(hour=16)
    await runner.on_candle(
        "BTC/USDT", "1h",
        _candle(settlement, high=101.0, low=99.0),
    )

    # Funding Bitget explicite = 0.01% de 1000$ = 0.10$.
    assert runner._total_funding_cost == pytest.approx(0.10)
    assert runner._realized_pnl == pytest.approx(-0.10)
    assert runner.get_status()["net_pnl"] == pytest.approx(-0.10)


@pytest.mark.asyncio
async def test_missing_funding_is_explicit_and_never_invented():
    runner = _runner()
    runner._positions["BTC/USDT"] = [
        GridPosition(
            level=0,
            direction=Direction.LONG,
            entry_price=100.0,
            quantity=10.0,
            entry_time=BASE_TS - timedelta(hours=1),
            entry_fee=0.6,
        ),
    ]
    runner._planned_grid_exits["BTC/USDT"] = PlannedGridExit(
        tp_price=150.0,
        sl_price=50.0,
        created_at=BASE_TS,
    )

    await runner.on_candle(
        "BTC/USDT", "1h",
        _candle(BASE_TS.replace(hour=16), high=101.0, low=99.0),
    )

    assert runner._total_funding_cost == 0.0
    assert runner._missing_funding_events == 1


@pytest.mark.asyncio
async def test_limit_order_persists_within_drift_and_replaces_above_threshold():
    runner = _runner()
    await runner.on_candle("BTC/USDT", "1h", _candle(BASE_TS))
    original = runner._pending_grid_orders["BTC/USDT"][0]

    runner._strategy.compute_grid.return_value = [GridLevel(
        index=0,
        entry_price=95.15,  # +0.158%: sous le seuil de 0.2%
        direction=Direction.LONG,
        size_fraction=0.5,
    )]
    await runner.on_candle(
        "BTC/USDT", "1h", _candle(BASE_TS + timedelta(hours=1), low=99.0),
    )
    retained = runner._pending_grid_orders["BTC/USDT"][0]
    assert retained.intent_id == original.intent_id
    assert retained.created_at == original.created_at

    runner._strategy.compute_grid.return_value = [GridLevel(
        index=0,
        entry_price=96.0,
        direction=Direction.LONG,
        size_fraction=0.5,
    )]
    await runner.on_candle(
        "BTC/USDT", "1h", _candle(BASE_TS + timedelta(hours=2), low=99.0),
    )
    replacement = runner._pending_grid_orders["BTC/USDT"][0]
    assert replacement.intent_id != original.intent_id
    assert any(
        event.status == OrderStatus.CANCELLED and event.reason == "price_drift"
        for event in runner._fill_events
    )


@pytest.mark.asyncio
async def test_partial_fill_keeps_remainder_and_does_not_recreate_filled_level():
    runner = _runner()
    runner._execution_spec = ExecutionSpec(
        partial_fill_probability=1.0,
        grid_order_expiry_minutes=180,
    )
    await runner.on_candle("BTC/USDT", "1h", _candle(BASE_TS))

    await runner.on_candle(
        "BTC/USDT", "1h", _candle(BASE_TS + timedelta(hours=1), low=94.0),
    )
    first_qty = runner._positions["BTC/USDT"][0].quantity
    pending = runner._pending_grid_orders["BTC/USDT"]
    assert len(pending) == 1
    assert pending[0].remaining_fraction == pytest.approx(0.5)
    assert runner._fill_events[-1].status == OrderStatus.PARTIALLY_FILLED

    await runner.on_candle(
        "BTC/USDT", "1h", _candle(BASE_TS + timedelta(hours=2), low=94.0),
    )
    assert len(runner._positions["BTC/USDT"]) == 1
    assert runner._positions["BTC/USDT"][0].quantity > first_qty
    assert runner._pending_grid_orders["BTC/USDT"][0].remaining_fraction == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_latency_does_not_drop_unactivated_order():
    runner = _runner()
    runner._execution_spec = ExecutionSpec(latency_ms=90 * 60 * 1000)
    await runner.on_candle("BTC/USDT", "1h", _candle(BASE_TS))
    intent_id = runner._pending_grid_orders["BTC/USDT"][0].intent_id

    await runner.on_candle(
        "BTC/USDT", "1h", _candle(BASE_TS + timedelta(hours=1), low=94.0),
    )
    assert runner._positions.get("BTC/USDT", []) == []
    assert runner._pending_grid_orders["BTC/USDT"][0].intent_id == intent_id
