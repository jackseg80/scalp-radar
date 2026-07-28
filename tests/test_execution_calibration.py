from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backend.core.database import Database
from backend.core.execution_calibration import (
    calibrate_execution,
    load_execution_calibration,
)


@pytest.mark.asyncio
async def test_calibration_is_traceable_and_deterministic(tmp_path):
    db_path = str(tmp_path / "calibration.db")
    db = Database(db_path)
    await db.init()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(30):
        intent = base + timedelta(minutes=index)
        await db.insert_live_trade({
            "timestamp": (intent + timedelta(milliseconds=100 + index)).isoformat(),
            "strategy_name": "grid_atr", "symbol": "BTC/USDT:USDT",
            "direction": "LONG", "trade_type": "entry", "side": "buy",
            "quantity": 1.0 if index else 0.5, "price": 100.05,
            "order_id": f"fill-{index}", "context": "grid_limit",
            "intent_timestamp": intent.isoformat(), "intent_price": 100.0,
            "requested_quantity": 1.0,
            "filled_quantity": 1.0 if index else 0.5,
            "fill_timestamp": (intent + timedelta(milliseconds=100 + index)).isoformat(),
            "latency_ms": 100 + index, "slippage_pct": 0.05,
            "fill_ratio": 1.0 if index else 0.5,
            "order_status": "canceled" if index == 0 else "filled",
        })
    for index in range(2):
        intent = base + timedelta(hours=1, minutes=index)
        await db.insert_live_trade({
            "timestamp": (intent + timedelta(hours=2)).isoformat(),
            "strategy_name": "grid_atr", "symbol": "ETH/USDT:USDT",
            "direction": "LONG", "trade_type": "entry_unfilled", "side": "buy",
            "quantity": 0.0, "price": 200.0, "order_id": f"miss-{index}",
            "context": "grid_limit_max_age", "intent_timestamp": intent.isoformat(),
            "intent_price": 200.0, "requested_quantity": 1.0,
            "filled_quantity": 0.0,
            "fill_timestamp": (intent + timedelta(hours=2)).isoformat(),
            "latency_ms": 7_200_000, "slippage_pct": 0.0,
            "fill_ratio": 0.0, "order_status": "expired",
        })
    await db.close()

    kwargs = dict(
        db_path=db_path, maker_fee_pct=0.02, taker_fee_pct=0.06,
        default_slippage_pct=0.03, strategy_name="grid_atr", seed=42,
    )
    first_id, first = calibrate_execution(**kwargs)
    second_id, second = calibrate_execution(**kwargs)
    assert first_id == second_id
    assert first.calibration_sample_size == 30
    # The first cancelled partial has a confirmed unfilled remainder.
    assert first.calibration_unfilled_sample_size == 3
    assert first.calibration_partial_sample_size == 1
    assert first.missed_fill_probability == pytest.approx(3 / 32)
    assert first.calibration_observation_hash
    assert second == first
    assert load_execution_calibration(db_path, first_id) == first


@pytest.mark.asyncio
async def test_calibration_refuses_fill_only_sample(tmp_path):
    db_path = str(tmp_path / "incomplete.db")
    db = Database(db_path)
    await db.init()
    now = datetime.now(tz=timezone.utc)
    await db.insert_live_trade({
        "timestamp": now.isoformat(), "strategy_name": "grid_atr",
        "symbol": "BTC/USDT:USDT", "direction": "LONG", "trade_type": "entry",
        "side": "buy", "quantity": 1, "price": 100, "order_id": "only-fill",
        "intent_timestamp": now.isoformat(), "intent_price": 100,
        "requested_quantity": 1, "filled_quantity": 1,
        "fill_timestamp": now.isoformat(), "latency_ms": 10,
        "slippage_pct": 0, "fill_ratio": 1, "order_status": "filled",
    })
    await db.close()
    with pytest.raises(ValueError, match="unfilled"):
        calibrate_execution(
            db_path=db_path, maker_fee_pct=0.02, taker_fee_pct=0.06,
            default_slippage_pct=0.03,
        )
