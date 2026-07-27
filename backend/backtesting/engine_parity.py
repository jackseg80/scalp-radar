"""Measured parity between WFO search engine and canonical portfolio replay."""

from __future__ import annotations

import copy
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from backend.backtesting.engine import BacktestConfig, BacktestResult
from backend.backtesting.extra_data_builder import build_extra_data_map
from backend.backtesting.multi_engine import run_multi_backtest_single
from backend.backtesting.portfolio_engine import PortfolioBacktester, PortfolioResult
from backend.core.database import Database
from backend.core.models import ExecutionSpec
from backend.optimization import create_strategy_with_params
from backend.optimization.fast_multi_backtest import (
    run_grid_multi_tf_audit_from_cache,
)
from backend.optimization.indicator_cache import (
    build_cache,
    slice_indicator_cache,
)
from backend.backtesting.certification_capabilities import (
    canonical_certification_capability,
)


def compare_engine_results(
    fast: BacktestResult,
    canonical: PortfolioResult,
    *,
    tolerance_pct: float = 0.5,
) -> dict[str, Any]:
    """Compare P&L, trade sequence and exit reasons without hiding drift."""
    initial = float(fast.config.initial_capital)
    fast_return = (fast.final_capital / initial - 1.0) * 100 if initial else 0.0
    canonical_return = canonical.total_return_pct
    delta = abs(fast_return - canonical_return)
    fast_sequence = [
        (trade.entry_time.isoformat(), trade.exit_time.isoformat(),
         trade.direction.value, trade.exit_reason)
        for trade in fast.trades
    ]
    canonical_sequence = [
        (trade.entry_time.isoformat(), trade.exit_time.isoformat(),
         trade.direction.value, trade.exit_reason)
        for _, trade in canonical.all_trades
    ]
    trade_count_equal = len(fast.trades) == canonical.total_trades
    sequence_equal = fast_sequence == canonical_sequence
    reasons_equal = (
        [trade.exit_reason for trade in fast.trades]
        == [trade.exit_reason for _, trade in canonical.all_trades]
    )
    within = (
        delta <= tolerance_pct
        and trade_count_equal
        and sequence_equal
        and reasons_equal
    )
    return {
        "within_tolerance": within,
        "tolerance_pct": tolerance_pct,
        "cumulative_return_delta_pct": round(delta, 8),
        "fast_return_pct": round(fast_return, 8),
        "canonical_return_pct": round(canonical_return, 8),
        "fast_trades": len(fast.trades),
        "canonical_trades": canonical.total_trades,
        "trade_count_equal": trade_count_equal,
        "sequence_equal": sequence_equal,
        "exit_reasons_equal": reasons_equal,
    }


async def _measure_wfo_window_parity(
    *,
    row: dict[str, Any],
    window: dict[str, Any],
    config: Any,
    db_path: str,
    exchange: str,
    seed: int,
    warmup_hours: int = 500,
) -> dict[str, Any]:
    """Measure one IS-selected external WFO window against the canonical path."""
    start = datetime.fromisoformat(
        window["oos_start"].replace("Z", "+00:00")
    )
    end = datetime.fromisoformat(
        window["oos_end"].replace("Z", "+00:00")
    )
    params = window.get("best_params") or {}
    if params.get("timeframe", "1h") != "1h":
        return {
            "within_tolerance": False,
            "error": "canonical portfolio currently requires 1h timeframe",
        }

    strategy = create_strategy_with_params(row["strategy_name"], params)
    db = Database(db_path)
    await db.init()
    warmup_start = start - timedelta(hours=warmup_hours)
    candles_by_tf: dict[str, list] = {}
    try:
        for timeframe in sorted(set(strategy.min_candles) | {"1h"}):
            candles_by_tf[timeframe] = await db.get_candles(
                row["asset"], timeframe, start=warmup_start,
                end=end - timedelta(microseconds=1), limit=1_000_000,
                exchange=exchange,
            )
        funding = await db.get_funding_rates(
            row["asset"], exchange=exchange,
            start_ts=int(start.timestamp() * 1000),
            end_ts=int(end.timestamp() * 1000) - 1,
        )
        oi = await db.get_open_interest(
            row["asset"], timeframe="5m", exchange=exchange,
        )
    finally:
        await db.close()
    full_main = candles_by_tf.get("1h", [])
    trading_by_tf = {
        timeframe: [candle for candle in candles if start <= candle.timestamp < end]
        for timeframe, candles in candles_by_tf.items()
    }
    if len(full_main) < 60 or len(trading_by_tf.get("1h", [])) < 10:
        return {"within_tolerance": False, "error": "insufficient parity candles"}

    precomputed = strategy.compute_indicators(candles_by_tf)
    extra = build_extra_data_map(trading_by_tf["1h"], funding, oi)
    strategy_config = getattr(config.strategies, row["strategy_name"])
    leverage = int(getattr(strategy_config, "leverage", 1))
    bt_config = BacktestConfig(
        symbol=row["asset"], start_date=start, end_date=end,
        initial_capital=10_000.0, leverage=leverage,
        maker_fee=config.risk.fees.maker_percent / 100,
        taker_fee=config.risk.fees.taker_percent / 100,
        slippage_pct=config.risk.slippage.default_estimate_percent / 100,
        high_vol_slippage_mult=config.risk.slippage.high_volatility_multiplier,
        max_risk_per_trade=config.risk.position.max_risk_per_trade_percent / 100,
    )
    fast = run_multi_backtest_single(
        row["strategy_name"], params, trading_by_tf, bt_config, "1h",
        precomputed_indicators=precomputed,
        extra_data_by_timestamp=extra,
    )

    search_metrics = None
    search_trace: list[dict[str, Any]] = []
    if row["strategy_name"] == "grid_multi_tf":
        grid_values = {
            key: value if isinstance(value, list) else [value]
            for key, value in params.items()
        }
        full_cache = build_cache(
            candles_by_tf,
            grid_values,
            row["strategy_name"],
            main_tf="1h",
        )
        first_trading_index = next(
            (
                index
                for index, candle in enumerate(full_main)
                if candle.timestamp >= start
            ),
            full_cache.n_candles,
        )
        trading_cache = slice_indicator_cache(full_cache, first_trading_index)
        search_metrics, search_trace = run_grid_multi_tf_audit_from_cache(
            params,
            trading_cache,
            bt_config,
        )

    canonical_config = copy.deepcopy(config)
    getattr(canonical_config.strategies, row["strategy_name"]).per_asset = {
        row["asset"]: copy.deepcopy(params),
    }
    spec = ExecutionSpec(
        maker_fee_pct=config.risk.fees.maker_percent,
        taker_fee_pct=config.risk.fees.taker_percent,
        slippage_pct=config.risk.slippage.default_estimate_percent,
        scenario="nominal",
        random_seed=seed,
    )
    canonical = await PortfolioBacktester(
        config=canonical_config,
        initial_capital=10_000.0,
        strategy_name=row["strategy_name"],
        assets=[row["asset"]],
        exchange=exchange,
        execution_spec=spec,
    ).run(
        start, end, db_path=db_path, warmup_start=warmup_start,
        end_exclusive=True,
    )
    event_result = compare_engine_results(fast, canonical)
    result = dict(event_result)
    if search_metrics is not None:
        search_return = float(search_metrics[2])
        search_trades = int(search_metrics[4])
        search_delta = abs(search_return - canonical.total_return_pct)
        search_exits = [
            event for event in search_trace if event["event"] == "exit"
        ]
        canonical_sequence = [
            {
                "entry_timestamp_ms": int(trade.entry_time.timestamp() * 1000),
                "exit_timestamp_ms": int(trade.exit_time.timestamp() * 1000),
                "direction": 1 if trade.direction.value == "LONG" else -1,
                "reason": trade.exit_reason,
            }
            for _, trade in canonical.all_trades
        ]
        search_sequence = [
            {
                "entry_timestamp_ms": (
                    int(trading_cache.candle_timestamps[event["entry_candle_index"]])
                    if trading_cache.candle_timestamps is not None
                    and event["entry_candle_index"] >= 0 else None
                ),
                "exit_timestamp_ms": event["timestamp_ms"],
                "direction": event["direction"],
                "reason": event["reason"],
            }
            for event in search_exits
        ]
        trade_count_equal = search_trades == canonical.total_trades
        sequence_equal = search_sequence == canonical_sequence
        reasons_equal = (
            [item["reason"] for item in search_sequence]
            == [item["reason"] for item in canonical_sequence]
        )
        result.update({
            "within_tolerance": (
                search_delta <= 0.5
                and trade_count_equal
                and sequence_equal
                and reasons_equal
                and bool(event_result["within_tolerance"])
            ),
            "cumulative_return_delta_pct": round(search_delta, 8),
            "fast_return_pct": round(search_return, 8),
            "canonical_return_pct": round(canonical.total_return_pct, 8),
            "fast_trades": search_trades,
            "canonical_trades": canonical.total_trades,
            "trade_count_equal": trade_count_equal,
            "sequence_equal": sequence_equal,
            "exit_reasons_equal": reasons_equal,
            "search_engine": {
                "return_pct": round(search_return, 8),
                "trades": search_trades,
                "selected_candidate_count": sum(
                    event["event"] == "entry_candidate"
                    and bool(event.get("selected"))
                    for event in search_trace
                ),
                "trace": search_trace,
                "trade_sequence": search_sequence,
            },
            "event_engine": event_result,
            "canonical_trade_sequence": canonical_sequence,
        })
    result.update({
        "asset": row["asset"],
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
    })
    return result


async def measure_wfo_row_parity(
    *,
    row: dict[str, Any],
    config: Any,
    db_path: str,
    exchange: str,
    seed: int,
    warmup_hours: int = 500,
) -> dict[str, Any]:
    """Validate every selected WFO window, never merely the newest one.

    Each external window may use different IS-selected parameters.  Collapsing
    this check to the latest one would leave earlier decisions uncertified.
    The aggregate keeps the legacy top-level fields consumed by robustness
    while retaining the complete per-window audit.
    """
    parsed = json.loads(row["wfo_windows"])
    windows = parsed.get("windows", [])
    if not windows:
        return {"within_tolerance": False, "error": "no WFO windows", "windows": []}

    results: list[dict[str, Any]] = []
    for window in sorted(windows, key=lambda item: item["oos_start"]):
        try:
            result = await _measure_wfo_window_parity(
                row=row,
                window=window,
                config=config,
                db_path=db_path,
                exchange=exchange,
                seed=seed,
                warmup_hours=warmup_hours,
            )
        except Exception as exc:
            result = {
                "asset": row["asset"],
                "window_start": window.get("oos_start"),
                "window_end": window.get("oos_end"),
                "within_tolerance": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)

    deltas = [
        float(item["cumulative_return_delta_pct"])
        for item in results
        if item.get("cumulative_return_delta_pct") is not None
    ]
    return {
        "asset": row["asset"],
        "within_tolerance": bool(results) and all(
            bool(item.get("within_tolerance")) for item in results
        ),
        "cumulative_return_delta_pct": max(deltas, default=float("inf")),
        "window_count": len(results),
        "windows": results,
    }


async def measure_and_store_strategy_parity(
    *,
    db_path: str,
    strategy_name: str,
    manifest_hash: str,
    config: Any,
    exchange: str,
    seed: int,
) -> list[dict[str, Any]]:
    supported, reason = canonical_certification_capability(strategy_name)
    if not supported:
        raise ValueError(f"Canonical parity unavailable for {strategy_name}: {reason}")
    """Measure all WFO assets and attach evidence to their existing rows."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT * FROM optimization_results
               WHERE strategy_name=? AND manifest_hash=?
                 AND result_status!='legacy' AND wfo_windows IS NOT NULL
               ORDER BY id DESC""",
            (strategy_name, manifest_hash),
        ).fetchall()
        newest: dict[str, dict[str, Any]] = {}
        for raw in rows:
            item = dict(raw)
            newest.setdefault(item["asset"], item)
    finally:
        conn.close()

    results: list[dict[str, Any]] = []
    for row in newest.values():
        try:
            result = await measure_wfo_row_parity(
                row=row, config=config, db_path=db_path,
                exchange=exchange, seed=seed,
            )
        except Exception as exc:
            result = {
                "asset": row["asset"],
                "within_tolerance": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "UPDATE optimization_results SET engine_parity_json=? WHERE id=?",
                (json.dumps(result, sort_keys=True), row["id"]),
            )
            conn.commit()
        finally:
            conn.close()
    return results
