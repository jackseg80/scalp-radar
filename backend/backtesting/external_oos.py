"""Nested external-OOS portfolio orchestration over the canonical engine.

This module does not implement another simulator. It turns the IS-selected
parameters already persisted by walk-forward optimization into chronological
window plans and replays every window through :class:`PortfolioBacktester`.
"""

from __future__ import annotations

import copy
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from backend.backtesting.portfolio_engine import (
    PortfolioBacktester,
    PortfolioResult,
    PortfolioSnapshot,
)
from backend.core.models import ExecutionSpec, UniverseSelectionSpec


@dataclass(frozen=True)
class ExternalWindowPlan:
    start: datetime
    end: datetime
    params_by_asset: dict[str, dict[str, Any]]
    is_diagnostics: dict[str, dict[str, Any]]
    selection_audit: dict[str, Any] = field(default_factory=dict)


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_wfo_rows(
    db_path: str,
    strategy_name: str,
    manifest_hash: str,
) -> list[dict[str, Any]]:
    """Load one newest snapshot-bound WFO row per asset."""
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
        for row in rows:
            item = dict(row)
            newest.setdefault(str(item["asset"]), item)
        return list(newest.values())
    finally:
        conn.close()


def require_complete_universe_wfo_rows(
    wfo_rows: list[dict[str, Any]],
    selection: UniverseSelectionSpec | None,
) -> None:
    """Fail closed when a declared discovery universe was only partly optimized.

    An asset may legitimately have *zero eligible windows* because it joined
    the market late.  It must nevertheless have a snapshot-bound WFO row: a
    missing row otherwise makes it impossible to tell an unavailable history
    from an interrupted optimisation.  The external OOS decision stream must
    never silently shrink the declared 28-asset universe for that reason.
    """
    if selection is None:
        return
    actual = {str(row["asset"]) for row in wfo_rows}
    expected = set(selection.universe_symbols)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected))
        raise ValueError(
            "Universe discovery WFO is incomplete (" + "; ".join(details) + ")"
        )


def build_external_window_plans(
    wfo_rows: list[dict[str, Any]],
    *,
    min_is_sharpe: float = 0.0,
    min_is_trades: int = 10,
    selection: UniverseSelectionSpec | None = None,
    cutoff: datetime | None = None,
) -> list[ExternalWindowPlan]:
    """Preselect each window's universe using IS evidence only.

    OOS returns, OOS grades and later portfolio results are intentionally not
    read here. The rule is fixed before any external window is replayed.
    """
    grouped: dict[tuple[datetime, datetime], dict[str, dict[str, Any]]] = defaultdict(dict)
    diagnostics: dict[tuple[datetime, datetime], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in wfo_rows:
        payload = row.get("wfo_windows")
        parsed = json.loads(payload) if isinstance(payload, str) else (payload or {})
        windows = parsed.get("windows", parsed if isinstance(parsed, list) else [])
        asset = str(row["asset"])
        for window in windows:
            start = _parse_datetime(window["oos_start"])
            end = _parse_datetime(window["oos_end"])
            if end <= start:
                raise ValueError(f"Invalid external window for {asset}: {start} -> {end}")
            is_sharpe = float(window.get("is_sharpe") or 0.0)
            is_trades = int(window.get("is_trades") or 0)
            is_return = float(window.get("is_net_return_pct") or 0.0)
            key = (start, end)
            diagnostics[key][asset] = {
                "is_sharpe": is_sharpe,
                "is_trades": float(is_trades),
                "is_net_return_pct": is_return,
                "availability": "available",
            }
            if is_sharpe > min_is_sharpe and is_trades >= min_is_trades:
                params = window.get("best_params")
                if not isinstance(params, dict) or not params:
                    raise ValueError(f"Missing IS-selected params for {asset} @ {start}")
                grouped[key][asset] = copy.deepcopy(params)

    if selection is None:
        plans = [
            ExternalWindowPlan(start, end, dict(sorted(params.items())), diagnostics[(start, end)])
            for (start, end), params in grouped.items()
            if params
        ]
    else:
        if cutoff is None:
            raise ValueError("universe selection requires the snapshot cutoff")
        require_complete_universe_wfo_rows(wfo_rows, selection)
        from backend.optimization.walk_forward import build_aligned_wfo_windows

        expected = build_aligned_wfo_windows(selection, cutoff)
        plans = []
        for start, end in [(item[2], item[3]) for item in expected]:
            key = (start, end)
            per_asset = diagnostics.get(key, {})
            candidates: list[tuple[str, dict[str, Any]]] = []
            for asset in selection.universe_symbols:
                diagnostic = per_asset.setdefault(asset, {"availability": "unavailable"})
                if diagnostic.get("availability") != "available":
                    diagnostic["eligibility"] = "unavailable"
                    continue
                eligible = (
                    diagnostic["is_sharpe"] > selection.min_is_sharpe
                    and diagnostic["is_net_return_pct"] > selection.min_is_net_return_pct
                    and diagnostic["is_trades"] >= selection.min_is_trades
                )
                diagnostic["eligibility"] = "eligible" if eligible else "threshold_reject"
                if eligible and asset in grouped.get(key, {}):
                    candidates.append((asset, diagnostic))

            candidates.sort(
                key=lambda item: (-item[1]["is_sharpe"], -item[1]["is_trades"], item[0]),
            )
            selected_assets = [asset for asset, _ in candidates[:selection.top_n]]
            for rank, (asset, diagnostic) in enumerate(candidates, start=1):
                diagnostic["is_rank"] = rank
                diagnostic["selection"] = "selected" if asset in selected_assets else "top_n_reject"
            selected_params = {
                asset: copy.deepcopy(grouped[key][asset]) for asset in selected_assets
            }
            if selected_params:
                plans.append(ExternalWindowPlan(
                    start,
                    end,
                    dict(sorted(selected_params.items())),
                    dict(sorted(per_asset.items())),
                    selection_audit={
                        "top_n": selection.top_n,
                        "selected_assets": selected_assets,
                        "available_assets": sorted(
                            asset for asset, value in per_asset.items()
                            if value.get("availability") == "available"
                        ),
                    },
                ))
    plans.sort(key=lambda plan: (plan.start, plan.end))
    for previous, current in zip(plans, plans[1:]):
        if current.start < previous.end:
            raise ValueError(
                f"Overlapping external windows: {previous.start}->{previous.end} and "
                f"{current.start}->{current.end}"
            )
    return plans


def clip_external_window_plans(
    plans: list[ExternalWindowPlan],
    *,
    start: datetime,
    end: datetime,
) -> list[ExternalWindowPlan]:
    """Intersect preselected external-OOS plans with a fresh-capital period.

    Parameter selection and universe selection remain untouched; only the
    replay boundaries are clipped.  This lets certification restart the same
    historical decision stream from a clean account without looking at OOS
    results or selecting a new combination post-hoc.
    """
    if end <= start:
        raise ValueError(f"Invalid fresh-capital period: {start} -> {end}")
    clipped: list[ExternalWindowPlan] = []
    for plan in plans:
        clipped_start = max(plan.start, start)
        clipped_end = min(plan.end, end)
        if clipped_end <= clipped_start:
            continue
        clipped.append(ExternalWindowPlan(
            start=clipped_start,
            end=clipped_end,
            params_by_asset=copy.deepcopy(plan.params_by_asset),
            is_diagnostics=copy.deepcopy(plan.is_diagnostics),
            selection_audit=copy.deepcopy(plan.selection_audit),
        ))
    return clipped


async def run_external_oos(
    *,
    config: Any,
    strategy_name: str,
    plans: list[ExternalWindowPlan],
    initial_capital: float,
    db_path: str,
    exchange: str,
    execution_spec: ExecutionSpec,
    kill_switch_pct: float = 45.0,
    kill_switch_window_hours: int = 24,
    warmup_hours: int = 500,
    leverage: int | None = None,
) -> PortfolioResult:
    """Replay and concatenate external windows with capital carried forward."""
    if not plans:
        raise ValueError("No eligible external WFO windows")
    capital = initial_capital
    results: list[PortfolioResult] = []
    selection_audit: list[dict[str, Any]] = []
    for plan in plans:
        window_config = copy.deepcopy(config)
        strategy_config = getattr(window_config.strategies, strategy_name, None)
        if strategy_config is None or not hasattr(strategy_config, "per_asset"):
            raise ValueError(f"Strategy {strategy_name} has no per_asset configuration")
        strategy_config.per_asset = copy.deepcopy(plan.params_by_asset)
        assets = sorted(plan.params_by_asset)
        backtester = PortfolioBacktester(
            config=window_config,
            initial_capital=capital,
            strategy_name=strategy_name,
            assets=assets,
            exchange=exchange,
            kill_switch_pct=kill_switch_pct,
            kill_switch_window_hours=kill_switch_window_hours,
            execution_spec=execution_spec,
            leverage=leverage,
        )
        result = await backtester.run(
            plan.start,
            plan.end,
            db_path=db_path,
            warmup_start=plan.start - timedelta(hours=warmup_hours),
            end_exclusive=True,
        )
        results.append(result)
        selection_audit.append({
            "start": plan.start.isoformat(),
            "end": plan.end.isoformat(),
            "selected_assets": sorted(plan.params_by_asset),
            "selection": copy.deepcopy(plan.selection_audit),
            "assets": copy.deepcopy(plan.is_diagnostics),
            "order_rejections": copy.deepcopy(result.order_rejections),
        })
        capital = result.final_equity
    return combine_external_results(
        results,
        initial_capital=initial_capital,
        kill_switch_pct=kill_switch_pct,
        kill_switch_window_hours=kill_switch_window_hours,
        universe_selection=selection_audit,
    )


def combine_external_results(
    results: list[PortfolioResult],
    *,
    initial_capital: float,
    kill_switch_pct: float,
    kill_switch_window_hours: int,
    universe_selection: list[dict[str, Any]] | None = None,
) -> PortfolioResult:
    """Combine window results while recomputing account-level risk metrics."""
    if not results:
        raise ValueError("No external results to combine")
    snapshots: list[PortfolioSnapshot] = []
    all_trades = []
    order_rejections: dict[str, int] = defaultdict(int)
    assets: set[str] = set()
    for result in results:
        snapshots.extend(result.snapshots)
        all_trades.extend(result.all_trades)
        assets.update(result.assets)
        for reason, count in result.order_rejections.items():
            order_rejections[reason] += count

    snapshots.sort(key=lambda snapshot: snapshot.timestamp)
    all_trades.sort(key=lambda item: item[1].exit_time)
    max_dd, max_dd_date, max_dd_duration = PortfolioBacktester._compute_drawdown(snapshots)
    checker = object.__new__(PortfolioBacktester)
    checker._kill_switch_pct = kill_switch_pct
    checker._kill_switch_window_hours = kill_switch_window_hours
    kill_events = checker._check_kill_switch(snapshots)

    per_asset: dict[str, dict[str, Any]] = {}
    for runner_key in sorted({key for key, _ in all_trades}):
        trades = [trade for key, trade in all_trades if key == runner_key]
        wins = sum(trade.net_pnl > 0 for trade in trades)
        pnl = sum(trade.net_pnl for trade in trades)
        per_asset[runner_key] = {
            "trades": len(trades),
            "wins": wins,
            "win_rate": wins / len(trades) * 100 if trades else 0.0,
            "net_pnl": round(pnl, 2),
        }

    final_equity = results[-1].final_equity
    total_trades = len(all_trades)
    wins = sum(trade.net_pnl > 0 for _, trade in all_trades)
    return PortfolioResult(
        initial_capital=initial_capital,
        n_assets=len(assets),
        period_days=sum(result.period_days for result in results),
        assets=sorted(assets),
        final_equity=final_equity,
        total_return_pct=round((final_equity / initial_capital - 1) * 100, 2),
        total_trades=total_trades,
        win_rate=round(wins / total_trades * 100, 1) if total_trades else 0.0,
        realized_pnl=round(sum(result.realized_pnl for result in results), 2),
        force_closed_pnl=round(sum(result.force_closed_pnl for result in results), 2),
        max_drawdown_pct=round(max_dd, 2),
        max_drawdown_date=max_dd_date,
        max_drawdown_duration_hours=round(max_dd_duration, 1),
        peak_margin_ratio=max((snapshot.margin_ratio for snapshot in snapshots), default=0.0),
        peak_open_positions=max((snapshot.n_open_positions for snapshot in snapshots), default=0),
        peak_concurrent_assets=max(
            (snapshot.n_assets_with_positions for snapshot in snapshots), default=0,
        ),
        kill_switch_triggers=len(kill_events),
        kill_switch_events=kill_events,
        snapshots=snapshots,
        per_asset_results=per_asset,
        all_trades=all_trades,
        funding_paid_total=round(sum(result.funding_paid_total for result in results), 2),
        leverage=results[0].leverage,
        was_liquidated=any(result.was_liquidated for result in results),
        liquidation_event=next(
            (result.liquidation_event for result in results if result.liquidation_event), None,
        ),
        min_liquidation_distance_pct=min(
            result.min_liquidation_distance_pct for result in results
        ),
        worst_case_sl_loss_pct=max(result.worst_case_sl_loss_pct for result in results),
        order_rejections=dict(order_rejections),
        missing_funding_events=sum(result.missing_funding_events for result in results),
        execution_scenario=results[0].execution_scenario,
        execution_spec=copy.deepcopy(results[0].execution_spec),
        execution_timeframe_used=results[0].execution_timeframe_used,
        universe_selection=universe_selection or [],
    )
