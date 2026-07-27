from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from backend.backtesting.external_oos import (
    build_external_window_plans,
    clip_external_window_plans,
    combine_external_results,
    require_complete_universe_wfo_rows,
)
from backend.core.models import UniverseSelectionSpec
from backend.backtesting.engine import BacktestConfig, BacktestResult
from backend.backtesting.engine_parity import compare_engine_results
from backend.backtesting.engine_parity import measure_wfo_row_parity
from backend.backtesting.portfolio_engine import PortfolioResult, PortfolioSnapshot
from scripts.external_oos_portfolio import _load_external_oos_config


def test_external_oos_explicit_config_ignores_local_env(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_get_config(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr("scripts.external_oos_portfolio.get_config", fake_get_config)

    _load_external_oos_config(str(tmp_path))

    assert captured["args"] == (tmp_path,)
    assert captured["kwargs"] == {"env_file": None, "force_reload": True}


def _window(start: datetime, *, is_sharpe: float, is_trades: int, oos_return: float) -> dict:
    return {
        "oos_start": start.isoformat(),
        "oos_end": (start + timedelta(days=30)).isoformat(),
        "best_params": {"atr_period": 14},
        "is_sharpe": is_sharpe,
        "is_trades": is_trades,
        "oos_net_return_pct": oos_return,
    }


def _selection(
    symbols: list[str],
    *,
    calendar_start: datetime,
    top_n: int = 8,
) -> UniverseSelectionSpec:
    return UniverseSelectionSpec(
        universe_symbols=symbols,
        calendar_start=calendar_start,
        is_window_days=10,
        embargo_days=0,
        oos_window_days=10,
        step_days=10,
        top_n=top_n,
        min_is_trades=10,
    )


def _universal_window(
    start: datetime,
    *,
    is_sharpe: float,
    is_trades: int,
    is_return: float,
    oos_return: float = 0.0,
) -> dict:
    return {
        "oos_start": start.isoformat(),
        "oos_end": (start + timedelta(days=10)).isoformat(),
        "best_params": {"atr_period": 14},
        "is_sharpe": is_sharpe,
        "is_trades": is_trades,
        "is_net_return_pct": is_return,
        # Poisoned deliberately: universal selection must never read it.
        "oos_net_return_pct": oos_return,
    }


def test_universe_selection_uses_is_only_not_oos_result():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = [
        {"asset": "AAA/USDT", "wfo_windows": json.dumps({"windows": [
            _window(start, is_sharpe=1.0, is_trades=20, oos_return=-99.0),
        ]})},
        {"asset": "BBB/USDT", "wfo_windows": json.dumps({"windows": [
            _window(start, is_sharpe=-0.1, is_trades=20, oos_return=999.0),
        ]})},
    ]
    plans = build_external_window_plans(rows)
    assert list(plans[0].params_by_asset) == ["AAA/USDT"]
    assert set(plans[0].is_diagnostics) == {"AAA/USDT", "BBB/USDT"}


def test_universal_top_n_is_deterministic_and_uses_only_is_metrics():
    calendar_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    symbols = [f"A{index:02}/USDT" for index in range(9)]
    selection = _selection(symbols, calendar_start=calendar_start, top_n=8)
    oos_start = calendar_start + timedelta(days=10)
    rows = [
        {
            "asset": symbol,
            "wfo_windows": {"windows": [_universal_window(
                oos_start,
                is_sharpe=9 - index,
                is_trades=10 + index,
                is_return=1.0,
                oos_return=10_000.0 if index == 8 else -10_000.0,
            )]},
        }
        for index, symbol in enumerate(symbols)
    ]

    plans = build_external_window_plans(
        rows,
        selection=selection,
        cutoff=calendar_start + timedelta(days=20),
    )

    assert len(plans) == 1
    assert list(plans[0].params_by_asset) == sorted(symbols[:-1])
    rejected = plans[0].is_diagnostics["A08/USDT"]
    assert rejected["selection"] == "top_n_reject"
    assert rejected["is_rank"] == 9


def test_universal_calendar_is_global_and_late_asset_joins_without_overlap():
    calendar_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    selection = _selection(["AAA/USDT", "BBB/USDT"], calendar_start=calendar_start)
    first_oos = calendar_start + timedelta(days=10)
    second_oos = first_oos + timedelta(days=10)
    rows = [
        {"asset": "AAA/USDT", "wfo_windows": {"windows": [
            _universal_window(first_oos, is_sharpe=2, is_trades=20, is_return=1),
            _universal_window(second_oos, is_sharpe=2, is_trades=20, is_return=1),
        ]}},
        {"asset": "BBB/USDT", "wfo_windows": {"windows": [
            _universal_window(second_oos, is_sharpe=3, is_trades=20, is_return=1),
        ]}},
    ]

    plans = build_external_window_plans(
        rows,
        selection=selection,
        cutoff=calendar_start + timedelta(days=30),
    )

    assert [(plan.start, plan.end) for plan in plans] == [
        (first_oos, second_oos),
        (second_oos, second_oos + timedelta(days=10)),
    ]
    assert list(plans[0].params_by_asset) == ["AAA/USDT"]
    assert list(plans[1].params_by_asset) == ["AAA/USDT", "BBB/USDT"]
    assert plans[0].is_diagnostics["BBB/USDT"]["eligibility"] == "unavailable"


def test_universe_replay_refuses_partial_wfo_results():
    selection = _selection(
        ["AAA/USDT", "BBB/USDT"],
        calendar_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    with pytest.raises(ValueError, match="incomplete.*missing=BBB/USDT"):
        require_complete_universe_wfo_rows([{"asset": "AAA/USDT"}], selection)


def test_universal_snapshot_requires_all_28_wfo_rows():
    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    symbols = [f"A{index:02}/USDT" for index in range(28)]
    selection = _selection(symbols, calendar_start=start)
    rows = [{"asset": symbol, "wfo_windows": {"windows": []}} for symbol in symbols]
    require_complete_universe_wfo_rows(rows, selection)
    with pytest.raises(ValueError, match="A27/USDT"):
        require_complete_universe_wfo_rows(rows[:-1], selection)


def test_overlapping_external_windows_are_rejected():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    first = _window(start, is_sharpe=1.0, is_trades=20, oos_return=1.0)
    second = _window(start + timedelta(days=20), is_sharpe=1.0, is_trades=20, oos_return=1.0)
    with pytest.raises(ValueError, match="Overlapping"):
        build_external_window_plans([
            {"asset": "AAA/USDT", "wfo_windows": {"windows": [first, second]}},
        ])


def test_fresh_capital_clips_existing_plans_without_changing_selection():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    plans = build_external_window_plans([
        {"asset": "AAA/USDT", "wfo_windows": {"windows": [
            _window(start, is_sharpe=1.0, is_trades=20, oos_return=-99.0),
            _window(start + timedelta(days=30), is_sharpe=1.0, is_trades=20, oos_return=99.0),
        ]}},
    ])

    clipped = clip_external_window_plans(
        plans,
        start=start + timedelta(days=20),
        end=start + timedelta(days=45),
    )

    assert [(plan.start, plan.end) for plan in clipped] == [
        (start + timedelta(days=20), start + timedelta(days=30)),
        (start + timedelta(days=30), start + timedelta(days=45)),
    ]
    assert all(plan.params_by_asset == {"AAA/USDT": {"atr_period": 14}} for plan in clipped)


def _result(initial: float, final: float, start: datetime) -> PortfolioResult:
    snapshots = [
        PortfolioSnapshot(
            timestamp=start,
            total_equity=initial,
            total_capital=initial,
            total_realized_pnl=0,
            total_unrealized_pnl=0,
            total_margin_used=0,
            margin_ratio=0,
            n_open_positions=0,
            n_assets_with_positions=0,
        ),
        PortfolioSnapshot(
            timestamp=start + timedelta(days=1),
            total_equity=final,
            total_capital=final,
            total_realized_pnl=final - initial,
            total_unrealized_pnl=0,
            total_margin_used=0,
            margin_ratio=0,
            n_open_positions=0,
            n_assets_with_positions=0,
        ),
    ]
    return PortfolioResult(
        initial_capital=initial, n_assets=1, period_days=1, assets=["AAA/USDT"],
        final_equity=final, total_return_pct=(final / initial - 1) * 100,
        total_trades=0, win_rate=0, realized_pnl=final - initial, force_closed_pnl=0,
        max_drawdown_pct=0, max_drawdown_date=None, max_drawdown_duration_hours=0,
        peak_margin_ratio=0, peak_open_positions=0, peak_concurrent_assets=0,
        kill_switch_triggers=0, kill_switch_events=[], snapshots=snapshots,
        per_asset_results={}, execution_scenario="nominal",
        execution_spec={"scenario": "nominal"},
    )


def test_external_windows_carry_capital_and_recompute_total_return():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    combined = combine_external_results(
        [_result(1000, 1100, start), _result(1100, 1210, start + timedelta(days=2))],
        initial_capital=1000,
        kill_switch_pct=45,
        kill_switch_window_hours=24,
    )
    assert combined.final_equity == 1210
    assert combined.total_return_pct == pytest.approx(21.0)
    assert len(combined.snapshots) == 4


def test_engine_parity_requires_matching_return_even_without_trades():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    fast = BacktestResult(
        config=BacktestConfig(
            symbol="AAA/USDT", start_date=start, end_date=start + timedelta(days=1),
            initial_capital=1000,
        ),
        strategy_name="grid_atr", strategy_params={}, trades=[],
        equity_curve=[], equity_timestamps=[], final_capital=1010,
    )
    canonical = _result(1000, 1010, start)
    parity = compare_engine_results(fast, canonical)
    assert parity["within_tolerance"] is True

    canonical.total_return_pct = 2.0
    parity = compare_engine_results(fast, canonical)
    assert parity["within_tolerance"] is False
    assert parity["cumulative_return_delta_pct"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_wfo_parity_checks_every_is_selected_window(monkeypatch):
    seen: list[str] = []

    async def fake_window_parity(*, window, **_kwargs):
        seen.append(window["oos_start"])
        return {
            "within_tolerance": window["oos_start"].endswith("00:00:00+00:00"),
            "cumulative_return_delta_pct": 0.2,
        }

    monkeypatch.setattr(
        "backend.backtesting.engine_parity._measure_wfo_window_parity",
        fake_window_parity,
    )
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    row = {
        "asset": "AAA/USDT",
        "wfo_windows": json.dumps({"windows": [
            _window(start, is_sharpe=1.0, is_trades=20, oos_return=1.0),
            _window(start + timedelta(days=30), is_sharpe=1.0, is_trades=20, oos_return=1.0),
        ]}),
    }

    result = await measure_wfo_row_parity(
        row=row, config=MagicMock(), db_path="unused.db", exchange="binance", seed=0,
    )

    assert seen == [
        start.isoformat(), (start + timedelta(days=30)).isoformat(),
    ]
    assert result["window_count"] == 2
    assert result["within_tolerance"] is True
