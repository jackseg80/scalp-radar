"""Synthetic study contracts, with no real market data or external transport."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from backend.backtesting.portfolio_engine import ResearchFundingProvider
from backend.core.certification import evaluate_historical_gates
from backend.core.database import Database
from backend.core.experiment import (
    CONFIG_FILES, create_snapshot, revalidate_snapshot, wfo_reuse_fingerprint,
)
from backend.core.funding_research import (
    FUNDING_RESEARCH_SCENARIOS, ResearchFundingSpec, require_observed_funding,
)
from backend.core.models import Candle, CertificationStatus, Direction, ExecutionSpec, UniverseSelectionSpec
from backend.optimization.walk_forward import WalkForwardOptimizer
from scripts.external_oos_portfolio import _execution_specs_for_study
from tests.test_certification import (
    _passing_backtest, _passing_fresh_backtests, _passing_robustness, _passing_snapshot,
)
from tests.test_paper_execution_realism import _candle, _runner


UTC = timezone.utc
BASE = datetime(2024, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize("scenario,first,eighth", [
    ("central", .01, .01), ("positive_stress", .03, .03),
    ("negative_stress", -.03, -.03), ("positive_shocks", .10, .01),
    ("negative_shocks", -.10, .01),
])
def test_frozen_rates_and_calendar_shocks(scenario, first, eighth):
    spec = ResearchFundingSpec(scenario=scenario)
    assert spec.rate_pct(BASE) == first
    assert spec.rate_pct(BASE + timedelta(days=7)) == eighth
    # Local Jan 8 is still Jan 7 UTC: shocks follow UTC, not the host calendar.
    local = datetime(2024, 1, 8, 1, tzinfo=timezone(timedelta(hours=2)))
    assert spec.rate_pct(local) == first
    assert spec.rate_pct(datetime(2024, 2, 1, tzinfo=UTC)) == first


def test_profile_is_frozen_and_validated():
    with pytest.raises(ValidationError):
        ResearchFundingSpec(central_rate_pct=.02)
    with pytest.raises(ValidationError):
        ResearchFundingSpec().scenario = "negative_stress"
    with pytest.raises(ValidationError):
        ResearchFundingSpec().for_scenario("best_after_oos")
    with pytest.raises(ValueError, match="timezone-aware"):
        ResearchFundingSpec().rate_pct(datetime(2024, 1, 1))


@pytest.mark.parametrize("direction", [Direction.LONG, Direction.SHORT])
@pytest.mark.parametrize("scenario", ["positive_stress", "negative_stress"])
def test_canonical_runner_funding_updates_capital_once_at_utc_boundary(direction, scenario):
    runner = _runner()
    spec = ResearchFundingSpec(scenario=scenario)
    provider = ResearchFundingProvider(spec)
    runner._data_engine = provider
    runner._execution_spec = ExecutionSpec(research_funding=spec)
    runner._positions["BTC/USDT"] = [
        SimpleNamespace(entry_price=100., quantity=50., direction=direction),
    ]
    initial = runner._capital
    due = BASE + timedelta(hours=8)
    provider.set_timestamp(due - timedelta(minutes=1))
    runner._apply_funding_if_due("BTC/USDT", _candle(due - timedelta(minutes=1)))
    assert runner._capital == initial
    provider.set_timestamp(due)
    runner._apply_funding_if_due("BTC/USDT", _candle(due))
    cost = 5000 * spec.rate_pct(due) / 100 * (1 if direction == Direction.LONG else -1)
    assert runner._capital == pytest.approx(initial - cost)
    assert runner._stats.capital == pytest.approx(initial - cost)
    assert runner._total_funding_cost == pytest.approx(cost)
    assert runner._missing_funding_events == 0
    runner._apply_funding_if_due("BTC/USDT", _candle(due))
    assert runner._capital == pytest.approx(initial - cost)
    runner._is_warming_up = True
    provider.set_timestamp(due + timedelta(hours=8))
    runner._apply_funding_if_due("BTC/USDT", _candle(due + timedelta(hours=8)))
    assert runner._capital == pytest.approx(initial - cost)


def test_fast_cache_uses_percent_to_decimal_without_reading_observed_funding(monkeypatch):
    cache = SimpleNamespace(n_candles=2)
    build = MagicMock(return_value=cache)
    monkeypatch.setattr("backend.optimization.indicator_cache.build_cache", build)
    monkeypatch.setattr("backend.optimization.fast_multi_backtest.run_multi_backtest_from_cache",
                        lambda *a: ({}, 1., 2., 3., 4))
    candles = [_candle(BASE), _candle(BASE + timedelta(hours=1))]
    WalkForwardOptimizer._run_fast(
        [{}], {"1h": candles}, "grid_boltrend",
        {"symbol": "BTC/USDT", "start_date": BASE, "end_date": BASE + timedelta(days=1)},
        "1h", db_path="must-not-open.db", symbol="BTC/USDT", exchange="bitget",
        research_funding=ResearchFundingSpec(),
    )
    assert build.call_args.kwargs["db_path"] is None
    np.testing.assert_allclose(cache.funding_rates_1h, [.0001, .0001])


def test_research_never_silently_falls_back_to_another_engine():
    optimizer = WalkForwardOptimizer.__new__(WalkForwardOptimizer)
    optimizer._run_fast = MagicMock(side_effect=ValueError("invalid research cache"))
    optimizer._run_sequential = MagicMock()
    with pytest.raises(ValueError, match="invalid research cache"):
        optimizer._parallel_backtest(
            [{}], {"1h": []}, "grid_boltrend", "BTC/USDT", {}, "1h", 1,
            "sharpe_ratio", research_funding=ResearchFundingSpec(),
        )
    optimizer._run_sequential.assert_not_called()


@pytest.mark.parametrize("where", ["snapshot", "result"])
@pytest.mark.parametrize("losing", [False, True])
def test_synthetic_evidence_never_becomes_a_certification_verdict(where, losing):
    result, snapshot = _passing_backtest(), _passing_snapshot()
    research = ResearchFundingSpec().model_dump(mode="json")
    if where == "snapshot":
        snapshot["metadata"]["execution_spec"] = {"research_funding": research}
        with pytest.raises(ValueError, match="RESEARCH_ONLY"):
            require_observed_funding(snapshot)
    else:
        spec = json.loads(result["execution_spec_json"])
        spec["research_funding"] = research
        result["execution_spec_json"] = json.dumps(spec)
    if losing:
        result["max_drawdown_pct"] = -80.
    status, details = evaluate_historical_gates(
        result, _passing_robustness(), snapshot, _passing_fresh_backtests(),
    )
    assert status == CertificationStatus.RESEARCH_ONLY
    assert "observed_funding" in details["failed"]


def test_scenario_family_is_complete_and_uses_nominal_execution():
    base = ExecutionSpec(research_funding=ResearchFundingSpec())
    family = _execution_specs_for_study(base, "nominal")
    assert [spec.research_funding.scenario for spec in family] == list(FUNDING_RESEARCH_SCENARIOS)
    assert all(spec.scenario == "nominal" for spec in family)
    assert base.research_funding.scenario == "central"
    with pytest.raises(ValueError, match="nominal"):
        _execution_specs_for_study(base, "adverse")
    real = _execution_specs_for_study(ExecutionSpec(), "adverse")
    assert len(real) == 1 and real[0].research_funding is None


@pytest.mark.asyncio
@pytest.mark.parametrize("synthetic", [False, True])
@pytest.mark.parametrize("missing_prefix", [False, True])
async def test_snapshot_keeps_candle_gates_and_never_inserts_synthetic_rates(
    tmp_path, monkeypatch, synthetic, missing_prefix,
):
    cfg = tmp_path / "config"
    cfg.mkdir()
    for name in CONFIG_FILES:
        (cfg / name).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("backend.core.experiment.git_provenance", lambda root: ("commit", [], "hash"))
    path = str(tmp_path / "research.db")
    db = Database(path)
    await db.init()
    candles = [Candle(timestamp=BASE, open=100, high=101, low=99, close=100, volume=10,
                      symbol="BTC/USDT", exchange="binance", timeframe="1h")]
    candles += [Candle(timestamp=BASE + timedelta(minutes=i), open=100, high=101, low=99,
                       close=100, volume=10, symbol="BTC/USDT", exchange="bitget", timeframe="1m")
                for i in range(int(missing_prefix), 60)]
    await db.insert_candles_batch(candles)
    await db.close()
    spec = ExecutionSpec(research_funding=ResearchFundingSpec() if synthetic else None)
    selection = UniverseSelectionSpec(strategy_name="grid_boltrend", universe_symbols=["BTC/USDT"],
                                      calendar_start=BASE, primary_leverage=5, leverage_scenarios=[3, 5, 8])
    ident, manifest = await create_snapshot(
        db_path=path, series=[("binance", "BTC/USDT", "1h"), ("bitget", "BTC/USDT", "1m")],
        cutoff=BASE + timedelta(hours=1), start=BASE, config_dir=cfg, repo_root=tmp_path,
        execution_spec=spec, universe_selection=selection,
    )
    assert (manifest["validation_status"] == "VALID") == (synthetic and not missing_prefix)
    if missing_prefix:
        assert any("after signal coverage" in e for e in manifest["validation_errors"])
    if not synthetic:
        assert any("no funding rates" in e for e in manifest["validation_errors"])
    _, errors = await revalidate_snapshot(path, ident, config_dir=cfg, repo_root=tmp_path)
    assert (not errors) == (synthetic and not missing_prefix)
    await db.init()
    assert await db.get_funding_rates("BTC/USDT", exchange="bitget") == []
    await db.close()
    real = json.loads(json.dumps(manifest))
    real["metadata"]["execution_spec"]["research_funding"] = None
    if synthetic:
        assert wfo_reuse_fingerprint(real) != wfo_reuse_fingerprint(manifest)


@pytest.mark.asyncio
async def test_external_replay_runs_all_scenarios_on_same_is_selections(monkeypatch):
    import scripts.external_oos_portfolio as cli
    selection = UniverseSelectionSpec(strategy_name="grid_boltrend", universe_symbols=["BTC/USDT"],
                                      calendar_start=BASE, primary_leverage=5, leverage_scenarios=[3, 5, 8],
                                      portfolio_initial_capital=1646.)
    manifest = {"cutoff": "2024-02-01T00:00:00Z", "metadata": {
        "universe_selection": selection.model_dump(mode="json"),
        "execution_spec": ExecutionSpec(research_funding=ResearchFundingSpec()).model_dump(mode="json"),
    }}
    monkeypatch.setattr(cli, "Database", lambda path: SimpleNamespace(init=AsyncMock(), close=AsyncMock()))
    monkeypatch.setattr(cli, "revalidate_snapshot", AsyncMock(return_value=(manifest, [])))
    monkeypatch.setattr(cli, "load_wfo_rows", lambda *a: [])
    monkeypatch.setattr(cli, "require_complete_universe_wfo_rows", lambda *a: None)
    plans = [SimpleNamespace(params_by_asset={"BTC/USDT": {"bol_window": 100}})]
    monkeypatch.setattr(cli, "build_external_window_plans", lambda *a, **k: plans)
    monkeypatch.setattr(cli, "require_snapshot_execution_series", lambda *a: None)
    monkeypatch.setattr(cli, "_load_external_oos_config", lambda *a: object())
    replay = AsyncMock(return_value=SimpleNamespace(universe_selection=[]))
    monkeypatch.setattr(cli, "run_external_oos", replay)
    save = MagicMock(return_value=1)
    monkeypatch.setattr(cli, "save_result_sync", save)
    monkeypatch.setattr(cli, "format_portfolio_report", lambda *a: "synthetic")
    args = SimpleNamespace(db="unused", snapshot="research", config_dir="unused", wfo_snapshot=None,
                           strategy="grid_boltrend", capital=None, leverage=None, all_leverages=False,
                           execution_scenario="nominal", exchange="binance", kill_switch=45.,
                           kill_switch_window=24, label=None)
    assert await cli.run(args) == 0
    assert replay.await_count == save.call_count == 15
    assert all(call.kwargs["plans"] is plans for call in replay.call_args_list)
    assert all(call.kwargs["initial_capital"] == 1646. for call in replay.call_args_list)
    assert len({call.kwargs["evaluation_scope"] for call in save.call_args_list}) == 15
    assert all(call.kwargs["result_status"] == "RESEARCH_ONLY" for call in save.call_args_list)
