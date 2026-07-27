from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from backend.core.certification import (
    _preserve_forward_progress,
    compute_forward_metrics,
    evaluate_historical_gates,
    review_next_canary_stage,
    review_paper_forward,
)
from backend.core.database import Database
from backend.core.models import CertificationStatus
from scripts.record_forward_observations import validate_observation
from scripts.certify_strategy import _certification_runtime_error


def _passing_backtest() -> dict:
    return {
        "result_status": "RESEARCH_ONLY",
        "max_drawdown_pct": -12.0,
        "kill_switch_triggers": 0,
        "worst_case_sl_loss_pct": 22.0,
        "peak_margin_ratio": 0.55,
        "min_liquidation_distance_pct": 75.0,
        "missing_funding_events": 0,
        "evaluation_scope": "external_oos",
        "execution_scenario": "nominal",
        "execution_timeframe_used": "1m",
        "execution_candles_processed": 1000,
        "intrabar_max_gap_bars": 0,
        "execution_spec_json": json.dumps({
            "calibration_id": "bitget-observations-1",
            "calibration_sample_size": 100,
            "calibration_unfilled_sample_size": 10,
            "calibration_observation_hash": "abc123",
        }),
    }


def _passing_robustness() -> dict:
    return {
        "external_oos_return_pct": 20.0,
        "bootstrap_ci95_return_low": 0.10,
        "bootstrap_prob_loss": 0.02,
        "adverse_max_drawdown_pct": -25.0,
        "degraded_cost_return_pct": 4.0,
        "engine_parity_passed": 1,
        "engine_parity_max_delta_pct": 0.2,
    }


def _passing_snapshot() -> dict:
    return {
        "validation_status": "VALID",
        "metadata": {
            "intrabar_missing": [],
            "execution_timeframe": "1m",
            "max_gap_bars": 1,
        },
    }


def _passing_fresh_backtests() -> dict[int, dict]:
    return {
        180: {
            "period_days": 180,
            "total_return_pct": 5.0,
            "max_drawdown_pct": -10.0,
            "kill_switch_triggers": 0,
        },
        365: {
            "period_days": 365,
            "total_return_pct": 9.0,
            "max_drawdown_pct": -15.0,
            "kill_switch_triggers": 0,
        },
    }


def test_all_historical_gates_reach_paper_ready():
    status, details = evaluate_historical_gates(
        _passing_backtest(), _passing_robustness(), _passing_snapshot(),
        _passing_fresh_backtests(),
    )
    assert status == CertificationStatus.PAPER_READY
    assert details["missing_evidence"] == []
    assert details["failed"] == []


def test_certification_status_enum_contains_explicit_forward_states():
    assert CertificationStatus.LIVE_CANARY_READY.value == "LIVE_CANARY_READY"
    assert CertificationStatus.LIVE_APPROVED.value == "LIVE_APPROVED"


def test_long_certification_reuses_windows_python_runtime_guard():
    error = _certification_runtime_error(
        "win32", (3, 13, 13), evaluate_only=False,
    )
    assert error is not None
    assert "--python 3.12" in error
    assert _certification_runtime_error(
        "win32", (3, 13, 13), evaluate_only=True,
    ) is None
    assert _certification_runtime_error(
        "win32", (3, 12, 13), evaluate_only=False,
    ) is None


def test_historical_reevaluation_preserves_forward_progress_but_not_failures():
    assert _preserve_forward_progress(
        CertificationStatus.PAPER_READY, "LIVE_APPROVED",
    ) == CertificationStatus.LIVE_APPROVED
    assert _preserve_forward_progress(
        CertificationStatus.HISTORICAL_FAIL, "LIVE_APPROVED",
    ) == CertificationStatus.HISTORICAL_FAIL


def test_missing_evidence_stays_research_only():
    robustness = _passing_robustness()
    robustness.pop("adverse_max_drawdown_pct")
    status, details = evaluate_historical_gates(
        _passing_backtest(), robustness, _passing_snapshot(),
        _passing_fresh_backtests(),
    )
    assert status == CertificationStatus.RESEARCH_ONLY
    assert "adverse_drawdown" in details["missing_evidence"]


def test_legacy_result_can_never_be_promoted():
    backtest = _passing_backtest()
    backtest["result_status"] = "legacy"
    status, details = evaluate_historical_gates(
        backtest, _passing_robustness(), _passing_snapshot(),
        _passing_fresh_backtests(),
    )
    assert status == CertificationStatus.RESEARCH_ONLY
    assert "certifiable_result" in details["failed"]
    assert "certifiable_result" in details["capability_blockers"]


def test_intrabar_and_fast_parity_are_mandatory():
    snapshot = _passing_snapshot()
    snapshot["metadata"] = {
        "intrabar_missing": ["BTC/USDT"],
        "execution_timeframe": "1m",
    }
    robustness = _passing_robustness()
    robustness["engine_parity_passed"] = 0
    robustness["engine_parity_max_delta_pct"] = 0.8
    status, details = evaluate_historical_gates(
        _passing_backtest(), robustness, snapshot, _passing_fresh_backtests(),
    )
    assert status == CertificationStatus.RESEARCH_ONLY
    assert {"intrabar_coverage", "fast_canonical_parity"} <= set(details["failed"])


def test_having_1m_data_without_consuming_it_blocks_certification():
    backtest = _passing_backtest()
    backtest["execution_timeframe_used"] = "1h"
    status, details = evaluate_historical_gates(
        backtest, _passing_robustness(), _passing_snapshot(),
        _passing_fresh_backtests(),
    )
    assert status == CertificationStatus.RESEARCH_ONLY
    assert "intrabar_execution_used" in details["failed"]


def test_intrabar_label_without_broker_events_blocks_certification():
    backtest = _passing_backtest()
    backtest["execution_candles_processed"] = 0
    status, details = evaluate_historical_gates(
        backtest, _passing_robustness(), _passing_snapshot(),
        _passing_fresh_backtests(),
    )
    assert status == CertificationStatus.RESEARCH_ONLY
    assert "intrabar_broker_events" in details["failed"]


def test_intrabar_gap_above_snapshot_bound_blocks_certification():
    backtest = _passing_backtest()
    backtest["intrabar_max_gap_bars"] = 2
    status, details = evaluate_historical_gates(
        backtest, _passing_robustness(), _passing_snapshot(),
        _passing_fresh_backtests(),
    )
    assert status == CertificationStatus.RESEARCH_ONLY
    assert "intrabar_gap_bound" in details["failed"]


@pytest.mark.parametrize("field", ["evaluation_scope", "execution_scenario"])
def test_static_or_non_nominal_portfolio_cannot_reach_paper(field):
    backtest = _passing_backtest()
    backtest[field] = "full_history" if field == "evaluation_scope" else "adverse"
    status, details = evaluate_historical_gates(
        backtest, _passing_robustness(), _passing_snapshot(),
        _passing_fresh_backtests(),
    )
    assert status == CertificationStatus.RESEARCH_ONLY


def test_universe_verdict_requires_the_declared_primary_leverage():
    snapshot = _passing_snapshot()
    snapshot["metadata"]["universe_selection"] = {"primary_leverage": 4}
    backtest = _passing_backtest()
    backtest["leverage"] = 6
    status, details = evaluate_historical_gates(
        backtest, _passing_robustness(), snapshot, _passing_fresh_backtests(),
    )
    assert status == CertificationStatus.RESEARCH_ONLY
    assert "declared_primary_leverage" in details["failed"]


def test_missing_execution_calibration_blocks_paper():
    backtest = _passing_backtest()
    backtest["execution_spec_json"] = "{}"
    status, details = evaluate_historical_gates(
        backtest, _passing_robustness(), _passing_snapshot(),
        _passing_fresh_backtests(),
    )
    assert status == CertificationStatus.RESEARCH_ONLY
    assert "execution_calibration" in details["failed"]


def test_performance_failure_dominates_missing_operational_broker():
    backtest = _passing_backtest()
    backtest["max_drawdown_pct"] = -47.5
    backtest["execution_timeframe_used"] = "1h"
    snapshot = _passing_snapshot()
    snapshot["metadata"]["intrabar_missing"] = ["BTC/USDT"]
    status, details = evaluate_historical_gates(
        backtest,
        _passing_robustness(),
        snapshot,
        _passing_fresh_backtests(),
    )
    assert status == CertificationStatus.HISTORICAL_FAIL
    assert "nominal_drawdown" in details["performance_failed"]
    assert {
        "intrabar_coverage",
        "intrabar_execution_used",
    } <= set(details["capability_blockers"])


def test_missing_fresh_capital_evidence_blocks_paper():
    status, details = evaluate_historical_gates(
        _passing_backtest(), _passing_robustness(), _passing_snapshot(), {},
    )
    assert status == CertificationStatus.RESEARCH_ONLY
    assert "fresh_180d_coverage" in details["missing_evidence"]
    assert "fresh_365d_return" in details["missing_evidence"]


@pytest.mark.parametrize("days", [180, 365])
def test_losing_fresh_capital_window_fails_historical_gate(days):
    fresh = _passing_fresh_backtests()
    fresh[days]["total_return_pct"] = -0.01
    status, details = evaluate_historical_gates(
        _passing_backtest(), _passing_robustness(), _passing_snapshot(), fresh,
    )
    assert status == CertificationStatus.HISTORICAL_FAIL
    assert f"fresh_{days}d_return" in details["failed"]


def test_forward_ingestion_rejects_summaries_and_naive_timestamps():
    with pytest.raises(ValueError, match="observation_type"):
        validate_observation(
            {"timestamp": "2026-01-01T00:00:00+00:00", "intent_id": "i", "explained": True},
            "cert-1", "paper",
        )
    with pytest.raises(ValueError, match="timezone"):
        validate_observation(
            {
                "timestamp": "2026-01-01T00:00:00", "intent_id": "i",
                "explained": True, "observation_type": "cycle",
            },
            "cert-1", "paper",
        )


@pytest.mark.asyncio
async def test_forward_observation_is_idempotent_but_conflicts_fail(tmp_path):
    db = Database(str(tmp_path / "forward-hash.db"))
    await db.init()
    assert db._conn is not None
    now = datetime.now(tz=timezone.utc).isoformat()
    await db._conn.execute(
        """INSERT INTO strategy_certifications
           (id, strategy_name, snapshot_id, status, created_at, updated_at,
            manifest_json, manifest_hash, gates_json)
           VALUES ('cert-1', 'grid_atr', 'snap-1', 'PAPER_READY', ?, ?, '{}', 'h', '{}')""",
        (now, now),
    )
    await db._conn.commit()
    observation = {
        "certification_id": "cert-1", "phase": "paper", "timestamp": now,
        "intent_id": "entry-1", "observation_type": "entry", "explained": True,
        "expected_quantity": 1.0, "actual_quantity": 1.0,
        "fill_in_simulated_interval": True, "has_server_sl": True,
    }
    first = await db.insert_forward_observation(observation)
    assert await db.insert_forward_observation(dict(observation)) == first
    changed = dict(observation, actual_quantity=0.9)
    with pytest.raises(ValueError, match="Conflicting forward observation"):
        await db.insert_forward_observation(changed)
    rows = await db.get_forward_observations("cert-1", "paper")
    assert len(rows) == 1
    assert len(rows[0]["observation_hash"]) == 64
    await db.close()


@pytest.mark.asyncio
async def test_paper_and_four_canary_reviews_require_raw_evidence(tmp_path):
    db_path = str(tmp_path / "forward.db")
    db = Database(db_path)
    await db.init()
    assert db._conn is not None
    now = datetime.now(tz=timezone.utc)
    await db._conn.execute(
        """INSERT INTO strategy_certifications
           (id, strategy_name, snapshot_id, status, created_at, updated_at,
            manifest_json, manifest_hash, gates_json)
           VALUES ('cert-1', 'grid_atr', 'snap-1', 'PAPER_READY', ?, ?, '{}', 'h', '{}')""",
        (now.isoformat(), now.isoformat()),
    )
    await db._conn.commit()
    for index in range(100):
        await db.insert_forward_observation({
            "certification_id": "cert-1", "phase": "paper",
            "timestamp": (now + timedelta(days=index * 61 / 99)).isoformat(),
            "intent_id": f"entry-{index}", "observation_type": "entry",
            "explained": True, "expected_quantity": 1,
            "actual_quantity": 1.005, "fill_in_simulated_interval": True,
            "shadow_pnl": 1, "actual_pnl": 1.1, "has_server_sl": True,
        })
    for index in range(30):
        await db.insert_forward_observation({
            "certification_id": "cert-1", "phase": "paper",
            "timestamp": (now + timedelta(days=index * 61 / 29)).isoformat(),
            "intent_id": f"cycle-{index}", "observation_type": "cycle",
            "explained": True, "shadow_pnl": 1, "actual_pnl": 1.1,
        })
    await db.close()

    status, metrics = review_paper_forward(db_path, "cert-1")
    assert status == CertificationStatus.LIVE_CANARY_READY
    assert metrics["forward_days"] >= 60

    fractions = [0.10, 0.25, 0.50, 1.0]
    for stage, fraction in enumerate(fractions, 1):
        db = Database(db_path)
        await db.init()
        for kind in ("entry", "cycle"):
            await db.insert_forward_observation({
                "certification_id": "cert-1", "phase": f"canary_{stage}",
                "timestamp": (now + timedelta(days=70 + stage)).isoformat(),
                "intent_id": f"{kind}-{stage}", "observation_type": kind,
                "explained": True, "expected_quantity": 1 if kind == "entry" else None,
                "actual_quantity": 1 if kind == "entry" else None,
                "fill_in_simulated_interval": True if kind == "entry" else None,
                "shadow_pnl": 1, "actual_pnl": 1.1,
                "has_server_sl": True if kind == "entry" else None,
                "capital_fraction": fraction,
            })
        await db.close()
        status, reviewed_stage, _ = review_next_canary_stage(db_path, "cert-1")
        assert reviewed_stage == stage
    assert status == CertificationStatus.LIVE_APPROVED


def test_empty_forward_metrics_fail_closed():
    metrics = compute_forward_metrics([])
    assert metrics["entry_orders"] == 0
    assert metrics["max_size_deviation_pct"] == float("inf")
