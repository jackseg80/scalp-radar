"""Strict certification gates over existing WFO/portfolio/robustness rows."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.core.experiment import canonical_json, config_hashes, git_provenance
from backend.core.models import CertificationStatus


def _gate(name: str, value: Any, passed: bool | None, limit: str) -> dict[str, Any]:
    return {"name": name, "value": value, "passed": passed, "limit": limit}


def evaluate_historical_gates(
    backtest: dict[str, Any],
    robustness: dict[str, Any] | None,
    snapshot: dict[str, Any],
    fresh_backtests: dict[int, dict[str, Any]] | None = None,
) -> tuple[CertificationStatus, dict[str, Any]]:
    """Evaluate without substituting missing evidence with optimistic values."""
    robustness = robustness or {}
    fresh_backtests = fresh_backtests or {}
    metadata = snapshot.get("metadata", {})
    universe_selection = metadata.get("universe_selection") or {}
    execution_spec = _load_json(backtest.get("execution_spec_json"), {})
    adverse_dd = robustness.get("adverse_max_drawdown_pct")
    degraded_return = robustness.get("degraded_cost_return_pct")
    external_return = robustness.get("external_oos_return_pct")
    bootstrap_low = robustness.get("bootstrap_ci95_return_low")
    prob_loss = robustness.get("bootstrap_prob_loss")
    performance_gate_names = {
        "external_oos_return",
        "bootstrap_ci95_low",
        "bootstrap_prob_loss",
        "nominal_drawdown",
        "adverse_drawdown",
        "kill_switch",
        "simultaneous_sl_loss",
        "margin",
        "liquidation_distance",
        "degraded_cost_return",
        "fresh_180d_return",
        "fresh_180d_drawdown",
        "fresh_180d_kill_switch",
        "fresh_365d_return",
        "fresh_365d_drawdown",
        "fresh_365d_kill_switch",
    }

    gates = [
        _gate("snapshot_valid", snapshot.get("validation_status"),
              snapshot.get("validation_status") == "VALID", "VALID"),
        _gate("certifiable_result", backtest.get("result_status"),
              backtest.get("result_status") != "legacy", "not legacy"),
        _gate("external_oos_scope", backtest.get("evaluation_scope"),
              backtest.get("evaluation_scope") == "external_oos", "external_oos"),
        _gate(
            "declared_primary_leverage",
            backtest.get("leverage"),
            (
                True if not universe_selection else
                int(backtest.get("leverage") or 0)
                == int(universe_selection.get("primary_leverage", 0))
            ),
            (
                str(universe_selection.get("primary_leverage"))
                if universe_selection else "snapshot default"
            ),
        ),
        _gate("nominal_execution", backtest.get("execution_scenario"),
              backtest.get("execution_scenario") == "nominal", "nominal"),
        _gate("execution_calibration", execution_spec.get("calibration_id"),
              bool(execution_spec.get("calibration_id"))
              and int(execution_spec.get("calibration_sample_size", 0)) >= 30,
              "calibration_id and >= 30 filled observations"),
        _gate("missed_fill_calibration",
              execution_spec.get("calibration_unfilled_sample_size"),
              int(execution_spec.get("calibration_unfilled_sample_size", 0)) > 0
              and bool(execution_spec.get("calibration_observation_hash")),
              ">= 1 confirmed unfilled observation and hash"),
        _gate("external_oos_return", external_return,
              None if external_return is None else external_return > 0, "> 0%"),
        _gate("bootstrap_ci95_low", bootstrap_low,
              None if bootstrap_low is None else bootstrap_low > 0, "> 0%"),
        _gate("bootstrap_prob_loss", prob_loss,
              None if prob_loss is None else prob_loss < 0.10, "< 10%"),
        _gate(
            "nominal_drawdown",
            backtest.get("max_drawdown_pct"),
            (
                None if backtest.get("max_drawdown_pct") is None
                else abs(float(backtest["max_drawdown_pct"])) <= 30
            ),
            "<= 30%",
        ),
        _gate("adverse_drawdown", adverse_dd,
              None if adverse_dd is None else abs(float(adverse_dd)) <= 40, "<= 40%"),
        _gate(
            "kill_switch",
            backtest.get("kill_switch_triggers"),
            (
                None if backtest.get("kill_switch_triggers") is None
                else int(backtest["kill_switch_triggers"]) == 0
            ),
            "0",
        ),
        _gate(
            "simultaneous_sl_loss",
            backtest.get("worst_case_sl_loss_pct"),
            (
                None if backtest.get("worst_case_sl_loss_pct") is None
                else float(backtest["worst_case_sl_loss_pct"]) <= 30
            ),
            "<= 30%",
        ),
        _gate(
            "margin",
            backtest.get("peak_margin_ratio"),
            (
                None if backtest.get("peak_margin_ratio") is None
                else float(backtest["peak_margin_ratio"]) <= 0.70
            ),
            "<= 70%",
        ),
        _gate(
            "liquidation_distance",
            backtest.get("min_liquidation_distance_pct"),
            (
                None if backtest.get("min_liquidation_distance_pct") is None
                else float(backtest["min_liquidation_distance_pct"]) > 50
            ),
            "> 50%",
        ),
        _gate("degraded_cost_return", degraded_return,
              None if degraded_return is None else degraded_return > 0, "> 0%"),
        _gate("funding_coverage", backtest.get("missing_funding_events"),
              int(backtest.get("missing_funding_events", 1)) == 0, "0 missing"),
        _gate("intrabar_coverage", metadata.get("intrabar_missing", []),
              not metadata.get("intrabar_missing", []), "complete"),
        _gate(
            "intrabar_execution_used",
            backtest.get("execution_timeframe_used"),
            bool(metadata.get("execution_timeframe"))
            and backtest.get("execution_timeframe_used") == metadata.get("execution_timeframe"),
            str(metadata.get("execution_timeframe") or "snapshot execution timeframe"),
        ),
        _gate(
            "intrabar_broker_events",
            backtest.get("execution_candles_processed"),
            (
                None
                if backtest.get("execution_candles_processed") is None
                else int(backtest["execution_candles_processed"]) > 0
            ),
            "> 0 broker candles consumed",
        ),
        _gate(
            "intrabar_gap_bound",
            backtest.get("intrabar_max_gap_bars"),
            (
                None
                if backtest.get("intrabar_max_gap_bars") is None
                else int(backtest["intrabar_max_gap_bars"])
                <= int(metadata.get("max_gap_bars", 0))
            ),
            f"<= {int(metadata.get('max_gap_bars', 0))} bars",
        ),
        _gate("fast_canonical_parity", robustness.get("engine_parity_max_delta_pct"),
              None if robustness.get("engine_parity_passed") is None
              else bool(robustness.get("engine_parity_passed")),
              "cumulative delta <= 0.5%"),
    ]
    for days in (180, 365):
        fresh = fresh_backtests.get(days)
        gates.extend([
            _gate(
                f"fresh_{days}d_coverage",
                None if fresh is None else fresh.get("period_days"),
                None if fresh is None else float(fresh.get("period_days", 0)) >= days * 0.95,
                f">= {days * 0.95:.1f} days",
            ),
            _gate(
                f"fresh_{days}d_return",
                None if fresh is None else fresh.get("total_return_pct"),
                None if fresh is None else float(fresh.get("total_return_pct", -999)) > 0,
                "> 0%",
            ),
            _gate(
                f"fresh_{days}d_drawdown",
                None if fresh is None else fresh.get("max_drawdown_pct"),
                None if fresh is None else abs(
                    float(fresh.get("max_drawdown_pct", -999))
                ) <= 30,
                "<= 30%",
            ),
            _gate(
                f"fresh_{days}d_kill_switch",
                None if fresh is None else fresh.get("kill_switch_triggers"),
                None if fresh is None else int(
                    fresh.get("kill_switch_triggers", 1)
                ) == 0,
                "0",
            ),
        ])
    missing = [gate["name"] for gate in gates if gate["passed"] is None]
    failed = [gate["name"] for gate in gates if gate["passed"] is False]
    performance_failed = [
        name for name in failed if name in performance_gate_names
    ]
    capability_blockers = [
        gate["name"]
        for gate in gates
        if gate["passed"] is not True and gate["name"] not in performance_gate_names
    ]
    if performance_failed:
        status = CertificationStatus.HISTORICAL_FAIL
    elif missing or capability_blockers:
        status = CertificationStatus.RESEARCH_ONLY
    else:
        status = CertificationStatus.PAPER_READY
    return status, {
        "gates": gates,
        "missing_evidence": missing,
        "failed": failed,
        "performance_failed": performance_failed,
        "capability_blockers": capability_blockers,
    }


def _load_json(value: str | None, fallback: Any) -> Any:
    return json.loads(value) if value else fallback


def _preserve_forward_progress(
    historical_status: CertificationStatus, existing_status: str | None,
) -> CertificationStatus:
    """Keep forward progress only while the historical gates still pass."""
    if historical_status == CertificationStatus.PAPER_READY and existing_status in {
        CertificationStatus.LIVE_CANARY_READY.value,
        CertificationStatus.LIVE_APPROVED.value,
    }:
        return CertificationStatus(existing_status)
    return historical_status


def evaluate_certification(
    db_path: str,
    strategy_name: str,
    snapshot_id: str,
) -> tuple[str, CertificationStatus, dict[str, Any]]:
    """Create/update one deterministic certification record."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        snapshot_row = conn.execute(
            "SELECT * FROM data_snapshots WHERE id=?", (snapshot_id,),
        ).fetchone()
        if snapshot_row is None:
            raise ValueError(f"Unknown snapshot: {snapshot_id}")
        snapshot = _load_json(snapshot_row["manifest_json"], {})
        snapshot["validation_status"] = snapshot_row["validation_status"]
        snapshot["validation_errors"] = _load_json(snapshot_row["validation_errors"], [])

        candidates = conn.execute(
            """SELECT * FROM portfolio_backtests
               WHERE strategy_name=? AND manifest_json IS NOT NULL
                 AND execution_scenario='nominal'
                 AND evaluation_scope='external_oos'
               ORDER BY id DESC""",
            (strategy_name,),
        ).fetchall()
        backtest = None
        for row in candidates:
            manifest = _load_json(row["manifest_json"], {})
            if manifest.get("snapshot_id") == snapshot_id:
                backtest = dict(row)
                break

        if backtest is None:
            status = CertificationStatus.RESEARCH_ONLY
            details = {
                "gates": [],
                "missing_evidence": ["portfolio_backtest"],
                "failed": [],
            }
            portfolio_id = None
        else:
            portfolio_id = int(backtest["id"])
            robustness_row = conn.execute(
                """SELECT * FROM portfolio_robustness
                   WHERE backtest_id=? ORDER BY id DESC LIMIT 1""",
                (portfolio_id,),
            ).fetchone() if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='portfolio_robustness'"
            ).fetchone() else None
            fresh_backtests: dict[int, dict[str, Any]] = {}
            for days in (180, 365):
                fresh_candidates = conn.execute(
                    """SELECT * FROM portfolio_backtests
                       WHERE strategy_name=? AND manifest_json IS NOT NULL
                         AND execution_scenario='nominal'
                         AND evaluation_scope=?
                       ORDER BY id DESC""",
                    (strategy_name, f"fresh_capital_{days}d"),
                ).fetchall()
                for fresh_row in fresh_candidates:
                    fresh_manifest = _load_json(fresh_row["manifest_json"], {})
                    if fresh_manifest.get("snapshot_id") == snapshot_id:
                        fresh_backtests[days] = dict(fresh_row)
                        break
            status, details = evaluate_historical_gates(
                backtest,
                dict(robustness_row) if robustness_row else None,
                snapshot,
                fresh_backtests,
            )
            if backtest.get("manifest_hash") != snapshot_row["manifest_hash"]:
                details["gates"].append(_gate(
                    "manifest_hash", backtest.get("manifest_hash"), False,
                    snapshot_row["manifest_hash"],
                ))
                details["failed"].append("manifest_hash")
                details.setdefault("capability_blockers", []).append("manifest_hash")
                if status != CertificationStatus.HISTORICAL_FAIL:
                    status = CertificationStatus.RESEARCH_ONLY

        cert_key = canonical_json({
            "strategy": strategy_name,
            "snapshot_id": snapshot_id,
            "manifest_hash": snapshot_row["manifest_hash"],
        })
        certification_id = f"cert-{hashlib.sha256(cert_key.encode()).hexdigest()[:16]}"
        now = datetime.now(tz=timezone.utc).isoformat()
        existing = conn.execute(
            "SELECT created_at, status FROM strategy_certifications WHERE id=?",
            (certification_id,),
        ).fetchone()
        created_at = existing["created_at"] if existing else now
        # Re-evaluation of unchanged historical evidence must be idempotent
        # and cannot erase completed forward/canary stages. A historical
        # regression, however, invalidates them immediately.
        status = _preserve_forward_progress(
            status, existing["status"] if existing else None,
        )
        conn.execute(
            """INSERT INTO strategy_certifications
               (id, strategy_name, snapshot_id, status, created_at, updated_at,
                manifest_json, manifest_hash, gates_json, portfolio_backtest_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 status=excluded.status, updated_at=excluded.updated_at,
                 gates_json=excluded.gates_json,
                 portfolio_backtest_id=excluded.portfolio_backtest_id""",
            (
                certification_id, strategy_name, snapshot_id, status.value,
                created_at, now, snapshot_row["manifest_json"],
                snapshot_row["manifest_hash"], canonical_json(details), portfolio_id,
            ),
        )
        conn.commit()
        return certification_id, status, details
    finally:
        conn.close()


def validate_promotion_environment(
    certification: dict[str, Any], repo_root: Path, config_dir: Path,
) -> list[str]:
    manifest = _load_json(certification.get("manifest_json"), {})
    errors: list[str] = []
    commit, dirty, _diff_hash = git_provenance(repo_root)
    if commit != manifest.get("git_commit"):
        errors.append("git commit differs from certification")
    if dirty:
        errors.append("worktree is not clean")
    if config_hashes(config_dir) != manifest.get("config_hashes"):
        errors.append("configuration hashes differ from certification")
    if certification.get("status") != CertificationStatus.LIVE_APPROVED.value:
        errors.append("certification is not LIVE_APPROVED")
    paper = _load_json(certification.get("paper_metrics_json"), {})
    required = {
        "forward_days": 60,
        "completed_cycles": 30,
        "entry_orders": 100,
    }
    for key, minimum in required.items():
        if float(paper.get(key, 0)) < minimum:
            errors.append(f"paper {key} < {minimum}")
    if int(certification.get("canary_stage", 0)) < 4:
        errors.append("canary has not completed 10/25/50/100% stages")
    return errors


def compute_forward_metrics(
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive paper/canary gates from immutable raw comparison observations."""
    if not observations:
        return {
            "forward_days": 0.0, "completed_cycles": 0, "entry_orders": 0,
            "explained_ratio": 0.0, "max_size_deviation_pct": float("inf"),
            "fill_interval_ratio": 0.0, "pnl_deviation_pct": float("inf"),
            "missing_sl": 0, "orphan_orders": 0, "state_divergences": 0,
            "capital_fraction": None,
        }
    timestamps = [datetime.fromisoformat(item["timestamp"]) for item in observations]
    entries = [item for item in observations if item["observation_type"] == "entry"]
    cycles = [item for item in observations if item["observation_type"] == "cycle"]
    sizes = [
        abs(float(item["actual_quantity"]) / float(item["expected_quantity"]) - 1) * 100
        for item in entries
        if item.get("expected_quantity") not in (None, 0)
        and item.get("actual_quantity") is not None
    ]
    interval_rows = [
        item for item in entries if item.get("fill_in_simulated_interval") is not None
    ]
    pnl_rows = [
        item for item in observations
        if item.get("shadow_pnl") is not None and item.get("actual_pnl") is not None
    ]
    shadow_pnl = sum(float(item["shadow_pnl"]) for item in pnl_rows)
    actual_pnl = sum(float(item["actual_pnl"]) for item in pnl_rows)
    if abs(shadow_pnl) <= 1e-12:
        pnl_deviation = 0.0 if abs(actual_pnl) <= 1e-12 else float("inf")
    else:
        pnl_deviation = abs(actual_pnl - shadow_pnl) / abs(shadow_pnl) * 100
    capital_values = [
        float(item["capital_fraction"]) for item in observations
        if item.get("capital_fraction") is not None
    ]
    return {
        "forward_days": (max(timestamps) - min(timestamps)).total_seconds() / 86400,
        "completed_cycles": len(cycles),
        "entry_orders": len(entries),
        "explained_ratio": sum(bool(item.get("explained")) for item in observations)
        / len(observations),
        "max_size_deviation_pct": max(sizes, default=float("inf")),
        "fill_interval_ratio": (
            sum(bool(item.get("fill_in_simulated_interval")) for item in interval_rows)
            / len(interval_rows) if interval_rows else 0.0
        ),
        "pnl_deviation_pct": pnl_deviation,
        "missing_sl": sum(item.get("has_server_sl") != 1 for item in entries),
        "orphan_orders": sum(bool(item.get("orphan_order")) for item in observations),
        "state_divergences": sum(
            bool(item.get("state_divergence")) for item in observations
        ),
        "capital_fraction": (
            sum(capital_values) / len(capital_values) if capital_values else None
        ),
    }


def _forward_quality_passes(metrics: dict[str, Any]) -> bool:
    return (
        metrics["explained_ratio"] >= 1.0
        and metrics["max_size_deviation_pct"] <= 1.0
        and metrics["fill_interval_ratio"] >= 0.90
        and metrics["pnl_deviation_pct"] <= 20.0
        and metrics["missing_sl"] == 0
        and metrics["orphan_orders"] == 0
        and metrics["state_divergences"] == 0
    )


def review_paper_forward(
    db_path: str, certification_id: str,
) -> tuple[CertificationStatus, dict[str, Any]]:
    """Promote PAPER_READY to LIVE_CANARY_READY from raw paper evidence."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        certification = conn.execute(
            "SELECT * FROM strategy_certifications WHERE id=?", (certification_id,),
        ).fetchone()
        if certification is None:
            raise ValueError(f"Unknown certification: {certification_id}")
        if certification["status"] not in {
            CertificationStatus.PAPER_READY.value,
            CertificationStatus.LIVE_CANARY_READY.value,
        }:
            raise ValueError("Certification is not PAPER_READY")
        observations = [dict(row) for row in conn.execute(
            """SELECT * FROM certification_forward_observations
               WHERE certification_id=? AND phase='paper'
               ORDER BY timestamp, id""",
            (certification_id,),
        ).fetchall()]
        metrics = compute_forward_metrics(observations)
        passed = (
            metrics["forward_days"] >= 60
            and metrics["completed_cycles"] >= 30
            and metrics["entry_orders"] >= 100
            and _forward_quality_passes(metrics)
        )
        status = (
            CertificationStatus.LIVE_CANARY_READY if passed
            else CertificationStatus.PAPER_READY
        )
        started = observations[0]["timestamp"] if observations else None
        conn.execute(
            """UPDATE strategy_certifications
               SET status=?, updated_at=?, paper_started_at=?, paper_metrics_json=?
               WHERE id=?""",
            (
                status.value, datetime.now(tz=timezone.utc).isoformat(), started,
                canonical_json(metrics), certification_id,
            ),
        )
        conn.commit()
        return status, metrics
    finally:
        conn.close()


def review_next_canary_stage(
    db_path: str, certification_id: str,
) -> tuple[CertificationStatus, int, dict[str, Any]]:
    """Review the next explicit 10/25/50/100% canary phase."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        certification = conn.execute(
            "SELECT * FROM strategy_certifications WHERE id=?", (certification_id,),
        ).fetchone()
        if certification is None:
            raise ValueError(f"Unknown certification: {certification_id}")
        if certification["status"] != CertificationStatus.LIVE_CANARY_READY.value:
            raise ValueError("Certification is not LIVE_CANARY_READY")
        current_stage = int(certification["canary_stage"])
        if current_stage >= 4:
            return CertificationStatus.LIVE_APPROVED, 4, {}
        next_stage = current_stage + 1
        expected_fraction = (0.10, 0.25, 0.50, 1.0)[current_stage]
        observations = [dict(row) for row in conn.execute(
            """SELECT * FROM certification_forward_observations
               WHERE certification_id=? AND phase=?
               ORDER BY timestamp, id""",
            (certification_id, f"canary_{next_stage}"),
        ).fetchall()]
        metrics = compute_forward_metrics(observations)
        fraction = metrics.get("capital_fraction")
        passed = (
            bool(observations)
            and metrics["entry_orders"] > 0
            and metrics["completed_cycles"] > 0
            and _forward_quality_passes(metrics)
            and fraction is not None
            and abs(float(fraction) - expected_fraction) <= 0.01
        )
        if not passed:
            return CertificationStatus.LIVE_CANARY_READY, current_stage, metrics
        status = (
            CertificationStatus.LIVE_APPROVED
            if next_stage == 4 else CertificationStatus.LIVE_CANARY_READY
        )
        conn.execute(
            """UPDATE strategy_certifications
               SET status=?, canary_stage=?, updated_at=? WHERE id=?""",
            (
                status.value, next_stage,
                datetime.now(tz=timezone.utc).isoformat(), certification_id,
            ),
        )
        conn.commit()
        return status, next_stage, metrics
    finally:
        conn.close()
