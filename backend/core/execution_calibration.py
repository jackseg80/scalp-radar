"""Build reproducible ExecutionSpec calibration from persisted Bitget orders."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

import numpy as np

from backend.core.experiment import canonical_json
from backend.core.models import ExecutionSpec


def calibrate_execution(
    *,
    db_path: str,
    maker_fee_pct: float,
    taker_fee_pct: float,
    default_slippage_pct: float,
    strategy_name: str | None = None,
    strategy_prefix: str | None = None,
    source_db_path: str | None = None,
    min_filled_observations: int = 30,
    min_unfilled_observations: int = 1,
    since: datetime | None = None,
    until: datetime | None = None,
    seed: int = 0,
) -> tuple[str, ExecutionSpec]:
    """Calibrate nominal medians and adverse p95 from real order outcomes."""
    source_path = source_db_path or db_path
    source_conn = sqlite3.connect(source_path)
    source_conn.row_factory = sqlite3.Row
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conditions = [
            "trade_type IN ('entry', 'entry_unfilled')",
            "order_id IS NOT NULL", "order_id!=''",
            "intent_timestamp IS NOT NULL",
            "requested_quantity IS NOT NULL",
            "order_status IS NOT NULL",
        ]
        params: list[Any] = []
        if strategy_name:
            conditions.append("strategy_name=?")
            params.append(strategy_name)
        if strategy_prefix:
            conditions.append("strategy_name LIKE ?")
            params.append(f"{strategy_prefix}%")
        if since:
            conditions.append("timestamp>=?")
            params.append(since.astimezone(timezone.utc).isoformat())
        if until:
            conditions.append("timestamp<=?")
            params.append(until.astimezone(timezone.utc).isoformat())
        rows = source_conn.execute(
            "SELECT * FROM live_trades WHERE " + " AND ".join(conditions)
            + " ORDER BY timestamp, order_id",
            params,
        ).fetchall()
        observations = [dict(row) for row in rows]
        filled = [
            row for row in observations
            if float(row.get("filled_quantity") or 0) > 0
            and row.get("latency_ms") is not None
            and row.get("slippage_pct") is not None
        ]
        unfilled = [
            row for row in observations
            # A cancelled partial order contains a confirmed non-filled
            # remainder.  Count it once for missed-fill calibration while it
            # remains separately represented in the partial-fill sample.
            if float(row.get("requested_quantity") or 0)
            > float(row.get("filled_quantity") or 0)
            and row.get("order_status") in {
                "canceled", "cancelled", "expired", "rejected",
            }
        ]
        partial = [
            row for row in filled if float(row.get("fill_ratio") or 0) < 0.99
        ]
        if len(unfilled) < min_unfilled_observations:
            raise ValueError(
                "Insufficient confirmed unfilled observations: "
                f"{len(unfilled)} < required {min_unfilled_observations}"
            )
        if len(filled) < min_filled_observations:
            raise ValueError(
                "Insufficient filled execution observations: "
                f"{len(filled)} < required {min_filled_observations}"
            )

        latency = np.asarray([float(row["latency_ms"]) for row in filled])
        slippage = np.asarray([abs(float(row["slippage_pct"])) for row in filled])
        observation_payload = [
            {
                key: row.get(key) for key in (
                    "order_id", "timestamp", "strategy_name", "symbol", "direction",
                    "intent_timestamp", "intent_price", "requested_quantity",
                    "filled_quantity", "fill_timestamp", "latency_ms",
                    "slippage_pct", "fill_ratio", "order_status", "context",
                )
            }
            for row in observations
        ]
        observation_hash = hashlib.sha256(
            canonical_json(observation_payload).encode("utf-8")
        ).hexdigest()
        window_start = min(str(row["timestamp"]) for row in observations)
        window_end = max(str(row["timestamp"]) for row in observations)
        nominal_slippage = max(default_slippage_pct, float(np.median(slippage)))
        spec = ExecutionSpec(
            maker_fee_pct=maker_fee_pct,
            taker_fee_pct=taker_fee_pct,
            slippage_pct=nominal_slippage,
            latency_ms=max(0, int(round(float(np.median(latency))))),
            missed_fill_probability=len(unfilled) / len(observations),
            partial_fill_probability=len(partial) / len(filled),
            calibration_sample_size=len(filled),
            calibration_unfilled_sample_size=len(unfilled),
            calibration_partial_sample_size=len(partial),
            calibration_observation_hash=observation_hash,
            calibration_window_start=datetime.fromisoformat(window_start),
            calibration_window_end=datetime.fromisoformat(window_end),
            calibration_latency_p95_ms=max(0, int(round(float(np.percentile(latency, 95))))),
            calibration_slippage_p95_pct=float(np.percentile(slippage, 95)),
            random_seed=seed,
        )
        identity = canonical_json({
            "observation_hash": observation_hash,
            "strategy_name": strategy_name,
            "strategy_prefix": strategy_prefix,
            "source_db_path": source_path,
            "spec": spec.model_dump(mode="json"),
        })
        calibration_id = f"cal-{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
        spec = spec.model_copy(update={"calibration_id": calibration_id})
        conn.execute(
            """INSERT OR IGNORE INTO execution_calibrations
               (id, created_at, exchange, strategy_name, window_start, window_end,
                sample_size, unfilled_sample_size, partial_sample_size,
                observation_hash, execution_spec_json)
               VALUES (?, ?, 'bitget', ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                calibration_id, datetime.now(tz=timezone.utc).isoformat(),
                strategy_name, window_start, window_end, len(filled), len(unfilled),
                len(partial), observation_hash,
                canonical_json(spec.model_dump(mode="json")),
            ),
        )
        conn.commit()
        return calibration_id, spec
    finally:
        source_conn.close()
        conn.close()


def load_execution_calibration(db_path: str, calibration_id: str) -> ExecutionSpec:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT execution_spec_json FROM execution_calibrations WHERE id=?",
            (calibration_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown execution calibration: {calibration_id}")
        return ExecutionSpec.model_validate(json.loads(row[0]))
    finally:
        conn.close()
