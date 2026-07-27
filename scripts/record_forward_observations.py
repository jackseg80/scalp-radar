"""Ingest raw shadow/paper/canary comparisons into certification evidence.

The producer (shadow runner or reconciliation export) must emit one JSON
object per observation.  This command validates and signs every raw row; it
never accepts pre-computed aggregate metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.core.database import Database


def load_observations(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    return payload if isinstance(payload, list) else [payload]


def validate_observation(
    item: dict[str, Any], certification_id: str, phase: str,
) -> dict[str, Any]:
    allowed_phases = {"paper", "canary_1", "canary_2", "canary_3", "canary_4"}
    if phase not in allowed_phases:
        raise ValueError(f"Unsupported forward phase: {phase}")
    observation_type = item.get("observation_type")
    if observation_type not in {"entry", "cycle"}:
        raise ValueError("observation_type must be entry or cycle")
    for field in ("timestamp", "intent_id", "explained"):
        if field not in item:
            raise ValueError(f"Missing raw observation field: {field}")
    timestamp = datetime.fromisoformat(str(item["timestamp"]).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Forward observation timestamps must include a timezone")
    normalized = dict(item)
    normalized["timestamp"] = timestamp.isoformat()
    normalized["certification_id"] = certification_id
    normalized["phase"] = phase
    if observation_type == "entry":
        for field in (
            "expected_quantity", "actual_quantity",
            "fill_in_simulated_interval", "has_server_sl",
        ):
            if normalized.get(field) is None:
                raise ValueError(f"Entry observation requires {field}")
    if phase.startswith("canary_") and normalized.get("capital_fraction") is None:
        raise ValueError("Canary observations require capital_fraction")
    return normalized


async def ingest(args: argparse.Namespace) -> int:
    db = Database(args.db)
    await db.init()
    try:
        assert db._conn is not None
        row = await (
            await db._conn.execute(
                "SELECT status FROM strategy_certifications WHERE id=?",
                (args.certification_id,),
            )
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown certification: {args.certification_id}")
        expected_status = "PAPER_READY" if args.phase == "paper" else "LIVE_CANARY_READY"
        if row[0] != expected_status:
            raise ValueError(
                f"Certification status {row[0]} cannot receive {args.phase} evidence; "
                f"expected {expected_status}"
            )
        observations = [
            validate_observation(item, args.certification_id, args.phase)
            for item in load_observations(Path(args.input))
        ]
        if args.dry_run:
            return len(observations)
        for item in observations:
            await db.insert_forward_observation(item)
        return len(observations)
    finally:
        await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record raw forward certification observations",
    )
    parser.add_argument("--certification-id", required=True)
    parser.add_argument(
        "--phase", required=True,
        choices=["paper", "canary_1", "canary_2", "canary_3", "canary_4"],
    )
    parser.add_argument("--input", required=True, help="JSON or JSONL raw observations")
    parser.add_argument("--db", default="data/scalp_radar.db")
    parser.add_argument("--dry-run", action="store_true")
    parsed = parser.parse_args()
    count = asyncio.run(ingest(parsed))
    print(json.dumps({"validated": count, "written": 0 if parsed.dry_run else count}))
