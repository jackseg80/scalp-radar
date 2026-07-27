"""Review paper or next canary phase from raw certification observations."""

from __future__ import annotations

import argparse
import asyncio
import json

from backend.core.certification import review_next_canary_stage, review_paper_forward
from backend.core.database import Database


async def run(args: argparse.Namespace) -> int:
    db = Database(args.db)
    await db.init()
    await db.close()
    if args.phase == "paper":
        status, metrics = review_paper_forward(args.db, args.certification_id)
        payload = {"status": status.value, "metrics": metrics}
    else:
        status, stage, metrics = review_next_canary_stage(
            args.db, args.certification_id,
        )
        payload = {"status": status.value, "canary_stage": stage, "metrics": metrics}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if status.value in {"LIVE_CANARY_READY", "LIVE_APPROVED"} else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review forward certification evidence")
    parser.add_argument("--certification-id", required=True)
    parser.add_argument("--phase", required=True, choices=["paper", "canary"])
    parser.add_argument("--db", default="data/scalp_radar.db")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
