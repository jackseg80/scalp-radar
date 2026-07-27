"""Create a traceable execution calibration from persisted Bitget outcomes."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone

from backend.core.config import get_config
from backend.core.database import Database
from backend.core.execution_calibration import calibrate_execution


def _date(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


async def run(args: argparse.Namespace) -> int:
    db = Database(args.db)
    await db.init()
    await db.close()
    config = get_config(args.config_dir, force_reload=True)
    calibration_id, spec = calibrate_execution(
        db_path=args.db,
        maker_fee_pct=config.risk.fees.maker_percent,
        taker_fee_pct=config.risk.fees.taker_percent,
        default_slippage_pct=config.risk.slippage.default_estimate_percent,
        strategy_name=args.strategy,
        since=_date(args.since),
        until=_date(args.until),
        seed=args.seed,
    )
    print(json.dumps({
        "calibration_id": calibration_id,
        "execution_spec": spec.model_dump(mode="json"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate execution from Bitget fills")
    parser.add_argument("--strategy")
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--db", default="data/scalp_radar.db")
    parser.add_argument("--config-dir", default="config")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
