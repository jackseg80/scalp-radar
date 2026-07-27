"""Measure fast/canonical parity on snapshot-bound WFO reference windows."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from backend.backtesting.engine_parity import measure_and_store_strategy_parity
from backend.core.config import get_config
from backend.core.database import Database
from backend.core.experiment import revalidate_snapshot


async def run(args: argparse.Namespace) -> int:
    db = Database(args.db)
    await db.init()
    await db.close()
    manifest, errors = await revalidate_snapshot(
        args.db, args.snapshot,
        config_dir=Path(args.config_dir), repo_root=Path.cwd(),
    )
    if errors:
        raise ValueError("Snapshot non reproductible: " + "; ".join(errors))
    results = await measure_and_store_strategy_parity(
        db_path=args.db,
        strategy_name=args.strategy,
        manifest_hash=manifest["manifest_hash"],
        config=get_config(args.config_dir, force_reload=True),
        exchange=args.exchange,
        seed=int(manifest.get("seed", 0)),
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if results and all(item.get("within_tolerance") for item in results) else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast/canonical engine parity")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--exchange", default="binance", choices=["binance", "bitget"])
    parser.add_argument("--db", default="data/scalp_radar.db")
    parser.add_argument("--config-dir", default="config")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
