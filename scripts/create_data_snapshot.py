"""Create a reproducible certification snapshot from the existing DB."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.core.config import get_config
from backend.core.experiment import create_snapshot
from backend.core.execution_calibration import load_execution_calibration
from backend.core.database import Database
from backend.core.models import ExecutionSpec, UniverseSelectionSpec


def _utc_datetime(raw: str) -> datetime:
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_snapshot_config(config_dir: str) -> object:
    """Load explicit evidence YAML without inheriting the developer's .env."""
    path = Path(config_dir)
    if not path.is_dir():
        raise ValueError(f"Configuration directory not found: {path}")
    env_file = ".env" if path == Path("config") else None
    return get_config(path, env_file=env_file, force_reload=True)


async def main(args: argparse.Namespace) -> int:
    config_dir = getattr(args, "config_dir", "config")
    config = _load_snapshot_config(config_dir)
    db = Database(args.db)
    await db.init()
    await db.close()
    symbols = (
        [value.strip() for value in args.symbols.split(",") if value.strip()]
        if args.symbols else [asset.symbol for asset in config.assets]
    )
    timeframes = [value.strip() for value in args.timeframes.split(",") if value.strip()]
    universe_selection = None
    if args.universe_discovery:
        if args.symbols:
            raise ValueError(
                "--universe-discovery uses all configured assets; do not pass --symbols"
            )
        if not args.calendar_start:
            raise ValueError("--universe-discovery requires --calendar-start")
        if "1h" not in timeframes:
            raise ValueError("--universe-discovery requires 1h signal candles")
        universe_selection = UniverseSelectionSpec(
            strategy_name=args.strategy,
            universe_symbols=symbols,
            calendar_start=_utc_datetime(args.calendar_start),
            is_window_days=args.is_window_days,
            embargo_days=args.embargo_days,
            oos_window_days=args.oos_window_days,
            step_days=args.step_days,
            top_n=args.top_n,
            min_is_sharpe=args.min_is_sharpe,
            min_is_net_return_pct=args.min_is_net_return_pct,
            min_is_trades=args.min_is_trades,
            search_mode="exhaustive",
            primary_leverage=args.primary_leverage,
            leverage_scenarios=[
                int(value.strip()) for value in args.leverage_scenarios.split(",")
                if value.strip()
            ],
        )
    series = [
        (args.exchange, symbol, timeframe)
        for symbol in symbols
        for timeframe in timeframes
    ]
    if args.validate:
        series.extend(
            (args.calibration_exchange, symbol, timeframe)
            for symbol in symbols
            for timeframe in sorted(set(timeframes + [args.execution_timeframe]))
        )
    if args.calibration_id:
        spec_data = load_execution_calibration(
            args.db, args.calibration_id,
        ).model_dump(mode="json")
        spec_data.update({
            "execution_timeframe": args.execution_timeframe,
            "random_seed": args.seed,
        })
        spec = ExecutionSpec.model_validate(spec_data)
    else:
        if args.validate:
            raise ValueError(
                "--validate requiert --calibration-id produit par "
                "scripts.calibrate_execution"
            )
        spec = ExecutionSpec(
            exchange="bitget",
            execution_timeframe=args.execution_timeframe,
            maker_fee_pct=config.risk.fees.maker_percent,
            taker_fee_pct=config.risk.fees.taker_percent,
            slippage_pct=config.risk.slippage.default_estimate_percent,
            latency_ms=args.latency_p95_ms,
            missed_fill_probability=args.missed_fill_probability,
            partial_fill_probability=args.partial_fill_probability,
            random_seed=args.seed,
        )
    snapshot_id, manifest = await create_snapshot(
        db_path=args.db,
        series=series,
        cutoff=_utc_datetime(args.cutoff),
        start=_utc_datetime(args.since) if args.since else None,
        config_dir=Path(config_dir),
        repo_root=Path.cwd(),
        seed=args.seed,
        max_gap_bars=args.max_gap_bars,
        require_execution_timeframe=args.validate,
        execution_spec=spec,
        universe_selection=universe_selection,
    )
    print(json.dumps({"snapshot_id": snapshot_id, **manifest}, indent=2, ensure_ascii=False))
    return 0 if manifest.get("validation_status", "VALID") == "VALID" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an immutable data snapshot")
    parser.add_argument("--cutoff", required=True, help="Exact ISO-8601 cutoff")
    parser.add_argument("--since", help="Optional exact ISO-8601 start")
    parser.add_argument("--symbols", help="Comma-separated symbols; default: all configured assets")
    parser.add_argument("--strategy", default="grid_atr")
    parser.add_argument("--universe-discovery", action="store_true")
    parser.add_argument("--calendar-start", help="UTC start of the aligned WFO calendar")
    parser.add_argument("--is-window-days", type=int, default=180)
    parser.add_argument("--embargo-days", type=int, default=7)
    parser.add_argument("--oos-window-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=60)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--min-is-sharpe", type=float, default=0.0)
    parser.add_argument("--min-is-net-return-pct", type=float, default=0.0)
    parser.add_argument("--min-is-trades", type=int, default=10)
    parser.add_argument("--primary-leverage", type=int, default=4)
    parser.add_argument("--leverage-scenarios", default="2,4,6")
    parser.add_argument("--timeframes", default="1h,1m")
    parser.add_argument("--execution-timeframe", default="1m", choices=["1m", "5m", "15m"])
    parser.add_argument("--exchange", default="binance", choices=["binance", "bitget"])
    parser.add_argument(
        "--calibration-exchange", default="bitget", choices=["bitget"],
        help="Execution calibration source added by --validate",
    )
    parser.add_argument("--db", default="data/scalp_radar.db")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-gap-bars", type=int, default=0)
    parser.add_argument("--latency-p95-ms", type=int, default=0)
    parser.add_argument("--missed-fill-probability", type=float, default=0.0)
    parser.add_argument("--partial-fill-probability", type=float, default=0.0)
    parser.add_argument("--calibration-id")
    parser.add_argument(
        "--config-dir",
        default="config",
        help="YAML directory; an explicit directory ignores the local .env",
    )
    parser.add_argument("--validate", action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
