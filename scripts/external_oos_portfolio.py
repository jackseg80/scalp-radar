"""Replay nested WFO external windows through the canonical portfolio engine."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from backend.backtesting.external_oos import (
    build_external_window_plans,
    load_wfo_rows,
    require_complete_universe_wfo_rows,
    run_external_oos,
)
from backend.backtesting.portfolio_db import save_result_sync
from backend.backtesting.portfolio_engine import format_portfolio_report
from backend.core.config import get_config
from backend.core.database import Database
from backend.core.experiment import (
    load_snapshot,
    revalidate_snapshot,
    snapshot_manifest_hash,
    wfo_reuse_fingerprint,
)
from backend.core.models import ExecutionSpec, UniverseSelectionSpec


def _load_external_oos_config(config_dir: str) -> object:
    """Load explicit evidence YAML without inheriting the developer's .env."""
    path = Path(config_dir)
    if not path.is_dir():
        raise ValueError(f"Configuration directory not found: {path}")
    env_file = ".env" if path == Path("config") else None
    return get_config(path, env_file=env_file, force_reload=True)


async def run(args: argparse.Namespace) -> int:
    db = Database(args.db)
    await db.init()
    await db.close()
    manifest, errors = await revalidate_snapshot(
        args.db,
        args.snapshot,
        config_dir=Path(args.config_dir),
        repo_root=Path.cwd(),
    )
    if errors:
        raise ValueError("Snapshot non reproductible: " + "; ".join(errors))
    wfo_manifest = manifest
    if args.wfo_snapshot:
        source = await load_snapshot(args.db, args.wfo_snapshot)
        if source is None:
            raise ValueError(f"Unknown WFO source snapshot: {args.wfo_snapshot}")
        if source.get("validation_status") != "VALID":
            raise ValueError(
                f"WFO source snapshot is not valid: {source.get('validation_status')}"
            )
        if wfo_reuse_fingerprint(source) != wfo_reuse_fingerprint(manifest):
            raise ValueError(
                "WFO source snapshot is incompatible: data/config/calendar/"
                "selection inputs differ"
            )
        wfo_manifest = source
        print(
            "[WFO REUSE] compatible source "
            f"{args.wfo_snapshot} -> canonical replay {args.snapshot}"
        )
    rows = load_wfo_rows(
        args.db, args.strategy, snapshot_manifest_hash(wfo_manifest),
    )
    selection_raw = manifest.get("metadata", {}).get("universe_selection")
    selection = UniverseSelectionSpec.model_validate(selection_raw) if selection_raw else None
    if selection and selection.strategy_name != args.strategy:
        raise ValueError(
            f"Snapshot universe selection targets {selection.strategy_name}, not {args.strategy}"
        )
    require_complete_universe_wfo_rows(rows, selection)
    cutoff = datetime.fromisoformat(str(manifest["cutoff"]).replace("Z", "+00:00"))
    plans = build_external_window_plans(rows, selection=selection, cutoff=cutoff)
    if not plans:
        raise ValueError(
            "Aucune fenêtre WFO liée au snapshot. Exécutez optimize --snapshot "
            "pour les assets de l'univers préenregistré."
        )
    config = _load_external_oos_config(args.config_dir)
    base_spec = ExecutionSpec.model_validate(manifest["metadata"]["execution_spec"])
    execution_spec = base_spec.with_scenario(args.execution_scenario)
    if args.all_leverages and not selection:
        raise ValueError("--all-leverages requires a universe-discovery snapshot")
    leverages = (
        selection.leverage_scenarios if args.all_leverages else
        [args.leverage or (selection.primary_leverage if selection else None)]
    )
    for leverage in leverages:
        result = await run_external_oos(
            config=config,
            strategy_name=args.strategy,
            plans=plans,
            initial_capital=args.capital,
            db_path=args.db,
            exchange=args.exchange,
            execution_spec=execution_spec,
            kill_switch_pct=args.kill_switch,
            kill_switch_window_hours=args.kill_switch_window,
            leverage=leverage,
        )
        if args.wfo_snapshot:
            for audit in result.universe_selection:
                audit["wfo_source_snapshot"] = args.wfo_snapshot
        suffix = f"_{leverage}x" if leverage else ""
        if args.label and args.all_leverages:
            label = f"{args.label}_{leverage}x"
        else:
            label = args.label or (
            f"{args.strategy}_{args.snapshot}_{args.execution_scenario}_external_oos{suffix}"
            )
        evaluation_scope = "external_oos"
        if selection and leverage != selection.primary_leverage:
            evaluation_scope = f"universe_sensitivity_{leverage}x"
        result_id = save_result_sync(
            db_path=args.db,
            result=result,
            strategy_name=args.strategy,
            exchange=args.exchange,
            kill_switch_pct=args.kill_switch,
            kill_switch_window_hours=args.kill_switch_window,
            label=label,
            manifest=manifest,
            result_status="RESEARCH_ONLY",
            evaluation_scope=evaluation_scope,
        )
        print(format_portfolio_report(result))
        print(f"\n[SAVE] external_oos id={result_id}, label={label}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Canonical nested external-OOS portfolio replay",
    )
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument(
        "--wfo-snapshot",
        help=(
            "Explicit compatible source for already-computed WFO evidence. "
            "Only accepted when its WFO input fingerprint matches --snapshot."
        ),
    )
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--exchange", default="binance", choices=["binance", "bitget"])
    parser.add_argument(
        "--execution-scenario", default="nominal",
        choices=["nominal", "favorable", "adverse"],
    )
    parser.add_argument("--kill-switch", type=float, default=45.0)
    parser.add_argument("--kill-switch-window", type=int, default=24)
    parser.add_argument("--leverage", type=int)
    parser.add_argument("--all-leverages", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--db", default="data/scalp_radar.db")
    parser.add_argument("--config-dir", default="config")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
