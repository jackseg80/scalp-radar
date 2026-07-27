"""Run or resume the strict historical strategy certification pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from backend.backtesting.engine_parity import measure_and_store_strategy_parity
from backend.backtesting.certification_capabilities import (
    canonical_certification_capability,
)
from backend.backtesting.external_oos import (
    build_external_window_plans,
    clip_external_window_plans,
    load_wfo_rows,
    require_complete_universe_wfo_rows,
    run_external_oos,
)
from backend.backtesting.portfolio_db import save_result_sync
from backend.core.certification import evaluate_certification
from backend.core.config import get_config
from backend.core.database import Database
from backend.core.experiment import canonical_json, revalidate_snapshot
from backend.core.models import ExecutionSpec, UniverseSelectionSpec
from scripts.portfolio_robustness import analyze_label
from scripts.portfolio_backtest import _long_replay_runtime_error


def _certification_runtime_error(
    platform_name: str,
    version_info: tuple[int, ...],
    *,
    evaluate_only: bool,
) -> str | None:
    """Reuse the canonical replay guard while allowing read-only evaluation."""
    if evaluate_only:
        return None
    return _long_replay_runtime_error(platform_name, version_info)


def _find_portfolio_evidence(
    db_path: str,
    strategy_name: str,
    manifest_hash: str,
    scenario: str,
    expected_spec: ExecutionSpec,
    evaluation_scope: str = "external_oos",
    leverage: int | None = None,
) -> dict | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT * FROM portfolio_backtests
               WHERE strategy_name=? AND manifest_hash=?
                 AND evaluation_scope=?
                 AND (? IS NULL OR leverage=?)
                 AND execution_scenario=? ORDER BY id DESC""",
            (strategy_name, manifest_hash, evaluation_scope, leverage, leverage, scenario),
        ).fetchall()
        expected = canonical_json(expected_spec.model_dump(mode="json"))
        for row in rows:
            actual = row["execution_spec_json"]
            if actual and canonical_json(json.loads(actual)) == expected:
                return dict(row)
        return None
    finally:
        conn.close()


def _parity_complete(rows: list[dict]) -> bool:
    if not rows:
        return False
    for row in rows:
        raw = row.get("engine_parity_json")
        if not raw:
            return False
    return True


async def _ensure_external_replay(
    *,
    args: argparse.Namespace,
    manifest: dict,
    config: object,
    plans: list,
    base_spec: ExecutionSpec,
    scenario: str,
    kill_switch_pct: float,
    kill_switch_hours: int,
    evaluation_scope: str = "external_oos",
    leverage: int | None = None,
) -> dict:
    spec = base_spec.with_scenario(scenario)
    existing = _find_portfolio_evidence(
        args.db, args.strategy, manifest["manifest_hash"], scenario, spec,
        evaluation_scope, leverage,
    )
    if existing is not None:
        print(f"[REUSE] Portfolio {scenario} id={existing['id']}")
        return existing
    result = await run_external_oos(
        config=config,
        strategy_name=args.strategy,
        plans=plans,
        initial_capital=args.capital,
        db_path=args.db,
        exchange=args.exchange,
        execution_spec=spec,
        kill_switch_pct=kill_switch_pct,
        kill_switch_window_hours=kill_switch_hours,
        leverage=leverage,
    )
    suffix = f"_{leverage}x" if leverage else ""
    label = f"{args.strategy}_{args.snapshot}_{scenario}_{evaluation_scope}{suffix}"
    result_id = save_result_sync(
        db_path=args.db,
        result=result,
        strategy_name=args.strategy,
        exchange=args.exchange,
        kill_switch_pct=kill_switch_pct,
        kill_switch_window_hours=kill_switch_hours,
        label=label,
        manifest=manifest,
        result_status="RESEARCH_ONLY",
        evaluation_scope=evaluation_scope,
    )
    print(f"[RUN] Portfolio {scenario} sauvegardé id={result_id}")
    return {"id": result_id, "label": label}


async def main(args: argparse.Namespace) -> int:
    runtime_error = _certification_runtime_error(
        sys.platform,
        tuple(sys.version_info),
        evaluate_only=args.evaluate_only,
    )
    if runtime_error:
        print(runtime_error, file=sys.stderr)
        return 2
    # Ensure every idempotent migration exists before sync readers/writers.
    db = Database(args.db)
    await db.init()
    await db.close()
    manifest, snapshot_errors = await revalidate_snapshot(
        args.db, args.snapshot,
        config_dir=Path(args.config_dir), repo_root=Path.cwd(),
    )
    if snapshot_errors:
        raise ValueError("Snapshot non reproductible: " + "; ".join(snapshot_errors))

    if not args.evaluate_only:
        supported, reason = canonical_certification_capability(args.strategy)
        if not supported:
            raise ValueError(
                f"{args.strategy} remains RESEARCH_ONLY: {reason}. "
                "No result was simulated with an incompatible engine."
            )
        config = get_config(
            args.config_dir,
            env_file=None if Path(args.config_dir) != Path("config") else ".env",
            force_reload=True,
        )
        rows = load_wfo_rows(args.db, args.strategy, manifest["manifest_hash"])
        if not rows:
            raise ValueError(
                "Aucun WFO snapshot-bound. Lancez scripts.optimize --snapshot "
                "pour chaque asset de l'univers préenregistré."
            )
        if _parity_complete(rows):
            print("[REUSE] Contrôles fast/canonique déjà présents")
        else:
            parity = await measure_and_store_strategy_parity(
                db_path=args.db,
                strategy_name=args.strategy,
                manifest_hash=manifest["manifest_hash"],
                config=config,
                exchange=args.exchange,
                seed=int(manifest.get("seed", 0)),
            )
            print(f"[RUN] Parité mesurée pour {len(parity)} assets")
        # Reload rows because parity evidence was attached in place.
        rows = load_wfo_rows(args.db, args.strategy, manifest["manifest_hash"])
        selection_raw = manifest.get("metadata", {}).get("universe_selection")
        selection = UniverseSelectionSpec.model_validate(selection_raw) if selection_raw else None
        require_complete_universe_wfo_rows(rows, selection)
        cutoff = datetime.fromisoformat(str(manifest["cutoff"]).replace("Z", "+00:00"))
        plans = build_external_window_plans(rows, selection=selection, cutoff=cutoff)
        primary_leverage = selection.primary_leverage if selection else None
        base_spec = ExecutionSpec.model_validate(manifest["metadata"]["execution_spec"])
        ks_config = getattr(config.risk, "kill_switch", None)
        kill_switch_pct = float(
            getattr(ks_config, "global_max_loss_pct", 45.0)
        )
        kill_switch_hours = int(getattr(ks_config, "global_window_hours", 24))
        nominal = await _ensure_external_replay(
            args=args, manifest=manifest, config=config, plans=plans,
            base_spec=base_spec, scenario="nominal",
            kill_switch_pct=kill_switch_pct, kill_switch_hours=kill_switch_hours,
            leverage=primary_leverage,
        )
        adverse = await _ensure_external_replay(
            args=args, manifest=manifest, config=config, plans=plans,
            base_spec=base_spec, scenario="adverse",
            kill_switch_pct=kill_switch_pct, kill_switch_hours=kill_switch_hours,
            leverage=primary_leverage,
        )
        if selection:
            for leverage in selection.leverage_scenarios:
                if leverage == selection.primary_leverage:
                    continue
                await _ensure_external_replay(
                    args=args, manifest=manifest, config=config, plans=plans,
                    base_spec=base_spec, scenario="nominal",
                    kill_switch_pct=kill_switch_pct, kill_switch_hours=kill_switch_hours,
                    evaluation_scope=f"universe_sensitivity_{leverage}x",
                    leverage=leverage,
                )
        for fresh_days in (180, 365):
            fresh_scope = f"fresh_capital_{fresh_days}d"
            fresh_plans = clip_external_window_plans(
                plans,
                start=cutoff - timedelta(days=fresh_days),
                end=cutoff,
            )
            if not fresh_plans:
                print(
                    f"[MISSING] Aucune fenêtre OOS pour {fresh_scope}; "
                    "la certification restera RESEARCH_ONLY"
                )
                continue
            await _ensure_external_replay(
                args=args,
                manifest=manifest,
                config=config,
                plans=fresh_plans,
                base_spec=base_spec,
                scenario="nominal",
                kill_switch_pct=kill_switch_pct,
                kill_switch_hours=kill_switch_hours,
                evaluation_scope=fresh_scope,
                leverage=primary_leverage,
            )

        conn = sqlite3.connect(args.db)
        conn.row_factory = sqlite3.Row
        try:
            existing_robustness = conn.execute(
                """SELECT id FROM portfolio_robustness
                   WHERE backtest_id=? AND adverse_backtest_id=?
                     AND manifest_hash=? ORDER BY id DESC LIMIT 1""",
                (nominal["id"], adverse["id"], manifest["manifest_hash"]),
            ).fetchone()
            if existing_robustness:
                print(f"[REUSE] Robustesse id={existing_robustness['id']}")
            else:
                np.random.seed(int(manifest.get("seed", 0)))
                analyze_label(
                    conn,
                    nominal.get("label") or f"{args.strategy}_{args.snapshot}_nominal_external_oos",
                    args.n_simulations,
                    args.block_size,
                    95.0,
                    True,
                    adverse_label=(
                        adverse.get("label")
                        or f"{args.strategy}_{args.snapshot}_adverse_external_oos"
                    ),
                )
        finally:
            conn.close()

    certification_id, status, details = evaluate_certification(
        args.db, args.strategy, args.snapshot,
    )
    print(json.dumps({
        "certification_id": certification_id,
        "status": status.value,
        **details,
    }, indent=2, ensure_ascii=False))
    return 0 if status.value in {"PAPER_READY", "LIVE_CANARY_READY", "LIVE_APPROVED"} else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Certify a strategy from immutable evidence")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--exchange", default="binance", choices=["binance", "bitget"])
    parser.add_argument("--n-simulations", type=int, default=5000)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--db", default="data/scalp_radar.db")
    parser.add_argument("--config-dir", default="config")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
