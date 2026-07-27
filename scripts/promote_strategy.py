"""Final local promotion gate. This command never deploys robot2."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from backend.backtesting.external_oos import build_external_window_plans, load_wfo_rows
from backend.core.certification import validate_promotion_environment
from backend.core.database import Database


async def _ensure_schema(db_path: str) -> None:
    """Run the normal idempotent migrations before the sync transaction."""
    db = Database(db_path)
    await db.init()
    await db.close()


def main(args: argparse.Namespace) -> int:
    asyncio.run(_ensure_schema(args.db))
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM strategy_certifications WHERE id=?",
            (args.certification_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown certification: {args.certification_id}")
        certification = dict(row)
        errors = validate_promotion_environment(
            certification, Path.cwd(), Path("config"),
        )
        if errors:
            print(json.dumps({"promoted": False, "errors": errors}, indent=2))
            return 2
        robustness = conn.execute(
            """SELECT * FROM portfolio_robustness
               WHERE backtest_id=? ORDER BY id DESC LIMIT 1""",
            (certification.get("portfolio_backtest_id"),),
        ).fetchone()
        if robustness is None:
            print(json.dumps({
                "promoted": False, "errors": ["missing robustness evidence"],
            }, indent=2))
            return 2
        robustness = dict(robustness)
        external_return = robustness.get("external_oos_return_pct")
        adverse_dd = robustness.get("adverse_max_drawdown_pct")
        if external_return is None or adverse_dd in (None, 0):
            print(json.dumps({
                "promoted": False,
                "errors": ["robust score cannot be computed"],
            }, indent=2))
            return 2
        robust_score = float(external_return) / abs(float(adverse_dd))

        champion = conn.execute(
            """SELECT * FROM strategy_certifications
               WHERE strategy_name=? AND promoted_at IS NOT NULL AND id!=?
               ORDER BY promoted_at DESC LIMIT 1""",
            (certification["strategy_name"], args.certification_id),
        ).fetchone()
        if champion is not None:
            champion = dict(champion)
            champion_score = champion.get("robust_score")
            champion_dd = champion.get("adverse_drawdown_pct")
            comparison_errors = []
            if champion_score is None or champion_dd is None:
                comparison_errors.append("champion lacks comparable robust evidence")
            else:
                if robust_score < float(champion_score) * 1.15:
                    comparison_errors.append("challenger robust score improvement < 15%")
                if abs(float(adverse_dd)) > abs(float(champion_dd)) + 3.0:
                    comparison_errors.append("challenger adverse DD degrades by > 3 points")
            if comparison_errors:
                print(json.dumps({
                    "promoted": False, "errors": comparison_errors,
                }, indent=2))
                return 2

        rows = load_wfo_rows(
            args.db, certification["strategy_name"], certification["manifest_hash"],
        )
        plans = build_external_window_plans(rows)
        if not plans:
            print(json.dumps({
                "promoted": False, "errors": ["missing external WFO candidate"],
            }, indent=2))
            return 2
        latest_plan = plans[-1]
        artifact = {
            "certification_id": args.certification_id,
            "strategy_name": certification["strategy_name"],
            "snapshot_id": certification["snapshot_id"],
            "manifest_hash": certification["manifest_hash"],
            "effective_from": latest_plan.start.isoformat(),
            "universe": sorted(latest_plan.params_by_asset),
            "per_asset": latest_plan.params_by_asset,
            "robust_score": robust_score,
            "adverse_drawdown_pct": adverse_dd,
            "robot2_changed": False,
        }
        artifact_dir = Path("data/promotions")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{args.certification_id}.json"
        artifact_path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        now = datetime.now(tz=timezone.utc).isoformat()
        conn.execute(
            """UPDATE strategy_certifications
               SET promoted_at=?, updated_at=?, promotion_artifact=?,
                   robust_score=?, adverse_drawdown_pct=? WHERE id=?""",
            (
                now, now, str(artifact_path), robust_score, adverse_dd,
                args.certification_id,
            ),
        )
        conn.execute(
            """UPDATE portfolio_backtests
               SET result_status='LIVE_APPROVED', certification_id=?
               WHERE manifest_hash=? AND strategy_name=?""",
            (
                args.certification_id, certification["manifest_hash"],
                certification["strategy_name"],
            ),
        )
        conn.execute(
            """UPDATE optimization_results
               SET result_status='LIVE_APPROVED', certification_id=?
               WHERE manifest_hash=? AND strategy_name=?""",
            (
                args.certification_id, certification["manifest_hash"],
                certification["strategy_name"],
            ),
        )
        conn.commit()
        print(json.dumps({
            "promoted": True,
            "certification_id": args.certification_id,
            "promotion_artifact": str(artifact_path),
            "robot2_changed": False,
        }, indent=2))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote a LIVE_APPROVED certification")
    parser.add_argument("--certification-id", required=True)
    parser.add_argument("--db", default="data/scalp_radar.db")
    raise SystemExit(main(parser.parse_args()))
