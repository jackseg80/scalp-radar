"""Acceptance tests for the local-only champion/challenger promotion gate."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace

from backend.core.database import Database
from scripts import promote_strategy


def _seed_candidate(db_path: str, *, champion: bool = False) -> None:
    async def initialize() -> None:
        db = Database(db_path)
        await db.init()
        await db.close()

    asyncio.run(initialize())
    now = datetime.now(tz=timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO strategy_certifications
           (id, strategy_name, snapshot_id, status, created_at, updated_at,
            manifest_json, manifest_hash, gates_json, paper_metrics_json,
            canary_stage, portfolio_backtest_id)
           VALUES ('candidate', 'grid_atr', 'snap', 'LIVE_APPROVED', ?, ?,
                   '{}', 'manifest', '{}',
                   '{"forward_days":61,"completed_cycles":31,"entry_orders":101}',
                   4, 99)""",
        (now, now),
    )
    conn.execute(
        """INSERT INTO portfolio_robustness
           (backtest_id, label, created_at, external_oos_return_pct,
            adverse_max_drawdown_pct)
           VALUES (99, 'candidate', ?, 30, -10)""",
        (now,),
    )
    if champion:
        conn.execute(
            """INSERT INTO strategy_certifications
               (id, strategy_name, snapshot_id, status, created_at, updated_at,
                manifest_json, manifest_hash, gates_json, promoted_at,
                robust_score, adverse_drawdown_pct)
               VALUES ('champion', 'grid_atr', 'old', 'LIVE_APPROVED', ?, ?,
                       '{}', 'old-manifest', '{}', ?, 2.9, -8)""",
            (now, now, now),
        )
    conn.commit()
    conn.close()


def _args(db_path: str) -> argparse.Namespace:
    return argparse.Namespace(certification_id="candidate", db=db_path)


def _mock_candidate(monkeypatch) -> None:
    monkeypatch.setattr(promote_strategy, "validate_promotion_environment", lambda *a: [])
    monkeypatch.setattr(promote_strategy, "load_wfo_rows", lambda *a: [{}])
    monkeypatch.setattr(
        promote_strategy,
        "build_external_window_plans",
        lambda rows: [SimpleNamespace(
            start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            params_by_asset={"SOL/USDT": {"atr_period": 14}},
        )],
    )


def test_promotion_creates_local_artifact_without_touching_robot2(tmp_path, monkeypatch):
    db_path = str(tmp_path / "promotion.db")
    _seed_candidate(db_path)
    _mock_candidate(monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert promote_strategy.main(_args(db_path)) == 0
    artifact = tmp_path / "data" / "promotions" / "candidate.json"
    assert artifact.exists()
    assert '"robot2_changed": false' in artifact.read_text(encoding="utf-8")


def test_challenger_below_15_percent_improvement_is_refused(tmp_path, monkeypatch):
    db_path = str(tmp_path / "promotion.db")
    _seed_candidate(db_path, champion=True)
    _mock_candidate(monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert promote_strategy.main(_args(db_path)) == 2
    assert not (tmp_path / "data" / "promotions" / "candidate.json").exists()
