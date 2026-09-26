"""Small synthetic fixtures only: no historical database or candidate runs."""
from datetime import datetime, timedelta, timezone

import pytest

from backend.core.database import Database
from backend.core.experiment import (
    CONFIG_FILES, _common_hour_start, create_snapshot, revalidate_snapshot,
    wfo_reuse_fingerprint,
)
from backend.core.funding_research import ResearchFundingSpec
from backend.core.models import Candle, ExecutionSpec, UniverseSelectionSpec
from backend.optimization.walk_forward import (
    _windows_with_complete_coverage, build_aligned_wfo_windows,
)
from scripts.optimize import _snapshot_bounds

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize("minute,expected", [(0, 0), (1, 1), (59, 1), (60, 1)])
def test_rounding_to_common_full_hour(minute, expected):
    assert _common_hour_start({"signal": BASE.isoformat(),
                               "execution": (BASE + timedelta(minutes=minute)).isoformat()}) == BASE + timedelta(hours=expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", [None, "gap", "missing", "tail", "ohlc", "boundary"])
async def test_common_snapshot_clips_only_prefix_and_revalidates(tmp_path, monkeypatch, damage):
    cfg = tmp_path / "config"
    cfg.mkdir()
    for name in CONFIG_FILES:
        (cfg / name).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("backend.core.experiment.git_provenance", lambda root: ("commit", [], "hash"))
    path = str(tmp_path / "research.db")
    db = Database(path)
    await db.init()
    candles = [Candle(timestamp=BASE + timedelta(hours=i), open=100, high=101,
                     low=99, close=100, volume=10, symbol="BTC/USDT",
                     exchange="binance", timeframe="1h") for i in range(3)]
    omitted = {"gap": 90, "tail": 179, "boundary": 60}.get(damage)
    if damage != "missing":
        candles += [Candle(timestamp=BASE + timedelta(minutes=i), open=100, high=101,
                          low=99, close=100, volume=10, symbol="BTC/USDT",
                          exchange="bitget", timeframe="1m")
                    for i in range(30, 180) if i != omitted]
    await db.insert_candles_batch(candles)
    if damage == "ohlc":
        await db._conn.execute("UPDATE candles SET high=90 WHERE exchange='bitget'")
        await db._conn.commit()
    await db.close()
    kwargs = dict(db_path=path, series=[("binance", "BTC/USDT", "1h"), ("bitget", "BTC/USDT", "1m")],
                  cutoff=BASE + timedelta(hours=3), start=BASE, config_dir=cfg,
                  repo_root=tmp_path, execution_spec=ExecutionSpec(research_funding=ResearchFundingSpec()),
                  universe_selection=UniverseSelectionSpec(strategy_name="grid_boltrend",
                      universe_symbols=["BTC/USDT"], calendar_start=BASE))
    ident, manifest = await create_snapshot(**kwargs, research_common_availability=True)
    assert (manifest["validation_status"] == "VALID") == (damage is None)
    _, errors = await revalidate_snapshot(path, ident, config_dir=cfg, repo_root=tmp_path)
    assert (not errors) == (damage is None)
    if damage is not None:
        return
    entry = manifest["metadata"]["research_availability"]["assets"]["BTC/USDT"]
    assert entry["observed_starts"]["bitget:BTC/USDT:1m"] == (BASE + timedelta(minutes=30)).isoformat()
    assert entry["available_from"] == (BASE + timedelta(hours=1)).isoformat()
    assert _snapshot_bounds(manifest, "binance", "BTC/USDT")["1h"][0] == BASE + timedelta(hours=1)
    assert sorted(s["row_count"] for s in manifest["metadata"]["series"]) == [2, 120]
    # No destructive cleanup of excluded data, and no opt-in => original strict gate.
    _, strict = await create_snapshot(**kwargs)
    assert strict["validation_status"] == "INVALID"
    assert any("after signal coverage" in e for e in strict["validation_errors"])
    assert wfo_reuse_fingerprint(strict) != wfo_reuse_fingerprint(manifest)
    await db.init()
    count = await (await db._conn.execute("SELECT COUNT(*) AS n FROM candles")).fetchone()
    assert count["n"] == 153
    await db._conn.execute("DELETE FROM candles WHERE exchange='bitget' AND timestamp=?",
                           ((BASE + timedelta(minutes=90)).isoformat(),))
    await db._conn.commit()
    await db.close()
    _, errors = await revalidate_snapshot(path, ident, config_dir=cfg, repo_root=tmp_path)
    assert any("hash changed" in e for e in errors)


@pytest.mark.asyncio
@pytest.mark.parametrize("synthetic,start,gaps", [(False, BASE, 0), (True, None, 0), (True, BASE, 1)])
async def test_policy_rejected_before_database_access(synthetic, start, gaps):
    with pytest.raises(ValueError, match="Common availability requires"):
        await create_snapshot(db_path="must-not-open.db", series=[], cutoff=BASE,
                              config_dir=None, repo_root=None, start=start, max_gap_bars=gaps,
                              execution_spec=ExecutionSpec(research_funding=ResearchFundingSpec() if synthetic else None),
                              research_common_availability=True)


@pytest.mark.parametrize("available,first_oos", [
    ("2022-06-13T04:00:00+00:00", "2023-01-03T00:00:00+00:00"),
    ("2023-03-02T09:00:00+00:00", "2023-10-30T00:00:00+00:00"),
    ("2023-05-04T10:00:00+00:00", "2023-12-29T00:00:00+00:00"),
])
def test_unchanged_global_calendar_admission(available, first_oos):
    selection = UniverseSelectionSpec(strategy_name="grid_boltrend", universe_symbols=["BTC/USDT"],
                                      calendar_start=datetime(2022, 1, 1, tzinfo=timezone.utc),
                                      is_window_days=180, embargo_days=7, oos_window_days=60, step_days=60)
    windows = build_aligned_wfo_windows(selection, datetime(2026, 7, 27, tzinfo=timezone.utc))
    start = datetime.fromisoformat(available)
    # Only timestamps are required by the existing coverage filter.
    from types import SimpleNamespace
    hours = int((windows[-1][3] - start).total_seconds() / 3600)
    candles = [SimpleNamespace(timestamp=start + timedelta(hours=i)) for i in range(hours)]
    admitted = _windows_with_complete_coverage(windows, candles, "1h")
    assert admitted[0][2] == datetime.fromisoformat(first_oos)
    assert admitted[0][0] >= start
    assert admitted[0][2] - timedelta(hours=500) >= start
