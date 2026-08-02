from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from backend.core.database import Database
from backend.core.experiment import (
    CONFIG_FILES,
    create_snapshot,
    funding_coverage_errors,
    require_snapshot_execution_series,
    revalidate_snapshot,
    validate_candle_rows,
    wfo_reuse_fingerprint,
)
from backend.core.models import (
    Candle,
    ExecutionSpec,
    TimeFrame,
    UniverseSelectionSpec,
)
from scripts.create_data_snapshot import _load_snapshot_config


def test_snapshot_explicit_config_directory_ignores_local_env(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_get_config(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr("scripts.create_data_snapshot.get_config", fake_get_config)

    _load_snapshot_config(str(tmp_path))

    assert captured["args"] == (tmp_path,)
    assert captured["kwargs"] == {"env_file": None, "force_reload": True}


def test_snapshot_replay_requires_frozen_execution_series():
    manifest = {
        "metadata": {
            "execution_spec": ExecutionSpec().model_dump(mode="json"),
            "series": [{
                "key": "binance:BTC/USDT:1h",
                "row_count": 100,
            }],
        },
    }
    with pytest.raises(ValueError, match="bitget 1m"):
        require_snapshot_execution_series(manifest, {"BTC/USDT"})

    manifest["metadata"]["series"].append({
        "key": "bitget:BTC/USDT:1m",
        "row_count": 1000,
    })
    require_snapshot_execution_series(manifest, {"BTC/USDT"})


def test_wfo_reuse_fingerprint_ignores_portfolio_only_provenance():
    """A fresh canonical replay may reuse WFO only when WFO inputs match."""
    manifest = {
        "snapshot_id": "old",
        "created_at": "2026-01-01T00:00:00Z",
        "strategy_name": "grid_atr",
        "cutoff": "2026-01-02T00:00:00Z",
        "config_hashes": {"risk.yaml": "risk"},
        "data_hashes": {"binance:BTC/USDT:1h": "candles"},
        "params_hash": "params",
        "seed": 0,
        "execution_model_version": "closed_bar_v3",
        "metadata": {
            "start": "2022-01-01T00:00:00Z",
            "series": [{"key": "binance:BTC/USDT:1h", "series_hash": "candles"}],
            "max_gap_bars": 1,
            "special_data": {},
            "universe_selection": {"top_n": 8},
            "worktree_diff_hash": "portfolio-fix-only",
            "execution_spec": {"scenario": "nominal"},
        },
    }
    refreshed = json.loads(json.dumps(manifest))
    refreshed["snapshot_id"] = "new"
    refreshed["created_at"] = "2026-01-02T00:00:00Z"
    refreshed["metadata"]["worktree_diff_hash"] = "new-portfolio-fix"
    refreshed["metadata"]["execution_spec"] = {"scenario": "adverse"}

    assert wfo_reuse_fingerprint(manifest) == wfo_reuse_fingerprint(refreshed)

    refreshed["data_hashes"]["binance:BTC/USDT:1h"] = "different"
    assert wfo_reuse_fingerprint(manifest) != wfo_reuse_fingerprint(refreshed)


def _row(timestamp: str, close: float = 100.0) -> dict:
    return {
        "timestamp": timestamp,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 10,
    }


def test_series_validation_detects_gaps_and_invalid_ohlc():
    rows = [
        _row("2026-01-01T00:00:00+00:00"),
        _row("2026-01-01T01:00:00+00:00"),
        _row("2026-01-01T03:00:00+00:00"),
    ]
    rows[-1]["high"] = 90.0
    validation = validate_candle_rows(
        rows, exchange="binance", symbol="BTC/USDT", timeframe="1h",
    )
    assert validation.missing_bars == 1
    assert validation.max_gap_bars == 1
    assert validation.invalid_ohlc_count == 1
    assert len(validation.series_hash) == 64


@pytest.mark.asyncio
async def test_snapshot_is_cutoff_safe_and_reproducible(tmp_path):
    db_path = str(tmp_path / "snapshot.db")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for name in CONFIG_FILES:
        (config_dir / name).write_text(f"name: {name}\n", encoding="utf-8")

    db = Database(db_path)
    await db.init()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [
        Candle(
            timestamp=base + timedelta(hours=index),
            open=100 + index,
            high=101 + index,
            low=99 + index,
            close=100.5 + index,
            volume=10,
            symbol="BTC/USDT",
            timeframe=TimeFrame.H1,
            exchange="binance",
        )
        for index in range(4)
    ]
    await db.insert_candles_batch(candles)
    await db.close()

    selection = UniverseSelectionSpec(
        universe_symbols=["BTC/USDT"],
        calendar_start=base,
        primary_leverage=4,
        leverage_scenarios=[6, 2, 4],
        portfolio_initial_capital=1646.0,
    )
    kwargs = {
        "db_path": db_path,
        "series": [("binance", "BTC/USDT", "1h")],
        "cutoff": base + timedelta(hours=3),
        "config_dir": config_dir,
        "repo_root": __import__("pathlib").Path.cwd(),
        "seed": 17,
        "max_gap_bars": 0,
        "require_execution_timeframe": False,
        "universe_selection": selection,
    }
    first_id, first = await create_snapshot(**kwargs)
    second_id, second = await create_snapshot(**kwargs)

    assert first_id == second_id
    assert first == second
    assert first["validation_status"] == "VALID"
    series = first["metadata"]["series"][0]
    # cutoff 03:00 includes only candles known closed at 03:00 (00, 01, 02).
    assert series["row_count"] == 3
    assert series["last_timestamp"] == "2026-01-01T02:00:00+00:00"
    assert first["metadata"]["special_data"]["binance:BTC/USDT:funding"]["row_count"] == 0
    assert "binance:BTC/USDT:open_interest" in first["data_hashes"]
    assert first["metadata"]["universe_selection"]["universe_symbols"] == ["BTC/USDT"]
    assert first["metadata"]["universe_selection"]["leverage_scenarios"] == [2, 4, 6]
    assert first["metadata"]["universe_selection"]["portfolio_initial_capital"] == 1646.0


def test_universe_selection_freezes_capital_and_leverage():
    selection = UniverseSelectionSpec(
        strategy_name="grid_multi_tf",
        universe_symbols=["BTC/USDT"],
        calendar_start=datetime(2022, 1, 1, tzinfo=timezone.utc),
        primary_leverage=3,
        leverage_scenarios=[2, 3, 4],
        portfolio_initial_capital=1646.0,
    )
    assert selection.resolve_portfolio_initial_capital(None) == 1646.0
    assert selection.resolve_portfolio_initial_capital(1646.0) == 1646.0
    with pytest.raises(ValueError, match="immutable snapshot"):
        selection.resolve_portfolio_initial_capital(1000.0)
    assert selection.primary_leverage == 3
    assert selection.leverage_scenarios == [2, 3, 4]


def test_legacy_universe_selection_can_still_be_read():
    selection = UniverseSelectionSpec(
        universe_symbols=["BTC/USDT"],
        calendar_start=datetime(2022, 1, 1, tzinfo=timezone.utc),
    )
    assert selection.portfolio_initial_capital is None
    assert selection.resolve_portfolio_initial_capital(None) == 1000.0


@pytest.mark.asyncio
async def test_snapshot_has_no_silent_exchange_fallback(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for name in CONFIG_FILES:
        (config_dir / name).write_text("{}\n", encoding="utf-8")
    db_path = str(tmp_path / "missing.db")

    _, manifest = await create_snapshot(
        db_path=db_path,
        series=[("bitget", "BTC/USDT", "1h")],
        cutoff=datetime(2026, 1, 1, tzinfo=timezone.utc),
        config_dir=config_dir,
        repo_root=__import__("pathlib").Path.cwd(),
        require_execution_timeframe=False,
    )
    assert manifest["validation_status"] == "INVALID"
    assert any("bitget:BTC/USDT:1h: no closed candles" in error for error in manifest["validation_errors"])


@pytest.mark.asyncio
async def test_certification_snapshot_rejects_stale_series_coverage(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for name in CONFIG_FILES:
        (config_dir / name).write_text("{}\n", encoding="utf-8")
    db_path = str(tmp_path / "stale.db")
    db = Database(db_path)
    await db.init()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    await db.insert_candles_batch([
        Candle(
            timestamp=base,
            open=100,
            high=101,
            low=99,
            close=100,
            volume=10,
            symbol="BTC/USDT",
            timeframe=TimeFrame.H1,
            exchange="binance",
        ),
    ])
    await db.close()

    _, manifest = await create_snapshot(
        db_path=db_path,
        series=[("binance", "BTC/USDT", "1h")],
        cutoff=base + timedelta(hours=2),
        config_dir=config_dir,
        repo_root=__import__("pathlib").Path.cwd(),
        require_execution_timeframe=False,
    )

    assert manifest["validation_status"] == "INVALID"
    assert any("coverage ends at" in error for error in manifest["validation_errors"])


@pytest.mark.asyncio
async def test_certification_snapshot_requires_execution_from_signal_start(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for name in CONFIG_FILES:
        (config_dir / name).write_text("{}\n", encoding="utf-8")
    db_path = str(tmp_path / "execution-coverage.db")
    db = Database(db_path)
    await db.init()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    signal = [
        Candle(
            timestamp=base + timedelta(hours=index),
            open=100,
            high=101,
            low=99,
            close=100,
            volume=10,
            symbol="BTC/USDT",
            timeframe=TimeFrame.H1,
            exchange="binance",
        )
        for index in range(2)
    ]
    execution = [
        Candle(
            timestamp=base + timedelta(minutes=index),
            open=100,
            high=101,
            low=99,
            close=100,
            volume=10,
            symbol="BTC/USDT",
            timeframe=TimeFrame.M1,
            exchange="bitget",
        )
        for index in range(1, 120)
    ]
    await db.insert_candles_batch(signal + execution)
    await db.close()

    _, manifest = await create_snapshot(
        db_path=db_path,
        series=[
            ("binance", "BTC/USDT", "1h"),
            ("bitget", "BTC/USDT", "1m"),
        ],
        cutoff=base + timedelta(hours=2),
        config_dir=config_dir,
        repo_root=__import__("pathlib").Path.cwd(),
        execution_spec=ExecutionSpec(exchange="bitget", execution_timeframe="1m"),
    )

    assert manifest["validation_status"] == "INVALID"
    assert any("after signal coverage begins" in error for error in manifest["validation_errors"])


def test_funding_coverage_rejects_recent_only_broker_sample():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    errors = funding_coverage_errors(
        [(int((base + timedelta(hours=25)).timestamp() * 1000), 0.01)],
        key="bitget:BTC/USDT:funding",
        coverage_start_ms=int(base.timestamp() * 1000),
        cutoff_ms=int((base + timedelta(hours=48)).timestamp() * 1000),
    )

    assert any("funding coverage begins" in error for error in errors)


@pytest.mark.asyncio
async def test_certification_snapshot_rejects_dirty_worktree(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for name in CONFIG_FILES:
        (config_dir / name).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "backend.core.experiment.git_provenance",
        lambda root: ("commit", [" M backend/code.py"], "diff-hash"),
    )
    _, manifest = await create_snapshot(
        db_path=str(tmp_path / "dirty.db"),
        series=[("bitget", "BTC/USDT", "1m")],
        cutoff=datetime(2026, 1, 1, tzinfo=timezone.utc),
        config_dir=config_dir,
        repo_root=tmp_path,
        require_execution_timeframe=True,
    )
    assert manifest["validation_status"] == "INVALID"
    assert any("clean worktree" in error for error in manifest["validation_errors"])


@pytest.mark.asyncio
async def test_snapshot_revalidation_detects_mutated_market_data(tmp_path):
    db_path = str(tmp_path / "snapshot.db")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for name in CONFIG_FILES:
        (config_dir / name).write_text(f"name: {name}\n", encoding="utf-8")

    db = Database(db_path)
    await db.init()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    await db.insert_candles_batch([
        Candle(
            timestamp=base,
            open=100,
            high=101,
            low=99,
            close=100,
            volume=10,
            symbol="BTC/USDT",
            timeframe=TimeFrame.H1,
            exchange="binance",
        ),
    ])
    await db.close()
    snapshot_id, _ = await create_snapshot(
        db_path=db_path,
        series=[("binance", "BTC/USDT", "1h")],
        cutoff=base + timedelta(hours=1),
        config_dir=config_dir,
        repo_root=__import__("pathlib").Path.cwd(),
        require_execution_timeframe=False,
    )
    _, errors = await revalidate_snapshot(
        db_path, snapshot_id, check_environment=False,
    )
    assert errors == []

    db = Database(db_path)
    await db.init()
    assert db._conn is not None
    await db._conn.execute(
        "UPDATE candles SET close=100.5 WHERE exchange='binance' AND symbol='BTC/USDT'",
    )
    await db._conn.commit()
    await db.close()
    _, errors = await revalidate_snapshot(
        db_path, snapshot_id, check_environment=False,
    )
    assert "binance:BTC/USDT:1h: series hash changed" in errors
