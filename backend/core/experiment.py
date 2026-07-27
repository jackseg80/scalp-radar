"""Reproducible data snapshots and experiment provenance.

This module extends the existing SQLite candle store.  A snapshot is a
manifest of exact rows and hashes; it does not copy market data into a second
store.  Certification consumers must always revalidate the hashes before use.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from backend.core.database import Database
from backend.core.models import (
    ExecutionSpec,
    ExperimentManifest,
    TimeFrame,
    UniverseSelectionSpec,
)


CONFIG_FILES = (
    "assets.yaml",
    "strategies.yaml",
    "risk.yaml",
    "exchanges.yaml",
    "param_grids.yaml",
)


def canonical_json(value: Any) -> str:
    """Serialize provenance deterministically across repeated runs."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def snapshot_manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return only the immutable payload signed in ``data_snapshots``.

    Loading and revalidation attach database status fields for callers. They
    are not part of the signed payload and must not alter downstream hashes.
    """
    return {
        key: value for key, value in manifest.items()
        if key not in {"manifest_hash", "validation_status", "validation_errors"}
    }


def snapshot_manifest_hash(manifest: dict[str, Any]) -> str:
    """Return the authoritative hash, or derive it from an unsigned payload."""
    if manifest.get("manifest_hash"):
        return str(manifest["manifest_hash"])
    return sha256_text(canonical_json(snapshot_manifest_payload(manifest)))


def wfo_reuse_fingerprint(manifest: dict[str, Any]) -> str:
    """Hash precisely the inputs that can affect a fast WFO decision.

    A portfolio/execution implementation change must require a fresh canonical
    replay, but it must not force millions of unchanged fast WFO simulations.
    This deliberately excludes snapshot identity, creation time and execution
    calibration while retaining every series/config/selection input used by
    WFO. Reuse remains explicit at the CLI and is rejected on any mismatch.
    """
    metadata = manifest.get("metadata") or {}
    payload = {
        "strategy_name": manifest.get("strategy_name"),
        "cutoff": manifest.get("cutoff"),
        "config_hashes": manifest.get("config_hashes"),
        "data_hashes": manifest.get("data_hashes"),
        "params_hash": manifest.get("params_hash"),
        "seed": manifest.get("seed"),
        "execution_model_version": manifest.get("execution_model_version"),
        "metadata": {
            key: metadata.get(key)
            for key in (
                "start", "series", "max_gap_bars", "special_data",
                "universe_selection",
            )
        },
    }
    return sha256_text(canonical_json(payload))


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hashes(config_dir: Path) -> dict[str, str]:
    """Hash every authoritative YAML config, failing on a missing file."""
    return {name: hash_file(config_dir / name) for name in CONFIG_FILES}


def _git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=True,
    )
    return completed.stdout.strip()


def git_provenance(repo_root: Path) -> tuple[str, list[str], str]:
    """Return commit, changed paths and a hash of all captured changes."""
    commit = _git_output(repo_root, "rev-parse", "HEAD")
    status = [line for line in _git_output(repo_root, "status", "--porcelain=v1").splitlines() if line]
    diff = _git_output(repo_root, "diff", "--binary", "HEAD")
    untracked: list[tuple[str, str]] = []
    for line in status:
        if not line.startswith("?? "):
            continue
        relative = line[3:].strip().strip('"')
        path = repo_root / relative
        if path.is_file():
            untracked.append((relative.replace("\\", "/"), hash_file(path)))
    captured = canonical_json({"diff": diff, "untracked": sorted(untracked)})
    return commit, status, sha256_text(captured)


def _parse_timestamp(raw: str) -> datetime:
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class SeriesValidation:
    key: str
    row_count: int
    first_timestamp: str | None
    last_timestamp: str | None
    duplicate_count: int
    divergent_duplicate_count: int
    invalid_ohlc_count: int
    irregular_timestamp_count: int
    missing_bars: int
    max_gap_bars: int
    series_hash: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def validate_candle_rows(
    rows: Iterable[Any],
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> SeriesValidation:
    """Validate and hash one already-cut-off candle series."""
    interval = timedelta(milliseconds=TimeFrame.from_string(timeframe).to_milliseconds())
    interval_seconds = interval.total_seconds()
    normalized: list[tuple[Any, ...]] = []
    timestamps: list[datetime] = []
    seen: dict[str, tuple[float, ...]] = {}
    duplicates = 0
    divergent_duplicates = 0
    invalid_ohlc = 0

    for row in rows:
        timestamp_raw = str(row["timestamp"])
        timestamp = _parse_timestamp(timestamp_raw)
        values = tuple(float(row[name]) for name in ("open", "high", "low", "close", "volume"))
        open_, high, low, close, volume = values
        if (
            min(open_, high, low, close) <= 0
            or volume < 0
            or high < max(open_, close, low)
            or low > min(open_, close, high)
        ):
            invalid_ohlc += 1
        previous = seen.get(timestamp.isoformat())
        if previous is not None:
            duplicates += 1
            if previous != values:
                divergent_duplicates += 1
        else:
            seen[timestamp.isoformat()] = values
        timestamps.append(timestamp)
        normalized.append((exchange, symbol, timeframe, timestamp.isoformat(), *values))

    ordered = sorted(zip(timestamps, normalized), key=lambda item: item[0])
    irregular = 0
    missing = 0
    max_gap = 0
    for index in range(1, len(ordered)):
        delta = (ordered[index][0] - ordered[index - 1][0]).total_seconds()
        if delta <= 0 or delta % interval_seconds != 0:
            irregular += 1
            continue
        gap = max(0, int(delta // interval_seconds) - 1)
        missing += gap
        max_gap = max(max_gap, gap)

    series_json = canonical_json([item[1] for item in ordered])
    return SeriesValidation(
        key=f"{exchange}:{symbol}:{timeframe}",
        row_count=len(ordered),
        first_timestamp=ordered[0][0].isoformat() if ordered else None,
        last_timestamp=ordered[-1][0].isoformat() if ordered else None,
        duplicate_count=duplicates,
        divergent_duplicate_count=divergent_duplicates,
        invalid_ohlc_count=invalid_ohlc,
        irregular_timestamp_count=irregular,
        missing_bars=missing,
        max_gap_bars=max_gap,
        series_hash=sha256_text(series_json),
    )


async def create_snapshot(
    *,
    db_path: str,
    series: list[tuple[str, str, str]],
    cutoff: datetime,
    config_dir: Path,
    repo_root: Path,
    seed: int = 0,
    max_gap_bars: int = 0,
    require_execution_timeframe: bool = True,
    execution_spec: ExecutionSpec | None = None,
    start: datetime | None = None,
    universe_selection: UniverseSelectionSpec | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create or reuse an immutable snapshot manifest.

    ``series`` contains explicit ``(exchange, symbol, timeframe)`` tuples, so
    an unavailable source can never silently fall back to another exchange.
    """
    cutoff = cutoff.astimezone(timezone.utc)
    execution_spec = execution_spec or ExecutionSpec(random_seed=seed)
    if universe_selection is not None:
        expected_symbols = sorted({symbol for _, symbol, _ in series})
        if universe_selection.universe_symbols != expected_symbols:
            raise ValueError(
                "universe selection symbols must match the snapshot candle series"
            )
        if universe_selection.calendar_start.tzinfo is None:
            raise ValueError("universe selection calendar_start must be timezone-aware")
        if universe_selection.calendar_start >= cutoff:
            raise ValueError("universe selection calendar_start must precede cutoff")
    db = Database(db_path)
    await db.init()
    assert db._conn is not None
    validations: list[SeriesValidation] = []
    errors: list[str] = []

    try:
        for exchange, symbol, timeframe in sorted(set(series)):
            tf = TimeFrame.from_string(timeframe)
            latest_open = cutoff - timedelta(milliseconds=tf.to_milliseconds())
            query = (
                "SELECT timestamp, open, high, low, close, volume FROM candles "
                "WHERE exchange=? AND symbol=? AND timeframe=? AND timestamp<=?"
            )
            params: list[Any] = [exchange, symbol, timeframe, latest_open.isoformat()]
            if start is not None:
                query += " AND timestamp>=?"
                params.append(start.astimezone(timezone.utc).isoformat())
            query += " ORDER BY timestamp ASC"
            rows = await (await db._conn.execute(query, params)).fetchall()
            validation = validate_candle_rows(
                rows, exchange=exchange, symbol=symbol, timeframe=timeframe,
            )
            validations.append(validation)
            if validation.row_count == 0:
                errors.append(f"{validation.key}: no closed candles")
            if validation.divergent_duplicate_count:
                errors.append(f"{validation.key}: divergent duplicates")
            if validation.invalid_ohlc_count:
                errors.append(f"{validation.key}: invalid OHLC rows")
            if validation.irregular_timestamp_count:
                errors.append(f"{validation.key}: irregular timestamps")
            if validation.max_gap_bars > max_gap_bars:
                errors.append(
                    f"{validation.key}: max gap {validation.max_gap_bars} > {max_gap_bars} bars"
                )

        execution_tf = execution_spec.execution_timeframe.value
        symbols = {symbol for _, symbol, _ in series}
        intrabar_missing = sorted(
            symbol
            for symbol in symbols
            if not any(
                validation.row_count > 0
                and validation.key.endswith(f":{symbol}:{execution_tf}")
                for validation in validations
            )
        )
        if require_execution_timeframe and intrabar_missing:
            errors.append(
                f"execution timeframe {execution_tf} missing for: {', '.join(intrabar_missing)}"
            )

        special_data: dict[str, dict[str, Any]] = {}
        cutoff_ms = int(cutoff.timestamp() * 1000)
        start_ms = int(start.timestamp() * 1000) if start is not None else None
        for exchange, symbol in sorted({(item[0], item[1]) for item in series}):
            funding_query = (
                "SELECT timestamp, funding_rate FROM funding_rates "
                "WHERE exchange=? AND symbol=? AND timestamp<=?"
            )
            funding_params: list[Any] = [exchange, symbol, cutoff_ms]
            if start_ms is not None:
                funding_query += " AND timestamp>=?"
                funding_params.append(start_ms)
            funding_query += " ORDER BY timestamp ASC"
            funding_rows = await (
                await db._conn.execute(funding_query, funding_params)
            ).fetchall()
            funding_values = [
                (int(row["timestamp"]), float(row["funding_rate"]))
                for row in funding_rows
            ]
            funding_key = f"{exchange}:{symbol}:funding"
            funding_hash = sha256_text(canonical_json(funding_values))
            special_data[funding_key] = {
                "row_count": len(funding_values), "hash": funding_hash,
            }

            oi_query = (
                "SELECT timeframe, timestamp, oi, oi_value FROM open_interest "
                "WHERE exchange=? AND symbol=? AND timestamp<=?"
            )
            oi_params: list[Any] = [exchange, symbol, cutoff_ms]
            if start_ms is not None:
                oi_query += " AND timestamp>=?"
                oi_params.append(start_ms)
            oi_query += " ORDER BY timeframe, timestamp ASC"
            oi_rows = await (await db._conn.execute(oi_query, oi_params)).fetchall()
            oi_values = [
                (
                    str(row["timeframe"]), int(row["timestamp"]),
                    float(row["oi"]), float(row["oi_value"]),
                )
                for row in oi_rows
            ]
            oi_key = f"{exchange}:{symbol}:open_interest"
            oi_hash = sha256_text(canonical_json(oi_values))
            special_data[oi_key] = {
                "row_count": len(oi_values), "hash": oi_hash,
            }

        commit, dirty_worktree, worktree_diff_hash = git_provenance(repo_root)
        if require_execution_timeframe and dirty_worktree:
            errors.append(
                "certification snapshot requires a clean worktree; "
                "commit or remove every captured change first"
            )
        cfg_hashes = config_hashes(config_dir)
        data_hashes = {v.key: v.series_hash for v in validations}
        data_hashes.update({key: value["hash"] for key, value in special_data.items()})
        created_at = datetime.now(tz=timezone.utc)
        content = {
            "cutoff": cutoff.isoformat(),
            "series": [v.as_dict() for v in validations],
            "config_hashes": cfg_hashes,
            "data_hashes": data_hashes,
            "git_commit": commit,
            "dirty_worktree": dirty_worktree,
            "worktree_diff_hash": worktree_diff_hash,
            "seed": seed,
            "execution_spec": execution_spec.model_dump(mode="json"),
            "max_gap_bars": max_gap_bars,
            "special_data": special_data,
            "universe_selection": (
                universe_selection.model_dump(mode="json")
                if universe_selection is not None else None
            ),
        }
        content_hash = sha256_text(canonical_json(content))
        snapshot_id = f"snapshot-{content_hash[:16]}"
        manifest_model = ExperimentManifest(
            snapshot_id=snapshot_id,
            cutoff=cutoff,
            git_commit=commit,
            config_hashes=cfg_hashes,
            data_hashes=data_hashes,
            seed=seed,
            execution_model_version=execution_spec.model_version,
            created_at=created_at,
            dirty_worktree=dirty_worktree,
            metadata={
                "series": [v.as_dict() for v in validations],
                "start": start.astimezone(timezone.utc).isoformat() if start else None,
                "worktree_diff_hash": worktree_diff_hash,
                "execution_spec": execution_spec.model_dump(mode="json"),
                "max_gap_bars": max_gap_bars,
                "intrabar_missing": intrabar_missing,
                "special_data": special_data,
                "universe_selection": (
                    universe_selection.model_dump(mode="json")
                    if universe_selection is not None else None
                ),
            },
        )
        manifest = manifest_model.model_dump(mode="json")
        manifest_json = canonical_json(manifest)
        manifest_hash = sha256_text(manifest_json)
        status = "VALID" if not errors else "INVALID"

        existing = await (
            await db._conn.execute(
                """SELECT manifest_json, manifest_hash, validation_status,
                          validation_errors
                   FROM data_snapshots WHERE id=?""",
                (snapshot_id,),
            )
        ).fetchone()
        if existing is not None:
            stored = json.loads(existing["manifest_json"])
            stored["manifest_hash"] = existing["manifest_hash"]
            stored["validation_status"] = existing["validation_status"]
            stored["validation_errors"] = json.loads(existing["validation_errors"])
            return snapshot_id, stored
        await db._conn.execute(
            """INSERT INTO data_snapshots
               (id, created_at, cutoff, manifest_json, manifest_hash,
                validation_status, validation_errors)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                snapshot_id, created_at.isoformat(), cutoff.isoformat(),
                manifest_json, manifest_hash, status, canonical_json(errors),
            ),
        )
        await db._conn.commit()
        manifest["validation_status"] = status
        manifest["validation_errors"] = errors
        manifest["manifest_hash"] = manifest_hash
        return snapshot_id, manifest
    finally:
        await db.close()


async def load_snapshot(db_path: str, snapshot_id: str) -> dict[str, Any] | None:
    db = Database(db_path)
    await db.init()
    assert db._conn is not None
    try:
        row = await (
            await db._conn.execute(
                "SELECT * FROM data_snapshots WHERE id=?", (snapshot_id,),
            )
        ).fetchone()
        if row is None:
            return None
        result = json.loads(row["manifest_json"])
        result["manifest_hash"] = row["manifest_hash"]
        result["validation_status"] = row["validation_status"]
        result["validation_errors"] = json.loads(row["validation_errors"])
        return result
    finally:
        await db.close()


async def revalidate_snapshot(
    db_path: str,
    snapshot_id: str,
    *,
    config_dir: Path | None = None,
    repo_root: Path | None = None,
    check_environment: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """Re-hash immutable snapshot inputs immediately before consumption.

    The snapshot references rows in the existing market-data database. This
    check detects a repaired/overwritten candle, special-data mutation, config
    drift, code drift, or a corrupted manifest instead of silently replaying a
    different experiment under the old snapshot id.
    """
    db = Database(db_path)
    await db.init()
    assert db._conn is not None
    errors: list[str] = []
    try:
        row = await (
            await db._conn.execute(
                "SELECT * FROM data_snapshots WHERE id=?", (snapshot_id,),
            )
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown snapshot: {snapshot_id}")
        manifest = json.loads(row["manifest_json"])
        actual_manifest_hash = sha256_text(canonical_json(manifest))
        if actual_manifest_hash != row["manifest_hash"]:
            errors.append("manifest hash mismatch")
        if row["validation_status"] != "VALID":
            errors.append(f"snapshot status is {row['validation_status']}")

        metadata = manifest.get("metadata", {})
        for expected in metadata.get("series", []):
            try:
                exchange, symbol, timeframe = expected["key"].split(":", 2)
            except (KeyError, ValueError):
                errors.append("invalid series key in manifest")
                continue
            rows = await (
                await db._conn.execute(
                    """SELECT timestamp, open, high, low, close, volume
                       FROM candles
                       WHERE exchange=? AND symbol=? AND timeframe=?
                         AND timestamp>=? AND timestamp<=?
                       ORDER BY timestamp ASC""",
                    (
                        exchange, symbol, timeframe,
                        expected["first_timestamp"], expected["last_timestamp"],
                    ),
                )
            ).fetchall()
            actual = validate_candle_rows(
                rows, exchange=exchange, symbol=symbol, timeframe=timeframe,
            )
            if actual.series_hash != expected.get("series_hash"):
                errors.append(f"{expected['key']}: series hash changed")
            if actual.row_count != int(expected.get("row_count", -1)):
                errors.append(f"{expected['key']}: row count changed")
            if (
                actual.divergent_duplicate_count
                or actual.invalid_ohlc_count
                or actual.irregular_timestamp_count
                or actual.max_gap_bars > int(metadata.get("max_gap_bars", 0))
            ):
                errors.append(f"{expected['key']}: validation no longer passes")

        cutoff = _parse_timestamp(str(manifest["cutoff"]))
        cutoff_ms = int(cutoff.timestamp() * 1000)
        start_raw = metadata.get("start")
        start_ms = int(_parse_timestamp(start_raw).timestamp() * 1000) if start_raw else None
        for key, expected in metadata.get("special_data", {}).items():
            exchange, symbol, kind = key.split(":", 2)
            if kind == "funding":
                query = (
                    "SELECT timestamp, funding_rate FROM funding_rates "
                    "WHERE exchange=? AND symbol=? AND timestamp<=?"
                )
                params: list[Any] = [exchange, symbol, cutoff_ms]
                if start_ms is not None:
                    query += " AND timestamp>=?"
                    params.append(start_ms)
                query += " ORDER BY timestamp ASC"
                special_rows = await (await db._conn.execute(query, params)).fetchall()
                values = [
                    (int(item["timestamp"]), float(item["funding_rate"]))
                    for item in special_rows
                ]
            elif kind == "open_interest":
                query = (
                    "SELECT timeframe, timestamp, oi, oi_value FROM open_interest "
                    "WHERE exchange=? AND symbol=? AND timestamp<=?"
                )
                params = [exchange, symbol, cutoff_ms]
                if start_ms is not None:
                    query += " AND timestamp>=?"
                    params.append(start_ms)
                query += " ORDER BY timeframe, timestamp ASC"
                special_rows = await (await db._conn.execute(query, params)).fetchall()
                values = [
                    (
                        str(item["timeframe"]), int(item["timestamp"]),
                        float(item["oi"]), float(item["oi_value"]),
                    )
                    for item in special_rows
                ]
            else:
                errors.append(f"unsupported special data kind: {kind}")
                continue
            if sha256_text(canonical_json(values)) != expected.get("hash"):
                errors.append(f"{key}: data hash changed")
            if len(values) != int(expected.get("row_count", -1)):
                errors.append(f"{key}: row count changed")

        if check_environment:
            if config_dir is None or repo_root is None:
                raise ValueError(
                    "config_dir and repo_root are required when check_environment=True"
                )
            if config_hashes(config_dir) != manifest.get("config_hashes"):
                errors.append("configuration hashes differ from snapshot")
            commit, dirty, diff_hash = git_provenance(repo_root)
            if commit != manifest.get("git_commit"):
                errors.append("git commit differs from snapshot")
            if dirty != manifest.get("dirty_worktree"):
                errors.append("dirty worktree paths differ from snapshot")
            if diff_hash != metadata.get("worktree_diff_hash"):
                errors.append("worktree content differs from snapshot")

        manifest["manifest_hash"] = row["manifest_hash"]
        manifest["validation_status"] = row["validation_status"]
        manifest["validation_errors"] = json.loads(row["validation_errors"])
        return manifest, errors
    finally:
        await db.close()


def require_snapshot_execution_series(
    manifest: dict[str, Any],
    symbols: list[str] | set[str],
) -> None:
    """Reject a replay whose broker candles are outside immutable provenance."""
    metadata = manifest.get("metadata", {})
    execution_spec = ExecutionSpec.model_validate(
        metadata.get("execution_spec", {}),
    )
    timeframe = execution_spec.execution_timeframe.value
    exchange = execution_spec.exchange
    available = {
        entry.get("key")
        for entry in metadata.get("series", [])
        if int(entry.get("row_count", 0)) > 0
    }
    missing = sorted(
        symbol
        for symbol in symbols
        if f"{exchange}:{symbol}:{timeframe}" not in available
    )
    if missing:
        raise ValueError(
            "Snapshot does not freeze canonical broker series "
            f"{exchange} {timeframe} for: {', '.join(missing)}"
        )
