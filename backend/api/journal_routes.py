"""Routes API pour le journal d'activité live (Sprint 25)."""

from __future__ import annotations

import json

from fastapi import APIRouter, Query, Request

router = APIRouter(prefix="/api/journal", tags=["journal"])


def _parse_json_field(row: dict, json_field: str, output_field: str) -> None:
    """Parse un champ JSON et le remplace dans le dict."""
    raw = row.get(json_field)
    if raw:
        try:
            row[output_field] = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            row[output_field] = None
    else:
        row[output_field] = None
    row.pop(json_field, None)


@router.get("/snapshots")
async def get_snapshots(
    request: Request,
    since: str | None = Query(None, description="ISO timestamp filter"),
    until: str | None = Query(None, description="ISO timestamp filter"),
    limit: int = Query(2000, ge=1, le=10000),
) -> dict:
    """Snapshots du portfolio pour courbe d'equity live."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return {"snapshots": [], "count": 0}

    snapshots = await db.get_portfolio_snapshots(
        since=since, until=until, limit=limit,
    )

    for s in snapshots:
        _parse_json_field(s, "breakdown_json", "breakdown")

    return {"snapshots": snapshots, "count": len(snapshots)}


@router.get("/events")
async def get_events(
    request: Request,
    since: str | None = Query(None),
    strategy: str | None = Query(None),
    symbol: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
) -> dict:
    """Événements de position (ouvertures, fermetures, DCA)."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return {"events": [], "count": 0}

    events = await db.get_position_events(
        since=since, limit=limit, strategy=strategy, symbol=symbol,
    )

    for e in events:
        _parse_json_field(e, "metadata_json", "metadata")

    return {"events": events, "count": len(events)}


@router.get("/summary")
async def get_summary(request: Request) -> dict:
    """Résumé rapide : dernier snapshot + derniers events."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return {"latest_snapshot": None, "recent_events": [], "total_events": 0}

    # Dernier snapshot (ORDER BY DESC LIMIT 1)
    latest = await db.get_latest_snapshot()
    if latest:
        _parse_json_field(latest, "breakdown_json", "breakdown")

    # 10 derniers events
    events = await db.get_position_events(limit=10)
    for e in events:
        _parse_json_field(e, "metadata_json", "metadata")

    return {
        "latest_snapshot": latest,
        "recent_events": events,
        "total_events": len(events),
    }


@router.get("/stats")
async def get_stats(
    request: Request,
    period: str = Query("all", description="today, 7d, 30d, all"),
) -> dict:
    """Stats agrégées du journal de trading (Sprint 32)."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return {"stats": None}

    if period not in {"today", "7d", "30d", "all"}:
        period = "all"

    stats = await db.get_journal_stats(period=period)
    return {"stats": stats}
