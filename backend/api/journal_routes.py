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


@router.get("/slippage")
async def get_slippage(request: Request) -> dict:
    """Rapport slippage paper vs live (Sprint Journal V2).

    Source : executor._order_history (deque max 200 en mémoire).
    Seuls les ordres avec average_price > 0 ET paper_price > 0 sont analysés.
    """
    executor = getattr(request.app.state, "executor", None)
    if executor is None:
        return {"slippage": None}

    order_history = getattr(executor, "_order_history", [])
    if not order_history:
        return {
            "slippage": {
                "orders_analyzed": 0,
                "avg_slippage_pct": 0.0,
                "total_slippage_cost": 0.0,
                "by_asset": {},
                "by_strategy": {},
                "by_type": {},
                "note": "No orders in memory.",
            },
        }

    # Filtrer les ordres exploitables
    valid_orders = []
    for o in order_history:
        avg = o.get("average_price", 0) or 0
        paper = o.get("paper_price", 0) or 0
        if avg > 0 and paper > 0:
            slip_pct = (avg - paper) / paper * 100
            valid_orders.append({
                "symbol": o.get("symbol", ""),
                "strategy": o.get("strategy", ""),
                "order_type": o.get("order_type", ""),
                "side": o.get("side", ""),
                "quantity": o.get("quantity", 0),
                "paper_price": paper,
                "average_price": avg,
                "slippage_pct": slip_pct,
                "slippage_cost": (avg - paper) * (o.get("quantity", 0) or 0),
            })

    if not valid_orders:
        return {
            "slippage": {
                "orders_analyzed": 0,
                "avg_slippage_pct": 0.0,
                "total_slippage_cost": 0.0,
                "by_asset": {},
                "by_strategy": {},
                "by_type": {},
                "note": f"0/{len(list(order_history))} orders have both real and paper prices.",
            },
        }

    # Agrégations
    avg_slip = sum(o["slippage_pct"] for o in valid_orders) / len(valid_orders)
    total_cost = sum(o["slippage_cost"] for o in valid_orders)

    by_asset: dict[str, dict] = {}
    by_strategy: dict[str, dict] = {}
    by_type: dict[str, dict] = {}

    for o in valid_orders:
        for key, bucket in [
            (o["symbol"], by_asset),
            (o["strategy"], by_strategy),
            (o["order_type"], by_type),
        ]:
            if key not in bucket:
                bucket[key] = {"sum_pct": 0.0, "count": 0}
            bucket[key]["sum_pct"] += o["slippage_pct"]
            bucket[key]["count"] += 1

    def _finalize(bucket: dict) -> dict:
        return {
            k: {"avg_pct": round(v["sum_pct"] / v["count"], 4), "count": v["count"]}
            for k, v in bucket.items()
        }

    return {
        "slippage": {
            "orders_analyzed": len(valid_orders),
            "avg_slippage_pct": round(avg_slip, 4),
            "total_slippage_cost": round(total_cost, 4),
            "by_asset": _finalize(by_asset),
            "by_strategy": _finalize(by_strategy),
            "by_type": _finalize(by_type),
            "note": f"Based on last {len(list(order_history))} orders in memory (max 200).",
        },
    }


@router.get("/per-asset")
async def get_per_asset_stats(
    request: Request,
    period: str = Query("all", description="today, 7d, 30d, all"),
) -> dict:
    """Performance par asset depuis simulation_trades (paper)."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return {"per_asset": []}

    if period not in {"today", "7d", "30d", "all"}:
        period = "all"

    per_asset = await db.get_journal_per_asset_stats(period=period)
    return {"per_asset": per_asset}
