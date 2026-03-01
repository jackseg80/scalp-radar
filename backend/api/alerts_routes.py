"""Routes API pour l'historique des alertes Telegram (Sprint 63b)."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("/telegram")
async def get_telegram_alerts(
    request: Request,
    alert_type: str | None = Query(None),
    strategy: str | None = Query(None),
    since: str | None = Query(None, description="ISO8601 timestamp"),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    """Historique des alertes Telegram envoyées."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return {"alerts": [], "count": 0}

    alerts = await db.get_telegram_alerts(
        alert_type=alert_type,
        strategy=strategy,
        since=since,
        limit=limit,
    )
    return {"alerts": alerts, "count": len(alerts)}
