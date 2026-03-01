"""API endpoints pour le Regime Monitor — Sprint 61."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

router = APIRouter(prefix="/api/regime", tags=["regime"])


@router.get("/snapshot")
async def get_regime_snapshot(request: Request):
    """Retourne le dernier RegimeSnapshot."""
    monitor = getattr(request.app.state, "regime_monitor", None)
    if monitor is None or monitor.latest is None:
        return {"snapshot": None, "message": "Regime monitor non disponible"}
    return {"snapshot": monitor._snapshot_to_dict(monitor.latest)}


@router.get("/history")
async def get_regime_history(
    request: Request,
    days: int = Query(default=30, ge=1, le=90),
):
    """Retourne l'historique des snapshots (max 30 en mémoire)."""
    monitor = getattr(request.app.state, "regime_monitor", None)
    if monitor is None:
        return {"history": [], "count": 0}
    history = monitor.history[-days:]
    return {"history": history, "count": len(history)}
