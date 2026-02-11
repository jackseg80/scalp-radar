"""API endpoints pour les signaux."""

from __future__ import annotations

from fastapi import APIRouter, Request, Query

router = APIRouter(prefix="/api/signals", tags=["signals"])


@router.get("/recent")
async def recent_signals(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> dict:
    """Derniers signaux émis par les stratégies."""
    simulator = getattr(request.app.state, "simulator", None)
    if simulator is None:
        return {"signals": []}

    # Les signaux récents sont les trades récents (un trade = un signal exécuté)
    all_trades = simulator.get_all_trades()
    return {"signals": all_trades[:limit]}
