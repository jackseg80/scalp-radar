"""API endpoints pour l'Arena (classement des stratégies)."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/arena", tags=["arena"])


@router.get("/ranking")
async def arena_ranking(request: Request) -> dict:
    """Classement des stratégies par performance."""
    arena = getattr(request.app.state, "arena", None)
    if arena is None:
        return {"ranking": []}

    return {"ranking": await arena.get_reporting_ranking(
        getattr(request.app.state, "db", None),
    )}


@router.get("/strategy/{name}")
async def arena_strategy_detail(name: str, request: Request) -> dict:
    """Détail d'une stratégie spécifique."""
    arena = getattr(request.app.state, "arena", None)
    if arena is None:
        return {"error": "Arena non disponible"}

    detail = await arena.get_reporting_detail(name, getattr(request.app.state, "db", None))
    if detail is None:
        return {"error": f"Stratégie '{name}' non trouvée"}

    return detail
