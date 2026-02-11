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

    ranking = arena.get_ranking()
    return {
        "ranking": [
            {
                "name": p.name,
                "capital": p.capital,
                "net_pnl": p.net_pnl,
                "net_return_pct": p.net_return_pct,
                "total_trades": p.total_trades,
                "win_rate": p.win_rate,
                "profit_factor": p.profit_factor,
                "max_drawdown_pct": p.max_drawdown_pct,
                "is_active": p.is_active,
            }
            for p in ranking
        ]
    }


@router.get("/strategy/{name}")
async def arena_strategy_detail(name: str, request: Request) -> dict:
    """Détail d'une stratégie spécifique."""
    arena = getattr(request.app.state, "arena", None)
    if arena is None:
        return {"error": "Arena non disponible"}

    detail = arena.get_strategy_detail(name)
    if detail is None:
        return {"error": f"Stratégie '{name}' non trouvée"}

    return detail
