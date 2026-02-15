"""API endpoints pour les conditions live et l'equity curve (Sprint 6)."""

from __future__ import annotations

from fastapi import APIRouter, Request, Query

from backend.core.config import get_config

router = APIRouter(tags=["conditions"])


@router.get("/api/simulator/conditions")
async def get_conditions(request: Request) -> dict:
    """Indicateurs courants par asset + conditions par stratégie.

    Retourne des données brutes structurées — le frontend formate.
    Cache côté Simulator, invalidé à chaque nouvelle bougie.
    """
    simulator = getattr(request.app.state, "simulator", None)
    if simulator is None:
        return {"assets": {}, "timestamp": None}

    return simulator.get_conditions()


@router.get("/api/signals/matrix")
async def get_signal_matrix(request: Request) -> dict:
    """Matrice simplifiée pour la Heatmap : dernier score par (stratégie, asset)."""
    simulator = getattr(request.app.state, "simulator", None)
    if simulator is None:
        return {"matrix": {}}

    return simulator.get_signal_matrix()


@router.get("/api/simulator/equity")
async def get_equity_curve(
    request: Request,
    since: str | None = Query(default=None, description="ISO8601 timestamp filter"),
) -> dict:
    """Courbe d'equity calculée depuis les trades.

    Priorité : mémoire (temps réel). Fallback : DB (robuste aux restarts).
    ?since= pour ne retourner que les nouveaux points (polling incrémental).
    """
    simulator = getattr(request.app.state, "simulator", None)
    if simulator is None:
        default_capital = get_config().risk.initial_capital
        return {"equity": [], "current_capital": default_capital, "initial_capital": default_capital}

    result = simulator.get_equity_curve(since=since)

    # Fallback DB si la mémoire est vide (après restart)
    if not result.get("equity"):
        db = getattr(request.app.state, "db", None)
        if db is not None:
            equity = await db.get_equity_curve_from_trades(since=since)
            if equity:
                result["equity"] = equity

    return result
