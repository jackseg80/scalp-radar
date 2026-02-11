"""API endpoints pour le Simulator (paper trading)."""

from __future__ import annotations

from fastapi import APIRouter, Request, Query

router = APIRouter(prefix="/api/simulator", tags=["simulator"])


@router.get("/status")
async def simulator_status(request: Request) -> dict:
    """Statut du simulateur : running, stratégies actives, uptime."""
    simulator = getattr(request.app.state, "simulator", None)
    if simulator is None:
        return {"running": False, "strategies": {}}

    return {
        "running": simulator._running,
        "strategies": simulator.get_all_status(),
        "kill_switch_triggered": simulator.is_kill_switch_triggered(),
    }


@router.get("/positions")
async def simulator_positions(request: Request) -> dict:
    """Positions ouvertes par stratégie."""
    simulator = getattr(request.app.state, "simulator", None)
    if simulator is None:
        return {"positions": []}

    positions = []
    for runner in simulator.runners:
        pos = runner._position
        if pos is not None:
            positions.append({
                "strategy": runner.name,
                "direction": pos.direction.value,
                "entry_price": pos.entry_price,
                "quantity": pos.quantity,
                "entry_time": pos.entry_time.isoformat(),
                "tp_price": pos.tp_price,
                "sl_price": pos.sl_price,
            })

    return {"positions": positions}


@router.get("/trades")
async def simulator_trades(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
) -> dict:
    """Trades récents (paginé)."""
    simulator = getattr(request.app.state, "simulator", None)
    if simulator is None:
        return {"trades": []}

    all_trades = simulator.get_all_trades()
    return {"trades": all_trades[:limit]}


@router.get("/performance")
async def simulator_performance(request: Request) -> dict:
    """Métriques de performance par stratégie."""
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
