"""API endpoints pour le Simulator (paper trading)."""

from __future__ import annotations

from fastapi import APIRouter, Request, Query
from loguru import logger

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
        "kill_switch_reason": simulator.kill_switch_reason,
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
    """Trades récents (paginé) — lit depuis la DB."""
    db = getattr(request.app.state, "db", None)
    if db is not None:
        # Lire depuis la DB (source permanente)
        trades = await db.get_simulation_trades(limit=limit)
        return {"trades": trades}

    # Fallback : mémoire (backward compat si DB absente)
    simulator = getattr(request.app.state, "simulator", None)
    if simulator is None:
        return {"trades": []}

    all_trades = simulator.get_all_trades()
    return {"trades": all_trades[:limit]}


@router.get("/grid-state")
async def simulator_grid_state(request: Request) -> dict:
    """État détaillé des grilles DCA actives avec P&L non réalisé."""
    simulator = getattr(request.app.state, "simulator", None)
    if simulator is None:
        return {
            "grid_positions": {},
            "summary": {
                "total_positions": 0,
                "total_assets": 0,
                "total_margin_used": 0,
                "total_unrealized_pnl": 0,
                "capital_available": 0,
            },
        }
    return simulator.get_grid_state()


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


# TODO: ajouter auth quand exposé hors réseau local
@router.post("/kill-switch/reset")
async def reset_kill_switch(request: Request) -> dict:
    """Reset le kill switch global et réactive tous les runners."""
    simulator = getattr(request.app.state, "simulator", None)
    if simulator is None:
        return {"error": "Simulator non disponible", "status": "error"}

    if not simulator._global_kill_switch:
        return {"status": "not_triggered", "message": "Kill switch non actif"}

    reactivated = simulator.reset_kill_switch()

    # Sauvegarder l'état immédiatement
    state_manager = getattr(request.app.state, "state_manager", None)
    if state_manager:
        await state_manager.save_runner_state(
            simulator.runners,
            global_kill_switch=simulator._global_kill_switch,
            kill_switch_reason=simulator._kill_switch_reason,
        )

    # Notification Telegram
    notifier = getattr(request.app.state, "notifier", None)
    if notifier:
        try:
            from backend.alerts.notifier import AnomalyType
            await notifier.notify_anomaly(
                AnomalyType.KILL_SWITCH_GLOBAL,
                f"Kill switch RESET manuellement — {reactivated} runners réactivés",
            )
        except Exception as e:
            logger.error("Erreur notification reset kill switch: {}", e)

    return {
        "status": "reset",
        "runners_reactivated": reactivated,
    }
