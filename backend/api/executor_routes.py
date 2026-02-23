"""Routes de l'Executor (Sprint 36b : Multi-Executor).

GET  /api/executor/status — statut agrégé ou par stratégie (?strategy=grid_atr)
POST /api/executor/refresh-balance — force refresh solde exchange
POST /api/executor/kill-switch/reset — reset le kill switch live (par stratégie)
GET  /api/executor/orders — historique des ordres mergé
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from loguru import logger

from backend.core.config import get_config

router = APIRouter(prefix="/api/executor", tags=["executor"])


async def verify_executor_key(x_api_key: str = Header(None, alias="X-API-Key")) -> None:
    """Vérifie l'API key pour les endpoints executor."""
    config = get_config()
    server_key = config.secrets.sync_api_key
    if not server_key:
        raise HTTPException(status_code=401, detail="API key non configurée")
    if not x_api_key or x_api_key != server_key:
        raise HTTPException(status_code=401, detail="API key invalide")


@router.get("/status", dependencies=[Depends(verify_executor_key)])
async def executor_status(
    request: Request,
    strategy: str | None = Query(default=None, description="Nom de stratégie pour statut individuel"),
) -> dict:
    """Statut agrégé ou par stratégie (?strategy=grid_atr)."""
    executor_mgr = getattr(request.app.state, "executor", None)
    if executor_mgr is None:
        return {"enabled": False, "message": "Executor non actif"}

    # Per-strategy status
    if strategy:
        ex = getattr(executor_mgr, "get", None)
        if ex is None:
            # Legacy single executor — pas de méthode get()
            return executor_mgr.get_status()
        single = executor_mgr.get(strategy)
        if single is None:
            raise HTTPException(404, f"Executor '{strategy}' non trouvé")
        return single.get_status()

    return executor_mgr.get_status()


@router.post("/refresh-balance", dependencies=[Depends(verify_executor_key)])
async def refresh_balance(request: Request) -> dict:
    """Force un refresh du solde exchange (tous les executors)."""
    executor_mgr = getattr(request.app.state, "executor", None)
    if executor_mgr is None:
        raise HTTPException(status_code=400, detail="Executor non actif")

    # Multi-executor : refresh tous les balances
    if hasattr(executor_mgr, "refresh_all_balances"):
        results = await executor_mgr.refresh_all_balances()
        total = sum(v for v in results.values() if v is not None)
        return {"status": "ok", "exchange_balance": total, "per_strategy": results}

    # Legacy single executor
    new_balance = await executor_mgr.refresh_balance()
    if new_balance is None:
        raise HTTPException(status_code=502, detail="Échec fetch balance exchange")
    return {"status": "ok", "exchange_balance": new_balance}


@router.post("/kill-switch/reset", dependencies=[Depends(verify_executor_key)])
async def reset_live_kill_switch(
    request: Request,
    strategy: str | None = Query(default=None, description="Stratégie ciblée (défaut: toutes)"),
) -> dict:
    """Reset le kill switch live et réactive le trading.

    Sprint 36b : supporte multi-executor — param ?strategy=grid_atr pour cibler.
    """
    executor_mgr = getattr(request.app.state, "executor", None)
    if executor_mgr is None:
        raise HTTPException(status_code=400, detail="Executor non actif")

    # Récupérer les risk_managers via executor_mgr
    risk_managers: dict = {}
    executors: dict = {}
    if hasattr(executor_mgr, "risk_managers"):
        risk_managers = executor_mgr.risk_managers
        executors = executor_mgr.executors
    else:
        # Legacy single executor
        rm = getattr(request.app.state, "risk_mgr", None)
        if rm is None:
            raise HTTPException(status_code=400, detail="RiskManager non disponible")
        risk_managers = {"_legacy": rm}
        executors = {"_legacy": executor_mgr}

    # Filtrer si stratégie spécifique
    if strategy:
        if strategy not in risk_managers:
            raise HTTPException(404, f"Executor '{strategy}' non trouvé")
        risk_managers = {strategy: risk_managers[strategy]}
        executors = {strategy: executors[strategy]}

    # Reset chaque risk_manager ciblé
    reset_results: list[dict] = []
    for name, rm in risk_managers.items():
        if not rm.is_kill_switch_triggered:
            reset_results.append({"strategy": name, "status": "not_triggered"})
            continue

        old_pnl = rm._session_pnl
        rm._kill_switch_triggered = False
        rm._session_pnl = 0.0

        # Sauvegarder l'état immédiatement
        state_manager = getattr(request.app.state, "state_manager", None)
        ex = executors.get(name)
        if state_manager and ex:
            strat_name = name if name != "_legacy" else None
            await state_manager.save_executor_state(ex, rm, strategy_name=strat_name)

        logger.info(
            "Kill switch live reset via API [{}] (session_pnl {:.2f} → 0.0)",
            name, old_pnl,
        )
        reset_results.append({
            "strategy": name,
            "status": "reset",
            "previous_session_pnl": old_pnl,
        })

    # Notification Telegram (une seule)
    reset_names = [r["strategy"] for r in reset_results if r["status"] == "reset"]
    if reset_names:
        notifier = getattr(request.app.state, "notifier", None)
        if notifier:
            from backend.alerts.notifier import AnomalyType

            await notifier.notify_anomaly(
                AnomalyType.KILL_SWITCH_LIVE,
                f"Kill switch LIVE reset manuellement ({', '.join(reset_names)}) "
                "— trading réactivé",
            )

    if not reset_names:
        return {"status": "not_triggered", "message": "Aucun kill switch actif"}

    return {
        "status": "reset",
        "message": f"Kill switch réinitialisé ({', '.join(reset_names)})",
        "details": reset_results,
    }


@router.get("/orders")
async def executor_orders(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    """Historique des ordres Bitget (read-only, sans auth — Sprint 32)."""
    executor_mgr = getattr(request.app.state, "executor", None)
    if executor_mgr is None:
        return {"orders": [], "count": 0}

    # Multi-executor : merge + tri
    if hasattr(executor_mgr, "get_all_order_history"):
        orders = executor_mgr.get_all_order_history(limit)
    else:
        orders = list(executor_mgr._order_history)[:limit]

    return {"orders": orders, "count": len(orders)}
