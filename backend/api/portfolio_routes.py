"""API endpoints pour le portfolio backtest — Sprint 20b-UI."""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel

from backend.api.websocket_routes import manager as ws_manager
from backend.backtesting.portfolio_db import (
    delete_backtest_async,
    get_backtest_by_id_async,
    get_backtests_async,
    save_result_async,
)
from backend.backtesting.portfolio_engine import PortfolioBacktester
from backend.core.config import get_config

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

# ─── Presets ─────────────────────────────────────────────────────────────

PORTFOLIO_PRESETS = [
    {
        "name": "conservative",
        "label": "Conservateur",
        "description": "Petit capital, top 5 assets Grade A",
        "capital": 1000,
        "days": 90,
        "assets": [
            "CRV/USDT", "FET/USDT", "ENJ/USDT", "AVAX/USDT", "DOGE/USDT",
        ],
    },
    {
        "name": "balanced",
        "label": "Équilibré",
        "description": "Capital moyen, top 10 Grade A+B",
        "capital": 5000,
        "days": 90,
        "assets": [
            "CRV/USDT", "ENJ/USDT", "FET/USDT", "XTZ/USDT", "AVAX/USDT",
            "ICP/USDT", "AR/USDT", "DYDX/USDT", "UNI/USDT", "DOGE/USDT",
        ],
    },
    {
        "name": "aggressive",
        "label": "Agressif",
        "description": "Capital max, diversification maximale",
        "capital": 10000,
        "days": 90,
        "assets": None,  # None = tous les assets per_asset
    },
    {
        "name": "longterm",
        "label": "Long terme",
        "description": "Backtest 1 an, top 7 Grade A",
        "capital": 10000,
        "days": 365,
        "assets": [
            "CRV/USDT", "ENJ/USDT", "FET/USDT", "XTZ/USDT", "AVAX/USDT",
            "ICP/USDT", "AR/USDT",
        ],
    },
]

# ─── Job tracker in-memory ───────────────────────────────────────────────

_current_job: dict | None = None


# ─── Pydantic models ────────────────────────────────────────────────────

class RunPortfolioRequest(BaseModel):
    strategy_name: str = "grid_atr"
    initial_capital: float = 10_000
    days: int = 90
    assets: list[str] | None = None
    exchange: str = "binance"
    kill_switch_pct: float = 30.0
    kill_switch_window: int = 24
    label: str | None = None


# ─── Helper ──────────────────────────────────────────────────────────────

def _get_db_path(request: Request) -> str:
    db = getattr(request.app.state, "db", None)
    if db and db.db_path:
        return str(db.db_path)
    return "data/scalp_radar.db"


# ─── Endpoints ───────────────────────────────────────────────────────────

@router.get("/presets")
async def get_presets():
    """Liste des presets portfolio."""
    return {"presets": PORTFOLIO_PRESETS}


@router.get("/backtests")
async def list_backtests(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
):
    """Liste des backtests sauvegardés (sans equity_curve)."""
    db_path = _get_db_path(request)
    results = await get_backtests_async(db_path, limit=limit)
    return {"backtests": results, "total": len(results)}


@router.get("/backtests/{backtest_id}")
async def get_backtest_detail(request: Request, backtest_id: int):
    """Détail complet d'un backtest (avec equity_curve)."""
    db_path = _get_db_path(request)
    result = await get_backtest_by_id_async(db_path, backtest_id)
    if result is None:
        raise HTTPException(404, f"Backtest {backtest_id} non trouvé")
    return result


@router.delete("/backtests/{backtest_id}")
async def delete_backtest(request: Request, backtest_id: int):
    """Supprime un backtest."""
    db_path = _get_db_path(request)
    deleted = await delete_backtest_async(db_path, backtest_id)
    if not deleted:
        raise HTTPException(404, f"Backtest {backtest_id} non trouvé")
    return {"status": "deleted", "id": backtest_id}


@router.get("/status")
async def get_status():
    """Status du backtest en cours."""
    if _current_job is None or _current_job["status"] != "running":
        return {"running": False}
    return {
        "running": True,
        "job_id": _current_job["id"],
        "progress_pct": _current_job["progress_pct"],
        "phase": _current_job["phase"],
    }


@router.post("/run")
async def run_portfolio_backtest(request: Request, body: RunPortfolioRequest):
    """Lance un backtest portfolio en background."""
    global _current_job

    if _current_job is not None and _current_job["status"] == "running":
        raise HTTPException(409, "Un backtest portfolio est déjà en cours")

    job_id = str(uuid.uuid4())[:8]
    _current_job = {
        "id": job_id,
        "status": "running",
        "progress_pct": 0.0,
        "phase": "Démarrage",
        "result_id": None,
    }

    db_path = _get_db_path(request)
    task = asyncio.create_task(_run_backtest(db_path, body, job_id))
    _current_job["task"] = task

    return {"job_id": job_id, "status": "running"}


@router.get("/compare")
async def compare_backtests(
    request: Request,
    ids: str = Query(..., description="IDs séparés par virgule (ex: 1,3)"),
):
    """Compare N backtests."""
    db_path = _get_db_path(request)
    try:
        id_list = [int(x.strip()) for x in ids.split(",")]
    except ValueError:
        raise HTTPException(400, "IDs invalides (format attendu: 1,3)")

    if len(id_list) < 2:
        raise HTTPException(400, "Au moins 2 IDs requis pour la comparaison")

    runs = []
    for bid in id_list:
        result = await get_backtest_by_id_async(db_path, bid)
        if result is None:
            raise HTTPException(404, f"Backtest {bid} non trouvé")
        runs.append(result)

    return {"runs": runs}


# ─── Background task ─────────────────────────────────────────────────────

async def _run_backtest(db_path: str, body: RunPortfolioRequest, job_id: str) -> None:
    """Exécute le backtest en background et sauvegarde en DB."""
    global _current_job

    t0 = time.monotonic()
    config = get_config()

    def progress_callback(pct: float, phase: str) -> None:
        if _current_job and _current_job["id"] == job_id:
            _current_job["progress_pct"] = pct
            _current_job["phase"] = phase
        # Broadcast async — on est déjà dans la boucle asyncio
        asyncio.ensure_future(ws_manager.broadcast({
            "type": "portfolio_progress",
            "job_id": job_id,
            "progress_pct": pct,
            "phase": phase,
        }))

    try:
        backtester = PortfolioBacktester(
            config=config,
            initial_capital=body.initial_capital,
            strategy_name=body.strategy_name,
            assets=body.assets,
            exchange=body.exchange,
            kill_switch_pct=body.kill_switch_pct,
            kill_switch_window_hours=body.kill_switch_window,
        )

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=body.days)

        result = await backtester.run(
            start, end, db_path=db_path, progress_callback=progress_callback,
        )

        duration = time.monotonic() - t0

        result_id = await save_result_async(
            db_path,
            result,
            strategy_name=body.strategy_name,
            exchange=body.exchange,
            kill_switch_pct=body.kill_switch_pct,
            kill_switch_window_hours=body.kill_switch_window,
            duration_seconds=round(duration, 1),
            label=body.label,
        )

        if _current_job and _current_job["id"] == job_id:
            _current_job["status"] = "completed"
            _current_job["result_id"] = result_id
            _current_job["progress_pct"] = 100.0

        await ws_manager.broadcast({
            "type": "portfolio_completed",
            "job_id": job_id,
            "result_id": result_id,
            "duration_seconds": round(duration, 1),
        })
        logger.info(
            "Portfolio backtest terminé (id={}, {:.0f}s)", result_id, duration
        )

    except Exception as e:
        logger.error("Portfolio backtest échoué : {}", e)
        if _current_job and _current_job["id"] == job_id:
            _current_job["status"] = "failed"
            _current_job["phase"] = str(e)[:200]

        await ws_manager.broadcast({
            "type": "portfolio_failed",
            "job_id": job_id,
            "error": str(e)[:200],
        })
