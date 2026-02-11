"""Endpoint /health pour vérifier le statut du système."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health_check(request: Request) -> dict:
    """Retourne le statut du DataEngine, de la DB et l'uptime."""
    engine = request.app.state.engine
    start_time: datetime = request.app.state.start_time

    uptime = (datetime.now(tz=timezone.utc) - start_time).total_seconds()

    # Statut DataEngine
    if engine is None:
        engine_status = {
            "enabled": False,
            "connected": False,
            "last_update": None,
            "symbols": [],
        }
    else:
        engine_status = {
            "enabled": True,
            "connected": engine.is_connected,
            "last_update": (
                engine.last_update.isoformat() if engine.last_update else None
            ),
            "symbols": engine.get_all_symbols(),
        }

    # Statut DB
    db = request.app.state.db
    db_connected = db._conn is not None if db else False

    # Statut global
    if engine and not engine.is_connected:
        status = "degraded"
    elif not db_connected:
        status = "error"
    else:
        status = "ok"

    # Statut Watchdog
    watchdog = getattr(request.app.state, "watchdog", None)
    watchdog_status = watchdog.get_status() if watchdog else None

    # Statut Executor (Sprint 5a)
    executor = getattr(request.app.state, "executor", None)
    executor_status = executor.get_status() if executor else None

    return {
        "status": status,
        "data_engine": engine_status,
        "database": {"connected": db_connected},
        "watchdog": watchdog_status,
        "executor": executor_status,
        "uptime_seconds": int(uptime),
    }
