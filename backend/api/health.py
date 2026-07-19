"""Endpoint /health pour vérifier le statut du système."""

from __future__ import annotations

import shutil
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
        health_snapshot = (
            engine.get_health_snapshot()
            if callable(getattr(engine, "get_health_snapshot", None))
            else {}
        )
        if not isinstance(health_snapshot, dict):
            health_snapshot = {}
        engine_status = {
            "enabled": True,
            "connected": engine.is_connected,
            "last_update": (
                engine.last_update.isoformat() if engine.last_update else None
            ),
            "symbols": engine.get_all_symbols(),
            **health_snapshot,
        }

    # Statut DB
    db = request.app.state.db
    db_connected = db._conn is not None if db else False

    # Statut composants (Sprint Audit-A)
    startup_components: dict = getattr(
        request.app.state, "startup_components", {},
    )
    failed_components = [
        k for k, v in startup_components.items()
        if isinstance(v, str) and v.startswith("error")
    ]

    # Statut global
    if failed_components:
        status = "degraded"
    elif engine and not engine.is_connected:
        status = "degraded"
    elif engine_status.get("last_flush_error"):
        status = "degraded"
    elif uptime > 300 and engine_status.get("stale_symbols"):
        status = "degraded"
    elif engine_status.get("abandoned_symbols"):
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

    # Statut disque (répertoire data/)
    try:
        disk_usage = shutil.disk_usage("data/")
        disk_info: dict | None = {
            "total_gb": round(disk_usage.total / (1024 ** 3), 1),
            "used_pct": round(disk_usage.used / disk_usage.total * 100, 1),
            "free_gb": round(disk_usage.free / (1024 ** 3), 1),
            "path": "data/",
        }
    except Exception:
        disk_info = None

    return {
        "status": status,
        "data_engine": engine_status,
        "database": {"connected": db_connected},
        "components": startup_components,
        "watchdog": watchdog_status,
        "executor": executor_status,
        "disk": disk_info,
        "uptime_seconds": int(uptime),
    }
