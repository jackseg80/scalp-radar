"""Routes DataEngine — monitoring per-symbol."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/data/status")
async def data_status(request: Request) -> dict:
    """Retourne le statut du DataEngine par symbol.

    - last_update_ago_s : secondes depuis la dernière candle reçue
    - status : "ok" (< 120s) | "stale" (>= 120s ou jamais reçu)
    """
    engine = request.app.state.engine
    if engine is None:
        return {
            "connected": False,
            "total_symbols": 0,
            "active": 0,
            "stale": 0,
            "symbols": {},
        }

    now = datetime.now(tz=timezone.utc)
    all_symbols = engine.get_all_symbols()
    symbols: dict[str, dict] = {}

    for sym in all_symbols:
        last = engine._last_update_per_symbol.get(sym)
        if last is not None:
            age = round((now - last).total_seconds(), 1)
            status = "ok" if age < 120 else "stale"
        else:
            age = None
            status = "stale"
        symbols[sym] = {"last_update_ago_s": age, "status": status}

    active = sum(1 for s in symbols.values() if s["status"] == "ok")
    stale = len(symbols) - active

    return {
        "connected": engine.is_connected,
        "total_symbols": len(symbols),
        "active": active,
        "stale": stale,
        "symbols": symbols,
    }
