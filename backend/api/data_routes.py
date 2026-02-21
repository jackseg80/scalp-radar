"""Routes DataEngine — monitoring per-symbol + backfill candles."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get("/status")
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


@router.get("/candle-status")
async def candle_status(request: Request) -> dict:
    """Retourne l'état des données historiques par asset/exchange."""
    updater = getattr(request.app.state, "candle_updater", None)
    if updater is None:
        return {"running": False, "assets": {}}
    return await updater.get_status()


@router.post("/backfill")
async def trigger_backfill(request: Request) -> JSONResponse:
    """Déclenche un backfill en background.

    Body optionnel: { "exchanges": ["binance","bitget"], "timeframes": ["1h"] }
    """
    updater = getattr(request.app.state, "candle_updater", None)
    if updater is None:
        return JSONResponse(
            {"status": "error", "message": "CandleUpdater non disponible"},
            status_code=503,
        )

    if updater.is_running:
        return JSONResponse(
            {"status": "already_running"},
            status_code=409,
        )

    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass

    exchanges = body.get("exchanges")
    timeframes = body.get("timeframes")

    asyncio.create_task(updater.run_backfill(exchanges=exchanges, timeframes=timeframes))

    return JSONResponse({"status": "started"}, status_code=202)
