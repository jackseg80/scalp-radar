"""WebSocket pour le push temps réel vers le frontend.

Sprint 31 : intégration log alerts WARNING+ via per-connection queue.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from backend.core.logging_setup import get_log_buffer, subscribe_logs, unsubscribe_logs

router = APIRouter()


def _get_current_prices(engine) -> dict:
    """Extrait le dernier prix et variation depuis les buffers du DataEngine."""
    prices = {}
    for symbol in engine.get_all_symbols():
        data = engine.get_data(symbol)
        candles_1m = data.candles.get("1m", [])
        if candles_1m:
            last = candles_1m[-1]
            change_pct = None
            if len(candles_1m) >= 2:
                prev = candles_1m[-2].close
                if prev > 0:
                    change_pct = round((last.close - prev) / prev * 100, 2)
            prices[symbol] = {"last": last.close, "change_pct": change_pct}
    return prices


def _build_update_data(simulator, arena, executor, engine) -> dict:
    """Construit le payload update standard (extrait pour lisibilité)."""
    data: dict = {"type": "update"}

    if simulator is not None:
        data["strategies"] = simulator.get_all_status()
        data["kill_switch"] = simulator.is_kill_switch_triggered()
        data["simulator_positions"] = simulator.get_open_positions()
        data["grid_state"] = simulator.get_grid_state()

    if arena is not None:
        ranking = arena.get_ranking()
        data["ranking"] = [
            {
                "name": p.name,
                "net_pnl": p.net_pnl,
                "net_return_pct": p.net_return_pct,
                "win_rate": p.win_rate,
                "is_active": p.is_active,
            }
            for p in ranking
        ]

    if executor is not None:
        data["executor"] = executor.get_status()

    if engine is not None:
        data["prices"] = _get_current_prices(engine)

    return data


class ConnectionManager:
    """Gère les connexions WebSocket du frontend."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info("WebSocket client connecté ({} total)", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info("WebSocket client déconnecté ({} restants)", len(self._connections))

    async def broadcast(self, data: dict) -> None:
        """Envoie des données à tous les clients connectés."""
        if not self._connections:
            return

        message = json.dumps(data)
        disconnected = []
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self.disconnect(ws)

    @property
    def client_count(self) -> int:
        return len(self._connections)


manager = ConnectionManager()

UPDATE_INTERVAL = 3.0  # secondes entre chaque update standard


@router.websocket("/ws/live")
async def live_feed(websocket: WebSocket) -> None:
    """Push temps réel : status simulator, signaux, trades, log alerts."""
    await manager.connect(websocket)

    simulator = getattr(websocket.app.state, "simulator", None)
    arena = getattr(websocket.app.state, "arena", None)
    executor = getattr(websocket.app.state, "executor", None)
    engine = getattr(websocket.app.state, "engine", None)

    # Sprint 31 : subscribe aux log alerts
    log_queue = subscribe_logs()

    try:
        # Envoyer le buffer initial (derniers WARNING/ERROR) au nouveau client
        for entry in get_log_buffer():
            await websocket.send_json({"type": "log_alert", "entry": entry})

        loop = asyncio.get_event_loop()
        next_update = loop.time() + UPDATE_INTERVAL

        while True:
            now = loop.time()
            remaining = max(0, next_update - now)

            try:
                # Attendre un log alert OU le timeout pour l'update régulier
                alert_entry = await asyncio.wait_for(
                    log_queue.get(), timeout=remaining
                )
                # Log alert reçu → envoyer immédiatement
                await websocket.send_json(
                    {"type": "log_alert", "entry": alert_entry}
                )
                # NE PAS reset next_update (garder la cadence 3s pour les updates)

            except asyncio.TimeoutError:
                # 3s écoulées → envoyer l'update standard
                data = _build_update_data(simulator, arena, executor, engine)
                await websocket.send_json(data)
                next_update = loop.time() + UPDATE_INTERVAL

    except WebSocketDisconnect:
        unsubscribe_logs(log_queue)
        manager.disconnect(websocket)
    except Exception as e:
        logger.error("WebSocket erreur: {}", e)
        unsubscribe_logs(log_queue)
        manager.disconnect(websocket)
