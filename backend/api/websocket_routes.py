"""WebSocket pour le push temps réel vers le frontend."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()


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


@router.websocket("/ws/live")
async def live_feed(websocket: WebSocket) -> None:
    """Push temps réel : status simulator, signaux, trades."""
    await manager.connect(websocket)

    simulator = getattr(websocket.app.state, "simulator", None)
    arena = getattr(websocket.app.state, "arena", None)

    try:
        while True:
            # Push toutes les 3 secondes
            data: dict = {"type": "update"}

            if simulator is not None:
                data["strategies"] = simulator.get_all_status()
                data["kill_switch"] = simulator.is_kill_switch_triggered()

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

            await websocket.send_json(data)
            await asyncio.sleep(3)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error("WebSocket erreur: {}", e)
        manager.disconnect(websocket)
