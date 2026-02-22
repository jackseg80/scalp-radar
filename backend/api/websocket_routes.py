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
    """Extrait le dernier prix et variation depuis les buffers du DataEngine.

    Fallback multi-timeframe : 1m → 5m → 1h → 4h pour les altcoins sans données 1m.
    """
    prices = {}
    for symbol in engine.get_all_symbols():
        data = engine.get_data(symbol)
        candles = None
        for tf in ("1m", "5m", "1h", "4h"):
            c = data.candles.get(tf, [])
            if c:
                candles = c
                break
        if not candles:
            continue
        last = candles[-1]
        change_pct = None
        if len(candles) >= 2:
            prev = candles[-2].close
            if prev > 0:
                change_pct = round((last.close - prev) / prev * 100, 2)
        prices[symbol] = {"last": last.close, "change_pct": change_pct}
    return prices


def _merge_live_grids_into_state(grid_state: dict, exec_status: dict) -> None:
    """Merge les positions live dans grid_state, remplaçant les paper pour les stratégies live.

    Sprint 39 : le grid_state résultant contient un mix paper + live.
    Les positions live (source="live") écrasent les paper du même strategy:symbol.
    Les positions paper de stratégies non-live restent avec source="paper".
    """
    exec_grid = exec_status.get("executor_grid_state")
    if not exec_grid or not exec_grid.get("grid_positions"):
        # Pas de grids live — tagger tout comme paper
        for g in grid_state.get("grid_positions", {}).values():
            g.setdefault("source", "paper")
        return

    gp = grid_state.setdefault("grid_positions", {})

    # Déterminer les stratégies live
    live_strategies = set()
    for g in exec_grid["grid_positions"].values():
        live_strategies.add(g.get("strategy_name", ""))

    # Supprimer les entrées paper des stratégies live (évite doublons)
    keys_to_remove = [
        k for k, g in gp.items() if g.get("strategy", "") in live_strategies
    ]
    for k in keys_to_remove:
        del gp[k]

    # Tagger les paper restantes
    for g in gp.values():
        g.setdefault("source", "paper")

    # Ajouter les entrées live converties au format paper
    for key, g in exec_grid["grid_positions"].items():
        spot_sym = g.get("symbol", "")
        if ":" in spot_sym:
            spot_sym = spot_sym.split(":")[0]
        gp[key] = {
            "symbol": spot_sym,
            "strategy": g.get("strategy_name", ""),
            "direction": g.get("direction", ""),
            "levels_open": g.get("levels", 0),
            "levels_max": g.get("levels_max", 3),
            "avg_entry": g.get("entry_price", 0),
            "current_price": g.get("current_price"),
            "unrealized_pnl": g.get("unrealized_pnl", 0),
            "unrealized_pnl_pct": g.get("unrealized_pnl_pct", 0),
            "tp_price": g.get("tp_price"),
            "sl_price": g.get("sl_price"),
            "tp_distance_pct": g.get("tp_distance_pct"),
            "sl_distance_pct": g.get("sl_distance_pct"),
            "margin_used": g.get("margin_used", 0),
            "leverage": g.get("leverage", 6),
            "duration_hours": g.get("duration_hours"),
            "source": "live",
            "positions": g.get("positions", []),
        }

    # Recalculer le summary
    all_grids = list(gp.values())
    grid_state["summary"] = {
        "total_positions": sum(
            g.get("levels_open", g.get("levels", 0)) for g in all_grids
        ),
        "total_assets": len(all_grids),
        "total_margin_used": round(
            sum(g.get("margin_used", 0) for g in all_grids), 2,
        ),
        "total_unrealized_pnl": round(
            sum(g.get("unrealized_pnl", 0) for g in all_grids), 2,
        ),
        "capital_available": grid_state.get("summary", {}).get("capital_available", 0),
    }


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
        exec_status = executor.get_status()
        data["executor"] = exec_status
        # Sprint 39 : merger positions live dans grid_state
        if "grid_state" in data:
            _merge_live_grids_into_state(data["grid_state"], exec_status)

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
