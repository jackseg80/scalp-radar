"""Routes de test pour l'Executor.

POST /api/executor/test-trade — injecte un TradeEvent OPEN directement
POST /api/executor/test-close — ferme la position ouverte
GET  /api/executor/status — statut détaillé de l'executor
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from backend.execution.executor import TradeEvent, TradeEventType, to_futures_symbol

router = APIRouter(prefix="/api/executor", tags=["executor"])


@router.get("/status")
async def executor_status(request: Request) -> dict:
    """Statut détaillé de l'executor."""
    executor = getattr(request.app.state, "executor", None)
    if executor is None:
        return {"enabled": False, "message": "Executor non actif"}
    return executor.get_status()


@router.post("/test-trade")
async def test_trade(request: Request) -> dict:
    """Injecte un TradeEvent OPEN dans l'executor.

    Récupère le prix BTC actuel, calcule SL/TP, et envoie un ordre réel
    avec la quantité minimale (0.001 BTC).
    """
    executor = getattr(request.app.state, "executor", None)
    if executor is None:
        raise HTTPException(status_code=400, detail="Executor non actif")

    if executor.position is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Position déjà ouverte: {executor.position.direction} "
                   f"{executor.position.symbol} @ {executor.position.entry_price}",
        )

    # Récupérer le prix actuel du BTC via le DataEngine
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=400, detail="DataEngine non actif")

    # Tenter d'obtenir le prix depuis le buffer du DataEngine
    data = engine.get_data("BTC/USDT")
    current_price = _get_current_price(data)

    if current_price is None or current_price <= 0:
        # Fallback: fetch ticker via l'exchange de l'executor
        try:
            ticker = await executor._exchange.fetch_ticker(
                to_futures_symbol("BTC/USDT"),
            )
            current_price = float(ticker.get("last", 0))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Impossible d'obtenir le prix BTC: {e}",
            )

    if current_price <= 0:
        raise HTTPException(status_code=500, detail="Prix BTC invalide")

    # Paramètres du trade test — quantité minimale
    quantity = 0.001
    direction = "LONG"
    sl_pct = 0.3 / 100  # 0.3%
    tp_pct = 0.8 / 100  # 0.8%

    sl_price = round(current_price * (1 - sl_pct), 2)
    tp_price = round(current_price * (1 + tp_pct), 2)

    event = TradeEvent(
        event_type=TradeEventType.OPEN,
        strategy_name="vwap_rsi",
        symbol="BTC/USDT",
        direction=direction,
        entry_price=current_price,
        quantity=quantity,
        tp_price=tp_price,
        sl_price=sl_price,
        score=0.75,
        timestamp=datetime.now(tz=timezone.utc),
        market_regime="RANGING",
    )

    # Injecter dans l'executor
    await executor.handle_event(event)

    # Vérifier si la position a bien été ouverte
    if executor.position is not None:
        return {
            "status": "ok",
            "message": "Trade test ouvert",
            "trade": {
                "direction": direction,
                "symbol": "BTC/USDT:USDT",
                "entry_price_target": current_price,
                "entry_price_real": executor.position.entry_price,
                "quantity": executor.position.quantity,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "entry_order_id": executor.position.entry_order_id,
                "sl_order_id": executor.position.sl_order_id,
                "tp_order_id": executor.position.tp_order_id,
            },
        }

    return {
        "status": "rejected",
        "message": "Trade rejeté (voir logs pour la raison)",
        "event": {
            "direction": direction,
            "price": current_price,
            "quantity": quantity,
            "sl": sl_price,
            "tp": tp_price,
        },
    }


@router.post("/test-close")
async def test_close(request: Request) -> dict:
    """Ferme la position ouverte par test-trade."""
    executor = getattr(request.app.state, "executor", None)
    if executor is None:
        raise HTTPException(status_code=400, detail="Executor non actif")

    if executor.position is None:
        raise HTTPException(status_code=404, detail="Aucune position ouverte")

    pos = executor.position
    event = TradeEvent(
        event_type=TradeEventType.CLOSE,
        strategy_name=pos.strategy_name,
        symbol="BTC/USDT",
        direction=pos.direction,
        entry_price=pos.entry_price,
        quantity=pos.quantity,
        tp_price=pos.tp_price,
        sl_price=pos.sl_price,
        score=0.0,
        timestamp=datetime.now(tz=timezone.utc),
        exit_reason="manual_test_close",
        exit_price=pos.entry_price,  # sera mis à jour par le market close
    )

    await executor.handle_event(event)

    return {
        "status": "ok",
        "message": "Position fermée (market close)",
    }


def _get_current_price(data) -> float | None:
    """Extrait le dernier prix close depuis les buffers du DataEngine."""
    # Chercher dans le timeframe 5m d'abord, puis 1m
    for tf in ("5m", "1m", "15m"):
        candles = data.candles.get(tf, [])
        if candles:
            return candles[-1].close
    return None
