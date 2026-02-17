"""Routes de test pour l'Executor.

POST /api/executor/test-trade — injecte un TradeEvent OPEN directement
POST /api/executor/test-close — ferme la position ouverte
GET  /api/executor/status — statut détaillé de l'executor
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request

from backend.core.config import get_config
from backend.execution.executor import TradeEvent, TradeEventType, to_futures_symbol

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
async def executor_status(request: Request) -> dict:
    """Statut détaillé de l'executor."""
    executor = getattr(request.app.state, "executor", None)
    if executor is None:
        return {"enabled": False, "message": "Executor non actif"}
    return executor.get_status()


@router.post("/refresh-balance", dependencies=[Depends(verify_executor_key)])
async def refresh_balance(request: Request) -> dict:
    """Force un refresh du solde exchange."""
    executor = getattr(request.app.state, "executor", None)
    if executor is None:
        raise HTTPException(status_code=400, detail="Executor non actif")

    new_balance = await executor.refresh_balance()
    if new_balance is None:
        raise HTTPException(status_code=502, detail="Échec fetch balance exchange")

    return {"status": "ok", "exchange_balance": new_balance}


@router.post("/test-trade", dependencies=[Depends(verify_executor_key)])
async def test_trade(
    request: Request,
    symbol: str = Query(default="BTC/USDT", description="Symbole spot (ex: BTC/USDT, ETH/USDT)"),
) -> dict:
    """Injecte un TradeEvent OPEN dans l'executor.

    Récupère le prix actuel, calcule SL/TP, et envoie un ordre réel
    avec la quantité minimale.
    """
    executor = getattr(request.app.state, "executor", None)
    if executor is None:
        raise HTTPException(status_code=400, detail="Executor non actif")

    futures_sym = to_futures_symbol(symbol)

    if futures_sym in executor._positions:
        pos = executor._positions[futures_sym]
        raise HTTPException(
            status_code=409,
            detail=f"Position déjà ouverte: {pos.direction} "
                   f"{pos.symbol} @ {pos.entry_price}",
        )

    # Récupérer le prix actuel via le DataEngine
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=400, detail="DataEngine non actif")

    data = engine.get_data(symbol)
    current_price = _get_current_price(data)

    if current_price is None or current_price <= 0:
        # Fallback: fetch ticker via l'exchange de l'executor
        try:
            ticker = await executor._exchange.fetch_ticker(futures_sym)
            current_price = float(ticker.get("last", 0))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Impossible d'obtenir le prix {symbol}: {e}",
            )

    if current_price <= 0:
        raise HTTPException(status_code=500, detail=f"Prix {symbol} invalide")

    # Quantité minimale selon l'asset
    min_quantities = {"BTC/USDT": 0.001, "ETH/USDT": 0.01, "SOL/USDT": 0.1}
    quantity = min_quantities.get(symbol, 0.001)

    direction = "LONG"
    sl_pct = 0.3 / 100  # 0.3%
    tp_pct = 0.8 / 100  # 0.8%

    sl_price = round(current_price * (1 - sl_pct), 2)
    tp_price = round(current_price * (1 + tp_pct), 2)

    event = TradeEvent(
        event_type=TradeEventType.OPEN,
        strategy_name="vwap_rsi",
        symbol=symbol,
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
    pos = executor._positions.get(futures_sym)
    if pos is not None:
        return {
            "status": "ok",
            "message": f"Trade test ouvert ({symbol})",
            "trade": {
                "direction": direction,
                "symbol": futures_sym,
                "entry_price_target": current_price,
                "entry_price_real": pos.entry_price,
                "quantity": pos.quantity,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "entry_order_id": pos.entry_order_id,
                "sl_order_id": pos.sl_order_id,
                "tp_order_id": pos.tp_order_id,
            },
        }

    return {
        "status": "rejected",
        "message": "Trade rejeté (voir logs pour la raison)",
        "event": {
            "direction": direction,
            "symbol": symbol,
            "price": current_price,
            "quantity": quantity,
            "sl": sl_price,
            "tp": tp_price,
        },
    }


@router.post("/test-close", dependencies=[Depends(verify_executor_key)])
async def test_close(
    request: Request,
    symbol: str = Query(default="BTC/USDT", description="Symbole spot (ex: BTC/USDT, ETH/USDT)"),
) -> dict:
    """Ferme la position ouverte par test-trade."""
    executor = getattr(request.app.state, "executor", None)
    if executor is None:
        raise HTTPException(status_code=400, detail="Executor non actif")

    futures_sym = to_futures_symbol(symbol)
    pos = executor._positions.get(futures_sym)
    if pos is None:
        raise HTTPException(status_code=404, detail=f"Aucune position ouverte sur {symbol}")

    event = TradeEvent(
        event_type=TradeEventType.CLOSE,
        strategy_name=pos.strategy_name,
        symbol=symbol,
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
        "message": f"Position {symbol} fermée (market close)",
    }


def _get_current_price(data) -> float | None:
    """Extrait le dernier prix close depuis les buffers du DataEngine."""
    # Chercher dans le timeframe 5m d'abord, puis 1m
    for tf in ("5m", "1m", "15m"):
        candles = data.candles.get(tf, [])
        if candles:
            return candles[-1].close
    return None
