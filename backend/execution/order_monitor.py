"""Surveillance des ordres et positions — extrait de executor.py (Sprint Audit-C).

Fonctions module-level appelées par les méthodes Executor._watch_orders_loop(),
_poll_positions_loop(), etc. Opèrent sur l'instance Executor passée en argument.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    pass  # Executor est duck-typed pour éviter l'import circulaire

_POLL_INTERVAL = 5


async def watch_orders_loop(ex: Any) -> None:
    """Mécanisme principal : watchOrders via ccxt Pro."""
    while ex._running:
        try:
            if not ex._positions and not ex._grid_states:
                await asyncio.sleep(1)
                continue

            orders = await ex._exchange.watch_orders()
            for order in orders:
                await process_watched_order(ex, order)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning("Executor: erreur watchOrders: {}", e)
            await asyncio.sleep(2)


async def process_watched_order(ex: Any, order: dict) -> None:
    """Traite un ordre détecté par watchOrders."""
    async with ex._state_lock:  # P1-RC-3 Audit
        await _process_watched_order_unlocked(ex, order)


async def _process_watched_order_unlocked(ex: Any, order: dict) -> None:
    """Implémentation interne (appelée sous _state_lock)."""
    order_id = order.get("id", "")
    status = order.get("status", "")

    if status not in ("closed", "filled"):
        return

    # Hotfix 34 : extraire fee du WS push
    fee_info = order.get("fee") or {}
    exit_fee: float | None = (
        float(fee_info.get("cost") or 0)
        if fee_info.get("cost") is not None
        else None
    )

    # Scanner toutes les positions pour matcher l'order_id
    for symbol, pos in list(ex._positions.items()):
        exit_reason = ""
        if order_id == pos.sl_order_id:
            exit_reason = "sl"
        elif order_id == pos.tp_order_id:
            exit_reason = "tp"
        else:
            continue

        exit_price = float(order.get("average") or order.get("price") or 0)

        # Sprint Journal V2 : enregistrer le SL/TP fill dans l'historique
        close_side = "sell" if pos.direction == "LONG" else "buy"
        ex._record_order(
            exit_reason, symbol, close_side, pos.quantity,
            order, pos.strategy_name, f"watched_{exit_reason}",
        )

        # Si fee absente du WS push (fréquent pour trigger orders Bitget), fetch le fill
        if exit_fee is None or exit_fee == 0:
            _, fetched_fee = await ex._fetch_fill_price(
                order_id, symbol, exit_price,
            )
            if fetched_fee is not None:
                exit_fee = fetched_fee

        await ex._handle_exchange_close(symbol, exit_price, exit_reason, exit_fee)
        return

    # Scanner les grid states pour SL match
    for futures_sym, grid_state in list(ex._grid_states.items()):
        if order_id == grid_state.sl_order_id:
            exit_price = float(
                order.get("average") or order.get("price") or grid_state.sl_price,
            )

            # Sprint Journal V2 : enregistrer le SL grid fill dans l'historique
            close_side = "sell" if grid_state.direction == "LONG" else "buy"
            ex._record_order(
                "sl", futures_sym, close_side, grid_state.total_quantity,
                order, grid_state.strategy_name, "watched_grid_sl",
            )

            # Si fee absente du WS push, fetch le fill
            if exit_fee is None or exit_fee == 0:
                _, fetched_fee = await ex._fetch_fill_price(
                    order_id, futures_sym, exit_price,
                )
                if fetched_fee is not None:
                    exit_fee = fetched_fee

            await ex._handle_grid_sl_executed(
                futures_sym, grid_state, exit_price, exit_fee,
            )
            return

    # Ordre non matché — log debug (ordres d'autres systèmes sur le sous-compte)
    logger.debug(
        "Executor: ordre non matché ignoré: id={}, symbol={}, status={}",
        order_id, order.get("symbol", "?"), status,
    )


async def poll_positions_loop(ex: Any) -> None:
    """Fallback : vérifie périodiquement l'état des positions."""
    while ex._running:
        try:
            await asyncio.sleep(_POLL_INTERVAL)
            if not ex._running:
                continue

            for symbol in list(ex._positions.keys()):
                await check_position_still_open(ex, symbol)
            for symbol in list(ex._grid_states.keys()):
                await check_grid_still_open(ex, symbol)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning("Executor: erreur polling: {}", e)


async def check_position_still_open(ex: Any, symbol: str) -> None:
    """Vérifie si une position est toujours ouverte sur l'exchange."""
    async with ex._state_lock:  # P1-RC-3 Audit
        await _check_position_still_open_unlocked(ex, symbol)


async def _check_position_still_open_unlocked(ex: Any, symbol: str) -> None:
    """Implémentation interne (appelée sous _state_lock)."""
    pos = ex._positions.get(symbol)
    if pos is None:
        return

    positions = await ex._fetch_positions_safe(symbol)
    has_open = any(
        float(p.get("contracts", 0)) > 0 for p in positions
    )

    if not has_open:
        logger.info(
            "Executor: position {} fermée côté exchange (détectée par polling)",
            symbol,
        )
        exit_price = await ex._fetch_exit_price(symbol)
        exit_reason = await ex._determine_exit_reason(symbol)

        # Hotfix 34 completion : extraire fees réelles
        exit_fee: float | None = None
        try:
            if pos.sl_order_id or pos.tp_order_id:
                order_id = pos.tp_order_id or pos.sl_order_id or ""
                _, exit_fee = await ex._fetch_fill_price(
                    order_id, symbol, exit_price,
                )
        except Exception as e:
            logger.warning(
                "Executor: fee extraction failed for {} ({}), using estimate",
                symbol, e,
            )
            exit_fee = None

        await ex._handle_exchange_close(symbol, exit_price, exit_reason, exit_fee)


async def check_grid_still_open(ex: Any, symbol: str) -> None:
    """Vérifie si la position grid est toujours ouverte sur l'exchange."""
    async with ex._state_lock:  # P1-RC-3 Audit
        await _check_grid_still_open_unlocked(ex, symbol)


async def _check_grid_still_open_unlocked(ex: Any, symbol: str) -> None:
    """Implémentation interne (appelée sous _state_lock)."""
    state = ex._grid_states.get(symbol)
    if state is None:
        return

    positions = await ex._fetch_positions_safe(symbol)
    has_open = any(float(p.get("contracts", 0)) > 0 for p in positions)

    if not has_open:
        logger.info(
            "Executor: grid {} fermée côté exchange (détectée par polling)", symbol,
        )
        exit_price = await ex._fetch_exit_price(symbol)

        # Hotfix 34 completion : extraire fees réelles
        exit_fee: float | None = None
        try:
            if state.sl_order_id:
                _, exit_fee = await ex._fetch_fill_price(
                    state.sl_order_id, symbol, exit_price,
                )
        except Exception as e:
            logger.warning(
                "Executor: fee extraction failed for grid {} ({}), using estimate",
                symbol, e,
            )
            exit_fee = None

        await ex._handle_grid_sl_executed(symbol, state, exit_price, exit_fee)


async def handle_exchange_close(
    ex: Any, symbol: str, exit_price: float, exit_reason: str,
    exit_fee: float | None = None,
) -> None:
    """Traite la fermeture d'une position par l'exchange (TP/SL hit)."""
    pos = ex._positions.get(symbol)
    if pos is None:
        return

    # Annuler l'autre ordre (si SL hit, annuler TP et vice-versa)
    await ex._cancel_pending_orders(symbol)

    # Hotfix 34 : fees réelles si disponibles
    if exit_fee is not None:
        net_pnl = ex._calculate_real_pnl(
            pos.direction, pos.entry_price, exit_price,
            pos.quantity, pos.entry_fee, exit_fee,
        )
    else:
        net_pnl = ex._calculate_pnl(
            pos.direction, pos.entry_price, exit_price, pos.quantity,
        )

    logger.info(
        "Executor: EXCHANGE CLOSE {} {} @ {:.2f} net={:+.2f} ({})",
        pos.direction, symbol, exit_price, net_pnl, exit_reason,
    )

    # Sprint 45 : persist exchange close
    close_side = "sell" if pos.direction == "LONG" else "buy"
    trade_type = "sl_close" if exit_reason == "sl" else "tp_close"
    leverage = ex._config.risk.position.default_leverage
    margin = pos.entry_price * pos.quantity / leverage
    pnl_pct = (net_pnl / margin * 100) if margin > 0 else 0.0
    await ex._persist_live_trade(
        trade_type, symbol, close_side, pos.direction,
        pos.quantity, exit_price,
        strategy_name=pos.strategy_name,
        fee=exit_fee or 0,
        pnl=round(net_pnl, 4),
        pnl_pct=round(pnl_pct, 2),
        leverage=leverage,
        context=f"exchange_{exit_reason}",
    )

    from backend.execution.risk_manager import LiveTradeResult

    ex._risk_manager.record_trade_result(LiveTradeResult(
        net_pnl=net_pnl,
        timestamp=datetime.now(tz=timezone.utc),
        symbol=symbol,
        direction=pos.direction,
        exit_reason=exit_reason,
        strategy_name=pos.strategy_name,
    ))
    ex._risk_manager.unregister_position(symbol)

    await ex._notifier.notify_live_order_closed(
        symbol, pos.direction,
        pos.entry_price, exit_price,
        net_pnl, exit_reason,
        pos.strategy_name,
    )

    del ex._positions[symbol]
