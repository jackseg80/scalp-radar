"""Réconciliation des positions au boot — extrait de executor.py (Sprint Audit-C).

Fonctions module-level appelées par Executor._reconcile_on_boot() et associés.
Opèrent sur l'instance Executor passée en argument (duck-typed).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    pass  # Executor est duck-typed pour éviter l'import circulaire


def _to_futures_symbol(spot_symbol: str) -> str:
    """Convertit spot → futures (copie locale pour éviter l'import circulaire)."""
    if ":" in spot_symbol:
        return spot_symbol
    if spot_symbol.endswith("/USDT"):
        return f"{spot_symbol}:USDT"
    raise ValueError(f"Symbole non supporté pour futures: {spot_symbol}")


async def reconcile_on_boot(ex: Any) -> None:
    """Synchronise l'état local avec les positions Bitget réelles."""
    ex._is_reconciling = True
    ex._reconciliation_pnl = 0.0
    ex._reconciliation_count = 0
    ks_before = ex._risk_manager.is_kill_switch_triggered

    try:
        configured_symbols = []
        for a in ex._config.assets:
            try:
                configured_symbols.append(_to_futures_symbol(a.symbol))
            except ValueError:
                pass

        for futures_sym in configured_symbols:
            await _reconcile_symbol(ex, futures_sym)

        for futures_sym in list(ex._grid_states.keys()):
            await _reconcile_grid_symbol(ex, futures_sym)

        await cancel_orphan_orders(ex)
    finally:
        ex._is_reconciling = False

    if (
        not ks_before
        and ex._risk_manager.is_kill_switch_triggered
        and ex._reconciliation_count > 0
    ):
        strat = ex._strategy_name or "unknown"
        msg = (
            f"Kill switch déclenché par réconciliation au boot "
            f"({ex._reconciliation_count} position(s) fermée(s) pendant "
            f"downtime, P&L total: {ex._reconciliation_pnl:+.2f}$). "
            f"Reset manuel via POST /api/executor/kill-switch/reset"
            f"?strategy={strat}"
        )
        logger.warning("{}: {}", ex._log_prefix, msg)
        await ex._notifier.notify_reconciliation(
            f"⚠️ KILL SWITCH (RÉCONCILIATION BOOT)\n"
            f"{ex._reconciliation_count} position(s) fermée(s) pendant downtime\n"
            f"P&L réconciliation: {ex._reconciliation_pnl:+.2f}$\n"
            f"→ Reset manuel si le marché est stable"
        )


async def _reconcile_symbol(ex: Any, futures_sym: str) -> None:
    """Réconcilie une paire spécifique."""
    positions = await ex._fetch_positions_safe(futures_sym)

    exchange_has_position = any(
        float(p.get("contracts", 0)) > 0 for p in positions
    )
    # FIX : check mono positions ET cycles grid (pour éviter spam orphelines dans Watchdog)
    # On considère une position comme "active" si elle est déjà dans le tracking local.
    local_has_position = (futures_sym in ex._positions) or (futures_sym in ex._grid_states)

    # Cas 1 : les deux côtés ont une position → reprendre le suivi
    if exchange_has_position and local_has_position:
        # Alerte uniquement au boot réel (ex._running est False pendant le bootstrapping)
        # Sinon c'est du spam du Watchdog toutes les 15 min.
        if not getattr(ex, "_running", False):
            await ex._notifier.notify_reconciliation(
                f"Position {futures_sym} trouvée sur exchange et en local — reprise."
            )
        logger.info(
            "Executor: réconciliation {} OK — position reprise", futures_sym,
        )

    # Cas 2 : exchange a une position, pas le local → orpheline
    # P1-CR-4 Audit : créer un tracking local + SL protectif au lieu d'ignorer
    elif exchange_has_position and not local_has_position:
        # Si on est au runtime (Watchdog) et qu'on découvre une orpheline, on alerte.
        # Mais si Case 1 a été correctement géré, Case 2 ne devrait pas spammer
        # car la position sera ajoutée à ex._grid_states ci-dessous.
        pos_data = next(
            p for p in positions if float(p.get("contracts", 0)) > 0
        )
        contracts = float(pos_data.get("contracts", 0))
        entry_price = float(pos_data.get("entryPrice") or pos_data.get("markPrice") or 0)
        side = pos_data.get("side", "long").lower()
        direction = "LONG" if side == "long" else "SHORT"

        if entry_price > 0 and contracts > 0:
            from backend.execution.executor import GridLiveState, GridLivePosition

            # Créer un state grid minimal pour le tracking
            orphan_state = GridLiveState(
                symbol=futures_sym,
                direction=direction,
                strategy_name=ex._strategy_name or "orphan_recovery",
                leverage=ex._get_grid_leverage(ex._strategy_name or "grid_atr"),
            )
            orphan_state.positions.append(GridLivePosition(
                level=0,
                entry_price=entry_price,
                quantity=contracts,
                entry_order_id="orphan_recovery",
            ))
            ex._grid_states[futures_sym] = orphan_state
            ex._risk_manager.register_position({
                "symbol": futures_sym,
                "direction": direction,
                "entry_price": entry_price,
                "quantity": contracts,
            })

            # Placer un SL protectif
            try:
                await ex._update_grid_sl(futures_sym, orphan_state)
                sl_msg = f"SL placé à {orphan_state.sl_price:.2f}"
            except Exception as sl_err:
                sl_msg = f"ÉCHEC placement SL: {sl_err}"
                logger.error(
                    "Executor: échec SL protectif orpheline {}: {}", futures_sym, sl_err,
                )

            await ex._notifier.notify_reconciliation(
                f"⚠️ Position orpheline {futures_sym} récupérée "
                f"({direction}, {contracts} contracts @ {entry_price:.2f}). {sl_msg}"
            )
            logger.warning(
                "Executor: position orpheline {} récupérée — {} {} contracts @ {:.2f}, {}",
                futures_sym, direction, contracts, entry_price, sl_msg,
            )
        else:
            await ex._notifier.notify_reconciliation(
                f"Position orpheline {futures_sym} détectée sur exchange "
                f"(contracts={pos_data.get('contracts')}). Prix invalide, non récupérable."
            )
            logger.warning(
                "Executor: position orpheline {} — prix invalide, non récupérable",
                futures_sym,
            )

    # Cas 3 : local a une position, pas l'exchange → fermée pendant downtime
    elif not exchange_has_position and local_has_position:
        pos = ex._positions[futures_sym]
        exit_price = await ex._fetch_exit_price(futures_sym)
        net_pnl = ex._calculate_pnl(
            pos.direction, pos.entry_price, exit_price, pos.quantity,
        )

        from backend.execution.risk_manager import LiveTradeResult

        ex._risk_manager.record_trade_result(LiveTradeResult(
            net_pnl=net_pnl,
            timestamp=datetime.now(tz=timezone.utc),
            symbol=pos.symbol,
            direction=pos.direction,
            exit_reason="closed_during_downtime",
            strategy_name=pos.strategy_name,
        ))
        ex._risk_manager.unregister_position(pos.symbol)
        ex._reconciliation_pnl += net_pnl
        ex._reconciliation_count += 1

        # Alerte uniquement au boot réel
        if not getattr(ex, "_running", False):
            await ex._notifier.notify_reconciliation(
                f"Position {futures_sym} fermée pendant downtime. "
                f"P&L estimé: {net_pnl:+.2f}$"
            )
        logger.info(
            "Executor: position {} fermée pendant downtime, P&L={:+.2f}",
            futures_sym, net_pnl,
        )
        del ex._positions[futures_sym]

    # Cas 4 : aucune position → clean
    else:
        logger.debug("Executor: réconciliation {} — aucune position", futures_sym)


async def _reconcile_grid_symbol(ex: Any, futures_sym: str) -> None:
    """Réconcilie un cycle grid restauré avec l'exchange."""
    state = ex._grid_states.get(futures_sym)
    if state is None:
        return

    positions = await ex._fetch_positions_safe(futures_sym)
    has_position = any(
        float(p.get("contracts", 0)) > 0 for p in positions
    )

    if has_position:
        # Vérifier si le SL est toujours actif
        if state.sl_order_id:
            try:
                sl_order = await ex._exchange.fetch_order(
                    state.sl_order_id, futures_sym,
                )
                if sl_order.get("status") in ("closed", "filled"):
                    exit_price = float(
                        sl_order.get("average") or state.sl_price,
                    )
                    logger.info(
                        "Executor: SL grid exécuté pendant downtime {}",
                        futures_sym,
                    )
                    await ex._handle_grid_sl_executed(
                        futures_sym, state, exit_price,
                    )
                    return
            except Exception as e:
                err_str = str(e)
                # Bitget code 40109 = Order not found
                if "40109" in err_str or "OrderNotFound" in err_str:
                    logger.warning(
                        "Executor: SL id {} introuvable (40109) pour {} — reset et purge",
                        state.sl_order_id, futures_sym
                    )
                    state.sl_order_id = None
                    # Nettoyer le terrain au cas où d'autres orphelins trainent
                    await ex._cancel_all_open_orders(futures_sym)
                else:
                    logger.error(
                        "Executor: erreur détection SL pendant downtime {}: {}",
                        futures_sym, e,
                    )

        # Rétablir le leverage
        try:
            await ex._exchange.set_leverage(
                state.leverage, futures_sym,
            )
        except Exception as e:
            logger.warning(
                "Executor: échec restauration leverage {} pour {}: {}",
                state.leverage, futures_sym, e,
            )

        # Replacer le SL si absent (positions restaurées depuis sync.py au boot)
        # On ne replace que si on n'est pas déjà en train de tourner (Watchdog)
        # pour éviter le spam API/Telegram si le placement échoue répétitivement.
        if not state.sl_order_id and state.total_quantity > 0 and not getattr(ex, "_running", False):
            logger.info(
                "Executor: SL manquant pour {} (restauré via sync) — replacement en cours",
                futures_sym,
            )
            await ex._update_grid_sl(futures_sym, state)

        logger.info(
            "Executor: cycle grid restauré {} ({} niveaux, SL={})",
            futures_sym, len(state.positions), state.sl_order_id,
        )
    else:
        # Position fermée pendant downtime (SL exécuté ou liquidation)
        exit_price = await ex._fetch_exit_price(futures_sym)
        net_pnl = ex._calculate_pnl(
            state.direction, state.avg_entry_price,
            exit_price, state.total_quantity,
        )
        from backend.execution.risk_manager import LiveTradeResult

        ex._risk_manager.record_trade_result(LiveTradeResult(
            net_pnl=net_pnl,
            timestamp=datetime.now(tz=timezone.utc),
            symbol=futures_sym,
            direction=state.direction,
            exit_reason="closed_during_downtime",
            strategy_name=state.strategy_name,
        ))
        ex._reconciliation_pnl += net_pnl
        ex._reconciliation_count += 1
        await ex._notifier.notify_reconciliation(
            f"Cycle grid {futures_sym} fermé pendant downtime. "
            f"P&L estimé: {net_pnl:+.2f}$"
        )
        logger.info(
            "Executor: grid {} fermée pendant downtime, net={:+.2f}",
            futures_sym, net_pnl,
        )
        ex._record_grid_close(futures_sym)
        ex._grid_states.pop(futures_sym, None)


async def cancel_orphan_orders(ex: Any) -> None:
    """Annule les ordres trigger orphelins sans position associée."""
    try:
        open_orders = await ex._exchange.fetch_open_orders(
            params={"type": "swap"},
        )
    except Exception as e:
        logger.warning("Executor: impossible de fetch open orders: {}", e)
        return

    if not open_orders:
        return

    # IDs des ordres trackés localement (positions mono + grid)
    tracked_ids: set[str] = set()
    for pos in ex._positions.values():
        if pos.sl_order_id:
            tracked_ids.add(pos.sl_order_id)
        if pos.tp_order_id:
            tracked_ids.add(pos.tp_order_id)
        if pos.entry_order_id:
            tracked_ids.add(pos.entry_order_id)
    for grid_state in ex._grid_states.values():
        if grid_state.sl_order_id:
            tracked_ids.add(grid_state.sl_order_id)
        for gp in grid_state.positions:
            if gp.entry_order_id:
                tracked_ids.add(gp.entry_order_id)

    orphans = [
        o for o in open_orders
        if o.get("id") and o["id"] not in tracked_ids
    ]

    if not orphans:
        return

    cancelled: list[str] = []
    for order in orphans:
        order_id = order["id"]
        symbol = order.get("symbol", "unknown")
        try:
            await ex._exchange.cancel_order(order_id, symbol)
            cancelled.append(f"{order_id} ({symbol})")
            logger.info("Executor: ordre orphelin annulé: {} ({})", order_id, symbol)
        except Exception as e:
            logger.warning(
                "Executor: échec annulation ordre orphelin {}: {}", order_id, e,
            )

    if cancelled:
        # Alerte uniquement au boot réel pour éviter le spam périodique
        if not getattr(ex, "_running", False):
            await ex._notifier.notify_reconciliation(
                f"Ordres trigger orphelins annulés ({len(cancelled)}): "
                + ", ".join(cancelled)
            )
