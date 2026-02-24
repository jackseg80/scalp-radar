"""Synchronise les trades Bitget historiques vers live_trades en DB.

Rattrapage pour les trades executés avant le Sprint 45.
Utilise ccxt Bitget fetchMyTrades() pour récupérer les fills,
classifie entry/close, et calcule le P&L par matching FIFO.

Lancement :
    uv run python -m scripts.sync_bitget_trades --strategy grid_atr --days 30
    uv run python -m scripts.sync_bitget_trades --days 7
    uv run python -m scripts.sync_bitget_trades --purge --strategy grid_atr --days 30
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import ccxt.pro as ccxtpro
from loguru import logger

from backend.core.config import get_config
from backend.core.database import Database
from backend.core.logging_setup import setup_logging


# Stratégies LONG-only : sell = toujours close
LONG_ONLY_STRATEGIES = {"grid_atr", "grid_boltrend", "grid_momentum"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync Bitget trades to live_trades DB")
    parser.add_argument("--strategy", type=str, default=None, help="Stratégie spécifique")
    parser.add_argument("--days", type=int, default=30, help="Nombre de jours à récupérer")
    parser.add_argument("--dry-run", action="store_true", help="Afficher sans insérer")
    parser.add_argument(
        "--purge", action="store_true",
        help="Purger les trades sync existants avant re-sync",
    )
    return parser.parse_args()


def _classify_trade(
    trade: dict,
    strategy: str | None,
) -> tuple[str, str]:
    """Classifie un fill Bitget en (direction, trade_type).

    Retourne (direction, trade_type) où :
    - direction: "LONG" ou "SHORT"
    - trade_type: "entry", "tp_close", "sl_close", ou "close"
    """
    side = trade.get("side", "buy")
    info = trade.get("info", {})

    # 1) Champs Bitget fiables
    trade_side = info.get("tradeSide", "")       # "open" ou "close"
    reduce_only = info.get("reduceOnly", False)
    if isinstance(reduce_only, str):
        reduce_only = reduce_only.lower() == "true"

    # 2) Déterminer si c'est un close
    is_close = trade_side == "close" or reduce_only

    # 3) Heuristique par stratégie pour LONG-only
    if not is_close and strategy in LONG_ONLY_STRATEGIES:
        # Pour LONG-only: sell = forcément un close
        if side == "sell":
            is_close = True

    # 4) Direction
    if is_close:
        # Close d'un LONG = sell, close d'un SHORT = buy
        direction = "LONG" if side == "sell" else "SHORT"
    else:
        # Entry LONG = buy, entry SHORT = sell
        direction = "LONG" if side == "buy" else "SHORT"

    # 5) Trade type
    if is_close:
        trade_type = "close"  # On raffinera en tp_close/sl_close si possible
    else:
        trade_type = "entry"

    return direction, trade_type


def _compute_pnl_for_closes(
    trades: list[dict],
) -> list[dict]:
    """Calcule le P&L des closes par matching FIFO des entries.

    Modifie in-place les trades de type close pour ajouter pnl et pnl_pct.
    Retourne la liste complète (entries + closes avec P&L).
    """
    # Trier par timestamp
    trades.sort(key=lambda t: t["timestamp"])

    # État par (symbol, direction) : file FIFO de {price, qty_remaining}
    positions: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for trade in trades:
        symbol = trade["symbol"]
        direction = trade["direction"]
        key = (symbol, direction)

        if trade["trade_type"] == "entry":
            positions[key].append({
                "price": trade["price"],
                "qty_remaining": trade["quantity"],
            })
        else:
            # Close : matcher FIFO
            close_qty = trade["quantity"]
            close_price = trade["price"]
            total_pnl = 0.0
            total_cost = 0.0  # Pour calculer avg_entry pondéré

            fifo = positions[key]
            matched_qty = 0.0

            while close_qty > 1e-12 and fifo:
                entry = fifo[0]
                match_qty = min(close_qty, entry["qty_remaining"])

                if direction == "LONG":
                    pnl_piece = (close_price - entry["price"]) * match_qty
                else:
                    pnl_piece = (entry["price"] - close_price) * match_qty

                total_pnl += pnl_piece
                total_cost += entry["price"] * match_qty
                matched_qty += match_qty

                entry["qty_remaining"] -= match_qty
                close_qty -= match_qty

                if entry["qty_remaining"] < 1e-12:
                    fifo.pop(0)

            # Soustraire les fees (entry + close)
            fee = trade.get("fee", 0) or 0
            total_pnl -= fee

            trade["pnl"] = round(total_pnl, 4)

            # pnl_pct = pnl / margin * 100, margin = cost / leverage
            leverage = trade.get("leverage") or 1
            if total_cost > 0:
                margin = total_cost / leverage
                trade["pnl_pct"] = round(total_pnl / margin * 100, 2)
            else:
                trade["pnl_pct"] = None

            if close_qty > 1e-12:
                logger.warning(
                    "  {} {} : {:.6f} qty non matchée (entries manquantes)",
                    symbol, direction, close_qty,
                )

    return trades


async def _fetch_all_trades(
    exchange: ccxtpro.Exchange,
    symbol: str,
    since: int,
) -> list[dict]:
    """Fetch tous les trades avec pagination (Bitget limite à 500/requête)."""
    all_trades: list[dict] = []
    current_since = since

    while True:
        try:
            batch = await exchange.fetch_my_trades(
                symbol, since=current_since, limit=500,
            )
        except Exception as e:
            logger.warning("Erreur fetch trades {}: {}", symbol, e)
            break

        if not batch:
            break

        all_trades.extend(batch)

        # Avancer le curseur après le dernier trade
        last_ts = batch[-1].get("timestamp", 0)
        if last_ts <= current_since:
            break
        current_since = last_ts + 1

        if len(batch) < 500:
            break

    return all_trades


async def main() -> None:
    args = parse_args()
    config = get_config()
    setup_logging(level="INFO")

    db = Database()
    await db.init()

    # Créer l'exchange Bitget
    if args.strategy:
        api_key, secret, passphrase = config.get_executor_keys(args.strategy)
    else:
        api_key = config.secrets.bitget_api_key
        secret = config.secrets.bitget_secret
        passphrase = config.secrets.bitget_passphrase

    exchange = ccxtpro.bitget({
        "apiKey": api_key,
        "secret": secret,
        "password": passphrase,
        "options": {"defaultType": "swap"},
    })

    try:
        await exchange.load_markets()

        # Purge si demandé
        if args.purge:
            count = await db.purge_live_trades(context="sync_bitget_trades")
            logger.info("Purgé {} trades sync existants", count)

        since = int(
            (datetime.now(tz=timezone.utc) - timedelta(days=args.days)).timestamp() * 1000,
        )

        # Récupérer les ordres déjà en DB pour dédupliquer
        existing = await db.get_live_trades(period="all", limit=10000)
        existing_order_ids = {t["order_id"] for t in existing if t.get("order_id")}

        # Récupérer les symbols des assets configurés
        symbols = []
        for asset in config.assets:
            futures_sym = f"{asset.symbol}:USDT"
            if futures_sym in exchange.markets:
                symbols.append(futures_sym)

        # Récupérer le leverage par stratégie
        strategy_name = args.strategy or "unknown"
        strategy_config = getattr(config.strategies, strategy_name, None)
        leverage = getattr(strategy_config, "leverage", None) or 3

        # ── Phase 1 : Fetch + Classify ───────────────────────────────────
        all_records: list[dict] = []
        total_skipped = 0

        for symbol in symbols:
            logger.info("Fetching trades for {} since {} days...", symbol, args.days)
            trades = await _fetch_all_trades(exchange, symbol, since)
            logger.info("  {} fills bruts pour {}", len(trades), symbol)

            for trade in trades:
                order_id = trade.get("order", "") or trade.get("id", "")
                if order_id in existing_order_ids:
                    total_skipped += 1
                    continue

                direction, trade_type = _classify_trade(trade, args.strategy)
                side = trade.get("side", "buy")
                fee_cost = float((trade.get("fee") or {}).get("cost", 0) or 0)

                record = {
                    "timestamp": trade.get(
                        "datetime",
                        datetime.now(tz=timezone.utc).isoformat(),
                    ),
                    "strategy_name": strategy_name,
                    "symbol": symbol,
                    "direction": direction,
                    "trade_type": trade_type,
                    "side": side,
                    "quantity": float(trade.get("amount", 0)),
                    "price": float(trade.get("price", 0)),
                    "order_id": order_id,
                    "fee": fee_cost,
                    "pnl": None,
                    "pnl_pct": None,
                    "leverage": leverage,
                    "grid_level": None,
                    "context": "sync_bitget_trades",
                }
                all_records.append(record)

        # ── Phase 2 : Calculer P&L pour les closes ───────────────────────
        logger.info(
            "Classification : {} entries, {} closes sur {} trades",
            sum(1 for r in all_records if r["trade_type"] == "entry"),
            sum(1 for r in all_records if r["trade_type"] != "entry"),
            len(all_records),
        )

        _compute_pnl_for_closes(all_records)

        # ── Phase 3 : Insérer en DB ──────────────────────────────────────
        total_inserted = 0
        total_entries = 0
        total_closes = 0
        total_pnl = 0.0

        for record in all_records:
            is_close = record["trade_type"] != "entry"

            if args.dry_run:
                pnl_str = f"P&L={record['pnl']:.2f}" if record["pnl"] is not None else ""
                logger.info(
                    "[DRY-RUN] {} {} {} {} @ {:.2f} {} (order={})",
                    record["trade_type"], record["direction"],
                    record["symbol"], record["side"],
                    record["price"], pnl_str, record["order_id"],
                )
            else:
                await db.insert_live_trade(record)
                existing_order_ids.add(record["order_id"])

            total_inserted += 1
            if is_close:
                total_closes += 1
                if record["pnl"] is not None:
                    total_pnl += record["pnl"]
            else:
                total_entries += 1

        logger.info("─" * 60)
        logger.info(
            "Sync terminée : {} trades ({} entries + {} closes), {} doublons ignorés",
            total_inserted, total_entries, total_closes, total_skipped,
        )
        logger.info("P&L total calculé : {:.2f} USDT", total_pnl)

        # Résumé par symbol
        pnl_by_symbol: dict[str, float] = defaultdict(float)
        trades_by_symbol: dict[str, int] = defaultdict(int)
        for r in all_records:
            if r["trade_type"] != "entry" and r["pnl"] is not None:
                pnl_by_symbol[r["symbol"]] += r["pnl"]
                trades_by_symbol[r["symbol"]] += 1

        for sym in sorted(pnl_by_symbol):
            logger.info(
                "  {} : {} closes, P&L {:.2f}",
                sym, trades_by_symbol[sym], pnl_by_symbol[sym],
            )

    finally:
        await exchange.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
