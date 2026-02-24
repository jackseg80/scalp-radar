"""Synchronise les trades Bitget historiques vers live_trades en DB.

Rattrapage pour les trades executés avant le Sprint 45.
Utilise ccxt Bitget fetchMyTrades() pour récupérer les fills.

Lancement :
    uv run python -m scripts.sync_bitget_trades --strategy grid_atr --days 30
    uv run python -m scripts.sync_bitget_trades --days 7
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone

import ccxt.pro as ccxtpro
from loguru import logger

from backend.core.config import get_config
from backend.core.database import Database
from backend.core.logging_setup import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync Bitget trades to live_trades DB")
    parser.add_argument("--strategy", type=str, default=None, help="Stratégie spécifique")
    parser.add_argument("--days", type=int, default=30, help="Nombre de jours à récupérer")
    parser.add_argument("--dry-run", action="store_true", help="Afficher sans insérer")
    return parser.parse_args()


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

        since = int((datetime.now(tz=timezone.utc) - timedelta(days=args.days)).timestamp() * 1000)

        # Récupérer les ordres déjà en DB pour dédupliquer
        existing = await db.get_live_trades(period="all", limit=10000)
        existing_order_ids = {t["order_id"] for t in existing if t.get("order_id")}

        # Récupérer les symbols des assets configurés
        symbols = []
        for asset in config.assets:
            futures_sym = f"{asset.symbol}:USDT"
            if futures_sym in exchange.markets:
                symbols.append(futures_sym)

        total_inserted = 0
        total_skipped = 0

        for symbol in symbols:
            logger.info("Fetching trades for {} since {} days...", symbol, args.days)
            try:
                trades = await exchange.fetch_my_trades(symbol, since=since, limit=500)
            except Exception as e:
                logger.warning("Erreur fetch trades {}: {}", symbol, e)
                continue

            for trade in trades:
                order_id = trade.get("order", "") or trade.get("id", "")
                if order_id in existing_order_ids:
                    total_skipped += 1
                    continue

                # Déterminer direction et trade_type depuis side + info
                side = trade.get("side", "buy")
                info = trade.get("info", {})
                trade_side = info.get("tradeSide", "")  # open, close

                # Heuristique pour direction
                is_close = trade_side == "close" or info.get("reduceOnly", False)
                if side == "buy":
                    direction = "SHORT" if is_close else "LONG"
                else:
                    direction = "LONG" if is_close else "SHORT"

                trade_type = "close" if is_close else "entry"
                fee_cost = float((trade.get("fee") or {}).get("cost", 0) or 0)

                record = {
                    "timestamp": trade.get("datetime", datetime.now(tz=timezone.utc).isoformat()),
                    "strategy_name": args.strategy or "unknown",
                    "symbol": symbol,
                    "direction": direction,
                    "trade_type": trade_type,
                    "side": side,
                    "quantity": float(trade.get("amount", 0)),
                    "price": float(trade.get("price", 0)),
                    "order_id": order_id,
                    "fee": fee_cost,
                    "pnl": None,  # P&L non calculable sans matching entry/exit
                    "pnl_pct": None,
                    "leverage": None,
                    "grid_level": None,
                    "context": "sync_bitget_trades",
                }

                if args.dry_run:
                    logger.info(
                        "[DRY-RUN] {} {} {} {} @ {} (order={})",
                        record["trade_type"], record["direction"],
                        record["symbol"], record["side"],
                        record["price"], order_id,
                    )
                else:
                    await db.insert_live_trade(record)
                    existing_order_ids.add(order_id)

                total_inserted += 1

        logger.info(
            "Sync terminée : {} trades insérés, {} doublons ignorés",
            total_inserted, total_skipped,
        )

    finally:
        await exchange.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
