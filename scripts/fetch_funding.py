"""Téléchargement des funding rates historiques depuis Binance via ccxt REST.

Script async avec asyncio.run() comme point d'entrée.
Gère la reprise incrémentale (ne télécharge que les données manquantes).
Les taux sont stockés en % (cohérent avec DataEngine).

Lancement :
    uv run python -m scripts.fetch_funding --days 720
    uv run python -m scripts.fetch_funding --symbol BTC/USDT --days 720
    uv run python -m scripts.fetch_funding --force --days 720
"""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime, timedelta, timezone

import ccxt
from loguru import logger
from tqdm import tqdm

from backend.core.config import get_config
from backend.core.database import Database
from backend.core.logging_setup import setup_logging


def create_exchange(exchange_name: str) -> ccxt.Exchange:
    """Create a futures client for the requested persisted funding source."""
    if exchange_name == "bitget":
        return ccxt.bitget({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
    return ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })


async def fetch_funding_for_symbol(
    exchange: ccxt.Exchange,
    db: Database,
    symbol: str,
    since_ms: int,
    end_ms: int,
    exchange_name: str,
    resume: bool = True,
) -> int:
    """Fetch incrémental des funding rates pour un symbol."""
    # Reprise incrémentale
    latest_ts = await db.get_latest_funding_timestamp(symbol, exchange_name)
    if resume and latest_ts is not None and latest_ts > since_ms:
        since_ms = latest_ts + 1

    if since_ms >= end_ms:
        logger.info("Pas de funding manquant pour {}", symbol)
        return 0

    # Estimation : 3 rates/jour
    days = (end_ms - since_ms) / 1000 / 86400
    expected = int(days * 3)
    pbar = tqdm(total=max(expected, 1), desc=f"{symbol} funding", unit="rates", leave=False)

    batch: list[dict] = []
    current = since_ms
    while current < end_ms:
        try:
            rates = exchange.fetch_funding_rate_history(
                symbol, since=current, limit=1000,
            )
        except Exception as e:
            logger.error("Erreur fetch funding {} : {}", symbol, e)
            time.sleep(2)
            continue

        if not rates:
            break

        for r in rates:
            batch.append({
                "symbol": symbol,
                "exchange": exchange_name,
                "timestamp": r["timestamp"],
                "funding_rate": r["fundingRate"] * 100,  # → en %, cohérent DataEngine
            })

        pbar.update(len(rates))
        current = rates[-1]["timestamp"] + 1
        time.sleep(0.1)  # rate limiting

    pbar.close()

    inserted = 0
    if batch:
        inserted = await db.insert_funding_rates_batch(batch)
    return inserted


async def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical funding rates")
    parser.add_argument("--symbol", type=str, help="Symbol spécifique (ex: BTC/USDT)")
    parser.add_argument("--days", type=int, default=720, help="Nombre de jours (défaut: 720)")
    parser.add_argument("--since", help="UTC ISO start; re-fetches the requested range")
    parser.add_argument("--until", help="UTC ISO end (exclusive); defaults to now")
    parser.add_argument("--symbols", help="CSV symbols; bypass assets.yaml")
    parser.add_argument("--exchange", choices=["binance", "bitget"], default="binance")
    parser.add_argument("--db", default="data/scalp_radar.db")
    parser.add_argument("--force", action="store_true", help="Supprimer et re-fetcher")
    args = parser.parse_args()

    config = get_config()
    setup_logging(level="INFO")

    db = Database(args.db)
    await db.init()

    exchange = create_exchange(args.exchange)

    end_date = (
        datetime.fromisoformat(args.until.replace("Z", "+00:00")).astimezone(timezone.utc)
        if args.until else datetime.now(tz=timezone.utc)
    )
    start_date = (
        datetime.fromisoformat(args.since.replace("Z", "+00:00")).astimezone(timezone.utc)
        if args.since else end_date - timedelta(days=args.days)
    )
    since_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)

    # Déterminer les assets
    symbols = (
        [value.strip() for value in args.symbols.split(",") if value.strip()]
        if args.symbols else []
    )
    if not symbols:
        for asset in config.assets:
            if args.symbol and asset.symbol != args.symbol:
                continue
            symbols.append(asset.symbol)

    logger.info(
        "Fetch funding rates {} : {} asset(s) sur {} jours ({} → {})",
        args.exchange, len(symbols), args.days,
        start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
    )

    # --force : pas de table delete pour funding, on re-insert (INSERT OR IGNORE)
    # Le flag force est géré par le fait qu'on commence depuis since_ms sans reprise

    total = 0
    for symbol in symbols:
        start = since_ms if args.force else since_ms
        count = await fetch_funding_for_symbol(
            exchange, db, symbol, start, end_ms,
            exchange_name=args.exchange, resume=not bool(args.since),
        )
        total += count
        logger.info("{} : {} funding rates insérés", symbol, count)

    await db.close()
    logger.info("Terminé : {} funding rates au total ({})", total, args.exchange)


if __name__ == "__main__":
    asyncio.run(main())
