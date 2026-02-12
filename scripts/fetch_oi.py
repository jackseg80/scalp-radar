"""Téléchargement de l'open interest historique depuis Binance via ccxt REST.

Script async avec asyncio.run() comme point d'entrée.
Gère la reprise incrémentale (ne télécharge que les données manquantes).

Lancement :
    uv run python -m scripts.fetch_oi --days 720
    uv run python -m scripts.fetch_oi --symbol BTC/USDT --days 720 --timeframe 5m
    uv run python -m scripts.fetch_oi --force --days 720
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


def create_exchange() -> ccxt.Exchange:
    """Crée une instance ccxt Binance futures."""
    return ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })


async def fetch_oi_for_symbol(
    exchange: ccxt.Exchange,
    db: Database,
    symbol: str,
    timeframe: str,
    since_ms: int,
    end_ms: int,
) -> int:
    """Fetch incrémental de l'OI pour un symbol."""
    # Reprise incrémentale
    latest_ts = await db.get_latest_oi_timestamp(symbol, timeframe, "binance")
    if latest_ts is not None and latest_ts > since_ms:
        since_ms = latest_ts + 1

    if since_ms >= end_ms:
        logger.info("Pas d'OI manquant pour {} {}", symbol, timeframe)
        return 0

    # Estimation du nombre de points
    days = (end_ms - since_ms) / 1000 / 86400
    tf_minutes = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
    minutes = tf_minutes.get(timeframe, 5)
    expected = int(days * 24 * 60 / minutes)
    pbar = tqdm(total=max(expected, 1), desc=f"{symbol} OI {timeframe}", unit="records", leave=False)

    batch: list[dict] = []
    current = since_ms
    total_inserted = 0
    consecutive_errors = 0
    max_consecutive_errors = 5
    # Binance limite le lookback OI (~30j pour 5m, ~60j pour 15m/30m, plus pour 1h/1d)
    jump_forward_ms = 30 * 24 * 3600 * 1000  # 30 jours en ms

    while current < end_ms:
        try:
            records = exchange.fetch_open_interest_history(
                symbol, timeframe=timeframe, since=current,
                params={"limit": 500},
            )
            consecutive_errors = 0  # reset on success
        except Exception as e:
            err_str = str(e)
            consecutive_errors += 1

            if "startTime" in err_str and "invalid" in err_str:
                # Binance rejette le startTime car trop ancien pour ce timeframe
                old_date = datetime.fromtimestamp(current / 1000, tz=timezone.utc)
                current += jump_forward_ms
                new_date = datetime.fromtimestamp(current / 1000, tz=timezone.utc)
                logger.warning(
                    "{} {} : startTime trop ancien ({}), saut → {}",
                    symbol, timeframe, old_date.strftime("%Y-%m-%d"),
                    new_date.strftime("%Y-%m-%d"),
                )
                consecutive_errors = 0  # pas une vraie erreur, on avance
                continue

            if consecutive_errors >= max_consecutive_errors:
                logger.error(
                    "{} {} : {} erreurs consécutives, abandon. Dernière : {}",
                    symbol, timeframe, max_consecutive_errors, e,
                )
                break

            logger.error("Erreur fetch OI {} {} ({}/{}) : {}",
                         symbol, timeframe, consecutive_errors,
                         max_consecutive_errors, e)
            time.sleep(2)
            continue

        if not records:
            break

        for r in records:
            batch.append({
                "symbol": symbol,
                "exchange": "binance",
                "timeframe": timeframe,
                "timestamp": r["timestamp"],
                "oi": float(r.get("baseVolume", 0) or r.get("openInterestAmount", 0)),
                "oi_value": float(r.get("quoteVolume", 0) or r.get("openInterestValue", 0)),
            })

        pbar.update(len(records))
        current = records[-1]["timestamp"] + 1
        time.sleep(0.1)  # rate limiting

        # Insert par batch de 5000 pour limiter la mémoire
        if len(batch) >= 5000:
            total_inserted += await db.insert_oi_batch(batch)
            batch = []

    if batch:
        total_inserted += await db.insert_oi_batch(batch)

    pbar.close()
    return total_inserted


async def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical open interest from Binance")
    parser.add_argument("--symbol", type=str, help="Symbol spécifique (ex: BTC/USDT)")
    parser.add_argument("--days", type=int, default=720, help="Nombre de jours (défaut: 720)")
    parser.add_argument("--timeframe", type=str, default="5m",
                        choices=["5m", "15m", "30m", "1h", "4h", "1d"],
                        help="Timeframe OI (défaut: 5m)")
    parser.add_argument("--force", action="store_true", help="Supprimer et re-fetcher")
    args = parser.parse_args()

    config = get_config()
    setup_logging(level="INFO")

    db = Database()
    await db.init()

    exchange = create_exchange()

    end_date = datetime.now(tz=timezone.utc)
    start_date = end_date - timedelta(days=args.days)
    since_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)

    # Déterminer les assets
    symbols = []
    for asset in config.assets:
        if args.symbol and asset.symbol != args.symbol:
            continue
        symbols.append(asset.symbol)

    # Avertissement : Binance limite le lookback OI à ~30 jours quel que soit le TF
    if args.days > 30:
        logger.warning(
            "Binance limite l'OI historique à ~30 jours (tous timeframes). "
            "Les données antérieures seront sautées automatiquement.",
        )

    logger.info(
        "Fetch OI Binance ({}) : {} asset(s) sur {} jours ({} → {})",
        args.timeframe, len(symbols), args.days,
        start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
    )

    total = 0
    for symbol in symbols:
        count = await fetch_oi_for_symbol(
            exchange, db, symbol, args.timeframe, since_ms, end_ms,
        )
        total += count
        logger.info("{} {} : {} OI records insérés", symbol, args.timeframe, count)

    await db.close()
    logger.info("Terminé : {} OI records au total", total)


if __name__ == "__main__":
    asyncio.run(main())
