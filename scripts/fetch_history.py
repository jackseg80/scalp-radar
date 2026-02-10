"""Téléchargement de l'historique des klines depuis Bitget via ccxt REST.

Script async avec asyncio.run() comme point d'entrée.
Gère la reprise (ne télécharge que les données manquantes).

Lancement :
    uv run python -m scripts.fetch_history
    uv run python -m scripts.fetch_history --symbol BTC/USDT --timeframe 5m --days 7
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone

import ccxt
from loguru import logger
from tqdm import tqdm

from backend.core.config import get_config
from backend.core.database import Database
from backend.core.logging_setup import setup_logging
from backend.core.models import Candle, TimeFrame


async def fetch_ohlcv(
    exchange: ccxt.bitget,
    symbol: str,
    timeframe: str,
    since_ms: int,
    limit: int = 200,
) -> list[list]:
    """Fetch OHLCV data depuis l'API REST (synchrone ccxt)."""
    return exchange.fetch_ohlcv(
        symbol, timeframe, since=since_ms, limit=limit
    )


async def fetch_symbol_timeframe(
    exchange: ccxt.bitget,
    db: Database,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
) -> int:
    """Télécharge les klines pour un (symbol, timeframe) et les persiste."""
    tf = TimeFrame.from_string(timeframe)
    interval_ms = tf.to_milliseconds()

    # Vérifier les données existantes pour la reprise
    latest = await db.get_latest_candle_timestamp(symbol, timeframe)
    if latest and latest.timestamp() * 1000 > start_date.timestamp() * 1000:
        actual_start_ms = int(latest.timestamp() * 1000) + interval_ms
        logger.info(
            "Reprise {} {} depuis {}",
            symbol,
            timeframe,
            datetime.fromtimestamp(actual_start_ms / 1000, tz=timezone.utc),
        )
    else:
        actual_start_ms = int(start_date.timestamp() * 1000)

    end_ms = int(end_date.timestamp() * 1000)
    total_expected = (end_ms - actual_start_ms) // interval_ms
    if total_expected <= 0:
        logger.info("Pas de données manquantes pour {} {}", symbol, timeframe)
        return 0

    total_inserted = 0
    current_ms = actual_start_ms
    pbar = tqdm(
        total=total_expected,
        desc=f"{symbol} {timeframe}",
        unit="candles",
        leave=False,
    )

    while current_ms < end_ms:
        try:
            ohlcv_list = await fetch_ohlcv(
                exchange, symbol, timeframe, current_ms, limit=200
            )
        except Exception as e:
            logger.error("Erreur fetch {} {} : {}", symbol, timeframe, e)
            await asyncio.sleep(2)
            continue

        if not ohlcv_list:
            break

        candles = []
        for ohlcv in ohlcv_list:
            try:
                candle = Candle(
                    timestamp=datetime.fromtimestamp(
                        ohlcv[0] / 1000, tz=timezone.utc
                    ),
                    open=float(ohlcv[1]),
                    high=float(ohlcv[2]),
                    low=float(ohlcv[3]),
                    close=float(ohlcv[4]),
                    volume=float(ohlcv[5]) if len(ohlcv) > 5 else 0.0,
                    symbol=symbol,
                    timeframe=tf,
                )
                candles.append(candle)
            except (ValueError, IndexError) as e:
                logger.warning("Candle ignorée: {}", e)

        if candles:
            inserted = await db.insert_candles_batch(candles)
            total_inserted += inserted
            pbar.update(len(candles))

        # Avancer au timestamp suivant
        last_ts = ohlcv_list[-1][0]
        current_ms = last_ts + interval_ms

        # Petit délai pour respecter le rate limit
        await asyncio.sleep(0.06)

    pbar.close()
    return total_inserted


async def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical klines")
    parser.add_argument("--symbol", type=str, help="Symbol spécifique (ex: BTC/USDT)")
    parser.add_argument("--timeframe", type=str, help="Timeframe spécifique (ex: 5m)")
    parser.add_argument("--days", type=int, default=180, help="Nombre de jours (défaut: 180)")
    args = parser.parse_args()

    config = get_config()
    setup_logging(level="INFO")

    db = Database()
    await db.init()

    exchange = ccxt.bitget({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

    end_date = datetime.now(tz=timezone.utc)
    start_date = end_date - timedelta(days=args.days)

    # Déterminer les paires à télécharger
    pairs: list[tuple[str, str]] = []
    for asset in config.assets:
        if args.symbol and asset.symbol != args.symbol:
            continue
        for tf in asset.timeframes:
            if args.timeframe and tf != args.timeframe:
                continue
            pairs.append((asset.symbol, tf))

    logger.info(
        "Téléchargement de {} paire(s) sur {} jours ({} → {})",
        len(pairs),
        args.days,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

    total = 0
    for symbol, tf in pairs:
        count = await fetch_symbol_timeframe(
            exchange, db, symbol, tf, start_date, end_date
        )
        total += count
        logger.info("{} {} : {} candles insérées", symbol, tf, count)

    await db.close()
    exchange.close()

    logger.info("Terminé : {} candles au total", total)


if __name__ == "__main__":
    asyncio.run(main())
