"""Backfill candles depuis l'API publique Binance (sans clé API).

Télécharge l'historique OHLCV via GET /api/v3/klines et stocke en DB
avec exchange='binance'. Idempotent (INSERT OR IGNORE).

Lancement :
    uv run python -m scripts.backfill_candles
    uv run python -m scripts.backfill_candles --symbol BTC/USDT
    uv run python -m scripts.backfill_candles --since 2023-01-01
    uv run python -m scripts.backfill_candles --timeframe 4h
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

import httpx
from loguru import logger
from tqdm import tqdm

from backend.core.config import get_config
from backend.core.database import Database
from backend.core.logging_setup import setup_logging
from backend.core.models import Candle, TimeFrame

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_RETRIES = 3
REQUEST_DELAY_S = 0.1  # 100ms entre requêtes (rate limit Binance)


def _symbol_to_binance(symbol: str) -> str:
    """Convertit BTC/USDT -> BTCUSDT pour l'API Binance."""
    return symbol.replace("/", "")


def _tf_to_binance(timeframe: str) -> str:
    """Convertit le timeframe interne vers le format Binance.

    Les noms sont identiques pour 1m, 5m, 15m, 1h, 4h.
    """
    valid = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"}
    if timeframe not in valid:
        raise ValueError(f"Timeframe '{timeframe}' non supporté par Binance. Valides : {valid}")
    return timeframe


def _tf_to_ms(timeframe: str) -> int:
    """Retourne la durée du timeframe en millisecondes."""
    mapping = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "2h": 7_200_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }
    return mapping[timeframe]


async def fetch_klines(
    client: httpx.AsyncClient,
    binance_symbol: str,
    interval: str,
    start_ms: int,
    limit: int = 1000,
) -> list[list]:
    """Fetch klines depuis l'API publique Binance avec retry."""
    params = {
        "symbol": binance_symbol,
        "interval": interval,
        "startTime": start_ms,
        "limit": limit,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.get(BINANCE_KLINES_URL, params=params)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as exc:
            if attempt == MAX_RETRIES:
                logger.error(
                    "Échec après {} tentatives pour {} : {}",
                    MAX_RETRIES, binance_symbol, exc,
                )
                raise
            delay = 2 ** (attempt - 1)  # 1s, 2s, 4s
            logger.warning(
                "Retry {}/{} pour {} ({}), attente {}s...",
                attempt, MAX_RETRIES, binance_symbol, type(exc).__name__, delay,
            )
            await asyncio.sleep(delay)
    return []  # unreachable


async def backfill_symbol(
    db: Database,
    client: httpx.AsyncClient,
    symbol: str,
    timeframe: str,
    since: datetime,
) -> int:
    """Backfill un (symbol, timeframe) depuis Binance. Retourne le nombre de candles insérées."""
    binance_symbol = _symbol_to_binance(symbol)
    interval = _tf_to_binance(timeframe)
    interval_ms = _tf_to_ms(timeframe)
    tf = TimeFrame.from_string(timeframe)

    # Reprise incrémentale
    latest = await db.get_latest_candle_timestamp(symbol, timeframe, exchange="binance")
    if latest and latest.timestamp() * 1000 > since.timestamp() * 1000:
        start_ms = int(latest.timestamp() * 1000) + interval_ms
        logger.info(
            "Reprise {} {} depuis {}",
            symbol, timeframe,
            datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
        )
    else:
        start_ms = int(since.timestamp() * 1000)

    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    total_expected = max(0, (now_ms - start_ms) // interval_ms)
    if total_expected <= 0:
        logger.info("Pas de données manquantes pour {} {} (binance)", symbol, timeframe)
        return 0

    total_inserted = 0
    current_ms = start_ms
    pbar = tqdm(
        total=total_expected,
        desc=f"{symbol} {timeframe}",
        unit="candles",
        leave=False,
    )

    while current_ms < now_ms:
        klines = await fetch_klines(client, binance_symbol, interval, current_ms)
        if not klines:
            break

        candles: list[Candle] = []
        for k in klines:
            try:
                candles.append(Candle(
                    timestamp=datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    symbol=symbol,
                    timeframe=tf,
                    exchange="binance",
                ))
            except (ValueError, IndexError) as exc:
                logger.warning("Kline ignorée : {}", exc)

        if candles:
            inserted = await db.insert_candles_batch(candles)
            total_inserted += inserted
            pbar.update(len(candles))

        # Avancer au prochain bloc
        last_open_time = klines[-1][0]
        current_ms = last_open_time + interval_ms

        await asyncio.sleep(REQUEST_DELAY_S)

    pbar.close()

    # Log final avec bornes temporelles
    first_ts = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    last_ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    logger.info(
        "{} {} : {} candles insérées ({} → {})",
        symbol, timeframe, total_inserted, first_ts, last_ts,
    )
    return total_inserted


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill candles depuis l'API publique Binance",
    )
    parser.add_argument("--symbol", type=str, help="Symbol spécifique (ex: BTC/USDT)")
    parser.add_argument(
        "--since", type=str, default="2020-08-01",
        help="Date de début (défaut: 2020-08-01)",
    )
    parser.add_argument(
        "--timeframe", type=str, default="1h",
        help="Timeframe (défaut: 1h). Ex: 1m, 5m, 15m, 1h, 4h",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")

    since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    config = get_config()
    symbols = [a.symbol for a in config.assets]
    if args.symbol:
        if args.symbol not in symbols:
            logger.warning("{} pas dans assets.yaml, on continue quand même", args.symbol)
        symbols = [args.symbol]

    db = Database()
    await db.init()

    logger.info(
        "Backfill Binance : {} asset(s), timeframe={}, depuis {}",
        len(symbols), args.timeframe, args.since,
    )

    total = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        for symbol in symbols:
            count = await backfill_symbol(db, client, symbol, args.timeframe, since)
            total += count

    await db.close()
    logger.info("Terminé : {} candles insérées au total (binance)", total)


if __name__ == "__main__":
    asyncio.run(main())
