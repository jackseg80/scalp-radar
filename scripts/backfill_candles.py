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


def _internal_gap_windows(
    timestamps: list[datetime],
    *,
    start: datetime,
    end: datetime,
    interval_ms: int,
) -> list[tuple[datetime, datetime]]:
    """Return exact missing candle windows strictly inside an observed series."""
    interval = interval_ms / 1000
    ordered = sorted(ts for ts in timestamps if start <= ts <= end)
    windows: list[tuple[datetime, datetime]] = []
    for previous, current in zip(ordered, ordered[1:]):
        if (current - previous).total_seconds() > interval:
            windows.append((
                datetime.fromtimestamp(previous.timestamp() + interval, tz=timezone.utc),
                current,
            ))
    return windows


def _candles_from_klines(
    klines: list[list],
    *,
    symbol: str,
    timeframe: TimeFrame,
) -> list[Candle]:
    candles: list[Candle] = []
    for k in klines:
        try:
            candles.append(Candle(
                timestamp=datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                open=float(k[1]), high=float(k[2]), low=float(k[3]),
                close=float(k[4]), volume=float(k[5]), symbol=symbol,
                timeframe=timeframe, exchange="binance",
            ))
        except (ValueError, IndexError) as exc:
            logger.warning("Kline ignorée : {}", exc)
    return candles


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

        candles = _candles_from_klines(
            klines, symbol=symbol, timeframe=tf,
        )

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


async def repair_internal_gaps(
    db: Database,
    client: httpx.AsyncClient,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> int:
    """Fetch and upsert only missing internal candles in the requested range."""
    tf = TimeFrame.from_string(timeframe)
    interval_ms = _tf_to_ms(timeframe)
    existing = await db.get_candles(
        symbol, timeframe, start=start, end=end,
        limit=1_000_000, exchange="binance",
    )
    windows = _internal_gap_windows(
        [candle.timestamp for candle in existing],
        start=start, end=end, interval_ms=interval_ms,
    )
    if not windows:
        logger.info("Aucun gap interne pour {} {}", symbol, timeframe)
        return 0

    total_inserted = 0
    for gap_start, gap_end in windows:
        remaining = int((gap_end - gap_start).total_seconds() * 1000 // interval_ms)
        current_ms = int(gap_start.timestamp() * 1000)
        while remaining > 0:
            klines = await fetch_klines(
                client, _symbol_to_binance(symbol), _tf_to_binance(timeframe),
                current_ms, limit=min(remaining, 1000),
            )
            if not klines:
                logger.error(
                    "Gap non réparé pour {} {} à {}",
                    symbol, timeframe, gap_start.isoformat(),
                )
                break
            candles = [
                candle for candle in _candles_from_klines(
                    klines, symbol=symbol, timeframe=tf,
                )
                if gap_start <= candle.timestamp < gap_end
            ]
            if not candles:
                logger.error(
                    "Gap non réparé pour {} {} à {} : source sans bougie",
                    symbol, timeframe, gap_start.isoformat(),
                )
                break
            total_inserted += await db.insert_candles_batch(candles)
            fetched = len(klines)
            current_ms = int(klines[-1][0]) + interval_ms
            remaining -= fetched
            await asyncio.sleep(REQUEST_DELAY_S)

    logger.info(
        "{} {} : {} bougies de gaps internes réparées",
        symbol, timeframe, total_inserted,
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
    parser.add_argument(
        "--repair-gaps", action="store_true",
        help="Réparer aussi les trous internes dans la plage --since",
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
            if args.repair_gaps:
                total += await repair_internal_gaps(
                    db, client, symbol, args.timeframe, since,
                    datetime.now(tz=timezone.utc),
                )

    await db.close()
    logger.info("Terminé : {} candles insérées au total (binance)", total)


if __name__ == "__main__":
    asyncio.run(main())
