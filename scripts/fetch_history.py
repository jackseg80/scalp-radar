"""Téléchargement de l'historique des klines depuis Bitget ou Binance via ccxt REST.

Script async avec asyncio.run() comme point d'entrée.
Gère la reprise (ne télécharge que les données manquantes).

Lancement :
    uv run python -m scripts.fetch_history
    uv run python -m scripts.fetch_history --symbol BTC/USDT --timeframe 5m --days 7
    uv run python -m scripts.fetch_history --exchange binance --days 730
    uv run python -m scripts.fetch_history --exchange binance --days 1800 --symbols ADA/USDT,AVAX/USDT --timeframe 1h
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
    limit: int = 1000,
) -> list[list]:
    """Fetch OHLCV data depuis l'API REST (synchrone ccxt)."""
    return exchange.fetch_ohlcv(
        symbol, timeframe, since=since_ms, limit=limit
    )


async def fetch_bitget_uta_history_page(
    exchange: ccxt.bitget,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
) -> list[list]:
    """Fetch one bounded page from Bitget's long-history UTA endpoint.

    CCXT's standard Bitget OHLCV route uses the classic endpoint, whose 1m
    retention is short.  The UTA v3 history endpoint is public, supports
    candles older than 90 days, and accepts 100 rows per request.  Keeping
    this mode explicit prevents an accidental multi-year download during
    ordinary scanner history refreshes.
    """
    intervals = {"1h": "1H", "4h": "4H", "6h": "6H", "12h": "12H", "1d": "1D"}
    response = exchange.publicUtaGetV3MarketHistoryCandles({
        "category": "USDT-FUTURES",
        "symbol": symbol.replace("/", ""),
        "interval": intervals.get(timeframe, timeframe),
        "startTime": str(start_ms),
        "endTime": str(end_ms),
        "limit": "100",
        "type": "market",
    })
    if response.get("code") != "00000":
        raise RuntimeError(
            "Bitget UTA history failed for "
            f"{symbol} {timeframe}: {response.get('code')} {response.get('msg')}"
        )
    # The raw UTA payload is already OHLCV-shaped except for string values.
    # Keep only the requested half-open interval so a rounded endpoint result
    # can never insert a candle beyond the declared cutoff.
    return [
        row for row in response.get("data", [])
        if start_ms <= int(row[0]) < end_ms
    ]


def create_exchange(exchange_name: str) -> ccxt.Exchange:
    """Crée une instance ccxt pour l'exchange demandé."""
    if exchange_name == "bitget":
        return ccxt.bitget({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
    elif exchange_name == "binance":
        return ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
    else:
        raise ValueError(f"Exchange non supporté : {exchange_name}")


async def fetch_symbol_timeframe(
    exchange: ccxt.Exchange,
    db: Database,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    exchange_name: str = "bitget",
    resume: bool = True,
    bitget_uta_history: bool = False,
) -> int:
    """Télécharge les klines pour un (symbol, timeframe) et les persiste."""
    tf = TimeFrame.from_string(timeframe)
    interval_ms = tf.to_milliseconds()

    # Vérifier les données existantes pour la reprise
    latest = await db.get_latest_candle_timestamp(symbol, timeframe, exchange=exchange_name)
    if resume and latest and latest.timestamp() * 1000 > start_date.timestamp() * 1000:
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
            if bitget_uta_history:
                page_end_ms = min(current_ms + interval_ms * 100, end_ms)
                ohlcv_list = await fetch_bitget_uta_history_page(
                    exchange, symbol, timeframe, current_ms, page_end_ms,
                )
            else:
                ohlcv_list = await fetch_ohlcv(
                    exchange, symbol, timeframe, current_ms, limit=1000
                )
        except Exception as e:
            logger.error("Erreur fetch {} {} : {}", symbol, timeframe, e)
            await asyncio.sleep(2)
            continue

        if not ohlcv_list:
            if bitget_uta_history:
                # Continue over an empty page.  The certification snapshot
                # will reject any resulting gap, but a late-listed contract
                # must not prevent collection of its later history.
                skipped_bars = (page_end_ms - current_ms) // interval_ms
                current_ms = page_end_ms
                pbar.update(skipped_bars)
                continue
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
                    exchange=exchange_name,
                )
                candles.append(candle)
            except (ValueError, IndexError) as e:
                logger.warning("Candle ignorée: {}", e)

        if candles:
            inserted = await db.insert_candles_batch(candles)
            total_inserted += inserted
            pbar.update(len(candles))

        # Avancer au timestamp suivant
        last_ts = max(int(item[0]) for item in ohlcv_list)
        current_ms = last_ts + interval_ms

        # Petit délai pour respecter le rate limit
        await asyncio.sleep(0.06)

    pbar.close()
    return total_inserted


async def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical klines")
    parser.add_argument("--symbol", type=str, help="Filtre un symbol de config/assets.yaml (ex: BTC/USDT)")
    parser.add_argument("--symbols", type=str,
                        help="Liste de symbols séparés par des virgules, bypass assets.yaml (ex: ADA/USDT,AVAX/USDT)")
    parser.add_argument("--timeframe", type=str, help="Timeframe spécifique (ex: 5m, 1h)")
    parser.add_argument("--days", type=int, default=180, help="Nombre de jours (défaut: 180)")
    parser.add_argument("--since", help="UTC ISO start; re-fetches the full range to repair prefix/gaps")
    parser.add_argument("--until", help="UTC ISO end (exclusive); defaults to now")
    parser.add_argument("--db", default="data/scalp_radar.db")
    parser.add_argument("--exchange", type=str, default="bitget", choices=["bitget", "binance"],
                        help="Exchange source (défaut: bitget)")
    parser.add_argument("--force", action="store_true", help="Supprimer les données existantes et re-fetcher")
    parser.add_argument(
        "--bitget-uta-history", action="store_true",
        help="Use Bitget UTA v3 long-history endpoint (explicit, public, Bitget only)",
    )
    args = parser.parse_args()

    if args.bitget_uta_history and args.exchange != "bitget":
        parser.error("--bitget-uta-history requires --exchange bitget")

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

    # Déterminer les paires à télécharger
    pairs: list[tuple[str, str]] = []

    if args.symbols:
        # --symbols : bypass config.assets, construction directe
        symbols_list = [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.timeframe:
            timeframes = [args.timeframe]
        else:
            timeframes = ["1m", "5m", "15m", "1h"]
        for sym in symbols_list:
            for tf in timeframes:
                pairs.append((sym, tf))
    else:
        # Mode normal : depuis config.assets avec filtre --symbol optionnel
        for asset in config.assets:
            if args.symbol and asset.symbol != args.symbol:
                continue
            for tf in asset.timeframes:
                if args.timeframe and tf != args.timeframe:
                    continue
                pairs.append((asset.symbol, tf))

    logger.info(
        "Téléchargement {} : {} paire(s) sur {} jours ({} → {})",
        args.exchange,
        len(pairs),
        args.days,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

    # --force : supprimer les données existantes
    if args.force:
        for symbol, tf in pairs:
            deleted = await db.delete_candles(symbol, tf, exchange=args.exchange)
            if deleted:
                logger.info("Supprimé {} candles existantes pour {} {} ({})", deleted, symbol, tf, args.exchange)

    total = 0
    for symbol, tf in pairs:
        count = await fetch_symbol_timeframe(
            exchange, db, symbol, tf, start_date, end_date,
            exchange_name=args.exchange, resume=not bool(args.since),
            bitget_uta_history=args.bitget_uta_history,
        )
        total += count
        logger.info("{} {} ({}) : {} candles insérées", symbol, tf, args.exchange, count)

    await db.close()

    logger.info("Terminé : {} candles au total ({})", total, args.exchange)


if __name__ == "__main__":
    asyncio.run(main())
