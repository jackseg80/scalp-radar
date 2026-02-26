"""Export BTC/USDT 4h candles vers CSV pour l'analyse de regime.

Lit directement la DB SQLite (table candles) et exporte en CSV standalone.
Le CSV est le dataset de reference pour toute l'analyse Sprint 50a.

Usage:
    uv run python -m scripts.export_btc_4h
    uv run python -m scripts.export_btc_4h --output data/btc_4h_custom.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path

import aiosqlite


DB_PATH = "data/scalp_radar.db"


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export BTC/USDT 4h candles vers CSV",
    )
    parser.add_argument(
        "--output", type=str, default="data/btc_4h_2017_2025.csv",
        help="Chemin du fichier CSV de sortie (defaut: data/btc_4h_2017_2025.csv)",
    )
    parser.add_argument(
        "--exchange", type=str, default="binance",
        help="Source exchange (defaut: binance)",
    )
    parser.add_argument(
        "--db", type=str, default=DB_PATH,
        help="Chemin de la base SQLite (defaut: data/scalp_radar.db)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(args.db) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute(
            "SELECT timestamp, open, high, low, close, volume "
            "FROM candles "
            "WHERE exchange = ? AND symbol = 'BTC/USDT' AND timeframe = '4h' "
            "ORDER BY timestamp ASC",
            [args.exchange],
        )
        rows = await cursor.fetchall()

    if not rows:
        print(f"Aucune candle BTC/USDT 4h trouvee (exchange={args.exchange})")
        sys.exit(1)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "open", "high", "low", "close", "volume"])
        for row in rows:
            writer.writerow([row["timestamp"], row["open"], row["high"],
                             row["low"], row["close"], row["volume"]])

    first_ts = rows[0]["timestamp"]
    last_ts = rows[-1]["timestamp"]
    print(f"{len(rows)} candles exportees -> {output_path}")
    print(f"Periode: {first_ts} -> {last_ts}")


if __name__ == "__main__":
    asyncio.run(main())
