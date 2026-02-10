"""Couche d'abstraction SQLite async pour Scalp Radar.

Gère le stockage des candles, signaux, trades et état de session.
100% async avec aiosqlite.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite
from loguru import logger

from backend.core.models import (
    Candle,
    Direction,
    MarketRegime,
    SessionState,
    Signal,
    SignalStrength,
    TimeFrame,
    Trade,
)


class Database:
    """Abstraction SQLite async."""

    def __init__(self, db_path: str | Path = "data/scalp_radar.db") -> None:
        self.db_path = str(db_path)
        self._conn: Optional[aiosqlite.Connection] = None

    async def init(self) -> None:
        """Crée la connexion et les tables."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._create_tables()
        logger.info("Database initialisée : {}", self.db_path)

    async def _create_tables(self) -> None:
        assert self._conn is not None
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                vwap REAL,
                mark_price REAL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            );

            CREATE INDEX IF NOT EXISTS idx_candles_lookup
                ON candles (symbol, timeframe, timestamp);

            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                strength TEXT NOT NULL,
                score REAL NOT NULL,
                entry_price REAL NOT NULL,
                tp_price REAL,
                sl_price REAL,
                regime TEXT,
                metadata_json TEXT
            );

            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                leverage INTEGER NOT NULL,
                gross_pnl REAL NOT NULL,
                fee_cost REAL NOT NULL,
                slippage_cost REAL NOT NULL,
                net_pnl REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                strategy TEXT NOT NULL,
                regime TEXT
            );

            CREATE TABLE IF NOT EXISTS session_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                start_time TEXT NOT NULL,
                total_pnl REAL NOT NULL DEFAULT 0,
                total_trades INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                max_drawdown REAL NOT NULL DEFAULT 0,
                available_margin REAL NOT NULL DEFAULT 0,
                kill_switch_triggered INTEGER NOT NULL DEFAULT 0,
                last_update TEXT NOT NULL
            );
        """)
        await self._conn.commit()

    # ─── CANDLES ────────────────────────────────────────────────────────────

    async def insert_candles_batch(self, candles: list[Candle]) -> int:
        """Insère un batch de candles. Ignore les doublons. Retourne le nombre inséré."""
        if not candles:
            return 0
        assert self._conn is not None
        data = [
            (
                c.symbol,
                c.timeframe.value,
                c.timestamp.isoformat(),
                c.open,
                c.high,
                c.low,
                c.close,
                c.volume,
                c.vwap,
                c.mark_price,
            )
            for c in candles
        ]
        cursor = await self._conn.executemany(
            """INSERT OR IGNORE INTO candles
               (symbol, timeframe, timestamp, open, high, low, close, volume, vwap, mark_price)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            data,
        )
        await self._conn.commit()
        return cursor.rowcount

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 500,
    ) -> list[Candle]:
        """Récupère des candles avec filtres optionnels."""
        assert self._conn is not None
        query = "SELECT * FROM candles WHERE symbol = ? AND timeframe = ?"
        params: list[object] = [symbol, timeframe]

        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())

        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [
            Candle(
                timestamp=datetime.fromisoformat(row["timestamp"]),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                symbol=row["symbol"],
                timeframe=TimeFrame.from_string(row["timeframe"]),
                vwap=row["vwap"],
                mark_price=row["mark_price"],
            )
            for row in rows
        ]

    async def get_latest_candle_timestamp(
        self, symbol: str, timeframe: str
    ) -> Optional[datetime]:
        """Retourne le timestamp de la dernière candle stockée."""
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT MAX(timestamp) as ts FROM candles WHERE symbol = ? AND timeframe = ?",
            [symbol, timeframe],
        )
        row = await cursor.fetchone()
        if row and row["ts"]:
            return datetime.fromisoformat(row["ts"])
        return None

    async def delete_candles(self, symbol: str, timeframe: str) -> int:
        """Supprime toutes les candles pour un (symbol, timeframe). Retourne le nombre supprimé."""
        assert self._conn is not None
        cursor = await self._conn.execute(
            "DELETE FROM candles WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe),
        )
        await self._conn.commit()
        return cursor.rowcount

    # ─── SIGNALS ────────────────────────────────────────────────────────────

    async def insert_signal(self, signal: Signal) -> None:
        assert self._conn is not None
        import json
        await self._conn.execute(
            """INSERT INTO signals
               (timestamp, strategy, symbol, direction, strength, score,
                entry_price, tp_price, sl_price, regime, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal.timestamp.isoformat(),
                signal.strategy_name,
                signal.symbol,
                signal.direction.value,
                signal.strength.value,
                signal.score,
                signal.entry_price,
                signal.tp_price,
                signal.sl_price,
                signal.market_regime.value if signal.market_regime else None,
                json.dumps(signal.metadata),
            ),
        )
        await self._conn.commit()

    # ─── TRADES ─────────────────────────────────────────────────────────────

    async def insert_trade(self, trade: Trade) -> None:
        assert self._conn is not None
        await self._conn.execute(
            """INSERT INTO trades
               (id, symbol, direction, entry_price, exit_price, quantity, leverage,
                gross_pnl, fee_cost, slippage_cost, net_pnl,
                entry_time, exit_time, strategy, regime)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade.id,
                trade.symbol,
                trade.direction.value,
                trade.entry_price,
                trade.exit_price,
                trade.quantity,
                trade.leverage,
                trade.gross_pnl,
                trade.fee_cost,
                trade.slippage_cost,
                trade.net_pnl,
                trade.entry_time.isoformat(),
                trade.exit_time.isoformat(),
                trade.strategy_name,
                trade.market_regime.value if trade.market_regime else None,
            ),
        )
        await self._conn.commit()

    # ─── SESSION STATE ──────────────────────────────────────────────────────

    async def save_session_state(self, state: SessionState) -> None:
        assert self._conn is not None
        await self._conn.execute(
            """INSERT OR REPLACE INTO session_state
               (id, start_time, total_pnl, total_trades, wins, losses,
                max_drawdown, available_margin, kill_switch_triggered, last_update)
               VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                state.start_time.isoformat(),
                state.total_pnl,
                state.total_trades,
                state.wins,
                state.losses,
                state.max_drawdown,
                state.available_margin,
                1 if state.kill_switch_triggered else 0,
                datetime.now().isoformat(),
            ),
        )
        await self._conn.commit()

    async def load_session_state(self) -> Optional[SessionState]:
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT * FROM session_state WHERE id = 1"
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return SessionState(
            start_time=datetime.fromisoformat(row["start_time"]),
            total_pnl=row["total_pnl"],
            total_trades=row["total_trades"],
            wins=row["wins"],
            losses=row["losses"],
            max_drawdown=row["max_drawdown"],
            available_margin=row["available_margin"],
            kill_switch_triggered=bool(row["kill_switch_triggered"]),
        )

    # ─── LIFECYCLE ──────────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("Database fermée")
