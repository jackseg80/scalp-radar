"""Couche d'abstraction SQLite async pour Scalp Radar.

Gère le stockage des candles, signaux, trades et état de session.
100% async avec aiosqlite.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timedelta, timezone
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
        self._conn = await aiosqlite.connect(self.db_path, timeout=10)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._migrate_candles_exchange()
        await self._create_tables()
        logger.info("Database initialisée : {}", self.db_path)

    async def _migrate_candles_exchange(self) -> None:
        """Migration idempotente : ajoute la colonne exchange à candles si absente.

        Recrée la table avec la nouvelle PK (exchange, symbol, timeframe, timestamp).
        Backup automatique horodaté avant migration.
        """
        assert self._conn is not None
        cursor = await self._conn.execute("PRAGMA table_info(candles)")
        columns = await cursor.fetchall()
        if not columns:
            return  # Table n'existe pas encore, _create_tables la créera
        col_names = [col["name"] for col in columns]
        if "exchange" in col_names:
            return  # Déjà migré

        # Backup automatique horodaté
        db_file = Path(self.db_path)
        if db_file.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = db_file.with_name(f"{db_file.stem}_backup_{ts}{db_file.suffix}")
            shutil.copy2(str(db_file), str(backup_path))
            logger.info("Backup DB avant migration : {}", backup_path)

        logger.info("Migration candles : ajout colonne exchange...")
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS candles_new (
                exchange TEXT NOT NULL DEFAULT 'bitget',
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
                PRIMARY KEY (exchange, symbol, timeframe, timestamp)
            );

            INSERT OR IGNORE INTO candles_new
                (exchange, symbol, timeframe, timestamp, open, high, low, close, volume, vwap, mark_price)
            SELECT 'bitget', symbol, timeframe, timestamp, open, high, low, close, volume, vwap, mark_price
            FROM candles;

            DROP TABLE candles;
            ALTER TABLE candles_new RENAME TO candles;

            CREATE INDEX IF NOT EXISTS idx_candles_exchange
                ON candles (exchange, symbol, timeframe, timestamp);
        """)
        await self._conn.commit()
        logger.info("Migration candles terminée (colonne exchange ajoutée)")

    async def _create_tables(self) -> None:
        assert self._conn is not None
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS candles (
                exchange TEXT NOT NULL DEFAULT 'bitget',
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
                PRIMARY KEY (exchange, symbol, timeframe, timestamp)
            );

            CREATE INDEX IF NOT EXISTS idx_candles_exchange
                ON candles (exchange, symbol, timeframe, timestamp);

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
                metadata_json TEXT,
                source TEXT DEFAULT 'backtest'
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
                regime TEXT,
                source TEXT DEFAULT 'backtest'
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
        await self._create_sprint7b_tables()
        await self._create_sprint13_tables()
        await self._create_sprint14_tables()
        await self._create_simulator_trades_table()
        await self._create_portfolio_tables()
        await self._create_journal_tables()
        await self._conn.commit()

    async def _create_sprint7b_tables(self) -> None:
        """Tables Sprint 7b : funding rates et open interest historiques."""
        assert self._conn is not None
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS funding_rates (
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL DEFAULT 'binance',
                timestamp INTEGER NOT NULL,
                funding_rate REAL NOT NULL,
                PRIMARY KEY (symbol, exchange, timestamp)
            );

            CREATE INDEX IF NOT EXISTS idx_funding_rates_lookup
                ON funding_rates (symbol, exchange, timestamp);

            CREATE TABLE IF NOT EXISTS open_interest (
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL DEFAULT 'binance',
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                oi REAL NOT NULL,
                oi_value REAL NOT NULL,
                PRIMARY KEY (symbol, exchange, timeframe, timestamp)
            );

            CREATE INDEX IF NOT EXISTS idx_oi_lookup
                ON open_interest (symbol, exchange, timeframe, timestamp);
        """)

    async def _create_sprint13_tables(self) -> None:
        """Tables Sprint 13 : résultats WFO en DB."""
        assert self._conn is not None
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS optimization_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                asset TEXT NOT NULL,
                timeframe TEXT NOT NULL,

                -- Métadonnées
                created_at TEXT NOT NULL,
                duration_seconds REAL,

                -- Grading
                grade TEXT NOT NULL,
                total_score REAL NOT NULL,
                oos_sharpe REAL,
                consistency REAL,
                oos_is_ratio REAL,
                dsr REAL,
                param_stability REAL,
                monte_carlo_pvalue REAL,
                mc_underpowered INTEGER DEFAULT 0,
                n_windows INTEGER NOT NULL,
                n_distinct_combos INTEGER,

                -- JSON blobs
                best_params TEXT NOT NULL,
                wfo_windows TEXT,
                monte_carlo_summary TEXT,
                validation_summary TEXT,
                warnings TEXT,

                -- Flags
                is_latest INTEGER DEFAULT 1,
                source TEXT DEFAULT 'local',

                UNIQUE(strategy_name, asset, timeframe, created_at)
            );

            CREATE INDEX IF NOT EXISTS idx_opt_strategy_asset
                ON optimization_results(strategy_name, asset);
            CREATE INDEX IF NOT EXISTS idx_opt_grade
                ON optimization_results(grade);
            CREATE INDEX IF NOT EXISTS idx_opt_latest
                ON optimization_results(is_latest) WHERE is_latest = 1;
            CREATE INDEX IF NOT EXISTS idx_opt_created
                ON optimization_results(created_at);
        """)
        await self._migrate_optimization_source()
        await self._migrate_regime_analysis()

    async def _migrate_optimization_source(self) -> None:
        """Migration idempotente : ajoute la colonne source à optimization_results si absente."""
        assert self._conn is not None
        cursor = await self._conn.execute("PRAGMA table_info(optimization_results)")
        columns = await cursor.fetchall()
        if not columns:
            return  # Table n'existe pas encore
        col_names = [col["name"] for col in columns]
        if "source" in col_names:
            return  # Déjà migré
        await self._conn.execute(
            "ALTER TABLE optimization_results ADD COLUMN source TEXT DEFAULT 'local'"
        )
        await self._conn.commit()
        logger.info("Migration optimization_results : colonne source ajoutée")

    async def _migrate_regime_analysis(self) -> None:
        """Migration idempotente : ajoute regime_analysis à optimization_results (Sprint 15b)."""
        assert self._conn is not None
        cursor = await self._conn.execute("PRAGMA table_info(optimization_results)")
        columns = await cursor.fetchall()
        if not columns:
            return
        col_names = [col["name"] for col in columns]
        if "regime_analysis" in col_names:
            return
        await self._conn.execute(
            "ALTER TABLE optimization_results ADD COLUMN regime_analysis TEXT"
        )
        await self._conn.commit()
        logger.info("Migration optimization_results : colonne regime_analysis ajoutée")

    async def _create_sprint14_tables(self) -> None:
        """Tables Sprint 14 : jobs d'optimisation WFO."""
        assert self._conn is not None
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS optimization_jobs (
                id TEXT PRIMARY KEY,
                strategy_name TEXT NOT NULL,
                asset TEXT NOT NULL,
                timeframe TEXT NOT NULL,

                -- Statut
                status TEXT NOT NULL DEFAULT 'pending',
                progress_pct REAL DEFAULT 0,
                current_phase TEXT DEFAULT '',

                -- Paramètres
                params_override TEXT,

                -- Timing
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                duration_seconds REAL,

                -- Résultat
                result_id INTEGER,
                error_message TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_status
                ON optimization_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_created
                ON optimization_jobs(created_at);
        """)
        await self._create_sprint14b_tables()

    async def _create_sprint14b_tables(self) -> None:
        """Tables Sprint 14b : résultats détaillés de chaque combo WFO."""
        assert self._conn is not None
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS wfo_combo_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                optimization_result_id INTEGER NOT NULL,
                params TEXT NOT NULL,
                oos_sharpe REAL,
                oos_return_pct REAL,
                oos_trades INTEGER,
                oos_win_rate REAL,
                is_sharpe REAL,
                is_return_pct REAL,
                is_trades INTEGER,
                consistency REAL,
                oos_is_ratio REAL,
                is_best INTEGER DEFAULT 0,
                n_windows_evaluated INTEGER,
                FOREIGN KEY (optimization_result_id) REFERENCES optimization_results(id)
            );

            CREATE INDEX IF NOT EXISTS idx_combo_opt_id
                ON wfo_combo_results(optimization_result_id);
            CREATE INDEX IF NOT EXISTS idx_combo_best
                ON wfo_combo_results(is_best) WHERE is_best = 1;
        """)

    async def _create_simulator_trades_table(self) -> None:
        """Table pour les trades du simulateur (paper trading)."""
        assert self._conn is not None
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS simulation_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                gross_pnl REAL NOT NULL,
                fee_cost REAL NOT NULL,
                slippage_cost REAL NOT NULL,
                net_pnl REAL NOT NULL,
                exit_reason TEXT NOT NULL,
                market_regime TEXT,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_sim_trades_strategy
                ON simulation_trades(strategy_name);
            CREATE INDEX IF NOT EXISTS idx_sim_trades_exit_time
                ON simulation_trades(exit_time);
        """)

    async def _create_portfolio_tables(self) -> None:
        """Tables Sprint 20b-UI : résultats portfolio backtest."""
        assert self._conn is not None
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS portfolio_backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                initial_capital REAL NOT NULL,
                n_assets INTEGER NOT NULL,
                period_days INTEGER NOT NULL,
                assets TEXT NOT NULL,
                exchange TEXT NOT NULL DEFAULT 'binance',
                kill_switch_pct REAL NOT NULL DEFAULT 30.0,
                kill_switch_window_hours INTEGER NOT NULL DEFAULT 24,
                final_equity REAL NOT NULL,
                total_return_pct REAL NOT NULL,
                total_trades INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                force_closed_pnl REAL NOT NULL,
                max_drawdown_pct REAL NOT NULL,
                max_drawdown_date TEXT,
                max_drawdown_duration_hours REAL NOT NULL,
                peak_margin_ratio REAL NOT NULL,
                peak_open_positions INTEGER NOT NULL,
                peak_concurrent_assets INTEGER NOT NULL,
                kill_switch_triggers INTEGER NOT NULL DEFAULT 0,
                kill_switch_events TEXT,
                equity_curve TEXT NOT NULL,
                per_asset_results TEXT NOT NULL,
                created_at TEXT NOT NULL,
                duration_seconds REAL,
                label TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_portfolio_created
                ON portfolio_backtests(created_at);
        """)

    # ─── CANDLES ────────────────────────────────────────────────────────────

    async def insert_candles_batch(self, candles: list[Candle]) -> int:
        """Insère un batch de candles. Ignore les doublons. Retourne le nombre inséré."""
        if not candles:
            return 0
        assert self._conn is not None
        data = [
            (
                c.exchange,
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
               (exchange, symbol, timeframe, timestamp, open, high, low, close, volume, vwap, mark_price)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
        exchange: str = "bitget",
    ) -> list[Candle]:
        """Récupère des candles avec filtres optionnels."""
        assert self._conn is not None
        query = "SELECT * FROM candles WHERE exchange = ? AND symbol = ? AND timeframe = ?"
        params: list[object] = [exchange, symbol, timeframe]

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
                exchange=row["exchange"],
                vwap=row["vwap"],
                mark_price=row["mark_price"],
            )
            for row in rows
        ]

    async def get_latest_candle_timestamp(
        self, symbol: str, timeframe: str, exchange: str = "bitget",
    ) -> Optional[datetime]:
        """Retourne le timestamp de la dernière candle stockée."""
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT MAX(timestamp) as ts FROM candles WHERE exchange = ? AND symbol = ? AND timeframe = ?",
            [exchange, symbol, timeframe],
        )
        row = await cursor.fetchone()
        if row and row["ts"]:
            return datetime.fromisoformat(row["ts"])
        return None

    async def delete_candles(
        self, symbol: str, timeframe: str, exchange: str = "bitget",
    ) -> int:
        """Supprime toutes les candles pour un (exchange, symbol, timeframe). Retourne le nombre supprimé."""
        assert self._conn is not None
        cursor = await self._conn.execute(
            "DELETE FROM candles WHERE exchange = ? AND symbol = ? AND timeframe = ?",
            (exchange, symbol, timeframe),
        )
        await self._conn.commit()
        return cursor.rowcount

    async def get_recent_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 50,
    ) -> list[Candle]:
        """Retourne les N dernières bougies (tous exchanges), triées par timestamp ASC.

        Utilisé pour le warm-up des stratégies grid au démarrage du Simulator.
        Pas de filtre exchange : les données 1h peuvent venir de Binance ou Bitget.
        """
        assert self._conn is not None
        cursor = await self._conn.execute(
            """SELECT exchange, symbol, timeframe, timestamp,
                      open, high, low, close, volume, vwap, mark_price
               FROM candles
               WHERE symbol = ? AND timeframe = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (symbol, timeframe, limit),
        )
        rows = await cursor.fetchall()

        # Inverser pour ASC (les plus anciennes en premier)
        rows = list(reversed(rows))

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
                exchange=row["exchange"],
                vwap=row["vwap"],
                mark_price=row["mark_price"],
            )
            for row in rows
        ]

    # ─── SIGNALS ────────────────────────────────────────────────────────────

    async def insert_signal(self, signal: Signal, source: str = "backtest") -> None:
        assert self._conn is not None
        import json
        await self._conn.execute(
            """INSERT INTO signals
               (timestamp, strategy, symbol, direction, strength, score,
                entry_price, tp_price, sl_price, regime, metadata_json, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                source,
            ),
        )
        await self._conn.commit()

    # ─── TRADES ─────────────────────────────────────────────────────────────

    async def insert_trade(self, trade: Trade, source: str = "backtest") -> None:
        assert self._conn is not None
        await self._conn.execute(
            """INSERT INTO trades
               (id, symbol, direction, entry_price, exit_price, quantity, leverage,
                gross_pnl, fee_cost, slippage_cost, net_pnl,
                entry_time, exit_time, strategy, regime, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                source,
            ),
        )
        await self._conn.commit()

    # ─── FUNDING RATES ─────────────────────────────────────────────────────

    async def insert_funding_rates_batch(self, rates: list[dict]) -> int:
        """Insère un batch de funding rates. Ignore les doublons. Retourne le nombre inséré."""
        if not rates:
            return 0
        assert self._conn is not None
        data = [
            (r["symbol"], r["exchange"], r["timestamp"], r["funding_rate"])
            for r in rates
        ]
        cursor = await self._conn.executemany(
            """INSERT OR IGNORE INTO funding_rates
               (symbol, exchange, timestamp, funding_rate)
               VALUES (?, ?, ?, ?)""",
            data,
        )
        await self._conn.commit()
        return cursor.rowcount

    async def get_funding_rates(
        self,
        symbol: str,
        exchange: str = "binance",
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict]:
        """Retourne les funding rates triés par timestamp ASC."""
        assert self._conn is not None
        query = "SELECT * FROM funding_rates WHERE symbol = ? AND exchange = ?"
        params: list[object] = [symbol, exchange]

        if start_ts is not None:
            query += " AND timestamp >= ?"
            params.append(start_ts)
        if end_ts is not None:
            query += " AND timestamp <= ?"
            params.append(end_ts)

        query += " ORDER BY timestamp ASC"
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [
            {
                "symbol": row["symbol"],
                "exchange": row["exchange"],
                "timestamp": row["timestamp"],
                "funding_rate": row["funding_rate"],
            }
            for row in rows
        ]

    async def get_latest_funding_timestamp(
        self, symbol: str, exchange: str = "binance",
    ) -> int | None:
        """Retourne le timestamp du dernier funding rate (pour reprise incrémentale)."""
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT MAX(timestamp) as ts FROM funding_rates WHERE symbol = ? AND exchange = ?",
            [symbol, exchange],
        )
        row = await cursor.fetchone()
        if row and row["ts"]:
            return row["ts"]
        return None

    # ─── OPEN INTEREST ───────────────────────────────────────────────────

    async def insert_oi_batch(self, records: list[dict]) -> int:
        """Insère un batch d'OI records. Ignore les doublons. Retourne le nombre inséré."""
        if not records:
            return 0
        assert self._conn is not None
        data = [
            (r["symbol"], r["exchange"], r["timeframe"], r["timestamp"], r["oi"], r["oi_value"])
            for r in records
        ]
        cursor = await self._conn.executemany(
            """INSERT OR IGNORE INTO open_interest
               (symbol, exchange, timeframe, timestamp, oi, oi_value)
               VALUES (?, ?, ?, ?, ?, ?)""",
            data,
        )
        await self._conn.commit()
        return cursor.rowcount

    async def get_open_interest(
        self,
        symbol: str,
        timeframe: str = "5m",
        exchange: str = "binance",
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict]:
        """Retourne les records OI triés par timestamp ASC."""
        assert self._conn is not None
        query = "SELECT * FROM open_interest WHERE symbol = ? AND exchange = ? AND timeframe = ?"
        params: list[object] = [symbol, exchange, timeframe]

        if start_ts is not None:
            query += " AND timestamp >= ?"
            params.append(start_ts)
        if end_ts is not None:
            query += " AND timestamp <= ?"
            params.append(end_ts)

        query += " ORDER BY timestamp ASC"
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [
            {
                "symbol": row["symbol"],
                "exchange": row["exchange"],
                "timeframe": row["timeframe"],
                "timestamp": row["timestamp"],
                "oi": row["oi"],
                "oi_value": row["oi_value"],
            }
            for row in rows
        ]

    async def get_latest_oi_timestamp(
        self, symbol: str, timeframe: str = "5m", exchange: str = "binance",
    ) -> int | None:
        """Retourne le timestamp du dernier record OI (pour reprise incrémentale)."""
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT MAX(timestamp) as ts FROM open_interest WHERE symbol = ? AND exchange = ? AND timeframe = ?",
            [symbol, exchange, timeframe],
        )
        row = await cursor.fetchone()
        if row and row["ts"]:
            return row["ts"]
        return None

    # ─── SIMULATION TRADES ──────────────────────────────────────────────────

    async def get_simulation_trades(
        self,
        strategy_name: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Retourne les trades du simulateur triés par exit_time DESC."""
        assert self._conn is not None
        query = "SELECT * FROM simulation_trades"
        params: list[object] = []

        if strategy_name is not None:
            query += " WHERE strategy_name = ?"
            params.append(strategy_name)

        query += " ORDER BY exit_time DESC LIMIT ?"
        params.append(limit)

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "strategy": row["strategy_name"],
                "symbol": row["symbol"],
                "direction": row["direction"],
                "entry_price": row["entry_price"],
                "exit_price": row["exit_price"],
                "quantity": row["quantity"],
                "gross_pnl": row["gross_pnl"],
                "fee_cost": row["fee_cost"],
                "slippage_cost": row["slippage_cost"],
                "net_pnl": row["net_pnl"],
                "exit_reason": row["exit_reason"],
                "market_regime": row["market_regime"],
                "entry_time": row["entry_time"],
                "exit_time": row["exit_time"],
            }
            for row in rows
        ]

    async def get_trade_counts_by_strategy(self) -> dict[str, int]:
        """Nombre de trades par stratégie dans simulation_trades.

        Utilisé par l'AdaptiveSelector pour connaître l'historique
        même après un --clean (qui supprime le JSON state mais pas la DB).
        """
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT strategy_name, COUNT(*) as cnt "
            "FROM simulation_trades GROUP BY strategy_name"
        )
        rows = await cursor.fetchall()
        return {row["strategy_name"]: row["cnt"] for row in rows}

    # ─── REGIME PROFILES (Sprint 27) ────────────────────────────────────────

    async def get_regime_profiles(
        self, strategy_name: str,
    ) -> dict[str, dict]:
        """Charge les regime_analysis WFO pour une stratégie.

        Retourne {symbol: {regime: {avg_oos_sharpe: ..., ...}}} pour les
        résultats is_latest=1 qui ont un regime_analysis non NULL.
        """
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT asset, regime_analysis FROM optimization_results "
            "WHERE strategy_name = ? AND is_latest = 1 "
            "AND regime_analysis IS NOT NULL",
            (strategy_name,),
        )
        rows = await cursor.fetchall()
        profiles: dict[str, dict] = {}
        for row in rows:
            try:
                ra = json.loads(row["regime_analysis"])
                if isinstance(ra, dict):
                    profiles[row["asset"]] = ra
            except (json.JSONDecodeError, TypeError):
                continue
        return profiles

    async def get_equity_curve_from_trades(self, since: str | None = None) -> list[dict]:
        """Calcule l'equity curve depuis les trades en DB (robuste aux restarts).

        Retourne une liste de points {timestamp, capital, trade_pnl}.
        """
        assert self._conn is not None
        query = "SELECT net_pnl, exit_time FROM simulation_trades"
        params: list[object] = []
        if since:
            query += " WHERE exit_time > ?"
            params.append(since)
        query += " ORDER BY exit_time ASC"

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()

        capital = 10_000.0
        # Si on filtre par since, on doit d'abord calculer le capital de base
        if since and rows:
            base_cursor = await self._conn.execute(
                "SELECT COALESCE(SUM(net_pnl), 0) FROM simulation_trades WHERE exit_time <= ?",
                [since],
            )
            base_row = await base_cursor.fetchone()
            capital += base_row[0]

        equity = []
        for row in rows:
            capital += row["net_pnl"]
            equity.append({
                "timestamp": row["exit_time"],
                "capital": round(capital, 2),
                "trade_pnl": round(row["net_pnl"], 2),
            })
        return equity

    async def clear_simulation_trades(self) -> int:
        """Vide la table simulation_trades. Retourne le nombre de lignes supprimées."""
        assert self._conn is not None
        cursor = await self._conn.execute("DELETE FROM simulation_trades")
        await self._conn.commit()
        return cursor.rowcount

    # ─── JOURNAL STATS (Sprint 32) ──────────────────────────────────────────

    async def get_journal_stats(self, period: str = "all") -> dict:
        """Stats agregees du journal de trading.

        Calcule win rate, profit factor, drawdown, streak, etc.
        depuis simulation_trades + portfolio_snapshots.
        """
        assert self._conn is not None

        # 1. Calculer le timestamp 'since' selon la periode
        since: str | None = None
        now = datetime.now(tz=timezone.utc)
        period_days = 0
        if period == "today":
            since = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            period_days = 1
        elif period == "7d":
            since = (now - timedelta(days=7)).isoformat()
            period_days = 7
        elif period == "30d":
            since = (now - timedelta(days=30)).isoformat()
            period_days = 30

        # 2. Query trades
        if since:
            cursor = await self._conn.execute(
                "SELECT net_pnl, exit_time, entry_time, symbol FROM simulation_trades "
                "WHERE exit_time >= ? ORDER BY exit_time ASC",
                [since],
            )
        else:
            cursor = await self._conn.execute(
                "SELECT net_pnl, exit_time, entry_time, symbol FROM simulation_trades "
                "ORDER BY exit_time ASC",
            )
        rows = await cursor.fetchall()

        total_trades = len(rows)
        if total_trades == 0:
            return {
                "period": period,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "profit_factor": 0.0,
                "best_trade": None,
                "worst_trade": None,
                "avg_duration_hours": 0.0,
                "trades_per_day": 0.0,
                "current_streak": {"type": "none", "count": 0},
                "max_drawdown_pct": 0.0,
            }

        # 3. Calculs
        pnls = [row["net_pnl"] for row in rows]
        wins = sum(1 for p in pnls if p > 0)
        losses = total_trades - wins
        win_rate = round(wins / total_trades * 100, 1)
        total_pnl = round(sum(pnls), 2)
        gross_profit = round(sum(p for p in pnls if p > 0), 2)
        gross_loss = round(sum(p for p in pnls if p <= 0), 2)
        profit_factor = round(gross_profit / abs(gross_loss), 2) if gross_loss < 0 else 0.0

        # Best / worst trade
        best_idx = max(range(total_trades), key=lambda i: pnls[i])
        worst_idx = min(range(total_trades), key=lambda i: pnls[i])
        best_trade = {"symbol": rows[best_idx]["symbol"], "pnl": round(pnls[best_idx], 2)}
        worst_trade = {"symbol": rows[worst_idx]["symbol"], "pnl": round(pnls[worst_idx], 2)}

        # Duree moyenne
        durations = []
        for row in rows:
            try:
                entry = datetime.fromisoformat(row["entry_time"])
                exit_ = datetime.fromisoformat(row["exit_time"])
                durations.append((exit_ - entry).total_seconds() / 3600)
            except (ValueError, TypeError):
                pass
        avg_duration_hours = round(sum(durations) / len(durations), 1) if durations else 0.0

        # Trades par jour
        if period_days > 0:
            trades_per_day = round(total_trades / period_days, 1)
        else:
            # Calculer depuis le premier au dernier trade
            try:
                first = datetime.fromisoformat(rows[0]["exit_time"])
                last = datetime.fromisoformat(rows[-1]["exit_time"])
                span_days = max(1, (last - first).days + 1)
                trades_per_day = round(total_trades / span_days, 1)
            except (ValueError, TypeError):
                trades_per_day = 0.0

        # Streak (depuis le plus recent)
        streak_type = "win" if pnls[-1] > 0 else "loss"
        streak_count = 0
        for p in reversed(pnls):
            if (streak_type == "win" and p > 0) or (streak_type == "loss" and p <= 0):
                streak_count += 1
            else:
                break

        # 4. Drawdown depuis portfolio_snapshots
        max_drawdown_pct = 0.0
        if since:
            snap_cursor = await self._conn.execute(
                "SELECT equity FROM portfolio_snapshots WHERE timestamp >= ? ORDER BY timestamp ASC",
                [since],
            )
        else:
            snap_cursor = await self._conn.execute(
                "SELECT equity FROM portfolio_snapshots ORDER BY timestamp ASC",
            )
        snap_rows = await snap_cursor.fetchall()
        if snap_rows:
            peak = snap_rows[0]["equity"]
            for sr in snap_rows:
                eq = sr["equity"]
                if eq > peak:
                    peak = eq
                if peak > 0:
                    dd = (peak - eq) / peak * 100
                    if dd > max_drawdown_pct:
                        max_drawdown_pct = dd
            max_drawdown_pct = round(max_drawdown_pct, 1)

        return {
            "period": period,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "avg_duration_hours": avg_duration_hours,
            "trades_per_day": trades_per_day,
            "current_streak": {"type": streak_type, "count": streak_count},
            "max_drawdown_pct": max_drawdown_pct,
        }

    # ─── PER-ASSET STATS (Sprint Journal V2) ────────────────────────────────

    async def get_journal_per_asset_stats(self, period: str = "all") -> list[dict]:
        """Stats par asset depuis simulation_trades (paper trades only)."""
        assert self._conn is not None

        since: str | None = None
        now = datetime.now(tz=timezone.utc)
        if period == "today":
            since = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        elif period == "7d":
            since = (now - timedelta(days=7)).isoformat()
        elif period == "30d":
            since = (now - timedelta(days=30)).isoformat()

        if since:
            cursor = await self._conn.execute(
                """SELECT symbol,
                          COUNT(*) as total_trades,
                          SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
                          SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END) as losses,
                          ROUND(SUM(net_pnl), 2) as total_pnl,
                          ROUND(AVG(net_pnl), 2) as avg_pnl
                   FROM simulation_trades
                   WHERE exit_time >= ?
                   GROUP BY symbol
                   ORDER BY total_pnl DESC""",
                [since],
            )
        else:
            cursor = await self._conn.execute(
                """SELECT symbol,
                          COUNT(*) as total_trades,
                          SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
                          SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END) as losses,
                          ROUND(SUM(net_pnl), 2) as total_pnl,
                          ROUND(AVG(net_pnl), 2) as avg_pnl
                   FROM simulation_trades
                   GROUP BY symbol
                   ORDER BY total_pnl DESC""",
            )

        rows = await cursor.fetchall()
        result = []
        for row in rows:
            total = row["total_trades"]
            wins = row["wins"]
            result.append({
                "symbol": row["symbol"],
                "total_trades": total,
                "wins": wins,
                "losses": row["losses"],
                "win_rate": round(wins / total * 100, 1) if total > 0 else 0.0,
                "total_pnl": row["total_pnl"],
                "avg_pnl": row["avg_pnl"],
            })
        return result

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

    # ─── JOURNAL (Sprint 25) ─────────────────────────────────────────────

    async def _create_journal_tables(self) -> None:
        """Tables Sprint 25 : journal d'activité live (snapshots + events)."""
        assert self._conn is not None
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL,
                capital REAL NOT NULL,
                margin_used REAL NOT NULL,
                margin_ratio REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                n_positions INTEGER NOT NULL,
                n_assets INTEGER NOT NULL,
                breakdown_json TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp
                ON portfolio_snapshots (timestamp);

            CREATE TABLE IF NOT EXISTS position_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                event_type TEXT NOT NULL,
                level INTEGER,
                direction TEXT NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                unrealized_pnl REAL,
                margin_used REAL,
                metadata_json TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_events_timestamp
                ON position_events (timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_symbol
                ON position_events (strategy_name, symbol);
        """)

    async def _execute_with_retry(
        self, query: str, params: tuple, *, max_retries: int = 3
    ) -> None:
        """Execute + commit avec retry sur 'database is locked'."""
        assert self._conn is not None
        backoff = 0.1
        for attempt in range(max_retries):
            try:
                await self._conn.execute(query, params)
                await self._conn.commit()
                return
            except Exception as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(
                        "DB locked (attempt {}/{}), retry in {}s",
                        attempt + 1, max_retries, backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    raise

    async def insert_portfolio_snapshot(self, snapshot: dict) -> None:
        """Insère un snapshot portfolio."""
        await self._execute_with_retry(
            """INSERT INTO portfolio_snapshots
               (timestamp, equity, capital, margin_used, margin_ratio,
                realized_pnl, unrealized_pnl, n_positions, n_assets, breakdown_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                snapshot["timestamp"],
                snapshot["equity"],
                snapshot["capital"],
                snapshot["margin_used"],
                snapshot["margin_ratio"],
                snapshot["realized_pnl"],
                snapshot["unrealized_pnl"],
                snapshot["n_positions"],
                snapshot["n_assets"],
                json.dumps(snapshot.get("breakdown")) if snapshot.get("breakdown") else None,
            ),
        )

    async def get_portfolio_snapshots(
        self, since: str | None = None, until: str | None = None, limit: int = 2000
    ) -> list[dict]:
        """Récupère les snapshots portfolio, triés par timestamp ASC."""
        assert self._conn is not None
        query = "SELECT * FROM portfolio_snapshots WHERE 1=1"
        params: list = []
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        if until:
            query += " AND timestamp <= ?"
            params.append(until)
        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_latest_snapshot(self) -> dict | None:
        """Récupère le snapshot le plus récent."""
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def insert_position_event(self, event: dict) -> None:
        """Insère un événement de position."""
        await self._execute_with_retry(
            """INSERT INTO position_events
               (timestamp, strategy_name, symbol, event_type, level,
                direction, price, quantity, unrealized_pnl, margin_used, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event["timestamp"],
                event["strategy_name"],
                event["symbol"],
                event["event_type"],
                event.get("level"),
                event["direction"],
                event["price"],
                event["quantity"],
                event.get("unrealized_pnl"),
                event.get("margin_used"),
                json.dumps(event.get("metadata")) if event.get("metadata") else None,
            ),
        )

    async def get_position_events(
        self, since: str | None = None, limit: int = 100,
        strategy: str | None = None, symbol: str | None = None,
    ) -> list[dict]:
        """Récupère les événements de position, triés par timestamp DESC."""
        assert self._conn is not None
        query = "SELECT * FROM position_events WHERE 1=1"
        params: list = []
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        if strategy:
            query += " AND strategy_name = ?"
            params.append(strategy)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # ─── LIFECYCLE ──────────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("Database fermée")
