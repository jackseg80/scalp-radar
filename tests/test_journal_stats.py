"""Tests pour get_journal_stats() — Sprint 32."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from backend.core.database import Database


@pytest_asyncio.fixture
async def db():
    database = Database(db_path=":memory:")
    await database.init()
    yield database
    await database.close()


async def _insert_trade(db: Database, net_pnl: float, symbol: str = "BTC/USDT",
                        entry_time: str | None = None, exit_time: str | None = None,
                        strategy: str = "grid_atr", direction: str = "LONG") -> None:
    """Insert un trade directement en SQL."""
    now = datetime.now(tz=timezone.utc)
    if exit_time is None:
        exit_time = now.isoformat()
    if entry_time is None:
        entry_time = (now - timedelta(hours=2)).isoformat()

    await db._conn.execute(
        """INSERT INTO simulation_trades
           (strategy_name, symbol, direction, entry_price, exit_price,
            quantity, gross_pnl, fee_cost, slippage_cost, net_pnl,
            entry_time, exit_time, exit_reason)
           VALUES (?, ?, ?, 100.0, 101.0, 1.0, ?, 0.0, 0.0, ?, ?, ?, 'tp')""",
        [strategy, symbol, direction, net_pnl, net_pnl, entry_time, exit_time],
    )
    await db._conn.commit()


async def _insert_snapshot(db: Database, equity: float,
                           timestamp: str | None = None) -> None:
    """Insert un snapshot portfolio directement en SQL."""
    if timestamp is None:
        timestamp = datetime.now(tz=timezone.utc).isoformat()
    await db._conn.execute(
        """INSERT INTO portfolio_snapshots
           (timestamp, equity, capital, margin_used, margin_ratio,
            realized_pnl, unrealized_pnl, n_positions, n_assets, breakdown_json)
           VALUES (?, ?, 10000.0, 0.0, 0.0, 0.0, 0.0, 0, 1, '{}')""",
        [timestamp, equity],
    )
    await db._conn.commit()


class TestJournalStatsEmpty:
    """Stats sur une DB vide."""

    @pytest.mark.asyncio
    async def test_stats_empty_db(self, db: Database):
        stats = await db.get_journal_stats()
        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["profit_factor"] == 0.0
        assert stats["max_drawdown_pct"] == 0.0
        assert stats["current_streak"]["type"] == "none"
        assert stats["current_streak"]["count"] == 0
        assert stats["best_trade"] is None
        assert stats["worst_trade"] is None


class TestJournalStatsWithTrades:
    """Stats avec trades inseres."""

    @pytest.mark.asyncio
    async def test_stats_with_trades(self, db: Database):
        """3 wins + 1 loss → WR=75%, PF correct."""
        now = datetime.now(tz=timezone.utc)
        for i, pnl in enumerate([10.0, 5.0, 8.0, -6.0]):
            await _insert_trade(
                db, pnl,
                symbol="BTC/USDT" if i < 2 else "ETH/USDT",
                exit_time=(now - timedelta(hours=4 - i)).isoformat(),
                entry_time=(now - timedelta(hours=6 - i)).isoformat(),
            )

        stats = await db.get_journal_stats()
        assert stats["total_trades"] == 4
        assert stats["wins"] == 3
        assert stats["losses"] == 1
        assert stats["win_rate"] == 75.0
        assert stats["total_pnl"] == 17.0  # 10+5+8-6
        assert stats["gross_profit"] == 23.0  # 10+5+8
        assert stats["gross_loss"] == -6.0
        assert stats["profit_factor"] == round(23.0 / 6.0, 2)
        assert stats["best_trade"]["pnl"] == 10.0
        assert stats["worst_trade"]["pnl"] == -6.0

    @pytest.mark.asyncio
    async def test_stats_period_filter_7d(self, db: Database):
        """Trades vieux exclus de '7d'."""
        now = datetime.now(tz=timezone.utc)
        # Trade recent (hier)
        await _insert_trade(
            db, 10.0,
            entry_time=(now - timedelta(days=1, hours=3)).isoformat(),
            exit_time=(now - timedelta(days=1)).isoformat(),
        )
        # Trade vieux (15 jours)
        await _insert_trade(
            db, 50.0,
            entry_time=(now - timedelta(days=15, hours=3)).isoformat(),
            exit_time=(now - timedelta(days=15)).isoformat(),
        )

        stats_7d = await db.get_journal_stats(period="7d")
        assert stats_7d["total_trades"] == 1
        assert stats_7d["total_pnl"] == 10.0

        stats_all = await db.get_journal_stats(period="all")
        assert stats_all["total_trades"] == 2
        assert stats_all["total_pnl"] == 60.0

    @pytest.mark.asyncio
    async def test_stats_period_filter_today(self, db: Database):
        """Seuls trades du jour inclus dans 'today'."""
        now = datetime.now(tz=timezone.utc)
        # Trade d'aujourd'hui
        await _insert_trade(
            db, 5.0,
            entry_time=(now - timedelta(hours=1)).isoformat(),
            exit_time=now.isoformat(),
        )
        # Trade d'hier
        await _insert_trade(
            db, 20.0,
            entry_time=(now - timedelta(days=1, hours=3)).isoformat(),
            exit_time=(now - timedelta(days=1)).isoformat(),
        )

        stats = await db.get_journal_stats(period="today")
        assert stats["total_trades"] == 1
        assert stats["total_pnl"] == 5.0


class TestJournalStatsStreak:
    """Tests specifiques pour le streak."""

    @pytest.mark.asyncio
    async def test_stats_streak_win(self, db: Database):
        """3 wins consecutifs recents."""
        now = datetime.now(tz=timezone.utc)
        # Loss ancienne, puis 3 wins recentes
        pnls = [-5.0, 10.0, 8.0, 3.0]
        for i, pnl in enumerate(pnls):
            await _insert_trade(
                db, pnl,
                exit_time=(now - timedelta(hours=len(pnls) - i)).isoformat(),
                entry_time=(now - timedelta(hours=len(pnls) - i + 2)).isoformat(),
            )

        stats = await db.get_journal_stats()
        assert stats["current_streak"]["type"] == "win"
        assert stats["current_streak"]["count"] == 3

    @pytest.mark.asyncio
    async def test_stats_streak_loss(self, db: Database):
        """2 losses consecutives recentes."""
        now = datetime.now(tz=timezone.utc)
        pnls = [10.0, -3.0, -7.0]
        for i, pnl in enumerate(pnls):
            await _insert_trade(
                db, pnl,
                exit_time=(now - timedelta(hours=len(pnls) - i)).isoformat(),
                entry_time=(now - timedelta(hours=len(pnls) - i + 2)).isoformat(),
            )

        stats = await db.get_journal_stats()
        assert stats["current_streak"]["type"] == "loss"
        assert stats["current_streak"]["count"] == 2


class TestJournalStatsProfitFactor:
    """Tests profit factor edge cases."""

    @pytest.mark.asyncio
    async def test_stats_profit_factor_no_losses(self, db: Database):
        """PF = 0.0 quand pas de losses (guard division par zero)."""
        await _insert_trade(db, 10.0)
        await _insert_trade(db, 5.0)

        stats = await db.get_journal_stats()
        assert stats["profit_factor"] == 0.0
        assert stats["gross_loss"] == 0.0
        assert stats["losses"] == 0


class TestJournalStatsDrawdown:
    """Tests drawdown depuis snapshots."""

    @pytest.mark.asyncio
    async def test_stats_drawdown_from_snapshots(self, db: Database):
        """Peak-to-trough sur equity."""
        now = datetime.now(tz=timezone.utc)
        # Insert un trade pour que stats ne retourne pas vide
        await _insert_trade(db, 1.0)

        # Snapshots : 10000 → 10500 (peak) → 9975 (trough) → 10200
        equities = [10000, 10500, 9975, 10200]
        for i, eq in enumerate(equities):
            await _insert_snapshot(
                db, float(eq),
                timestamp=(now - timedelta(hours=len(equities) - i)).isoformat(),
            )

        stats = await db.get_journal_stats()
        # Drawdown = (10500 - 9975) / 10500 * 100 = 5.0%
        assert stats["max_drawdown_pct"] == 5.0

    @pytest.mark.asyncio
    async def test_stats_drawdown_no_snapshots(self, db: Database):
        """0 snapshots → drawdown = 0.0."""
        await _insert_trade(db, 1.0)

        stats = await db.get_journal_stats()
        assert stats["max_drawdown_pct"] == 0.0
