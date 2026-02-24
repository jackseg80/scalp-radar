"""Tests Sprint 46 — P&L Jour, filtre stratégie, balance snapshots, Max DD."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from backend.core.database import Database


@pytest_asyncio.fixture
async def db():
    """DB en mémoire initialisée avec toutes les tables."""
    database = Database(db_path=":memory:")
    await database.init()
    yield database
    await database.close()


def _make_live_trade(**overrides) -> dict:
    """Helper pour créer un live trade."""
    base = {
        "timestamp": "2026-02-25T10:00:00+00:00",
        "strategy_name": "grid_atr",
        "symbol": "BTC/USDT:USDT",
        "direction": "LONG",
        "trade_type": "cycle_close",
        "side": "sell",
        "quantity": 0.01,
        "price": 96000.0,
        "order_id": "order_123",
        "fee": 0.57,
        "pnl": 5.0,
        "pnl_pct": 1.2,
        "leverage": 7,
        "grid_level": 2,
        "context": "grid",
    }
    base.update(overrides)
    return base


def _make_snapshot(**overrides) -> dict:
    """Helper pour créer un balance snapshot."""
    base = {
        "timestamp": "2026-02-25T10:00:00+00:00",
        "strategy_name": "grid_atr",
        "balance": 1000.0,
        "unrealized_pnl": 0.0,
        "margin_used": 0.0,
        "equity": 1000.0,
    }
    base.update(overrides)
    return base


# ─── Fix 1 : daily-pnl-summary ──────────────────────────────────────────


class TestDailyPnlSummary:
    @pytest.mark.asyncio
    async def test_daily_pnl_returns_today(self, db):
        """get_daily_pnl_summary retourne le P&L du jour."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%dT12:00:00+00:00")
        await db.insert_live_trade(_make_live_trade(timestamp=today, pnl=2.5))
        await db.insert_live_trade(_make_live_trade(timestamp=today, pnl=-0.5))

        result = await db.get_daily_pnl_summary()
        assert result["daily_pnl"] == 2.0
        assert result["total_pnl"] == 2.0
        assert result["first_trade_date"] is not None

    @pytest.mark.asyncio
    async def test_daily_pnl_no_trades(self, db):
        """get_daily_pnl_summary retourne 0 sans trades."""
        result = await db.get_daily_pnl_summary()
        assert result["daily_pnl"] == 0
        assert result["total_pnl"] == 0
        assert result["first_trade_date"] is None

    @pytest.mark.asyncio
    async def test_daily_pnl_separates_today_from_history(self, db):
        """P&L jour vs P&L total sont distincts."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%dT12:00:00+00:00")
        await db.insert_live_trade(_make_live_trade(timestamp=today, pnl=3.0))
        await db.insert_live_trade(
            _make_live_trade(timestamp="2026-02-20T10:00:00+00:00", pnl=-10.0),
        )

        result = await db.get_daily_pnl_summary()
        assert result["daily_pnl"] == 3.0
        assert result["total_pnl"] == -7.0


# ─── Fix 2 : filtre stratégie ───────────────────────────────────────────


class TestLiveStatsStrategyFilter:
    @pytest.mark.asyncio
    async def test_filter_by_strategy(self, db):
        """get_live_stats filtre par stratégie."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%dT12:00:00+00:00")
        await db.insert_live_trade(
            _make_live_trade(timestamp=today, strategy_name="grid_atr", pnl=5.0),
        )
        await db.insert_live_trade(
            _make_live_trade(timestamp=today, strategy_name="grid_multi_tf", pnl=-3.0),
        )

        stats_all = await db.get_live_stats(period="all")
        assert stats_all["total_trades"] == 2

        stats_atr = await db.get_live_stats(period="all", strategy="grid_atr")
        assert stats_atr["total_trades"] == 1
        assert stats_atr["total_pnl"] == 5.0

        stats_mtf = await db.get_live_stats(period="all", strategy="grid_multi_tf")
        assert stats_mtf["total_trades"] == 1
        assert stats_mtf["total_pnl"] == -3.0


# ─── Fix 3 : balance snapshots ──────────────────────────────────────────


class TestBalanceSnapshots:
    @pytest.mark.asyncio
    async def test_insert_and_read_roundtrip(self, db):
        """Insert un snapshot et le relire."""
        snap = _make_snapshot()
        await db.insert_balance_snapshot(snap)

        snapshots = await db.get_balance_snapshots(days=30)
        assert len(snapshots) == 1
        assert snapshots[0]["equity"] == 1000.0
        assert snapshots[0]["strategy_name"] == "grid_atr"

    @pytest.mark.asyncio
    async def test_filter_by_strategy(self, db):
        """get_balance_snapshots filtre par stratégie."""
        await db.insert_balance_snapshot(
            _make_snapshot(strategy_name="grid_atr", equity=1000.0),
        )
        await db.insert_balance_snapshot(
            _make_snapshot(strategy_name="grid_multi_tf", equity=2000.0),
        )

        snaps_atr = await db.get_balance_snapshots(strategy="grid_atr", days=30)
        assert len(snaps_atr) == 1
        assert snaps_atr[0]["equity"] == 1000.0

        snaps_all = await db.get_balance_snapshots(days=30)
        assert len(snaps_all) == 2


class TestMaxDrawdownFromSnapshots:
    @pytest.mark.asyncio
    async def test_max_dd_calculation(self, db):
        """Calcul correct du max drawdown depuis les snapshots."""
        # Equity: 1000 → 1100 → 900 → 950
        # Peak at 1100, drawdown to 900 = (900-1100)/1100 = -18.18%
        for eq, h in [(1000, "08"), (1100, "09"), (900, "10"), (950, "11")]:
            await db.insert_balance_snapshot(
                _make_snapshot(
                    timestamp=f"2026-02-25T{h}:00:00+00:00",
                    equity=eq,
                    balance=eq,
                ),
            )

        max_dd = await db.get_max_drawdown_from_snapshots()
        assert max_dd is not None
        assert abs(max_dd - (-18.18)) < 0.1

    @pytest.mark.asyncio
    async def test_max_dd_no_snapshots(self, db):
        """Retourne None sans snapshots."""
        max_dd = await db.get_max_drawdown_from_snapshots()
        assert max_dd is None

    @pytest.mark.asyncio
    async def test_max_dd_single_snapshot(self, db):
        """Retourne None avec un seul snapshot (pas de comparaison possible)."""
        await db.insert_balance_snapshot(_make_snapshot())
        max_dd = await db.get_max_drawdown_from_snapshots()
        assert max_dd is None

    @pytest.mark.asyncio
    async def test_max_dd_monotone_up(self, db):
        """Max DD = 0 si equity monte toujours."""
        for eq, h in [(1000, "08"), (1050, "09"), (1100, "10")]:
            await db.insert_balance_snapshot(
                _make_snapshot(
                    timestamp=f"2026-02-25T{h}:00:00+00:00",
                    equity=eq,
                    balance=eq,
                ),
            )

        max_dd = await db.get_max_drawdown_from_snapshots()
        assert max_dd == 0.0


class TestBalanceSnapshotPersist:
    @pytest.mark.asyncio
    async def test_persist_balance_snapshot_best_effort(self):
        """_persist_balance_snapshot ne crashe pas si DB échoue."""
        from backend.execution.executor import Executor

        executor = Executor.__new__(Executor)
        executor._db = AsyncMock()
        executor._db.insert_balance_snapshot = AsyncMock(
            side_effect=Exception("DB error"),
        )
        executor._strategy_name = "grid_atr"

        # Ne doit PAS lever d'exception
        await executor._persist_balance_snapshot(1000.0)
        executor._db.insert_balance_snapshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_balance_snapshot_no_db(self):
        """_persist_balance_snapshot ne fait rien sans DB."""
        from backend.execution.executor import Executor

        executor = Executor.__new__(Executor)
        executor._db = None
        executor._strategy_name = "grid_atr"

        # Ne doit PAS lever d'exception
        await executor._persist_balance_snapshot(1000.0)


# ─── Bugfix : cycle_close dans get_live_daily_pnl ───────────────────────


class TestDailyPnlCycleClose:
    @pytest.mark.asyncio
    async def test_cycle_close_included_in_daily_pnl(self, db):
        """cycle_close est bien comptabilisé dans get_live_daily_pnl."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%dT12:00:00+00:00")
        await db.insert_live_trade(
            _make_live_trade(
                timestamp=today, trade_type="cycle_close", pnl=5.0,
            ),
        )
        await db.insert_live_trade(
            _make_live_trade(
                timestamp=today, trade_type="tp_close", pnl=3.0,
            ),
        )

        daily = await db.get_live_daily_pnl(days=1)
        assert len(daily) == 1
        assert daily[0]["pnl"] == 8.0  # 5 + 3
