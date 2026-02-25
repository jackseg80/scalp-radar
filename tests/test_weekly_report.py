"""Tests du rapport Telegram hebdomadaire — Sprint 49."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.alerts.weekly_reporter import (
    WeeklyReporter,
    _classify_strategies,
    generate_report,
)


# ─── HELPERS ──────────────────────────────────────────────────────────────────


def _make_config(
    live: list[str] | None = None,
    paper: list[str] | None = None,
) -> MagicMock:
    """Construit un config mock avec les stratégies spécifiées."""
    live = live or []
    paper = paper or []
    all_names = live + paper

    config = MagicMock()
    # model_fields doit être un dict (comme Pydantic)
    config.strategies.model_fields = {
        name: None for name in all_names
    }
    for name in live:
        strat = MagicMock()
        strat.enabled = True
        strat.live_eligible = True
        strat.leverage = 4
        setattr(config.strategies, name, strat)
    for name in paper:
        strat = MagicMock()
        strat.enabled = True
        strat.live_eligible = False
        strat.leverage = 6
        setattr(config.strategies, name, strat)
    return config


def _make_db(
    live_stats: dict | None = None,
    per_asset: list[dict] | None = None,
    daily_summary: dict | None = None,
    max_dd: float | None = None,
    balance_snapshots: list[dict] | None = None,
    paper_week_row: dict | None = None,
    paper_total: float = 0.0,
) -> AsyncMock:
    """Construit un db mock avec les données spécifiées."""
    db = AsyncMock()

    # Live stats
    db.get_live_stats.return_value = live_stats or {
        "total_trades": 0, "wins": 0, "win_rate": 0.0, "total_pnl": 0.0,
    }
    db.get_daily_pnl_summary.return_value = daily_summary or {
        "daily_pnl": 0.0, "total_pnl": 0.0,
    }
    db.get_live_per_asset_stats.return_value = per_asset or []
    db.get_max_drawdown_from_snapshots.return_value = max_dd
    db.get_balance_snapshots.return_value = balance_snapshots or []

    # Paper : mock _conn pour les requêtes SQL directes
    mock_conn = AsyncMock()

    if paper_week_row:
        week_cursor = AsyncMock()
        week_cursor.fetchone.return_value = paper_week_row
    else:
        week_cursor = AsyncMock()
        week_cursor.fetchone.return_value = {
            "total_trades": 0, "wins": 0, "total_pnl_week": 0.0,
        }

    total_cursor = AsyncMock()
    total_row = MagicMock()
    total_row.__getitem__ = lambda self, idx: paper_total
    total_cursor.fetchone.return_value = total_row

    # execute retourne alternativement week_cursor puis total_cursor
    mock_conn.execute = AsyncMock(side_effect=[week_cursor, total_cursor])
    db._conn = mock_conn

    return db


# ─── TESTS ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_weekly_report_format():
    """Le rapport contient les sections GLOBAL et le nom de la stratégie."""
    config = _make_config(live=["grid_atr"])
    db = _make_db(
        live_stats={
            "total_trades": 28, "wins": 22, "win_rate": 78.6, "total_pnl": 12.34,
        },
        daily_summary={"daily_pnl": 5.0, "total_pnl": -0.29},
        balance_snapshots=[{"equity": 625.0, "timestamp": "2026-02-24T12:00:00+00:00"}],
        max_dd=-2.1,
        per_asset=[
            {"symbol": "BNB/USDT", "total_trades": 8, "total_pnl": 1.33},
            {"symbol": "AAVE/USDT", "total_trades": 5, "total_pnl": 0.68},
        ],
    )

    report = await generate_report(db, config)

    assert "SCALP-RADAR" in report
    assert "GLOBAL" in report
    assert "GRID_ATR" in report
    assert "+12.34$" in report
    assert "28" in report


@pytest.mark.asyncio
async def test_weekly_report_no_trades():
    """Semaine sans trades → P&L 0 et Trades 0."""
    config = _make_config(live=["grid_atr"])
    db = _make_db(
        balance_snapshots=[{"equity": 500.0, "timestamp": "2026-02-24T12:00:00+00:00"}],
    )

    report = await generate_report(db, config)

    assert "+0.00$" in report
    assert "Trades          : 0" in report


@pytest.mark.asyncio
async def test_weekly_report_multiple_strategies():
    """2 stratégies (1 live, 1 paper) → sections séparées."""
    config = _make_config(live=["grid_atr"], paper=["grid_boltrend"])

    # DB mock avec side_effect pour 2 appels paper SQL
    db = _make_db(
        live_stats={
            "total_trades": 10, "wins": 8, "win_rate": 80.0, "total_pnl": 5.0,
        },
        daily_summary={"daily_pnl": 2.0, "total_pnl": 15.0},
        balance_snapshots=[{"equity": 625.0, "timestamp": "2026-02-24T12:00:00+00:00"}],
        paper_week_row={"total_trades": 3, "wins": 2, "total_pnl_week": 1.5},
        paper_total=4.2,
    )

    report = await generate_report(db, config)

    # Live avec ⚡
    assert "\u26a1 GRID_ATR" in report
    # Paper avec 👁️
    assert "\U0001f441\ufe0f GRID_BOLTREND (paper)" in report


@pytest.mark.asyncio
async def test_weekly_report_dry_run():
    """generate_report retourne un str sans dépendance Telegram."""
    config = _make_config(live=["grid_atr"])
    db = _make_db()

    report = await generate_report(db, config)

    assert isinstance(report, str)
    assert len(report) > 0
    assert "SCALP-RADAR" in report


@pytest.mark.asyncio
async def test_weekly_report_top_worst_assets():
    """Top et Worst assets correctement extraits."""
    config = _make_config(live=["grid_atr"])
    db = _make_db(
        live_stats={
            "total_trades": 15, "wins": 12, "win_rate": 80.0, "total_pnl": 8.5,
        },
        daily_summary={"daily_pnl": 3.0, "total_pnl": 20.0},
        balance_snapshots=[{"equity": 1000.0, "timestamp": "2026-02-24T12:00:00+00:00"}],
        per_asset=[
            {"symbol": "BNB/USDT", "total_trades": 5, "total_pnl": 3.5},
            {"symbol": "AAVE/USDT", "total_trades": 4, "total_pnl": 2.1},
            {"symbol": "SOL/USDT", "total_trades": 3, "total_pnl": 0.5},
            {"symbol": "NEAR/USDT", "total_trades": 3, "total_pnl": -0.8},
        ],
    )

    report = await generate_report(db, config)

    assert "BNB" in report
    assert "AAVE" in report
    assert "NEAR" in report
    assert "Worst" in report


# ─── TEST CLASSIFY ────────────────────────────────────────────────────────────


def test_classify_strategies():
    """_classify_strategies sépare correctement live et paper."""
    config = _make_config(live=["grid_atr", "grid_multi_tf"], paper=["grid_boltrend"])
    live, paper = _classify_strategies(config)
    assert "grid_atr" in live
    assert "grid_multi_tf" in live
    assert "grid_boltrend" in paper


# ─── TEST SCHEDULER ──────────────────────────────────────────────────────────


def test_seconds_until_next_monday():
    """Le calcul du prochain lundi est toujours dans le futur."""
    seconds = WeeklyReporter._seconds_until_next_monday_8utc()
    assert seconds > 0
    # Max 7 jours = 604800 secondes
    assert seconds <= 7 * 24 * 3600
