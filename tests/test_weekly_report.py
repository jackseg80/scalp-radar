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
    config = MagicMock()
    config.strategies.model_fields = {name: None for name in live + paper}
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


def _make_db() -> MagicMock:
    """Construit un db mock minimal (conn non-None pour l'assert)."""
    db = MagicMock()
    db._conn = MagicMock()
    return db


def _default_live_stats() -> dict:
    return {
        "total_trades": 0, "wins": 0, "win_rate": 0.0,
        "total_pnl": 0.0, "per_asset": [],
    }


def _default_paper_stats() -> dict:
    return {
        "total_trades": 0, "wins": 0, "win_rate": 0.0,
        "total_pnl_week": 0.0, "total_pnl_all": 0.0,
    }


# ─── TESTS ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_weekly_report_format():
    """Le rapport contient les sections GLOBAL et le nom de la stratégie."""
    config = _make_config(live=["grid_atr"])
    db = _make_db()
    live_stats = {
        "total_trades": 28, "wins": 22, "win_rate": 78.6, "total_pnl": 12.34,
        "per_asset": [
            {"symbol": "BNB/USDT", "total_pnl": 1.33},
            {"symbol": "AAVE/USDT", "total_pnl": 0.68},
        ],
    }

    with (
        patch("backend.alerts.weekly_reporter._live_week_stats",
              new=AsyncMock(return_value=live_stats)),
        patch("backend.alerts.weekly_reporter._live_total_pnl",
              new=AsyncMock(return_value=-0.29)),
        patch("backend.alerts.weekly_reporter._latest_balance",
              new=AsyncMock(return_value=625.0)),
        patch("backend.alerts.weekly_reporter._max_drawdown_week",
              new=AsyncMock(return_value=-2.1)),
        patch("backend.alerts.weekly_reporter._compute_uptime",
              new=AsyncMock(return_value=99.8)),
    ):
        report = await generate_report(db, config)

    assert "SCALP-RADAR" in report
    assert "GLOBAL" in report
    assert "GRID_ATR" in report
    assert "+12.34$" in report
    assert "28" in report
    assert "625" in report   # balance
    assert "x4" in report    # leverage


@pytest.mark.asyncio
async def test_weekly_report_no_trades():
    """Semaine sans trades → P&L 0 et Trades 0."""
    config = _make_config(live=["grid_atr"])
    db = _make_db()

    with (
        patch("backend.alerts.weekly_reporter._live_week_stats",
              new=AsyncMock(return_value=_default_live_stats())),
        patch("backend.alerts.weekly_reporter._live_total_pnl",
              new=AsyncMock(return_value=0.0)),
        patch("backend.alerts.weekly_reporter._latest_balance",
              new=AsyncMock(return_value=None)),
        patch("backend.alerts.weekly_reporter._max_drawdown_week",
              new=AsyncMock(return_value=None)),
        patch("backend.alerts.weekly_reporter._compute_uptime",
              new=AsyncMock(return_value=None)),
    ):
        report = await generate_report(db, config)

    assert "+0.00$" in report
    assert "Trades          : 0" in report


@pytest.mark.asyncio
async def test_weekly_report_multiple_strategies():
    """1 stratégie live + 1 paper → sections séparées, GLOBAL = live only."""
    config = _make_config(live=["grid_atr"], paper=["grid_boltrend"])
    db = _make_db()
    live_stats = {
        "total_trades": 10, "wins": 8, "win_rate": 80.0, "total_pnl": 5.0,
        "per_asset": [],
    }

    with (
        patch("backend.alerts.weekly_reporter._live_week_stats",
              new=AsyncMock(return_value=live_stats)),
        patch("backend.alerts.weekly_reporter._live_total_pnl",
              new=AsyncMock(return_value=15.0)),
        patch("backend.alerts.weekly_reporter._latest_balance",
              new=AsyncMock(return_value=625.0)),
        patch("backend.alerts.weekly_reporter._max_drawdown_week",
              new=AsyncMock(return_value=None)),
        patch("backend.alerts.weekly_reporter._paper_week_stats",
              new=AsyncMock(return_value={
                  "total_trades": 3, "wins": 2, "win_rate": 66.7,
                  "total_pnl_week": 1.5, "total_pnl_all": 4.2,
              })),
        patch("backend.alerts.weekly_reporter._compute_uptime",
              new=AsyncMock(return_value=None)),
    ):
        report = await generate_report(db, config)

    # LIVE avec ⚡
    assert "\u26a1 GRID_ATR" in report
    # PAPER avec 👁️ — ne pollue pas le GLOBAL
    assert "GRID_BOLTREND (paper)" in report
    # GLOBAL = live only (5.0$ semaine, pas 5+1.5$)
    assert "+5.00$" in report


@pytest.mark.asyncio
async def test_weekly_report_dry_run():
    """generate_report retourne un str sans aucune dépendance Telegram."""
    config = _make_config(live=["grid_atr"])
    db = _make_db()

    with (
        patch("backend.alerts.weekly_reporter._live_week_stats",
              new=AsyncMock(return_value=_default_live_stats())),
        patch("backend.alerts.weekly_reporter._live_total_pnl",
              new=AsyncMock(return_value=0.0)),
        patch("backend.alerts.weekly_reporter._latest_balance",
              new=AsyncMock(return_value=None)),
        patch("backend.alerts.weekly_reporter._max_drawdown_week",
              new=AsyncMock(return_value=None)),
        patch("backend.alerts.weekly_reporter._compute_uptime",
              new=AsyncMock(return_value=None)),
    ):
        report = await generate_report(db, config)

    assert isinstance(report, str)
    assert len(report) > 0
    assert "SCALP-RADAR" in report


@pytest.mark.asyncio
async def test_weekly_report_top_worst_assets():
    """Top et Worst assets correctement extraits depuis per_asset."""
    config = _make_config(live=["grid_atr"])
    db = _make_db()
    live_stats = {
        "total_trades": 15, "wins": 12, "win_rate": 80.0, "total_pnl": 8.5,
        "per_asset": [
            {"symbol": "BNB/USDT", "total_pnl": 3.5},
            {"symbol": "AAVE/USDT", "total_pnl": 2.1},
            {"symbol": "SOL/USDT", "total_pnl": 0.5},
            {"symbol": "NEAR/USDT", "total_pnl": -0.8},
        ],
    }

    with (
        patch("backend.alerts.weekly_reporter._live_week_stats",
              new=AsyncMock(return_value=live_stats)),
        patch("backend.alerts.weekly_reporter._live_total_pnl",
              new=AsyncMock(return_value=20.0)),
        patch("backend.alerts.weekly_reporter._latest_balance",
              new=AsyncMock(return_value=1000.0)),
        patch("backend.alerts.weekly_reporter._max_drawdown_week",
              new=AsyncMock(return_value=-1.2)),
        patch("backend.alerts.weekly_reporter._compute_uptime",
              new=AsyncMock(return_value=None)),
    ):
        report = await generate_report(db, config)

    assert "BNB" in report
    assert "AAVE" in report
    assert "NEAR" in report
    assert "Worst" in report
    assert "Top" in report


# ─── TEST CURRENT WEEK ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_weekly_report_current_week():
    """Le flag current_week=True change le label de la période."""
    config = _make_config(live=["grid_atr"])
    db = _make_db()

    with (
        patch("backend.alerts.weekly_reporter._live_week_stats",
              new=AsyncMock(return_value=_default_live_stats())),
        patch("backend.alerts.weekly_reporter._live_total_pnl",
              new=AsyncMock(return_value=0.0)),
        patch("backend.alerts.weekly_reporter._latest_balance",
              new=AsyncMock(return_value=None)),
        patch("backend.alerts.weekly_reporter._max_drawdown_week",
              new=AsyncMock(return_value=None)),
        patch("backend.alerts.weekly_reporter._compute_uptime",
              new=AsyncMock(return_value=None)),
    ):
        report = await generate_report(db, config, current_week=True)

    assert "Semaine en cours" in report


# ─── TEST CLASSIFY ────────────────────────────────────────────────────────────


def test_classify_strategies():
    """_classify_strategies sépare correctement live et paper."""
    config = _make_config(
        live=["grid_atr", "grid_multi_tf"],
        paper=["grid_boltrend"],
    )
    live, paper = _classify_strategies(config)
    assert "grid_atr" in live
    assert "grid_multi_tf" in live
    assert "grid_boltrend" in paper


# ─── TEST SCHEDULER ──────────────────────────────────────────────────────────


def test_seconds_until_next_monday_positive():
    """Le calcul du prochain lundi est toujours dans le futur, max 7 jours."""
    seconds = WeeklyReporter._seconds_until_next_monday_8utc()
    assert 0 < seconds <= 7 * 24 * 3600
