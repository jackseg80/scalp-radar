"""Persistence DB des résultats portfolio backtest — Sprint 20b-UI.

Fonctions sync (sqlite3) pour le CLI portfolio_backtest.py.
Fonctions async (aiosqlite) pour l'API FastAPI.

Même pattern que optimization_db.py.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

import aiosqlite
from loguru import logger

from backend.backtesting.portfolio_engine import PortfolioResult

# ─── Colonnes de la liste (sans equity_curve pour la perf) ───────────────

_LIST_COLUMNS = (
    "id",
    "strategy_name",
    "initial_capital",
    "n_assets",
    "period_days",
    "assets",
    "exchange",
    "final_equity",
    "total_return_pct",
    "total_trades",
    "win_rate",
    "realized_pnl",
    "force_closed_pnl",
    "max_drawdown_pct",
    "peak_margin_ratio",
    "peak_open_positions",
    "peak_concurrent_assets",
    "kill_switch_triggers",
    "created_at",
    "duration_seconds",
    "label",
)

_LIST_SELECT = ", ".join(_LIST_COLUMNS)


# ─── Sérialisation ──────────────────────────────────────────────────────

def _result_to_row(
    result: PortfolioResult,
    strategy_name: str,
    exchange: str,
    kill_switch_pct: float,
    kill_switch_window_hours: int,
    duration_seconds: float | None,
    label: str | None,
    created_at: str,
) -> dict[str, Any]:
    """Convertit un PortfolioResult en dict prêt pour INSERT."""
    # Sous-échantillonner les snapshots (max 500 points)
    step = max(1, len(result.snapshots) // 500)
    equity_curve = [
        {
            "timestamp": s.timestamp.isoformat(),
            "equity": round(s.total_equity, 2),
            "capital": round(s.total_capital, 2),
            "realized_pnl": round(s.total_realized_pnl, 2),
            "unrealized_pnl": round(s.total_unrealized_pnl, 2),
            "margin_ratio": round(s.margin_ratio, 4),
            "positions": s.n_open_positions,
            "assets_active": s.n_assets_with_positions,
        }
        for s in result.snapshots[::step]
    ]

    return {
        "strategy_name": strategy_name,
        "initial_capital": result.initial_capital,
        "n_assets": result.n_assets,
        "period_days": result.period_days,
        "assets": json.dumps(result.assets),
        "exchange": exchange,
        "kill_switch_pct": kill_switch_pct,
        "kill_switch_window_hours": kill_switch_window_hours,
        "final_equity": result.final_equity,
        "total_return_pct": result.total_return_pct,
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "realized_pnl": result.realized_pnl,
        "force_closed_pnl": result.force_closed_pnl,
        "max_drawdown_pct": result.max_drawdown_pct,
        "max_drawdown_date": (
            result.max_drawdown_date.isoformat() if result.max_drawdown_date else None
        ),
        "max_drawdown_duration_hours": result.max_drawdown_duration_hours,
        "peak_margin_ratio": result.peak_margin_ratio,
        "peak_open_positions": result.peak_open_positions,
        "peak_concurrent_assets": result.peak_concurrent_assets,
        "kill_switch_triggers": result.kill_switch_triggers,
        "kill_switch_events": json.dumps(result.kill_switch_events),
        "equity_curve": json.dumps(equity_curve),
        "per_asset_results": json.dumps(result.per_asset_results),
        "created_at": created_at,
        "duration_seconds": duration_seconds,
        "label": label,
    }


_INSERT_SQL = """
    INSERT INTO portfolio_backtests (
        strategy_name, initial_capital, n_assets, period_days, assets,
        exchange, kill_switch_pct, kill_switch_window_hours,
        final_equity, total_return_pct, total_trades, win_rate,
        realized_pnl, force_closed_pnl,
        max_drawdown_pct, max_drawdown_date, max_drawdown_duration_hours,
        peak_margin_ratio, peak_open_positions, peak_concurrent_assets,
        kill_switch_triggers, kill_switch_events,
        equity_curve, per_asset_results,
        created_at, duration_seconds, label
    ) VALUES (
        :strategy_name, :initial_capital, :n_assets, :period_days, :assets,
        :exchange, :kill_switch_pct, :kill_switch_window_hours,
        :final_equity, :total_return_pct, :total_trades, :win_rate,
        :realized_pnl, :force_closed_pnl,
        :max_drawdown_pct, :max_drawdown_date, :max_drawdown_duration_hours,
        :peak_margin_ratio, :peak_open_positions, :peak_concurrent_assets,
        :kill_switch_triggers, :kill_switch_events,
        :equity_curve, :per_asset_results,
        :created_at, :duration_seconds, :label
    )
"""


# ─── Sync (CLI) ─────────────────────────────────────────────────────────

def save_result_sync(
    db_path: str,
    result: PortfolioResult,
    strategy_name: str = "grid_atr",
    exchange: str = "binance",
    kill_switch_pct: float = 30.0,
    kill_switch_window_hours: int = 24,
    duration_seconds: float | None = None,
    label: str | None = None,
) -> int:
    """Sauvegarde un PortfolioResult en DB (sync pour le CLI)."""
    created_at = datetime.now(tz=timezone.utc).isoformat()
    row = _result_to_row(
        result, strategy_name, exchange,
        kill_switch_pct, kill_switch_window_hours,
        duration_seconds, label, created_at,
    )
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(_INSERT_SQL, row)
        conn.commit()
        result_id = cursor.lastrowid
        logger.info("Portfolio backtest sauvegardé (id={})", result_id)
        return result_id
    finally:
        conn.close()


# ─── Async (API) ────────────────────────────────────────────────────────

async def save_result_async(
    db_path: str,
    result: PortfolioResult,
    strategy_name: str = "grid_atr",
    exchange: str = "binance",
    kill_switch_pct: float = 30.0,
    kill_switch_window_hours: int = 24,
    duration_seconds: float | None = None,
    label: str | None = None,
) -> int:
    """Sauvegarde un PortfolioResult en DB (async pour l'API)."""
    created_at = datetime.now(tz=timezone.utc).isoformat()
    row = _result_to_row(
        result, strategy_name, exchange,
        kill_switch_pct, kill_switch_window_hours,
        duration_seconds, label, created_at,
    )
    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute(_INSERT_SQL, row)
        await conn.commit()
        result_id = cursor.lastrowid
        logger.info("Portfolio backtest sauvegardé async (id={})", result_id)
        return result_id


async def get_backtests_async(
    db_path: str,
    limit: int = 20,
) -> list[dict]:
    """Liste des backtests (sans equity_curve pour la perf)."""
    query = f"SELECT {_LIST_SELECT} FROM portfolio_backtests ORDER BY created_at DESC LIMIT ?"
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute(query, (limit,))
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["assets"] = json.loads(d["assets"])
            results.append(d)
        return results


async def get_backtest_by_id_async(
    db_path: str,
    backtest_id: int,
) -> dict | None:
    """Détail complet d'un backtest (avec equity_curve parsée)."""
    query = "SELECT * FROM portfolio_backtests WHERE id = ?"
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute(query, (backtest_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        d = dict(row)
        # Parser les blobs JSON
        d["assets"] = json.loads(d["assets"])
        d["equity_curve"] = json.loads(d["equity_curve"])
        d["per_asset_results"] = json.loads(d["per_asset_results"])
        d["kill_switch_events"] = (
            json.loads(d["kill_switch_events"]) if d["kill_switch_events"] else []
        )
        return d


async def delete_backtest_async(
    db_path: str,
    backtest_id: int,
) -> bool:
    """Supprime un backtest. Retourne True si supprimé."""
    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute(
            "DELETE FROM portfolio_backtests WHERE id = ?", (backtest_id,)
        )
        await conn.commit()
        return cursor.rowcount > 0
