"""Persistence DB des résultats portfolio backtest — Sprint 20b-UI.

Fonctions sync (sqlite3) pour le CLI portfolio_backtest.py.
Fonctions async (aiosqlite) pour l'API FastAPI.
Sync local → serveur : push_portfolio_to_server() + save_portfolio_from_payload_sync().

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
    "leverage",
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
        "leverage": result.leverage,
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
        exchange, leverage, kill_switch_pct, kill_switch_window_hours,
        final_equity, total_return_pct, total_trades, win_rate,
        realized_pnl, force_closed_pnl,
        max_drawdown_pct, max_drawdown_date, max_drawdown_duration_hours,
        peak_margin_ratio, peak_open_positions, peak_concurrent_assets,
        kill_switch_triggers, kill_switch_events,
        equity_curve, per_asset_results,
        created_at, duration_seconds, label
    ) VALUES (
        :strategy_name, :initial_capital, :n_assets, :period_days, :assets,
        :exchange, :leverage, :kill_switch_pct, :kill_switch_window_hours,
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
    kill_switch_pct: float = 45.0,
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
    kill_switch_pct: float = 45.0,
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


# ─── Sync local → serveur ────────────────────────────────────────────────


def build_portfolio_payload_from_row(row: dict) -> dict:
    """Construit un payload POST depuis une row DB portfolio_backtests.

    Les colonnes DB correspondent au format POST attendu.
    Les JSON blobs restent en string (le serveur les stocke tels quels).
    """
    return {
        "strategy_name": row["strategy_name"],
        "initial_capital": row["initial_capital"],
        "n_assets": row["n_assets"],
        "period_days": row["period_days"],
        "assets": row["assets"],
        "exchange": row.get("exchange", "binance"),
        "kill_switch_pct": row.get("kill_switch_pct", 45.0),
        "kill_switch_window_hours": row.get("kill_switch_window_hours", 24),
        "final_equity": row["final_equity"],
        "total_return_pct": row["total_return_pct"],
        "total_trades": row["total_trades"],
        "win_rate": row["win_rate"],
        "realized_pnl": row["realized_pnl"],
        "force_closed_pnl": row["force_closed_pnl"],
        "max_drawdown_pct": row["max_drawdown_pct"],
        "max_drawdown_date": row.get("max_drawdown_date"),
        "max_drawdown_duration_hours": row["max_drawdown_duration_hours"],
        "peak_margin_ratio": row["peak_margin_ratio"],
        "peak_open_positions": row["peak_open_positions"],
        "peak_concurrent_assets": row["peak_concurrent_assets"],
        "kill_switch_triggers": row.get("kill_switch_triggers", 0),
        "kill_switch_events": row.get("kill_switch_events"),
        "equity_curve": row["equity_curve"],
        "per_asset_results": row["per_asset_results"],
        "created_at": row["created_at"],
        "duration_seconds": row.get("duration_seconds"),
        "label": row.get("label"),
        "source": row.get("source", "local"),
    }


def save_portfolio_from_payload_sync(db_path: str, payload: dict) -> str:
    """Insère un portfolio backtest depuis un payload JSON (récepteur serveur).

    Retourne 'created' ou 'exists' (dédupliqué sur created_at).
    """
    conn = sqlite3.connect(db_path)
    try:
        # Vérifier doublon par created_at (ISO timestamp unique en pratique)
        cursor = conn.execute(
            "SELECT id FROM portfolio_backtests WHERE created_at = ?",
            (payload["created_at"],),
        )
        if cursor.fetchone() is not None:
            return "exists"

        row = {
            "strategy_name": payload["strategy_name"],
            "initial_capital": payload["initial_capital"],
            "n_assets": payload["n_assets"],
            "period_days": payload["period_days"],
            "assets": payload["assets"],
            "exchange": payload.get("exchange", "binance"),
            "leverage": payload.get("leverage"),
            "kill_switch_pct": payload.get("kill_switch_pct", 45.0),
            "kill_switch_window_hours": payload.get("kill_switch_window_hours", 24),
            "final_equity": payload["final_equity"],
            "total_return_pct": payload["total_return_pct"],
            "total_trades": payload["total_trades"],
            "win_rate": payload["win_rate"],
            "realized_pnl": payload["realized_pnl"],
            "force_closed_pnl": payload["force_closed_pnl"],
            "max_drawdown_pct": payload["max_drawdown_pct"],
            "max_drawdown_date": payload.get("max_drawdown_date"),
            "max_drawdown_duration_hours": payload["max_drawdown_duration_hours"],
            "peak_margin_ratio": payload["peak_margin_ratio"],
            "peak_open_positions": payload["peak_open_positions"],
            "peak_concurrent_assets": payload["peak_concurrent_assets"],
            "kill_switch_triggers": payload.get("kill_switch_triggers", 0),
            "kill_switch_events": payload.get("kill_switch_events"),
            "equity_curve": payload["equity_curve"],
            "per_asset_results": payload["per_asset_results"],
            "created_at": payload["created_at"],
            "duration_seconds": payload.get("duration_seconds"),
            "label": payload.get("label"),
        }
        conn.execute(_INSERT_SQL, row)
        conn.commit()
        logger.info(
            "Portfolio backtest reçu et inséré : {} @ {}",
            payload["strategy_name"], payload["created_at"],
        )
        return "created"
    finally:
        conn.close()


def push_portfolio_to_server(
    db_row: dict,
) -> None:
    """Pousse un portfolio backtest vers le serveur de production (best-effort).

    Ne crashe JAMAIS le run local. Log warning si erreur.
    Réutilise la même config sync que les résultats WFO.
    """
    try:
        from backend.core.config import get_config
        config = get_config()

        if not config.secrets.sync_enabled:
            return
        if not config.secrets.sync_server_url:
            logger.warning("sync_enabled=true mais sync_server_url vide — push portfolio ignoré")
            return
        if not config.secrets.sync_api_key:
            logger.warning("sync_enabled=true mais sync_api_key vide — push portfolio ignoré")
            return

        import httpx

        payload = build_portfolio_payload_from_row(db_row)
        url = f"{config.secrets.sync_server_url.rstrip('/')}/api/portfolio/results"

        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                url,
                json=payload,
                headers={"X-API-Key": config.secrets.sync_api_key},
            )

        if resp.status_code in (200, 201):
            status = resp.json().get("status", "ok")
            logger.info(
                "Portfolio pushé au serveur : {} → {} ({})",
                payload["strategy_name"], resp.status_code, status,
            )
        else:
            logger.warning(
                "Push portfolio échoué : {} → HTTP {} : {}",
                payload["strategy_name"], resp.status_code, resp.text[:200],
            )
    except Exception as exc:
        logger.warning(
            "Push portfolio échoué (réseau) : {}",
            exc,
        )
