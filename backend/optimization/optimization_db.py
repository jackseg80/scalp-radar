"""Persistence DB des résultats WFO — Sprint 13.

Fonctions sync (sqlite3) pour optimize.py CLI.
Fonctions async (aiosqlite) pour l'API FastAPI.
"""

from __future__ import annotations

import json
import math
import sqlite3
from typing import Any

import aiosqlite
from loguru import logger

from backend.optimization.report import FinalReport, compute_grade


# ─── Helpers ────────────────────────────────────────────────────────────────


def _sanitize_json_value(value: Any) -> Any:
    """Remplace NaN/Infinity par None pour éviter les erreurs json.dumps."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def _sanitize_dict(data: dict) -> dict:
    """Nettoie récursivement un dict de ses NaN/Infinity."""
    result = {}
    for k, v in data.items():
        if isinstance(v, dict):
            result[k] = _sanitize_dict(v)
        elif isinstance(v, list):
            result[k] = [_sanitize_json_value(item) if not isinstance(item, dict) else _sanitize_dict(item) for item in v]
        else:
            result[k] = _sanitize_json_value(v)
    return result


# ─── Fonctions SYNC (pour optimize.py CLI) ─────────────────────────────────


def save_result_sync(
    db_path: str,
    report: FinalReport,
    wfo_windows: list[dict] | None,
    duration: float | None,
    timeframe: str,
) -> None:
    """Sauvegarde un résultat WFO en DB (sync pour optimize.py CLI).

    Args:
        db_path: Chemin vers la DB SQLite
        report: FinalReport complet
        wfo_windows: WindowResult sérialisés (ou None)
        duration: Durée du run en secondes (ou None)
        timeframe: Timeframe de la stratégie (ex: "5m", "1h")
    """
    conn = sqlite3.connect(db_path)
    try:
        # Sanitize JSON values (NaN/Infinity → None)
        best_params_json = json.dumps(_sanitize_dict(report.recommended_params))
        wfo_windows_json = json.dumps(_sanitize_dict({"windows": wfo_windows})) if wfo_windows else None

        # Monte Carlo summary
        mc_summary = {
            "p_value": _sanitize_json_value(report.mc_p_value),
            "significant": report.mc_significant,
            "underpowered": report.mc_underpowered,
        }
        mc_summary_json = json.dumps(mc_summary)

        # Validation summary
        val_summary = {
            "bitget_sharpe": _sanitize_json_value(report.validation.bitget_sharpe),
            "bitget_net_return_pct": _sanitize_json_value(report.validation.bitget_net_return_pct),
            "bitget_trades": report.validation.bitget_trades,
            "bitget_sharpe_ci_low": _sanitize_json_value(report.validation.bitget_sharpe_ci_low),
            "bitget_sharpe_ci_high": _sanitize_json_value(report.validation.bitget_sharpe_ci_high),
            "binance_oos_avg_sharpe": _sanitize_json_value(report.validation.binance_oos_avg_sharpe),
            "transfer_ratio": _sanitize_json_value(report.validation.transfer_ratio),
            "transfer_significant": report.validation.transfer_significant,
            "volume_warning": report.validation.volume_warning,
            "volume_warning_detail": report.validation.volume_warning_detail,
        }
        val_summary_json = json.dumps(val_summary)

        warnings_json = json.dumps(report.warnings)

        # Transaction is_latest
        conn.execute("BEGIN")

        # 1. Mettre is_latest=0 sur l'ancien (s'il existe)
        conn.execute(
            """UPDATE optimization_results SET is_latest=0
               WHERE strategy_name=? AND asset=? AND timeframe=? AND is_latest=1""",
            (report.strategy_name, report.symbol, timeframe),
        )

        # 2. Insérer le nouveau avec is_latest=1
        conn.execute(
            """INSERT INTO optimization_results (
                strategy_name, asset, timeframe, created_at, duration_seconds,
                grade, total_score, oos_sharpe, consistency, oos_is_ratio, dsr,
                param_stability, monte_carlo_pvalue, mc_underpowered, n_windows, n_distinct_combos,
                best_params, wfo_windows, monte_carlo_summary, validation_summary, warnings, is_latest
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
            (
                report.strategy_name,
                report.symbol,
                timeframe,
                report.timestamp.isoformat(),
                duration,
                report.grade,
                report.total_score,
                _sanitize_json_value(report.wfo_avg_oos_sharpe),
                report.wfo_consistency_rate,
                _sanitize_json_value(report.oos_is_ratio),
                _sanitize_json_value(report.dsr),
                _sanitize_json_value(report.stability),
                _sanitize_json_value(report.mc_p_value),
                1 if report.mc_underpowered else 0,
                report.wfo_n_windows,
                report.n_distinct_combos,
                best_params_json,
                wfo_windows_json,
                mc_summary_json,
                val_summary_json,
                warnings_json,
            ),
        )

        conn.commit()
        logger.info(
            "Résultat WFO sauvé en DB : {} × {} (grade {}, score {})",
            report.strategy_name, report.symbol, report.grade, report.total_score,
        )
    finally:
        conn.close()


# ─── Fonctions ASYNC (pour l'API FastAPI) ──────────────────────────────────


async def get_results_async(
    db_path: str,
    strategy: str | None = None,
    asset: str | None = None,
    min_grade: str | None = None,
    latest_only: bool = True,
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    """Retourne les résultats avec filtres et pagination.

    Returns:
        {"results": [...], "total": int}
    """
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row

        # Build query
        where_clauses = []
        params: list[Any] = []

        if latest_only:
            where_clauses.append("is_latest = 1")
        if strategy:
            where_clauses.append("strategy_name = ?")
            params.append(strategy)
        if asset:
            where_clauses.append("asset = ?")
            params.append(asset)
        if min_grade:
            # Grades: A=85, B=70, C=55, D=40, F=0
            grade_thresholds = {"A": 85, "B": 70, "C": 55, "D": 40, "F": 0}
            threshold = grade_thresholds.get(min_grade, 0)
            where_clauses.append("total_score >= ?")
            params.append(threshold)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Count total
        count_query = f"SELECT COUNT(*) as total FROM optimization_results WHERE {where_sql}"
        cursor = await conn.execute(count_query, params)
        row = await cursor.fetchone()
        total = row["total"]

        # Fetch page
        query = f"""
            SELECT id, strategy_name, asset, timeframe, created_at, grade, total_score,
                   oos_sharpe, consistency, oos_is_ratio, dsr, param_stability,
                   n_windows, is_latest
            FROM optimization_results
            WHERE {where_sql}
            ORDER BY total_score DESC, created_at DESC
            LIMIT ? OFFSET ?
        """
        cursor = await conn.execute(query, params + [limit, offset])
        rows = await cursor.fetchall()

        results = [dict(row) for row in rows]
        return {"results": results, "total": total}


async def get_result_by_id_async(db_path: str, result_id: int) -> dict[str, Any] | None:
    """Retourne le détail complet d'un résultat (avec JSON parsés)."""
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute(
            "SELECT * FROM optimization_results WHERE id = ?", (result_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None

        result = dict(row)

        # Parse JSON blobs
        if result["best_params"]:
            result["best_params"] = json.loads(result["best_params"])
        if result["wfo_windows"]:
            windows_data = json.loads(result["wfo_windows"])
            result["wfo_windows"] = windows_data.get("windows", [])
        if result["monte_carlo_summary"]:
            result["monte_carlo_summary"] = json.loads(result["monte_carlo_summary"])
        if result["validation_summary"]:
            result["validation_summary"] = json.loads(result["validation_summary"])
        if result["warnings"]:
            result["warnings"] = json.loads(result["warnings"])

        return result


async def get_comparison_async(db_path: str) -> dict[str, Any]:
    """Retourne un tableau croisé strategies × assets (is_latest=1 seulement).

    Returns:
        {
            "strategies": ["vwap_rsi", "envelope_dca", ...],
            "assets": ["BTC/USDT", "ETH/USDT", ...],
            "matrix": {
                "vwap_rsi": {
                    "BTC/USDT": {"grade": "F", "total_score": 15, "oos_sharpe": -0.5, ...},
                    ...
                }
            }
        }

    Cases vides (stratégie non testée sur un asset) : clé absente du dict.
    """
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute(
            """SELECT strategy_name, asset, grade, total_score, oos_sharpe,
                      consistency, oos_is_ratio, dsr, param_stability, n_windows
               FROM optimization_results
               WHERE is_latest = 1
               ORDER BY strategy_name, asset"""
        )
        rows = await cursor.fetchall()

        strategies = set()
        assets = set()
        matrix: dict[str, dict[str, dict]] = {}

        for row in rows:
            strat = row["strategy_name"]
            asset = row["asset"]
            strategies.add(strat)
            assets.add(asset)

            if strat not in matrix:
                matrix[strat] = {}

            matrix[strat][asset] = {
                "grade": row["grade"],
                "total_score": row["total_score"],
                "oos_sharpe": row["oos_sharpe"],
                "consistency": row["consistency"],
                "oos_is_ratio": row["oos_is_ratio"],
                "dsr": row["dsr"],
                "param_stability": row["param_stability"],
                "n_windows": row["n_windows"],
            }

        return {
            "strategies": sorted(strategies),
            "assets": sorted(assets),
            "matrix": matrix,
        }
