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
    source: str = "local",
    regime_analysis: dict | None = None,
) -> int:
    """Sauvegarde un résultat WFO en DB (sync pour optimize.py CLI).

    Args:
        db_path: Chemin vers la DB SQLite
        report: FinalReport complet
        wfo_windows: WindowResult sérialisés (ou None)
        duration: Durée du run en secondes (ou None)
        timeframe: Timeframe de la stratégie (ex: "5m", "1h")
        source: Origine du résultat ("local" ou "server")
        regime_analysis: Analyse par régime du best combo (Sprint 15b, optionnel)

    Returns:
        result_id (int) : ID du résultat inséré
    """
    conn = sqlite3.connect(db_path, timeout=30)
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

        warnings_json = json.dumps(
            [_sanitize_dict(w) if isinstance(w, dict) else w for w in report.warnings]
            if isinstance(report.warnings, list) else report.warnings
        )
        regime_analysis_json = json.dumps(_sanitize_dict(regime_analysis)) if regime_analysis else None

        # Transaction is_latest
        conn.execute("BEGIN")

        n_combos = report.n_distinct_combos or 0

        # Protection : un run avec très peu de combos (< 10, ex: Explorer avec
        # grille restreinte) ne doit PAS voler is_latest d'un run complet.
        if n_combos >= 10:
            # 1. Mettre is_latest=0 sur l'ancien (s'il existe)
            conn.execute(
                """UPDATE optimization_results SET is_latest=0
                   WHERE strategy_name=? AND asset=? AND is_latest=1""",
                (report.strategy_name, report.symbol),
            )
            is_latest_val = 1
        else:
            is_latest_val = 0
            logger.warning(
                "Run local avec peu de combos (n={}), is_latest non modifié : {} × {}",
                n_combos, report.strategy_name, report.symbol,
            )

        # 2. Insérer le nouveau
        cursor = conn.execute(
            """INSERT INTO optimization_results (
                strategy_name, asset, timeframe, created_at, duration_seconds,
                grade, total_score, oos_sharpe, consistency, oos_is_ratio, dsr,
                param_stability, monte_carlo_pvalue, mc_underpowered, n_windows, n_distinct_combos,
                best_params, wfo_windows, monte_carlo_summary, validation_summary, warnings,
                is_latest, source, regime_analysis
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                is_latest_val,
                source,
                regime_analysis_json,
            ),
        )
        result_id = cursor.lastrowid

        conn.commit()
        logger.info(
            "Résultat WFO sauvé en DB : {} × {} (grade {}, score {}, id={}, is_latest={})",
            report.strategy_name, report.symbol, report.grade, report.total_score,
            result_id, is_latest_val,
        )
        return result_id
    finally:
        conn.close()


def save_result_from_payload_sync(db_path: str, payload: dict) -> str:
    """Insère un résultat WFO depuis un payload JSON brut (endpoint POST serveur).

    Transaction sûre : INSERT d'abord, UPDATE is_latest ensuite seulement si inséré.
    Évite de perdre le flag is_latest sur un doublon (INSERT OR IGNORE).

    Returns:
        "created" si inséré, "exists" si doublon (UNIQUE constraint).
    """
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute("BEGIN")

        # Sérialiser regime_analysis si présent
        ra = payload.get("regime_analysis")
        regime_analysis_val = ra if isinstance(ra, str) else json.dumps(ra) if ra is not None else None

        # 1. Tenter l'INSERT (OR IGNORE pour les doublons)
        cursor = conn.execute(
            """INSERT OR IGNORE INTO optimization_results (
                strategy_name, asset, timeframe, created_at, duration_seconds,
                grade, total_score, oos_sharpe, consistency, oos_is_ratio, dsr,
                param_stability, monte_carlo_pvalue, mc_underpowered, n_windows, n_distinct_combos,
                best_params, wfo_windows, monte_carlo_summary, validation_summary, warnings,
                is_latest, source, regime_analysis
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (
                payload["strategy_name"],
                payload["asset"],
                payload["timeframe"],
                payload["created_at"],
                payload.get("duration_seconds"),
                payload["grade"],
                payload["total_score"],
                payload.get("oos_sharpe"),
                payload.get("consistency"),
                payload.get("oos_is_ratio"),
                payload.get("dsr"),
                payload.get("param_stability"),
                payload.get("monte_carlo_pvalue"),
                payload.get("mc_underpowered", 0),
                payload["n_windows"],
                payload.get("n_distinct_combos"),
                payload["best_params"] if isinstance(payload["best_params"], str) else json.dumps(payload["best_params"]),
                payload.get("wfo_windows") if isinstance(payload.get("wfo_windows"), str) else json.dumps(payload["wfo_windows"]) if payload.get("wfo_windows") is not None else None,
                payload.get("monte_carlo_summary") if isinstance(payload.get("monte_carlo_summary"), str) else json.dumps(payload["monte_carlo_summary"]) if payload.get("monte_carlo_summary") is not None else None,
                payload.get("validation_summary") if isinstance(payload.get("validation_summary"), str) else json.dumps(payload["validation_summary"]) if payload.get("validation_summary") is not None else None,
                payload.get("warnings") if isinstance(payload.get("warnings"), str) else json.dumps(payload["warnings"]) if payload.get("warnings") is not None else None,
                payload.get("source", "local"),
                regime_analysis_val,
            ),
        )

        if cursor.rowcount == 0:
            # Doublon — ne pas toucher à is_latest
            conn.commit()
            return "exists"

        # 2. Inséré avec succès → conditionner is_latest
        new_id = cursor.lastrowid
        n_combos = payload.get("n_distinct_combos") or 0

        # Protection : un run avec très peu de combos (< 10) ne doit PAS
        # voler le flag is_latest d'un run complet existant (push serveur
        # avec grille restreinte → ne doit pas écraser un run local à 324 combos)
        if n_combos >= 10:
            conn.execute(
                """UPDATE optimization_results SET is_latest=0
                   WHERE strategy_name=? AND asset=? AND is_latest=1 AND id!=?""",
                (payload["strategy_name"], payload["asset"], new_id),
            )
        else:
            # Garder is_latest=0 sur le nouveau run (ne pas écraser le bon)
            conn.execute(
                "UPDATE optimization_results SET is_latest=0 WHERE id=?",
                (new_id,),
            )
            logger.warning(
                "Run pushé avec peu de combos (n={}), is_latest non modifié : {} × {}",
                n_combos, payload["strategy_name"], payload["asset"],
            )

        conn.commit()

        # 3. Sauver les combo_results si présents (Sprint 14b)
        combo_results = payload.get("combo_results")
        if combo_results and new_id:
            n_saved = save_combo_results_sync(db_path, new_id, combo_results)
            logger.info("Combo results sauvés (POST) : {} combos pour result_id={}", n_saved, new_id)

        logger.info(
            "Résultat WFO reçu (POST) : {} × {} (grade {})",
            payload["strategy_name"], payload["asset"], payload["grade"],
        )
        return "created"
    finally:
        conn.close()


def build_push_payload(
    report: FinalReport,
    wfo_windows: list[dict] | None,
    duration: float | None,
    timeframe: str,
    source: str = "local",
    combo_results: list[dict] | None = None,
    regime_analysis: dict | None = None,
) -> dict:
    """Construit le payload JSON pour POST vers le serveur.

    Réutilise _sanitize_dict/_sanitize_json_value pour nettoyer NaN/Infinity.

    Args:
        combo_results: Combo results du WFO (Sprint 14b, optionnel)
        regime_analysis: Analyse par régime du best combo (Sprint 15b, optionnel)
    """
    best_params_json = json.dumps(_sanitize_dict(report.recommended_params))
    wfo_windows_json = json.dumps(_sanitize_dict({"windows": wfo_windows})) if wfo_windows else None

    mc_summary = json.dumps({
        "p_value": _sanitize_json_value(report.mc_p_value),
        "significant": report.mc_significant,
        "underpowered": report.mc_underpowered,
    })

    val_summary = json.dumps({
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
    })

    payload = {
        "strategy_name": report.strategy_name,
        "asset": report.symbol,
        "timeframe": timeframe,
        "created_at": report.timestamp.isoformat(),
        "duration_seconds": duration,
        "grade": report.grade,
        "total_score": report.total_score,
        "oos_sharpe": _sanitize_json_value(report.wfo_avg_oos_sharpe),
        "consistency": report.wfo_consistency_rate,
        "oos_is_ratio": _sanitize_json_value(report.oos_is_ratio),
        "dsr": _sanitize_json_value(report.dsr),
        "param_stability": _sanitize_json_value(report.stability),
        "monte_carlo_pvalue": _sanitize_json_value(report.mc_p_value),
        "mc_underpowered": 1 if report.mc_underpowered else 0,
        "n_windows": report.wfo_n_windows,
        "n_distinct_combos": report.n_distinct_combos,
        "best_params": best_params_json,
        "wfo_windows": wfo_windows_json,
        "monte_carlo_summary": mc_summary,
        "validation_summary": val_summary,
        "warnings": json.dumps(report.warnings),
        "source": source,
    }

    # Ajouter combo_results si présent (Sprint 14b)
    if combo_results:
        payload["combo_results"] = combo_results

    # Ajouter regime_analysis si présent (Sprint 15b)
    if regime_analysis:
        payload["regime_analysis"] = json.dumps(_sanitize_dict(regime_analysis))

    return payload


def build_payload_from_db_row(row: dict) -> dict:
    """Construit un payload POST depuis une row DB (pour sync_to_server.py).

    Les colonnes DB correspondent déjà au format POST attendu.
    Les JSON blobs restent en string (le serveur les stocke tels quels).
    """
    return {
        "strategy_name": row["strategy_name"],
        "asset": row["asset"],
        "timeframe": row["timeframe"],
        "created_at": row["created_at"],
        "duration_seconds": row.get("duration_seconds"),
        "grade": row["grade"],
        "total_score": row["total_score"],
        "oos_sharpe": row.get("oos_sharpe"),
        "consistency": row.get("consistency"),
        "oos_is_ratio": row.get("oos_is_ratio"),
        "dsr": row.get("dsr"),
        "param_stability": row.get("param_stability"),
        "monte_carlo_pvalue": row.get("monte_carlo_pvalue"),
        "mc_underpowered": row.get("mc_underpowered", 0),
        "n_windows": row["n_windows"],
        "n_distinct_combos": row.get("n_distinct_combos"),
        "best_params": row["best_params"],
        "wfo_windows": row.get("wfo_windows"),
        "monte_carlo_summary": row.get("monte_carlo_summary"),
        "validation_summary": row.get("validation_summary"),
        "warnings": row.get("warnings"),
        "source": row.get("source", "local"),
        "regime_analysis": row.get("regime_analysis"),
    }


def push_to_server(
    report: FinalReport,
    wfo_windows: list[dict] | None,
    duration: float | None,
    timeframe: str,
    combo_results: list[dict] | None = None,
    regime_analysis: dict | None = None,
) -> None:
    """Pousse un résultat WFO vers le serveur de production (best-effort).

    Ne crashe JAMAIS le run local. Log warning si erreur.

    Args:
        combo_results: Combo results du WFO (Sprint 14b, optionnel)
        regime_analysis: Analyse par régime du best combo (Sprint 15b, optionnel)
    """
    try:
        from backend.core.config import get_config
        config = get_config()

        if not config.secrets.sync_enabled:
            return
        if not config.secrets.sync_server_url:
            logger.warning("sync_enabled=true mais sync_server_url vide — push ignoré")
            return
        if not config.secrets.sync_api_key:
            logger.warning("sync_enabled=true mais sync_api_key vide — push ignoré")
            return

        import httpx

        payload = build_push_payload(
            report, wfo_windows, duration, timeframe,
            combo_results=combo_results, regime_analysis=regime_analysis,
        )
        url = f"{config.secrets.sync_server_url.rstrip('/')}/api/optimization/results"

        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                url,
                json=payload,
                headers={"X-API-Key": config.secrets.sync_api_key},
            )

        if resp.status_code in (200, 201):
            status = resp.json().get("status", "ok")
            logger.info(
                "Résultat pushé au serveur : {} × {} → {} ({})",
                report.strategy_name, report.symbol, resp.status_code, status,
            )
        else:
            logger.warning(
                "Push serveur échoué : {} × {} → HTTP {} : {}",
                report.strategy_name, report.symbol, resp.status_code, resp.text[:200],
            )
    except Exception as exc:
        logger.warning(
            "Push serveur échoué (réseau) : {} × {} → {}",
            report.strategy_name, report.symbol, exc,
        )


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

        # Build query (préfixé r. pour compatibilité LEFT JOIN)
        where_clauses = []
        params: list[Any] = []

        if latest_only:
            where_clauses.append("r.is_latest = 1")
        if strategy:
            where_clauses.append("r.strategy_name = ?")
            params.append(strategy)
        if asset:
            where_clauses.append("r.asset = ?")
            params.append(asset)
        if min_grade:
            # Grades: A=85, B=70, C=55, D=40, F=0
            grade_thresholds = {"A": 85, "B": 70, "C": 55, "D": 40, "F": 0}
            threshold = grade_thresholds.get(min_grade, 0)
            where_clauses.append("r.total_score >= ?")
            params.append(threshold)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Count total
        count_query = f"SELECT COUNT(*) as total FROM optimization_results r WHERE {where_sql}"
        cursor = await conn.execute(count_query, params)
        row = await cursor.fetchone()
        total = row["total"]

        # Fetch page (avec combo_count = nb réel de combos en DB)
        query = f"""
            SELECT r.id, r.strategy_name, r.asset, r.timeframe, r.created_at, r.grade, r.total_score,
                   r.oos_sharpe, r.consistency, r.oos_is_ratio, r.dsr, r.param_stability,
                   r.n_windows, r.is_latest, r.n_distinct_combos,
                   COUNT(c.id) as combo_count
            FROM optimization_results r
            LEFT JOIN wfo_combo_results c ON c.optimization_result_id = r.id
            WHERE {where_sql}
            GROUP BY r.id
            ORDER BY r.created_at DESC
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
        if result.get("regime_analysis"):
            result["regime_analysis"] = json.loads(result["regime_analysis"])

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


# ─── Combo Results (Sprint 14b) ────────────────────────────────────────────


def save_combo_results_sync(db_path: str, result_id: int, combo_results: list[dict]) -> int:
    """Insère les combo results en DB (sync). Retourne le nombre inséré.

    Args:
        db_path: Chemin vers la DB SQLite
        result_id: ID du résultat WFO parent
        combo_results: Liste des combos avec métriques agrégées

    Returns:
        Nombre de combos insérées
    """
    if not combo_results:
        return 0

    conn = sqlite3.connect(db_path, timeout=30)
    try:
        data = [
            (
                result_id,
                json.dumps(cr["params"], sort_keys=True),
                _sanitize_json_value(cr.get("oos_sharpe")),
                _sanitize_json_value(cr.get("oos_return_pct")),
                cr.get("oos_trades"),
                _sanitize_json_value(cr.get("oos_win_rate")),
                _sanitize_json_value(cr.get("is_sharpe")),
                _sanitize_json_value(cr.get("is_return_pct")),
                cr.get("is_trades"),
                _sanitize_json_value(cr.get("consistency")),
                _sanitize_json_value(cr.get("oos_is_ratio")),
                1 if cr.get("is_best") else 0,
                cr.get("n_windows_evaluated"),
                json.dumps(cr["per_window_sharpes"]) if cr.get("per_window_sharpes") is not None else None,
            )
            for cr in combo_results
        ]
        conn.executemany(
            """INSERT INTO wfo_combo_results
               (optimization_result_id, params, oos_sharpe, oos_return_pct, oos_trades,
                oos_win_rate, is_sharpe, is_return_pct, is_trades, consistency, oos_is_ratio, is_best,
                n_windows_evaluated, per_window_sharpes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            data,
        )
        conn.commit()
        return len(data)
    finally:
        conn.close()


async def get_combo_results_async(db_path: str, result_id: int) -> list[dict]:
    """Retourne tous les combo results pour un résultat WFO donné.

    Args:
        db_path: Chemin vers la DB SQLite
        result_id: ID du résultat WFO

    Returns:
        Liste des combos triées par OOS Sharpe décroissant
    """
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute(
            """SELECT params, oos_sharpe, oos_return_pct, oos_trades, oos_win_rate,
                      is_sharpe, is_return_pct, is_trades, consistency, oos_is_ratio, is_best,
                      n_windows_evaluated, per_window_sharpes
               FROM wfo_combo_results
               WHERE optimization_result_id = ?
               ORDER BY oos_sharpe DESC""",
            (result_id,),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            d = {**dict(row), "params": json.loads(row["params"])}
            pws = row["per_window_sharpes"]
            d["per_window_sharpes"] = json.loads(pws) if pws else None
            results.append(d)
        return results


# ─── Strategy Summary (Sprint 36) ────────────────────────────────────────


async def get_strategy_summary_async(db_path: str, strategy_name: str) -> dict:
    """Résumé agrégé d'une stratégie : grades, red flags, convergence params, portfolio runs.

    Args:
        db_path: Chemin vers la DB SQLite
        strategy_name: Nom de la stratégie (ex: "grid_atr")

    Returns:
        Dict complet avec grades, red_flags, param_convergence, portfolio_runs
    """
    from collections import Counter

    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row

        # 1. Tous les résultats is_latest=1 pour cette stratégie
        cursor = await conn.execute(
            """SELECT asset, grade, total_score, oos_sharpe, consistency,
                      oos_is_ratio, monte_carlo_pvalue, mc_underpowered,
                      param_stability, best_params, created_at
               FROM optimization_results
               WHERE is_latest = 1 AND strategy_name = ?
               ORDER BY total_score DESC""",
            (strategy_name,),
        )
        rows = await cursor.fetchall()

        if not rows:
            return {"strategy_name": strategy_name, "total_assets": 0}

        total = len(rows)

        # 2. Grade distribution
        grade_counts = Counter(r["grade"] for r in rows)

        # 3. Red flags
        red_flags = {
            "oos_is_ratio_suspect": sum(
                1 for r in rows if r["oos_is_ratio"] and r["oos_is_ratio"] > 1.5
            ),
            "sharpe_anomalous": sum(
                1 for r in rows if r["oos_sharpe"] and r["oos_sharpe"] > 20
            ),
            "underpowered": sum(1 for r in rows if r["mc_underpowered"]),
            "low_consistency": sum(
                1 for r in rows if r["consistency"] is not None and r["consistency"] < 0.5
            ),
            "low_stability": sum(
                1 for r in rows
                if r["param_stability"] is not None and r["param_stability"] < 0.3
            ),
        }

        # 4. Convergence params (double-JSON guard)
        param_vals: dict[str, list] = {}
        for r in rows:
            raw = r["best_params"]
            if not raw:
                continue
            params = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(params, str):
                params = json.loads(params)  # double-encoded
            if isinstance(params, dict):
                for k, v in params.items():
                    param_vals.setdefault(k, []).append(v)

        convergence = []
        for k, vals in sorted(param_vals.items()):
            counter = Counter(str(v) for v in vals)
            mode_val, mode_count = counter.most_common(1)[0]
            convergence.append({
                "param": k,
                "n_unique": len(set(str(v) for v in vals)),
                "mode": mode_val,
                "mode_pct": round(mode_count / len(vals) * 100),
            })

        # 5. Portfolio runs
        cursor = await conn.execute(
            """SELECT id, label, total_return_pct, period_days, created_at
               FROM portfolio_backtests
               WHERE strategy_name = ?
               ORDER BY created_at DESC LIMIT 10""",
            (strategy_name,),
        )
        portfolio_rows = await cursor.fetchall()
        portfolio_runs = [
            {
                "id": r["id"],
                "label": r["label"],
                "return_pct": r["total_return_pct"],
                "days": r["period_days"],
                "created_at": r["created_at"],
            }
            for r in portfolio_rows
        ]

        ab_count = grade_counts.get("A", 0) + grade_counts.get("B", 0)
        avg_sharpe = sum(r["oos_sharpe"] or 0 for r in rows) / total
        avg_consistency = sum(r["consistency"] or 0 for r in rows) / total

        return {
            "strategy_name": strategy_name,
            "total_assets": total,
            "grades": {g: grade_counts.get(g, 0) for g in ["A", "B", "C", "D", "F"]},
            "ab_count": ab_count,
            "ab_pct": round(ab_count / total * 100, 1) if total else 0,
            "avg_oos_sharpe": round(avg_sharpe, 2),
            "avg_consistency": round(avg_consistency, 2),
            "underpowered_count": red_flags["underpowered"],
            "underpowered_pct": round(red_flags["underpowered"] / total * 100, 1),
            "red_flags": red_flags,
            "red_flags_total": sum(red_flags.values()),
            "param_convergence": convergence,
            "latest_wfo_date": max(r["created_at"][:10] for r in rows),
            "portfolio_runs": portfolio_runs,
        }
