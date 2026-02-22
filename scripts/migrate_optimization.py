"""Migration des JSON existants vers optimization_results — Sprint 13.

Usage:
    uv run python -m scripts.migrate_optimization
    uv run python -m scripts.migrate_optimization --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
from datetime import datetime
from pathlib import Path

from loguru import logger

from backend.core.config import get_config
from backend.core.database import Database
from backend.core.logging_setup import setup_logging
from backend.optimization import STRATEGY_REGISTRY
from backend.optimization.report import compute_grade


def _sanitize_json_value(value):
    """Remplace NaN/Infinity par None."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


async def migrate_json_files(
    data_dir: str = "data/optimization",
    dry_run: bool = False,
    db_path: str | None = None,
) -> None:
    """Importe les final reports JSON existants dans la DB.

    Args:
        data_dir: Répertoire contenant les JSON
        dry_run: Si True, liste les fichiers sans écrire en DB
        db_path: Chemin DB (pour les tests), None = défaut
    """
    db = Database(db_path=db_path) if db_path else Database()
    await db.init()

    pattern = f"{data_dir}/*_*.json"
    all_files = glob.glob(pattern)

    # Filtrer : seulement les fichiers finaux (pas intermediate)
    # IMPORTANT : checker le nom du fichier, pas le chemin complet (sinon "test_migrate_with_intermediate" trigger le filtre)
    final_files = [f for f in all_files if "intermediate" not in Path(f).name]
    logger.info("Fichiers trouvés : {} (finaux) sur {} (total)", len(final_files), len(all_files))

    if dry_run:
        logger.info("Mode dry-run : aucune écriture en DB")
        for filepath in sorted(final_files):
            logger.info("  - {}", Path(filepath).name)
        await db.close()
        return

    imported = 0
    skipped = 0
    errors = 0
    no_windows = 0

    for filepath in sorted(final_files):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extraire les champs nécessaires (.get() défensif)
            strategy_name = data.get("strategy_name")
            symbol = data.get("symbol")
            timestamp_str = data.get("timestamp")
            grade = data.get("grade")
            total_score = data.get("total_score")  # Peut être absent (anciens JSON)

            if not all([strategy_name, symbol, timestamp_str, grade]):
                logger.warning("Fichier incomplet (champs manquants) : {}", Path(filepath).name)
                errors += 1
                continue

            # Déduire timeframe via STRATEGY_REGISTRY
            if strategy_name not in STRATEGY_REGISTRY:
                logger.warning("Stratégie inconnue '{}' dans {}", strategy_name, Path(filepath).name)
                errors += 1
                continue

            config_cls, _ = STRATEGY_REGISTRY[strategy_name]
            timeframe = config_cls().timeframe

            # Recalculer total_score si absent
            if total_score is None:
                wfo_data = data.get("wfo", {})
                overfit_data = data.get("overfitting", {})
                val_data = data.get("validation", {})

                _result = compute_grade(
                    oos_is_ratio=wfo_data.get("oos_is_ratio", 0.0),
                    mc_p_value=overfit_data.get("mc_p_value", 1.0),
                    dsr=overfit_data.get("dsr", 0.0),
                    stability=overfit_data.get("stability", 0.0),
                    bitget_transfer=val_data.get("transfer_ratio", 0.0),
                    mc_underpowered=overfit_data.get("mc_underpowered", False),
                )
                total_score = _result.score

            created_at = datetime.fromisoformat(timestamp_str)

            # Chercher l'intermediate correspondant pour wfo_windows
            base_name = Path(filepath).stem  # ex: "vwap_rsi_BTC_USDT_20260213_001736"
            parts = base_name.rsplit("_", 2)  # Retirer les 2 derniers blocs (timestamp)
            strat_sym = "_".join(parts[:-2])
            intermediate_path = Path(data_dir) / f"wfo_{strat_sym}_intermediate.json"

            wfo_windows = None
            if intermediate_path.exists():
                try:
                    with open(intermediate_path, "r", encoding="utf-8") as f_inter:
                        inter_data = json.load(f_inter)
                    wfo_windows = inter_data.get("windows", [])
                except Exception as e:
                    logger.warning("Erreur lecture intermediate {} : {}", intermediate_path.name, e)
            else:
                no_windows += 1

            # WFO data
            wfo_data = data.get("wfo", {})
            oos_sharpe = _sanitize_json_value(wfo_data.get("avg_oos_sharpe"))
            consistency = wfo_data.get("consistency_rate")
            oos_is_ratio = _sanitize_json_value(wfo_data.get("oos_is_ratio"))
            n_windows = wfo_data.get("n_windows", 0)

            # Overfitting data
            overfit_data = data.get("overfitting", {})
            dsr = _sanitize_json_value(overfit_data.get("dsr"))
            param_stability = _sanitize_json_value(overfit_data.get("stability"))
            mc_pvalue = _sanitize_json_value(overfit_data.get("mc_p_value"))
            mc_underpowered = 1 if overfit_data.get("mc_underpowered", False) else 0
            n_distinct_combos = overfit_data.get("n_distinct_combos")

            # Params recommandés
            recommended_params = data.get("recommended_params", {})
            best_params_json = json.dumps(recommended_params)

            # WFO windows JSON
            wfo_windows_json = json.dumps({"windows": wfo_windows}) if wfo_windows else None

            # Monte Carlo summary
            mc_summary = {
                "p_value": mc_pvalue,
                "significant": overfit_data.get("mc_significant", False),
                "underpowered": mc_underpowered == 1,
            }
            mc_summary_json = json.dumps(mc_summary)

            # Validation summary
            val_data = data.get("validation", {})
            val_summary = {
                "bitget_sharpe": _sanitize_json_value(val_data.get("bitget_sharpe")),
                "bitget_net_return_pct": _sanitize_json_value(val_data.get("bitget_net_return_pct")),
                "bitget_trades": val_data.get("bitget_trades"),
                "bitget_sharpe_ci_low": _sanitize_json_value(val_data.get("bitget_sharpe_ci_low")),
                "bitget_sharpe_ci_high": _sanitize_json_value(val_data.get("bitget_sharpe_ci_high")),
                "binance_oos_avg_sharpe": _sanitize_json_value(val_data.get("binance_oos_avg_sharpe")),
                "transfer_ratio": _sanitize_json_value(val_data.get("transfer_ratio")),
                "transfer_significant": val_data.get("transfer_significant", False),
                "volume_warning": val_data.get("volume_warning", False),
                "volume_warning_detail": val_data.get("volume_warning_detail", ""),
            }
            val_summary_json = json.dumps(val_summary)

            # Warnings
            warnings = data.get("warnings", [])
            warnings_json = json.dumps(warnings)

            # INSERT OR IGNORE (is_latest=0 par défaut, sera mis à jour après)
            await db._conn.execute(
                """INSERT OR IGNORE INTO optimization_results (
                    strategy_name, asset, timeframe, created_at, duration_seconds,
                    grade, total_score, oos_sharpe, consistency, oos_is_ratio, dsr,
                    param_stability, monte_carlo_pvalue, mc_underpowered, n_windows, n_distinct_combos,
                    best_params, wfo_windows, monte_carlo_summary, validation_summary, warnings, is_latest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (
                    strategy_name,
                    symbol,
                    timeframe,
                    created_at.isoformat(),
                    None,  # duration non dispo dans les anciens JSON
                    grade,
                    total_score,
                    oos_sharpe,
                    consistency,
                    oos_is_ratio,
                    dsr,
                    param_stability,
                    mc_pvalue,
                    mc_underpowered,
                    n_windows,
                    n_distinct_combos,
                    best_params_json,
                    wfo_windows_json,
                    mc_summary_json,
                    val_summary_json,
                    warnings_json,
                ),
            )
            imported += 1

        except Exception as e:
            logger.error("Erreur import {} : {}", Path(filepath).name, e)
            errors += 1

    await db._conn.commit()

    # Pass finale is_latest : pour chaque (strategy, asset, timeframe), mettre is_latest=1 sur le plus récent
    logger.info("Mise à jour is_latest sur les runs les plus récents...")
    cursor = await db._conn.execute(
        """SELECT DISTINCT strategy_name, asset, timeframe FROM optimization_results"""
    )
    combos = await cursor.fetchall()

    for combo in combos:
        strategy_name, asset, timeframe = combo
        # Récupérer le created_at MAX
        cursor2 = await db._conn.execute(
            """SELECT id FROM optimization_results
               WHERE strategy_name=? AND asset=? AND timeframe=?
               ORDER BY created_at DESC LIMIT 1""",
            (strategy_name, asset, timeframe),
        )
        row = await cursor2.fetchone()
        if row:
            await db._conn.execute(
                "UPDATE optimization_results SET is_latest=1 WHERE id=?", (row[0],)
            )

    await db._conn.commit()
    await db.close()

    logger.info("=" * 60)
    logger.info("Migration terminée")
    logger.info("  Fichiers importés    : {}", imported)
    logger.info("  Fichiers skippés     : {}", skipped)
    logger.info("  Erreurs              : {}", errors)
    logger.info("  Reports sans windows : {}", no_windows)
    logger.info("=" * 60)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Migration JSON → DB optimization_results")
    parser.add_argument("--dry-run", action="store_true", help="Liste les fichiers sans écrire en DB")
    args = parser.parse_args()

    setup_logging(level="INFO")

    await migrate_json_files(dry_run=args.dry_run)


if __name__ == "__main__":
    asyncio.run(main())
