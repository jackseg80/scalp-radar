"""Purge des doublons is_latest dans optimization_results — Sprint 47b.

Le bug antérieur filtrait le UPDATE par (strategy_name, asset, timeframe),
ce qui laissait plusieurs is_latest=1 pour le même (strategy, asset) quand
le meilleur timeframe changeait entre deux runs WFO.

Ce script détecte ces doublons et ne garde que l'entrée avec le meilleur
total_score (le plus récent en cas d'égalité).

Usage:
    uv run python -m scripts.purge_wfo_duplicates
    uv run python -m scripts.purge_wfo_duplicates --dry-run
    uv run python -m scripts.purge_wfo_duplicates --db-path data/optimization.db
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from loguru import logger

from backend.core.logging_setup import setup_logging


def find_duplicates(conn: sqlite3.Connection) -> list[dict]:
    """Retourne les paires (strategy_name, asset) avec plusieurs is_latest=1."""
    rows = conn.execute(
        """SELECT strategy_name, asset, COUNT(*) as cnt
           FROM optimization_results
           WHERE is_latest=1
           GROUP BY strategy_name, asset
           HAVING cnt > 1
           ORDER BY strategy_name, asset"""
    ).fetchall()
    return [{"strategy_name": r[0], "asset": r[1], "count": r[2]} for r in rows]


def purge_duplicates(
    db_path: str,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Déduplique is_latest par (strategy_name, asset).

    Pour chaque doublon, garde uniquement l'entrée avec MAX(total_score).
    En cas d'égalité de score, garde la plus récente (MAX(id)).

    Returns:
        (n_pairs_fixed, n_rows_cleared) — nombre de paires corrigées et de lignes mises à 0.
    """
    conn = sqlite3.connect(db_path)
    try:
        duplicates = find_duplicates(conn)

        if not duplicates:
            logger.info("Aucun doublon is_latest détecté dans {}", db_path)
            return 0, 0

        logger.warning(
            "{} paire(s) avec doublons is_latest détectée(s) :",
            len(duplicates),
        )
        for d in duplicates:
            logger.warning(
                "  {} × {} → {} entrées is_latest=1",
                d["strategy_name"], d["asset"], d["count"],
            )

        n_pairs_fixed = 0
        n_rows_cleared = 0

        for d in duplicates:
            strat = d["strategy_name"]
            asset = d["asset"]

            # Sélectionner le meilleur id (MAX total_score, puis MAX id pour égalité)
            best_row = conn.execute(
                """SELECT id, timeframe, total_score, created_at
                   FROM optimization_results
                   WHERE strategy_name=? AND asset=? AND is_latest=1
                   ORDER BY total_score DESC NULLS LAST, id DESC
                   LIMIT 1""",
                (strat, asset),
            ).fetchone()

            if best_row is None:
                continue

            best_id = best_row[0]
            best_tf = best_row[1]
            best_score = best_row[2]

            # Compter combien seront mis à 0
            to_clear = conn.execute(
                """SELECT id, timeframe, total_score FROM optimization_results
                   WHERE strategy_name=? AND asset=? AND is_latest=1 AND id!=?""",
                (strat, asset, best_id),
            ).fetchall()

            logger.info(
                "  Conserve : {} × {} tf={} score={:.4f} (id={})",
                strat, asset, best_tf, best_score or 0.0, best_id,
            )
            for row in to_clear:
                logger.info(
                    "  Efface   : {} × {} tf={} score={:.4f} (id={})",
                    strat, asset, row[1], row[2] or 0.0, row[0],
                )

            if not dry_run:
                conn.execute(
                    """UPDATE optimization_results SET is_latest=0
                       WHERE strategy_name=? AND asset=? AND is_latest=1 AND id!=?""",
                    (strat, asset, best_id),
                )

            n_pairs_fixed += 1
            n_rows_cleared += len(to_clear)

        if not dry_run:
            conn.commit()
            logger.success(
                "Purge terminée : {} paire(s) corrigée(s), {} ligne(s) is_latest mise(s) à 0.",
                n_pairs_fixed, n_rows_cleared,
            )
        else:
            logger.info(
                "[DRY-RUN] Aurait corrigé {} paire(s), {} ligne(s).",
                n_pairs_fixed, n_rows_cleared,
            )

        return n_pairs_fixed, n_rows_cleared

    finally:
        conn.close()


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Purge des doublons is_latest dans optimization_results"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher sans modifier la base",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Chemin vers la DB SQLite (défaut : data/optimization.db)",
    )
    args = parser.parse_args()

    if args.db_path:
        db_path = args.db_path
    else:
        # Chercher la DB dans les emplacements habituels
        candidates = [
            Path("data/optimization.db"),
            Path("data/scalp_radar.db"),
        ]
        db_path = None
        for c in candidates:
            if c.exists():
                db_path = str(c)
                break
        if db_path is None:
            logger.error(
                "Aucune DB trouvée dans data/. Spécifier --db-path explicitement."
            )
            return

    logger.info("DB : {}", db_path)
    purge_duplicates(db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
