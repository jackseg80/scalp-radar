"""Sync one-shot des résultats WFO locaux vers le serveur de production.

Pousse tous les résultats de la DB locale vers le serveur via POST API.
Idempotent : INSERT OR IGNORE côté serveur, relanceable à volonté.

Usage :
    uv run python -m scripts.sync_to_server
    uv run python -m scripts.sync_to_server --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3

import httpx
from loguru import logger

from backend.core.config import get_config
from backend.core.logging_setup import setup_logging
from backend.optimization.optimization_db import build_payload_from_db_row


def _get_local_db_path() -> str:
    """Résout le chemin DB locale depuis la config."""
    config = get_config()
    db_url = config.secrets.database_url
    if db_url.startswith("sqlite:///"):
        return db_url[10:]
    return "data/scalp_radar.db"


def _load_all_results(db_path: str) -> list[dict]:
    """Charge tous les résultats de la DB locale."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "SELECT * FROM optimization_results ORDER BY created_at"
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync résultats WFO locaux vers le serveur"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Afficher ce qui serait envoyé sans envoyer",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")

    config = get_config()
    if not config.secrets.sync_server_url:
        logger.error("SYNC_SERVER_URL non configuré dans .env")
        return
    if not config.secrets.sync_api_key:
        logger.error("SYNC_API_KEY non configuré dans .env")
        return

    db_path = _get_local_db_path()
    results = _load_all_results(db_path)

    if not results:
        logger.info("Aucun résultat en DB locale")
        return

    logger.info("{} résultats trouvés en DB locale", len(results))

    if args.dry_run:
        print(f"\n  Mode dry-run — {len(results)} résultats seraient envoyés :")
        print(f"  {'─' * 55}")
        for r in results:
            latest = "★" if r.get("is_latest") else " "
            print(
                f"  {latest} {r['strategy_name']:<16s} × {r['asset']:<12s} "
                f"Grade {r['grade']}  Score {r['total_score']:>5.1f}  "
                f"{r['created_at']}"
            )
        print()
        return

    url = f"{config.secrets.sync_server_url.rstrip('/')}/api/optimization/results"
    headers = {"X-API-Key": config.secrets.sync_api_key}

    created = 0
    exists = 0
    errors = 0

    with httpx.Client(timeout=15.0) as client:
        for r in results:
            payload = build_payload_from_db_row(r)
            try:
                resp = client.post(url, json=payload, headers=headers)
                if resp.status_code == 201:
                    created += 1
                    logger.info(
                        "  ✓ {} × {} → créé",
                        r["strategy_name"], r["asset"],
                    )
                elif resp.status_code == 200:
                    exists += 1
                    logger.debug(
                        "  = {} × {} → déjà existant",
                        r["strategy_name"], r["asset"],
                    )
                else:
                    errors += 1
                    logger.warning(
                        "  ✗ {} × {} → HTTP {} : {}",
                        r["strategy_name"], r["asset"],
                        resp.status_code, resp.text[:200],
                    )
            except Exception as exc:
                errors += 1
                logger.warning(
                    "  ✗ {} × {} → {}",
                    r["strategy_name"], r["asset"], exc,
                )

    print(f"\n  Résultat : {created} créés, {exists} déjà existants, {errors} erreurs")
    print()


if __name__ == "__main__":
    main()
