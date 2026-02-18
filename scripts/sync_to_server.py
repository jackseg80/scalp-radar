"""Sync one-shot des résultats WFO + portfolio backtests vers le serveur de production.

Pousse tous les résultats de la DB locale vers le serveur via POST API.
Idempotent : dédupliqué côté serveur, relanceable à volonté.

Usage :
    uv run python -m scripts.sync_to_server
    uv run python -m scripts.sync_to_server --dry-run
    uv run python -m scripts.sync_to_server --only wfo
    uv run python -m scripts.sync_to_server --only portfolio
"""

from __future__ import annotations

import argparse
import sqlite3

import httpx
from loguru import logger

from backend.backtesting.portfolio_db import build_portfolio_payload_from_row
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
    """Charge tous les résultats WFO de la DB locale."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "SELECT * FROM optimization_results ORDER BY created_at"
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def _load_all_portfolio_backtests(db_path: str) -> list[dict]:
    """Charge tous les portfolio backtests de la DB locale."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # Table peut ne pas exister sur les anciennes DB
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_backtests'"
        )
        if cursor.fetchone() is None:
            return []
        cursor = conn.execute(
            "SELECT * FROM portfolio_backtests ORDER BY created_at"
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def _sync_wfo(
    client: httpx.Client,
    results: list[dict],
    base_url: str,
    headers: dict,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Sync les résultats WFO. Retourne (created, exists, errors)."""
    if not results:
        logger.info("Aucun résultat WFO en DB locale")
        return 0, 0, 0

    logger.info("{} résultats WFO trouvés en DB locale", len(results))

    if dry_run:
        print(f"\n  Mode dry-run — {len(results)} résultats WFO seraient envoyés :")
        print(f"  {'─' * 55}")
        for r in results:
            latest = "★" if r.get("is_latest") else " "
            print(
                f"  {latest} {r['strategy_name']:<16s} × {r['asset']:<12s} "
                f"Grade {r['grade']}  Score {r['total_score']:>5.1f}  "
                f"{r['created_at']}"
            )
        print()
        return 0, 0, 0

    url = f"{base_url}/api/optimization/results"
    created = 0
    exists = 0
    errors = 0

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

    return created, exists, errors


def _sync_portfolio(
    client: httpx.Client,
    backtests: list[dict],
    base_url: str,
    headers: dict,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Sync les portfolio backtests. Retourne (created, exists, errors)."""
    if not backtests:
        logger.info("Aucun portfolio backtest en DB locale")
        return 0, 0, 0

    logger.info("{} portfolio backtests trouvés en DB locale", len(backtests))

    if dry_run:
        print(f"\n  Mode dry-run — {len(backtests)} portfolio backtests seraient envoyés :")
        print(f"  {'─' * 55}")
        for r in backtests:
            label = r.get("label") or ""
            print(
                f"    {r['strategy_name']:<16s}  {r['n_assets']} assets  "
                f"{r['total_return_pct']:>+6.1f}%  DD {r['max_drawdown_pct']:>5.1f}%  "
                f"{label}  {r['created_at']}"
            )
        print()
        return 0, 0, 0

    url = f"{base_url}/api/portfolio/results"
    created = 0
    exists = 0
    errors = 0

    for r in backtests:
        payload = build_portfolio_payload_from_row(r)
        try:
            resp = client.post(url, json=payload, headers=headers)
            if resp.status_code == 201:
                created += 1
                logger.info(
                    "  ✓ portfolio {} ({} assets) → créé",
                    r["strategy_name"], r["n_assets"],
                )
            elif resp.status_code == 200:
                exists += 1
                logger.debug(
                    "  = portfolio {} → déjà existant",
                    r["strategy_name"],
                )
            else:
                errors += 1
                logger.warning(
                    "  ✗ portfolio {} → HTTP {} : {}",
                    r["strategy_name"],
                    resp.status_code, resp.text[:200],
                )
        except Exception as exc:
            errors += 1
            logger.warning(
                "  ✗ portfolio {} → {}",
                r["strategy_name"], exc,
            )

    return created, exists, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync résultats WFO + portfolio backtests vers le serveur"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Afficher ce qui serait envoyé sans envoyer",
    )
    parser.add_argument(
        "--only", choices=["wfo", "portfolio"],
        help="Sync uniquement WFO ou portfolio (par défaut: les deux)",
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
    base_url = config.secrets.sync_server_url.rstrip("/")
    headers = {"X-API-Key": config.secrets.sync_api_key}

    wfo_created = wfo_exists = wfo_errors = 0
    pf_created = pf_exists = pf_errors = 0

    with httpx.Client(timeout=15.0) as client:
        # WFO
        if args.only != "portfolio":
            wfo_results = _load_all_results(db_path)
            wfo_created, wfo_exists, wfo_errors = _sync_wfo(
                client, wfo_results, base_url, headers, args.dry_run,
            )

        # Portfolio
        if args.only != "wfo":
            pf_results = _load_all_portfolio_backtests(db_path)
            pf_created, pf_exists, pf_errors = _sync_portfolio(
                client, pf_results, base_url, headers, args.dry_run,
            )

    if not args.dry_run:
        total_created = wfo_created + pf_created
        total_exists = wfo_exists + pf_exists
        total_errors = wfo_errors + pf_errors
        print(f"\n  Résultat : {total_created} créés, {total_exists} déjà existants, {total_errors} erreurs")
        if args.only != "portfolio":
            print(f"    WFO       : {wfo_created} créés, {wfo_exists} existants, {wfo_errors} erreurs")
        if args.only != "wfo":
            print(f"    Portfolio : {pf_created} créés, {pf_exists} existants, {pf_errors} erreurs")
        print()


if __name__ == "__main__":
    main()
