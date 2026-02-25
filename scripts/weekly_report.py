"""CLI pour générer et envoyer le rapport Telegram hebdomadaire.

Usage :
    uv run python -m scripts.weekly_report --dry-run           # semaine précédente
    uv run python -m scripts.weekly_report --dry-run --current # semaine en cours
    uv run python -m scripts.weekly_report                      # envoi Telegram
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from loguru import logger

from backend.alerts.weekly_reporter import generate_report
from backend.core.config import get_config
from backend.core.database import Database
from backend.core.logging_setup import setup_logging

# Forcer UTF-8 sur Windows (évite UnicodeEncodeError avec les emoji)
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


async def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Rapport Telegram hebdomadaire Scalp Radar",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche le rapport sans l'envoyer",
    )
    parser.add_argument(
        "--current",
        action="store_true",
        help="Rapport semaine en cours (défaut : semaine précédente complète)",
    )
    args = parser.parse_args()

    config = get_config()
    db = Database()
    await db.init()

    try:
        report = await generate_report(db, config, current_week=args.current)

        if args.dry_run:
            print(report)
        else:
            token = config.secrets.telegram_bot_token
            chat_id = config.secrets.telegram_chat_id
            if not token or not chat_id:
                logger.error("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID manquant")
                sys.exit(1)

            from backend.alerts.telegram import TelegramClient

            telegram = TelegramClient(token, chat_id)
            ok = await telegram.send_message(report)
            if ok:
                logger.info("Rapport envoyé avec succès")
            else:
                logger.error("Échec envoi Telegram")
                sys.exit(1)
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
