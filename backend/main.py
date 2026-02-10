"""Point d'entrée standalone pour le DataEngine (sans API FastAPI).

Utile en prod ou pour le debugging. En dev, préférer uvicorn qui
intègre le DataEngine via lifespan.

Lancement : uv run python -m backend.main
"""

from __future__ import annotations

import asyncio
import signal
import sys

from loguru import logger

from backend.core.config import get_config
from backend.core.data_engine import DataEngine
from backend.core.database import Database
from backend.core.logging_setup import setup_logging


async def main() -> None:
    config = get_config()
    setup_logging(level=config.secrets.log_level)

    db = Database()
    await db.init()

    engine = DataEngine(config, db)

    # Gestion du shutdown
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Signal d'arrêt reçu")
        stop_event.set()

    # Linux : SIGTERM
    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, _signal_handler)

    try:
        await engine.start()
        logger.info("DataEngine standalone démarré. CTRL+C pour arrêter.")

        # Tourne jusqu'au signal d'arrêt
        while not stop_event.is_set():
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt reçu")
    finally:
        await engine.stop()
        await db.close()
        logger.info("Shutdown complet")


if __name__ == "__main__":
    asyncio.run(main())
