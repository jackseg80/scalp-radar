"""Configuration du logging avec loguru.

Console colorée pour le dev, fichiers JSON structurés pour la prod.
Sprint 31 : sink WS pour broadcast WARNING+ en temps réel.
"""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from pathlib import Path

from loguru import logger


# ─── WS Log Broadcast (Sprint 31) ────────────────────────────────────────

# Buffer circulaire des derniers WARNING/ERROR pour les nouveaux clients WS
_log_buffer: deque[dict] = deque(maxlen=20)

# Set de queues asyncio, une par client WS connecté
_log_subscribers: set[asyncio.Queue] = set()


def _ws_log_sink(message):
    """Handler loguru : capture WARNING+ et push aux subscribers WS.

    Appelé de manière synchrone depuis n'importe quel thread.
    put_nowait() sur asyncio.Queue est thread-safe (CPython GIL).
    """
    record = message.record
    if record["level"].no < 30:  # 30 = WARNING
        return

    entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
    }
    _log_buffer.append(entry)

    # Snapshot copy pour thread-safety (évite RuntimeError si un client
    # se déconnecte pendant l'itération)
    for queue in list(_log_subscribers):
        try:
            queue.put_nowait(entry)
        except asyncio.QueueFull:
            pass  # Client lent, on drop le log


def subscribe_logs() -> asyncio.Queue:
    """Enregistre un nouveau subscriber WS. Retourne sa queue."""
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    _log_subscribers.add(q)
    return q


def unsubscribe_logs(q: asyncio.Queue) -> None:
    """Supprime un subscriber WS."""
    _log_subscribers.discard(q)


def get_log_buffer() -> list[dict]:
    """Retourne les derniers WARNING/ERROR en buffer (pour init client WS)."""
    return list(_log_buffer)


# ─── Setup principal ──────────────────────────────────────────────────────


def setup_logging(
    level: str = "DEBUG",
    log_dir: str | Path = "logs",
) -> None:
    """Configure loguru avec sortie console + fichiers.

    Args:
        level: Niveau de log minimum (DEBUG, INFO, WARNING, ERROR).
        log_dir: Répertoire pour les fichiers de log.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Supprimer le handler par défaut
    logger.remove()

    # Console : format lisible, coloré
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Fichier principal : JSON structuré, rotation 50MB, rétention 30 jours
    logger.add(
        log_path / "scalp_radar.log",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="50 MB",
        retention="30 days",
        compression="gz",
        serialize=True,
    )

    # Fichier erreurs séparé
    logger.add(
        log_path / "errors.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="50 MB",
        retention="30 days",
        compression="gz",
        serialize=True,
    )

    # Sink WS : broadcast WARNING+ vers les clients WebSocket connectés
    logger.add(_ws_log_sink, level="WARNING", format="{message}")

    logger.info("Logging initialisé (level={})", level)
