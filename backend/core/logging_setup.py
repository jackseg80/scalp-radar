"""Configuration du logging avec loguru.

Console colorée pour le dev, fichiers JSON structurés pour la prod.
"""

import sys
from pathlib import Path

from loguru import logger


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

    logger.info("Logging initialisé (level={})", level)
