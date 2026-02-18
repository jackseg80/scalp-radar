"""Routes API pour la lecture des logs backend.

GET /api/logs — Lit les N dernières lignes du fichier log JSON (loguru serialize=True)
et les retourne filtrées par niveau, module, texte et date.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Query
from loguru import logger

router = APIRouter(prefix="/api/logs", tags=["logs"])

# Chemin du fichier log (même que logging_setup.py)
LOG_FILE = Path("logs") / "scalp_radar.log"

# Constantes de sécurité
MAX_LIMIT = 500
MAX_FILE_READ = 10 * 1024 * 1024  # 10 MB max à lire
CHUNK_SIZE = 8 * 1024  # 8 KB chunks pour la lecture inversée

# Mapping niveau → numéro pour le filtrage
LEVEL_ORDER = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


def _parse_log_line(line: str) -> dict | None:
    """Parse une ligne JSON loguru sérialisée. Retourne None si invalide."""
    try:
        data = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None

    record = data.get("record")
    if not record:
        return None

    level_info = record.get("level", {})
    time_info = record.get("time", {})

    # loguru serialize=True met le timestamp dans record.time.repr ou record.time.timestamp
    timestamp = time_info.get("repr", "")
    if not timestamp and "timestamp" in time_info:
        try:
            timestamp = datetime.fromtimestamp(
                time_info["timestamp"], tz=timezone.utc
            ).isoformat()
        except (ValueError, TypeError, OSError):
            timestamp = ""

    return {
        "timestamp": timestamp,
        "level": level_info.get("name", "UNKNOWN"),
        "module": record.get("name", ""),
        "function": record.get("function", ""),
        "line": record.get("line", 0),
        "message": record.get("message", ""),
    }


def _read_last_lines(file_path: Path, max_bytes: int = MAX_FILE_READ) -> list[str]:
    """Lit les dernières lignes d'un fichier par chunks depuis la fin.

    Ne charge jamais tout le fichier en mémoire.
    """
    if not file_path.exists():
        return []

    file_size = file_path.stat().st_size
    if file_size == 0:
        return []

    # Limiter la lecture aux derniers max_bytes
    start_pos = max(0, file_size - max_bytes)

    lines: list[str] = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(start_pos)

        # Si on ne lit pas depuis le début, skip la première ligne partielle
        if start_pos > 0:
            f.readline()

        for line in f:
            stripped = line.strip()
            if stripped:
                lines.append(stripped)

    return lines


@router.get("")
async def get_logs(
    limit: int = Query(default=100, ge=1, le=MAX_LIMIT),
    level: str | None = Query(default=None),
    search: str | None = Query(default=None),
    module: str | None = Query(default=None),
    since: str | None = Query(default=None),
):
    """Retourne les N dernières lignes de log filtrées.

    Args:
        limit: Nombre max de lignes (1-500).
        level: Niveau minimum (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        search: Filtre texte sur le message (case-insensitive).
        module: Filtre par nom de module.
        since: ISO datetime, logs après cette date uniquement.
    """
    # Parser le seuil de niveau
    min_level_no = 0
    if level:
        level_upper = level.upper()
        min_level_no = LEVEL_ORDER.get(level_upper, 0)

    # Parser la date since
    since_dt: datetime | None = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            pass

    search_lower = search.lower() if search else None
    module_lower = module.lower() if module else None

    # Lire les lignes depuis le fichier
    raw_lines = _read_last_lines(LOG_FILE)

    # Parser et filtrer en partant de la fin (les plus récentes d'abord)
    results: list[dict] = []
    for raw_line in reversed(raw_lines):
        if len(results) >= limit:
            break

        entry = _parse_log_line(raw_line)
        if entry is None:
            continue

        # Filtre niveau
        entry_level_no = LEVEL_ORDER.get(entry["level"], 0)
        if entry_level_no < min_level_no:
            continue

        # Filtre module
        if module_lower and module_lower not in entry["module"].lower():
            continue

        # Filtre texte
        if search_lower and search_lower not in entry["message"].lower():
            continue

        # Filtre date
        if since_dt and entry["timestamp"]:
            try:
                entry_dt = datetime.fromisoformat(entry["timestamp"])
                if entry_dt <= since_dt:
                    continue
            except ValueError:
                pass

        results.append(entry)

    # Retourner dans l'ordre chronologique (les plus anciennes en premier)
    results.reverse()

    return {"logs": results}
