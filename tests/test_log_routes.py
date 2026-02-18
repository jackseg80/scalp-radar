"""Tests pour l'endpoint GET /api/logs — Sprint 31."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.api.log_routes import (
    LOG_FILE,
    _parse_log_line,
    _read_last_lines,
)


# ─── Helpers ──────────────────────────────────────────────────────────────


def _make_loguru_json_line(
    message: str,
    level: str = "INFO",
    module: str = "test_module",
    function: str = "test_func",
    line: int = 42,
    timestamp: str | None = None,
) -> str:
    """Construit une ligne JSON au format loguru serialize=True."""
    if timestamp is None:
        timestamp = datetime.now(tz=timezone.utc).isoformat()
    record = {
        "text": f"{timestamp} | {level:<8} | {module}:{function}:{line} | {message}\n",
        "record": {
            "elapsed": {"repr": "0:00:01", "seconds": 1.0},
            "exception": None,
            "extra": {},
            "file": {"name": "test.py", "path": "test.py"},
            "function": function,
            "level": {"icon": "", "name": level, "no": {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}.get(level, 20)},
            "line": line,
            "message": message,
            "module": module.split(".")[-1],
            "name": module,
            "process": {"id": 1234, "name": "MainProcess"},
            "thread": {"id": 5678, "name": "MainThread"},
            "time": {"repr": timestamp, "timestamp": 1700000000.0},
        },
    }
    return json.dumps(record)


def _write_log_file(tmp_path: Path, lines: list[str]) -> Path:
    """Écrit un fichier log temporaire et retourne son chemin."""
    log_file = tmp_path / "scalp_radar.log"
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_file


# ─── Tests unitaires parsing ─────────────────────────────────────────────


class TestParseLogLine:
    """Tests de parsing des lignes JSON loguru."""

    def test_valid_line(self):
        line = _make_loguru_json_line("Test message", level="WARNING", module="executor")
        result = _parse_log_line(line)
        assert result is not None
        assert result["level"] == "WARNING"
        assert result["message"] == "Test message"
        assert result["module"] == "executor"
        assert result["function"] == "test_func"
        assert result["line"] == 42

    def test_invalid_json(self):
        assert _parse_log_line("not json at all") is None

    def test_missing_record(self):
        assert _parse_log_line('{"text": "hello"}') is None


# ─── Tests endpoint HTTP ─────────────────────────────────────────────────


class TestLogEndpoint:
    """Tests de l'endpoint GET /api/logs."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Crée un fichier log temporaire avec des entrées de test."""
        lines = [
            _make_loguru_json_line("Debug msg", level="DEBUG", module="core.db", timestamp="2025-01-01T10:00:00+00:00"),
            _make_loguru_json_line("Info msg", level="INFO", module="simulator", timestamp="2025-01-01T10:01:00+00:00"),
            _make_loguru_json_line("Warn msg", level="WARNING", module="executor", timestamp="2025-01-01T10:02:00+00:00"),
            _make_loguru_json_line("Error msg", level="ERROR", module="watchdog", timestamp="2025-01-01T10:03:00+00:00"),
            _make_loguru_json_line("Critical msg", level="CRITICAL", module="notifier", timestamp="2025-01-01T10:04:00+00:00"),
        ]
        self.log_file = _write_log_file(tmp_path, lines)
        self._patcher = patch("backend.api.log_routes.LOG_FILE", self.log_file)
        self._patcher.start()

        from backend.api.server import app
        self.client = TestClient(app, raise_server_exceptions=False)

    @pytest.fixture(autouse=True)
    def _teardown(self):
        yield
        self._patcher.stop()

    def test_returns_json_valid(self):
        """GET /api/logs retourne du JSON valide avec le bon format."""
        resp = self.client.get("/api/logs")
        assert resp.status_code == 200
        data = resp.json()
        assert "logs" in data
        assert len(data["logs"]) == 5
        # Vérifier le format des entrées
        entry = data["logs"][0]
        assert "timestamp" in entry
        assert "level" in entry
        assert "module" in entry
        assert "message" in entry

    def test_filter_level(self):
        """Filtre par niveau minimum fonctionne."""
        resp = self.client.get("/api/logs?level=WARNING")
        assert resp.status_code == 200
        logs = resp.json()["logs"]
        assert len(logs) == 3
        levels = [l["level"] for l in logs]
        assert "DEBUG" not in levels
        assert "INFO" not in levels
        assert "WARNING" in levels
        assert "ERROR" in levels

    def test_filter_search(self):
        """Filtre texte case-insensitive fonctionne."""
        resp = self.client.get("/api/logs?search=warn")
        assert resp.status_code == 200
        logs = resp.json()["logs"]
        assert len(logs) == 1
        assert logs[0]["message"] == "Warn msg"

    def test_filter_module(self):
        """Filtre par module fonctionne."""
        resp = self.client.get("/api/logs?module=executor")
        assert resp.status_code == 200
        logs = resp.json()["logs"]
        assert len(logs) == 1
        assert logs[0]["module"] == "executor"

    def test_limit_respected(self):
        """Le paramètre limit est respecté."""
        resp = self.client.get("/api/logs?limit=2")
        assert resp.status_code == 200
        logs = resp.json()["logs"]
        assert len(logs) == 2
        # Les 2 plus récentes (retournées chronologiquement)
        assert logs[0]["level"] == "ERROR"
        assert logs[1]["level"] == "CRITICAL"
