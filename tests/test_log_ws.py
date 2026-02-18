"""Tests pour le sink loguru WS et le système subscribe/unsubscribe — Sprint 31."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from backend.core.logging_setup import (
    _log_buffer,
    _log_subscribers,
    _ws_log_sink,
    get_log_buffer,
    subscribe_logs,
    unsubscribe_logs,
)


@pytest.fixture(autouse=True)
def _cleanup():
    """Nettoie le buffer et les subscribers entre chaque test."""
    _log_buffer.clear()
    _log_subscribers.clear()
    yield
    _log_buffer.clear()
    _log_subscribers.clear()


def _make_message(level_name: str = "WARNING", level_no: int = 30, message: str = "test"):
    """Crée un mock de message loguru."""
    msg = MagicMock()
    msg.record = {
        "level": MagicMock(name=level_name, no=level_no),
        "time": MagicMock(),
        "name": "test_module",
        "function": "test_func",
        "line": 42,
        "message": message,
    }
    # loguru level.name est un attribut, pas la prop name de MagicMock
    msg.record["level"].name = level_name
    msg.record["level"].no = level_no
    msg.record["time"].isoformat.return_value = "2025-01-01T10:00:00+00:00"
    return msg


class TestWsSinkCapture:
    """Vérifie que le sink capture uniquement WARNING+."""

    def test_captures_warning(self):
        """WARNING est capturé dans le buffer."""
        _ws_log_sink(_make_message("WARNING", 30, "warn msg"))
        assert len(_log_buffer) == 1
        assert _log_buffer[0]["level"] == "WARNING"
        assert _log_buffer[0]["message"] == "warn msg"

    def test_captures_error(self):
        """ERROR est capturé dans le buffer."""
        _ws_log_sink(_make_message("ERROR", 40, "err msg"))
        assert len(_log_buffer) == 1
        assert _log_buffer[0]["level"] == "ERROR"

    def test_skips_info(self):
        """INFO n'est pas capturé (niveau < 30)."""
        _ws_log_sink(_make_message("INFO", 20, "info msg"))
        assert len(_log_buffer) == 0

    def test_skips_debug(self):
        """DEBUG n'est pas capturé."""
        _ws_log_sink(_make_message("DEBUG", 10, "debug msg"))
        assert len(_log_buffer) == 0


class TestBufferMaxlen:
    """Vérifie que le buffer circulaire respecte maxlen=20."""

    def test_buffer_maxlen(self):
        """Le buffer ne dépasse pas 20 éléments."""
        for i in range(30):
            _ws_log_sink(_make_message("WARNING", 30, f"msg {i}"))
        assert len(_log_buffer) == 20
        # Les plus anciens sont éjectés
        assert _log_buffer[0]["message"] == "msg 10"
        assert _log_buffer[-1]["message"] == "msg 29"


class TestSubscribeUnsubscribe:
    """Vérifie le mécanisme subscribe/unsubscribe."""

    def test_subscribe_creates_queue(self):
        """subscribe_logs() retourne une queue et l'ajoute aux subscribers."""
        q = subscribe_logs()
        assert isinstance(q, asyncio.Queue)
        assert q in _log_subscribers

    def test_unsubscribe_removes_queue(self):
        """unsubscribe_logs() retire la queue des subscribers."""
        q = subscribe_logs()
        assert q in _log_subscribers
        unsubscribe_logs(q)
        assert q not in _log_subscribers

    def test_unsubscribe_idempotent(self):
        """unsubscribe sur une queue déjà retirée ne crash pas."""
        q = subscribe_logs()
        unsubscribe_logs(q)
        unsubscribe_logs(q)  # pas d'erreur

    def test_sink_pushes_to_subscriber(self):
        """Le sink pousse les entrées dans les queues des subscribers."""
        q = subscribe_logs()
        _ws_log_sink(_make_message("ERROR", 40, "pushed msg"))
        assert not q.empty()
        entry = q.get_nowait()
        assert entry["message"] == "pushed msg"
        assert entry["level"] == "ERROR"

    def test_sink_skips_full_queue(self):
        """Si la queue est pleine, le sink ne bloque pas (drop silencieux)."""
        q = subscribe_logs()
        # Remplir la queue (maxsize=50)
        for i in range(50):
            _ws_log_sink(_make_message("WARNING", 30, f"fill {i}"))
        # La queue est pleine, le prochain ne doit pas bloquer
        _ws_log_sink(_make_message("ERROR", 40, "overflow"))
        assert q.qsize() == 50  # Pas plus que maxsize

    def test_get_log_buffer_returns_copy(self):
        """get_log_buffer() retourne une copie du buffer."""
        _ws_log_sink(_make_message("WARNING", 30, "buf msg"))
        buf = get_log_buffer()
        assert len(buf) == 1
        assert buf[0]["message"] == "buf msg"
        # C'est une copie, pas une référence
        buf.clear()
        assert len(get_log_buffer()) == 1
