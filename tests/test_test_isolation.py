"""Guard regression tests: no actual external connection is attempted."""
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys

import pytest


@pytest.mark.parametrize("uri", [False, True])
def test_repository_database_is_blocked(uri):
    target = Path(__file__).resolve().parents[1] / "data/scalp_radar.db"
    path = target.as_uri() + "?mode=ro" if uri else str(target)
    with pytest.raises(RuntimeError, match="repository database"):
        sqlite3.connect(path, uri=uri)


def test_temporary_database_is_allowed(tmp_path):
    with sqlite3.connect(tmp_path / "test.db") as db:
        assert db.execute("SELECT 1").fetchone() == (1,)


def test_repository_output_is_blocked():
    target = Path(__file__).resolve().parents[1] / "data/forbidden-test-output.json"
    with pytest.raises(RuntimeError, match="repository data/config"):
        target.write_text("must never be written", encoding="utf-8")


def test_external_network_is_blocked():
    with pytest.raises(RuntimeError, match="external network"):
        socket.getaddrinfo("api.telegram.org", 443)
    with socket.socket() as sock:
        with pytest.raises(RuntimeError, match="external network"):
            sock.connect(("198.51.100.1", 443))


def test_real_wfo_worker_is_blocked():
    with pytest.raises(RuntimeError, match="real WFO worker"):
        subprocess.Popen([sys.executable, "scripts/wfo_worker.py", "--help"])
