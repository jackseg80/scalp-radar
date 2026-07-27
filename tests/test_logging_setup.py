"""Regression tests for Loguru setup modes."""

from __future__ import annotations

from pathlib import Path

import backend.core.logging_setup as logging_setup


class _FakeLogger:
    def __init__(self) -> None:
        self.add_calls: list[tuple[tuple, dict]] = []

    def remove(self) -> None:
        pass

    def add(self, *args, **kwargs) -> int:
        self.add_calls.append((args, kwargs))
        return len(self.add_calls)

    def info(self, *args, **kwargs) -> None:
        pass


def test_batch_logging_can_disable_multiprocessing_enqueue(monkeypatch, tmp_path):
    """A batch replay must not create queued multiprocessing file writers."""
    fake = _FakeLogger()
    monkeypatch.setattr(logging_setup, "logger", fake)

    logging_setup.setup_logging(
        level="WARNING", log_dir=tmp_path, enqueue=False,
    )

    file_sink_options = [
        kwargs for args, kwargs in fake.add_calls
        if args and isinstance(args[0], Path)
    ]
    assert len(file_sink_options) == 2
    assert all(options["enqueue"] is False for options in file_sink_options)
