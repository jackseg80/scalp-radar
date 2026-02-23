"""Tests Phase 1 — Kill switch global ne gèle plus les indicateurs.

Vérifie que _dispatch_candle() met à jour les indicateurs même quand
le kill switch global est actif, et que update_indicators_only() est
appelé sur les runners.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.core.models import Candle, TimeFrame


def _make_candle(close: float = 50_000.0) -> Candle:
    return Candle(
        timestamp=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        open=close,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=100.0,
        symbol="BTC/USDT",
        timeframe=TimeFrame.H1,
    )


def _make_simulator() -> MagicMock:
    """Crée un Simulator mock avec les attributs nécessaires pour _dispatch_candle."""
    from backend.backtesting.simulator import Simulator

    sim = MagicMock(spec=Simulator)
    sim._running = True
    sim._indicator_engine = MagicMock()
    sim._global_kill_switch = False
    sim._conditions_cache = {"old": True}
    sim._runners = []
    sim._db = None
    sim._notifier = None
    sim._collision_warnings = []
    return sim


class TestKillSwitchIndicators:
    """Kill switch global ne doit pas empêcher la mise à jour des indicateurs."""

    @pytest.mark.asyncio
    async def test_dispatch_updates_indicator_engine_during_kill_switch(self):
        """indicator_engine.update() est appelé même quand kill switch actif."""
        from backend.backtesting.simulator import Simulator

        sim = _make_simulator()
        sim._global_kill_switch = True

        runner = MagicMock()
        runner.update_indicators_only = MagicMock()
        sim._runners = [runner]

        candle = _make_candle()
        # Appel direct de la vraie méthode avec le mock comme self
        await Simulator._dispatch_candle(sim, "BTC/USDT", "1h", candle)

        sim._indicator_engine.update.assert_called_once_with("BTC/USDT", "1h", candle)

    @pytest.mark.asyncio
    async def test_dispatch_calls_update_indicators_only_during_kill_switch(self):
        """update_indicators_only() appelé sur chaque runner quand kill switch actif."""
        from backend.backtesting.simulator import Simulator

        sim = _make_simulator()
        sim._global_kill_switch = True

        runner1 = MagicMock()
        runner1.update_indicators_only = MagicMock()
        runner2 = MagicMock()
        runner2.update_indicators_only = MagicMock()
        sim._runners = [runner1, runner2]

        candle = _make_candle()
        await Simulator._dispatch_candle(sim, "BTC/USDT", "1h", candle)

        runner1.update_indicators_only.assert_called_once_with("BTC/USDT", "1h", candle)
        runner2.update_indicators_only.assert_called_once_with("BTC/USDT", "1h", candle)

    @pytest.mark.asyncio
    async def test_dispatch_does_not_call_on_candle_during_kill_switch(self):
        """runner.on_candle() NE doit PAS être appelé quand kill switch actif."""
        from backend.backtesting.simulator import Simulator

        sim = _make_simulator()
        sim._global_kill_switch = True

        runner = MagicMock()
        runner.on_candle = AsyncMock()
        runner.update_indicators_only = MagicMock()
        sim._runners = [runner]

        candle = _make_candle()
        await Simulator._dispatch_candle(sim, "BTC/USDT", "1h", candle)

        runner.on_candle.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_normal_without_kill_switch(self):
        """Sans kill switch, on_candle() est appelé normalement."""
        from backend.backtesting.simulator import Simulator

        sim = _make_simulator()
        sim._global_kill_switch = False

        runner = MagicMock()
        runner.name = "grid_atr"
        runner.on_candle = AsyncMock()
        runner._stats = MagicMock()
        runner._stats.total_trades = 0
        runner._circuit_breaker_open = False
        runner._last_indicator_error = None
        runner._pending_journal_events = []
        sim._runners = [runner]

        def get_position_symbols(r):
            return set()
        sim._get_position_symbols = get_position_symbols

        candle = _make_candle()
        await Simulator._dispatch_candle(sim, "BTC/USDT", "1h", candle)

        sim._indicator_engine.update.assert_called_once()
        runner.on_candle.assert_called_once()

    @pytest.mark.asyncio
    async def test_conditions_cache_invalidated_during_kill_switch(self):
        """Le cache conditions est invalidé même pendant le kill switch."""
        from backend.backtesting.simulator import Simulator

        sim = _make_simulator()
        sim._global_kill_switch = True
        sim._conditions_cache = {"old": True}
        sim._runners = [MagicMock()]

        await Simulator._dispatch_candle(sim, "BTC/USDT", "1h", _make_candle())

        assert sim._conditions_cache is None


class TestRunnerKillSwitchIndicators:
    """Kill switch RUNNER ne doit pas geler le close_buffer / SMA."""

    @pytest.mark.asyncio
    async def test_runner_kill_switch_updates_close_buffer(self):
        """on_candle() avec kill switch runner met à jour _close_buffer."""
        from backend.backtesting.simulator import GridStrategyRunner
        from unittest.mock import MagicMock, patch
        from collections import deque

        runner = MagicMock(spec=GridStrategyRunner)
        runner._kill_switch_triggered = True
        runner._circuit_breaker_open = False
        runner._strategy_tf = "1h"
        runner._ma_period = 20
        runner._close_buffer = {}

        # Redirige vers la vraie implémentation de update_indicators_only
        runner.update_indicators_only = lambda s, tf, c: GridStrategyRunner.update_indicators_only(
            runner, s, tf, c
        )

        candle = _make_candle(close=79_000.0)
        await GridStrategyRunner.on_candle(runner, "SOL/USDT", "1h", candle)

        assert "SOL/USDT" in runner._close_buffer
        assert runner._close_buffer["SOL/USDT"][-1] == 79_000.0

    @pytest.mark.asyncio
    async def test_runner_kill_switch_no_trades(self):
        """on_candle() avec kill switch runner ne crée aucun trade ni position."""
        from backend.backtesting.simulator import GridStrategyRunner

        runner = MagicMock(spec=GridStrategyRunner)
        runner._kill_switch_triggered = True
        runner._circuit_breaker_open = False
        runner._strategy_tf = "1h"
        runner._ma_period = 20
        runner._close_buffer = {}
        runner._trades = []
        runner._positions = {}

        runner.update_indicators_only = lambda s, tf, c: GridStrategyRunner.update_indicators_only(
            runner, s, tf, c
        )
        runner._on_candle_inner = MagicMock()

        candle = _make_candle(close=79_000.0)
        await GridStrategyRunner.on_candle(runner, "SOL/USDT", "1h", candle)

        assert len(runner._trades) == 0
        assert len(runner._positions) == 0
        runner._on_candle_inner.assert_not_called()
