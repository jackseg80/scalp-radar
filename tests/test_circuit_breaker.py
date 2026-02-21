"""Tests Circuit Breaker Runner Isolation (Sprint 36a).

Couvre :
- Section 1 : GridStrategyRunner circuit breaker (~5 tests)
- Section 2 : LiveStrategyRunner circuit breaker (~2 tests)
- Section 3 : Status exposure (~2 tests)
- Section 4 : Crash recording via on_candle (~1 test)
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.backtesting.simulator import GridStrategyRunner, LiveStrategyRunner
from backend.core.grid_position_manager import GridPositionManager
from backend.core.incremental_indicators import IncrementalIndicatorEngine
from backend.core.models import Candle, TimeFrame
from backend.core.position_manager import PositionManager, PositionManagerConfig
from backend.strategies.base_grid import BaseGridStrategy


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_candle(
    close: float = 100_000.0,
    ts: datetime | None = None,
    symbol: str = "BTC/USDT",
    tf: TimeFrame = TimeFrame.H1,
) -> Candle:
    return Candle(
        timestamp=ts or datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=100.0,
        symbol=symbol,
        timeframe=tf,
    )


def _make_gpm_config() -> PositionManagerConfig:
    return PositionManagerConfig(
        leverage=6,
        maker_fee=0.0002,
        taker_fee=0.0006,
        slippage_pct=0.0005,
        high_vol_slippage_mult=2.0,
        max_risk_per_trade=0.02,
    )


def _make_pm_config() -> PositionManagerConfig:
    return PositionManagerConfig(
        leverage=15,
        maker_fee=0.0002,
        taker_fee=0.0006,
        slippage_pct=0.0005,
        high_vol_slippage_mult=2.0,
        max_risk_per_trade=0.02,
    )


def _make_mock_strategy(name: str = "envelope_dca") -> MagicMock:
    strategy = MagicMock(spec=BaseGridStrategy)
    strategy.name = name
    config = MagicMock()
    config.timeframe = "1h"
    config.ma_period = 7
    config.leverage = 6
    strategy._config = config
    strategy.min_candles = {"1h": 50}
    strategy.max_positions = 2
    strategy.compute_grid.return_value = []
    strategy.should_close_all.return_value = None
    strategy.get_tp_price.return_value = float("nan")
    strategy.get_sl_price.return_value = float("nan")
    strategy.get_current_conditions.return_value = []
    return strategy


def _make_mock_config() -> MagicMock:
    config = MagicMock()
    config.risk.initial_capital = 10_000.0
    config.risk.max_margin_ratio = 0.70
    config.risk.kill_switch.max_session_loss_percent = 5.0
    config.risk.kill_switch.max_daily_loss_percent = 10.0
    config.risk.position.max_risk_per_trade_percent = 2.0
    return config


def _make_grid_runner(strategy=None, config=None) -> GridStrategyRunner:
    if strategy is None:
        strategy = _make_mock_strategy()
    if config is None:
        config = _make_mock_config()

    indicator_engine = MagicMock(spec=IncrementalIndicatorEngine)
    indicator_engine.get_indicators.return_value = {}
    indicator_engine.update = MagicMock()

    gpm = GridPositionManager(_make_gpm_config())
    data_engine = MagicMock()
    data_engine.get_funding_rate.return_value = None
    data_engine.get_open_interest.return_value = []

    runner = GridStrategyRunner(
        strategy=strategy,
        config=config,
        indicator_engine=indicator_engine,
        grid_position_manager=gpm,
        data_engine=data_engine,
    )
    runner._is_warming_up = False
    runner._warmup_ended_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return runner


def _make_live_runner(strategy=None, config=None) -> LiveStrategyRunner:
    if strategy is None:
        strategy = MagicMock()
        strategy.name = "vwap_rsi"
        strategy.min_candles = {"5m": 20}
    if config is None:
        config = _make_mock_config()

    indicator_engine = MagicMock(spec=IncrementalIndicatorEngine)
    indicator_engine.get_indicators.return_value = {
        "5m": {
            "rsi_14": 50.0,
            "vwap_distance_pct": 0.0,
            "adx": 15.0,
            "di_plus": 10.0,
            "di_minus": 12.0,
            "atr": 500.0,
            "atr_sma": 450.0,
            "close": 100_000.0,
        }
    }

    data_engine = MagicMock()
    data_engine.get_funding_rate.return_value = None
    data_engine.get_open_interest.return_value = []

    return LiveStrategyRunner(
        strategy=strategy,
        config=config,
        indicator_engine=indicator_engine,
        position_manager=PositionManager(_make_pm_config()),
        data_engine=data_engine,
    )


# ─── Section 1 : GridStrategyRunner Circuit Breaker ──────────────────────────


class TestGridRunnerCircuitBreaker:
    def test_circuit_breaker_disabled_by_default(self):
        """_circuit_breaker_open est False au init."""
        runner = _make_grid_runner()
        assert runner._circuit_breaker_open is False
        assert runner._crash_times == []

    def test_record_crash_increments(self):
        """1 crash enregistré, crash_count = 1, circuit_breaker_open = False."""
        runner = _make_grid_runner()
        runner._record_crash("BTC/USDT", RuntimeError("test"))
        assert len(runner._crash_times) == 1
        assert runner._circuit_breaker_open is False

    def test_circuit_breaker_triggers_after_3_crashes(self):
        """3 crashes en <10min → circuit_breaker_open = True."""
        runner = _make_grid_runner()
        for i in range(3):
            runner._record_crash("BTC/USDT", RuntimeError(f"crash {i}"))
        assert runner._circuit_breaker_open is True
        assert len(runner._crash_times) == 3

    def test_circuit_breaker_not_triggered_outside_window(self):
        """3 crashes espacés de >10min → reste False."""
        runner = _make_grid_runner()
        # Simuler 2 crashes anciens (hors fenêtre)
        old_time = time.monotonic() - 700  # > 600s window
        runner._crash_times = [old_time, old_time + 1]
        # 1 crash récent
        runner._record_crash("BTC/USDT", RuntimeError("recent"))
        # Seul le crash récent est dans la fenêtre
        assert len(runner._crash_times) == 1
        assert runner._circuit_breaker_open is False

    @pytest.mark.asyncio
    async def test_on_candle_skipped_when_disabled(self):
        """Quand _circuit_breaker_open = True, on_candle return immédiatement."""
        runner = _make_grid_runner()
        runner._circuit_breaker_open = True
        candle = _make_candle()

        # Remplir le buffer pour que la logique interne passerait normalement
        runner._close_buffer["BTC/USDT"] = deque(maxlen=50)
        for i in range(20):
            runner._close_buffer["BTC/USDT"].append(100_000.0)

        await runner.on_candle("BTC/USDT", "1h", candle)

        # compute_grid ne doit JAMAIS être appelé
        runner._strategy.compute_grid.assert_not_called()


# ─── Section 2 : LiveStrategyRunner Circuit Breaker ──────────────────────────


class TestLiveRunnerCircuitBreaker:
    def test_live_runner_circuit_breaker(self):
        """3 crashes en <10min → circuit_breaker_open = True."""
        runner = _make_live_runner()
        for i in range(3):
            runner._record_crash("BTC/USDT", RuntimeError(f"crash {i}"))
        assert runner._circuit_breaker_open is True

    @pytest.mark.asyncio
    async def test_live_runner_skips_when_disabled(self):
        """Quand circuit breaker ouvert, on_candle return immédiatement."""
        runner = _make_live_runner()
        runner._circuit_breaker_open = True
        candle = _make_candle()

        await runner.on_candle("BTC/USDT", "5m", candle)

        # evaluate ne doit pas être appelé
        runner._strategy.evaluate.assert_not_called()


# ─── Section 3 : Status exposure ─────────────────────────────────────────────


class TestCircuitBreakerStatus:
    def test_grid_status_exposes_circuit_breaker(self):
        """get_status() retourne circuit_breaker et crash_count."""
        runner = _make_grid_runner()
        status = runner.get_status()
        assert "circuit_breaker" in status
        assert "crash_count" in status
        assert status["circuit_breaker"] is False
        assert status["crash_count"] == 0

    def test_grid_status_after_crashes(self):
        """get_status() reflète l'état après des crashes."""
        runner = _make_grid_runner()
        runner._record_crash("BTC/USDT", RuntimeError("test1"))
        runner._record_crash("BTC/USDT", RuntimeError("test2"))
        status = runner.get_status()
        assert status["circuit_breaker"] is False
        assert status["crash_count"] == 2

    def test_live_status_exposes_circuit_breaker(self):
        """LiveStrategyRunner.get_status() inclut circuit_breaker."""
        runner = _make_live_runner()
        status = runner.get_status()
        assert "circuit_breaker" in status
        assert status["circuit_breaker"] is False
        assert status["crash_count"] == 0


# ─── Section 4 : Crash recording via on_candle ──────────────────────────────


class TestCrashRecordingOnCandle:
    @pytest.mark.asyncio
    async def test_crash_in_on_candle_recorded(self):
        """Exception dans _on_candle_inner est capturée par le circuit breaker."""
        runner = _make_grid_runner()
        # Remplir le buffer pour dépasser la SMA check
        runner._close_buffer["BTC/USDT"] = deque(maxlen=50)
        for i in range(20):
            runner._close_buffer["BTC/USDT"].append(100_000.0)

        # Injecter une exception dans compute_grid
        runner._strategy.compute_grid.side_effect = RuntimeError("boom")

        candle = _make_candle()
        await runner.on_candle("BTC/USDT", "1h", candle)

        # Le crash doit être enregistré
        assert len(runner._crash_times) == 1
        assert runner._circuit_breaker_open is False
        # L'erreur doit être signalée pour Telegram
        assert runner._last_indicator_error is not None
        assert "CRASH on_candle" in runner._last_indicator_error[1]
