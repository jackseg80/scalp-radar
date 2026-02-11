"""Tests pour backend/strategies/momentum.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from backend.core.config import MomentumConfig
from backend.core.models import Candle, Direction, MarketRegime, TimeFrame
from backend.strategies.base import OpenPosition, StrategyContext
from backend.strategies.momentum import MomentumStrategy


def _make_config(**overrides) -> MomentumConfig:
    defaults = {
        "enabled": True,
        "timeframe": "5m",
        "trend_filter_timeframe": "15m",
        "breakout_lookback": 20,
        "volume_confirmation_multiplier": 2.0,
        "atr_multiplier_tp": 2.0,
        "atr_multiplier_sl": 1.0,
        "tp_percent": 0.6,
        "sl_percent": 0.3,
        "weight": 0.20,
    }
    defaults.update(overrides)
    return MomentumConfig(**defaults)


def _make_context(
    main_indicators: dict | None = None,
    filter_indicators: dict | None = None,
) -> StrategyContext:
    indicators = {}
    if main_indicators:
        indicators["5m"] = main_indicators
    if filter_indicators:
        indicators["15m"] = filter_indicators

    return StrategyContext(
        symbol="BTC/USDT",
        timestamp=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
        candles={},
        indicators=indicators,
        current_position=None,
        capital=10_000.0,
        config=None,  # type: ignore[arg-type]
    )


class TestEvaluate:
    def test_long_breakout(self):
        """Prix > rolling_high, 15m bullish trending, volume spike → LONG."""
        strategy = MomentumStrategy(_make_config())
        ctx = _make_context(
            main_indicators={
                "close": 101_000.0,
                "high": 101_500.0,
                "low": 100_500.0,
                "atr": 200.0,
                "atr_sma": 200.0,
                "adx": 20.0,
                "di_plus": 15.0,
                "di_minus": 15.0,
                "volume": 3000.0,
                "volume_sma": 1000.0,
                "rolling_high": 100_800.0,  # close > rolling_high → breakout
                "rolling_low": 99_000.0,
            },
            filter_indicators={
                "adx": 30.0,  # Trending (> 25)
                "di_plus": 25.0,  # Bullish
                "di_minus": 10.0,
            },
        )
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert signal.direction == Direction.LONG
        assert signal.tp_price > signal.entry_price
        assert signal.sl_price < signal.entry_price

    def test_short_breakout(self):
        """Prix < rolling_low, 15m bearish trending, volume spike → SHORT."""
        strategy = MomentumStrategy(_make_config())
        ctx = _make_context(
            main_indicators={
                "close": 98_500.0,
                "high": 99_000.0,
                "low": 98_000.0,
                "atr": 200.0,
                "atr_sma": 200.0,
                "adx": 20.0,
                "di_plus": 15.0,
                "di_minus": 15.0,
                "volume": 3000.0,
                "volume_sma": 1000.0,
                "rolling_high": 100_000.0,
                "rolling_low": 99_000.0,  # close < rolling_low → breakdown
            },
            filter_indicators={
                "adx": 30.0,
                "di_plus": 10.0,
                "di_minus": 25.0,  # Bearish
            },
        )
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert signal.direction == Direction.SHORT
        assert signal.tp_price < signal.entry_price
        assert signal.sl_price > signal.entry_price

    def test_no_breakout(self):
        """Prix entre rolling_high et rolling_low → pas de signal."""
        strategy = MomentumStrategy(_make_config())
        ctx = _make_context(
            main_indicators={
                "close": 100_000.0,
                "high": 100_200.0,
                "low": 99_800.0,
                "atr": 200.0,
                "atr_sma": 200.0,
                "adx": 20.0,
                "di_plus": 15.0,
                "di_minus": 15.0,
                "volume": 3000.0,
                "volume_sma": 1000.0,
                "rolling_high": 100_500.0,  # close < rolling_high
                "rolling_low": 99_500.0,    # close > rolling_low
            },
            filter_indicators={
                "adx": 30.0,
                "di_plus": 25.0,
                "di_minus": 10.0,
            },
        )
        assert strategy.evaluate(ctx) is None

    def test_no_trend_15m_filter(self):
        """ADX 15m < 25 → pas de tendance → pas de momentum trade."""
        strategy = MomentumStrategy(_make_config())
        ctx = _make_context(
            main_indicators={
                "close": 101_000.0,
                "high": 101_500.0,
                "low": 100_500.0,
                "atr": 200.0,
                "atr_sma": 200.0,
                "adx": 20.0,
                "di_plus": 15.0,
                "di_minus": 15.0,
                "volume": 3000.0,
                "volume_sma": 1000.0,
                "rolling_high": 100_800.0,
                "rolling_low": 99_000.0,
            },
            filter_indicators={
                "adx": 18.0,  # < 25 → pas de tendance
                "di_plus": 25.0,
                "di_minus": 10.0,
            },
        )
        assert strategy.evaluate(ctx) is None

    def test_no_volume_filter(self):
        """Volume pas assez élevé → pas de confirmation → pas de signal."""
        strategy = MomentumStrategy(_make_config())
        ctx = _make_context(
            main_indicators={
                "close": 101_000.0,
                "high": 101_500.0,
                "low": 100_500.0,
                "atr": 200.0,
                "atr_sma": 200.0,
                "adx": 20.0,
                "di_plus": 15.0,
                "di_minus": 15.0,
                "volume": 1500.0,
                "volume_sma": 1000.0,  # 1500/1000 = 1.5 < 2.0
                "rolling_high": 100_800.0,
                "rolling_low": 99_000.0,
            },
            filter_indicators={
                "adx": 30.0,
                "di_plus": 25.0,
                "di_minus": 10.0,
            },
        )
        assert strategy.evaluate(ctx) is None

    def test_score_has_components(self):
        """Le signal doit avoir des sous-scores détaillés."""
        strategy = MomentumStrategy(_make_config())
        ctx = _make_context(
            main_indicators={
                "close": 101_000.0,
                "high": 101_500.0,
                "low": 100_500.0,
                "atr": 200.0,
                "atr_sma": 200.0,
                "adx": 20.0,
                "di_plus": 15.0,
                "di_minus": 15.0,
                "volume": 3000.0,
                "volume_sma": 1000.0,
                "rolling_high": 100_800.0,
                "rolling_low": 99_000.0,
            },
            filter_indicators={
                "adx": 30.0,
                "di_plus": 25.0,
                "di_minus": 10.0,
            },
        )
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert "breakout_score" in signal.signals_detail
        assert "volume_score" in signal.signals_detail
        assert "trend_score" in signal.signals_detail


class TestCheckExit:
    def test_exit_adx_drop(self):
        """ADX < 20 → momentum essoufflé → signal_exit."""
        strategy = MomentumStrategy(_make_config())
        position = OpenPosition(
            direction=Direction.LONG,
            entry_price=100_000.0,
            quantity=0.01,
            entry_time=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            tp_price=100_500.0,
            sl_price=99_700.0,
            entry_fee=0.5,
        )
        ctx = _make_context(
            main_indicators={
                "adx": 15.0,  # < 20
                "close": 100_200.0,
            },
        )
        ctx.current_position = position
        assert strategy.check_exit(ctx, position) == "signal_exit"

    def test_no_exit_adx_strong(self):
        """ADX > 20 → momentum encore fort → pas de sortie."""
        strategy = MomentumStrategy(_make_config())
        position = OpenPosition(
            direction=Direction.LONG,
            entry_price=100_000.0,
            quantity=0.01,
            entry_time=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            tp_price=100_500.0,
            sl_price=99_700.0,
            entry_fee=0.5,
        )
        ctx = _make_context(
            main_indicators={
                "adx": 25.0,
                "close": 100_200.0,
            },
        )
        assert strategy.check_exit(ctx, position) is None


class TestMisc:
    def test_min_candles(self):
        strategy = MomentumStrategy(_make_config())
        mc = strategy.min_candles
        assert "5m" in mc
        assert "15m" in mc
        assert mc["5m"] >= 70  # breakout_lookback(20) + 50

    def test_get_params(self):
        config = _make_config()
        strategy = MomentumStrategy(config)
        params = strategy.get_params()
        assert params["breakout_lookback"] == 20
        assert params["atr_multiplier_tp"] == 2.0
