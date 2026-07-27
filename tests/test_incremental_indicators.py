"""Tests pour backend/core/incremental_indicators.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from backend.core.incremental_indicators import IncrementalIndicatorEngine
from backend.core.indicators import adx, atr, rsi
from backend.core.models import Candle, TimeFrame
from backend.strategies.base import BaseStrategy, OpenPosition, StrategyContext, StrategySignal


class _DummyStrategy(BaseStrategy):
    """Stratégie minimale pour tester l'engine."""

    name = "dummy"

    @property
    def min_candles(self) -> dict[str, int]:
        return {"5m": 50, "15m": 20}

    def evaluate(self, ctx: StrategyContext) -> StrategySignal | None:
        return None

    def check_exit(self, ctx: StrategyContext, position: OpenPosition) -> str | None:
        return None

    def get_current_conditions(self, ctx: StrategyContext) -> list[dict]:
        return []

    def compute_indicators(self, candles_by_tf):
        return {}


def _make_candle(
    i: int,
    timeframe: TimeFrame = TimeFrame.M5,
    tf_minutes: int = 5,
) -> Candle:
    base = datetime(2024, 1, 15, tzinfo=timezone.utc)
    price = 100.0 + np.sin(i / 20) * 5
    return Candle(
        timestamp=base + timedelta(minutes=tf_minutes * i),
        open=price - 0.5,
        high=price + 1.0,
        low=price - 1.0,
        close=price + 0.5,
        volume=100.0 + i,
        symbol="BTC/USDT",
        timeframe=timeframe,
    )


class TestIncrementalIndicatorEngine:
    def test_update_and_get_indicators(self):
        """Après suffisamment de candles, get_indicators retourne des valeurs valides."""
        engine = IncrementalIndicatorEngine([_DummyStrategy()])

        # Ajouter 350 candles 5m et 100 candles 15m
        for i in range(350):
            engine.update("BTC/USDT", "5m", _make_candle(i, TimeFrame.M5, 5))
        for i in range(100):
            engine.update("BTC/USDT", "15m", _make_candle(i, TimeFrame.M15, 15))

        result = engine.get_indicators("BTC/USDT")

        assert "5m" in result
        assert "15m" in result

        # Vérifier les champs clés
        ind_5m = result["5m"]
        assert "rsi" in ind_5m
        assert "vwap" in ind_5m
        assert "adx" in ind_5m
        assert "close" in ind_5m
        assert "regime" in ind_5m
        assert not np.isnan(ind_5m["rsi"])
        assert not np.isnan(ind_5m["close"])

    def test_rolling_window_trim(self):
        """Le buffer est borné à max_buffer."""
        engine = IncrementalIndicatorEngine([_DummyStrategy()], max_buffer=100)

        for i in range(200):
            engine.update("BTC/USDT", "5m", _make_candle(i, TimeFrame.M5, 5))

        sizes = engine.get_buffer_sizes()
        assert sizes[("BTC/USDT", "5m")] == 100

    def test_current_candle_update_replaces_last_value(self):
        """Une mise à jour au même timestamp remplace la candle courante."""
        engine = IncrementalIndicatorEngine([_DummyStrategy()])
        candle = _make_candle(0, TimeFrame.M5, 5)
        updated = Candle(
            timestamp=candle.timestamp,
            open=candle.open,
            high=candle.high + 5,
            low=candle.low,
            close=candle.close + 5,
            volume=candle.volume + 10,
            symbol=candle.symbol,
            timeframe=candle.timeframe,
        )

        engine.update("BTC/USDT", "5m", candle)
        engine.update("BTC/USDT", "5m", updated)

        sizes = engine.get_buffer_sizes()
        assert sizes[("BTC/USDT", "5m")] == 1
        assert engine._buffers[("BTC/USDT", "5m")][-1].close == updated.close

    def test_empty_buffer_returns_empty(self):
        """Pas de candles → pas d'indicateurs."""
        engine = IncrementalIndicatorEngine([_DummyStrategy()])
        result = engine.get_indicators("BTC/USDT")
        assert result == {}

    def test_timeframes_property(self):
        """Les timeframes gérés doivent correspondre à ceux des stratégies."""
        engine = IncrementalIndicatorEngine([_DummyStrategy()])
        assert "5m" in engine.timeframes
        assert "15m" in engine.timeframes

    @pytest.mark.parametrize("period", [7, 10, 14, 20])
    def test_atr_matches_batch_for_optimizable_periods(self, period):
        """Every grid ATR candidate uses the exact batch definition."""
        engine = IncrementalIndicatorEngine([_DummyStrategy()])
        candles = [_make_candle(i) for i in range(120)]
        for candle in candles:
            engine.update("BTC/USDT", "5m", candle)

        actual = engine.get_indicators(
            "BTC/USDT", parameters={"atr_period": period},
        )["5m"]["atr"]
        expected = atr(
            np.array([c.high for c in candles]),
            np.array([c.low for c in candles]),
            np.array([c.close for c in candles]),
            period,
        )[-1]
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)

    @pytest.mark.parametrize("period", [7, 14, 20])
    def test_rsi_and_adx_match_batch_periods(self, period):
        engine = IncrementalIndicatorEngine([_DummyStrategy()])
        candles = [_make_candle(i) for i in range(160)]
        for candle in candles:
            engine.update("BTC/USDT", "5m", candle)
        actual = engine.get_indicators(
            "BTC/USDT",
            parameters={"rsi_period": period, "adx_period": period},
        )["5m"]
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])
        expected_adx, expected_plus, expected_minus = adx(
            highs, lows, closes, period,
        )
        assert actual["rsi"] == pytest.approx(rsi(closes, period)[-1], rel=1e-12)
        assert actual["adx"] == pytest.approx(expected_adx[-1], rel=1e-12)
        assert actual["di_plus"] == pytest.approx(expected_plus[-1], rel=1e-12)
        assert actual["di_minus"] == pytest.approx(expected_minus[-1], rel=1e-12)

    def test_adx_long_running_scalar_path_is_stable(self):
        """Repeated portfolio-style ADX calls remain deterministic.

        This protects the allocation-free scalar path used for long Windows
        backtests, where the former per-candle temporary arrays caused severe
        allocator and logging-queue pressure.
        """
        candles = [_make_candle(i) for i in range(500)]
        expected = IncrementalIndicatorEngine._adx_last(candles, 14)

        for _ in range(1_000):
            actual = IncrementalIndicatorEngine._adx_last(candles, 14)
            assert actual == pytest.approx(expected, rel=1e-15, abs=1e-15)

    def test_period_profile_changes_output_without_second_buffer(self):
        engine = IncrementalIndicatorEngine([_DummyStrategy()])
        for i in range(100):
            base = _make_candle(i)
            variable = Candle(
                timestamp=base.timestamp,
                open=base.open,
                high=base.high + (i % 11) * 0.2,
                low=base.low - (i % 7) * 0.1,
                close=base.close,
                volume=base.volume,
                symbol=base.symbol,
                timeframe=base.timeframe,
            )
            engine.update("BTC/USDT", "5m", variable)
        atr_7 = engine.get_indicators(
            "BTC/USDT", parameters={"atr_period": 7},
        )["5m"]["atr"]
        atr_20 = engine.get_indicators(
            "BTC/USDT", parameters={"atr_period": 20},
        )["5m"]["atr"]
        assert atr_7 != pytest.approx(atr_20)
        assert len(engine._buffers) == 1
