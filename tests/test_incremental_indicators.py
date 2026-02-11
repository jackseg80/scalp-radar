"""Tests pour backend/core/incremental_indicators.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from backend.core.incremental_indicators import IncrementalIndicatorEngine
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

    def test_duplicate_candle_ignored(self):
        """Les bougies avec timestamp <= dernier sont ignorées."""
        engine = IncrementalIndicatorEngine([_DummyStrategy()])
        candle = _make_candle(0, TimeFrame.M5, 5)

        engine.update("BTC/USDT", "5m", candle)
        engine.update("BTC/USDT", "5m", candle)  # Doublon

        sizes = engine.get_buffer_sizes()
        assert sizes[("BTC/USDT", "5m")] == 1

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
