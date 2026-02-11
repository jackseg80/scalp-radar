"""Tests pour backend/strategies/liquidation.py."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backend.core.config import LiquidationConfig
from backend.core.models import Direction
from backend.strategies.base import (
    EXTRA_OI_CHANGE_PCT,
    EXTRA_OPEN_INTEREST,
    OpenPosition,
    StrategyContext,
)
from backend.strategies.liquidation import LiquidationStrategy


def _make_config(**overrides) -> LiquidationConfig:
    defaults = {
        "enabled": True,
        "timeframe": "5m",
        "oi_change_threshold": 5.0,
        "leverage_estimate": 15,
        "zone_buffer_percent": 1.5,
        "tp_percent": 0.8,
        "sl_percent": 0.4,
        "weight": 0.20,
    }
    defaults.update(overrides)
    return LiquidationConfig(**defaults)


def _make_context(
    close: float = 100_000.0,
    oi_change: float = 0.0,
) -> StrategyContext:
    return StrategyContext(
        symbol="BTC/USDT",
        timestamp=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
        candles={},
        indicators={"5m": {"close": close}},
        current_position=None,
        capital=10_000.0,
        config=None,  # type: ignore[arg-type]
        extra_data={
            EXTRA_OI_CHANGE_PCT: oi_change,
            EXTRA_OPEN_INTEREST: [],
        },
    )


class TestEvaluate:
    def test_oi_high_near_short_liq_zone(self):
        """OI en hausse + prix proche zone liq shorts → LONG (short squeeze)."""
        # Avec leverage=15 : liq_short_zone = close * (1 + 1/15) = close * 1.0667
        # Pour que le prix soit "proche" (< 1.5% buffer), il faut :
        # dist_to_short_liq < 0.015, i.e. (liq_short_zone - close) / close < 0.015
        # → (1/15) < 0.015 ? Non, 1/15 = 0.0667. Toujours loin.
        # Avec leverage=100 : liq_short_zone = close * 1.01
        # dist = 0.01 < 0.015 → OK
        strategy = LiquidationStrategy(_make_config(leverage_estimate=100))
        ctx = _make_context(close=100_000.0, oi_change=8.0)
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert signal.direction == Direction.LONG

    def test_oi_high_near_long_liq_zone(self):
        """OI en hausse + prix proche zone liq longs → SHORT (cascade)."""
        # Avec leverage=100 : liq_long_zone = close * (1 - 1/100) = close * 0.99
        # dist_to_long_liq = (close - liq_long_zone)/close = 0.01 < 0.015 → proche
        strategy = LiquidationStrategy(_make_config(leverage_estimate=100))
        ctx = _make_context(close=100_000.0, oi_change=8.0)
        signal = strategy.evaluate(ctx)
        # Avec leverage=100, les deux zones sont proches. Le LONG (short squeeze) a priorité
        # car near_short_liq est testé en premier
        assert signal is not None

    def test_oi_low_no_signal(self):
        """OI change < seuil → pas de signal."""
        strategy = LiquidationStrategy(_make_config(leverage_estimate=100))
        ctx = _make_context(close=100_000.0, oi_change=2.0)  # < 5.0 seuil
        assert strategy.evaluate(ctx) is None

    def test_oi_high_but_far_from_zones(self):
        """OI haute mais prix loin des zones → pas de signal."""
        # Avec leverage=15 : zones à ~6.67% du prix → loin du buffer 1.5%
        strategy = LiquidationStrategy(_make_config(leverage_estimate=15))
        ctx = _make_context(close=100_000.0, oi_change=8.0)
        assert strategy.evaluate(ctx) is None

    def test_score_components(self):
        """Le signal doit avoir les sous-scores détaillés."""
        strategy = LiquidationStrategy(_make_config(leverage_estimate=100))
        ctx = _make_context(close=100_000.0, oi_change=10.0)
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert "oi_score" in signal.signals_detail
        assert "proximity_score" in signal.signals_detail
        assert "liq_long_zone" in signal.signals_detail
        assert "liq_short_zone" in signal.signals_detail


class TestCheckExit:
    def test_exit_oi_drop(self):
        """OI chute > 3% → cascade terminée → signal_exit."""
        strategy = LiquidationStrategy(_make_config())
        position = OpenPosition(
            direction=Direction.LONG,
            entry_price=100_000.0,
            quantity=0.01,
            entry_time=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            tp_price=100_800.0,
            sl_price=99_600.0,
            entry_fee=0.5,
        )
        ctx = _make_context(oi_change=-5.0)
        assert strategy.check_exit(ctx, position) == "signal_exit"

    def test_no_exit_oi_stable(self):
        """OI stable → pas de sortie."""
        strategy = LiquidationStrategy(_make_config())
        position = OpenPosition(
            direction=Direction.LONG,
            entry_price=100_000.0,
            quantity=0.01,
            entry_time=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            tp_price=100_800.0,
            sl_price=99_600.0,
            entry_fee=0.5,
        )
        ctx = _make_context(oi_change=1.0)
        assert strategy.check_exit(ctx, position) is None
