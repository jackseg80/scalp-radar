"""Tests pour backend/strategies/funding.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backend.core.config import FundingConfig
from backend.core.models import Direction
from backend.strategies.base import EXTRA_FUNDING_RATE, OpenPosition, StrategyContext
from backend.strategies.funding import FundingStrategy


def _make_config(**overrides) -> FundingConfig:
    defaults = {
        "enabled": True,
        "timeframe": "15m",
        "extreme_positive_threshold": 0.03,
        "extreme_negative_threshold": -0.03,
        "entry_delay_minutes": 5,
        "tp_percent": 0.4,
        "sl_percent": 0.2,
        "weight": 0.15,
    }
    defaults.update(overrides)
    return FundingConfig(**defaults)


def _make_context(
    funding_rate: float | None = None,
    close: float = 100_000.0,
    timestamp: datetime | None = None,
) -> StrategyContext:
    ts = timestamp or datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
    extra_data = {}
    if funding_rate is not None:
        extra_data[EXTRA_FUNDING_RATE] = funding_rate

    return StrategyContext(
        symbol="BTC/USDT",
        timestamp=ts,
        candles={},
        indicators={"15m": {"close": close}},
        current_position=None,
        capital=10_000.0,
        config=None,  # type: ignore[arg-type]
        extra_data=extra_data,
    )


class TestEvaluate:
    def test_extreme_negative_long(self):
        """Funding rate < -0.03% → LONG après entry delay."""
        strategy = FundingStrategy(_make_config(entry_delay_minutes=0))
        # Première détection
        ctx1 = _make_context(funding_rate=-0.05, timestamp=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc))
        strategy.evaluate(ctx1)  # Enregistre la détection

        # Deuxième appel : delay écoulé (0 min)
        ctx2 = _make_context(funding_rate=-0.05, timestamp=datetime(2024, 1, 15, 12, 1, tzinfo=timezone.utc))
        signal = strategy.evaluate(ctx2)
        assert signal is not None
        assert signal.direction == Direction.LONG
        assert signal.tp_price > signal.entry_price
        assert signal.sl_price < signal.entry_price

    def test_extreme_positive_short(self):
        """Funding rate > 0.03% → SHORT après entry delay."""
        strategy = FundingStrategy(_make_config(entry_delay_minutes=0))
        ctx1 = _make_context(funding_rate=0.05, timestamp=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc))
        strategy.evaluate(ctx1)

        ctx2 = _make_context(funding_rate=0.05, timestamp=datetime(2024, 1, 15, 12, 1, tzinfo=timezone.utc))
        signal = strategy.evaluate(ctx2)
        assert signal is not None
        assert signal.direction == Direction.SHORT
        assert signal.tp_price < signal.entry_price
        assert signal.sl_price > signal.entry_price

    def test_neutral_no_signal(self):
        """Funding rate neutre → pas de signal."""
        strategy = FundingStrategy(_make_config())
        ctx = _make_context(funding_rate=0.01)
        assert strategy.evaluate(ctx) is None

    def test_no_funding_data(self):
        """Pas de funding rate dans extra_data → pas de signal."""
        strategy = FundingStrategy(_make_config())
        ctx = _make_context(funding_rate=None)
        assert strategy.evaluate(ctx) is None

    def test_entry_delay(self):
        """Signal confirmé seulement après le délai."""
        strategy = FundingStrategy(_make_config(entry_delay_minutes=5))
        ts0 = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

        # Première détection → enregistrement
        ctx1 = _make_context(funding_rate=-0.05, timestamp=ts0)
        assert strategy.evaluate(ctx1) is None

        # 3 min plus tard → trop tôt
        ctx2 = _make_context(funding_rate=-0.05, timestamp=ts0 + timedelta(minutes=3))
        assert strategy.evaluate(ctx2) is None

        # 6 min plus tard → OK
        ctx3 = _make_context(funding_rate=-0.05, timestamp=ts0 + timedelta(minutes=6))
        signal = strategy.evaluate(ctx3)
        assert signal is not None
        assert signal.direction == Direction.LONG


class TestCheckExit:
    def test_exit_neutral_funding(self):
        """Funding revient à neutre → signal_exit."""
        strategy = FundingStrategy(_make_config())
        position = OpenPosition(
            direction=Direction.LONG,
            entry_price=100_000.0,
            quantity=0.01,
            entry_time=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            tp_price=100_400.0,
            sl_price=99_800.0,
            entry_fee=0.5,
        )
        ctx = _make_context(funding_rate=0.005)  # |0.005| < 0.01 → neutre
        assert strategy.check_exit(ctx, position) == "signal_exit"

    def test_no_exit_still_extreme(self):
        """Funding encore extrême → pas de sortie."""
        strategy = FundingStrategy(_make_config())
        position = OpenPosition(
            direction=Direction.LONG,
            entry_price=100_000.0,
            quantity=0.01,
            entry_time=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            tp_price=100_400.0,
            sl_price=99_800.0,
            entry_fee=0.5,
        )
        ctx = _make_context(funding_rate=-0.04)  # Encore extrême
        assert strategy.check_exit(ctx, position) is None
