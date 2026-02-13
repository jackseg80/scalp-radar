"""Tests pour les 3 nouvelles stratégies 1h : Bollinger MR, Donchian Breakout, SuperTrend.

Couvre : indicateurs, signaux, check_exit, fast engine, configs, registry.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from backend.core.config import (
    BollingerMRConfig,
    DonchianBreakoutConfig,
    SuperTrendConfig,
)
from backend.core.indicators import bollinger_bands, supertrend
from backend.core.models import Candle, Direction
from backend.strategies.base import OpenPosition, StrategyContext, StrategySignal
from backend.strategies.bollinger_mr import BollingerMRStrategy
from backend.strategies.donchian_breakout import DonchianBreakoutStrategy
from backend.strategies.supertrend import SuperTrendStrategy


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _make_candles(
    n: int,
    start_price: float = 100.0,
    step: float = 0.0,
    tf_minutes: int = 60,
    high_offset: float = 1.0,
    low_offset: float = 1.0,
    volume: float = 100.0,
) -> list[Candle]:
    """Génère N candles synthétiques."""
    candles = []
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        price = start_price + i * step
        candles.append(Candle(
            symbol="BTC/USDT",
            exchange="binance",
            timeframe="1h",
            timestamp=base + timedelta(minutes=tf_minutes * i),
            open=price,
            high=price + high_offset,
            low=price - low_offset,
            close=price,
            volume=volume,
        ))
    return candles


def _make_ctx(
    main_indicators: dict[str, float],
    tf: str = "1h",
) -> StrategyContext:
    """Crée un StrategyContext avec indicateurs."""
    return StrategyContext(
        symbol="BTC/USDT",
        timestamp=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        candles={},
        indicators={tf: main_indicators},
        current_position=None,
        capital=10_000.0,
        config=None,  # type: ignore[arg-type]
    )


def _make_position(
    direction: Direction,
    entry_price: float = 100.0,
) -> OpenPosition:
    return OpenPosition(
        direction=direction,
        entry_price=entry_price,
        quantity=0.01,
        entry_time=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
        tp_price=200.0 if direction == Direction.LONG else 50.0,
        sl_price=90.0 if direction == Direction.LONG else 110.0,
        entry_fee=0.5,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BOLLINGER MR
# ═══════════════════════════════════════════════════════════════════════════════


class TestBollingerBandsIndicator:
    def test_basic_computation(self):
        """bollinger_bands retourne 3 arrays (sma, upper, lower)."""
        closes = np.array([10, 11, 12, 11, 10, 11, 12, 13, 12, 11], dtype=float)
        sma_arr, upper, lower = bollinger_bands(closes, period=5, std_dev=2.0)
        assert len(sma_arr) == len(closes)
        assert len(upper) == len(closes)
        assert len(lower) == len(closes)

    def test_nan_before_period(self):
        """NaN avant que la fenêtre soit remplie."""
        closes = np.array([10, 11, 12, 13, 14, 15], dtype=float)
        sma_arr, upper, lower = bollinger_bands(closes, period=5)
        # Premiers 4 éléments = NaN (period=5 → premier valide à index 4)
        assert np.isnan(sma_arr[3])
        assert not np.isnan(sma_arr[4])

    def test_bands_symmetry(self):
        """upper - sma == sma - lower (symétrie par rapport à la SMA)."""
        closes = np.array([10, 12, 11, 13, 14, 12, 11, 10, 13, 15], dtype=float)
        sma_arr, upper, lower = bollinger_bands(closes, period=5, std_dev=2.0)
        for i in range(4, len(closes)):
            if not np.isnan(sma_arr[i]):
                assert abs((upper[i] - sma_arr[i]) - (sma_arr[i] - lower[i])) < 1e-10

    def test_wider_std_wider_bands(self):
        """Plus grand std_dev → bandes plus larges."""
        closes = np.array([10, 12, 11, 13, 14, 12, 11, 10, 13, 15], dtype=float)
        _, upper_2, lower_2 = bollinger_bands(closes, period=5, std_dev=2.0)
        _, upper_3, lower_3 = bollinger_bands(closes, period=5, std_dev=3.0)
        # À l'index 6 (après warmup)
        width_2 = upper_2[6] - lower_2[6]
        width_3 = upper_3[6] - lower_3[6]
        assert width_3 > width_2


class TestBollingerMRStrategy:
    def _make_strategy(self, **overrides) -> BollingerMRStrategy:
        defaults = {
            "enabled": True,
            "timeframe": "1h",
            "bb_period": 20,
            "bb_std": 2.0,
            "sl_percent": 5.0,
            "weight": 0.15,
        }
        defaults.update(overrides)
        return BollingerMRStrategy(BollingerMRConfig(**defaults))

    def test_long_below_lower_band(self):
        """Close < lower band → LONG."""
        strategy = self._make_strategy()
        ctx = _make_ctx({
            "close": 95.0,
            "bb_sma": 100.0,
            "bb_upper": 105.0,
            "bb_lower": 96.0,  # close < lower
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 20.0,
            "di_plus": 15.0,
            "di_minus": 15.0,
        })
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert signal.direction == Direction.LONG

    def test_short_above_upper_band(self):
        """Close > upper band → SHORT."""
        strategy = self._make_strategy()
        ctx = _make_ctx({
            "close": 107.0,
            "bb_sma": 100.0,
            "bb_upper": 105.0,
            "bb_lower": 95.0,
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 20.0,
            "di_plus": 15.0,
            "di_minus": 15.0,
        })
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert signal.direction == Direction.SHORT

    def test_no_signal_inside_bands(self):
        """Close entre les bandes → pas de signal."""
        strategy = self._make_strategy()
        ctx = _make_ctx({
            "close": 100.0,
            "bb_sma": 100.0,
            "bb_upper": 105.0,
            "bb_lower": 95.0,
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 20.0,
            "di_plus": 15.0,
            "di_minus": 15.0,
        })
        assert strategy.evaluate(ctx) is None

    def test_tp_very_far_long(self):
        """TP = entry × 2 pour LONG (jamais touché)."""
        strategy = self._make_strategy()
        ctx = _make_ctx({
            "close": 95.0,
            "bb_sma": 100.0,
            "bb_upper": 105.0,
            "bb_lower": 96.0,
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 20.0,
            "di_plus": 15.0,
            "di_minus": 15.0,
        })
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert signal.tp_price == pytest.approx(95.0 * 2.0)

    def test_tp_very_far_short(self):
        """TP = entry × 0.5 pour SHORT (jamais touché)."""
        strategy = self._make_strategy()
        ctx = _make_ctx({
            "close": 107.0,
            "bb_sma": 100.0,
            "bb_upper": 105.0,
            "bb_lower": 95.0,
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 20.0,
            "di_plus": 15.0,
            "di_minus": 15.0,
        })
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert signal.tp_price == pytest.approx(107.0 * 0.5)

    def test_check_exit_long_crosses_sma(self):
        """LONG : close >= SMA → signal_exit."""
        strategy = self._make_strategy()
        position = _make_position(Direction.LONG, entry_price=95.0)
        ctx = _make_ctx({
            "close": 100.5,
            "bb_sma": 100.0,
            "bb_upper": 105.0,
            "bb_lower": 95.0,
        })
        assert strategy.check_exit(ctx, position) == "signal_exit"

    def test_check_exit_long_below_sma(self):
        """LONG : close < SMA → pas de sortie."""
        strategy = self._make_strategy()
        position = _make_position(Direction.LONG, entry_price=95.0)
        ctx = _make_ctx({
            "close": 98.0,
            "bb_sma": 100.0,
            "bb_upper": 105.0,
            "bb_lower": 95.0,
        })
        assert strategy.check_exit(ctx, position) is None

    def test_check_exit_short_crosses_sma(self):
        """SHORT : close <= SMA → signal_exit."""
        strategy = self._make_strategy()
        position = _make_position(Direction.SHORT, entry_price=107.0)
        ctx = _make_ctx({
            "close": 99.5,
            "bb_sma": 100.0,
            "bb_upper": 105.0,
            "bb_lower": 95.0,
        })
        assert strategy.check_exit(ctx, position) == "signal_exit"

    def test_min_candles(self):
        strategy = self._make_strategy(bb_period=30)
        mc = strategy.min_candles
        assert "1h" in mc
        assert mc["1h"] >= 50  # max(30+20, 50)

    def test_get_params(self):
        strategy = self._make_strategy()
        params = strategy.get_params()
        assert params["bb_period"] == 20
        assert params["bb_std"] == 2.0
        assert params["sl_percent"] == 5.0

    def test_get_current_conditions(self):
        strategy = self._make_strategy()
        ctx = _make_ctx({
            "close": 94.0,
            "bb_upper": 105.0,
            "bb_lower": 95.0,
        })
        conditions = strategy.get_current_conditions(ctx)
        assert len(conditions) == 2
        assert conditions[0]["name"] == "bb_position"
        assert conditions[0]["met"] is True  # close < lower
        assert conditions[0]["value"] == "below"


# ═══════════════════════════════════════════════════════════════════════════════
# DONCHIAN BREAKOUT
# ═══════════════════════════════════════════════════════════════════════════════


class TestDonchianBreakoutStrategy:
    def _make_strategy(self, **overrides) -> DonchianBreakoutStrategy:
        defaults = {
            "enabled": True,
            "timeframe": "1h",
            "entry_lookback": 20,
            "atr_period": 14,
            "atr_tp_multiple": 3.0,
            "atr_sl_multiple": 1.5,
            "weight": 0.15,
        }
        defaults.update(overrides)
        return DonchianBreakoutStrategy(DonchianBreakoutConfig(**defaults))

    def test_long_breakout(self):
        """Close > rolling_high → LONG."""
        strategy = self._make_strategy()
        ctx = _make_ctx({
            "close": 105.0,
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 25.0,
            "di_plus": 20.0,
            "di_minus": 10.0,
            "rolling_high": 104.0,  # close > rolling_high
            "rolling_low": 96.0,
        })
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert signal.direction == Direction.LONG

    def test_short_breakout(self):
        """Close < rolling_low → SHORT."""
        strategy = self._make_strategy()
        ctx = _make_ctx({
            "close": 95.0,
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 25.0,
            "di_plus": 10.0,
            "di_minus": 20.0,
            "rolling_high": 104.0,
            "rolling_low": 96.0,
        })
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert signal.direction == Direction.SHORT

    def test_no_signal_inside_channel(self):
        """Close dans le canal → pas de signal."""
        strategy = self._make_strategy()
        ctx = _make_ctx({
            "close": 100.0,
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 25.0,
            "di_plus": 15.0,
            "di_minus": 15.0,
            "rolling_high": 104.0,
            "rolling_low": 96.0,
        })
        assert strategy.evaluate(ctx) is None

    def test_tp_sl_atr_multiples(self):
        """TP et SL basés sur ATR × multiples."""
        strategy = self._make_strategy(atr_tp_multiple=3.0, atr_sl_multiple=1.5)
        ctx = _make_ctx({
            "close": 105.0,
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 25.0,
            "di_plus": 20.0,
            "di_minus": 10.0,
            "rolling_high": 104.0,
            "rolling_low": 96.0,
        })
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert signal.tp_price == pytest.approx(105.0 + 2.0 * 3.0)
        assert signal.sl_price == pytest.approx(105.0 - 2.0 * 1.5)

    def test_check_exit_returns_none(self):
        """Pas de sortie anticipée pour Donchian."""
        strategy = self._make_strategy()
        position = _make_position(Direction.LONG)
        ctx = _make_ctx({"close": 100.0})
        assert strategy.check_exit(ctx, position) is None

    def test_min_candles(self):
        strategy = self._make_strategy(entry_lookback=40)
        mc = strategy.min_candles
        assert "1h" in mc
        assert mc["1h"] >= 60  # max(40+20, 50)

    def test_get_params(self):
        strategy = self._make_strategy()
        params = strategy.get_params()
        assert params["entry_lookback"] == 20
        assert params["atr_period"] == 14
        assert params["atr_tp_multiple"] == 3.0

    def test_get_current_conditions(self):
        strategy = self._make_strategy()
        ctx = _make_ctx({
            "close": 105.0,
            "rolling_high": 104.0,
            "rolling_low": 96.0,
            "atr": 2.0,
        })
        conditions = strategy.get_current_conditions(ctx)
        assert len(conditions) == 2
        assert conditions[0]["name"] == "channel_position"
        assert conditions[0]["met"] is True
        assert conditions[0]["value"] == "above"

    def test_compute_indicators_structure(self):
        """compute_indicators retourne la structure attendue."""
        strategy = self._make_strategy(entry_lookback=5, atr_period=5)
        candles = _make_candles(60, start_price=100.0, step=0.5)
        result = strategy.compute_indicators({"1h": candles})
        assert "1h" in result
        first_valid_ts = list(result["1h"].keys())[-1]
        ind = result["1h"][first_valid_ts]
        assert "close" in ind
        assert "atr" in ind
        assert "rolling_high" in ind
        assert "rolling_low" in ind


# ═══════════════════════════════════════════════════════════════════════════════
# SUPERTREND
# ═══════════════════════════════════════════════════════════════════════════════


class TestSuperTrendIndicator:
    def test_basic_computation(self):
        """supertrend retourne 2 arrays (st_values, direction)."""
        from backend.core.indicators import atr as atr_fn

        n = 50
        highs = np.random.default_rng(42).uniform(100, 110, n)
        lows = highs - np.random.default_rng(43).uniform(1, 5, n)
        closes = (highs + lows) / 2
        atr_arr = atr_fn(highs, lows, closes, period=10)
        st_values, st_direction = supertrend(highs, lows, closes, atr_arr, 3.0)
        assert len(st_values) == n
        assert len(st_direction) == n

    def test_direction_values(self):
        """Direction est 1.0 ou -1.0 (après warmup)."""
        from backend.core.indicators import atr as atr_fn

        n = 50
        rng = np.random.default_rng(42)
        highs = rng.uniform(100, 110, n)
        lows = highs - rng.uniform(1, 5, n)
        closes = (highs + lows) / 2
        atr_arr = atr_fn(highs, lows, closes, period=10)
        _, st_direction = supertrend(highs, lows, closes, atr_arr, 3.0)
        valid = ~np.isnan(st_direction)
        unique_dirs = set(st_direction[valid])
        assert unique_dirs.issubset({1.0, -1.0})

    def test_nan_before_atr_ready(self):
        """NaN dans direction quand ATR n'est pas encore calculé."""
        from backend.core.indicators import atr as atr_fn

        n = 20
        highs = np.full(n, 105.0)
        lows = np.full(n, 95.0)
        closes = np.full(n, 100.0)
        atr_arr = atr_fn(highs, lows, closes, period=14)
        _, st_direction = supertrend(highs, lows, closes, atr_arr, 3.0)
        # Les premiers éléments doivent être NaN (ATR pas prêt)
        assert np.isnan(st_direction[0])


class TestSuperTrendStrategy:
    def _make_strategy(self, **overrides) -> SuperTrendStrategy:
        defaults = {
            "enabled": True,
            "timeframe": "1h",
            "atr_period": 10,
            "atr_multiplier": 3.0,
            "tp_percent": 4.0,
            "sl_percent": 2.0,
            "weight": 0.15,
        }
        defaults.update(overrides)
        return SuperTrendStrategy(SuperTrendConfig(**defaults))

    def test_long_flip_down_to_up(self):
        """Flip direction -1 → +1 → LONG."""
        strategy = self._make_strategy()
        ctx = _make_ctx({
            "close": 100.0,
            "st_value": 98.0,
            "st_direction": 1.0,
            "st_prev_direction": -1.0,  # Flip DOWN→UP
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 25.0,
            "di_plus": 20.0,
            "di_minus": 10.0,
        })
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert signal.direction == Direction.LONG

    def test_short_flip_up_to_down(self):
        """Flip direction +1 → -1 → SHORT."""
        strategy = self._make_strategy()
        ctx = _make_ctx({
            "close": 100.0,
            "st_value": 102.0,
            "st_direction": -1.0,
            "st_prev_direction": 1.0,  # Flip UP→DOWN
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 25.0,
            "di_plus": 10.0,
            "di_minus": 20.0,
        })
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert signal.direction == Direction.SHORT

    def test_no_signal_without_flip(self):
        """Même direction → pas de signal."""
        strategy = self._make_strategy()
        ctx = _make_ctx({
            "close": 100.0,
            "st_value": 98.0,
            "st_direction": 1.0,
            "st_prev_direction": 1.0,  # Pas de flip
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 25.0,
            "di_plus": 20.0,
            "di_minus": 10.0,
        })
        assert strategy.evaluate(ctx) is None

    def test_tp_sl_percent_long(self):
        """TP et SL % fixe pour LONG."""
        strategy = self._make_strategy(tp_percent=4.0, sl_percent=2.0)
        ctx = _make_ctx({
            "close": 100.0,
            "st_value": 98.0,
            "st_direction": 1.0,
            "st_prev_direction": -1.0,
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 25.0,
            "di_plus": 20.0,
            "di_minus": 10.0,
        })
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert signal.tp_price == pytest.approx(100.0 * 1.04)
        assert signal.sl_price == pytest.approx(100.0 * 0.98)

    def test_tp_sl_percent_short(self):
        """TP et SL % fixe pour SHORT."""
        strategy = self._make_strategy(tp_percent=4.0, sl_percent=2.0)
        ctx = _make_ctx({
            "close": 100.0,
            "st_value": 102.0,
            "st_direction": -1.0,
            "st_prev_direction": 1.0,
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 25.0,
            "di_plus": 10.0,
            "di_minus": 20.0,
        })
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert signal.tp_price == pytest.approx(100.0 * 0.96)
        assert signal.sl_price == pytest.approx(100.0 * 1.02)

    def test_check_exit_returns_none(self):
        """Pas de sortie anticipée pour SuperTrend."""
        strategy = self._make_strategy()
        position = _make_position(Direction.LONG)
        ctx = _make_ctx({"close": 100.0})
        assert strategy.check_exit(ctx, position) is None

    def test_min_candles(self):
        strategy = self._make_strategy(atr_period=20)
        mc = strategy.min_candles
        assert "1h" in mc
        assert mc["1h"] >= 50  # max(20+20, 50)

    def test_get_params(self):
        strategy = self._make_strategy()
        params = strategy.get_params()
        assert params["atr_period"] == 10
        assert params["atr_multiplier"] == 3.0
        assert params["tp_percent"] == 4.0

    def test_get_current_conditions(self):
        strategy = self._make_strategy()
        ctx = _make_ctx({
            "close": 100.0,
            "st_value": 98.0,
            "st_direction": 1.0,
            "st_prev_direction": -1.0,
        })
        conditions = strategy.get_current_conditions(ctx)
        assert len(conditions) == 2
        assert conditions[0]["name"] == "st_direction"
        assert conditions[0]["met"] is True  # flip detected

    def test_compute_indicators_structure(self):
        """compute_indicators retourne la structure attendue."""
        strategy = self._make_strategy(atr_period=5)
        candles = _make_candles(60, start_price=100.0, step=0.5)
        result = strategy.compute_indicators({"1h": candles})
        assert "1h" in result
        last_ts = list(result["1h"].keys())[-1]
        ind = result["1h"][last_ts]
        assert "st_direction" in ind
        assert "st_prev_direction" in ind
        assert "st_value" in ind


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY / CONFIG / FACTORY
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistryAndConfig:
    def test_strategies_in_registry(self):
        """Les 3 stratégies sont dans le STRATEGY_REGISTRY."""
        from backend.optimization import STRATEGY_REGISTRY
        assert "bollinger_mr" in STRATEGY_REGISTRY
        assert "donchian_breakout" in STRATEGY_REGISTRY
        assert "supertrend" in STRATEGY_REGISTRY

    def test_create_strategy_with_params(self):
        """create_strategy_with_params fonctionne pour les 3."""
        from backend.optimization import create_strategy_with_params
        for name in ("bollinger_mr", "donchian_breakout", "supertrend"):
            strategy = create_strategy_with_params(name, {})
            assert strategy.name == name

    def test_bollinger_config_validation(self):
        """BollingerMRConfig valide les bornes."""
        with pytest.raises(Exception):
            BollingerMRConfig(bb_period=1)  # ge=2

    def test_donchian_config_validation(self):
        """DonchianBreakoutConfig valide les bornes."""
        with pytest.raises(Exception):
            DonchianBreakoutConfig(atr_sl_multiple=-1)  # gt=0

    def test_supertrend_config_validation(self):
        """SuperTrendConfig valide les bornes."""
        with pytest.raises(Exception):
            SuperTrendConfig(atr_period=0)  # ge=2

    def test_per_asset_override(self):
        """get_params_for_symbol applique les overrides."""
        cfg = BollingerMRConfig(
            per_asset={"SOL/USDT": {"sl_percent": 8.0}},
        )
        params = cfg.get_params_for_symbol("SOL/USDT")
        assert params["sl_percent"] == 8.0
        params_btc = cfg.get_params_for_symbol("BTC/USDT")
        assert params_btc["sl_percent"] == 5.0  # default


# ═══════════════════════════════════════════════════════════════════════════════
# FAST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class TestFastEngine:
    """Tests du fast backtest engine pour les 3 nouvelles stratégies."""

    def _make_cache_candles(
        self, n: int = 200, trend: str = "up",
    ) -> dict[str, list[Candle]]:
        """Génère des candles 1h avec tendance pour le cache."""
        candles: list[Candle] = []
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        price = 100.0
        rng = np.random.default_rng(42)

        for i in range(n):
            if trend == "up":
                change = rng.uniform(-0.5, 1.0)
            elif trend == "down":
                change = rng.uniform(-1.0, 0.5)
            else:
                change = rng.uniform(-1.0, 1.0)

            price += change
            high = price + rng.uniform(0.5, 2.0)
            low = price - rng.uniform(0.5, 2.0)
            volume = rng.uniform(50, 200)

            candles.append(Candle(
                symbol="BTC/USDT",
                exchange="binance",
                timeframe="1h",
                timestamp=base + timedelta(hours=i),
                open=price - change * 0.5,
                high=high,
                low=low,
                close=price,
                volume=volume,
            ))

        return {"1h": candles}

    def test_bollinger_mr_fast_runs(self):
        """Fast engine bollinger_mr s'exécute sans erreur."""
        from backend.backtesting.engine import BacktestConfig
        from backend.optimization.fast_backtest import run_backtest_from_cache
        from backend.optimization.indicator_cache import build_cache

        candles_by_tf = self._make_cache_candles(200)
        params = {"bb_period": 20, "bb_std": 2.0, "sl_percent": 5.0}
        param_grid_values = {k: [v] for k, v in params.items()}

        cache = build_cache(candles_by_tf, param_grid_values, "bollinger_mr", "1h")
        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=candles_by_tf["1h"][0].timestamp,
            end_date=candles_by_tf["1h"][-1].timestamp,
        )

        result = run_backtest_from_cache("bollinger_mr", params, cache, bt_config)
        assert len(result) == 5  # (params, sharpe, return, pf, n_trades)
        assert result[4] >= 0  # n_trades >= 0

    def test_donchian_fast_runs(self):
        """Fast engine donchian_breakout s'exécute sans erreur."""
        from backend.backtesting.engine import BacktestConfig
        from backend.optimization.fast_backtest import run_backtest_from_cache
        from backend.optimization.indicator_cache import build_cache

        candles_by_tf = self._make_cache_candles(200, trend="up")
        params = {
            "entry_lookback": 20,
            "atr_period": 14,
            "atr_tp_multiple": 3.0,
            "atr_sl_multiple": 1.5,
        }
        param_grid_values = {k: [v] for k, v in params.items()}

        cache = build_cache(candles_by_tf, param_grid_values, "donchian_breakout", "1h")
        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=candles_by_tf["1h"][0].timestamp,
            end_date=candles_by_tf["1h"][-1].timestamp,
        )

        result = run_backtest_from_cache("donchian_breakout", params, cache, bt_config)
        assert len(result) == 5
        assert result[4] >= 0

    def test_supertrend_fast_runs(self):
        """Fast engine supertrend s'exécute sans erreur."""
        from backend.backtesting.engine import BacktestConfig
        from backend.optimization.fast_backtest import run_backtest_from_cache
        from backend.optimization.indicator_cache import build_cache

        candles_by_tf = self._make_cache_candles(200)
        params = {
            "atr_period": 10,
            "atr_multiplier": 3.0,
            "tp_percent": 4.0,
            "sl_percent": 2.0,
        }
        param_grid_values = {k: [v] for k, v in params.items()}

        cache = build_cache(candles_by_tf, param_grid_values, "supertrend", "1h")
        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=candles_by_tf["1h"][0].timestamp,
            end_date=candles_by_tf["1h"][-1].timestamp,
        )

        result = run_backtest_from_cache("supertrend", params, cache, bt_config)
        assert len(result) == 5
        assert result[4] >= 0

    def test_bollinger_mr_check_exit_in_fast(self):
        """Le fast engine ferme les positions Bollinger MR sur SMA crossing."""
        from backend.optimization.fast_backtest import _check_exit
        from backend.optimization.indicator_cache import IndicatorCache

        n = 10
        closes = np.array([95, 96, 97, 98, 99, 100, 101, 102, 103, 104], dtype=float)
        bb_sma = np.full(n, 100.0)

        cache = IndicatorCache(
            n_candles=n,
            opens=closes,
            highs=closes + 1,
            lows=closes - 1,
            closes=closes,
            volumes=np.full(n, 100.0),
            total_days=10.0,
            rsi={},
            vwap=np.full(n, np.nan),
            vwap_distance_pct=np.full(n, np.nan),
            adx_arr=np.full(n, np.nan),
            di_plus=np.full(n, np.nan),
            di_minus=np.full(n, np.nan),
            atr_arr=np.full(n, np.nan),
            atr_sma=np.full(n, np.nan),
            volume_sma_arr=np.full(n, np.nan),
            regime=np.zeros(n, dtype=np.int8),
            rolling_high={},
            rolling_low={},
            filter_adx=np.full(n, np.nan),
            filter_di_plus=np.full(n, np.nan),
            filter_di_minus=np.full(n, np.nan),
            bb_sma={20: bb_sma},
            bb_upper={},
            bb_lower={},
            supertrend_direction={},
            atr_by_period={},
        )

        params = {"bb_period": 20}

        # Index 4 (close=99) : LONG, below SMA → pas de sortie
        assert _check_exit("bollinger_mr", cache, 4, 1, 95.0, params) is False
        # Index 5 (close=100) : LONG, close >= SMA → sortie
        assert _check_exit("bollinger_mr", cache, 5, 1, 95.0, params) is True
        # Index 7 (close=102) : SHORT, above SMA → pas de sortie
        assert _check_exit("bollinger_mr", cache, 7, -1, 105.0, params) is False
        # Index 3 (close=98) : SHORT, close <= SMA → sortie
        assert _check_exit("bollinger_mr", cache, 3, -1, 105.0, params) is True


class TestParamGridsYaml:
    def test_grids_load(self):
        """Les grilles param_grids.yaml se chargent correctement."""
        import yaml
        with open("config/param_grids.yaml", encoding="utf-8") as f:
            grids = yaml.safe_load(f)

        assert "bollinger_mr" in grids
        assert "donchian_breakout" in grids
        assert "supertrend" in grids

        # Vérifier structure WFO
        assert grids["bollinger_mr"]["wfo"]["is_days"] == 180
        assert grids["donchian_breakout"]["wfo"]["oos_days"] == 60
        assert grids["supertrend"]["wfo"]["step_days"] == 60

        # Vérifier grids default
        assert len(grids["bollinger_mr"]["default"]["bb_period"]) >= 3
        assert len(grids["donchian_breakout"]["default"]["entry_lookback"]) >= 3
        assert len(grids["supertrend"]["default"]["atr_period"]) >= 3
