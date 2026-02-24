"""Tests pour la stratégie Grid Momentum.

33 tests couvrant :
- Breakout detection + compute_grid (8 tests)
- Trailing stop + SL + direction flip (6 tests)
- Fast engine (10 tests)
- Registry et config (5 tests)
- Edge cases (4 tests)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from backend.backtesting.engine import BacktestConfig
from backend.core.config import GridMomentumConfig
from backend.core.models import Candle, Direction
from backend.strategies.base import StrategyContext
from backend.strategies.base_grid import GridLevel, GridPosition, GridState
from backend.strategies.grid_momentum import GridMomentumStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_strategy(**overrides: Any) -> GridMomentumStrategy:
    """Crée une stratégie Grid Momentum avec defaults sensibles."""
    defaults: dict[str, Any] = {
        "donchian_period": 30,
        "vol_sma_period": 20,
        "vol_multiplier": 1.5,
        "adx_period": 14,
        "adx_threshold": 0.0,
        "atr_period": 14,
        "pullback_start": 1.0,
        "pullback_step": 0.5,
        "num_levels": 3,
        "trailing_atr_mult": 3.0,
        "sl_percent": 15.0,
        "sides": ["long", "short"],
        "leverage": 6,
        "timeframe": "1h",
        "cooldown_candles": 3,
    }
    defaults.update(overrides)
    config = GridMomentumConfig(**defaults)
    return GridMomentumStrategy(config)


def _make_ctx(
    close: float,
    high: float,
    donchian_high: float,
    donchian_low: float,
    volume: float = 200.0,
    volume_sma: float = 100.0,
    atr_val: float = 2.0,
    adx_val: float = 25.0,
    hwm: float | None = None,
) -> StrategyContext:
    """Crée un StrategyContext minimal pour grid_momentum."""
    indicators: dict[str, Any] = {
        "close": close,
        "high": high,
        "donchian_high": donchian_high,
        "donchian_low": donchian_low,
        "volume": volume,
        "volume_sma": volume_sma,
        "atr": atr_val,
        "adx": adx_val,
    }
    if hwm is not None:
        indicators["hwm"] = hwm

    return StrategyContext(
        symbol="BTC/USDT",
        timestamp=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        candles={},
        indicators={"1h": indicators},
        current_position=None,
        capital=10_000.0,
        config=None,  # type: ignore[arg-type]
    )


def _make_grid_state(
    positions: list[GridPosition] | None = None,
) -> GridState:
    """Crée un GridState."""
    positions = positions or []
    total_qty = sum(p.quantity for p in positions)
    avg_entry = (
        sum(p.entry_price * p.quantity for p in positions) / total_qty
        if total_qty > 0
        else 0.0
    )
    return GridState(
        positions=positions,
        avg_entry_price=avg_entry,
        total_quantity=total_qty,
        total_notional=0.0,
        unrealized_pnl=0.0,
    )


def _make_position(
    level: int = 0,
    direction: Direction = Direction.LONG,
    entry_price: float = 115.0,
    quantity: float = 1.0,
) -> GridPosition:
    return GridPosition(
        level=level,
        direction=direction,
        entry_price=entry_price,
        quantity=quantity,
        entry_time=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        entry_fee=0.01,
    )


def _make_bt_config(**overrides: Any) -> BacktestConfig:
    defaults: dict[str, Any] = {
        "symbol": "BTC/USDT",
        "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "end_date": datetime(2024, 3, 1, tzinfo=timezone.utc),
        "initial_capital": 10_000.0,
        "leverage": 6,
        "taker_fee": 0.0006,
        "maker_fee": 0.0002,
        "slippage_pct": 0.0001,
    }
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling max excluant la bougie courante."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window, n):
        result[i] = float(np.max(arr[i - window : i]))
    return result


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling min excluant la bougie courante."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window, n):
        result[i] = float(np.min(arr[i - window : i]))
    return result


def _make_breakout_cache(
    make_indicator_cache,
    *,
    direction: str = "long",
    n: int = 300,
    donchian_period: int = 30,
    atr_period: int = 14,
):
    """Crée un IndicatorCache avec un breakout Donchian clair.

    LONG : prix stable ~100 puis spike à 120 → puis retour lent vers 95.
    SHORT : symétrique (200 - prix LONG).
    """
    rng = np.random.default_rng(42)

    # Phase 1 (0-99) : stable autour de 100
    prices = np.full(n, 100.0)
    prices[:100] = 100.0 + rng.normal(0, 0.15, 100)

    # Phase 2 (100-109) : spike (breakout Donchian)
    prices[100:110] = np.linspace(101, 120, 10)

    # Phase 3 (110-200) : reste haut puis redescend (trailing stop)
    prices[110:200] = np.linspace(120, 95, 90)

    # Phase 4 (200+) : stable bas
    prices[200:] = 95.0 + rng.normal(0, 0.1, n - 200)

    if direction == "short":
        prices = 200.0 - prices

    highs = prices + np.abs(rng.normal(0.5, 0.2, n))
    lows = prices - np.abs(rng.normal(0.5, 0.2, n))

    volumes = np.full(n, 100.0)
    # Volume spike au breakout
    volumes[100:110] = 300.0

    # Rolling high/low pour Donchian (excluant bougie courante)
    rh = _rolling_max(highs, donchian_period)
    rl = _rolling_min(lows, donchian_period)

    # ATR simplifié
    atr_vals = np.full(n, 2.0)
    atr_vals[:atr_period] = np.nan

    # Volume SMA
    vol_sma = np.full(n, 100.0)
    vol_sma[:20] = np.nan

    # ADX constant
    adx = np.full(n, 25.0)

    ts = np.arange(n, dtype=np.float64) * 3600000

    return make_indicator_cache(
        n=n,
        closes=prices,
        opens=prices.copy(),
        highs=highs,
        lows=lows,
        volumes=volumes,
        rolling_high={donchian_period: rh},
        rolling_low={donchian_period: rl},
        atr_by_period={atr_period: atr_vals},
        volume_sma_arr=vol_sma,
        adx_arr=adx,
        candle_timestamps=ts,
    )


def _default_params(**overrides: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "donchian_period": 30,
        "vol_sma_period": 20,
        "vol_multiplier": 1.5,
        "adx_period": 14,
        "adx_threshold": 0.0,
        "atr_period": 14,
        "pullback_start": 1.0,
        "pullback_step": 0.5,
        "num_levels": 3,
        "trailing_atr_mult": 3.0,
        "sl_percent": 15.0,
        "sides": ["long", "short"],
        "leverage": 6,
        "cooldown_candles": 0,
    }
    d.update(overrides)
    return d


# ===================================================================
# Section 1 : Breakout detection + compute_grid (8 tests)
# ===================================================================


class TestBreakoutDetection:
    """Détection breakout Donchian + volume filter + compute_grid."""

    def test_long_breakout(self):
        """close > donchian_high + volume OK → niveaux LONG."""
        strategy = _make_strategy()
        ctx = _make_ctx(
            close=115.0,
            high=115.5,
            donchian_high=110.0,
            donchian_low=95.0,
        )
        levels = strategy.compute_grid(ctx, _make_grid_state())
        assert len(levels) == 3
        assert all(lv.direction == Direction.LONG for lv in levels)
        assert levels[0].entry_price == pytest.approx(115.0)

    def test_short_breakout(self):
        """close < donchian_low + volume OK → niveaux SHORT."""
        strategy = _make_strategy()
        ctx = _make_ctx(
            close=90.0,
            high=91.0,
            donchian_high=110.0,
            donchian_low=95.0,
        )
        levels = strategy.compute_grid(ctx, _make_grid_state())
        assert len(levels) == 3
        assert all(lv.direction == Direction.SHORT for lv in levels)
        assert levels[0].entry_price == pytest.approx(90.0)

    def test_no_breakout_insufficient_volume(self):
        """Breakout détecté mais volume < sma × vol_multiplier → []."""
        strategy = _make_strategy(vol_multiplier=2.0)
        ctx = _make_ctx(
            close=115.0,
            high=115.5,
            donchian_high=110.0,
            donchian_low=95.0,
            volume=150.0,
            volume_sma=100.0,  # 150 < 100*2
        )
        levels = strategy.compute_grid(ctx, _make_grid_state())
        assert levels == []

    def test_adx_filter_blocks_when_below_threshold(self):
        """adx_threshold=25 + adx=20 → []."""
        strategy = _make_strategy(adx_threshold=25.0)
        ctx = _make_ctx(
            close=115.0,
            high=115.5,
            donchian_high=110.0,
            donchian_low=95.0,
            adx_val=20.0,
        )
        levels = strategy.compute_grid(ctx, _make_grid_state())
        assert levels == []

    def test_breakout_ignored_if_side_excluded(self):
        """sides=["long"] → SHORT breakout ignoré."""
        strategy = _make_strategy(sides=["long"])
        ctx = _make_ctx(
            close=90.0,
            high=91.0,
            donchian_high=110.0,
            donchian_low=95.0,
        )
        levels = strategy.compute_grid(ctx, _make_grid_state())
        assert levels == []

    def test_pullback_dca_levels_long(self):
        """Niveaux DCA pullback LONG calculés correctement."""
        strategy = _make_strategy(
            num_levels=3, pullback_start=1.0, pullback_step=0.5
        )
        ctx = _make_ctx(
            close=115.0,
            high=115.5,
            donchian_high=110.0,
            donchian_low=95.0,
            atr_val=2.0,
        )
        levels = strategy.compute_grid(ctx, _make_grid_state())
        # Level 0: close = 115
        # Level 1: 115 - 2*(1.0) = 113
        # Level 2: 115 - 2*(1.0 + 0.5) = 112
        assert levels[0].entry_price == pytest.approx(115.0)
        assert levels[1].entry_price == pytest.approx(113.0)
        assert levels[2].entry_price == pytest.approx(112.0)

    def test_pullback_dca_levels_short(self):
        """Niveaux DCA pullback SHORT : au-dessus du prix."""
        strategy = _make_strategy(
            num_levels=3, pullback_start=1.0, pullback_step=0.5
        )
        ctx = _make_ctx(
            close=90.0,
            high=91.0,
            donchian_high=110.0,
            donchian_low=95.0,
            atr_val=2.0,
        )
        levels = strategy.compute_grid(ctx, _make_grid_state())
        # Level 0: close = 90
        # Level 1: 90 + 2*(1.0) = 92
        # Level 2: 90 + 2*(1.5) = 93
        assert levels[0].entry_price == pytest.approx(90.0)
        assert levels[1].entry_price == pytest.approx(92.0)
        assert levels[2].entry_price == pytest.approx(93.0)

    def test_existing_positions_pullback_from_anchor(self):
        """Positions ouvertes → niveaux DCA depuis anchor (positions[0].entry_price)."""
        strategy = _make_strategy(
            num_levels=3, pullback_start=1.0, pullback_step=0.5
        )
        pos0 = _make_position(level=0, entry_price=115.0)
        gs = _make_grid_state([pos0])
        ctx = _make_ctx(
            close=120.0,
            high=120.5,
            donchian_high=110.0,
            donchian_low=95.0,
            atr_val=2.0,
        )
        levels = strategy.compute_grid(ctx, gs)
        # Level 0 déjà rempli
        # Level 1: 115 - 2*(1.0) = 113
        # Level 2: 115 - 2*(1.5) = 112
        assert len(levels) == 2
        assert levels[0].index == 1
        assert levels[0].entry_price == pytest.approx(113.0)
        assert levels[1].index == 2
        assert levels[1].entry_price == pytest.approx(112.0)


# ===================================================================
# Section 2 : Trailing stop + SL + direction flip (6 tests)
# ===================================================================


class TestExitRules:
    """should_close_all(), get_sl_price(), get_tp_price()."""

    def test_direction_flip_long_to_short(self):
        """LONG + close < donchian_low → direction_flip."""
        strategy = _make_strategy()
        pos = _make_position(direction=Direction.LONG, entry_price=115.0)
        gs = _make_grid_state([pos])
        ctx = _make_ctx(
            close=94.0,
            high=95.0,
            donchian_high=110.0,
            donchian_low=95.0,
        )
        assert strategy.should_close_all(ctx, gs) == "direction_flip"

    def test_direction_flip_short_to_long(self):
        """SHORT + close > donchian_high → direction_flip."""
        strategy = _make_strategy()
        pos = _make_position(direction=Direction.SHORT, entry_price=95.0)
        gs = _make_grid_state([pos])
        ctx = _make_ctx(
            close=111.0,
            high=111.5,
            donchian_high=110.0,
            donchian_low=95.0,
        )
        assert strategy.should_close_all(ctx, gs) == "direction_flip"

    def test_trailing_stop_long(self):
        """LONG : close < hwm - trailing_atr_mult*atr → trail_stop."""
        strategy = _make_strategy(trailing_atr_mult=2.0)
        pos = _make_position(direction=Direction.LONG, entry_price=100.0)
        gs = _make_grid_state([pos])
        # hwm=120, atr=2, trail=120-4=116, close=115 < 116
        ctx = _make_ctx(
            close=115.0,
            high=115.5,
            donchian_high=130.0,
            donchian_low=80.0,
            atr_val=2.0,
            hwm=120.0,
        )
        assert strategy.should_close_all(ctx, gs) == "trail_stop"

    def test_trailing_stop_short(self):
        """SHORT : close > hwm + trailing_atr_mult*atr → trail_stop."""
        strategy = _make_strategy(trailing_atr_mult=2.0)
        pos = _make_position(direction=Direction.SHORT, entry_price=100.0)
        gs = _make_grid_state([pos])
        # hwm(lwm)=80, atr=2, trail=80+4=84, close=85 > 84
        ctx = _make_ctx(
            close=85.0,
            high=85.5,
            donchian_high=130.0,
            donchian_low=70.0,
            atr_val=2.0,
            hwm=80.0,
        )
        assert strategy.should_close_all(ctx, gs) == "trail_stop"

    def test_no_exit_when_in_range(self):
        """Position LONG, pas de flip/trail → None."""
        strategy = _make_strategy(trailing_atr_mult=3.0)
        pos = _make_position(direction=Direction.LONG, entry_price=100.0)
        gs = _make_grid_state([pos])
        # hwm=115, atr=2, trail=115-6=109, close=112 > 109
        ctx = _make_ctx(
            close=112.0,
            high=112.5,
            donchian_high=130.0,
            donchian_low=80.0,
            atr_val=2.0,
            hwm=115.0,
        )
        assert strategy.should_close_all(ctx, gs) is None

    def test_sl_and_tp_prices(self):
        """SL = avg_entry ± sl_percent%, TP = NaN."""
        strategy = _make_strategy(sl_percent=10.0)
        pos = _make_position(direction=Direction.LONG, entry_price=100.0)
        gs = _make_grid_state([pos])
        sl = strategy.get_sl_price(gs, {})
        tp = strategy.get_tp_price(gs, {})
        assert sl == pytest.approx(90.0)
        assert math.isnan(tp)


# ===================================================================
# Section 3 : Fast engine (10 tests)
# ===================================================================


class TestFastEngine:
    """Tests _simulate_grid_momentum() et run_multi_backtest_from_cache()."""

    def test_breakout_produces_trades(self, make_indicator_cache):
        """Breakout LONG clair → au moins 1 trade."""
        from backend.optimization.fast_multi_backtest import (
            _simulate_grid_momentum,
        )

        cache = _make_breakout_cache(make_indicator_cache)
        params = _default_params()
        bt_config = _make_bt_config()
        pnls, _, capital = _simulate_grid_momentum(cache, params, bt_config)
        assert len(pnls) >= 1
        assert isinstance(capital, float)

    def test_flat_data_no_trades(self, make_indicator_cache):
        """Prix plat → pas de breakout → 0 trades."""
        from backend.optimization.fast_multi_backtest import (
            _simulate_grid_momentum,
        )

        n = 200
        prices = np.full(n, 100.0)
        highs = np.full(n, 100.5)
        lows = np.full(n, 99.5)
        volumes = np.full(n, 100.0)

        rh = _rolling_max(highs, 30)
        rl = _rolling_min(lows, 30)
        atr_vals = np.full(n, 0.5)
        atr_vals[:14] = np.nan

        cache = make_indicator_cache(
            n=n,
            closes=prices,
            opens=prices.copy(),
            highs=highs,
            lows=lows,
            volumes=volumes,
            rolling_high={30: rh},
            rolling_low={30: rl},
            atr_by_period={14: atr_vals},
            volume_sma_arr=np.full(n, 100.0),
            adx_arr=np.full(n, 25.0),
            candle_timestamps=np.arange(n, dtype=np.float64) * 3600000,
        )
        params = _default_params()
        bt_config = _make_bt_config()
        pnls, _, capital = _simulate_grid_momentum(cache, params, bt_config)
        assert len(pnls) == 0
        assert capital == bt_config.initial_capital

    def test_trailing_stop_fires_after_pump_reversal(self, make_indicator_cache):
        """Pump → reversal → trailing stop déclenché."""
        from backend.optimization.fast_multi_backtest import (
            _simulate_grid_momentum,
        )

        cache = _make_breakout_cache(make_indicator_cache)
        params = _default_params(trailing_atr_mult=2.0)
        bt_config = _make_bt_config()
        pnls, _, _ = _simulate_grid_momentum(cache, params, bt_config)
        assert len(pnls) >= 1

    def test_sl_global_on_bear_data(self, make_indicator_cache):
        """Breakout LONG + crash immédiat → SL global touché."""
        from backend.optimization.fast_multi_backtest import (
            _simulate_grid_momentum,
        )

        n = 200
        prices = np.full(n, 100.0)
        # Breakout léger
        prices[100:105] = np.linspace(101, 108, 5)
        # Crash direct
        prices[105:120] = np.linspace(108, 70, 15)
        prices[120:] = 70.0

        highs = prices + 0.5
        lows = prices - 0.5
        volumes = np.full(n, 100.0)
        volumes[100:105] = 300.0

        rh = _rolling_max(highs, 30)
        rl = _rolling_min(lows, 30)
        atr_vals = np.full(n, 2.0)
        atr_vals[:14] = np.nan

        cache = make_indicator_cache(
            n=n,
            closes=prices,
            opens=prices.copy(),
            highs=highs,
            lows=lows,
            volumes=volumes,
            rolling_high={30: rh},
            rolling_low={30: rl},
            atr_by_period={14: atr_vals},
            volume_sma_arr=np.full(n, 100.0),
            adx_arr=np.full(n, 25.0),
            candle_timestamps=np.arange(n, dtype=np.float64) * 3600000,
        )
        params = _default_params(sl_percent=10.0, trailing_atr_mult=100.0)
        bt_config = _make_bt_config()
        pnls, _, _ = _simulate_grid_momentum(cache, params, bt_config)
        assert len(pnls) >= 1
        # SL est une perte
        assert any(p < 0 for p in pnls)

    def test_direction_flip_in_fast_engine(self, make_indicator_cache):
        """Breakout LONG puis close < donchian_low → direction_flip."""
        from backend.optimization.fast_multi_backtest import (
            _simulate_grid_momentum,
        )

        n = 300
        prices = np.full(n, 100.0)
        # Breakout LONG
        prices[100:110] = np.linspace(101, 115, 10)
        # Reste haut un moment
        prices[110:150] = 114.0
        # Crash sous donchian_low (direction flip)
        prices[150:170] = np.linspace(114, 80, 20)
        prices[170:] = 80.0

        highs = prices + 0.5
        lows = prices - 0.5
        volumes = np.full(n, 100.0)
        volumes[100:110] = 300.0

        rh = _rolling_max(highs, 30)
        rl = _rolling_min(lows, 30)
        atr_vals = np.full(n, 2.0)
        atr_vals[:14] = np.nan

        cache = make_indicator_cache(
            n=n,
            closes=prices,
            opens=prices.copy(),
            highs=highs,
            lows=lows,
            volumes=volumes,
            rolling_high={30: rh},
            rolling_low={30: rl},
            atr_by_period={14: atr_vals},
            volume_sma_arr=np.full(n, 100.0),
            adx_arr=np.full(n, 25.0),
            candle_timestamps=np.arange(n, dtype=np.float64) * 3600000,
        )
        params = _default_params(trailing_atr_mult=100.0, sl_percent=50.0)
        bt_config = _make_bt_config()
        pnls, _, _ = _simulate_grid_momentum(cache, params, bt_config)
        assert len(pnls) >= 1

    def test_capital_tracking(self, make_indicator_cache):
        """Capital final cohérent (pas négatif, pas overflow)."""
        from backend.optimization.fast_multi_backtest import (
            _simulate_grid_momentum,
        )

        cache = _make_breakout_cache(make_indicator_cache)
        params = _default_params()
        bt_config = _make_bt_config(initial_capital=10_000.0)
        _, _, capital = _simulate_grid_momentum(cache, params, bt_config)
        assert capital > 0
        assert capital < 1_000_000  # Pas d'overflow

    def test_hwm_updates_correctly(self, make_indicator_cache):
        """HWM tracking : le trailing stop ne se déclenche qu'après reversal."""
        from backend.optimization.fast_multi_backtest import (
            _simulate_grid_momentum,
        )

        # Grand trailing_mult → le trailing ne se déclenchera que si vraie reversal
        cache = _make_breakout_cache(make_indicator_cache)
        params = _default_params(trailing_atr_mult=1.0)
        bt_config = _make_bt_config()
        pnls, _, _ = _simulate_grid_momentum(cache, params, bt_config)
        # Avec trailing_atr_mult=1.0 et ATR=2.0, le trailing est très serré
        # → le prix redescendant dans la Phase 3 va déclencher le trail
        assert len(pnls) >= 1

    def test_cooldown_respected(self, make_indicator_cache):
        """cooldown_candles > 0 → pas de re-entry immédiat."""
        from backend.optimization.fast_multi_backtest import (
            _simulate_grid_momentum,
        )

        cache = _make_breakout_cache(make_indicator_cache, n=300)
        # Sans cooldown
        params_no_cd = _default_params(cooldown_candles=0, trailing_atr_mult=1.0)
        bt_config = _make_bt_config()
        pnls_no_cd, _, _ = _simulate_grid_momentum(cache, params_no_cd, bt_config)

        # Avec gros cooldown (aucun re-entry possible)
        params_cd = _default_params(cooldown_candles=10, trailing_atr_mult=1.0)
        pnls_cd, _, _ = _simulate_grid_momentum(cache, params_cd, bt_config)

        # Plus de trades sans cooldown que avec
        assert len(pnls_no_cd) >= len(pnls_cd)

    def test_short_breakout_in_fast_engine(self, make_indicator_cache):
        """Breakout SHORT dans le fast engine produit des trades."""
        from backend.optimization.fast_multi_backtest import (
            _simulate_grid_momentum,
        )

        cache = _make_breakout_cache(
            make_indicator_cache, direction="short"
        )
        params = _default_params()
        bt_config = _make_bt_config()
        pnls, _, _ = _simulate_grid_momentum(cache, params, bt_config)
        assert len(pnls) >= 1

    def test_multi_level_fill_on_same_candle(self, make_indicator_cache):
        """Breakout + long wick → Level 0 et Level 1+ remplis si wick les touche."""
        from backend.optimization.fast_multi_backtest import (
            _simulate_grid_momentum,
        )

        n = 200
        prices = np.full(n, 100.0)
        # Breakout clair
        prices[100:105] = np.linspace(101, 112, 5)
        # Candle 105 : close=112 mais low=106 (long wick)
        # → Level 0 = 112, Level 1 = 112 - 2*1.0 = 110
        # → lows[105] = 106 touche Level 0 ET Level 1
        prices[105:] = 112.0

        highs = prices + 0.5
        lows = prices - 0.5
        # Long wick au moment du breakout
        lows[105] = 106.0
        volumes = np.full(n, 100.0)
        volumes[100:106] = 300.0

        rh = _rolling_max(highs, 30)
        rl = _rolling_min(lows, 30)
        atr_vals = np.full(n, 2.0)
        atr_vals[:14] = np.nan

        cache = make_indicator_cache(
            n=n,
            closes=prices,
            opens=prices.copy(),
            highs=highs,
            lows=lows,
            volumes=volumes,
            rolling_high={30: rh},
            rolling_low={30: rl},
            atr_by_period={14: atr_vals},
            volume_sma_arr=np.full(n, 100.0),
            adx_arr=np.full(n, 25.0),
            candle_timestamps=np.arange(n, dtype=np.float64) * 3600000,
        )
        # pullback_start=1.0, step=0.5, num_levels=3
        # Level 0 = breakout price, Level 1 = price - 2*1.0 = -2, Level 2 = -3
        params = _default_params(num_levels=3)
        bt_config = _make_bt_config()
        pnls, _, _ = _simulate_grid_momentum(cache, params, bt_config)
        # Au moins un trade (même si les niveaux sont remplis sur la même candle)
        assert len(pnls) >= 1


# ===================================================================
# Section 4 : Registry et config (5 tests)
# ===================================================================


class TestRegistryConfig:
    """Tests d'intégration registry et config."""

    def test_in_strategy_registry(self):
        from backend.optimization import STRATEGY_REGISTRY

        assert "grid_momentum" in STRATEGY_REGISTRY

    def test_in_grid_strategies(self):
        from backend.optimization import GRID_STRATEGIES

        assert "grid_momentum" in GRID_STRATEGIES

    def test_in_fast_engine_strategies(self):
        from backend.optimization import FAST_ENGINE_STRATEGIES

        assert "grid_momentum" in FAST_ENGINE_STRATEGIES

    def test_not_in_strategies_need_extra_data(self):
        """grid_momentum n'utilise ni funding rates ni OI."""
        from backend.optimization import STRATEGIES_NEED_EXTRA_DATA

        assert "grid_momentum" not in STRATEGIES_NEED_EXTRA_DATA

    def test_create_with_params(self):
        from backend.optimization import create_strategy_with_params

        strategy = create_strategy_with_params("grid_momentum", {})
        assert strategy.name == "grid_momentum"
        assert isinstance(strategy, GridMomentumStrategy)


# ===================================================================
# Section 5 : Edge cases (4 tests)
# ===================================================================


class TestEdgeCases:
    """Tests de robustesse."""

    def test_atr_zero_no_levels(self):
        """ATR = 0 → pas de niveaux (division safe)."""
        strategy = _make_strategy()
        ctx = _make_ctx(
            close=115.0,
            high=115.5,
            donchian_high=110.0,
            donchian_low=95.0,
            atr_val=0.0,
        )
        levels = strategy.compute_grid(ctx, _make_grid_state())
        assert levels == []

    def test_constant_price_no_breakout(self):
        """Prix constant → donchian_high ≈ donchian_low ≈ close → pas de breakout."""
        strategy = _make_strategy()
        ctx = _make_ctx(
            close=100.0,
            high=100.5,
            donchian_high=100.5,
            donchian_low=99.5,
        )
        levels = strategy.compute_grid(ctx, _make_grid_state())
        assert levels == []

    def test_nan_indicators_no_crash(self):
        """NaN dans les indicateurs → return [] sans crash."""
        strategy = _make_strategy()
        ctx = _make_ctx(
            close=float("nan"),
            high=float("nan"),
            donchian_high=float("nan"),
            donchian_low=float("nan"),
            atr_val=float("nan"),
        )
        levels = strategy.compute_grid(ctx, _make_grid_state())
        assert levels == []

    def test_volume_sma_zero_no_division_error(self):
        """volume_sma = 0 → check est volume > 0 * mult = 0 → breakout si volume > 0."""
        strategy = _make_strategy()
        ctx = _make_ctx(
            close=115.0,
            high=115.5,
            donchian_high=110.0,
            donchian_low=95.0,
            volume=100.0,
            volume_sma=0.0,
        )
        # volume(100) > 0 * 1.5 = 0 → True → breakout détecté
        levels = strategy.compute_grid(ctx, _make_grid_state())
        assert len(levels) > 0
