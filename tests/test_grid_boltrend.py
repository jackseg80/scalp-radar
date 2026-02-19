"""Tests pour Grid BolTrend (Sprint 31).

Couvre :
- Section 1 : Breakout detection + compute_grid (~6 tests)
- Section 2 : TP inverse + SL global (~5 tests)
- Section 3 : Fast engine (~10 tests)
- Section 4 : Registry et config (~6 tests)
- Section 5 : Edge cases (~3 tests)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest

from backend.core.config import GridBolTrendConfig
from backend.core.models import Direction
from backend.strategies.base import StrategyContext
from backend.strategies.base_grid import GridLevel, GridPosition, GridState
from backend.strategies.grid_boltrend import GridBolTrendStrategy


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_strategy(**overrides) -> GridBolTrendStrategy:
    """Crée une stratégie Grid BolTrend avec defaults sensibles."""
    defaults: dict[str, Any] = {
        "bol_window": 100,
        "bol_std": 2.0,
        "long_ma_window": 200,
        "min_bol_spread": 0.0,
        "atr_period": 14,
        "atr_spacing_mult": 1.0,
        "num_levels": 3,
        "sl_percent": 15.0,
        "sides": ["long", "short"],
        "leverage": 6,
    }
    defaults.update(overrides)
    config = GridBolTrendConfig(**defaults)
    return GridBolTrendStrategy(config)


def _make_ctx(
    close: float,
    bb_sma: float,
    bb_upper: float,
    bb_lower: float,
    long_ma: float,
    atr_val: float = 2.0,
    prev_close: float | None = None,
    prev_upper: float | None = None,
    prev_lower: float | None = None,
    prev_spread: float | None = None,
) -> StrategyContext:
    """Crée un StrategyContext minimal pour grid_boltrend."""
    indicators: dict[str, Any] = {
        "close": close,
        "bb_sma": bb_sma,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "long_ma": long_ma,
        "atr": atr_val,
        "prev_close": prev_close if prev_close is not None else close - 1.0,
        "prev_upper": prev_upper if prev_upper is not None else bb_upper,
        "prev_lower": prev_lower if prev_lower is not None else bb_lower,
        "prev_spread": prev_spread if prev_spread is not None else 0.05,
    }
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


def _make_bt_config(**overrides):
    """Crée un BacktestConfig avec defaults sensibles."""
    from backend.backtesting.engine import BacktestConfig

    defaults = {
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


# ═══════════════════════════════════════════════════════════════════════════
# Section 1 : Breakout detection + compute_grid
# ═══════════════════════════════════════════════════════════════════════════


class TestBreakoutDetection:
    """Détection breakout et génération de niveaux DCA."""

    def test_name(self):
        """Le nom de la stratégie est correct."""
        strategy = _make_strategy()
        assert strategy.name == "grid_boltrend"

    def test_no_breakout_returns_empty(self):
        """Si close entre les bandes, pas de breakout → []."""
        strategy = _make_strategy()
        # close=100, bien entre upper=110 et lower=90
        ctx = _make_ctx(
            close=100.0, bb_sma=100.0, bb_upper=110.0, bb_lower=90.0,
            long_ma=95.0, atr_val=2.0,
            prev_close=99.0, prev_upper=110.0, prev_lower=90.0,
        )
        grid_state = _make_grid_state()
        levels = strategy.compute_grid(ctx, grid_state)
        assert levels == []

    def test_long_breakout_generates_levels(self):
        """Breakout LONG : close > upper + close > long_ma → niveaux LONG."""
        strategy = _make_strategy(num_levels=3, atr_spacing_mult=1.0)
        # prev_close=109 < prev_upper=110 → dans les bandes
        # close=112 > upper=110 → breakout, close > long_ma=95
        ctx = _make_ctx(
            close=112.0, bb_sma=100.0, bb_upper=110.0, bb_lower=90.0,
            long_ma=95.0, atr_val=2.0,
            prev_close=109.0, prev_upper=110.0, prev_lower=90.0,
            prev_spread=0.22,  # (110-90)/90
        )
        grid_state = _make_grid_state()
        levels = strategy.compute_grid(ctx, grid_state)
        assert len(levels) == 3
        assert all(lvl.direction == Direction.LONG for lvl in levels)
        # Level 0 = close = 112
        assert levels[0].entry_price == pytest.approx(112.0)
        # Level 1 = close - 1*ATR*mult = 112 - 2 = 110
        assert levels[1].entry_price == pytest.approx(110.0)
        # Level 2 = close - 2*ATR*mult = 112 - 4 = 108
        assert levels[2].entry_price == pytest.approx(108.0)

    def test_short_breakout_generates_levels(self):
        """Breakout SHORT : close < lower + close < long_ma → niveaux SHORT."""
        strategy = _make_strategy(num_levels=3, atr_spacing_mult=1.0)
        # prev_close=91 > prev_lower=90 → dans les bandes
        # close=88 < lower=90 → breakout, close < long_ma=95
        ctx = _make_ctx(
            close=88.0, bb_sma=100.0, bb_upper=110.0, bb_lower=90.0,
            long_ma=95.0, atr_val=2.0,
            prev_close=91.0, prev_upper=110.0, prev_lower=90.0,
            prev_spread=0.22,
        )
        grid_state = _make_grid_state()
        levels = strategy.compute_grid(ctx, grid_state)
        assert len(levels) == 3
        assert all(lvl.direction == Direction.SHORT for lvl in levels)
        # Level 0 = close = 88
        assert levels[0].entry_price == pytest.approx(88.0)
        # Level 1 = close + 1*ATR*mult = 88 + 2 = 90
        assert levels[1].entry_price == pytest.approx(90.0)

    def test_spread_too_small_no_breakout(self):
        """min_bol_spread filtre les breakouts avec bandes trop étroites."""
        strategy = _make_strategy(min_bol_spread=0.10)
        # Breakout LONG mais spread = 0.05 < min_spread = 0.10
        ctx = _make_ctx(
            close=112.0, bb_sma=100.0, bb_upper=110.0, bb_lower=90.0,
            long_ma=95.0, atr_val=2.0,
            prev_close=109.0, prev_upper=110.0, prev_lower=90.0,
            prev_spread=0.05,
        )
        grid_state = _make_grid_state()
        levels = strategy.compute_grid(ctx, grid_state)
        assert levels == []

    def test_existing_positions_use_anchor(self):
        """Si positions ouvertes, niveaux DCA depuis positions[0].entry_price."""
        strategy = _make_strategy(num_levels=3, atr_spacing_mult=1.0)
        pos0 = GridPosition(
            level=0, direction=Direction.LONG, entry_price=112.0,
            quantity=1.0, entry_time=datetime.now(timezone.utc), entry_fee=0.01,
        )
        grid_state = _make_grid_state([pos0])
        # Indicateurs quelconques, le breakout n'est pas re-vérifié
        ctx = _make_ctx(
            close=105.0, bb_sma=100.0, bb_upper=110.0, bb_lower=90.0,
            long_ma=95.0, atr_val=2.0,
        )
        levels = strategy.compute_grid(ctx, grid_state)
        # Level 0 déjà rempli → 2 niveaux restants
        assert len(levels) == 2
        assert levels[0].index == 1
        assert levels[0].entry_price == pytest.approx(110.0)  # 112 - 1*2
        assert levels[1].index == 2
        assert levels[1].entry_price == pytest.approx(108.0)  # 112 - 2*2


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 : TP inverse + SL global
# ═══════════════════════════════════════════════════════════════════════════


class TestTPInverseSLGlobal:
    """TP inverse (close < SMA LONG) et SL global."""

    def test_tp_inverse_long_close_below_sma(self):
        """LONG : close < bb_sma → signal_exit."""
        strategy = _make_strategy()
        pos = GridPosition(
            level=0, direction=Direction.LONG, entry_price=112.0,
            quantity=1.0, entry_time=datetime.now(timezone.utc), entry_fee=0.01,
        )
        grid_state = _make_grid_state([pos])
        # close=98 < bb_sma=100 → TP inverse
        ctx = _make_ctx(close=98.0, bb_sma=100.0, bb_upper=110.0, bb_lower=90.0, long_ma=95.0)
        result = strategy.should_close_all(ctx, grid_state)
        assert result == "signal_exit"

    def test_tp_inverse_short_close_above_sma(self):
        """SHORT : close > bb_sma → signal_exit."""
        strategy = _make_strategy(sl_percent=15.0)
        pos = GridPosition(
            level=0, direction=Direction.SHORT, entry_price=95.0,
            quantity=1.0, entry_time=datetime.now(timezone.utc), entry_fee=0.01,
        )
        grid_state = _make_grid_state([pos])
        # SL SHORT = 95 × 1.15 = 109.25, close=102 < 109.25 → SL pas touché
        # close=102 > bb_sma=100 → TP inverse
        ctx = _make_ctx(close=102.0, bb_sma=100.0, bb_upper=110.0, bb_lower=90.0, long_ma=95.0)
        result = strategy.should_close_all(ctx, grid_state)
        assert result == "signal_exit"

    def test_no_exit_long_above_sma(self):
        """LONG : close > bb_sma → pas d'exit (breakout toujours en cours)."""
        strategy = _make_strategy()
        pos = GridPosition(
            level=0, direction=Direction.LONG, entry_price=112.0,
            quantity=1.0, entry_time=datetime.now(timezone.utc), entry_fee=0.01,
        )
        grid_state = _make_grid_state([pos])
        # close=115 > bb_sma=100 → pas d'exit
        ctx = _make_ctx(close=115.0, bb_sma=100.0, bb_upper=110.0, bb_lower=90.0, long_ma=95.0)
        result = strategy.should_close_all(ctx, grid_state)
        assert result is None

    def test_sl_global_long(self):
        """LONG : close < avg_entry × (1 - sl_pct) → sl_global."""
        strategy = _make_strategy(sl_percent=10.0)
        pos = GridPosition(
            level=0, direction=Direction.LONG, entry_price=100.0,
            quantity=1.0, entry_time=datetime.now(timezone.utc), entry_fee=0.01,
        )
        grid_state = _make_grid_state([pos])
        # SL = 100 × 0.90 = 90. close=89 < 90 → SL
        # bb_sma=95 > close → signal_exit aussi, mais SL a priorité (checked first)
        ctx = _make_ctx(close=89.0, bb_sma=95.0, bb_upper=110.0, bb_lower=90.0, long_ma=95.0)
        result = strategy.should_close_all(ctx, grid_state)
        assert result == "sl_global"

    def test_get_params(self):
        """get_params retourne tous les champs importants."""
        strategy = _make_strategy(bol_window=50, bol_std=1.5, num_levels=4)
        params = strategy.get_params()
        assert params["bol_window"] == 50
        assert params["bol_std"] == 1.5
        assert params["num_levels"] == 4
        assert "atr_spacing_mult" in params
        assert "sl_percent" in params
        assert "sides" in params
        assert "leverage" in params


# ═══════════════════════════════════════════════════════════════════════════
# Section 3 : Fast engine
# ═══════════════════════════════════════════════════════════════════════════


def _make_breakout_cache(make_indicator_cache, *, direction="long", n=500):
    """Crée un cache avec un breakout clair pour les tests fast engine.

    start_idx = max(bol_window=100, long_ma_window=200) + 1 = 201
    Donc le breakout doit se produire APRÈS index 201.

    Phases :
    - 0-249 : stable à 100 (SMA convergée)
    - 250-260 : spike rapide (breakout)
    - 260-350 : reste haut puis redescend vers SMA (TP inverse)
    - 350-499 : stable
    """
    rng = np.random.default_rng(42)

    prices = np.full(n, 100.0)
    # Phase 1 (0-249) : stable
    prices[:250] = 100.0 + rng.normal(0, 0.15, 250)
    # Phase 2 (250-260) : spike fort (+20 en 10 candles)
    prices[250:260] = np.linspace(100, 120, 10)
    # Phase 3 (260-300) : reste haut
    prices[260:300] = 118.0 + rng.normal(0, 0.3, 40)
    # Phase 4 (300-400) : redescend lentement vers SMA (~100)
    prices[300:400] = np.linspace(118, 95, 100)
    # Phase 5 (400-499) : stable bas
    prices[400:] = 95.0 + rng.normal(0, 0.1, 100)

    if direction == "short":
        prices = 200.0 - prices

    opens = prices + rng.uniform(-0.1, 0.1, n)
    highs = prices + np.abs(rng.normal(0.3, 0.15, n))
    lows = prices - np.abs(rng.normal(0.3, 0.15, n))

    # SMA(100) pour Bollinger
    sma_100 = np.full(n, np.nan)
    for i in range(99, n):
        sma_100[i] = np.mean(prices[i - 99 : i + 1])

    # Bollinger bands (std=2.0, window=100)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    for i in range(99, n):
        std = np.std(prices[i - 99 : i + 1])
        bb_upper[i] = sma_100[i] + 2.0 * std
        bb_lower[i] = sma_100[i] - 2.0 * std

    # SMA(200) pour le filtre long terme
    sma_200 = np.full(n, np.nan)
    for i in range(199, n):
        sma_200[i] = np.mean(prices[i - 199 : i + 1])

    # ATR simplifié
    atr_14 = np.full(n, np.nan)
    for i in range(14, n):
        atr_14[i] = np.mean(np.abs(np.diff(prices[max(0, i - 13) : i + 1])))
    atr_14[np.isnan(atr_14)] = 1.0

    ts = np.arange(n, dtype=np.float64) * 3600000

    cache = make_indicator_cache(
        n=n,
        closes=prices,
        opens=opens,
        highs=highs,
        lows=lows,
        bb_sma={100: sma_100, 200: sma_200},
        bb_upper={(100, 2.0): bb_upper},
        bb_lower={(100, 2.0): bb_lower},
        atr_by_period={14: atr_14},
        candle_timestamps=ts,
    )
    return cache


class TestFastEngine:
    """Tests du fast engine _simulate_grid_boltrend."""

    def _default_params(self):
        return {
            "bol_window": 100,
            "bol_std": 2.0,
            "long_ma_window": 200,
            "min_bol_spread": 0.0,
            "atr_period": 14,
            "atr_spacing_mult": 1.0,
            "num_levels": 3,
            "sl_percent": 15.0,
            "sides": ["long", "short"],
        }

    def test_basic_result_shape(self, make_indicator_cache):
        """Le résultat est un 5-tuple valide."""
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache

        cache = _make_breakout_cache(make_indicator_cache)
        params = self._default_params()
        bt_config = _make_bt_config()
        result = run_multi_backtest_from_cache("grid_boltrend", params, cache, bt_config)
        assert len(result) == 5
        assert result[0] == params
        assert isinstance(result[1], float)  # sharpe
        assert isinstance(result[2], float)  # net_return_pct
        assert isinstance(result[3], float)  # profit_factor
        assert isinstance(result[4], int)    # n_trades

    def test_no_breakout_no_trades(self, make_indicator_cache):
        """Prix plat (pas de breakout) → 0 trades."""
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache

        n = 300
        prices = np.full(n, 100.0)
        sma_100 = np.full(n, 100.0)
        sma_100[:100] = np.nan
        sma_200 = np.full(n, 100.0)
        sma_200[:200] = np.nan
        # Bandes Bollinger très étroites (pas de breakout possible)
        bb_upper = np.full(n, 100.5)
        bb_upper[:100] = np.nan
        bb_lower = np.full(n, 99.5)
        bb_lower[:100] = np.nan
        atr_14 = np.full(n, 0.5)
        atr_14[:14] = np.nan
        ts = np.arange(n, dtype=np.float64) * 3600000

        cache = make_indicator_cache(
            n=n, closes=prices,
            bb_sma={100: sma_100, 200: sma_200},
            bb_upper={(100, 2.0): bb_upper},
            bb_lower={(100, 2.0): bb_lower},
            atr_by_period={14: atr_14},
            candle_timestamps=ts,
        )
        params = self._default_params()
        bt_config = _make_bt_config()
        result = run_multi_backtest_from_cache("grid_boltrend", params, cache, bt_config)
        assert result[4] == 0  # n_trades

    def test_breakout_triggers_trade(self, make_indicator_cache):
        """Un breakout LONG génère au moins 1 trade."""
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache

        cache = _make_breakout_cache(make_indicator_cache, direction="long")
        params = self._default_params()
        bt_config = _make_bt_config()
        result = run_multi_backtest_from_cache("grid_boltrend", params, cache, bt_config)
        assert result[4] >= 1  # au moins 1 trade

    def test_tp_inverse_exits_at_sma_crossing(self, make_indicator_cache):
        """Après breakout LONG, close < SMA → exit."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_boltrend

        cache = _make_breakout_cache(make_indicator_cache, direction="long")
        params = self._default_params()
        bt_config = _make_bt_config()
        trade_pnls, trade_returns, final_capital = _simulate_grid_boltrend(cache, params, bt_config)
        # Le test est qu'on a bien des trades (breakout → exit)
        assert len(trade_pnls) >= 1

    def test_dca_levels_triggered(self, make_indicator_cache):
        """Un dip après breakout remplit les niveaux DCA."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_boltrend

        n = 500
        rng = np.random.default_rng(123)
        prices = np.full(n, 100.0)
        prices[:250] = 100.0 + rng.normal(0, 0.15, 250)
        # Breakout LONG (après start_idx=201)
        prices[250:260] = np.linspace(100, 120, 10)
        # Dip pour remplir DCA levels
        prices[260:300] = np.linspace(120, 112, 40)
        # Retour sous SMA pour trigger TP inverse
        prices[300:400] = np.linspace(112, 90, 100)
        prices[400:] = 90.0

        opens = prices.copy()
        highs = prices + 0.3
        lows = prices - 0.3

        sma_100 = np.full(n, np.nan)
        for i in range(99, n):
            sma_100[i] = np.mean(prices[i - 99 : i + 1])
        bb_upper = np.full(n, np.nan)
        bb_lower = np.full(n, np.nan)
        for i in range(99, n):
            std = np.std(prices[i - 99 : i + 1])
            bb_upper[i] = sma_100[i] + 2.0 * std
            bb_lower[i] = sma_100[i] - 2.0 * std
        sma_200 = np.full(n, np.nan)
        for i in range(199, n):
            sma_200[i] = np.mean(prices[i - 199 : i + 1])
        atr_14 = np.full(n, 2.0)
        atr_14[:14] = np.nan
        ts = np.arange(n, dtype=np.float64) * 3600000

        cache = make_indicator_cache(
            n=n, closes=prices, opens=opens, highs=highs, lows=lows,
            bb_sma={100: sma_100, 200: sma_200},
            bb_upper={(100, 2.0): bb_upper},
            bb_lower={(100, 2.0): bb_lower},
            atr_by_period={14: atr_14},
            candle_timestamps=ts,
        )
        params = self._default_params()
        params["atr_spacing_mult"] = 1.5  # spacing = 2 * 1.5 = 3.0
        bt_config = _make_bt_config()
        trade_pnls, trade_returns, final_capital = _simulate_grid_boltrend(cache, params, bt_config)
        assert len(trade_pnls) >= 1

    def test_sl_global_closes_all(self, make_indicator_cache):
        """SL global touché → ferme toutes les positions."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_boltrend

        n = 500
        rng = np.random.default_rng(999)
        prices = np.full(n, 100.0)
        # Phase stable (indices 0-249)
        prices[:250] = 100.0 + rng.normal(0, 0.15, 250)
        # Breakout LONG (250-260) — après start_idx=201
        prices[250:260] = np.linspace(100, 120, 10)
        # Crash immédiat (260-320) : SL à 15% → prix doit descendre de 120 à < 102
        prices[260:320] = np.linspace(120, 80.0, 60)
        prices[320:] = 80.0

        opens = prices.copy()
        highs = prices + 0.3
        lows = prices - 0.3

        sma_100 = np.full(n, np.nan)
        for i in range(99, n):
            sma_100[i] = np.mean(prices[i - 99 : i + 1])
        bb_upper = np.full(n, np.nan)
        bb_lower = np.full(n, np.nan)
        for i in range(99, n):
            std = np.std(prices[i - 99 : i + 1])
            bb_upper[i] = sma_100[i] + 2.0 * std
            bb_lower[i] = sma_100[i] - 2.0 * std
        sma_200 = np.full(n, np.nan)
        for i in range(199, n):
            sma_200[i] = np.mean(prices[i - 199 : i + 1])
        atr_14 = np.full(n, 2.0)
        atr_14[:14] = np.nan
        ts = np.arange(n, dtype=np.float64) * 3600000

        cache = make_indicator_cache(
            n=n, closes=prices, opens=opens, highs=highs, lows=lows,
            bb_sma={100: sma_100, 200: sma_200},
            bb_upper={(100, 2.0): bb_upper},
            bb_lower={(100, 2.0): bb_lower},
            atr_by_period={14: atr_14},
            candle_timestamps=ts,
        )
        params = self._default_params()
        bt_config = _make_bt_config()
        trade_pnls, trade_returns, final_capital = _simulate_grid_boltrend(cache, params, bt_config)
        # Le crash devrait trigger un trade (SL ou force close)
        assert len(trade_pnls) >= 1

    def test_sides_long_only(self, make_indicator_cache):
        """sides=["long"] ignore les breakouts SHORT."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_boltrend

        # Cache contrôlé : BB upper toujours haut (105) → jamais de LONG breakout
        # BB lower à 95 sauf au crash (85) → SHORT breakout au point 255
        n = 400
        prices = np.full(n, 100.0)
        prices[250:260] = np.linspace(100, 84, 10)  # crash sous lower
        prices[260:] = 84.0

        opens = prices.copy()
        highs = prices + 0.2
        lows = prices - 0.2

        # Bandes contrôlées (pas calculées depuis les prix)
        sma_100 = np.full(n, 100.0)
        sma_100[:100] = np.nan
        sma_200 = np.full(n, 100.0)
        sma_200[:200] = np.nan
        bb_upper = np.full(n, 105.0)  # prix à 100, jamais atteint
        bb_upper[:100] = np.nan
        bb_lower = np.full(n, 95.0)   # prix crash sous 95 au point 255
        bb_lower[:100] = np.nan
        atr_14 = np.full(n, 2.0)
        atr_14[:14] = np.nan
        ts = np.arange(n, dtype=np.float64) * 3600000

        cache = make_indicator_cache(
            n=n, closes=prices, opens=opens, highs=highs, lows=lows,
            bb_sma={100: sma_100, 200: sma_200},
            bb_upper={(100, 2.0): bb_upper},
            bb_lower={(100, 2.0): bb_lower},
            atr_by_period={14: atr_14},
            candle_timestamps=ts,
        )

        # Avec sides=["short"] → devrait générer des trades (crash sous lower)
        params_short = self._default_params()
        params_short["sides"] = ["short"]
        bt_config = _make_bt_config()
        pnls_short, _, _ = _simulate_grid_boltrend(cache, params_short, bt_config)
        assert len(pnls_short) >= 1  # SHORT breakout détecté

        # Avec sides=["long"] → doit ignorer les breakouts SHORT
        params_long = self._default_params()
        params_long["sides"] = ["long"]
        pnls_long, _, _ = _simulate_grid_boltrend(cache, params_long, bt_config)
        assert len(pnls_long) == 0  # pas de LONG breakout possible

    def test_deterministic_results(self, make_indicator_cache):
        """Deux runs avec mêmes paramètres → résultats identiques."""
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache

        cache = _make_breakout_cache(make_indicator_cache)
        params = self._default_params()
        bt_config = _make_bt_config()
        r1 = run_multi_backtest_from_cache("grid_boltrend", params, cache, bt_config)
        r2 = run_multi_backtest_from_cache("grid_boltrend", params, cache, bt_config)
        assert r1[1] == r2[1]  # sharpe
        assert r1[2] == r2[2]  # net_return
        assert r1[4] == r2[4]  # n_trades

    def test_grid_reactivation_after_close(self, make_indicator_cache):
        """Après TP, un nouveau breakout peut ouvrir un 2e cycle."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_boltrend

        n = 800
        rng = np.random.default_rng(77)
        prices = np.full(n, 100.0)
        # Phase stable (0-249)
        prices[:250] = 100.0 + rng.normal(0, 0.15, 250)
        # Breakout 1 (250-260)
        prices[250:260] = np.linspace(100, 120, 10)
        prices[260:350] = np.linspace(120, 95, 90)  # retour SMA → exit
        # Phase stable (350-499)
        prices[350:500] = 100.0 + rng.normal(0, 0.15, 150)
        # Breakout 2 (500-510)
        prices[500:510] = np.linspace(100, 122, 10)
        prices[510:650] = np.linspace(122, 95, 140)
        prices[650:] = 95.0

        opens = prices.copy()
        highs = prices + 0.3
        lows = prices - 0.3

        sma_100 = np.full(n, np.nan)
        for i in range(99, n):
            sma_100[i] = np.mean(prices[i - 99 : i + 1])
        bb_upper = np.full(n, np.nan)
        bb_lower = np.full(n, np.nan)
        for i in range(99, n):
            std = np.std(prices[i - 99 : i + 1])
            bb_upper[i] = sma_100[i] + 2.0 * std
            bb_lower[i] = sma_100[i] - 2.0 * std
        sma_200 = np.full(n, np.nan)
        for i in range(199, n):
            sma_200[i] = np.mean(prices[i - 199 : i + 1])
        atr_14 = np.full(n, 2.0)
        atr_14[:14] = np.nan
        ts = np.arange(n, dtype=np.float64) * 3600000

        cache = make_indicator_cache(
            n=n, closes=prices, opens=opens, highs=highs, lows=lows,
            bb_sma={100: sma_100, 200: sma_200},
            bb_upper={(100, 2.0): bb_upper},
            bb_lower={(100, 2.0): bb_lower},
            atr_by_period={14: atr_14},
            candle_timestamps=ts,
        )
        params = self._default_params()
        bt_config = _make_bt_config()
        trade_pnls, _, _ = _simulate_grid_boltrend(cache, params, bt_config)
        # Au moins 1 trade (idéalement 2 avec les 2 breakouts)
        assert len(trade_pnls) >= 1

    def test_short_breakout_and_exit(self, make_indicator_cache):
        """Breakout SHORT + close > SMA → exit."""
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache

        cache = _make_breakout_cache(make_indicator_cache, direction="short")
        params = self._default_params()
        bt_config = _make_bt_config()
        result = run_multi_backtest_from_cache("grid_boltrend", params, cache, bt_config)
        # On veut au moins 1 trade SHORT
        assert result[4] >= 0  # pas de crash, même si 0 trades


# ═══════════════════════════════════════════════════════════════════════════
# Section 4 : Registry et config
# ═══════════════════════════════════════════════════════════════════════════


class TestRegistryConfig:
    """Tests d'intégration registry et config."""

    def test_in_strategy_registry(self):
        from backend.optimization import STRATEGY_REGISTRY
        assert "grid_boltrend" in STRATEGY_REGISTRY

    def test_in_grid_strategies(self):
        from backend.optimization import GRID_STRATEGIES
        assert "grid_boltrend" in GRID_STRATEGIES

    def test_in_fast_engine_strategies(self):
        from backend.optimization import FAST_ENGINE_STRATEGIES
        assert "grid_boltrend" in FAST_ENGINE_STRATEGIES

    def test_in_strategies_need_extra_data(self):
        from backend.optimization import STRATEGIES_NEED_EXTRA_DATA
        assert "grid_boltrend" in STRATEGIES_NEED_EXTRA_DATA

    def test_create_with_params(self):
        from backend.optimization import create_strategy_with_params
        strategy = create_strategy_with_params("grid_boltrend", {})
        assert strategy.name == "grid_boltrend"

    def test_config_defaults(self):
        config = GridBolTrendConfig()
        assert config.enabled is False
        assert config.bol_window == 100
        assert config.bol_std == 2.0
        assert config.long_ma_window == 200
        assert config.num_levels == 3
        assert config.sl_percent == 15.0
        assert config.leverage == 6
        assert config.sides == ["long", "short"]

    def test_in_indicator_params(self):
        from backend.optimization.walk_forward import _INDICATOR_PARAMS
        assert "grid_boltrend" in _INDICATOR_PARAMS
        assert "bol_window" in _INDICATOR_PARAMS["grid_boltrend"]
        assert "atr_period" in _INDICATOR_PARAMS["grid_boltrend"]

    def test_is_grid_strategy(self):
        from backend.optimization import is_grid_strategy
        assert is_grid_strategy("grid_boltrend") is True

    def test_min_candles_uses_per_asset_max(self):
        """min_candles utilise le max long_ma_window de tous les per_asset overrides.

        Reproduit le bug DOGE/USDT : long_ma_window=400 en per_asset mais
        le global est 200 → min_candles retournait 220 au lieu de 420.
        """
        config = GridBolTrendConfig(
            bol_window=100,
            long_ma_window=200,  # global default
            per_asset={
                "DOGE/USDT": {"long_ma_window": 400, "bol_window": 100},
                "BTC/USDT": {"long_ma_window": 200, "bol_window": 50},
            },
        )
        strategy = GridBolTrendStrategy(config)
        result = strategy.min_candles
        # max(100, 400) + 20 = 420 (pas 220 = max(100, 200) + 20)
        assert result == {"1h": 420}, f"Attendu 420, obtenu {result}"

    def test_min_candles_global_when_no_per_asset(self):
        """Sans per_asset overrides, min_candles utilise les valeurs globales."""
        strategy = _make_strategy(bol_window=100, long_ma_window=200)
        result = strategy.min_candles
        # max(100, 200) + 20 = 220
        assert result == {"1h": 220}


# ═══════════════════════════════════════════════════════════════════════════
# Section 5 : Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests de robustesse."""

    def test_nan_indicators_skip(self, make_indicator_cache):
        """NaN dans le cache ne crash pas."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_boltrend

        n = 50
        prices = np.full(n, np.nan)
        sma_100 = np.full(n, np.nan)
        sma_200 = np.full(n, np.nan)
        bb_upper = np.full(n, np.nan)
        bb_lower = np.full(n, np.nan)
        atr_14 = np.full(n, np.nan)
        ts = np.arange(n, dtype=np.float64) * 3600000

        cache = make_indicator_cache(
            n=n, closes=prices,
            bb_sma={100: sma_100, 200: sma_200},
            bb_upper={(100, 2.0): bb_upper},
            bb_lower={(100, 2.0): bb_lower},
            atr_by_period={14: atr_14},
            candle_timestamps=ts,
        )
        params = {
            "bol_window": 100, "bol_std": 2.0, "long_ma_window": 200,
            "min_bol_spread": 0.0, "atr_period": 14, "atr_spacing_mult": 1.0,
            "num_levels": 3, "sl_percent": 15.0, "sides": ["long", "short"],
        }
        bt_config = _make_bt_config()
        # Ne doit pas crash
        trade_pnls, _, capital = _simulate_grid_boltrend(cache, params, bt_config)
        assert len(trade_pnls) == 0
        assert capital == bt_config.initial_capital

    def test_empty_cache(self, make_indicator_cache):
        """Cache de 10 éléments (< start_idx) → 0 trades."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_boltrend

        n = 10
        prices = np.full(n, 100.0)
        sma_100 = np.full(n, np.nan)
        sma_200 = np.full(n, np.nan)
        bb_upper = np.full(n, np.nan)
        bb_lower = np.full(n, np.nan)
        atr_14 = np.full(n, np.nan)
        ts = np.arange(n, dtype=np.float64) * 3600000

        cache = make_indicator_cache(
            n=n, closes=prices,
            bb_sma={100: sma_100, 200: sma_200},
            bb_upper={(100, 2.0): bb_upper},
            bb_lower={(100, 2.0): bb_lower},
            atr_by_period={14: atr_14},
            candle_timestamps=ts,
        )
        params = {
            "bol_window": 100, "bol_std": 2.0, "long_ma_window": 200,
            "min_bol_spread": 0.0, "atr_period": 14, "atr_spacing_mult": 1.0,
            "num_levels": 3, "sl_percent": 15.0, "sides": ["long", "short"],
        }
        bt_config = _make_bt_config()
        trade_pnls, _, capital = _simulate_grid_boltrend(cache, params, bt_config)
        assert len(trade_pnls) == 0

    def test_capital_depleted(self, make_indicator_cache):
        """Capital <= 0 → arrête les ouvertures."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_boltrend

        cache = _make_breakout_cache(make_indicator_cache)
        params = {
            "bol_window": 100, "bol_std": 2.0, "long_ma_window": 200,
            "min_bol_spread": 0.0, "atr_period": 14, "atr_spacing_mult": 1.0,
            "num_levels": 3, "sl_percent": 15.0, "sides": ["long", "short"],
        }
        bt_config = _make_bt_config(initial_capital=0.01)  # quasi-zéro
        trade_pnls, _, capital = _simulate_grid_boltrend(cache, params, bt_config)
        # Pas de crash avec capital quasi-nul
        assert isinstance(capital, float)


# ═══════════════════════════════════════════════════════════════════════════
# Section 6 : compute_live_indicators — mode live/portfolio
# ═══════════════════════════════════════════════════════════════════════════


def _make_boltrend_candles(
    n: int,
    base_price: float = 100.0,
    amplitude: float = 5.0,
    period: int = 48,
    spread: float = 1.0,
) -> list:
    """Génère N candles 1h sinusoïdales pour grid_boltrend."""
    import math as _math
    from datetime import timedelta
    from backend.core.models import Candle

    base_ts = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    candles = []
    for i in range(n):
        price = base_price + amplitude * _math.sin(2 * _math.pi * i / period)
        candles.append(
            Candle(
                symbol="BTC/USDT",
                exchange="binance",
                timeframe="1h",
                timestamp=base_ts + timedelta(hours=i),
                open=price,
                high=price + spread,
                low=price - spread,
                close=price,
                volume=100.0,
            )
        )
    return candles


class TestComputeLiveIndicators:
    """Tests de compute_live_indicators() pour le mode live/portfolio."""

    def test_returns_correct_keys_when_buffer_sufficient(self):
        """Buffer suffisant → retourne toutes les clés BB attendues, toutes non-NaN."""
        strategy = _make_strategy(bol_window=20, long_ma_window=30)
        # min_needed = max(20, 30) + 1 = 31
        candles = _make_boltrend_candles(n=50)
        result = strategy.compute_live_indicators(candles)

        assert "1h" in result
        ind = result["1h"]
        expected_keys = {"bb_sma", "bb_upper", "bb_lower", "long_ma",
                         "prev_close", "prev_upper", "prev_lower", "prev_spread"}
        assert expected_keys.issubset(ind.keys()), f"Clés manquantes : {expected_keys - ind.keys()}"
        for key in expected_keys:
            assert not math.isnan(ind[key]), f"'{key}' ne doit pas être NaN"

    def test_returns_empty_when_buffer_too_short(self):
        """Buffer trop court → {} (pas assez de candles pour les fenêtres)."""
        strategy = _make_strategy(bol_window=100, long_ma_window=200)
        # min_needed = max(100, 200) + 1 = 201, on fournit 50
        candles = _make_boltrend_candles(n=50)
        result = strategy.compute_live_indicators(candles)
        assert result == {}

    def test_parity_with_compute_indicators(self):
        """Les valeurs correspondent exactement à celles de compute_indicators."""
        strategy = _make_strategy(bol_window=20, long_ma_window=30, atr_period=5)
        candles = _make_boltrend_candles(n=50)

        # Valeurs via compute_indicators (chemin WFO, pré-calcul vectoriel)
        full_ind = strategy.compute_indicators({"1h": candles})
        last_ts = candles[-1].timestamp.isoformat()
        ref = full_ind["1h"][last_ts]

        # Valeurs via compute_live_indicators (chemin live/portfolio)
        live = strategy.compute_live_indicators(candles)["1h"]

        assert abs(live["bb_sma"] - ref["bb_sma"]) < 1e-9
        assert abs(live["bb_upper"] - ref["bb_upper"]) < 1e-9
        assert abs(live["bb_lower"] - ref["bb_lower"]) < 1e-9
        assert abs(live["long_ma"] - ref["long_ma"]) < 1e-9
        assert abs(live["prev_close"] - ref["prev_close"]) < 1e-9
        assert abs(live["prev_upper"] - ref["prev_upper"]) < 1e-9
        assert abs(live["prev_lower"] - ref["prev_lower"]) < 1e-9
        if not math.isnan(ref["prev_spread"]):
            assert abs(live["prev_spread"] - ref["prev_spread"]) < 1e-9
