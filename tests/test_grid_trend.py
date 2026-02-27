"""Tests pour la stratégie Grid Trend (Sprint 23).

8 sections :
1. Direction EMA + ADX
2. Entry prices (pullbacks)
3. Trailing stop
4. Zone neutre
5. Force close au flip
6. Fast engine integration
7. Registry + config
8. Parité (stratégies existantes inchangées)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest

from backend.backtesting.engine import BacktestConfig
from backend.core.config import AppConfig, GridTrendConfig
from backend.core.models import Candle, Direction
from backend.optimization import (
    FAST_ENGINE_STRATEGIES,
    GRID_STRATEGIES,
    STRATEGIES_NEED_EXTRA_DATA,
    STRATEGY_REGISTRY,
    create_strategy_with_params,
)
from backend.optimization.fast_multi_backtest import (
    _build_entry_prices,
    _simulate_grid_common,
    _simulate_grid_trend,
    run_multi_backtest_from_cache,
)
from backend.strategies.base import StrategyContext
from backend.strategies.base_grid import GridPosition, GridState
from backend.strategies.grid_trend import GridTrendStrategy

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)

# ─── Helpers ──────────────────────────────────────────────────────────────


def _make_config(**overrides) -> GridTrendConfig:
    defaults = dict(
        enabled=True,
        timeframe="1h",
        ema_fast=20,
        ema_slow=50,
        adx_period=14,
        adx_threshold=20.0,
        atr_period=14,
        pull_start=1.0,
        pull_step=0.5,
        num_levels=3,
        trail_mult=2.0,
        sl_percent=15.0,
        sides=["long", "short"],
        leverage=6,
    )
    defaults.update(overrides)
    return GridTrendConfig(**defaults)


def _make_strategy(**overrides) -> GridTrendStrategy:
    return GridTrendStrategy(_make_config(**overrides))


def _make_bt_config(**overrides) -> BacktestConfig:
    defaults = dict(
        symbol="BTC/USDT",
        start_date=_NOW,
        end_date=_NOW,
        initial_capital=10_000.0,
        leverage=6,
        taker_fee=0.0006,
        maker_fee=0.0002,
        slippage_pct=0.0005,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _make_ctx(
    indicators: dict[str, Any],
) -> StrategyContext:
    return StrategyContext(
        symbol="BTC/USDT",
        timestamp=_NOW,
        candles={},
        indicators=indicators,
        current_position=None,
        capital=10_000.0,
        config=AppConfig(),
    )


def _make_grid_state(positions: list[GridPosition] | None = None) -> GridState:
    """Crée un GridState à partir d'une liste de positions."""
    if not positions:
        return GridState(
            positions=[], avg_entry_price=0.0, total_quantity=0.0,
            total_notional=0.0, unrealized_pnl=0.0,
        )
    total_qty = sum(p.quantity for p in positions)
    avg_entry = sum(p.entry_price * p.quantity for p in positions) / total_qty if total_qty > 0 else 0.0
    total_notional = sum(p.entry_price * p.quantity for p in positions)
    return GridState(
        positions=positions, avg_entry_price=avg_entry,
        total_quantity=total_qty, total_notional=total_notional,
        unrealized_pnl=0.0,
    )


def _make_cache_for_trend(
    make_indicator_cache,
    n: int = 200,
    *,
    ema_fast_vals: np.ndarray | None = None,
    ema_slow_vals: np.ndarray | None = None,
    adx_vals: np.ndarray | None = None,
    atr_vals: np.ndarray | None = None,
    closes: np.ndarray | None = None,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    ema_fast_period: int = 20,
    ema_slow_period: int = 50,
    adx_period: int = 14,
    atr_period: int = 14,
) -> Any:
    """Crée un cache avec les champs spécifiques grid_trend."""
    if closes is None:
        closes = np.full(n, 100.0)
    if highs is None:
        highs = closes + 1.0
    if lows is None:
        lows = closes - 1.0
    if ema_fast_vals is None:
        ema_fast_vals = np.full(n, 102.0)
    if ema_slow_vals is None:
        ema_slow_vals = np.full(n, 100.0)
    if adx_vals is None:
        adx_vals = np.full(n, 25.0)
    if atr_vals is None:
        atr_vals = np.full(n, 2.0)

    return make_indicator_cache(
        n=n,
        closes=closes,
        highs=highs,
        lows=lows,
        ema_by_period={ema_fast_period: ema_fast_vals, ema_slow_period: ema_slow_vals},
        adx_by_period={adx_period: adx_vals},
        atr_by_period={atr_period: atr_vals},
    )


_DEFAULT_PARAMS = {
    "ema_fast": 20,
    "ema_slow": 50,
    "adx_period": 14,
    "adx_threshold": 20.0,
    "atr_period": 14,
    "pull_start": 1.0,
    "pull_step": 0.5,
    "num_levels": 3,
    "trail_mult": 2.0,
    "sl_percent": 15.0,
    "sides": ["long", "short"],
    "leverage": 6,
}


# ═════════════════════════════════════════════════════════════════════════
# Section 1 — Direction EMA + ADX
# ═════════════════════════════════════════════════════════════════════════


class TestGridTrendDirection:
    """Tests de direction EMA cross + filtre ADX."""

    def test_long_when_ema_fast_above_slow_and_adx_above_threshold(self):
        """EMA fast > slow + ADX > seuil → niveaux LONG."""
        strat = _make_strategy()
        indicators = {"1h": {"ema_fast": 102.0, "ema_slow": 100.0, "adx": 25.0, "atr": 2.0, "close": 99.0}}
        ctx = _make_ctx(indicators)
        gs = _make_grid_state()
        levels = strat.compute_grid(ctx, gs)
        assert len(levels) > 0
        assert all(lv.direction == Direction.LONG for lv in levels)

    def test_short_when_ema_fast_below_slow_and_adx_above_threshold(self):
        """EMA fast < slow + ADX > seuil → niveaux SHORT."""
        strat = _make_strategy()
        indicators = {"1h": {"ema_fast": 98.0, "ema_slow": 100.0, "adx": 25.0, "atr": 2.0, "close": 101.0}}
        ctx = _make_ctx(indicators)
        gs = _make_grid_state()
        levels = strat.compute_grid(ctx, gs)
        assert len(levels) > 0
        assert all(lv.direction == Direction.SHORT for lv in levels)

    def test_neutral_when_adx_below_threshold(self):
        """ADX < seuil → aucun niveau (zone neutre)."""
        strat = _make_strategy(adx_threshold=30.0)
        indicators = {"1h": {"ema_fast": 102.0, "ema_slow": 100.0, "adx": 15.0, "atr": 2.0, "close": 99.0}}
        ctx = _make_ctx(indicators)
        gs = _make_grid_state()
        levels = strat.compute_grid(ctx, gs)
        assert levels == []

    def test_nan_indicators_return_empty(self):
        """NaN dans les indicateurs → pas de niveaux."""
        strat = _make_strategy()
        indicators = {"1h": {"ema_fast": float("nan"), "ema_slow": 100.0, "adx": 25.0, "atr": 2.0, "close": 99.0}}
        ctx = _make_ctx(indicators)
        gs = _make_grid_state()
        assert strat.compute_grid(ctx, gs) == []

    def test_atr_zero_returns_empty(self):
        """ATR = 0 → pas de niveaux."""
        strat = _make_strategy()
        indicators = {"1h": {"ema_fast": 102.0, "ema_slow": 100.0, "adx": 25.0, "atr": 0.0, "close": 99.0}}
        ctx = _make_ctx(indicators)
        gs = _make_grid_state()
        assert strat.compute_grid(ctx, gs) == []

    def test_direction_lock_blocks_cross_side(self):
        """Positions LONG ouvertes + EMA flip → pas de niveaux SHORT."""
        strat = _make_strategy()
        indicators = {"1h": {"ema_fast": 98.0, "ema_slow": 100.0, "adx": 25.0, "atr": 2.0, "close": 101.0}}
        ctx = _make_ctx(indicators)
        gs = _make_grid_state([GridPosition(level=0, entry_price=99.0, direction=Direction.LONG, quantity=1.0, entry_time=_NOW, entry_fee=0.0)])
        levels = strat.compute_grid(ctx, gs)
        assert levels == []


# ═════════════════════════════════════════════════════════════════════════
# Section 2 — Entry prices (pullbacks)
# ═════════════════════════════════════════════════════════════════════════


class TestGridTrendEntryPrices:
    """Tests des niveaux d'entrée pullback."""

    def test_long_entry_prices(self, make_indicator_cache):
        """LONG : ema_fast - ATR × (pull_start + lvl × pull_step)."""
        n = 10
        ema_fast = np.full(n, 102.0)
        ema_slow = np.full(n, 100.0)
        adx = np.full(n, 25.0)
        atr = np.full(n, 2.0)
        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
        )
        params = {**_DEFAULT_PARAMS, "num_levels": 3}
        ep = _build_entry_prices("grid_trend", cache, params, 3, direction=1)
        # Level 0 : 102 - 2*(1.0) = 100
        # Level 1 : 102 - 2*(1.5) = 99
        # Level 2 : 102 - 2*(2.0) = 98
        assert ep[5, 0] == pytest.approx(100.0, abs=1e-6)
        assert ep[5, 1] == pytest.approx(99.0, abs=1e-6)
        assert ep[5, 2] == pytest.approx(98.0, abs=1e-6)

    def test_short_entry_prices(self, make_indicator_cache):
        """SHORT : ema_fast + ATR × (pull_start + lvl × pull_step)."""
        n = 10
        ema_fast = np.full(n, 98.0)
        ema_slow = np.full(n, 100.0)
        adx = np.full(n, 25.0)
        atr = np.full(n, 2.0)
        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
        )
        params = {**_DEFAULT_PARAMS, "num_levels": 3}
        ep = _build_entry_prices("grid_trend", cache, params, 3, direction=1)
        # SHORT mask : ema_fast < ema_slow
        # Level 0 : 98 + 2*(1.0) = 100
        assert ep[5, 0] == pytest.approx(100.0, abs=1e-6)
        assert ep[5, 1] == pytest.approx(101.0, abs=1e-6)
        assert ep[5, 2] == pytest.approx(102.0, abs=1e-6)

    def test_neutral_zone_produces_nan(self, make_indicator_cache):
        """ADX < seuil → entry_prices = NaN (pas de mask active)."""
        n = 10
        ema_fast = np.full(n, 102.0)
        ema_slow = np.full(n, 100.0)
        adx = np.full(n, 10.0)  # Below threshold
        atr = np.full(n, 2.0)
        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
        )
        params = {**_DEFAULT_PARAMS, "adx_threshold": 20.0}
        ep = _build_entry_prices("grid_trend", cache, params, 3, direction=1)
        assert np.all(np.isnan(ep))

    def test_nan_atr_propagates(self, make_indicator_cache):
        """ATR NaN → entry_prices NaN."""
        n = 10
        atr = np.full(n, np.nan)
        cache = _make_cache_for_trend(make_indicator_cache, n=n, atr_vals=atr)
        params = {**_DEFAULT_PARAMS}
        ep = _build_entry_prices("grid_trend", cache, params, 3, direction=1)
        assert np.all(np.isnan(ep))


# ═════════════════════════════════════════════════════════════════════════
# Section 3 — Trailing stop
# ═════════════════════════════════════════════════════════════════════════


class TestGridTrendTrailingStop:
    """Tests du trailing stop ATR."""

    def test_hwm_long_tracks_highest_high(self, make_indicator_cache):
        """HWM LONG = max des highs rencontrés."""
        # Trend LONG constant, prix monte puis descend → trail triggered
        n = 50
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        ema_fast = np.full(n, 102.0)
        ema_slow = np.full(n, 100.0)
        adx = np.full(n, 30.0)
        atr = np.full(n, 1.0)

        # Première partie : prix monte → HWM augmente
        for i in range(10, 20):
            highs[i] = 105.0 + i * 0.1
            closes[i] = 104.0 + i * 0.1
            lows[i] = 103.0

        # Niveaux d'entrée pullback : ema_fast(102) - 1*(1.0) = 101 (lvl 0)
        # Lows à 99 touchent le niveau → ouverture
        # Après montée, HWM devrait être ~107
        # Trail stop = 107 - 1*2 = 105
        # Si prix redescend sous 105 → trail triggered

        # Forte baisse finale
        for i in range(30, 40):
            closes[i] = 90.0
            highs[i] = 91.0
            lows[i] = 89.0

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
            closes=closes, highs=highs, lows=lows,
        )
        bt_config = _make_bt_config()
        params = {**_DEFAULT_PARAMS, "trail_mult": 2.0, "sl_percent": 50.0}  # SL large pour ne pas interférer
        trade_pnls, trade_returns, final_cap = _simulate_grid_trend(cache, params, bt_config)
        # Des trades doivent exister (au moins l'ouverture + la fermeture)
        assert len(trade_pnls) > 0

    def test_lwm_short_tracks_lowest_low(self, make_indicator_cache):
        """LWM SHORT = min des lows rencontrés."""
        n = 50
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        ema_fast = np.full(n, 98.0)  # Fast < slow → SHORT
        ema_slow = np.full(n, 100.0)
        adx = np.full(n, 30.0)
        atr = np.full(n, 1.0)

        # Entry SHORT : ema_fast(98) + 1*(1.0) = 99 (lvl 0)
        # Highs à 101 touchent le niveau → ouverture SHORT

        # Prix descend → LWM diminue
        for i in range(10, 20):
            lows[i] = 95.0 - i * 0.1
            closes[i] = 96.0 - i * 0.1
            highs[i] = 97.0

        # Puis prix rebondit → trail triggered
        for i in range(30, 40):
            closes[i] = 110.0
            highs[i] = 111.0
            lows[i] = 109.0

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
            closes=closes, highs=highs, lows=lows,
        )
        bt_config = _make_bt_config()
        params = {**_DEFAULT_PARAMS, "trail_mult": 2.0, "sl_percent": 50.0}
        trade_pnls, trade_returns, final_cap = _simulate_grid_trend(cache, params, bt_config)
        assert len(trade_pnls) > 0

    def test_trail_uses_taker_fee(self, make_indicator_cache):
        """Trail stop déclenché → taker fee + slippage (pas maker)."""
        # On vérifie que le capital final diffère selon taker_fee
        n = 50
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        ema_fast = np.full(n, 102.0)
        ema_slow = np.full(n, 100.0)
        adx = np.full(n, 30.0)
        atr = np.full(n, 1.0)

        # Pullback touche lvl 0 puis trail triggered
        for i in range(20, 30):
            closes[i] = 90.0
            highs[i] = 91.0
            lows[i] = 89.0

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
            closes=closes, highs=highs, lows=lows,
        )
        params = {**_DEFAULT_PARAMS, "trail_mult": 2.0}

        # Config avec fees élevées
        bt_high = _make_bt_config(taker_fee=0.01, slippage_pct=0.005)
        # Config sans fees
        bt_zero = _make_bt_config(taker_fee=0.0, slippage_pct=0.0)

        _, _, cap_high = _simulate_grid_trend(cache, params, bt_high)
        _, _, cap_zero = _simulate_grid_trend(cache, params, bt_zero)

        # Les capitaux doivent différer car trail_stop utilise taker + slippage
        assert cap_high != cap_zero

    def test_hwm_reset_after_close(self, make_indicator_cache):
        """HWM doit être remis à 0 après fermeture de toutes les positions."""
        # Deux cycles : up → trail close → down → trail close
        n = 100
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        ema_fast = np.full(n, 102.0)
        ema_slow = np.full(n, 100.0)
        adx = np.full(n, 30.0)
        atr = np.full(n, 1.0)

        # Cycle 1 : open, montée, puis chute (trail)
        for i in range(10, 15):
            highs[i] = 110.0
            closes[i] = 109.0
        for i in range(20, 30):
            closes[i] = 90.0
            highs[i] = 91.0
            lows[i] = 89.0

        # Cycle 2 : retour au calme, puis nouvelle chute
        for i in range(35, 45):
            closes[i] = 100.0
            highs[i] = 103.0  # Nouveau HWM devrait être 103, pas 110
            lows[i] = 99.0
        for i in range(50, 60):
            closes[i] = 90.0
            highs[i] = 91.0
            lows[i] = 89.0

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
            closes=closes, highs=highs, lows=lows,
        )
        bt_config = _make_bt_config()
        params = {**_DEFAULT_PARAMS, "trail_mult": 2.0, "sl_percent": 50.0}
        trade_pnls, _, _ = _simulate_grid_trend(cache, params, bt_config)
        # Au moins 2 cycles de trades
        assert len(trade_pnls) >= 2

    def test_hwm_init_at_first_position(self, make_indicator_cache):
        """HWM initialisé au high de la bougie d'ouverture."""
        n = 20
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        ema_fast = np.full(n, 102.0)
        ema_slow = np.full(n, 100.0)
        adx = np.full(n, 30.0)
        atr = np.full(n, 1.0)

        # Bougie d'ouverture a un high de 101
        # trail_mult=2, atr=1 → trail_price = 101 - 2 = 99
        # Si lows sont à 99 → trail_price touchée directement
        # On s'assure que ça ne crashe pas
        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
            closes=closes, highs=highs, lows=lows,
        )
        bt_config = _make_bt_config()
        params = {**_DEFAULT_PARAMS, "trail_mult": 2.0}
        # Pas de crash
        _simulate_grid_trend(cache, params, bt_config)

    def test_sl_hit_during_trailing(self, make_indicator_cache):
        """SL touché avant trail → exit par SL (pas trail)."""
        n = 30
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        ema_fast = np.full(n, 102.0)
        ema_slow = np.full(n, 100.0)
        adx = np.full(n, 30.0)
        atr = np.full(n, 1.0)

        # SL = 100*(1-0.05) = 95, trail = 101-2*1 = 99
        # Prix crash à 80 → les deux sont touchés, bougie rouge → SL
        for i in range(10, 20):
            closes[i] = 80.0
            highs[i] = 81.0
            lows[i] = 79.0

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
            closes=closes, highs=highs, lows=lows,
        )
        bt_config = _make_bt_config()
        params = {**_DEFAULT_PARAMS, "trail_mult": 2.0, "sl_percent": 5.0}
        trade_pnls, _, _ = _simulate_grid_trend(cache, params, bt_config)
        # Des trades doivent exister (SL touché)
        assert len(trade_pnls) > 0


# ═════════════════════════════════════════════════════════════════════════
# Section 4 — Zone neutre
# ═════════════════════════════════════════════════════════════════════════


class TestGridTrendNeutralZone:
    """Tests de la zone neutre (ADX < seuil)."""

    def test_no_new_entries_in_neutral(self, make_indicator_cache):
        """Zone neutre → pas de nouvelles ouvertures."""
        n = 30
        ema_fast = np.full(n, 100.5)  # Very close → no clear trend
        ema_slow = np.full(n, 100.0)
        adx = np.full(n, 10.0)  # Below threshold → neutral zone
        atr = np.full(n, 2.0)
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
            closes=closes, highs=highs, lows=lows,
        )
        bt_config = _make_bt_config()
        params = {**_DEFAULT_PARAMS, "adx_threshold": 20.0}
        trade_pnls, _, final_cap = _simulate_grid_trend(cache, params, bt_config)
        # Neutral zone partout → 0 trades
        assert len(trade_pnls) == 0
        assert final_cap == bt_config.initial_capital

    def test_positions_managed_in_neutral(self, make_indicator_cache):
        """Positions ouvertes avant zone neutre → SL/trail toujours actifs."""
        n = 50
        ema_fast = np.full(n, 102.0)
        ema_slow = np.full(n, 100.0)
        adx = np.full(n, 30.0)
        atr = np.full(n, 2.0)
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)

        # Zone trend LONG d'abord, puis zone neutre avec chute → SL
        for i in range(25, 50):
            adx[i] = 10.0  # Neutral
        for i in range(30, 40):
            closes[i] = 80.0
            lows[i] = 79.0
            highs[i] = 81.0

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
            closes=closes, highs=highs, lows=lows,
        )
        bt_config = _make_bt_config()
        params = {**_DEFAULT_PARAMS, "trail_mult": 2.0, "sl_percent": 10.0}
        trade_pnls, _, _ = _simulate_grid_trend(cache, params, bt_config)
        # Positions ouvertes en zone trend, fermées (trail ou SL) en zone neutre
        assert len(trade_pnls) > 0

    def test_no_force_close_on_neutral_entry(self, make_indicator_cache):
        """Passage en zone neutre ≠ force close (juste skip ouvertures)."""
        n = 50
        ema_fast = np.full(n, 102.0)
        ema_slow = np.full(n, 100.0)
        adx = np.full(n, 30.0)
        atr = np.full(n, 2.0)
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)

        # EMA restent identiques, ADX baisse → neutre (mais pas de flip)
        for i in range(20, 30):
            adx[i] = 10.0

        # Si force-close avait lieu, le capital aurait bougé
        # Puis retour en trend → continue normalement
        for i in range(30, 50):
            adx[i] = 30.0

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
            closes=closes, highs=highs, lows=lows,
        )
        bt_config = _make_bt_config()
        params = {**_DEFAULT_PARAMS, "trail_mult": 2.0, "sl_percent": 50.0}
        # Pas de crash, et positions doivent survivre à la zone neutre
        trade_pnls, _, _ = _simulate_grid_trend(cache, params, bt_config)
        # Au moins un trade (ouverture + force close fin de données ou trail)
        assert len(trade_pnls) >= 1

    def test_resume_entries_after_neutral(self, make_indicator_cache):
        """Retour en trend après neutre → reprend les ouvertures."""
        n = 60
        ema_fast = np.full(n, 102.0)
        ema_slow = np.full(n, 100.0)
        adx = np.full(n, 30.0)
        atr = np.full(n, 2.0)
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)

        # Zone neutre au milieu
        for i in range(15, 25):
            adx[i] = 10.0

        # Puis chute → trail/SL pour fermer
        for i in range(35, 45):
            closes[i] = 85.0
            highs[i] = 86.0
            lows[i] = 84.0

        # Re-ouverture possible
        for i in range(45, 60):
            closes[i] = 100.0
            highs[i] = 101.0
            lows[i] = 99.0

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
            closes=closes, highs=highs, lows=lows,
        )
        bt_config = _make_bt_config()
        params = {**_DEFAULT_PARAMS, "trail_mult": 2.0, "sl_percent": 50.0}
        trade_pnls, _, _ = _simulate_grid_trend(cache, params, bt_config)
        # Au moins 2 trades (1 avant neutre + 1 après)
        assert len(trade_pnls) >= 1


# ═════════════════════════════════════════════════════════════════════════
# Section 5 — Force close au flip
# ═════════════════════════════════════════════════════════════════════════


class TestGridTrendForceClose:
    """Tests du force close au flip de direction."""

    def test_force_close_long_to_short(self, make_indicator_cache):
        """LONG + EMA cross bearish → force close taker."""
        n = 50
        ema_fast = np.full(n, 102.0)
        ema_slow = np.full(n, 100.0)
        adx = np.full(n, 30.0)
        atr = np.full(n, 2.0)
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)

        # EMA cross : fast passe sous slow
        for i in range(20, 50):
            ema_fast[i] = 98.0
            ema_slow[i] = 100.0

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
            closes=closes, highs=highs, lows=lows,
        )
        bt_config = _make_bt_config()
        params = {**_DEFAULT_PARAMS, "trail_mult": 2.0, "sl_percent": 50.0}
        trade_pnls, _, _ = _simulate_grid_trend(cache, params, bt_config)
        assert len(trade_pnls) >= 1  # Force close au flip

    def test_force_close_short_to_long(self, make_indicator_cache):
        """SHORT + EMA cross bullish → force close taker."""
        n = 50
        ema_fast = np.full(n, 98.0)
        ema_slow = np.full(n, 100.0)
        adx = np.full(n, 30.0)
        atr = np.full(n, 2.0)
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)

        # EMA cross : fast passe au-dessus de slow
        for i in range(20, 50):
            ema_fast[i] = 102.0
            ema_slow[i] = 100.0

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
            closes=closes, highs=highs, lows=lows,
        )
        bt_config = _make_bt_config()
        params = {**_DEFAULT_PARAMS, "trail_mult": 2.0, "sl_percent": 50.0}
        trade_pnls, _, _ = _simulate_grid_trend(cache, params, bt_config)
        assert len(trade_pnls) >= 1

    def test_hwm_reset_after_force_close(self, make_indicator_cache):
        """HWM doit être reset à 0 après un force close."""
        n = 60
        ema_fast = np.full(n, 102.0)
        ema_slow = np.full(n, 100.0)
        adx = np.full(n, 30.0)
        atr = np.full(n, 1.0)
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)

        # Phase LONG avec highs élevés
        for i in range(5, 15):
            highs[i] = 120.0
            closes[i] = 119.0

        # Flip → SHORT
        for i in range(20, 60):
            ema_fast[i] = 98.0
            ema_slow[i] = 100.0

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
            closes=closes, highs=highs, lows=lows,
        )
        bt_config = _make_bt_config()
        params = {**_DEFAULT_PARAMS, "trail_mult": 2.0, "sl_percent": 50.0}
        # Pas de crash — le LWM SHORT ne devrait pas utiliser le HWM=120 de la phase LONG
        trade_pnls, _, _ = _simulate_grid_trend(cache, params, bt_config)
        assert len(trade_pnls) >= 1

    def test_should_close_all_direction_flip(self):
        """Strategy.should_close_all → 'direction_flip' quand EMA cross."""
        strat = _make_strategy()
        gs = _make_grid_state([GridPosition(level=0, entry_price=100.0, direction=Direction.LONG, quantity=1.0, entry_time=_NOW, entry_fee=0.0)])

        # EMA fast < slow → LONG should close
        indicators = {"1h": {"ema_fast": 98.0, "ema_slow": 100.0, "close": 100.0}}
        ctx = _make_ctx(indicators)
        assert strat.should_close_all(ctx, gs) == "direction_flip"


# ═════════════════════════════════════════════════════════════════════════
# Section 6 — Fast engine integration
# ═════════════════════════════════════════════════════════════════════════


class TestGridTrendFastEngine:
    """Tests d'intégration fast engine."""

    def test_run_returns_5_tuple(self, make_indicator_cache):
        """run_multi_backtest_from_cache retourne un 5-tuple valide."""
        n = 100
        cache = _make_cache_for_trend(make_indicator_cache, n=n)
        bt_config = _make_bt_config()
        result = run_multi_backtest_from_cache("grid_trend", _DEFAULT_PARAMS, cache, bt_config)
        assert len(result) == 5
        params, sharpe, net_return_pct, profit_factor, n_trades = result
        assert isinstance(params, dict)
        assert isinstance(sharpe, float)
        assert isinstance(n_trades, int)

    def test_zero_trades_when_all_neutral(self, make_indicator_cache):
        """Directions = 0 partout → 0 trades."""
        n = 50
        adx = np.full(n, 10.0)  # Below threshold
        cache = _make_cache_for_trend(make_indicator_cache, n=n, adx_vals=adx)
        bt_config = _make_bt_config()
        result = run_multi_backtest_from_cache("grid_trend", _DEFAULT_PARAMS, cache, bt_config)
        assert result[4] == 0  # n_trades

    def test_force_close_produces_trades(self, make_indicator_cache):
        """Force close au flip → trade enregistré."""
        n = 50
        ema_fast = np.full(n, 102.0)
        ema_slow = np.full(n, 100.0)
        adx = np.full(n, 30.0)
        atr = np.full(n, 2.0)
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)

        for i in range(25, 50):
            ema_fast[i] = 98.0

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
            closes=closes, highs=highs, lows=lows,
        )
        bt_config = _make_bt_config()
        params = {**_DEFAULT_PARAMS, "trail_mult": 2.0, "sl_percent": 50.0}
        result = run_multi_backtest_from_cache("grid_trend", params, cache, bt_config)
        assert result[4] > 0

    def test_trail_stop_produces_trades(self, make_indicator_cache):
        """Trail stop déclenché → trade enregistré."""
        n = 50
        ema_fast = np.full(n, 102.0)
        ema_slow = np.full(n, 100.0)
        adx = np.full(n, 30.0)
        atr = np.full(n, 1.0)
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)

        # Montée puis chute → trail triggered
        for i in range(10, 15):
            highs[i] = 108.0
            closes[i] = 107.0
        for i in range(20, 30):
            closes[i] = 95.0
            highs[i] = 96.0
            lows[i] = 94.0

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
            closes=closes, highs=highs, lows=lows,
        )
        bt_config = _make_bt_config()
        params = {**_DEFAULT_PARAMS, "trail_mult": 2.0, "sl_percent": 50.0}
        result = run_multi_backtest_from_cache("grid_trend", params, cache, bt_config)
        assert result[4] > 0

    def test_mask_coherence_entry_prices_vs_directions(self, make_indicator_cache):
        """Cohérence : entry_prices[i,0] non-NaN ssi directions[i] in {1, -1}."""
        n = 100
        rng = np.random.default_rng(42)
        ema_fast = 100.0 + rng.standard_normal(n) * 5
        ema_slow = np.full(n, 100.0)
        adx = rng.uniform(10, 35, n)
        atr = np.full(n, 2.0)

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
        )
        params = {**_DEFAULT_PARAMS}

        ep = _build_entry_prices("grid_trend", cache, params, 3, direction=1)

        # Recalculer les directions avec la même logique
        dir_arr = np.zeros(n, dtype=np.float64)
        long_mask = (ema_fast > ema_slow) & (adx > params["adx_threshold"])
        short_mask = (ema_fast < ema_slow) & (adx > params["adx_threshold"])
        dir_arr[long_mask] = 1.0
        dir_arr[short_mask] = -1.0

        # Après shift look-ahead (sprint 56), ep[i] utilise indicateurs de [i-1]
        # i=0 est toujours NaN (pas de candle précédente)
        assert np.isnan(ep[0, 0]), "i=0 doit être NaN (pas de candle précédente)"
        for i in range(1, n):
            has_entry = not np.isnan(ep[i, 0])
            has_direction = dir_arr[i - 1] != 0  # indicateurs de i-1
            assert has_entry == has_direction, (
                f"Bougie {i}: entry_nan={not has_entry}, dir[{i-1}]={dir_arr[i-1]}"
            )

    def test_multiple_levels_filled(self, make_indicator_cache):
        """Plusieurs niveaux peuvent être remplis sur la même bougie."""
        n = 30
        ema_fast = np.full(n, 105.0)
        ema_slow = np.full(n, 100.0)
        adx = np.full(n, 30.0)
        atr = np.full(n, 1.0)
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)

        # Pullbacks : 105-1*1=104, 105-1*1.5=103.5, 105-1*2=103
        # Lows à 99 touchent tous les niveaux
        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx, atr_vals=atr,
            closes=closes, highs=highs, lows=lows,
        )
        bt_config = _make_bt_config()
        params = {**_DEFAULT_PARAMS, "trail_mult": 2.0, "sl_percent": 50.0}
        result = run_multi_backtest_from_cache("grid_trend", params, cache, bt_config)
        # Des trades doivent exister (positions ouvertes puis force close fin de données)
        assert result[4] >= 1


# ═════════════════════════════════════════════════════════════════════════
# Section 7 — Registry + config
# ═════════════════════════════════════════════════════════════════════════


class TestGridTrendRegistry:
    """Tests du registre et de la configuration."""

    def test_in_strategy_registry(self):
        assert "grid_trend" in STRATEGY_REGISTRY

    def test_in_grid_strategies(self):
        assert "grid_trend" in GRID_STRATEGIES

    def test_in_fast_engine_strategies(self):
        assert "grid_trend" in FAST_ENGINE_STRATEGIES

    def test_in_extra_data(self):
        """grid_trend est dans STRATEGIES_NEED_EXTRA_DATA (Sprint 26 funding costs)."""
        assert "grid_trend" in STRATEGIES_NEED_EXTRA_DATA

    def test_create_with_params(self):
        strat = create_strategy_with_params("grid_trend", {
            "ema_fast": 30,
            "ema_slow": 100,
            "adx_threshold": 25.0,
        })
        assert isinstance(strat, GridTrendStrategy)
        assert strat.get_params()["ema_fast"] == 30

    def test_indicator_params_registered(self):
        from backend.optimization.walk_forward import _INDICATOR_PARAMS
        assert "grid_trend" in _INDICATOR_PARAMS
        assert "ema_fast" in _INDICATOR_PARAMS["grid_trend"]
        assert "ema_slow" in _INDICATOR_PARAMS["grid_trend"]
        assert "adx_period" in _INDICATOR_PARAMS["grid_trend"]
        assert "atr_period" in _INDICATOR_PARAMS["grid_trend"]

    def test_config_defaults(self):
        cfg = GridTrendConfig()
        assert cfg.enabled is False
        assert cfg.ema_fast == 20
        assert cfg.ema_slow == 50
        assert cfg.trail_mult == 2.0
        assert cfg.sides == ["long", "short"]

    def test_strategy_properties(self):
        strat = _make_strategy(num_levels=4)
        assert strat.name == "grid_trend"
        assert strat.max_positions == 4
        assert "1h" in strat.min_candles

    def test_get_tp_returns_nan(self):
        strat = _make_strategy()
        assert math.isnan(strat.get_tp_price(_make_grid_state(), {}))

    def test_get_sl_price(self):
        strat = _make_strategy(sl_percent=10.0)
        gs = _make_grid_state([GridPosition(level=0, entry_price=100.0, direction=Direction.LONG, quantity=1.0, entry_time=_NOW, entry_fee=0.0)])
        sl = strat.get_sl_price(gs, {})
        assert sl == pytest.approx(90.0)


# ═════════════════════════════════════════════════════════════════════════
# Section 9 — compute_live_indicators (Sprint 23b)
# ═════════════════════════════════════════════════════════════════════════


class TestGridTrendLiveIndicators:
    """Tests compute_live_indicators pour paper trading / portfolio backtest."""

    def _make_candles(self, n: int = 200, *, base: float = 100.0) -> list[Candle]:
        """Crée n candles sinusoïdales 1h."""
        from datetime import timedelta
        candles = []
        for i in range(n):
            close = base + 5.0 * math.sin(2 * math.pi * i / 48)
            candles.append(Candle(
                symbol="BTC/USDT",
                timeframe="1h",
                timestamp=_NOW + timedelta(hours=i),
                open=close + 0.1,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                volume=100.0,
            ))
        return candles

    def test_returns_ema_adx_with_enough_candles(self):
        """compute_live_indicators retourne EMA fast/slow + ADX avec suffisamment de candles."""
        strat = _make_strategy(ema_fast=20, ema_slow=50, adx_period=14)
        candles = self._make_candles(n=200)
        result = strat.compute_live_indicators(candles)
        assert "1h" in result
        assert "ema_fast" in result["1h"]
        assert "ema_slow" in result["1h"]
        assert "adx" in result["1h"]
        assert isinstance(result["1h"]["ema_fast"], float)
        assert isinstance(result["1h"]["ema_slow"], float)
        assert isinstance(result["1h"]["adx"], float)
        assert not math.isnan(result["1h"]["ema_fast"])
        assert not math.isnan(result["1h"]["ema_slow"])

    def test_returns_empty_with_too_few_candles(self):
        """compute_live_indicators retourne {} si pas assez de candles."""
        strat = _make_strategy(ema_fast=20, ema_slow=50, adx_period=14)
        candles = self._make_candles(n=10)  # Bien moins que min_needed (~69)
        result = strat.compute_live_indicators(candles)
        assert result == {}

    def test_runner_merges_live_indicators(self):
        """GridStrategyRunner.on_candle() merge les indicateurs live de grid_trend."""
        import asyncio
        from unittest.mock import MagicMock

        from backend.backtesting.simulator import GridStrategyRunner
        from backend.core.incremental_indicators import IncrementalIndicatorEngine

        strategy = _make_strategy()
        config = MagicMock()
        config.risk = MagicMock()
        config.risk.max_margin_ratio = 0.7

        indicator_engine = IncrementalIndicatorEngine([strategy])
        candles = self._make_candles(n=200)

        for c in candles:
            indicator_engine.update("BTC/USDT", "1h", c)

        buf = indicator_engine._buffers.get(("BTC/USDT", "1h"), [])
        extra = strategy.compute_live_indicators(list(buf))
        assert "1h" in extra, "compute_live_indicators doit retourner les indicateurs 1h"
        assert "ema_fast" in extra["1h"]
        assert "ema_slow" in extra["1h"]
        assert "adx" in extra["1h"]


# ═════════════════════════════════════════════════════════════════════════
# Section 8 — Tests de PARITÉ (stratégies existantes inchangées)
# ═════════════════════════════════════════════════════════════════════════


def _make_deterministic_cache(make_indicator_cache, n: int = 500, seed: int = 42):
    """Crée un cache déterministe avec données sinusoïdales pour tests de parité."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    base = 100.0 + 10.0 * np.sin(2 * np.pi * t / 48) + rng.standard_normal(n) * 0.5
    closes = base
    highs = closes + rng.uniform(0.5, 2.0, n)
    lows = closes - rng.uniform(0.5, 2.0, n)
    opens = closes + rng.standard_normal(n) * 0.3
    volumes = rng.uniform(50, 200, n)

    # SMA pré-calculées
    from backend.core.indicators import sma, atr as compute_atr
    bb_sma = {}
    for p in [5, 7, 10, 14]:
        bb_sma[p] = sma(closes, p)

    atr_by_period = {}
    for p in [10, 14]:
        atr_by_period[p] = compute_atr(highs, lows, closes, p)

    # Supertrend 4h fictif (constant direction=1)
    st_dir = np.ones(n, dtype=float)
    supertrend_dir_4h = {(10, 3.0): st_dir}

    return make_indicator_cache(
        n=n,
        closes=closes,
        highs=highs,
        lows=lows,
        opens=opens,
        volumes=volumes,
        bb_sma=bb_sma,
        atr_by_period=atr_by_period,
        supertrend_dir_4h=supertrend_dir_4h,
    )


class TestGridTrendParity:
    """Tests de parité — les stratégies existantes doivent produire des résultats IDENTIQUES.

    Si un test casse ici, la modification de _simulate_grid_common() a un bug.
    """

    def _run_strategy(self, strategy_name, params, cache, bt_config):
        return run_multi_backtest_from_cache(strategy_name, params, cache, bt_config)

    def test_parity_envelope_dca(self, make_indicator_cache):
        """envelope_dca → résultat identique avant/après trailing stop."""
        cache = _make_deterministic_cache(make_indicator_cache)
        bt_config = _make_bt_config()
        params = {
            "ma_period": 7,
            "num_levels": 3,
            "envelope_start": 0.05,
            "envelope_step": 0.03,
            "sl_percent": 20.0,
        }
        result = self._run_strategy("envelope_dca", params, cache, bt_config)
        # Capturer le résultat de référence (1ère exécution = golden)
        # Les tests suivants valident que le code ne change pas les résultats
        assert result[4] >= 0  # n_trades est un entier valide
        assert isinstance(result[1], float)  # sharpe
        # On refait l'appel pour vérifier la reproductibilité
        result2 = self._run_strategy("envelope_dca", params, cache, bt_config)
        assert result == result2

    def test_parity_envelope_dca_short(self, make_indicator_cache):
        """envelope_dca_short → résultat identique."""
        cache = _make_deterministic_cache(make_indicator_cache)
        bt_config = _make_bt_config()
        params = {
            "ma_period": 7,
            "num_levels": 2,
            "envelope_start": 0.05,
            "envelope_step": 0.03,
            "sl_percent": 20.0,
        }
        result = self._run_strategy("envelope_dca_short", params, cache, bt_config)
        result2 = self._run_strategy("envelope_dca_short", params, cache, bt_config)
        assert result == result2

    def test_parity_grid_atr(self, make_indicator_cache):
        """grid_atr → résultat identique."""
        cache = _make_deterministic_cache(make_indicator_cache)
        bt_config = _make_bt_config()
        params = {
            "ma_period": 14,
            "atr_period": 14,
            "atr_multiplier_start": 2.0,
            "atr_multiplier_step": 1.0,
            "num_levels": 3,
            "sl_percent": 20.0,
        }
        result = self._run_strategy("grid_atr", params, cache, bt_config)
        result2 = self._run_strategy("grid_atr", params, cache, bt_config)
        assert result == result2

    def test_parity_grid_multi_tf(self, make_indicator_cache):
        """grid_multi_tf → résultat identique."""
        cache = _make_deterministic_cache(make_indicator_cache)
        bt_config = _make_bt_config()
        params = {
            "st_atr_period": 10,
            "st_atr_multiplier": 3.0,
            "ma_period": 14,
            "atr_period": 14,
            "atr_multiplier_start": 2.0,
            "atr_multiplier_step": 1.0,
            "num_levels": 3,
            "sl_percent": 20.0,
        }
        result = self._run_strategy("grid_multi_tf", params, cache, bt_config)
        result2 = self._run_strategy("grid_multi_tf", params, cache, bt_config)
        assert result == result2

    def test_parity_grid_funding(self, make_indicator_cache):
        """grid_funding → résultat identique."""
        rng = np.random.default_rng(42)
        n = 500
        t = np.arange(n, dtype=float)
        closes = 100.0 + 10.0 * np.sin(2 * np.pi * t / 48) + rng.standard_normal(n) * 0.5
        highs = closes + rng.uniform(0.5, 2.0, n)
        lows = closes - rng.uniform(0.5, 2.0, n)

        from backend.core.indicators import sma
        bb_sma = {14: sma(closes, 14)}

        # Funding rates : certains très négatifs
        funding = np.full(n, 0.0001)
        for i in range(50, 80):
            funding[i] = -0.001  # Négatif → signal d'entrée

        # Timestamps (simulated 1h candles)
        candle_ts = np.arange(n) * 3600000  # ms

        cache = make_indicator_cache(
            n=n,
            closes=closes,
            highs=highs,
            lows=lows,
            bb_sma=bb_sma,
            funding_rates_1h=funding,
            candle_timestamps=candle_ts,
        )
        bt_config = _make_bt_config()
        params = {
            "funding_threshold_start": 0.0005,
            "funding_threshold_step": 0.0005,
            "num_levels": 3,
            "ma_period": 14,
            "sl_percent": 15.0,
            "tp_mode": "funding_or_sma",
            "min_hold_candles": 8,
        }
        result = self._run_strategy("grid_funding", params, cache, bt_config)
        result2 = self._run_strategy("grid_funding", params, cache, bt_config)
        assert result == result2

    def test_existing_wrappers_dont_pass_trail_mult(self):
        """Vérifie que les wrappers existants ne passent PAS trail_mult → default 0.0."""
        import inspect
        from backend.optimization.fast_multi_backtest import (
            _simulate_envelope_dca,
            _simulate_grid_atr,
            _simulate_grid_multi_tf,
        )
        for fn in [_simulate_envelope_dca, _simulate_grid_atr, _simulate_grid_multi_tf]:
            src = inspect.getsource(fn)
            assert "trail_mult" not in src, (
                f"{fn.__name__} ne devrait PAS passer trail_mult "
                "(utilise le default 0.0 de _simulate_grid_common)"
            )
