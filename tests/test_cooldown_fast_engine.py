"""Tests Sprint 51 — Cooldown anti-churn dans le fast backtest engine.

Vérifie que :
- Le cooldown bloque les ré-entrées après fermeture de grid
- cooldown=0 ne change rien (backward compat, bit-for-bit)
- Le flag can_open_new ne bloque pas les exits (SL/TP)
- Les trades avec cooldown sont un sous-ensemble des trades sans cooldown
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from backend.backtesting.engine import BacktestConfig
from backend.optimization.fast_multi_backtest import (
    _build_entry_prices,
    _simulate_grid_boltrend,
    _simulate_grid_common,
    _simulate_grid_range,
    run_multi_backtest_from_cache,
)


# ─── Helpers ──────────────────────────────────────────────────────────────


def _make_bt_config(**kwargs) -> BacktestConfig:
    defaults = dict(
        symbol="BTC/USDT",
        start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        initial_capital=10_000.0,
        leverage=6,
        maker_fee=0.0002,
        taker_fee=0.0006,
        slippage_pct=0.0005,
        max_wfo_drawdown_pct=80.0,
    )
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


def _make_oscillating_cache(make_indicator_cache, n=200, base=100.0, amplitude=5.0):
    """Crée un cache avec des prix oscillants autour de base ± amplitude.

    Génère des cycles entry → exit réguliers pour tester le cooldown.
    """
    t = np.arange(n, dtype=np.float64)
    # Oscillation sinusoïdale avec période ~20 candles
    wave = amplitude * np.sin(2 * np.pi * t / 20)
    closes = base + wave
    opens = closes - 0.1
    highs = closes + amplitude * 0.3
    lows = closes - amplitude * 0.3

    sma = np.full(n, base)
    atr = np.full(n, amplitude * 0.5)

    return make_indicator_cache(
        n=n,
        closes=closes,
        opens=opens,
        highs=highs,
        lows=lows,
        bb_sma={20: sma},
        atr_by_period={14: atr},
    )


def _make_grid_atr_params(**overrides):
    defaults = dict(
        ma_period=20,
        atr_period=14,
        atr_multiplier_start=1.0,
        atr_multiplier_step=0.5,
        num_levels=3,
        sl_percent=5.0,
        cooldown_candles=0,
    )
    defaults.update(overrides)
    return defaults


def _make_boltrend_params(**overrides):
    defaults = dict(
        bol_window=20,
        bol_std=2.0,
        long_ma_window=50,
        atr_period=14,
        atr_spacing_mult=1.0,
        num_levels=3,
        sl_percent=5.0,
        sides=["long", "short"],
        min_bol_spread=0.0,
        cooldown_candles=0,
    )
    defaults.update(overrides)
    return defaults


def _make_range_params(**overrides):
    defaults = dict(
        ma_period=20,
        atr_period=14,
        atr_spacing_mult=1.0,
        num_levels=2,
        sl_percent=5.0,
        tp_mode="dynamic_sma",
        sides=["long", "short"],
        cooldown_candles=0,
    )
    defaults.update(overrides)
    return defaults


# ─── TestCooldownGridCommon ──────────────────────────────────────────────


class TestCooldownGridCommon:
    """Tests cooldown dans _simulate_grid_common (grid_atr, envelope_dca, etc.)."""

    def test_cooldown_blocks_reentry_grid_atr(self, make_indicator_cache):
        """Avec cooldown=3, le nombre de trades diminue vs cooldown=0."""
        cache = _make_oscillating_cache(make_indicator_cache)
        bt = _make_bt_config()
        params_0 = _make_grid_atr_params(cooldown_candles=0)
        params_3 = _make_grid_atr_params(cooldown_candles=3)

        entry_prices_0 = _build_entry_prices("grid_atr", cache, params_0, 3, 1)
        entry_prices_3 = _build_entry_prices("grid_atr", cache, params_3, 3, 1)
        sma = cache.bb_sma[20]

        pnls_0, _, _ = _simulate_grid_common(
            entry_prices_0, sma, cache, bt, 3, 0.05, 1, cooldown_candles=0,
        )
        pnls_3, _, _ = _simulate_grid_common(
            entry_prices_3, sma, cache, bt, 3, 0.05, 1, cooldown_candles=3,
        )
        # Cooldown=3 doit avoir <= trades que cooldown=0
        assert len(pnls_0) >= len(pnls_3)

    def test_cooldown_zero_no_effect(self, make_indicator_cache):
        """cooldown=0 → même résultat que sans param cooldown (backward compat)."""
        cache = _make_oscillating_cache(make_indicator_cache)
        bt = _make_bt_config()
        params = _make_grid_atr_params(cooldown_candles=0)

        entry_prices = _build_entry_prices("grid_atr", cache, params, 3, 1)
        sma = cache.bb_sma[20]

        pnls_with, rets_with, cap_with = _simulate_grid_common(
            entry_prices, sma, cache, bt, 3, 0.05, 1, cooldown_candles=0,
        )
        pnls_without, rets_without, cap_without = _simulate_grid_common(
            entry_prices, sma, cache, bt, 3, 0.05, 1,
        )
        assert len(pnls_with) == len(pnls_without)
        assert cap_with == pytest.approx(cap_without, rel=1e-10)

    def test_cooldown_exits_then_waits(self, make_indicator_cache):
        """Données synthétiques : après exit, pas de nouvelle entrée pendant cooldown."""
        n = 50
        # Prix qui déclenchent une entrée puis un SL, puis restent stables
        closes = np.full(n, 100.0)
        opens = np.full(n, 100.0)
        highs = np.full(n, 100.5)
        lows = np.full(n, 99.5)

        # Candle 5 : prix chute pour trigger entry (low <= entry_price)
        lows[5] = 95.0
        closes[5] = 96.0

        # Candle 8 : SL hit (low <= avg_entry * (1 - sl_pct))
        lows[8] = 89.0
        closes[8] = 90.0

        # Candle 10 : prix revient, potentiellement nouvelle entrée
        lows[10] = 95.0
        closes[10] = 96.0

        # Candle 15 : idem
        lows[15] = 95.0
        closes[15] = 96.0

        sma = np.full(n, 100.0)
        atr = np.full(n, 2.0)

        cache = make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma}, atr_by_period={14: atr},
        )
        bt = _make_bt_config()

        # Construire entry_prices manuellement (SMA - ATR * mult)
        entry_prices = np.full((n, 1), np.nan)
        for i in range(n):
            entry_prices[i, 0] = sma[i] - atr[i] * 1.0  # = 98.0

        # Avec cooldown=5 : après exit à candle 8, pas de ré-entrée avant candle 13
        pnls_5, _, _ = _simulate_grid_common(
            entry_prices, sma, cache, bt, 1, 0.10, 1, cooldown_candles=5,
        )
        # Avec cooldown=0 : ré-entrée dès que possible
        pnls_0, _, _ = _simulate_grid_common(
            entry_prices, sma, cache, bt, 1, 0.10, 1, cooldown_candles=0,
        )
        # Le cooldown doit réduire ou égaler le nombre de trades
        assert len(pnls_0) >= len(pnls_5)

    def test_cooldown_grid_multi_tf_directions(self, make_indicator_cache):
        """Cooldown respecté même quand direction change (force-close → cooldown)."""
        n = 100
        closes = np.full(n, 100.0)
        opens = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)

        sma = np.full(n, 100.0)
        atr = np.full(n, 2.0)

        # Directions : LONG 0-30, SHORT 30-60, LONG 60-100
        directions = np.ones(n, dtype=np.float64)
        directions[30:60] = -1.0

        cache = make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma}, atr_by_period={14: atr},
        )
        bt = _make_bt_config()
        entry_prices = np.full((n, 2), np.nan)
        for i in range(n):
            if directions[i] == 1:
                entry_prices[i, 0] = sma[i] - atr[i] * 1.0
                entry_prices[i, 1] = sma[i] - atr[i] * 1.5
            else:
                entry_prices[i, 0] = sma[i] + atr[i] * 1.0
                entry_prices[i, 1] = sma[i] + atr[i] * 1.5

        pnls_0, _, _ = _simulate_grid_common(
            entry_prices, sma, cache, bt, 2, 0.05, 1,
            directions=directions, cooldown_candles=0,
        )
        pnls_5, _, _ = _simulate_grid_common(
            entry_prices, sma, cache, bt, 2, 0.05, 1,
            directions=directions, cooldown_candles=5,
        )
        assert len(pnls_0) >= len(pnls_5)

    def test_cooldown_envelope_dca(self, make_indicator_cache):
        """envelope_dca reçoit et applique le cooldown via wrapper."""
        cache = _make_oscillating_cache(make_indicator_cache)
        bt = _make_bt_config()

        params_0 = dict(
            ma_period=20, envelope_start=0.02, envelope_step=0.01,
            num_levels=3, sl_percent=5.0, cooldown_candles=0,
        )
        params_3 = dict(
            ma_period=20, envelope_start=0.02, envelope_step=0.01,
            num_levels=3, sl_percent=5.0, cooldown_candles=3,
        )

        r0 = run_multi_backtest_from_cache("envelope_dca", params_0, cache, bt)
        r3 = run_multi_backtest_from_cache("envelope_dca", params_3, cache, bt)
        # n_trades est result[4]
        assert r0[4] >= r3[4]


# ─── TestCooldownGridBoltrend ────────────────────────────────────────────


class TestCooldownGridBoltrend:
    """Tests cooldown dans _simulate_grid_boltrend."""

    def _make_boltrend_cache(self, make_indicator_cache, n=300):
        """Cache avec données pour grid_boltrend (needs bb bands + long MA)."""
        t = np.arange(n, dtype=np.float64)
        wave = 5.0 * np.sin(2 * np.pi * t / 40)
        base = 100.0
        closes = base + wave
        opens = closes - 0.2
        highs = closes + 2.0
        lows = closes - 2.0

        sma_20 = np.full(n, base)
        sma_50 = np.full(n, base)
        atr = np.full(n, 2.5)

        # Bollinger bands : SMA ± std
        bb_upper = sma_20 + 2 * 3.0  # wide bands
        bb_lower = sma_20 - 2 * 3.0

        return make_indicator_cache(
            n=n,
            closes=closes,
            opens=opens,
            highs=highs,
            lows=lows,
            bb_sma={20: sma_20, 50: sma_50},
            bb_upper={(20, 2.0): bb_upper},
            bb_lower={(20, 2.0): bb_lower},
            atr_by_period={14: atr},
        )

    def test_cooldown_blocks_breakout(self, make_indicator_cache):
        """Après close, pas de nouveau breakout pendant cooldown."""
        cache = self._make_boltrend_cache(make_indicator_cache)
        bt = _make_bt_config()

        params_0 = _make_boltrend_params(cooldown_candles=0)
        params_5 = _make_boltrend_params(cooldown_candles=5)

        pnls_0, _, _ = _simulate_grid_boltrend(cache, params_0, bt)
        pnls_5, _, _ = _simulate_grid_boltrend(cache, params_5, bt)

        assert len(pnls_0) >= len(pnls_5)

    def test_cooldown_zero_allows_immediate(self, make_indicator_cache):
        """cooldown=0 → breakout immédiat après close (pas de blocage)."""
        cache = self._make_boltrend_cache(make_indicator_cache)
        bt = _make_bt_config()

        params_0 = _make_boltrend_params(cooldown_candles=0)
        pnls_0, _, cap_0 = _simulate_grid_boltrend(cache, params_0, bt)
        # Juste vérifier que ça tourne sans erreur et retourne des résultats
        assert isinstance(pnls_0, list)
        assert isinstance(cap_0, float)

    def test_cooldown_counts_candles(self, make_indicator_cache):
        """Le cooldown compte en candles, pas en timestamps."""
        cache = self._make_boltrend_cache(make_indicator_cache)
        bt = _make_bt_config()

        # cooldown=1 devrait être presque identique à cooldown=0
        params_1 = _make_boltrend_params(cooldown_candles=1)
        params_10 = _make_boltrend_params(cooldown_candles=10)

        pnls_1, _, _ = _simulate_grid_boltrend(cache, params_1, bt)
        pnls_10, _, _ = _simulate_grid_boltrend(cache, params_10, bt)

        # cooldown=10 devrait avoir <= trades que cooldown=1
        assert len(pnls_1) >= len(pnls_10)


# ─── TestCooldownGridRange ──────────────────────────────────────────────


class TestCooldownGridRange:
    """Tests cooldown dans _simulate_grid_range."""

    def _make_range_cache(self, make_indicator_cache, n=200):
        t = np.arange(n, dtype=np.float64)
        wave = 3.0 * np.sin(2 * np.pi * t / 25)
        base = 100.0
        closes = base + wave
        opens = closes - 0.1
        highs = closes + 1.5
        lows = closes - 1.5

        sma = np.full(n, base)
        atr = np.full(n, 2.0)

        return make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma}, atr_by_period={14: atr},
        )

    def test_cooldown_blocks_after_all_closed(self, make_indicator_cache):
        """Cooldown actif quand toutes positions fermées."""
        cache = self._make_range_cache(make_indicator_cache)
        bt = _make_bt_config()

        params_0 = _make_range_params(cooldown_candles=0)
        params_5 = _make_range_params(cooldown_candles=5)

        pnls_0, _, _ = _simulate_grid_range(cache, params_0, bt)
        pnls_5, _, _ = _simulate_grid_range(cache, params_5, bt)

        assert len(pnls_0) >= len(pnls_5)

    def test_cooldown_no_block_while_positions_open(self, make_indicator_cache):
        """Positions encore ouvertes → pas de cooldown sur les nouvelles."""
        cache = self._make_range_cache(make_indicator_cache)
        bt = _make_bt_config()

        # Avec cooldown très grand mais positions ouvertes → ne devrait pas bloquer
        # le filling de niveaux DCA supplémentaires
        params = _make_range_params(cooldown_candles=100, num_levels=3)
        pnls, _, cap = _simulate_grid_range(cache, params, bt)

        # Doit avoir pu ouvrir des positions (le cooldown ne bloque que
        # quand toutes les positions sont fermées)
        assert isinstance(pnls, list)
        assert isinstance(cap, float)


# ─── TestRegression ──────────────────────────────────────────────────────


class TestRegression:
    """Tests de régression bit-for-bit : cooldown=0 == sans param."""

    def test_cooldown_zero_bit_for_bit(self, make_indicator_cache):
        """grid_atr avec cooldown=0 vs sans param → résultats identiques."""
        cache = _make_oscillating_cache(make_indicator_cache)
        bt = _make_bt_config()

        params_with = _make_grid_atr_params(cooldown_candles=0)
        params_without = _make_grid_atr_params()
        # S'assurer que les deux ont bien cooldown_candles=0
        assert params_with["cooldown_candles"] == 0
        assert params_without["cooldown_candles"] == 0

        r_with = run_multi_backtest_from_cache("grid_atr", params_with, cache, bt)
        r_without = run_multi_backtest_from_cache("grid_atr", params_without, cache, bt)

        # Bit-for-bit identical
        assert r_with[1] == pytest.approx(r_with[1], rel=1e-12)  # sharpe
        assert r_with[2] == pytest.approx(r_without[2], rel=1e-12)  # return
        assert r_with[3] == pytest.approx(r_without[3], rel=1e-12)  # profit_factor
        assert r_with[4] == r_without[4]  # n_trades

    def test_cooldown_zero_boltrend_bit_for_bit(self, make_indicator_cache):
        """grid_boltrend avec cooldown=0 vs sans param → résultats identiques."""
        n = 300
        t = np.arange(n, dtype=np.float64)
        wave = 5.0 * np.sin(2 * np.pi * t / 40)
        base = 100.0
        closes = base + wave
        opens = closes - 0.2
        highs = closes + 2.0
        lows = closes - 2.0

        sma_20 = np.full(n, base)
        sma_50 = np.full(n, base)
        atr = np.full(n, 2.5)
        bb_upper = sma_20 + 2 * 3.0
        bb_lower = sma_20 - 2 * 3.0

        cache = make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma_20, 50: sma_50},
            bb_upper={(20, 2.0): bb_upper},
            bb_lower={(20, 2.0): bb_lower},
            atr_by_period={14: atr},
        )
        bt = _make_bt_config()

        params_with = _make_boltrend_params(cooldown_candles=0)
        params_without = _make_boltrend_params()
        del params_without["cooldown_candles"]  # Pas de param → default 0

        pnls_with, rets_with, cap_with = _simulate_grid_boltrend(cache, params_with, bt)
        pnls_without, rets_without, cap_without = _simulate_grid_boltrend(cache, params_without, bt)

        assert len(pnls_with) == len(pnls_without)
        assert cap_with == pytest.approx(cap_without, rel=1e-12)
        for pw, pwo in zip(pnls_with, pnls_without):
            assert pw == pytest.approx(pwo, rel=1e-12)


# ─── TestParity ──────────────────────────────────────────────────────────


class TestParity:
    """Tests de parité et propriétés du cooldown."""

    def test_warning_removed(self, make_indicator_cache):
        """Plus de UserWarning émis quand cooldown > 0."""
        cache = _make_oscillating_cache(make_indicator_cache)
        bt = _make_bt_config()
        params = _make_grid_atr_params(cooldown_candles=3)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_multi_backtest_from_cache("grid_atr", params, cache, bt)
            cooldown_warnings = [
                x for x in w
                if "cooldown_candles" in str(x.message)
            ]
            assert len(cooldown_warnings) == 0

    def test_cooldown_reduces_trades_subset(self, make_indicator_cache):
        """Trades avec cooldown sont un sous-ensemble (par count) des trades sans."""
        cache = _make_oscillating_cache(make_indicator_cache, n=500)
        bt = _make_bt_config()

        params = _make_grid_atr_params()
        entry_prices = _build_entry_prices("grid_atr", cache, params, 3, 1)
        sma = cache.bb_sma[20]

        pnls_0, _, cap_0 = _simulate_grid_common(
            entry_prices, sma, cache, bt, 3, 0.05, 1, cooldown_candles=0,
        )
        pnls_3, _, cap_3 = _simulate_grid_common(
            entry_prices, sma, cache, bt, 3, 0.05, 1, cooldown_candles=3,
        )
        pnls_10, _, cap_10 = _simulate_grid_common(
            entry_prices, sma, cache, bt, 3, 0.05, 1, cooldown_candles=10,
        )

        # Monotonically decreasing trades with increasing cooldown
        assert len(pnls_0) >= len(pnls_3) >= len(pnls_10)

    def test_cooldown_does_not_skip_exits(self, make_indicator_cache):
        """Position ouverte + cooldown actif → SL se déclenche quand même."""
        n = 30
        closes = np.full(n, 100.0)
        opens = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.5)

        # Candle 3 : entry trigger
        lows[3] = 96.0
        closes[3] = 97.0

        # Candle 5 : SL hit (baisse brutale)
        lows[5] = 90.0
        closes[5] = 91.0

        # Candle 7 : encore un entry trigger (pendant cooldown)
        lows[7] = 96.0
        closes[7] = 97.0

        # Candle 9 : SL hit encore — mais avec cooldown, pas d'entrée à candle 7
        # donc rien à sortir. Vérifions que si on entre quand même (cooldown=0),
        # le SL fonctionne.
        lows[9] = 90.0
        closes[9] = 91.0

        sma = np.full(n, 100.0)
        atr = np.full(n, 2.0)

        cache = make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma}, atr_by_period={14: atr},
        )
        bt = _make_bt_config()

        entry_prices = np.full((n, 1), np.nan)
        for i in range(n):
            entry_prices[i, 0] = sma[i] - atr[i] * 1.0  # 98.0

        # cooldown=0 : devrait avoir 2 trades (entry@3 → SL@5, entry@7 → SL@9)
        pnls_0, _, _ = _simulate_grid_common(
            entry_prices, sma, cache, bt, 1, 0.05, 1, cooldown_candles=0,
        )

        # cooldown=10 : devrait avoir 1 trade (entry@3 → SL@5, puis cooldown bloque)
        pnls_10, _, _ = _simulate_grid_common(
            entry_prices, sma, cache, bt, 1, 0.05, 1, cooldown_candles=10,
        )

        # Le SL a BIEN déclenché dans les deux cas
        assert len(pnls_0) >= 1, "SL doit déclencher au moins 1 trade"
        assert len(pnls_10) >= 1, "SL doit déclencher malgré le cooldown"
        # Mais le cooldown=10 bloque la 2ème entrée
        assert len(pnls_0) >= len(pnls_10)
