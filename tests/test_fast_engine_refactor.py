"""Tests Sprint 20c — Factorisation fast engine (_build_entry_prices + _simulate_grid_common).

Couvre :
- Section 1 : _build_entry_prices (7 tests)
- Section 2 : FAST_ENGINE_STRATEGIES (3 tests)
- Section 3 : Parité bit-à-bit (3 tests)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.backtesting.engine import BacktestConfig
from backend.optimization.fast_multi_backtest import (
    _build_entry_prices,
    _simulate_grid_common,
    run_multi_backtest_from_cache,
)
from backend.optimization.indicator_cache import IndicatorCache
from datetime import datetime, timezone


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_cache(
    n: int = 500,
    seed: int = 42,
    *,
    factory=None,
    bb_sma: dict | None = None,
    atr_by_period: dict | None = None,
) -> IndicatorCache:
    """Crée un cache synthétique reproductible.

    Si factory (make_indicator_cache fixture) est fourni, l'utilise.
    Sinon, construit le cache directement.
    """
    rng = np.random.default_rng(seed)
    prices = 100.0 + np.cumsum(rng.normal(0, 2.0, n))

    opens = prices + rng.uniform(-0.3, 0.3, n)
    highs = prices + np.abs(rng.normal(3.0, 1.5, n))
    lows = prices - np.abs(rng.normal(3.0, 1.5, n))

    sma_7 = np.full(n, np.nan)
    sma_14 = np.full(n, np.nan)
    atr_14 = np.full(n, np.nan)

    for i in range(7, n):
        sma_7[i] = np.mean(prices[max(0, i - 6) : i + 1])
    for i in range(14, n):
        sma_14[i] = np.mean(prices[max(0, i - 13) : i + 1])
        atr_14[i] = np.mean(np.abs(np.diff(prices[max(0, i - 13) : i + 1])))

    if factory is not None:
        return factory(
            n=n,
            closes=prices,
            opens=opens,
            highs=highs,
            lows=lows,
            bb_sma=bb_sma if bb_sma is not None else {7: sma_7, 14: sma_14},
            atr_by_period=atr_by_period if atr_by_period is not None else {14: atr_14},
        )

    return IndicatorCache(
        n_candles=n,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=prices,
        volumes=np.full(n, 100.0),
        total_days=n / 24,
        rsi={14: np.full(n, 50.0)},
        vwap=np.full(n, np.nan),
        vwap_distance_pct=np.full(n, np.nan),
        adx_arr=np.full(n, 25.0),
        di_plus=np.full(n, 15.0),
        di_minus=np.full(n, 10.0),
        atr_arr=np.full(n, 1.0),
        atr_sma=np.full(n, 1.0),
        volume_sma_arr=np.full(n, 100.0),
        regime=np.zeros(n, dtype=np.int8),
        rolling_high={},
        rolling_low={},
        filter_adx=np.full(n, np.nan),
        filter_di_plus=np.full(n, np.nan),
        filter_di_minus=np.full(n, np.nan),
        bb_sma=bb_sma if bb_sma is not None else {7: sma_7, 14: sma_14},
        bb_upper={},
        bb_lower={},
        supertrend_direction={},
        atr_by_period=atr_by_period if atr_by_period is not None else {14: atr_14},
        supertrend_dir_4h={},
    )


def _make_bt_config() -> BacktestConfig:
    return BacktestConfig(
        symbol="TEST/USDT",
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
        leverage=6,
        initial_capital=10_000,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Section 1 : _build_entry_prices
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildEntryPrices:
    """Tests unitaires pour _build_entry_prices."""

    def test_envelope_dca_long(self):
        """Shape correcte et valeurs = sma * (1 - offset) pour envelope_dca LONG."""
        n = 50
        sma = np.full(n, 100.0)
        sma[:7] = np.nan
        cache = _make_cache(n, bb_sma={7: sma}, atr_by_period={14: np.full(n, 1.0)})

        params = {"ma_period": 7, "envelope_start": 0.05, "envelope_step": 0.02, "num_levels": 3}
        ep = _build_entry_prices("envelope_dca", cache, params, 3, direction=1)

        assert ep.shape == (n, 3)
        # NaN pour les premières candles (SMA NaN)
        assert math.isnan(ep[0, 0])
        # Valeurs valides
        assert ep[10, 0] == pytest.approx(100.0 * (1 - 0.05))  # lvl 0 : 95.0
        assert ep[10, 1] == pytest.approx(100.0 * (1 - 0.07))  # lvl 1 : 93.0
        assert ep[10, 2] == pytest.approx(100.0 * (1 - 0.09))  # lvl 2 : 91.0

    def test_envelope_dca_short(self):
        """Offsets asymétriques round(1/(1-e)-1, 3) pour SHORT."""
        n = 50
        sma = np.full(n, 100.0)
        sma[:7] = np.nan
        cache = _make_cache(n, bb_sma={7: sma}, atr_by_period={14: np.full(n, 1.0)})

        params = {"ma_period": 7, "envelope_start": 0.05, "envelope_step": 0.02, "num_levels": 2}
        ep = _build_entry_prices("envelope_dca_short", cache, params, 2, direction=-1)

        # Asymétrie : offset_upper = round(1/(1-lower)-1, 3)
        # lvl 0 : lower=0.05, upper=round(1/0.95-1, 3)=round(0.0526.., 3)=0.053
        # lvl 1 : lower=0.07, upper=round(1/0.93-1, 3)=round(0.0752.., 3)=0.075
        expected_0 = 100.0 * (1 + round(1 / (1 - 0.05) - 1, 3))
        expected_1 = 100.0 * (1 + round(1 / (1 - 0.07) - 1, 3))
        assert ep[10, 0] == pytest.approx(expected_0)
        assert ep[10, 1] == pytest.approx(expected_1)

    def test_grid_atr_long(self):
        """Valeurs = sma - atr * multiplier pour grid_atr LONG."""
        n = 50
        sma = np.full(n, 100.0)
        sma[:14] = np.nan
        atr = np.full(n, 3.0)
        atr[:14] = np.nan
        cache = _make_cache(n, bb_sma={14: sma}, atr_by_period={14: atr})

        params = {
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 2.0, "atr_multiplier_step": 1.0, "num_levels": 3,
        }
        ep = _build_entry_prices("grid_atr", cache, params, 3, direction=1)

        assert ep.shape == (n, 3)
        assert math.isnan(ep[0, 0])
        # lvl 0 : 100 - 3*2 = 94
        assert ep[20, 0] == pytest.approx(100.0 - 3.0 * 2.0)
        # lvl 1 : 100 - 3*3 = 91
        assert ep[20, 1] == pytest.approx(100.0 - 3.0 * 3.0)
        # lvl 2 : 100 - 3*4 = 88
        assert ep[20, 2] == pytest.approx(100.0 - 3.0 * 4.0)

    def test_grid_atr_nan_atr(self):
        """ATR NaN ou <= 0 → entry_prices NaN."""
        n = 30
        sma = np.full(n, 100.0)
        atr = np.full(n, 3.0)
        atr[5] = np.nan   # NaN
        atr[10] = 0.0     # <= 0
        atr[15] = -1.0    # <= 0
        cache = _make_cache(n, bb_sma={14: sma}, atr_by_period={14: atr})

        params = {
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 1.0, "atr_multiplier_step": 0.5, "num_levels": 2,
        }
        ep = _build_entry_prices("grid_atr", cache, params, 2, direction=1)

        # Après le shift look-ahead (i utilise i-1), les NaN sont à i+1
        assert math.isnan(ep[6, 0])   # ATR NaN at [5] → visible at [6]
        assert math.isnan(ep[11, 0])  # ATR = 0 at [10] → visible at [11]
        assert math.isnan(ep[16, 0])  # ATR < 0 at [15] → visible at [16]
        # i=0 toujours NaN (pas de candle précédente)
        assert math.isnan(ep[0, 0])
        # Candle valide
        assert not math.isnan(ep[20, 0])

    def test_sma_nan_propagation(self):
        """SMA NaN → entry_prices NaN pour les 2 types de stratégie."""
        n = 20
        sma = np.full(n, np.nan)  # Tout NaN
        atr = np.full(n, 3.0)
        cache = _make_cache(n, bb_sma={7: sma, 14: sma}, atr_by_period={14: atr})

        # envelope_dca
        params_env = {"ma_period": 7, "envelope_start": 0.05, "envelope_step": 0.02, "num_levels": 2}
        ep_env = _build_entry_prices("envelope_dca", cache, params_env, 2, direction=1)
        assert all(math.isnan(ep_env[i, 0]) for i in range(n))

        # grid_atr
        params_grid = {
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 1.0, "atr_multiplier_step": 0.5, "num_levels": 2,
        }
        ep_grid = _build_entry_prices("grid_atr", cache, params_grid, 2, direction=1)
        assert all(math.isnan(ep_grid[i, 0]) for i in range(n))

    def test_entry_price_lte_zero_handled_by_common(self):
        """Prix négatifs dans entry_prices : gérés en aval par _simulate_grid_common."""
        n = 30
        sma = np.full(n, 10.0)  # SMA faible
        atr = np.full(n, 100.0)  # ATR énorme → entry = 10 - 100*2 = -190
        cache = _make_cache(n, bb_sma={14: sma}, atr_by_period={14: atr})

        params = {
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 2.0, "atr_multiplier_step": 1.0, "num_levels": 2,
        }
        ep = _build_entry_prices("grid_atr", cache, params, 2, direction=1)

        # Entry price négatif (mais pas NaN)
        assert ep[20, 0] < 0
        assert not math.isnan(ep[20, 0])

        # _simulate_grid_common doit ignorer ces niveaux (ep <= 0 check)
        bt_config = _make_bt_config()
        result = run_multi_backtest_from_cache("grid_atr", {**params, "sl_percent": 20.0}, cache, bt_config)
        assert result[4] == 0  # 0 trades (aucun niveau valide)

    def test_unknown_strategy_raises(self):
        """Stratégie inconnue lève ValueError."""
        cache = _make_cache(20)
        with pytest.raises(ValueError, match="inconnue"):
            _build_entry_prices("unknown", cache, {"ma_period": 7}, 2, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 : FAST_ENGINE_STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════


class TestFastEngineStrategies:
    """Tests pour la constante FAST_ENGINE_STRATEGIES."""

    def test_content(self):
        """Contient les 15 stratégies attendues."""
        from backend.optimization import FAST_ENGINE_STRATEGIES

        expected = {
            "vwap_rsi", "momentum", "bollinger_mr", "donchian_breakout",
            "supertrend", "boltrend", "envelope_dca", "envelope_dca_short",
            "grid_atr", "grid_range_atr", "grid_multi_tf", "grid_funding",
            "grid_trend", "grid_boltrend", "grid_momentum",
        }
        assert FAST_ENGINE_STRATEGIES == expected

    def test_excludes_extra_data(self):
        """funding et liquidation absents (nécessitent extra_data)."""
        from backend.optimization import FAST_ENGINE_STRATEGIES

        assert "funding" not in FAST_ENGINE_STRATEGIES
        assert "liquidation" not in FAST_ENGINE_STRATEGIES

    def test_derived_from_registry(self):
        """Dérivée automatiquement du registre (REGISTRY - _NO_FAST_ENGINE)."""
        from backend.optimization import (
            FAST_ENGINE_STRATEGIES,
            STRATEGY_REGISTRY,
            _NO_FAST_ENGINE,
        )

        assert FAST_ENGINE_STRATEGIES == set(STRATEGY_REGISTRY.keys()) - _NO_FAST_ENGINE


# ═══════════════════════════════════════════════════════════════════════════
# Section 3 : Parité bit-à-bit
# ═══════════════════════════════════════════════════════════════════════════


# Valeurs capturées post-Sprint 56 (look-ahead fix, slippage, margin guard, seed=42, n=500, vol=2.0)
_EXPECTED_ENVELOPE_DCA = (55.25512965761102, 25667.36397824845, 35.060915572448664, 47)
_EXPECTED_ENVELOPE_DCA_SHORT = (45.571870934729766, 4266.514200648141, 10.619894422789638, 38)
_EXPECTED_GRID_ATR = (33.29853014825827, 5633.459636806401, 3.2603291267265373, 54)


class TestParityBitwise:
    """Tests de parité bit-à-bit : résultats post-refactoring = pré-refactoring."""

    def test_parity_envelope_dca_long(self):
        """envelope_dca LONG produit des résultats identiques."""
        cache = _make_cache(500, seed=42)
        bt_config = _make_bt_config()

        params = {
            "ma_period": 7, "num_levels": 3,
            "envelope_start": 0.07, "envelope_step": 0.03,
            "sl_percent": 25.0,
        }
        result = run_multi_backtest_from_cache("envelope_dca", params, cache, bt_config)

        sharpe, ret, pf, n_trades = _EXPECTED_ENVELOPE_DCA
        assert result[1] == sharpe, f"Sharpe: {result[1]} != {sharpe}"
        assert result[2] == pytest.approx(ret, abs=1e-10), f"Return: {result[2]} != {ret}"
        assert result[3] == pytest.approx(pf, abs=1e-10), f"PF: {result[3]} != {pf}"
        assert result[4] == n_trades, f"Trades: {result[4]} != {n_trades}"

    def test_parity_envelope_dca_short(self):
        """envelope_dca_short produit des résultats identiques."""
        cache = _make_cache(500, seed=42)
        bt_config = _make_bt_config()

        params = {
            "ma_period": 7, "num_levels": 3,
            "envelope_start": 0.07, "envelope_step": 0.03,
            "sl_percent": 25.0,
        }
        result = run_multi_backtest_from_cache("envelope_dca_short", params, cache, bt_config)

        sharpe, ret, pf, n_trades = _EXPECTED_ENVELOPE_DCA_SHORT
        assert result[1] == sharpe, f"Sharpe: {result[1]} != {sharpe}"
        assert result[2] == pytest.approx(ret, abs=1e-10), f"Return: {result[2]} != {ret}"
        assert result[3] == pytest.approx(pf, abs=1e-10), f"PF: {result[3]} != {pf}"
        assert result[4] == n_trades, f"Trades: {result[4]} != {n_trades}"

    def test_parity_grid_atr(self):
        """grid_atr LONG produit des résultats identiques."""
        cache = _make_cache(500, seed=42)
        bt_config = _make_bt_config()

        params = {
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 2.0, "atr_multiplier_step": 1.0,
            "num_levels": 3, "sl_percent": 20.0,
        }
        result = run_multi_backtest_from_cache("grid_atr", params, cache, bt_config)

        sharpe, ret, pf, n_trades = _EXPECTED_GRID_ATR
        assert result[1] == sharpe, f"Sharpe: {result[1]} != {sharpe}"
        assert result[2] == pytest.approx(ret, abs=1e-10), f"Return: {result[2]} != {ret}"
        assert result[3] == pytest.approx(pf, abs=1e-10), f"PF: {result[3]} != {pf}"
        assert result[4] == n_trades, f"Trades: {result[4]} != {n_trades}"
