"""Tests Grid ATR — support sides SHORT dans le WFO.

Couvre :
- Test 1 : sides=["short"] → direction=-1, entries au-dessus de la SMA
- Test 2 : sides=["long"] backward compat (identique à sans sides)
- Test 3 : sides=["long","short"] → direction=1 (LONG prioritaire)
- Test 4 : run_multi_backtest_from_cache avec sides=["short"] → 5-tuple valide
- Test 5 : param_grids.yaml contient sides dans grid_atr.default
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from backend.backtesting.engine import BacktestConfig
from backend.optimization.fast_multi_backtest import (
    _build_entry_prices,
    _simulate_grid_atr,
    run_multi_backtest_from_cache,
)


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_bt_config(**overrides):
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


_DEFAULT_PARAMS = {
    "ma_period": 14,
    "atr_period": 14,
    "atr_multiplier_start": 2.0,
    "atr_multiplier_step": 1.0,
    "num_levels": 3,
    "sl_percent": 20.0,
}


def _make_cache(make_indicator_cache, n=200, seed=42):
    """Crée un cache avec prix sinusoïdaux et SMA/ATR réalistes."""
    rng = np.random.default_rng(seed)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    sma_arr = np.full(n, np.nan)
    atr_arr = np.full(n, np.nan)
    for i in range(14, n):
        sma_arr[i] = np.mean(prices[max(0, i - 13) : i + 1])
        atr_arr[i] = np.mean(np.abs(np.diff(prices[max(0, i - 13) : i + 1])))

    return make_indicator_cache(
        n=n,
        closes=prices,
        opens=prices + rng.uniform(-0.3, 0.3, n),
        highs=prices + np.abs(rng.normal(1.0, 0.5, n)),
        lows=prices - np.abs(rng.normal(1.0, 0.5, n)),
        bb_sma={14: sma_arr},
        atr_by_period={14: atr_arr},
    )


# ─── Tests ─────────────────────────────────────────────────────────────────


class TestGridATRShort:
    """Tests sides SHORT pour grid_atr."""

    def test_short_direction_entries_above_sma(self, make_indicator_cache):
        """sides=["short"] → direction=-1 → entries au-dessus de la SMA."""
        cache = _make_cache(make_indicator_cache)
        params = {**_DEFAULT_PARAMS, "sides": ["short"]}

        entry_prices = _build_entry_prices("grid_atr", cache, params, 3, direction=-1)

        # Vérifier sur les candles valides (SMA non-NaN)
        sma = cache.bb_sma[14]
        valid = ~np.isnan(sma) & ~np.isnan(entry_prices[:, 0])
        assert valid.sum() > 50, "Pas assez de candles valides"

        # SHORT : toutes les entries doivent être AU-DESSUS de la SMA
        for lvl in range(3):
            assert np.all(entry_prices[valid, lvl] > sma[valid]), (
                f"Niveau {lvl} : certaines entries SHORT sont sous la SMA"
            )

    def test_simulate_short_uses_direction_minus1(self, make_indicator_cache):
        """_simulate_grid_atr avec sides=["short"] utilise direction=-1."""
        cache = _make_cache(make_indicator_cache)
        bt = _make_bt_config()
        params_short = {**_DEFAULT_PARAMS, "sides": ["short"]}
        params_long = {**_DEFAULT_PARAMS, "sides": ["long"]}

        pnls_short, _, _ = _simulate_grid_atr(cache, params_short, bt)
        pnls_long, _, _ = _simulate_grid_atr(cache, params_long, bt)

        # Les PnL doivent être différents (directions opposées)
        assert pnls_short != pnls_long, "SHORT et LONG devraient avoir des PnL différents"

    def test_long_backward_compat(self, make_indicator_cache):
        """sides=["long"] donne le même résultat que sans sides (backward compat)."""
        cache = _make_cache(make_indicator_cache)
        bt = _make_bt_config()

        params_with_sides = {**_DEFAULT_PARAMS, "sides": ["long"]}
        params_without_sides = {**_DEFAULT_PARAMS}

        pnls_with, rets_with, cap_with = _simulate_grid_atr(cache, params_with_sides, bt)
        pnls_without, rets_without, cap_without = _simulate_grid_atr(cache, params_without_sides, bt)

        assert pnls_with == pnls_without
        assert rets_with == rets_without
        assert cap_with == cap_without

    def test_both_sides_defaults_to_long(self, make_indicator_cache):
        """sides=["long","short"] → direction=1 (LONG prioritaire)."""
        cache = _make_cache(make_indicator_cache)
        bt = _make_bt_config()

        params_both = {**_DEFAULT_PARAMS, "sides": ["long", "short"]}
        params_long = {**_DEFAULT_PARAMS, "sides": ["long"]}

        pnls_both, rets_both, cap_both = _simulate_grid_atr(cache, params_both, bt)
        pnls_long, rets_long, cap_long = _simulate_grid_atr(cache, params_long, bt)

        assert pnls_both == pnls_long
        assert rets_both == rets_long
        assert cap_both == cap_long

    def test_run_multi_backtest_short_5tuple(self, make_indicator_cache):
        """run_multi_backtest_from_cache avec sides=["short"] → 5-tuple valide."""
        cache = _make_cache(make_indicator_cache)
        bt = _make_bt_config()
        params = {**_DEFAULT_PARAMS, "sides": ["short"]}

        result = run_multi_backtest_from_cache("grid_atr", params, cache, bt)
        assert len(result) == 5
        assert result[0] == params  # params retournés
        assert isinstance(result[1], float)  # sharpe
        assert isinstance(result[4], int)  # n_trades

    def test_param_grids_yaml_has_sides(self):
        """param_grids.yaml contient sides dans grid_atr.default."""
        yaml_path = Path(__file__).parent.parent / "config" / "param_grids.yaml"
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        assert "grid_atr" in cfg
        assert "default" in cfg["grid_atr"]
        assert "sides" in cfg["grid_atr"]["default"]
        sides_values = cfg["grid_atr"]["default"]["sides"]
        assert ["long"] in sides_values
        assert ["short"] in sides_values
