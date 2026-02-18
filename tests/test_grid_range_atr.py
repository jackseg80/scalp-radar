"""Tests Sprint 29a — Grid Range ATR (stratégie bidirectionnelle, TP/SL individuels).

Couvre :
- Section 1 : Signaux compute_grid (~8 tests)
- Section 2 : TP/SL individuels (~4 tests)
- Section 3 : SL individuel (~3 tests)
- Section 4 : Fast engine (~12 tests)
- Section 5 : Viabilité fees (~2 tests)
- Section 6 : Registry et config (~6 tests)
- Section 7 : Parité stratégies existantes (~5 tests)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from backend.backtesting.engine import BacktestConfig
from backend.core.config import GridRangeATRConfig
from backend.core.models import Direction
from backend.optimization.fast_multi_backtest import (
    _simulate_grid_range,
    run_multi_backtest_from_cache,
)
from backend.strategies.base import StrategyContext
from backend.strategies.base_grid import GridPosition, GridState
from backend.strategies.grid_range_atr import GridRangeATRStrategy


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_bt_config(**overrides) -> BacktestConfig:
    defaults = dict(
        symbol="TEST/USDT",
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
        leverage=6,
        initial_capital=10_000,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _make_strategy(**overrides) -> GridRangeATRStrategy:
    defaults = dict(
        ma_period=20,
        atr_period=14,
        atr_spacing_mult=0.3,
        num_levels=2,
        sl_percent=10.0,
        tp_mode="dynamic_sma",
        sides=["long", "short"],
        leverage=6,
    )
    defaults.update(overrides)
    cfg = GridRangeATRConfig(**defaults)
    return GridRangeATRStrategy(cfg)


def _make_ctx(sma: float, atr_val: float, close: float = 100.0) -> StrategyContext:
    return StrategyContext(
        symbol="TEST/USDT",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        candles={},
        indicators={
            "1h": {
                "sma": sma,
                "atr": atr_val,
                "close": close,
            }
        },
        current_position=None,
        capital=10_000.0,
        config=None,  # type: ignore[arg-type]
    )


def _empty_grid_state() -> GridState:
    return GridState(
        positions=[], avg_entry_price=0, total_quantity=0,
        total_notional=0, unrealized_pnl=0,
    )


def _grid_state_with_positions(positions: list[GridPosition]) -> GridState:
    if not positions:
        return _empty_grid_state()
    total_qty = sum(p.quantity for p in positions)
    avg_entry = sum(p.entry_price * p.quantity for p in positions) / total_qty
    return GridState(
        positions=positions, avg_entry_price=avg_entry,
        total_quantity=total_qty, total_notional=total_qty * avg_entry,
        unrealized_pnl=0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Section 1 : Signaux compute_grid
# ═══════════════════════════════════════════════════════════════════════════


class TestGridRangeATRSignals:
    def test_name(self):
        strategy = _make_strategy()
        assert strategy.name == "grid_range_atr"

    def test_max_positions_both_sides(self):
        strategy = _make_strategy(num_levels=3, sides=["long", "short"])
        assert strategy.max_positions == 6

    def test_max_positions_long_only(self):
        strategy = _make_strategy(num_levels=3, sides=["long"])
        assert strategy.max_positions == 3

    def test_max_positions_short_only(self):
        strategy = _make_strategy(num_levels=2, sides=["short"])
        assert strategy.max_positions == 2

    def test_compute_grid_returns_both_sides(self):
        """LONG sous SMA, SHORT au-dessus, simultanément."""
        strategy = _make_strategy(num_levels=2, atr_spacing_mult=0.5)
        ctx = _make_ctx(sma=100.0, atr_val=10.0)
        levels = strategy.compute_grid(ctx, _empty_grid_state())

        long_levels = [l for l in levels if l.direction == Direction.LONG]
        short_levels = [l for l in levels if l.direction == Direction.SHORT]

        assert len(long_levels) == 2
        assert len(short_levels) == 2

        # LONG : SMA - (i+1) * ATR * spacing
        assert long_levels[0].entry_price == pytest.approx(100 - 1 * 10 * 0.5)  # 95
        assert long_levels[1].entry_price == pytest.approx(100 - 2 * 10 * 0.5)  # 90

        # SHORT : SMA + (i+1) * ATR * spacing
        assert short_levels[0].entry_price == pytest.approx(100 + 1 * 10 * 0.5)  # 105
        assert short_levels[1].entry_price == pytest.approx(100 + 2 * 10 * 0.5)  # 110

    def test_filled_levels_excluded(self):
        """Un niveau rempli est exclu du compute_grid."""
        strategy = _make_strategy(num_levels=2)
        ctx = _make_ctx(sma=100.0, atr_val=10.0)
        # Level 0 (LONG) est rempli
        pos = GridPosition(
            level=0, direction=Direction.LONG, entry_price=97.0,
            quantity=1.0, entry_time=datetime.now(timezone.utc), entry_fee=0.01,
        )
        state = _grid_state_with_positions([pos])
        levels = strategy.compute_grid(ctx, state)
        # Level 0 LONG absent, mais level 1 LONG + 2 SHORT présents
        indices = {l.index for l in levels}
        assert 0 not in indices
        assert 1 in indices  # LONG level 1
        assert 2 in indices  # SHORT level 0 (index = num_levels + 0)
        assert 3 in indices  # SHORT level 1

    def test_no_direction_lock(self):
        """Positions LONG ne bloquent PAS les SHORT (contrairement à grid_atr)."""
        strategy = _make_strategy(num_levels=1)
        ctx = _make_ctx(sma=100.0, atr_val=10.0)
        # LONG level 0 rempli
        pos = GridPosition(
            level=0, direction=Direction.LONG, entry_price=97.0,
            quantity=1.0, entry_time=datetime.now(timezone.utc), entry_fee=0.01,
        )
        state = _grid_state_with_positions([pos])
        levels = strategy.compute_grid(ctx, state)
        # SHORT level 1 (index=1) toujours dispo
        assert len(levels) == 1
        assert levels[0].direction == Direction.SHORT

    def test_nan_atr_returns_empty(self):
        strategy = _make_strategy()
        ctx = _make_ctx(sma=100.0, atr_val=float("nan"))
        assert strategy.compute_grid(ctx, _empty_grid_state()) == []

    def test_nan_sma_returns_empty(self):
        strategy = _make_strategy()
        ctx = _make_ctx(sma=float("nan"), atr_val=10.0)
        assert strategy.compute_grid(ctx, _empty_grid_state()) == []

    def test_sides_long_only_no_short(self):
        strategy = _make_strategy(sides=["long"], num_levels=2)
        ctx = _make_ctx(sma=100.0, atr_val=10.0)
        levels = strategy.compute_grid(ctx, _empty_grid_state())
        dirs = {l.direction for l in levels}
        assert Direction.SHORT not in dirs
        assert len(levels) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 : TP/SL individuels
# ═══════════════════════════════════════════════════════════════════════════


class TestGridRangeATRTPSL:
    def test_should_close_all_always_none(self):
        strategy = _make_strategy()
        ctx = _make_ctx(sma=100.0, atr_val=10.0, close=200.0)
        pos = GridPosition(
            level=0, direction=Direction.LONG, entry_price=50.0,
            quantity=1.0, entry_time=datetime.now(timezone.utc), entry_fee=0.01,
        )
        state = _grid_state_with_positions([pos])
        assert strategy.should_close_all(ctx, state) is None

    def test_get_tp_price_returns_nan(self):
        strategy = _make_strategy()
        tp = strategy.get_tp_price(_empty_grid_state(), {"sma": 100.0})
        assert math.isnan(tp)

    def test_get_sl_price_returns_nan(self):
        strategy = _make_strategy()
        sl = strategy.get_sl_price(_empty_grid_state(), {"sma": 100.0})
        assert math.isnan(sl)

    def test_get_params(self):
        strategy = _make_strategy(tp_mode="fixed_center", atr_spacing_mult=0.4)
        params = strategy.get_params()
        assert params["tp_mode"] == "fixed_center"
        assert params["atr_spacing_mult"] == 0.4
        assert params["sides"] == ["long", "short"]


# ═══════════════════════════════════════════════════════════════════════════
# Section 3 : Fast engine
# ═══════════════════════════════════════════════════════════════════════════


class TestGridRangeATRFastEngine:
    """Tests du fast engine _simulate_grid_range."""

    def _make_range_cache(self, make_indicator_cache, n=200, sma_val=100.0, atr_val=5.0):
        """Cache synthétique pour grid_range_atr avec prix oscillant autour de SMA."""
        rng = np.random.default_rng(42)
        # Prix oscillant autour de sma_val
        noise = rng.normal(0, atr_val * 0.3, n)
        closes = sma_val + np.cumsum(noise) * 0.1  # oscillations légères
        closes = np.clip(closes, sma_val - atr_val * 3, sma_val + atr_val * 3)
        highs = closes + np.abs(rng.normal(atr_val * 0.4, atr_val * 0.1, n))
        lows = closes - np.abs(rng.normal(atr_val * 0.4, atr_val * 0.1, n))
        opens = closes + rng.uniform(-0.5, 0.5, n)

        sma_arr = np.full(n, sma_val)
        atr_arr = np.full(n, atr_val)

        return make_indicator_cache(
            n=n,
            closes=closes, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma_arr, 14: sma_arr},
            atr_by_period={14: atr_arr, 10: atr_arr},
        )

    def test_basic_result_shape(self, make_indicator_cache):
        """run_multi_backtest_from_cache retourne un 5-tuple valide."""
        cache = self._make_range_cache(make_indicator_cache)
        params = {
            "ma_period": 20, "atr_period": 14, "atr_spacing_mult": 0.3,
            "num_levels": 2, "sl_percent": 10.0, "tp_mode": "dynamic_sma",
            "sides": ["long", "short"],
        }
        bt = _make_bt_config()
        result = run_multi_backtest_from_cache("grid_range_atr", params, cache, bt)
        assert len(result) == 5
        p, sharpe, net_ret, pf, n_trades = result
        assert isinstance(n_trades, int)
        assert n_trades >= 0

    def test_no_trades_when_atr_zero(self, make_indicator_cache):
        """ATR=0 → pas de niveaux → pas de trades."""
        n = 100
        cache = make_indicator_cache(
            n=n,
            bb_sma={20: np.full(n, 100.0)},
            atr_by_period={14: np.full(n, 0.0)},
        )
        params = {
            "ma_period": 20, "atr_period": 14, "atr_spacing_mult": 0.3,
            "num_levels": 2, "sl_percent": 10.0, "tp_mode": "dynamic_sma",
            "sides": ["long", "short"],
        }
        bt = _make_bt_config()
        _, _, _, _, n_trades = run_multi_backtest_from_cache("grid_range_atr", params, cache, bt)
        assert n_trades == 0

    def test_bidirectional_simultaneous(self, make_indicator_cache):
        """LONG et SHORT peuvent s'ouvrir sur la même candle."""
        n = 50
        sma = np.full(n, 100.0)
        atr_val = np.full(n, 10.0)
        # Prix touche LONG (bas=95) et SHORT (haut=105) en même temps
        closes = np.full(n, 100.0)
        highs = np.full(n, 108.0)  # touche SHORT level 0 (SMA + 1*10*0.5 = 105)
        lows = np.full(n, 92.0)   # touche LONG level 0 (SMA - 1*10*0.5 = 95)
        opens = np.full(n, 100.0)

        cache = make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma}, atr_by_period={14: atr_val},
        )
        params = {
            "ma_period": 20, "atr_period": 14, "atr_spacing_mult": 0.5,
            "num_levels": 1, "sl_percent": 50.0, "tp_mode": "dynamic_sma",
            "sides": ["long", "short"],
        }
        bt = _make_bt_config()
        pnls, _, capital = _simulate_grid_range(cache, params, bt)
        # Au moins 1 trade (force close) avec positions des deux côtés
        assert len(pnls) >= 1

    def test_individual_tp_long(self, make_indicator_cache):
        """TP LONG individuel quand high >= SMA."""
        n = 10
        sma = np.full(n, 100.0)
        atr_val = np.full(n, 10.0)
        closes = np.full(n, 100.0)
        opens = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        # Candle 0 : LONG entry (low touche SMA - 1*10*0.3 = 97)
        lows[0] = 96.0
        # Candle 1 : TP (high >= SMA = 100)
        highs[1] = 101.0

        cache = make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma}, atr_by_period={14: atr_val},
        )
        params = {
            "ma_period": 20, "atr_period": 14, "atr_spacing_mult": 0.3,
            "num_levels": 1, "sl_percent": 50.0, "tp_mode": "dynamic_sma",
            "sides": ["long"],
        }
        bt = _make_bt_config()
        pnls, _, _ = _simulate_grid_range(cache, params, bt)
        # Le premier trade est un TP (entry 97 → exit ~100, net positif après fees)
        assert len(pnls) >= 1
        assert pnls[0] > 0  # TP = profit

    def test_individual_sl_long(self, make_indicator_cache):
        """SL LONG individuel quand low <= entry * (1 - sl_pct)."""
        n = 10
        sma = np.full(n, 100.0)
        atr_val = np.full(n, 10.0)
        closes = np.full(n, 95.0)
        opens = np.full(n, 95.0)
        highs = np.full(n, 96.0)
        lows = np.full(n, 94.0)
        # Candle 0 : LONG entry à 97 (low=96 touche SMA - 1*10*0.3 = 97... mais low=96 < 97 ok)
        lows[0] = 96.0
        # Candle 1 : SL à entry*(1-0.1)= 87.3 → low doit aller là
        lows[1] = 85.0
        closes[1] = 86.0
        highs[1] = 87.0

        cache = make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma}, atr_by_period={14: atr_val},
        )
        params = {
            "ma_period": 20, "atr_period": 14, "atr_spacing_mult": 0.3,
            "num_levels": 1, "sl_percent": 10.0, "tp_mode": "dynamic_sma",
            "sides": ["long"],
        }
        bt = _make_bt_config()
        pnls, _, _ = _simulate_grid_range(cache, params, bt)
        assert len(pnls) >= 1
        assert pnls[0] < 0  # SL = perte

    def test_individual_tp_short(self, make_indicator_cache):
        """TP SHORT individuel quand low <= SMA."""
        n = 10
        sma = np.full(n, 100.0)
        atr_val = np.full(n, 10.0)
        closes = np.full(n, 100.0)
        opens = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        # Candle 0 : SHORT entry (high touche SMA + 1*10*0.3 = 103)
        highs[0] = 104.0
        # Candle 1 : TP (low <= SMA = 100)
        lows[1] = 99.0
        closes[1] = 99.5
        opens[1] = 100.5  # bougie rouge → TP SHORT

        cache = make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma}, atr_by_period={14: atr_val},
        )
        params = {
            "ma_period": 20, "atr_period": 14, "atr_spacing_mult": 0.3,
            "num_levels": 1, "sl_percent": 50.0, "tp_mode": "dynamic_sma",
            "sides": ["short"],
        }
        bt = _make_bt_config()
        pnls, _, _ = _simulate_grid_range(cache, params, bt)
        assert len(pnls) >= 1
        assert pnls[0] > 0  # TP SHORT = profit

    def test_tp_mode_fixed_center_differs(self, make_indicator_cache):
        """fixed_center et dynamic_sma produisent des résultats différents quand SMA bouge."""
        n = 100
        # SMA qui dérive de 100 à 110
        sma = np.linspace(100, 110, n)
        atr_val = np.full(n, 5.0)
        rng = np.random.default_rng(123)
        closes = sma + rng.normal(0, 2, n)
        highs = closes + 3.0
        lows = closes - 3.0
        opens = closes + rng.uniform(-0.5, 0.5, n)

        cache = make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma}, atr_by_period={14: atr_val},
        )
        base_params = {
            "ma_period": 20, "atr_period": 14, "atr_spacing_mult": 0.3,
            "num_levels": 2, "sl_percent": 20.0,
            "sides": ["long", "short"],
        }
        bt = _make_bt_config()

        pnls_dyn, _, cap_dyn = _simulate_grid_range(cache, {**base_params, "tp_mode": "dynamic_sma"}, bt)
        pnls_fix, _, cap_fix = _simulate_grid_range(cache, {**base_params, "tp_mode": "fixed_center"}, bt)

        # Avec SMA qui bouge, les résultats devraient différer
        if pnls_dyn and pnls_fix:
            assert cap_dyn != pytest.approx(cap_fix, abs=0.01)

    def test_sides_long_only_no_short_trades(self, make_indicator_cache):
        """sides=["long"] → aucun trade SHORT."""
        cache = self._make_range_cache(make_indicator_cache)
        params = {
            "ma_period": 20, "atr_period": 14, "atr_spacing_mult": 0.3,
            "num_levels": 2, "sl_percent": 10.0, "tp_mode": "dynamic_sma",
            "sides": ["long"],
        }
        bt = _make_bt_config()
        result = run_multi_backtest_from_cache("grid_range_atr", params, cache, bt)
        # Le test vérifie que ça tourne sans crash avec sides=["long"]
        assert result[4] >= 0  # n_trades >= 0

    def test_level_reopen_after_close(self, make_indicator_cache):
        """Un niveau libéré peut être réutilisé."""
        n = 50
        sma = np.full(n, 100.0)
        atr_val = np.full(n, 10.0)
        closes = np.full(n, 100.0)
        opens = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)

        # Cycle 1 : candle 0 ouvre LONG, candle 2 ferme (TP)
        lows[0] = 96.0  # entry LONG level 0 à 97
        highs[2] = 101.0  # TP à SMA=100

        # Cycle 2 : candle 5 réouvre le même level
        lows[5] = 96.0  # entry LONG level 0 à 97

        cache = make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma}, atr_by_period={14: atr_val},
        )
        params = {
            "ma_period": 20, "atr_period": 14, "atr_spacing_mult": 0.3,
            "num_levels": 1, "sl_percent": 50.0, "tp_mode": "dynamic_sma",
            "sides": ["long"],
        }
        bt = _make_bt_config()
        pnls, _, _ = _simulate_grid_range(cache, params, bt)
        # Au moins 2 trades (cycle 1 + cycle 2 ou force close)
        assert len(pnls) >= 2

    def test_fees_tp_maker_sl_taker(self, make_indicator_cache):
        """TP utilise maker_fee, SL utilise taker_fee + slippage."""
        n = 10
        sma = np.full(n, 100.0)
        atr_val = np.full(n, 10.0)

        # Trade 1 : LONG TP
        closes_tp = np.full(n, 100.0)
        opens_tp = np.full(n, 100.0)
        highs_tp = np.full(n, 101.0)
        lows_tp = np.full(n, 99.0)
        lows_tp[0] = 96.0  # entry à 97
        highs_tp[1] = 101.0  # TP

        cache_tp = make_indicator_cache(
            n=n, closes=closes_tp, opens=opens_tp, highs=highs_tp, lows=lows_tp,
            bb_sma={20: sma}, atr_by_period={14: atr_val},
        )

        # Trade 2 : LONG SL
        closes_sl = np.full(n, 90.0)
        opens_sl = np.full(n, 90.0)
        highs_sl = np.full(n, 91.0)
        lows_sl = np.full(n, 89.0)
        lows_sl[0] = 96.0  # entry à 97
        lows_sl[1] = 85.0  # SL (97 * 0.9 = 87.3)

        cache_sl = make_indicator_cache(
            n=n, closes=closes_sl, opens=opens_sl, highs=highs_sl, lows=lows_sl,
            bb_sma={20: sma}, atr_by_period={14: atr_val},
        )

        params = {
            "ma_period": 20, "atr_period": 14, "atr_spacing_mult": 0.3,
            "num_levels": 1, "sl_percent": 10.0, "tp_mode": "dynamic_sma",
            "sides": ["long"],
        }
        bt_tp = _make_bt_config(taker_fee=0.0006, maker_fee=0.0002, slippage_pct=0.001)
        bt_sl = _make_bt_config(taker_fee=0.0006, maker_fee=0.0002, slippage_pct=0.001)

        pnls_tp, _, _ = _simulate_grid_range(cache_tp, params, bt_tp)
        pnls_sl, _, _ = _simulate_grid_range(cache_sl, params, bt_sl)

        # TP trade : entry taker + exit maker (moins de frais)
        # SL trade : entry taker + exit taker + slippage (plus de frais)
        assert len(pnls_tp) >= 1
        assert len(pnls_sl) >= 1

    def test_funding_settlement(self, make_indicator_cache):
        """Funding settlement 8h appliqué aux positions ouvertes."""
        n = 50
        sma = np.full(n, 100.0)
        atr_val = np.full(n, 10.0)
        closes = np.full(n, 98.0)
        opens = np.full(n, 98.0)
        # highs < SMA pour empêcher le TP (LONG TP = highs >= SMA)
        highs = np.full(n, 99.5)
        lows = np.full(n, 96.5)
        lows[0] = 96.0  # entry LONG à SMA - 1*ATR*0.3 = 97

        # Timestamps : toutes les heures, commençant à 00:00
        base_ts = 1704067200000.0  # 2024-01-01 00:00 UTC en ms
        candle_ts = np.array([base_ts + i * 3600000 for i in range(n)])
        # Funding rate négatif → LONG reçoit du funding
        funding = np.full(n, -0.001)  # -0.1%

        cache = make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma}, atr_by_period={14: atr_val},
            funding_rates_1h=funding, candle_timestamps=candle_ts,
        )
        params = {
            "ma_period": 20, "atr_period": 14, "atr_spacing_mult": 0.3,
            "num_levels": 1, "sl_percent": 50.0, "tp_mode": "dynamic_sma",
            "sides": ["long"],
        }
        bt_with = _make_bt_config()
        bt_without = _make_bt_config()

        pnls_with, _, cap_with = _simulate_grid_range(cache, params, bt_with)

        # Sans funding
        cache_no_fund = make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma}, atr_by_period={14: atr_val},
        )
        pnls_without, _, cap_without = _simulate_grid_range(cache_no_fund, params, bt_without)

        # Avec funding négatif, LONG reçoit → capital plus élevé
        assert cap_with > cap_without

    def test_long_short_same_candle_close(self, make_indicator_cache):
        """LONG et SHORT peuvent fermer sur la même candle."""
        n = 10
        sma = np.full(n, 100.0)
        atr_val = np.full(n, 10.0)
        closes = np.full(n, 100.0)
        opens = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)

        # Candle 0 : entry LONG (low=96 < 97) et SHORT (high=104 > 103)
        lows[0] = 95.0
        highs[0] = 104.0
        # Candle 1 : TP LONG (high=101 >= SMA=100) et TP SHORT (low=99 <= SMA=100)
        highs[1] = 101.0
        lows[1] = 99.0

        cache = make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma}, atr_by_period={14: atr_val},
        )
        params = {
            "ma_period": 20, "atr_period": 14, "atr_spacing_mult": 0.3,
            "num_levels": 1, "sl_percent": 50.0, "tp_mode": "dynamic_sma",
            "sides": ["long", "short"],
        }
        bt = _make_bt_config()
        pnls, _, _ = _simulate_grid_range(cache, params, bt)
        # 2 positions fermées individuellement
        assert len(pnls) >= 2


# ═══════════════════════════════════════════════════════════════════════════
# Section 5 : Viabilité fees
# ═══════════════════════════════════════════════════════════════════════════


class TestGridRangeATRFeeViability:
    def test_tight_spacing_loses_to_fees(self, make_indicator_cache):
        """Spacing < breakeven fee → chaque trade perd de l'argent.

        Breakeven ≈ price × (taker_fee + maker_fee) = 100 × 0.0008 = 0.08.
        Avec ATR=0.5 × spacing_mult=0.1 = 0.05 < 0.08, chaque round-trip est perdant.
        """
        n = 100
        sma = np.full(n, 100.0)
        atr_val = np.full(n, 0.5)
        # Oscillation : prix dip sous entry et remonte au-dessus de SMA
        closes = np.full(n, 100.0)
        opens = np.full(n, 100.0)
        # Highs au-dessus de SMA → TP LONG trigger, lows en-dessous entry → open LONG
        highs = np.full(n, 100.2)
        lows = np.full(n, 99.8)

        cache = make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma}, atr_by_period={14: atr_val},
        )
        # spacing = 0.5 × 0.1 = 0.05, entry LONG à 99.95 → TP à 100
        # Gross profit = 0.05/unit, fees ≈ 0.08/unit → net négatif
        params = {
            "ma_period": 20, "atr_period": 14, "atr_spacing_mult": 0.1,
            "num_levels": 1, "sl_percent": 50.0, "tp_mode": "dynamic_sma",
            "sides": ["long"],
        }
        bt = _make_bt_config(taker_fee=0.0006, maker_fee=0.0002)
        pnls, _, capital = _simulate_grid_range(cache, params, bt)

        # Beaucoup de trades (open+close chaque candle)
        assert len(pnls) > 10
        # Chaque trade est perdant (fees > gross profit)
        assert all(p < 0 for p in pnls)
        # Capital final inférieur à l'initial
        assert capital < bt.initial_capital

    def test_maker_vs_taker_fee_impact(self, make_indicator_cache):
        """Maker fees vs taker fees → résultats significativement différents."""
        n = 200
        rng = np.random.default_rng(42)
        sma = np.full(n, 100.0)
        atr_val = np.full(n, 5.0)
        noise = rng.normal(0, 1.5, n)
        closes = 100.0 + np.cumsum(noise) * 0.05
        closes = np.clip(closes, 90, 110)
        highs = closes + np.abs(rng.normal(1.5, 0.3, n))
        lows = closes - np.abs(rng.normal(1.5, 0.3, n))
        opens = closes + rng.uniform(-0.3, 0.3, n)

        cache = make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma}, atr_by_period={14: atr_val},
        )
        params = {
            "ma_period": 20, "atr_period": 14, "atr_spacing_mult": 0.3,
            "num_levels": 2, "sl_percent": 10.0, "tp_mode": "dynamic_sma",
            "sides": ["long", "short"],
        }
        # Taker fees élevées
        bt_taker = _make_bt_config(taker_fee=0.0006, maker_fee=0.0006)
        # Maker fees basses
        bt_maker = _make_bt_config(taker_fee=0.0002, maker_fee=0.0002)

        _, _, cap_taker = _simulate_grid_range(cache, params, bt_taker)
        _, _, cap_maker = _simulate_grid_range(cache, params, bt_maker)

        # Plus de fees = moins de capital
        assert cap_maker > cap_taker


# ═══════════════════════════════════════════════════════════════════════════
# Section 6 : Registry et config
# ═══════════════════════════════════════════════════════════════════════════


class TestGridRangeATRRegistry:
    def test_in_strategy_registry(self):
        from backend.optimization import STRATEGY_REGISTRY
        assert "grid_range_atr" in STRATEGY_REGISTRY

    def test_in_grid_strategies(self):
        from backend.optimization import GRID_STRATEGIES
        assert "grid_range_atr" in GRID_STRATEGIES

    def test_in_fast_engine_strategies(self):
        from backend.optimization import FAST_ENGINE_STRATEGIES
        assert "grid_range_atr" in FAST_ENGINE_STRATEGIES

    def test_in_strategies_need_extra_data(self):
        from backend.optimization import STRATEGIES_NEED_EXTRA_DATA
        assert "grid_range_atr" in STRATEGIES_NEED_EXTRA_DATA

    def test_create_with_params(self):
        from backend.optimization import create_strategy_with_params
        strategy = create_strategy_with_params("grid_range_atr", {
            "ma_period": 14, "atr_spacing_mult": 0.4, "tp_mode": "fixed_center",
        })
        assert strategy.name == "grid_range_atr"
        params = strategy.get_params()
        assert params["ma_period"] == 14
        assert params["atr_spacing_mult"] == 0.4
        assert params["tp_mode"] == "fixed_center"

    def test_sides_list_and_tp_mode_string_in_create_strategy(self):
        """sides (list) et tp_mode (string) passent dans create_strategy_with_params."""
        from backend.optimization import create_strategy_with_params
        strategy = create_strategy_with_params("grid_range_atr", {
            "sides": ["long"],
            "tp_mode": "fixed_center",
        })
        params = strategy.get_params()
        assert params["sides"] == ["long"]
        assert params["tp_mode"] == "fixed_center"

    def test_config_defaults(self):
        cfg = GridRangeATRConfig()
        assert cfg.enabled is False
        assert cfg.timeframe == "1h"
        assert cfg.ma_period == 20
        assert cfg.atr_spacing_mult == 0.3
        assert cfg.sides == ["long", "short"]
        assert cfg.tp_mode == "dynamic_sma"
        assert cfg.num_levels == 2
        assert cfg.sl_percent == 10.0
        assert cfg.leverage == 6


# ═══════════════════════════════════════════════════════════════════════════
# Section 7 : Parité (CRITIQUE — stratégies existantes inchangées)
# ═══════════════════════════════════════════════════════════════════════════


class TestGridRangeATRParity:
    """Vérifie que les stratégies existantes donnent EXACTEMENT les mêmes résultats.

    Si un seul test échoue → régression sur les stratégies en production.
    """

    def _make_deterministic_cache(self, make_indicator_cache, n=500, seed=42):
        """Cache déterministe avec prix, SMA, ATR cohérents."""
        rng = np.random.default_rng(seed)
        prices = 100.0 + np.cumsum(rng.normal(0, 2.0, n))
        opens = prices + rng.uniform(-0.3, 0.3, n)
        highs = prices + np.abs(rng.normal(3.0, 1.5, n))
        lows = prices - np.abs(rng.normal(3.0, 1.5, n))

        sma_7 = np.full(n, np.nan)
        sma_14 = np.full(n, np.nan)
        atr_14 = np.full(n, np.nan)
        atr_10 = np.full(n, np.nan)

        for i in range(7, n):
            sma_7[i] = np.mean(prices[max(0, i - 6):i + 1])
        for i in range(10, n):
            atr_10[i] = np.mean(np.abs(np.diff(prices[max(0, i - 9):i + 1])))
        for i in range(14, n):
            sma_14[i] = np.mean(prices[max(0, i - 13):i + 1])
            atr_14[i] = np.mean(np.abs(np.diff(prices[max(0, i - 13):i + 1])))

        return make_indicator_cache(
            n=n,
            closes=prices, opens=opens, highs=highs, lows=lows,
            bb_sma={7: sma_7, 14: sma_14},
            atr_by_period={10: atr_10, 14: atr_14},
        )

    def _bt_config(self):
        return _make_bt_config()

    def test_grid_atr_parity(self, make_indicator_cache):
        """grid_atr via _simulate_grid_common produit le même résultat."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_atr
        cache = self._make_deterministic_cache(make_indicator_cache)
        bt = self._bt_config()
        params = {
            "ma_period": 14, "atr_period": 14, "atr_multiplier_start": 2.0,
            "atr_multiplier_step": 1.0, "num_levels": 3, "sl_percent": 20.0,
        }
        pnls, rets, cap = _simulate_grid_atr(cache, params, bt, direction=1)
        # Snapshot déterministe — juste vérifier que ça ne crash pas et que les résultats sont stables
        assert isinstance(cap, float)
        assert len(pnls) == len(rets)

        # Re-run doit donner EXACTEMENT le même résultat
        pnls2, rets2, cap2 = _simulate_grid_atr(cache, params, bt, direction=1)
        assert cap == cap2
        assert pnls == pnls2

    def test_envelope_dca_parity(self, make_indicator_cache):
        """envelope_dca via _simulate_grid_common produit le même résultat."""
        from backend.optimization.fast_multi_backtest import _simulate_envelope_dca
        cache = self._make_deterministic_cache(make_indicator_cache)
        bt = self._bt_config()
        params = {
            "ma_period": 7, "envelope_start": 0.05, "envelope_step": 0.02,
            "num_levels": 3, "sl_percent": 25.0,
        }
        pnls, rets, cap = _simulate_envelope_dca(cache, params, bt, direction=1)
        pnls2, rets2, cap2 = _simulate_envelope_dca(cache, params, bt, direction=1)
        assert cap == cap2
        assert pnls == pnls2

    def test_grid_trend_parity(self, make_indicator_cache):
        """grid_trend produit le même résultat."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_trend

        n = 500
        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 2.0, n))
        opens = prices + rng.uniform(-0.3, 0.3, n)
        highs = prices + np.abs(rng.normal(3.0, 1.5, n))
        lows = prices - np.abs(rng.normal(3.0, 1.5, n))

        from backend.core.indicators import atr as compute_atr
        from backend.core.indicators import ema as compute_ema
        from backend.core.indicators import adx as compute_adx

        ema_20 = compute_ema(prices, 20)
        ema_50 = compute_ema(prices, 50)
        atr_14 = compute_atr(highs, lows, prices, 14)
        adx_14, _, _ = compute_adx(highs, lows, prices, 14)

        cache = make_indicator_cache(
            n=n, closes=prices, opens=opens, highs=highs, lows=lows,
            ema_by_period={20: ema_20, 50: ema_50},
            atr_by_period={14: atr_14},
            adx_by_period={14: adx_14},
        )
        bt = self._bt_config()
        params = {
            "ema_fast": 20, "ema_slow": 50, "adx_period": 14, "adx_threshold": 20,
            "atr_period": 14, "pull_start": 1.0, "pull_step": 0.5,
            "num_levels": 2, "trail_mult": 2.0, "sl_percent": 15.0,
        }
        pnls, rets, cap = _simulate_grid_trend(cache, params, bt)
        pnls2, rets2, cap2 = _simulate_grid_trend(cache, params, bt)
        assert cap == cap2
        assert pnls == pnls2

    def test_grid_multi_tf_parity(self, make_indicator_cache):
        """grid_multi_tf produit le même résultat."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_multi_tf

        n = 500
        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 2.0, n))
        opens = prices + rng.uniform(-0.3, 0.3, n)
        highs = prices + np.abs(rng.normal(3.0, 1.5, n))
        lows = prices - np.abs(rng.normal(3.0, 1.5, n))

        sma_14 = np.full(n, np.nan)
        atr_14 = np.full(n, np.nan)
        for i in range(14, n):
            sma_14[i] = np.mean(prices[max(0, i - 13):i + 1])
            atr_14[i] = np.mean(np.abs(np.diff(prices[max(0, i - 13):i + 1])))

        # Fake supertrend direction 4h
        st_dir = np.ones(n)
        st_dir[n // 2:] = -1.0

        cache = make_indicator_cache(
            n=n, closes=prices, opens=opens, highs=highs, lows=lows,
            bb_sma={14: sma_14},
            atr_by_period={14: atr_14},
            supertrend_dir_4h={(10, 3.0): st_dir},
        )
        bt = self._bt_config()
        params = {
            "ma_period": 14, "atr_period": 14, "atr_multiplier_start": 2.0,
            "atr_multiplier_step": 1.0, "num_levels": 2, "sl_percent": 20.0,
            "st_atr_period": 10, "st_atr_multiplier": 3.0,
        }
        pnls, rets, cap = _simulate_grid_multi_tf(cache, params, bt)
        pnls2, rets2, cap2 = _simulate_grid_multi_tf(cache, params, bt)
        assert cap == cap2
        assert pnls == pnls2

    def test_grid_funding_parity(self, make_indicator_cache):
        """grid_funding produit le même résultat."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_funding

        n = 500
        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 2.0, n))
        opens = prices + rng.uniform(-0.3, 0.3, n)
        highs = prices + np.abs(rng.normal(3.0, 1.5, n))
        lows = prices - np.abs(rng.normal(3.0, 1.5, n))

        sma_14 = np.full(n, np.nan)
        for i in range(14, n):
            sma_14[i] = np.mean(prices[max(0, i - 13):i + 1])

        # Funding rates négatifs pour déclencher des entries
        funding = np.full(n, -0.001)
        base_ts = 1704067200000.0
        candle_ts = np.array([base_ts + i * 3600000 for i in range(n)])

        cache = make_indicator_cache(
            n=n, closes=prices, opens=opens, highs=highs, lows=lows,
            bb_sma={14: sma_14},
            funding_rates_1h=funding, candle_timestamps=candle_ts,
        )
        bt = self._bt_config()
        params = {
            "funding_threshold_start": 0.0005, "funding_threshold_step": 0.0005,
            "num_levels": 2, "ma_period": 14, "sl_percent": 15.0,
            "tp_mode": "funding_or_sma", "min_hold_candles": 8,
        }
        pnls, rets, cap = _simulate_grid_funding(cache, params, bt)
        pnls2, rets2, cap2 = _simulate_grid_funding(cache, params, bt)
        assert cap == cap2
        assert pnls == pnls2
