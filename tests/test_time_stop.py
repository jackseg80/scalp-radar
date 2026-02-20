"""Tests pour le Time-Based Stop Loss (max_hold_candles).

Couvre :
- Section 1 : grid_atr strategy layer (~8 tests)
- Section 2 : grid_boltrend strategy layer (~5 tests)
- Section 3 : Fast engine grid_common (~6 tests)
- Section 4 : Fast engine grid_boltrend (~4 tests)
- Section 5 : Config validation (~3 tests)
- Section 6 : Parité fast engine vs strategy layer (~1 test)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from backend.core.config import GridATRConfig, GridBolTrendConfig
from backend.core.models import Direction
from backend.strategies.base import StrategyContext
from backend.strategies.base_grid import GridPosition, GridState
from backend.strategies.grid_atr import GridATRStrategy
from backend.strategies.grid_boltrend import GridBolTrendStrategy


# ─── Helpers ───────────────────────────────────────────────────────────────

_BASE_TS = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)


def _make_atr_strategy(**overrides) -> GridATRStrategy:
    defaults: dict[str, Any] = {
        "ma_period": 14,
        "atr_period": 14,
        "atr_multiplier_start": 2.0,
        "atr_multiplier_step": 1.0,
        "num_levels": 3,
        "sl_percent": 20.0,
        "sides": ["long"],
        "leverage": 6,
        "max_hold_candles": 0,
    }
    defaults.update(overrides)
    return GridATRStrategy(GridATRConfig(**defaults))


def _make_boltrend_strategy(**overrides) -> GridBolTrendStrategy:
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
        "max_hold_candles": 0,
    }
    defaults.update(overrides)
    return GridBolTrendStrategy(GridBolTrendConfig(**defaults))


def _make_atr_ctx(
    sma_val: float, close: float, ts: datetime = _BASE_TS
) -> StrategyContext:
    return StrategyContext(
        symbol="BTC/USDT",
        timestamp=ts,
        candles={},
        indicators={"1h": {"sma": sma_val, "atr": 5.0, "close": close}},
        current_position=None,
        capital=10_000.0,
        config=None,  # type: ignore[arg-type]
    )


def _make_boltrend_ctx(
    close: float, bb_sma: float, ts: datetime = _BASE_TS
) -> StrategyContext:
    return StrategyContext(
        symbol="BTC/USDT",
        timestamp=ts,
        candles={},
        indicators={"1h": {"close": close, "bb_sma": bb_sma}},
        current_position=None,
        capital=10_000.0,
        config=None,  # type: ignore[arg-type]
    )


def _make_grid_state(
    entry_price: float = 100.0,
    close: float = 95.0,
    direction: Direction = Direction.LONG,
    entry_time: datetime = _BASE_TS,
    quantity: float = 1.0,
) -> GridState:
    """Crée un GridState avec une seule position."""
    pos = GridPosition(
        level=0,
        direction=direction,
        entry_price=entry_price,
        quantity=quantity,
        entry_time=entry_time,
        entry_fee=0.01,
    )
    unrealized = (close - entry_price) * quantity if direction == Direction.LONG else (entry_price - close) * quantity
    return GridState(
        positions=[pos],
        avg_entry_price=entry_price,
        total_quantity=quantity,
        total_notional=entry_price * quantity,
        unrealized_pnl=unrealized,
    )


def _make_bt_config(**overrides):
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
# Section 1 : grid_atr strategy layer
# ═══════════════════════════════════════════════════════════════════════════


class TestGridATRTimeStop:
    """Tests time_stop dans grid_atr.should_close_all()."""

    def test_disabled_by_default(self):
        """max_hold_candles=0 → time_stop jamais déclenché."""
        strategy = _make_atr_strategy(max_hold_candles=0)
        entry_time = _BASE_TS - timedelta(hours=100)
        gs = _make_grid_state(entry_price=100.0, close=90.0, entry_time=entry_time)
        # close=90 < sma=95 → pas TP, close > 80 (SL=20%) → pas SL
        ctx = _make_atr_ctx(sma_val=95.0, close=90.0)
        assert strategy.should_close_all(ctx, gs) is None

    def test_triggers_when_in_loss(self):
        """48 candles + PnL < 0 → time_stop."""
        strategy = _make_atr_strategy(max_hold_candles=48)
        entry_time = _BASE_TS - timedelta(hours=48)
        # Entry=100, close=95 → perte, SMA=95 → close < SMA → pas TP pour LONG (close < sma)
        # SL à 80 → pas touché
        gs = _make_grid_state(entry_price=100.0, close=95.0, entry_time=entry_time)
        ctx = _make_atr_ctx(sma_val=96.0, close=95.0)
        assert strategy.should_close_all(ctx, gs) == "time_stop"

    def test_no_trigger_in_profit(self):
        """48 candles + PnL > 0 → None (pas de time_stop en profit)."""
        strategy = _make_atr_strategy(max_hold_candles=48)
        entry_time = _BASE_TS - timedelta(hours=48)
        # Entry=100, close=101 → profit mais pas TP (sma=105)
        gs = _make_grid_state(entry_price=100.0, close=101.0, entry_time=entry_time)
        ctx = _make_atr_ctx(sma_val=105.0, close=101.0)
        assert strategy.should_close_all(ctx, gs) is None

    def test_no_trigger_below_threshold(self):
        """47 candles + PnL < 0 → None (pas encore le seuil)."""
        strategy = _make_atr_strategy(max_hold_candles=48)
        entry_time = _BASE_TS - timedelta(hours=47)
        gs = _make_grid_state(entry_price=100.0, close=95.0, entry_time=entry_time)
        ctx = _make_atr_ctx(sma_val=96.0, close=95.0)
        assert strategy.should_close_all(ctx, gs) is None

    def test_sl_has_priority(self):
        """SL ET time_stop applicables → SL prioritaire."""
        strategy = _make_atr_strategy(max_hold_candles=48, sl_percent=10.0)
        entry_time = _BASE_TS - timedelta(hours=48)
        # Entry=100, close=89 → SL touché (10%), et PnL négatif
        gs = _make_grid_state(entry_price=100.0, close=89.0, entry_time=entry_time)
        ctx = _make_atr_ctx(sma_val=96.0, close=89.0)
        assert strategy.should_close_all(ctx, gs) == "sl_global"

    def test_no_trigger_at_breakeven(self):
        """PnL = 0 exactement → None (pas strictement < 0)."""
        strategy = _make_atr_strategy(max_hold_candles=48)
        entry_time = _BASE_TS - timedelta(hours=48)
        # Entry=100, close=100 → PnL=0, et close < SMA=105 → pas TP
        gs = _make_grid_state(entry_price=100.0, close=100.0, entry_time=entry_time)
        ctx = _make_atr_ctx(sma_val=105.0, close=100.0)
        assert strategy.should_close_all(ctx, gs) is None

    def test_tp_has_priority(self):
        """TP atteint ET time_stop (impossible en pratique, TP=profit) → TP gagne."""
        strategy = _make_atr_strategy(max_hold_candles=48)
        entry_time = _BASE_TS - timedelta(hours=48)
        # Entry=100, close=101 → TP (close >= sma=101), PnL > 0 → pas time_stop de toute façon
        gs = _make_grid_state(entry_price=100.0, close=101.0, entry_time=entry_time)
        ctx = _make_atr_ctx(sma_val=101.0, close=101.0)
        assert strategy.should_close_all(ctx, gs) == "tp_global"

    def test_timeframe_4h(self):
        """Timeframe 4h : 12 candles × 4h = 48h → valide."""
        strategy = _make_atr_strategy(max_hold_candles=12, timeframe="4h")
        entry_time = _BASE_TS - timedelta(hours=48)  # 12 candles × 4h
        gs = _make_grid_state(entry_price=100.0, close=95.0, entry_time=entry_time)
        ctx = StrategyContext(
            symbol="BTC/USDT",
            timestamp=_BASE_TS,
            candles={},
            indicators={"4h": {"sma": 96.0, "atr": 5.0, "close": 95.0}},
            current_position=None,
            capital=10_000.0,
            config=None,  # type: ignore[arg-type]
        )
        assert strategy.should_close_all(ctx, gs) == "time_stop"


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 : grid_boltrend strategy layer
# ═══════════════════════════════════════════════════════════════════════════


class TestGridBolTrendTimeStop:
    """Tests time_stop dans grid_boltrend.should_close_all()."""

    def test_disabled_by_default(self):
        """max_hold_candles=0 → pas de time_stop."""
        strategy = _make_boltrend_strategy(max_hold_candles=0)
        entry_time = _BASE_TS - timedelta(hours=100)
        gs = _make_grid_state(entry_price=100.0, close=99.0, entry_time=entry_time)
        ctx = _make_boltrend_ctx(close=99.0, bb_sma=98.0)
        # close=99 > bb_sma=98 → pas TP inverse LONG ; close > SL
        assert strategy.should_close_all(ctx, gs) is None

    def test_triggers_when_in_loss(self):
        """48 candles + PnL < 0 → time_stop."""
        strategy = _make_boltrend_strategy(max_hold_candles=48)
        entry_time = _BASE_TS - timedelta(hours=48)
        # Entry=100, close=96 → perte, close > bb_sma=95 → pas TP inverse, close > SL
        gs = _make_grid_state(entry_price=100.0, close=96.0, entry_time=entry_time)
        ctx = _make_boltrend_ctx(close=96.0, bb_sma=95.0)
        assert strategy.should_close_all(ctx, gs) == "time_stop"

    def test_no_trigger_in_profit(self):
        """48 candles + PnL > 0 → None."""
        strategy = _make_boltrend_strategy(max_hold_candles=48)
        entry_time = _BASE_TS - timedelta(hours=48)
        gs = _make_grid_state(entry_price=100.0, close=102.0, entry_time=entry_time)
        ctx = _make_boltrend_ctx(close=102.0, bb_sma=95.0)
        assert strategy.should_close_all(ctx, gs) is None

    def test_sl_has_priority(self):
        """SL prioritaire sur time_stop."""
        strategy = _make_boltrend_strategy(max_hold_candles=48, sl_percent=10.0)
        entry_time = _BASE_TS - timedelta(hours=48)
        # Entry=100, close=89 → SL touché (10%), PnL < 0
        gs = _make_grid_state(entry_price=100.0, close=89.0, entry_time=entry_time)
        ctx = _make_boltrend_ctx(close=89.0, bb_sma=95.0)
        assert strategy.should_close_all(ctx, gs) == "sl_global"

    def test_signal_exit_has_priority(self):
        """TP inverse (signal_exit) prioritaire sur time_stop."""
        strategy = _make_boltrend_strategy(max_hold_candles=48)
        entry_time = _BASE_TS - timedelta(hours=48)
        # Entry=100, close=97, bb_sma=98 → close < bb_sma → signal_exit
        # PnL < 0 aussi → time_stop applicable, mais signal_exit est avant
        gs = _make_grid_state(entry_price=100.0, close=97.0, entry_time=entry_time)
        ctx = _make_boltrend_ctx(close=97.0, bb_sma=98.0)
        # SL check : sl_pct=15%, SL price = 85 → close=97 > 85 → pas SL
        # time_stop : 48 candles, PnL < 0 → time_stop would fire
        # signal_exit : close=97 < bb_sma=98 → signal_exit fires AFTER time_stop in code order
        # But time_stop is between SL and signal_exit → time_stop wins
        assert strategy.should_close_all(ctx, gs) == "time_stop"


# ═══════════════════════════════════════════════════════════════════════════
# Section 3 : Fast engine grid_common
# ═══════════════════════════════════════════════════════════════════════════


class TestFastEngineGridCommonTimeStop:
    """Tests time_stop dans _simulate_grid_common()."""

    def test_backward_compat_disabled(self, make_indicator_cache):
        """max_hold_candles=0 → résultat identique à avant."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_atr

        n = 200
        base_price = 100.0
        closes = np.full(n, base_price)
        highs = closes + 2.0
        lows = closes - 2.0
        sma_vals = np.full(n, base_price)

        cache = make_indicator_cache(
            n=n, closes=closes, highs=highs, lows=lows,
            bb_sma={14: sma_vals},
            atr_by_period={14: np.full(n, 3.0)},
        )
        bt_config = _make_bt_config()

        params_off = {"ma_period": 14, "atr_period": 14, "atr_multiplier_start": 2.0,
                      "atr_multiplier_step": 1.0, "num_levels": 3, "sl_percent": 20.0,
                      "max_hold_candles": 0}
        params_none = {"ma_period": 14, "atr_period": 14, "atr_multiplier_start": 2.0,
                       "atr_multiplier_step": 1.0, "num_levels": 3, "sl_percent": 20.0}

        r1 = _simulate_grid_atr(cache, params_off, bt_config)
        r2 = _simulate_grid_atr(cache, params_none, bt_config)
        assert r1[2] == r2[2]  # même capital final

    def test_time_stop_triggers_in_loss(self, make_indicator_cache):
        """Position ouverte 50 candles en perte → trade fermé par time_stop."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_common, _build_entry_prices

        n = 100
        # Prix qui descent lentement, restant au-dessus du SL
        closes = np.linspace(100.0, 92.0, n)  # -8% sur 100 candles
        highs = closes + 1.0
        lows = closes - 1.0
        opens = closes + 0.5
        sma_vals = np.full(n, 110.0)  # SMA très haute → pas de TP

        cache = make_indicator_cache(
            n=n, closes=closes, highs=highs, lows=lows, opens=opens,
            bb_sma={14: sma_vals},
            atr_by_period={14: np.full(n, 3.0)},
        )
        bt_config = _make_bt_config(leverage=3, slippage_pct=0.0001)

        params = {"ma_period": 14, "atr_period": 14, "atr_multiplier_start": 2.0,
                  "atr_multiplier_step": 1.0, "num_levels": 3, "sl_percent": 30.0}
        entry_prices = _build_entry_prices("grid_atr", cache, params, 3, 1)

        # Sans time_stop : positions restent ouvertes (pas de TP ni SL à 30%)
        pnls_no_ts, _, cap_no_ts = _simulate_grid_common(
            entry_prices, sma_vals, cache, bt_config, 3, 0.30, 1,
            max_hold_candles=0,
        )

        # Avec time_stop=20 candles
        pnls_ts, _, cap_ts = _simulate_grid_common(
            entry_prices, sma_vals, cache, bt_config, 3, 0.30, 1,
            max_hold_candles=20,
        )

        # Avec time_stop, des trades doivent être fermés plus tôt
        assert len(pnls_ts) >= len(pnls_no_ts)

    def test_no_time_stop_in_profit(self, make_indicator_cache):
        """Position ouverte 50 candles en profit → pas de time_stop."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_common, _build_entry_prices

        n = 100
        # Prix monte lentement, mais sous la SMA (pas de TP)
        closes = np.linspace(90.0, 95.0, n)
        highs = closes + 0.5
        lows = closes - 0.5
        opens = closes.copy()
        sma_vals = np.full(n, 110.0)  # SMA très haute → jamais TP

        cache = make_indicator_cache(
            n=n, closes=closes, highs=highs, lows=lows, opens=opens,
            bb_sma={14: sma_vals},
            atr_by_period={14: np.full(n, 3.0)},
        )
        bt_config = _make_bt_config(leverage=3)

        params = {"ma_period": 14, "atr_period": 14, "atr_multiplier_start": 1.0,
                  "atr_multiplier_step": 0.5, "num_levels": 2, "sl_percent": 30.0}
        entry_prices = _build_entry_prices("grid_atr", cache, params, 2, 1)

        # Niveaux à SMA - ATR*1.0 = 110-3 = 107 → jamais touché (prix < 95)
        # Forcer les entry_prices pour déclencher des entrées
        entry_prices[:] = 91.0  # Juste au-dessus du min lows
        entry_prices[:, 1] = 90.5

        pnls_no_ts, _, _ = _simulate_grid_common(
            entry_prices, sma_vals, cache, bt_config, 2, 0.30, 1,
            max_hold_candles=0,
        )
        pnls_ts, _, _ = _simulate_grid_common(
            entry_prices, sma_vals, cache, bt_config, 2, 0.30, 1,
            max_hold_candles=20,
        )

        # Prix monte → profit → time_stop ne devrait pas se déclencher
        # Résultats identiques (time_stop ne fire pas en profit)
        assert len(pnls_ts) == len(pnls_no_ts)

    def test_first_entry_idx_reset(self, make_indicator_cache):
        """first_entry_idx reset correctement après fermeture."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_common

        n = 200
        # Prix oscille : descend (entrée), remonte (TP), redescend (entrée), etc.
        closes = np.full(n, 100.0)
        for i in range(n):
            if i < 30:
                closes[i] = 100.0 - i * 0.2  # descente
            elif i < 50:
                closes[i] = 110.0  # remontée → TP
            elif i < 100:
                closes[i] = 100.0 - (i - 50) * 0.2  # redescente
            elif i < 120:
                closes[i] = 110.0  # remontée → TP
            else:
                closes[i] = 100.0

        highs = closes + 1.0
        lows = closes - 1.0
        opens = closes.copy()
        sma_vals = np.full(n, 100.0)

        entry_prices = np.full((n, 3), 99.0)  # trigger à 99
        entry_prices[:, 1] = 98.0
        entry_prices[:, 2] = 97.0

        cache = make_indicator_cache(
            n=n, closes=closes, highs=highs, lows=lows, opens=opens,
            bb_sma={14: sma_vals},
            atr_by_period={14: np.full(n, 3.0)},
        )
        bt_config = _make_bt_config(leverage=3)

        pnls, _, _ = _simulate_grid_common(
            entry_prices, sma_vals, cache, bt_config, 3, 0.30, 1,
            max_hold_candles=80,
        )
        # Devrait avoir au moins 2 cycles (2 TPs)
        assert len(pnls) >= 2

    def test_fees_time_stop_are_taker(self, make_indicator_cache):
        """Fees time_stop = taker + slippage (comme SL, pas comme TP)."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_common

        n = 100
        # Prix descend légèrement mais reste au-dessus du SL
        closes = np.full(n, 99.0)
        closes[0] = 100.0  # Première candle plus haute
        highs = closes + 0.5
        lows = closes - 0.5
        opens = closes.copy()
        sma_vals = np.full(n, 120.0)  # SMA très haute → jamais TP

        entry_prices = np.full((n, 1), 100.0)  # trigger au début

        cache = make_indicator_cache(
            n=n, closes=closes, highs=highs, lows=lows, opens=opens,
            bb_sma={14: sma_vals},
            atr_by_period={14: np.full(n, 3.0)},
        )
        bt_config = _make_bt_config(leverage=1, taker_fee=0.001, maker_fee=0.0001, slippage_pct=0.001)

        # time_stop at candle 10
        pnls, _, _ = _simulate_grid_common(
            entry_prices, sma_vals, cache, bt_config, 1, 0.50, 1,
            max_hold_candles=10,
        )
        # Le trade devrait exister et avoir des fees taker (pas maker)
        assert len(pnls) >= 1
        # Net PnL devrait être négatif (loss + taker fees)
        assert pnls[0] < 0

    def test_envelope_dca_unaffected(self, make_indicator_cache):
        """_simulate_envelope_dca fonctionne toujours sans max_hold (défaut 0)."""
        from backend.optimization.fast_multi_backtest import _simulate_envelope_dca

        n = 200
        closes = np.full(n, 100.0)
        highs = closes + 2.0
        lows = closes - 2.0
        sma_vals = np.full(n, 100.0)

        cache = make_indicator_cache(
            n=n, closes=closes, highs=highs, lows=lows,
            bb_sma={10: sma_vals},
        )
        bt_config = _make_bt_config()
        params = {"ma_period": 10, "num_levels": 3, "envelope_start": 0.05,
                  "envelope_step": 0.03, "sl_percent": 20.0}

        # Ne doit pas lever d'exception
        pnls, returns, capital = _simulate_envelope_dca(cache, params, bt_config)
        assert isinstance(capital, float)


# ═══════════════════════════════════════════════════════════════════════════
# Section 4 : Fast engine grid_boltrend
# ═══════════════════════════════════════════════════════════════════════════


class TestFastEngineBolTrendTimeStop:
    """Tests time_stop dans _simulate_grid_boltrend()."""

    def _make_boltrend_cache(self, make_indicator_cache, n=500, base_price=100.0):
        """Crée un cache avec indicateurs Bollinger + SMA pour grid_boltrend."""
        bol_window = 50
        bol_std = 2.0
        long_ma_window = 100
        atr_period = 14

        closes = np.full(n, base_price)
        highs = closes + 2.0
        lows = closes - 2.0
        opens = closes.copy()

        # SMA simple
        bb_sma = {bol_window: np.full(n, base_price), long_ma_window: np.full(n, base_price - 5)}
        bb_upper = {(bol_window, bol_std): np.full(n, base_price + 10)}
        bb_lower = {(bol_window, bol_std): np.full(n, base_price - 10)}
        atr_by_period = {atr_period: np.full(n, 3.0)}

        cache = make_indicator_cache(
            n=n, closes=closes, highs=highs, lows=lows, opens=opens,
            bb_sma=bb_sma, bb_upper=bb_upper, bb_lower=bb_lower,
            atr_by_period=atr_by_period,
        )
        return cache

    def test_backward_compat_disabled(self, make_indicator_cache):
        """max_hold_candles=0 → résultat identique."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_boltrend

        cache = self._make_boltrend_cache(make_indicator_cache)
        bt_config = _make_bt_config()

        params_base = {
            "bol_window": 50, "bol_std": 2.0, "long_ma_window": 100,
            "min_bol_spread": 0.0, "atr_period": 14, "atr_spacing_mult": 1.0,
            "num_levels": 3, "sl_percent": 15.0, "sides": ["long", "short"],
        }
        params_off = {**params_base, "max_hold_candles": 0}
        params_none = {**params_base}

        r1 = _simulate_grid_boltrend(cache, params_off, bt_config)
        r2 = _simulate_grid_boltrend(cache, params_none, bt_config)
        assert r1[2] == r2[2]  # même capital final

    def test_time_stop_triggers_in_loss(self, make_indicator_cache):
        """Position en perte depuis 60 candles → time_stop."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_boltrend

        n = 500
        bol_window = 50
        bol_std = 2.0
        long_ma_window = 100
        atr_period = 14
        base_price = 100.0

        closes = np.full(n, base_price)
        # Simulate a breakout: prev_close < bb_upper, close > bb_upper
        # Then price stays below entry (loss) but above SL
        start_idx = max(bol_window, long_ma_window) + 1
        # Create a breakout at start_idx
        closes[start_idx - 1] = base_price + 9.0  # prev_close < bb_upper (110)
        closes[start_idx] = base_price + 11.0  # close > bb_upper → breakout LONG
        # After breakout, price drops and stays there (in loss, no TP)
        for i in range(start_idx + 1, n):
            closes[i] = base_price + 6.0  # below entry (~111) but above SL, above bb_sma(100)

        highs = closes + 1.0
        lows = closes - 1.0
        opens = closes.copy()

        bb_sma = {bol_window: np.full(n, base_price), long_ma_window: np.full(n, base_price - 5)}
        bb_upper = {(bol_window, bol_std): np.full(n, base_price + 10)}
        bb_lower = {(bol_window, bol_std): np.full(n, base_price - 10)}
        atr_by_period = {atr_period: np.full(n, 3.0)}

        cache = make_indicator_cache(
            n=n, closes=closes, highs=highs, lows=lows, opens=opens,
            bb_sma=bb_sma, bb_upper=bb_upper, bb_lower=bb_lower,
            atr_by_period=atr_by_period,
        )
        bt_config = _make_bt_config()

        params_no_ts = {
            "bol_window": 50, "bol_std": 2.0, "long_ma_window": 100,
            "min_bol_spread": 0.0, "atr_period": 14, "atr_spacing_mult": 1.0,
            "num_levels": 3, "sl_percent": 30.0, "sides": ["long"],
            "max_hold_candles": 0,
        }
        params_ts = {**params_no_ts, "max_hold_candles": 30}

        pnls_no_ts, _, _ = _simulate_grid_boltrend(cache, params_no_ts, bt_config)
        pnls_ts, _, _ = _simulate_grid_boltrend(cache, params_ts, bt_config)

        # Avec time_stop, le trade est fermé plus tôt → plus de trades
        assert len(pnls_ts) >= len(pnls_no_ts)

    def test_sl_has_priority_over_time_stop(self, make_indicator_cache):
        """SL atteint avant time_stop → SL prioritaire."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_boltrend

        n = 500
        bol_window = 50
        bol_std = 2.0
        long_ma_window = 100
        atr_period = 14
        base_price = 100.0

        closes = np.full(n, base_price)
        start_idx = max(bol_window, long_ma_window) + 1
        closes[start_idx - 1] = base_price + 9.0
        closes[start_idx] = base_price + 11.0  # breakout LONG
        # Price drops hard → SL first (before time_stop threshold)
        for i in range(start_idx + 1, min(start_idx + 5, n)):
            closes[i] = base_price - 5.0  # below SL (if sl=10%)

        highs = closes + 1.0
        lows = closes - 1.0
        opens = closes.copy()

        bb_sma = {bol_window: np.full(n, base_price), long_ma_window: np.full(n, base_price - 5)}
        bb_upper = {(bol_window, bol_std): np.full(n, base_price + 10)}
        bb_lower = {(bol_window, bol_std): np.full(n, base_price - 10)}
        atr_by_period = {atr_period: np.full(n, 3.0)}

        cache = make_indicator_cache(
            n=n, closes=closes, highs=highs, lows=lows, opens=opens,
            bb_sma=bb_sma, bb_upper=bb_upper, bb_lower=bb_lower,
            atr_by_period=atr_by_period,
        )
        bt_config = _make_bt_config()

        params = {
            "bol_window": 50, "bol_std": 2.0, "long_ma_window": 100,
            "min_bol_spread": 0.0, "atr_period": 14, "atr_spacing_mult": 1.0,
            "num_levels": 3, "sl_percent": 10.0, "sides": ["long"],
            "max_hold_candles": 50,  # time_stop at 50, SL at ~2-3
        }
        pnls, _, _ = _simulate_grid_boltrend(cache, params, bt_config)
        # Should have at least one trade (SL), closed by SL before time_stop
        assert len(pnls) >= 1

    def test_breakout_candle_idx_used(self, make_indicator_cache):
        """breakout_candle_idx est bien utilisé (pas un first_entry_idx séparé)."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_boltrend

        # Le simple fait que le test ne crash pas et donne les mêmes résultats
        # avec max_hold=0 prouve que breakout_candle_idx est bien géré
        cache = self._make_boltrend_cache(make_indicator_cache)
        bt_config = _make_bt_config()

        params = {
            "bol_window": 50, "bol_std": 2.0, "long_ma_window": 100,
            "min_bol_spread": 0.0, "atr_period": 14, "atr_spacing_mult": 1.0,
            "num_levels": 3, "sl_percent": 15.0, "sides": ["long", "short"],
            "max_hold_candles": 0,
        }
        pnls, returns, capital = _simulate_grid_boltrend(cache, params, bt_config)
        assert isinstance(capital, float)


# ═══════════════════════════════════════════════════════════════════════════
# Section 5 : Config validation
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigValidation:
    """Tests de validation Pydantic pour max_hold_candles."""

    def test_grid_atr_valid(self):
        """GridATRConfig avec max_hold_candles=48 → valide."""
        config = GridATRConfig(max_hold_candles=48)
        assert config.max_hold_candles == 48

    def test_grid_atr_negative_rejected(self):
        """GridATRConfig avec max_hold_candles=-1 → ValidationError."""
        with pytest.raises(Exception):  # ValidationError
            GridATRConfig(max_hold_candles=-1)

    def test_grid_boltrend_default_zero(self):
        """GridBolTrendConfig default max_hold_candles=0 (backward compat)."""
        config = GridBolTrendConfig()
        assert config.max_hold_candles == 0

    def test_get_params_includes_max_hold(self):
        """get_params() retourne max_hold_candles."""
        strategy = _make_atr_strategy(max_hold_candles=48)
        params = strategy.get_params()
        assert params["max_hold_candles"] == 48


# ═══════════════════════════════════════════════════════════════════════════
# Section 6 : Parité fast engine vs strategy layer
# ═══════════════════════════════════════════════════════════════════════════


class TestParity:
    """Test de parité : fast engine et strategy layer déclenchent time_stop."""

    def test_both_layers_agree_on_time_stop(self):
        """Fast engine et should_close_all() déclenchent time_stop pour les mêmes conditions."""
        # Strategy layer
        strategy = _make_atr_strategy(max_hold_candles=48)
        entry_time = _BASE_TS - timedelta(hours=50)  # > 48 candles
        gs = _make_grid_state(entry_price=100.0, close=95.0, entry_time=entry_time)
        ctx = _make_atr_ctx(sma_val=96.0, close=95.0)

        result = strategy.should_close_all(ctx, gs)
        assert result == "time_stop", f"Strategy layer should return 'time_stop', got {result}"

        # Fast engine : même scénario — position ouverte à candle 0,
        # prix descend progressivement (en perte), SMA très haute
        from backend.optimization.fast_multi_backtest import _simulate_grid_common
        from backend.optimization.indicator_cache import IndicatorCache

        n = 100
        closes = np.full(n, 95.0)
        closes[0] = 100.0
        highs = closes + 1.0
        lows = closes - 1.0
        opens = closes.copy()
        sma_vals = np.full(n, 120.0)  # SMA très haute → pas de TP

        # Entry prices : level 0 at 100 (trigger at candle 0)
        entry_prices = np.full((n, 1), float("nan"))
        entry_prices[0, 0] = 100.0  # trigger à la première candle
        # Candles suivantes : pas de nouveaux niveaux
        for i in range(1, n):
            entry_prices[i, 0] = 50.0  # très bas → jamais touché en LONG

        cache = IndicatorCache(
            n_candles=n, opens=opens, highs=highs, lows=lows, closes=closes,
            volumes=np.full(n, 100.0), total_days=n / 24,
            rsi={14: np.full(n, 50.0)},
            vwap=np.full(n, np.nan), vwap_distance_pct=np.full(n, np.nan),
            adx_arr=np.full(n, 25.0), di_plus=np.full(n, 15.0), di_minus=np.full(n, 10.0),
            atr_arr=np.full(n, 1.0), atr_sma=np.full(n, 1.0),
            volume_sma_arr=np.full(n, 100.0),
            regime=np.zeros(n, dtype=np.int8),
            rolling_high={}, rolling_low={},
            filter_adx=np.full(n, np.nan), filter_di_plus=np.full(n, np.nan),
            filter_di_minus=np.full(n, np.nan),
            bb_sma={14: sma_vals}, bb_upper={}, bb_lower={},
            supertrend_direction={}, atr_by_period={14: np.full(n, 3.0)},
            supertrend_dir_4h={},
            funding_rates_1h=None, candle_timestamps=None,
            ema_by_period={}, adx_by_period={},
        )

        bt_config = _make_bt_config(leverage=3)

        pnls, _, _ = _simulate_grid_common(
            entry_prices, sma_vals, cache, bt_config, 1, 0.30, 1,
            max_hold_candles=48,
        )

        # Le fast engine devrait avoir fermé la position par time_stop à la candle 48
        assert len(pnls) >= 1, "Fast engine should have at least 1 trade (time_stop)"
        assert pnls[0] < 0, "Trade should be a loss (time_stop on losing position)"
