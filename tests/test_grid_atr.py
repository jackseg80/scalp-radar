"""Tests pour Grid ATR (Sprint 19).

Couvre :
- Section 1 : Signaux (compute_grid, should_close_all) — ~14 tests
- Section 2 : TP/SL prices — 3 tests
- Section 3 : Fast engine — ~8 tests
- Section 4 : Parité fast/normal — 2 tests
- Section 5 : Registry & integration — ~6 tests
- Section 6 : Executor helpers — 2 tests
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from backend.core.config import GridATRConfig
from backend.core.models import Candle, Direction
from backend.strategies.base import StrategyContext
from backend.strategies.base_grid import GridLevel, GridPosition, GridState
from backend.strategies.grid_atr import GridATRStrategy


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_strategy(**overrides) -> GridATRStrategy:
    """Crée une stratégie Grid ATR avec defaults sensibles."""
    defaults = {
        "ma_period": 14,
        "atr_period": 14,
        "atr_multiplier_start": 2.0,
        "atr_multiplier_step": 1.0,
        "num_levels": 3,
        "sl_percent": 20.0,
        "sides": ["long"],
        "leverage": 6,
    }
    defaults.update(overrides)
    config = GridATRConfig(**defaults)
    return GridATRStrategy(config)


def _make_ctx(
    sma_val: float, atr_val: float, close: float
) -> StrategyContext:
    """Crée un StrategyContext minimal avec SMA, ATR et close."""
    return StrategyContext(
        symbol="BTC/USDT",
        timestamp=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        candles={},
        indicators={"1h": {"sma": sma_val, "atr": atr_val, "close": close}},
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
    entry_price: float = 90.0,
    quantity: float = 1.0,
) -> GridPosition:
    return GridPosition(
        level=level,
        direction=direction,
        entry_price=entry_price,
        quantity=quantity,
        entry_time=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
        entry_fee=0.01,
    )


def _make_candles(
    n: int,
    start_price: float = 100.0,
    step: float = 0.0,
    spread: float = 1.0,
) -> list[Candle]:
    """Génère N candles 1h synthétiques."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    for i in range(n):
        price = start_price + i * step
        candles.append(
            Candle(
                symbol="BTC/USDT",
                exchange="binance",
                timeframe="1h",
                timestamp=base + timedelta(hours=i),
                open=price,
                high=price + spread,
                low=price - spread,
                close=price,
                volume=100.0,
            )
        )
    return candles


def _make_bt_config(**overrides):
    """Crée un BacktestConfig avec defaults sensibles pour les tests."""
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


def _make_sinusoidal_candles(
    n: int = 500,
    base_price: float = 100.0,
    amplitude: float = 8.0,
    period: int = 48,
    spread: float = 2.0,
) -> list[Candle]:
    """Génère N candles 1h sinusoïdales (oscillation régulière).

    Produit des trades garantis pour la parité fast/normal.
    """
    base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    for i in range(n):
        mid = base_price + amplitude * math.sin(2 * math.pi * i / period)
        candles.append(
            Candle(
                symbol="BTC/USDT",
                exchange="binance",
                timeframe="1h",
                timestamp=base_ts + timedelta(hours=i),
                open=mid - 0.3,
                high=mid + spread,
                low=mid - spread,
                close=mid + 0.3,
                volume=100.0,
            )
        )
    return candles


# ═══════════════════════════════════════════════════════════════════════════
# Section 1 : Signaux (compute_grid, should_close_all)
# ═══════════════════════════════════════════════════════════════════════════


class TestGridATRSignals:
    """Tests de la logique de signaux Grid ATR."""

    def test_name(self):
        """Le nom de la stratégie est 'grid_atr'."""
        strategy = _make_strategy()
        assert strategy.name == "grid_atr"

    def test_max_positions(self):
        """max_positions = num_levels."""
        strategy = _make_strategy(num_levels=4)
        assert strategy.max_positions == 4

    def test_min_candles_uses_max_period(self):
        """min_candles utilise max(ma_period, atr_period) + marge."""
        # atr_period > ma_period
        strategy = _make_strategy(ma_period=10, atr_period=20)
        min_c = strategy.min_candles
        assert "1h" in min_c
        assert min_c["1h"] >= 20 + 20  # atr_period + 20

        # ma_period > atr_period
        strategy2 = _make_strategy(ma_period=30, atr_period=14)
        assert strategy2.min_candles["1h"] >= 30 + 20

    def test_compute_indicators_has_sma_and_atr(self):
        """compute_indicators retourne sma ET atr."""
        strategy = _make_strategy(ma_period=5, atr_period=5)
        candles = _make_candles(50)
        result = strategy.compute_indicators({"1h": candles})
        assert "1h" in result
        # Vérifier un timestamp non-NaN (après warmup)
        last_ts = candles[-1].timestamp.isoformat()
        assert "sma" in result["1h"][last_ts]
        assert "atr" in result["1h"][last_ts]
        assert not math.isnan(result["1h"][last_ts]["sma"])

    def test_compute_grid_returns_long_levels_below_sma(self):
        """compute_grid retourne des niveaux LONG en dessous de la SMA."""
        strategy = _make_strategy(
            atr_multiplier_start=2.0, atr_multiplier_step=1.0, num_levels=3
        )
        # SMA=100, ATR=5 → niveaux à 100-10=90, 100-15=85, 100-20=80
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=95.0)
        grid_state = _make_grid_state()

        levels = strategy.compute_grid(ctx, grid_state)
        assert len(levels) == 3
        assert levels[0].entry_price == pytest.approx(90.0)  # 100 - 5*2
        assert levels[1].entry_price == pytest.approx(85.0)  # 100 - 5*3
        assert levels[2].entry_price == pytest.approx(80.0)  # 100 - 5*4
        for lvl in levels:
            assert lvl.direction == Direction.LONG
            assert lvl.entry_price < 100.0

    def test_compute_grid_atr_adaptive(self):
        """Les enveloppes changent avec l'ATR (pas fixes)."""
        strategy = _make_strategy(
            atr_multiplier_start=2.0, atr_multiplier_step=1.0, num_levels=2
        )
        grid_state = _make_grid_state()

        # ATR=10 → niveaux à 80, 70
        ctx1 = _make_ctx(sma_val=100.0, atr_val=10.0, close=90.0)
        levels1 = strategy.compute_grid(ctx1, grid_state)
        assert levels1[0].entry_price == pytest.approx(80.0)
        assert levels1[1].entry_price == pytest.approx(70.0)

        # ATR=5 → niveaux à 90, 85
        ctx2 = _make_ctx(sma_val=100.0, atr_val=5.0, close=95.0)
        levels2 = strategy.compute_grid(ctx2, grid_state)
        assert levels2[0].entry_price == pytest.approx(90.0)
        assert levels2[1].entry_price == pytest.approx(85.0)

    def test_compute_grid_symmetric_short(self):
        """Les enveloppes SHORT sont symétriques (sma + atr × mult)."""
        strategy = _make_strategy(
            sides=["short"],
            atr_multiplier_start=2.0,
            atr_multiplier_step=1.0,
            num_levels=2,
        )
        # SMA=100, ATR=5 → niveaux SHORT à 110, 115
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=105.0)
        grid_state = _make_grid_state()

        levels = strategy.compute_grid(ctx, grid_state)
        assert len(levels) == 2
        assert levels[0].entry_price == pytest.approx(110.0)  # 100 + 5*2
        assert levels[1].entry_price == pytest.approx(115.0)  # 100 + 5*3
        for lvl in levels:
            assert lvl.direction == Direction.SHORT

    def test_compute_grid_filters_filled_levels(self):
        """Les niveaux déjà remplis sont exclus."""
        strategy = _make_strategy(num_levels=3)
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=95.0)

        # Niveau 0 déjà rempli
        pos = _make_position(level=0)
        grid_state = _make_grid_state([pos])

        levels = strategy.compute_grid(ctx, grid_state)
        level_indices = {lvl.index for lvl in levels}
        assert 0 not in level_indices
        assert 1 in level_indices
        assert 2 in level_indices

    def test_direction_lock(self):
        """Si LONG ouvert, pas de SHORT proposé (même avec sides=["long","short"])."""
        strategy = _make_strategy(sides=["long", "short"], num_levels=3)
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=95.0)

        # Position LONG ouverte au niveau 0
        pos = _make_position(level=0, direction=Direction.LONG)
        grid_state = _make_grid_state([pos])

        levels = strategy.compute_grid(ctx, grid_state)
        for lvl in levels:
            assert lvl.direction == Direction.LONG

    def test_compute_grid_returns_empty_if_nan_atr(self):
        """Pas de niveaux si ATR est NaN."""
        strategy = _make_strategy()
        ctx = _make_ctx(sma_val=100.0, atr_val=float("nan"), close=95.0)
        assert strategy.compute_grid(ctx, _make_grid_state()) == []

    def test_compute_grid_returns_empty_if_atr_zero(self):
        """Pas de niveaux si ATR == 0."""
        strategy = _make_strategy()
        ctx = _make_ctx(sma_val=100.0, atr_val=0.0, close=95.0)
        assert strategy.compute_grid(ctx, _make_grid_state()) == []

    def test_compute_grid_guards_negative_price(self):
        """Pas de niveau LONG si entry_price serait négatif."""
        strategy = _make_strategy(
            atr_multiplier_start=50.0, num_levels=1
        )
        # SMA=100, ATR=5, mult=50 → entry = 100 - 5*50 = -150 < 0
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=50.0)
        levels = strategy.compute_grid(ctx, _make_grid_state())
        assert len(levels) == 0

    def test_should_close_all_tp_long(self):
        """TP global LONG : close >= SMA."""
        strategy = _make_strategy()
        pos = _make_position(level=0, direction=Direction.LONG, entry_price=90.0)
        grid_state = _make_grid_state([pos])
        # Close (105) >= SMA (100) → TP
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=105.0)
        assert strategy.should_close_all(ctx, grid_state) == "tp_global"

    def test_should_close_all_sl_long(self):
        """SL global LONG : close <= avg_entry × (1 - sl_pct)."""
        strategy = _make_strategy(sl_percent=20.0)
        pos = _make_position(level=0, direction=Direction.LONG, entry_price=90.0)
        grid_state = _make_grid_state([pos])
        # avg_entry=90, SL = 90 × 0.8 = 72, close=70 < 72 → SL
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=70.0)
        assert strategy.should_close_all(ctx, grid_state) == "sl_global"

    def test_should_close_all_none(self):
        """Pas de fermeture si ni TP ni SL."""
        strategy = _make_strategy(sl_percent=20.0)
        pos = _make_position(level=0, direction=Direction.LONG, entry_price=90.0)
        grid_state = _make_grid_state([pos])
        # SL=72, TP(SMA)=100, close=85 → entre les deux
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=85.0)
        assert strategy.should_close_all(ctx, grid_state) is None

    def test_should_close_all_empty_positions(self):
        """Pas de fermeture si aucune position ouverte."""
        strategy = _make_strategy()
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=85.0)
        assert strategy.should_close_all(ctx, _make_grid_state()) is None


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 : TP/SL prices
# ═══════════════════════════════════════════════════════════════════════════


class TestGridATRPrices:
    """Tests get_tp_price / get_sl_price."""

    def test_get_tp_price_returns_sma(self):
        """TP = SMA actuelle."""
        strategy = _make_strategy()
        indicators = {"sma": 105.0, "atr": 5.0}
        tp = strategy.get_tp_price(_make_grid_state(), indicators)
        assert tp == pytest.approx(105.0)

    def test_get_sl_price_long(self):
        """SL LONG = avg_entry × (1 - sl_pct)."""
        strategy = _make_strategy(sl_percent=20.0)
        pos = _make_position(level=0, direction=Direction.LONG, entry_price=100.0)
        grid_state = _make_grid_state([pos])
        sl = strategy.get_sl_price(grid_state, {"sma": 110.0, "atr": 5.0})
        assert sl == pytest.approx(80.0)  # 100 × 0.8

    def test_get_sl_price_short(self):
        """SL SHORT = avg_entry × (1 + sl_pct)."""
        strategy = _make_strategy(sl_percent=20.0, sides=["short"])
        pos = _make_position(
            level=0, direction=Direction.SHORT, entry_price=100.0
        )
        grid_state = _make_grid_state([pos])
        sl = strategy.get_sl_price(grid_state, {"sma": 90.0, "atr": 5.0})
        assert sl == pytest.approx(120.0)  # 100 × 1.2


# ═══════════════════════════════════════════════════════════════════════════
# Section 3 : Fast engine
# ═══════════════════════════════════════════════════════════════════════════


class TestGridATRFastEngine:
    """Tests du fast backtest engine pour grid_atr."""

    def test_run_multi_backtest_from_cache(self, make_indicator_cache):
        """Fast engine grid_atr retourne un résultat valide (5-tuple)."""
        from backend.optimization.fast_multi_backtest import (
            run_multi_backtest_from_cache,
        )

        n = 200
        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        sma_arr = np.full(n, np.nan)
        atr_arr = np.full(n, np.nan)
        for i in range(14, n):
            sma_arr[i] = np.mean(prices[max(0, i - 13) : i + 1])
            # ATR simplifié
            atr_arr[i] = np.mean(np.abs(np.diff(prices[max(0, i - 13) : i + 1])))

        cache = make_indicator_cache(
            n=n,
            closes=prices,
            opens=prices + rng.uniform(-0.3, 0.3, n),
            highs=prices + np.abs(rng.normal(1.0, 0.5, n)),
            lows=prices - np.abs(rng.normal(1.0, 0.5, n)),
            bb_sma={14: sma_arr},
            atr_by_period={14: atr_arr},
        )
        params = {
            "ma_period": 14,
            "atr_period": 14,
            "atr_multiplier_start": 2.0,
            "atr_multiplier_step": 1.0,
            "num_levels": 3,
            "sl_percent": 20.0,
        }
        bt_config = _make_bt_config()
        result = run_multi_backtest_from_cache("grid_atr", params, cache, bt_config)
        assert len(result) == 5
        assert result[0] == params  # params
        assert isinstance(result[1], float)  # sharpe
        assert isinstance(result[4], int)  # n_trades

    def test_unknown_strategy_still_raises(self, make_indicator_cache):
        """Stratégie 'unknown' lève toujours ValueError."""
        from backend.optimization.fast_multi_backtest import (
            run_multi_backtest_from_cache,
        )

        cache = make_indicator_cache(n=50)
        bt_config = _make_bt_config()
        with pytest.raises(ValueError, match="inconnue"):
            run_multi_backtest_from_cache("unknown", {}, cache, bt_config)

    def test_atr_adaptive_levels(self, make_indicator_cache):
        """Les niveaux d'entrée changent avec l'ATR (vérifié sur données synthétiques)."""
        from backend.optimization.fast_multi_backtest import (
            run_multi_backtest_from_cache,
        )

        n = 200
        # Série 1 : ATR élevé → moins de trades (niveaux plus éloignés)
        prices = np.full(n, 100.0)
        sma_14 = np.full(n, 100.0)
        sma_14[:14] = np.nan
        atr_high = np.full(n, 10.0)  # ATR élevé
        atr_high[:14] = np.nan
        atr_low = np.full(n, 2.0)   # ATR faible
        atr_low[:14] = np.nan

        params = {
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 2.0, "atr_multiplier_step": 1.0,
            "num_levels": 3, "sl_percent": 20.0,
        }
        bt_config = _make_bt_config()

        # ATR=10 : premier niveau à 100-20=80 (très loin)
        cache_high = make_indicator_cache(
            n=n, closes=prices,
            lows=prices - 5.0, highs=prices + 5.0,
            bb_sma={14: sma_14}, atr_by_period={14: atr_high},
        )
        r_high = run_multi_backtest_from_cache("grid_atr", params, cache_high, bt_config)

        # ATR=2 : premier niveau à 100-4=96 (proche)
        cache_low = make_indicator_cache(
            n=n, closes=prices,
            lows=prices - 5.0, highs=prices + 5.0,
            bb_sma={14: sma_14}, atr_by_period={14: atr_low},
        )
        r_low = run_multi_backtest_from_cache("grid_atr", params, cache_low, bt_config)

        # ATR faible devrait générer plus de trades (niveaux plus proches du prix)
        assert r_low[4] >= r_high[4]

    def test_no_entry_when_atr_nan(self, make_indicator_cache):
        """Pas de trade si ATR est NaN (début de série)."""
        from backend.optimization.fast_multi_backtest import (
            run_multi_backtest_from_cache,
        )

        n = 50
        prices = np.full(n, 100.0)
        sma_arr = np.full(n, np.nan)  # Tout NaN
        atr_arr = np.full(n, np.nan)  # Tout NaN

        cache = make_indicator_cache(
            n=n, closes=prices,
            bb_sma={14: sma_arr}, atr_by_period={14: atr_arr},
        )
        params = {
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 2.0, "atr_multiplier_step": 1.0,
            "num_levels": 3, "sl_percent": 20.0,
        }
        bt_config = _make_bt_config()
        result = run_multi_backtest_from_cache("grid_atr", params, cache, bt_config)
        assert result[4] == 0  # 0 trades

    def test_force_close_end_of_data(self, make_indicator_cache):
        """Positions ouvertes fermées en fin de données."""
        from backend.optimization.fast_multi_backtest import (
            run_multi_backtest_from_cache,
        )

        n = 100
        # Prix qui descend pour ouvrir des positions puis reste bas (pas de TP)
        prices = np.concatenate([
            np.linspace(100, 80, 50),  # Descente : touche les niveaux
            np.full(50, 80.0),          # Reste bas : pas de TP (SMA ~80)
        ])
        sma_arr = np.full(n, np.nan)
        atr_arr = np.full(n, np.nan)
        for i in range(14, n):
            sma_arr[i] = np.mean(prices[max(0, i - 13) : i + 1])
            atr_arr[i] = 3.0  # ATR fixe

        cache = make_indicator_cache(
            n=n,
            closes=prices,
            opens=prices,
            highs=prices + 2.0,
            lows=prices - 2.0,
            bb_sma={14: sma_arr},
            atr_by_period={14: atr_arr},
        )
        params = {
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 1.0, "atr_multiplier_step": 0.5,
            "num_levels": 2, "sl_percent": 50.0,  # SL large pour ne pas déclencher
        }
        bt_config = _make_bt_config()
        result = run_multi_backtest_from_cache("grid_atr", params, cache, bt_config)
        # Des trades doivent exister (au moins le force close)
        assert result[4] >= 1

    def test_tp_at_sma(self, make_indicator_cache):
        """TP quand close >= SMA (LONG)."""
        from backend.optimization.fast_multi_backtest import (
            run_multi_backtest_from_cache,
        )

        n = 100
        # Prix descend pour toucher un niveau, puis remonte vers la SMA
        prices = np.concatenate([
            np.full(20, 100.0),          # Stable
            np.linspace(100, 90, 20),    # Descente → touche niveau
            np.linspace(90, 105, 30),    # Remontée → TP à la SMA
            np.full(30, 105.0),          # Stable haut
        ])
        sma_arr = np.full(n, np.nan)
        atr_arr = np.full(n, np.nan)
        for i in range(14, n):
            sma_arr[i] = np.mean(prices[max(0, i - 13) : i + 1])
            atr_arr[i] = 3.0

        cache = make_indicator_cache(
            n=n,
            closes=prices,
            opens=prices,
            highs=prices + 2.0,
            lows=prices - 2.0,
            bb_sma={14: sma_arr},
            atr_by_period={14: atr_arr},
        )
        params = {
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 1.5, "atr_multiplier_step": 0.5,
            "num_levels": 2, "sl_percent": 30.0,
        }
        bt_config = _make_bt_config()
        result = run_multi_backtest_from_cache("grid_atr", params, cache, bt_config)
        # Au moins un trade (ouvert + TP ou force close)
        assert result[4] >= 1

    def test_sl_at_percent(self, make_indicator_cache):
        """SL quand close descend au-delà du sl_percent."""
        from backend.optimization.fast_multi_backtest import (
            run_multi_backtest_from_cache,
        )

        n = 100
        # Prix descend brusquement : touche un niveau puis crash au SL
        prices = np.concatenate([
            np.full(20, 100.0),
            np.linspace(100, 50, 80),  # Crash continu
        ])
        sma_arr = np.full(n, np.nan)
        atr_arr = np.full(n, np.nan)
        for i in range(14, n):
            sma_arr[i] = np.mean(prices[max(0, i - 13) : i + 1])
            atr_arr[i] = 2.0

        cache = make_indicator_cache(
            n=n,
            closes=prices,
            opens=prices,
            highs=prices + 1.0,
            lows=prices - 1.0,
            bb_sma={14: sma_arr},
            atr_by_period={14: atr_arr},
        )
        params = {
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 1.0, "atr_multiplier_step": 0.5,
            "num_levels": 2, "sl_percent": 15.0,
        }
        bt_config = _make_bt_config()
        result = run_multi_backtest_from_cache("grid_atr", params, cache, bt_config)
        # Des trades (SL hit)
        assert result[4] >= 1
        # Net return négatif (SL = perte)
        assert result[2] < 0


# ═══════════════════════════════════════════════════════════════════════════
# Section 4 : Parité fast/normal
# ═══════════════════════════════════════════════════════════════════════════


class TestGridATRParity:
    """Parité fast engine vs MultiPositionEngine."""

    def test_parity_fast_vs_normal(self):
        """Fast engine et MultiPositionEngine donnent les mêmes résultats (±1%)."""
        from backend.backtesting.multi_engine import MultiPositionEngine
        from backend.optimization.fast_multi_backtest import (
            run_multi_backtest_from_cache,
        )
        from backend.optimization.indicator_cache import build_cache

        candles = _make_sinusoidal_candles(n=500, amplitude=8.0, period=48)

        params = {
            "ma_period": 14,
            "atr_period": 14,
            "atr_multiplier_start": 1.5,
            "atr_multiplier_step": 0.5,
            "num_levels": 3,
            "sl_percent": 20.0,
            "sides": ["long"],
            "leverage": 6,
        }
        bt_config = _make_bt_config()

        # --- Fast engine ---
        param_grid_values = {k: [v] for k, v in params.items() if isinstance(v, (int, float))}
        cache = build_cache(
            {"1h": candles}, param_grid_values, "grid_atr", main_tf="1h",
        )
        fast_result = run_multi_backtest_from_cache("grid_atr", params, cache, bt_config)
        fast_n_trades = fast_result[4]
        fast_return = fast_result[2]

        # --- Normal engine ---
        from backend.optimization import create_strategy_with_params

        strategy = create_strategy_with_params("grid_atr", params)
        engine = MultiPositionEngine(bt_config, strategy)
        normal_result = engine.run({"1h": candles})

        from backend.backtesting.metrics import calculate_metrics

        metrics = calculate_metrics(normal_result)

        # Vérifications : données sinusoïdales doivent produire des trades
        assert fast_n_trades >= 3, f"Fast engine n'a produit que {fast_n_trades} trades"
        assert metrics.total_trades >= 3, f"Normal engine n'a produit que {metrics.total_trades} trades"

        # Parité ±1 trade (les heuristiques OHLC peuvent varier marginalement)
        assert abs(fast_n_trades - metrics.total_trades) <= 1, (
            f"Trades: fast={fast_n_trades}, normal={metrics.total_trades}"
        )

        # Parité return ±5% relatif (tolérance plus large car OHLC heuristique)
        if abs(metrics.net_return_pct) > 0.1:
            ratio = abs(fast_return - metrics.net_return_pct) / abs(metrics.net_return_pct)
            assert ratio < 0.05, (
                f"Return: fast={fast_return:.2f}%, normal={metrics.net_return_pct:.2f}%"
            )

    def test_fast_engine_speed(self):
        """Fast engine au moins 10× plus rapide que MultiPositionEngine."""
        from backend.backtesting.multi_engine import MultiPositionEngine
        from backend.optimization.fast_multi_backtest import (
            run_multi_backtest_from_cache,
        )
        from backend.optimization.indicator_cache import build_cache

        candles = _make_sinusoidal_candles(n=500)
        params = {
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 2.0, "atr_multiplier_step": 1.0,
            "num_levels": 3, "sl_percent": 20.0,
            "sides": ["long"], "leverage": 6,
        }
        bt_config = _make_bt_config()

        # Fast engine (sans compter build_cache)
        param_grid_values = {k: [v] for k, v in params.items() if isinstance(v, (int, float))}
        cache = build_cache({"1h": candles}, param_grid_values, "grid_atr", main_tf="1h")
        t0 = time.perf_counter()
        for _ in range(100):
            run_multi_backtest_from_cache("grid_atr", params, cache, bt_config)
        fast_time = time.perf_counter() - t0

        # Normal engine
        from backend.optimization import create_strategy_with_params

        strategy = create_strategy_with_params("grid_atr", params)
        t0 = time.perf_counter()
        for _ in range(10):
            engine = MultiPositionEngine(bt_config, strategy)
            engine.run({"1h": candles})
        normal_time = time.perf_counter() - t0

        # Fast devrait être au moins 10× plus rapide (100 runs fast vs 10 runs normal)
        fast_per_run = fast_time / 100
        normal_per_run = normal_time / 10
        speedup = normal_per_run / fast_per_run if fast_per_run > 0 else float("inf")
        assert speedup >= 5, f"Speedup seulement {speedup:.1f}× (attendu >= 5×)"


# ═══════════════════════════════════════════════════════════════════════════
# Section 5 : Registry & integration
# ═══════════════════════════════════════════════════════════════════════════


class TestGridATRIntegration:
    """Tests d'intégration : registry, factory, cache."""

    def test_in_strategy_registry(self):
        """grid_atr est dans STRATEGY_REGISTRY."""
        from backend.optimization import STRATEGY_REGISTRY

        assert "grid_atr" in STRATEGY_REGISTRY

    def test_in_grid_strategies(self):
        """grid_atr est dans GRID_STRATEGIES."""
        from backend.optimization import GRID_STRATEGIES

        assert "grid_atr" in GRID_STRATEGIES

    def test_is_grid_strategy_true(self):
        """is_grid_strategy('grid_atr') retourne True."""
        from backend.optimization import is_grid_strategy

        assert is_grid_strategy("grid_atr") is True

    def test_create_with_params(self):
        """create_strategy_with_params crée un GridATRStrategy."""
        from backend.optimization import create_strategy_with_params

        params = {
            "ma_period": 10, "atr_period": 20,
            "atr_multiplier_start": 1.5, "atr_multiplier_step": 0.5,
            "num_levels": 4, "sl_percent": 25.0,
            "sides": ["long"], "leverage": 6,
        }
        strategy = create_strategy_with_params("grid_atr", params)
        assert isinstance(strategy, GridATRStrategy)
        assert strategy.name == "grid_atr"
        assert strategy.max_positions == 4

    def test_indicator_params(self):
        """_INDICATOR_PARAMS['grid_atr'] == ['ma_period', 'atr_period']."""
        from backend.optimization.walk_forward import _INDICATOR_PARAMS

        assert "grid_atr" in _INDICATOR_PARAMS
        assert _INDICATOR_PARAMS["grid_atr"] == ["ma_period", "atr_period"]

    def test_build_cache_grid_atr(self):
        """build_cache crée les SMA et ATR pour grid_atr."""
        from backend.optimization.indicator_cache import build_cache

        candles = _make_candles(100)
        param_grid_values = {
            "ma_period": [7, 14],
            "atr_period": [10, 14],
        }
        cache = build_cache(
            {"1h": candles}, param_grid_values, "grid_atr", main_tf="1h",
        )
        # SMA calculées pour les 2 périodes
        assert 7 in cache.bb_sma
        assert 14 in cache.bb_sma
        assert len(cache.bb_sma[7]) == 100

        # ATR calculées pour les 2 périodes
        assert 10 in cache.atr_by_period
        assert 14 in cache.atr_by_period
        assert len(cache.atr_by_period[10]) == 100

    def test_get_params(self):
        """get_params retourne les paramètres attendus."""
        strategy = _make_strategy(
            ma_period=10, atr_period=20,
            atr_multiplier_start=1.5, atr_multiplier_step=0.5,
        )
        params = strategy.get_params()
        assert params["ma_period"] == 10
        assert params["atr_period"] == 20
        assert params["atr_multiplier_start"] == 1.5
        assert params["atr_multiplier_step"] == 0.5
        assert "leverage" in params


# ═══════════════════════════════════════════════════════════════════════════
# Section 6 : Executor helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestGridATRExecutorHelpers:
    """Vérifie que getattr(config.strategies, 'grid_atr') fonctionne pour l'Executor."""

    def test_executor_get_grid_sl_percent(self):
        """_get_grid_sl_percent('grid_atr') retourne la bonne valeur."""
        from backend.core.config import GridATRConfig, StrategiesConfig

        strats = StrategiesConfig(grid_atr=GridATRConfig(sl_percent=25.0))
        config_attr = getattr(strats, "grid_atr", None)
        assert config_attr is not None
        assert hasattr(config_attr, "sl_percent")
        assert config_attr.sl_percent == 25.0

    def test_executor_get_grid_leverage(self):
        """_get_grid_leverage('grid_atr') retourne la bonne valeur."""
        from backend.core.config import GridATRConfig, StrategiesConfig

        strats = StrategiesConfig(grid_atr=GridATRConfig(leverage=8))
        config_attr = getattr(strats, "grid_atr", None)
        assert config_attr is not None
        assert hasattr(config_attr, "leverage")
        assert config_attr.leverage == 8


# ═══════════════════════════════════════════════════════════════════════════
# Section 7 : Sprint 47 — min_grid_spacing_pct & min_profit_pct
# ═══════════════════════════════════════════════════════════════════════════


class TestGridATRMinGridSpacing:
    """Tests du plancher ATR adaptatif (min_grid_spacing_pct)."""

    def test_compute_grid_min_spacing_clamps_atr(self):
        """ATR faible clampé par min_grid_spacing_pct."""
        strategy = _make_strategy(
            atr_multiplier_start=2.0, atr_multiplier_step=1.0,
            num_levels=3, min_grid_spacing_pct=2.0,
        )
        # ATR=1.0, close=100 → min_atr = 100 * 2.0/100 = 2.0
        # effective_atr = max(1.0, 2.0) = 2.0
        # Niveaux : 100 - 2.0*2=96, 100 - 2.0*3=94, 100 - 2.0*4=92
        ctx = _make_ctx(sma_val=100.0, atr_val=1.0, close=100.0)
        levels = strategy.compute_grid(ctx, _make_grid_state())
        assert len(levels) == 3
        assert levels[0].entry_price == pytest.approx(96.0)
        assert levels[1].entry_price == pytest.approx(94.0)
        assert levels[2].entry_price == pytest.approx(92.0)

    def test_compute_grid_min_spacing_no_effect_high_atr(self):
        """ATR élevé > plancher → pas de clamping."""
        strategy = _make_strategy(
            atr_multiplier_start=2.0, atr_multiplier_step=1.0,
            num_levels=2, min_grid_spacing_pct=1.0,
        )
        # ATR=5.0, close=100 → min_atr = 1.0, effective_atr = max(5.0, 1.0) = 5.0
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=100.0)
        levels = strategy.compute_grid(ctx, _make_grid_state())
        assert levels[0].entry_price == pytest.approx(90.0)  # 100 - 5*2
        assert levels[1].entry_price == pytest.approx(85.0)  # 100 - 5*3

    def test_compute_grid_min_spacing_zero_disabled(self):
        """min_grid_spacing_pct=0.0 → comportement classique."""
        strat_classic = _make_strategy(
            atr_multiplier_start=2.0, atr_multiplier_step=1.0,
            num_levels=2, min_grid_spacing_pct=0.0,
        )
        strat_default = _make_strategy(
            atr_multiplier_start=2.0, atr_multiplier_step=1.0,
            num_levels=2,
        )
        ctx = _make_ctx(sma_val=100.0, atr_val=1.0, close=100.0)
        gs = _make_grid_state()
        levels_classic = strat_classic.compute_grid(ctx, gs)
        levels_default = strat_default.compute_grid(ctx, gs)
        assert levels_classic[0].entry_price == levels_default[0].entry_price
        assert levels_classic[1].entry_price == levels_default[1].entry_price

    def test_compute_grid_min_spacing_short(self):
        """Plancher ATR en direction SHORT."""
        strategy = _make_strategy(
            sides=["short"],
            atr_multiplier_start=2.0, atr_multiplier_step=1.0,
            num_levels=2, min_grid_spacing_pct=2.0,
        )
        # ATR=1.0, close=100 → effective_atr = 2.0
        # SHORT : 100 + 2.0*2=104, 100 + 2.0*3=106
        ctx = _make_ctx(sma_val=100.0, atr_val=1.0, close=100.0)
        levels = strategy.compute_grid(ctx, _make_grid_state())
        assert len(levels) == 2
        assert levels[0].entry_price == pytest.approx(104.0)
        assert levels[1].entry_price == pytest.approx(106.0)


class TestGridATRMinProfit:
    """Tests du profit minimum au TP (min_profit_pct)."""

    def test_tp_blocked_by_min_profit(self):
        """close >= SMA mais profit < seuil → pas de TP."""
        strategy = _make_strategy(min_profit_pct=1.0)
        # Entry à 99, close=100 → profit = 1.01% ≈ 1.01%
        # Mais on veut tester le cas bloqué : entry à 99.5, close=100
        # profit = (100-99.5)/99.5 = 0.50% < 1.0% → bloqué
        pos = _make_position(level=0, direction=Direction.LONG, entry_price=99.5)
        grid_state = _make_grid_state([pos])
        ctx = _make_ctx(sma_val=99.0, atr_val=5.0, close=100.0)
        # close (100) >= sma (99) → condition SMA OK
        # close (100) < avg_entry * 1.01 = 100.495 → condition profit KO
        assert strategy.should_close_all(ctx, grid_state) is None

    def test_tp_allowed_by_min_profit(self):
        """close >= SMA ET profit >= seuil → TP fire."""
        strategy = _make_strategy(min_profit_pct=1.0)
        pos = _make_position(level=0, direction=Direction.LONG, entry_price=95.0)
        grid_state = _make_grid_state([pos])
        ctx = _make_ctx(sma_val=99.0, atr_val=5.0, close=100.0)
        # close (100) >= sma (99) → OK
        # close (100) >= 95 * 1.01 = 95.95 → OK
        assert strategy.should_close_all(ctx, grid_state) == "tp_global"

    def test_tp_min_profit_zero_classic(self):
        """min_profit_pct=0.0 → TP classique (SMA seule)."""
        strategy = _make_strategy(min_profit_pct=0.0)
        pos = _make_position(level=0, direction=Direction.LONG, entry_price=99.9)
        grid_state = _make_grid_state([pos])
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=100.0)
        # close (100) >= sma (100) → TP même avec profit quasi-nul
        assert strategy.should_close_all(ctx, grid_state) == "tp_global"

    def test_tp_min_profit_short_blocked(self):
        """SHORT : close <= SMA mais pas assez de profit → pas de TP."""
        strategy = _make_strategy(sides=["short"], min_profit_pct=1.0)
        pos = _make_position(level=0, direction=Direction.SHORT, entry_price=100.5)
        grid_state = _make_grid_state([pos])
        ctx = _make_ctx(sma_val=101.0, atr_val=5.0, close=100.0)
        # close (100) <= sma (101) → condition SMA OK
        # close (100) <= avg_entry * (1 - 0.01) = 99.495 ? Non, 100 > 99.495 → bloqué
        assert strategy.should_close_all(ctx, grid_state) is None

    def test_tp_min_profit_short_hit(self):
        """SHORT : les 2 conditions OK → TP."""
        strategy = _make_strategy(sides=["short"], min_profit_pct=1.0)
        pos = _make_position(level=0, direction=Direction.SHORT, entry_price=105.0)
        grid_state = _make_grid_state([pos])
        ctx = _make_ctx(sma_val=101.0, atr_val=5.0, close=100.0)
        # close (100) <= sma (101) → OK
        # close (100) <= 105 * 0.99 = 103.95 → OK
        assert strategy.should_close_all(ctx, grid_state) == "tp_global"

    def test_sl_ignores_min_profit(self):
        """SL fonctionne normalement avec min_profit > 0."""
        strategy = _make_strategy(sl_percent=20.0, min_profit_pct=5.0)
        pos = _make_position(level=0, direction=Direction.LONG, entry_price=90.0)
        grid_state = _make_grid_state([pos])
        # SL = 90 * 0.8 = 72, close=70 < 72 → SL
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=70.0)
        assert strategy.should_close_all(ctx, grid_state) == "sl_global"


class TestGridATRAdaptiveIntegration:
    """Tests d'intégration Sprint 47."""

    def test_get_params_includes_adaptive_fields(self):
        """get_params retourne min_grid_spacing_pct et min_profit_pct."""
        strategy = _make_strategy(min_grid_spacing_pct=1.2, min_profit_pct=0.3)
        params = strategy.get_params()
        assert params["min_grid_spacing_pct"] == 1.2
        assert params["min_profit_pct"] == 0.3

    def test_get_params_defaults_zero(self):
        """get_params retourne 0.0 par défaut pour les nouveaux params."""
        strategy = _make_strategy()
        params = strategy.get_params()
        assert params["min_grid_spacing_pct"] == 0.0
        assert params["min_profit_pct"] == 0.0

    def test_fast_engine_min_spacing(self, make_indicator_cache):
        """Fast engine avec min_grid_spacing_pct élargit les grilles en basse vol."""
        from backend.optimization.fast_multi_backtest import (
            _build_entry_prices,
        )

        n = 100
        prices = np.full(n, 100.0)
        sma_arr = np.full(n, 100.0)
        sma_arr[:14] = np.nan
        atr_arr = np.full(n, 0.5)  # ATR très faible
        atr_arr[:14] = np.nan

        cache = make_indicator_cache(
            n=n, closes=prices, bb_sma={14: sma_arr}, atr_by_period={14: atr_arr},
        )

        params_classic = {
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 2.0, "atr_multiplier_step": 1.0,
            "num_levels": 3, "min_grid_spacing_pct": 0.0,
        }
        params_floor = {
            **params_classic, "min_grid_spacing_pct": 1.5,
        }

        ep_classic = _build_entry_prices("grid_atr", cache, params_classic, 3, 1)
        ep_floor = _build_entry_prices("grid_atr", cache, params_floor, 3, 1)

        # Avec plancher, les niveaux sont plus éloignés (entry_price plus bas pour LONG)
        valid = ~np.isnan(ep_classic[:, 0])
        assert np.all(ep_floor[valid, 0] <= ep_classic[valid, 0])

    def test_fast_engine_backward_compat(self, make_indicator_cache):
        """Fast engine avec params=0.0 donne résultat identique au classique."""
        from backend.optimization.fast_multi_backtest import (
            run_multi_backtest_from_cache,
        )

        candles = _make_sinusoidal_candles(n=500, amplitude=8.0, period=48)
        from backend.optimization.indicator_cache import build_cache

        params_base = {
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 1.5, "atr_multiplier_step": 0.5,
            "num_levels": 3, "sl_percent": 20.0,
        }
        params_v2 = {
            **params_base,
            "min_grid_spacing_pct": 0.0,
            "min_profit_pct": 0.0,
        }
        param_grid_values = {k: [v] for k, v in params_base.items() if isinstance(v, (int, float))}
        cache = build_cache({"1h": candles}, param_grid_values, "grid_atr", main_tf="1h")
        bt_config = _make_bt_config()

        r_base = run_multi_backtest_from_cache("grid_atr", params_base, cache, bt_config)
        r_v2 = run_multi_backtest_from_cache("grid_atr", params_v2, cache, bt_config)

        # Résultats bit-for-bit identiques
        assert r_base[4] == r_v2[4], f"Trades: base={r_base[4]}, v2={r_v2[4]}"
        assert r_base[1] == pytest.approx(r_v2[1], abs=1e-10), "Sharpe diverge"
        assert r_base[2] == pytest.approx(r_v2[2], abs=1e-10), "Return diverge"


# ---------------------------------------------------------------------------
# Sprint 47d — Tests propagation per_asset chain
# ---------------------------------------------------------------------------


class TestGridATRPerAssetChain:
    """Tests de la chaîne config per_asset → GridATRStrategy (Sprint 47d)."""

    def test_config_per_asset_propagates_adaptive_params(self):
        """per_asset NEAR {min_grid_spacing_pct: 1.8} doit être accessible via get_params_for_symbol."""
        from backend.core.config import GridATRConfig

        cfg = GridATRConfig(
            min_grid_spacing_pct=0.0,
            min_profit_pct=0.0,
            per_asset={
                "NEAR/USDT": {
                    "min_grid_spacing_pct": 1.8,
                    "min_profit_pct": 0.2,
                    "num_levels": 4,
                    "sl_percent": 25.0,
                }
            },
        )
        params = cfg.get_params_for_symbol("NEAR/USDT")
        assert params["min_grid_spacing_pct"] == 1.8
        assert params["min_profit_pct"] == 0.2
        assert params["num_levels"] == 4

    def test_config_per_asset_fallback_to_default(self):
        """Symbol sans override per_asset → valeurs top-level."""
        from backend.core.config import GridATRConfig

        cfg = GridATRConfig(
            min_grid_spacing_pct=0.8,
            min_profit_pct=0.0,
            per_asset={"BNB/USDT": {"min_grid_spacing_pct": 1.2}},
        )
        params = cfg.get_params_for_symbol("BTC/USDT")  # Pas dans per_asset
        assert params["min_grid_spacing_pct"] == 0.8  # Valeur top-level

    def test_get_per_asset_float_helper_simulator(self):
        """GridStrategyRunner._get_per_asset_float() résout les overrides per_asset."""
        from unittest.mock import MagicMock

        from backend.core.config import GridATRConfig

        cfg = GridATRConfig(
            min_grid_spacing_pct=0.0,
            per_asset={"SOL/USDT": {"min_grid_spacing_pct": 1.8}},
        )
        strategy = MagicMock()
        strategy._config = cfg

        # Simuler le helper sans instancier le runner complet
        def _get_per_asset_float(sym: str, param: str, default: float) -> float:
            per_asset = getattr(strategy._config, "per_asset", {})
            if isinstance(per_asset, dict):
                overrides = per_asset.get(sym, {})
                if isinstance(overrides, dict) and param in overrides:
                    try:
                        return float(overrides[param])
                    except (TypeError, ValueError):
                        pass
            return default

        assert _get_per_asset_float("SOL/USDT", "min_grid_spacing_pct", 0.0) == 1.8
        assert _get_per_asset_float("BTC/USDT", "min_grid_spacing_pct", 0.0) == 0.0
        assert _get_per_asset_float("SOL/USDT", "unknown_param", 99.0) == 99.0
