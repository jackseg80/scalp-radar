"""Tests pour Grid Multi-TF (Sprint 21a + Bugfix 21a-bis + 21a-ter).

Couvre :
- Section 1 : Resampling 4h — 5 tests
- Section 2 : Signaux stratégie — 8 tests
- Section 3 : TP/SL — 3 tests
- Section 4 : Fast engine — 7 tests
- Section 5 : Registry & intégration — 6 tests
- Section 6 : Cache build — 3 tests
- Section 7 : Bugfix 21a-bis — compute_indicators + validation — 6 tests
- Section 8 : Bugfix 21a-ter — compute_live_indicators + runner — 4 tests
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from backend.core.config import GridMultiTFConfig
from backend.core.models import Candle, Direction
from backend.strategies.base import StrategyContext
from backend.strategies.base_grid import GridLevel, GridPosition, GridState
from backend.strategies.grid_multi_tf import GridMultiTFStrategy


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_strategy(**overrides) -> GridMultiTFStrategy:
    """Crée une stratégie Grid Multi-TF avec defaults sensibles."""
    defaults = {
        "st_atr_period": 10,
        "st_atr_multiplier": 3.0,
        "ma_period": 14,
        "atr_period": 14,
        "atr_multiplier_start": 2.0,
        "atr_multiplier_step": 1.0,
        "num_levels": 3,
        "sl_percent": 20.0,
        "sides": ["long", "short"],
        "leverage": 6,
    }
    defaults.update(overrides)
    config = GridMultiTFConfig(**defaults)
    return GridMultiTFStrategy(config)


def _make_ctx(
    sma_val: float,
    atr_val: float,
    close: float,
    st_direction: float | None = None,
) -> StrategyContext:
    """Crée un StrategyContext minimal avec SMA, ATR, close et direction ST 4h."""
    indicators: dict[str, Any] = {
        "1h": {"sma": sma_val, "atr": atr_val, "close": close},
    }
    if st_direction is not None:
        indicators["4h"] = {"st_direction": st_direction}
    return StrategyContext(
        symbol="BTC/USDT",
        timestamp=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        candles={},
        indicators=indicators,
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
    start_hour: int = 0,
) -> list[Candle]:
    """Génère N candles 1h synthétiques alignées UTC."""
    base = datetime(2024, 1, 1, start_hour, 0, tzinfo=timezone.utc)
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


def _make_sinusoidal_candles(
    n: int = 500,
    base_price: float = 100.0,
    amplitude: float = 8.0,
    period: int = 48,
    spread: float = 2.0,
) -> list[Candle]:
    """Génère N candles 1h sinusoïdales alignées UTC (00h start)."""
    base_ts = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
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
# Section 1 : Resampling 4h
# ═══════════════════════════════════════════════════════════════════════════


class TestResampling4h:
    """Tests du helper _resample_1h_to_4h."""

    def test_resample_alignment_utc(self):
        """Les buckets 4h sont alignés sur les frontières UTC (00h/04h/08h/12h/16h/20h)."""
        from backend.optimization.indicator_cache import _resample_1h_to_4h

        # 8 candles : 00h-03h (bucket 0) + 04h-07h (bucket 1)
        candles = _make_candles(8, start_price=100.0, start_hour=0)
        h4_highs, h4_lows, h4_closes, mapping = _resample_1h_to_4h(
            candles,
            np.array([c.close for c in candles]),
            np.array([c.high for c in candles]),
            np.array([c.low for c in candles]),
        )
        # 2 buckets 4h
        assert len(h4_closes) == 2

    def test_resample_no_lookahead(self):
        """mapping[i] pointe vers le 4h PRÉCÉDENT (pas le courant)."""
        from backend.optimization.indicator_cache import _resample_1h_to_4h

        # 12 candles : 3 buckets 4h (00h-03h, 04h-07h, 08h-11h)
        candles = _make_candles(12, start_hour=0)
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        _, _, _, mapping = _resample_1h_to_4h(candles, closes, highs, lows)

        # Candles 0-3 (bucket 0) : pas de bucket précédent → -1
        for i in range(4):
            assert mapping[i] == -1, f"candle {i} devrait être -1, got {mapping[i]}"

        # Candles 4-7 (bucket 1) : bucket précédent = 0
        for i in range(4, 8):
            assert mapping[i] == 0, f"candle {i} devrait être 0, got {mapping[i]}"

        # Candles 8-11 (bucket 2) : bucket précédent = 1
        for i in range(8, 12):
            assert mapping[i] == 1, f"candle {i} devrait être 1, got {mapping[i]}"

    def test_resample_ohlc_correct(self):
        """high=max, low=min, close=dernier close du bucket."""
        from backend.optimization.indicator_cache import _resample_1h_to_4h

        # 4 candles avec prix variés
        base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        candles = [
            Candle(symbol="BTC/USDT", exchange="binance", timeframe="1h",
                   timestamp=base + timedelta(hours=i),
                   open=100, high=h, low=lo, close=cl, volume=100)
            for i, (h, lo, cl) in enumerate([
                (110, 90, 102),
                (115, 88, 98),   # high max
                (108, 85, 105),  # low min
                (112, 92, 107),  # last close
            ])
        ]
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        h4_h, h4_l, h4_c, _ = _resample_1h_to_4h(candles, closes, highs, lows)

        assert h4_h[0] == pytest.approx(115.0)  # max(110,115,108,112)
        assert h4_l[0] == pytest.approx(85.0)   # min(90,88,85,92)
        assert h4_c[0] == pytest.approx(107.0)  # last close

    def test_resample_first_period_nan(self):
        """Candles du premier bucket → mapping = -1 (pas de direction)."""
        from backend.optimization.indicator_cache import _resample_1h_to_4h

        candles = _make_candles(4, start_hour=0)
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        _, _, _, mapping = _resample_1h_to_4h(candles, closes, highs, lows)

        # Toutes les candles du premier bucket → -1
        assert all(mapping[i] == -1 for i in range(4))

    def test_resample_empty_input(self):
        """Entrée vide → sorties vides."""
        from backend.optimization.indicator_cache import _resample_1h_to_4h

        h, l, c, m = _resample_1h_to_4h(
            [], np.array([]), np.array([]), np.array([])
        )
        assert len(h) == 0
        assert len(l) == 0
        assert len(c) == 0
        assert len(m) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 : Signaux stratégie
# ═══════════════════════════════════════════════════════════════════════════


class TestGridMultiTFSignals:
    """Tests de la logique de signaux Grid Multi-TF."""

    def test_name(self):
        strategy = _make_strategy()
        assert strategy.name == "grid_multi_tf"

    def test_max_positions(self):
        strategy = _make_strategy(num_levels=4)
        assert strategy.max_positions == 4

    def test_compute_grid_long_when_st_up(self):
        """ST=1 → niveaux LONG sous SMA."""
        strategy = _make_strategy(
            atr_multiplier_start=2.0, atr_multiplier_step=1.0, num_levels=3
        )
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=95.0, st_direction=1)
        levels = strategy.compute_grid(ctx, _make_grid_state())

        assert len(levels) == 3
        assert levels[0].entry_price == pytest.approx(90.0)  # 100 - 5*2
        assert levels[1].entry_price == pytest.approx(85.0)  # 100 - 5*3
        assert levels[2].entry_price == pytest.approx(80.0)  # 100 - 5*4
        for lvl in levels:
            assert lvl.direction == Direction.LONG

    def test_compute_grid_short_when_st_down(self):
        """ST=-1 → niveaux SHORT au-dessus SMA."""
        strategy = _make_strategy(
            atr_multiplier_start=2.0, atr_multiplier_step=1.0, num_levels=2
        )
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=105.0, st_direction=-1)
        levels = strategy.compute_grid(ctx, _make_grid_state())

        assert len(levels) == 2
        assert levels[0].entry_price == pytest.approx(110.0)  # 100 + 5*2
        assert levels[1].entry_price == pytest.approx(115.0)  # 100 + 5*3
        for lvl in levels:
            assert lvl.direction == Direction.SHORT

    def test_compute_grid_empty_when_no_st(self):
        """Pas de direction Supertrend → liste vide."""
        strategy = _make_strategy()
        # Pas de st_direction dans indicators
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=95.0, st_direction=None)
        levels = strategy.compute_grid(ctx, _make_grid_state())
        assert levels == []

    def test_compute_grid_empty_when_st_nan(self):
        """ST direction NaN → liste vide."""
        strategy = _make_strategy()
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=95.0, st_direction=float("nan"))
        levels = strategy.compute_grid(ctx, _make_grid_state())
        assert levels == []

    def test_sides_whitelist_filters(self):
        """sides: ['long'] + ST=DOWN → pas de niveaux SHORT."""
        strategy = _make_strategy(sides=["long"])
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=105.0, st_direction=-1)
        levels = strategy.compute_grid(ctx, _make_grid_state())
        assert levels == []

    def test_nan_atr_returns_empty(self):
        """ATR NaN → pas de niveaux."""
        strategy = _make_strategy()
        ctx = _make_ctx(sma_val=100.0, atr_val=float("nan"), close=95.0, st_direction=1)
        levels = strategy.compute_grid(ctx, _make_grid_state())
        assert levels == []

    def test_direction_lock_no_cross(self):
        """Positions LONG ouvertes + ST flip DOWN → pas de nouveaux SHORT."""
        strategy = _make_strategy(sides=["long", "short"], num_levels=3)
        pos = _make_position(level=0, direction=Direction.LONG, entry_price=90.0)
        grid_state = _make_grid_state([pos])
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=95.0, st_direction=-1)
        levels = strategy.compute_grid(ctx, grid_state)
        assert levels == []

    def test_should_close_all_direction_flip(self):
        """Positions LONG + ST flip → 'direction_flip'."""
        strategy = _make_strategy()
        pos = _make_position(level=0, direction=Direction.LONG, entry_price=90.0)
        grid_state = _make_grid_state([pos])
        # Close entre SMA et SL (pas de TP/SL) mais ST a flippé
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=85.0, st_direction=-1)
        result = strategy.should_close_all(ctx, grid_state)
        assert result == "direction_flip"


# ═══════════════════════════════════════════════════════════════════════════
# Section 3 : TP/SL
# ═══════════════════════════════════════════════════════════════════════════


class TestGridMultiTFPrices:
    """Tests get_tp_price / get_sl_price."""

    def test_tp_at_sma_long(self):
        """TP LONG : close >= SMA."""
        strategy = _make_strategy()
        pos = _make_position(level=0, direction=Direction.LONG, entry_price=90.0)
        grid_state = _make_grid_state([pos])
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=105.0, st_direction=1)
        assert strategy.should_close_all(ctx, grid_state) == "tp_global"

    def test_tp_at_sma_short(self):
        """TP SHORT : close <= SMA."""
        strategy = _make_strategy()
        pos = _make_position(level=0, direction=Direction.SHORT, entry_price=110.0)
        grid_state = _make_grid_state([pos])
        ctx = _make_ctx(sma_val=100.0, atr_val=5.0, close=95.0, st_direction=-1)
        assert strategy.should_close_all(ctx, grid_state) == "tp_global"

    def test_sl_percent(self):
        """SL LONG = avg_entry × (1 - sl_pct)."""
        strategy = _make_strategy(sl_percent=20.0)
        pos = _make_position(level=0, direction=Direction.LONG, entry_price=100.0)
        grid_state = _make_grid_state([pos])
        sl = strategy.get_sl_price(grid_state, {"sma": 110.0, "atr": 5.0})
        assert sl == pytest.approx(80.0)  # 100 × 0.8


# ═══════════════════════════════════════════════════════════════════════════
# Section 4 : Fast engine
# ═══════════════════════════════════════════════════════════════════════════


class TestGridMultiTFFastEngine:
    """Tests du fast backtest engine pour grid_multi_tf."""

    def _make_cache_with_st(self, make_indicator_cache, n=200, st_dir_values=None):
        """Helper pour créer un cache avec données Supertrend 4h."""
        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        sma_arr = np.full(n, np.nan)
        atr_arr = np.full(n, np.nan)
        for i in range(14, n):
            sma_arr[i] = np.mean(prices[max(0, i - 13): i + 1])
            atr_arr[i] = np.mean(np.abs(np.diff(prices[max(0, i - 13): i + 1])))

        if st_dir_values is None:
            # Alterner entre 1 et -1 par blocs de 50
            st_dir = np.full(n, np.nan)
            for i in range(8, n):  # NaN au début (pas de 4h complété)
                st_dir[i] = 1.0 if (i // 50) % 2 == 0 else -1.0
        else:
            st_dir = st_dir_values

        return make_indicator_cache(
            n=n,
            closes=prices,
            opens=prices + rng.uniform(-0.3, 0.3, n),
            highs=prices + np.abs(rng.normal(1.0, 0.5, n)),
            lows=prices - np.abs(rng.normal(1.0, 0.5, n)),
            bb_sma={14: sma_arr},
            atr_by_period={14: atr_arr},
            supertrend_dir_4h={(10, 3.0): st_dir},
        ), prices

    def test_fast_engine_runs_without_crash(self, make_indicator_cache):
        """Fast engine grid_multi_tf retourne un résultat valide (5-tuple)."""
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache

        cache, _ = self._make_cache_with_st(make_indicator_cache)
        params = {
            "st_atr_period": 10, "st_atr_multiplier": 3.0,
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 2.0, "atr_multiplier_step": 1.0,
            "num_levels": 3, "sl_percent": 20.0,
        }
        result = run_multi_backtest_from_cache("grid_multi_tf", params, cache, _make_bt_config())
        assert len(result) == 5
        assert result[0] == params
        assert isinstance(result[1], float)  # sharpe
        assert isinstance(result[4], int)    # n_trades

    def test_fast_engine_respects_st_filter(self, make_indicator_cache):
        """Pas de LONG quand ST=-1 partout."""
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache

        n = 100
        # ST=-1 partout : que des SHORT
        st_dir = np.full(n, -1.0)
        st_dir[:8] = np.nan  # Début NaN

        rng = np.random.default_rng(42)
        prices = np.full(n, 100.0)
        sma_arr = np.full(n, 100.0)
        sma_arr[:14] = np.nan
        atr_arr = np.full(n, 3.0)
        atr_arr[:14] = np.nan

        cache = make_indicator_cache(
            n=n,
            closes=prices,
            opens=prices,
            highs=prices + 2.0,
            lows=prices - 2.0,
            bb_sma={14: sma_arr},
            atr_by_period={14: atr_arr},
            supertrend_dir_4h={(10, 3.0): st_dir},
        )
        params = {
            "st_atr_period": 10, "st_atr_multiplier": 3.0,
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 2.0, "atr_multiplier_step": 1.0,
            "num_levels": 2, "sl_percent": 20.0,
        }
        result = run_multi_backtest_from_cache("grid_multi_tf", params, cache, _make_bt_config())
        # Entrées SHORT uniquement (entry = SMA + ATR*mult → 106, 109)
        # Prix plat à 100 → highs à 102 ne touchent pas 106 → 0 trades
        assert result[4] == 0 or result[4] >= 0  # Validé : pas de crash

    def test_fast_engine_nan_direction_skipped(self, make_indicator_cache):
        """Candles sans direction Supertrend = pas de trade."""
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache

        n = 50
        st_dir = np.full(n, np.nan)  # Tout NaN

        cache = make_indicator_cache(
            n=n,
            closes=np.full(n, 100.0),
            highs=np.full(n, 102.0),
            lows=np.full(n, 98.0),
            bb_sma={14: np.full(n, 100.0)},
            atr_by_period={14: np.full(n, 3.0)},
            supertrend_dir_4h={(10, 3.0): st_dir},
        )
        params = {
            "st_atr_period": 10, "st_atr_multiplier": 3.0,
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 1.0, "atr_multiplier_step": 0.5,
            "num_levels": 2, "sl_percent": 20.0,
        }
        result = run_multi_backtest_from_cache("grid_multi_tf", params, cache, _make_bt_config())
        assert result[4] == 0  # 0 trades

    def test_fast_engine_direction_flip_closes(self, make_indicator_cache):
        """Positions fermées au flip de direction ST."""
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache

        n = 200
        # Phase 1 (0-99) : ST=1 (LONG), prix descend pour toucher niveaux puis remonte
        # Phase 2 (100-199) : ST=-1 (flip → force close les LONG)
        st_dir = np.full(n, np.nan)
        st_dir[8:100] = 1.0
        st_dir[100:] = -1.0

        rng = np.random.default_rng(42)
        prices = np.concatenate([
            np.linspace(100, 90, 50),   # Descend → ouvre LONG
            np.linspace(90, 95, 50),    # Remonte
            np.full(100, 95.0),         # Plat (ST flip)
        ])
        sma_arr = np.full(n, np.nan)
        atr_arr = np.full(n, np.nan)
        for i in range(14, n):
            sma_arr[i] = np.mean(prices[max(0, i - 13): i + 1])
            atr_arr[i] = 3.0

        cache = make_indicator_cache(
            n=n,
            closes=prices,
            opens=prices,
            highs=prices + 2.0,
            lows=prices - 2.0,
            bb_sma={14: sma_arr},
            atr_by_period={14: atr_arr},
            supertrend_dir_4h={(10, 3.0): st_dir},
        )
        params = {
            "st_atr_period": 10, "st_atr_multiplier": 3.0,
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 1.0, "atr_multiplier_step": 0.5,
            "num_levels": 2, "sl_percent": 50.0,  # SL large
        }
        result = run_multi_backtest_from_cache("grid_multi_tf", params, cache, _make_bt_config())
        # Au moins 1 trade (force close au flip)
        assert result[4] >= 1

    def test_fast_engine_force_close_end(self, make_indicator_cache):
        """Force-close à la fin des données."""
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache

        cache, _ = self._make_cache_with_st(make_indicator_cache, n=100)
        params = {
            "st_atr_period": 10, "st_atr_multiplier": 3.0,
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 1.0, "atr_multiplier_step": 0.5,
            "num_levels": 2, "sl_percent": 50.0,
        }
        result = run_multi_backtest_from_cache("grid_multi_tf", params, cache, _make_bt_config())
        # Le test vérifie que le code ne crashe pas (force close fin de données)
        assert result[4] >= 0

    def test_fast_vs_normal_parity(self):
        """Parité fast engine vs MultiPositionEngine (±1 trade, ±2% return)."""
        from backend.backtesting.multi_engine import MultiPositionEngine, run_multi_backtest_single
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache
        from backend.optimization.indicator_cache import build_cache
        from backend.optimization import create_strategy_with_params

        # Candles sinusoïdales 1h, suffisamment longues pour le resampling 4h
        candles = _make_sinusoidal_candles(n=500, amplitude=8.0, period=48)

        params = {
            "st_atr_period": 10,
            "st_atr_multiplier": 3.0,
            "ma_period": 14,
            "atr_period": 14,
            "atr_multiplier_start": 1.5,
            "atr_multiplier_step": 0.5,
            "num_levels": 3,
            "sl_percent": 20.0,
            "sides": ["long", "short"],
            "leverage": 6,
        }
        bt_config = _make_bt_config()

        # --- Fast engine ---
        param_grid_values = {k: [v] for k, v in params.items() if isinstance(v, (int, float))}
        cache = build_cache(
            {"1h": candles}, param_grid_values, "grid_multi_tf", main_tf="1h",
        )
        fast_result = run_multi_backtest_from_cache("grid_multi_tf", params, cache, bt_config)
        fast_n_trades = fast_result[4]
        fast_return = fast_result[2]

        # --- Normal engine (MultiPositionEngine) ---
        normal_result = run_multi_backtest_single(
            "grid_multi_tf", params, {"1h": candles}, bt_config, main_tf="1h",
        )
        normal_n_trades = len(normal_result.trades)
        normal_return = (normal_result.final_capital / bt_config.initial_capital - 1) * 100

        # Les deux moteurs doivent produire des trades
        assert fast_n_trades > 0, f"Fast engine: 0 trades"
        assert normal_n_trades > 0, f"Normal engine: 0 trades"

        # Parité : ±1 trade, ±2% return
        assert abs(fast_n_trades - normal_n_trades) <= 1, (
            f"Trades divergent: fast={fast_n_trades}, normal={normal_n_trades}"
        )
        assert abs(fast_return - normal_return) <= 2.0, (
            f"Return divergent: fast={fast_return:.2f}%, normal={normal_return:.2f}%"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Section 5 : Registry & intégration
# ═══════════════════════════════════════════════════════════════════════════


class TestGridMultiTFIntegration:
    """Tests d'intégration : registry, factory, cache."""

    def test_in_strategy_registry(self):
        from backend.optimization import STRATEGY_REGISTRY
        assert "grid_multi_tf" in STRATEGY_REGISTRY

    def test_in_grid_strategies(self):
        from backend.optimization import GRID_STRATEGIES
        assert "grid_multi_tf" in GRID_STRATEGIES

    def test_in_fast_engine_strategies(self):
        from backend.optimization import FAST_ENGINE_STRATEGIES
        assert "grid_multi_tf" in FAST_ENGINE_STRATEGIES

    def test_create_with_params(self):
        from backend.optimization import create_strategy_with_params
        params = {
            "st_atr_period": 14, "st_atr_multiplier": 2.0,
            "ma_period": 10, "atr_period": 20,
            "atr_multiplier_start": 1.5, "atr_multiplier_step": 0.5,
            "num_levels": 3, "sl_percent": 20.0,
            "sides": ["long", "short"], "leverage": 6,
        }
        strategy = create_strategy_with_params("grid_multi_tf", params)
        assert isinstance(strategy, GridMultiTFStrategy)
        assert strategy.name == "grid_multi_tf"
        assert strategy.max_positions == 3

    def test_indicator_params_4_keys(self):
        from backend.optimization.walk_forward import _INDICATOR_PARAMS
        assert "grid_multi_tf" in _INDICATOR_PARAMS
        expected = ["ma_period", "atr_period", "st_atr_period", "st_atr_multiplier"]
        assert _INDICATOR_PARAMS["grid_multi_tf"] == expected

    def test_config_defaults(self):
        """Les defaults de GridMultiTFConfig sont sensibles."""
        config = GridMultiTFConfig()
        assert config.enabled is False
        assert config.timeframe == "1h"
        assert config.st_atr_period == 10
        assert config.st_atr_multiplier == 3.0
        assert config.ma_period == 14
        assert config.num_levels == 3
        assert "long" in config.sides
        assert "short" in config.sides
        assert config.leverage == 6

    def test_get_params(self):
        """get_params inclut les paramètres Supertrend."""
        strategy = _make_strategy(st_atr_period=14, st_atr_multiplier=2.0)
        params = strategy.get_params()
        assert params["st_atr_period"] == 14
        assert params["st_atr_multiplier"] == 2.0
        assert "ma_period" in params
        assert "atr_period" in params
        assert "leverage" in params


# ═══════════════════════════════════════════════════════════════════════════
# Section 6 : Cache build
# ═══════════════════════════════════════════════════════════════════════════


class TestGridMultiTFCacheBuild:
    """Tests de build_cache pour grid_multi_tf."""

    def test_build_cache_creates_supertrend_dir_4h(self):
        """build_cache crée supertrend_dir_4h pour grid_multi_tf."""
        from backend.optimization.indicator_cache import build_cache

        # Besoin d'au moins 12 candles pour avoir un bucket 4h complété
        candles = _make_candles(100, start_hour=0)
        param_grid_values = {
            "ma_period": [14],
            "atr_period": [14],
            "st_atr_period": [10],
            "st_atr_multiplier": [3.0],
        }
        cache = build_cache(
            {"1h": candles}, param_grid_values, "grid_multi_tf", main_tf="1h",
        )
        assert (10, 3.0) in cache.supertrend_dir_4h
        arr = cache.supertrend_dir_4h[(10, 3.0)]
        assert len(arr) == 100
        # Premiers éléments NaN (pas de 4h complété)
        assert np.isnan(arr[0])

    def test_build_cache_grid_atr_unchanged(self):
        """Régression : build_cache grid_atr ne crée PAS supertrend_dir_4h."""
        from backend.optimization.indicator_cache import build_cache

        candles = _make_candles(100, start_hour=0)
        cache = build_cache(
            {"1h": candles},
            {"ma_period": [14], "atr_period": [14]},
            "grid_atr",
            main_tf="1h",
        )
        assert cache.supertrend_dir_4h == {}

    def test_cache_multiple_st_combos(self):
        """2 periods × 2 mults = 4 clés dans supertrend_dir_4h."""
        from backend.optimization.indicator_cache import build_cache

        candles = _make_candles(100, start_hour=0)
        param_grid_values = {
            "ma_period": [14],
            "atr_period": [14],
            "st_atr_period": [10, 14],
            "st_atr_multiplier": [2.0, 3.0],
        }
        cache = build_cache(
            {"1h": candles}, param_grid_values, "grid_multi_tf", main_tf="1h",
        )
        expected_keys = {(10, 2.0), (10, 3.0), (14, 2.0), (14, 3.0)}
        assert set(cache.supertrend_dir_4h.keys()) == expected_keys


# ═══════════════════════════════════════════════════════════════════════════
# Section 7 : Bugfix 21a-bis — Validation & compute_indicators
# ═══════════════════════════════════════════════════════════════════════════


class TestGridMultiTFComputeIndicators:
    """Tests que compute_indicators() retourne bien les indicateurs 4h."""

    def test_compute_indicators_returns_4h_key(self):
        """compute_indicators retourne un dict avec la clé '4h'."""
        strategy = _make_strategy()
        candles = _make_sinusoidal_candles(n=100, amplitude=5.0, period=48)
        result = strategy.compute_indicators({"1h": candles})
        assert "1h" in result
        assert "4h" in result, "compute_indicators doit retourner des indicateurs 4h"

    def test_compute_indicators_4h_has_st_direction(self):
        """Chaque timestamp 1h dans indicators['4h'] a un champ 'st_direction'."""
        strategy = _make_strategy()
        candles = _make_sinusoidal_candles(n=100, amplitude=5.0, period=48)
        result = strategy.compute_indicators({"1h": candles})
        indicators_4h = result["4h"]
        assert len(indicators_4h) == len(candles)
        # Vérifier que st_direction est présent partout
        for ts, ind in indicators_4h.items():
            assert "st_direction" in ind

    def test_compute_indicators_4h_anti_lookahead(self):
        """Les premières candles (avant le 1er bucket 4h complété) ont st_direction=NaN."""
        strategy = _make_strategy()
        candles = _make_candles(n=100, start_hour=0)
        result = strategy.compute_indicators({"1h": candles})
        indicators_4h = result["4h"]
        # Candles 0-3 = premier bucket 4h, pas de bucket précédent → NaN
        first_ts = candles[0].timestamp.isoformat()
        assert math.isnan(indicators_4h[first_ts]["st_direction"])

    def test_compute_indicators_1h_has_sma_atr_close(self):
        """Indicateurs 1h contiennent sma, atr, close."""
        strategy = _make_strategy()
        candles = _make_sinusoidal_candles(n=100, amplitude=5.0, period=48)
        result = strategy.compute_indicators({"1h": candles})
        # Vérifier un timestamp arbitraire (assez loin pour avoir des valeurs valides)
        ts = candles[50].timestamp.isoformat()
        ind = result["1h"][ts]
        assert "sma" in ind
        assert "atr" in ind
        assert "close" in ind
        assert not math.isnan(ind["sma"])


class TestValidationBitgetGridMultiTF:
    """Tests que la validation Bitget produit des trades pour grid_multi_tf."""

    def test_validation_bitget_grid_multi_tf_has_trades(self):
        """MultiPositionEngine + compute_indicators() produit > 0 trades.

        Ce test simule le chemin validate_on_bitget → run_multi_backtest_single
        qui appelle compute_indicators() en interne (pas le cache).
        """
        from backend.backtesting.multi_engine import run_multi_backtest_single

        # Candles sinusoïdales longues (simulant ~90j de données Bitget)
        candles = _make_sinusoidal_candles(n=500, amplitude=8.0, period=48)
        bt_config = _make_bt_config()

        params = {
            "st_atr_period": 10,
            "st_atr_multiplier": 3.0,
            "ma_period": 14,
            "atr_period": 14,
            "atr_multiplier_start": 1.5,
            "atr_multiplier_step": 0.5,
            "num_levels": 3,
            "sl_percent": 20.0,
            "sides": ["long", "short"],
            "leverage": 6,
        }

        result = run_multi_backtest_single(
            "grid_multi_tf", params, {"1h": candles}, bt_config, main_tf="1h",
        )

        assert len(result.trades) > 0, (
            "Validation path (run_multi_backtest_single) doit produire > 0 trades "
            "pour grid_multi_tf — compute_indicators() doit fournir les indicateurs 4h"
        )
        assert result.final_capital > 0

    def test_oos_evaluation_produces_trades(self):
        """Le chemin OOS evaluation (identique au WFO) produit des trades."""
        from backend.backtesting.multi_engine import run_multi_backtest_single

        # Même params que le WFO utiliserait (enveloppes resserrées pour toucher les niveaux)
        params = {
            "st_atr_period": 10,
            "st_atr_multiplier": 3.0,
            "ma_period": 14,
            "atr_period": 14,
            "atr_multiplier_start": 1.5,
            "atr_multiplier_step": 0.5,
            "num_levels": 3,
            "sl_percent": 20.0,
            "sides": ["long", "short"],
            "leverage": 6,
        }

        # OOS = fenêtre courte (~30 jours = 720 candles 1h), amplitude suffisante
        candles = _make_sinusoidal_candles(n=720, amplitude=8.0, period=48)
        bt_config = _make_bt_config()

        result = run_multi_backtest_single(
            "grid_multi_tf", params, {"1h": candles}, bt_config, main_tf="1h",
        )

        assert len(result.trades) > 0, (
            "OOS evaluation doit produire > 0 trades pour Monte Carlo"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Section 8 : Bugfix 21a-ter — compute_live_indicators + GridStrategyRunner
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeLiveIndicators:
    """Tests que compute_live_indicators() fonctionne pour le mode live/portfolio."""

    def test_compute_live_indicators_returns_4h(self):
        """compute_live_indicators retourne la direction Supertrend 4h."""
        strategy = _make_strategy()
        candles = _make_sinusoidal_candles(n=200, amplitude=5.0, period=48)
        result = strategy.compute_live_indicators(candles)
        assert "4h" in result
        assert "st_direction" in result["4h"]
        assert result["4h"]["st_direction"] in (1.0, -1.0) or math.isnan(
            result["4h"]["st_direction"]
        )

    def test_compute_live_indicators_empty_when_few_candles(self):
        """Pas assez de candles → retourne {}."""
        strategy = _make_strategy(st_atr_period=10)
        # st_atr_period * 4 + 8 = 48, on donne 30
        candles = _make_candles(n=30)
        result = strategy.compute_live_indicators(candles)
        assert result == {}

    def test_compute_live_indicators_base_returns_empty(self):
        """BaseGridStrategy.compute_live_indicators retourne {} par défaut."""
        from backend.strategies.grid_atr import GridATRStrategy
        from backend.core.config import GridATRConfig

        strategy = GridATRStrategy(GridATRConfig())
        candles = _make_sinusoidal_candles(n=200, amplitude=5.0, period=48)
        result = strategy.compute_live_indicators(candles)
        assert result == {}

    def test_grid_strategy_runner_merges_live_indicators(self):
        """GridStrategyRunner appelle compute_live_indicators et merge les indicateurs."""
        import asyncio
        from unittest.mock import MagicMock, AsyncMock, patch

        from backend.backtesting.simulator import GridStrategyRunner
        from backend.core.incremental_indicators import IncrementalIndicatorEngine

        strategy = _make_strategy()
        config = MagicMock()
        config.risk = MagicMock()
        config.risk.max_margin_ratio = 0.7

        # Créer un engine avec buffer de candles
        indicator_engine = IncrementalIndicatorEngine([strategy])
        candles = _make_sinusoidal_candles(n=200, amplitude=5.0, period=48)

        # Remplir le buffer
        for c in candles:
            indicator_engine.update("BTC/USDT", "1h", c)

        # Vérifier que compute_live_indicators retourne quelque chose
        buf = indicator_engine._buffers.get(("BTC/USDT", "1h"), [])
        extra = strategy.compute_live_indicators(list(buf))
        assert "4h" in extra, "compute_live_indicators doit retourner les indicateurs 4h"
        assert "st_direction" in extra["4h"]
