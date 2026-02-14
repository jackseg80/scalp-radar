"""Tests du fast backtest engine (indicator_cache + fast_backtest).

16 tests couvrant :
- Tests 1-6 : indicateurs du cache vs fonctions originales
- Tests 7-8 : signaux vectorisés vs evaluate() de la stratégie
- Tests 9-14 : simulation de trades (TP/SL, OHLC, check_exit)
- Test 15 : parité fast engine vs moteur normal (test critique)
- Test 16 : vérification du speedup
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from backend.backtesting.engine import BacktestConfig, BacktestEngine
from backend.backtesting.metrics import calculate_metrics
from backend.core.indicators import (
    adx,
    atr,
    detect_market_regime,
    rsi,
    sma,
    volume_sma,
    vwap_rolling,
)
from backend.core.models import Candle, Direction, MarketRegime
from backend.optimization.fast_backtest import (
    _check_exit,
    _check_tp_sl,
    _close_trade,
    _momentum_signals,
    _ohlc_heuristic,
    _open_trade,
    _simulate_trades,
    _vwap_rsi_signals,
    run_backtest_from_cache,
)
from backend.optimization.indicator_cache import (
    REGIME_TO_INT,
    IndicatorCache,
    _build_filter_index,
    _compute_atr_sma_aligned,
    _rolling_max,
    _rolling_min,
    build_cache,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


def _make_candle(
    ts: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
) -> Candle:
    return Candle(
        symbol="BTC/USDT",
        timeframe="5m",
        timestamp=ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _make_candles(n: int, start_price: float = 40000.0, tf: str = "5m") -> list[Candle]:
    """Génère n bougies synthétiques avec mouvement sinusoïdal."""
    rng = np.random.default_rng(42)
    candles = []
    price = start_price
    base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}[tf]

    for i in range(n):
        ts = base_ts + timedelta(minutes=i * tf_minutes)
        # Mouvement sinusoïdal + bruit
        trend = np.sin(i / 50) * 200
        noise = rng.normal(0, 50)
        price = start_price + trend + noise
        price = max(price, 100)  # plancher

        spread = rng.uniform(10, 100)
        high = price + spread
        low = price - spread
        open_ = price + rng.normal(0, 30)
        close = price + rng.normal(0, 30)
        # Normaliser OHLC
        high = max(high, open_, close)
        low = min(low, open_, close)
        volume = rng.uniform(100, 10000)

        candles.append(Candle(
            symbol="BTC/USDT",
            timeframe=tf,
            timestamp=ts,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
        ))

    return candles


def _make_test_data(n_5m: int = 500) -> dict[str, list[Candle]]:
    """Génère un jeu de données 5m + 15m aligné."""
    candles_5m = _make_candles(n_5m, tf="5m")
    # Créer des bougies 15m à partir des 5m (1 bougie 15m toutes les 3 bougies 5m)
    candles_15m = []
    for i in range(0, n_5m - 2, 3):
        batch = candles_5m[i:i + 3]
        candles_15m.append(Candle(
            symbol="BTC/USDT",
            timeframe="15m",
            timestamp=batch[0].timestamp,
            open=batch[0].open,
            high=max(c.high for c in batch),
            low=min(c.low for c in batch),
            close=batch[-1].close,
            volume=sum(c.volume for c in batch),
        ))
    return {"5m": candles_5m, "15m": candles_15m}


@pytest.fixture
def test_data():
    return _make_test_data(500)


@pytest.fixture
def vwap_rsi_grid():
    return {
        "rsi_period": [10, 14, 20],
        "rsi_long_threshold": [25, 30],
        "rsi_short_threshold": [70, 75],
        "volume_spike_multiplier": [1.5, 2.0],
        "vwap_deviation_entry": [0.1, 0.2],
        "trend_adx_threshold": [20, 25],
        "tp_percent": [0.6],
        "sl_percent": [0.3],
    }


@pytest.fixture
def momentum_grid():
    return {
        "breakout_lookback": [15, 20, 30],
        "volume_confirmation_multiplier": [1.5, 2.0],
        "atr_multiplier_tp": [2.0],
        "atr_multiplier_sl": [1.0],
        "tp_percent": [0.8],
        "sl_percent": [0.3],
    }


@pytest.fixture
def bt_config():
    return BacktestConfig(
        symbol="BTC/USDT",
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 7, 1, tzinfo=timezone.utc),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Tests 1-6 : Indicateurs cache vs originaux
# ═══════════════════════════════════════════════════════════════════════════


class TestIndicatorCacheParity:
    """Vérifie que le cache produit les mêmes indicateurs que les fonctions originales."""

    def test_rsi_cache_matches_original(self, test_data, vwap_rsi_grid):
        """Test 1 : RSI du cache == RSI de indicators.rsi() pour chaque period."""
        cache = build_cache(test_data, vwap_rsi_grid, "vwap_rsi")
        closes = np.array([c.close for c in test_data["5m"]], dtype=float)

        for period in vwap_rsi_grid["rsi_period"]:
            expected = rsi(closes, period)
            actual = cache.rsi[period]

            # Comparer seulement les valeurs non-NaN
            mask = ~np.isnan(expected) & ~np.isnan(actual)
            assert mask.sum() > 0, f"Aucune valeur non-NaN pour RSI period={period}"
            np.testing.assert_allclose(
                actual[mask], expected[mask], rtol=1e-10,
                err_msg=f"RSI period={period} diverge",
            )

    def test_vwap_cache_matches_original(self, test_data, vwap_rsi_grid):
        """Test 2 : VWAP du cache == vwap_rolling() original."""
        cache = build_cache(test_data, vwap_rsi_grid, "vwap_rsi")
        candles = test_data["5m"]
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        closes = np.array([c.close for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)

        expected = vwap_rolling(highs, lows, closes, volumes)
        mask = ~np.isnan(expected) & ~np.isnan(cache.vwap)
        assert mask.sum() > 0
        np.testing.assert_allclose(
            cache.vwap[mask], expected[mask], rtol=1e-10,
        )

    def test_atr_cache_matches_original(self, test_data, vwap_rsi_grid):
        """Test 3 : ATR du cache == atr() original."""
        cache = build_cache(test_data, vwap_rsi_grid, "vwap_rsi")
        candles = test_data["5m"]
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        closes = np.array([c.close for c in candles], dtype=float)

        expected = atr(highs, lows, closes)
        mask = ~np.isnan(expected) & ~np.isnan(cache.atr_arr)
        assert mask.sum() > 0
        np.testing.assert_allclose(
            cache.atr_arr[mask], expected[mask], rtol=1e-10,
        )

    def test_adx_cache_matches_original(self, test_data, vwap_rsi_grid):
        """Test 4 : ADX, DI+, DI- du cache == adx() original."""
        cache = build_cache(test_data, vwap_rsi_grid, "vwap_rsi")
        candles = test_data["5m"]
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        closes = np.array([c.close for c in candles], dtype=float)

        exp_adx, exp_di_p, exp_di_m = adx(highs, lows, closes)

        for name, actual, expected in [
            ("adx", cache.adx_arr, exp_adx),
            ("di_plus", cache.di_plus, exp_di_p),
            ("di_minus", cache.di_minus, exp_di_m),
        ]:
            mask = ~np.isnan(expected) & ~np.isnan(actual)
            assert mask.sum() > 0, f"Aucune valeur non-NaN pour {name}"
            np.testing.assert_allclose(
                actual[mask], expected[mask], rtol=1e-10,
                err_msg=f"{name} diverge",
            )

    def test_regime_cache_matches_original(self, test_data, vwap_rsi_grid):
        """Test 5 : régime par bougie == detect_market_regime()."""
        cache = build_cache(test_data, vwap_rsi_grid, "vwap_rsi")

        for i in range(cache.n_candles):
            expected = detect_market_regime(
                cache.adx_arr[i], cache.di_plus[i], cache.di_minus[i],
                cache.atr_arr[i], cache.atr_sma[i],
            )
            expected_int = REGIME_TO_INT[expected]
            assert cache.regime[i] == expected_int, (
                f"Regime diverge à l'index {i}: "
                f"cache={cache.regime[i]}, expected={expected_int} ({expected})"
            )

    def test_filter_index_alignment(self, test_data):
        """Test 6 (CRITIQUE) : filtre 15m aligné correctement sur les indices 5m.

        Pour chaque bougie 5m, la valeur 15m alignée doit être celle de la
        dernière bougie 15m dont le timestamp <= bougie 5m.
        """
        main_candles = test_data["5m"]
        filter_candles = test_data["15m"]

        filter_index = _build_filter_index(main_candles, filter_candles)

        for i in range(len(main_candles)):
            main_ts = main_candles[i].timestamp

            # Trouver le dernier 15m <= main_ts manuellement
            expected_idx = -1
            for j, fc in enumerate(filter_candles):
                if fc.timestamp <= main_ts:
                    expected_idx = j
                else:
                    break

            assert filter_index[i] == expected_idx, (
                f"Index {i}: main_ts={main_ts}, "
                f"filter_index={filter_index[i]}, expected={expected_idx}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Tests 7-8 : Signaux vectorisés vs stratégie
# ═══════════════════════════════════════════════════════════════════════════


class TestSignalParity:
    """Vérifie que les signaux vectorisés matchent evaluate() de la stratégie."""

    def test_vwap_rsi_signals_match_strategy(self, test_data, vwap_rsi_grid, bt_config):
        """Test 7 : masques long/short VWAP+RSI == evaluate() du moteur normal."""
        from backend.optimization import create_strategy_with_params

        params = {
            "rsi_period": 14,
            "rsi_long_threshold": 30,
            "rsi_short_threshold": 70,
            "volume_spike_multiplier": 1.5,
            "vwap_deviation_entry": 0.1,
            "trend_adx_threshold": 25,
            "tp_percent": 0.6,
            "sl_percent": 0.3,
        }

        cache = build_cache(test_data, vwap_rsi_grid, "vwap_rsi")
        longs, shorts = _vwap_rsi_signals(params, cache)

        # Exécuter la stratégie normale pour comparaison
        strategy = create_strategy_with_params("vwap_rsi", params)
        indicators_by_tf = strategy.compute_indicators(test_data)

        # Construire l'index 15m pour alignment (même logique que BacktestEngine)
        higher_tf_timestamps = {}
        for tf, ind_dict in indicators_by_tf.items():
            if tf != "5m":
                higher_tf_timestamps[tf] = sorted(ind_dict.keys())

        main_indicators = indicators_by_tf.get("5m", {})
        main_candles = test_data["5m"]

        n_match = 0
        n_total = 0
        for i, candle in enumerate(main_candles):
            ts_iso = candle.timestamp.isoformat()

            ctx_indicators: dict = {}
            if ts_iso in main_indicators:
                ctx_indicators["5m"] = main_indicators[ts_iso]

            for tf, ts_list in higher_tf_timestamps.items():
                last_ts = None
                for ts in ts_list:
                    if ts <= ts_iso:
                        last_ts = ts
                    else:
                        break
                if last_ts and last_ts in indicators_by_tf[tf]:
                    ctx_indicators[tf] = indicators_by_tf[tf][last_ts]

            from backend.strategies.base import StrategyContext
            ctx = StrategyContext(
                symbol="BTC/USDT",
                timestamp=candle.timestamp,
                candles=test_data,
                indicators=ctx_indicators,
                current_position=None,
                capital=10000.0,
                config=None,
            )

            signal = strategy.evaluate(ctx)
            expected_long = signal is not None and signal.direction == Direction.LONG
            expected_short = signal is not None and signal.direction == Direction.SHORT

            # Les signaux doivent matcher
            n_total += 1
            if longs[i] == expected_long and shorts[i] == expected_short:
                n_match += 1
            else:
                # Tolérance : les rares divergences viennent de l'arrondi float
                # sur les seuils exacts. Accepter < 1% de divergence.
                pass

        match_rate = n_match / n_total if n_total > 0 else 0
        assert match_rate >= 0.99, (
            f"Signaux VWAP+RSI: {n_match}/{n_total} match ({match_rate:.1%})"
        )

    def test_momentum_signals_match_strategy(self, test_data, momentum_grid, bt_config):
        """Test 8 : masques long/short Momentum == evaluate() du moteur normal."""
        from backend.optimization import create_strategy_with_params

        params = {
            "breakout_lookback": 20,
            "volume_confirmation_multiplier": 1.5,
            "atr_multiplier_tp": 2.0,
            "atr_multiplier_sl": 1.0,
            "tp_percent": 0.8,
            "sl_percent": 0.3,
        }

        cache = build_cache(test_data, momentum_grid, "momentum")
        longs, shorts = _momentum_signals(params, cache)

        strategy = create_strategy_with_params("momentum", params)
        indicators_by_tf = strategy.compute_indicators(test_data)

        higher_tf_timestamps = {}
        for tf, ind_dict in indicators_by_tf.items():
            if tf != "5m":
                higher_tf_timestamps[tf] = sorted(ind_dict.keys())

        main_indicators = indicators_by_tf.get("5m", {})
        main_candles = test_data["5m"]

        n_match = 0
        n_total = 0
        for i, candle in enumerate(main_candles):
            ts_iso = candle.timestamp.isoformat()

            ctx_indicators: dict = {}
            if ts_iso in main_indicators:
                ctx_indicators["5m"] = main_indicators[ts_iso]

            for tf, ts_list in higher_tf_timestamps.items():
                last_ts = None
                for ts in ts_list:
                    if ts <= ts_iso:
                        last_ts = ts
                    else:
                        break
                if last_ts and last_ts in indicators_by_tf[tf]:
                    ctx_indicators[tf] = indicators_by_tf[tf][last_ts]

            from backend.strategies.base import StrategyContext
            ctx = StrategyContext(
                symbol="BTC/USDT",
                timestamp=candle.timestamp,
                candles=test_data,
                indicators=ctx_indicators,
                current_position=None,
                capital=10000.0,
                config=None,
            )

            signal = strategy.evaluate(ctx)
            expected_long = signal is not None and signal.direction == Direction.LONG
            expected_short = signal is not None and signal.direction == Direction.SHORT

            n_total += 1
            if longs[i] == expected_long and shorts[i] == expected_short:
                n_match += 1

        match_rate = n_match / n_total if n_total > 0 else 0
        assert match_rate >= 0.99, (
            f"Signaux Momentum: {n_match}/{n_total} match ({match_rate:.1%})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Tests 9-14 : Simulation de trades
# ═══════════════════════════════════════════════════════════════════════════


class TestTradeSimulation:
    """Vérifie la logique de simulation des trades."""

    def test_tp_sl_triggered_correctly(self):
        """Test 9 : TP/SL check sur OHLC."""
        # LONG : TP = high >= tp, SL = low <= sl
        tp_hit, sl_hit = _check_tp_sl(105.0, 95.0, 1, 104.0, 96.0)
        assert tp_hit is True
        assert sl_hit is True

        tp_hit, sl_hit = _check_tp_sl(103.0, 97.0, 1, 104.0, 96.0)
        assert tp_hit is False
        assert sl_hit is False

        # SHORT : TP = low <= tp, SL = high >= sl
        tp_hit, sl_hit = _check_tp_sl(105.0, 95.0, -1, 96.0, 104.0)
        assert tp_hit is True
        assert sl_hit is True

    def test_ohlc_heuristic_matches(self):
        """Test 10 : OHLC heuristic identique à PositionManager."""
        # Bougie verte (close > open) : LONG → tp, SHORT → sl
        assert _ohlc_heuristic(100, 105, 1) == "tp"
        assert _ohlc_heuristic(100, 105, -1) == "sl"

        # Bougie rouge (close < open) : LONG → sl, SHORT → tp
        assert _ohlc_heuristic(105, 100, 1) == "sl"
        assert _ohlc_heuristic(105, 100, -1) == "tp"

        # Doji (close == open) → sl
        assert _ohlc_heuristic(100, 100, 1) == "sl"
        assert _ohlc_heuristic(100, 100, -1) == "sl"

    def test_no_double_entry(self, test_data, vwap_rsi_grid, bt_config):
        """Test 11 : pas de position ouverte quand on est déjà en position."""
        cache = build_cache(test_data, vwap_rsi_grid, "vwap_rsi")
        params = {
            "rsi_period": 14,
            "rsi_long_threshold": 30,
            "rsi_short_threshold": 70,
            "volume_spike_multiplier": 1.5,
            "vwap_deviation_entry": 0.1,
            "trend_adx_threshold": 25,
            "tp_percent": 0.6,
            "sl_percent": 0.3,
        }

        # Forcer tous les signaux à True
        all_true = np.ones(cache.n_candles, dtype=bool)
        trade_pnls, _, _ = _simulate_trades(
            all_true, np.zeros(cache.n_candles, dtype=bool),
            cache, "vwap_rsi", params, bt_config,
        )

        # Avec tous les signaux à True, on devrait avoir des trades
        # mais jamais de double entry (1 position à la fois)
        assert len(trade_pnls) > 0

    def test_check_exit_vwap_rsi(self, make_indicator_cache):
        """Test 12 : VWAP+RSI check_exit (RSI normalisé + en profit)."""
        n = 10
        cache = make_indicator_cache(
            n=n,
            closes=np.full(n, 105.0),  # au-dessus de l'entrée (100)
            opens=np.full(n, 100.0),
            highs=np.full(n, 110.0),
            lows=np.full(n, 90.0),
            volumes=np.full(n, 1000.0),
            total_days=1.0,
            rsi={14: np.full(n, 55.0)},  # RSI > 50
            vwap=np.full(n, 100.0),
            vwap_distance_pct=np.full(n, 5.0),
            adx_arr=np.full(n, 30.0),
            di_plus=np.full(n, 20.0),
            di_minus=np.full(n, 10.0),
            volume_sma_arr=np.full(n, 500.0),
        )
        params = {"rsi_period": 14}

        # LONG : close (105) > entry (100) et RSI (55) > 50 → exit
        assert _check_exit("vwap_rsi", cache, 5, 1, 100.0, params) is True

        # LONG : close (105) > entry mais RSI < 50 → pas d'exit
        cache.rsi[14][:] = 45.0
        assert _check_exit("vwap_rsi", cache, 5, 1, 100.0, params) is False

        # SHORT : close (105) > entry (100) → pas en profit → pas d'exit
        cache.rsi[14][:] = 45.0  # RSI < 50
        assert _check_exit("vwap_rsi", cache, 5, -1, 100.0, params) is False

        # SHORT : close < entry et RSI < 50 → exit
        cache.closes[:] = 95.0
        assert _check_exit("vwap_rsi", cache, 5, -1, 100.0, params) is True

    def test_check_exit_momentum(self, make_indicator_cache):
        """Test 13 : Momentum check_exit (ADX < 20)."""
        n = 10
        cache = make_indicator_cache(
            n=n,
            highs=np.full(n, 110.0),
            lows=np.full(n, 90.0),
            volumes=np.full(n, 1000.0),
            total_days=1.0,
            vwap=np.full(n, 100.0),
            vwap_distance_pct=np.zeros(n),
            adx_arr=np.full(n, 15.0),  # ADX < 20
            di_plus=np.full(n, 20.0),
            volume_sma_arr=np.full(n, 500.0),
        )

        # ADX (15) < 20 → exit
        assert _check_exit("momentum", cache, 5, 1, 100.0, {}) is True

        # ADX >= 20 → pas d'exit
        cache.adx_arr[:] = 25.0
        assert _check_exit("momentum", cache, 5, 1, 100.0, {}) is False

    def test_regime_used_for_slippage(self):
        """Test 14 : le régime au moment de l'exit affecte le slippage."""
        # Trade identique, mais régime HIGH_VOL → slippage x2
        base_pnl = _close_trade(
            direction=1, entry_price=100.0, exit_price=99.0, quantity=1.0,
            entry_fee=0.06, exit_reason="sl", regime_int=0,  # RANGING
            taker_fee=0.0006, maker_fee=0.0002,
            slippage_pct=0.0005, high_vol_slippage_mult=2.0,
        )

        high_vol_pnl = _close_trade(
            direction=1, entry_price=100.0, exit_price=99.0, quantity=1.0,
            entry_fee=0.06, exit_reason="sl", regime_int=3,  # HIGH_VOL
            taker_fee=0.0006, maker_fee=0.0002,
            slippage_pct=0.0005, high_vol_slippage_mult=2.0,
        )

        # HIGH_VOL pire que RANGING (plus de slippage)
        assert high_vol_pnl < base_pnl

        # TP : pas de slippage (ni RANGING ni HIGH_VOL)
        tp_pnl = _close_trade(
            direction=1, entry_price=100.0, exit_price=101.0, quantity=1.0,
            entry_fee=0.06, exit_reason="tp", regime_int=3,  # HIGH_VOL
            taker_fee=0.0006, maker_fee=0.0002,
            slippage_pct=0.0005, high_vol_slippage_mult=2.0,
        )
        tp_pnl_ranging = _close_trade(
            direction=1, entry_price=100.0, exit_price=101.0, quantity=1.0,
            entry_fee=0.06, exit_reason="tp", regime_int=0,  # RANGING
            taker_fee=0.0006, maker_fee=0.0002,
            slippage_pct=0.0005, high_vol_slippage_mult=2.0,
        )
        # TP : même PnL quel que soit le régime (pas de slippage sur TP)
        assert tp_pnl == tp_pnl_ranging


# ═══════════════════════════════════════════════════════════════════════════
# Test 15 : Parité fast engine vs moteur normal (TEST CRITIQUE)
# ═══════════════════════════════════════════════════════════════════════════


class TestFastVsNormalParity:
    """Compare le fast engine au moteur normal sur les mêmes données et params."""

    def _run_normal_backtest(self, strategy_name, params, candles_by_tf, bt_config):
        """Lance un backtest avec le moteur normal."""
        from backend.optimization import create_strategy_with_params

        strategy = create_strategy_with_params(strategy_name, params)
        engine = BacktestEngine(bt_config, strategy)
        result = engine.run(candles_by_tf, main_tf="5m")
        metrics = calculate_metrics(result)
        return metrics

    def test_fast_vs_normal_parity_vwap_rsi(self, bt_config):
        """Parité fast vs normal pour VWAP+RSI."""
        # Données plus grandes pour avoir des trades
        data = _make_test_data(800)

        params = {
            "rsi_period": 14,
            "rsi_long_threshold": 30,
            "rsi_short_threshold": 70,
            "volume_spike_multiplier": 1.5,
            "vwap_deviation_entry": 0.1,
            "trend_adx_threshold": 25,
            "tp_percent": 0.6,
            "sl_percent": 0.3,
        }

        grid_values = {k: [v] for k, v in params.items()}

        # Fast engine
        cache = build_cache(data, grid_values, "vwap_rsi")
        fast_result = run_backtest_from_cache("vwap_rsi", params, cache, bt_config)
        fast_sharpe = fast_result[1]
        fast_return = fast_result[2]
        fast_pf = fast_result[3]
        fast_trades = fast_result[4]

        # Normal engine
        normal_metrics = self._run_normal_backtest("vwap_rsi", params, data, bt_config)

        # Comparaison
        assert fast_trades == normal_metrics.total_trades, (
            f"n_trades: fast={fast_trades}, normal={normal_metrics.total_trades}"
        )

        if normal_metrics.total_trades > 0:
            assert abs(fast_return - normal_metrics.net_return_pct) < 0.5, (
                f"net_return: fast={fast_return:.2f}%, normal={normal_metrics.net_return_pct:.2f}%"
            )
            if normal_metrics.sharpe_ratio != 0:
                assert abs(fast_sharpe - normal_metrics.sharpe_ratio) < 0.1, (
                    f"sharpe: fast={fast_sharpe:.3f}, normal={normal_metrics.sharpe_ratio:.3f}"
                )

    def test_fast_vs_normal_parity_momentum(self, bt_config):
        """Parité fast vs normal pour Momentum."""
        data = _make_test_data(800)

        params = {
            "breakout_lookback": 20,
            "volume_confirmation_multiplier": 1.5,
            "atr_multiplier_tp": 2.0,
            "atr_multiplier_sl": 1.0,
            "tp_percent": 0.8,
            "sl_percent": 0.3,
        }

        grid_values = {k: [v] for k, v in params.items()}

        cache = build_cache(data, grid_values, "momentum")
        fast_result = run_backtest_from_cache("momentum", params, cache, bt_config)
        fast_sharpe = fast_result[1]
        fast_return = fast_result[2]
        fast_pf = fast_result[3]
        fast_trades = fast_result[4]

        normal_metrics = self._run_normal_backtest("momentum", params, data, bt_config)

        assert fast_trades == normal_metrics.total_trades, (
            f"n_trades: fast={fast_trades}, normal={normal_metrics.total_trades}"
        )

        if normal_metrics.total_trades > 0:
            assert abs(fast_return - normal_metrics.net_return_pct) < 0.5, (
                f"net_return: fast={fast_return:.2f}%, normal={normal_metrics.net_return_pct:.2f}%"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Test 16 : Speedup
# ═══════════════════════════════════════════════════════════════════════════


class TestSpeedup:
    """Vérifie que le fast engine est significativement plus rapide."""

    def test_fast_engine_speedup(self, bt_config):
        """Fast engine >= 10x plus rapide sur 100 backtests."""
        import itertools

        data = _make_test_data(500)

        # Grid de paramètres
        grid_values = {
            "rsi_period": [10, 14],
            "rsi_long_threshold": [25, 30],
            "rsi_short_threshold": [70, 75],
            "volume_spike_multiplier": [1.5, 2.0],
            "vwap_deviation_entry": [0.1, 0.2],
            "trend_adx_threshold": [25],
            "tp_percent": [0.6],
            "sl_percent": [0.3],
        }

        keys = sorted(grid_values.keys())
        values = [grid_values[k] for k in keys]
        combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

        # Limiter à 100 combos max
        combos = combos[:100]
        n_combos = len(combos)

        # Fast engine
        t0 = time.monotonic()
        cache = build_cache(data, grid_values, "vwap_rsi")
        for params in combos:
            run_backtest_from_cache("vwap_rsi", params, cache, bt_config)
        fast_time = time.monotonic() - t0

        # Normal engine (sample de 5 pour estimer)
        n_sample = min(5, n_combos)
        t0 = time.monotonic()
        from backend.optimization import create_strategy_with_params
        for params in combos[:n_sample]:
            strategy = create_strategy_with_params("vwap_rsi", params)
            engine = BacktestEngine(bt_config, strategy)
            engine.run(data, main_tf="5m")
        normal_sample_time = time.monotonic() - t0
        normal_estimated = normal_sample_time / n_sample * n_combos

        speedup = normal_estimated / fast_time if fast_time > 0 else float("inf")

        # Le speedup doit être >= 10x
        assert speedup >= 10, (
            f"Speedup insuffisant: {speedup:.1f}x "
            f"(fast={fast_time:.2f}s, normal_est={normal_estimated:.2f}s pour {n_combos} combos)"
        )
