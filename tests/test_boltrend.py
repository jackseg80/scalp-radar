"""Tests pour la stratégie BolTrend (Bollinger Trend Following)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from backend.core.config import BolTrendConfig
from backend.core.models import Candle, Direction, TimeFrame
from backend.strategies.base import OpenPosition, StrategyContext
from backend.strategies.boltrend import BolTrendStrategy


# ─── Fixtures ────────────────────────────────────────────────────────────────


def _make_candle(close: float, high: float | None = None, low: float | None = None,
                 ts_offset_hours: int = 0) -> Candle:
    """Crée une candle 1h simple."""
    return Candle(
        timestamp=datetime(2024, 1, 1, ts_offset_hours, tzinfo=timezone.utc),
        open=close,
        high=high if high is not None else close + 1.0,
        low=low if low is not None else close - 1.0,
        close=close,
        volume=1000.0,
        symbol="ETH/USDT",
        timeframe=TimeFrame.from_string("1h"),
        exchange="bitget",
    )


def _make_config(**overrides) -> BolTrendConfig:
    defaults = {
        "enabled": True,
        "bol_window": 5,
        "bol_std": 2.0,
        "min_bol_spread": 0.0,
        "long_ma_window": 10,
        "sl_percent": 15.0,
        "leverage": 2,
    }
    defaults.update(overrides)
    return BolTrendConfig(**defaults)


def _make_strategy(**overrides) -> BolTrendStrategy:
    return BolTrendStrategy(_make_config(**overrides))


def _dummy_app_config():
    """Retourne une AppConfig minimale via YAML."""
    from backend.core.config import AppConfig
    return AppConfig()


# ─── Tests Config ────────────────────────────────────────────────────────────


class TestBolTrendConfig:
    def test_defaults(self):
        cfg = BolTrendConfig()
        assert cfg.enabled is False
        assert cfg.bol_window == 100
        assert cfg.bol_std == 2.2
        assert cfg.min_bol_spread == 0.0
        assert cfg.long_ma_window == 550
        assert cfg.sl_percent == 15.0
        assert cfg.leverage == 2
        assert cfg.timeframe == "1h"

    def test_per_asset(self):
        cfg = BolTrendConfig(per_asset={"ETH/USDT": {"sl_percent": 10.0}})
        params = cfg.get_params_for_symbol("ETH/USDT")
        assert params["sl_percent"] == 10.0
        params_default = cfg.get_params_for_symbol("BTC/USDT")
        assert params_default["sl_percent"] == 15.0


# ─── Tests Stratégie ─────────────────────────────────────────────────────────


class TestBolTrendStrategy:
    def test_name(self):
        s = _make_strategy()
        assert s.name == "boltrend"

    def test_min_candles(self):
        s = _make_strategy(bol_window=100, long_ma_window=550)
        assert s.min_candles == {"1h": 570}

    def test_get_params(self):
        s = _make_strategy()
        params = s.get_params()
        assert params["bol_window"] == 5
        assert params["bol_std"] == 2.0
        assert params["long_ma_window"] == 10
        assert params["sl_percent"] == 15.0

    def test_compute_indicators_returns_keys(self):
        """compute_indicators retourne les indicateurs attendus."""
        s = _make_strategy(bol_window=3, long_ma_window=5)
        candles = [_make_candle(100.0 + i, ts_offset_hours=i) for i in range(20)]
        result = s.compute_indicators({"1h": candles})
        assert "1h" in result
        # Vérifier les clés du premier indicateur valide
        first_ts = list(result["1h"].keys())[-1]
        ind = result["1h"][first_ts]
        expected_keys = {"close", "bb_sma", "bb_upper", "bb_lower", "long_ma",
                         "prev_close", "prev_upper", "prev_lower", "prev_spread",
                         "atr", "atr_sma", "adx", "di_plus", "di_minus"}
        assert expected_keys.issubset(ind.keys())


class TestBolTrendSignals:
    """Tests des conditions d'entrée."""

    def _make_ctx(self, indicators: dict, symbol: str = "ETH/USDT") -> StrategyContext:
        return StrategyContext(
            symbol=symbol,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            candles={},
            indicators={"1h": indicators},
            current_position=None,
            capital=10000.0,
            config=_dummy_app_config(),
        )

    def test_long_signal_breakout(self):
        """Signal LONG : prev_close < prev_upper AND close > upper AND close > long_ma."""
        s = _make_strategy()
        ctx = self._make_ctx({
            "close": 110.0,
            "bb_upper": 108.0,
            "bb_lower": 92.0,
            "bb_sma": 100.0,
            "long_ma": 95.0,
            "prev_close": 107.0,  # < prev_upper
            "prev_upper": 108.0,
            "prev_lower": 92.0,
            "prev_spread": (108.0 - 92.0) / 92.0,
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 25.0,
            "di_plus": 20.0,
            "di_minus": 10.0,
        })
        signal = s.evaluate(ctx)
        assert signal is not None
        assert signal.direction == Direction.LONG

    def test_short_signal_breakout(self):
        """Signal SHORT : prev_close > prev_lower AND close < lower AND close < long_ma."""
        s = _make_strategy()
        ctx = self._make_ctx({
            "close": 88.0,
            "bb_upper": 108.0,
            "bb_lower": 92.0,
            "bb_sma": 100.0,
            "long_ma": 105.0,
            "prev_close": 93.0,  # > prev_lower
            "prev_upper": 108.0,
            "prev_lower": 92.0,
            "prev_spread": (108.0 - 92.0) / 92.0,
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 25.0,
            "di_plus": 10.0,
            "di_minus": 20.0,
        })
        signal = s.evaluate(ctx)
        assert signal is not None
        assert signal.direction == Direction.SHORT

    def test_no_long_if_below_long_ma(self):
        """Pas de signal LONG si close < long_ma (trend baissière)."""
        s = _make_strategy()
        ctx = self._make_ctx({
            "close": 110.0,
            "bb_upper": 108.0,
            "bb_lower": 92.0,
            "bb_sma": 100.0,
            "long_ma": 115.0,  # close < long_ma → pas de LONG
            "prev_close": 107.0,
            "prev_upper": 108.0,
            "prev_lower": 92.0,
            "prev_spread": (108.0 - 92.0) / 92.0,
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 25.0,
            "di_plus": 20.0,
            "di_minus": 10.0,
        })
        signal = s.evaluate(ctx)
        assert signal is None

    def test_no_short_if_above_long_ma(self):
        """Pas de signal SHORT si close > long_ma (trend haussière)."""
        s = _make_strategy()
        ctx = self._make_ctx({
            "close": 88.0,
            "bb_upper": 108.0,
            "bb_lower": 92.0,
            "bb_sma": 100.0,
            "long_ma": 85.0,  # close > long_ma → pas de SHORT
            "prev_close": 93.0,
            "prev_upper": 108.0,
            "prev_lower": 92.0,
            "prev_spread": (108.0 - 92.0) / 92.0,
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 25.0,
            "di_plus": 10.0,
            "di_minus": 20.0,
        })
        signal = s.evaluate(ctx)
        assert signal is None

    def test_spread_filter_blocks(self):
        """Pas de signal si spread < min_bol_spread."""
        s = _make_strategy(min_bol_spread=0.5)
        ctx = self._make_ctx({
            "close": 110.0,
            "bb_upper": 108.0,
            "bb_lower": 92.0,
            "bb_sma": 100.0,
            "long_ma": 95.0,
            "prev_close": 107.0,
            "prev_upper": 101.0,  # spread = (101-99)/99 ≈ 0.02 < 0.5
            "prev_lower": 99.0,
            "prev_spread": (101.0 - 99.0) / 99.0,
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 25.0,
            "di_plus": 20.0,
            "di_minus": 10.0,
        })
        signal = s.evaluate(ctx)
        assert signal is None

    def test_no_signal_prev_close_outside_band(self):
        """Pas de signal LONG si prev_close >= prev_upper (déjà breakouté)."""
        s = _make_strategy()
        ctx = self._make_ctx({
            "close": 112.0,
            "bb_upper": 108.0,
            "bb_lower": 92.0,
            "bb_sma": 100.0,
            "long_ma": 95.0,
            "prev_close": 109.0,  # >= prev_upper → pas un nouveau breakout
            "prev_upper": 108.0,
            "prev_lower": 92.0,
            "prev_spread": (108.0 - 92.0) / 92.0,
            "atr": 2.0,
            "atr_sma": 2.0,
            "adx": 25.0,
            "di_plus": 20.0,
            "di_minus": 10.0,
        })
        signal = s.evaluate(ctx)
        assert signal is None


class TestBolTrendExit:
    """Tests des conditions de sortie."""

    def _make_ctx_with_ind(self, close: float, bb_sma: float) -> StrategyContext:
        return StrategyContext(
            symbol="ETH/USDT",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            candles={},
            indicators={"1h": {"close": close, "bb_sma": bb_sma}},
            current_position=None,
            capital=10000.0,
            config=_dummy_app_config(),
        )

    def test_exit_long_when_close_below_sma(self):
        """LONG exit quand close < bb_sma."""
        s = _make_strategy()
        ctx = self._make_ctx_with_ind(close=98.0, bb_sma=100.0)
        pos = OpenPosition(
            direction=Direction.LONG, entry_price=105.0, quantity=1.0,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            tp_price=210.0, sl_price=90.0, entry_fee=0.1,
        )
        result = s.check_exit(ctx, pos)
        assert result == "signal_exit"

    def test_exit_short_when_close_above_sma(self):
        """SHORT exit quand close > bb_sma."""
        s = _make_strategy()
        ctx = self._make_ctx_with_ind(close=102.0, bb_sma=100.0)
        pos = OpenPosition(
            direction=Direction.SHORT, entry_price=95.0, quantity=1.0,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            tp_price=47.5, sl_price=110.0, entry_fee=0.1,
        )
        result = s.check_exit(ctx, pos)
        assert result == "signal_exit"

    def test_no_exit_long_above_sma(self):
        """Pas d'exit LONG quand close > bb_sma (breakout continue)."""
        s = _make_strategy()
        ctx = self._make_ctx_with_ind(close=105.0, bb_sma=100.0)
        pos = OpenPosition(
            direction=Direction.LONG, entry_price=102.0, quantity=1.0,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            tp_price=210.0, sl_price=90.0, entry_fee=0.1,
        )
        result = s.check_exit(ctx, pos)
        assert result is None

    def test_no_exit_short_below_sma(self):
        """Pas d'exit SHORT quand close < bb_sma (breakout continue)."""
        s = _make_strategy()
        ctx = self._make_ctx_with_ind(close=95.0, bb_sma=100.0)
        pos = OpenPosition(
            direction=Direction.SHORT, entry_price=98.0, quantity=1.0,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            tp_price=47.5, sl_price=115.0, entry_fee=0.1,
        )
        result = s.check_exit(ctx, pos)
        assert result is None


# ─── Tests Fast Engine ───────────────────────────────────────────────────────


class TestBolTrendFastEngine:
    def test_signals_basic(self, make_indicator_cache):
        """_boltrend_signals génère des longs/shorts corrects."""
        from backend.optimization.fast_backtest import _boltrend_signals

        n = 20
        # Prix qui monte puis descend — simule un breakout
        closes = np.array([100.0] * 5 + [101.0, 102.0, 103.0, 105.0, 108.0,
                                          112.0, 110.0, 107.0, 104.0, 100.0,
                                          97.0, 94.0, 91.0, 88.0, 85.0])

        # BB avec window=5 — calculer manuellement
        from backend.core.indicators import bollinger_bands, sma
        bb_sma_arr, bb_upper, bb_lower = bollinger_bands(closes, 5, 2.0)
        long_ma = sma(closes, 10)

        cache = make_indicator_cache(
            n=n,
            closes=closes,
            bb_sma={5: bb_sma_arr, 10: long_ma},
            bb_upper={(5, 2.0): bb_upper},
            bb_lower={(5, 2.0): bb_lower},
        )

        params = {
            "bol_window": 5,
            "bol_std": 2.0,
            "long_ma_window": 10,
            "min_bol_spread": 0.0,
        }
        longs, shorts = _boltrend_signals(params, cache)

        assert len(longs) == n
        assert len(shorts) == n
        # Position 0 doit toujours être False (np.roll wraparound)
        assert longs[0] == False
        assert shorts[0] == False

    def test_valid_zero_false(self, make_indicator_cache):
        """Position 0 ne produit jamais de signal (np.roll wraparound)."""
        from backend.optimization.fast_backtest import _boltrend_signals

        n = 10
        closes = np.linspace(100.0, 150.0, n)
        from backend.core.indicators import bollinger_bands, sma
        bb_sma_arr, bb_upper, bb_lower = bollinger_bands(closes, 3, 2.0)
        long_ma = sma(closes, 5)

        cache = make_indicator_cache(
            n=n,
            closes=closes,
            bb_sma={3: bb_sma_arr, 5: long_ma},
            bb_upper={(3, 2.0): bb_upper},
            bb_lower={(3, 2.0): bb_lower},
        )

        params = {"bol_window": 3, "bol_std": 2.0, "long_ma_window": 5, "min_bol_spread": 0.0}
        longs, shorts = _boltrend_signals(params, cache)
        assert longs[0] == False
        assert shorts[0] == False

    def test_backtest_returns_5_tuple(self, make_indicator_cache):
        """run_backtest_from_cache retourne un 5-tuple valide."""
        from backend.backtesting.engine import BacktestConfig
        from backend.optimization.fast_backtest import run_backtest_from_cache
        from backend.core.indicators import bollinger_bands, sma

        n = 100
        # Prix avec tendance haussière puis baissière — devrait générer des trades
        prices_up = np.linspace(100, 130, 50)
        prices_down = np.linspace(130, 90, 50)
        closes = np.concatenate([prices_up, prices_down])

        bb_sma_arr, bb_upper, bb_lower = bollinger_bands(closes, 10, 2.0)
        long_ma = sma(closes, 20)

        cache = make_indicator_cache(
            n=n,
            closes=closes,
            opens=closes,
            highs=closes + 2.0,
            lows=closes - 2.0,
            bb_sma={10: bb_sma_arr, 20: long_ma},
            bb_upper={(10, 2.0): bb_upper},
            bb_lower={(10, 2.0): bb_lower},
        )

        params = {
            "bol_window": 10,
            "bol_std": 2.0,
            "long_ma_window": 20,
            "min_bol_spread": 0.0,
            "sl_percent": 15.0,
        }
        bt_config = BacktestConfig(
            symbol="ETH/USDT",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            initial_capital=10000.0,
            taker_fee=0.0006,
            maker_fee=0.0002,
            slippage_pct=0.0002,
            high_vol_slippage_mult=2.0,
            max_risk_per_trade=0.02,
            leverage=2,
        )

        result = run_backtest_from_cache("boltrend", params, cache, bt_config)
        assert isinstance(result, tuple)
        assert len(result) == 5
        # (params_dict, sharpe, net_return_pct, profit_factor, n_trades)
        assert isinstance(result[0], dict)
        assert isinstance(result[4], int)
        assert result[4] >= 0


# ─── Tests Registre ──────────────────────────────────────────────────────────


class TestBolTrendRegistry:
    def test_in_strategy_registry(self):
        from backend.optimization import STRATEGY_REGISTRY
        assert "boltrend" in STRATEGY_REGISTRY

    def test_in_fast_engine_strategies(self):
        from backend.optimization import FAST_ENGINE_STRATEGIES
        assert "boltrend" in FAST_ENGINE_STRATEGIES

    def test_not_in_grid_strategies(self):
        from backend.optimization import GRID_STRATEGIES
        assert "boltrend" not in GRID_STRATEGIES

    def test_create_strategy_with_params(self):
        from backend.optimization import create_strategy_with_params
        s = create_strategy_with_params("boltrend", {
            "bol_window": 50,
            "bol_std": 1.5,
            "long_ma_window": 200,
            "sl_percent": 10.0,
        })
        assert s.name == "boltrend"
        params = s.get_params()
        assert params["bol_window"] == 50
        assert params["long_ma_window"] == 200


class TestBolTrendFactory:
    def test_factory_create(self):
        """create_strategy crée bien une BolTrendStrategy."""
        from backend.strategies.factory import create_strategy
        config = _dummy_app_config()
        s = create_strategy("boltrend", config)
        assert s.name == "boltrend"


class TestBolTrendAdaptiveSelector:
    def test_mapping_present(self):
        from backend.execution.adaptive_selector import _STRATEGY_CONFIG_ATTR
        assert "boltrend" in _STRATEGY_CONFIG_ATTR
