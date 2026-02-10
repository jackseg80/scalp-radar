"""Tests pour backend/strategies/vwap_rsi.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from backend.core.config import VwapRsiConfig
from backend.core.models import Candle, Direction, MarketRegime, SignalStrength, TimeFrame
from backend.strategies.base import OpenPosition, StrategyContext
from backend.strategies.vwap_rsi import VwapRsiStrategy


def _make_config(**overrides) -> VwapRsiConfig:
    defaults = {
        "enabled": True,
        "timeframe": "5m",
        "trend_filter_timeframe": "15m",
        "rsi_period": 14,
        "rsi_long_threshold": 25,
        "rsi_short_threshold": 75,
        "volume_spike_multiplier": 3.0,
        "vwap_deviation_entry": 0.5,
        "tp_percent": 0.5,
        "sl_percent": 0.25,
        "weight": 0.25,
    }
    defaults.update(overrides)
    return VwapRsiConfig(**defaults)


def _make_context(
    main_indicators: dict | None = None,
    filter_indicators: dict | None = None,
    position: OpenPosition | None = None,
) -> StrategyContext:
    """Helper pour créer un StrategyContext de test."""
    indicators = {}
    if main_indicators:
        indicators["5m"] = main_indicators
    if filter_indicators:
        indicators["15m"] = filter_indicators

    return StrategyContext(
        symbol="BTC/USDT",
        timestamp=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
        candles={},
        indicators=indicators,
        current_position=position,
        capital=10_000.0,
        config=None,  # type: ignore[arg-type]
    )


# ─── Signaux d'entrée ───────────────────────────────────────────────────────


class TestEvaluate:
    def test_long_signal(self):
        """RSI < 25, prix sous VWAP, volume spike → signal LONG."""
        strategy = VwapRsiStrategy(_make_config())
        ctx = _make_context(
            main_indicators={
                "rsi": 20.0,
                "vwap": 100_000.0,
                "close": 99_400.0,  # 0.6% sous VWAP > 0.5% deviation
                "volume": 1500.0,
                "volume_sma": 400.0,  # 1500/400 = 3.75 > 3.0
                "adx": 15.0,
                "di_plus": 12.0,
                "di_minus": 12.0,
                "atr": 100.0,
                "atr_sma": 100.0,
            },
            filter_indicators={
                "rsi": 50.0,
                "adx": 15.0,  # Pas de tendance 15m → ne filtre pas
                "di_plus": 12.0,
                "di_minus": 12.0,
            },
        )
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert signal.direction == Direction.LONG
        assert signal.score > 0
        assert signal.tp_price > signal.entry_price
        assert signal.sl_price < signal.entry_price

    def test_short_signal(self):
        """RSI > 75, prix au-dessus VWAP, volume spike → signal SHORT."""
        strategy = VwapRsiStrategy(_make_config())
        ctx = _make_context(
            main_indicators={
                "rsi": 80.0,
                "vwap": 100_000.0,
                "close": 100_600.0,  # 0.6% au-dessus VWAP
                "volume": 1500.0,
                "volume_sma": 400.0,
                "adx": 15.0,
                "di_plus": 12.0,
                "di_minus": 12.0,
                "atr": 100.0,
                "atr_sma": 100.0,
            },
            filter_indicators={
                "rsi": 50.0,
                "adx": 15.0,
                "di_plus": 12.0,
                "di_minus": 12.0,
            },
        )
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert signal.direction == Direction.SHORT
        assert signal.tp_price < signal.entry_price
        assert signal.sl_price > signal.entry_price

    def test_no_signal_rsi_normal(self):
        """RSI entre 40-60 → pas de signal."""
        strategy = VwapRsiStrategy(_make_config())
        ctx = _make_context(
            main_indicators={
                "rsi": 50.0,
                "vwap": 100_000.0,
                "close": 99_400.0,
                "volume": 1500.0,
                "volume_sma": 400.0,
                "adx": 15.0,
                "di_plus": 12.0,
                "di_minus": 12.0,
                "atr": 100.0,
                "atr_sma": 100.0,
            },
            filter_indicators={
                "rsi": 50.0,
                "adx": 15.0,
                "di_plus": 12.0,
                "di_minus": 12.0,
            },
        )
        assert strategy.evaluate(ctx) is None

    def test_filtered_by_15m_bearish(self):
        """15m bearish (DI- > DI+, ADX > 20) filtre les LONG."""
        strategy = VwapRsiStrategy(_make_config())
        ctx = _make_context(
            main_indicators={
                "rsi": 20.0,
                "vwap": 100_000.0,
                "close": 99_400.0,
                "volume": 1500.0,
                "volume_sma": 400.0,
                "adx": 15.0,
                "di_plus": 12.0,
                "di_minus": 12.0,
                "atr": 100.0,
                "atr_sma": 100.0,
            },
            filter_indicators={
                "rsi": 30.0,
                "adx": 30.0,  # Tendance forte
                "di_plus": 10.0,
                "di_minus": 25.0,  # Bearish
            },
        )
        assert strategy.evaluate(ctx) is None

    def test_not_filtered_by_15m_bullish(self):
        """15m bullish ne filtre pas les LONG."""
        strategy = VwapRsiStrategy(_make_config())
        ctx = _make_context(
            main_indicators={
                "rsi": 20.0,
                "vwap": 100_000.0,
                "close": 99_400.0,
                "volume": 1500.0,
                "volume_sma": 400.0,
                "adx": 15.0,
                "di_plus": 12.0,
                "di_minus": 12.0,
                "atr": 100.0,
                "atr_sma": 100.0,
            },
            filter_indicators={
                "rsi": 60.0,
                "adx": 30.0,
                "di_plus": 25.0,  # Bullish
                "di_minus": 10.0,
            },
        )
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert signal.direction == Direction.LONG
        assert signal.signals_detail["trend_score"] == 1.0

    def test_score_has_components(self):
        """Le signal doit avoir des sous-scores détaillés."""
        strategy = VwapRsiStrategy(_make_config())
        ctx = _make_context(
            main_indicators={
                "rsi": 15.0,
                "vwap": 100_000.0,
                "close": 99_200.0,
                "volume": 2000.0,
                "volume_sma": 400.0,
                "adx": 15.0,
                "di_plus": 12.0,
                "di_minus": 12.0,
                "atr": 100.0,
                "atr_sma": 100.0,
            },
            filter_indicators={
                "rsi": 50.0,
                "adx": 15.0,
                "di_plus": 12.0,
                "di_minus": 12.0,
            },
        )
        signal = strategy.evaluate(ctx)
        assert signal is not None
        assert "rsi_score" in signal.signals_detail
        assert "vwap_score" in signal.signals_detail
        assert "volume_score" in signal.signals_detail
        assert "trend_score" in signal.signals_detail


# ─── Sortie anticipée ────────────────────────────────────────────────────────


class TestCheckExit:
    def test_exit_rsi_normalized_in_profit(self):
        """RSI > 50 en profit → signal_exit."""
        strategy = VwapRsiStrategy(_make_config())
        position = OpenPosition(
            direction=Direction.LONG,
            entry_price=99_000.0,
            quantity=0.01,
            entry_time=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            tp_price=99_500.0,
            sl_price=98_750.0,
            entry_fee=0.5,
        )
        ctx = _make_context(
            main_indicators={
                "rsi": 55.0,  # RSI normalisé
                "close": 99_200.0,  # En profit (> 99_000)
                "vwap": 99_000.0,
                "adx": 15.0,
                "di_plus": 12.0,
                "di_minus": 12.0,
                "atr": 100.0,
                "atr_sma": 100.0,
                "volume": 500.0,
                "volume_sma": 500.0,
            },
            position=position,
        )
        result = strategy.check_exit(ctx, position)
        assert result == "signal_exit"

    def test_no_exit_rsi_normalized_in_loss(self):
        """RSI > 50 mais en perte → pas de sortie."""
        strategy = VwapRsiStrategy(_make_config())
        position = OpenPosition(
            direction=Direction.LONG,
            entry_price=99_000.0,
            quantity=0.01,
            entry_time=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            tp_price=99_500.0,
            sl_price=98_750.0,
            entry_fee=0.5,
        )
        ctx = _make_context(
            main_indicators={
                "rsi": 55.0,
                "close": 98_900.0,  # En perte (< 99_000)
                "vwap": 99_000.0,
                "adx": 15.0,
                "di_plus": 12.0,
                "di_minus": 12.0,
                "atr": 100.0,
                "atr_sma": 100.0,
                "volume": 500.0,
                "volume_sma": 500.0,
            },
            position=position,
        )
        result = strategy.check_exit(ctx, position)
        assert result is None


# ─── Divers ──────────────────────────────────────────────────────────────────


class TestMisc:
    def test_min_candles(self):
        strategy = VwapRsiStrategy(_make_config())
        mc = strategy.min_candles
        assert "5m" in mc
        assert "15m" in mc
        assert mc["5m"] >= 288  # VWAP window

    def test_compute_indicators_returns_dict(self):
        """compute_indicators retourne un dict par TF indexé par timestamp."""
        strategy = VwapRsiStrategy(_make_config())

        # Créer des candles synthétiques avec timestamps uniques
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        candles_5m = []
        for i in range(350):
            ts = base_ts + timedelta(minutes=5 * i)
            price = 100.0 + np.sin(i / 20) * 5
            candles_5m.append(Candle(
                timestamp=ts,
                open=price - 0.5,
                high=price + 1,
                low=price - 1,
                close=price + 0.5,
                volume=100.0 + i,
                symbol="BTC/USDT",
                timeframe=TimeFrame.M5,
            ))

        candles_15m = []
        for i in range(100):
            ts = base_ts + timedelta(minutes=15 * i)
            price = 100.0 + np.sin(i / 7) * 5
            candles_15m.append(Candle(
                timestamp=ts,
                open=price - 0.5,
                high=price + 1,
                low=price - 1,
                close=price + 0.5,
                volume=300.0 + i,
                symbol="BTC/USDT",
                timeframe=TimeFrame.M15,
            ))

        result = strategy.compute_indicators({"5m": candles_5m, "15m": candles_15m})
        assert "5m" in result
        assert "15m" in result
        assert len(result["5m"]) == 350
        # Vérifier les champs
        first_ts = list(result["5m"].keys())[0]
        assert "rsi" in result["5m"][first_ts]
        assert "vwap" in result["5m"][first_ts]

    def test_get_params(self):
        config = _make_config()
        strategy = VwapRsiStrategy(config)
        params = strategy.get_params()
        assert params["rsi_period"] == 14
        assert params["tp_percent"] == 0.5
