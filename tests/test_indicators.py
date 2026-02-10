"""Tests pour backend/core/indicators.py."""

import numpy as np
import pytest

from backend.core.indicators import (
    adx,
    atr,
    detect_market_regime,
    ema,
    rsi,
    sma,
    volume_sma,
    vwap_rolling,
)
from backend.core.models import MarketRegime


# ─── SMA ─────────────────────────────────────────────────────────────────────


class TestSMA:
    def test_basic(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(values, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_too_short(self):
        values = np.array([1.0, 2.0])
        result = sma(values, 5)
        assert all(np.isnan(result))

    def test_single_period(self):
        values = np.array([10.0, 20.0, 30.0])
        result = sma(values, 1)
        np.testing.assert_array_almost_equal(result, values)


# ─── EMA ─────────────────────────────────────────────────────────────────────


class TestEMA:
    def test_basic(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ema(values, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # Seed = SMA(1,2,3) = 2.0
        assert result[2] == pytest.approx(2.0)
        # EMA = 4 * 0.5 + 2.0 * 0.5 = 3.0
        assert result[3] == pytest.approx(3.0)
        # EMA = 5 * 0.5 + 3.0 * 0.5 = 4.0
        assert result[4] == pytest.approx(4.0)

    def test_too_short(self):
        values = np.array([1.0])
        result = ema(values, 5)
        assert all(np.isnan(result))


# ─── RSI ─────────────────────────────────────────────────────────────────────


class TestRSI:
    def test_warmup_nan(self):
        """Les period premières valeurs doivent être NaN."""
        closes = np.arange(1.0, 30.0)  # 29 valeurs
        result = rsi(closes, period=14)
        assert all(np.isnan(result[:14]))
        assert not np.isnan(result[14])

    def test_all_up(self):
        """Si le prix monte constamment, RSI doit être proche de 100."""
        closes = np.arange(100.0, 130.0)  # 30 valeurs montantes
        result = rsi(closes, period=14)
        # Après warmup, RSI devrait être 100 (que des gains, pas de pertes)
        assert result[-1] == pytest.approx(100.0)

    def test_all_down(self):
        """Si le prix descend constamment, RSI doit être proche de 0."""
        closes = np.arange(130.0, 100.0, -1.0)  # 30 valeurs descendantes
        result = rsi(closes, period=14)
        assert result[-1] == pytest.approx(0.0, abs=0.1)

    def test_range_0_100(self):
        """RSI doit rester entre 0 et 100."""
        np.random.seed(42)
        closes = np.cumsum(np.random.randn(200)) + 100
        result = rsi(closes, period=14)
        valid = result[~np.isnan(result)]
        assert all(0 <= v <= 100 for v in valid)

    def test_too_short(self):
        closes = np.array([1.0, 2.0, 3.0])
        result = rsi(closes, period=14)
        assert all(np.isnan(result))


# ─── VWAP Rolling ────────────────────────────────────────────────────────────


class TestVWAPRolling:
    def test_known_values(self):
        """Test sur 5 bougies avec fenêtre de 3."""
        highs = np.array([12.0, 14.0, 13.0, 15.0, 16.0])
        lows = np.array([10.0, 11.0, 11.0, 12.0, 13.0])
        closes = np.array([11.0, 13.0, 12.0, 14.0, 15.0])
        volumes = np.array([100.0, 200.0, 150.0, 300.0, 250.0])

        result = vwap_rolling(highs, lows, closes, volumes, window=3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])

        # Window [0:3] : typical = (12+10+11)/3, (14+11+13)/3, (13+11+12)/3
        tp0 = (12 + 10 + 11) / 3  # 11.0
        tp1 = (14 + 11 + 13) / 3  # 12.667
        tp2 = (13 + 11 + 12) / 3  # 12.0
        expected = (tp0 * 100 + tp1 * 200 + tp2 * 150) / (100 + 200 + 150)
        assert result[2] == pytest.approx(expected, rel=1e-4)

    def test_too_short(self):
        closes = np.array([1.0, 2.0])
        result = vwap_rolling(closes, closes, closes, closes, window=5)
        assert all(np.isnan(result))

    def test_zero_volume(self):
        """Avec volume 0, le VWAP doit retourner le typical price."""
        highs = np.array([10.0] * 5)
        lows = np.array([10.0] * 5)
        closes = np.array([10.0] * 5)
        volumes = np.array([0.0] * 5)
        result = vwap_rolling(highs, lows, closes, volumes, window=3)
        # Volume = 0 → fallback au typical price
        assert result[2] == pytest.approx(10.0)


# ─── ATR ─────────────────────────────────────────────────────────────────────


class TestATR:
    def test_constant_range(self):
        """Avec des bougies de range constant, ATR devrait être stable."""
        n = 30
        highs = np.full(n, 110.0)
        lows = np.full(n, 100.0)
        closes = np.full(n, 105.0)
        result = atr(highs, lows, closes, period=14)
        # ATR devrait être ~10 (range constant)
        assert result[-1] == pytest.approx(10.0, rel=0.1)

    def test_warmup_nan(self):
        highs = np.full(20, 110.0)
        lows = np.full(20, 100.0)
        closes = np.full(20, 105.0)
        result = atr(highs, lows, closes, period=14)
        assert all(np.isnan(result[:14]))
        assert not np.isnan(result[14])

    def test_too_short(self):
        result = atr(np.array([10.0]), np.array([9.0]), np.array([9.5]), period=14)
        assert all(np.isnan(result))


# ─── ADX + DI ────────────────────────────────────────────────────────────────


class TestADX:
    def test_returns_tuple(self):
        """adx() doit retourner un tuple de 3 arrays."""
        n = 50
        highs = np.cumsum(np.random.RandomState(42).rand(n)) + 100
        lows = highs - 2
        closes = (highs + lows) / 2
        adx_arr, di_plus, di_minus = adx(highs, lows, closes, period=14)
        assert len(adx_arr) == n
        assert len(di_plus) == n
        assert len(di_minus) == n

    def test_trending_up(self):
        """Sur une tendance haussière, DI+ devrait être > DI-."""
        n = 60
        # Prix montant régulièrement
        base = np.arange(n, dtype=float) * 2 + 100
        highs = base + 1
        lows = base - 1
        closes = base
        _, di_plus, di_minus = adx(highs, lows, closes, period=14)
        # Vérifier sur les dernières valeurs (après warmup)
        valid_plus = di_plus[~np.isnan(di_plus)]
        valid_minus = di_minus[~np.isnan(di_minus)]
        if len(valid_plus) > 0 and len(valid_minus) > 0:
            assert valid_plus[-1] > valid_minus[-1]

    def test_trending_down(self):
        """Sur une tendance baissière, DI- devrait être > DI+."""
        n = 60
        base = np.arange(n, dtype=float) * (-2) + 200
        highs = base + 1
        lows = base - 1
        closes = base
        _, di_plus, di_minus = adx(highs, lows, closes, period=14)
        valid_plus = di_plus[~np.isnan(di_plus)]
        valid_minus = di_minus[~np.isnan(di_minus)]
        if len(valid_plus) > 0 and len(valid_minus) > 0:
            assert valid_minus[-1] > valid_plus[-1]

    def test_too_short(self):
        adx_arr, di_plus, di_minus = adx(
            np.array([10.0]), np.array([9.0]), np.array([9.5]), period=14
        )
        assert all(np.isnan(adx_arr))


# ─── Volume SMA ──────────────────────────────────────────────────────────────


class TestVolumeSMA:
    def test_basic(self):
        volumes = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        result = volume_sma(volumes, period=3)
        assert result[2] == pytest.approx(200.0)


# ─── Detect Market Regime ────────────────────────────────────────────────────


class TestDetectMarketRegime:
    def test_trending_up(self):
        result = detect_market_regime(
            adx_value=30.0, di_plus_value=25.0, di_minus_value=10.0,
            atr_value=5.0, atr_sma_value=5.0,
        )
        assert result == MarketRegime.TRENDING_UP

    def test_trending_down(self):
        result = detect_market_regime(
            adx_value=30.0, di_plus_value=10.0, di_minus_value=25.0,
            atr_value=5.0, atr_sma_value=5.0,
        )
        assert result == MarketRegime.TRENDING_DOWN

    def test_ranging(self):
        result = detect_market_regime(
            adx_value=15.0, di_plus_value=12.0, di_minus_value=12.0,
            atr_value=5.0, atr_sma_value=5.0,
        )
        assert result == MarketRegime.RANGING

    def test_high_volatility_priority(self):
        """HIGH_VOLATILITY est prioritaire sur trending."""
        result = detect_market_regime(
            adx_value=30.0, di_plus_value=25.0, di_minus_value=10.0,
            atr_value=15.0, atr_sma_value=5.0,  # ATR > 2× SMA
        )
        assert result == MarketRegime.HIGH_VOLATILITY

    def test_low_volatility(self):
        result = detect_market_regime(
            adx_value=15.0, di_plus_value=10.0, di_minus_value=10.0,
            atr_value=2.0, atr_sma_value=10.0,  # ATR < 0.5× SMA
        )
        assert result == MarketRegime.LOW_VOLATILITY

    def test_nan_values(self):
        result = detect_market_regime(
            adx_value=float("nan"), di_plus_value=float("nan"),
            di_minus_value=float("nan"), atr_value=float("nan"),
            atr_sma_value=float("nan"),
        )
        assert result == MarketRegime.RANGING
