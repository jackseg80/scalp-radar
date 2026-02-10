"""Indicateurs techniques en pur numpy pour Scalp Radar.

Toutes les fonctions sont pures (pas d'état interne), prennent et retournent
des np.ndarray. Les premières valeurs sont NaN (période d'échauffement).
"""

from __future__ import annotations

import numpy as np

from backend.core.models import MarketRegime


# ─── MOYENNES ────────────────────────────────────────────────────────────────


def sma(values: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average. Les period-1 premières valeurs sont NaN."""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full_like(values, np.nan, dtype=float)
    cumsum = np.cumsum(values)
    result[period - 1 :] = (cumsum[period - 1 :] - np.concatenate(([0], cumsum[:-period]))) / period
    return result


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average. Les period-1 premières valeurs sont NaN."""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full_like(values, np.nan, dtype=float)
    multiplier = 2.0 / (period + 1)
    # Seed : SMA des period premières valeurs
    result[period - 1] = np.mean(values[:period])
    for i in range(period, len(values)):
        result[i] = values[i] * multiplier + result[i - 1] * (1 - multiplier)
    return result


# ─── RSI (Wilder smoothing) ─────────────────────────────────────────────────


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI avec lissage de Wilder (exponentiel).

    Formule Wilder : avg = prev_avg × (n-1)/n + current/n
    Les period premières valeurs sont NaN.
    """
    if len(closes) < period + 1:
        return np.full_like(closes, np.nan, dtype=float)

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    result = np.full(len(closes), np.nan, dtype=float)

    # Seed : moyenne simple des period premières variations
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)

    # Wilder smoothing
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return result


# ─── VWAP Rolling ────────────────────────────────────────────────────────────


def vwap_rolling(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    window: int = 288,
) -> np.ndarray:
    """VWAP rolling sur fenêtre glissante.

    typical_price = (H + L + C) / 3
    VWAP = Σ(typical_price × volume) / Σ(volume) sur la fenêtre.
    window=288 pour 24h en bougies 5min.
    """
    if len(closes) < window:
        return np.full_like(closes, np.nan, dtype=float)

    typical = (highs + lows + closes) / 3.0
    tp_vol = typical * volumes

    result = np.full(len(closes), np.nan, dtype=float)

    # Sommes cumulées pour calcul glissant
    cumsum_tp_vol = np.cumsum(tp_vol)
    cumsum_vol = np.cumsum(volumes)

    for i in range(window - 1, len(closes)):
        start = i - window + 1
        sum_tp_vol = cumsum_tp_vol[i] - (cumsum_tp_vol[start - 1] if start > 0 else 0)
        sum_vol = cumsum_vol[i] - (cumsum_vol[start - 1] if start > 0 else 0)
        if sum_vol > 0:
            result[i] = sum_tp_vol / sum_vol
        else:
            result[i] = typical[i]

    return result


# ─── ATR (Wilder smoothing) ─────────────────────────────────────────────────


def atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average True Range avec lissage de Wilder.

    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    Les period premières valeurs sont NaN.
    """
    if len(closes) < period + 1:
        return np.full_like(closes, np.nan, dtype=float)

    result = np.full(len(closes), np.nan, dtype=float)

    # True Range
    tr = np.empty(len(closes))
    tr[0] = highs[0] - lows[0]
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    # Seed : moyenne simple
    atr_val = np.mean(tr[1 : period + 1])
    result[period] = atr_val

    # Wilder smoothing
    for i in range(period + 1, len(closes)):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        result[i] = atr_val

    return result


# ─── ADX + DI+/DI- ──────────────────────────────────────────────────────────


def adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average Directional Index.

    Retourne (adx, di_plus, di_minus).
    Les 2×period premières valeurs environ sont NaN.
    """
    n = len(closes)
    if n < 2 * period + 1:
        nan_arr = np.full(n, np.nan, dtype=float)
        return nan_arr.copy(), nan_arr.copy(), nan_arr.copy()

    adx_arr = np.full(n, np.nan, dtype=float)
    di_plus_arr = np.full(n, np.nan, dtype=float)
    di_minus_arr = np.full(n, np.nan, dtype=float)

    # +DM et -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    tr[0] = highs[0] - lows[0]

    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]

        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    # Smoothed TR, +DM, -DM (Wilder smoothing)
    sm_tr = np.mean(tr[1 : period + 1]) * period
    sm_plus = np.mean(plus_dm[1 : period + 1]) * period
    sm_minus = np.mean(minus_dm[1 : period + 1]) * period

    # DI+/DI- à partir de period
    if sm_tr > 0:
        di_plus_arr[period] = 100.0 * sm_plus / sm_tr
        di_minus_arr[period] = 100.0 * sm_minus / sm_tr
    else:
        di_plus_arr[period] = 0.0
        di_minus_arr[period] = 0.0

    dx_values = []
    di_sum = di_plus_arr[period] + di_minus_arr[period]
    if di_sum > 0:
        dx_values.append(abs(di_plus_arr[period] - di_minus_arr[period]) / di_sum * 100.0)
    else:
        dx_values.append(0.0)

    for i in range(period + 1, n):
        sm_tr = sm_tr - sm_tr / period + tr[i]
        sm_plus = sm_plus - sm_plus / period + plus_dm[i]
        sm_minus = sm_minus - sm_minus / period + minus_dm[i]

        if sm_tr > 0:
            di_plus_arr[i] = 100.0 * sm_plus / sm_tr
            di_minus_arr[i] = 100.0 * sm_minus / sm_tr
        else:
            di_plus_arr[i] = 0.0
            di_minus_arr[i] = 0.0

        di_sum = di_plus_arr[i] + di_minus_arr[i]
        if di_sum > 0:
            dx_values.append(abs(di_plus_arr[i] - di_minus_arr[i]) / di_sum * 100.0)
        else:
            dx_values.append(0.0)

    # ADX = Wilder smoothed DX
    # dx_values[j] correspond à l'index original period + j
    # Le seed utilise dx_values[0:period] → placé à l'index 2*period - 1
    if len(dx_values) >= period:
        adx_val = np.mean(dx_values[:period])
        adx_arr[2 * period - 1] = adx_val
        for i in range(period, len(dx_values)):
            adx_val = (adx_val * (period - 1) + dx_values[i]) / period
            target_idx = period + i
            if target_idx < n:
                adx_arr[target_idx] = adx_val

    return adx_arr, di_plus_arr, di_minus_arr


# ─── Volume SMA ─────────────────────────────────────────────────────────────


def volume_sma(volumes: np.ndarray, period: int = 20) -> np.ndarray:
    """SMA du volume pour détecter les spikes."""
    return sma(volumes, period)


# ─── Détection de régime de marché ───────────────────────────────────────────


def detect_market_regime(
    adx_value: float,
    di_plus_value: float,
    di_minus_value: float,
    atr_value: float,
    atr_sma_value: float,
) -> MarketRegime:
    """Détecte le régime de marché actuel à partir des valeurs scalaires.

    Priorité :
    1. ATR > 2× SMA(ATR) → HIGH_VOLATILITY
    2. ATR < 0.5× SMA(ATR) → LOW_VOLATILITY
    3. ADX > 25 et DI+ > DI- → TRENDING_UP
    4. ADX > 25 et DI- > DI+ → TRENDING_DOWN
    5. ADX < 20 → RANGING
    6. Sinon → RANGING (zone neutre 20-25)
    """
    # Vérifier les NaN
    if any(np.isnan(v) for v in [adx_value, di_plus_value, di_minus_value, atr_value, atr_sma_value]):
        return MarketRegime.RANGING

    # Volatilité (prioritaire)
    if atr_sma_value > 0:
        if atr_value > 2.0 * atr_sma_value:
            return MarketRegime.HIGH_VOLATILITY
        if atr_value < 0.5 * atr_sma_value:
            return MarketRegime.LOW_VOLATILITY

    # Tendance
    if adx_value > 25:
        if di_plus_value > di_minus_value:
            return MarketRegime.TRENDING_UP
        return MarketRegime.TRENDING_DOWN

    return MarketRegime.RANGING
