"""Indicateurs techniques en pur numpy pour Scalp Radar.

Toutes les fonctions sont pures (pas d'état interne), prennent et retournent
des np.ndarray. Les premières valeurs sont NaN (période d'échauffement).
"""

from __future__ import annotations

import numpy as np

from backend.core.models import MarketRegime

# Numba JIT DÉSACTIVÉ pour indicators.py — segfault aléatoire sur Python 3.13
# (access violation dans _rsi_wilder_loop / _adx_wilder_loop).
# Le vrai speedup JIT est dans fast_backtest.py (boucle trades), pas ici.
# Ces boucles tournent sur des arrays de 500 éléments max (IncrementalIndicatorEngine)
# → la différence Python pur vs JIT est négligeable.
try:
    from numba import njit as _njit_real  # noqa: F401 — gardé pour fast_backtest.py

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False


def njit(*args, **kwargs):  # type: ignore[misc]
    """No-op decorator — remplace @njit pour éviter les segfaults Numba/Py3.13."""
    if args and callable(args[0]):
        return args[0]
    return lambda func: func


# ─── MOYENNES ────────────────────────────────────────────────────────────────


def sma(values: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average. Les period-1 premières valeurs sont NaN."""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full_like(values, np.nan, dtype=float)
    cumsum = np.cumsum(values)
    result[period - 1 :] = (cumsum[period - 1 :] - np.concatenate(([0], cumsum[:-period]))) / period
    return result


@njit(cache=False)
def _ema_loop(values, result, period, multiplier):
    for i in range(period, len(values)):
        result[i] = values[i] * multiplier + result[i - 1] * (1 - multiplier)
    return result


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average. Les period-1 premières valeurs sont NaN."""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    values = np.ascontiguousarray(values, dtype=np.float64)
    result = np.full_like(values, np.nan, dtype=np.float64)
    multiplier = 2.0 / (period + 1)
    # Seed : SMA des period premières valeurs
    result[period - 1] = np.mean(values[:period])
    _ema_loop(values, result, period, multiplier)
    return result


# ─── RSI (Wilder smoothing) ─────────────────────────────────────────────────


@njit(cache=False)
def _rsi_wilder_loop(gains, losses, result, period, avg_gain, avg_loss):
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return result


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI avec lissage de Wilder (exponentiel).

    Formule Wilder : avg = prev_avg × (n-1)/n + current/n
    Les period premières valeurs sont NaN.
    """
    if len(closes) < period + 1:
        return np.full_like(closes, np.nan, dtype=float)

    # Garantir contiguïté + dtype pour JIT (évite segfault Numba)
    closes = np.ascontiguousarray(closes, dtype=np.float64)

    deltas = np.diff(closes)
    gains = np.ascontiguousarray(np.where(deltas > 0, deltas, 0.0), dtype=np.float64)
    losses = np.ascontiguousarray(np.where(deltas < 0, -deltas, 0.0), dtype=np.float64)

    result = np.full(len(closes), np.nan, dtype=np.float64)

    # Seed : moyenne simple des period premières variations
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)

    # Wilder smoothing (JIT-compiled)
    _rsi_wilder_loop(gains, losses, result, period, avg_gain, avg_loss)
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


@njit(cache=False)
def _wilder_smooth(data, result, period, seed_val, start_idx):
    val = seed_val
    for i in range(start_idx, len(data)):
        val = (val * (period - 1) + data[i]) / period
        result[i] = val
    return result


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

    # Garantir contiguïté + dtype pour JIT
    highs = np.ascontiguousarray(highs, dtype=np.float64)
    lows = np.ascontiguousarray(lows, dtype=np.float64)
    closes = np.ascontiguousarray(closes, dtype=np.float64)

    result = np.full(len(closes), np.nan, dtype=np.float64)

    # True Range (vectorisé)
    tr = np.empty(len(closes), dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    tr[1:] = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
    )

    # Seed : moyenne simple
    atr_val = float(np.mean(tr[1 : period + 1]))
    result[period] = atr_val

    # Wilder smoothing (JIT-compiled)
    _wilder_smooth(tr, result, period, atr_val, period + 1)
    return result


# ─── ADX + DI+/DI- ──────────────────────────────────────────────────────────


@njit(cache=False)
def _adx_wilder_loop(tr, plus_dm, minus_dm, period, n, adx_arr, di_plus_arr, di_minus_arr):
    # Seeds : somme des period premières valeurs (= mean × period)
    sm_tr = 0.0
    sm_plus = 0.0
    sm_minus = 0.0
    for j in range(1, period + 1):
        sm_tr += tr[j]
        sm_plus += plus_dm[j]
        sm_minus += minus_dm[j]

    # DI+/DI- au premier index (period)
    if sm_tr > 0.0:
        di_plus_arr[period] = 100.0 * sm_plus / sm_tr
        di_minus_arr[period] = 100.0 * sm_minus / sm_tr
    else:
        di_plus_arr[period] = 0.0
        di_minus_arr[period] = 0.0

    # DX pré-alloué (pas de list.append)
    dx_arr = np.empty(n - period, dtype=np.float64)
    di_sum = di_plus_arr[period] + di_minus_arr[period]
    if di_sum > 0.0:
        dx_arr[0] = abs(di_plus_arr[period] - di_minus_arr[period]) / di_sum * 100.0
    else:
        dx_arr[0] = 0.0
    dx_count = 1

    for i in range(period + 1, n):
        sm_tr = sm_tr - sm_tr / period + tr[i]
        sm_plus = sm_plus - sm_plus / period + plus_dm[i]
        sm_minus = sm_minus - sm_minus / period + minus_dm[i]

        if sm_tr > 0.0:
            di_plus_arr[i] = 100.0 * sm_plus / sm_tr
            di_minus_arr[i] = 100.0 * sm_minus / sm_tr
        else:
            di_plus_arr[i] = 0.0
            di_minus_arr[i] = 0.0

        di_sum = di_plus_arr[i] + di_minus_arr[i]
        if di_sum > 0.0:
            dx_arr[dx_count] = abs(di_plus_arr[i] - di_minus_arr[i]) / di_sum * 100.0
        else:
            dx_arr[dx_count] = 0.0
        dx_count += 1

    # ADX = Wilder smoothed DX
    if dx_count >= period:
        adx_val = 0.0
        for k in range(period):
            adx_val += dx_arr[k]
        adx_val /= period
        adx_arr[2 * period - 1] = adx_val
        for k in range(period, dx_count):
            adx_val = (adx_val * (period - 1) + dx_arr[k]) / period
            target_idx = period + k
            if target_idx < n:
                adx_arr[target_idx] = adx_val


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
        nan_arr = np.full(n, np.nan, dtype=np.float64)
        return nan_arr.copy(), nan_arr.copy(), nan_arr.copy()

    # Garantir contiguïté + dtype pour JIT
    highs = np.ascontiguousarray(highs, dtype=np.float64)
    lows = np.ascontiguousarray(lows, dtype=np.float64)
    closes = np.ascontiguousarray(closes, dtype=np.float64)

    adx_arr = np.full(n, np.nan, dtype=np.float64)
    di_plus_arr = np.full(n, np.nan, dtype=np.float64)
    di_minus_arr = np.full(n, np.nan, dtype=np.float64)

    # +DM, -DM et TR (vectorisé)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)

    tr[0] = highs[0] - lows[0]
    tr[1:] = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
    )

    up = highs[1:] - highs[:-1]
    down = lows[:-1] - lows[1:]
    plus_dm[1:] = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm[1:] = np.where((down > up) & (down > 0), down, 0.0)

    # Wilder smoothing DI+/DI-/DX + ADX (JIT-compiled)
    _adx_wilder_loop(tr, plus_dm, minus_dm, period, n, adx_arr, di_plus_arr, di_minus_arr)
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


# ─── Bollinger Bands ─────────────────────────────────────────────────────────


def bollinger_bands(
    closes: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands : SMA ± std_dev × rolling std.

    Retourne (sma_arr, upper_band, lower_band).
    Les period-1 premières valeurs sont NaN.
    """
    n = len(closes)
    if n < period:
        nan_arr = np.full(n, np.nan, dtype=float)
        return nan_arr.copy(), nan_arr.copy(), nan_arr.copy()

    sma_arr = sma(closes, period)

    # Rolling standard deviation (vectorisé via sliding_window_view)
    from numpy.lib.stride_tricks import sliding_window_view

    std_arr = np.full(n, np.nan, dtype=float)
    if n >= period:
        windows = sliding_window_view(closes, period)
        std_arr[period - 1 :] = np.std(windows, axis=1, ddof=0)

    upper = sma_arr + std_dev * std_arr
    lower = sma_arr - std_dev * std_arr

    return sma_arr, upper, lower


# ─── SuperTrend ──────────────────────────────────────────────────────────────


@njit(cache=False)
def _supertrend_loop(highs, lows, closes, atr_arr, multiplier, st_values, direction):
    n = len(closes)

    # Trouver le premier index avec ATR valide
    first_valid = -1
    for i in range(n):
        if not np.isnan(atr_arr[i]):
            first_valid = i
            break

    if first_valid < 0 or first_valid >= n - 1:
        return st_values, direction

    # Initialiser au premier index valide
    hl2 = (highs[first_valid] + lows[first_valid]) / 2.0
    upper_band = hl2 + multiplier * atr_arr[first_valid]
    lower_band = hl2 - multiplier * atr_arr[first_valid]
    if closes[first_valid] > upper_band:
        direction[first_valid] = 1.0
        st_values[first_valid] = lower_band
    else:
        direction[first_valid] = -1.0
        st_values[first_valid] = upper_band

    prev_upper = upper_band
    prev_lower = lower_band

    for i in range(first_valid + 1, n):
        if np.isnan(atr_arr[i]):
            direction[i] = direction[i - 1]
            st_values[i] = st_values[i - 1]
            continue

        hl2 = (highs[i] + lows[i]) / 2.0
        upper_band = hl2 + multiplier * atr_arr[i]
        lower_band = hl2 - multiplier * atr_arr[i]

        # Ajuster les bandes : ne pas s'éloigner du prix
        if closes[i - 1] <= prev_upper:
            upper_band = min(upper_band, prev_upper)
        if closes[i - 1] >= prev_lower:
            lower_band = max(lower_band, prev_lower)

        # Déterminer la direction
        prev_dir = direction[i - 1]
        if prev_dir == 1.0:  # Était UP (bullish)
            if closes[i] < lower_band:
                direction[i] = -1.0  # Flip DOWN
                st_values[i] = upper_band
            else:
                direction[i] = 1.0
                st_values[i] = lower_band
        else:  # Était DOWN (bearish)
            if closes[i] > upper_band:
                direction[i] = 1.0  # Flip UP
                st_values[i] = lower_band
            else:
                direction[i] = -1.0
                st_values[i] = upper_band

        prev_upper = upper_band
        prev_lower = lower_band

    return st_values, direction


def supertrend(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr_arr: np.ndarray,
    multiplier: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """SuperTrend indicator.

    Calcul itératif (la direction précédente détermine la bande active).

    Retourne (supertrend_values, direction).
    direction[i] = 1 (UP/bullish) ou -1 (DOWN/bearish).
    NaN tant que l'ATR n'est pas disponible.
    """
    n = len(closes)
    # Garantir contiguïté + dtype pour JIT
    highs = np.ascontiguousarray(highs, dtype=np.float64)
    lows = np.ascontiguousarray(lows, dtype=np.float64)
    closes = np.ascontiguousarray(closes, dtype=np.float64)
    atr_arr = np.ascontiguousarray(atr_arr, dtype=np.float64)
    st_values = np.full(n, np.nan, dtype=np.float64)
    direction = np.full(n, np.nan, dtype=np.float64)
    _supertrend_loop(highs, lows, closes, atr_arr, multiplier, st_values, direction)
    return st_values, direction
