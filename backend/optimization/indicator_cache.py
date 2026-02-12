"""Cache d'indicateurs pré-calculés pour le fast backtest engine.

Construit TOUS les indicateurs une seule fois par fenêtre WFO, pour toutes
les variantes de paramètres du grid. Les fonctions existantes de indicators.py
sont réutilisées (pas de réimplémentation).

Usage dans le WFO :
    cache = build_cache(candles_by_tf, param_grid_values, strategy_name)
    for params in grid:
        result = run_backtest_from_cache(strategy_name, params, cache, bt_config)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.core.indicators import (
    adx,
    atr,
    detect_market_regime,
    rsi,
    sma,
    volume_sma,
    vwap_rolling,
)
from backend.core.models import Candle, MarketRegime

# Mapping MarketRegime → int8 pour stockage compact
REGIME_TO_INT: dict[MarketRegime, int] = {
    MarketRegime.RANGING: 0,
    MarketRegime.TRENDING_UP: 1,
    MarketRegime.TRENDING_DOWN: 2,
    MarketRegime.HIGH_VOLATILITY: 3,
    MarketRegime.LOW_VOLATILITY: 4,
}


@dataclass
class IndicatorCache:
    """Cache numpy de tous les indicateurs pour une fenêtre de données.

    Tous les arrays ont shape (n_candles,) sauf les dicts {param_variant: array}.
    """

    n_candles: int
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray
    total_days: float  # durée totale en jours (pour annualisation Sharpe)

    # Indicateurs par variante de paramètre
    rsi: dict[int, np.ndarray]          # {period: array}
    vwap: np.ndarray                     # VWAP rolling 24h (288 bougies 5m)
    vwap_distance_pct: np.ndarray        # (close - vwap) / vwap * 100
    adx_arr: np.ndarray                  # ADX period 14
    di_plus: np.ndarray
    di_minus: np.ndarray
    atr_arr: np.ndarray                  # ATR period 14
    atr_sma: np.ndarray                  # SMA(20) sur ATR valides (réaligné)
    volume_sma_arr: np.ndarray           # SMA(20) volume
    regime: np.ndarray                   # int8 : 0-4 (voir REGIME_TO_INT)

    # Momentum spécifique
    rolling_high: dict[int, np.ndarray]  # {lookback: rolling max sur highs}
    rolling_low: dict[int, np.ndarray]   # {lookback: rolling min sur lows}

    # Filtre 15m aligné sur les indices 5m
    filter_adx: np.ndarray               # (n_main,)
    filter_di_plus: np.ndarray
    filter_di_minus: np.ndarray


def build_cache(
    candles_by_tf: dict[str, list[Candle]],
    param_grid_values: dict[str, list],
    strategy_name: str,
    main_tf: str = "5m",
    filter_tf: str = "15m",
) -> IndicatorCache:
    """Construit le cache d'indicateurs pour une fenêtre de données.

    Réutilise les fonctions existantes de indicators.py.

    Args:
        candles_by_tf: Bougies par timeframe (au minimum main_tf).
        param_grid_values: Valeurs du grid {param_name: [values]}.
        strategy_name: "vwap_rsi" ou "momentum".
        main_tf: Timeframe principal (défaut "5m").
        filter_tf: Timeframe filtre (défaut "15m").
    """
    main_candles = candles_by_tf[main_tf]
    n = len(main_candles)

    opens = np.array([c.open for c in main_candles], dtype=float)
    highs = np.array([c.high for c in main_candles], dtype=float)
    lows = np.array([c.low for c in main_candles], dtype=float)
    closes = np.array([c.close for c in main_candles], dtype=float)
    volumes = np.array([c.volume for c in main_candles], dtype=float)

    # Durée totale en jours
    if n >= 2:
        total_days = (
            main_candles[-1].timestamp - main_candles[0].timestamp
        ).total_seconds() / 86400
    else:
        total_days = 1.0

    # --- RSI pour chaque period du grid ---
    rsi_periods: set[int] = set()
    if "rsi_period" in param_grid_values:
        rsi_periods.update(param_grid_values["rsi_period"])
    if not rsi_periods:
        rsi_periods.add(14)
    rsi_dict = {p: rsi(closes, p) for p in rsi_periods}

    # --- ADX, DI+, DI-, ATR (period fixe 14) ---
    adx_arr, di_plus_arr, di_minus_arr = adx(highs, lows, closes)
    atr_arr = atr(highs, lows, closes)

    # --- ATR SMA aligné (même logique que les stratégies) ---
    atr_sma_full = _compute_atr_sma_aligned(atr_arr)

    # --- VWAP et volume SMA ---
    vwap_arr = vwap_rolling(highs, lows, closes, volumes)
    vol_sma_arr = volume_sma(volumes)

    # VWAP distance %
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap_dist = np.where(
            (~np.isnan(vwap_arr)) & (vwap_arr > 0),
            (closes - vwap_arr) / vwap_arr * 100,
            np.nan,
        )

    # --- Régime par bougie ---
    regime_arr = np.empty(n, dtype=np.int8)
    for i in range(n):
        r = detect_market_regime(
            adx_arr[i], di_plus_arr[i], di_minus_arr[i],
            atr_arr[i], atr_sma_full[i],
        )
        regime_arr[i] = REGIME_TO_INT[r]

    # --- Rolling high/low pour momentum ---
    lookbacks: set[int] = set()
    if "breakout_lookback" in param_grid_values:
        lookbacks.update(param_grid_values["breakout_lookback"])
    rolling_high_dict = {lb: _rolling_max(highs, lb) for lb in lookbacks}
    rolling_low_dict = {lb: _rolling_min(lows, lb) for lb in lookbacks}

    # --- Filtre 15m aligné ---
    filter_candles = candles_by_tf.get(filter_tf, [])
    filter_adx_aligned, filter_di_plus_aligned, filter_di_minus_aligned = (
        _build_aligned_filter(main_candles, filter_candles)
    )

    return IndicatorCache(
        n_candles=n,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
        total_days=total_days,
        rsi=rsi_dict,
        vwap=vwap_arr,
        vwap_distance_pct=vwap_dist,
        adx_arr=adx_arr,
        di_plus=di_plus_arr,
        di_minus=di_minus_arr,
        atr_arr=atr_arr,
        atr_sma=atr_sma_full,
        volume_sma_arr=vol_sma_arr,
        regime=regime_arr,
        rolling_high=rolling_high_dict,
        rolling_low=rolling_low_dict,
        filter_adx=filter_adx_aligned,
        filter_di_plus=filter_di_plus_aligned,
        filter_di_minus=filter_di_minus_aligned,
    )


def _compute_atr_sma_aligned(atr_arr: np.ndarray) -> np.ndarray:
    """Calcule SMA(20) sur les valeurs ATR valides et réaligne sur l'array original.

    Reproduit exactement la logique des stratégies (vwap_rsi, momentum).
    """
    atr_sma_full = np.full_like(atr_arr, np.nan)
    valid_mask = ~np.isnan(atr_arr)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) >= 20:
        atr_valid = atr_arr[valid_mask]
        atr_sma_valid = sma(atr_valid, 20)
        for j, idx in enumerate(valid_indices):
            if not np.isnan(atr_sma_valid[j]):
                atr_sma_full[idx] = atr_sma_valid[j]
    return atr_sma_full


def _build_filter_index(
    main_candles: list[Candle], filter_candles: list[Candle],
) -> np.ndarray:
    """Pré-calcule filter_index[i] = dernier index 15m dont timestamp <= 5m[i].

    Équivalent vectorisé de BacktestEngine._last_available_before().
    Utilise searchsorted sur les timestamps epoch pour O(n log m).
    Retourne -1 si aucune bougie filtre disponible.
    """
    n_main = len(main_candles)
    if not filter_candles:
        return np.full(n_main, -1, dtype=np.int32)

    main_ts = np.array([c.timestamp.timestamp() for c in main_candles])
    filter_ts = np.array([c.timestamp.timestamp() for c in filter_candles])

    # searchsorted('right') - 1 : dernier index avec ts <= target
    indices = np.searchsorted(filter_ts, main_ts, side="right") - 1
    return indices.astype(np.int32)


def _build_aligned_filter(
    main_candles: list[Candle], filter_candles: list[Candle],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit les arrays ADX/DI+ /DI- du 15m alignés sur les indices 5m."""
    n_main = len(main_candles)

    if not filter_candles:
        nan_arr = np.full(n_main, np.nan)
        return nan_arr.copy(), nan_arr.copy(), nan_arr.copy()

    # Calculer indicateurs 15m
    f_closes = np.array([c.close for c in filter_candles], dtype=float)
    f_highs = np.array([c.high for c in filter_candles], dtype=float)
    f_lows = np.array([c.low for c in filter_candles], dtype=float)

    f_adx, f_di_plus, f_di_minus = adx(f_highs, f_lows, f_closes)

    # Aligner sur les indices 5m
    filter_index = _build_filter_index(main_candles, filter_candles)

    # Indices valides (>= 0), clip pour indexing sûr
    safe_index = np.clip(filter_index, 0, len(filter_candles) - 1)
    mask_valid = filter_index >= 0

    filter_adx_aligned = np.where(mask_valid, f_adx[safe_index], np.nan)
    filter_di_plus_aligned = np.where(mask_valid, f_di_plus[safe_index], np.nan)
    filter_di_minus_aligned = np.where(mask_valid, f_di_minus[safe_index], np.nan)

    return filter_adx_aligned, filter_di_plus_aligned, filter_di_minus_aligned


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling max sur fenêtre glissante (exclut l'élément courant).

    rolling_max[i] = max(arr[i-window:i]). NaN avant window.
    Même logique que MomentumStrategy._compute_main.
    """
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window, len(arr)):
        result[i] = np.max(arr[i - window:i])
    return result


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling min sur fenêtre glissante (exclut l'élément courant).

    rolling_min[i] = min(arr[i-window:i]). NaN avant window.
    """
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window, len(arr)):
        result[i] = np.min(arr[i - window:i])
    return result
