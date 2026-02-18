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

from dataclasses import dataclass, field

import numpy as np

import sqlite3

from backend.core.indicators import (
    adx,
    atr,
    bollinger_bands,
    detect_market_regime,
    rsi,
    sma,
    supertrend,
    volume_sma,
    vwap_rolling,
)
from backend.core.models import Candle, MarketRegime, TimeFrame
from loguru import logger as _ic_logger

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

    # Bollinger MR
    bb_sma: dict[int, np.ndarray]                            # {period: sma_array}
    bb_upper: dict[tuple[int, float], np.ndarray]             # {(period, std): upper_band}
    bb_lower: dict[tuple[int, float], np.ndarray]             # {(period, std): lower_band}

    # SuperTrend
    supertrend_direction: dict[tuple[int, float], np.ndarray]  # {(atr_period, mult): direction}

    # ATR multi-period (pour Donchian/SuperTrend avec atr_period variable)
    atr_by_period: dict[int, np.ndarray]                       # {period: atr_array}

    # Grid Multi-TF : Supertrend 4h mappé sur indices 1h (anti-lookahead)
    supertrend_dir_4h: dict[tuple[int, float], np.ndarray]     # {(st_atr_period, st_mult): dir_1h}

    # Grid Funding : funding rates alignés sur candles 1h (forward-fill, raw decimal)
    funding_rates_1h: np.ndarray | None = None    # shape (n,), raw decimal (/100 depuis DB)
    candle_timestamps: np.ndarray | None = None   # epoch ms, shape (n,)

    # Grid Trend : EMA et ADX multi-period
    ema_by_period: dict[int, np.ndarray] = field(default_factory=dict)   # {period: ema_array}
    adx_by_period: dict[int, np.ndarray] = field(default_factory=dict)   # {period: adx_array}


# ─── Resampling générique 1h → 4h/1d ──────────────────────────────────────

_TF_BUCKET_SIZES: dict[str, tuple[int, int]] = {
    # target_tf: (bucket_seconds, candles_per_bucket)
    "4h": (14400, 4),
    "1d": (86400, 24),
}


def resample_candles(candles_1h: list[Candle], target_tf: str) -> list[Candle]:
    """Resample des candles 1h vers un timeframe supérieur.

    Args:
        candles_1h: Candles source en 1h (triées chronologiquement).
        target_tf: ``"1h"`` (passthrough), ``"4h"``, ou ``"1d"``.

    Returns:
        Liste de Candle resamplees. Seuls les buckets COMPLETS sont inclus
        (4 candles pour 4h, 24 candles pour 1d).

    Les candles retournées ont :
        - timestamp = timestamp de la PREMIÈRE candle 1h du bucket
        - open = open de la première candle 1h
        - high = max des highs
        - low = min des lows
        - close = close de la dernière candle 1h
        - volume = somme des volumes
        - timeframe = target_tf
        - symbol/exchange = copiés de la source
    """
    if target_tf == "1h":
        return candles_1h

    if not candles_1h:
        return []

    if target_tf not in _TF_BUCKET_SIZES:
        raise ValueError(f"Timeframe cible non supporté: {target_tf} (supportés: 1h, 4h, 1d)")

    bucket_seconds, expected_count = _TF_BUCKET_SIZES[target_tf]
    target_timeframe = TimeFrame.from_string(target_tf)

    # Référence pour symbol/exchange
    ref = candles_1h[0]

    # Grouper les candles par bucket
    buckets: dict[int, list[Candle]] = {}
    bucket_order: list[int] = []
    for c in candles_1h:
        bucket_id = int(c.timestamp.timestamp()) // bucket_seconds
        if bucket_id not in buckets:
            buckets[bucket_id] = []
            bucket_order.append(bucket_id)
        buckets[bucket_id].append(c)

    # Construire les candles resamplees (buckets complets uniquement)
    result: list[Candle] = []
    n_buckets = len(bucket_order)
    for idx, bucket_id in enumerate(bucket_order):
        group = buckets[bucket_id]
        if len(group) != expected_count:
            # Warning si bucket incomplet au milieu des données (pas début/fin)
            if 0 < idx < n_buckets - 1:
                ts_str = group[0].timestamp.isoformat()
                _ic_logger.warning(
                    "resample_candles: bucket {} incomplet ({}/{} candles) au milieu des données, exclu",
                    ts_str, len(group), expected_count,
                )
            continue

        result.append(Candle(
            timestamp=group[0].timestamp,
            open=group[0].open,
            high=max(c.high for c in group),
            low=min(c.low for c in group),
            close=group[-1].close,
            volume=sum(c.volume for c in group),
            symbol=ref.symbol,
            timeframe=target_timeframe,
            exchange=ref.exchange,
        ))

    return result


def _load_funding_rates_aligned(
    symbol: str,
    exchange: str,
    candle_timestamps: np.ndarray,
    db_path: str,
) -> np.ndarray:
    """Charge les funding rates et les aligne sur les candles 1h.

    Anti-lookahead : searchsorted direct (le taux settlé à T est connu à T).
    Forward-fill via l'index searchsorted (chaque candle utilise le dernier taux connu).
    Les valeurs DB sont en % (×100) — on divise par 100 → raw decimal.

    Returns:
        np.ndarray shape (n,) — funding rate en raw decimal, NaN si indisponible.
    """
    n = len(candle_timestamps)
    if n == 0:
        return np.array([], dtype=float)

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT timestamp, funding_rate FROM funding_rates "
        "WHERE symbol = ? AND exchange = ? ORDER BY timestamp",
        (symbol, exchange),
    ).fetchall()
    conn.close()

    if not rows:
        return np.full(n, np.nan)

    fr_timestamps = np.array([r[0] for r in rows], dtype=np.float64)
    fr_values = np.array([r[1] for r in rows], dtype=np.float64) / 100  # % → raw decimal

    # Forward-fill : pour chaque candle, trouver le dernier funding connu
    indices = np.searchsorted(fr_timestamps, candle_timestamps, side="right") - 1
    result = np.full(n, np.nan)
    valid = indices >= 0
    result[valid] = fr_values[indices[valid]]

    return result


def build_cache(
    candles_by_tf: dict[str, list[Candle]],
    param_grid_values: dict[str, list],
    strategy_name: str,
    main_tf: str = "5m",
    filter_tf: str = "15m",
    db_path: str | None = None,
    symbol: str | None = None,
    exchange: str | None = None,
) -> IndicatorCache:
    """Construit le cache d'indicateurs pour une fenêtre de données.

    Args:
        candles_by_tf: Bougies par timeframe (au minimum main_tf).
        param_grid_values: Valeurs du grid {param_name: [values]}.
        strategy_name: Nom de la stratégie.
        main_tf: Timeframe principal (défaut "5m").
        filter_tf: Timeframe filtre (défaut "15m").
        db_path: Chemin DB pour charger les funding rates (grid_funding).
        symbol: Symbol pour la query funding (grid_funding).
        exchange: Exchange pour la query funding (grid_funding).
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

    # --- Rolling high/low pour momentum + donchian ---
    lookbacks: set[int] = set()
    if "breakout_lookback" in param_grid_values:
        lookbacks.update(param_grid_values["breakout_lookback"])
    if "entry_lookback" in param_grid_values:
        lookbacks.update(param_grid_values["entry_lookback"])
    rolling_high_dict = {lb: _rolling_max(highs, lb) for lb in lookbacks}
    rolling_low_dict = {lb: _rolling_min(lows, lb) for lb in lookbacks}

    # --- Filtre 15m aligné ---
    filter_candles = candles_by_tf.get(filter_tf, [])
    filter_adx_aligned, filter_di_plus_aligned, filter_di_minus_aligned = (
        _build_aligned_filter(main_candles, filter_candles)
    )

    # --- Bollinger Bands / SMA pour envelope_dca ---
    bb_sma_dict: dict[int, np.ndarray] = {}
    bb_upper_dict: dict[tuple[int, float], np.ndarray] = {}
    bb_lower_dict: dict[tuple[int, float], np.ndarray] = {}

    # Envelope DCA : seulement SMA par period (enveloppes calculées à la volée)
    if strategy_name in ("envelope_dca", "envelope_dca_short"):
        ma_periods: set[int] = set()
        if "ma_period" in param_grid_values:
            ma_periods.update(param_grid_values["ma_period"])
        if not ma_periods:
            ma_periods.add(7)
        for period in ma_periods:
            bb_sma_dict[period] = sma(closes, period)

    # Grid ATR / Grid Multi-TF : SMA + ATR multi-period
    if strategy_name in ("grid_atr", "grid_multi_tf", "grid_range_atr"):
        ma_periods_atr: set[int] = set()
        if "ma_period" in param_grid_values:
            ma_periods_atr.update(param_grid_values["ma_period"])
        if not ma_periods_atr:
            ma_periods_atr.add(14)
        for period in ma_periods_atr:
            if period not in bb_sma_dict:
                bb_sma_dict[period] = sma(closes, period)

    if strategy_name == "bollinger_mr":
        bb_periods: set[int] = set()
        bb_stds: set[float] = set()
        if "bb_period" in param_grid_values:
            bb_periods.update(param_grid_values["bb_period"])
        if "bb_std" in param_grid_values:
            bb_stds.update(param_grid_values["bb_std"])
        if not bb_periods:
            bb_periods.add(20)
        if not bb_stds:
            bb_stds.add(2.0)
        for period in bb_periods:
            bb_sma_arr, _, _ = bollinger_bands(closes, period, 1.0)
            bb_sma_dict[period] = bb_sma_arr
            for std_dev in bb_stds:
                _, upper, lower = bollinger_bands(closes, period, std_dev)
                bb_upper_dict[(period, std_dev)] = upper
                bb_lower_dict[(period, std_dev)] = lower

    if strategy_name == "boltrend":
        bol_windows: set[int] = set()
        bol_stds: set[float] = set()
        long_ma_windows: set[int] = set()
        if "bol_window" in param_grid_values:
            bol_windows.update(param_grid_values["bol_window"])
        if "bol_std" in param_grid_values:
            bol_stds.update(param_grid_values["bol_std"])
        if "long_ma_window" in param_grid_values:
            long_ma_windows.update(param_grid_values["long_ma_window"])
        if not bol_windows:
            bol_windows.add(100)
        if not bol_stds:
            bol_stds.add(2.2)
        if not long_ma_windows:
            long_ma_windows.add(550)

        for period in bol_windows:
            bb_sma_arr_bt, _, _ = bollinger_bands(closes, period, 1.0)
            bb_sma_dict[period] = bb_sma_arr_bt
            for std_dev in bol_stds:
                _, upper, lower = bollinger_bands(closes, period, std_dev)
                bb_upper_dict[(period, std_dev)] = upper
                bb_lower_dict[(period, std_dev)] = lower

        # SMA long terme (réutilise bb_sma_dict)
        for period in long_ma_windows:
            if period not in bb_sma_dict:
                bb_sma_dict[period] = sma(closes, period)

    if strategy_name == "grid_boltrend":
        bol_windows_gb: set[int] = set()
        bol_stds_gb: set[float] = set()
        long_ma_windows_gb: set[int] = set()
        if "bol_window" in param_grid_values:
            bol_windows_gb.update(param_grid_values["bol_window"])
        if "bol_std" in param_grid_values:
            bol_stds_gb.update(param_grid_values["bol_std"])
        if "long_ma_window" in param_grid_values:
            long_ma_windows_gb.update(param_grid_values["long_ma_window"])
        if not bol_windows_gb:
            bol_windows_gb.add(100)
        if not bol_stds_gb:
            bol_stds_gb.add(2.0)
        if not long_ma_windows_gb:
            long_ma_windows_gb.add(200)

        for period in bol_windows_gb:
            bb_sma_arr_gb, _, _ = bollinger_bands(closes, period, 1.0)
            bb_sma_dict[period] = bb_sma_arr_gb
            for std_dev in bol_stds_gb:
                _, upper, lower = bollinger_bands(closes, period, std_dev)
                bb_upper_dict[(period, std_dev)] = upper
                bb_lower_dict[(period, std_dev)] = lower

        # SMA long terme (réutilise bb_sma_dict)
        for period in long_ma_windows_gb:
            if period not in bb_sma_dict:
                bb_sma_dict[period] = sma(closes, period)

    # --- ATR multi-period (pour donchian/supertrend/grid_atr/grid_multi_tf) ---
    atr_by_period_dict: dict[int, np.ndarray] = {}
    if strategy_name in ("donchian_breakout", "supertrend", "grid_atr", "grid_multi_tf", "grid_trend", "grid_range_atr", "grid_boltrend"):
        atr_periods: set[int] = set()
        if "atr_period" in param_grid_values:
            atr_periods.update(param_grid_values["atr_period"])
        if not atr_periods:
            atr_periods.add(14)
        for p in atr_periods:
            if p not in atr_by_period_dict:
                atr_by_period_dict[p] = atr(highs, lows, closes, p)

    # --- SuperTrend (1h, pour stratégie supertrend) ---
    st_direction_dict: dict[tuple[int, float], np.ndarray] = {}
    if strategy_name == "supertrend":
        atr_multipliers: set[float] = set()
        if "atr_multiplier" in param_grid_values:
            atr_multipliers.update(param_grid_values["atr_multiplier"])
        if not atr_multipliers:
            atr_multipliers.add(3.0)
        for p in atr_by_period_dict:
            for mult in atr_multipliers:
                _, direction_arr = supertrend(highs, lows, closes, atr_by_period_dict[p], mult)
                st_direction_dict[(p, mult)] = direction_arr

    # --- Funding rates depuis DB (toutes stratégies grid) ---
    _GRID_STRATEGIES_WITH_FUNDING = {
        "grid_funding", "grid_atr", "grid_range_atr", "envelope_dca",
        "envelope_dca_short", "grid_multi_tf", "grid_trend", "grid_boltrend",
    }
    funding_1h: np.ndarray | None = None
    candle_ts: np.ndarray | None = None
    if strategy_name in _GRID_STRATEGIES_WITH_FUNDING:
        # candle_timestamps toujours nécessaire (settlement mask fast engine)
        candle_ts = np.array(
            [c.timestamp.timestamp() * 1000 for c in main_candles], dtype=np.float64,
        )
        if db_path is not None and symbol is not None and exchange is not None:
            funding_1h = _load_funding_rates_aligned(symbol, exchange, candle_ts, db_path)
        # SMA supplémentaire pour grid_funding TP
        if strategy_name == "grid_funding":
            ma_periods_fund: set[int] = set()
            if "ma_period" in param_grid_values:
                ma_periods_fund.update(param_grid_values["ma_period"])
            if not ma_periods_fund:
                ma_periods_fund.add(14)
            for period in ma_periods_fund:
                if period not in bb_sma_dict:
                    bb_sma_dict[period] = sma(closes, period)

    # --- SuperTrend 4h (pour grid_multi_tf, resampleé depuis 1h) ---
    st_dir_4h_dict: dict[tuple[int, float], np.ndarray] = {}
    if strategy_name == "grid_multi_tf":
        h4_highs, h4_lows, h4_closes, mapping_1h = _resample_1h_to_4h(
            main_candles, closes, highs, lows,
        )
        if len(h4_closes) > 0:
            st_atr_periods: set[int] = set()
            if "st_atr_period" in param_grid_values:
                st_atr_periods.update(param_grid_values["st_atr_period"])
            if not st_atr_periods:
                st_atr_periods.add(10)

            st_multipliers: set[float] = set()
            if "st_atr_multiplier" in param_grid_values:
                st_multipliers.update(param_grid_values["st_atr_multiplier"])
            if not st_multipliers:
                st_multipliers.add(3.0)

            for st_period in st_atr_periods:
                atr_4h = atr(h4_highs, h4_lows, h4_closes, st_period)
                for st_mult in st_multipliers:
                    _, st_dir = supertrend(h4_highs, h4_lows, h4_closes, atr_4h, st_mult)
                    # Mapper sur les indices 1h via le mapping anti-lookahead
                    st_dir_1h = np.full(n, np.nan)
                    for i in range(n):
                        idx_4h = mapping_1h[i]
                        if idx_4h >= 0 and not np.isnan(st_dir[idx_4h]):
                            st_dir_1h[i] = st_dir[idx_4h]
                    st_dir_4h_dict[(st_period, st_mult)] = st_dir_1h

    # --- Grid Trend : EMA + ADX multi-period ---
    ema_by_period_dict: dict[int, np.ndarray] = {}
    adx_by_period_dict: dict[int, np.ndarray] = {}
    if strategy_name == "grid_trend":
        from backend.core.indicators import ema as compute_ema
        for p in set(
            param_grid_values.get("ema_fast", []) + param_grid_values.get("ema_slow", [])
        ):
            if p not in ema_by_period_dict:
                ema_by_period_dict[p] = compute_ema(closes, p)

        for p in param_grid_values.get("adx_period", [14]):
            if p not in adx_by_period_dict:
                adx_p, _, _ = adx(highs, lows, closes, p)
                adx_by_period_dict[p] = adx_p

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
        bb_sma=bb_sma_dict,
        bb_upper=bb_upper_dict,
        bb_lower=bb_lower_dict,
        supertrend_direction=st_direction_dict,
        atr_by_period=atr_by_period_dict,
        supertrend_dir_4h=st_dir_4h_dict,
        funding_rates_1h=funding_1h,
        candle_timestamps=candle_ts,
        ema_by_period=ema_by_period_dict,
        adx_by_period=adx_by_period_dict,
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
    Vectorisé via sliding_window_view.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    result = np.full_like(arr, np.nan, dtype=float)
    n = len(arr)
    if n > window:
        views = sliding_window_view(arr, window)  # shape (n - window + 1, window)
        # views[j] = arr[j:j+window], on veut max(arr[i-window:i]) pour i=window..n-1
        # soit views[0..n-window-1] → result[window..n-1]
        result[window:] = np.max(views[: n - window], axis=1)
    return result


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling min sur fenêtre glissante (exclut l'élément courant).

    rolling_min[i] = min(arr[i-window:i]). NaN avant window.
    Vectorisé via sliding_window_view.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    result = np.full_like(arr, np.nan, dtype=float)
    n = len(arr)
    if n > window:
        views = sliding_window_view(arr, window)
        result[window:] = np.min(views[: n - window], axis=1)
    return result


def _resample_1h_to_4h(
    main_candles: list[Candle],
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample 1h → 4h aligné aux frontières UTC (00h, 04h, 08h, 12h, 16h, 20h).

    Returns:
        (highs_4h, lows_4h, closes_4h, mapping_1h_to_4h)
        mapping_1h_to_4h[i] = index du dernier 4h COMPLÉTÉ avant candle 1h[i], ou -1.
        Anti-lookahead : une candle 4h n'est utilisable qu'après sa clôture.
    """
    n = len(main_candles)
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, np.full(0, -1, dtype=np.int32)

    # Calculer le bucket 4h pour chaque candle 1h (14400s = 4h)
    bucket_size = 14400
    timestamps = np.array([c.timestamp.timestamp() for c in main_candles])
    buckets = (timestamps // bucket_size).astype(np.int64)

    # Identifier les buckets uniques dans l'ordre
    unique_buckets = []
    prev_bucket = -1
    for b in buckets:
        if b != prev_bucket:
            unique_buckets.append(int(b))
            prev_bucket = b
    unique_buckets_arr = np.array(unique_buckets)

    # Construire OHLC 4h pour chaque bucket
    h4_highs_list = []
    h4_lows_list = []
    h4_closes_list = []
    bucket_ids = []  # bucket id correspondant

    for bucket_id in unique_buckets_arr:
        mask = buckets == bucket_id
        h4_highs_list.append(float(np.max(highs[mask])))
        h4_lows_list.append(float(np.min(lows[mask])))
        # Close = dernier close du bucket
        indices = np.where(mask)[0]
        h4_closes_list.append(float(closes[indices[-1]]))
        bucket_ids.append(bucket_id)

    h4_highs_out = np.array(h4_highs_list, dtype=float)
    h4_lows_out = np.array(h4_lows_list, dtype=float)
    h4_closes_out = np.array(h4_closes_list, dtype=float)

    # Mapping anti-lookahead : pour chaque candle 1h[i],
    # trouver l'index du dernier bucket 4h COMPLÉTÉ (= bucket précédent).
    # Le bucket courant n'est pas encore complété.
    bucket_ids_arr = np.array(bucket_ids)
    mapping = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        current_bucket = buckets[i]
        # Trouver l'index du bucket PRÉCÉDENT le bucket courant
        idx = np.searchsorted(bucket_ids_arr, current_bucket, side="left") - 1
        if idx >= 0:
            mapping[i] = idx

    return h4_highs_out, h4_lows_out, h4_closes_out, mapping
