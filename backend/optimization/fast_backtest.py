"""Fast backtest engine pour le WFO — signaux vectorisés + boucle de trades minimale.

Pré-requis : un IndicatorCache construit par build_cache() (tous les indicateurs
pré-calculés une seule fois). Chaque combinaison de paramètres ne fait que des
comparaisons de seuils numpy (~0.1ms) + une boucle Python légère pour les trades.

Le moteur normal BacktestEngine reste intact comme référence pour la validation Bitget.
Ce module est utilisé UNIQUEMENT par le WFO pour accélérer le grid search IS.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from backend.backtesting.engine import BacktestConfig
from backend.optimization.indicator_cache import IndicatorCache

# Numba JIT (optionnel) — fallback transparent si non installé
try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover

    def njit(*args, **kwargs):  # type: ignore[misc]
        if args and callable(args[0]):
            return args[0]
        return lambda func: func

    NUMBA_AVAILABLE = False

# Type retour léger (même que walk_forward._ISResult)
_ISResult = tuple[dict[str, Any], float, float, float, int]

# Régimes en int8 (voir indicator_cache.REGIME_TO_INT)
_RANGING = 0
_HIGH_VOL = 3


# ─── Entry point ─────────────────────────────────────────────────────────────


def run_backtest_from_cache(
    strategy_name: str,
    params: dict[str, Any],
    cache: IndicatorCache,
    bt_config: BacktestConfig,
) -> _ISResult:
    """Lance un backtest rapide depuis le cache pré-calculé.

    Retourne (params, sharpe, net_return_pct, profit_factor, n_trades).
    """
    if strategy_name == "vwap_rsi":
        longs, shorts = _vwap_rsi_signals(params, cache)
    elif strategy_name == "momentum":
        longs, shorts = _momentum_signals(params, cache)
    elif strategy_name == "bollinger_mr":
        longs, shorts = _bollinger_mr_signals(params, cache)
    elif strategy_name == "donchian_breakout":
        longs, shorts = _donchian_signals(params, cache)
    elif strategy_name == "supertrend":
        longs, shorts = _supertrend_signals(params, cache)
    elif strategy_name == "boltrend":
        longs, shorts = _boltrend_signals(params, cache)
    else:
        raise ValueError(f"Stratégie inconnue pour fast engine: {strategy_name}")

    trade_pnls, trade_returns, final_capital = _simulate_trades(
        longs, shorts, cache, strategy_name, params, bt_config,
    )

    return _compute_fast_metrics(
        params, trade_pnls, trade_returns, final_capital,
        bt_config.initial_capital, cache.total_days,
    )


# ─── Signal generation (vectorisé) ──────────────────────────────────────────


def _vwap_rsi_signals(
    params: dict[str, Any], cache: IndicatorCache,
) -> tuple[np.ndarray, np.ndarray]:
    """Génère les masques long/short pour VWAP+RSI.

    Reproduit exactement VwapRsiStrategy.evaluate() en vectorisé.
    """
    rsi_arr = cache.rsi[params["rsi_period"]]

    # Indicateurs valides (pas NaN)
    valid = (
        ~np.isnan(rsi_arr)
        & ~np.isnan(cache.vwap)
        & ~np.isnan(cache.volume_sma_arr)
        & ~np.isnan(cache.closes)  # toujours vrai en pratique
    )

    # Volume spike (vol_sma > 0 implicite si pas NaN et > 0)
    vol_spike = (
        (cache.volume_sma_arr > 0)
        & (cache.volumes > cache.volume_sma_arr * params["volume_spike_multiplier"])
    )

    # Filtre 15m : ADX > trend_threshold → marché en tendance → PAS de mean reversion
    trend_filter = (
        ~np.isnan(cache.filter_adx)
        & (cache.filter_adx > params["trend_adx_threshold"])
    )

    # Direction 15m (pour filtrer longs/shorts individuellement)
    is_15m_bearish = (
        ~np.isnan(cache.filter_adx)
        & (cache.filter_adx > 20)
        & ~np.isnan(cache.filter_di_minus)
        & ~np.isnan(cache.filter_di_plus)
        & (cache.filter_di_minus > cache.filter_di_plus)
    )
    is_15m_bullish = (
        ~np.isnan(cache.filter_adx)
        & (cache.filter_adx > 20)
        & ~np.isnan(cache.filter_di_plus)
        & ~np.isnan(cache.filter_di_minus)
        & (cache.filter_di_plus > cache.filter_di_minus)
    )

    # Régime 5m : seulement RANGING (0) et LOW_VOLATILITY (4)
    regime_ok = (cache.regime == 0) | (cache.regime == 4)

    # VWAP deviation %
    vwap_dev = cache.vwap_distance_pct

    base = valid & vol_spike & ~trend_filter & regime_ok

    longs = (
        base
        & (rsi_arr < params["rsi_long_threshold"])
        & (vwap_dev < -params["vwap_deviation_entry"])
        & ~is_15m_bearish
    )
    shorts = (
        base
        & (rsi_arr > params["rsi_short_threshold"])
        & (vwap_dev > params["vwap_deviation_entry"])
        & ~is_15m_bullish
    )

    return longs, shorts


def _momentum_signals(
    params: dict[str, Any], cache: IndicatorCache,
) -> tuple[np.ndarray, np.ndarray]:
    """Génère les masques long/short pour Momentum Breakout.

    Reproduit exactement MomentumStrategy.evaluate() en vectorisé.
    """
    lookback = params["breakout_lookback"]
    rolling_high = cache.rolling_high[lookback]
    rolling_low = cache.rolling_low[lookback]

    # Indicateurs valides
    valid = (
        ~np.isnan(cache.closes)
        & ~np.isnan(cache.atr_arr)
        & ~np.isnan(rolling_high)
        & ~np.isnan(rolling_low)
    )

    # Filtre 15m : ADX >= 25 obligatoire (momentum = trend following)
    filter_trend = ~np.isnan(cache.filter_adx) & (cache.filter_adx >= 25)

    # Direction 15m
    is_15m_bullish = cache.filter_di_plus > cache.filter_di_minus
    is_15m_bearish = cache.filter_di_minus > cache.filter_di_plus

    # Volume spike
    vol_spike = (
        ~np.isnan(cache.volume_sma_arr)
        & (cache.volume_sma_arr > 0)
        & (cache.volumes > cache.volume_sma_arr * params["volume_confirmation_multiplier"])
    )

    base = valid & filter_trend & vol_spike

    longs = base & (cache.closes > rolling_high) & is_15m_bullish
    shorts = base & (cache.closes < rolling_low) & is_15m_bearish

    return longs, shorts


def _bollinger_mr_signals(
    params: dict[str, Any], cache: IndicatorCache,
) -> tuple[np.ndarray, np.ndarray]:
    """Génère les masques long/short pour Bollinger Mean Reversion.

    LONG si close < lower band, SHORT si close > upper band.
    """
    bb_period = params["bb_period"]
    bb_std = params["bb_std"]

    bb_sma_arr = cache.bb_sma[bb_period]
    bb_lower = cache.bb_lower[(bb_period, bb_std)]
    bb_upper = cache.bb_upper[(bb_period, bb_std)]

    valid = ~np.isnan(bb_sma_arr) & ~np.isnan(bb_lower) & ~np.isnan(bb_upper)

    longs = valid & (cache.closes < bb_lower)
    shorts = valid & (cache.closes > bb_upper)

    return longs, shorts


def _donchian_signals(
    params: dict[str, Any], cache: IndicatorCache,
) -> tuple[np.ndarray, np.ndarray]:
    """Génère les masques long/short pour Donchian Breakout.

    LONG si close > rolling_high (N bougies précédentes),
    SHORT si close < rolling_low.
    """
    lookback = params["entry_lookback"]
    rolling_high = cache.rolling_high[lookback]
    rolling_low = cache.rolling_low[lookback]

    valid = ~np.isnan(rolling_high) & ~np.isnan(rolling_low)

    longs = valid & (cache.closes > rolling_high)
    shorts = valid & (cache.closes < rolling_low)

    return longs, shorts


def _supertrend_signals(
    params: dict[str, Any], cache: IndicatorCache,
) -> tuple[np.ndarray, np.ndarray]:
    """Génère les masques long/short pour SuperTrend.

    LONG sur flip direction DOWN→UP, SHORT sur flip UP→DOWN.
    """
    key = (params["atr_period"], params["atr_multiplier"])
    direction = cache.supertrend_direction[key]

    n = len(direction)
    longs = np.zeros(n, dtype=bool)
    shorts = np.zeros(n, dtype=bool)

    if n < 2:
        return longs, shorts

    # Flip detection : comparer direction[i] avec direction[i-1]
    prev_dir = np.empty(n)
    prev_dir[0] = np.nan
    prev_dir[1:] = direction[:-1]

    valid = ~np.isnan(prev_dir) & ~np.isnan(direction)
    longs = valid & (prev_dir == -1.0) & (direction == 1.0)
    shorts = valid & (prev_dir == 1.0) & (direction == -1.0)

    return longs, shorts


def _boltrend_signals(
    params: dict[str, Any], cache: IndicatorCache,
) -> tuple[np.ndarray, np.ndarray]:
    """Génère les masques long/short pour Bollinger Trend Following.

    LONG : prev_close < prev_upper AND close > upper AND spread > min AND close > long_ma.
    SHORT : prev_close > prev_lower AND close < lower AND spread > min AND close < long_ma.
    """
    bol_window = params["bol_window"]
    bol_std = params["bol_std"]
    long_ma_window = params["long_ma_window"]
    min_bol_spread = params["min_bol_spread"]

    bb_upper = cache.bb_upper[(bol_window, bol_std)]
    bb_lower = cache.bb_lower[(bol_window, bol_std)]
    long_ma = cache.bb_sma[long_ma_window]

    prev_close = np.roll(cache.closes, 1)
    prev_upper = np.roll(bb_upper, 1)
    prev_lower = np.roll(bb_lower, 1)

    # Spread = (prev_upper - prev_lower) / prev_lower
    bb_spread = (prev_upper - prev_lower) / np.where(prev_lower > 0, prev_lower, 1.0)

    valid = ~np.isnan(bb_upper) & ~np.isnan(bb_lower) & ~np.isnan(long_ma)
    valid[0] = False  # np.roll wraparound : closes[-1] en position 0

    longs = (
        valid
        & (prev_close < prev_upper)
        & (cache.closes > bb_upper)
        & (bb_spread > min_bol_spread)
        & (cache.closes > long_ma)
    )
    shorts = (
        valid
        & (prev_close > prev_lower)
        & (cache.closes < bb_lower)
        & (bb_spread > min_bol_spread)
        & (cache.closes < long_ma)
    )

    return longs, shorts


# ─── Trade simulation JIT (Numba) ────────────────────────────────────────────


@njit(cache=True)
def _close_trade_numba(
    direction, entry_price, exit_price, quantity, entry_fee,
    exit_reason, regime_int, taker_fee, maker_fee,
    slippage_pct, high_vol_slippage_mult,
):
    """Calcule le net_pnl d'un trade fermé (JIT-compiled).

    exit_reason: 0=tp, 1=sl, 2=signal_exit, 3=end_of_data
    """
    # Gross PnL sur prix brut (pas d'actual_exit ajusté)
    if direction == 1:
        gross_pnl = (exit_price - entry_price) * quantity
    else:
        gross_pnl = (entry_price - exit_price) * quantity

    # Slippage : flat cost 1 seule fois (exit seulement, sauf TP)
    slippage_cost = 0.0
    if exit_reason != 0:  # Pas TP → appliquer slippage
        slippage_rate = slippage_pct
        if regime_int == 3:  # HIGH_VOLATILITY
            slippage_rate *= high_vol_slippage_mult
        slippage_cost = quantity * exit_price * slippage_rate

    if exit_reason == 0:  # TP → maker fee
        exit_fee = quantity * exit_price * maker_fee
    else:
        exit_fee = quantity * exit_price * taker_fee

    return gross_pnl - (entry_fee + exit_fee) - slippage_cost


@njit(cache=True)
def _simulate_vwap_rsi_numba(
    longs, shorts,
    opens, highs, lows, closes, regime,
    rsi_arr,
    tp_pct, sl_pct,
    initial_capital, taker_fee, maker_fee,
    slippage_pct, high_vol_slippage_mult,
    max_risk_per_trade,
):
    """Boucle de trades complète pour vwap_rsi (JIT-compiled)."""
    n = len(closes)
    trade_pnls = np.empty(n, dtype=np.float64)
    trade_returns = np.empty(n, dtype=np.float64)
    n_trades = 0
    capital = initial_capital

    in_position = False
    direction = 0
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0
    quantity = 0.0
    entry_fee = 0.0

    for i in range(n):
        # 1. Si position ouverte : vérifier TP/SL puis check_exit
        if in_position:
            exit_reason = -1  # pas de sortie

            # Check TP/SL inline
            if direction == 1:
                tp_hit = highs[i] >= tp_price
                sl_hit = lows[i] <= sl_price
            else:
                tp_hit = lows[i] <= tp_price
                sl_hit = highs[i] >= sl_price

            if tp_hit and sl_hit:
                # OHLC heuristic inline
                if closes[i] > opens[i]:
                    exit_reason = 0 if direction == 1 else 1
                elif closes[i] < opens[i]:
                    exit_reason = 1 if direction == 1 else 0
                else:
                    exit_reason = 1  # Doji → SL
            elif tp_hit:
                exit_reason = 0  # tp
            elif sl_hit:
                exit_reason = 1  # sl
            else:
                # check_exit vwap_rsi : RSI normalisé + en profit
                rsi_val = rsi_arr[i]
                close = closes[i]
                if not np.isnan(rsi_val) and not np.isnan(close):
                    if direction == 1:
                        if close > entry_price and rsi_val > 50.0:
                            exit_reason = 2  # signal_exit
                    else:
                        if close < entry_price and rsi_val < 50.0:
                            exit_reason = 2  # signal_exit

            if exit_reason >= 0:
                if exit_reason == 0:
                    exit_price = tp_price
                elif exit_reason == 1:
                    exit_price = sl_price
                else:
                    exit_price = closes[i]

                net_pnl = _close_trade_numba(
                    direction, entry_price, exit_price, quantity,
                    entry_fee, exit_reason, int(regime[i]),
                    taker_fee, maker_fee, slippage_pct, high_vol_slippage_mult,
                )
                if capital > 0.0:
                    trade_returns[n_trades] = net_pnl / capital
                else:
                    trade_returns[n_trades] = 0.0
                capital += net_pnl
                trade_pnls[n_trades] = net_pnl
                n_trades += 1
                in_position = False

        # 2. Si pas de position : entrée
        if not in_position and (longs[i] or shorts[i]):
            direction = 1 if longs[i] else -1
            ep = closes[i]

            if ep <= 0.0 or capital <= 0.0:
                continue

            tp_dist = ep * tp_pct / 100.0
            sl_dist = ep * sl_pct / 100.0

            if direction == 1:
                tp_price = ep + tp_dist
                sl_price = ep - sl_dist
            else:
                tp_price = ep - tp_dist
                sl_price = ep + sl_dist

            if sl_price <= 0.0 or tp_price <= 0.0:
                continue

            sl_distance_pct = abs(ep - sl_price) / ep
            sl_real_cost = sl_distance_pct + taker_fee + slippage_pct
            if sl_real_cost <= 0.0:
                continue

            risk_amount = capital * max_risk_per_trade
            notional = risk_amount / sl_real_cost
            quantity = notional / ep

            if quantity <= 0.0:
                continue

            entry_fee = quantity * ep * taker_fee
            if entry_fee >= capital:
                continue

            entry_price = ep
            capital -= entry_fee
            in_position = True

    # Force close fin de données
    if in_position:
        exit_price = closes[n - 1]
        net_pnl = _close_trade_numba(
            direction, entry_price, exit_price, quantity,
            entry_fee, 3, int(regime[n - 1]),
            taker_fee, maker_fee, slippage_pct, high_vol_slippage_mult,
        )
        if capital > 0.0:
            trade_returns[n_trades] = net_pnl / capital
        else:
            trade_returns[n_trades] = 0.0
        capital += net_pnl
        trade_pnls[n_trades] = net_pnl
        n_trades += 1

    return trade_pnls, trade_returns, n_trades, capital


def _run_simulate_vwap_rsi(
    longs: np.ndarray,
    shorts: np.ndarray,
    cache: IndicatorCache,
    params: dict[str, Any],
    bt_config: BacktestConfig,
) -> tuple[list[float], list[float], float]:
    """Wrapper Python pour _simulate_vwap_rsi_numba — extrait les arrays."""
    rsi_arr = cache.rsi[params["rsi_period"]]
    trade_pnls, trade_returns, n_trades, final_capital = _simulate_vwap_rsi_numba(
        longs, shorts,
        cache.opens, cache.highs, cache.lows, cache.closes, cache.regime,
        rsi_arr,
        float(params["tp_percent"]), float(params["sl_percent"]),
        bt_config.initial_capital, bt_config.taker_fee, bt_config.maker_fee,
        bt_config.slippage_pct, bt_config.high_vol_slippage_mult,
        bt_config.max_risk_per_trade,
    )
    return (
        trade_pnls[:n_trades].tolist(),
        trade_returns[:n_trades].tolist(),
        final_capital,
    )


@njit(cache=True)
def _simulate_momentum_numba(
    longs, shorts,
    opens, highs, lows, closes, regime,
    atr_arr, adx_arr,
    atr_mult_tp, atr_mult_sl, tp_pct, sl_pct,
    initial_capital, taker_fee, maker_fee,
    slippage_pct, high_vol_slippage_mult,
    max_risk_per_trade,
):
    """Boucle de trades complète pour momentum (JIT-compiled)."""
    n = len(closes)
    trade_pnls = np.empty(n, dtype=np.float64)
    trade_returns = np.empty(n, dtype=np.float64)
    n_trades = 0
    capital = initial_capital

    in_position = False
    direction = 0
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0
    quantity = 0.0
    entry_fee = 0.0

    for i in range(n):
        if in_position:
            exit_reason = -1

            if direction == 1:
                tp_hit = highs[i] >= tp_price
                sl_hit = lows[i] <= sl_price
            else:
                tp_hit = lows[i] <= tp_price
                sl_hit = highs[i] >= sl_price

            if tp_hit and sl_hit:
                if closes[i] > opens[i]:
                    exit_reason = 0 if direction == 1 else 1
                elif closes[i] < opens[i]:
                    exit_reason = 1 if direction == 1 else 0
                else:
                    exit_reason = 1
            elif tp_hit:
                exit_reason = 0
            elif sl_hit:
                exit_reason = 1
            else:
                # check_exit momentum : ADX < 20
                adx_val = adx_arr[i]
                if not np.isnan(adx_val) and adx_val < 20.0:
                    exit_reason = 2

            if exit_reason >= 0:
                if exit_reason == 0:
                    exit_price = tp_price
                elif exit_reason == 1:
                    exit_price = sl_price
                else:
                    exit_price = closes[i]

                net_pnl = _close_trade_numba(
                    direction, entry_price, exit_price, quantity,
                    entry_fee, exit_reason, int(regime[i]),
                    taker_fee, maker_fee, slippage_pct, high_vol_slippage_mult,
                )
                if capital > 0.0:
                    trade_returns[n_trades] = net_pnl / capital
                else:
                    trade_returns[n_trades] = 0.0
                capital += net_pnl
                trade_pnls[n_trades] = net_pnl
                n_trades += 1
                in_position = False

        if not in_position and (longs[i] or shorts[i]):
            direction = 1 if longs[i] else -1
            ep = closes[i]
            atr_val = atr_arr[i]

            if ep <= 0.0 or capital <= 0.0:
                continue

            # ATR-based TP/SL avec caps
            if not np.isnan(atr_val) and atr_val > 0.0:
                atr_tp = atr_val * atr_mult_tp
                atr_sl = atr_val * atr_mult_sl
            else:
                atr_tp = ep * tp_pct / 100.0
                atr_sl = ep * sl_pct / 100.0

            max_tp = ep * tp_pct / 100.0
            max_sl = ep * sl_pct / 100.0
            tp_dist = min(atr_tp, max_tp)
            sl_dist = min(atr_sl, max_sl)

            if direction == 1:
                tp_price = ep + tp_dist
                sl_price = ep - sl_dist
            else:
                tp_price = ep - tp_dist
                sl_price = ep + sl_dist

            if sl_price <= 0.0 or tp_price <= 0.0:
                continue

            sl_distance_pct = abs(ep - sl_price) / ep
            sl_real_cost = sl_distance_pct + taker_fee + slippage_pct
            if sl_real_cost <= 0.0:
                continue

            risk_amount = capital * max_risk_per_trade
            notional = risk_amount / sl_real_cost
            quantity = notional / ep

            if quantity <= 0.0:
                continue

            entry_fee = quantity * ep * taker_fee
            if entry_fee >= capital:
                continue

            entry_price = ep
            capital -= entry_fee
            in_position = True

    if in_position:
        exit_price = closes[n - 1]
        net_pnl = _close_trade_numba(
            direction, entry_price, exit_price, quantity,
            entry_fee, 3, int(regime[n - 1]),
            taker_fee, maker_fee, slippage_pct, high_vol_slippage_mult,
        )
        if capital > 0.0:
            trade_returns[n_trades] = net_pnl / capital
        else:
            trade_returns[n_trades] = 0.0
        capital += net_pnl
        trade_pnls[n_trades] = net_pnl
        n_trades += 1

    return trade_pnls, trade_returns, n_trades, capital


def _run_simulate_momentum(
    longs: np.ndarray, shorts: np.ndarray,
    cache: IndicatorCache, params: dict[str, Any], bt_config: BacktestConfig,
) -> tuple[list[float], list[float], float]:
    """Wrapper Python pour _simulate_momentum_numba."""
    pnls, rets, nt, cap = _simulate_momentum_numba(
        longs, shorts,
        cache.opens, cache.highs, cache.lows, cache.closes, cache.regime,
        cache.atr_arr, cache.adx_arr,
        float(params["atr_multiplier_tp"]), float(params["atr_multiplier_sl"]),
        float(params["tp_percent"]), float(params["sl_percent"]),
        bt_config.initial_capital, bt_config.taker_fee, bt_config.maker_fee,
        bt_config.slippage_pct, bt_config.high_vol_slippage_mult,
        bt_config.max_risk_per_trade,
    )
    return pnls[:nt].tolist(), rets[:nt].tolist(), cap


@njit(cache=True)
def _simulate_bollinger_numba(
    longs, shorts,
    opens, highs, lows, closes, regime,
    bb_sma_arr,
    sl_pct,
    initial_capital, taker_fee, maker_fee,
    slippage_pct, high_vol_slippage_mult,
    max_risk_per_trade,
):
    """Boucle de trades complète pour bollinger_mr (JIT-compiled)."""
    n = len(closes)
    trade_pnls = np.empty(n, dtype=np.float64)
    trade_returns = np.empty(n, dtype=np.float64)
    n_trades = 0
    capital = initial_capital

    in_position = False
    direction = 0
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0
    quantity = 0.0
    entry_fee = 0.0

    for i in range(n):
        if in_position:
            exit_reason = -1

            if direction == 1:
                tp_hit = highs[i] >= tp_price
                sl_hit = lows[i] <= sl_price
            else:
                tp_hit = lows[i] <= tp_price
                sl_hit = highs[i] >= sl_price

            if tp_hit and sl_hit:
                if closes[i] > opens[i]:
                    exit_reason = 0 if direction == 1 else 1
                elif closes[i] < opens[i]:
                    exit_reason = 1 if direction == 1 else 0
                else:
                    exit_reason = 1
            elif tp_hit:
                exit_reason = 0
            elif sl_hit:
                exit_reason = 1
            else:
                # check_exit bollinger_mr : SMA crossing
                sma_val = bb_sma_arr[i]
                close = closes[i]
                if not np.isnan(sma_val) and not np.isnan(close):
                    if direction == 1:
                        if close >= sma_val:
                            exit_reason = 2
                    else:
                        if close <= sma_val:
                            exit_reason = 2

            if exit_reason >= 0:
                if exit_reason == 0:
                    exit_price = tp_price
                elif exit_reason == 1:
                    exit_price = sl_price
                else:
                    exit_price = closes[i]

                net_pnl = _close_trade_numba(
                    direction, entry_price, exit_price, quantity,
                    entry_fee, exit_reason, int(regime[i]),
                    taker_fee, maker_fee, slippage_pct, high_vol_slippage_mult,
                )
                if capital > 0.0:
                    trade_returns[n_trades] = net_pnl / capital
                else:
                    trade_returns[n_trades] = 0.0
                capital += net_pnl
                trade_pnls[n_trades] = net_pnl
                n_trades += 1
                in_position = False

        if not in_position and (longs[i] or shorts[i]):
            direction = 1 if longs[i] else -1
            ep = closes[i]

            if ep <= 0.0 or capital <= 0.0:
                continue

            # TP très éloigné (SMA exit gère le vrai TP), SL % fixe
            sl_dist = ep * sl_pct / 100.0
            tp_dist = ep  # LONG: tp=2×entry, SHORT: tp=0 (won't open)

            if direction == 1:
                tp_price = ep + tp_dist
                sl_price = ep - sl_dist
            else:
                tp_price = ep - tp_dist
                sl_price = ep + sl_dist

            if sl_price <= 0.0 or tp_price <= 0.0:
                continue

            sl_distance_pct = abs(ep - sl_price) / ep
            sl_real_cost = sl_distance_pct + taker_fee + slippage_pct
            if sl_real_cost <= 0.0:
                continue

            risk_amount = capital * max_risk_per_trade
            notional = risk_amount / sl_real_cost
            quantity = notional / ep

            if quantity <= 0.0:
                continue

            entry_fee = quantity * ep * taker_fee
            if entry_fee >= capital:
                continue

            entry_price = ep
            capital -= entry_fee
            in_position = True

    if in_position:
        exit_price = closes[n - 1]
        net_pnl = _close_trade_numba(
            direction, entry_price, exit_price, quantity,
            entry_fee, 3, int(regime[n - 1]),
            taker_fee, maker_fee, slippage_pct, high_vol_slippage_mult,
        )
        if capital > 0.0:
            trade_returns[n_trades] = net_pnl / capital
        else:
            trade_returns[n_trades] = 0.0
        capital += net_pnl
        trade_pnls[n_trades] = net_pnl
        n_trades += 1

    return trade_pnls, trade_returns, n_trades, capital


def _run_simulate_bollinger(
    longs: np.ndarray, shorts: np.ndarray,
    cache: IndicatorCache, params: dict[str, Any], bt_config: BacktestConfig,
) -> tuple[list[float], list[float], float]:
    """Wrapper Python pour _simulate_bollinger_numba."""
    bb_sma_arr = cache.bb_sma[params["bb_period"]]
    pnls, rets, nt, cap = _simulate_bollinger_numba(
        longs, shorts,
        cache.opens, cache.highs, cache.lows, cache.closes, cache.regime,
        bb_sma_arr,
        float(params["sl_percent"]),
        bt_config.initial_capital, bt_config.taker_fee, bt_config.maker_fee,
        bt_config.slippage_pct, bt_config.high_vol_slippage_mult,
        bt_config.max_risk_per_trade,
    )
    return pnls[:nt].tolist(), rets[:nt].tolist(), cap


@njit(cache=True)
def _simulate_boltrend_numba(
    longs, shorts,
    opens, highs, lows, closes, regime,
    bb_sma_arr,
    sl_pct,
    initial_capital, taker_fee, maker_fee,
    slippage_pct, high_vol_slippage_mult,
    max_risk_per_trade,
):
    """Boucle de trades complète pour boltrend (JIT-compiled).

    check_exit INVERSÉ vs bollinger_mr :
    - LONG exit: close < SMA (breakout s'essouffle)
    - SHORT exit: close > SMA (breakout s'essouffle)
    """
    n = len(closes)
    trade_pnls = np.empty(n, dtype=np.float64)
    trade_returns = np.empty(n, dtype=np.float64)
    n_trades = 0
    capital = initial_capital

    in_position = False
    direction = 0
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0
    quantity = 0.0
    entry_fee = 0.0

    for i in range(n):
        if in_position:
            exit_reason = -1

            if direction == 1:
                tp_hit = highs[i] >= tp_price
                sl_hit = lows[i] <= sl_price
            else:
                tp_hit = lows[i] <= tp_price
                sl_hit = highs[i] >= sl_price

            if tp_hit and sl_hit:
                if closes[i] > opens[i]:
                    exit_reason = 0 if direction == 1 else 1
                elif closes[i] < opens[i]:
                    exit_reason = 1 if direction == 1 else 0
                else:
                    exit_reason = 1
            elif tp_hit:
                exit_reason = 0
            elif sl_hit:
                exit_reason = 1
            else:
                # check_exit boltrend : SMA crossing (INVERSÉ vs bollinger_mr)
                sma_val = bb_sma_arr[i]
                close = closes[i]
                if not np.isnan(sma_val) and not np.isnan(close):
                    if direction == 1:
                        if close < sma_val:  # LONG: breakout s'essouffle
                            exit_reason = 2
                    else:
                        if close > sma_val:  # SHORT: breakout s'essouffle
                            exit_reason = 2

            if exit_reason >= 0:
                if exit_reason == 0:
                    exit_price = tp_price
                elif exit_reason == 1:
                    exit_price = sl_price
                else:
                    exit_price = closes[i]

                net_pnl = _close_trade_numba(
                    direction, entry_price, exit_price, quantity,
                    entry_fee, exit_reason, int(regime[i]),
                    taker_fee, maker_fee, slippage_pct, high_vol_slippage_mult,
                )
                if capital > 0.0:
                    trade_returns[n_trades] = net_pnl / capital
                else:
                    trade_returns[n_trades] = 0.0
                capital += net_pnl
                trade_pnls[n_trades] = net_pnl
                n_trades += 1
                in_position = False

        if not in_position and (longs[i] or shorts[i]):
            direction = 1 if longs[i] else -1
            ep = closes[i]

            if ep <= 0.0 or capital <= 0.0:
                continue

            # TP très éloigné (SMA exit gère le vrai TP), SL % fixe
            sl_dist = ep * sl_pct / 100.0
            tp_dist = ep  # LONG: tp=2×entry, SHORT: tp≈0

            if direction == 1:
                tp_price = ep + tp_dist
                sl_price = ep - sl_dist
            else:
                tp_price = ep - tp_dist
                sl_price = ep + sl_dist

            if sl_price <= 0.0 or tp_price <= 0.0:
                continue

            sl_distance_pct = abs(ep - sl_price) / ep
            sl_real_cost = sl_distance_pct + taker_fee + slippage_pct
            if sl_real_cost <= 0.0:
                continue

            risk_amount = capital * max_risk_per_trade
            notional = risk_amount / sl_real_cost
            quantity = notional / ep

            if quantity <= 0.0:
                continue

            entry_fee = quantity * ep * taker_fee
            if entry_fee >= capital:
                continue

            entry_price = ep
            capital -= entry_fee
            in_position = True

    if in_position:
        exit_price = closes[n - 1]
        net_pnl = _close_trade_numba(
            direction, entry_price, exit_price, quantity,
            entry_fee, 3, int(regime[n - 1]),
            taker_fee, maker_fee, slippage_pct, high_vol_slippage_mult,
        )
        if capital > 0.0:
            trade_returns[n_trades] = net_pnl / capital
        else:
            trade_returns[n_trades] = 0.0
        capital += net_pnl
        trade_pnls[n_trades] = net_pnl
        n_trades += 1

    return trade_pnls, trade_returns, n_trades, capital


def _run_simulate_boltrend(
    longs: np.ndarray, shorts: np.ndarray,
    cache: IndicatorCache, params: dict[str, Any], bt_config: BacktestConfig,
) -> tuple[list[float], list[float], float]:
    """Wrapper Python pour _simulate_boltrend_numba."""
    bb_sma_arr = cache.bb_sma[params["bol_window"]]
    pnls, rets, nt, cap = _simulate_boltrend_numba(
        longs, shorts,
        cache.opens, cache.highs, cache.lows, cache.closes, cache.regime,
        bb_sma_arr,
        float(params["sl_percent"]),
        bt_config.initial_capital, bt_config.taker_fee, bt_config.maker_fee,
        bt_config.slippage_pct, bt_config.high_vol_slippage_mult,
        bt_config.max_risk_per_trade,
    )
    return pnls[:nt].tolist(), rets[:nt].tolist(), cap


@njit(cache=True)
def _simulate_donchian_numba(
    longs, shorts,
    opens, highs, lows, closes, regime,
    atr_arr,
    atr_tp_mult, atr_sl_mult,
    initial_capital, taker_fee, maker_fee,
    slippage_pct, high_vol_slippage_mult,
    max_risk_per_trade,
):
    """Boucle de trades complète pour donchian_breakout (JIT-compiled)."""
    n = len(closes)
    trade_pnls = np.empty(n, dtype=np.float64)
    trade_returns = np.empty(n, dtype=np.float64)
    n_trades = 0
    capital = initial_capital

    in_position = False
    direction = 0
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0
    quantity = 0.0
    entry_fee = 0.0

    for i in range(n):
        if in_position:
            exit_reason = -1

            if direction == 1:
                tp_hit = highs[i] >= tp_price
                sl_hit = lows[i] <= sl_price
            else:
                tp_hit = lows[i] <= tp_price
                sl_hit = highs[i] >= sl_price

            if tp_hit and sl_hit:
                if closes[i] > opens[i]:
                    exit_reason = 0 if direction == 1 else 1
                elif closes[i] < opens[i]:
                    exit_reason = 1 if direction == 1 else 0
                else:
                    exit_reason = 1
            elif tp_hit:
                exit_reason = 0
            elif sl_hit:
                exit_reason = 1
            # Pas de check_exit pour donchian

            if exit_reason >= 0:
                if exit_reason == 0:
                    exit_price = tp_price
                elif exit_reason == 1:
                    exit_price = sl_price
                else:
                    exit_price = closes[i]

                net_pnl = _close_trade_numba(
                    direction, entry_price, exit_price, quantity,
                    entry_fee, exit_reason, int(regime[i]),
                    taker_fee, maker_fee, slippage_pct, high_vol_slippage_mult,
                )
                if capital > 0.0:
                    trade_returns[n_trades] = net_pnl / capital
                else:
                    trade_returns[n_trades] = 0.0
                capital += net_pnl
                trade_pnls[n_trades] = net_pnl
                n_trades += 1
                in_position = False

        if not in_position and (longs[i] or shorts[i]):
            direction = 1 if longs[i] else -1
            ep = closes[i]
            atr_val = atr_arr[i]

            if ep <= 0.0 or capital <= 0.0:
                continue

            # ATR-based TP/SL
            if np.isnan(atr_val) or atr_val <= 0.0:
                continue
            tp_dist = atr_val * atr_tp_mult
            sl_dist = atr_val * atr_sl_mult

            if direction == 1:
                tp_price = ep + tp_dist
                sl_price = ep - sl_dist
            else:
                tp_price = ep - tp_dist
                sl_price = ep + sl_dist

            if sl_price <= 0.0 or tp_price <= 0.0:
                continue

            sl_distance_pct = abs(ep - sl_price) / ep
            sl_real_cost = sl_distance_pct + taker_fee + slippage_pct
            if sl_real_cost <= 0.0:
                continue

            risk_amount = capital * max_risk_per_trade
            notional = risk_amount / sl_real_cost
            quantity = notional / ep

            if quantity <= 0.0:
                continue

            entry_fee = quantity * ep * taker_fee
            if entry_fee >= capital:
                continue

            entry_price = ep
            capital -= entry_fee
            in_position = True

    if in_position:
        exit_price = closes[n - 1]
        net_pnl = _close_trade_numba(
            direction, entry_price, exit_price, quantity,
            entry_fee, 3, int(regime[n - 1]),
            taker_fee, maker_fee, slippage_pct, high_vol_slippage_mult,
        )
        if capital > 0.0:
            trade_returns[n_trades] = net_pnl / capital
        else:
            trade_returns[n_trades] = 0.0
        capital += net_pnl
        trade_pnls[n_trades] = net_pnl
        n_trades += 1

    return trade_pnls, trade_returns, n_trades, capital


def _run_simulate_donchian(
    longs: np.ndarray, shorts: np.ndarray,
    cache: IndicatorCache, params: dict[str, Any], bt_config: BacktestConfig,
) -> tuple[list[float], list[float], float]:
    """Wrapper Python pour _simulate_donchian_numba."""
    atr_arr = cache.atr_by_period[params["atr_period"]]
    pnls, rets, nt, cap = _simulate_donchian_numba(
        longs, shorts,
        cache.opens, cache.highs, cache.lows, cache.closes, cache.regime,
        atr_arr,
        float(params["atr_tp_multiple"]), float(params["atr_sl_multiple"]),
        bt_config.initial_capital, bt_config.taker_fee, bt_config.maker_fee,
        bt_config.slippage_pct, bt_config.high_vol_slippage_mult,
        bt_config.max_risk_per_trade,
    )
    return pnls[:nt].tolist(), rets[:nt].tolist(), cap


@njit(cache=True)
def _simulate_supertrend_numba(
    longs, shorts,
    opens, highs, lows, closes, regime,
    tp_pct, sl_pct,
    initial_capital, taker_fee, maker_fee,
    slippage_pct, high_vol_slippage_mult,
    max_risk_per_trade,
):
    """Boucle de trades complète pour supertrend (JIT-compiled)."""
    n = len(closes)
    trade_pnls = np.empty(n, dtype=np.float64)
    trade_returns = np.empty(n, dtype=np.float64)
    n_trades = 0
    capital = initial_capital

    in_position = False
    direction = 0
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0
    quantity = 0.0
    entry_fee = 0.0

    for i in range(n):
        if in_position:
            exit_reason = -1

            if direction == 1:
                tp_hit = highs[i] >= tp_price
                sl_hit = lows[i] <= sl_price
            else:
                tp_hit = lows[i] <= tp_price
                sl_hit = highs[i] >= sl_price

            if tp_hit and sl_hit:
                if closes[i] > opens[i]:
                    exit_reason = 0 if direction == 1 else 1
                elif closes[i] < opens[i]:
                    exit_reason = 1 if direction == 1 else 0
                else:
                    exit_reason = 1
            elif tp_hit:
                exit_reason = 0
            elif sl_hit:
                exit_reason = 1
            # Pas de check_exit pour supertrend

            if exit_reason >= 0:
                if exit_reason == 0:
                    exit_price = tp_price
                elif exit_reason == 1:
                    exit_price = sl_price
                else:
                    exit_price = closes[i]

                net_pnl = _close_trade_numba(
                    direction, entry_price, exit_price, quantity,
                    entry_fee, exit_reason, int(regime[i]),
                    taker_fee, maker_fee, slippage_pct, high_vol_slippage_mult,
                )
                if capital > 0.0:
                    trade_returns[n_trades] = net_pnl / capital
                else:
                    trade_returns[n_trades] = 0.0
                capital += net_pnl
                trade_pnls[n_trades] = net_pnl
                n_trades += 1
                in_position = False

        if not in_position and (longs[i] or shorts[i]):
            direction = 1 if longs[i] else -1
            ep = closes[i]

            if ep <= 0.0 or capital <= 0.0:
                continue

            tp_dist = ep * tp_pct / 100.0
            sl_dist = ep * sl_pct / 100.0

            if direction == 1:
                tp_price = ep + tp_dist
                sl_price = ep - sl_dist
            else:
                tp_price = ep - tp_dist
                sl_price = ep + sl_dist

            if sl_price <= 0.0 or tp_price <= 0.0:
                continue

            sl_distance_pct = abs(ep - sl_price) / ep
            sl_real_cost = sl_distance_pct + taker_fee + slippage_pct
            if sl_real_cost <= 0.0:
                continue

            risk_amount = capital * max_risk_per_trade
            notional = risk_amount / sl_real_cost
            quantity = notional / ep

            if quantity <= 0.0:
                continue

            entry_fee = quantity * ep * taker_fee
            if entry_fee >= capital:
                continue

            entry_price = ep
            capital -= entry_fee
            in_position = True

    if in_position:
        exit_price = closes[n - 1]
        net_pnl = _close_trade_numba(
            direction, entry_price, exit_price, quantity,
            entry_fee, 3, int(regime[n - 1]),
            taker_fee, maker_fee, slippage_pct, high_vol_slippage_mult,
        )
        if capital > 0.0:
            trade_returns[n_trades] = net_pnl / capital
        else:
            trade_returns[n_trades] = 0.0
        capital += net_pnl
        trade_pnls[n_trades] = net_pnl
        n_trades += 1

    return trade_pnls, trade_returns, n_trades, capital


def _run_simulate_supertrend(
    longs: np.ndarray, shorts: np.ndarray,
    cache: IndicatorCache, params: dict[str, Any], bt_config: BacktestConfig,
) -> tuple[list[float], list[float], float]:
    """Wrapper Python pour _simulate_supertrend_numba."""
    pnls, rets, nt, cap = _simulate_supertrend_numba(
        longs, shorts,
        cache.opens, cache.highs, cache.lows, cache.closes, cache.regime,
        float(params["tp_percent"]), float(params["sl_percent"]),
        bt_config.initial_capital, bt_config.taker_fee, bt_config.maker_fee,
        bt_config.slippage_pct, bt_config.high_vol_slippage_mult,
        bt_config.max_risk_per_trade,
    )
    return pnls[:nt].tolist(), rets[:nt].tolist(), cap


# ─── Trade simulation (boucle Python fallback) ──────────────────────────────


def _simulate_trades(
    longs: np.ndarray,
    shorts: np.ndarray,
    cache: IndicatorCache,
    strategy_name: str,
    params: dict[str, Any],
    bt_config: BacktestConfig,
) -> tuple[list[float], list[float], float]:
    """Simule les trades séquentiellement.

    Dispatch vers la version JIT si numba est disponible, sinon
    boucle Python fallback.

    Retourne (trade_pnls, trade_returns, final_capital).
    """
    # Dispatch numba (par stratégie)
    if NUMBA_AVAILABLE:
        if strategy_name == "vwap_rsi":
            return _run_simulate_vwap_rsi(longs, shorts, cache, params, bt_config)
        if strategy_name == "momentum":
            return _run_simulate_momentum(longs, shorts, cache, params, bt_config)
        if strategy_name == "bollinger_mr":
            return _run_simulate_bollinger(longs, shorts, cache, params, bt_config)
        if strategy_name == "donchian_breakout":
            return _run_simulate_donchian(longs, shorts, cache, params, bt_config)
        if strategy_name == "supertrend":
            return _run_simulate_supertrend(longs, shorts, cache, params, bt_config)
        if strategy_name == "boltrend":
            return _run_simulate_boltrend(longs, shorts, cache, params, bt_config)

    n = cache.n_candles
    capital = bt_config.initial_capital
    trade_pnls: list[float] = []
    trade_returns: list[float] = []

    # Config fees/slippage
    taker_fee = bt_config.taker_fee
    maker_fee = bt_config.maker_fee
    slippage_pct = bt_config.slippage_pct
    high_vol_slippage_mult = bt_config.high_vol_slippage_mult
    max_risk_per_trade = bt_config.max_risk_per_trade
    leverage = bt_config.leverage

    # État position
    in_position = False
    direction = 0  # 1=LONG, -1=SHORT
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0
    quantity = 0.0
    entry_fee = 0.0

    for i in range(n):
        # 1. Si position ouverte : vérifier TP/SL puis check_exit
        if in_position:
            exit_reason = None
            exit_price = 0.0

            # Check TP/SL
            tp_hit, sl_hit = _check_tp_sl(
                cache.highs[i], cache.lows[i], direction, tp_price, sl_price,
            )

            if tp_hit and sl_hit:
                exit_reason = _ohlc_heuristic(
                    cache.opens[i], cache.closes[i], direction,
                )
            elif tp_hit:
                exit_reason = "tp"
            elif sl_hit:
                exit_reason = "sl"
            else:
                # check_exit stratégie-spécifique
                if _check_exit(strategy_name, cache, i, direction, entry_price, params):
                    exit_reason = "signal_exit"

            if exit_reason is not None:
                if exit_reason == "tp":
                    exit_price = tp_price
                elif exit_reason == "sl":
                    exit_price = sl_price
                else:
                    exit_price = cache.closes[i]

                net_pnl = _close_trade(
                    direction, entry_price, exit_price, quantity,
                    entry_fee, exit_reason, int(cache.regime[i]),
                    taker_fee, maker_fee, slippage_pct, high_vol_slippage_mult,
                )
                if capital > 0:
                    trade_returns.append(net_pnl / capital)
                capital += net_pnl
                trade_pnls.append(net_pnl)
                in_position = False

        # 2. Si pas de position : entrée
        if not in_position and (longs[i] or shorts[i]):
            direction = 1 if longs[i] else -1

            # ATR variable selon la stratégie
            if strategy_name in ("donchian_breakout", "supertrend"):
                atr_p = params["atr_period"]
                atr_val = float(cache.atr_by_period[atr_p][i])
            else:
                atr_val = cache.atr_arr[i]

            result = _open_trade(
                direction, cache.closes[i], atr_val,
                strategy_name, params, capital,
                max_risk_per_trade, taker_fee, slippage_pct,
            )
            if result is not None:
                tp_price, sl_price, quantity, entry_fee = result
                entry_price = cache.closes[i]
                capital -= entry_fee
                in_position = True

    # Force close fin de données
    if in_position:
        exit_price = cache.closes[n - 1]
        regime_at_exit = int(cache.regime[n - 1])
        net_pnl = _close_trade(
            direction, entry_price, exit_price, quantity,
            entry_fee, "end_of_data", regime_at_exit,
            taker_fee, maker_fee, slippage_pct, high_vol_slippage_mult,
        )
        if capital > 0:
            trade_returns.append(net_pnl / capital)
        capital += net_pnl
        trade_pnls.append(net_pnl)

    return trade_pnls, trade_returns, capital


# ─── Helpers de la boucle ────────────────────────────────────────────────────


def _check_tp_sl(
    high: float, low: float, direction: int, tp: float, sl: float,
) -> tuple[bool, bool]:
    """Vérifie si TP et/ou SL sont touchés sur la bougie."""
    if direction == 1:  # LONG
        return high >= tp, low <= sl
    else:  # SHORT
        return low <= tp, high >= sl


def _ohlc_heuristic(open_price: float, close_price: float, direction: int) -> str:
    """Heuristique OHLC quand TP et SL sont touchés sur la même bougie.

    Reproduit exactement PositionManager._ohlc_heuristic().
    """
    if close_price > open_price:  # Bougie verte
        return "tp" if direction == 1 else "sl"
    elif close_price < open_price:  # Bougie rouge
        return "sl" if direction == 1 else "tp"
    else:  # Doji
        return "sl"


def _check_exit(
    strategy_name: str,
    cache: IndicatorCache,
    i: int,
    direction: int,
    entry_price: float,
    params: dict[str, Any],
) -> bool:
    """Vérifie la sortie anticipée stratégie-spécifique.

    VWAP+RSI : RSI normalisé (> 50 LONG, < 50 SHORT) ET en profit.
    Momentum : ADX 5m < 20.
    """
    if strategy_name == "vwap_rsi":
        rsi_val = float(cache.rsi[params["rsi_period"]][i])
        close = float(cache.closes[i])
        if math.isnan(rsi_val) or math.isnan(close):
            return False
        if direction == 1:  # LONG
            return close > entry_price and rsi_val > 50
        else:  # SHORT
            return close < entry_price and rsi_val < 50

    elif strategy_name == "momentum":
        adx_val = float(cache.adx_arr[i])
        if math.isnan(adx_val):
            return False
        return adx_val < 20

    elif strategy_name == "bollinger_mr":
        bb_period = params["bb_period"]
        bb_sma_val = float(cache.bb_sma[bb_period][i])
        close = float(cache.closes[i])
        if math.isnan(bb_sma_val) or math.isnan(close):
            return False
        if direction == 1:  # LONG : close a croisé au-dessus de la SMA
            return close >= bb_sma_val
        else:  # SHORT : close a croisé en-dessous de la SMA
            return close <= bb_sma_val

    elif strategy_name == "boltrend":
        # INVERSÉ vs bollinger_mr : breakout s'essouffle → retour à la SMA
        bb_sma_val = float(cache.bb_sma[params["bol_window"]][i])
        close = float(cache.closes[i])
        if math.isnan(bb_sma_val) or math.isnan(close):
            return False
        if direction == 1:  # LONG exit: close < SMA
            return close < bb_sma_val
        else:  # SHORT exit: close > SMA
            return close > bb_sma_val

    return False


def _open_trade(
    direction: int,
    entry_price: float,
    atr_val: float,
    strategy_name: str,
    params: dict[str, Any],
    capital: float,
    max_risk_per_trade: float,
    taker_fee: float,
    slippage_pct: float,
) -> tuple[float, float, float, float] | None:
    """Calcule TP/SL et position sizing.

    Reproduit exactement PositionManager.open_position() + TP/SL de la stratégie.
    Retourne (tp_price, sl_price, quantity, entry_fee) ou None si invalide.
    """
    if entry_price <= 0 or capital <= 0:
        return None

    # TP/SL selon la stratégie
    if strategy_name == "vwap_rsi":
        tp_dist = entry_price * params["tp_percent"] / 100
        sl_dist = entry_price * params["sl_percent"] / 100
    elif strategy_name == "momentum":
        # ATR-based avec caps
        tp_pct = params["tp_percent"]
        sl_pct = params["sl_percent"]
        atr_mult_tp = params["atr_multiplier_tp"]
        atr_mult_sl = params["atr_multiplier_sl"]

        if not math.isnan(atr_val) and atr_val > 0:
            atr_tp = atr_val * atr_mult_tp
            atr_sl = atr_val * atr_mult_sl
        else:
            atr_tp = entry_price * tp_pct / 100
            atr_sl = entry_price * sl_pct / 100

        max_tp = entry_price * tp_pct / 100
        max_sl = entry_price * sl_pct / 100
        tp_dist = min(atr_tp, max_tp)
        sl_dist = min(atr_sl, max_sl)
    elif strategy_name == "bollinger_mr":
        # TP très éloigné (désactivé), SL % fixe. check_exit gère le vrai TP.
        sl_dist = entry_price * params["sl_percent"] / 100
        tp_dist = entry_price  # tp = entry × 2 (LONG) ou entry × 0.5 (SHORT)
    elif strategy_name == "donchian_breakout":
        if math.isnan(atr_val) or atr_val <= 0:
            return None
        tp_dist = atr_val * params["atr_tp_multiple"]
        sl_dist = atr_val * params["atr_sl_multiple"]
    elif strategy_name == "supertrend":
        tp_dist = entry_price * params["tp_percent"] / 100
        sl_dist = entry_price * params["sl_percent"] / 100
    elif strategy_name == "boltrend":
        # TP très éloigné (SMA exit gère le vrai TP), SL % fixe
        sl_dist = entry_price * params["sl_percent"] / 100
        tp_dist = entry_price  # tp = entry × 2 (LONG) ou entry × 0.5 (SHORT)
    else:
        return None

    if direction == 1:  # LONG
        tp_price = entry_price + tp_dist
        sl_price = entry_price - sl_dist
    else:  # SHORT
        tp_price = entry_price - tp_dist
        sl_price = entry_price + sl_dist

    if sl_price <= 0 or tp_price <= 0:
        return None

    # Position sizing (même formule que PositionManager)
    sl_distance_pct = abs(entry_price - sl_price) / entry_price
    sl_real_cost = sl_distance_pct + taker_fee + slippage_pct

    if sl_real_cost <= 0:
        return None

    risk_amount = capital * max_risk_per_trade
    notional = risk_amount / sl_real_cost
    quantity = notional / entry_price

    if quantity <= 0:
        return None

    entry_fee = quantity * entry_price * taker_fee
    if entry_fee >= capital:
        return None

    return tp_price, sl_price, quantity, entry_fee


def _close_trade(
    direction: int,
    entry_price: float,
    exit_price: float,
    quantity: float,
    entry_fee: float,
    exit_reason: str,
    regime_int: int,
    taker_fee: float,
    maker_fee: float,
    slippage_pct: float,
    high_vol_slippage_mult: float,
) -> float:
    """Calcule le net_pnl d'un trade fermé.

    Reproduit exactement PositionManager.close_position().
    """
    # Gross PnL sur prix brut (pas d'actual_exit ajusté)
    if direction == 1:  # LONG
        gross_pnl = (exit_price - entry_price) * quantity
    else:  # SHORT
        gross_pnl = (entry_price - exit_price) * quantity

    # Slippage : flat cost 1 seule fois (exit seulement, sauf TP)
    slippage_cost = 0.0
    if exit_reason in ("sl", "signal_exit", "end_of_data"):
        slippage_rate = slippage_pct
        if regime_int == _HIGH_VOL:
            slippage_rate *= high_vol_slippage_mult
        slippage_cost = quantity * exit_price * slippage_rate

    # Fee de sortie
    if exit_reason == "tp":
        exit_fee = quantity * exit_price * maker_fee
    else:
        exit_fee = quantity * exit_price * taker_fee

    fee_cost = entry_fee + exit_fee
    return gross_pnl - fee_cost - slippage_cost


# ─── Métriques rapides ──────────────────────────────────────────────────────


def _compute_fast_metrics(
    params: dict[str, Any],
    trade_pnls: list[float],
    trade_returns: list[float],
    final_capital: float,
    initial_capital: float,
    total_days: float,
) -> _ISResult:
    """Calcule les métriques (sharpe, return, PF) sans objets lourds.

    Reproduit exactement calculate_metrics() pour les 4 champs de _ISResult.
    """
    n_trades = len(trade_pnls)

    if n_trades == 0:
        return (params, 0.0, 0.0, 0.0, 0)

    # Net return %
    net_return_pct = sum(trade_pnls) / initial_capital * 100

    # Profit factor
    net_wins = sum(p for p in trade_pnls if p > 0)
    net_losses = abs(sum(p for p in trade_pnls if p <= 0))
    profit_factor = net_wins / net_losses if net_losses > 0 else float("inf")

    # Sharpe ratio annualisé (même formule que metrics._sharpe)
    sharpe = 0.0
    if n_trades >= 3 and len(trade_returns) >= 2:
        arr = np.array(trade_returns)
        std = float(np.std(arr))
        if std > 1e-10:
            trades_per_year = n_trades / max(total_days, 1) * 365
            sharpe = float(np.mean(arr) / std * np.sqrt(trades_per_year))
            sharpe = min(100.0, sharpe)

    return (params, sharpe, net_return_pct, profit_factor, n_trades)
