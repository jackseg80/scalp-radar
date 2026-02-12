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


# ─── Trade simulation (boucle minimale) ─────────────────────────────────────


def _simulate_trades(
    longs: np.ndarray,
    shorts: np.ndarray,
    cache: IndicatorCache,
    strategy_name: str,
    params: dict[str, Any],
    bt_config: BacktestConfig,
) -> tuple[list[float], list[float], float]:
    """Simule les trades séquentiellement.

    Seule boucle Python restante. Pas de calcul d'indicateurs, juste des
    comparaisons scalaires sur les arrays pré-indexés.

    Retourne (trade_pnls, trade_returns, final_capital).
    """
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

            result = _open_trade(
                direction, cache.closes[i], cache.atr_arr[i],
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
    slippage_cost = 0.0
    actual_exit_price = exit_price

    if exit_reason in ("sl", "signal_exit", "end_of_data"):
        slippage_rate = slippage_pct
        if regime_int == _HIGH_VOL:
            slippage_rate *= high_vol_slippage_mult

        slippage_cost = quantity * exit_price * slippage_rate

        if direction == 1:  # LONG
            actual_exit_price = exit_price * (1 - slippage_rate)
        else:  # SHORT
            actual_exit_price = exit_price * (1 + slippage_rate)

    # Gross PnL
    if direction == 1:  # LONG
        gross_pnl = (actual_exit_price - entry_price) * quantity
    else:  # SHORT
        gross_pnl = (entry_price - actual_exit_price) * quantity

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
