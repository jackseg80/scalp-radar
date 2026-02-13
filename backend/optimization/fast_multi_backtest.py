"""Fast backtest engine multi-position pour le WFO — grid/DCA.

Pré-requis : un IndicatorCache avec SMA pré-calculées (bb_sma).
Chaque combinaison calcule les enveloppes à la volée (multiplication triviale).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from backend.backtesting.engine import BacktestConfig
from backend.optimization.indicator_cache import IndicatorCache

# Type retour léger (même que walk_forward._ISResult)
_ISResult = tuple[dict[str, Any], float, float, float, int]


def run_multi_backtest_from_cache(
    strategy_name: str,
    params: dict[str, Any],
    cache: IndicatorCache,
    bt_config: BacktestConfig,
) -> _ISResult:
    """Backtest rapide multi-position sur cache numpy.

    Retourne (params, sharpe, net_return_pct, profit_factor, n_trades).
    """
    if strategy_name == "envelope_dca":
        trade_pnls, trade_returns, final_capital = _simulate_envelope_dca(
            cache, params, bt_config,
        )
    else:
        raise ValueError(f"Stratégie grid inconnue pour fast engine: {strategy_name}")

    return _compute_fast_metrics(
        params, trade_pnls, trade_returns, final_capital,
        bt_config.initial_capital, cache.total_days,
    )


def _simulate_envelope_dca(
    cache: IndicatorCache,
    params: dict[str, Any],
    bt_config: BacktestConfig,
) -> tuple[list[float], list[float], float]:
    """Simulation multi-position Envelope DCA.

    SMA depuis le cache, enveloppes à la volée.
    Allocation fixe par niveau : notional = capital/levels × leverage.
    """
    capital = bt_config.initial_capital
    leverage = bt_config.leverage
    taker_fee = bt_config.taker_fee
    maker_fee = bt_config.maker_fee
    slippage_pct = bt_config.slippage_pct

    num_levels = params["num_levels"]
    sl_pct = params["sl_percent"] / 100

    sma_arr = cache.bb_sma[params["ma_period"]]
    n = cache.n_candles

    # Pré-calculer les offsets d'enveloppe (constantes)
    lower_offsets = [
        params["envelope_start"] + lvl * params["envelope_step"]
        for lvl in range(num_levels)
    ]

    trade_pnls: list[float] = []
    trade_returns: list[float] = []

    # Positions : list of (level_idx, entry_price, quantity, entry_fee)
    positions: list[tuple[int, float, float, float]] = []

    for i in range(n):
        if math.isnan(sma_arr[i]):
            continue

        # 1. Check TP/SL global si positions ouvertes
        if positions:
            total_qty = sum(p[2] for p in positions)
            avg_entry = sum(p[1] * p[2] for p in positions) / total_qty

            # OHLC heuristic : bougie verte → low d'abord (SL check), puis high (TP)
            # Bougie rouge → high d'abord (TP), puis low (SL)
            is_green = cache.closes[i] > cache.opens[i]

            sl_price = avg_entry * (1 - sl_pct)
            tp_price = sma_arr[i]  # Dynamique

            sl_hit = cache.lows[i] <= sl_price
            tp_hit = cache.highs[i] >= tp_price

            exit_reason = None
            exit_price = 0.0

            if tp_hit and sl_hit:
                # Heuristique OHLC
                if is_green:
                    # Bougie verte : LONG favorable → TP
                    exit_reason = "tp_global"
                    exit_price = tp_price
                elif cache.closes[i] < cache.opens[i]:
                    # Bougie rouge : LONG défavorable → SL
                    exit_reason = "sl_global"
                    exit_price = sl_price
                else:
                    # Doji → SL (conservateur)
                    exit_reason = "sl_global"
                    exit_price = sl_price
            elif sl_hit:
                exit_reason = "sl_global"
                exit_price = sl_price
            elif tp_hit:
                exit_reason = "tp_global"
                exit_price = tp_price

            if exit_reason is not None:
                pnl = _calc_grid_pnl(
                    positions, exit_price,
                    maker_fee if exit_reason == "tp_global" else taker_fee,
                    slippage_pct if exit_reason != "tp_global" else 0.0,
                    1,  # LONG
                )
                trade_pnls.append(pnl)
                if capital > 0:
                    trade_returns.append(pnl / capital)
                capital += pnl
                positions = []
                continue

        # 2. Ouvrir de nouvelles positions si niveaux touchés
        if len(positions) < num_levels:
            filled = {p[0] for p in positions}
            for lvl in range(num_levels):
                if lvl in filled:
                    continue
                if len(positions) >= num_levels:
                    break

                entry_price = sma_arr[i] * (1 - lower_offsets[lvl])
                if math.isnan(entry_price) or entry_price <= 0:
                    continue

                if cache.lows[i] <= entry_price:
                    # Allocation fixe par niveau
                    notional = capital * (1.0 / num_levels) * leverage
                    qty = notional / entry_price
                    if qty <= 0:
                        continue
                    entry_fee = qty * entry_price * taker_fee
                    positions.append((lvl, entry_price, qty, entry_fee))

    # Force close fin de données
    if positions:
        exit_price = float(cache.closes[n - 1])
        pnl = _calc_grid_pnl(positions, exit_price, taker_fee, slippage_pct, 1)
        trade_pnls.append(pnl)
        if capital > 0:
            trade_returns.append(pnl / capital)
        capital += pnl

    return trade_pnls, trade_returns, capital


def _calc_grid_pnl(
    positions: list[tuple[int, float, float, float]],
    exit_price: float,
    exit_fee_rate: float,
    slippage_rate: float,
    direction: int,
) -> float:
    """Calcule le net PnL agrégé pour fermer toutes les positions."""
    total_pnl = 0.0
    for _lvl, entry_price, qty, entry_fee in positions:
        actual_exit = exit_price
        slippage_cost = 0.0

        if slippage_rate > 0:
            slippage_cost = qty * exit_price * slippage_rate
            if direction == 1:  # LONG
                actual_exit = exit_price * (1 - slippage_rate)
            else:
                actual_exit = exit_price * (1 + slippage_rate)

        if direction == 1:
            gross = (actual_exit - entry_price) * qty
        else:
            gross = (entry_price - actual_exit) * qty

        exit_fee = qty * exit_price * exit_fee_rate
        net = gross - entry_fee - exit_fee - slippage_cost
        total_pnl += net

    return total_pnl


def _compute_fast_metrics(
    params: dict[str, Any],
    trade_pnls: list[float],
    trade_returns: list[float],
    final_capital: float,
    initial_capital: float,
    total_days: float,
) -> _ISResult:
    """Calcule les métriques (sharpe, return, PF) sans objets lourds."""
    n_trades = len(trade_pnls)

    if n_trades == 0:
        return (params, 0.0, 0.0, 0.0, 0)

    net_return_pct = sum(trade_pnls) / initial_capital * 100

    net_wins = sum(p for p in trade_pnls if p > 0)
    net_losses = abs(sum(p for p in trade_pnls if p <= 0))
    profit_factor = net_wins / net_losses if net_losses > 0 else float("inf")

    sharpe = 0.0
    if n_trades >= 3 and len(trade_returns) >= 2:
        arr = np.array(trade_returns)
        std = float(np.std(arr))
        if std > 1e-10:
            trades_per_year = n_trades / max(total_days, 1) * 365
            sharpe = float(np.mean(arr) / std * np.sqrt(trades_per_year))
            sharpe = min(100.0, sharpe)

    return (params, sharpe, net_return_pct, profit_factor, n_trades)
