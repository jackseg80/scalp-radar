"""Fast backtest engine multi-position pour le WFO — grid/DCA.

Pré-requis : un IndicatorCache avec SMA pré-calculées (bb_sma).
Chaque combinaison calcule les enveloppes à la volée (multiplication triviale).

Architecture (Sprint 20c) :
- _build_entry_prices() : factory retournant un 2D array (n_candles, num_levels)
- _simulate_grid_common() : boucle chaude unifiée (TP/SL, allocation, force close)
- _simulate_envelope_dca / _simulate_grid_atr : wrappers backward-compat
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from backend.backtesting.engine import BacktestConfig
from backend.optimization.indicator_cache import IndicatorCache

# Type retour léger (même que walk_forward._ISResult)
_ISResult = tuple[dict[str, Any], float, float, float, int]


# ─── Factory entry prices ──────────────────────────────────────────────────


def _build_entry_prices(
    strategy_name: str,
    cache: IndicatorCache,
    params: dict[str, Any],
    num_levels: int,
    direction: int,
) -> np.ndarray:
    """Factory retournant un array 2D (n_candles, num_levels) de prix d'entrée.

    NaN propagé pour les candles invalides (SMA NaN, ATR NaN ou <= 0).
    Chaque nouvelle stratégie grid = ajouter un elif de 3-5 lignes ici.
    """
    n = cache.n_candles
    sma_arr = cache.bb_sma[params["ma_period"]]
    entry_prices = np.full((n, num_levels), np.nan)

    if strategy_name in ("envelope_dca", "envelope_dca_short"):
        lower_offsets = [
            params["envelope_start"] + lvl * params["envelope_step"]
            for lvl in range(num_levels)
        ]
        if direction == -1:
            # SHORT : enveloppes hautes asymétriques (comme EnvelopeDCAStrategy.compute_grid)
            envelope_offsets = [round(1 / (1 - e) - 1, 3) for e in lower_offsets]
        else:
            envelope_offsets = lower_offsets

        for lvl in range(num_levels):
            if direction == 1:
                entry_prices[:, lvl] = sma_arr * (1 - envelope_offsets[lvl])
            else:
                entry_prices[:, lvl] = sma_arr * (1 + envelope_offsets[lvl])

    elif strategy_name == "grid_atr":
        atr_arr = cache.atr_by_period[params["atr_period"]]
        multipliers = [
            params["atr_multiplier_start"] + lvl * params["atr_multiplier_step"]
            for lvl in range(num_levels)
        ]
        for lvl in range(num_levels):
            if direction == 1:
                entry_prices[:, lvl] = sma_arr - atr_arr * multipliers[lvl]
            else:
                entry_prices[:, lvl] = sma_arr + atr_arr * multipliers[lvl]
        # ATR NaN ou <= 0 : forcer NaN (SMA NaN déjà propagé naturellement)
        invalid = np.isnan(atr_arr) | (atr_arr <= 0)
        entry_prices[invalid, :] = np.nan

    elif strategy_name == "grid_multi_tf":
        atr_arr = cache.atr_by_period[params["atr_period"]]
        st_key = (params["st_atr_period"], params["st_atr_multiplier"])
        st_dir = cache.supertrend_dir_4h[st_key]
        multipliers = [
            params["atr_multiplier_start"] + lvl * params["atr_multiplier_step"]
            for lvl in range(num_levels)
        ]
        long_mask = st_dir == 1
        short_mask = st_dir == -1
        for lvl in range(num_levels):
            entry_prices[long_mask, lvl] = sma_arr[long_mask] - atr_arr[long_mask] * multipliers[lvl]
            entry_prices[short_mask, lvl] = sma_arr[short_mask] + atr_arr[short_mask] * multipliers[lvl]
        # NaN propagation : ATR invalide ou pas de direction Supertrend
        invalid = np.isnan(atr_arr) | (atr_arr <= 0) | np.isnan(st_dir)
        entry_prices[invalid, :] = np.nan

    else:
        raise ValueError(f"Stratégie grid inconnue pour _build_entry_prices: {strategy_name}")

    return entry_prices


# ─── Boucle chaude unifiée ─────────────────────────────────────────────────


def _simulate_grid_common(
    entry_prices: np.ndarray,
    sma_arr: np.ndarray,
    cache: IndicatorCache,
    bt_config: BacktestConfig,
    num_levels: int,
    sl_pct: float,
    direction: int,
    directions: np.ndarray | None = None,
) -> tuple[list[float], list[float], float]:
    """Boucle chaude unifiée pour toutes les stratégies grid/DCA.

    Args:
        entry_prices: (n_candles, num_levels) pré-calculé par _build_entry_prices.
        sma_arr: SMA pour TP dynamique (retour vers la SMA).
        sl_pct: déjà divisé par 100.
        direction: 1 = LONG, -1 = SHORT (scalar fixe, ou initial pour directions dynamiques).
        directions: si fourni, array 1D de directions par candle (1/-1/NaN).
            Override le scalar `direction` à chaque candle. Force-close au flip.
    """
    capital = bt_config.initial_capital
    leverage = bt_config.leverage
    taker_fee = bt_config.taker_fee
    maker_fee = bt_config.maker_fee
    slippage_pct = bt_config.slippage_pct
    n = cache.n_candles

    trade_pnls: list[float] = []
    trade_returns: list[float] = []

    # Positions : list of (level_idx, entry_price, quantity, entry_fee)
    positions: list[tuple[int, float, float, float]] = []

    # Direction tracking pour le mode dynamique
    last_dir = 0  # 0 = pas encore initialisé

    for i in range(n):
        # --- Directions dynamiques (grid_multi_tf) ---
        if directions is not None:
            cur_dir = directions[i]
            if math.isnan(cur_dir):
                continue  # Pas de direction Supertrend → skip
            cur_dir_int = int(cur_dir)
            # Force-close si direction a flippé
            if positions and last_dir != 0 and cur_dir_int != last_dir:
                pnl = _calc_grid_pnl(
                    positions, cache.closes[i], taker_fee, slippage_pct, last_dir,
                )
                trade_pnls.append(pnl)
                if capital > 0:
                    trade_returns.append(pnl / capital)
                capital += pnl
                positions = []
            last_dir = cur_dir_int
            direction = cur_dir_int  # Override le scalar pour TP/SL et entry

        # Skip candles invalides (NaN propagé depuis _build_entry_prices)
        if math.isnan(entry_prices[i, 0]):
            continue

        # 1. Check TP/SL global si positions ouvertes
        if positions:
            total_qty = sum(p[2] for p in positions)
            avg_entry = sum(p[1] * p[2] for p in positions) / total_qty

            is_green = cache.closes[i] > cache.opens[i]

            tp_price = sma_arr[i]  # Dynamique (retour vers la SMA)

            if direction == 1:
                # LONG : SL en dessous, TP au-dessus
                sl_price = avg_entry * (1 - sl_pct)
                sl_hit = cache.lows[i] <= sl_price
                tp_hit = cache.highs[i] >= tp_price
            else:
                # SHORT : SL au-dessus, TP en dessous
                sl_price = avg_entry * (1 + sl_pct)
                sl_hit = cache.highs[i] >= sl_price
                tp_hit = cache.lows[i] <= tp_price

            exit_reason = None
            exit_price = 0.0

            if tp_hit and sl_hit:
                # Heuristique OHLC
                if direction == 1:
                    if is_green:
                        exit_reason = "tp_global"
                        exit_price = tp_price
                    else:
                        exit_reason = "sl_global"
                        exit_price = sl_price
                else:
                    if cache.closes[i] < cache.opens[i]:
                        exit_reason = "tp_global"
                        exit_price = tp_price
                    else:
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
                    direction,
                )
                trade_pnls.append(pnl)
                if capital > 0:
                    trade_returns.append(pnl / capital)
                capital += pnl
                positions = []
                continue

        # 2. Guard capital épuisé
        if capital <= 0:
            continue

        # 3. Ouvrir de nouvelles positions si niveaux touchés
        if len(positions) < num_levels:
            filled = {p[0] for p in positions}
            for lvl in range(num_levels):
                if lvl in filled:
                    continue
                if len(positions) >= num_levels:
                    break

                ep = float(entry_prices[i, lvl])
                if math.isnan(ep) or ep <= 0:
                    continue

                if direction == 1:
                    triggered = cache.lows[i] <= ep
                else:
                    triggered = cache.highs[i] >= ep

                if triggered:
                    # Allocation fixe par niveau
                    notional = capital * (1.0 / num_levels) * leverage
                    qty = notional / ep
                    if qty <= 0:
                        continue
                    entry_fee = qty * ep * taker_fee
                    positions.append((lvl, ep, qty, entry_fee))

    # Force close fin de données
    if positions:
        exit_price = float(cache.closes[n - 1])
        pnl = _calc_grid_pnl(positions, exit_price, taker_fee, slippage_pct, direction)
        trade_pnls.append(pnl)
        if capital > 0:
            trade_returns.append(pnl / capital)
        capital += pnl

    return trade_pnls, trade_returns, capital


# ─── Grid Funding ─────────────────────────────────────────────────────────


def _build_entry_signals(
    cache: IndicatorCache,
    params: dict[str, Any],
    num_levels: int,
) -> np.ndarray:
    """Retourne entry_signals[i, lvl] = True si le funding est assez négatif.

    Shape: (n, num_levels) dtype bool. NaN funding = pas de signal.
    """
    n = cache.n_candles
    signals = np.zeros((n, num_levels), dtype=bool)

    funding = cache.funding_rates_1h
    if funding is None:
        return signals

    threshold_start = params["funding_threshold_start"]
    threshold_step = params["funding_threshold_step"]

    for lvl in range(num_levels):
        threshold = -(threshold_start + lvl * threshold_step)
        signals[:, lvl] = funding <= threshold

    # NaN funding = pas de signal
    nan_mask = np.isnan(funding)
    signals[nan_mask, :] = False

    return signals


def _calc_grid_pnl_with_funding(
    positions: list[tuple[float, float, int]],
    exit_price: float,
    exit_idx: int,
    funding_rates: np.ndarray | None,
    candle_timestamps: np.ndarray | None,
    taker_fee: float,
    slippage_pct: float,
) -> float:
    """PnL avec funding payments accumulés (LONG-only).

    Funding payment par position : notional × funding_rate à chaque frontière 8h.
    Pour LONG avec funding négatif : on REÇOIT |funding_rate| × notional.
    """
    total_pnl = 0.0
    for entry_price, quantity, entry_idx in positions:
        notional = entry_price * quantity
        # PnL prix classique (LONG)
        price_pnl = (exit_price - entry_price) * quantity

        # Fees (entry + exit)
        entry_fee = notional * taker_fee
        exit_fee = exit_price * quantity * taker_fee
        slippage = notional * slippage_pct + exit_price * quantity * slippage_pct

        # Funding payments accumulés entre entry et exit
        funding_pnl = 0.0
        if (
            funding_rates is not None
            and candle_timestamps is not None
            and entry_idx < exit_idx
        ):
            for j in range(entry_idx, exit_idx):
                ts = candle_timestamps[j]
                # Frontière 8h : 00:00, 08:00, 16:00 UTC
                hour = int((ts / 3600000) % 24)
                if hour % 8 == 0:
                    fr = funding_rates[j]
                    if not np.isnan(fr):
                        # LONG : -fr × notional (fr négatif → bonus positif)
                        funding_pnl -= fr * notional

        total_pnl += price_pnl + funding_pnl - entry_fee - exit_fee - slippage

    return total_pnl


def _simulate_grid_funding(
    cache: IndicatorCache,
    params: dict[str, Any],
    bt_config: BacktestConfig,
) -> tuple[list[float], list[float], float]:
    """Boucle de simulation Grid Funding (LONG-only, signal funding rate)."""
    n = cache.n_candles
    num_levels = params["num_levels"]
    sl_pct = params["sl_percent"] / 100
    ma_period = params["ma_period"]
    min_hold = params.get("min_hold_candles", 8)
    tp_mode = params.get("tp_mode", "funding_or_sma")

    sma_arr = cache.bb_sma[ma_period]
    funding = cache.funding_rates_1h
    candle_ts = cache.candle_timestamps
    entry_signals = _build_entry_signals(cache, params, num_levels)

    capital = bt_config.initial_capital
    leverage = bt_config.leverage
    taker_fee = bt_config.taker_fee
    slippage_pct = bt_config.slippage_pct

    # Positions : list of (entry_price, quantity, entry_candle_idx, level)
    positions: list[tuple[float, float, int, int]] = []
    filled_levels: set[int] = set()
    trade_pnls: list[float] = []
    trade_returns: list[float] = []

    start_idx = ma_period + 1  # attendre SMA valide

    for i in range(start_idx, n):
        close = cache.closes[i]
        if capital <= 0 or math.isnan(close):
            continue

        # === CHECK EXIT ===
        if positions:
            avg_entry = sum(p[0] * p[1] for p in positions) / sum(p[1] for p in positions)
            min_candles_held = min(i - p[2] for p in positions)
            fr = 0.0
            if funding is not None and not np.isnan(funding[i]):
                fr = funding[i]

            should_exit = False

            # SL (toujours actif)
            sl_price = avg_entry * (1 - sl_pct)
            if close <= sl_price:
                should_exit = True

            # TP (seulement après min_hold)
            if not should_exit and min_candles_held >= min_hold:
                if tp_mode in ("funding_positive", "funding_or_sma") and fr > 0:
                    should_exit = True
                if tp_mode in ("sma_cross", "funding_or_sma") and close >= sma_arr[i]:
                    should_exit = True

            if should_exit:
                pnl = _calc_grid_pnl_with_funding(
                    [(p[0], p[1], p[2]) for p in positions],
                    close, i, funding, candle_ts,
                    taker_fee, slippage_pct,
                )
                trade_pnls.append(pnl)
                if capital > 0:
                    trade_returns.append(pnl / capital)
                capital += pnl
                positions = []
                filled_levels = set()
                continue

        # === CHECK ENTRY ===
        if capital > 0:
            for lvl in range(num_levels):
                if lvl not in filled_levels and entry_signals[i, lvl]:
                    margin_per_level = capital / num_levels
                    notional = margin_per_level * leverage
                    qty = notional / close
                    fee = notional * taker_fee
                    capital -= fee
                    positions.append((close, qty, i, lvl))
                    filled_levels.add(lvl)

        # === FORCE CLOSE DERNIÈRE BOUGIE ===
        if i == n - 1 and positions:
            pnl = _calc_grid_pnl_with_funding(
                [(p[0], p[1], p[2]) for p in positions],
                close, i, funding, candle_ts,
                taker_fee, slippage_pct,
            )
            trade_pnls.append(pnl)
            if capital > 0:
                trade_returns.append(pnl / capital)
            capital += pnl

    return trade_pnls, trade_returns, capital


# ─── Entry point ───────────────────────────────────────────────────────────


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
            cache, params, bt_config, direction=1,
        )
    elif strategy_name == "envelope_dca_short":
        trade_pnls, trade_returns, final_capital = _simulate_envelope_dca(
            cache, params, bt_config, direction=-1,
        )
    elif strategy_name == "grid_atr":
        trade_pnls, trade_returns, final_capital = _simulate_grid_atr(
            cache, params, bt_config, direction=1,
        )
    elif strategy_name == "grid_multi_tf":
        trade_pnls, trade_returns, final_capital = _simulate_grid_multi_tf(
            cache, params, bt_config,
        )
    elif strategy_name == "grid_funding":
        trade_pnls, trade_returns, final_capital = _simulate_grid_funding(
            cache, params, bt_config,
        )
    else:
        raise ValueError(f"Stratégie grid inconnue pour fast engine: {strategy_name}")

    return _compute_fast_metrics(
        params, trade_pnls, trade_returns, final_capital,
        bt_config.initial_capital, cache.total_days,
    )


# ─── Wrappers backward-compat ─────────────────────────────────────────────


def _simulate_envelope_dca(
    cache: IndicatorCache,
    params: dict[str, Any],
    bt_config: BacktestConfig,
    direction: int = 1,
) -> tuple[list[float], list[float], float]:
    """Simulation multi-position Envelope DCA (LONG ou SHORT).

    Wrapper backward-compat — délègue à _build_entry_prices + _simulate_grid_common.
    """
    strategy_name = "envelope_dca_short" if direction == -1 else "envelope_dca"
    num_levels = params["num_levels"]
    sl_pct = params["sl_percent"] / 100
    sma_arr = cache.bb_sma[params["ma_period"]]
    entry_prices = _build_entry_prices(strategy_name, cache, params, num_levels, direction)
    return _simulate_grid_common(
        entry_prices, sma_arr, cache, bt_config, num_levels, sl_pct, direction,
    )


def _simulate_grid_atr(
    cache: IndicatorCache,
    params: dict[str, Any],
    bt_config: BacktestConfig,
    direction: int = 1,
) -> tuple[list[float], list[float], float]:
    """Simulation multi-position Grid ATR (LONG ou SHORT).

    Wrapper backward-compat — délègue à _build_entry_prices + _simulate_grid_common.
    """
    num_levels = params["num_levels"]
    sl_pct = params["sl_percent"] / 100
    sma_arr = cache.bb_sma[params["ma_period"]]
    entry_prices = _build_entry_prices("grid_atr", cache, params, num_levels, direction)
    return _simulate_grid_common(
        entry_prices, sma_arr, cache, bt_config, num_levels, sl_pct, direction,
    )


def _simulate_grid_multi_tf(
    cache: IndicatorCache,
    params: dict[str, Any],
    bt_config: BacktestConfig,
) -> tuple[list[float], list[float], float]:
    """Simulation Grid Multi-TF (direction dynamique Supertrend 4h).

    La direction change au cours du temps : force-close au flip.
    """
    num_levels = params["num_levels"]
    sl_pct = params["sl_percent"] / 100
    sma_arr = cache.bb_sma[params["ma_period"]]
    st_key = (params["st_atr_period"], params["st_atr_multiplier"])
    dir_arr = cache.supertrend_dir_4h[st_key]
    entry_prices = _build_entry_prices("grid_multi_tf", cache, params, num_levels, direction=1)
    return _simulate_grid_common(
        entry_prices, sma_arr, cache, bt_config, num_levels, sl_pct,
        direction=1,
        directions=dir_arr,
    )


# ─── Helpers ───────────────────────────────────────────────────────────────


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
