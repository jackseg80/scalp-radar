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
    entry_prices = np.full((n, num_levels), np.nan)

    if strategy_name in ("envelope_dca", "envelope_dca_short"):
        sma_arr = cache.bb_sma[params["ma_period"]]
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
        sma_arr = cache.bb_sma[params["ma_period"]]
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
        sma_arr = cache.bb_sma[params["ma_period"]]
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

    elif strategy_name == "grid_trend":
        ema_fast_arr = cache.ema_by_period[params["ema_fast"]]
        ema_slow_arr = cache.ema_by_period[params["ema_slow"]]
        atr_arr = cache.atr_by_period[params["atr_period"]]
        adx_arr = cache.adx_by_period[params["adx_period"]]
        adx_threshold = params["adx_threshold"]

        multipliers = [
            params["pull_start"] + lvl * params["pull_step"]
            for lvl in range(num_levels)
        ]
        long_mask = (ema_fast_arr > ema_slow_arr) & (adx_arr > adx_threshold)
        short_mask = (ema_fast_arr < ema_slow_arr) & (adx_arr > adx_threshold)
        for lvl in range(num_levels):
            entry_prices[long_mask, lvl] = (
                ema_fast_arr[long_mask] - atr_arr[long_mask] * multipliers[lvl]
            )
            entry_prices[short_mask, lvl] = (
                ema_fast_arr[short_mask] + atr_arr[short_mask] * multipliers[lvl]
            )
        invalid = (
            np.isnan(ema_fast_arr) | np.isnan(ema_slow_arr)
            | np.isnan(atr_arr) | np.isnan(adx_arr) | (atr_arr <= 0)
        )
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
    trail_mult: float = 0.0,
    trail_atr_arr: np.ndarray | None = None,
) -> tuple[list[float], list[float], float]:
    """Boucle chaude unifiée pour toutes les stratégies grid/DCA.

    Args:
        entry_prices: (n_candles, num_levels) pré-calculé par _build_entry_prices.
        sma_arr: SMA pour TP dynamique (retour vers la SMA). Ignoré si trail_mult > 0.
        sl_pct: déjà divisé par 100.
        direction: 1 = LONG, -1 = SHORT (scalar fixe, ou initial pour directions dynamiques).
        directions: si fourni, array 1D de directions par candle (1/-1/0/NaN).
            Override le scalar `direction` à chaque candle. Force-close au flip.
            0 = zone neutre (pas de nouvelles ouvertures, positions gérées).
        trail_mult: multiplicateur ATR pour trailing stop (0 = désactivé → TP SMA classique).
        trail_atr_arr: array ATR pour le calcul du trailing stop distance.
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

    # Trailing stop state
    hwm = 0.0  # High Water Mark (LONG) ou Low Water Mark (SHORT)
    neutral_zone = False

    # Funding settlement mask (00:00, 08:00, 16:00 UTC)
    funding_rates = cache.funding_rates_1h
    settlement_mask = np.zeros(n, dtype=bool)
    if funding_rates is not None and cache.candle_timestamps is not None:
        hours = ((cache.candle_timestamps / 3600000) % 24).astype(int)
        settlement_mask = (hours % 8 == 0)

    for i in range(n):
        # --- Directions dynamiques (grid_multi_tf, grid_trend) ---
        if directions is not None:
            cur_dir = directions[i]
            if math.isnan(cur_dir):
                continue  # Pas de data → skip total
            cur_dir_int = int(cur_dir)

            if cur_dir_int == 0:
                # Zone neutre : gérer positions existantes, ne pas en ouvrir
                neutral_zone = True
                # NE PAS mettre à jour direction ni last_dir
            else:
                neutral_zone = False
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
                    hwm = 0.0
                last_dir = cur_dir_int
                direction = cur_dir_int  # Override le scalar pour TP/SL et entry

        # Skip candles invalides (NaN propagé depuis _build_entry_prices)
        if math.isnan(entry_prices[i, 0]):
            continue

        # 1. Check sorties si positions ouvertes
        if positions:
            total_qty = sum(p[2] for p in positions)
            avg_entry = sum(p[1] * p[2] for p in positions) / total_qty

            is_green = cache.closes[i] > cache.opens[i]

            exit_reason = None
            exit_price = 0.0

            # SL classique (toujours actif)
            if direction == 1:
                sl_price = avg_entry * (1 - sl_pct)
                sl_hit = cache.lows[i] <= sl_price
            else:
                sl_price = avg_entry * (1 + sl_pct)
                sl_hit = cache.highs[i] >= sl_price

            if trail_mult > 0 and trail_atr_arr is not None:
                # --- MODE TRAILING STOP (grid_trend) ---
                if direction == 1:
                    hwm = max(hwm, cache.highs[i])
                else:
                    # LWM : init si hwm == 0, puis toujours min
                    hwm = min(hwm, cache.lows[i]) if hwm > 0 else cache.lows[i]

                trail_distance = trail_atr_arr[i] * trail_mult
                if direction == 1:
                    trail_price = hwm - trail_distance
                    trail_hit = trail_price > 0 and cache.lows[i] <= trail_price
                else:
                    trail_price = hwm + trail_distance
                    trail_hit = cache.highs[i] >= trail_price

                if trail_hit and sl_hit:
                    # Heuristique OHLC : bougie verte → trail (prix montait), rouge → SL
                    if direction == 1:
                        exit_reason = "sl_global" if not is_green else "trail_stop"
                    else:
                        exit_reason = "sl_global" if is_green else "trail_stop"
                elif sl_hit:
                    exit_reason = "sl_global"
                elif trail_hit:
                    exit_reason = "trail_stop"

                if exit_reason == "trail_stop":
                    exit_price = trail_price
                elif exit_reason == "sl_global":
                    exit_price = sl_price

            else:
                # --- MODE TP CLASSIQUE (SMA) ---
                tp_price = sma_arr[i]  # Dynamique (retour vers la SMA)

                if direction == 1:
                    tp_hit = cache.highs[i] >= tp_price
                else:
                    tp_hit = cache.lows[i] <= tp_price

                if tp_hit and sl_hit:
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
                # trail_stop et sl_global → taker fee + slippage
                # tp_global → maker fee, pas de slippage
                if exit_reason == "tp_global":
                    fee = maker_fee
                    slip = 0.0
                else:
                    fee = taker_fee
                    slip = slippage_pct

                pnl = _calc_grid_pnl(positions, exit_price, fee, slip, direction)
                trade_pnls.append(pnl)
                if capital > 0:
                    trade_returns.append(pnl / capital)
                capital += pnl
                positions = []
                hwm = 0.0
                continue

        # 2. Funding costs aux settlements 8h
        if positions and settlement_mask[i] and funding_rates is not None:
            fr = funding_rates[i]
            if not math.isnan(fr):
                for _lvl, entry_price, quantity, _fee in positions:
                    notional = entry_price * quantity
                    capital += -fr * notional * direction

        # 3. Guard capital épuisé
        if capital <= 0:
            continue

        # 4. Zone neutre : pas de nouvelles ouvertures
        if neutral_zone:
            continue

        # 5. Ouvrir de nouvelles positions si niveaux touchés
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
                    # Init HWM à la première ouverture (trailing stop)
                    if trail_mult > 0 and hwm == 0.0:
                        if direction == 1:
                            hwm = cache.highs[i]
                        else:
                            hwm = cache.lows[i]

    # Force close fin de données
    if positions:
        exit_price = float(cache.closes[n - 1])
        pnl = _calc_grid_pnl(positions, exit_price, taker_fee, slippage_pct, direction)
        trade_pnls.append(pnl)
        if capital > 0:
            trade_returns.append(pnl / capital)
        capital += pnl

    return trade_pnls, trade_returns, capital


# ─── Grid Trend ───────────────────────────────────────────────────────────


def _simulate_grid_trend(
    cache: IndicatorCache,
    params: dict[str, Any],
    bt_config: BacktestConfig,
) -> tuple[list[float], list[float], float]:
    """Simulation Grid Trend (trend following DCA avec trailing stop ATR).

    Direction dynamique EMA cross + filtre ADX. Trailing stop ATR.
    """
    num_levels = params["num_levels"]
    sl_pct = params["sl_percent"] / 100
    trail_mult = params["trail_mult"]

    ema_fast_arr = cache.ema_by_period[params["ema_fast"]]
    ema_slow_arr = cache.ema_by_period[params["ema_slow"]]
    atr_arr = cache.atr_by_period[params["atr_period"]]
    adx_arr = cache.adx_by_period[params["adx_period"]]
    adx_threshold = params["adx_threshold"]

    # Directions array : +1 (LONG), -1 (SHORT), 0 (neutre)
    n = cache.n_candles
    dir_arr = np.zeros(n, dtype=np.float64)
    long_mask = (ema_fast_arr > ema_slow_arr) & (adx_arr > adx_threshold)
    short_mask = (ema_fast_arr < ema_slow_arr) & (adx_arr > adx_threshold)
    nan_mask = np.isnan(ema_fast_arr) | np.isnan(ema_slow_arr) | np.isnan(adx_arr)
    dir_arr[long_mask] = 1.0
    dir_arr[short_mask] = -1.0
    dir_arr[nan_mask] = np.nan

    entry_prices = _build_entry_prices("grid_trend", cache, params, num_levels, direction=1)

    # EMA fast comme sma_arr placeholder (non utilisé pour TP car trail_mult > 0)
    return _simulate_grid_common(
        entry_prices, ema_fast_arr, cache, bt_config, num_levels, sl_pct,
        direction=1,
        directions=dir_arr,
        trail_mult=trail_mult,
        trail_atr_arr=atr_arr,
    )


# ─── Grid Range ATR ───────────────────────────────────────────────────────


def _simulate_grid_range(
    cache: IndicatorCache,
    params: dict[str, Any],
    bt_config: BacktestConfig,
) -> tuple[list[float], list[float], float]:
    """Simulation Grid Range ATR (bidirectional, individual TP/SL).

    LONG et SHORT simultanés. Chaque position a son propre TP (retour à SMA)
    et SL (% depuis entry). Les positions se ferment indépendamment.
    """
    num_levels = params["num_levels"]  # par côté
    sl_pct = params["sl_percent"] / 100
    tp_mode = params.get("tp_mode", "dynamic_sma")
    sides = params.get("sides", ["long", "short"])

    sma_arr = cache.bb_sma[params["ma_period"]]
    atr_arr = cache.atr_by_period[params["atr_period"]]
    spacing_mult = params["atr_spacing_mult"]

    capital = bt_config.initial_capital
    leverage = bt_config.leverage
    taker_fee = bt_config.taker_fee
    maker_fee = bt_config.maker_fee
    slippage_pct = bt_config.slippage_pct
    n = cache.n_candles

    trade_pnls: list[float] = []
    trade_returns: list[float] = []

    # Slots : 0..N-1 = LONG, N..2N-1 = SHORT
    n_long = num_levels if "long" in sides else 0
    n_short = num_levels if "short" in sides else 0
    total_slots = n_long + n_short
    if total_slots == 0:
        return trade_pnls, trade_returns, capital

    # Positions : (slot_idx, direction, entry_price, qty, entry_fee, entry_sma)
    positions: list[tuple[int, int, float, float, float, float]] = []

    # Funding settlement mask (00:00, 08:00, 16:00 UTC)
    funding_rates = cache.funding_rates_1h
    settlement_mask = np.zeros(n, dtype=bool)
    if funding_rates is not None and cache.candle_timestamps is not None:
        hours = ((cache.candle_timestamps / 3600000) % 24).astype(int)
        settlement_mask = (hours % 8 == 0)

    for i in range(n):
        cur_sma = sma_arr[i]
        cur_atr = atr_arr[i]
        if math.isnan(cur_sma) or math.isnan(cur_atr) or cur_atr <= 0:
            # Pas d'ouverture mais check exits/funding quand même
            if not positions:
                continue
            # Skip exit checks si pas de SMA valide pour TP
            # (funding quand même traité plus bas)
            pass
        else:
            pass  # indicateurs valides, tout est bon

        # --- 1. Check TP/SL individuel pour chaque position ---
        if positions:
            closed_indices: list[int] = []
            for pos_idx, (slot, direction, ep, qty, efee, esma) in enumerate(positions):
                # TP price
                if tp_mode == "fixed_center":
                    tp_price = esma
                else:
                    tp_price = cur_sma

                # SL price
                if direction == 1:
                    sl_price = ep * (1 - sl_pct)
                else:
                    sl_price = ep * (1 + sl_pct)

                # Check hits
                if direction == 1:
                    tp_hit = (not math.isnan(tp_price)) and cache.highs[i] >= tp_price
                    sl_hit = cache.lows[i] <= sl_price
                else:
                    tp_hit = (not math.isnan(tp_price)) and cache.lows[i] <= tp_price
                    sl_hit = cache.highs[i] >= sl_price

                # Heuristique OHLC si les deux sont touchés
                is_green = cache.closes[i] > cache.opens[i]
                exit_reason = None
                exit_price = 0.0

                if tp_hit and sl_hit:
                    if direction == 1:
                        exit_reason = "tp" if is_green else "sl"
                    else:
                        exit_reason = "sl" if is_green else "tp"
                elif sl_hit:
                    exit_reason = "sl"
                elif tp_hit:
                    exit_reason = "tp"

                if exit_reason == "tp":
                    exit_price = tp_price
                elif exit_reason == "sl":
                    exit_price = sl_price

                if exit_reason is not None:
                    # Fee model : TP = maker (limit, 0 slippage), SL = taker + slippage
                    if exit_reason == "tp":
                        fee = maker_fee
                        slip = 0.0
                    else:
                        fee = taker_fee
                        slip = slippage_pct

                    # PnL individuel (même pattern que _calc_grid_pnl)
                    actual_exit = exit_price
                    slippage_cost = 0.0
                    if slip > 0:
                        slippage_cost = qty * exit_price * slip
                        if direction == 1:
                            actual_exit = exit_price * (1 - slip)
                        else:
                            actual_exit = exit_price * (1 + slip)

                    if direction == 1:
                        gross = (actual_exit - ep) * qty
                    else:
                        gross = (ep - actual_exit) * qty

                    exit_fee = qty * exit_price * fee
                    net = gross - efee - exit_fee - slippage_cost
                    trade_pnls.append(net)
                    if capital > 0:
                        trade_returns.append(net / capital)
                    capital += net
                    closed_indices.append(pos_idx)

            # Retirer les positions fermées (ordre inverse)
            for idx in reversed(closed_indices):
                positions.pop(idx)

        # --- 2. Funding costs aux settlements 8h ---
        if positions and settlement_mask[i] and funding_rates is not None:
            fr = funding_rates[i]
            if not math.isnan(fr):
                for _slot, direction, ep, qty, _efee, _esma in positions:
                    notional = ep * qty
                    capital += -fr * notional * direction

        # --- 3. Guard capital épuisé ---
        if capital <= 0:
            continue

        # --- 4. Skip ouvertures si indicateurs invalides ---
        if math.isnan(cur_sma) or math.isnan(cur_atr) or cur_atr <= 0:
            continue

        # --- 5. Ouvrir de nouvelles positions si niveaux touchés ---
        if len(positions) < total_slots:
            filled_slots = {p[0] for p in positions}
            spacing = cur_atr * spacing_mult

            for lvl in range(num_levels):
                if len(positions) >= total_slots or capital <= 0:
                    break

                # LONG slot
                if "long" in sides and lvl not in filled_slots:
                    ep = cur_sma - (lvl + 1) * spacing
                    if ep > 0 and cache.lows[i] <= ep:
                        notional = capital * (1.0 / total_slots) * leverage
                        qty = notional / ep
                        if qty > 0:
                            entry_fee = qty * ep * taker_fee
                            positions.append((lvl, 1, ep, qty, entry_fee, cur_sma))
                            filled_slots.add(lvl)

                # SHORT slot
                short_slot = num_levels + lvl
                if "short" in sides and short_slot not in filled_slots:
                    ep = cur_sma + (lvl + 1) * spacing
                    if cache.highs[i] >= ep:
                        notional = capital * (1.0 / total_slots) * leverage
                        qty = notional / ep
                        if qty > 0:
                            entry_fee = qty * ep * taker_fee
                            positions.append((short_slot, -1, ep, qty, entry_fee, cur_sma))
                            filled_slots.add(short_slot)

    # Force close fin de données
    if positions:
        exit_price = float(cache.closes[n - 1])
        for _slot, direction, ep, qty, efee, _esma in positions:
            actual_exit = exit_price
            slippage_cost = qty * exit_price * slippage_pct
            if direction == 1:
                actual_exit = exit_price * (1 - slippage_pct)
                gross = (actual_exit - ep) * qty
            else:
                actual_exit = exit_price * (1 + slippage_pct)
                gross = (ep - actual_exit) * qty

            exit_fee = qty * exit_price * taker_fee
            net = gross - efee - exit_fee - slippage_cost
            trade_pnls.append(net)
            if capital > 0:
                trade_returns.append(net / capital)
            capital += net

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
    elif strategy_name == "grid_trend":
        trade_pnls, trade_returns, final_capital = _simulate_grid_trend(
            cache, params, bt_config,
        )
    elif strategy_name == "grid_range_atr":
        trade_pnls, trade_returns, final_capital = _simulate_grid_range(
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
