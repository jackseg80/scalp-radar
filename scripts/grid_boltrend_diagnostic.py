"""Script diagnostic grid_boltrend — trade log détaillé.

Lance _simulate_grid_boltrend() avec logging des 10 premiers trades :
- Candle index d'entrée, prix d'entrée, direction
- Candle index de sortie, prix de sortie, raison (TP/SL)
- PnL du trade
- Frais payés

Usage :
    uv run python scripts/grid_boltrend_diagnostic.py
    uv run python scripts/grid_boltrend_diagnostic.py --output trade_log.txt
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ajouter la racine du projet au path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.backtesting.engine import BacktestConfig
from backend.core.indicators import atr as compute_atr
from backend.core.indicators import bollinger_bands, sma
from backend.optimization.indicator_cache import IndicatorCache


# ─── Params WFO optimisés ─────────────────────────────────────────────────

WFO_PARAMS = {
    "bol_window": 50,
    "bol_std": 2.0,
    "long_ma_window": 200,
    "min_bol_spread": 0.0,
    "atr_period": 10,
    "atr_spacing_mult": 0.5,
    "num_levels": 4,
    "sl_percent": 10.0,
    "sides": ["long", "short"],
}

BT_CONFIG = BacktestConfig(
    symbol="BTC/USDT",
    start_date=None,  # type: ignore[arg-type]
    end_date=None,  # type: ignore[arg-type]
    initial_capital=10_000.0,
    leverage=6,
    taker_fee=0.0006,
    maker_fee=0.0002,
    slippage_pct=0.0001,
)


# ─── Trade log dataclass ──────────────────────────────────────────────────


@dataclass
class TradeLog:
    trade_idx: int
    direction: str  # "LONG" ou "SHORT"
    n_positions: int
    entry_candle_idx: int
    exit_candle_idx: int
    avg_entry_price: float
    exit_price: float
    exit_reason: str
    gross_pnl: float
    total_entry_fees: float
    exit_fee: float
    slippage_cost: float
    net_pnl: float


# ─── Version instrumentée de _simulate_grid_boltrend ──────────────────────


def _simulate_grid_boltrend_with_log(
    cache: IndicatorCache,
    params: dict,
    bt_config: BacktestConfig,
    max_trades: int = 10,
) -> list[TradeLog]:
    """Version de _simulate_grid_boltrend qui collecte les détails des trades."""
    n = cache.n_candles
    num_levels = params["num_levels"]
    sl_pct = params["sl_percent"] / 100
    spacing_mult = params["atr_spacing_mult"]
    sides = params.get("sides", ["long", "short"])
    min_bol_spread = params.get("min_bol_spread", 0.0)

    bol_window = params["bol_window"]
    bol_std = params["bol_std"]
    long_ma_window = params["long_ma_window"]
    atr_period = params["atr_period"]

    bb_sma = cache.bb_sma[bol_window]
    bb_upper = cache.bb_upper[(bol_window, bol_std)]
    bb_lower = cache.bb_lower[(bol_window, bol_std)]
    long_ma = cache.bb_sma[long_ma_window]
    atr_arr = cache.atr_by_period[atr_period]

    closes = cache.closes
    highs = cache.highs
    lows = cache.lows

    capital = bt_config.initial_capital
    leverage = bt_config.leverage
    taker_fee = bt_config.taker_fee
    maker_fee = bt_config.maker_fee
    slippage_pct = bt_config.slippage_pct

    trade_logs: list[TradeLog] = []
    trade_idx = 0

    # State machine
    positions: list[tuple[int, float, float, float]] = []
    entry_levels: list[float] = []
    direction = 0
    breakout_candle_idx = -1

    start_idx = max(bol_window, long_ma_window) + 1

    for i in range(start_idx, n):
        close_i = closes[i]
        if capital <= 0 or math.isnan(close_i):
            continue

        # === EXIT CHECK ===
        if positions:
            avg_entry = sum(p[1] * p[2] for p in positions) / sum(
                p[2] for p in positions
            )

            sl_hit = False
            if direction == 1:
                sl_price = avg_entry * (1 - sl_pct)
                sl_hit = lows[i] <= sl_price
            else:
                sl_price = avg_entry * (1 + sl_pct)
                sl_hit = highs[i] >= sl_price

            tp_hit = False
            sma_val = bb_sma[i]
            if not math.isnan(sma_val):
                if direction == 1:
                    tp_hit = closes[i] < sma_val
                else:
                    tp_hit = closes[i] > sma_val

            exit_reason = None
            exit_price = close_i
            if sl_hit and tp_hit:
                is_green = closes[i] >= cache.opens[i]
                if direction == 1:
                    exit_reason = "signal_exit" if is_green else "sl_global"
                    exit_price = close_i if is_green else sl_price
                else:
                    exit_reason = "signal_exit" if not is_green else "sl_global"
                    exit_price = close_i if not is_green else sl_price
            elif sl_hit:
                exit_reason = "sl_global"
                exit_price = sl_price
            elif tp_hit:
                exit_reason = "signal_exit"
                exit_price = close_i  # prix réel de marché, pas la SMA

            if exit_reason is not None:
                # signal_exit = clôture marché (taker fee + slippage)
                fee_rate = taker_fee
                slip = slippage_pct

                # Calcul PnL détaillé
                total_gross = 0.0
                total_entry_fees = 0.0
                total_exit_fees = 0.0
                total_slippage = 0.0

                for _lvl, ep, qty, ef in positions:
                    actual_exit = exit_price
                    if slip > 0:
                        if direction == 1:
                            actual_exit = exit_price * (1 - slip)
                        else:
                            actual_exit = exit_price * (1 + slip)
                    if direction == 1:
                        gross = (actual_exit - ep) * qty
                    else:
                        gross = (ep - actual_exit) * qty
                    exit_f = qty * exit_price * fee_rate
                    slip_cost = qty * exit_price * slip if slip > 0 else 0.0

                    total_gross += gross
                    total_entry_fees += ef
                    total_exit_fees += exit_f
                    total_slippage += slip_cost

                net_pnl = total_gross - total_entry_fees - total_exit_fees - total_slippage

                if trade_idx < max_trades:
                    trade_logs.append(
                        TradeLog(
                            trade_idx=trade_idx,
                            direction="LONG" if direction == 1 else "SHORT",
                            n_positions=len(positions),
                            entry_candle_idx=breakout_candle_idx,
                            exit_candle_idx=i,
                            avg_entry_price=avg_entry,
                            exit_price=exit_price,
                            exit_reason=exit_reason,
                            gross_pnl=total_gross,
                            total_entry_fees=total_entry_fees,
                            exit_fee=total_exit_fees,
                            slippage_cost=total_slippage,
                            net_pnl=net_pnl,
                        )
                    )

                capital += net_pnl
                trade_idx += 1
                positions = []
                entry_levels = []
                direction = 0
                continue

        # === GUARD ===
        if capital <= 0:
            continue

        # === DCA FILLING ===
        if direction != 0 and entry_levels:
            filled = {p[0] for p in positions}
            for lvl in range(num_levels):
                if lvl in filled or lvl >= len(entry_levels):
                    continue
                if len(positions) >= num_levels:
                    break
                ep = entry_levels[lvl]
                if math.isnan(ep) or ep <= 0:
                    continue
                triggered = False
                if direction == 1:
                    triggered = lows[i] <= ep
                else:
                    triggered = highs[i] >= ep
                if triggered:
                    notional = capital * (1.0 / num_levels) * leverage
                    qty = notional / ep
                    if qty <= 0:
                        continue
                    entry_fee = qty * ep * taker_fee
                    positions.append((lvl, ep, qty, entry_fee))

        # === BREAKOUT DETECTION ===
        if direction == 0 and not positions:
            prev_close = closes[i - 1]
            prev_upper = bb_upper[i - 1]
            prev_lower = bb_lower[i - 1]
            curr_upper = bb_upper[i]
            curr_lower = bb_lower[i]
            long_ma_val = long_ma[i]
            atr_val = atr_arr[i]

            if any(
                math.isnan(v)
                for v in [
                    prev_close,
                    prev_upper,
                    prev_lower,
                    curr_upper,
                    curr_lower,
                    long_ma_val,
                    atr_val,
                ]
            ):
                continue

            if prev_lower > 0:
                prev_spread = (prev_upper - prev_lower) / prev_lower
            else:
                continue
            spread_ok = prev_spread > min_bol_spread

            long_bo = (
                "long" in sides
                and prev_close < prev_upper
                and close_i > curr_upper
                and spread_ok
                and close_i > long_ma_val
            )
            short_bo = (
                "short" in sides
                and prev_close > prev_lower
                and close_i < curr_lower
                and spread_ok
                and close_i < long_ma_val
            )

            if long_bo or short_bo:
                direction = 1 if long_bo else -1
                spacing = atr_val * spacing_mult
                breakout_candle_idx = i

                entry_levels = []
                for k in range(num_levels):
                    if direction == 1:
                        entry_levels.append(close_i - k * spacing)
                    else:
                        entry_levels.append(close_i + k * spacing)

                ep0 = entry_levels[0]
                if ep0 > 0 and capital > 0:
                    notional = capital * (1.0 / num_levels) * leverage
                    qty = notional / ep0
                    if qty > 0:
                        entry_fee = qty * ep0 * taker_fee
                        positions.append((0, ep0, qty, entry_fee))

    # Force close
    if positions and trade_idx < max_trades:
        avg_entry = sum(p[1] * p[2] for p in positions) / sum(p[2] for p in positions)
        exit_price = float(closes[n - 1])

        total_gross = 0.0
        total_entry_fees = 0.0
        total_exit_fees = 0.0
        total_slippage = 0.0

        for _lvl, ep, qty, ef in positions:
            actual_exit = exit_price * (1 - slippage_pct) if direction == 1 else exit_price * (1 + slippage_pct)
            if direction == 1:
                gross = (actual_exit - ep) * qty
            else:
                gross = (ep - actual_exit) * qty
            exit_f = qty * exit_price * taker_fee
            slip_cost = qty * exit_price * slippage_pct

            total_gross += gross
            total_entry_fees += ef
            total_exit_fees += exit_f
            total_slippage += slip_cost

        net_pnl = total_gross - total_entry_fees - total_exit_fees - total_slippage

        trade_logs.append(
            TradeLog(
                trade_idx=trade_idx,
                direction="LONG" if direction == 1 else "SHORT",
                n_positions=len(positions),
                entry_candle_idx=breakout_candle_idx,
                exit_candle_idx=n - 1,
                avg_entry_price=avg_entry,
                exit_price=exit_price,
                exit_reason="end_of_data",
                gross_pnl=total_gross,
                total_entry_fees=total_entry_fees,
                exit_fee=total_exit_fees,
                slippage_cost=total_slippage,
                net_pnl=net_pnl,
            )
        )

    return trade_logs


# ─── Construction données synthétiques ────────────────────────────────────


def build_synthetic_data(n: int = 500, seed: int = 42) -> IndicatorCache:
    """Construit un IndicatorCache avec données BTC/USDT synthétiques."""
    rng = np.random.default_rng(seed)
    params = WFO_PARAMS

    closes = np.full(n, 100.0)
    closes[:250] = 100.0 + rng.normal(0, 0.1, 250)
    closes[250:260] = np.linspace(100, 120, 10)
    closes[260:300] = 118.0 + rng.normal(0, 0.3, 40)
    closes[300:400] = np.linspace(118, 95, 100)
    closes[400:] = 95.0 + rng.normal(0, 0.1, n - 400)

    noise_open = rng.uniform(-0.1, 0.1, n)
    opens = closes + noise_open
    spread = np.abs(rng.normal(0.3, 0.1, n))
    highs = np.maximum(opens, closes) + spread
    lows = np.minimum(opens, closes) - spread

    bb_sma_arr, bb_upper_arr, bb_lower_arr = bollinger_bands(
        closes, params["bol_window"], params["bol_std"]
    )
    long_ma_arr = sma(closes, params["long_ma_window"])
    atr_arr = compute_atr(highs, lows, closes, params["atr_period"])

    ts = np.arange(n, dtype=np.float64) * 3600000
    volumes = np.full(n, 1000.0)

    return IndicatorCache(
        n_candles=n,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
        total_days=n / 24,
        rsi={14: np.full(n, 50.0)},
        vwap=np.full(n, np.nan),
        vwap_distance_pct=np.full(n, np.nan),
        adx_arr=np.full(n, 25.0),
        di_plus=np.full(n, 15.0),
        di_minus=np.full(n, 10.0),
        atr_arr=np.full(n, 1.0),
        atr_sma=np.full(n, 1.0),
        volume_sma_arr=np.full(n, 100.0),
        regime=np.zeros(n, dtype=np.int8),
        rolling_high={},
        rolling_low={},
        filter_adx=np.full(n, np.nan),
        filter_di_plus=np.full(n, np.nan),
        filter_di_minus=np.full(n, np.nan),
        bb_sma={
            params["bol_window"]: bb_sma_arr,
            params["long_ma_window"]: long_ma_arr,
        },
        bb_upper={(params["bol_window"], params["bol_std"]): bb_upper_arr},
        bb_lower={(params["bol_window"], params["bol_std"]): bb_lower_arr},
        supertrend_direction={},
        atr_by_period={params["atr_period"]: atr_arr},
        supertrend_dir_4h={},
        funding_rates_1h=None,
        candle_timestamps=ts,
        ema_by_period={},
        adx_by_period={},
    )


# ─── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Diagnostic grid_boltrend trade log")
    parser.add_argument("--output", "-o", type=str, help="Fichier de sortie")
    parser.add_argument("--n-candles", type=int, default=500, help="Nombre de candles")
    parser.add_argument("--max-trades", type=int, default=10, help="Max trades à logger")
    args = parser.parse_args()

    print("=== Grid BolTrend Diagnostic ===")
    print(f"Params WFO : {WFO_PARAMS}")
    print(f"Config : capital={BT_CONFIG.initial_capital}, leverage={BT_CONFIG.leverage}")
    print(f"Fees : taker={BT_CONFIG.taker_fee}, maker={BT_CONFIG.maker_fee}")
    print()

    cache = build_synthetic_data(n=args.n_candles)
    trade_logs = _simulate_grid_boltrend_with_log(
        cache, WFO_PARAMS, BT_CONFIG, max_trades=args.max_trades
    )

    lines = []
    lines.append(f"{'#':>3} {'Dir':>5} {'Pos':>3} {'Entry_i':>7} {'Exit_i':>6} "
                  f"{'Avg_entry':>10} {'Exit_px':>10} {'Reason':>12} "
                  f"{'Gross':>10} {'Entry_fee':>10} {'Exit_fee':>10} "
                  f"{'Slippage':>10} {'Net_PnL':>10}")
    lines.append("-" * 120)

    for t in trade_logs:
        lines.append(
            f"{t.trade_idx:>3} {t.direction:>5} {t.n_positions:>3} "
            f"{t.entry_candle_idx:>7} {t.exit_candle_idx:>6} "
            f"{t.avg_entry_price:>10.2f} {t.exit_price:>10.2f} {t.exit_reason:>12} "
            f"{t.gross_pnl:>+10.2f} {t.total_entry_fees:>10.4f} {t.exit_fee:>10.4f} "
            f"{t.slippage_cost:>10.6f} {t.net_pnl:>+10.2f}"
        )

    if not trade_logs:
        lines.append("  Aucun trade détecté — vérifier les données")

    output = "\n".join(lines)
    print(output)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"\nSauvegardé dans {args.output}")


if __name__ == "__main__":
    main()
