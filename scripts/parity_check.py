"""Script de diagnostic : compare fast engine vs normal engine + échange bitget vs binance.

Usage :
    uv run python scripts/parity_check.py
"""
import os
os.environ.setdefault("PYTHON_JIT", "0")

import asyncio
from datetime import datetime, timedelta, timezone

from backend.backtesting.engine import BacktestConfig, run_backtest_single
from backend.backtesting.metrics import calculate_metrics
from backend.core.database import Database
from backend.core.models import Candle
from backend.optimization.indicator_cache import build_cache
from backend.optimization.fast_backtest import run_backtest_from_cache


# Paramètres par défaut (strategies.yaml)
DEFAULT_PARAMS = {
    "rsi_period": 14,
    "rsi_long_threshold": 30,
    "rsi_short_threshold": 70,
    "volume_spike_multiplier": 2.0,
    "vwap_deviation_entry": 0.3,
    "trend_adx_threshold": 25.0,
    "tp_percent": 0.8,
    "sl_percent": 0.3,
}

GRID_VALUES = {
    "rsi_period": [14],
    "rsi_long_threshold": [30],
    "rsi_short_threshold": [70],
    "volume_spike_multiplier": [2.0],
    "vwap_deviation_entry": [0.3],
    "trend_adx_threshold": [25.0],
    "tp_percent": [0.8],
    "sl_percent": [0.3],
}


async def load_data(exchange: str, symbol: str = "BTC/USDT", days: int = 180):
    """Charge les données depuis la DB."""
    db = Database()
    await db.init()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    candles_by_tf = {}
    for tf in ["5m", "15m"]:
        candles = await db.get_candles(symbol, tf, start=start, end=end,
                                       exchange=exchange, limit=1_000_000)
        candles_by_tf[tf] = candles

    await db.close()
    return candles_by_tf


def run_normal_engine(candles_by_tf, params=None):
    """Lance le moteur normal."""
    if params is None:
        params = DEFAULT_PARAMS

    main_candles = candles_by_tf["5m"]
    if not main_candles:
        return None

    bt_config = BacktestConfig(
        symbol="BTC/USDT",
        start_date=main_candles[0].timestamp,
        end_date=main_candles[-1].timestamp,
    )

    result = run_backtest_single("vwap_rsi", params, candles_by_tf, bt_config, "5m")
    metrics = calculate_metrics(result)
    return {
        "trades": metrics.total_trades,
        "sharpe": round(metrics.sharpe_ratio, 4),
        "net_return_pct": round(metrics.net_return_pct, 4),
        "profit_factor": round(metrics.profit_factor, 4),
        "net_pnl": round(metrics.net_pnl, 2),
    }


def run_fast_engine(candles_by_tf, params=None):
    """Lance le fast engine."""
    if params is None:
        params = DEFAULT_PARAMS

    main_candles = candles_by_tf["5m"]
    if not main_candles:
        return None

    bt_config = BacktestConfig(
        symbol="BTC/USDT",
        start_date=main_candles[0].timestamp,
        end_date=main_candles[-1].timestamp,
    )

    cache = build_cache(candles_by_tf, GRID_VALUES, "vwap_rsi")
    result = run_backtest_from_cache("vwap_rsi", params, cache, bt_config)
    _, sharpe, net_return_pct, profit_factor, n_trades = result
    return {
        "trades": n_trades,
        "sharpe": round(sharpe, 4),
        "net_return_pct": round(net_return_pct, 4),
        "profit_factor": round(profit_factor, 4),
    }


def count_signal_conditions(candles_by_tf):
    """Diagnostic : combien de bougies remplissent chaque condition d'entrée."""
    import numpy as np
    from backend.core.indicators import (
        rsi, vwap_rolling, adx, atr, volume_sma, sma, detect_market_regime
    )
    from backend.core.models import MarketRegime

    main_candles = candles_by_tf["5m"]
    filter_candles = candles_by_tf.get("15m", [])
    n = len(main_candles)

    closes = np.array([c.close for c in main_candles], dtype=float)
    highs = np.array([c.high for c in main_candles], dtype=float)
    lows = np.array([c.low for c in main_candles], dtype=float)
    volumes = np.array([c.volume for c in main_candles], dtype=float)

    rsi_arr = rsi(closes, 14)
    vwap_arr = vwap_rolling(highs, lows, closes, volumes)
    adx_arr, di_plus, di_minus = adx(highs, lows, closes)
    atr_arr = atr(highs, lows, closes)
    vol_sma_arr = volume_sma(volumes)

    # ATR SMA aligned
    atr_sma_full = np.full_like(atr_arr, np.nan)
    valid_mask = ~np.isnan(atr_arr)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) >= 20:
        atr_valid = atr_arr[valid_mask]
        atr_sma_valid = sma(atr_valid, 20)
        for j, idx in enumerate(valid_indices):
            if not np.isnan(atr_sma_valid[j]):
                atr_sma_full[idx] = atr_sma_valid[j]

    # VWAP deviation %
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap_dev = np.where(
            (~np.isnan(vwap_arr)) & (vwap_arr > 0),
            (closes - vwap_arr) / vwap_arr * 100,
            np.nan,
        )

    # Regimes
    regimes = []
    for i in range(n):
        r = detect_market_regime(adx_arr[i], di_plus[i], di_minus[i],
                                  atr_arr[i], atr_sma_full[i])
        regimes.append(r)

    # Filtrer 15m
    from backend.optimization.indicator_cache import _build_aligned_filter
    f_adx, f_di_plus, f_di_minus = _build_aligned_filter(main_candles, filter_candles)

    # Conditions individuelles
    valid = ~np.isnan(rsi_arr) & ~np.isnan(vwap_arr) & ~np.isnan(vol_sma_arr)
    rsi_long = rsi_arr < 30
    rsi_short = rsi_arr > 70
    vwap_long = vwap_dev < -0.3
    vwap_short = vwap_dev > 0.3
    vol_spike = (vol_sma_arr > 0) & (volumes > vol_sma_arr * 2.0)
    regime_ok = np.array([r in (MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY) for r in regimes])
    trend_filter = ~np.isnan(f_adx) & (f_adx > 25.0)
    is_15m_bearish = ~np.isnan(f_adx) & (f_adx > 20) & (f_di_minus > f_di_plus)
    is_15m_bullish = ~np.isnan(f_adx) & (f_adx > 20) & (f_di_plus > f_di_minus)

    base = valid & vol_spike & ~trend_filter & regime_ok
    longs = base & rsi_long & vwap_long & ~is_15m_bearish
    shorts = base & rsi_short & vwap_short & ~is_15m_bullish

    print(f"  Bougies totales        : {n}")
    print(f"  Indicateurs valides    : {np.sum(valid)}")
    print(f"  RSI < 30 (long)        : {np.sum(rsi_long & valid)}")
    print(f"  RSI > 70 (short)       : {np.sum(rsi_short & valid)}")
    print(f"  VWAP dev < -0.3% (long): {np.sum(vwap_long & valid)}")
    print(f"  VWAP dev > +0.3% (short): {np.sum(vwap_short & valid)}")
    print(f"  Volume spike           : {np.sum(vol_spike & valid)}")
    print(f"  Regime OK (RANGING/LOW): {np.sum(regime_ok & valid)}")
    print(f"  15m PAS en tendance    : {np.sum(~trend_filter & valid)}")
    print(f"  Base (valid+vol+regime+15m): {np.sum(base)}")
    print(f"  Signaux LONG           : {np.sum(longs)}")
    print(f"  Signaux SHORT          : {np.sum(shorts)}")
    print(f"  Total signaux          : {np.sum(longs) + np.sum(shorts)}")


async def main():
    print("=" * 60)
    print("  DIAGNOSTIC PARITE FAST ENGINE vs NORMAL ENGINE")
    print("=" * 60)

    for exchange in ["bitget", "binance"]:
        print(f"\n{'-' * 60}")
        print(f"  Exchange : {exchange.upper()}")
        print(f"{'-' * 60}")

        candles_by_tf = await load_data(exchange)

        n_5m = len(candles_by_tf.get("5m", []))
        n_15m = len(candles_by_tf.get("15m", []))
        print(f"\n  Donnees : {n_5m} bougies 5m, {n_15m} bougies 15m")

        if n_5m == 0:
            print(f"  !! PAS DE DONNEES {exchange} — skip")
            continue

        first = candles_by_tf["5m"][0].timestamp
        last = candles_by_tf["5m"][-1].timestamp
        days = (last - first).days
        print(f"  Periode : {first.strftime('%Y-%m-%d')} -> {last.strftime('%Y-%m-%d')} ({days}j)")

        print(f"\n  --- Diagnostic conditions d'entrée ---")
        count_signal_conditions(candles_by_tf)

        print(f"\n  --- Moteur normal (BacktestEngine) ---")
        normal = run_normal_engine(candles_by_tf)
        if normal:
            for k, v in normal.items():
                print(f"    {k:<18s}: {v}")

        print(f"\n  --- Fast engine (IndicatorCache) ---")
        fast = run_fast_engine(candles_by_tf)
        if fast:
            for k, v in fast.items():
                print(f"    {k:<18s}: {v}")

        if normal and fast:
            print(f"\n  --- Comparaison ---")
            trade_match = normal["trades"] == fast["trades"]
            sharpe_close = abs(normal["sharpe"] - fast["sharpe"]) < 0.1
            print(f"    Trades   : {'OK MATCH' if trade_match else 'XX DIVERGENT'} "
                  f"(normal={normal['trades']}, fast={fast['trades']})")
            print(f"    Sharpe   : {'OK PROCHE' if sharpe_close else 'XX DIVERGENT'} "
                  f"(normal={normal['sharpe']}, fast={fast['sharpe']})")
            pnl_match = abs(normal.get("net_return_pct", 0) - fast["net_return_pct"]) < 0.5
            print(f"    Return % : {'OK PROCHE' if pnl_match else 'XX DIVERGENT'} "
                  f"(normal={normal.get('net_return_pct', 'N/A')}, fast={fast['net_return_pct']})")

    # Test sur fenêtre 120j (taille IS du WFO) avec données binance
    print(f"\n{'-' * 60}")
    print(f"  TEST FENETRE 120j (taille IS du WFO) — BINANCE")
    print(f"{'-' * 60}")

    candles_binance = await load_data("binance", days=120)
    n_5m = len(candles_binance.get("5m", []))
    if n_5m > 0:
        print(f"\n  Donnees : {n_5m} bougies 5m")
        print(f"\n  --- Diagnostic conditions d'entrée ---")
        count_signal_conditions(candles_binance)

        print(f"\n  --- Moteur normal (120j) ---")
        normal = run_normal_engine(candles_binance)
        if normal:
            for k, v in normal.items():
                print(f"    {k:<18s}: {v}")

        print(f"\n  --- Fast engine (120j) ---")
        fast = run_fast_engine(candles_binance)
        if fast:
            for k, v in fast.items():
                print(f"    {k:<18s}: {v}")
    else:
        print("  !! PAS DE DONNEES BINANCE 120j")

    print(f"\n{'=' * 60}")
    print("  FIN DIAGNOSTIC")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
