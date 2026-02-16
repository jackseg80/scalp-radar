"""Benchmark du fast engine (indicateurs + trade simulation).

Génère des données synthétiques, construit un cache, exécute N combos,
et mesure le temps total. 3 runs, exclut le 1er (compilation numba),
reporte mean ± std.

Usage :
    uv run python -m scripts.benchmark_fast_engine
    uv run python -m scripts.benchmark_fast_engine --combos 500 --candles 8000
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from backend.backtesting.engine import BacktestConfig
from backend.core.models import Candle
from backend.optimization.fast_backtest import (
    NUMBA_AVAILABLE,
    run_backtest_from_cache,
)
from backend.optimization.indicator_cache import build_cache


@dataclass
class BenchResult:
    strategy: str
    n_combos: int
    n_candles: int
    times: list[float]
    numba: bool

    @property
    def mean(self) -> float:
        return np.mean(self.times)

    @property
    def std(self) -> float:
        return np.std(self.times)


def _make_candles(n: int, tf_minutes: int = 5) -> list[Candle]:
    """Génère N candles synthétiques réalistes."""
    rng = np.random.default_rng(42)
    base_price = 50000.0
    returns = rng.normal(0, 0.002, n)
    prices = base_price * np.cumprod(1 + returns)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    delta = timedelta(minutes=tf_minutes)

    candles = []
    for i in range(n):
        c = prices[i]
        spread = c * rng.uniform(0.0005, 0.003)
        h = c + spread * rng.uniform(0.3, 1.0)
        l = c - spread * rng.uniform(0.3, 1.0)
        o = l + (h - l) * rng.uniform(0.2, 0.8)
        vol = rng.uniform(100, 10000)
        tf_str = {5: "5m", 15: "15m", 60: "1h"}.get(tf_minutes, "5m")
        candles.append(Candle(
            timestamp=start + delta * i,
            symbol="BTCUSDT", timeframe=tf_str,
            open=o, high=h, low=l, close=c, volume=vol,
        ))
    return candles


def _make_param_combos(strategy: str, n_combos: int) -> list[dict]:
    """Génère N combinaisons de paramètres pour la stratégie."""
    rng = np.random.default_rng(123)

    if strategy == "vwap_rsi":
        combos = []
        for _ in range(n_combos):
            combos.append({
                "rsi_period": int(rng.choice([10, 14, 20])),
                "rsi_long_threshold": float(rng.uniform(25, 40)),
                "rsi_short_threshold": float(rng.uniform(60, 75)),
                "vwap_deviation_entry": float(rng.uniform(0.1, 0.5)),
                "volume_spike_multiplier": float(rng.uniform(1.2, 2.5)),
                "trend_adx_threshold": float(rng.uniform(20, 35)),
                "tp_percent": float(rng.uniform(0.3, 1.0)),
                "sl_percent": float(rng.uniform(0.3, 1.0)),
            })
        return combos

    if strategy == "momentum":
        combos = []
        for _ in range(n_combos):
            combos.append({
                "breakout_lookback": int(rng.choice([10, 20, 30])),
                "volume_confirmation_multiplier": float(rng.uniform(1.2, 2.5)),
                "atr_multiplier_tp": float(rng.uniform(1.5, 3.0)),
                "atr_multiplier_sl": float(rng.uniform(1.0, 2.0)),
                "tp_percent": float(rng.uniform(0.5, 2.0)),
                "sl_percent": float(rng.uniform(0.3, 1.0)),
            })
        return combos

    if strategy == "bollinger_mr":
        combos = []
        for _ in range(n_combos):
            combos.append({
                "bb_period": int(rng.choice([15, 20, 25, 30])),
                "bb_std": float(rng.choice([1.5, 2.0, 2.5, 3.0])),
                "sl_percent": float(rng.uniform(1.0, 5.0)),
            })
        return combos

    if strategy == "donchian_breakout":
        combos = []
        for _ in range(n_combos):
            combos.append({
                "entry_lookback": int(rng.choice([10, 20, 30, 40])),
                "atr_period": int(rng.choice([10, 14, 20])),
                "atr_tp_multiple": float(rng.uniform(1.5, 4.0)),
                "atr_sl_multiple": float(rng.uniform(1.0, 2.5)),
            })
        return combos

    if strategy == "supertrend":
        combos = []
        for _ in range(n_combos):
            combos.append({
                "atr_period": int(rng.choice([10, 14, 20])),
                "atr_multiplier": float(rng.choice([2.0, 2.5, 3.0, 3.5])),
                "tp_percent": float(rng.uniform(1.0, 5.0)),
                "sl_percent": float(rng.uniform(1.0, 3.0)),
            })
        return combos

    raise ValueError(f"Stratégie inconnue: {strategy}")


def _extract_grid_values(combos: list[dict]) -> dict[str, list]:
    """Extrait les valeurs uniques par paramètre (pour build_cache)."""
    grid: dict[str, set] = {}
    for c in combos:
        for k, v in c.items():
            grid.setdefault(k, set()).add(v)
    return {k: sorted(v) for k, v in grid.items()}


def _run_benchmark(
    strategy: str,
    n_candles: int,
    n_combos: int,
    n_runs: int,
) -> BenchResult:
    """Exécute le benchmark pour une stratégie."""
    # Générer données
    main_tf = "5m" if strategy in ("vwap_rsi", "momentum") else "1h"
    tf_minutes = 5 if main_tf == "5m" else 60
    candles = _make_candles(n_candles, tf_minutes)

    # Candles 15m pour le filtre (si nécessaire)
    filter_candles = _make_candles(n_candles // 3, 15) if main_tf == "5m" else []
    candles_by_tf = {main_tf: candles}
    if filter_candles:
        candles_by_tf["15m"] = filter_candles

    # Combos et grid
    combos = _make_param_combos(strategy, n_combos)
    grid_values = _extract_grid_values(combos)

    # Config
    bt_config = BacktestConfig(
        symbol="BTCUSDT",
        start_date=candles[0].timestamp,
        end_date=candles[-1].timestamp,
    )

    times = []
    for run_idx in range(n_runs):
        # Build cache (inclus dans le timing)
        t0 = time.perf_counter()
        cache = build_cache(
            candles_by_tf, grid_values, strategy,
            main_tf=main_tf,
            filter_tf="15m" if main_tf == "5m" else "1h",
        )
        # Run all combos
        for combo in combos:
            run_backtest_from_cache(strategy, combo, cache, bt_config)
        t1 = time.perf_counter()
        times.append(t1 - t0)

        label = "WARM" if run_idx == 0 else f"RUN {run_idx}"
        print(f"  [{label}] {strategy:20s} : {times[-1]:.3f}s "
              f"({n_combos} combos × {n_candles} candles)")

    return BenchResult(
        strategy=strategy,
        n_combos=n_combos,
        n_candles=n_candles,
        times=times[1:],  # Exclure le 1er run (compilation numba)
        numba=NUMBA_AVAILABLE,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark fast engine")
    parser.add_argument("--combos", type=int, default=200)
    parser.add_argument("--candles", type=int, default=5000)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--strategies", nargs="*", default=None)
    args = parser.parse_args()

    all_strategies = ["vwap_rsi", "momentum", "bollinger_mr", "donchian_breakout", "supertrend"]
    strategies = args.strategies or all_strategies

    print(f"=== Benchmark Fast Engine ===")
    print(f"Numba: {'OUI' if NUMBA_AVAILABLE else 'NON'}")
    print(f"Candles: {args.candles}, Combos: {args.combos}, Runs: {args.runs} (1er exclu)")
    print()

    results: list[BenchResult] = []
    for strat in strategies:
        print(f"--- {strat} ---")
        r = _run_benchmark(strat, args.candles, args.combos, args.runs)
        results.append(r)
        print()

    # Résumé
    print("=" * 70)
    print(f"{'Stratégie':20s} | {'Mean':>8s} | {'Std':>8s} | {'Per combo':>10s} | Numba")
    print("-" * 70)
    for r in results:
        per_combo_ms = r.mean / r.n_combos * 1000
        print(f"{r.strategy:20s} | {r.mean:7.3f}s | {r.std:7.4f}s | {per_combo_ms:8.2f}ms | "
              f"{'OUI' if r.numba else 'NON'}")
    print("=" * 70)

    total_mean = sum(r.mean for r in results)
    print(f"\nTotal (5 stratégies) : {total_mean:.3f}s "
          f"({args.combos} combos × {args.candles} candles)")


if __name__ == "__main__":
    main()
