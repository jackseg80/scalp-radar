"""Diagnostic rapide - Grid Range ATR fast engine.

Usage :
  uv run python -m scripts.test_grid_range_fast --symbol BTC/USDT --days 365
  uv run python -m scripts.test_grid_range_fast --symbol BTC/USDT --days 365 --sweep
  uv run python -m scripts.test_grid_range_fast --symbol DOGE/USDT --days 365 --sweep
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone

from loguru import logger

from backend.backtesting.engine import BacktestConfig
from backend.core.config import get_config
from backend.core.database import Database
from backend.core.logging_setup import setup_logging
from backend.core.models import Candle
from backend.optimization.fast_multi_backtest import _simulate_grid_range
from backend.optimization.indicator_cache import build_cache


# ─── Data loading ─────────────────────────────────────────────────────────


async def _load_candles(
    symbol: str, start: datetime, end: datetime,
) -> dict[str, list[Candle]]:
    db = Database()
    await db.init()
    try:
        candles = await db.get_candles(
            symbol=symbol, timeframe="1h",
            start=start, end=end, limit=999_999,
        )
        logger.info("  1h : {} bougies chargées", len(candles))
        return {"1h": candles}
    finally:
        await db.close()


# ─── Metrics ──────────────────────────────────────────────────────────────


def _compute_metrics(
    trade_pnls: list[float], initial_capital: float, final_capital: float,
) -> dict[str, float]:
    n = len(trade_pnls)
    if n == 0:
        return {
            "trades": 0, "win_rate": 0.0, "net_pnl": 0.0,
            "profit_factor": 0.0, "avg_trade": 0.0, "fees_total": 0.0,
        }
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    net_pnl = final_capital - initial_capital
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    # Estimation fees : net_pnl = gross_total - fees → fees ≈ gross_total - net_pnl
    gross_total = sum(trade_pnls)
    # Mais trade_pnls est déjà net des fees → on calcule les fees autrement :
    # fees_total = gross_moves - net_moves (impossible sans données brutes)
    # Approximation : fees = abs(sum_pnls) - net_pnl si sign differ, sinon 0
    # Plus simple : on affiche juste le net
    return {
        "trades": n,
        "win_rate": len(wins) / n * 100,
        "net_pnl": net_pnl,
        "profit_factor": pf,
        "avg_trade": gross_total / n,
        "fees_total": 0.0,  # non calculable depuis trade_pnls seul
    }


# ─── Display ──────────────────────────────────────────────────────────────


def _print_single(m: dict, params: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Params : spacing={params['atr_spacing_mult']}, "
          f"levels={params['num_levels']}, sides={params['sides']}, "
          f"tp_mode={params['tp_mode']}")
    print(f"{'=' * 60}")
    print(f"  Trades     : {m['trades']}")
    print(f"  Win rate   : {m['win_rate']:.1f}%")
    print(f"  Net PnL    : {m['net_pnl']:+.2f}")
    print(f"  PF         : {m['profit_factor']:.2f}")
    print(f"  Avg trade  : {m['avg_trade']:.4f}")
    print()


def _print_sweep(rows: list[dict]) -> None:
    header = f"{'Spacing':>8} | {'Trades':>6} | {'WR':>6} | {'Net PnL':>12} | {'PF':>6} | {'Avg Trade':>10}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 1000 else "inf"
        print(
            f"{r['spacing']:>8.2f} | {r['trades']:>6} | {r['win_rate']:>5.1f}% | "
            f"{r['net_pnl']:>+12.2f} | {pf_str:>6} | {r['avg_trade']:>+10.4f}"
        )
    print(sep)
    print()


# ─── Main ─────────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> None:
    config = get_config()

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.days)

    # BacktestConfig
    leverage = args.leverage or 6
    bt_config = BacktestConfig(
        symbol=args.symbol,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        leverage=leverage,
        maker_fee=config.risk.fees.maker_percent / 100,
        taker_fee=config.risk.fees.taker_percent / 100,
        slippage_pct=config.risk.slippage.default_estimate_percent / 100,
        high_vol_slippage_mult=config.risk.slippage.high_volatility_multiplier,
        max_risk_per_trade=config.risk.position.max_risk_per_trade_percent / 100,
    )

    # Charger candles
    logger.info("Chargement {} ({} jours)...", args.symbol, args.days)
    candles_by_tf = asyncio.run(_load_candles(args.symbol, start_date, end_date))
    n_candles = len(candles_by_tf.get("1h", []))
    if n_candles < 50:
        logger.error("Pas assez de bougies 1h ({} < 50)", n_candles)
        sys.exit(1)

    # Spacings à tester
    if args.sweep:
        spacings = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    else:
        spacings = [args.spacing]

    # Params de base
    base_params = {
        "ma_period": args.ma_period,
        "atr_period": args.atr_period,
        "num_levels": args.num_levels,
        "sl_percent": args.sl_percent,
        "tp_mode": args.tp_mode,
        "sides": args.sides.split(","),
    }

    # Construire param_grid_values pour couvrir toutes les variantes
    param_grid_values = {
        "ma_period": [base_params["ma_period"]],
        "atr_period": [base_params["atr_period"]],
    }

    # Construire le cache (une seule fois)
    db_path = "data/scalp_radar.db"
    logger.info("Construction IndicatorCache...")
    cache = build_cache(
        candles_by_tf, param_grid_values, "grid_range_atr",
        main_tf="1h",
        db_path=db_path,
        symbol=args.symbol,
        exchange="bitget",
    )
    logger.info("Cache : {} bougies", cache.n_candles)

    # Run
    if args.sweep:
        rows = []
        for sp in spacings:
            params = {**base_params, "atr_spacing_mult": sp}
            # Copier bt_config pour éviter mutation
            bt = BacktestConfig(
                symbol=bt_config.symbol,
                start_date=bt_config.start_date,
                end_date=bt_config.end_date,
                initial_capital=bt_config.initial_capital,
                leverage=bt_config.leverage,
                maker_fee=bt_config.maker_fee,
                taker_fee=bt_config.taker_fee,
                slippage_pct=bt_config.slippage_pct,
                high_vol_slippage_mult=bt_config.high_vol_slippage_mult,
                max_risk_per_trade=bt_config.max_risk_per_trade,
            )
            pnls, returns, cap = _simulate_grid_range(cache, params, bt)
            m = _compute_metrics(pnls, bt.initial_capital, cap)
            rows.append({"spacing": sp, **m})

        print(f"\n  Grid Range ATR - {args.symbol} - {args.days}j")
        print(f"  Levels={base_params['num_levels']}, sides={base_params['sides']}, "
              f"tp_mode={base_params['tp_mode']}, leverage={leverage}")
        _print_sweep(rows)
    else:
        params = {**base_params, "atr_spacing_mult": args.spacing}
        pnls, returns, cap = _simulate_grid_range(cache, params, bt_config)
        m = _compute_metrics(pnls, bt_config.initial_capital, cap)
        print(f"\n  Grid Range ATR - {args.symbol} - {args.days}j")
        _print_single(m, params)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Diagnostic Grid Range ATR - fast engine direct",
    )
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--capital", type=float, default=10_000)
    parser.add_argument("--leverage", type=int, default=None)
    parser.add_argument("--spacing", type=float, default=0.3,
                        help="atr_spacing_mult (défaut: 0.3)")
    parser.add_argument("--ma-period", type=int, default=20)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--num-levels", type=int, default=2)
    parser.add_argument("--sl-percent", type=float, default=10.0)
    parser.add_argument("--tp-mode", default="dynamic_sma",
                        choices=["dynamic_sma", "fixed_center"])
    parser.add_argument("--sides", default="long,short",
                        help="Côtés actifs (défaut: long,short)")
    parser.add_argument("--sweep", action="store_true",
                        help="Tester plusieurs spacings automatiquement")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
