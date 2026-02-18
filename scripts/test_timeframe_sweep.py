"""Diagnostic — Compare les performances d'une stratégie grid sur 1h, 4h, 1d.

Usage :
  uv run python -m scripts.test_timeframe_sweep --symbol BTC/USDT --days 730
  uv run python -m scripts.test_timeframe_sweep --symbol BTC/USDT --days 730 --strategy envelope_dca
  uv run python -m scripts.test_timeframe_sweep --symbol DOGE/USDT --days 365 --strategy grid_trend
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone

import yaml
from loguru import logger

from backend.backtesting.engine import BacktestConfig
from backend.core.config import get_config
from backend.core.database import Database
from backend.core.logging_setup import setup_logging
from backend.core.models import Candle
from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache
from backend.optimization.indicator_cache import build_cache, resample_candles


# ─── Stratégies supportées ────────────────────────────────────────────────

# Stratégies compatibles multi-TF (exclues : grid_multi_tf, grid_funding, grid_range_atr)
SUPPORTED_STRATEGIES = {
    "grid_atr", "envelope_dca", "envelope_dca_short", "grid_trend",
}

TIMEFRAMES = ["1h", "4h", "1d"]

# Mapping stratégie → clés param_grid_values nécessaires pour build_cache
_INDICATOR_KEYS: dict[str, list[str]] = {
    "grid_atr": ["ma_period", "atr_period"],
    "envelope_dca": ["ma_period"],
    "envelope_dca_short": ["ma_period"],
    "grid_trend": ["ema_fast", "ema_slow", "adx_period", "atr_period"],
}


# ─── Data loading ─────────────────────────────────────────────────────────


async def _load_candles(
    symbol: str, start: datetime, end: datetime,
) -> list[Candle]:
    db = Database()
    await db.init()
    try:
        candles = await db.get_candles(
            symbol=symbol, timeframe="1h",
            start=start, end=end, limit=999_999,
        )
        logger.info("  1h : {} bougies chargées", len(candles))
        return candles
    finally:
        await db.close()


# ─── Params par défaut depuis strategies.yaml ─────────────────────────────


def _load_default_params(strategy_name: str, symbol: str | None = None) -> dict:
    """Charge les params par défaut d'une stratégie depuis strategies.yaml."""
    with open("config/strategies.yaml", encoding="utf-8") as f:
        strats = yaml.safe_load(f)

    if strategy_name not in strats:
        logger.error("Stratégie '{}' introuvable dans strategies.yaml", strategy_name)
        sys.exit(1)

    cfg = strats[strategy_name]
    per_asset = cfg.get("per_asset", {})

    # Extraire seulement les params numériques/listes utiles (pas enabled, weight, etc.)
    skip_keys = {
        "enabled", "live_eligible", "timeframe", "weight", "per_asset",
        "description",
    }
    params = {k: v for k, v in cfg.items() if k not in skip_keys}

    # Override per_asset si symbol spécifié et présent
    if symbol and symbol in per_asset:
        pa = per_asset[symbol]
        params.update(pa)
        logger.info("  per_asset override pour {} : {}", symbol, pa)

    return params


def _build_param_grid_values(strategy_name: str, params: dict) -> dict[str, list]:
    """Construit le param_grid_values pour build_cache à partir des params scalaires."""
    keys = _INDICATOR_KEYS.get(strategy_name, ["ma_period", "atr_period"])
    grid = {}
    for k in keys:
        if k in params:
            grid[k] = [params[k]]
    return grid


# ─── Display ──────────────────────────────────────────────────────────────


def _print_results(
    strategy_name: str,
    symbol: str,
    days: int,
    params: dict,
    rows: list[dict],
) -> None:
    # Titre
    print(f"\n  {strategy_name} - {symbol} - {days}j (params défaut)")

    # Params compacts
    param_str = ", ".join(f"{k}={v}" for k, v in sorted(params.items())
                         if k not in ("sides", "timeframe"))
    print(f"  [{param_str}]")

    header = f"  {'TF':>4} | {'Trades':>6} | {'WR':>6} | {'Net PnL':>12} | {'PF':>6} | {'Sharpe':>7}"
    sep = "  " + "-" * (len(header) - 2)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        pf_str = f"{r['pf']:.2f}" if r["pf"] < 1000 else "inf"
        wr_str = f"{r['wr']:.1f}%" if r["trades"] > 0 else "  n/a"
        sharpe_str = f"{r['sharpe']:+.2f}" if r["trades"] > 0 else "  n/a"
        net_pnl_str = f"{r['net_pnl']:+.1f}" if r["trades"] > 0 else "0.0"
        print(
            f"  {r['tf']:>4} | {r['trades']:>6} | {wr_str:>6} | "
            f"{net_pnl_str:>12} | {pf_str:>6} | {sharpe_str:>7}"
        )
    print(sep)
    print()


# ─── Main ─────────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> None:
    config = get_config()

    strategy_name = args.strategy
    if strategy_name not in SUPPORTED_STRATEGIES:
        logger.error(
            "Stratégie '{}' non supportée. Supportées : {}",
            strategy_name, ", ".join(sorted(SUPPORTED_STRATEGIES)),
        )
        sys.exit(1)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.days)

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

    # Charger candles 1h
    logger.info("Chargement {} ({} jours)...", args.symbol, args.days)
    candles_1h = asyncio.run(_load_candles(args.symbol, start_date, end_date))
    if len(candles_1h) < 50:
        logger.error("Pas assez de bougies 1h ({} < 50)", len(candles_1h))
        sys.exit(1)

    # Params par défaut
    params = _load_default_params(strategy_name, args.symbol)
    param_grid_values = _build_param_grid_values(strategy_name, params)

    db_path = "data/scalp_radar.db"
    rows: list[dict] = []

    for tf in TIMEFRAMES:
        # Resampler
        if tf == "1h":
            candles = candles_1h
        else:
            candles = resample_candles(candles_1h, tf)

        if len(candles) < 30:
            logger.warning("  {} : seulement {} bougies — skip", tf, len(candles))
            rows.append({
                "tf": tf, "trades": 0, "wr": 0.0,
                "net_pnl": 0.0, "pf": 0.0, "sharpe": 0.0,
            })
            continue

        logger.info("  {} : {} bougies", tf, len(candles))

        # build_cache — funding rates seulement pour 1h
        cache = build_cache(
            {tf: candles},
            param_grid_values,
            strategy_name,
            main_tf=tf,
            db_path=db_path if tf == "1h" else None,
            symbol=args.symbol,
            exchange="bitget",
        )

        # Run backtest
        result = run_multi_backtest_from_cache(strategy_name, params, cache, bt_config)
        # result = (params, sharpe, net_return_pct, profit_factor, n_trades)
        _, sharpe, net_return_pct, profit_factor, n_trades = result

        net_pnl = net_return_pct / 100 * bt_config.initial_capital
        # Win rate : non disponible directement, on l'estime depuis le fast engine
        # Pour le WR, on doit relancer la simulation directement
        from backend.optimization import fast_multi_backtest as fmb
        trade_pnls: list[float] = []
        if strategy_name == "grid_atr":
            trade_pnls, _, _ = fmb._simulate_grid_atr(cache, params, bt_config, direction=1)
        elif strategy_name == "envelope_dca":
            trade_pnls, _, _ = fmb._simulate_envelope_dca(cache, params, bt_config, direction=1)
        elif strategy_name == "envelope_dca_short":
            trade_pnls, _, _ = fmb._simulate_envelope_dca(cache, params, bt_config, direction=-1)
        elif strategy_name == "grid_trend":
            trade_pnls, _, _ = fmb._simulate_grid_trend(cache, params, bt_config)

        wins = sum(1 for p in trade_pnls if p > 0)
        wr = (wins / len(trade_pnls) * 100) if trade_pnls else 0.0

        rows.append({
            "tf": tf,
            "trades": n_trades,
            "wr": wr,
            "net_pnl": net_pnl,
            "pf": profit_factor,
            "sharpe": sharpe,
        })

    _print_results(strategy_name, args.symbol, args.days, params, rows)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Diagnostic multi-timeframe — compare 1h vs 4h vs 1d",
    )
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--capital", type=float, default=10_000)
    parser.add_argument("--leverage", type=int, default=None)
    parser.add_argument("--strategy", default="grid_atr",
                        choices=sorted(SUPPORTED_STRATEGIES),
                        help="Stratégie à tester (défaut: grid_atr)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
