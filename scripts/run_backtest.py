"""CLI pour lancer un backtest Scalp Radar.

Usage :
    uv run python -m scripts.run_backtest --symbol BTC/USDT --days 90
    uv run python -m scripts.run_backtest --symbol ETH/USDT --days 30 --capital 5000 --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone

from loguru import logger

from backend.backtesting.engine import BacktestConfig, BacktestEngine
from backend.backtesting.metrics import calculate_metrics, format_metrics_table
from backend.backtesting.multi_engine import MultiPositionEngine
from backend.core.config import get_config
from backend.core.database import Database
from backend.core.logging_setup import setup_logging
from backend.core.models import Candle
from backend.optimization import GRID_STRATEGIES
from backend.strategies.factory import create_strategy


async def load_candles(
    db: Database,
    symbol: str,
    timeframes: list[str],
    start: datetime,
    end: datetime,
) -> dict[str, list[Candle]]:
    """Charge les candles depuis la DB pour chaque timeframe."""
    candles_by_tf: dict[str, list[Candle]] = {}
    for tf in timeframes:
        candles = await db.get_candles(
            symbol=symbol,
            timeframe=tf,
            start=start,
            end=end,
            limit=999_999,
        )
        candles_by_tf[tf] = candles
        logger.info("  {} : {} bougies chargées", tf, len(candles))
    return candles_by_tf


def run_backtest(args: argparse.Namespace) -> None:
    """Point d'entrée principal du backtest."""
    setup_logging()
    config = get_config()

    # Stratégie
    strategy_name = args.strategy
    try:
        strategy = create_strategy(strategy_name, config)
    except ValueError:
        logger.error("Stratégie inconnue : {}", strategy_name)
        sys.exit(1)

    # Dates
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.days)

    # Leverage : CLI > config stratégie > risk.yaml
    strat_cfg = getattr(config.strategies, strategy_name, None)
    strategy_leverage = getattr(strat_cfg, 'leverage', None) if strat_cfg else None
    if args.leverage:
        leverage = args.leverage
    elif strategy_leverage is not None:
        leverage = strategy_leverage
    else:
        leverage = config.risk.position.default_leverage

    # Config backtest
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

    # Charger les données
    logger.info("Chargement des données {} ({} jours)...", args.symbol, args.days)

    # Timeframes nécessaires (depuis la config stratégie)
    main_tf = strat_cfg.timeframe if strat_cfg else "5m"
    timeframes = [main_tf]
    if hasattr(strat_cfg, 'trend_filter_timeframe'):
        timeframes.append(strat_cfg.trend_filter_timeframe)
    timeframes = list(dict.fromkeys(timeframes))  # Dédoublonner en gardant l'ordre

    db = Database()
    candles_by_tf = asyncio.run(_load_data(db, args.symbol, timeframes, start_date, end_date))

    # Vérifier les données
    main_count = len(candles_by_tf.get(main_tf, []))
    if main_count == 0:
        logger.error(
            "Aucune donnée pour {} en {}. Lancez d'abord :\n"
            "  uv run python -m scripts.fetch_history --symbol {} --days {}",
            args.symbol, main_tf, args.symbol, args.days,
        )
        sys.exit(1)

    if main_count < strategy.min_candles.get(main_tf, 0):
        logger.warning(
            "Données insuffisantes : {} bougies (minimum recommandé : {})",
            main_count, strategy.min_candles.get(main_tf, 0),
        )

    # Lancer le backtest (moteur adapté au type de stratégie)
    if strategy_name in GRID_STRATEGIES:
        engine = MultiPositionEngine(bt_config, strategy)  # type: ignore[arg-type]
    else:
        engine = BacktestEngine(bt_config, strategy)
    result = engine.run(candles_by_tf, main_tf=main_tf)

    # Calculer les métriques
    metrics = calculate_metrics(result)

    # Afficher les résultats
    if args.json:
        output = _metrics_to_json(metrics, result)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            logger.info("Résultats JSON écrits dans {}", args.output)
        else:
            print(output)
    else:
        title = f"BACKTEST — {strategy.name.upper()} · {args.symbol} · {args.days}j"
        table = format_metrics_table(metrics, title=title)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(table)
            logger.info("Résultats écrits dans {}", args.output)
        else:
            print(table)


async def _load_data(
    db: Database,
    symbol: str,
    timeframes: list[str],
    start: datetime,
    end: datetime,
) -> dict[str, list[Candle]]:
    """Charge les données de manière async."""
    await db.init()
    try:
        return await load_candles(db, symbol, timeframes, start, end)
    finally:
        await db.close()


def _metrics_to_json(metrics, result) -> str:
    """Convertit les métriques en JSON."""
    data = {
        "strategy": result.strategy_name,
        "strategy_params": result.strategy_params,
        "symbol": result.config.symbol,
        "initial_capital": result.config.initial_capital,
        "final_capital": result.final_capital,
        "performance": {
            "total_trades": metrics.total_trades,
            "win_rate": round(metrics.win_rate, 2),
            "net_pnl": round(metrics.net_pnl, 2),
            "net_return_pct": round(metrics.net_return_pct, 2),
            "profit_factor": round(metrics.profit_factor, 2),
            "gross_profit_factor": round(metrics.gross_profit_factor, 2),
        },
        "fees": {
            "gross_pnl": round(metrics.gross_pnl, 2),
            "total_fees": round(metrics.total_fees, 2),
            "total_slippage": round(metrics.total_slippage, 2),
            "fee_drag_pct": round(metrics.fee_drag_pct, 2),
        },
        "risk": {
            "max_drawdown_pct": round(metrics.max_drawdown_pct, 2),
            "max_drawdown_duration_hours": round(
                metrics.max_drawdown_duration.total_seconds() / 3600, 1
            ),
            "sharpe_ratio": round(metrics.sharpe_ratio, 2),
            "sortino_ratio": round(metrics.sortino_ratio, 2),
        },
        "regime_stats": metrics.regime_stats,
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scalp Radar — Backtest CLI")
    parser.add_argument("--symbol", default="BTC/USDT", help="Symbole (défaut: BTC/USDT)")
    parser.add_argument("--strategy", default="vwap_rsi", help="Stratégie (défaut: vwap_rsi)")
    parser.add_argument("--days", type=int, default=90, help="Jours de données (défaut: 90)")
    parser.add_argument("--capital", type=float, default=10_000, help="Capital initial (défaut: 10000)")
    parser.add_argument("--leverage", type=int, default=None, help="Levier (défaut: depuis risk.yaml)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--output", type=str, default=None, help="Fichier de sortie")
    args = parser.parse_args()
    run_backtest(args)


if __name__ == "__main__":
    main()
