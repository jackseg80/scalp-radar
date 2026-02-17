"""CLI pour lancer un backtest portfolio multi-asset.

Simule N assets avec capital partagé en réutilisant GridStrategyRunner
(le même code que la prod).

Usage :
    uv run python -m scripts.portfolio_backtest
    uv run python -m scripts.portfolio_backtest --days 180 --capital 5000
    uv run python -m scripts.portfolio_backtest --assets BTC/USDT,ETH/USDT,SOL/USDT
    uv run python -m scripts.portfolio_backtest --json --output portfolio.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timedelta, timezone

from loguru import logger

from backend.backtesting.portfolio_engine import (
    PortfolioBacktester,
    PortfolioResult,
    format_portfolio_report,
)
from backend.core.config import get_config
from backend.core.logging_setup import setup_logging


def _result_to_dict(result: PortfolioResult) -> dict:
    """Convertit le résultat en dict JSON-serializable."""
    d = {
        "initial_capital": result.initial_capital,
        "n_assets": result.n_assets,
        "period_days": result.period_days,
        "assets": result.assets,
        "final_equity": result.final_equity,
        "total_return_pct": result.total_return_pct,
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "realized_pnl": result.realized_pnl,
        "force_closed_pnl": result.force_closed_pnl,
        "max_drawdown_pct": result.max_drawdown_pct,
        "max_drawdown_date": (
            result.max_drawdown_date.isoformat() if result.max_drawdown_date else None
        ),
        "max_drawdown_duration_hours": result.max_drawdown_duration_hours,
        "peak_margin_ratio": result.peak_margin_ratio,
        "peak_open_positions": result.peak_open_positions,
        "peak_concurrent_assets": result.peak_concurrent_assets,
        "kill_switch_triggers": result.kill_switch_triggers,
        "kill_switch_events": result.kill_switch_events,
        "per_asset_results": result.per_asset_results,
    }
    # Equity curve résumée (pas tous les snapshots)
    d["equity_curve"] = [
        {
            "timestamp": s.timestamp.isoformat(),
            "equity": round(s.total_equity, 2),
            "margin_ratio": round(s.margin_ratio, 4),
            "positions": s.n_open_positions,
        }
        for s in result.snapshots[:: max(1, len(result.snapshots) // 500)]
    ]
    return d


def _parse_multi_strategies(raw: str) -> list[tuple[str, list[str]]]:
    """Parse le format 'strat1:sym1,sym2+strat2:sym3,sym4'."""
    result = []
    for part in raw.split("+"):
        part = part.strip()
        if ":" not in part:
            raise ValueError(f"Format invalide '{part}' — attendu 'strategy:sym1,sym2'")
        strat_name, symbols_str = part.split(":", 1)
        symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        if not symbols:
            raise ValueError(f"Aucun symbol pour la stratégie '{strat_name}'")
        result.append((strat_name.strip(), symbols))
    return result


def _resolve_preset(preset: str, config) -> list[tuple[str, list[str]]]:
    """Résout un preset en multi_strategies."""
    if preset == "combined":
        strats = []
        for sname in ["grid_atr", "grid_trend"]:
            scfg = getattr(config.strategies, sname, None)
            pa = getattr(scfg, "per_asset", {}) if scfg else {}
            if pa:
                strats.append((sname, sorted(pa.keys())))
        if not strats:
            raise ValueError("Preset 'combined' : aucun per_asset trouvé pour grid_atr/grid_trend")
        return strats
    raise ValueError(f"Preset inconnu : '{preset}' (disponibles : combined)")


async def main(args: argparse.Namespace) -> None:
    """Point d'entrée principal."""
    setup_logging(level="INFO")
    config = get_config()

    # Résoudre multi_strategies
    multi_strategies = None
    strategy_label = args.strategy

    if args.preset:
        multi_strategies = _resolve_preset(args.preset, config)
        strategy_label = args.preset
    elif args.strategies:
        multi_strategies = _parse_multi_strategies(args.strategies)
        strategy_label = "+".join(s for s, _ in multi_strategies)

    assets = args.assets.split(",") if args.assets else None

    backtester = PortfolioBacktester(
        config=config,
        initial_capital=args.capital,
        strategy_name=args.strategy,
        assets=assets,
        exchange=args.exchange,
        kill_switch_pct=args.kill_switch_pct,
        kill_switch_window_hours=args.kill_switch_window,
        multi_strategies=multi_strategies,
    )

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)

    t0 = time.monotonic()
    result = await backtester.run(start, end, db_path=args.db)
    duration = time.monotonic() - t0

    # Sauvegarder en DB si demandé
    if args.save:
        from backend.backtesting.portfolio_db import save_result_sync

        result_id = save_result_sync(
            db_path=args.db,
            result=result,
            strategy_name=strategy_label,
            exchange=args.exchange,
            kill_switch_pct=args.kill_switch_pct,
            kill_switch_window_hours=args.kill_switch_window,
            duration_seconds=round(duration, 1),
            label=args.label,
        )
        logger.info("Résultat sauvegardé en DB (id={}, {:.0f}s)", result_id, duration)

    if args.json:
        output = json.dumps(_result_to_dict(result), indent=2, ensure_ascii=False)
    else:
        output = format_portfolio_report(result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        logger.info("Résultat écrit dans {}", args.output)
    else:
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Portfolio backtest multi-asset (capital partagé)"
    )
    parser.add_argument(
        "--days", type=int, default=90, help="Période de backtest (jours)"
    )
    parser.add_argument(
        "--capital", type=float, default=10_000, help="Capital initial ($)"
    )
    parser.add_argument(
        "--strategy", type=str, default="grid_atr", help="Nom de la stratégie"
    )
    parser.add_argument(
        "--assets",
        type=str,
        default=None,
        help="Assets séparés par virgule (défaut: tous per_asset)",
    )
    parser.add_argument(
        "--exchange", type=str, default="binance", help="Source des candles"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/scalp_radar.db",
        help="Chemin de la base de données",
    )
    parser.add_argument(
        "--json", action="store_true", help="Sortie JSON au lieu de tableau"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Écrire dans un fichier"
    )
    parser.add_argument(
        "--kill-switch-pct",
        type=float,
        default=30.0,
        help="Seuil kill switch (%%)",
    )
    parser.add_argument(
        "--kill-switch-window",
        type=int,
        default=24,
        help="Fenêtre kill switch (heures)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Sauvegarder le résultat en DB",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label pour identifier le run",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Multi-stratégie : 'strat1:sym1,sym2+strat2:sym3,sym4'",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Preset multi-stratégie (ex: 'combined' = grid_atr + grid_trend per_asset)",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
