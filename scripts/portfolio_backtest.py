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
    TimeframeConflictError,
    format_portfolio_report,
)
from backend.core.config import get_config
from backend.core.database import Database
from backend.core.logging_setup import setup_logging


async def _detect_max_days(
    config,
    strategy_name: str,
    exchange: str,
    db_path: str,
    multi_strategies: list[tuple[str, list[str]]] | None = None,
) -> tuple[int, dict[str, int]]:
    """Détecte le nombre de jours max couverts par tous les assets.

    Scanne la DB pour trouver la première candle 1h de chaque asset.
    Retourne la couverture commune (= asset le plus récent = goulot).

    Returns:
        (common_days, per_asset_days) — jours communs et détail par asset.
    """
    # Collecter tous les assets depuis per_asset ou multi_strategies
    all_assets: set[str] = set()

    if multi_strategies:
        for _, symbols in multi_strategies:
            all_assets.update(symbols)
    else:
        strat_config = getattr(config.strategies, strategy_name, None)
        per_asset = getattr(strat_config, "per_asset", {}) if strat_config else {}
        all_assets.update(per_asset.keys())

    if not all_assets:
        return 90, {}

    db = Database(db_path)
    await db.init()

    per_asset_days: dict[str, int] = {}
    latest_start: datetime | None = None  # la date de début la plus récente

    exchanges_to_try = [exchange]
    if exchange == "binance":
        exchanges_to_try.append("bitget")
    elif exchange == "bitget":
        exchanges_to_try.append("binance")

    for symbol in sorted(all_assets):
        candles = None
        for ex in exchanges_to_try:
            candles = await db.get_candles(symbol, "1h", exchange=ex, limit=1)
            if candles:
                break

        if candles:
            first_ts = candles[0].timestamp
            days = (datetime.now(timezone.utc) - first_ts).days
            per_asset_days[symbol] = days
            if latest_start is None or first_ts > latest_start:
                latest_start = first_ts
        else:
            per_asset_days[symbol] = 0

    await db.close()

    if latest_start is None:
        return 90, per_asset_days

    common_days = (datetime.now(timezone.utc) - latest_start).days
    # Soustraire le warm-up (~50 candles 1h ≈ 2 jours)
    common_days = max(common_days - 3, 30)

    return common_days, per_asset_days


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
        # Cross-margin risk
        "was_liquidated": result.was_liquidated,
        "liquidation_event": result.liquidation_event,
        "min_liquidation_distance_pct": result.min_liquidation_distance_pct,
        "worst_case_sl_loss_pct": result.worst_case_sl_loss_pct,
        "funding_paid_total": result.funding_paid_total,
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

    # Résoudre les valeurs kill switch : CLI override > risk.yaml > fallback
    ks_cfg = getattr(config.risk, "kill_switch", None)
    ks_pct: float = (
        args.kill_switch
        if args.kill_switch is not None
        else getattr(ks_cfg, "global_max_loss_pct", 30.0)
    )
    ks_hours: int = (
        args.kill_switch_window
        if args.kill_switch_window is not None
        else int(getattr(ks_cfg, "global_window_hours", 24))
    )

    # Résoudre multi_strategies
    multi_strategies = None
    strategy_label = args.strategy

    if args.preset:
        multi_strategies = _resolve_preset(args.preset, config)
        strategy_label = args.preset
    elif args.strategies:
        multi_strategies = _parse_multi_strategies(args.strategies)
        strategy_label = "+".join(s for s, _ in multi_strategies)

    # Auto-label si --params sans --label explicite
    if getattr(args, "params", None) and not args.label:
        args.label = f"{strategy_label}_{args.params.replace(',', '_').replace('=', '')}"

    assets = args.assets.split(",") if args.assets else None

    # --regime override --leverage (leverage piloté par le signal)
    if getattr(args, "regime", False) and args.leverage is not None:
        print("  ⚠ --leverage ignoré car --regime est actif")
        args.leverage = None

    # Override leverage dans la config (sans toucher au YAML)
    if args.leverage is not None:
        # Détermine les noms de stratégies impliquées
        if multi_strategies:
            strat_names = list({s for s, _ in multi_strategies})
        else:
            strat_names = [args.strategy]
        for sname in strat_names:
            strat_cfg = getattr(config.strategies, sname, None)
            if strat_cfg is not None and hasattr(strat_cfg, "leverage"):
                strat_cfg.leverage = args.leverage
        print(f"  Leverage override   : {args.leverage}x")

    # Override params stratégie (sans toucher au YAML)
    if args.params:
        params_override: dict[str, int | float | str] = {}
        for item in args.params.split(","):
            key, val_str = item.strip().split("=", 1)
            try:
                val: int | float | str = int(val_str)
            except ValueError:
                try:
                    val = float(val_str)
                except ValueError:
                    val = val_str
            params_override[key.strip()] = val

        strat_names = list({s for s, _ in multi_strategies}) if multi_strategies else [args.strategy]
        for sname in strat_names:
            strat_cfg = getattr(config.strategies, sname, None)
            if strat_cfg is None:
                continue
            for k, v in params_override.items():
                if hasattr(strat_cfg, k):
                    setattr(strat_cfg, k, v)
                else:
                    logger.warning("Param '{}' inconnu pour {}, ignoré", k, sname)
            # Patcher aussi les per_asset (sinon per_asset écrase l'override)
            if hasattr(strat_cfg, "per_asset"):
                for asset_params in strat_cfg.per_asset.values():
                    for k, v in params_override.items():
                        if k in asset_params or hasattr(strat_cfg, k):
                            asset_params[k] = v

        print(f"  Params override     : {params_override}")

    # Résoudre --days : "auto" ou nombre
    if args.days == "auto":
        common_days, detail = await _detect_max_days(
            config,
            args.strategy,
            args.exchange,
            args.db,
            multi_strategies=multi_strategies,
        )
        if detail:
            min_days = min(detail.values()) if detail else 0
            print("\n  Auto-détection historique :")
            for sym, d in sorted(detail.items(), key=lambda x: x[1]):
                marker = " <- goulot" if d == min_days and d > 0 else ""
                if d == 0:
                    marker = " <- ABSENT"
                print(f"    {sym:15s} : {d:5d} jours{marker}")
            print(f"  → Période commune : {common_days} jours\n")
        else:
            print(f"\n  Aucun per_asset trouvé, fallback {common_days} jours\n")
        days = common_days
    else:
        days = int(args.days)

    print(f"  Kill switch         : {ks_pct:.0f}% / {ks_hours}h")

    # --- Regime signal (Sprint 50b) ---
    regime_signal = None
    if getattr(args, "regime", False):
        from backend.regime.btc_regime_signal import compute_regime_signal

        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days) if days else None
        regime_signal = await compute_regime_signal(
            db_path=args.db,
            start=start_dt,
            end=end_dt,
            exchange=args.exchange,
        )
        n_trans = len(regime_signal.transitions)
        print(f"  Regime signal       : {n_trans} transitions")
        print(f"  Normal leverage     : {args.regime_normal}x")
        print(f"  Defensive leverage  : {args.regime_defensive}x")

    backtester = PortfolioBacktester(
        config=config,
        initial_capital=args.capital,
        strategy_name=args.strategy,
        assets=assets,
        exchange=args.exchange,
        kill_switch_pct=ks_pct,
        kill_switch_window_hours=ks_hours,
        multi_strategies=multi_strategies,
        regime_signal=regime_signal,
    )

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    t0 = time.monotonic()
    try:
        result = await backtester.run(start, end, db_path=args.db)
    except TimeframeConflictError as e:
        print(f"\n  ❌  TIMEFRAME CONFLICT — portfolio backtest ANNULÉ\n")
        print(f"  {len(e.mismatched)} runner(s) incompatible(s) "
              f"(portfolio = {e.expected_tf}) :\n")
        for key, tf in e.mismatched:
            print(f"    {key} (WFO timeframe = {tf})")
        bad_strats = sorted({key.split(":", 1)[0] for key, _ in e.mismatched})
        print(f"\n  💡 Corrigez avec --force-timeframe :")
        for strat in bad_strats:
            strat_bads = sorted({
                key.split(":", 1)[1] for key, _ in e.mismatched
                if key.startswith(strat + ":")
            })
            print(f"     uv run python -m scripts.optimize --strategy {strat} "
                  f"--symbols {','.join(strat_bads)} "
                  f"--force-timeframe {e.expected_tf}")
        if e.valid_keys:
            valid_assets = sorted({
                k.split(":", 1)[1] if ":" in k else k
                for k in e.valid_keys
            })
            print(f"\n  Ou relancez sans les assets conflictuels :")
            print(f"     --assets {','.join(valid_assets)}")
        print()
        sys.exit(1)
    duration = time.monotonic() - t0

    # Sauvegarder en DB si demandé
    if args.save:
        from backend.backtesting.portfolio_db import save_result_sync

        result_id = save_result_sync(
            db_path=args.db,
            result=result,
            strategy_name=strategy_label,
            exchange=args.exchange,
            kill_switch_pct=ks_pct,
            kill_switch_window_hours=ks_hours,
            duration_seconds=round(duration, 1),
            label=args.label,
        )
        logger.info("Résultat sauvegardé en DB (id={}, {:.0f}s)", result_id, duration)

        # Push vers le serveur (best-effort, même pattern que WFO)
        try:
            import sqlite3 as _sqlite3

            from backend.backtesting.portfolio_db import push_portfolio_to_server

            _conn = _sqlite3.connect(args.db)
            _conn.row_factory = _sqlite3.Row
            _row = _conn.execute(
                "SELECT * FROM portfolio_backtests WHERE id = ?", (result_id,)
            ).fetchone()
            _conn.close()
            if _row:
                push_portfolio_to_server(dict(_row))
        except Exception as push_exc:
            logger.warning("Push portfolio serveur échoué : {}", push_exc)

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
        "--days",
        type=str,
        default="auto",
        help="Période de backtest en jours (défaut: 'auto' = max historique commun)",
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
        "--kill-switch",
        type=float,
        default=None,
        help="Override seuil kill switch %% (défaut: global_max_loss_pct depuis risk.yaml)",
    )
    parser.add_argument(
        "--kill-switch-window",
        type=int,
        default=None,
        help="Override fenêtre kill switch en heures (défaut: global_window_hours depuis risk.yaml)",
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
    parser.add_argument(
        "--leverage",
        type=int,
        default=None,
        help="Override leverage pour tous les runners (défaut: depuis strategies.yaml)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Override params stratégie (ex: 'max_hold_candles=48,sl_percent=15')",
    )

    # --- Regime BTC (Sprint 50b) ---
    parser.add_argument(
        "--regime",
        action="store_true",
        help="Leverage dynamique piloté par régime BTC (ema_atr)",
    )
    parser.add_argument(
        "--regime-normal",
        type=int,
        default=7,
        help="Leverage en mode normal (défaut: 7)",
    )
    parser.add_argument(
        "--regime-defensive",
        type=int,
        default=4,
        help="Leverage en mode defensive (défaut: 4)",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
