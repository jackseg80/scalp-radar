"""CLI d'optimisation des paramètres de stratégies.

Lancement :
    uv run python -m scripts.optimize --check-data
    uv run python -m scripts.optimize --strategy vwap_rsi --symbol BTC/USDT
    uv run python -m scripts.optimize --strategy vwap_rsi --all-symbols
    uv run python -m scripts.optimize --all
    uv run python -m scripts.optimize --all --dry-run
    uv run python -m scripts.optimize --all --apply
    uv run python -m scripts.optimize --strategy vwap_rsi --symbol BTC/USDT -v
"""

from __future__ import annotations

# Désactiver le JIT expérimental de Python 3.13 — cause des segfaults
# et des BrokenProcessPool sur calculs longs (bug connu du specializer).
# Doit être fait AVANT tout import lourd.
import os
os.environ.setdefault("PYTHON_JIT", "0")

import argparse
import asyncio
import itertools
import math as _math
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

from loguru import logger

from backend.core.config import get_config
from backend.core.database import Database
from backend.core.logging_setup import setup_logging
from backend.optimization import STRATEGY_REGISTRY
from backend.optimization.overfitting import OverfitDetector
from backend.optimization.report import (
    FinalReport,
    apply_to_yaml,
    build_final_report,
    save_report,
    validate_on_bitget,
)
from backend.optimization.walk_forward import WalkForwardOptimizer, _build_grid, _load_param_grids


async def check_data(config_dir: str = "config") -> None:
    """Vérifie les données disponibles pour l'optimisation."""
    config = get_config(config_dir)
    grids = _load_param_grids(f"{config_dir}/param_grids.yaml")
    opt_config = grids.get("optimization", {})
    main_exchange = opt_config.get("main_exchange", "binance")
    val_exchange = opt_config.get("validation_exchange", "bitget")

    db = Database()
    await db.init()

    symbols = [a.symbol for a in config.assets]

    print("\nVerification des donnees pour l'optimisation")
    print("-" * 50)

    for exchange in [main_exchange, val_exchange]:
        for symbol in symbols:
            # Compter les candles 5m
            candles = await db.get_candles(symbol, "5m", exchange=exchange, limit=1_000_000)
            n_candles = len(candles)
            if n_candles > 0:
                first = candles[0].timestamp
                last = candles[-1].timestamp
                days = (last - first).days
                mark = "OK"
            else:
                days = 0
                mark = "X"

            status = f"{symbol:<12s} {exchange:<8s} candles : {days:>4d} jours"
            if n_candles > 0:
                status += f" (5m: {n_candles // 1000}k)"
            status += f"  {mark}"

            if n_candles == 0:
                cmd = f"uv run python -m scripts.fetch_history --exchange {exchange} --symbol {symbol}"
                if exchange == main_exchange:
                    cmd += " --days 720"
                else:
                    cmd += " --days 90"
                status += f"  → {cmd}"

            print(f"  {status}")

    # Données funding/OI (Binance seulement)
    print(f"\n  Funding Rates & Open Interest ({main_exchange})")
    print("  " + "-" * 48)
    for symbol in symbols:
        # Funding rates
        funding = await db.get_funding_rates(symbol, exchange=main_exchange)
        n_funding = len(funding)
        if n_funding > 0:
            f_days = (funding[-1]["timestamp"] - funding[0]["timestamp"]) / 1000 / 86400
            f_mark = "OK" if f_days >= 360 else "!"
            f_status = f"{symbol:<12s} funding  : {f_days:>4.0f} jours ({n_funding} rates) {f_mark}"
        else:
            f_status = f"{symbol:<12s} funding  :    0 jours  X  -> uv run python -m scripts.fetch_funding --symbol {symbol}"
        print(f"  {f_status}")

        # Open Interest
        oi = await db.get_open_interest(symbol, timeframe="5m", exchange=main_exchange)
        n_oi = len(oi)
        if n_oi > 0:
            o_days = (oi[-1]["timestamp"] - oi[0]["timestamp"]) / 1000 / 86400
            o_mark = "OK" if o_days >= 360 else "!"
            o_status = f"{symbol:<12s} OI 5m    : {o_days:>4.0f} jours ({n_oi // 1000}k records) {o_mark}"
        else:
            o_status = f"{symbol:<12s} OI 5m    :    0 jours  X  -> uv run python -m scripts.fetch_oi --symbol {symbol}"
        print(f"  {o_status}")

    await db.close()
    print()


def _save_wfo_intermediate(wfo: "WFOResult", output_dir: str = "data/optimization") -> Path:
    """Sauvegarde intermédiaire du WFO (avant overfitting) pour ne pas perdre le travail."""
    import json as _json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    filename = f"wfo_{wfo.strategy_name}_{wfo.symbol.replace('/', '_')}_intermediate.json"
    filepath = out / filename

    data = {
        "strategy_name": wfo.strategy_name,
        "symbol": wfo.symbol,
        "avg_is_sharpe": wfo.avg_is_sharpe,
        "avg_oos_sharpe": wfo.avg_oos_sharpe,
        "oos_is_ratio": wfo.oos_is_ratio,
        "consistency_rate": wfo.consistency_rate,
        "recommended_params": wfo.recommended_params,
        "n_distinct_combos": wfo.n_distinct_combos,
        "windows": [
            {
                "window_index": w.window_index,
                "is_start": w.is_start.isoformat(),
                "is_end": w.is_end.isoformat(),
                "oos_start": w.oos_start.isoformat(),
                "oos_end": w.oos_end.isoformat(),
                "best_params": w.best_params,
                "is_sharpe": w.is_sharpe,
                "is_net_return_pct": w.is_net_return_pct,
                "is_profit_factor": w.is_profit_factor,
                "is_trades": w.is_trades,
                "oos_sharpe": w.oos_sharpe if not _math.isnan(w.oos_sharpe) else None,
                "oos_net_return_pct": w.oos_net_return_pct,
                "oos_profit_factor": w.oos_profit_factor,
                "oos_trades": w.oos_trades,
            }
            for w in wfo.windows
        ],
    }
    with open(filepath, "w", encoding="utf-8") as f:
        _json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    logger.info("WFO intermédiaire sauvé : {}", filepath)
    return filepath


async def run_optimization(
    strategy_name: str,
    symbol: str,
    config_dir: str = "config",
    verbose: bool = False,
    all_symbols_results: dict[str, dict] | None = None,
    db: Database | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_event: threading.Event | None = None,
    params_override: dict | None = None,
) -> tuple[FinalReport, int | None]:
    """Optimise une stratégie sur un asset.

    Returns:
        (report, result_id) : FinalReport + ID DB (ou None si pas sauvé)
    """
    logger.info("═" * 55)
    logger.info("  Optimisation {} × {}", strategy_name.upper(), symbol)
    logger.info("═" * 55)

    # Phase 1 : WFO
    logger.info("")
    logger.info(">>> PHASE 1/3 : WALK-FORWARD OPTIMIZATION <<<")
    logger.info("")
    optimizer = WalkForwardOptimizer(config_dir)
    wfo = await optimizer.optimize(
        strategy_name, symbol,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        params_override=params_override,
    )

    logger.info("")
    logger.info(">>> PHASE 1/3 TERMINEE — WFO : {} fenêtres, OOS/IS ratio = {:.2f} <<<", len(wfo.windows), wfo.oos_is_ratio)
    logger.info("")

    # Sauvegarde intermédiaire (ne plus jamais perdre 55 min de WFO)
    _save_wfo_intermediate(wfo)

    if verbose:
        for w in wfo.windows:
            logger.info(
                "  Window {} : IS Sharpe={:.2f}, OOS Sharpe={:.2f}",
                w.window_index, w.is_sharpe, w.oos_sharpe,
            )

    # Phase 2 : Overfitting
    logger.info("")
    logger.info(">>> PHASE 2/3 : DETECTION OVERFITTING <<<")
    logger.info("")
    detector = OverfitDetector()

    # Pour la stabilité, utiliser la dernière fenêtre IS (pas tout le dataset)
    grids = _load_param_grids(f"{config_dir}/param_grids.yaml")
    opt_config = grids.get("optimization", {})
    main_exchange = opt_config.get("main_exchange", "binance")
    is_window_days = opt_config.get("is_window_days", 120)

    close_db = False
    if db is None:
        db = Database()
        await db.init()
        close_db = True

    from backend.optimization import STRATEGY_REGISTRY as reg
    config_cls, _ = reg[strategy_name]
    default_cfg = config_cls()
    main_tf = default_cfg.timeframe
    tfs = [main_tf]
    if hasattr(default_cfg, "trend_filter_timeframe"):
        tfs.append(default_cfg.trend_filter_timeframe)

    # Charger seulement les IS_WINDOW_DAYS derniers jours pour la stabilité
    # (34k bougies au lieu de 207k → ~6x plus rapide)
    from datetime import timedelta

    all_candles_by_tf: dict = {}
    for tf in tfs:
        all_candles_by_tf[tf] = await db.get_candles(
            symbol, tf, exchange=main_exchange, limit=1_000_000,
        )

    # Découper aux derniers is_window_days pour la stabilité
    stability_candles_by_tf: dict = {}
    from backend.optimization.walk_forward import _slice_candles
    if all_candles_by_tf.get(main_tf):
        last_candle = all_candles_by_tf[main_tf][-1]
        stability_start = last_candle.timestamp - timedelta(days=is_window_days)
        for tf in tfs:
            stability_candles_by_tf[tf] = _slice_candles(
                all_candles_by_tf[tf], stability_start, last_candle.timestamp,
            )
        logger.info(
            "Stabilité : {} bougies {} (derniers {} jours)",
            len(stability_candles_by_tf.get(main_tf, [])), main_tf, is_window_days,
        )
    else:
        stability_candles_by_tf = all_candles_by_tf

    from backend.backtesting.engine import BacktestConfig
    stab_candles = stability_candles_by_tf.get(main_tf, [])
    if stab_candles:
        bt_config = BacktestConfig(
            symbol=symbol,
            start_date=stab_candles[0].timestamp,
            end_date=stab_candles[-1].timestamp,
        )
    else:
        bt_config = BacktestConfig(
            symbol=symbol,
            start_date=datetime.now(),
            end_date=datetime.now(),
        )
    # Override leverage si la stratégie en spécifie un (ex: envelope_dca=6)
    if hasattr(default_cfg, 'leverage'):
        bt_config.leverage = default_cfg.leverage

    # Charger extra_data (funding/OI) si nécessaire pour la stabilité
    from backend.optimization import STRATEGIES_NEED_EXTRA_DATA
    stability_extra_data = None
    if strategy_name in STRATEGIES_NEED_EXTRA_DATA and stab_candles:
        from backend.backtesting.extra_data_builder import build_extra_data_map
        funding_rates = await db.get_funding_rates(symbol, exchange=main_exchange)
        oi_records = await db.get_open_interest(symbol, timeframe="5m", exchange=main_exchange)
        if funding_rates or oi_records:
            stability_extra_data = build_extra_data_map(
                stab_candles, funding_rates, oi_records,
            )

    overfit = detector.full_analysis(
        trades=wfo.all_oos_trades,
        observed_sharpe=wfo.avg_is_sharpe,  # DSR teste le IS Sharpe (pas OOS)
        n_distinct_combos=wfo.n_distinct_combos,
        strategy_name=strategy_name,
        symbol=symbol,
        optimal_params=wfo.recommended_params,
        candles_by_tf=stability_candles_by_tf,
        bt_config=bt_config,
        all_symbols_results=all_symbols_results,
        extra_data_by_timestamp=stability_extra_data,
        main_tf=main_tf,
    )

    mc_suffix = " (underpowered)" if overfit.monte_carlo.underpowered else ""
    logger.info(
        "Overfitting : MC p={:.3f}{}, DSR={:.2f}, Stabilité={:.2f}",
        overfit.monte_carlo.p_value, mc_suffix, overfit.dsr.dsr,
        overfit.stability.overall_stability,
    )

    if progress_callback:
        progress_callback(90.0, "Overfitting detection terminée")

    # Phase 3 : Validation Bitget
    logger.info("")
    logger.info(">>> PHASE 3/3 : VALIDATION BITGET <<<")
    logger.info("")
    validation = await validate_on_bitget(
        strategy_name, symbol, wfo.recommended_params,
        wfo.avg_oos_sharpe, db=db,
    )

    logger.info(
        "Bitget : Sharpe={:.2f} [CI: {:.2f} — {:.2f}], transfer={:.2f}",
        validation.bitget_sharpe,
        validation.bitget_sharpe_ci_low, validation.bitget_sharpe_ci_high,
        validation.transfer_ratio,
    )

    if progress_callback:
        progress_callback(95.0, "Validation Bitget terminée")

    # Build report
    report = build_final_report(wfo, overfit, validation)

    # Sérialiser les windows pour la DB
    windows_serialized = [
        {
            "window_index": w.window_index,
            "is_start": w.is_start.isoformat(),
            "is_end": w.is_end.isoformat(),
            "oos_start": w.oos_start.isoformat(),
            "oos_end": w.oos_end.isoformat(),
            "best_params": w.best_params,
            "is_sharpe": w.is_sharpe,
            "is_net_return_pct": w.is_net_return_pct,
            "is_profit_factor": w.is_profit_factor,
            "is_trades": w.is_trades,
            "oos_sharpe": w.oos_sharpe if not _math.isnan(w.oos_sharpe) else None,
            "oos_net_return_pct": w.oos_net_return_pct,
            "oos_profit_factor": w.oos_profit_factor,
            "oos_trades": w.oos_trades,
        }
        for w in wfo.windows
    ]

    # Sauvegarde JSON + DB
    filepath, result_id = save_report(
        report,
        wfo_windows=windows_serialized,
        duration=None,  # TODO: tracker la durée si nécessaire
        timeframe=main_tf,
    )

    if progress_callback:
        progress_callback(100.0, "Terminé")

    # Affichage console
    _print_report(report)

    if close_db:
        await db.close()

    return report, result_id


def _print_report(report: FinalReport) -> None:
    """Affichage console du rapport."""
    print(f"\n  {'=' * 55}")
    print(f"  Optimisation {report.strategy_name.upper()} x {report.symbol}")
    print(f"  {'=' * 55}")

    print(f"\n  Walk-Forward ({report.wfo_n_windows} fenetres)")
    print(f"  {'-' * 40}")
    print(f"  IS Sharpe moyen     : {report.wfo_avg_is_sharpe:.2f}")
    print(f"  OOS Sharpe moyen    : {report.wfo_avg_oos_sharpe:.2f}")
    print(f"  OOS/IS ratio        : {report.oos_is_ratio:.2f}")
    print(f"  Consistance OOS+    : {report.wfo_consistency_rate:.0%}")
    print(f"  Combinaisons testees: {report.n_distinct_combos}")

    print(f"\n  Parametres recommandes")
    print(f"  {'-' * 40}")
    for k, v in sorted(report.recommended_params.items()):
        print(f"  {k:<25s}: {v}")

    print(f"\n  Detection d'overfitting")
    print(f"  {'-' * 40}")
    if report.mc_underpowered:
        mc_mark = "!"
        mc_extra = " (underpowered, score neutre 12/25)"
    elif report.mc_significant:
        mc_mark = "OK"
        mc_extra = ""
    else:
        mc_mark = "X"
        mc_extra = ""
    print(f"  Monte Carlo p-value  : {report.mc_p_value:.3f} {mc_mark}{mc_extra}")
    dsr_mark = "OK" if report.dsr > 0.95 else ("~" if report.dsr > 0.80 else "X")
    print(f"  DSR (n={report.n_distinct_combos}){'':>10s}: {report.dsr:.2f} {dsr_mark}")
    stab_mark = "OK" if report.stability > 0.80 else ("~" if report.stability > 0.60 else "X")
    print(f"  Stabilite parametres : {report.stability:.2f} {stab_mark}")
    if report.convergence is not None:
        conv_mark = "OK" if report.convergence > 0.70 else "X"
        print(f"  Convergence cross    : {report.convergence:.2f} {conv_mark}")

    print(f"\n  Validation Bitget")
    print(f"  {'-' * 40}")
    v = report.validation
    print(f"  Sharpe Bitget        : {v.bitget_sharpe:.2f} [CI 95%: {v.bitget_sharpe_ci_low:.2f} - {v.bitget_sharpe_ci_high:.2f}]")
    transfer_mark = "OK" if v.transfer_ratio > 0.50 else "X"
    print(f"  Transfer ratio       : {v.transfer_ratio:.2f} {transfer_mark}")
    vol_str = "Oui" if v.volume_warning else "Non"
    print(f"  Volume divergence    : {vol_str}")

    print(f"\n  {'=' * 25}")
    print(f"  GRADE : {report.grade}")
    print(f"  LIVE ELIGIBLE : {'Oui' if report.live_eligible else 'Non'}")
    print(f"  {'=' * 25}")

    if report.warnings:
        print(f"\n  Warnings :")
        for w in report.warnings:
            print(f"    ! {w}")
    print()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Optimisation des paramètres de stratégies")
    parser.add_argument("--strategy", type=str, help="Stratégie à optimiser (ex: vwap_rsi)")
    parser.add_argument("--symbol", type=str, help="Symbol spécifique (ex: BTC/USDT)")
    parser.add_argument("--all-symbols", action="store_true", help="Optimiser sur tous les assets")
    parser.add_argument("--all", action="store_true", help="Optimiser toutes les stratégies optimisables")
    parser.add_argument("--check-data", action="store_true", help="Vérifier les données disponibles")
    parser.add_argument("--dry-run", action="store_true", help="Afficher le plan sans exécuter")
    parser.add_argument("--apply", action="store_true", help="Appliquer les paramètres grade A/B")
    parser.add_argument("-v", "--verbose", action="store_true", help="Affichage détaillé")
    parser.add_argument("--config-dir", type=str, default="config", help="Répertoire de config")
    args = parser.parse_args()

    setup_logging(level="INFO")

    if args.check_data:
        await check_data(args.config_dir)
        return

    config = get_config(args.config_dir)
    symbols = [a.symbol for a in config.assets]
    available_strategies = list(STRATEGY_REGISTRY.keys())
    excluded = [
        name for name in ["orderflow"]
        if name not in STRATEGY_REGISTRY
    ]

    # Déterminer quoi optimiser
    if args.all:
        strategies = available_strategies
        target_symbols = symbols
        if excluded:
            logger.info(
                "Stratégies exclues (pas de données historiques) : {}",
                ", ".join(excluded),
            )
    elif args.strategy:
        if args.strategy not in STRATEGY_REGISTRY:
            logger.error(
                "Stratégie '{}' non optimisable. Disponibles : {}",
                args.strategy, available_strategies,
            )
            return
        strategies = [args.strategy]
        if args.all_symbols:
            target_symbols = symbols
        elif args.symbol:
            target_symbols = [args.symbol]
        else:
            logger.error("Spécifier --symbol ou --all-symbols")
            return
    else:
        parser.print_help()
        return

    # Dry run
    if args.dry_run:
        grids = _load_param_grids(f"{args.config_dir}/param_grids.yaml")
        print("\nPlan d'optimisation (dry-run)")
        print("-" * 50)
        total_combos = 0
        for strat in strategies:
            strat_grids = grids.get(strat, {})
            for sym in target_symbols:
                grid = _build_grid(strat_grids, sym)
                n = len(grid)
                total_combos += n
                coarse = min(n, 500)
                print(f"  {strat} x {sym} : {n} combos (coarse: {coarse})")
        print(f"\n  Total : {total_combos} combinaisons")
        print(f"  Workers : {__import__('os').cpu_count()}")
        print()
        return

    # Exécution
    all_reports: list[FinalReport] = []

    for strat in strategies:
        # Collecter les résultats par symbole pour convergence cross-asset
        symbol_results: dict[str, dict] = {}

        for sym in target_symbols:
            logger.info("Optimisation {} × {} ...", strat, sym)
            try:
                report, result_id = await run_optimization(
                    strat, sym, args.config_dir, args.verbose,
                    all_symbols_results=symbol_results if len(symbol_results) >= 1 else None,
                )
                all_reports.append(report)
                symbol_results[sym] = report.recommended_params
            except Exception as exc:
                logger.error(
                    "ERREUR {} × {} : {} — on continue avec les suivants",
                    strat, sym, exc,
                )
                import traceback
                traceback.print_exc()

    # Recapitulatif
    print(f"\n{'=' * 55}")
    print("  Recapitulatif")
    print(f"{'=' * 55}")
    for r in all_reports:
        elig = "OK LIVE" if r.live_eligible else "X"
        print(f"  {r.strategy_name:<12s} x {r.symbol:<12s} : Grade {r.grade} {elig}")
    print()

    # Apply
    if args.apply:
        applied = apply_to_yaml(all_reports, f"{args.config_dir}/strategies.yaml")
        if applied:
            logger.info("Paramètres appliqués dans strategies.yaml")
        else:
            logger.warning("Aucun paramètre appliqué (pas de grade A/B)")


if __name__ == "__main__":
    import sys
    import traceback

    def _unhandled_exception(exc_type, exc_value, exc_tb):
        """Attrape les exceptions non gérées pour éviter les crashs silencieux."""
        logger.error("Exception non gérée: {}", exc_value)
        traceback.print_exception(exc_type, exc_value, exc_tb)
        sys.exit(1)

    sys.excepthook = _unhandled_exception

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Optimisation interrompue par l'utilisateur")
    except Exception as exc:
        logger.error("Erreur fatale: {}", exc)
        traceback.print_exc()
        sys.exit(1)
