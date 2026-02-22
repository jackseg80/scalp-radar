"""CLI d'optimisation des paramètres de stratégies.

Lancement :
    uv run python -m scripts.optimize --check-data
    uv run python -m scripts.optimize --strategy vwap_rsi --symbol BTC/USDT
    uv run python -m scripts.optimize --strategy vwap_rsi --all-symbols
    uv run python -m scripts.optimize --all
    uv run python -m scripts.optimize --all --dry-run
    uv run python -m scripts.optimize --all --apply
    uv run python -m scripts.optimize --apply --strategy envelope_dca
    uv run python -m scripts.optimize --apply  (toutes les stratégies)
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
import sys
import threading
from collections import Counter
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
    main_exchange = "binance"
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
                "regime": wfo.window_regimes[i]["regime"] if i < len(wfo.window_regimes) else None,
                "regime_return_pct": wfo.window_regimes[i]["return_pct"] if i < len(wfo.window_regimes) else None,
                "regime_max_dd_pct": wfo.window_regimes[i]["max_dd_pct"] if i < len(wfo.window_regimes) else None,
            }
            for i, w in enumerate(wfo.windows)
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
    exchange: str | None = None,
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
        exchange=exchange,  # None = auto-détection (exchange avec le plus de candles)
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

    # Exchange pour la Phase 2 : CLI > binance par défaut
    main_exchange = exchange or "binance"

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
        observed_sharpe=wfo.avg_oos_sharpe,  # DSR teste le OOS Sharpe (pas IS)
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
    report = build_final_report(wfo, overfit, validation, regime_analysis=wfo.regime_analysis)

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
            "regime": wfo.window_regimes[i]["regime"] if i < len(wfo.window_regimes) else None,
            "regime_return_pct": wfo.window_regimes[i]["return_pct"] if i < len(wfo.window_regimes) else None,
            "regime_max_dd_pct": wfo.window_regimes[i]["max_dd_pct"] if i < len(wfo.window_regimes) else None,
        }
        for i, w in enumerate(wfo.windows)
    ]

    # Sauvegarde JSON + DB
    filepath, result_id = save_report(
        report,
        wfo_windows=windows_serialized,
        duration=None,  # TODO: tracker la durée si nécessaire
        timeframe=main_tf,
        combo_results=wfo.combo_results,  # Sprint 14b
        regime_analysis=wfo.regime_analysis,  # Sprint 15b
    )

    if progress_callback:
        progress_callback(100.0, "Terminé")

    # Affichage console
    _print_report(report, combo_results=wfo.combo_results)

    if close_db:
        await db.close()

    return report, result_id


def _print_report(
    report: FinalReport,
    combo_results: list[dict] | None = None,
) -> None:
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

    # TOP 5 COMBOS (score composite)
    if combo_results:
        from backend.optimization.walk_forward import combo_score
        max_win = max((c.get("n_windows_evaluated", 1) for c in combo_results), default=1)
        top5 = sorted(
            combo_results,
            key=lambda c: combo_score(
                c.get("oos_sharpe", 0), c.get("consistency", 0), c.get("oos_trades", 0),
                n_windows=c.get("n_windows_evaluated"), max_windows=max_win,
            ),
            reverse=True,
        )[:5]
        print(f"\n  TOP 5 COMBOS (score composite)")
        print(f"  {'-' * 40}")
        for i, c in enumerate(top5, 1):
            params = c.get("params", {})
            param_str = " ".join(f"{k}={v}" for k, v in params.items())
            consist_pct = round((c.get("consistency", 0)) * 100)
            oos_is = c.get("oos_is_ratio", 0)
            print(f"  #{i}: {param_str}")
            print(f"      OOS Sharpe: {c.get('oos_sharpe', 0):.2f} | Consist: {consist_pct}% | Trades: {c.get('oos_trades', 0)} | OOS/IS: {oos_is:.2f}")

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
    if v.funding_paid_total != 0.0:
        print(f"  Funding total        : {v.funding_paid_total:+.2f}")
    vol_str = "Oui" if v.volume_warning else "Non"
    print(f"  Volume divergence    : {vol_str}")

    print(f"\n  {'=' * 25}")
    if report.shallow:
        penalty = report.raw_score - report.total_score
        print(f"  GRADE : {report.grade}  ⚠ SHALLOW ({report.wfo_n_windows} fenêtres, raw: {report.raw_score}, pénalité: -{penalty})")
    else:
        print(f"  GRADE : {report.grade} (score: {report.total_score})")
    print(f"  LIVE ELIGIBLE : {'Oui' if report.live_eligible else 'Non'}")
    print(f"  {'=' * 25}")

    if report.warnings:
        print(f"\n  Warnings :")
        for w in report.warnings:
            print(f"    ! {w}")
    print()


def _fetch_market_specs(symbols: list[str]) -> dict[str, dict]:
    """Fetch tick_size, min_order_size, max_leverage depuis ccxt Bitget (sync).

    Un seul load_markets() pour tous les symbols.
    Retourne {} si ccxt échoue (réseau down, etc.).
    """
    try:
        import ccxt as _ccxt

        exchange = _ccxt.bitget({"options": {"defaultType": "swap"}})
        exchange.load_markets()
    except Exception as e:
        logger.warning("ccxt load_markets() échoué — skip auto-add assets.yaml : {}", e)
        return {}

    result: dict[str, dict] = {}
    for symbol in symbols:
        key = f"{symbol}:USDT"
        if key not in exchange.markets:
            logger.warning("{} non trouvé sur Bitget (clé {})", symbol, key)
            continue
        m = exchange.markets[key]
        max_lev = int(m.get("limits", {}).get("leverage", {}).get("max", 20))
        result[symbol] = {
            "tick_size": m["precision"]["price"],
            "min_order_size": m["limits"]["amount"]["min"],
            "max_leverage": min(max_lev, 20),
        }
    return result


def _params_equal(old: dict, new: dict) -> bool:
    """Compare deux dicts de paramètres avec normalisation des types.

    - Évite les faux "inchangé" quand int 25 vs float 25.0 (→ égaux)
    - Évite les faux "changé" entre "1h" et "1h" (→ égaux)
    - Détecte bien "4h" vs "1h", 20 vs 25, etc.
    - Si les clés diffèrent → différents (ex: YAML a des clés en moins/plus)
    """
    if set(old.keys()) != set(new.keys()):
        return False
    for key, v_new in new.items():
        v_old = old.get(key)
        # Normaliser int/float : 25 == 25.0 → pas de faux "changé"
        if isinstance(v_old, (int, float)) and isinstance(v_new, (int, float)):
            if float(v_old) != float(v_new):
                return False
        elif str(v_old) != str(v_new):
            return False
    return True


def apply_from_db(
    strategy_names: list[str],
    config_dir: str = "config",
    db_path: str | None = None,
    exclude_symbols: list[str] | None = None,
    ignore_tf_conflicts: bool = False,
) -> dict:
    """Lit les résultats is_latest=1 en DB et écrit per_asset dans strategies.yaml.

    - Grade A/B → best_params écrits dans per_asset
    - Grade C/D/F → retirés de per_asset
    - Champs non-optimisés préservés (enabled, leverage, weight, sides, timeframe)
    - Bloque si plusieurs timeframes détectés parmi les Grade A/B (sauf --ignore-tf-conflicts)

    Returns:
        dict avec clés: changed, applied, removed, excluded, grades, backup.
        Si conflit timeframe : blocked=True, reason="tf_conflict", tf_outliers=[...].
    """
    import json
    import shutil
    import sqlite3

    import yaml

    # Résoudre db_path depuis config
    if db_path is None:
        cfg = get_config(config_dir)
        db_url = cfg.secrets.database_url
        if db_url.startswith("sqlite:///"):
            db_path = db_url[10:]
        else:
            db_path = "data/scalp_radar.db"

    # Lire les résultats is_latest=1 pour les stratégies demandées
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in strategy_names)
        rows = conn.execute(
            f"""SELECT strategy_name, asset, timeframe, grade, total_score, best_params
                FROM optimization_results
                WHERE is_latest = 1 AND strategy_name IN ({placeholders})
                ORDER BY strategy_name, asset""",
            strategy_names,
        ).fetchall()
    finally:
        conn.close()

    # Organiser par stratégie
    by_strategy: dict[str, list[dict]] = {}
    for row in rows:
        best_params = json.loads(row["best_params"]) if row["best_params"] else {}
        tf_col = row["timeframe"] or "1h"
        # Synchroniser timeframe dans best_params avec la colonne (Hotfix 37d)
        if "timeframe" in best_params:
            best_params["timeframe"] = tf_col
        entry = {
            "asset": row["asset"],
            "grade": row["grade"],
            "total_score": row["total_score"],
            "best_params": best_params,
            "timeframe": tf_col,
        }
        by_strategy.setdefault(row["strategy_name"], []).append(entry)

    # Charger strategies.yaml
    yaml_path = Path(f"{config_dir}/strategies.yaml")
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Backup horodaté
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = yaml_path.with_name(f"{yaml_path.stem}.yaml.bak.{ts}")
    shutil.copy2(str(yaml_path), str(backup_path))

    changed = False
    all_applied: list[str] = []
    all_removed: list[str] = []
    all_excluded: list[str] = []
    all_grades: dict[str, str] = {}

    print(f"\n{'=' * 55}")
    print("  --apply : mise à jour per_asset depuis la DB")
    print(f"{'=' * 55}")

    TF_ORDER = {"1m": 0, "5m": 1, "15m": 2, "1h": 3, "4h": 4, "1d": 5}

    for strat_name in strategy_names:
        if strat_name not in data:
            print(f"\n  {strat_name} : absent de strategies.yaml, skip")
            continue

        strat_data = data[strat_name]
        old_per_asset: dict = strat_data.get("per_asset", {}) or {}
        new_per_asset: dict = {}
        results = by_strategy.get(strat_name, [])

        # 1. Déterminer les éligibles (Grade A/B, minus exclusions manuelles)
        if exclude_symbols:
            eligible = [r for r in results if r["grade"] in ("A", "B") and r["asset"] not in exclude_symbols]
        else:
            eligible = [r for r in results if r["grade"] in ("A", "B")]

        # 2. Vérifier la cohérence des timeframes AVANT d'écrire
        if eligible:
            tf_counts = Counter(r["timeframe"] for r in eligible)
            majority_tf = tf_counts.most_common(1)[0][0]
            # Tiebreak : prendre le plus petit timeframe si égalité
            if len(tf_counts) > 1 and tf_counts.most_common(2)[0][1] == tf_counts.most_common(2)[1][1]:
                majority_tf = min(tf_counts.keys(), key=lambda tf: TF_ORDER.get(tf, 99))

            outliers = [r for r in eligible if r["timeframe"] != majority_tf]

            if outliers and not ignore_tf_conflicts:
                outlier_symbols = [r["asset"] for r in outliers]
                symbols_csv = ",".join(outlier_symbols)
                print(f"\n  ❌  TIMEFRAME CONFLICT — --apply BLOQUÉ\n")
                print(f"  Timeframe majoritaire : {majority_tf} "
                      f"({tf_counts[majority_tf]}/{len(eligible)} assets A/B)\n")
                print("  Outliers :")
                for r in outliers:
                    print(f"    {r['asset']:15s} Grade {r['grade']} ({r['total_score']})  "
                          f"timeframe={r['timeframe']}")
                print(f"\n  Actions requises :")
                print(f"    1. Re-tester en {majority_tf} :")
                print(f"       uv run python -m scripts.optimize --strategy {strat_name} "
                      f"--symbols {symbols_csv} --force-timeframe {majority_tf}")
                print(f"    2. Exclure :")
                print(f"       uv run python -m scripts.optimize --strategy {strat_name} "
                      f"--apply --exclude {symbols_csv}")
                print(f"    3. Forcer (exclut les outliers silencieusement) :")
                print(f"       uv run python -m scripts.optimize --strategy {strat_name} "
                      f"--apply --ignore-tf-conflicts")
                print(f"\n  Aucune modification effectuée.\n")
                # Supprimer le backup créé inutilement
                if backup_path.exists():
                    backup_path.unlink()
                return {
                    "changed": False,
                    "blocked": True,
                    "reason": "tf_conflict",
                    "majority_tf": majority_tf,
                    "tf_outliers": outlier_symbols,
                    "applied": [],
                    "removed": [],
                    "excluded": [],
                    "grades": {},
                    "backup": None,
                    "assets_added": [],
                }

            if outliers and ignore_tf_conflicts:
                outlier_assets = {r["asset"] for r in outliers}
                eligible = [r for r in eligible if r["asset"] not in outlier_assets]
                print(f"  ℹ️  {len(outliers)} outliers timeframe exclus (--ignore-tf-conflicts)")

        added = []
        updated = []
        removed = []
        unchanged = []

        # Bug fix 2 : retirer du YAML les assets exclus sans résultat DB
        # (les assets exclus qui ONT un résultat DB sont traités plus bas
        # par le loop "Grade C/D/F" — pas de double-comptage)
        if exclude_symbols:
            result_asset_set = {r["asset"] for r in results}
            for sym in exclude_symbols:
                if sym in old_per_asset and sym not in result_asset_set:
                    del old_per_asset[sym]
                    removed.append(f"{sym} (exclu via --exclude)")
                    all_removed.append(sym)

        # 3. Écrire new_per_asset depuis les éligibles (filtrés)
        eligible_assets = {r["asset"] for r in eligible}
        for r in eligible:
            asset = r["asset"]
            all_grades[asset] = r["grade"]
            old_params = old_per_asset.get(asset, {})
            new_params = r["best_params"]
            new_per_asset[asset] = new_params

            if asset not in old_per_asset:
                added.append(f"{asset} (Grade {r['grade']}, score {r['total_score']})")
                all_applied.append(asset)
            elif not _params_equal(old_params, new_params):
                updated.append(f"{asset} (Grade {r['grade']}, score {r['total_score']})")
                all_applied.append(asset)
            else:
                unchanged.append(asset)

        # Grade C/D/F (ou exclus explicitement) → retrait implicite
        for r in results:
            asset = r["asset"]
            all_grades[asset] = r["grade"]
            if asset not in eligible_assets:
                if asset in old_per_asset:
                    if exclude_symbols and asset in exclude_symbols:
                        removed.append(f"{asset} (exclu via --exclude, Grade {r['grade']})")
                    else:
                        removed.append(f"{asset} (Grade {r['grade']})")
                    all_removed.append(asset)
                else:
                    all_excluded.append(asset)

        # Assets dans per_asset sans résultat DB → conserver
        for asset, params in old_per_asset.items():
            if asset not in eligible_assets and asset not in {r["asset"] for r in results}:
                new_per_asset[asset] = params

        strat_data["per_asset"] = new_per_asset if new_per_asset else {}
        data[strat_name] = strat_data

        # Affichage résumé
        print(f"\n  {strat_name}")
        print(f"  {'-' * 40}")
        if not results:
            print("    Aucun résultat is_latest=1 en DB")
            continue

        for r in results:
            mark = "OK" if r["asset"] in eligible_assets else "X"
            print(f"    {r['asset']:<12s} Grade {r['grade']} ({r['total_score']}) {mark}")

        if added:
            changed = True
            for a in added:
                print(f"    + AJOUTÉ    : {a}")
        if updated:
            changed = True
            for u in updated:
                print(f"    ~ MIS À JOUR: {u}")
        if removed:
            changed = True
            for r_item in removed:
                print(f"    - RETIRÉ    : {r_item}")
        if unchanged:
            print(f"    = Inchangés : {', '.join(unchanged)}")

    # Auto-add assets manquants dans assets.yaml
    assets_yaml_path = Path(f"{config_dir}/assets.yaml")
    assets_added: list[str] = []

    if all_applied and assets_yaml_path.exists():
        with open(assets_yaml_path, encoding="utf-8") as f:
            assets_data = yaml.safe_load(f) or {}
        existing_symbols = {a["symbol"] for a in assets_data.get("assets", [])}

        missing = [s for s in all_applied if s not in existing_symbols]
        if missing:
            specs = _fetch_market_specs(missing)
            for symbol in missing:
                if symbol not in specs:
                    continue
                sp = specs[symbol]
                new_entry = {
                    "symbol": symbol,
                    "exchange": "bitget",
                    "type": "futures",
                    "timeframes": ["1h"],
                    "max_leverage": sp["max_leverage"],
                    "min_order_size": sp["min_order_size"],
                    "tick_size": sp["tick_size"],
                    "correlation_group": "altcoins",
                }
                assets_data.setdefault("assets", []).append(new_entry)
                assets_added.append(symbol)
                print(f"    + AUTO-AJOUTÉ dans assets.yaml : {symbol}"
                      f" (tick={sp['tick_size']}, min_order={sp['min_order_size']}, lev={sp['max_leverage']})")

            if assets_added:
                assets_bak = assets_yaml_path.with_name(f"assets.yaml.bak.{ts}")
                shutil.copy2(str(assets_yaml_path), str(assets_bak))
                with open(assets_yaml_path, "w", encoding="utf-8") as f:
                    yaml.dump(assets_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Sauvegarder strategies.yaml
    backup_name: str | None = None
    if changed:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        backup_name = backup_path.name
        print(f"\n  strategies.yaml mis à jour (backup: {backup_name})")
    else:
        print("\n  Aucun changement détecté — strategies.yaml inchangé")
        backup_path.unlink()  # Supprimer le backup inutile

    print()
    return {
        "changed": changed,
        "applied": all_applied,
        "removed": all_removed,
        "excluded": all_excluded,
        "grades": all_grades,
        "backup": backup_name,
        "assets_added": assets_added,
    }


def _get_done_assets(strategy_name: str, db_path: str = "data/scalp_radar.db") -> set[str]:
    """Retourne les assets qui ont déjà un résultat is_latest=1 pour cette stratégie."""
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT DISTINCT asset FROM optimization_results "
            "WHERE strategy_name = ? AND is_latest = 1",
            (strategy_name,),
        )
        done = {row[0] for row in cursor.fetchall()}
        conn.close()
        return done
    except Exception:
        return set()


def _get_best_combo_trades(conn: "sqlite3.Connection", result_id: int) -> int:
    """Extrait oos_trades du best combo dans wfo_combo_results."""
    try:
        cursor = conn.execute(
            "SELECT oos_trades FROM wfo_combo_results "
            "WHERE optimization_result_id = ? AND is_best = 1 LIMIT 1",
            (result_id,),
        )
        row = cursor.fetchone()
        return row[0] if row and row[0] else 0
    except Exception:
        # Table inexistante ou erreur → fallback
        return 0


def _regrade_from_db(strategy_name: str, config_dir: str = "config") -> None:
    """Relit les résultats is_latest=1 et recalcule les grades avec la formule actuelle."""
    import json
    import sqlite3

    from backend.optimization.report import compute_grade

    config = get_config(config_dir)
    db_url = config.secrets.database_url
    db_path = db_url[10:] if db_url.startswith("sqlite:///") else "data/scalp_radar.db"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT r.id, r.asset, r.grade, r.total_score, r.n_windows,
                      r.oos_is_ratio, r.consistency, r.dsr, r.param_stability,
                      r.monte_carlo_pvalue, r.mc_underpowered,
                      r.validation_summary, r.wfo_windows
               FROM optimization_results r
               WHERE r.strategy_name = ? AND r.is_latest = 1
               ORDER BY r.asset""",
            (strategy_name,),
        ).fetchall()
    except Exception as exc:
        print(f"Erreur DB : {exc}")
        conn.close()
        return

    if not rows:
        print(f"Aucun résultat is_latest=1 pour {strategy_name}")
        conn.close()
        return

    print(f"\n{'=' * 72}")
    print(f"  REGRADE {strategy_name.upper()} ({len(rows)} assets)")
    print(f"{'=' * 72}")
    print(f"  {'Asset':<15s} {'Old':>4s} {'New':>4s} {'OldS':>5s} {'NewS':>5s} {'RawS':>5s} {'Win':>4s}  Shallow")
    print(f"  {'─' * 64}")

    changes = 0
    skipped = 0
    for row in rows:
        asset = row["asset"]

        # Vérifier métriques clés
        if row["oos_is_ratio"] is None:
            logger.warning("  {} : oos_is_ratio manquant → skip", asset)
            skipped += 1
            continue

        # Parse validation_summary JSON
        val_summary = {}
        if row["validation_summary"]:
            try:
                val_summary = json.loads(row["validation_summary"])
            except (json.JSONDecodeError, TypeError):
                pass
        bitget_trades = val_summary.get("bitget_trades", 0)
        transfer_significant = val_summary.get("transfer_significant", True)
        transfer_ratio = val_summary.get("transfer_ratio", 0.0)

        # total_trades : best combo → wfo_windows sum → 0
        best_combo_trades = _get_best_combo_trades(conn, row["id"])
        if best_combo_trades == 0:
            wfo_windows_json = row["wfo_windows"]
            if wfo_windows_json:
                try:
                    windows_data = json.loads(wfo_windows_json)
                    windows = windows_data.get("windows", []) if isinstance(windows_data, dict) else windows_data
                    best_combo_trades = sum(w.get("oos_trades", 0) for w in windows if isinstance(w, dict))
                except (json.JSONDecodeError, TypeError):
                    pass

        result = compute_grade(
            oos_is_ratio=row["oos_is_ratio"] or 0.0,
            mc_p_value=row["monte_carlo_pvalue"] if row["monte_carlo_pvalue"] is not None else 1.0,
            dsr=row["dsr"] or 0.0,
            stability=row["param_stability"] or 0.0,
            bitget_transfer=transfer_ratio or 0.0,
            mc_underpowered=bool(row["mc_underpowered"]),
            total_trades=best_combo_trades,
            consistency=row["consistency"] or 1.0,
            transfer_significant=transfer_significant,
            bitget_trades=bitget_trades or 0,
            n_windows=row["n_windows"],
        )

        old_grade = row["grade"]
        old_score = int(row["total_score"] or 0)
        changed = (result.grade != old_grade or result.score != old_score)
        penalty = result.raw_score - result.score
        shallow_str = f"-{penalty}pts" if result.is_shallow and penalty > 0 else ""
        mark = " *" if changed else ""

        print(
            f"  {asset:<15s} {old_grade:>4s} {result.grade:>4s} "
            f"{int(old_score):>5d} {result.score:>5d} {result.raw_score:>5d} "
            f"{row['n_windows'] or 0:>4d}  {shallow_str:<10s}{mark}"
        )

        if changed:
            conn.execute(
                "UPDATE optimization_results SET grade = ?, total_score = ? WHERE id = ?",
                (result.grade, result.score, row["id"]),
            )
            changes += 1

    conn.commit()
    conn.close()

    print(f"\n  {changes} grade(s) mis à jour sur {len(rows)} assets"
          f"{f' ({skipped} skippé(s))' if skipped else ''}.")
    print()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Optimisation des paramètres de stratégies")
    parser.add_argument("--strategy", type=str, help="Stratégie à optimiser (ex: vwap_rsi)")
    parser.add_argument("--symbol", type=str, help="Symbol spécifique (ex: BTC/USDT)")
    parser.add_argument("--all-symbols", action="store_true", help="Optimiser sur tous les assets")
    parser.add_argument("--all", action="store_true", help="Optimiser toutes les stratégies optimisables")
    parser.add_argument("--check-data", action="store_true", help="Vérifier les données disponibles")
    parser.add_argument("--dry-run", action="store_true", help="Afficher le plan sans exécuter")
    parser.add_argument("--apply", action="store_true",
                        help="Appliquer les paramètres grade A/B dans strategies.yaml (depuis la DB)")
    parser.add_argument("--resume", action="store_true",
                        help="Skipper les assets déjà optimisés en DB (is_latest=1) pour cette stratégie")
    parser.add_argument("-v", "--verbose", action="store_true", help="Affichage détaillé")
    parser.add_argument("--config-dir", type=str, default="config", help="Répertoire de config")
    parser.add_argument("--exchange", type=str, default=None, help="Exchange source des candles (défaut: binance)")
    parser.add_argument(
        "--force-timeframe", type=str, default=None,
        help="Forcer le timeframe WFO (ex: 1h). Override la grid.",
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Liste de symbols séparés par virgule (ex: BCH/USDT,BNB/USDT)",
    )
    parser.add_argument(
        "--exclude", type=str, default=None,
        help="Assets à exclure de --apply (CSV). Ex: BCH/USDT,BNB/USDT",
    )
    parser.add_argument(
        "--ignore-tf-conflicts", action="store_true", default=False,
        help="Forcer --apply en ignorant les outliers timeframe (les exclut silencieusement)",
    )
    parser.add_argument(
        "--regrade", action="store_true",
        help="Recalculer les grades is_latest=1 avec la formule actuelle (shallow penalty). Requiert --strategy.",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")

    # Mutex : --symbol, --symbols et --all-symbols sont exclusifs
    if sum(bool(x) for x in [args.symbol, args.symbols, args.all_symbols]) > 1:
        parser.error("Utilisez --symbol, --symbols OU --all-symbols (pas plusieurs à la fois)")

    # --regrade : recalcul des grades sans relancer le WFO
    if args.regrade:
        if not args.strategy:
            parser.error("--regrade requiert --strategy")
        if any([args.apply, args.symbol, args.all_symbols, args.resume,
                args.all, args.symbols]):
            parser.error("--regrade est incompatible avec --apply, --symbol, --symbols, --all-symbols, --all, --resume")
        _regrade_from_db(args.strategy, args.config_dir)
        return

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

    # --apply standalone (sans optimisation préalable)
    if args.apply and not args.all and not args.symbol and not args.all_symbols and not args.symbols:
        exclude_list = [s.strip() for s in args.exclude.split(",")] if args.exclude else None
        if args.strategy:
            result = apply_from_db(
                [args.strategy], args.config_dir,
                exclude_symbols=exclude_list,
                ignore_tf_conflicts=args.ignore_tf_conflicts,
            )
        else:
            result = apply_from_db(
                available_strategies, args.config_dir,
                exclude_symbols=exclude_list,
                ignore_tf_conflicts=args.ignore_tf_conflicts,
            )
        if result.get("blocked"):
            sys.exit(1)
        return

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
        elif args.symbols:
            target_symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        elif args.symbol:
            target_symbols = [args.symbol]
        else:
            logger.error("Spécifier --symbol, --symbols ou --all-symbols")
            return
    else:
        parser.print_help()
        return

    # Dry run
    if args.dry_run:
        grids = _load_param_grids(f"{args.config_dir}/param_grids.yaml")
        print("\nPlan d'optimisation (dry-run)")
        print("-" * 50)

        # --resume : calculer quels assets sont déjà faits par stratégie
        done_by_strat: dict[str, set[str]] = {}
        if args.resume:
            db_url = config.secrets.database_url
            db_path_resume = db_url[10:] if db_url.startswith("sqlite:///") else "data/scalp_radar.db"
            for strat in strategies:
                done_by_strat[strat] = _get_done_assets(strat, db_path_resume)

        total_combos = 0
        n_skipped = 0
        for strat in strategies:
            strat_grids = grids.get(strat, {})
            done = done_by_strat.get(strat, set())
            for sym in target_symbols:
                if sym in done:
                    print(f"  {strat} x {sym} : déjà en DB (skippé)")
                    n_skipped += 1
                    continue
                grid = _build_grid(strat_grids, sym)
                n = len(grid)
                total_combos += n
                coarse = min(n, 500)
                print(f"  {strat} x {sym} : {n} combos (coarse: {coarse})")
        n_total = len(strategies) * len(target_symbols)
        print(f"\n  Total : {total_combos} combinaisons ({n_total - n_skipped} assets à faire, {n_skipped} skippés)")
        print(f"  Workers : {__import__('os').cpu_count()}")
        print()
        return

    # Exécution
    all_reports: list[FinalReport] = []

    # Résoudre db_path une seule fois pour --resume
    _db_path_for_resume: str | None = None
    if args.resume:
        db_url = config.secrets.database_url
        _db_path_for_resume = db_url[10:] if db_url.startswith("sqlite:///") else "data/scalp_radar.db"

    for strat in strategies:
        # Collecter les résultats par symbole pour convergence cross-asset
        symbol_results: dict[str, dict] = {}

        # --resume : filtrer les assets déjà terminés en DB (is_latest=1)
        run_symbols = target_symbols
        if args.resume:
            done = _get_done_assets(strat, _db_path_for_resume or "data/scalp_radar.db")
            skipped = [s for s in target_symbols if s in done]
            run_symbols = [s for s in target_symbols if s not in done]
            if skipped:
                print(f"\n  --resume : {len(skipped)} assets déjà en DB pour {strat}, skippés :")
                for s in skipped:
                    print(f"    v {s}")
                print(f"  Restants : {len(run_symbols)} assets\n")
            if not run_symbols:
                print(f"  Tous les assets sont déjà terminés pour {strat}. Rien à faire.\n")
                continue

        # Construire params_override pour --force-timeframe
        force_tf_override: dict | None = None
        if args.force_timeframe:
            force_tf_override = {"timeframe": [args.force_timeframe]}
            logger.info("Timeframe forcé à [{}] pour tous les symbols", args.force_timeframe)

        for sym in run_symbols:
            logger.info("Optimisation {} × {} ...", strat, sym)
            try:
                report, result_id = await run_optimization(
                    strat, sym, args.config_dir, args.verbose,
                    all_symbols_results=symbol_results if len(symbol_results) >= 1 else None,
                    exchange=args.exchange,
                    params_override=force_tf_override,
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
        shallow_mark = " [SHALLOW]" if r.shallow else ""
        print(f"  {r.strategy_name:<12s} x {r.symbol:<12s} : Grade {r.grade}{shallow_mark} {elig}")
    print()

    # Apply (après optimisation : relire depuis la DB fraîche)
    if args.apply:
        run_strategies = list({r.strategy_name for r in all_reports})
        if run_strategies:
            exclude_list = [s.strip() for s in args.exclude.split(",")] if args.exclude else None
            result = apply_from_db(
                run_strategies, args.config_dir,
                exclude_symbols=exclude_list,
                ignore_tf_conflicts=args.ignore_tf_conflicts,
            )
            if result.get("blocked"):
                sys.exit(1)
        else:
            logger.warning("Aucun résultat à appliquer")


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
