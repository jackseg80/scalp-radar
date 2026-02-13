"""Rapport de confiance et validation Bitget pour Scalp Radar.

Grading A-F, validation croisée Binance→Bitget avec bootstrap CI,
application des paramètres dans strategies.yaml (per_asset).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger

from backend.backtesting.engine import BacktestConfig, run_backtest_single
from backend.backtesting.extra_data_builder import build_extra_data_map
from backend.backtesting.metrics import calculate_metrics
from backend.core.database import Database
from backend.core.models import Candle
from backend.core.position_manager import TradeResult
from backend.optimization.overfitting import OverfitReport
from backend.optimization.walk_forward import WFOResult


# ─── Dataclasses ───────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    """Résultat de la validation Bitget 90j."""

    bitget_sharpe: float
    bitget_net_return_pct: float
    bitget_trades: int
    bitget_sharpe_ci_low: float
    bitget_sharpe_ci_high: float
    binance_oos_avg_sharpe: float
    transfer_ratio: float
    transfer_significant: bool
    volume_warning: bool
    volume_warning_detail: str


@dataclass
class FinalReport:
    """Rapport complet d'une optimisation stratégie × asset."""

    strategy_name: str
    symbol: str
    timestamp: datetime
    grade: str

    # WFO
    wfo_avg_is_sharpe: float
    wfo_avg_oos_sharpe: float
    wfo_consistency_rate: float
    wfo_n_windows: int
    recommended_params: dict[str, Any]

    # Overfitting
    mc_p_value: float
    mc_significant: bool
    mc_underpowered: bool
    dsr: float
    dsr_max_expected_sharpe: float
    stability: float
    cliff_params: list[str]
    convergence: float | None
    divergent_params: list[str]

    # Validation
    validation: ValidationResult

    # Métriques de décision
    oos_is_ratio: float
    bitget_transfer: float

    # Décision
    live_eligible: bool
    warnings: list[str]
    n_distinct_combos: int


# ─── Grading ───────────────────────────────────────────────────────────────


def compute_grade(
    oos_is_ratio: float,
    mc_p_value: float,
    dsr: float,
    stability: float,
    bitget_transfer: float,
    mc_underpowered: bool = False,
) -> str:
    """Calcule le grade A-F selon les critères."""
    score = 0

    # OOS/IS ratio (max 25 points)
    if oos_is_ratio > 0.6:
        score += 25
    elif oos_is_ratio > 0.5:
        score += 20
    elif oos_is_ratio > 0.4:
        score += 15
    elif oos_is_ratio > 0.3:
        score += 10

    # Monte Carlo (max 25 points)
    if mc_underpowered:
        # Pas assez de trades pour un test MC fiable → score neutre (12/25)
        score += 12
    elif mc_p_value < 0.05:
        score += 25
    elif mc_p_value < 0.10:
        score += 15

    # DSR (max 20 points)
    if dsr > 0.95:
        score += 20
    elif dsr > 0.90:
        score += 15
    elif dsr > 0.80:
        score += 10

    # Stability (max 15 points)
    if stability > 0.80:
        score += 15
    elif stability > 0.60:
        score += 10
    elif stability > 0.40:
        score += 5

    # Bitget transfer (max 15 points)
    if bitget_transfer > 0.50:
        score += 15
    elif bitget_transfer > 0.30:
        score += 8

    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


# ─── Validation Bitget ─────────────────────────────────────────────────────


async def validate_on_bitget(
    strategy_name: str,
    symbol: str,
    recommended_params: dict[str, Any],
    binance_oos_avg_sharpe: float,
    db: Database | None = None,
    n_bootstrap: int = 1000,
    seed: int | None = 42,
) -> ValidationResult:
    """Backtest les paramètres optimaux sur Bitget 90j avec bootstrap CI."""
    close_db = False
    if db is None:
        db = Database()
        await db.init()
        close_db = True

    try:
        # Timeframes nécessaires
        from backend.optimization import STRATEGY_REGISTRY
        config_cls, _ = STRATEGY_REGISTRY[strategy_name]
        default_cfg = config_cls()
        main_tf = default_cfg.timeframe
        tfs = [main_tf]
        if hasattr(default_cfg, "trend_filter_timeframe"):
            tfs.append(default_cfg.trend_filter_timeframe)

        candles_by_tf: dict[str, list[Candle]] = {}
        for tf in tfs:
            candles = await db.get_candles(
                symbol, tf, exchange="bitget", limit=1_000_000,
            )
            candles_by_tf[tf] = candles

        if not candles_by_tf.get(main_tf):
            return ValidationResult(
                bitget_sharpe=0.0, bitget_net_return_pct=0.0, bitget_trades=0,
                bitget_sharpe_ci_low=0.0, bitget_sharpe_ci_high=0.0,
                binance_oos_avg_sharpe=binance_oos_avg_sharpe,
                transfer_ratio=0.0, transfer_significant=False,
                volume_warning=True,
                volume_warning_detail="Pas de données Bitget disponibles",
            )

        main_candles = candles_by_tf[main_tf]
        bt_config = BacktestConfig(
            symbol=symbol,
            start_date=main_candles[0].timestamp,
            end_date=main_candles[-1].timestamp,
        )
        # Override leverage si la stratégie en spécifie un (ex: envelope_dca=6)
        if hasattr(default_cfg, 'leverage'):
            bt_config.leverage = default_cfg.leverage

        # Charger extra_data si nécessaire (funding/OI Binance comme proxy)
        from backend.optimization import STRATEGIES_NEED_EXTRA_DATA, is_grid_strategy
        extra_data_map: dict[str, dict[str, Any]] | None = None
        if strategy_name in STRATEGIES_NEED_EXTRA_DATA:
            funding_rates = await db.get_funding_rates(symbol, exchange="binance")
            oi_records = await db.get_open_interest(symbol, timeframe="5m", exchange="binance")
            if funding_rates or oi_records:
                extra_data_map = build_extra_data_map(
                    main_candles, funding_rates, oi_records,
                )

        # Backtest (moteur adapté au type de stratégie)
        if is_grid_strategy(strategy_name):
            from backend.backtesting.multi_engine import run_multi_backtest_single
            result = run_multi_backtest_single(
                strategy_name, recommended_params, candles_by_tf, bt_config, main_tf,
            )
        else:
            result = run_backtest_single(
                strategy_name, recommended_params, candles_by_tf, bt_config, main_tf,
                extra_data_by_timestamp=extra_data_map,
            )
        metrics = calculate_metrics(result)

        # Bootstrap CI sur le Sharpe Bitget
        ci_low, ci_high = _bootstrap_sharpe_ci(
            result.trades, n_bootstrap=n_bootstrap, seed=seed
        )

        # Transfer ratio
        transfer_ratio = (
            metrics.sharpe_ratio / binance_oos_avg_sharpe
            if binance_oos_avg_sharpe > 0 else 0.0
        )
        transfer_significant = ci_low > 0

        # Volume warning
        volume_warning, volume_detail = _check_volume_divergence(
            candles_by_tf.get(main_tf, []), symbol,
        )

        return ValidationResult(
            bitget_sharpe=metrics.sharpe_ratio,
            bitget_net_return_pct=metrics.net_return_pct,
            bitget_trades=metrics.total_trades,
            bitget_sharpe_ci_low=ci_low,
            bitget_sharpe_ci_high=ci_high,
            binance_oos_avg_sharpe=binance_oos_avg_sharpe,
            transfer_ratio=transfer_ratio,
            transfer_significant=transfer_significant,
            volume_warning=volume_warning,
            volume_warning_detail=volume_detail,
        )
    finally:
        if close_db:
            await db.close()


def _bootstrap_sharpe_ci(
    trades: list[TradeResult],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = 42,
) -> tuple[float, float]:
    """Bootstrap CI sur le Sharpe ratio des trades."""
    if len(trades) < 5:
        return (0.0, 0.0)

    returns = []
    capital = 10_000.0
    for trade in trades:
        if capital > 0:
            returns.append(trade.net_pnl / capital)
        capital += trade.net_pnl

    returns_arr = np.array(returns)
    rng = np.random.default_rng(seed)
    bootstrap_sharpes = []

    for _ in range(n_bootstrap):
        sample = rng.choice(returns_arr, size=len(returns_arr), replace=True)
        std = np.std(sample)
        if std > 0:
            bootstrap_sharpes.append(float(np.mean(sample) / std))
        else:
            bootstrap_sharpes.append(0.0)

    alpha = (1 - confidence) / 2
    ci_low = float(np.percentile(bootstrap_sharpes, alpha * 100))
    ci_high = float(np.percentile(bootstrap_sharpes, (1 - alpha) * 100))
    return (ci_low, ci_high)


def _check_volume_divergence(
    bitget_candles: list[Candle],
    symbol: str,
) -> tuple[bool, str]:
    """Vérifie si le profil de volume Bitget diverge significativement."""
    if len(bitget_candles) < 100:
        return (True, f"Trop peu de candles Bitget ({len(bitget_candles)}) pour évaluer le volume")

    volumes = np.array([c.volume for c in bitget_candles])
    mean_vol = np.mean(volumes)
    if mean_vol == 0:
        return (True, "Volume moyen nul sur Bitget")

    # Vérifier la stabilité du volume (coefficient de variation)
    cv = float(np.std(volumes) / mean_vol)
    if cv > 3.0:
        return (
            True,
            f"Volume très instable sur Bitget (CV={cv:.1f}). "
            "Le volume_multiplier pourrait nécessiter un ajustement manuel."
        )

    return (False, "")


# ─── Sauvegarde JSON ──────────────────────────────────────────────────────


def save_report(report: FinalReport, output_dir: str = "data/optimization") -> Path:
    """Sauvegarde le rapport en JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = report.timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"{report.strategy_name}_{report.symbol.replace('/', '_')}_{ts}.json"
    filepath = out / filename

    # Convertir en dict sérialisable
    data = _report_to_dict(report)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)

    logger.info("Rapport sauvé : {}", filepath)
    return filepath


def _report_to_dict(report: FinalReport) -> dict[str, Any]:
    """Convertit un FinalReport en dict JSON-sérialisable."""
    validation = asdict(report.validation)
    return {
        "strategy_name": report.strategy_name,
        "symbol": report.symbol,
        "timestamp": report.timestamp.isoformat(),
        "grade": report.grade,
        "live_eligible": report.live_eligible,
        "recommended_params": report.recommended_params,
        "wfo": {
            "avg_is_sharpe": report.wfo_avg_is_sharpe,
            "avg_oos_sharpe": report.wfo_avg_oos_sharpe,
            "consistency_rate": report.wfo_consistency_rate,
            "n_windows": report.wfo_n_windows,
            "oos_is_ratio": report.oos_is_ratio,
        },
        "overfitting": {
            "mc_p_value": report.mc_p_value,
            "mc_significant": report.mc_significant,
            "mc_underpowered": report.mc_underpowered,
            "dsr": report.dsr,
            "dsr_max_expected_sharpe": report.dsr_max_expected_sharpe,
            "stability": report.stability,
            "cliff_params": report.cliff_params,
            "convergence": report.convergence,
            "divergent_params": report.divergent_params,
            "n_distinct_combos": report.n_distinct_combos,
        },
        "validation": validation,
        "warnings": report.warnings,
    }


# ─── Application YAML ──────────────────────────────────────────────────────


def apply_to_yaml(
    reports: list[FinalReport],
    strategies_yaml_path: str = "config/strategies.yaml",
) -> bool:
    """Écrit les paramètres grade A/B dans strategies.yaml avec per_asset.

    Crée un backup horodaté avant écriture.
    Retourne True si au moins un asset appliqué.
    """
    eligible = [r for r in reports if r.grade in ("A", "B")]
    if not eligible:
        logger.warning("Aucun rapport grade A/B — paramètres NON appliqués")
        return False

    # Backup horodaté
    yaml_path = Path(strategies_yaml_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = yaml_path.with_name(f"{yaml_path.stem}.yaml.bak.{ts}")
    shutil.copy2(str(yaml_path), str(backup_path))
    logger.info("Backup strategies.yaml : {}", backup_path)

    # Charger le YAML actuel
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Grouper par stratégie
    by_strategy: dict[str, list[FinalReport]] = {}
    for report in eligible:
        by_strategy.setdefault(report.strategy_name, []).append(report)

    for strat_name, strat_reports in by_strategy.items():
        if strat_name not in data:
            logger.warning("Stratégie '{}' absente du YAML, skip", strat_name)
            continue

        strat_data = data[strat_name]

        # Déterminer les paramètres convergents vs divergents
        if len(strat_reports) >= 2:
            # Chercher convergence dans le premier report qui en a
            convergent_params: set[str] = set()
            divergent_params: set[str] = set()
            for r in strat_reports:
                divergent_params.update(r.divergent_params)
            all_param_keys = set()
            for r in strat_reports:
                all_param_keys.update(r.recommended_params.keys())
            convergent_params = all_param_keys - divergent_params
        else:
            convergent_params = set(strat_reports[0].recommended_params.keys())
            divergent_params = set()

        # Paramètres convergents → médiane cross-asset comme default
        if len(strat_reports) >= 2 and convergent_params:
            for param_name in convergent_params:
                values = [
                    r.recommended_params[param_name]
                    for r in strat_reports
                    if param_name in r.recommended_params
                    and isinstance(r.recommended_params[param_name], (int, float))
                ]
                if values:
                    median_val = float(np.median(values))
                    if all(isinstance(v, int) for v in values):
                        median_val = int(round(median_val))
                    old_val = strat_data.get(param_name)
                    strat_data[param_name] = median_val
                    logger.info(
                        "  {} {} (default) : {} → {}",
                        strat_name, param_name, old_val, median_val,
                    )

        # Paramètres divergents → per_asset
        per_asset = strat_data.get("per_asset", {})
        for report in strat_reports:
            symbol = report.symbol
            asset_overrides = per_asset.get(symbol, {})

            for param_name in report.recommended_params:
                if param_name in divergent_params or len(strat_reports) == 1:
                    old_val = asset_overrides.get(param_name)
                    new_val = report.recommended_params[param_name]
                    asset_overrides[param_name] = new_val
                    if old_val != new_val:
                        logger.info(
                            "  {} {} per_asset[{}] : {} → {}",
                            strat_name, param_name, symbol, old_val, new_val,
                        )

            if asset_overrides:
                per_asset[symbol] = asset_overrides

        strat_data["per_asset"] = per_asset
        data[strat_name] = strat_data

    # Sauvegarder
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info(
        "Paramètres appliqués pour {} asset(s) grade A/B", len(eligible),
    )
    return True


# ─── Build FinalReport ─────────────────────────────────────────────────────


def build_final_report(
    wfo: WFOResult,
    overfit: OverfitReport,
    validation: ValidationResult,
) -> FinalReport:
    """Construit le rapport final à partir des résultats WFO, overfitting, validation."""
    convergence_score = (
        overfit.convergence.convergence_score if overfit.convergence else None
    )
    divergent = (
        overfit.convergence.divergent_params if overfit.convergence else []
    )

    grade = compute_grade(
        oos_is_ratio=wfo.oos_is_ratio,
        mc_p_value=overfit.monte_carlo.p_value,
        dsr=overfit.dsr.dsr,
        stability=overfit.stability.overall_stability,
        bitget_transfer=validation.transfer_ratio,
        mc_underpowered=overfit.monte_carlo.underpowered,
    )

    warnings: list[str] = []
    if overfit.monte_carlo.underpowered:
        n_trades = len(wfo.all_oos_trades)
        warnings.append(
            f"Monte Carlo sous-puissant ({n_trades} trades OOS < 30) — score MC neutre (12/25)"
        )
    if overfit.stability.cliff_params:
        warnings.append(
            f"Paramètres instables : {', '.join(overfit.stability.cliff_params)}"
        )
    if validation.volume_warning:
        warnings.append(validation.volume_warning_detail)
    if wfo.consistency_rate < 0.5:
        warnings.append(
            f"Consistance OOS faible : {wfo.consistency_rate:.0%}"
        )
    if overfit.dsr.dsr < 0.80:
        warnings.append(
            f"DSR faible ({overfit.dsr.dsr:.2f}) — risque de data mining"
        )

    return FinalReport(
        strategy_name=wfo.strategy_name,
        symbol=wfo.symbol,
        timestamp=datetime.now(),
        grade=grade,
        wfo_avg_is_sharpe=wfo.avg_is_sharpe,
        wfo_avg_oos_sharpe=wfo.avg_oos_sharpe,
        wfo_consistency_rate=wfo.consistency_rate,
        wfo_n_windows=len(wfo.windows),
        recommended_params=wfo.recommended_params,
        mc_p_value=overfit.monte_carlo.p_value,
        mc_significant=overfit.monte_carlo.significant,
        mc_underpowered=overfit.monte_carlo.underpowered,
        dsr=overfit.dsr.dsr,
        dsr_max_expected_sharpe=overfit.dsr.max_expected_sharpe,
        stability=overfit.stability.overall_stability,
        cliff_params=overfit.stability.cliff_params,
        convergence=convergence_score,
        divergent_params=divergent,
        validation=validation,
        oos_is_ratio=wfo.oos_is_ratio,
        bitget_transfer=validation.transfer_ratio,
        live_eligible=grade in ("A", "B"),
        warnings=warnings,
        n_distinct_combos=wfo.n_distinct_combos,
    )
