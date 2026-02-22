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
    funding_paid_total: float = 0.0


@dataclass
class FinalReport:
    """Rapport complet d'une optimisation stratégie × asset."""

    strategy_name: str
    symbol: str
    timestamp: datetime
    grade: str
    total_score: int  # Score numérique 0-100

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
    shallow: bool = False
    raw_score: int = 0
    fee_sensitivity: dict[str, float] | None = None  # scenario → Sharpe re-simulé


@dataclass
class GradeResult:
    """Résultat du grading avec pénalité shallow."""

    grade: str        # Lettre finale (A-F) après pénalité + caps
    score: int        # Score final après pénalité shallow (cap trades ne modifie PAS le score)
    is_shallow: bool  # True si n_windows < 24
    raw_score: int    # Score brut avant pénalité


# ─── Grading ───────────────────────────────────────────────────────────────


def compute_grade(
    oos_is_ratio: float,
    mc_p_value: float,
    dsr: float,
    stability: float,
    bitget_transfer: float,
    mc_underpowered: bool = False,
    total_trades: int = 0,
    consistency: float = 1.0,
    transfer_significant: bool = True,
    bitget_trades: int = 0,
    n_windows: int | None = None,
) -> GradeResult:
    """Calcule le grade A-F et le score numérique 0-100.

    Barème (100 pts) :
        OOS/IS ratio    20 pts
        Monte Carlo     20 pts
        DSR             15 pts
        Consistance     20 pts
        Stabilité       10 pts
        Bitget transfer 15 pts

    Pénalité shallow validation (soustraite du score brut) :
        ≥ 24 fenêtres : 0 pts
        18–23          : -10 pts
        12–17          : -20 pts
        < 12           : -25 pts

    Garde-fou trades : < 30 trades → plafond C, < 50 trades → plafond B.
    Note : le cap trades modifie la lettre, pas le score.

    Args:
        n_windows: Nombre de fenêtres WFO. None = pas de pénalité (backward compat).

    Returns:
        GradeResult avec grade, score, is_shallow, raw_score.
    """
    score = 0
    breakdown: dict[str, int] = {}

    # OOS/IS ratio (max 20 points)
    if oos_is_ratio > 0.6:
        breakdown["oos_is_ratio"] = 20
    elif oos_is_ratio > 0.5:
        breakdown["oos_is_ratio"] = 16
    elif oos_is_ratio > 0.4:
        breakdown["oos_is_ratio"] = 12
    elif oos_is_ratio > 0.3:
        breakdown["oos_is_ratio"] = 8
    else:
        breakdown["oos_is_ratio"] = 0

    # Monte Carlo (max 20 points)
    if mc_underpowered:
        breakdown["monte_carlo"] = 10
    elif mc_p_value < 0.05:
        breakdown["monte_carlo"] = 20
    elif mc_p_value < 0.10:
        breakdown["monte_carlo"] = 12
    else:
        breakdown["monte_carlo"] = 0

    # DSR (max 15 points)
    if dsr > 0.95:
        breakdown["dsr"] = 15
    elif dsr > 0.90:
        breakdown["dsr"] = 12
    elif dsr > 0.80:
        breakdown["dsr"] = 8
    else:
        breakdown["dsr"] = 0

    # Consistance OOS (max 20 points)
    if consistency >= 0.90:
        breakdown["consistency"] = 20
    elif consistency >= 0.80:
        breakdown["consistency"] = 16
    elif consistency >= 0.70:
        breakdown["consistency"] = 12
    elif consistency >= 0.60:
        breakdown["consistency"] = 8
    elif consistency >= 0.50:
        breakdown["consistency"] = 4
    else:
        breakdown["consistency"] = 0

    # Stability (max 10 points)
    if stability > 0.80:
        breakdown["stability"] = 10
    elif stability > 0.60:
        breakdown["stability"] = 7
    elif stability > 0.40:
        breakdown["stability"] = 4
    else:
        breakdown["stability"] = 0

    # Bitget transfer (max 15 points)
    if bitget_transfer > 0.50 and transfer_significant:
        breakdown["bitget_transfer"] = 15
    elif bitget_transfer > 0.50:  # ratio bon mais CI inclut zéro
        breakdown["bitget_transfer"] = 10
    elif bitget_transfer > 0.30:
        breakdown["bitget_transfer"] = 5
    else:
        breakdown["bitget_transfer"] = 0

    # Guard : peu de trades Bitget → cap le score transfer
    if 0 < bitget_trades < 15:
        breakdown["bitget_transfer"] = min(breakdown["bitget_transfer"], 8)

    score = sum(breakdown.values())
    raw_score = score

    # Pénalité shallow validation (n_windows insuffisantes)
    shallow_penalty = 0
    if n_windows is not None:
        if n_windows >= 24:
            shallow_penalty = 0
        elif n_windows >= 18:
            shallow_penalty = 10
        elif n_windows >= 12:
            shallow_penalty = 20
        else:
            shallow_penalty = 25
        score = max(0, score - shallow_penalty)
    is_shallow = n_windows is not None and n_windows < 24

    # Déterminer la lettre
    if score >= 85:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 55:
        grade = "C"
    elif score >= 40:
        grade = "D"
    else:
        grade = "F"

    # Garde-fou trades minimum — plafonnement du grade
    grade_before_cap = grade
    _GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
    if total_trades > 0:
        if total_trades < 30:
            max_grade = "C"
            if _GRADE_ORDER[grade] < _GRADE_ORDER[max_grade]:
                grade = max_grade
        elif total_trades < 50:
            max_grade = "B"
            if _GRADE_ORDER[grade] < _GRADE_ORDER[max_grade]:
                grade = max_grade

    logger.info(
        "compute_grade: {} ({}/100{}, trades={}{}) — oos_is={:.2f}→{}/20, "
        "mc_p={:.3f}(underpow={})→{}/20, dsr={:.2f}→{}/15, "
        "consistency={:.2f}→{}/20, stability={:.2f}→{}/10, bitget={:.2f}→{}/15",
        grade, score,
        f" raw={raw_score} shallow=-{shallow_penalty} n_win={n_windows}" if shallow_penalty > 0 else "",
        total_trades,
        f" cap {grade_before_cap}→{grade}" if grade != grade_before_cap else "",
        oos_is_ratio, breakdown["oos_is_ratio"],
        mc_p_value, mc_underpowered, breakdown["monte_carlo"],
        dsr, breakdown["dsr"],
        consistency, breakdown["consistency"],
        stability, breakdown["stability"],
        bitget_transfer, breakdown["bitget_transfer"],
    )

    return GradeResult(grade=grade, score=score, is_shallow=is_shallow, raw_score=raw_score)


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
                extra_data_by_timestamp=extra_data_map,
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
            funding_paid_total=getattr(result, "funding_paid_total", 0.0),
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


def save_report(
    report: FinalReport,
    wfo_windows: list[dict] | None = None,
    duration: float | None = None,
    timeframe: str | None = None,
    output_dir: str = "data/optimization",
    db_path: str | None = None,
    combo_results: list[dict] | None = None,
    regime_analysis: dict | None = None,
) -> tuple[Path, int | None]:
    """Sauvegarde le rapport en JSON et en DB.

    Args:
        report: FinalReport complet
        wfo_windows: WindowResult sérialisés (pour la DB)
        duration: Durée du run en secondes (ou None)
        timeframe: Timeframe de la stratégie (ex: "5m", "1h")
        output_dir: Répertoire JSON
        db_path: Chemin DB SQLite (None = depuis config)
        combo_results: Combo results du WFO (Sprint 14b, optionnel)
        regime_analysis: Analyse par régime du best combo (Sprint 15b, optionnel)

    Returns:
        (filepath, result_id) : chemin JSON + ID DB (ou None si pas sauvé en DB)
    """
    # 1. Sauvegarde JSON (existant)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = report.timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"{report.strategy_name}_{report.symbol.replace('/', '_')}_{ts}.json"
    filepath = out / filename

    data = _report_to_dict(report)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)

    logger.info("Rapport JSON sauvé : {}", filepath)

    # 2. Sauvegarde DB (nouveau)
    result_id = None
    if timeframe is not None:
        from backend.core.config import get_config
        from backend.optimization.optimization_db import save_result_sync

        # Résoudre db_path depuis config si non fourni
        if db_path is None:
            config = get_config()
            db_url = config.secrets.database_url
            if db_url.startswith("sqlite:///"):
                db_path = db_url[10:]  # Retirer "sqlite:///"
            else:
                db_path = "data/scalp_radar.db"  # Fallback

        result_id = save_result_sync(
            db_path, report, wfo_windows, duration, timeframe,
            regime_analysis=regime_analysis,
        )

        # Sauver les combo results si présents (Sprint 14b)
        if result_id and combo_results:
            from backend.optimization.optimization_db import save_combo_results_sync
            n_saved = save_combo_results_sync(db_path, result_id, combo_results)
            logger.info("Combo results sauvés : {} combos pour result_id={}", n_saved, result_id)

    # 3. Push serveur (best-effort, ne crashe jamais)
    if timeframe is not None:
        from backend.optimization.optimization_db import push_to_server
        push_to_server(
            report, wfo_windows, duration, timeframe,
            combo_results=combo_results, regime_analysis=regime_analysis,
        )

    return filepath, result_id


def _report_to_dict(report: FinalReport) -> dict[str, Any]:
    """Convertit un FinalReport en dict JSON-sérialisable."""
    validation = asdict(report.validation)
    return {
        "strategy_name": report.strategy_name,
        "symbol": report.symbol,
        "timestamp": report.timestamp.isoformat(),
        "grade": report.grade,
        "total_score": report.total_score,
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
        # Certains params sont des décisions de risk management globales : ne jamais
        # les écrire dans per_asset (sinon ils écrasent le top-level lors du merge).
        _EXCLUDED_FROM_PER_ASSET = {"leverage", "enabled", "live_eligible", "weight", "timeframe"}

        per_asset = strat_data.get("per_asset", {})
        for report in strat_reports:
            symbol = report.symbol
            asset_overrides = per_asset.get(symbol, {})

            for param_name in report.recommended_params:
                if param_name in _EXCLUDED_FROM_PER_ASSET:
                    continue
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


# ─── Leverage validation ──────────────────────────────────────────────────


def _validate_leverage_sl(strategy_name: str, params: dict) -> list[str]:
    """Valide que la combinaison SL × leverage est viable en cross margin."""
    from backend.optimization import STRATEGY_REGISTRY

    warnings: list[str] = []

    sl_pct = params.get("sl_percent")
    if sl_pct is None:
        return warnings

    entry = STRATEGY_REGISTRY.get(strategy_name)
    if entry is None:
        return warnings

    config_cls, _ = entry
    default_cfg = config_cls()
    leverage = getattr(default_cfg, "leverage", 6)

    loss_per_margin = sl_pct * leverage / 100

    if loss_per_margin > 1.0:
        warnings.append(
            f"SL {sl_pct}% x leverage {leverage}x = {loss_per_margin:.0%} de la marge "
            f"(depasse 100% — chaque SL coute plus que sa marge en cross margin)"
        )
    elif loss_per_margin > 0.8:
        warnings.append(
            f"SL {sl_pct}% x leverage {leverage}x = {loss_per_margin:.0%} de la marge "
            f"(risque, peu de marge de securite)"
        )

    return warnings


# ─── Fee Sensitivity Analysis ──────────────────────────────────────────────

# Multiplicateurs de fees par scénario (relatif au nominal 0.06%/0.05%)
_FEE_SCENARIOS: dict[str, dict[str, float]] = {
    "nominal":  {"fee_mult": 1.0,    "slip_mult": 1.0},
    "degraded": {"fee_mult": 4 / 3,  "slip_mult": 2.0},   # 0.06→0.08%, 0.05→0.10%
    "stress":   {"fee_mult": 5 / 3,  "slip_mult": 4.0},   # 0.06→0.10%, 0.05→0.20%
}


def _fee_sensitivity_analysis(trades: list[TradeResult]) -> dict[str, float]:
    """Calcule le Sharpe OOS re-simulé pour 3 scénarios de fees.

    Utilise les champs fee_cost et slippage_cost de chaque trade pour scaler
    les frais alternatifs. Ne touche pas au grading — diagnostic uniquement.
    """
    if len(trades) < 5:
        return {}

    results: dict[str, float] = {}
    for scenario, mults in _FEE_SCENARIOS.items():
        capital = 10_000.0
        returns: list[float] = []
        for trade in trades:
            adj_net = (
                trade.gross_pnl
                - trade.fee_cost * mults["fee_mult"]
                - trade.slippage_cost * mults["slip_mult"]
            )
            if capital > 0:
                returns.append(adj_net / capital)
            capital += adj_net

        if len(returns) < 2:
            results[scenario] = 0.0
            continue

        arr = np.array(returns)
        std = float(np.std(arr))
        results[scenario] = float(np.mean(arr) / std) if std > 0 else 0.0

    return results


# ─── Build FinalReport ─────────────────────────────────────────────────────


def build_final_report(
    wfo: WFOResult,
    overfit: OverfitReport,
    validation: ValidationResult,
    regime_analysis: dict | None = None,
) -> FinalReport:
    """Construit le rapport final à partir des résultats WFO, overfitting, validation."""
    convergence_score = (
        overfit.convergence.convergence_score if overfit.convergence else None
    )
    divergent = (
        overfit.convergence.divergent_params if overfit.convergence else []
    )

    # Nombre de trades OOS du best combo (ou fallback all_oos_trades)
    best_combo_trades = 0
    for c in wfo.combo_results:
        if c.get("is_best"):
            best_combo_trades = c.get("oos_trades") or 0
            break
    if best_combo_trades == 0:
        best_combo_trades = len(wfo.all_oos_trades)

    grade_result = compute_grade(
        oos_is_ratio=wfo.oos_is_ratio,
        mc_p_value=overfit.monte_carlo.p_value,
        dsr=overfit.dsr.dsr,
        stability=overfit.stability.overall_stability,
        bitget_transfer=validation.transfer_ratio,
        mc_underpowered=overfit.monte_carlo.underpowered,
        total_trades=best_combo_trades,
        consistency=wfo.consistency_rate,
        transfer_significant=validation.transfer_significant,
        bitget_trades=validation.bitget_trades,
        n_windows=len(wfo.windows),
    )
    grade = grade_result.grade
    total_score = grade_result.score

    warnings: list[str] = []
    if best_combo_trades < 30:
        warnings.append(
            f"Seulement {best_combo_trades} trades OOS — grade plafonné à C"
        )
    elif best_combo_trades < 50:
        warnings.append(
            f"Seulement {best_combo_trades} trades OOS — grade plafonné à B"
        )
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

    # --- Diagnostic régimes incomplets ---
    ALL_REGIMES = {"crash", "bull", "range", "bear"}
    if regime_analysis:
        tested = set(regime_analysis.keys())
        missing = ALL_REGIMES - tested
        n_tested = len(tested)
        weak = [
            r for r, d in regime_analysis.items()
            if d.get("avg_oos_sharpe", 0) < 0
        ]

        if weak:
            warnings.append(
                f"Faible en {', '.join(weak)} — Sharpe négatif dans "
                f"{'ce régime' if len(weak) == 1 else 'ces régimes'}"
            )
        if n_tested == 4 and not weak:
            pass  # Robuste tous régimes — pas de warning
        elif n_tested == 3 and not weak:
            missing_names = ", ".join(sorted(missing))
            warnings.append(
                f"Couverture régimes 3/4 — {missing_names} non couvert dans les données"
            )
        elif n_tested <= 2:
            missing_names = ", ".join(sorted(missing))
            warnings.append(
                f"Couverture partielle — testé sur {n_tested}/4 régimes seulement "
                f"({missing_names} non couverts)"
            )

    # --- Diagnostic trades par fenêtre OOS ---
    n_windows = len(wfo.windows)
    if n_windows > 0:
        total_oos_trades = sum(w.oos_trades for w in wfo.windows)
        avg_trades = total_oos_trades / n_windows
        if avg_trades < 5:
            warnings.append(
                f"Volume faible — moyenne de {avg_trades:.1f} trades/fenêtre OOS. "
                f"Significativité statistique limitée"
            )
        elif avg_trades < 10:
            warnings.append(
                f"Volume modéré — moyenne de {avg_trades:.1f} trades/fenêtre OOS. "
                f"Résultats à confirmer avec plus de données"
            )

    # Shallow validation warning
    if grade_result.is_shallow:
        penalty = grade_result.raw_score - grade_result.score
        warnings.append(
            f"Shallow validation ({len(wfo.windows)} fenêtres < 24) — pénalité -{penalty} pts"
        )

    # Validation leverage × SL en cross margin
    leverage_warnings = _validate_leverage_sl(wfo.strategy_name, wfo.recommended_params)
    warnings.extend(leverage_warnings)

    # Fee sensitivity analysis (diagnostic uniquement — ne touche pas au grade)
    fee_sensitivity = _fee_sensitivity_analysis(wfo.all_oos_trades)
    if fee_sensitivity:
        degraded_sharpe = fee_sensitivity.get("degraded", 0.0)
        nominal_sharpe = fee_sensitivity.get("nominal", 0.0)
        if degraded_sharpe < 0.5 and nominal_sharpe >= 0.5:
            warnings.append(
                f"Sensible aux frais : Sharpe nominal={nominal_sharpe:.2f} "
                f"→ degraded={degraded_sharpe:.2f} (fees +33%, slip ×2)"
            )

    return FinalReport(
        strategy_name=wfo.strategy_name,
        symbol=wfo.symbol,
        timestamp=datetime.now(),
        grade=grade,
        total_score=total_score,
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
        shallow=grade_result.is_shallow,
        raw_score=grade_result.raw_score,
        fee_sensitivity=fee_sensitivity if fee_sensitivity else None,
    )
