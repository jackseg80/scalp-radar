"""Walk-Forward Optimizer pour Scalp Radar.

Optimisation par fenêtres glissantes IS→OOS avec grid search en 2 passes
(coarse Latin Hypercube → fine autour du top 20).
Parallélisé via ProcessPoolExecutor.
"""

from __future__ import annotations

import asyncio
import gc
import itertools
import json
import os
import time

# Désactiver le JIT Python 3.13 (segfaults sur calculs longs)
os.environ.setdefault("PYTHON_JIT", "0")
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
import threading

import numpy as np
import yaml
from loguru import logger

from backend.backtesting.engine import BacktestConfig, BacktestResult, run_backtest_single
from backend.backtesting.extra_data_builder import build_extra_data_map
from backend.backtesting.metrics import calculate_metrics
from backend.core.database import Database
from backend.core.models import Candle, TimeFrame
from backend.core.position_manager import TradeResult


# ─── Dataclasses résultat ──────────────────────────────────────────────────


@dataclass
class WindowResult:
    """Résultat d'une fenêtre IS+OOS."""

    window_index: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime
    best_params: dict[str, Any]
    is_sharpe: float
    is_net_return_pct: float
    is_profit_factor: float
    is_trades: int
    oos_sharpe: float
    oos_net_return_pct: float
    oos_profit_factor: float
    oos_trades: int
    top_n_params: list[dict] = field(default_factory=list)


@dataclass
class WFOResult:
    """Résultat complet du walk-forward."""

    strategy_name: str
    symbol: str
    windows: list[WindowResult]
    avg_is_sharpe: float
    avg_oos_sharpe: float
    oos_is_ratio: float
    consistency_rate: float
    recommended_params: dict[str, Any]
    all_oos_trades: list[TradeResult]
    n_distinct_combos: int
    combo_results: list[dict[str, Any]] = field(default_factory=list)
    window_regimes: list[dict[str, Any]] = field(default_factory=list)
    regime_analysis: dict[str, dict[str, Any]] | None = None


# ─── Helpers ────────────────────────────────────────────────────────────────


def _load_param_grids(config_path: str = "config/param_grids.yaml") -> dict[str, Any]:
    """Charge les grilles de paramètres."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_grid(
    strategy_grids: dict[str, Any], symbol: str
) -> list[dict[str, Any]]:
    """Construit la grille complète pour un (strategy, symbol).

    Merge default + overrides spécifiques au symbol.
    """
    default = strategy_grids.get("default", {})
    overrides = strategy_grids.get(symbol, {})
    merged = {**default, **overrides}

    if not merged:
        return []

    keys = sorted(merged.keys())
    values = [merged[k] for k in keys]
    combos = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combos]


def _latin_hypercube_sample(
    grid: list[dict[str, Any]], n_samples: int, seed: int = 42
) -> list[dict[str, Any]]:
    """Sous-échantillonnage stratifié de la grille (approximation LHS).

    Sélectionne n_samples combinaisons en essayant de couvrir l'espace.
    """
    if len(grid) <= n_samples:
        return grid

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(grid), size=n_samples, replace=False)
    return [grid[i] for i in sorted(indices)]


def _fine_grid_around_top(
    top_params: list[dict[str, Any]],
    full_grid_values: dict[str, list],
) -> list[dict[str, Any]]:
    """Génère un grid fin autour des top N combinaisons.

    Pour chaque paramètre de chaque top combo, explore ±1 step dans le grid.
    """
    fine_combos: set[tuple] = set()
    sorted_keys = sorted(full_grid_values.keys())

    for params in top_params:
        for key in sorted_keys:
            values = full_grid_values[key]
            current_val = params.get(key)
            if current_val not in values:
                continue
            idx = values.index(current_val)
            # ±1 step
            neighbors = set()
            for offset in [-1, 0, 1]:
                ni = idx + offset
                if 0 <= ni < len(values):
                    neighbors.add(values[ni])

            # Construire les combos voisines pour ce paramètre
            for neighbor_val in neighbors:
                combo = dict(params)
                combo[key] = neighbor_val
                combo_tuple = tuple(combo[k] for k in sorted_keys)
                fine_combos.add(combo_tuple)

    return [dict(zip(sorted_keys, combo)) for combo in fine_combos]


def _median_params(
    all_best_params: list[dict[str, Any]],
    grid_values: dict[str, list],
) -> dict[str, Any]:
    """Calcule la médiane des paramètres et snappe à la valeur du grid la plus proche."""
    if not all_best_params:
        return {}

    result: dict[str, Any] = {}
    keys = all_best_params[0].keys()

    for key in keys:
        values = [p[key] for p in all_best_params if key in p]
        if not values:
            continue

        # Vérifier si numérique
        if all(isinstance(v, (int, float)) for v in values):
            median_val = float(np.median(values))
            # Snapper au grid le plus proche
            if key in grid_values and grid_values[key]:
                grid_vals = sorted(grid_values[key])
                closest = min(grid_vals, key=lambda x: abs(x - median_val))
                # Garder le type d'origine
                if all(isinstance(v, int) for v in values):
                    result[key] = int(closest)
                else:
                    result[key] = closest
            else:
                result[key] = median_val
        else:
            # Non-numérique : mode (valeur la plus fréquente)
            from collections import Counter
            result[key] = Counter(values).most_common(1)[0][0]

    return result


def _slice_candles(
    candles: list[Candle], start: datetime, end: datetime
) -> list[Candle]:
    """Extrait les candles dans [start, end)."""
    return [c for c in candles if start <= c.timestamp < end]



def _classify_regime(candles: list[Candle]) -> dict[str, Any]:
    """Classifie le régime de marché d'une période OOS.

    Ordre d'évaluation : Crash (prioritaire) → Bull → Bear → Range.

    Critères :
    - Crash : max drawdown > 30% survenant en < 14 jours
    - Bull : rendement fenêtre > +20%
    - Bear : rendement fenêtre < -20%
    - Range : ni bull ni bear (rendement entre -20% et +20%)
    """
    if len(candles) < 2:
        return {"regime": "range", "return_pct": 0.0, "max_dd_pct": 0.0}

    closes = [c.close for c in candles]
    timestamps = [c.timestamp for c in candles]
    return_pct = (closes[-1] - closes[0]) / closes[0] * 100

    # Max drawdown global
    peak = closes[0]
    max_dd = 0.0
    for close in closes:
        if close > peak:
            peak = close
        dd = (close - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    # Fast crash detection : peak-to-trough > 30% en ≤ 14 jours
    # Algorithme O(n) : sliding window maximum via deque
    from collections import deque

    is_crash = False
    max_seconds = 14 * 86400
    peak_deque: deque[int] = deque()

    for i in range(len(closes)):
        ts_i = timestamps[i].timestamp()

        # Retirer les éléments hors fenêtre de 14 jours
        while peak_deque and (ts_i - timestamps[peak_deque[0]].timestamp()) > max_seconds:
            peak_deque.popleft()

        # Retirer les éléments plus petits que le courant (ne seront jamais le max)
        while peak_deque and closes[peak_deque[-1]] <= closes[i]:
            peak_deque.pop()

        peak_deque.append(i)

        # Le max dans la fenêtre glissante est closes[peak_deque[0]]
        local_peak = closes[peak_deque[0]]
        if local_peak > 0:
            dd_14d = (closes[i] - local_peak) / local_peak * 100
            if dd_14d < -30:
                is_crash = True
                break

    # Classification (crash prioritaire)
    if is_crash:
        regime = "crash"
    elif return_pct > 20:
        regime = "bull"
    elif return_pct < -20:
        regime = "bear"
    else:
        regime = "range"

    return {
        "regime": regime,
        "return_pct": round(return_pct, 2),
        "max_dd_pct": round(max_dd, 2),
    }


# ─── Worker pool avec initializer (candles chargées 1 fois par worker) ─────

# Type retour léger : pas de trades (inutiles pour la sélection IS)
_ISResult = tuple[dict[str, Any], float, float, float, int]

_worker_candles: dict[str, list[Candle]] = {}
_worker_strategy: str = ""
_worker_symbol: str = ""
_worker_bt_config: BacktestConfig | None = None
_worker_main_tf: str = ""
_worker_extra_data: dict[str, dict[str, Any]] | None = None


def _init_worker(
    candles_serialized: dict[str, list[dict]],
    strategy_name: str,
    symbol: str,
    bt_config_dict: dict,
    main_tf: str,
    extra_data_map: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Initialise les données partagées dans chaque worker (appelé 1x par process)."""
    global _worker_candles, _worker_strategy, _worker_symbol, _worker_bt_config, _worker_main_tf, _worker_extra_data

    # Couper les logs dans les workers (seul le process principal affiche la progression)
    import logging
    logging.disable(logging.INFO)
    logger.disable("backend")

    _worker_candles = {}
    for tf_str, candle_dicts in candles_serialized.items():
        _worker_candles[tf_str] = [Candle(**cd) for cd in candle_dicts]
    _worker_strategy = strategy_name
    _worker_symbol = symbol
    _worker_bt_config = BacktestConfig(**bt_config_dict)
    _worker_main_tf = main_tf
    _worker_extra_data = extra_data_map


def _run_single_backtest_worker(params: dict[str, Any]) -> _ISResult:
    """Fonction top-level pour ProcessPoolExecutor.

    Retour léger (pas de trades) pour minimiser le pickling inter-process.
    """
    from backend.optimization import is_grid_strategy

    if is_grid_strategy(_worker_strategy):
        from backend.backtesting.multi_engine import run_multi_backtest_single
        result = run_multi_backtest_single(
            _worker_strategy, params, _worker_candles, _worker_bt_config, _worker_main_tf,
        )
    else:
        result = run_backtest_single(
            _worker_strategy, params, _worker_candles, _worker_bt_config, _worker_main_tf,
            extra_data_by_timestamp=_worker_extra_data,
        )
    metrics = calculate_metrics(result)

    return (
        params,
        metrics.sharpe_ratio,
        metrics.net_return_pct,
        metrics.profit_factor,
        metrics.total_trades,
    )


def _run_single_backtest_sequential(
    params: dict[str, Any],
    candles_by_tf: dict[str, list[Candle]],
    strategy_name: str,
    bt_config: BacktestConfig,
    main_tf: str,
    precomputed_indicators: dict | None = None,
    extra_data_by_timestamp: dict[str, dict[str, Any]] | None = None,
) -> _ISResult:
    """Version séquentielle (fallback si ProcessPoolExecutor crashe)."""
    from backend.optimization import is_grid_strategy

    if is_grid_strategy(strategy_name):
        from backend.backtesting.multi_engine import run_multi_backtest_single
        result = run_multi_backtest_single(
            strategy_name, params, candles_by_tf, bt_config, main_tf,
            precomputed_indicators=precomputed_indicators,
        )
    else:
        result = run_backtest_single(
            strategy_name, params, candles_by_tf, bt_config, main_tf,
            precomputed_indicators=precomputed_indicators,
            extra_data_by_timestamp=extra_data_by_timestamp,
        )
    metrics = calculate_metrics(result)
    return (
        params,
        metrics.sharpe_ratio,
        metrics.net_return_pct,
        metrics.profit_factor,
        metrics.total_trades,
    )


# Paramètres qui affectent compute_indicators() (tout le reste = seuils evaluate())
# Funding/liquidation : aucun paramètre n'affecte les indicateurs (pas de compute_indicators)
_INDICATOR_PARAMS: dict[str, list[str]] = {
    "vwap_rsi": ["rsi_period"],
    "momentum": ["breakout_lookback"],
    "funding": [],
    "liquidation": [],
    "bollinger_mr": ["bb_period", "bb_std"],
    "donchian_breakout": ["entry_lookback", "atr_period"],
    "supertrend": ["atr_period", "atr_multiplier"],
    "envelope_dca": ["ma_period"],
    "envelope_dca_short": ["ma_period"],
}


def _serialize_candles_by_tf(candles_by_tf: dict[str, list[Candle]]) -> dict[str, list[dict]]:
    """Sérialise les candles pour passage inter-process."""
    result: dict[str, list[dict]] = {}
    for tf, candles in candles_by_tf.items():
        result[tf] = [c.model_dump() for c in candles]
    return result


# ─── WalkForwardOptimizer ──────────────────────────────────────────────────


class WalkForwardOptimizer:
    """Optimiseur Walk-Forward avec grid search en 2 passes."""

    def __init__(self, config_dir: str = "config") -> None:
        self._config_dir = config_dir
        self._grids = _load_param_grids(str(Path(config_dir) / "param_grids.yaml"))

    async def optimize(
        self,
        strategy_name: str,
        symbol: str,
        exchange: str | None = None,
        is_window_days: int = 120,
        oos_window_days: int = 30,
        step_days: int = 30,
        max_workers: int | None = None,
        metric: str = "sharpe_ratio",
        progress_callback: Callable[[float, str], None] | None = None,
        cancel_event: threading.Event | None = None,
        params_override: dict | None = None,
    ) -> WFOResult:
        """Walk-forward optimization complète.

        Args:
            params_override: Sous-grille custom {param_name: [values]} fusionnée dans "default".
        """
        opt_config = self._grids.get("optimization", {})
        is_window_days = opt_config.get("is_window_days", is_window_days)
        oos_window_days = opt_config.get("oos_window_days", oos_window_days)
        step_days = opt_config.get("step_days", step_days)
        metric = opt_config.get("metric", metric)

        # Per-strategy WFO config override (section `wfo:` dans param_grids.yaml)
        strategy_grids_wfo = self._grids.get(strategy_name, {}).get("wfo", {})
        if strategy_grids_wfo:
            is_window_days = strategy_grids_wfo.get("is_days", is_window_days)
            oos_window_days = strategy_grids_wfo.get("oos_days", oos_window_days)
            step_days = strategy_grids_wfo.get("step_days", step_days)
        max_workers_cfg = opt_config.get("max_workers")
        if max_workers is None:
            max_workers = max_workers_cfg

        # Stratégie config pour le timeframe
        from backend.optimization import STRATEGY_REGISTRY, is_grid_strategy
        if strategy_name not in STRATEGY_REGISTRY:
            raise ValueError(f"Stratégie '{strategy_name}' non optimisable")

        config_cls, _ = STRATEGY_REGISTRY[strategy_name]
        default_cfg = config_cls()
        main_tf = default_cfg.timeframe
        tfs_needed = [main_tf]
        if hasattr(default_cfg, "trend_filter_timeframe"):
            tfs_needed.append(default_cfg.trend_filter_timeframe)

        # Exchange par défaut : binance (données profondes depuis 2020)
        # Override possible via --exchange en CLI
        db = Database()
        await db.init()
        if exchange is None:
            exchange = "binance"

        # Charger les candles depuis la DB
        logger.info(
            "Chargement candles {} {} depuis {} ...",
            symbol, tfs_needed, exchange,
        )

        all_candles_by_tf: dict[str, list[Candle]] = {}
        for tf in tfs_needed:
            candles = await db.get_candles(
                symbol, tf, exchange=exchange, limit=1_000_000
            )
            all_candles_by_tf[tf] = candles
            logger.info("  {} : {} candles", tf, len(candles))

        # Charger funding/OI si la stratégie en a besoin
        from backend.optimization import STRATEGIES_NEED_EXTRA_DATA
        all_funding_rates: list[dict] = []
        all_oi_records: list[dict] = []
        needs_extra = strategy_name in STRATEGIES_NEED_EXTRA_DATA

        if needs_extra:
            logger.info("Chargement données extra (funding/OI) depuis {} ...", exchange)
            all_funding_rates = await db.get_funding_rates(symbol, exchange=exchange)
            all_oi_records = await db.get_open_interest(symbol, timeframe="5m", exchange=exchange)
            logger.info(
                "  funding: {} rates, OI: {} records",
                len(all_funding_rates), len(all_oi_records),
            )
            if not all_funding_rates and strategy_name == "funding":
                logger.warning(
                    "Aucun funding rate en DB pour {} — "
                    "lancez: uv run python -m scripts.fetch_funding --symbol {}",
                    symbol, symbol,
                )
            if not all_oi_records and strategy_name == "liquidation":
                logger.warning(
                    "Aucun record OI en DB pour {} — "
                    "lancez: uv run python -m scripts.fetch_oi --symbol {}",
                    symbol, symbol,
                )

        await db.close()

        if not all_candles_by_tf.get(main_tf):
            raise ValueError(
                f"Pas de candles {main_tf} pour {symbol} sur {exchange}"
            )

        # Déterminer les bornes temporelles
        main_candles = all_candles_by_tf[main_tf]
        data_start = main_candles[0].timestamp
        data_end = main_candles[-1].timestamp

        # Construire les fenêtres
        windows = self._build_windows(
            data_start, data_end, is_window_days, oos_window_days, step_days
        )
        logger.info("{} fenêtres WFO", len(windows))

        if not windows:
            raise ValueError("Pas assez de données pour au moins une fenêtre WFO")

        # Construire le grid
        strategy_grids = self._grids.get(strategy_name, {})

        # Fusionner params_override dans "default" si fourni
        if params_override:
            merged_default = {**strategy_grids.get("default", {}), **params_override}
            strategy_grids = {**strategy_grids, "default": merged_default}

        full_grid = _build_grid(strategy_grids, symbol)
        grid_values = {**strategy_grids.get("default", {}), **strategy_grids.get(symbol, {})}
        logger.info("Grid complet : {} combinaisons", len(full_grid))

        # Grid search en 2 passes
        coarse_max = 200
        if len(full_grid) > coarse_max:
            coarse_grid = _latin_hypercube_sample(full_grid, coarse_max)
            logger.info("Coarse pass : {} combinaisons (LHS)", len(coarse_grid))
        else:
            coarse_grid = full_grid

        # 4 workers = bon compromis performance/thermique sur laptop
        # (8 workers = seulement ~1.5x plus rapide mais double la charge thermique)
        n_workers = max_workers or min(os.cpu_count() or 4, 4)
        n_distinct_combos = len(coarse_grid)

        # Backtest config
        bt_config = BacktestConfig(
            symbol=symbol,
            start_date=data_start,
            end_date=data_end,
        )
        # Override leverage si la stratégie en spécifie un (ex: envelope_dca=6)
        if hasattr(default_cfg, 'leverage'):
            bt_config.leverage = default_cfg.leverage
        bt_config_dict = {
            "symbol": bt_config.symbol,
            "start_date": bt_config.start_date,
            "end_date": bt_config.end_date,
            "initial_capital": bt_config.initial_capital,
            "leverage": bt_config.leverage,
            "maker_fee": bt_config.maker_fee,
            "taker_fee": bt_config.taker_fee,
            "slippage_pct": bt_config.slippage_pct,
            "high_vol_slippage_mult": bt_config.high_vol_slippage_mult,
            "max_risk_per_trade": bt_config.max_risk_per_trade,
        }

        # Optimisation par fenêtre
        window_results: list[WindowResult] = []
        all_oos_trades: list[TradeResult] = []

        # Accumulateur combo results cross-fenêtre (Sprint 14b)
        # Skip si stratégie sans fast engine (trop lent)
        collect_combo_results = strategy_name in ("vwap_rsi", "momentum", "bollinger_mr", "donchian_breakout", "supertrend", "envelope_dca", "envelope_dca_short")
        combo_accumulator: dict[str, list[dict]] = {}
        window_regimes: list[dict[str, Any]] = []

        for w_idx, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
            logger.info(
                "Fenêtre {}/{} : IS {} → {} | OOS {} → {}",
                w_idx + 1, len(windows),
                is_start.strftime("%Y-%m-%d"), is_end.strftime("%Y-%m-%d"),
                oos_start.strftime("%Y-%m-%d"), oos_end.strftime("%Y-%m-%d"),
            )

            # Check annulation
            if cancel_event and cancel_event.is_set():
                logger.warning("Optimisation annulée (cancel_event set)")
                raise asyncio.CancelledError("Optimisation annulée par l'utilisateur")

            try:
                # Découper les candles IS
                is_candles_by_tf: dict[str, list[Candle]] = {}
                for tf, candles in all_candles_by_tf.items():
                    is_candles_by_tf[tf] = _slice_candles(candles, is_start, is_end)

                if not is_candles_by_tf.get(main_tf):
                    logger.warning("Fenêtre {} : pas de candles IS, skip", w_idx)
                    continue

                # Build extra_data_map pour IS window (funding/OI)
                is_extra_data_map: dict[str, dict[str, Any]] | None = None
                if needs_extra:
                    is_extra_data_map = build_extra_data_map(
                        is_candles_by_tf[main_tf],
                        all_funding_rates, all_oi_records,
                    )

                # --- Coarse pass ---
                coarse_results = self._parallel_backtest(
                    coarse_grid, is_candles_by_tf, strategy_name, symbol,
                    bt_config_dict, main_tf, n_workers, metric,
                    extra_data_map=is_extra_data_map,
                )

                # Top 20
                coarse_results.sort(key=lambda r: r[1], reverse=True)
                top_20 = coarse_results[:20]

                # --- Fine pass ---
                top_20_params = [r[0] for r in top_20]
                fine_grid = _fine_grid_around_top(top_20_params, grid_values)
                n_distinct_combos = max(n_distinct_combos, len(coarse_grid) + len(fine_grid))

                if fine_grid:
                    fine_results = self._parallel_backtest(
                        fine_grid, is_candles_by_tf, strategy_name, symbol,
                        bt_config_dict, main_tf, n_workers, metric,
                        extra_data_map=is_extra_data_map,
                    )
                    all_is_results = coarse_results + fine_results
                else:
                    all_is_results = coarse_results

                # Meilleur IS
                all_is_results.sort(key=lambda r: r[1], reverse=True)
                best_is = all_is_results[0]
                best_params = best_is[0]
                is_sharpe = best_is[1]
                is_net_return = best_is[2]
                is_pf = best_is[3]
                is_n_trades = best_is[4]

                # Top 5 pour stabilité
                top_5 = [{"params": r[0], "sharpe": r[1]} for r in all_is_results[:5]]

                # --- Découper les candles OOS (commun à tous les chemins) ---
                oos_candles_by_tf: dict[str, list[Candle]] = {}
                for tf, candles in all_candles_by_tf.items():
                    oos_candles_by_tf[tf] = _slice_candles(candles, oos_start, oos_end)

                if not oos_candles_by_tf.get(main_tf):
                    logger.warning("Fenêtre {} : pas de candles OOS, skip", w_idx)
                    continue

                # Classifier le régime de la fenêtre OOS (Sprint 15b)
                oos_regime = _classify_regime(oos_candles_by_tf[main_tf])
                window_regimes.append(oos_regime)
                logger.info(
                    "  Régime OOS : {} (return={:.1f}%, max_dd={:.1f}%)",
                    oos_regime["regime"], oos_regime["return_pct"], oos_regime["max_dd_pct"],
                )

                # Build extra_data_map pour OOS window (commun)
                oos_extra_data_map: dict[str, dict[str, Any]] | None = None
                if needs_extra:
                    oos_extra_data_map = build_extra_data_map(
                        oos_candles_by_tf[main_tf],
                        all_funding_rates, all_oi_records,
                    )

                # --- OOS batch pour combo results (Sprint 14b) ---
                if collect_combo_results:
                    # Dédupliquer les params testés (coarse + fine)
                    seen_keys = set()
                    unique_params = []
                    for r in all_is_results:
                        key = json.dumps(r[0], sort_keys=True)
                        if key not in seen_keys:
                            seen_keys.add(key)
                            unique_params.append(r[0])

                    # OOS batch (via fast engine)
                    oos_batch_results = self._parallel_backtest(
                        unique_params, oos_candles_by_tf, strategy_name, symbol,
                        bt_config_dict, main_tf, n_workers, metric,
                        extra_data_map=oos_extra_data_map,
                    )

                    # Index les résultats IS et OOS par params_key
                    is_by_key = {json.dumps(r[0], sort_keys=True): r for r in all_is_results}
                    oos_by_key = {json.dumps(r[0], sort_keys=True): r for r in oos_batch_results}

                    for params_key in is_by_key:
                        is_r = is_by_key[params_key]
                        oos_r = oos_by_key.get(params_key)

                        if params_key not in combo_accumulator:
                            combo_accumulator[params_key] = []

                        combo_accumulator[params_key].append({
                            "is_sharpe": is_r[1],
                            "is_return_pct": is_r[2],
                            "is_trades": is_r[4],
                            "oos_sharpe": oos_r[1] if oos_r else None,
                            "oos_return_pct": oos_r[2] if oos_r else None,
                            "oos_trades": oos_r[4] if oos_r else None,
                            "window_idx": w_idx,
                        })

                # --- OOS evaluation (best params uniquement) ---

                if is_grid_strategy(strategy_name):
                    from backend.backtesting.multi_engine import run_multi_backtest_single
                    oos_result = run_multi_backtest_single(
                        strategy_name, best_params, oos_candles_by_tf, bt_config, main_tf,
                    )
                else:
                    oos_result = run_backtest_single(
                        strategy_name, best_params, oos_candles_by_tf, bt_config, main_tf,
                        extra_data_by_timestamp=oos_extra_data_map,
                    )
                oos_metrics = calculate_metrics(oos_result)

                all_oos_trades.extend(oos_result.trades)

                # OOS Sharpe = NaN si < 3 trades (non significatif)
                oos_sharpe = (
                    oos_metrics.sharpe_ratio
                    if oos_metrics.total_trades >= 3
                    else float("nan")
                )

                window_results.append(WindowResult(
                    window_index=w_idx,
                    is_start=is_start,
                    is_end=is_end,
                    oos_start=oos_start,
                    oos_end=oos_end,
                    best_params=best_params,
                    is_sharpe=is_sharpe,
                    is_net_return_pct=is_net_return,
                    is_profit_factor=is_pf,
                    is_trades=is_n_trades,
                    oos_sharpe=oos_sharpe,
                    oos_net_return_pct=oos_metrics.net_return_pct,
                    oos_profit_factor=oos_metrics.profit_factor,
                    oos_trades=oos_metrics.total_trades,
                    top_n_params=top_5,
                ))

            except Exception as exc:
                logger.error(
                    "Fenêtre {}/{} ERREUR: {} — skip",
                    w_idx + 1, len(windows), exc,
                )
                continue
            finally:
                # Libérer mémoire entre fenêtres
                gc.collect()

                # Progress callback (WFO = 80% du total)
                if progress_callback:
                    pct = (w_idx + 1) / len(windows) * 80.0
                    phase = f"WFO Fenêtre {w_idx + 1}/{len(windows)}"
                    progress_callback(pct, phase)

        # Agréger les résultats
        if not window_results:
            raise ValueError("Aucune fenêtre WFO valide")

        avg_is = float(np.nanmedian([w.is_sharpe for w in window_results]))
        oos_sharpes = np.array([w.oos_sharpe for w in window_results])
        n_valid_oos = int(np.sum(~np.isnan(oos_sharpes)))
        avg_oos = float(np.nanmedian(oos_sharpes)) if n_valid_oos > 0 else 0.0
        oos_is_ratio = avg_oos / avg_is if avg_is > 0 else 0.0
        # Consistance = % des fenêtres OOS valides (≥3 trades) qui sont positives
        consistency = (
            sum(1 for s in oos_sharpes if not np.isnan(s) and s > 0) / n_valid_oos
            if n_valid_oos > 0
            else 0.0
        )

        # Paramètres recommandés = médiane des best_params
        all_best_params = [w.best_params for w in window_results]
        recommended = _median_params(all_best_params, grid_values)

        # Agrégation des combo results (Sprint 14b)
        combo_results: list[dict[str, Any]] = []
        if collect_combo_results and combo_accumulator:
            recommended_key = json.dumps(recommended, sort_keys=True)

            for params_key, window_data in combo_accumulator.items():
                params = json.loads(params_key)

                is_sharpes = [d["is_sharpe"] for d in window_data]
                oos_sharpes = [d["oos_sharpe"] for d in window_data if d["oos_sharpe"] is not None]
                oos_returns = [d["oos_return_pct"] for d in window_data if d["oos_return_pct"] is not None]
                oos_trades_list = [d["oos_trades"] for d in window_data if d["oos_trades"] is not None]

                avg_is_sharpe = float(np.nanmean(is_sharpes)) if is_sharpes else 0.0
                avg_oos_sharpe = float(np.nanmean(oos_sharpes)) if oos_sharpes else 0.0
                total_oos_return = sum(oos_returns) if oos_returns else 0.0
                total_oos_trades = sum(oos_trades_list) if oos_trades_list else 0
                n_oos_positive = sum(1 for s in oos_sharpes if s > 0)
                consistency_combo = n_oos_positive / len(oos_sharpes) if oos_sharpes else 0.0
                oos_is_ratio_combo = avg_oos_sharpe / avg_is_sharpe if avg_is_sharpe > 0 else 0.0

                combo_results.append({
                    "params": params,
                    "is_sharpe": round(avg_is_sharpe, 4),
                    "is_return_pct": round(sum(d["is_return_pct"] for d in window_data), 4),
                    "is_trades": sum(d["is_trades"] for d in window_data),
                    "oos_sharpe": round(avg_oos_sharpe, 4),
                    "oos_return_pct": round(total_oos_return, 4),
                    "oos_trades": total_oos_trades,
                    "oos_win_rate": None,  # Non disponible via fast engine
                    "consistency": round(consistency_combo, 4),
                    "oos_is_ratio": round(oos_is_ratio_combo, 4),
                    "is_best": params_key == recommended_key,
                    "n_windows_evaluated": len(window_data),
                })

            logger.info("Sprint 14b : {} combo results agrégés", len(combo_results))

        # Agrégation regime_analysis pour le best combo (Sprint 15b)
        regime_analysis: dict[str, dict[str, Any]] | None = None
        if collect_combo_results and combo_accumulator and window_regimes:
            recommended_key = json.dumps(recommended, sort_keys=True)
            best_window_data = combo_accumulator.get(recommended_key, [])

            if best_window_data:
                regime_groups: dict[str, list[dict]] = {}
                for wd in best_window_data:
                    w_idx = wd.get("window_idx", -1)
                    if 0 <= w_idx < len(window_regimes):
                        regime = window_regimes[w_idx]["regime"]
                        regime_groups.setdefault(regime, []).append(wd)

                regime_analysis = {}
                for regime, entries in regime_groups.items():
                    oos_sharpes = [e["oos_sharpe"] for e in entries if e["oos_sharpe"] is not None]
                    oos_returns = [e["oos_return_pct"] for e in entries if e["oos_return_pct"] is not None]
                    n_positive = sum(1 for s in oos_sharpes if s > 0)

                    regime_analysis[regime] = {
                        "n_windows": len(entries),
                        "avg_oos_sharpe": round(float(np.nanmean(oos_sharpes)), 4) if oos_sharpes else 0.0,
                        "consistency": round(n_positive / len(oos_sharpes), 4) if oos_sharpes else 0.0,
                        "avg_return_pct": round(float(np.mean(oos_returns)), 4) if oos_returns else 0.0,
                    }

                logger.info(
                    "Sprint 15b : regime_analysis = {}",
                    {r: f"n={d['n_windows']}, sharpe={d['avg_oos_sharpe']:.2f}" for r, d in regime_analysis.items()},
                )

        return WFOResult(
            strategy_name=strategy_name,
            symbol=symbol,
            windows=window_results,
            avg_is_sharpe=avg_is,
            avg_oos_sharpe=avg_oos,
            oos_is_ratio=oos_is_ratio,
            consistency_rate=consistency,
            recommended_params=recommended,
            all_oos_trades=all_oos_trades,
            n_distinct_combos=n_distinct_combos,
            combo_results=combo_results,
            window_regimes=window_regimes,
            regime_analysis=regime_analysis,
        )

    def _build_windows(
        self,
        data_start: datetime,
        data_end: datetime,
        is_days: int,
        oos_days: int,
        step_days: int,
    ) -> list[tuple[datetime, datetime, datetime, datetime]]:
        """Construit les fenêtres IS+OOS glissantes."""
        windows = []
        current_start = data_start

        while True:
            is_start = current_start
            is_end = is_start + timedelta(days=is_days)
            oos_start = is_end
            oos_end = oos_start + timedelta(days=oos_days)

            if oos_end > data_end:
                break

            windows.append((is_start, is_end, oos_start, oos_end))
            current_start += timedelta(days=step_days)

        return windows

    def _parallel_backtest(
        self,
        grid: list[dict[str, Any]],
        candles_by_tf: dict[str, list[Candle]],
        strategy_name: str,
        symbol: str,
        bt_config_dict: dict,
        main_tf: str,
        n_workers: int,
        metric: str,
        extra_data_map: dict[str, dict[str, Any]] | None = None,
    ) -> list[_ISResult]:
        """Lance les backtests avec chaîne de fallback :

        1. Fast engine (pré-calcul indicateurs + boucle minimale)
        2. ProcessPool (parallèle, si fast échoue)
        3. Séquentiel (si pool crashe)
        """
        # 1. Tenter le fast engine (stratégies supportées uniquement)
        if strategy_name in ("vwap_rsi", "momentum", "bollinger_mr", "donchian_breakout", "supertrend", "envelope_dca", "envelope_dca_short"):
            try:
                results = self._run_fast(
                    grid, candles_by_tf, strategy_name, bt_config_dict, main_tf,
                )
                # Trier par métrique
                metric_idx = {"sharpe_ratio": 1, "net_return_pct": 2, "profit_factor": 3}
                sort_idx = metric_idx.get(metric, 1)
                results.sort(key=lambda r: r[sort_idx], reverse=True)
                return results
            except Exception as exc:
                logger.warning(
                    "Fast engine échoué ({}), fallback pool/séquentiel...",
                    exc,
                )

        # 2. Fallback : pool + séquentiel (code existant)
        # Sérialiser les candles ici seulement (pas avant, pour économiser la mémoire)
        candles_serialized = _serialize_candles_by_tf(candles_by_tf) if n_workers > 1 else {}
        results: list[_ISResult] = []
        remaining_grid = list(grid)

        if n_workers > 1:
            results, remaining_grid = self._run_pool(
                grid, candles_serialized, strategy_name, symbol,
                bt_config_dict, main_tf, n_workers,
                extra_data_map=extra_data_map,
            )
            # Libérer les candles sérialisées dès que le pool est fini
            del candles_serialized
            gc.collect()
            if remaining_grid:
                logger.info(
                    "  Continuation séquentielle : {} combos restantes...",
                    len(remaining_grid),
                )

        if remaining_grid:
            seq_results = self._run_sequential(
                remaining_grid, candles_by_tf, strategy_name, bt_config_dict, main_tf,
                extra_data_map=extra_data_map,
            )
            results.extend(seq_results)

        # Trier par métrique
        metric_idx = {"sharpe_ratio": 1, "net_return_pct": 2, "profit_factor": 3}
        sort_idx = metric_idx.get(metric, 1)
        results.sort(key=lambda r: r[sort_idx], reverse=True)

        return results

    @staticmethod
    def _run_fast(
        grid: list[dict[str, Any]],
        candles_by_tf: dict[str, list[Candle]],
        strategy_name: str,
        bt_config_dict: dict,
        main_tf: str,
    ) -> list[_ISResult]:
        """Fast engine : pré-calcul indicateurs + boucle de trades minimale.

        ~200x plus rapide que le moteur normal pour le grid search IS.
        Cache construit une seule fois, puis chaque combo = seuils numpy + boucle légère.
        """
        from backend.optimization.fast_backtest import run_backtest_from_cache
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache
        from backend.optimization.indicator_cache import build_cache
        from backend.optimization import is_grid_strategy

        bt_config = BacktestConfig(**bt_config_dict)

        # Extraire les valeurs du grid pour le cache (toutes les variantes)
        param_grid_values: dict[str, list] = {}
        for params in grid:
            for k, v in params.items():
                if k not in param_grid_values:
                    param_grid_values[k] = []
                if v not in param_grid_values[k]:
                    param_grid_values[k].append(v)

        # Construire le cache (une seule fois)
        t0 = time.monotonic()
        cache = build_cache(candles_by_tf, param_grid_values, strategy_name, main_tf)
        cache_time = time.monotonic() - t0
        logger.info(
            "  Fast cache: {} bougies, {:.1f}ms",
            cache.n_candles, cache_time * 1000,
        )

        # Exécuter tous les combos
        results: list[_ISResult] = []
        total = len(grid)
        t_start = time.monotonic()

        is_grid = is_grid_strategy(strategy_name)

        for i, params in enumerate(grid):
            if is_grid:
                results.append(run_multi_backtest_from_cache(strategy_name, params, cache, bt_config))
            else:
                results.append(run_backtest_from_cache(strategy_name, params, cache, bt_config))
            if (i + 1) % 50 == 0 or i == total - 1:
                elapsed = time.monotonic() - t_start
                avg_ms = elapsed / (i + 1) * 1000
                remaining = (total - i - 1) * elapsed / (i + 1)
                logger.info(
                    "  Fast {}/{} — {:.1f}ms/bt — {:.1f}s restant",
                    i + 1, total, avg_ms, remaining,
                )

        # Libérer le cache numpy explicitement (évite accumulation mémoire sur 20 fenêtres)
        del cache
        gc.collect()

        return results

    @staticmethod
    def _run_pool(
        grid: list[dict[str, Any]],
        candles_serialized: dict[str, list[dict]],
        strategy_name: str,
        symbol: str,
        bt_config_dict: dict,
        main_tf: str,
        n_workers: int,
        extra_data_map: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[list[_ISResult], list[dict[str, Any]]]:
        """Exécution parallèle par lots avec cooldown.

        Retourne (résultats obtenus, combos restantes non traitées).
        Si le pool crash, les résultats des lots terminés sont conservés.
        """
        batch_size = 20  # 20 tasks par lot
        total = len(grid)

        results: list[_ISResult] = []
        n_batches = (total + batch_size - 1) // batch_size
        done_count = 0
        t_start = time.monotonic()

        try:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=(candles_serialized, strategy_name, symbol, bt_config_dict, main_tf, extra_data_map),
                max_tasks_per_child=50,
            ) as executor:
                for batch_idx in range(n_batches):
                    batch_start = batch_idx * batch_size
                    batch = grid[batch_start : batch_start + batch_size]
                    chunksize = max(1, len(batch) // (n_workers * 2))

                    logger.info(
                        "    Lot {}/{} ({} backtests)...",
                        batch_idx + 1, n_batches, len(batch),
                    )

                    for result in executor.map(
                        _run_single_backtest_worker, batch, chunksize=chunksize,
                    ):
                        results.append(result)
                        done_count += 1

                        # Log progression toutes les 10 complétions ou fin de lot
                        if done_count % 10 == 0 or done_count == total:
                            elapsed = time.monotonic() - t_start
                            avg_s = elapsed / done_count
                            remaining = (total - done_count) * avg_s
                            eta_min, eta_sec = divmod(int(remaining), 60)
                            logger.info(
                                "    Pool {}/{} — {:.1f}s/bt — ETA: {}m{:02d}s",
                                done_count, total, avg_s, eta_min, eta_sec,
                            )

        except (BrokenExecutor, OSError) as exc:
            logger.warning(
                "ProcessPool crashé ({}) après {} backtests, "
                "continuation séquentielle pour les {} restants...",
                type(exc).__name__, done_count, len(grid) - done_count,
            )

        remaining = grid[done_count:]
        return results, remaining

    @staticmethod
    def _run_sequential(
        grid: list[dict[str, Any]],
        candles_by_tf: dict[str, list[Candle]],
        strategy_name: str,
        bt_config_dict: dict,
        main_tf: str,
        extra_data_map: dict[str, dict[str, Any]] | None = None,
    ) -> list[_ISResult]:
        """Exécution séquentielle avec pré-calcul des indicateurs par groupe.

        Groupe les combinaisons par paramètres qui affectent compute_indicators()
        (ex: rsi_period pour vwap_rsi). Les indicateurs sont calculés une seule fois
        par groupe, puis réutilisés pour toutes les combinaisons de seuils.
        """
        from backend.optimization import create_strategy_with_params

        bt_config = BacktestConfig(**bt_config_dict)
        indicator_keys = _INDICATOR_PARAMS.get(strategy_name, [])

        # Grouper les combos par paramètres indicateurs
        groups: dict[tuple, list[dict[str, Any]]] = {}
        for params in grid:
            key = tuple(params.get(k) for k in indicator_keys) if indicator_keys else ()
            groups.setdefault(key, []).append(params)

        n_groups = len(groups)
        total = len(grid)
        logger.info(
            "  {} groupes d'indicateurs ({} combos total)",
            n_groups, total,
        )

        # Couper les logs répétitifs en mode séquentiel (engine + stratégies)
        import logging as _logging
        _quiet_loggers = [
            _logging.getLogger("backend.backtesting.engine"),
            _logging.getLogger("backend.strategies"),
        ]
        _prev_levels = [lg.level for lg in _quiet_loggers]
        for lg in _quiet_loggers:
            lg.setLevel(_logging.WARNING)

        results: list[_ISResult] = []
        done = 0
        t_start = time.monotonic()

        for group_idx, (ind_key, combos) in enumerate(groups.items()):
            # Pré-calculer les indicateurs pour ce groupe
            representative = combos[0]
            strategy = create_strategy_with_params(strategy_name, representative)
            logger.info(
                "  Groupe {}/{} ({} combos) — indicateurs: {}",
                group_idx + 1, n_groups, len(combos),
                dict(zip(indicator_keys, ind_key)) if indicator_keys else "aucun",
            )
            precomputed = strategy.compute_indicators(candles_by_tf)

            for params in combos:
                results.append(
                    _run_single_backtest_sequential(
                        params, candles_by_tf, strategy_name, bt_config, main_tf,
                        precomputed_indicators=precomputed,
                        extra_data_by_timestamp=extra_data_map,
                    )
                )
                done += 1
                if done % 10 == 0 or done == total:
                    elapsed = time.monotonic() - t_start
                    avg_s = elapsed / done
                    remaining = (total - done) * avg_s
                    eta_min, eta_sec = divmod(int(remaining), 60)
                    logger.info(
                        "  Backtest {}/{} — {:.1f}s/bt — ETA: {}m{:02d}s",
                        done, total, avg_s, eta_min, eta_sec,
                    )

            # Libérer mémoire entre groupes
            del precomputed
            gc.collect()

        for lg, lvl in zip(_quiet_loggers, _prev_levels):
            lg.setLevel(lvl)
        return results
