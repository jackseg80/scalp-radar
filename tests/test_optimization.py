"""Tests du module d'optimisation (Sprint 7).

Couvre : registre stratégies, grid, WFO dataclasses, overfitting detection
(Monte Carlo, DSR, stabilité, convergence), grading, report, per_asset config,
migration DB, run_backtest_single.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from backend.backtesting.engine import BacktestConfig, BacktestResult, run_backtest_single
from backend.backtesting.metrics import calculate_metrics
from backend.core.config import MomentumConfig, VwapRsiConfig
from backend.core.models import Candle, Direction, MarketRegime, TimeFrame
from backend.core.position_manager import TradeResult
from backend.optimization.overfitting import (
    ConvergenceResult,
    DSRResult,
    MonteCarloResult,
    OverfitDetector,
    StabilityResult,
)
from backend.optimization.report import (
    FinalReport,
    ValidationResult,
    apply_to_yaml,
    build_final_report,
    compute_grade,
    save_report,
    _bootstrap_sharpe_ci,
)
from scripts.optimize import apply_from_db
from backend.optimization.walk_forward import (
    WFOResult,
    WindowResult,
    _build_grid,
    _fine_grid_around_top,
    _latin_hypercube_sample,
    _median_params,
    _slice_candles,
)


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_candle(
    close: float, ts: datetime, symbol: str = "BTC/USDT",
    tf: TimeFrame = TimeFrame.M5, exchange: str = "bitget",
) -> Candle:
    return Candle(
        timestamp=ts, open=close * 0.999, high=close * 1.001,
        low=close * 0.998, close=close, volume=100.0,
        symbol=symbol, timeframe=tf, exchange=exchange,
    )


def _make_candles(n: int, base_price: float = 50000.0, symbol: str = "BTC/USDT",
                  tf: TimeFrame = TimeFrame.M5, exchange: str = "bitget") -> list[Candle]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    delta = timedelta(minutes=5) if tf == TimeFrame.M5 else timedelta(minutes=15)
    candles = []
    for i in range(n):
        price = base_price + i * 10 * np.sin(i * 0.1)
        candles.append(_make_candle(price, start + delta * i, symbol, tf, exchange))
    return candles


def _make_trade(net_pnl: float, idx: int = 0) -> TradeResult:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=idx)
    return TradeResult(
        direction=Direction.LONG,
        entry_price=50000.0,
        exit_price=50000.0 + net_pnl * 10,
        quantity=0.01,
        entry_time=ts,
        exit_time=ts + timedelta(minutes=30),
        gross_pnl=net_pnl + 1.0,
        fee_cost=0.5,
        slippage_cost=0.5,
        net_pnl=net_pnl,
        exit_reason="tp" if net_pnl > 0 else "sl",
        market_regime=MarketRegime.RANGING,
    )


# ─── Tests Registre → centralisés dans test_strategy_registry.py ──────────


# ─── Tests Grid ────────────────────────────────────────────────────────────


class TestGrid:
    def test_build_grid_default(self):
        grids = {"default": {"a": [1, 2], "b": [10, 20]}}
        result = _build_grid(grids, "BTC/USDT")
        assert len(result) == 4  # 2 × 2

    def test_build_grid_with_symbol_override(self):
        grids = {
            "default": {"a": [1, 2], "b": [10, 20]},
            "BTC/USDT": {"b": [10, 20, 30]},
        }
        result = _build_grid(grids, "BTC/USDT")
        assert len(result) == 6  # 2 × 3

    def test_build_grid_empty(self):
        assert _build_grid({}, "BTC/USDT") == []

    def test_latin_hypercube_sample(self):
        grid = [{"a": i} for i in range(1000)]
        sample = _latin_hypercube_sample(grid, 100)
        assert len(sample) == 100

    def test_latin_hypercube_small_grid(self):
        grid = [{"a": i} for i in range(5)]
        sample = _latin_hypercube_sample(grid, 100)
        assert len(sample) == 5  # Retourne tout si grid < n_samples

    def test_fine_grid_around_top(self):
        top = [{"a": 2, "b": 20}]
        grid_values = {"a": [1, 2, 3], "b": [10, 20, 30]}
        fine = _fine_grid_around_top(top, grid_values)
        assert len(fine) > 0
        # Doit contenir les voisins
        param_a_values = {c["a"] for c in fine}
        assert 1 in param_a_values  # a-1
        assert 2 in param_a_values  # a
        assert 3 in param_a_values  # a+1

    def test_median_params(self):
        params_list = [
            {"rsi": 14, "sl": 0.3},
            {"rsi": 20, "sl": 0.5},
            {"rsi": 10, "sl": 0.4},
        ]
        grid_values = {"rsi": [10, 14, 20], "sl": [0.3, 0.4, 0.5]}
        result = _median_params(params_list, grid_values)
        assert result["rsi"] == 14  # médiane
        assert result["sl"] == 0.4  # médiane

    def test_slice_candles(self):
        candles = _make_candles(100)
        start = candles[10].timestamp
        end = candles[50].timestamp
        sliced = _slice_candles(candles, start, end)
        assert len(sliced) == 40  # [10, 50)
        assert sliced[0].timestamp == start


# ─── Tests Monte Carlo ─────────────────────────────────────────────────────


class TestMonteCarlo:
    def test_mc_with_profitable_trades(self):
        trades = [_make_trade(10.0, i) for i in range(50)]
        detector = OverfitDetector()
        result = detector.monte_carlo_block_bootstrap(trades, n_sims=500, seed=42)
        assert isinstance(result, MonteCarloResult)
        assert 0 <= result.p_value <= 1
        assert result.real_sharpe > 0

    def test_mc_with_random_trades(self):
        rng = np.random.default_rng(123)
        trades = [_make_trade(rng.normal(0, 5), i) for i in range(50)]
        detector = OverfitDetector()
        result = detector.monte_carlo_block_bootstrap(trades, n_sims=500, seed=42)
        # P-value devrait être élevée pour des trades random
        assert result.p_value > 0.01

    def test_mc_few_trades(self):
        trades = [_make_trade(10.0, i) for i in range(3)]
        detector = OverfitDetector()
        result = detector.monte_carlo_block_bootstrap(trades)
        assert result.p_value == 1.0
        assert not result.significant

    def test_mc_reproducible_with_seed(self):
        trades = [_make_trade(10.0, i) for i in range(30)]
        detector = OverfitDetector()
        r1 = detector.monte_carlo_block_bootstrap(trades, seed=42)
        r2 = detector.monte_carlo_block_bootstrap(trades, seed=42)
        assert r1.p_value == r2.p_value

    def test_mc_underpowered_10_trades(self):
        """< 30 trades → underpowered=True, p=0.50, pas de simulation."""
        trades = [_make_trade(10.0, i) for i in range(10)]
        detector = OverfitDetector()
        result = detector.monte_carlo_block_bootstrap(trades, seed=42)
        assert result.underpowered is True
        assert result.p_value == 0.50
        assert not result.significant
        assert result.distribution == []
        assert result.real_sharpe > 0  # trades profitables

    def test_mc_underpowered_25_trades(self):
        """25 trades (< 30) → underpowered=True, p=0.50."""
        trades = [_make_trade(10.0, i) for i in range(25)]
        detector = OverfitDetector()
        result = detector.monte_carlo_block_bootstrap(trades, seed=42)
        assert result.underpowered is True
        assert result.p_value == 0.50
        assert result.distribution == []

    def test_mc_30_trades_not_underpowered(self):
        """30 trades (>= 30) → test MC complet, block_size=7."""
        trades = [_make_trade(10.0, i) for i in range(30)]
        detector = OverfitDetector()
        result = detector.monte_carlo_block_bootstrap(trades, n_sims=300, seed=42)
        assert not result.underpowered
        assert len(result.distribution) == 300

    def test_mc_block_size_default_7(self):
        """Le block_size par défaut est 7 pour tous les cas >= 30 trades."""
        trades = [_make_trade(5.0, i) for i in range(150)]
        detector = OverfitDetector()
        r_default = detector.monte_carlo_block_bootstrap(trades, n_sims=300, seed=42)
        r_explicit = detector.monte_carlo_block_bootstrap(trades, n_sims=300, seed=42, block_size=7)
        assert r_default.p_value == r_explicit.p_value
        assert not r_default.underpowered

    def test_mc_observed_sharpe_high_gives_low_pvalue(self):
        """Quand observed_sharpe (best combo OOS) est élevé, p-value doit être bas.

        Bug historique : MC ne recevait pas observed_sharpe → comparait les shuffles
        au real_sharpe calculé depuis des trades mélangés (multi-params) → p-value
        artificiellement élevé (0.889 pour DOGE avec 93% consistance).
        """
        # Trades modérément profitables (mix de params, comme wfo.all_oos_trades)
        rng = np.random.default_rng(99)
        trades = [_make_trade(rng.normal(2.0, 5.0), i) for i in range(300)]
        detector = OverfitDetector()

        # Sans observed_sharpe : real_sharpe est calculé depuis les trades mixtes
        r_without = detector.monte_carlo_block_bootstrap(trades, n_sims=500, seed=42)

        # Avec observed_sharpe élevé (best combo OOS Sharpe = 7.40, comme DOGE)
        r_with = detector.monte_carlo_block_bootstrap(
            trades, n_sims=500, seed=42, observed_sharpe=7.40,
        )

        # Avec un observed_sharpe élevé, très peu de shuffles le dépassent
        assert r_with.p_value < 0.5, (
            f"p-value ({r_with.p_value:.3f}) devrait être < 0.5 avec observed_sharpe=7.40"
        )
        assert r_with.real_sharpe == 7.40


# ─── Tests DSR ─────────────────────────────────────────────────────────────


class TestDSR:
    def test_dsr_high_sharpe(self):
        trades = [_make_trade(20.0, i) for i in range(100)]
        detector = OverfitDetector()
        result = detector.deflated_sharpe_ratio(
            observed_sharpe=2.5, n_trials=500, n_trades=100, trades=trades,
        )
        assert isinstance(result, DSRResult)
        assert 0 <= result.dsr <= 1
        assert result.n_trials == 500

    def test_dsr_low_sharpe(self):
        trades = [_make_trade(0.1, i) for i in range(50)]
        detector = OverfitDetector()
        result = detector.deflated_sharpe_ratio(
            observed_sharpe=0.1, n_trials=700, n_trades=50, trades=trades,
        )
        # Sharpe faible + beaucoup de trials → DSR faible
        assert result.dsr < 0.5

    def test_dsr_few_trades(self):
        trades = [_make_trade(10.0)]
        detector = OverfitDetector()
        result = detector.deflated_sharpe_ratio(
            observed_sharpe=1.0, n_trials=100, n_trades=1, trades=trades,
        )
        assert result.dsr == 0.0

    def test_expected_max_sharpe(self):
        ems = OverfitDetector._expected_max_sharpe(700)
        assert ems > 2.0  # sqrt(2*log(700)) ≈ 3.6

    def test_dsr_oos_sharpe_not_inflated_by_is(self):
        """Quand OOS > IS, le DSR avec OOS Sharpe ne doit pas être artificiellement bas.

        Bug historique : observed_sharpe=IS (plus bas) faisait chuter le DSR
        alors que le OOS Sharpe est supérieur.
        """
        trades = [_make_trade(15.0, i) for i in range(100)]
        detector = OverfitDetector()

        # DSR avec IS Sharpe (plus bas) — ancien comportement bugué
        dsr_is = detector.deflated_sharpe_ratio(
            observed_sharpe=1.0, n_trials=500, n_trades=100, trades=trades,
        )
        # DSR avec OOS Sharpe (plus haut) — comportement correct
        dsr_oos = detector.deflated_sharpe_ratio(
            observed_sharpe=2.5, n_trials=500, n_trades=100, trades=trades,
        )

        # Le DSR avec le OOS Sharpe (plus élevé) doit être >= DSR avec IS
        assert dsr_oos.dsr >= dsr_is.dsr, (
            f"DSR(oos={dsr_oos.dsr:.3f}) devrait être >= DSR(is={dsr_is.dsr:.3f})"
        )


# ─── Tests Stabilité ──────────────────────────────────────────────────────


class TestStability:
    def test_stability_basic(self):
        # Utiliser des candles synthétiques
        candles_5m = _make_candles(400, symbol="BTC/USDT", tf=TimeFrame.M5)
        candles_15m = _make_candles(100, symbol="BTC/USDT", tf=TimeFrame.M15)
        candles_by_tf = {"5m": candles_5m, "15m": candles_15m}
        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=candles_5m[0].timestamp,
            end_date=candles_5m[-1].timestamp,
        )
        params = {"rsi_period": 14, "tp_percent": 0.8, "sl_percent": 0.3}

        detector = OverfitDetector()
        result = detector.parameter_stability(
            "vwap_rsi", "BTC/USDT", params, candles_by_tf, bt_config,
        )
        assert isinstance(result, StabilityResult)
        assert "rsi_period" in result.stability_map


# ─── Tests Convergence ────────────────────────────────────────────────────


class TestConvergence:
    def test_convergence_identical_params(self):
        detector = OverfitDetector()
        params = {
            "BTC/USDT": {"rsi": 14, "sl": 0.3},
            "ETH/USDT": {"rsi": 14, "sl": 0.3},
            "SOL/USDT": {"rsi": 14, "sl": 0.3},
        }
        result = detector.cross_asset_convergence(params)
        assert result.convergence_score == 1.0
        assert len(result.divergent_params) == 0

    def test_convergence_divergent_params(self):
        detector = OverfitDetector()
        # Valeurs très divergentes pour que CV > 0.5 → score < 0.5
        params = {
            "BTC/USDT": {"rsi": 5, "sl": 0.1},
            "ETH/USDT": {"rsi": 30, "sl": 0.9},
            "SOL/USDT": {"rsi": 50, "sl": 1.5},
        }
        result = detector.cross_asset_convergence(params)
        assert result.convergence_score < 0.8
        assert len(result.divergent_params) > 0

    def test_convergence_single_asset(self):
        detector = OverfitDetector()
        result = detector.cross_asset_convergence({"BTC/USDT": {"rsi": 14}})
        assert result.convergence_score == 1.0


# ─── Tests Grading ─────────────────────────────────────────────────────────


class TestGrading:
    def test_grade_a(self):
        grade, score = compute_grade(
            oos_is_ratio=0.65, mc_p_value=0.01, dsr=0.97,
            stability=0.85, bitget_transfer=0.60,
        )
        assert grade == "A"
        assert score == 100

    def test_grade_f(self):
        grade, score = compute_grade(
            oos_is_ratio=0.1, mc_p_value=0.5, dsr=0.3,
            stability=0.2, bitget_transfer=0.1,
            consistency=0.0,
        )
        assert grade == "F"
        assert score == 0

    def test_grade_c(self):
        grade, score = compute_grade(
            oos_is_ratio=0.45, mc_p_value=0.08, dsr=0.85,
            stability=0.65, bitget_transfer=0.35,
        )
        assert grade in ("C", "D")  # ~55 points

    def test_grade_boundary_b(self):
        grade, score = compute_grade(
            oos_is_ratio=0.55, mc_p_value=0.03, dsr=0.92,
            stability=0.70, bitget_transfer=0.40,
        )
        assert grade in ("B", "C")

    def test_grade_mc_underpowered(self):
        """mc_underpowered=True → 10/20 pts MC (neutre), pas de pénalité."""
        # Même params que test_grade_a, mais avec underpowered au lieu de mc_p < 0.05
        grade_underpowered, score_underpowered = compute_grade(
            oos_is_ratio=0.65, mc_p_value=0.50, dsr=0.97,
            stability=0.85, bitget_transfer=0.60,
            mc_underpowered=True,
        )
        # Sans underpowered, mc_p=0.50 donnerait 0/20 pts → grade plus bas
        grade_penalized, score_penalized = compute_grade(
            oos_is_ratio=0.65, mc_p_value=0.50, dsr=0.97,
            stability=0.85, bitget_transfer=0.60,
            mc_underpowered=False,
        )
        # underpowered donne 10 pts de plus que pénalisé (0 → 10)
        assert grade_underpowered < grade_penalized  # "A" < "B" alphabétiquement = meilleur grade
        # Vérifier que underpowered ne bloque pas un bon grade
        assert grade_underpowered in ("A", "B")

    def test_grade_capped_at_c_under_30_trades(self):
        """6 trades OOS, score 100 → Grade C (pas A)."""
        grade, score = compute_grade(
            oos_is_ratio=0.65, mc_p_value=0.01, dsr=0.97,
            stability=0.85, bitget_transfer=0.60,
            total_trades=6,
        )
        assert score == 100  # Score brut inchangé
        assert grade == "C"  # Plafonné à C

    def test_grade_capped_at_b_under_50_trades(self):
        """40 trades OOS, score 100 → Grade B (pas A)."""
        grade, score = compute_grade(
            oos_is_ratio=0.65, mc_p_value=0.01, dsr=0.97,
            stability=0.85, bitget_transfer=0.60,
            total_trades=40,
        )
        assert score == 100
        assert grade == "B"  # Plafonné à B

    def test_grade_not_capped_above_50_trades(self):
        """100 trades OOS, score 100 → Grade A (pas de plafond)."""
        grade, score = compute_grade(
            oos_is_ratio=0.65, mc_p_value=0.01, dsr=0.97,
            stability=0.85, bitget_transfer=0.60,
            total_trades=100,
        )
        assert score == 100
        assert grade == "A"


# ─── Tests Bootstrap CI ──────────────────────────────────────────────────


class TestBootstrapCI:
    def test_bootstrap_profitable(self):
        trades = [_make_trade(5.0, i) for i in range(50)]
        ci_low, ci_high = _bootstrap_sharpe_ci(trades, n_bootstrap=500, seed=42)
        assert ci_low < ci_high
        assert ci_low > 0  # Trades profitables → CI positive

    def test_bootstrap_few_trades(self):
        trades = [_make_trade(5.0)]
        ci_low, ci_high = _bootstrap_sharpe_ci(trades)
        assert ci_low == 0.0 and ci_high == 0.0


# ─── Tests Report ─────────────────────────────────────────────────────────


class TestReport:
    def test_save_report(self, tmp_path):
        validation = ValidationResult(
            bitget_sharpe=0.8, bitget_net_return_pct=5.0, bitget_trades=20,
            bitget_sharpe_ci_low=0.3, bitget_sharpe_ci_high=1.2,
            binance_oos_avg_sharpe=1.0, transfer_ratio=0.8,
            transfer_significant=True, volume_warning=False, volume_warning_detail="",
        )
        report = FinalReport(
            strategy_name="vwap_rsi", symbol="BTC/USDT",
            timestamp=datetime.now(), grade="A", total_score=87,
            wfo_avg_is_sharpe=1.8, wfo_avg_oos_sharpe=0.9,
            wfo_consistency_rate=0.75, wfo_n_windows=20,
            recommended_params={"rsi_period": 14}, mc_p_value=0.01,
            mc_significant=True, mc_underpowered=False, dsr=0.96, dsr_max_expected_sharpe=3.5,
            stability=0.85, cliff_params=[], convergence=0.80,
            divergent_params=[], validation=validation,
            oos_is_ratio=0.50, bitget_transfer=0.80,
            live_eligible=True, warnings=[], n_distinct_combos=700,
        )
        filepath, result_id = save_report(report, output_dir=str(tmp_path))
        assert filepath.exists()
        assert "vwap_rsi" in filepath.name

    def test_build_final_report(self):
        wfo = WFOResult(
            strategy_name="vwap_rsi", symbol="BTC/USDT",
            windows=[], avg_is_sharpe=1.5, avg_oos_sharpe=0.8,
            oos_is_ratio=0.53, consistency_rate=0.7,
            recommended_params={"rsi_period": 14},
            all_oos_trades=[], n_distinct_combos=500,
        )
        from backend.optimization.overfitting import OverfitReport
        overfit = OverfitReport(
            monte_carlo=MonteCarloResult(p_value=0.02, real_sharpe=0.8, distribution=[], significant=True),
            dsr=DSRResult(dsr=0.95, max_expected_sharpe=3.0, observed_sharpe=0.8, n_trials=500),
            stability=StabilityResult(stability_map={}, overall_stability=0.85, cliff_params=[]),
            convergence=None,
        )
        validation = ValidationResult(
            bitget_sharpe=0.7, bitget_net_return_pct=4.0, bitget_trades=15,
            bitget_sharpe_ci_low=0.2, bitget_sharpe_ci_high=1.1,
            binance_oos_avg_sharpe=0.8, transfer_ratio=0.875,
            transfer_significant=True, volume_warning=False, volume_warning_detail="",
        )
        report = build_final_report(wfo, overfit, validation)
        assert report.grade in ("A", "B", "C")
        assert report.strategy_name == "vwap_rsi"


# ─── Tests per_asset Config ──────────────────────────────────────────────


class TestPerAssetConfig:
    def test_vwap_rsi_config_per_asset(self):
        cfg = VwapRsiConfig(per_asset={"SOL/USDT": {"sl_percent": 0.5}})
        params = cfg.get_params_for_symbol("SOL/USDT")
        assert params["sl_percent"] == 0.5
        assert params["rsi_period"] == 14  # default

    def test_vwap_rsi_config_no_override(self):
        cfg = VwapRsiConfig(per_asset={"SOL/USDT": {"sl_percent": 0.5}})
        params = cfg.get_params_for_symbol("BTC/USDT")
        assert params["sl_percent"] == 0.3  # default

    def test_momentum_config_per_asset(self):
        cfg = MomentumConfig(per_asset={"DOGE/USDT": {"tp_percent": 1.2}})
        params = cfg.get_params_for_symbol("DOGE/USDT")
        assert params["tp_percent"] == 1.2

    def test_per_asset_empty(self):
        cfg = VwapRsiConfig()
        params = cfg.get_params_for_symbol("BTC/USDT")
        assert params["tp_percent"] == 0.8


# ─── Tests resolve_param ─────────────────────────────────────────────────


class TestResolveParam:
    def test_resolve_with_override(self):
        cfg = VwapRsiConfig(per_asset={"SOL/USDT": {"sl_percent": 0.5}})
        from backend.strategies.vwap_rsi import VwapRsiStrategy
        strategy = VwapRsiStrategy(cfg)
        assert strategy._resolve_param("sl_percent", "SOL/USDT") == 0.5

    def test_resolve_without_override(self):
        cfg = VwapRsiConfig()
        from backend.strategies.vwap_rsi import VwapRsiStrategy
        strategy = VwapRsiStrategy(cfg)
        assert strategy._resolve_param("sl_percent", "BTC/USDT") == 0.3

    def test_resolve_default(self):
        cfg = VwapRsiConfig(per_asset={})
        from backend.strategies.vwap_rsi import VwapRsiStrategy
        strategy = VwapRsiStrategy(cfg)
        assert strategy._resolve_param("tp_percent", "ETH/USDT") == 0.8


# ─── Tests Candle exchange field ─────────────────────────────────────────


class TestCandleExchange:
    def test_candle_default_exchange(self):
        candle = _make_candle(50000.0, datetime.now(tz=timezone.utc))
        assert candle.exchange == "bitget"

    def test_candle_custom_exchange(self):
        candle = _make_candle(50000.0, datetime.now(tz=timezone.utc), exchange="binance")
        assert candle.exchange == "binance"


# ─── Tests run_backtest_single ───────────────────────────────────────────


class TestRunBacktestSingle:
    def test_run_backtest_single_basic(self):
        candles_5m = _make_candles(400)
        candles_15m = _make_candles(100, tf=TimeFrame.M15)
        candles_by_tf = {"5m": candles_5m, "15m": candles_15m}
        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=candles_5m[0].timestamp,
            end_date=candles_5m[-1].timestamp,
        )
        result = run_backtest_single(
            "vwap_rsi", {"rsi_period": 14}, candles_by_tf, bt_config,
        )
        assert isinstance(result, BacktestResult)


# ─── Tests Apply YAML ───────────────────────────────────────────────────


class TestApplyYaml:
    def test_apply_creates_backup(self, tmp_path):
        yaml_path = tmp_path / "strategies.yaml"
        yaml_path.write_text(yaml.dump({
            "vwap_rsi": {
                "enabled": True, "tp_percent": 0.8, "sl_percent": 0.3,
                "per_asset": {},
            },
        }))

        validation = ValidationResult(
            bitget_sharpe=0.8, bitget_net_return_pct=5.0, bitget_trades=20,
            bitget_sharpe_ci_low=0.3, bitget_sharpe_ci_high=1.2,
            binance_oos_avg_sharpe=1.0, transfer_ratio=0.8,
            transfer_significant=True, volume_warning=False, volume_warning_detail="",
        )
        report = FinalReport(
            strategy_name="vwap_rsi", symbol="BTC/USDT",
            timestamp=datetime.now(), grade="A", total_score=87,
            wfo_avg_is_sharpe=1.8, wfo_avg_oos_sharpe=0.9,
            wfo_consistency_rate=0.75, wfo_n_windows=20,
            recommended_params={"tp_percent": 0.6, "sl_percent": 0.25},
            mc_p_value=0.01, mc_significant=True, mc_underpowered=False, dsr=0.96,
            dsr_max_expected_sharpe=3.5, stability=0.85, cliff_params=[],
            convergence=0.80, divergent_params=[],
            validation=validation, oos_is_ratio=0.50, bitget_transfer=0.80,
            live_eligible=True, warnings=[], n_distinct_combos=700,
        )
        result = apply_to_yaml([report], str(yaml_path))
        assert result is True

        # Vérifier backup
        backups = list(tmp_path.glob("strategies.yaml.bak.*"))
        assert len(backups) == 1

    def test_apply_no_eligible(self, tmp_path):
        yaml_path = tmp_path / "strategies.yaml"
        yaml_path.write_text(yaml.dump({"vwap_rsi": {"per_asset": {}}}))

        validation = ValidationResult(
            bitget_sharpe=0.1, bitget_net_return_pct=0.5, bitget_trades=5,
            bitget_sharpe_ci_low=-0.5, bitget_sharpe_ci_high=0.7,
            binance_oos_avg_sharpe=0.5, transfer_ratio=0.2,
            transfer_significant=False, volume_warning=True,
            volume_warning_detail="test",
        )
        report = FinalReport(
            strategy_name="vwap_rsi", symbol="BTC/USDT",
            timestamp=datetime.now(), grade="F", total_score=15,
            wfo_avg_is_sharpe=0.5, wfo_avg_oos_sharpe=0.1,
            wfo_consistency_rate=0.3, wfo_n_windows=10,
            recommended_params={}, mc_p_value=0.5, mc_significant=False,
            mc_underpowered=False, dsr=0.2, dsr_max_expected_sharpe=3.5, stability=0.3,
            cliff_params=["rsi_period"], convergence=None,
            divergent_params=[], validation=validation,
            oos_is_ratio=0.2, bitget_transfer=0.2,
            live_eligible=False, warnings=[], n_distinct_combos=500,
        )
        result = apply_to_yaml([report], str(yaml_path))
        assert result is False


# ─── Tests apply_from_db ──────────────────────────────────────────────────


class TestApplyFromDb:
    def test_apply_writes_per_asset(self, tmp_path):
        """2 assets Grade A + 1 Grade C → seuls les A/B sont écrits dans per_asset."""
        import json
        import sqlite3

        # Créer la DB avec 3 résultats is_latest=1
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE optimization_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                asset TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                created_at TEXT NOT NULL,
                grade TEXT NOT NULL,
                total_score REAL NOT NULL,
                best_params TEXT NOT NULL,
                is_latest INTEGER DEFAULT 1,
                duration_seconds REAL,
                oos_sharpe REAL,
                consistency REAL,
                oos_is_ratio REAL,
                dsr REAL,
                param_stability REAL,
                monte_carlo_pvalue REAL,
                mc_underpowered INTEGER DEFAULT 0,
                n_windows INTEGER NOT NULL,
                n_distinct_combos INTEGER,
                wfo_windows TEXT,
                monte_carlo_summary TEXT,
                validation_summary TEXT,
                warnings TEXT,
                source TEXT DEFAULT 'local',
                regime_analysis TEXT,
                UNIQUE(strategy_name, asset, timeframe, created_at)
            )
        """)

        # BTC → Grade A
        conn.execute(
            """INSERT INTO optimization_results
               (strategy_name, asset, timeframe, created_at, grade, total_score,
                best_params, is_latest, n_windows)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1, 12)""",
            ("envelope_dca", "BTC/USDT", "1h", "2026-02-15T10:00:00",
             "A", 87, json.dumps({"ma_period": 7, "num_levels": 3, "envelope_start": 0.07,
                                   "envelope_step": 0.03, "sl_percent": 25.0})),
        )
        # DOGE → Grade A
        conn.execute(
            """INSERT INTO optimization_results
               (strategy_name, asset, timeframe, created_at, grade, total_score,
                best_params, is_latest, n_windows)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1, 12)""",
            ("envelope_dca", "DOGE/USDT", "1h", "2026-02-15T11:00:00",
             "A", 90, json.dumps({"ma_period": 5, "num_levels": 4, "envelope_start": 0.05,
                                   "envelope_step": 0.05, "sl_percent": 20.0})),
        )
        # SOL → Grade C (ne doit PAS être dans per_asset)
        conn.execute(
            """INSERT INTO optimization_results
               (strategy_name, asset, timeframe, created_at, grade, total_score,
                best_params, is_latest, n_windows)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1, 12)""",
            ("envelope_dca", "SOL/USDT", "1h", "2026-02-15T12:00:00",
             "C", 55, json.dumps({"ma_period": 10, "num_levels": 2, "envelope_start": 0.10,
                                   "envelope_step": 0.02, "sl_percent": 30.0})),
        )
        conn.commit()
        conn.close()

        # Créer strategies.yaml avec per_asset existant (SOL doit être retiré)
        yaml_path = tmp_path / "strategies.yaml"
        yaml_path.write_text(yaml.dump({
            "envelope_dca": {
                "enabled": True,
                "leverage": 6,
                "timeframe": "1h",
                "sides": ["long"],
                "ma_period": 7,
                "per_asset": {
                    "SOL/USDT": {"ma_period": 7, "sl_percent": 30.0},
                },
            },
        }))

        # Exécuter apply_from_db
        result = apply_from_db(
            ["envelope_dca"],
            config_dir=str(tmp_path),
            db_path=db_path,
        )

        assert result is True

        # Relire le YAML
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        per_asset = data["envelope_dca"]["per_asset"]

        # BTC et DOGE doivent être présents (Grade A)
        assert "BTC/USDT" in per_asset
        assert per_asset["BTC/USDT"]["ma_period"] == 7
        assert per_asset["BTC/USDT"]["num_levels"] == 3
        assert per_asset["BTC/USDT"]["sl_percent"] == 25.0

        assert "DOGE/USDT" in per_asset
        assert per_asset["DOGE/USDT"]["ma_period"] == 5
        assert per_asset["DOGE/USDT"]["num_levels"] == 4

        # SOL ne doit PAS être présent (Grade C → retiré)
        assert "SOL/USDT" not in per_asset

        # Les champs non-optimisés doivent être préservés
        assert data["envelope_dca"]["enabled"] is True
        assert data["envelope_dca"]["leverage"] == 6
        assert data["envelope_dca"]["sides"] == ["long"]

        # Vérifier qu'un backup a été créé
        backups = list(tmp_path.glob("strategies.yaml.bak.*"))
        assert len(backups) == 1
