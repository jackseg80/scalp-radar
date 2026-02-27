"""Tests Sprint 40b #5 : Guard max DD -80% dans le fast engine.

Si le capital chute de plus de 80% par rapport au peak, la simulation
s'arrête prématurément (break). Cela évite de scorer positivement des
combos qui explosent en cours de simulation.
"""
from __future__ import annotations

import numpy as np
import pytest
from datetime import datetime

from backend.backtesting.engine import BacktestConfig


def _make_bt_config(**kwargs) -> BacktestConfig:
    defaults = dict(
        symbol="BTC/USDT",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1),
        initial_capital=10_000.0,
        leverage=6,
        maker_fee=0.0002,
        taker_fee=0.0006,
        slippage_pct=0.0005,
    )
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


# ─── Test champ max_wfo_drawdown_pct ──────────────────────────────────────


class TestBacktestConfigDDGuard:

    def test_default_dd_guard_is_80(self):
        """BacktestConfig.max_wfo_drawdown_pct = 80.0 par défaut."""
        bt = _make_bt_config()
        assert bt.max_wfo_drawdown_pct == 80.0

    def test_custom_dd_guard_accepted(self):
        """max_wfo_drawdown_pct peut être surchargé."""
        bt = _make_bt_config(max_wfo_drawdown_pct=50.0)
        assert bt.max_wfo_drawdown_pct == 50.0


# ─── Tests comportement guard DD dans _simulate_grid_common ───────────────


class TestDDGuardBehavior:

    def test_normal_strategy_not_stopped(self, make_indicator_cache):
        """Stratégie normale (DD < 80%) → simulation complète, aucun arrêt."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_common

        n = 50
        ma_period = 5
        sma = np.full(n, 100.0)
        closes = np.full(n, 95.0)
        lows = np.full(n, 90.0)
        highs = np.full(n, 99.0)
        highs[-1] = 105.0  # TP final

        entry_prices = np.full((n, 1), np.nan)
        for i in range(ma_period + 1, n):
            entry_prices[i, 0] = 95.0  # lows=90 ≤ 95 → entry

        cache = make_indicator_cache(
            n=n, closes=closes, lows=lows, highs=highs,
            bb_sma={ma_period: sma},
        )

        bt = _make_bt_config(
            initial_capital=10_000.0,
            leverage=6,
            max_wfo_drawdown_pct=80.0,
        )

        pnls, returns, final_capital = _simulate_grid_common(
            entry_prices, sma, cache, bt,
            num_levels=1, sl_pct=0.30, direction=1,
        )

        # Capital final > 0 (pas liquidé)
        assert final_capital > 0, f"Capital final doit être > 0, got {final_capital}"

    def test_progressive_dd_stops_simulation(self, make_indicator_cache):
        """Guard DD 15% stoppe avant le kill switch 25% (Sprint 53).

        Avec leverage=2 et sl_pct=0.05, chaque SL perd ~10% du capital courant.
        Guard 15% : stoppe après ~2 trades (DD ≈ 19%).
        Kill switch 25% (sans guard) : stoppe après ~3 trades (DD ≈ 27%).
        """
        from backend.optimization.fast_multi_backtest import _simulate_grid_common

        n = 200
        ma_period = 5
        sma = np.full(n, 100.0)
        closes = np.full(n, 95.0)
        # SL déclenché immédiatement : sl_price = entry*(1-0.05) = 90.25
        # lows=80 ≤ 90.25 → SL à chaque bougie après entrée
        lows = np.full(n, 80.0)
        highs = np.full(n, 99.0)  # < SMA=100 → pas de TP

        entry_prices = np.full((n, 1), np.nan)
        for i in range(ma_period + 1, n):
            entry_prices[i, 0] = 95.0

        cache = make_indicator_cache(
            n=n, closes=closes, lows=lows, highs=highs,
            bb_sma={ma_period: sma},
        )

        # leverage=2, sl_pct=0.05 : perte ~10%/trade (du capital courant)
        # Guard à 15% : stoppe avant le kill switch à 25%
        bt_with_guard = _make_bt_config(
            initial_capital=10_000.0,
            leverage=2,
            maker_fee=0.0,
            taker_fee=0.0,
            slippage_pct=0.0,
            max_wfo_drawdown_pct=15.0,
        )

        bt_without_guard = _make_bt_config(
            initial_capital=10_000.0,
            leverage=2,
            maker_fee=0.0,
            taker_fee=0.0,
            slippage_pct=0.0,
            max_wfo_drawdown_pct=9999.0,  # Guard désactivé, kill switch 25% reste actif
        )

        pnls_with, _, cap_with = _simulate_grid_common(
            entry_prices, sma, cache, bt_with_guard,
            num_levels=1, sl_pct=0.05, direction=1,
        )

        pnls_without, _, cap_without = _simulate_grid_common(
            entry_prices, sma, cache, bt_without_guard,
            num_levels=1, sl_pct=0.05, direction=1,
        )

        # Guard 15% stoppe avant kill switch 25%
        assert len(pnls_with) < len(pnls_without), (
            f"Guard DD 15% doit arrêter avant kill switch 25%. "
            f"Avec guard: {len(pnls_with)} trades, sans: {len(pnls_without)}"
        )
        # Le guard 15% doit s'arrêter après ~2-3 trades
        assert len(pnls_with) <= 5, (
            f"Guard 15% doit s'arrêter dans les 5 premiers trades, "
            f"got {len(pnls_with)}"
        )

    def test_dd_guard_propagated_in_walk_forward_config(self):
        """walk_forward.py propage max_wfo_drawdown_pct dans bt_config_dict."""
        import inspect
        from backend.optimization.walk_forward import WalkForwardOptimizer
        source = inspect.getsource(WalkForwardOptimizer.optimize)
        assert "max_wfo_drawdown_pct" in source, (
            "walk_forward.py doit propager max_wfo_drawdown_pct dans bt_config_dict"
        )

    def test_risk_yaml_has_dd_guard(self):
        """risk.yaml contient max_wfo_drawdown_pct."""
        import yaml
        with open("config/risk.yaml") as f:
            data = yaml.safe_load(f)
        assert "max_wfo_drawdown_pct" in data, (
            "risk.yaml doit avoir max_wfo_drawdown_pct"
        )
        assert data["max_wfo_drawdown_pct"] == 80
