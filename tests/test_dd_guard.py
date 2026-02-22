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
        """Pertes progressives (~20%/trade) → guard 80% s'arrête avant guard 9999%.

        Avec leverage=2 et sl_pct=0.1, chaque SL perd ~20% du capital courant.
        Après ~8 trades, DD peak→courant dépasse 80% → guard déclenche.
        Sans guard (9999%), la simulation continue sur 90+ bougies.
        """
        from backend.optimization.fast_multi_backtest import _simulate_grid_common

        n = 200
        ma_period = 5
        sma = np.full(n, 100.0)
        closes = np.full(n, 95.0)
        # SL déclenché immédiatement : sl_price = entry*(1-0.1) = 85.5
        # lows=80 ≤ 85.5 → SL à chaque bougie après entrée
        lows = np.full(n, 80.0)
        highs = np.full(n, 99.0)  # < SMA=100 → pas de TP

        entry_prices = np.full((n, 1), np.nan)
        for i in range(ma_period + 1, n):
            entry_prices[i, 0] = 95.0

        cache = make_indicator_cache(
            n=n, closes=closes, lows=lows, highs=highs,
            bb_sma={ma_period: sma},
        )

        # leverage=2, sl_pct=0.1 : perte ~20%/trade (du capital courant)
        # → ~8 trades pour DD > 80% du peak
        bt_with_guard = _make_bt_config(
            initial_capital=10_000.0,
            leverage=2,
            maker_fee=0.0,
            taker_fee=0.0,
            slippage_pct=0.0,
            max_wfo_drawdown_pct=80.0,
        )

        bt_without_guard = _make_bt_config(
            initial_capital=10_000.0,
            leverage=2,
            maker_fee=0.0,
            taker_fee=0.0,
            slippage_pct=0.0,
            max_wfo_drawdown_pct=9999.0,  # Guard essentiellement désactivé
        )

        pnls_with, _, cap_with = _simulate_grid_common(
            entry_prices, sma, cache, bt_with_guard,
            num_levels=1, sl_pct=0.1, direction=1,
        )

        pnls_without, _, cap_without = _simulate_grid_common(
            entry_prices, sma, cache, bt_without_guard,
            num_levels=1, sl_pct=0.1, direction=1,
        )

        # Avec guard DD : ~8 trades puis arrêt
        # Sans guard : ~95 trades (200-2*ma_period bougies = ~190/2=95 cycles)
        assert len(pnls_with) < len(pnls_without), (
            f"Guard DD doit arrêter la simulation tôt. "
            f"Avec guard: {len(pnls_with)} trades, sans: {len(pnls_without)}"
        )
        # Le guard 80% doit s'arrêter après ~8-12 trades
        assert len(pnls_with) <= 15, (
            f"Guard 80% doit s'arrêter dans les 15 premiers trades, "
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
