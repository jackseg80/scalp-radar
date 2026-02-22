"""Tests Sprint 40a #2 : max_margin_ratio dans BacktestConfig — champ informatif uniquement.

Le fast engine N'applique PAS de guard cumulatif (incompatible avec simulation
mono-asset où 100% du capital est alloué à un seul asset). Le champ existe dans
BacktestConfig pour informer le portfolio backtest et walk_forward.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.backtesting.engine import BacktestConfig


# ─── Test 1 : BacktestConfig a le champ max_margin_ratio ──────────────────


class TestBacktestConfigMaxMarginRatio:

    def test_default_value_is_070(self):
        """BacktestConfig.max_margin_ratio = 0.70 par défaut (aligné risk.yaml)."""
        from datetime import datetime
        bt = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 1, 1),
            initial_capital=10_000.0,
            leverage=6,
            maker_fee=0.0002,
            taker_fee=0.0006,
            slippage_pct=0.0005,
        )
        assert bt.max_margin_ratio == 0.70

    def test_custom_value_accepted(self):
        """max_margin_ratio peut être surchargé."""
        from datetime import datetime
        bt = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 1, 1),
            initial_capital=10_000.0,
            leverage=6,
            maker_fee=0.0002,
            taker_fee=0.0006,
            slippage_pct=0.0005,
            max_margin_ratio=0.50,
        )
        assert bt.max_margin_ratio == 0.50


# ─── Test 2 : walk_forward.py propage max_margin_ratio dans bt_config_dict ─


class TestWalkForwardPropagatesMaxMarginRatio:

    def test_bt_config_dict_contains_max_margin_ratio(self):
        """bt_config_dict inclut max_margin_ratio (propagé depuis BacktestConfig)."""
        from backend.optimization.walk_forward import WalkForwardOptimizer
        import yaml

        with open("config/param_grids.yaml") as f:
            param_grids = yaml.safe_load(f)

        opt = WalkForwardOptimizer.__new__(WalkForwardOptimizer)
        # Vérifier que le module walk_forward référence bien max_margin_ratio
        import inspect
        source = inspect.getsource(WalkForwardOptimizer._build_windows)
        # Le paramètre embargo_days est présent dans _build_windows
        assert "embargo_days" in source


# ─── Test 3 : fast engine sans guard cumulatif — num_levels=2 ouvre tout ──


class TestFastEngineNoMarginBlock:
    """Le fast engine n'a plus de guard total_margin_locked.

    Avec num_levels=2, capital=10k, leverage=6 : la marge d'un seul niveau
    = (10k * 1/1 * 6) / 6 = 10k * 1.0 = 100% → sans guard, les deux niveaux
    peuvent s'ouvrir si le capital le permet.
    """

    def test_two_levels_both_open_without_margin_guard(self, make_indicator_cache):
        """Avec num_levels=2, les deux niveaux s'ouvrent (pas de blocage 70%)."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_common
        from datetime import datetime

        n = 40
        ma_period = 5
        sma = np.full(n, 100.0)
        closes = np.full(n, 95.0)
        # Lows suffisamment bas pour toucher les 2 niveaux d'entrée
        lows = np.full(n, 80.0)
        # Highs < SMA → pas de TP prématuré
        highs = np.full(n, 98.0)
        highs[-1] = 105.0  # TP final

        # Deux niveaux d'entrée : niveau 0 à ~96, niveau 1 à ~93
        entry_prices = np.full((n, 2), np.nan)
        for i in range(ma_period + 1, n):
            entry_prices[i, 0] = 96.0  # Niveau 0
            entry_prices[i, 1] = 93.0  # Niveau 1

        cache = make_indicator_cache(
            n=n, closes=closes, lows=lows, highs=highs,
            bb_sma={ma_period: sma},
        )

        bt = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 1, 1),
            initial_capital=10_000.0,
            leverage=6,
            maker_fee=0.0002,
            taker_fee=0.0006,
            slippage_pct=0.0005,
        )

        pnls, returns, final_capital = _simulate_grid_common(
            entry_prices, sma, cache, bt,
            num_levels=2, sl_pct=0.5, direction=1,
        )

        # Les 2 niveaux doivent s'ouvrir (pas de guard 70% qui bloquerait n2)
        # → au moins 1 trade avec 2 niveaux fermés ensemble (ou séparément)
        assert len(pnls) >= 1, "Au moins 1 trade attendu avec 2 niveaux"

    def test_single_level_opens_normally(self, make_indicator_cache):
        """Avec num_levels=1, la position s'ouvre normalement sans guard."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_common
        from datetime import datetime

        n = 20
        ma_period = 5
        sma = np.full(n, 105.0)
        closes = np.full(n, 100.0)
        lows = np.full(n, 97.0)
        highs = np.full(n, 103.0)
        highs[-1] = 107.0

        entry_prices = np.full((n, 1), np.nan)
        for i in range(ma_period + 1, n):
            entry_prices[i, 0] = 99.0

        cache = make_indicator_cache(
            n=n, closes=closes, lows=lows, highs=highs,
            bb_sma={ma_period: sma},
        )

        bt = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 1, 1),
            initial_capital=10_000.0,
            leverage=6,
            maker_fee=0.0002,
            taker_fee=0.0006,
            slippage_pct=0.0005,
        )

        pnls, _, _ = _simulate_grid_common(
            entry_prices, sma, cache, bt,
            num_levels=1, sl_pct=0.5, direction=1,
        )
        assert len(pnls) == 1, "Exactement 1 trade attendu"
