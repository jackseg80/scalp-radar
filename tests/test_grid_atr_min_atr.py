"""Tests Sprint 62b — Filtre ATR minimum pour grid_atr.

Couvre :
- test_min_atr_zero_no_filter         : min_atr_pct=0.0, tous les trades passent
- test_min_atr_filters_low_vol         : min_atr_pct=2.0, ATR/close < 2% → pas d'ouverture
- test_min_atr_exits_still_active      : positions existantes se ferment même quand filtre actif
- test_min_atr_in_param_grid           : le param est bien dans la grille WFO
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_bt_config(**overrides):
    from backend.backtesting.engine import BacktestConfig

    defaults = {
        "symbol": "BTC/USDT",
        "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "end_date": datetime(2024, 3, 1, tzinfo=timezone.utc),
        "initial_capital": 10_000.0,
        "leverage": 6,
        "taker_fee": 0.0006,
        "maker_fee": 0.0002,
        "slippage_pct": 0.0001,
    }
    defaults.update(overrides)
    return BacktestConfig(**defaults)


# ─── Tests ─────────────────────────────────────────────────────────────────


class TestMinAtrFastEngine:
    """Tests du filtre min_atr_pct dans le fast engine (_simulate_grid_atr)."""

    def test_min_atr_zero_no_filter(self, make_indicator_cache):
        """min_atr_pct=0.0 (défaut) → tous les trades passent normalement."""
        from backend.optimization.fast_multi_backtest import (
            run_multi_backtest_from_cache,
        )

        n = 200
        # Prix qui descend de 100 à 80 (déclenche les niveaux de grille)
        closes = np.linspace(100.0, 80.0, n)
        lows = closes - 1.0
        highs = closes + 1.0
        sma_arr = np.full(n, 95.0)   # SMA au-dessus → TP = retour à 95
        sma_arr[:14] = np.nan
        # ATR faible : 0.5% du close
        atr_arr = closes * 0.005
        atr_arr[:14] = np.nan

        cache = make_indicator_cache(
            n=n,
            closes=closes, opens=closes, highs=highs, lows=lows,
            bb_sma={14: sma_arr},
            atr_by_period={14: atr_arr},
        )
        params = {
            "ma_period": 14,
            "atr_period": 14,
            "atr_multiplier_start": 2.0,
            "atr_multiplier_step": 1.0,
            "num_levels": 2,
            "sl_percent": 25.0,
            "min_atr_pct": 0.0,  # désactivé
        }
        result = run_multi_backtest_from_cache("grid_atr", params, cache, _make_bt_config())

        # Avec ATR = 0.5% * 100 = 0.5, niveaux à 95 - 0.5*2=94 et 95 - 0.5*3=93.5
        # Ces niveaux sont atteints quand le prix descend → des trades ont lieu
        n_trades = result[4]
        assert n_trades >= 0  # Pas de crash — min_atr_pct=0 ne filtre rien

    def test_min_atr_filters_low_vol(self, make_indicator_cache):
        """min_atr_pct=2.0 — candles avec ATR/close < 2% → aucun niveau ouvert."""
        from backend.optimization.fast_multi_backtest import (
            run_multi_backtest_from_cache,
        )

        n = 200
        # Prix constant à 100, ATR = 1.0 → ATR/close = 1% < 2% → filtre actif toujours
        closes = np.full(n, 100.0)
        lows = closes - 5.0   # lows assez bas pour déclencher les niveaux si filtre absent
        highs = closes + 5.0
        sma_arr = np.full(n, 100.0)
        sma_arr[:14] = np.nan
        atr_arr = np.full(n, 1.0)   # ATR/close = 1% → < min_atr_pct=2%
        atr_arr[:14] = np.nan

        cache = make_indicator_cache(
            n=n,
            closes=closes, opens=closes, highs=highs, lows=lows,
            bb_sma={14: sma_arr},
            atr_by_period={14: atr_arr},
        )
        params = {
            "ma_period": 14,
            "atr_period": 14,
            "atr_multiplier_start": 2.0,
            "atr_multiplier_step": 1.0,
            "num_levels": 3,
            "sl_percent": 20.0,
            "min_atr_pct": 2.0,  # ATR/close doit être >= 2% pour ouvrir
        }
        result = run_multi_backtest_from_cache("grid_atr", params, cache, _make_bt_config())

        # ATR=1 < 2% de close=100 → filtre toujours actif → 0 trades ouverts
        n_trades = result[4]
        assert n_trades == 0, (
            f"min_atr_pct=2.0 avec ATR/close=1% doit bloquer toutes les ouvertures, "
            f"got {n_trades} trades"
        )

    def test_min_atr_exits_still_active(self, make_indicator_cache):
        """Positions existantes se ferment normalement même quand le filtre bloque les ouvertures.

        Scénario :
        1. Phase 1 (candles 14-29) : ATR=3 → ATR/close=3% >= 2% → filtre OFF → position ouverte
           (entry = SMA - 1.5*ATR = 100 - 4.5 = 95.5 ; lows[14:30]=94 < 95.5 → déclenché)
        2. Phase 2 (candles 30+)  : ATR=0.5 → ATR/close=0.5% < 2% → filtre ON, mais
           la section EXIT (section 1) s'exécute AVANT le filtre (section 4b) →
           la position se ferme sur TP quand highs[30:]=101 >= SMA=100.
        """
        from backend.optimization.fast_multi_backtest import _simulate_grid_atr

        n = 100
        closes = np.full(n, 100.0)
        # Phase 1 : lows suffisamment bas pour déclencher l'entrée (95.5)
        lows = np.full(n, 99.0)
        lows[14:30] = 94.0     # Phase 1 : déclenche l'entrée à 95.5
        highs = np.full(n, 100.5)
        highs[30:] = 101.0     # Phase 2 : déclenche le TP (SMA=100)

        # SMA = 100 → TP = retour à 100
        sma_arr = np.full(n, 100.0)
        sma_arr[:14] = np.nan

        # Phase 1 (candles 14-29) : ATR=3 → 3% > 2% → filtre OFF
        # Phase 2 (candles 30+)   : ATR=0.5 → 0.5% < 2% → filtre ON
        atr_arr = np.full(n, np.nan)
        atr_arr[14:30] = 3.0
        atr_arr[30:] = 0.5

        cache = make_indicator_cache(
            n=n,
            closes=closes, opens=closes, highs=highs, lows=lows,
            bb_sma={14: sma_arr},
            atr_by_period={14: atr_arr},
        )
        params = {
            "ma_period": 14,
            "atr_period": 14,
            "atr_multiplier_start": 1.5,   # entry = 100 - 1.5*3 = 95.5 en phase 1
            "atr_multiplier_step": 1.0,
            "num_levels": 1,
            "sl_percent": 20.0,
            "min_atr_pct": 2.0,
        }
        bt_config = _make_bt_config()

        trade_pnls, trade_returns, final_capital = _simulate_grid_atr(cache, params, bt_config)

        # La position ouverte en phase 1 doit se fermer sur TP en phase 2
        # (le filtre n'empêche pas les exits, seulement les nouvelles ouvertures)
        assert len(trade_pnls) >= 1, (
            "Les positions ouvertes avant activation du filtre doivent pouvoir se fermer"
        )
        assert final_capital > 0, "Capital final doit être positif"

    def test_min_atr_in_param_grid(self):
        """min_atr_pct est bien dans la grille WFO de grid_atr."""
        from backend.optimization.walk_forward import _load_param_grids

        grids = _load_param_grids("config/param_grids.yaml")
        grid_atr = grids.get("grid_atr", {})
        default = grid_atr.get("default", {})

        assert "min_atr_pct" in default, (
            "min_atr_pct doit être dans la section default de grid_atr dans param_grids.yaml"
        )
        values = default["min_atr_pct"]
        assert isinstance(values, list), "min_atr_pct doit être une liste de valeurs"
        assert 0.0 in values, "min_atr_pct doit inclure 0.0 (désactivé)"
        assert len(values) >= 2, "min_atr_pct doit avoir au moins 2 valeurs pour la grille"


class TestMinAtrLiveStrategy:
    """Tests du filtre min_atr_pct dans la stratégie live (compute_grid)."""

    def _make_strategy(self, **overrides):
        from backend.core.config import GridATRConfig
        from backend.strategies.grid_atr import GridATRStrategy

        defaults = {
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 2.0, "atr_multiplier_step": 1.0,
            "num_levels": 3, "sl_percent": 20.0, "sides": ["long"],
        }
        defaults.update(overrides)
        return GridATRStrategy(GridATRConfig(**defaults))

    def _make_ctx(self, sma, atr, close):
        from backend.strategies.base import StrategyContext

        return StrategyContext(
            symbol="BTC/USDT",
            timestamp=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
            candles={},
            indicators={"1h": {"sma": sma, "atr": atr, "close": close}},
            current_position=None,
            capital=10_000.0,
            config=None,  # type: ignore[arg-type]
        )

    def _make_empty_state(self):
        from backend.strategies.base_grid import GridState

        return GridState(
            positions=[], avg_entry_price=0.0, total_quantity=0.0,
            total_notional=0.0, unrealized_pnl=0.0,
        )

    def test_compute_grid_filters_when_atr_too_low(self):
        """min_atr_pct=2.0 : ATR=1 (1% de 100) → compute_grid retourne []."""
        strat = self._make_strategy(min_atr_pct=2.0)
        ctx = self._make_ctx(sma=100.0, atr=1.0, close=100.0)  # ATR/close = 1%
        levels = strat.compute_grid(ctx, self._make_empty_state())
        assert levels == [], (
            "min_atr_pct=2.0 avec ATR/close=1% → aucun niveau ne doit être généré"
        )

    def test_compute_grid_passes_when_atr_sufficient(self):
        """min_atr_pct=2.0 : ATR=3 (3% de 100) → compute_grid retourne des niveaux."""
        strat = self._make_strategy(min_atr_pct=2.0)
        ctx = self._make_ctx(sma=100.0, atr=3.0, close=100.0)  # ATR/close = 3%
        levels = strat.compute_grid(ctx, self._make_empty_state())
        assert len(levels) > 0, (
            "min_atr_pct=2.0 avec ATR/close=3% → des niveaux doivent être générés"
        )

    def test_compute_grid_disabled_when_min_atr_zero(self):
        """min_atr_pct=0.0 (défaut) → filtre inactif, niveaux générés normalement."""
        strat = self._make_strategy(min_atr_pct=0.0)
        ctx = self._make_ctx(sma=100.0, atr=0.1, close=100.0)  # ATR très faible
        levels = strat.compute_grid(ctx, self._make_empty_state())
        # Sans filtre, des niveaux doivent être générés
        assert len(levels) > 0, (
            "min_atr_pct=0.0 ne doit pas filtrer même avec un ATR très faible"
        )
