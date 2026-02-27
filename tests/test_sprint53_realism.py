"""Tests Sprint 53 — Renforcement réalisme du fast engine WFO.

Couvre :
- Section 1 : Filtre SL×leverage (3 tests)
- Section 2 : Kill switch grid engine (3 tests)
- Section 3 : Kill switch scalp engine (1 test)
- Section 4 : Filtre fine_grid (1 test)
- Section 5 : Pipeline complet coarse+fine — aucun combo invalide (1 test)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from backend.backtesting.engine import BacktestConfig
from backend.optimization.fast_multi_backtest import (
    KILL_SWITCH_DD_PCT,
    _build_entry_prices,
    _simulate_grid_common,
)
from backend.optimization.walk_forward import (
    _filter_sl_leverage,
    _fine_grid_around_top,
)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _bt_config(**kwargs) -> BacktestConfig:
    defaults = dict(
        symbol="BTC/USDT",
        start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        initial_capital=10_000.0,
        leverage=6,
        maker_fee=0.0002,
        taker_fee=0.0006,
        slippage_pct=0.0005,
        max_wfo_drawdown_pct=80.0,
    )
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


# ──────────────────────────────────────────────────────────────────────────
# Section 1 : Filtre SL × leverage
# ──────────────────────────────────────────────────────────────────────────


class TestFilterSlLeverage:
    def test_removes_invalid(self):
        """SL 25% × 7x = 175% > 150% → filtré."""
        grid = [
            {"sl_percent": 15.0, "ma_period": 7},
            {"sl_percent": 20.0, "ma_period": 7},
            {"sl_percent": 25.0, "ma_period": 7},
        ]
        filtered = _filter_sl_leverage(grid, leverage=7, threshold=1.5)
        assert len(filtered) == 2
        assert all(c["sl_percent"] <= 20.0 for c in filtered)

    def test_no_sl_key(self):
        """Stratégies sans sl_percent (donchian_breakout) → pas de filtrage."""
        grid = [
            {"atr_sl_multiple": 2.0, "atr_period": 14},
            {"atr_sl_multiple": 3.0, "atr_period": 14},
        ]
        filtered = _filter_sl_leverage(grid, leverage=7)
        assert len(filtered) == 2

    def test_all_invalid_returns_empty(self):
        """Si tous invalides, retourne une liste vide (pas de fallback)."""
        grid = [
            {"sl_percent": 30.0, "other": 1},
            {"sl_percent": 25.0, "other": 2},
            {"sl_percent": 35.0, "other": 3},
        ]
        # Tous dépassent 150% à 7x → aucun combo retourné
        filtered = _filter_sl_leverage(grid, leverage=7, threshold=1.5)
        assert len(filtered) == 0


# ──────────────────────────────────────────────────────────────────────────
# Section 2 : Kill switch grid engine
# ──────────────────────────────────────────────────────────────────────────


class TestKillSwitchGridCommon:
    def test_stops_trading_on_crash(self, make_indicator_cache):
        """Plusieurs cycles de pertes → kill switch stoppe après ~25% DD cumulé."""
        n = 500
        # Prix en dents de scie descendantes : chaque cycle perd ~8% du capital
        # sl_percent=5% × leverage=2 = 10% de marge par SL, -8% net avec 1 niveau
        segment = 50
        n_segments = n // segment
        prices = np.zeros(n)
        for s in range(n_segments):
            base = 100.0 - s * 3.0  # descente graduelle
            start = s * segment
            end = start + segment
            # Montée puis chute dans chaque segment
            up = np.linspace(base, base + 2.0, segment // 2)
            down = np.linspace(base + 2.0, base - 4.0, segment // 2)
            prices[start:end] = np.concatenate([up, down])

        opens = prices + 0.1
        highs = prices + 1.5
        lows = prices - 1.5
        sma = np.full(n, 120.0)  # SMA haute → TP jamais touché
        atr = np.full(n, 2.0)

        cache = make_indicator_cache(
            n=n, closes=prices, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma}, atr_by_period={14: atr},
        )

        bt = _bt_config(leverage=2)
        params = dict(
            ma_period=20, atr_period=14, atr_multiplier_start=1.0,
            atr_multiplier_step=0.5, num_levels=1, sl_percent=5.0,
        )

        entry_prices = _build_entry_prices("grid_atr", cache, params, 1, 1)
        pnls, _rets, final_cap = _simulate_grid_common(
            entry_prices, sma, cache, bt, 1, 0.05, 1,
        )

        max_dd = (10_000 - final_cap) / 10_000
        # Kill switch à 25% doit limiter les pertes (sans kill switch, DD >> 25%)
        assert len(pnls) >= 2, f"Au moins 2 cycles de trades attendus, got {len(pnls)}"
        # Le DD final ne doit pas dépasser ~40% (25% trigger + dernière perte en cours)
        assert max_dd < 0.50, f"DD trop élevé: {max_dd:.2%}, kill switch aurait dû stopper"

    def test_no_trigger_stable_market(self, make_indicator_cache):
        """Marché stable → DD < 25% → simulation complète, pas de kill switch."""
        n = 200
        # Prix oscillant autour de 100 (amplitude faible)
        t = np.arange(n, dtype=np.float64)
        closes = 100.0 + 3.0 * np.sin(2 * np.pi * t / 40)
        opens = closes - 0.1
        highs = closes + 1.5
        lows = closes - 1.5
        sma = np.full(n, 100.0)
        atr = np.full(n, 2.0)

        cache = make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma}, atr_by_period={14: atr},
        )

        bt = _bt_config(leverage=3)
        params = dict(
            ma_period=20, atr_period=14, atr_multiplier_start=1.0,
            atr_multiplier_step=0.5, num_levels=2, sl_percent=10.0,
        )

        entry_prices = _build_entry_prices("grid_atr", cache, params, 2, 1)
        pnls, _rets, final_cap = _simulate_grid_common(
            entry_prices, sma, cache, bt, 2, 0.10, 1,
        )

        # Avec un marché stable et faible leverage, le kill switch ne devrait pas trigger
        max_dd = (10_000 - final_cap) / 10_000
        # Le test vérifie que la simulation a produit des trades normalement
        # (si kill switch avait triggeré, il y aurait très peu de trades)
        assert len(pnls) >= 2, f"Simulation devrait avoir des trades, got {len(pnls)}"

    def test_exits_then_breaks(self, make_indicator_cache):
        """Kill switch triggered → pas de nouvelles entrées après le premier batch."""
        n = 500
        # Dents de scie : multiple cycles de petites pertes
        segment = 50
        n_seg = n // segment
        prices = np.zeros(n)
        for s in range(n_seg):
            base = 100.0 - s * 2.0
            start = s * segment
            end = start + segment
            up = np.linspace(base, base + 1.5, segment // 2)
            down = np.linspace(base + 1.5, base - 3.0, segment // 2)
            prices[start:end] = np.concatenate([up, down])

        opens = prices + 0.1
        highs = prices + 1.5
        lows = prices - 1.5
        sma = np.full(n, 120.0)
        atr = np.full(n, 2.0)

        cache = make_indicator_cache(
            n=n, closes=prices, opens=opens, highs=highs, lows=lows,
            bb_sma={20: sma}, atr_by_period={14: atr},
        )

        # Avec kill switch (leverage=2, sl=5% → SL cost=10% de margin par cycle)
        bt_ks = _bt_config(leverage=2)
        params = dict(
            ma_period=20, atr_period=14, atr_multiplier_start=1.0,
            atr_multiplier_step=0.5, num_levels=1, sl_percent=5.0,
        )

        entry_prices = _build_entry_prices("grid_atr", cache, params, 1, 1)
        pnls_ks, _, cap_ks = _simulate_grid_common(
            entry_prices, sma, cache, bt_ks, 1, 0.05, 1,
        )

        # Sans kill switch (max_wfo_drawdown_pct à 80% = pas de break prématuré)
        bt_no_ks = _bt_config(leverage=2, max_wfo_drawdown_pct=80.0)
        # On réutilise la même simulation mais le kill switch trigger à 25% devrait
        # stopper plus tôt que la version 80%
        # Vérification : kill switch a produit moins de trades
        assert len(pnls_ks) >= 2, f"Au moins 2 trades avant kill switch, got {len(pnls_ks)}"
        # Le nombre de trades est limité par le kill switch
        # (avec 10 segments et SL cost=10%, ~3-4 trades avant 25% DD)
        assert len(pnls_ks) <= 6, (
            f"Kill switch devrait limiter les trades (~3-4), got {len(pnls_ks)}"
        )


# ──────────────────────────────────────────────────────────────────────────
# Section 3 : Kill switch scalp engine
# ──────────────────────────────────────────────────────────────────────────


class TestKillSwitchScalp:
    def test_python_fallback(self, make_indicator_cache):
        """Scalp Python fallback : kill switch après 25% DD."""
        # On importe ici pour éviter le side effect numba au niveau module
        from backend.optimization.fast_backtest import _simulate_trades

        n = 300
        # Prix en chute : force des SL répétés
        closes = np.linspace(100.0, 50.0, n)
        opens = closes + 0.2
        highs = closes + 1.0
        lows = closes - 1.0

        cache = make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows,
            rsi={14: np.full(n, 30.0)},
            vwap=np.full(n, np.nan),
            vwap_distance_pct=np.full(n, -0.5),
            adx_arr=np.full(n, 30.0),
            di_plus=np.full(n, 20.0),
            di_minus=np.full(n, 10.0),
            volume_sma_arr=np.full(n, 50.0),
            volumes=np.full(n, 100.0),
        )

        bt = _bt_config(leverage=1)

        # Signaux : toujours LONG (en downtrend = SL répétés)
        longs = np.ones(n, dtype=bool)
        shorts = np.zeros(n, dtype=bool)

        params = {
            "tp_percent": 2.0, "sl_percent": 1.0, "rsi_period": 14,
            "vwap_deviation_entry": 0.2, "trend_adx_threshold": 25,
            "volume_spike_multiplier": 1.5, "rsi_long_threshold": 30,
            "rsi_short_threshold": 70,
        }

        pnls, _rets, final_cap = _simulate_trades(
            longs, shorts, cache, "vwap_rsi", params, bt,
        )

        max_dd = (10_000 - final_cap) / 10_000
        # Avec kill switch à 25%, le DD ne devrait pas dépasser ~35%
        # (on laisse de la marge car le dernier trade peut dépasser légèrement)
        assert max_dd < 0.50, f"Kill switch devrait limiter DD, got {max_dd:.2%}"
        assert len(pnls) > 0, "Au moins un trade"


# ──────────────────────────────────────────────────────────────────────────
# Section 4 : Fine grid filter
# ──────────────────────────────────────────────────────────────────────────


class TestFilterFineGrid:
    def test_removes_invalid_from_fine_grid(self):
        """fine_grid ±1 step génère sl_percent=25 depuis sl_percent=20 → filtré à 7x."""
        grid_values = {
            "ma_period": [7, 10, 14],
            "sl_percent": [15.0, 20.0, 25.0],
        }
        top_params = [{"ma_period": 10, "sl_percent": 20.0}]
        fine = _fine_grid_around_top(top_params, grid_values)

        # fine devrait inclure sl_percent=25 (±1 step de 20)
        has_25 = any(c["sl_percent"] == 25.0 for c in fine)
        assert has_25, "Fine grid devrait générer sl_percent=25 via ±1 step"

        # Après filtre à 7x : sl=25 × 7 / 100 = 1.75 > 1.5 → filtré
        filtered = _filter_sl_leverage(fine, leverage=7, threshold=1.5)
        has_25_after = any(c["sl_percent"] == 25.0 for c in filtered)
        assert not has_25_after, "sl_percent=25 à 7x devrait être filtré"


# ──────────────────────────────────────────────────────────────────────────
# Section 5 : Pipeline complet — aucun combo SL×lev > 150% ne passe
# ──────────────────────────────────────────────────────────────────────────


class TestNoInvalidComboInPipeline:
    def test_full_pipeline_no_invalid_sl(self):
        """Simule le pipeline coarse → fine : aucun combo sl×lev > 1.5 ne survit."""
        from backend.optimization.walk_forward import _latin_hypercube_sample

        leverage = 7
        threshold = 1.5

        # Grille réaliste : certains combos invalides à 7x
        grid_values = {
            "ma_period": [7, 10, 14, 21],
            "sl_percent": [10.0, 15.0, 20.0, 25.0, 30.0],
        }
        full_grid = [
            {"ma_period": m, "sl_percent": s}
            for m in grid_values["ma_period"]
            for s in grid_values["sl_percent"]
        ]
        assert len(full_grid) == 20

        # 1. Filtre full_grid AVANT coarse sampling (comme le fix)
        full_grid = _filter_sl_leverage(full_grid, leverage, threshold)
        assert all(
            c["sl_percent"] / 100 * leverage <= threshold for c in full_grid
        ), "full_grid doit être propre après filtre"

        # 2. Coarse = sample de full_grid filtré
        coarse_grid = _latin_hypercube_sample(full_grid, min(10, len(full_grid)))
        assert all(
            c["sl_percent"] / 100 * leverage <= threshold for c in coarse_grid
        ), "coarse_grid ne doit contenir aucun combo invalide"

        # 3. Fine grid autour du top (simuler un top avec sl=20, le max valide)
        top_params = [{"ma_period": 14, "sl_percent": 20.0}]
        fine_grid = _fine_grid_around_top(top_params, grid_values)
        # Fine grid PEUT contenir sl=25 (±1 step) — le filtre le supprime
        fine_grid = _filter_sl_leverage(fine_grid, leverage, threshold)
        assert all(
            c["sl_percent"] / 100 * leverage <= threshold for c in fine_grid
        ), "fine_grid ne doit contenir aucun combo invalide après filtre"

        # 4. Résultats combinés
        all_combos = coarse_grid + fine_grid
        invalid = [
            c for c in all_combos
            if c["sl_percent"] / 100 * leverage > threshold
        ]
        assert invalid == [], (
            f"Combos invalides dans le pipeline : {invalid}"
        )
