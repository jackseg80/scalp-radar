"""Tests Sprint 40a #1 : taker fee sur tous les exits du fast engine grid.

En live, l'executor détecte le signal TP sur la candle puis envoie un market
order → taker fee + slippage. Avant ce fix, tp_global et tp individuel
utilisaient maker_fee, sous-estimant les frais de ~0.04% par trade.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.backtesting.engine import BacktestConfig
from backend.optimization.fast_multi_backtest import (
    _simulate_grid_common,
    _simulate_grid_range,
)


def _make_bt_config(**kwargs) -> BacktestConfig:
    from datetime import datetime
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


# ─── Test 1 : tp_global utilise taker fee ──────────────────────────────────


class TestTpGlobalUsesTakerFee:
    """Sprint 40a #1 : sortie TP dans _simulate_grid_common = taker fee."""

    def test_tp_exit_uses_taker_not_maker(self, make_indicator_cache):
        """Un exit TP (tp_global) doit coûter taker_fee, pas maker_fee."""
        n = 20
        ma_period = 5
        sma = np.full(n, 105.0)
        closes = np.full(n, 100.0)
        lows = np.full(n, 97.0)
        # highs en dessous de SMA sauf la dernière bougie → 1 seul TP
        highs = np.full(n, 103.0)  # 103 < 105=SMA → pas de TP
        highs[-1] = 107.0          # seule la dernière déclenche le TP

        entry_prices = np.full((n, 1), np.nan)
        for i in range(ma_period + 1, n):
            entry_prices[i, 0] = 99.0  # entry à 99, lows=97 ≤ 99 → triggered

        cache = make_indicator_cache(
            n=n, closes=closes, lows=lows, highs=highs,
            bb_sma={ma_period: sma},
        )

        # Cas A : taker_fee = 0.0006, maker_fee = 0.0 → taker appliqué si TP=taker
        bt_taker_only = _make_bt_config(
            taker_fee=0.0006, maker_fee=0.0, slippage_pct=0.0,
        )
        pnls_taker, _, _ = _simulate_grid_common(
            entry_prices, sma, cache, bt_taker_only,
            num_levels=1, sl_pct=0.5, direction=1,
        )

        # Cas B : taker_fee = 0.0, maker_fee = 0.0006 → maker appliqué si TP=maker
        bt_maker_only = _make_bt_config(
            taker_fee=0.0, maker_fee=0.0006, slippage_pct=0.0,
        )
        pnls_maker, _, _ = _simulate_grid_common(
            entry_prices, sma, cache, bt_maker_only,
            num_levels=1, sl_pct=0.5, direction=1,
        )

        assert len(pnls_taker) == 1, "1 trade attendu (exit TP)"
        assert len(pnls_maker) == 1, "1 trade attendu (exit TP)"

        # Sprint 40a #1 : TP utilise taker_fee
        # → cas A (taker=0.06%) coûte plus que cas B (maker=0.0) → pnl_A < pnl_B
        # Avant le fix, TP utilisait maker_fee
        # → cas A (taker=0.0) coûterait moins que cas B → pnl_A > pnl_B (WRONG)
        assert pnls_taker[0] < pnls_maker[0], (
            f"tp_global doit utiliser taker_fee (Sprint 40a #1). "
            f"pnl_taker={pnls_taker[0]:.4f}, pnl_maker={pnls_maker[0]:.4f}"
        )

    def test_sl_exit_still_uses_taker(self, make_indicator_cache):
        """Exit SL utilise toujours taker_fee (comportement inchangé)."""
        n = 20
        ma_period = 5
        sma = np.full(n, 105.0)
        closes = np.full(n, 100.0)
        lows = np.full(n, 97.0)
        highs = np.full(n, 103.0)
        # SL = entry * (1 - sl_pct) = 99 * 0.5 = 49.5 → lows[10] = 45 ≤ 49.5
        lows[10] = 45.0

        entry_prices = np.full((n, 1), np.nan)
        for i in range(ma_period + 1, n):
            entry_prices[i, 0] = 99.0

        cache = make_indicator_cache(
            n=n, closes=closes, lows=lows, highs=highs,
            bb_sma={ma_period: sma},
        )

        bt_taker_only = _make_bt_config(
            taker_fee=0.0006, maker_fee=0.0, slippage_pct=0.0,
        )
        pnls_taker, _, _ = _simulate_grid_common(
            entry_prices, sma, cache, bt_taker_only,
            num_levels=1, sl_pct=0.5, direction=1,
        )

        bt_zero_fees = _make_bt_config(
            taker_fee=0.0, maker_fee=0.0, slippage_pct=0.0,
        )
        pnls_zero, _, _ = _simulate_grid_common(
            entry_prices, sma, cache, bt_zero_fees,
            num_levels=1, sl_pct=0.5, direction=1,
        )

        assert len(pnls_taker) >= 1, "1 trade attendu (exit SL)"
        # Avec taker_fee = 0.06%, SL pnl < SL pnl sans frais
        assert pnls_taker[0] < pnls_zero[0], (
            f"SL exit doit appliquer taker_fee. "
            f"pnl_taker={pnls_taker[0]:.4f}, pnl_zero={pnls_zero[0]:.4f}"
        )


# ─── Test 2 : grid_range TP utilise taker fee ──────────────────────────────


class TestGridRangeTpUsesTakerFee:
    """Sprint 40a #1 : sortie TP dans _simulate_grid_range = taker fee."""

    def test_range_tp_uses_taker_not_maker(self, make_indicator_cache):
        """grid_range : exit TP individuel = taker_fee."""
        n = 50
        ma_period = 5
        atr_period = 5

        # SMA stable, ATR stable
        sma = np.full(n, 100.0)
        atr = np.full(n, 2.0)

        closes = np.full(n, 98.0)
        lows = np.full(n, 96.0)
        # highs en dessous de SMA (=100) sauf la dernière → 1 seul TP
        # LONG entry ep = SMA - 1*ATR*0.5 = 99; lows=96 <= 99 → entrée dès i=0
        # TP : highs[i] >= SMA → seulement highs[-1]=102
        highs = np.full(n, 97.0)   # 97 < 100=SMA → pas de TP
        highs[-1] = 102.0           # seule la dernière déclenche le TP

        cache = make_indicator_cache(
            n=n, closes=closes, lows=lows, highs=highs,
            bb_sma={ma_period: sma},
            atr_by_period={atr_period: atr},
        )

        params = {
            "ma_period": ma_period,
            "atr_period": atr_period,
            "atr_spacing_mult": 0.5,
            "num_levels": 1,
            "sl_percent": 10.0,
            "sides": ["long"],
            "tp_mode": "dynamic_sma",
        }

        bt_taker_only = _make_bt_config(
            taker_fee=0.0006, maker_fee=0.0, slippage_pct=0.0,
        )
        pnls_taker, _, _ = _simulate_grid_range(cache, params, bt_taker_only)

        bt_maker_only = _make_bt_config(
            taker_fee=0.0, maker_fee=0.0006, slippage_pct=0.0,
        )
        pnls_maker, _, _ = _simulate_grid_range(cache, params, bt_maker_only)

        if len(pnls_taker) == 0 or len(pnls_maker) == 0:
            pytest.skip("Aucun trade déclenché — vérifier les prix d'entrée")

        # Sprint 40a #1 : TP utilise taker_fee → pnl_taker < pnl_maker
        assert pnls_taker[0] < pnls_maker[0], (
            f"grid_range TP exit doit utiliser taker_fee. "
            f"pnl_taker={pnls_taker[0]:.4f}, pnl_maker={pnls_maker[0]:.4f}"
        )
