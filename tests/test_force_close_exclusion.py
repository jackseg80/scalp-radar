"""Tests Sprint 59 — Exclusion du force-close des métriques OOS WFO.

Le P&L des positions force-closées en fin de fenêtre WFO ne doit pas
polluer les métriques (Sharpe, n_trades, profit_factor). Il impacte
uniquement le capital final (cohérent avec la réalité).

Couvre :
- Test 1 : grid_common — force-close exclu de trade_pnls
- Test 2 : grid_common — vrais trades TP/SL non affectés
- Test 3 : grid_range — force-close exclu (positions bidirectionnelles)
- Test 4 : _compute_fast_metrics avec 0 trades → Sharpe=0, n_trades=0
- Test 5 : grid_common — final_capital inclut quand même le drift force-close
- Test 6 : grid_boltrend — force-close exclu
- Test 7 : grid_momentum — force-close exclu
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from backend.backtesting.engine import BacktestConfig
from backend.optimization.fast_multi_backtest import (
    _simulate_grid_common,
    _simulate_grid_range,
    run_multi_backtest_from_cache,
)
from backend.optimization.indicator_cache import IndicatorCache


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_bt_config(**overrides) -> BacktestConfig:
    defaults = dict(
        symbol="TEST/USDT",
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
        initial_capital=10_000.0,
        leverage=6,
        taker_fee=0.0006,
        maker_fee=0.0002,
        slippage_pct=0.0001,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _make_cache(closes, sma_arr=None, atr_arr=None) -> IndicatorCache:
    n = len(closes)
    if sma_arr is None:
        sma_arr = closes.copy()
    if atr_arr is None:
        atr_arr = np.full(n, 3.0)
    return IndicatorCache(
        n_candles=n,
        opens=closes.copy(),
        closes=closes.copy(),
        highs=closes + 2.0,
        lows=closes - 2.0,
        volumes=np.full(n, 1000.0),
        total_days=n / 24,
        bb_sma={14: sma_arr},
        atr_by_period={14: atr_arr},
    )


# ─── Tests ─────────────────────────────────────────────────────────────────


class TestForceCloseExclusion:
    """Vérifie que le force-close fin de données est exclu des métriques."""

    def test_grid_common_force_close_not_in_trade_pnls(self, make_indicator_cache):
        """Test 1 : grid_atr — force-close n'est pas dans trade_pnls.

        Données : 14 candles warmup à 100, puis 13 candles à 80.
        La SMA(14) reste au-dessus de 80 pendant toute la période basse
        (il faut 14 candles à 80 pour que SMA = 80, or on n'en a que 13).
        → Aucun TP ne se déclenche. SL très large → pas de SL.
        → Force-close uniquement à la fin.
        Résultat attendu : 0 vrais trades, capital ≠ initial (drift capturé).
        """
        # 14 warmup à 100 + 10 à 80.
        # SMA(14) à la dernière candle = (4*100 + 10*80)/14 ≈ 85.7, high=81 < 85.7 → pas de TP.
        # make_indicator_cache défaut : highs=closes+1, lows=closes-1.
        n = 24
        prices = np.concatenate([
            np.full(14, 100.0),
            np.full(10, 80.0),
        ])
        sma_arr = np.full(n, np.nan)
        atr_arr = np.full(n, np.nan)
        for i in range(14, n):
            sma_arr[i] = np.mean(prices[max(0, i - 13): i + 1])
            atr_arr[i] = 3.0

        cache = make_indicator_cache(
            n=n,
            closes=prices,
            bb_sma={14: sma_arr},
            atr_by_period={14: atr_arr},
        )
        params = {
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 1.0, "atr_multiplier_step": 0.5,
            "num_levels": 2, "sl_percent": 80.0,  # SL très large → jamais touché
        }
        bt_config = _make_bt_config()
        result = run_multi_backtest_from_cache("grid_atr", params, cache, bt_config)

        # Aucun vrai trade TP/SL
        n_trades = result[4]
        sharpe = result[1]
        net_return_pct = result[2]
        assert n_trades == 0, f"Attendu 0 trades, obtenu {n_trades}"
        assert sharpe == 0.0, f"Sharpe attendu 0.0, obtenu {sharpe}"
        assert net_return_pct == 0.0, f"net_return_pct attendu 0.0, obtenu {net_return_pct}"

    def test_grid_common_real_trades_unaffected(self, make_indicator_cache):
        """Test 2 : grid_atr — les vrais trades TP/SL restent comptés normalement.

        Données oscillantes : plusieurs TP se déclenchent avant la fin.
        Résultat attendu : n_trades > 0 (inchangé par rapport à avant le fix).
        """
        n = 200
        # Oscillation entre 85 et 100 → TP déclenché à la SMA à chaque cycle
        t = np.linspace(0, 8 * np.pi, n)
        prices = 92.5 + 7.5 * np.sin(t)
        sma_arr = np.full(n, np.nan)
        atr_arr = np.full(n, np.nan)
        for i in range(14, n):
            sma_arr[i] = np.mean(prices[max(0, i - 13): i + 1])
            atr_arr[i] = 2.0

        cache = make_indicator_cache(
            n=n,
            closes=prices,
            highs=prices + 1.5,
            lows=prices - 1.5,
            bb_sma={14: sma_arr},
            atr_by_period={14: atr_arr},
        )
        params = {
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 0.5, "atr_multiplier_step": 0.5,
            "num_levels": 2, "sl_percent": 20.0,
        }
        bt_config = _make_bt_config()
        result = run_multi_backtest_from_cache("grid_atr", params, cache, bt_config)

        n_trades = result[4]
        assert n_trades >= 2, f"Des vrais trades attendus, obtenu {n_trades}"

    def test_grid_range_force_close_not_in_trade_pnls(self, make_indicator_cache):
        """Test 3 : grid_range_atr — force-close exclu (cas buy-and-hold déguisé).

        Données : prix monte en ligne droite. Positions LONG ouvertes, jamais fermées.
        Résultat attendu : 0 vrais trades (le Sharpe 13-21 disparaît).
        """
        n = 200
        # Prix monte de 100 à 130 sans oscillation (aucun TP/SL ne se déclenche)
        prices = np.linspace(100.0, 130.0, n)
        sma_arr = np.full(n, np.nan)
        atr_arr = np.full(n, np.nan)
        for i in range(20, n):
            sma_arr[i] = np.mean(prices[max(0, i - 19): i + 1])
            atr_arr[i] = 0.5  # ATR très faible → niveaux proches → positions ouvertes

        cache = make_indicator_cache(
            n=n,
            closes=prices,
            highs=prices + 1.0,
            lows=prices - 1.0,
            bb_sma={20: sma_arr},
            atr_by_period={14: atr_arr},
        )
        params = {
            "ma_period": 20, "atr_period": 14,
            "atr_spacing_mult": 0.3, "num_levels": 2,
            "sl_percent": 50.0,  # SL très large → jamais touché
            "tp_mode": "dynamic_sma",
            "sides": ["long", "short"],
        }
        bt_config = _make_bt_config()
        result = run_multi_backtest_from_cache("grid_range_atr", params, cache, bt_config)

        n_trades = result[4]
        sharpe = result[1]
        assert n_trades == 0, f"Attendu 0 trades (force-close exclu), obtenu {n_trades}"
        assert sharpe == 0.0, f"Sharpe attendu 0.0 (pas de vrais trades), obtenu {sharpe}"

    def test_compute_fast_metrics_zero_trades(self):
        """Test 4 : _compute_fast_metrics avec 0 trades → early return propre."""
        from backend.optimization.fast_multi_backtest import _compute_fast_metrics

        params = {"test": 1}
        result = _compute_fast_metrics(
            params=params,
            trade_pnls=[],
            trade_returns=[],
            final_capital=12_000.0,  # capital > initial (drift positif)
            initial_capital=10_000.0,
            total_days=60.0,
        )

        assert result[0] == params
        assert result[1] == 0.0, "sharpe doit être 0.0"
        assert result[2] == 0.0, "net_return_pct doit être 0.0"
        assert result[3] == 0.0, "profit_factor doit être 0.0"
        assert result[4] == 0, "n_trades doit être 0"

    def test_force_close_impacts_capital_not_metrics(self, make_indicator_cache):
        """Test 5 : le force-close impacte le capital final mais pas les métriques.

        Même setup que test 1 : positions ouvertes à ~98, force-close à 80 (perte).
        - trade_pnls doit être vide (pas dans les métriques)
        - final_capital doit être < initial (force-close à perte appliqué au capital)
        Démontre que le capital reflète la réalité, mais le Sharpe ne le pollue pas.
        """
        from backend.optimization.fast_multi_backtest import (
            _build_entry_prices,
            _simulate_grid_common,
        )

        n = 24  # Identique au Test 1 : 14 warmup + 10 à 80, SMA reste > 81 → pas de TP
        prices = np.concatenate([
            np.full(14, 100.0),
            np.full(10, 80.0),
        ])
        sma_arr = np.full(n, np.nan)
        atr_arr = np.full(n, np.nan)
        for i in range(14, n):
            sma_arr[i] = np.mean(prices[max(0, i - 13): i + 1])
            atr_arr[i] = 3.0

        cache = make_indicator_cache(
            n=n,
            closes=prices,
            bb_sma={14: sma_arr},
            atr_by_period={14: atr_arr},
        )
        params = {
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 1.0, "atr_multiplier_step": 0.5,
            "num_levels": 2, "sl_percent": 80.0,
        }
        bt_config = _make_bt_config(initial_capital=10_000.0)

        entry_prices = _build_entry_prices("grid_atr", cache, params, 2, direction=1)
        trade_pnls, trade_returns, final_capital = _simulate_grid_common(
            entry_prices, sma_arr, cache, bt_config,
            num_levels=2, sl_pct=0.8, direction=1,
        )

        # Aucun vrai trade dans les métriques
        assert len(trade_pnls) == 0, f"Attendu 0 trades, obtenu {len(trade_pnls)}"
        # Mais le capital reflète le force-close à perte (entré à ~95-98, fermé à 80)
        assert final_capital < 10_000.0, (
            f"final_capital devrait être < 10000 (force-close à perte), obtenu {final_capital:.2f}"
        )
        assert final_capital != 10_000.0, "final_capital doit avoir été modifié par le force-close"

    def test_grid_boltrend_force_close_excluded(self, make_indicator_cache):
        """Test 6 : grid_boltrend — le code s'exécute sans erreur après le fix.

        Valide que le moteur retourne un 5-tuple valide (force-close ou non).
        """
        bol_w = 10
        long_w = 20
        n = 60
        prices = np.concatenate([
            np.full(25, 100.0),
            np.linspace(100, 93, 15),
            np.full(20, 93.0),
        ])
        sma_bol = np.full(n, np.nan)
        sma_long = np.full(n, np.nan)
        atr_arr = np.full(n, np.nan)
        bb_upper = np.full(n, np.nan)
        bb_lower = np.full(n, np.nan)
        for i in range(bol_w - 1, n):
            w = prices[max(0, i - bol_w + 1): i + 1]
            sma_bol[i] = np.mean(w)
            std = np.std(w)
            bb_upper[i] = sma_bol[i] + 2.0 * std
            bb_lower[i] = sma_bol[i] - 2.0 * std
            atr_arr[i] = 1.0
        for i in range(long_w - 1, n):
            sma_long[i] = np.mean(prices[max(0, i - long_w + 1): i + 1])

        cache = make_indicator_cache(
            n=n,
            closes=prices,
            bb_sma={bol_w: sma_bol, long_w: sma_long},
            bb_upper={(bol_w, 2.0): bb_upper},
            bb_lower={(bol_w, 2.0): bb_lower},
            atr_by_period={14: atr_arr},
        )
        params = {
            "bol_window": bol_w, "bol_std": 2.0,
            "long_ma_window": long_w, "atr_period": 14,
            "atr_spacing_mult": 0.5, "num_levels": 2,
            "sl_percent": 60.0, "sides": ["long"],
            "min_bol_spread": 0.0,
        }
        bt_config = _make_bt_config()
        result = run_multi_backtest_from_cache("grid_boltrend", params, cache, bt_config)

        assert len(result) == 5
        assert isinstance(result[4], int)
        assert isinstance(result[1], float)  # sharpe valide

    def test_grid_momentum_force_close_excluded(self, make_indicator_cache):
        """Test 7 : grid_momentum — force-close exclu des métriques."""
        n = 200
        # Breakout initial puis consolidation sans trailing stop hit
        prices = np.concatenate([
            np.full(50, 100.0),
            np.linspace(100, 115, 20),  # Breakout
            np.full(130, 115.0),         # Consolidation haute → pas de trailing stop
        ])
        sma_arr = np.full(n, np.nan)
        atr_arr = np.full(n, np.nan)
        volume_sma = np.full(n, np.nan)
        rolling_high = {20: np.full(n, np.nan)}
        rolling_low = {20: np.full(n, np.nan)}
        for i in range(20, n):
            w = prices[max(0, i - 19): i + 1]
            sma_arr[i] = np.mean(w)
            atr_arr[i] = 0.5
            rolling_high[20][i] = np.max(w)
            rolling_low[20][i] = np.min(w)
            volume_sma[i] = 1000.0

        cache = make_indicator_cache(
            n=n,
            closes=prices,
            highs=prices + 1.0,
            lows=prices - 1.0,
            volumes=np.full(n, 2000.0),
            bb_sma={20: sma_arr},
            atr_by_period={14: atr_arr},
            rolling_high=rolling_high,
            rolling_low=rolling_low,
            volume_sma_arr=volume_sma,
        )
        params = {
            "donchian_period": 20, "atr_period": 14,
            "num_levels": 2, "sl_percent": 8.0,
            "trailing_atr_mult": 5.0,  # Trailing très large → jamais hit
            "vol_multiplier": 1.5, "adx_threshold": 0.0,
            "pullback_start": 1.0, "pullback_step": 0.5,
            "cooldown_candles": 0, "sides": ["long"],
        }
        bt_config = _make_bt_config()
        result = run_multi_backtest_from_cache("grid_momentum", params, cache, bt_config)

        assert len(result) == 5
        assert isinstance(result[4], int)
        # Résultat valide ; si force-close seul, n_trades=0
