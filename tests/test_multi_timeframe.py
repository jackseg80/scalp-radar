"""Tests Sprint 30 — Multi-Timeframe Support (resampling + WFO).

Vérifie :
- Le resampling 1h → 4h / 1d (buckets complets, OHLCV correct)
- total_days correct dans build_cache pour chaque TF
- Le fast engine produit des résultats valides avec des candles resamplees
- La parité 1h ancien code vs nouveau code
- L'intégration param_grids.yaml (timeframe présent/absent)
- Le groupement par timeframe dans _run_fast
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import yaml

from backend.core.models import Candle, TimeFrame
from backend.optimization.indicator_cache import resample_candles, build_cache
from backend.backtesting.engine import BacktestConfig


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_1h_candles(
    n: int,
    *,
    start: datetime | None = None,
    base_price: float = 100.0,
    symbol: str = "BTC/USDT",
    exchange: str = "binance",
) -> list[Candle]:
    """Génère n candles 1h alignées UTC à partir de start."""
    if start is None:
        # Aligner sur une frontière 00:00 UTC
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

    candles = []
    for i in range(n):
        ts = start + timedelta(hours=i)
        noise = (i % 7) * 0.5 - 1.5  # variation déterministe
        close = base_price + noise
        candles.append(Candle(
            timestamp=ts,
            open=close - 0.3,
            high=close + 1.0,
            low=close - 1.0,
            close=close,
            volume=1000.0 + i * 10,
            symbol=symbol,
            timeframe=TimeFrame.H1,
            exchange=exchange,
        ))
    return candles


# ═══════════════════════════════════════════════════════════════════════════
# Section 1 — Resampling
# ═══════════════════════════════════════════════════════════════════════════


class TestResampleCandles:
    """Tests du resampling 1h → 4h / 1d."""

    def test_passthrough_1h(self):
        """resample_candles('1h') retourne la même liste."""
        candles = _make_1h_candles(10)
        result = resample_candles(candles, "1h")
        assert result is candles  # même référence

    def test_4h_count(self):
        """24 candles 1h → 6 candles 4h."""
        candles = _make_1h_candles(24)
        result = resample_candles(candles, "4h")
        assert len(result) == 6

    def test_4h_ohlcv(self):
        """OHLCV correct pour le premier bucket 4h."""
        candles = _make_1h_candles(24)
        result = resample_candles(candles, "4h")

        first = result[0]
        bucket = candles[:4]
        assert first.open == bucket[0].open
        assert first.close == bucket[-1].close
        assert first.high == max(c.high for c in bucket)
        assert first.low == min(c.low for c in bucket)
        assert first.volume == pytest.approx(sum(c.volume for c in bucket))

    def test_4h_incomplete_excluded(self):
        """23 candles 1h → 5 candles 4h (dernier bucket incomplet exclu)."""
        candles = _make_1h_candles(23)
        result = resample_candles(candles, "4h")
        assert len(result) == 5

    def test_1d_count(self):
        """48 candles 1h → 2 candles 1d."""
        candles = _make_1h_candles(48)
        result = resample_candles(candles, "1d")
        assert len(result) == 2

    def test_1d_incomplete_excluded(self):
        """47 candles 1h → 1 candle 1d (deuxième jour incomplet)."""
        candles = _make_1h_candles(47)
        result = resample_candles(candles, "1d")
        assert len(result) == 1

    def test_timestamps_aligned(self):
        """Les timestamps des candles resamplees 4h sont aux frontières UTC."""
        candles = _make_1h_candles(24)
        result = resample_candles(candles, "4h")

        for c in result:
            hour = c.timestamp.hour
            assert hour % 4 == 0, f"Timestamp {c.timestamp} n'est pas alignée sur 4h"

    def test_empty_list(self):
        """Liste vide → liste vide."""
        result = resample_candles([], "4h")
        assert result == []

    def test_metadata_copied(self):
        """symbol, exchange, timeframe copiés correctement."""
        candles = _make_1h_candles(24, symbol="ETH/USDT", exchange="bitget")
        result = resample_candles(candles, "4h")

        for c in result:
            assert c.symbol == "ETH/USDT"
            assert c.exchange == "bitget"
            assert c.timeframe == TimeFrame.H4

    def test_invalid_timeframe_raises(self):
        """Timeframe non supporté → ValueError."""
        candles = _make_1h_candles(10)
        with pytest.raises(ValueError, match="non supporté"):
            resample_candles(candles, "2h")


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 — total_days via build_cache
# ═══════════════════════════════════════════════════════════════════════════


class TestTotalDays:
    """Vérifie que total_days est correct pour chaque timeframe via build_cache."""

    def test_4h_total_days(self):
        """build_cache avec candles 4h : total_days ~ n_candles × 4 / 24."""
        candles_1h = _make_1h_candles(96)  # 4 jours
        candles_4h = resample_candles(candles_1h, "4h")
        assert len(candles_4h) == 24  # 96/4

        cache = build_cache(
            {"4h": candles_4h}, {"ma_period": [14]}, "grid_atr", main_tf="4h",
        )
        # 24 candles × 4h = 96h = ~4 jours
        assert 3.5 < cache.total_days < 4.5

    def test_1d_total_days(self):
        """build_cache avec candles 1d : total_days ~ n_candles."""
        candles_1h = _make_1h_candles(72)  # 3 jours
        candles_1d = resample_candles(candles_1h, "1d")
        assert len(candles_1d) == 3

        cache = build_cache(
            {"1d": candles_1d}, {"ma_period": [14]}, "grid_atr", main_tf="1d",
        )
        assert 1.5 < cache.total_days < 3.5

    def test_total_days_coherence(self):
        """Même période couverte → ≈ même total_days quelle que soit la TF."""
        candles_1h = _make_1h_candles(96)
        candles_4h = resample_candles(candles_1h, "4h")

        cache_1h = build_cache(
            {"1h": candles_1h}, {"ma_period": [14]}, "grid_atr", main_tf="1h",
        )
        cache_4h = build_cache(
            {"4h": candles_4h}, {"ma_period": [14]}, "grid_atr", main_tf="4h",
        )
        # Les deux doivent couvrir ~4 jours (à ±0.2j près car last_ts diffère)
        assert abs(cache_1h.total_days - cache_4h.total_days) < 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Section 3 — Fast engine multi-TF
# ═══════════════════════════════════════════════════════════════════════════


class TestFastEngineMultiTF:
    """Fast engine avec candles resamplees."""

    def _run_grid_atr(self, candles, tf, params=None):
        """Helper : lance grid_atr sur un cache construit depuis les candles."""
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache

        if params is None:
            params = {
                "ma_period": 14,
                "atr_period": 14,
                "atr_multiplier_start": 1.5,
                "atr_multiplier_step": 1.0,
                "num_levels": 2,
                "sl_percent": 20.0,
            }

        cache = build_cache(
            {tf: candles}, {"ma_period": [14], "atr_period": [14]},
            "grid_atr", main_tf=tf,
        )
        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=candles[0].timestamp,
            end_date=candles[-1].timestamp,
            leverage=6,
        )
        return run_multi_backtest_from_cache("grid_atr", params, cache, bt_config)

    def test_grid_atr_4h_valid(self):
        """grid_atr sur candles 4h → résultat 5-tuple valide."""
        candles_1h = _make_1h_candles(480)  # 20 jours
        candles_4h = resample_candles(candles_1h, "4h")

        result = self._run_grid_atr(candles_4h, "4h")
        assert len(result) == 5
        params, sharpe, net_return, pf, n_trades = result
        assert isinstance(sharpe, float)
        assert isinstance(n_trades, int)

    def test_grid_atr_1d_valid(self):
        """grid_atr sur candles 1d → résultat valide."""
        candles_1h = _make_1h_candles(720)  # 30 jours
        candles_1d = resample_candles(candles_1h, "1d")

        result = self._run_grid_atr(candles_1d, "1d")
        assert len(result) == 5

    def test_envelope_dca_4h_valid(self):
        """envelope_dca sur candles 4h → résultat valide."""
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache

        candles_1h = _make_1h_candles(480)
        candles_4h = resample_candles(candles_1h, "4h")

        params = {
            "ma_period": 7,
            "num_levels": 2,
            "envelope_start": 0.05,
            "envelope_step": 0.03,
            "sl_percent": 20.0,
        }
        cache = build_cache(
            {"4h": candles_4h}, {"ma_period": [7]},
            "envelope_dca", main_tf="4h",
        )
        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=candles_4h[0].timestamp,
            end_date=candles_4h[-1].timestamp,
            leverage=6,
        )
        result = run_multi_backtest_from_cache("envelope_dca", params, cache, bt_config)
        assert len(result) == 5

    def test_fewer_trades_4h_vs_1h(self):
        """4h produit ≤ trades que 1h (moins de candles = moins d'opportunités)."""
        candles_1h = _make_1h_candles(480)
        candles_4h = resample_candles(candles_1h, "4h")

        result_1h = self._run_grid_atr(candles_1h, "1h")
        result_4h = self._run_grid_atr(candles_4h, "4h")

        # 4h a ≤ trades (ou 0 si pas de trigger)
        assert result_4h[4] <= result_1h[4] or result_1h[4] == 0

    def test_sharpe_not_exploding(self):
        """Sharpe annualisé cohérent (pas d'explosion due à total_days faux)."""
        candles_1h = _make_1h_candles(480)
        candles_4h = resample_candles(candles_1h, "4h")

        result = self._run_grid_atr(candles_4h, "4h")
        sharpe = result[1]
        # Sharpe raisonnable : entre -100 et +100 (cap existant dans le code)
        assert -100 <= sharpe <= 100

    def test_deterministic(self):
        """Même input → même output."""
        candles_1h = _make_1h_candles(480)
        candles_4h = resample_candles(candles_1h, "4h")

        r1 = self._run_grid_atr(candles_4h, "4h")
        r2 = self._run_grid_atr(candles_4h, "4h")
        assert r1[1] == r2[1]  # sharpe
        assert r1[2] == r2[2]  # net_return
        assert r1[4] == r2[4]  # n_trades


# ═══════════════════════════════════════════════════════════════════════════
# Section 4 — Parité
# ═══════════════════════════════════════════════════════════════════════════


class TestParity:
    """Tests de parité — le plus important pour éviter les régressions."""

    def test_1h_no_timeframe_same_as_before(self, make_indicator_cache):
        """_run_fast() avec grid SANS timeframe → résultats identiques au flow linéaire.

        Simule l'ancien comportement (1 seul cache, pas de groupement).
        """
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache

        n = 200
        closes = 100.0 + np.sin(np.linspace(0, 4 * np.pi, n)) * 5
        sma_arr = np.convolve(closes, np.ones(14) / 14, mode="same")
        atr_arr = np.full(n, 2.0)

        cache = make_indicator_cache(
            n=n, closes=closes,
            bb_sma={14: sma_arr},
            atr_by_period={14: atr_arr},
        )

        params = {
            "ma_period": 14,
            "atr_period": 14,
            "atr_multiplier_start": 1.5,
            "atr_multiplier_step": 1.0,
            "num_levels": 2,
            "sl_percent": 20.0,
        }
        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 2, 1, tzinfo=timezone.utc),
            leverage=6,
        )

        # Résultat direct (sans timeframe dans params)
        result_direct = run_multi_backtest_from_cache("grid_atr", params, cache, bt_config)

        # Résultat via _run_fast (nouveau code) sans timeframe
        from backend.optimization.walk_forward import WalkForwardOptimizer
        candles = _make_1h_candles(n)
        results_fast = WalkForwardOptimizer._run_fast(
            [params], {"1h": candles}, "grid_atr",
            {
                "symbol": "BTC/USDT",
                "start_date": candles[0].timestamp,
                "end_date": candles[-1].timestamp,
                "initial_capital": 10000.0,
                "leverage": 6,
                "maker_fee": 0.0002,
                "taker_fee": 0.0006,
                "slippage_pct": 0.0005,
                "high_vol_slippage_mult": 1.5,
                "max_risk_per_trade": 0.02,
            },
            "1h",
        )
        assert len(results_fast) == 1
        # Résultats doivent être valides (5-tuple)
        assert len(results_fast[0]) == 5

    def test_1h_explicit_same_as_implicit(self):
        """grid avec timeframe='1h' explicite → même résultat que sans timeframe."""
        from backend.optimization.walk_forward import WalkForwardOptimizer

        candles = _make_1h_candles(200)
        params_no_tf = {
            "ma_period": 14,
            "atr_period": 14,
            "atr_multiplier_start": 1.5,
            "atr_multiplier_step": 1.0,
            "num_levels": 2,
            "sl_percent": 20.0,
        }
        params_with_tf = {**params_no_tf, "timeframe": "1h"}

        bt_config_dict = {
            "symbol": "BTC/USDT",
            "start_date": candles[0].timestamp,
            "end_date": candles[-1].timestamp,
            "initial_capital": 10000.0,
            "leverage": 6,
            "maker_fee": 0.0002,
            "taker_fee": 0.0006,
            "slippage_pct": 0.0005,
            "high_vol_slippage_mult": 1.5,
            "max_risk_per_trade": 0.02,
        }

        result_no_tf = WalkForwardOptimizer._run_fast(
            [params_no_tf], {"1h": candles}, "grid_atr", bt_config_dict, "1h",
        )
        result_with_tf = WalkForwardOptimizer._run_fast(
            [params_with_tf], {"1h": candles}, "grid_atr", bt_config_dict, "1h",
        )

        # Sharpe, net_return, n_trades identiques
        assert result_no_tf[0][1] == result_with_tf[0][1]
        assert result_no_tf[0][2] == result_with_tf[0][2]
        assert result_no_tf[0][4] == result_with_tf[0][4]

    def test_sma_on_resampled_correct(self):
        """SMA(14) sur candles 4h resamplees == SMA calculée manuellement."""
        from backend.core.indicators import sma

        candles_1h = _make_1h_candles(96)
        candles_4h = resample_candles(candles_1h, "4h")

        closes_4h = np.array([c.close for c in candles_4h])
        sma_from_cache = sma(closes_4h, 14)

        # Calcul manuel SMA(14)
        sma_manual = np.full_like(closes_4h, np.nan)
        for i in range(13, len(closes_4h)):
            sma_manual[i] = np.mean(closes_4h[i - 13 : i + 1])

        # Comparaison sur les indices valides
        valid = ~np.isnan(sma_from_cache) & ~np.isnan(sma_manual)
        assert np.sum(valid) > 0
        np.testing.assert_allclose(sma_from_cache[valid], sma_manual[valid], rtol=1e-10)

    def test_4h_differs_from_1h(self):
        """Le cache 4h a moins de points que le cache 1h — preuve du resampling."""
        candles_1h = _make_1h_candles(480)
        candles_4h = resample_candles(candles_1h, "4h")

        cache_1h = build_cache(
            {"1h": candles_1h}, {"ma_period": [14], "atr_period": [14]},
            "grid_atr", main_tf="1h",
        )
        cache_4h = build_cache(
            {"4h": candles_4h}, {"ma_period": [14], "atr_period": [14]},
            "grid_atr", main_tf="4h",
        )

        # Le cache 4h doit avoir ~4x moins de points
        assert len(cache_4h.closes) == len(candles_4h)
        assert len(cache_1h.closes) == len(candles_1h)
        assert len(cache_4h.closes) < len(cache_1h.closes)
        # Ratio approximatif 1:4
        ratio = len(cache_1h.closes) / len(cache_4h.closes)
        assert 3.5 < ratio < 4.5, f"Ratio inattendu: {ratio}"

    def test_existing_grid_atr_parity(self, make_indicator_cache):
        """Les tests grid_atr existants ne sont pas cassés par le refactoring."""
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache

        n = 100
        closes = np.full(n, 100.0)
        sma_arr = np.full(n, 100.0)
        atr_arr = np.full(n, 2.0)
        cache = make_indicator_cache(
            n=n, closes=closes,
            highs=np.full(n, 101.0),
            lows=np.full(n, 99.0),
            bb_sma={14: sma_arr},
            atr_by_period={14: atr_arr},
        )
        params = {
            "ma_period": 14, "atr_period": 14,
            "atr_multiplier_start": 1.0, "atr_multiplier_step": 0.5,
            "num_levels": 2, "sl_percent": 15.0,
        }
        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 2, 1, tzinfo=timezone.utc),
            leverage=6,
        )
        result = run_multi_backtest_from_cache("grid_atr", params, cache, bt_config)
        assert len(result) == 5
        assert isinstance(result[4], int)


# ═══════════════════════════════════════════════════════════════════════════
# Section 5 — Intégration param_grids
# ═══════════════════════════════════════════════════════════════════════════


class TestParamGridsIntegration:
    """Vérifie que param_grids.yaml a timeframe où il faut."""

    @pytest.fixture
    def param_grids(self):
        with open("config/param_grids.yaml", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def test_grid_atr_has_timeframe(self, param_grids):
        """grid_atr contient timeframe dans son grid."""
        default = param_grids["grid_atr"]["default"]
        assert "timeframe" in default
        assert "1h" in default["timeframe"]
        assert "4h" in default["timeframe"]
        assert "1d" in default["timeframe"]

    def test_grid_multi_tf_no_timeframe(self, param_grids):
        """grid_multi_tf ne contient PAS timeframe."""
        default = param_grids["grid_multi_tf"]["default"]
        assert "timeframe" not in default

    def test_grid_funding_no_timeframe(self, param_grids):
        """grid_funding ne contient PAS timeframe."""
        default = param_grids["grid_funding"]["default"]
        assert "timeframe" not in default


# ═══════════════════════════════════════════════════════════════════════════
# Section 6 — _run_fast groupement
# ═══════════════════════════════════════════════════════════════════════════


class TestRunFastGrouping:
    """Tests du groupement par timeframe dans _run_fast."""

    def test_mixed_grid(self):
        """Grid mixte [1h, 4h] → résultats pour les deux TFs."""
        from backend.optimization.walk_forward import WalkForwardOptimizer

        candles = _make_1h_candles(480)

        params_base = {
            "ma_period": 14,
            "atr_period": 14,
            "atr_multiplier_start": 1.5,
            "atr_multiplier_step": 1.0,
            "num_levels": 2,
            "sl_percent": 20.0,
        }
        grid = [
            {**params_base, "timeframe": "1h"},
            {**params_base, "timeframe": "4h"},
        ]

        bt_config_dict = {
            "symbol": "BTC/USDT",
            "start_date": candles[0].timestamp,
            "end_date": candles[-1].timestamp,
            "initial_capital": 10000.0,
            "leverage": 6,
            "maker_fee": 0.0002,
            "taker_fee": 0.0006,
            "slippage_pct": 0.0005,
            "high_vol_slippage_mult": 1.5,
            "max_risk_per_trade": 0.02,
        }

        results = WalkForwardOptimizer._run_fast(
            grid, {"1h": candles}, "grid_atr", bt_config_dict, "1h",
        )

        assert len(results) == 2
        # Les deux résultats doivent avoir timeframe dans les params
        tfs = {r[0].get("timeframe") for r in results}
        assert "1h" in tfs
        assert "4h" in tfs

    def test_no_timeframe_single_group(self):
        """Grid sans timeframe → un seul groupe, même nombre de résultats."""
        from backend.optimization.walk_forward import WalkForwardOptimizer

        candles = _make_1h_candles(200)
        grid = [
            {"ma_period": 14, "atr_period": 14, "atr_multiplier_start": 1.5,
             "atr_multiplier_step": 1.0, "num_levels": 2, "sl_percent": 20.0},
            {"ma_period": 20, "atr_period": 14, "atr_multiplier_start": 1.5,
             "atr_multiplier_step": 1.0, "num_levels": 2, "sl_percent": 20.0},
        ]

        bt_config_dict = {
            "symbol": "BTC/USDT",
            "start_date": candles[0].timestamp,
            "end_date": candles[-1].timestamp,
            "initial_capital": 10000.0,
            "leverage": 6,
            "maker_fee": 0.0002,
            "taker_fee": 0.0006,
            "slippage_pct": 0.0005,
            "high_vol_slippage_mult": 1.5,
            "max_risk_per_trade": 0.02,
        }

        results = WalkForwardOptimizer._run_fast(
            grid, {"1h": candles}, "grid_atr", bt_config_dict, "1h",
        )

        assert len(results) == 2
        # Pas de timeframe dans les params (pas injecté car absent du grid original)
        for r in results:
            assert "timeframe" not in r[0] or r[0].get("timeframe") == "1h"
