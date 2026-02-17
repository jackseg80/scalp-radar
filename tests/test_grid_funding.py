"""Tests pour la stratégie Grid Funding (Sprint 22).

Sections :
1. Funding rate alignment (cache loader)
2. Entry signals (fast engine)
3. TP/SL (stratégie class + fast engine)
4. Funding payments PnL
5. Fast engine simulation complète
6. Registry + config
7. Cache + DB
"""

from __future__ import annotations

import math
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from backend.backtesting.engine import BacktestConfig
from backend.core.config import GridFundingConfig
from backend.core.models import Candle, Direction
from backend.strategies.base import StrategyContext
from backend.strategies.base_grid import GridPosition, GridState
from backend.strategies.grid_funding import GridFundingStrategy


# ─── Helpers ──────────────────────────────────────────────────────────────

def _make_candles(n: int, *, base_price: float = 100.0, tf_hours: int = 1) -> list[Candle]:
    """Crée n bougies 1h à prix constant."""
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        Candle(
            timestamp=t0 + timedelta(hours=i * tf_hours),
            open=base_price,
            high=base_price + 1,
            low=base_price - 1,
            close=base_price,
            volume=100.0,
            symbol="TEST/USDT",
            timeframe="1h",
        )
        for i in range(n)
    ]


def _make_context(
    *,
    close: float = 100.0,
    sma: float = 100.0,
    funding_rate: float | None = None,
    timestamp: datetime | None = None,
) -> StrategyContext:
    """Crée un StrategyContext minimal pour les tests."""
    extra: dict[str, Any] = {}
    if funding_rate is not None:
        extra["funding_rate"] = funding_rate  # raw decimal (comme build_extra_data_map)
    return StrategyContext(
        symbol="TEST/USDT",
        timestamp=timestamp or datetime(2024, 1, 1, tzinfo=timezone.utc),
        candles={"1h": []},
        indicators={"1h": {"sma": sma, "close": close}},
        current_position=None,
        capital=10000.0,
        config=None,  # type: ignore[arg-type]
        extra_data=extra,
    )


def _make_grid_state(
    *,
    avg_entry: float = 100.0,
    n_positions: int = 1,
    first_entry_time: datetime | None = None,
) -> GridState:
    """Crée un GridState minimal avec positions."""
    if first_entry_time is None:
        first_entry_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    positions = [
        GridPosition(
            level=i,
            direction=Direction.LONG,
            entry_price=avg_entry,
            quantity=1.0,
            entry_time=first_entry_time + timedelta(hours=i),
            entry_fee=0.06,
        )
        for i in range(n_positions)
    ]
    return GridState(
        positions=positions,
        avg_entry_price=avg_entry,
        total_quantity=float(n_positions),
        total_notional=avg_entry * n_positions,
        unrealized_pnl=0.0,
    )


def _make_funding_db(db_path: str, rows: list[tuple]) -> None:
    """Crée une table funding_rates et insère les données."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS funding_rates ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  symbol TEXT NOT NULL,"
        "  exchange TEXT NOT NULL,"
        "  timestamp REAL NOT NULL,"
        "  funding_rate REAL NOT NULL"
        ")"
    )
    conn.executemany(
        "INSERT INTO funding_rates (symbol, exchange, timestamp, funding_rate) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


# ─── Section 1 : Funding rate alignment ──────────────────────────────────


class TestFundingAlignment:
    """Tests pour _load_funding_rates_aligned()."""

    def test_searchsorted_direct_no_lookahead(self):
        """Le taux settlé à T est utilisé pour la candle T (pas T+8h)."""
        from backend.optimization.indicator_cache import _load_funding_rates_aligned

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        # Funding à T=1000 (epoch ms) avec rate=-0.05%
        _make_funding_db(db_path, [("TEST/USDT", "binance", 1000.0, -0.05)])
        candle_ts = np.array([900.0, 1000.0, 1100.0, 2000.0])

        result = _load_funding_rates_aligned("TEST/USDT", "binance", candle_ts, db_path)

        assert np.isnan(result[0])  # Avant le premier funding
        assert result[1] == pytest.approx(-0.0005)  # À T=1000, /100
        assert result[2] == pytest.approx(-0.0005)  # Forward-fill
        assert result[3] == pytest.approx(-0.0005)  # Forward-fill

    def test_forward_fill_on_hourly(self):
        """Forward-fill entre deux funding rates à 8h d'intervalle."""
        from backend.optimization.indicator_cache import _load_funding_rates_aligned

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        t0 = 1_700_000_000_000.0  # epoch ms
        _make_funding_db(db_path, [
            ("TEST/USDT", "binance", t0, -0.10),           # -0.001 raw
            ("TEST/USDT", "binance", t0 + 8 * 3600_000, 0.02),  # +0.0002 raw
        ])
        # 12 candles horaires
        candle_ts = np.array([t0 + i * 3600_000 for i in range(12)])

        result = _load_funding_rates_aligned("TEST/USDT", "binance", candle_ts, db_path)

        # Les 8 premières → -0.001
        assert result[0] == pytest.approx(-0.001)
        assert result[7] == pytest.approx(-0.001)
        # À partir de h8 → +0.0002
        assert result[8] == pytest.approx(0.0002)
        assert result[11] == pytest.approx(0.0002)

    def test_nan_before_first_funding(self):
        """NaN pour les candles avant le premier funding connu."""
        from backend.optimization.indicator_cache import _load_funding_rates_aligned

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        _make_funding_db(db_path, [("TEST/USDT", "binance", 5000.0, -0.03)])
        candle_ts = np.array([1000.0, 2000.0, 3000.0, 5000.0, 6000.0])

        result = _load_funding_rates_aligned("TEST/USDT", "binance", candle_ts, db_path)

        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])
        assert result[3] == pytest.approx(-0.0003)
        assert result[4] == pytest.approx(-0.0003)

    def test_db_percent_to_raw_decimal(self):
        """Conversion DB percent → raw decimal (/100)."""
        from backend.optimization.indicator_cache import _load_funding_rates_aligned

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        # -0.05% en DB = -0.0005 raw
        _make_funding_db(db_path, [("TEST/USDT", "binance", 100.0, -0.05)])
        candle_ts = np.array([100.0])

        result = _load_funding_rates_aligned("TEST/USDT", "binance", candle_ts, db_path)
        assert result[0] == pytest.approx(-0.0005)

    def test_empty_db_returns_nan(self):
        """Si pas de funding en DB, retourne NaN partout."""
        from backend.optimization.indicator_cache import _load_funding_rates_aligned

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        _make_funding_db(db_path, [])  # table vide
        candle_ts = np.array([1000.0, 2000.0])

        result = _load_funding_rates_aligned("TEST/USDT", "binance", candle_ts, db_path)
        assert np.all(np.isnan(result))


# ─── Section 2 : Entry signals ───────────────────────────────────────────


class TestEntrySignals:
    """Tests pour _build_entry_signals()."""

    def test_single_level_signal(self, make_indicator_cache):
        """Signal level 0 quand funding <= -threshold_start."""
        from backend.optimization.fast_multi_backtest import _build_entry_signals

        n = 10
        funding = np.full(n, -0.001)  # raw decimal, très négatif
        cache = make_indicator_cache(n=n, funding_rates_1h=funding)
        params = {"funding_threshold_start": 0.0005, "funding_threshold_step": 0.0005}

        signals = _build_entry_signals(cache, params, num_levels=3)

        assert signals.shape == (n, 3)
        assert signals[0, 0] is np.True_  # -0.001 <= -0.0005
        assert signals[0, 1] is np.True_  # -0.001 <= -0.001
        assert signals[0, 2] is np.False_  # -0.001 > -0.0015

    def test_no_signal_if_positive(self, make_indicator_cache):
        """Pas de signal si funding positif."""
        from backend.optimization.fast_multi_backtest import _build_entry_signals

        n = 5
        funding = np.full(n, 0.001)  # positif
        cache = make_indicator_cache(n=n, funding_rates_1h=funding)
        params = {"funding_threshold_start": 0.0005, "funding_threshold_step": 0.0005}

        signals = _build_entry_signals(cache, params, num_levels=2)
        assert not signals.any()

    def test_no_signal_if_nan(self, make_indicator_cache):
        """NaN funding = pas de signal."""
        from backend.optimization.fast_multi_backtest import _build_entry_signals

        n = 5
        funding = np.full(n, np.nan)
        cache = make_indicator_cache(n=n, funding_rates_1h=funding)
        params = {"funding_threshold_start": 0.0005, "funding_threshold_step": 0.0005}

        signals = _build_entry_signals(cache, params, num_levels=2)
        assert not signals.any()

    def test_multi_level_progressive(self, make_indicator_cache):
        """Chaque niveau a un seuil plus négatif."""
        from backend.optimization.fast_multi_backtest import _build_entry_signals

        n = 5
        # Funding = -0.0007 → level 0 (-0.0005) oui, level 1 (-0.001) non
        funding = np.full(n, -0.0007)
        cache = make_indicator_cache(n=n, funding_rates_1h=funding)
        params = {"funding_threshold_start": 0.0005, "funding_threshold_step": 0.0005}

        signals = _build_entry_signals(cache, params, num_levels=3)

        assert signals[:, 0].all()   # level 0 actif
        assert not signals[:, 1].any()  # level 1 inactif
        assert not signals[:, 2].any()  # level 2 inactif

    def test_no_funding_cache_returns_false(self, make_indicator_cache):
        """Si funding_rates_1h est None, aucun signal."""
        from backend.optimization.fast_multi_backtest import _build_entry_signals

        cache = make_indicator_cache(n=5)  # pas de funding_rates_1h
        params = {"funding_threshold_start": 0.0005, "funding_threshold_step": 0.0005}

        signals = _build_entry_signals(cache, params, num_levels=2)
        assert not signals.any()


# ─── Section 3 : TP/SL ───────────────────────────────────────────────────


class TestTPSL:
    """Tests TP/SL pour la stratégie class."""

    def test_sl_triggers(self):
        """SL se déclenche quand close <= avg_entry * (1 - sl_pct)."""
        cfg = GridFundingConfig(sl_percent=10.0)
        strat = GridFundingStrategy(cfg)

        gs = _make_grid_state(avg_entry=100.0)
        # close=89 < 90 (= 100 * 0.9) → SL
        ctx = _make_context(close=89.0, sma=110.0, funding_rate=-0.05)
        assert strat.should_close_all(ctx, gs) == "sl_global"

    def test_sl_during_min_hold(self):
        """SL reste actif même pendant min_hold."""
        cfg = GridFundingConfig(sl_percent=10.0, min_hold_candles=100)
        strat = GridFundingStrategy(cfg)

        gs = _make_grid_state(avg_entry=100.0)
        # timestamp = juste 1h après l'entrée → min_hold pas passé
        ctx = _make_context(
            close=85.0, sma=110.0, funding_rate=-0.05,
            timestamp=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
        )
        assert strat.should_close_all(ctx, gs) == "sl_global"

    def test_tp_funding_positive(self):
        """TP = funding > 0 (mode funding_positive)."""
        cfg = GridFundingConfig(tp_mode="funding_positive", min_hold_candles=0)
        strat = GridFundingStrategy(cfg)

        gs = _make_grid_state(avg_entry=100.0)
        ctx = _make_context(
            close=100.0, sma=110.0, funding_rate=0.01,  # positif en %
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        assert strat.should_close_all(ctx, gs) == "tp_funding"

    def test_tp_sma_cross(self):
        """TP = close >= SMA (mode sma_cross)."""
        cfg = GridFundingConfig(tp_mode="sma_cross", min_hold_candles=0)
        strat = GridFundingStrategy(cfg)

        gs = _make_grid_state(avg_entry=95.0)
        ctx = _make_context(
            close=101.0, sma=100.0, funding_rate=-0.05,  # négatif mais SMA touchée
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        assert strat.should_close_all(ctx, gs) == "tp_sma"

    def test_tp_funding_or_sma(self):
        """TP = funding > 0 OU SMA (mode funding_or_sma)."""
        cfg = GridFundingConfig(tp_mode="funding_or_sma", min_hold_candles=0)
        strat = GridFundingStrategy(cfg)

        gs = _make_grid_state(avg_entry=95.0)

        # Funding positif → tp_funding
        ctx1 = _make_context(
            close=90.0, sma=100.0, funding_rate=0.01,
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        assert strat.should_close_all(ctx1, gs) == "tp_funding"

        # SMA cross → tp_sma
        ctx2 = _make_context(
            close=101.0, sma=100.0, funding_rate=-0.01,
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        assert strat.should_close_all(ctx2, gs) == "tp_sma"

    def test_min_hold_blocks_tp(self):
        """Min hold bloque le TP (mais pas le SL)."""
        cfg = GridFundingConfig(tp_mode="funding_positive", min_hold_candles=10)
        strat = GridFundingStrategy(cfg)

        gs = _make_grid_state(avg_entry=100.0)
        # Seulement 1h après l'entrée → min_hold pas atteint
        ctx = _make_context(
            close=100.0, sma=90.0, funding_rate=0.01,
            timestamp=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
        )
        assert strat.should_close_all(ctx, gs) is None


# ─── Section 4 : Funding payments PnL ────────────────────────────────────


class TestFundingPaymentsPnL:
    """Tests pour _calc_grid_pnl_with_funding()."""

    def test_bonus_with_negative_funding(self):
        """Funding négatif LONG = bonus (on reçoit des shorts)."""
        from backend.optimization.fast_multi_backtest import _calc_grid_pnl_with_funding

        # 3 candles, entry=100, qty=1, entry_idx=0, exit_idx=2, exit_price=100
        # Frontière 8h à candle 0 (ts=0) : hour=0, 0%8==0 → oui
        funding = np.array([-0.001, 0.0, 0.0])  # raw decimal
        timestamps = np.array([0.0, 3600_000.0, 7200_000.0])  # epoch ms

        pnl = _calc_grid_pnl_with_funding(
            [(100.0, 1.0, 0)],  # positions
            100.0,  # exit_price (flat)
            2,  # exit_idx
            funding, timestamps,
            taker_fee=0.0, slippage_pct=0.0,
        )
        # Prix flat → price_pnl = 0
        # Funding : -(-0.001) * 100 = +0.1
        assert pnl == pytest.approx(0.1)

    def test_cost_with_positive_funding(self):
        """Funding positif LONG = coût (on paie les shorts)."""
        from backend.optimization.fast_multi_backtest import _calc_grid_pnl_with_funding

        funding = np.array([0.001, 0.0, 0.0])
        timestamps = np.array([0.0, 3600_000.0, 7200_000.0])

        pnl = _calc_grid_pnl_with_funding(
            [(100.0, 1.0, 0)],
            100.0, 2, funding, timestamps,
            taker_fee=0.0, slippage_pct=0.0,
        )
        # -0.001 * 100 = -0.1
        assert pnl == pytest.approx(-0.1)

    def test_zero_funding_classic_pnl(self):
        """Funding = 0 → PnL classique."""
        from backend.optimization.fast_multi_backtest import _calc_grid_pnl_with_funding

        funding = np.zeros(5)
        timestamps = np.array([i * 3600_000 for i in range(5)])

        # entry=100, exit=110, qty=1 → price_pnl = +10
        pnl = _calc_grid_pnl_with_funding(
            [(100.0, 1.0, 0)],
            110.0, 4, funding, timestamps,
            taker_fee=0.0, slippage_pct=0.0,
        )
        assert pnl == pytest.approx(10.0)

    def test_8h_boundary_detection(self):
        """Les payments ne se produisent qu'aux frontières 8h (00:00, 08:00, 16:00 UTC)."""
        from backend.optimization.fast_multi_backtest import _calc_grid_pnl_with_funding

        # 24 candles horaires commençant à 00:00 UTC
        timestamps = np.array([i * 3600_000 for i in range(24)])
        funding = np.full(24, -0.001)  # -0.1% constant

        pnl = _calc_grid_pnl_with_funding(
            [(100.0, 1.0, 0)],
            100.0, 23, funding, timestamps,
            taker_fee=0.0, slippage_pct=0.0,
        )
        # Frontières 8h dans [0, 23) : h=0, h=8, h=16 → 3 payments
        # Chaque : -(-0.001) * 100 = +0.1
        assert pnl == pytest.approx(0.3)

    def test_multi_positions_accumulate(self):
        """Plusieurs positions accumulent les funding payments."""
        from backend.optimization.fast_multi_backtest import _calc_grid_pnl_with_funding

        funding = np.array([-0.001, 0.0, 0.0])
        timestamps = np.array([0.0, 3600_000.0, 7200_000.0])

        pnl = _calc_grid_pnl_with_funding(
            [(100.0, 1.0, 0), (100.0, 1.0, 0)],  # 2 positions
            100.0, 2, funding, timestamps,
            taker_fee=0.0, slippage_pct=0.0,
        )
        # 2 positions × +0.1 chacune = +0.2
        assert pnl == pytest.approx(0.2)


# ─── Section 5 : Fast engine simulation ──────────────────────────────────


class TestFastEngineSimulation:
    """Tests pour _simulate_grid_funding() et run_multi_backtest_from_cache()."""

    def _make_bt_config(self) -> BacktestConfig:
        return BacktestConfig(
            symbol="TEST/USDT",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            initial_capital=10000.0,
            leverage=6,
        )

    def test_run_without_crash(self, make_indicator_cache):
        """La simulation tourne sans crash."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_funding

        n = 100
        funding = np.full(n, -0.001)  # Négatif constant
        sma = np.full(n, 100.0)
        ts = np.array([i * 3600_000 for i in range(n)])

        cache = make_indicator_cache(
            n=n,
            bb_sma={14: sma},
            funding_rates_1h=funding,
            candle_timestamps=ts,
        )

        params = {
            "funding_threshold_start": 0.0005,
            "funding_threshold_step": 0.0005,
            "num_levels": 3,
            "ma_period": 14,
            "sl_percent": 15.0,
            "tp_mode": "funding_or_sma",
            "min_hold_candles": 8,
        }
        bt_config = self._make_bt_config()

        trade_pnls, trade_returns, final_capital = _simulate_grid_funding(cache, params, bt_config)

        assert isinstance(trade_pnls, list)
        assert isinstance(final_capital, float)
        assert final_capital > 0

    def test_entry_on_negative_funding(self, make_indicator_cache):
        """Des positions sont ouvertes quand le funding est négatif."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_funding

        n = 50
        # Funding très négatif = signal d'entrée
        funding = np.full(n, -0.005)
        # SMA au-dessus → pas de TP SMA
        sma = np.full(n, 200.0)
        ts = np.array([i * 3600_000 for i in range(n)])
        closes = np.full(n, 100.0)

        cache = make_indicator_cache(
            n=n, closes=closes,
            bb_sma={14: sma},
            funding_rates_1h=funding,
            candle_timestamps=ts,
        )

        params = {
            "funding_threshold_start": 0.001,
            "funding_threshold_step": 0.001,
            "num_levels": 2,
            "ma_period": 14,
            "sl_percent": 50.0,  # SL très large → pas de SL
            "tp_mode": "sma_cross",
            "min_hold_candles": 0,
        }
        bt_config = self._make_bt_config()

        trade_pnls, _, final_capital = _simulate_grid_funding(cache, params, bt_config)

        # Au minimum un trade (force close en fin)
        assert len(trade_pnls) >= 1

    def test_no_entry_if_funding_positive(self, make_indicator_cache):
        """Aucune position si le funding est toujours positif."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_funding

        n = 50
        funding = np.full(n, 0.001)  # Positif
        sma = np.full(n, 100.0)
        ts = np.array([i * 3600_000 for i in range(n)])

        cache = make_indicator_cache(
            n=n, bb_sma={14: sma},
            funding_rates_1h=funding,
            candle_timestamps=ts,
        )

        params = {
            "funding_threshold_start": 0.0005,
            "funding_threshold_step": 0.0005,
            "num_levels": 2,
            "ma_period": 14,
            "sl_percent": 15.0,
            "tp_mode": "funding_or_sma",
            "min_hold_candles": 0,
        }
        bt_config = self._make_bt_config()

        trade_pnls, _, final_capital = _simulate_grid_funding(cache, params, bt_config)

        # Aucun trade, capital inchangé
        assert len(trade_pnls) == 0
        assert final_capital == pytest.approx(bt_config.initial_capital)

    def test_exit_on_sl(self, make_indicator_cache):
        """Sortie sur SL quand le prix chute."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_funding

        n = 50
        funding = np.full(n, -0.005)
        sma = np.full(n, 200.0)  # SMA très haut → pas de TP
        ts = np.array([i * 3600_000 for i in range(n)])

        # Prix qui chute après 20 bougies
        closes = np.full(n, 100.0)
        closes[20:] = 80.0  # -20% → SL à 15% devrait trigger

        cache = make_indicator_cache(
            n=n, closes=closes,
            bb_sma={14: sma},
            funding_rates_1h=funding,
            candle_timestamps=ts,
        )

        params = {
            "funding_threshold_start": 0.001,
            "funding_threshold_step": 0.001,
            "num_levels": 2,
            "ma_period": 14,
            "sl_percent": 15.0,
            "tp_mode": "sma_cross",
            "min_hold_candles": 0,
        }
        bt_config = self._make_bt_config()

        trade_pnls, _, _ = _simulate_grid_funding(cache, params, bt_config)

        # Au moins un trade négatif (SL)
        assert any(pnl < 0 for pnl in trade_pnls)

    def test_exit_on_funding_positive(self, make_indicator_cache):
        """Sortie TP quand le funding redevient positif."""
        from backend.optimization.fast_multi_backtest import _simulate_grid_funding

        n = 50
        # Funding négatif au début, positif à partir de la bougie 30
        funding = np.full(n, -0.005)
        funding[30:] = 0.001  # Positif → TP

        sma = np.full(n, 200.0)  # SMA haut → pas de SMA TP
        ts = np.array([i * 3600_000 for i in range(n)])
        closes = np.full(n, 100.0)

        cache = make_indicator_cache(
            n=n, closes=closes,
            bb_sma={14: sma},
            funding_rates_1h=funding,
            candle_timestamps=ts,
        )

        params = {
            "funding_threshold_start": 0.001,
            "funding_threshold_step": 0.001,
            "num_levels": 2,
            "ma_period": 14,
            "sl_percent": 50.0,  # SL très large
            "tp_mode": "funding_positive",
            "min_hold_candles": 0,
        }
        bt_config = self._make_bt_config()

        trade_pnls, _, _ = _simulate_grid_funding(cache, params, bt_config)

        # Au moins un trade (TP funding)
        assert len(trade_pnls) >= 1

    def test_run_multi_backtest_from_cache_dispatch(self, make_indicator_cache):
        """run_multi_backtest_from_cache dispatche correctement grid_funding."""
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache

        n = 50
        funding = np.full(n, -0.005)
        sma = np.full(n, 100.0)
        ts = np.array([i * 3600_000 for i in range(n)])

        cache = make_indicator_cache(
            n=n, bb_sma={14: sma},
            funding_rates_1h=funding,
            candle_timestamps=ts,
        )

        params = {
            "funding_threshold_start": 0.001,
            "funding_threshold_step": 0.001,
            "num_levels": 2,
            "ma_period": 14,
            "sl_percent": 15.0,
            "tp_mode": "funding_or_sma",
            "min_hold_candles": 0,
        }
        bt_config = self._make_bt_config()

        result = run_multi_backtest_from_cache("grid_funding", params, cache, bt_config)

        assert isinstance(result, tuple)
        assert len(result) == 5
        assert result[0] == params  # params retournés


# ─── Section 6 : Registry + config ───────────────────────────────────────


class TestRegistryConfig:
    """Tests d'intégration registry et config."""

    def test_in_strategy_registry(self):
        """grid_funding est dans STRATEGY_REGISTRY."""
        from backend.optimization import STRATEGY_REGISTRY
        assert "grid_funding" in STRATEGY_REGISTRY

    def test_in_grid_strategies(self):
        """grid_funding est dans GRID_STRATEGIES."""
        from backend.optimization import GRID_STRATEGIES
        assert "grid_funding" in GRID_STRATEGIES

    def test_in_strategies_need_extra_data(self):
        """grid_funding est dans STRATEGIES_NEED_EXTRA_DATA."""
        from backend.optimization import STRATEGIES_NEED_EXTRA_DATA
        assert "grid_funding" in STRATEGIES_NEED_EXTRA_DATA

    def test_in_fast_engine_strategies(self):
        """grid_funding est dans FAST_ENGINE_STRATEGIES (décuplé de NEED_EXTRA_DATA)."""
        from backend.optimization import FAST_ENGINE_STRATEGIES
        assert "grid_funding" in FAST_ENGINE_STRATEGIES

    def test_create_with_params(self):
        """create_strategy_with_params fonctionne pour grid_funding."""
        from backend.optimization import create_strategy_with_params

        strat = create_strategy_with_params("grid_funding", {
            "funding_threshold_start": 0.0005,
            "ma_period": 14,
        })
        assert strat.name == "grid_funding"

    def test_config_defaults(self):
        """Les defaults GridFundingConfig sont corrects."""
        cfg = GridFundingConfig()
        assert cfg.enabled is False
        assert cfg.sides == ["long"]
        assert cfg.leverage == 6
        assert cfg.tp_mode == "funding_or_sma"
        assert cfg.min_hold_candles == 8

    def test_indicator_params_registered(self):
        """grid_funding est dans _INDICATOR_PARAMS."""
        from backend.optimization.walk_forward import _INDICATOR_PARAMS
        assert "grid_funding" in _INDICATOR_PARAMS
        assert _INDICATOR_PARAMS["grid_funding"] == ["ma_period"]


# ─── Section 7 : Cache + DB ──────────────────────────────────────────────


class TestCacheDB:
    """Tests pour build_cache avec grid_funding."""

    def test_build_cache_loads_funding(self):
        """build_cache charge funding_rates_1h (non None)."""
        from backend.optimization.indicator_cache import build_cache

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        # Créer des candles
        n = 30
        candles = _make_candles(n)
        t0 = candles[0].timestamp.timestamp() * 1000

        # Funding rate
        _make_funding_db(db_path, [
            ("TEST/USDT", "binance", t0, -0.05),
            ("TEST/USDT", "binance", t0 + 8 * 3600_000, -0.03),
        ])

        cache = build_cache(
            {"1h": candles},
            {"ma_period": [14], "funding_threshold_start": [0.0005]},
            "grid_funding",
            main_tf="1h",
            db_path=db_path,
            symbol="TEST/USDT",
            exchange="binance",
        )

        assert cache.funding_rates_1h is not None
        assert len(cache.funding_rates_1h) == n
        assert cache.candle_timestamps is not None

    def test_grid_atr_cache_unchanged(self, make_indicator_cache):
        """grid_atr cache n'a pas de funding (None par défaut)."""
        cache = make_indicator_cache(n=10)
        assert cache.funding_rates_1h is None
        assert cache.candle_timestamps is None

    def test_no_error_without_db_path(self):
        """Sans db_path, build_cache charge sans funding (pas d'erreur)."""
        from backend.optimization.indicator_cache import build_cache

        candles = _make_candles(30)
        cache = build_cache(
            {"1h": candles},
            {"ma_period": [14]},
            "grid_funding",
            main_tf="1h",
        )
        # Funding non chargé mais pas de crash
        assert cache.funding_rates_1h is None
        assert cache.candle_timestamps is not None  # timestamps toujours remplis

    def test_candle_timestamps_filled(self):
        """candle_timestamps est rempli pour grid_funding."""
        from backend.optimization.indicator_cache import build_cache

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        n = 20
        candles = _make_candles(n)
        _make_funding_db(db_path, [("TEST/USDT", "binance", 0.0, -0.01)])

        cache = build_cache(
            {"1h": candles},
            {"ma_period": [14]},
            "grid_funding",
            main_tf="1h",
            db_path=db_path,
            symbol="TEST/USDT",
            exchange="binance",
        )

        assert cache.candle_timestamps is not None
        assert len(cache.candle_timestamps) == n
        # Vérifier que c'est en epoch ms
        assert cache.candle_timestamps[0] > 1_000_000_000  # epoch ms > 1970


# ─── Section bonus : compute_grid (strategy class) ───────────────────────


class TestComputeGrid:
    """Tests pour GridFundingStrategy.compute_grid()."""

    def test_grid_levels_on_negative_funding(self):
        """Retourne des GridLevel quand le funding est assez négatif."""
        cfg = GridFundingConfig(
            funding_threshold_start=0.0005,
            funding_threshold_step=0.0005,
            num_levels=3,
        )
        strat = GridFundingStrategy(cfg)

        gs = GridState(positions=[], avg_entry_price=0.0, total_quantity=0.0, total_notional=0.0, unrealized_pnl=0.0)
        # Funding = -0.1% → raw decimal = -0.001 (extra_data_builder divise par 100)
        ctx = _make_context(close=100.0, funding_rate=-0.001)

        levels = strat.compute_grid(ctx, gs)

        assert len(levels) == 2  # -0.001 <= -0.0005 (L0), <= -0.001 (L1), > -0.0015 (L2 non)
        assert all(lv.direction == Direction.LONG for lv in levels)
        assert levels[0].entry_price == 100.0

    def test_no_grid_if_funding_positive(self):
        """Pas de niveaux si funding positif."""
        cfg = GridFundingConfig()
        strat = GridFundingStrategy(cfg)

        gs = GridState(positions=[], avg_entry_price=0.0, total_quantity=0.0, total_notional=0.0, unrealized_pnl=0.0)
        ctx = _make_context(close=100.0, funding_rate=0.01)

        levels = strat.compute_grid(ctx, gs)
        assert len(levels) == 0

    def test_no_grid_if_no_funding(self):
        """Pas de niveaux si pas de funding data."""
        cfg = GridFundingConfig()
        strat = GridFundingStrategy(cfg)

        gs = GridState(positions=[], avg_entry_price=0.0, total_quantity=0.0, total_notional=0.0, unrealized_pnl=0.0)
        ctx = _make_context(close=100.0)  # pas de funding_rate

        levels = strat.compute_grid(ctx, gs)
        assert len(levels) == 0

    def test_skip_filled_levels(self):
        """Les niveaux déjà remplis sont exclus."""
        cfg = GridFundingConfig(
            funding_threshold_start=0.0005,
            funding_threshold_step=0.0005,
            num_levels=3,
        )
        strat = GridFundingStrategy(cfg)

        # Level 0 déjà rempli
        gs = GridState(
            positions=[GridPosition(
                level=0, direction=Direction.LONG,
                entry_price=100.0, quantity=1.0,
                entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                entry_fee=0.06,
            )],
            avg_entry_price=100.0, total_quantity=1.0, total_notional=100.0, unrealized_pnl=0.0,
        )
        ctx = _make_context(close=100.0, funding_rate=-0.001)

        levels = strat.compute_grid(ctx, gs)

        # Level 0 déjà rempli → seulement level 1 (et pas level 2 car -0.001 > -0.0015)
        assert len(levels) == 1
        assert levels[0].index == 1
