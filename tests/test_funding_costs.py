"""Tests Sprint 26 — Funding Costs dans le Backtest.

25 tests couvrant :
- Settlement detection (mask 8h UTC)
- Funding calculation (LONG/SHORT × positive/negative)
- Convention /100 (extra_data_builder, indicator_cache, grid_funding)
- Backward compat (BacktestResult.funding_paid_total default)
- Parity event-driven vs fast engine
- Portfolio aggregation (PortfolioResult.funding_paid_total)
- Edge cases (NaN funding)
"""

from __future__ import annotations

import dataclasses
import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from backend.backtesting.engine import BacktestConfig, BacktestResult
from backend.backtesting.multi_engine import MultiPositionEngine
from backend.core.models import Candle, Direction, TimeFrame
from backend.optimization.fast_multi_backtest import _simulate_grid_common
from backend.strategies.base import EXTRA_FUNDING_RATE
from backend.strategies.base_grid import BaseGridStrategy, GridLevel, GridState


# ─── Helpers ─────────────────────────────────────────────────────────────


def _make_candle(
    ts: datetime,
    price: float = 100.0,
    symbol: str = "BTC/USDT",
    tf: TimeFrame = TimeFrame.H1,
) -> Candle:
    return Candle(
        timestamp=ts,
        open=price,
        high=price + 1,
        low=price - 1,
        close=price,
        volume=1000.0,
        symbol=symbol,
        timeframe=tf,
    )


def _make_candles_24h(base: datetime, price: float = 100.0) -> list[Candle]:
    """24 bougies 1h consécutives."""
    return [_make_candle(base + timedelta(hours=i), price=price) for i in range(24)]


class _TestGridStrategy(BaseGridStrategy):
    """Stratégie grid minimale pour tests event-driven."""

    name = "grid_atr"  # Doit être dans _GRID_STRATEGIES_WITH_FUNDING

    def __init__(self, *, open_at_first: bool = True) -> None:
        self._open_at_first = open_at_first
        self._opened = False

    @property
    def max_positions(self) -> int:
        return 3

    @property
    def min_candles(self) -> dict[str, int]:
        return {"1h": 1}

    def compute_indicators(
        self, candles_by_tf: dict[str, list[Candle]]
    ) -> dict[str, dict[str, dict[str, Any]]]:
        result: dict[str, dict[str, dict[str, Any]]] = {}
        for tf, candles in candles_by_tf.items():
            indicators: dict[str, dict[str, Any]] = {}
            for c in candles:
                indicators[c.timestamp.isoformat()] = {
                    "sma": c.close + 50,
                    "close": c.close,
                }
            result[tf] = indicators
        return result

    def compute_grid(self, ctx: Any, grid_state: GridState) -> list[GridLevel]:
        if not self._open_at_first or self._opened:
            return []
        self._opened = True
        close = ctx.indicators.get("1h", {}).get("close", 100.0)
        return [
            GridLevel(
                index=0,
                entry_price=close,
                direction=Direction.LONG,
                size_fraction=1.0 / 3,
            )
        ]

    def should_close_all(self, ctx: Any, grid_state: GridState) -> str | None:
        return None

    def get_tp_price(self, grid_state: GridState, current_indicators: dict) -> float:
        return float("nan")

    def get_sl_price(self, grid_state: GridState, current_indicators: dict) -> float:
        return 0.0

    def get_params(self) -> dict[str, Any]:
        return {}


def _run_fast_sim(
    make_indicator_cache,
    funding_rate: float,
    direction: int = 1,
    n: int = 24,
) -> float:
    """Lance une simulation fast engine avec funding et retourne le capital final."""
    closes = np.full(n, 100.0)
    highs = np.full(n, 101.0)
    lows = np.full(n, 99.0)
    opens = np.full(n, 100.0)

    # Entry prices : trigger au candle 1, valide mais non-triggering ensuite
    # NE PAS utiliser NaN — la boucle skip les candles NaN entièrement (exit+funding)
    if direction == 1:
        entry_prices = np.full((n, 1), 50.0)  # 50 < low=99 → jamais touché
        entry_prices[1, 0] = 99.5  # low=99 <= 99.5 → LONG trigger
    else:
        entry_prices = np.full((n, 1), 200.0)  # 200 > high=101 → jamais touché
        entry_prices[1, 0] = 100.5  # high=101 >= 100.5 → SHORT trigger

    # SMA loin → pas de TP
    sma = np.full(n, 200.0) if direction == 1 else np.full(n, 10.0)

    funding = np.full(n, funding_rate)
    base_ms = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    candle_ts = np.array([base_ms + i * 3600000 for i in range(n)], dtype=np.float64)

    cache = make_indicator_cache(
        n=n,
        closes=closes,
        opens=opens,
        highs=highs,
        lows=lows,
        bb_sma={20: sma},
        funding_rates_1h=funding,
        candle_timestamps=candle_ts,
    )

    bt_config = BacktestConfig(
        symbol="BTC/USDT",
        start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2025, 1, 2, tzinfo=timezone.utc),
        initial_capital=10_000.0,
        leverage=10,
    )

    _, _, capital = _simulate_grid_common(
        entry_prices, sma, cache, bt_config, num_levels=1, sl_pct=0.5, direction=direction,
    )
    return capital


# ─── Section 1 : Settlement Detection (5 tests) ─────────────────────────


class TestSettlementDetection:
    def test_settlement_hour_detection_utc(self):
        """Candle 08:00 UTC est détectée comme settlement."""
        ts = datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
        assert ts.hour % 8 == 0

    def test_settlement_hour_detection_non_utc(self):
        """Candle 09:00 CET (= 08:00 UTC) détectée via conversion."""
        cet = timezone(timedelta(hours=1))
        ts = datetime(2025, 1, 1, 9, 0, 0, tzinfo=cet)
        utc_hour = ts.astimezone(timezone.utc).hour
        assert utc_hour == 8
        assert utc_hour % 8 == 0

    def test_no_settlement_between_hours(self):
        """09:00 UTC n'est pas un settlement."""
        ts = datetime(2025, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        assert ts.hour % 8 != 0

    def test_settlement_all_three_hours(self):
        """00:00, 08:00, 16:00 UTC sont tous des settlements."""
        for hour in (0, 8, 16):
            ts = datetime(2025, 1, 1, hour, 0, 0, tzinfo=timezone.utc)
            assert ts.hour % 8 == 0, f"Hour {hour} should be settlement"

    def test_settlement_mask_vectorized(self):
        """Le mask numpy settlement est correct pour 24h."""
        base_ms = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        candle_ts = np.array(
            [base_ms + i * 3600000 for i in range(24)], dtype=np.float64
        )
        hours = ((candle_ts / 3600000) % 24).astype(int)
        mask = hours % 8 == 0

        assert mask[0] is np.True_  # 00:00
        assert mask[1] is np.False_  # 01:00
        assert mask[7] is np.False_  # 07:00
        assert mask[8] is np.True_  # 08:00
        assert mask[15] is np.False_  # 15:00
        assert mask[16] is np.True_  # 16:00
        assert mask.sum() == 3  # Exactement 3 settlements en 24h


# ─── Section 2 : Funding Calculation (6 tests) ──────────────────────────


class TestFundingCalculation:
    def test_funding_long_positive_rate(self, make_indicator_cache):
        """LONG + funding positif → capital diminue (on paie)."""
        cap_with = _run_fast_sim(make_indicator_cache, funding_rate=0.001, direction=1)
        cap_zero = _run_fast_sim(make_indicator_cache, funding_rate=0.0, direction=1)
        assert cap_with < cap_zero

    def test_funding_long_negative_rate(self, make_indicator_cache):
        """LONG + funding négatif → capital augmente (on reçoit)."""
        cap_with = _run_fast_sim(make_indicator_cache, funding_rate=-0.001, direction=1)
        cap_zero = _run_fast_sim(make_indicator_cache, funding_rate=0.0, direction=1)
        assert cap_with > cap_zero

    def test_funding_short_positive_rate(self, make_indicator_cache):
        """SHORT + funding positif → capital augmente (on reçoit)."""
        cap_with = _run_fast_sim(make_indicator_cache, funding_rate=0.001, direction=-1)
        cap_zero = _run_fast_sim(make_indicator_cache, funding_rate=0.0, direction=-1)
        assert cap_with > cap_zero

    def test_funding_short_negative_rate(self, make_indicator_cache):
        """SHORT + funding négatif → capital diminue (on paie)."""
        cap_with = _run_fast_sim(make_indicator_cache, funding_rate=-0.001, direction=-1)
        cap_zero = _run_fast_sim(make_indicator_cache, funding_rate=0.0, direction=-1)
        assert cap_with < cap_zero

    def test_notional_uses_entry_price(self, make_indicator_cache):
        """Le notional est calculé avec entry_price (99.5), pas candle.close (110)."""
        n = 24
        closes = np.full(n, 100.0)
        closes[2:] = 110.0  # Prix monte après ouverture
        highs = closes + 1
        lows = closes - 1
        opens = closes.copy()

        # Non-triggering sauf candle 1
        entry_prices = np.full((n, 1), 50.0)  # 50 < min(lows) → jamais touché
        entry_prices[1, 0] = 99.5  # Ouverture à 99.5
        sma = np.full(n, 200.0)
        funding = np.full(n, 0.001)
        base_ms = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        candle_ts = np.array(
            [base_ms + i * 3600000 for i in range(n)], dtype=np.float64
        )

        cache = make_indicator_cache(
            n=n,
            closes=closes,
            opens=opens,
            highs=highs,
            lows=lows,
            bb_sma={20: sma},
            funding_rates_1h=funding,
            candle_timestamps=candle_ts,
        )

        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2025, 1, 2, tzinfo=timezone.utc),
            initial_capital=10_000.0,
            leverage=10,
        )

        _, _, capital = _simulate_grid_common(
            entry_prices, sma, cache, bt_config, num_levels=1, sl_pct=0.5, direction=1,
        )

        # entry_price=99.5, notional≈100000, funding/settlement≈100
        # 2 settlements → ~200 total funding cost
        # Trade PnL: (110-99.5)*qty - fees → positive (price up 10%)
        # Net = positive (price gain >> funding cost)
        assert capital > bt_config.initial_capital

    def test_funding_accumulated_multi_settlements(self, make_indicator_cache):
        """48h a plus de settlements que 24h → plus de funding payé."""
        cap_24h = _run_fast_sim(make_indicator_cache, funding_rate=0.001, direction=1, n=24)
        cap_48h = _run_fast_sim(make_indicator_cache, funding_rate=0.001, direction=1, n=48)
        # 48h = 5 settlements vs 24h = 2 settlements → plus de coûts
        assert cap_48h < cap_24h


# ─── Section 3 : Convention /100 (4 tests) ──────────────────────────────


class TestDivision100:
    def test_extra_data_builder_divides_by_100(self):
        """extra_data_builder divise le funding rate DB par 100."""
        from backend.backtesting.extra_data_builder import build_extra_data_map

        candles = [_make_candle(datetime(2025, 1, 1, tzinfo=timezone.utc))]
        funding = [
            {
                "symbol": "BTC/USDT",
                "exchange": "binance",
                "timestamp": int(
                    datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
                ),
                "funding_rate": 0.05,  # 0.05% en DB
            }
        ]

        result = build_extra_data_map(candles, funding_rates=funding)
        ts_iso = candles[0].timestamp.isoformat()
        assert result[ts_iso][EXTRA_FUNDING_RATE] == pytest.approx(0.0005)  # 0.05/100

    def test_grid_funding_no_double_division(self):
        """grid_funding ne divise plus par 100 (fait par extra_data_builder)."""
        from backend.core.config import GridFundingConfig
        from backend.strategies.base_grid import GridState
        from backend.strategies.grid_funding import GridFundingStrategy

        cfg = GridFundingConfig(funding_threshold_start=0.0005, num_levels=1)
        strat = GridFundingStrategy(cfg)
        gs = GridState(
            positions=[],
            avg_entry_price=0.0,
            total_quantity=0.0,
            total_notional=0.0,
            unrealized_pnl=0.0,
        )

        from backend.strategies.base import StrategyContext

        ctx = StrategyContext(
            symbol="BTC/USDT",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            candles={},
            indicators={"1h": {"sma": 100.0, "close": 90.0}},
            current_position=None,
            capital=10000.0,
            config=None,  # type: ignore[arg-type]
            extra_data={"funding_rate": -0.001},  # -0.1% en decimal
        )

        levels = strat.compute_grid(ctx, gs)
        # -0.001 <= -0.0005 → trigger level 0
        assert len(levels) == 1

    def test_indicator_cache_load_function_exists(self):
        """_load_funding_rates_aligned divise par 100 (interface vérifiée)."""
        from backend.optimization.indicator_cache import _load_funding_rates_aligned

        assert callable(_load_funding_rates_aligned)

    def test_end_to_end_funding_value(self):
        """DB (0.05%) → extra_data_builder → decimal (0.0005) → payment correct."""
        from backend.backtesting.extra_data_builder import build_extra_data_map

        candles = [_make_candle(datetime(2025, 1, 1, tzinfo=timezone.utc))]
        funding = [
            {
                "symbol": "BTC/USDT",
                "exchange": "binance",
                "timestamp": int(
                    datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
                ),
                "funding_rate": -0.05,  # -0.05% en DB
            }
        ]

        result = build_extra_data_map(candles, funding_rates=funding)
        ts_iso = candles[0].timestamp.isoformat()
        fr = result[ts_iso][EXTRA_FUNDING_RATE]
        assert fr == pytest.approx(-0.0005)

        # LONG : payment = -fr × notional = -(-0.0005) × 10000 = +5.0 (on reçoit)
        payment = -fr * 10000
        assert payment == pytest.approx(5.0)


# ─── Section 4 : Backward Compat (3 tests) ──────────────────────────────


class TestBackwardCompat:
    def test_backtest_result_default_funding_zero(self):
        """BacktestResult sans funding_paid_total → default 0.0."""
        config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 2),
        )
        result = BacktestResult(
            config=config,
            strategy_name="vwap_rsi",
            strategy_params={},
            trades=[],
            equity_curve=[10000.0],
            equity_timestamps=[datetime(2025, 1, 1)],
            final_capital=10000.0,
        )
        assert result.funding_paid_total == 0.0

    def test_backtest_result_with_funding(self):
        """BacktestResult avec funding_paid_total explicite."""
        config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 2),
        )
        result = BacktestResult(
            config=config,
            strategy_name="grid_atr",
            strategy_params={},
            trades=[],
            equity_curve=[10000.0],
            equity_timestamps=[datetime(2025, 1, 1)],
            final_capital=9800.0,
            funding_paid_total=-200.0,
        )
        assert result.funding_paid_total == -200.0

    def test_old_constructors_positional_backward_compat(self):
        """Constructeur positionnel (7 args) reste backward-compat."""
        config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 2),
        )
        result = BacktestResult(
            config,
            "test",
            {},
            [],
            [10000.0],
            [datetime(2025, 1, 1)],
            10000.0,
        )
        assert result.funding_paid_total == 0.0
        assert result.final_capital == 10000.0


# ─── Section 5 : Parity Event-Driven vs Fast (4 tests) ──────────────────


class TestParityEngines:
    def test_no_funding_data_unchanged(self, make_indicator_cache):
        """Sans données funding, le résultat est identique."""
        n = 24
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        opens = np.full(n, 100.0)
        sma = np.full(n, 200.0)
        entry_prices = np.full((n, 1), 50.0)  # Non-triggering
        entry_prices[1, 0] = 99.5

        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2025, 1, 2, tzinfo=timezone.utc),
            initial_capital=10_000.0,
            leverage=10,
        )

        # Sans funding
        cache_no = make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows, bb_sma={20: sma},
        )
        _, _, cap_no = _simulate_grid_common(
            entry_prices, sma, cache_no, bt_config, num_levels=1, sl_pct=0.5, direction=1,
        )

        # Avec funding=0.0
        base_ms = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        candle_ts = np.array(
            [base_ms + i * 3600000 for i in range(n)], dtype=np.float64
        )
        cache_zero = make_indicator_cache(
            n=n,
            closes=closes,
            opens=opens,
            highs=highs,
            lows=lows,
            bb_sma={20: sma},
            funding_rates_1h=np.zeros(n),
            candle_timestamps=candle_ts,
        )
        _, _, cap_zero = _simulate_grid_common(
            entry_prices, sma, cache_zero, bt_config, num_levels=1, sl_pct=0.5, direction=1,
        )

        assert cap_no == pytest.approx(cap_zero, rel=1e-10)

    def test_event_driven_applies_funding(self):
        """MultiPositionEngine applique le funding aux settlements."""
        base = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        candles = _make_candles_24h(base, price=100.0)

        config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=candles[0].timestamp,
            end_date=candles[-1].timestamp,
            initial_capital=10_000.0,
            leverage=10,
        )

        extra_data: dict[str, dict[str, Any]] = {}
        for c in candles:
            extra_data[c.timestamp.isoformat()] = {"funding_rate": 0.001}

        strategy = _TestGridStrategy(open_at_first=True)
        engine = MultiPositionEngine(config, strategy, extra_data_by_timestamp=extra_data)
        result = engine.run({"1h": candles}, main_tf="1h")

        # LONG + funding positif → on paie → funding_paid_total < 0
        assert result.funding_paid_total < 0

    def test_event_driven_no_funding_zero(self):
        """MultiPositionEngine sans extra_data → funding_paid_total = 0."""
        base = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        candles = _make_candles_24h(base, price=100.0)

        config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=candles[0].timestamp,
            end_date=candles[-1].timestamp,
            initial_capital=10_000.0,
            leverage=10,
        )

        strategy = _TestGridStrategy(open_at_first=True)
        engine = MultiPositionEngine(config, strategy)
        result = engine.run({"1h": candles}, main_tf="1h")

        assert result.funding_paid_total == 0.0

    def test_fast_engine_direction_sign(self, make_indicator_cache):
        """LONG + positive funding = perte, SHORT + positive funding = gain."""
        cap_long = _run_fast_sim(make_indicator_cache, funding_rate=0.001, direction=1)
        cap_long_ref = _run_fast_sim(make_indicator_cache, funding_rate=0.0, direction=1)
        cap_short = _run_fast_sim(make_indicator_cache, funding_rate=0.001, direction=-1)
        cap_short_ref = _run_fast_sim(make_indicator_cache, funding_rate=0.0, direction=-1)

        long_effect = cap_long - cap_long_ref
        short_effect = cap_short - cap_short_ref

        assert long_effect < 0, "LONG should pay with positive funding"
        assert short_effect > 0, "SHORT should receive with positive funding"


# ─── Section 6 : Portfolio Aggregation (2 tests) ────────────────────────


class TestPortfolioAggregation:
    def test_portfolio_result_has_funding_field(self):
        """PortfolioResult a un champ funding_paid_total."""
        from backend.backtesting.portfolio_engine import PortfolioResult

        fields = {f.name: f for f in dataclasses.fields(PortfolioResult)}
        assert "funding_paid_total" in fields
        assert fields["funding_paid_total"].default == 0.0

    def test_portfolio_result_default_zero(self):
        """PortfolioResult.funding_paid_total est 0.0 par défaut."""
        from backend.backtesting.portfolio_engine import PortfolioResult

        result = PortfolioResult(
            initial_capital=10000.0,
            n_assets=1,
            period_days=30,
            assets=["BTC/USDT"],
            final_equity=10500.0,
            total_return_pct=5.0,
            total_trades=10,
            win_rate=60.0,
            realized_pnl=500.0,
            force_closed_pnl=0.0,
            max_drawdown_pct=5.0,
            max_drawdown_date=None,
            max_drawdown_duration_hours=0.0,
            peak_margin_ratio=0.3,
            peak_open_positions=3,
            peak_concurrent_assets=1,
            kill_switch_triggers=0,
            kill_switch_events=[],
            snapshots=[],
            per_asset_results={},
        )
        assert result.funding_paid_total == 0.0


# ─── Section 7 : Edge Cases (1 test) ────────────────────────────────────


class TestEdgeCases:
    def test_funding_nan_skipped(self, make_indicator_cache):
        """NaN funding rate → pas de payment, identique à sans funding."""
        n = 24
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        opens = np.full(n, 100.0)
        sma = np.full(n, 200.0)
        entry_prices = np.full((n, 1), 50.0)  # Non-triggering
        entry_prices[1, 0] = 99.5

        base_ms = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        candle_ts = np.array(
            [base_ms + i * 3600000 for i in range(n)], dtype=np.float64
        )

        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2025, 1, 2, tzinfo=timezone.utc),
            initial_capital=10_000.0,
            leverage=10,
        )

        # Funding NaN
        cache_nan = make_indicator_cache(
            n=n,
            closes=closes,
            opens=opens,
            highs=highs,
            lows=lows,
            bb_sma={20: sma},
            funding_rates_1h=np.full(n, np.nan),
            candle_timestamps=candle_ts,
        )
        _, _, cap_nan = _simulate_grid_common(
            entry_prices, sma, cache_nan, bt_config, num_levels=1, sl_pct=0.5, direction=1,
        )

        # Sans funding du tout
        cache_no = make_indicator_cache(
            n=n, closes=closes, opens=opens, highs=highs, lows=lows, bb_sma={20: sma},
        )
        _, _, cap_no = _simulate_grid_common(
            entry_prices, sma, cache_no, bt_config, num_levels=1, sl_pct=0.5, direction=1,
        )

        assert cap_nan == pytest.approx(cap_no, rel=1e-10)


# ─── Section 7 : MultiPositionEngine reçoit extra_data (1 test) ─────


class TestMultiEngineExtraData:
    """Vérifie que MultiPositionEngine utilise extra_data pour calculer le funding."""

    def test_multi_engine_funding_with_extra_data(self):
        """MultiPositionEngine avec extra_data produit funding_paid_total != 0."""
        from backend.backtesting.extra_data_builder import build_extra_data_map
        from backend.core.config import EnvelopeDCAConfig
        from backend.strategies.envelope_dca import EnvelopeDCAStrategy

        # 100 candles 1h avec prix sinusoïdal (amplitude large) → positions ouvertes/fermées
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        n = 100
        candles = []
        for i in range(n):
            price = 100 + 20 * np.sin(2 * np.pi * i / 30)
            candles.append(Candle(
                symbol="BTC/USDT", exchange="binance", timeframe="1h",
                timestamp=base + timedelta(hours=i),
                open=price, high=price + 2, low=price - 2,
                close=price, volume=1000.0,
            ))

        # Funding rates toutes les 8h, taux positif → LONG paie
        funding_rates = []
        for i in range(0, n, 8):
            ts = base + timedelta(hours=i)
            funding_rates.append({
                "timestamp": int(ts.timestamp() * 1000),
                "funding_rate": 0.05,  # 0.05% en DB → /100 par builder
            })

        extra_map = build_extra_data_map(candles, funding_rates=funding_rates)

        config = EnvelopeDCAConfig(
            ma_period=5, num_levels=3, envelope_start=0.03,
            envelope_step=0.02, sl_percent=25.0, sides=["long"],
            leverage=6,
        )
        strategy = EnvelopeDCAStrategy(config)

        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=base,
            end_date=base + timedelta(hours=n),
            initial_capital=10_000.0,
            leverage=6,
        )

        # Avec extra_data → funding non-zero
        engine_with = MultiPositionEngine(
            bt_config, strategy, extra_data_by_timestamp=extra_map,
        )
        result_with = engine_with.run({"1h": candles}, main_tf="1h")

        # Sans extra_data → funding = 0
        engine_without = MultiPositionEngine(bt_config, strategy)
        result_without = engine_without.run({"1h": candles}, main_tf="1h")

        # Vérifications
        assert result_without.funding_paid_total == 0.0
        assert result_with.funding_paid_total != 0.0
        assert result_with.funding_paid_total < 0  # LONG paie sur taux positif
