"""Tests Sprint Backtest Réalisme — Liquidation, Funding, Leverage Validation.

Fix 1 : Simulation liquidation cross margin (portfolio_engine)
Fix 2 : Funding costs dans GridStrategyRunner (simulator)
Fix 3 : Validation leverage × SL (report)
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from backend.backtesting.portfolio_engine import (
    PortfolioBacktester,
    PortfolioResult,
    PortfolioSnapshot,
    format_portfolio_report,
)
from backend.backtesting.simulator import GridStrategyRunner, RunnerStats
from backend.core.grid_position_manager import GridPositionManager
from backend.core.incremental_indicators import IncrementalIndicatorEngine
from backend.core.models import Candle, Direction, TimeFrame
from backend.core.position_manager import PositionManagerConfig
from backend.strategies.base_grid import BaseGridStrategy, GridPosition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gpm_config(leverage: int = 6) -> PositionManagerConfig:
    return PositionManagerConfig(
        leverage=leverage,
        maker_fee=0.0002,
        taker_fee=0.0006,
        slippage_pct=0.0005,
        high_vol_slippage_mult=2.0,
        max_risk_per_trade=0.02,
    )


def _make_mock_strategy(
    name: str = "grid_atr",
    timeframe: str = "1h",
    ma_period: int = 7,
    max_positions: int = 3,
    leverage: int = 6,
    sl_percent: float = 20.0,
) -> MagicMock:
    strategy = MagicMock(spec=BaseGridStrategy)
    strategy.name = name
    config = MagicMock()
    config.timeframe = timeframe
    config.ma_period = ma_period
    config.leverage = leverage
    config.sl_percent = sl_percent
    config.per_asset = {}
    strategy._config = config
    strategy.min_candles = {"1h": 50}
    strategy.max_positions = max_positions
    strategy.compute_grid.return_value = []
    strategy.should_close_all.return_value = None
    strategy.get_tp_price.return_value = float("nan")
    strategy.get_sl_price.return_value = float("nan")
    strategy.get_current_conditions.return_value = []
    strategy.compute_live_indicators.return_value = {}
    return strategy


def _make_mock_config(initial_capital: float = 10_000.0) -> MagicMock:
    config = MagicMock()
    config.risk.initial_capital = initial_capital
    config.risk.max_margin_ratio = 0.70
    config.risk.kill_switch.max_session_loss_percent = 5.0
    config.risk.kill_switch.grid_max_session_loss_percent = 25.0
    config.risk.kill_switch.max_daily_loss_percent = 10.0
    config.risk.kill_switch.grid_max_daily_loss_percent = 25.0
    config.risk.position.max_risk_per_trade_percent = 2.0
    config.risk.fees.maker_percent = 0.02
    config.risk.fees.taker_percent = 0.06
    config.risk.slippage.default_estimate_percent = 0.05
    config.risk.slippage.high_volatility_multiplier = 2.0
    config.risk.regime_filter_enabled = False
    config.assets = []
    return config


def _make_grid_runner(
    strategy=None,
    config=None,
    leverage: int = 6,
    initial_capital: float = 10_000.0,
) -> GridStrategyRunner:
    if strategy is None:
        strategy = _make_mock_strategy(leverage=leverage)
    if config is None:
        config = _make_mock_config(initial_capital=initial_capital)
    indicator_engine = MagicMock(spec=IncrementalIndicatorEngine)
    indicator_engine.get_indicators.return_value = {}
    indicator_engine.update = MagicMock()
    gpm_config = _make_gpm_config(leverage=leverage)
    gpm = GridPositionManager(gpm_config)
    data_engine = MagicMock()
    runner = GridStrategyRunner(
        strategy=strategy,
        config=config,
        indicator_engine=indicator_engine,
        grid_position_manager=gpm,
        data_engine=data_engine,
    )
    runner._is_warming_up = False
    runner._warmup_ended_at = datetime.now(tz=timezone.utc) - timedelta(hours=4)
    return runner


def _make_candle(
    symbol: str = "BTC/USDT",
    close: float = 50_000.0,
    hour: int = 12,
    **kwargs,
) -> Candle:
    ts = datetime(2024, 6, 1, hour, 0, 0, tzinfo=timezone.utc)
    return Candle(
        timestamp=ts,
        open=kwargs.get("open", close),
        high=kwargs.get("high", close * 1.01),
        low=kwargs.get("low", close * 0.99),
        close=close,
        volume=kwargs.get("volume", 100.0),
        symbol=symbol,
        timeframe=TimeFrame.H1,
    )


def _make_position(
    direction: Direction = Direction.LONG,
    entry_price: float = 50_000.0,
    quantity: float = 0.01,
    level: int = 0,
) -> GridPosition:
    return GridPosition(
        level=level,
        direction=direction,
        entry_price=entry_price,
        quantity=quantity,
        entry_time=datetime(2024, 5, 30, 12, 0, 0),
        entry_fee=entry_price * quantity * 0.0006,
    )


# ===========================================================================
# Fix 1 — Liquidation cross margin
# ===========================================================================


class TestLiquidationSimulation:
    """Tests pour la simulation de liquidation cross margin."""

    def test_snapshot_liquidation_distance(self):
        """Snapshot avec positions → liquidation_distance_pct > 0."""
        snap = PortfolioSnapshot(
            timestamp=datetime(2024, 6, 1),
            total_equity=10_000.0,
            total_capital=9_000.0,
            total_realized_pnl=0.0,
            total_unrealized_pnl=1_000.0,
            total_margin_used=2_000.0,
            margin_ratio=0.20,
            n_open_positions=3,
            n_assets_with_positions=2,
            total_notional=12_000.0,
            maintenance_margin=12_000.0 * 0.004,  # 48$
            liquidation_distance_pct=(10_000.0 - 48.0) / 10_000.0 * 100,
            is_liquidated=False,
        )
        assert snap.liquidation_distance_pct == pytest.approx(99.52, abs=0.01)
        assert not snap.is_liquidated

    def test_snapshot_liquidated_when_equity_below_maintenance(self):
        """Equity < maintenance → is_liquidated=True."""
        notional = 500_000.0  # Très gros notional
        maintenance = notional * 0.004  # 2000$
        equity = 1_500.0  # < maintenance

        snap = PortfolioSnapshot(
            timestamp=datetime(2024, 6, 1),
            total_equity=equity,
            total_capital=500.0,
            total_realized_pnl=-8_500.0,
            total_unrealized_pnl=1_000.0,
            total_margin_used=5_000.0,
            margin_ratio=0.50,
            n_open_positions=5,
            n_assets_with_positions=3,
            total_notional=notional,
            maintenance_margin=maintenance,
            liquidation_distance_pct=(equity - maintenance) / equity * 100,
            is_liquidated=equity <= maintenance,
        )
        assert snap.is_liquidated
        assert snap.liquidation_distance_pct < 0

    def test_simulation_stops_on_liquidation(self):
        """La simulation _take_snapshot détecte une liquidation."""
        # Créer un backtester minimal avec un runner en situation de liquidation
        config = _make_mock_config(initial_capital=10_000.0)
        bt = PortfolioBacktester(
            config=config,
            initial_capital=10_000.0,
            strategy_name="grid_atr",
            assets=["BTC/USDT"],
        )

        # Runner avec capital très bas et gros notional
        runner = _make_grid_runner(initial_capital=10_000.0)
        runner._capital = 100.0  # Presque plus de capital
        runner._realized_pnl = -9_900.0
        # Grosse position → notional élevé → maintenance > equity
        runner._positions = {
            "BTC/USDT": [
                _make_position(entry_price=50_000.0, quantity=0.1),  # notional = 5000
                _make_position(entry_price=48_000.0, quantity=0.1, level=1),  # 4800
                _make_position(entry_price=46_000.0, quantity=0.1, level=2),  # 4600
            ]
        }
        # Total notional = 14400, maintenance = 57.6$
        # Unrealized PNL très négatif (prix actuel 30000 vs entries ~48000)
        # equity = capital + unrealized

        runners = {"grid_atr:BTC/USDT": runner}
        last_closes = {"BTC/USDT": 30_000.0}

        snap = bt._take_snapshot(runners, datetime(2024, 6, 1), last_closes)
        # equity = 100 + unrealized(très négatif) → devrait être liquidé
        # unrealized = sum((30000 - entry) * qty) pour LONG
        expected_upnl = (
            (30_000 - 50_000) * 0.1
            + (30_000 - 48_000) * 0.1
            + (30_000 - 46_000) * 0.1
        )
        assert expected_upnl < -3000  # sanity check
        assert snap.total_equity < snap.maintenance_margin
        assert snap.is_liquidated

    def test_no_liquidation_normal_conditions(self):
        """Conditions normales → is_liquidated=False."""
        snap = PortfolioSnapshot(
            timestamp=datetime(2024, 6, 1),
            total_equity=10_000.0,
            total_capital=9_000.0,
            total_realized_pnl=0.0,
            total_unrealized_pnl=1_000.0,
            total_margin_used=2_000.0,
            margin_ratio=0.20,
            n_open_positions=3,
            n_assets_with_positions=2,
            total_notional=12_000.0,
            maintenance_margin=48.0,
            liquidation_distance_pct=99.52,
            is_liquidated=False,
        )
        assert not snap.is_liquidated
        assert snap.liquidation_distance_pct > 90

    def test_maintenance_margin_calculation(self):
        """notional 100k × 0.004 = 400$ maintenance."""
        notional = 100_000.0
        rate = 0.004
        assert notional * rate == 400.0

    def test_worst_case_sl_calculation(self):
        """3 positions × SL 20% × notional → perte correcte."""
        config = _make_mock_config(initial_capital=10_000.0)
        bt = PortfolioBacktester(
            config=config,
            initial_capital=10_000.0,
            strategy_name="grid_atr",
            assets=["BTC/USDT"],
        )

        # Runner avec SL 20%
        strategy = _make_mock_strategy(sl_percent=20.0, leverage=6)
        runner = _make_grid_runner(strategy=strategy, initial_capital=10_000.0)
        runner._capital = 9_000.0
        runner._positions = {
            "BTC/USDT": [
                _make_position(entry_price=50_000.0, quantity=0.01, level=0),
                _make_position(entry_price=48_000.0, quantity=0.01, level=1),
                _make_position(entry_price=46_000.0, quantity=0.01, level=2),
            ]
        }

        runners = {"grid_atr:BTC/USDT": runner}
        last_closes = {"BTC/USDT": 49_000.0}

        snap = bt._take_snapshot(runners, datetime(2024, 6, 1), last_closes)

        # worst_case = sum(notional × sl_pct) / initial_capital × 100
        # notional = 500 + 480 + 460 = 1440
        # worst_case = 1440 × 0.20 / 10000 × 100 = 2.88%
        expected = (500 + 480 + 460) * 0.20 / 10_000 * 100
        assert snap.worst_case_sl_loss_pct == pytest.approx(expected, abs=0.1)


# ===========================================================================
# Fix 2 — Funding costs
# ===========================================================================


class TestFundingCosts:
    """Tests pour le funding dans GridStrategyRunner.on_candle()."""

    @pytest.mark.asyncio
    async def test_funding_at_8h_intervals(self):
        """Capital diminue à h=0,8,16 UTC."""
        runner = _make_grid_runner()
        runner._capital = 10_000.0
        symbol = "BTC/USDT"

        # Ajouter des positions et le close buffer
        runner._positions[symbol] = [
            _make_position(entry_price=50_000.0, quantity=0.01),  # notional = 500
        ]
        runner._close_buffer[symbol] = deque(
            [50_000.0] * 20, maxlen=30
        )

        capital_before = runner._capital

        # Candle à h=8 → funding appliqué
        candle = _make_candle(symbol=symbol, close=50_000.0, hour=8)
        await runner.on_candle(symbol, "1h", candle)

        # Funding = 500 × 0.0001 = 0.05$ (LONG paie)
        assert runner._capital < capital_before
        expected_cost = 500.0 * 0.0001
        assert runner._total_funding_cost == pytest.approx(expected_cost, abs=0.001)

    @pytest.mark.asyncio
    async def test_funding_not_during_warmup(self):
        """Pas de funding pendant le warm-up."""
        runner = _make_grid_runner()
        runner._is_warming_up = True
        symbol = "BTC/USDT"

        runner._positions[symbol] = [
            _make_position(entry_price=50_000.0, quantity=0.01),
        ]
        runner._close_buffer[symbol] = deque(
            [50_000.0] * 20, maxlen=30
        )

        capital_before = runner._capital
        candle = _make_candle(symbol=symbol, close=50_000.0, hour=8)
        await runner.on_candle(symbol, "1h", candle)

        # Pas de funding pendant warmup
        assert runner._total_funding_cost == 0.0

    @pytest.mark.asyncio
    async def test_funding_long_pays_short_receives(self):
        """LONG → capital ↓, SHORT → capital ↑."""
        # Test LONG
        runner_long = _make_grid_runner()
        runner_long._capital = 10_000.0
        symbol = "BTC/USDT"
        runner_long._positions[symbol] = [
            _make_position(direction=Direction.LONG, entry_price=50_000.0, quantity=0.01),
        ]
        runner_long._close_buffer[symbol] = deque([50_000.0] * 20, maxlen=30)

        candle = _make_candle(symbol=symbol, close=50_000.0, hour=0)
        await runner_long.on_candle(symbol, "1h", candle)

        assert runner_long._total_funding_cost > 0  # LONG paie

        # Test SHORT
        runner_short = _make_grid_runner()
        runner_short._capital = 10_000.0
        runner_short._positions[symbol] = [
            _make_position(direction=Direction.SHORT, entry_price=50_000.0, quantity=0.01),
        ]
        runner_short._close_buffer[symbol] = deque([50_000.0] * 20, maxlen=30)

        candle = _make_candle(symbol=symbol, close=50_000.0, hour=0)
        await runner_short.on_candle(symbol, "1h", candle)

        assert runner_short._total_funding_cost < 0  # SHORT reçoit

    @pytest.mark.asyncio
    async def test_funding_total_tracked(self):
        """_total_funding_cost correct après N heures de settlement."""
        runner = _make_grid_runner()
        runner._capital = 10_000.0
        symbol = "BTC/USDT"
        runner._positions[symbol] = [
            _make_position(entry_price=50_000.0, quantity=0.01),  # notional = 500
        ]
        runner._close_buffer[symbol] = deque([50_000.0] * 20, maxlen=30)

        # 3 settlements : h=0, h=8, h=16
        for hour in (0, 8, 16):
            candle = _make_candle(symbol=symbol, close=50_000.0, hour=hour)
            await runner.on_candle(symbol, "1h", candle)

        expected = 500.0 * 0.0001 * 3  # 3 settlements
        assert runner._total_funding_cost == pytest.approx(expected, abs=0.001)

    @pytest.mark.asyncio
    async def test_no_funding_without_positions(self):
        """Pas de positions → pas de funding."""
        runner = _make_grid_runner()
        runner._capital = 10_000.0
        symbol = "BTC/USDT"
        # Pas de positions !
        runner._close_buffer[symbol] = deque([50_000.0] * 20, maxlen=30)

        candle = _make_candle(symbol=symbol, close=50_000.0, hour=8)
        await runner.on_candle(symbol, "1h", candle)

        assert runner._total_funding_cost == 0.0


# ===========================================================================
# Fix 3 — Leverage validation
# ===========================================================================


class TestLeverageValidation:
    """Tests pour _validate_leverage_sl et intégration build_final_report."""

    def test_leverage_sl_warning_above_100pct(self):
        """SL 20% × 6x = 120% → warning 'dépasse 100%'."""
        from backend.optimization.report import _validate_leverage_sl

        warnings = _validate_leverage_sl("grid_atr", {"sl_percent": 20.0})
        assert len(warnings) == 1
        assert "depasse 100%" in warnings[0]

    def test_leverage_sl_warning_above_80pct(self):
        """SL × leverage > 80% → warning 'risque'."""
        from backend.optimization.report import _validate_leverage_sl

        # 14% × 6x = 84% > 80% mais < 100% → warning risque
        warnings = _validate_leverage_sl("grid_atr", {"sl_percent": 14.0})
        assert len(warnings) == 1
        assert "risque" in warnings[0].lower() or "84%" in warnings[0]

    def test_leverage_sl_ok(self):
        """SL × leverage < 80% → pas de warning."""
        from backend.optimization.report import _validate_leverage_sl

        # 10% × 6x = 60% < 80% → OK
        warnings = _validate_leverage_sl("grid_atr", {"sl_percent": 10.0})
        assert len(warnings) == 0

    def test_leverage_sl_in_final_report(self):
        """Warning apparaît dans FinalReport.warnings."""
        from backend.optimization.overfitting import (
            DSRResult,
            MonteCarloResult,
            OverfitReport,
            StabilityResult,
        )
        from backend.optimization.report import (
            ValidationResult,
            build_final_report,
        )
        from backend.optimization.walk_forward import WFOResult

        wfo = WFOResult(
            strategy_name="grid_atr",
            symbol="BTC/USDT",
            windows=[],
            avg_is_sharpe=1.5,
            avg_oos_sharpe=0.8,
            oos_is_ratio=0.53,
            consistency_rate=0.7,
            recommended_params={"sl_percent": 20.0, "ma_period": 14},
            all_oos_trades=[],
            n_distinct_combos=500,
        )
        overfit = OverfitReport(
            monte_carlo=MonteCarloResult(
                p_value=0.02,
                real_sharpe=0.8,
                distribution=[],
                significant=True,
            ),
            dsr=DSRResult(
                dsr=0.95,
                max_expected_sharpe=3.0,
                observed_sharpe=0.8,
                n_trials=500,
            ),
            stability=StabilityResult(
                stability_map={},
                overall_stability=0.85,
                cliff_params=[],
            ),
            convergence=None,
        )
        validation = ValidationResult(
            bitget_sharpe=0.7,
            bitget_net_return_pct=4.0,
            bitget_trades=15,
            bitget_sharpe_ci_low=0.2,
            bitget_sharpe_ci_high=1.1,
            binance_oos_avg_sharpe=0.8,
            transfer_ratio=0.875,
            transfer_significant=True,
            volume_warning=False,
            volume_warning_detail="",
        )

        report = build_final_report(wfo, overfit, validation)
        leverage_warns = [w for w in report.warnings if "leverage" in w.lower() or "marge" in w.lower()]
        assert len(leverage_warns) >= 1
        assert "depasse 100%" in leverage_warns[0]


# ===========================================================================
# Fix 4 — Report format
# ===========================================================================


class TestReportFormat:
    """Tests pour format_portfolio_report avec les nouvelles métriques."""

    def test_format_report_includes_cross_margin(self):
        """Le rapport contient la section Cross-Margin Risk."""
        result = PortfolioResult(
            initial_capital=10_000.0,
            n_assets=3,
            period_days=90,
            assets=["BTC/USDT", "ETH/USDT", "DOGE/USDT"],
            final_equity=11_000.0,
            total_return_pct=10.0,
            total_trades=50,
            win_rate=60.0,
            realized_pnl=1_200.0,
            force_closed_pnl=-200.0,
            max_drawdown_pct=-15.0,
            max_drawdown_date=datetime(2024, 6, 15),
            max_drawdown_duration_hours=48.0,
            peak_margin_ratio=0.45,
            peak_open_positions=9,
            peak_concurrent_assets=3,
            kill_switch_triggers=0,
            kill_switch_events=[],
            snapshots=[],
            per_asset_results={},
            funding_paid_total=-45.50,
            was_liquidated=False,
            liquidation_event=None,
            min_liquidation_distance_pct=95.2,
            worst_case_sl_loss_pct=28.8,
        )

        report = format_portfolio_report(result)
        assert "Cross-Margin Risk" in report
        assert "95.2%" in report
        assert "28.8%" in report
        assert "-45.50" in report
        assert "LIQUIDE" not in report

    def test_format_report_liquidation(self):
        """Le rapport affiche la liquidation si elle a eu lieu."""
        result = PortfolioResult(
            initial_capital=10_000.0,
            n_assets=3,
            period_days=90,
            assets=["BTC/USDT"],
            final_equity=0.0,
            total_return_pct=-100.0,
            total_trades=10,
            win_rate=30.0,
            realized_pnl=-5_000.0,
            force_closed_pnl=0.0,
            max_drawdown_pct=-100.0,
            max_drawdown_date=datetime(2024, 6, 15),
            max_drawdown_duration_hours=100.0,
            peak_margin_ratio=0.80,
            peak_open_positions=9,
            peak_concurrent_assets=3,
            kill_switch_triggers=0,
            kill_switch_events=[],
            snapshots=[],
            per_asset_results={},
            funding_paid_total=-20.0,
            was_liquidated=True,
            liquidation_event={
                "timestamp": "2024-06-15T08:00:00",
                "equity": 150.0,
                "maintenance_margin": 200.0,
                "notional": 50_000.0,
                "n_positions": 5,
            },
            min_liquidation_distance_pct=-33.3,
            worst_case_sl_loss_pct=120.0,
        )

        report = format_portfolio_report(result)
        assert "LIQUIDE" in report
        assert "150.00" in report
        assert "200.00" in report
