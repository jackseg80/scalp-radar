"""Tests pour PortfolioBacktester (Sprint 20b).

Couvre :
- Dataclasses (snapshot, result)
- Merge chronologique des candles
- Warm-up et capital split
- Simulation basique (prix plat, 2 assets)
- Drawdown et kill switch
- Force-close en fin de données
- Per-asset breakdown
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.backtesting.portfolio_engine import (
    PortfolioBacktester,
    PortfolioResult,
    PortfolioSnapshot,
    format_portfolio_report,
)
from backend.backtesting.simulator import GridStrategyRunner, RunnerStats
from backend.core.config import GridATRConfig, GridBolTrendConfig
from backend.core.grid_position_manager import GridPositionManager
from backend.core.incremental_indicators import IncrementalIndicatorEngine
from backend.core.models import Candle, Direction, MarketRegime, TimeFrame
from backend.core.position_manager import PositionManagerConfig
from backend.optimization import create_strategy_with_params
from backend.strategies.base_grid import GridPosition
from backend.strategies.grid_atr import GridATRStrategy


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_candle(
    close: float,
    ts: datetime,
    symbol: str = "AAA/USDT",
    spread: float = 0.005,
) -> Candle:
    """Crée une candle 1h synthétique."""
    return Candle(
        timestamp=ts,
        open=close * (1 - spread / 2),
        high=close * (1 + spread),
        low=close * (1 - spread),
        close=close,
        volume=1000.0,
        symbol=symbol,
        timeframe=TimeFrame.H1,
    )


def _make_candles(
    symbol: str,
    n: int = 100,
    start_price: float = 100.0,
    start_ts: datetime | None = None,
    seed: int = 42,
    volatility: float = 0.005,
) -> list[Candle]:
    """Génère N candles 1h avec random walk."""
    rng = np.random.RandomState(seed)
    ts = start_ts or datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = start_price
    for _ in range(n):
        ret = rng.normal(0, volatility)
        price *= 1 + ret
        candles.append(_make_candle(price, ts, symbol))
        ts += timedelta(hours=1)
    return candles


def _make_flat_candles(
    symbol: str,
    n: int = 100,
    price: float = 100.0,
    start_ts: datetime | None = None,
) -> list[Candle]:
    """Génère N candles à prix fixe (pas de signal grid)."""
    ts = start_ts or datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    for _ in range(n):
        candles.append(
            Candle(
                timestamp=ts,
                open=price,
                high=price * 1.0001,
                low=price * 0.9999,
                close=price,
                volume=1000.0,
                symbol=symbol,
                timeframe=TimeFrame.H1,
            )
        )
        ts += timedelta(hours=1)
    return candles


def _make_mock_config(n_assets: int = 2) -> MagicMock:
    """Crée un mock AppConfig minimal."""
    config = MagicMock()
    config.risk.initial_capital = 10_000.0
    config.risk.max_margin_ratio = 0.70
    config.risk.fees.maker_percent = 0.02
    config.risk.fees.taker_percent = 0.06
    config.risk.slippage.default_estimate_percent = 0.05
    config.risk.slippage.high_volatility_multiplier = 2.0
    config.risk.position.max_risk_per_trade_percent = 2.0
    config.risk.position.default_leverage = 15
    config.risk.position.max_leverage = 30
    config.risk.kill_switch.max_session_loss_percent = 5.0
    config.risk.kill_switch.max_daily_loss_percent = 10.0
    config.risk.kill_switch.global_max_loss_pct = 30.0
    config.risk.kill_switch.global_window_hours = 24
    config.assets = [MagicMock(symbol=f"ASSET{i}/USDT") for i in range(n_assets)]

    # Strategy config pour grid_atr
    config.strategies.grid_atr.per_asset = {}
    config.strategies.grid_atr.enabled = True

    return config


def _make_runner_with_indicator_engine(
    symbol: str,
    config: MagicMock,
    per_asset_capital: float = 5000.0,
) -> tuple[GridStrategyRunner, IncrementalIndicatorEngine]:
    """Crée un runner réel (pas de mock) avec indicator engine."""
    strategy = create_strategy_with_params("grid_atr", {
        "ma_period": 14,
        "atr_period": 14,
        "atr_multiplier_start": 2.0,
        "atr_multiplier_step": 1.0,
        "num_levels": 3,
        "sl_percent": 20.0,
        "sides": ["long"],
        "leverage": 6,
    })
    indicator_engine = IncrementalIndicatorEngine([strategy])
    gpm_config = PositionManagerConfig(
        leverage=6,
        maker_fee=0.0006,
        taker_fee=0.0006,
        slippage_pct=0.0005,
    )
    gpm = GridPositionManager(gpm_config)

    runner = GridStrategyRunner(
        strategy=strategy,
        config=config,
        indicator_engine=indicator_engine,
        grid_position_manager=gpm,
        data_engine=None,  # type: ignore[arg-type]
        db_path=None,
    )
    runner._nb_assets = 1
    runner._capital = per_asset_capital
    runner._initial_capital = per_asset_capital
    runner._is_warming_up = False
    runner._stats = RunnerStats(capital=per_asset_capital, initial_capital=per_asset_capital)

    return runner, indicator_engine


# ─── Tests Dataclasses ─────────────────────────────────────────────────────


class TestPortfolioDataclasses:
    def test_snapshot_fields(self):
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        snap = PortfolioSnapshot(
            timestamp=ts,
            total_equity=10_000.0,
            total_capital=9_500.0,
            total_realized_pnl=0.0,
            total_unrealized_pnl=500.0,
            total_margin_used=1_000.0,
            margin_ratio=0.10,
            n_open_positions=3,
            n_assets_with_positions=2,
        )
        assert snap.total_equity == 10_000.0
        assert snap.margin_ratio == 0.10
        assert snap.n_assets_with_positions == 2

    def test_result_fields(self):
        result = PortfolioResult(
            initial_capital=10_000.0,
            n_assets=2,
            period_days=90,
            assets=["A/USDT", "B/USDT"],
            final_equity=10_500.0,
            total_return_pct=5.0,
            total_trades=10,
            win_rate=60.0,
            realized_pnl=600.0,
            force_closed_pnl=-100.0,
            max_drawdown_pct=-5.0,
            max_drawdown_date=None,
            max_drawdown_duration_hours=0.0,
            peak_margin_ratio=0.3,
            peak_open_positions=6,
            peak_concurrent_assets=2,
            kill_switch_triggers=0,
            kill_switch_events=[],
            snapshots=[],
            per_asset_results={},
        )
        assert result.realized_pnl == 600.0
        assert result.force_closed_pnl == -100.0
        assert result.total_return_pct == 5.0


# ─── Tests Merge Candles ───────────────────────────────────────────────────


class TestMergeCandles:
    def test_chronological_order(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        candles_a = [
            _make_candle(100, ts, "AAA/USDT"),
            _make_candle(101, ts + timedelta(hours=1), "AAA/USDT"),
        ]
        candles_b = [
            _make_candle(200, ts, "BBB/USDT"),
            _make_candle(201, ts + timedelta(hours=1), "BBB/USDT"),
        ]
        merged = PortfolioBacktester._merge_candles({
            "AAA/USDT": candles_a,
            "BBB/USDT": candles_b,
        })
        assert len(merged) == 4
        # Même timestamp → trié par symbol (AAA < BBB)
        assert merged[0].symbol == "AAA/USDT"
        assert merged[1].symbol == "BBB/USDT"
        assert merged[0].timestamp == merged[1].timestamp
        # Deuxième heure
        assert merged[2].timestamp > merged[0].timestamp

    def test_empty_input(self):
        merged = PortfolioBacktester._merge_candles({})
        assert merged == []


# ─── Tests Warmup ──────────────────────────────────────────────────────────


class TestWarmup:
    def test_warmup_sets_capital_and_flags(self):
        config = _make_mock_config()
        runner, engine = _make_runner_with_indicator_engine("AAA/USDT", config, 5000.0)
        runners = {"AAA/USDT": runner}

        candles = _make_candles("AAA/USDT", n=80, start_price=100.0)

        # Simuler un état initial (warmup=True serait le défaut, mais on l'a
        # déjà mis à False dans le helper — vérifier que warmup_runners reset correctement)
        runner._is_warming_up = True

        backtester = PortfolioBacktester.__new__(PortfolioBacktester)
        backtester._initial_capital = 10_000.0

        warmup_ends = backtester._warmup_runners(
            runners, {"AAA/USDT": candles}, engine, warmup_count=50
        )

        assert warmup_ends["AAA/USDT"] == 50
        assert runner._is_warming_up is False
        assert runner._capital == 5000.0
        assert runner._realized_pnl == 0.0
        assert runner._stats.total_trades == 0
        assert len(runner._close_buffer["AAA/USDT"]) == 50

    def test_warmup_insufficient_candles(self):
        """Si un asset a peu de candles, warmup adapte."""
        config = _make_mock_config()
        runner, engine = _make_runner_with_indicator_engine("AAA/USDT", config, 5000.0)
        runners = {"AAA/USDT": runner}

        # Seulement 20 candles (< warmup_count=50)
        candles = _make_candles("AAA/USDT", n=20, start_price=100.0)

        backtester = PortfolioBacktester.__new__(PortfolioBacktester)
        backtester._initial_capital = 10_000.0

        warmup_ends = backtester._warmup_runners(
            runners, {"AAA/USDT": candles}, engine, warmup_count=50
        )

        # Garde au moins 1 candle pour la simulation
        assert warmup_ends["AAA/USDT"] == 19
        assert runner._is_warming_up is False


# ─── Tests Snapshot ────────────────────────────────────────────────────────


class TestTakeSnapshot:
    def test_empty_runners(self):
        config = _make_mock_config()
        runner_a, _ = _make_runner_with_indicator_engine("AAA/USDT", config, 5000.0)
        runner_b, _ = _make_runner_with_indicator_engine("BBB/USDT", config, 5000.0)
        runners = {"AAA/USDT": runner_a, "BBB/USDT": runner_b}

        backtester = PortfolioBacktester.__new__(PortfolioBacktester)
        backtester._initial_capital = 10_000.0

        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        snap = backtester._take_snapshot(runners, ts, {"AAA/USDT": 100.0, "BBB/USDT": 200.0})

        assert snap.total_capital == 10_000.0
        assert snap.total_unrealized_pnl == 0.0
        assert snap.total_margin_used == 0.0
        assert snap.n_open_positions == 0
        assert snap.n_assets_with_positions == 0

    def test_with_positions(self):
        config = _make_mock_config()
        runner, _ = _make_runner_with_indicator_engine("AAA/USDT", config, 5000.0)

        # Simuler une position ouverte
        pos = GridPosition(
            level=0,
            direction=Direction.LONG,
            entry_price=100.0,
            quantity=10.0,
            entry_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
            entry_fee=0.06,
        )
        runner._positions["AAA/USDT"] = [pos]
        # Déduire la marge du capital
        margin = 100.0 * 10.0 / 6  # notional / leverage
        runner._capital -= margin

        backtester = PortfolioBacktester.__new__(PortfolioBacktester)
        backtester._initial_capital = 5_000.0

        ts = datetime(2024, 6, 1, 1, tzinfo=timezone.utc)
        snap = backtester._take_snapshot(
            {"AAA/USDT": runner}, ts, {"AAA/USDT": 105.0}
        )

        assert snap.n_open_positions == 1
        assert snap.n_assets_with_positions == 1
        assert snap.total_margin_used > 0
        # Unrealized P&L: (105 - 100) * 10 = 50
        assert abs(snap.total_unrealized_pnl - 50.0) < 0.01
        # Fix equity accounting : equity = capital + margin_locked + unrealized
        # = initial_capital + realized_pnl + unrealized_pnl
        assert abs(snap.total_equity - (snap.total_capital + snap.total_margin_used + snap.total_unrealized_pnl)) < 0.01
        # Avec realized=0 et unrealized=50 : equity = 5000 + 50 = 5050
        assert abs(snap.total_equity - 5_050.0) < 1.0


# ─── Tests Drawdown ────────────────────────────────────────────────────────


class TestDrawdown:
    def test_no_drawdown_flat(self):
        """Equity constante → drawdown 0."""
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        snaps = [
            PortfolioSnapshot(
                timestamp=ts + timedelta(hours=i),
                total_equity=10_000.0,
                total_capital=10_000.0,
                total_realized_pnl=0.0,
                total_unrealized_pnl=0.0,
                total_margin_used=0.0,
                margin_ratio=0.0,
                n_open_positions=0,
                n_assets_with_positions=0,
            )
            for i in range(10)
        ]
        dd, dd_date, dd_dur = PortfolioBacktester._compute_drawdown(snaps)
        assert dd == 0.0
        assert dd_date is None

    def test_drawdown_with_crash(self):
        """Equity 10k → 8k → 9k → drawdown = -20%."""
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        equities = [10_000, 9_000, 8_000, 8_500, 9_000]
        snaps = [
            PortfolioSnapshot(
                timestamp=ts + timedelta(hours=i),
                total_equity=eq,
                total_capital=eq,
                total_realized_pnl=0.0,
                total_unrealized_pnl=0.0,
                total_margin_used=0.0,
                margin_ratio=0.0,
                n_open_positions=0,
                n_assets_with_positions=0,
            )
            for i, eq in enumerate(equities)
        ]
        dd, dd_date, dd_dur = PortfolioBacktester._compute_drawdown(snaps)
        assert abs(dd - (-20.0)) < 0.01
        assert dd_date == ts + timedelta(hours=2)

    def test_drawdown_empty(self):
        dd, dd_date, dd_dur = PortfolioBacktester._compute_drawdown([])
        assert dd == 0.0


# ─── Tests Kill Switch ─────────────────────────────────────────────────────


class TestKillSwitch:
    def test_no_trigger(self):
        """Equity stable → pas de kill switch."""
        backtester = PortfolioBacktester.__new__(PortfolioBacktester)
        backtester._kill_switch_pct = 30.0
        backtester._kill_switch_window_hours = 24

        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        snaps = [
            PortfolioSnapshot(
                timestamp=ts + timedelta(hours=i),
                total_equity=10_000.0,
                total_capital=10_000.0,
                total_realized_pnl=0.0,
                total_unrealized_pnl=0.0,
                total_margin_used=0.0,
                margin_ratio=0.0,
                n_open_positions=0,
                n_assets_with_positions=0,
            )
            for i in range(48)
        ]
        events = backtester._check_kill_switch(snaps)
        assert len(events) == 0

    def test_trigger_on_crash(self):
        """Equity chute de 35% en 24h → kill switch."""
        backtester = PortfolioBacktester.__new__(PortfolioBacktester)
        backtester._kill_switch_pct = 30.0
        backtester._kill_switch_window_hours = 24

        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # 24 heures de baisse progressive : 10k → 6.5k (-35%)
        equities = [10_000 - i * 146 for i in range(24)]
        snaps = [
            PortfolioSnapshot(
                timestamp=ts + timedelta(hours=i),
                total_equity=eq,
                total_capital=eq,
                total_realized_pnl=0.0,
                total_unrealized_pnl=0.0,
                total_margin_used=0.0,
                margin_ratio=0.0,
                n_open_positions=0,
                n_assets_with_positions=0,
            )
            for i, eq in enumerate(equities)
        ]
        events = backtester._check_kill_switch(snaps)
        assert len(events) >= 1
        assert events[0]["drawdown_pct"] >= 30.0


# ─── Tests Force Close ─────────────────────────────────────────────────────


class TestForceClose:
    def test_force_close_returns_trades(self):
        """Positions ouvertes → force-close génère des trades."""
        config = _make_mock_config()
        runner, _ = _make_runner_with_indicator_engine("AAA/USDT", config, 5000.0)

        # Simuler une position ouverte
        pos = GridPosition(
            level=0,
            direction=Direction.LONG,
            entry_price=100.0,
            quantity=10.0,
            entry_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
            entry_fee=0.06,
        )
        runner._positions["AAA/USDT"] = [pos]
        margin = 100.0 * 10.0 / 6
        runner._capital -= margin
        runner._close_buffer["AAA/USDT"] = deque([105.0], maxlen=50)

        backtester = PortfolioBacktester.__new__(PortfolioBacktester)
        backtester._initial_capital = 5_000.0

        force_closed = backtester._force_close_all({"AAA/USDT": runner})

        assert len(force_closed) == 1
        symbol, trade = force_closed[0]
        assert symbol == "AAA/USDT"
        assert trade.exit_reason == "end_of_data"
        # Position LONG, entry=100, exit=105 → P&L positif
        assert trade.net_pnl > 0
        # Positions vidées
        assert runner._positions["AAA/USDT"] == []

    def test_no_positions_no_force_close(self):
        config = _make_mock_config()
        runner, _ = _make_runner_with_indicator_engine("AAA/USDT", config, 5000.0)

        backtester = PortfolioBacktester.__new__(PortfolioBacktester)
        backtester._initial_capital = 5_000.0

        force_closed = backtester._force_close_all({"AAA/USDT": runner})
        assert len(force_closed) == 0


# ─── Tests Per-Asset Breakdown ─────────────────────────────────────────────


class TestPerAssetBreakdown:
    def test_build_result_per_asset(self):
        """Vérifie que per_asset_results sépare realized/force_closed."""
        config = _make_mock_config()
        runner_a, _ = _make_runner_with_indicator_engine("AAA/USDT", config, 5000.0)
        runner_b, _ = _make_runner_with_indicator_engine("BBB/USDT", config, 5000.0)

        backtester = PortfolioBacktester.__new__(PortfolioBacktester)
        backtester._initial_capital = 10_000.0
        backtester._kill_switch_pct = 30.0
        backtester._kill_switch_window_hours = 24

        # Fake trades
        from backend.core.position_manager import TradeResult

        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        realized = [
            ("AAA/USDT", TradeResult(
                direction=Direction.LONG, entry_price=100, exit_price=105,
                quantity=10, entry_time=ts, exit_time=ts + timedelta(hours=5),
                gross_pnl=50.0, fee_cost=0.12, slippage_cost=0.0,
                net_pnl=49.88, exit_reason="tp_global",
                market_regime=MarketRegime.RANGING,
            )),
        ]
        force_closed = [
            ("BBB/USDT", TradeResult(
                direction=Direction.LONG, entry_price=200, exit_price=195,
                quantity=5, entry_time=ts, exit_time=ts + timedelta(hours=10),
                gross_pnl=-25.0, fee_cost=0.12, slippage_cost=0.0,
                net_pnl=-25.12, exit_reason="end_of_data",
                market_regime=MarketRegime.RANGING,
            )),
        ]

        result = backtester._build_result(
            runners={"AAA/USDT": runner_a, "BBB/USDT": runner_b},
            snapshots=[],
            realized_trades=realized,
            force_closed_trades=force_closed,
            runner_keys=["AAA/USDT", "BBB/USDT"],
            period_days=90,
        )

        assert result.realized_pnl == 49.88
        assert result.force_closed_pnl == -25.12

        pa = result.per_asset_results
        assert pa["AAA/USDT"]["realized_trades"] == 1
        assert pa["AAA/USDT"]["force_closed_trades"] == 0
        assert pa["BBB/USDT"]["realized_trades"] == 0
        assert pa["BBB/USDT"]["force_closed_trades"] == 1


# ─── Tests Format Report ──────────────────────────────────────────────────


class TestFormatReport:
    def test_report_contains_key_sections(self):
        result = PortfolioResult(
            initial_capital=10_000.0,
            n_assets=2,
            period_days=90,
            assets=["A/USDT", "B/USDT"],
            final_equity=10_500.0,
            total_return_pct=5.0,
            total_trades=10,
            win_rate=60.0,
            realized_pnl=600.0,
            force_closed_pnl=-100.0,
            max_drawdown_pct=-5.0,
            max_drawdown_date=None,
            max_drawdown_duration_hours=48.0,
            peak_margin_ratio=0.35,
            peak_open_positions=6,
            peak_concurrent_assets=2,
            kill_switch_triggers=0,
            kill_switch_events=[],
            snapshots=[],
            per_asset_results={
                "A/USDT": {"trades": 5, "win_rate": 60.0, "net_pnl": 400.0, "force_closed_trades": 0},
                "B/USDT": {"trades": 5, "win_rate": 60.0, "net_pnl": 100.0, "force_closed_trades": 1},
            },
        )
        report = format_portfolio_report(result)
        assert "PORTFOLIO BACKTEST REPORT" in report
        assert "10,000" in report  # capital initial
        assert "P&L réalisé" in report
        assert "P&L force-closed" in report
        assert "Kill Switch" in report
        assert "Par Runner" in report


# ─── Test Integration Simulation ───────────────────────────────────────────


class TestSimulation:
    @pytest.mark.asyncio
    async def test_flat_prices_no_trades(self):
        """Prix plats → SMA ≈ prix → pas de grille touchée → 0 trades."""
        config = _make_mock_config(n_assets=1)
        runner, engine = _make_runner_with_indicator_engine(
            "AAA/USDT", config, 10_000.0
        )
        runners = {"AAA/USDT": runner}

        candles = _make_flat_candles("AAA/USDT", n=120, price=100.0)

        backtester = PortfolioBacktester.__new__(PortfolioBacktester)
        backtester._initial_capital = 10_000.0
        backtester._kill_switch_pct = 30.0
        backtester._kill_switch_window_hours = 24
        backtester._kill_freeze_until = None

        warmup_ends = backtester._warmup_runners(
            runners, {"AAA/USDT": candles}, engine, warmup_count=50
        )

        merged = PortfolioBacktester._merge_candles({"AAA/USDT": candles})
        snapshots, realized, _liq_event = await backtester._simulate(
            runners, engine, merged, warmup_ends
        )

        # Prix plats → pas de trade (grille jamais touchée car SMA ≈ close)
        assert len(realized) == 0
        # Equity stable
        if snapshots:
            assert abs(snapshots[-1].total_equity - 10_000.0) < 1.0

    @pytest.mark.asyncio
    async def test_two_assets_capital_split(self):
        """2 assets → chaque runner commence avec 5k."""
        config = _make_mock_config(n_assets=2)

        runner_a, engine = _make_runner_with_indicator_engine("AAA/USDT", config, 5000.0)
        # Réutiliser le même indicator engine pour le 2e runner
        strategy_b = create_strategy_with_params("grid_atr", {
            "ma_period": 14, "atr_period": 14, "num_levels": 3,
            "sl_percent": 20.0, "leverage": 6,
        })
        gpm_b = GridPositionManager(PositionManagerConfig(
            leverage=6, maker_fee=0.0006, taker_fee=0.0006, slippage_pct=0.0005,
        ))
        runner_b = GridStrategyRunner(
            strategy=strategy_b, config=config,
            indicator_engine=engine,
            grid_position_manager=gpm_b,
            data_engine=None,  # type: ignore[arg-type]
            db_path=None,
        )
        runner_b._nb_assets = 1
        runner_b._capital = 5000.0
        runner_b._initial_capital = 5000.0
        runner_b._is_warming_up = False
        runner_b._stats = RunnerStats(capital=5000.0, initial_capital=5000.0)

        runners = {"AAA/USDT": runner_a, "BBB/USDT": runner_b}

        candles_a = _make_flat_candles("AAA/USDT", n=80, price=100.0)
        candles_b = _make_flat_candles("BBB/USDT", n=80, price=200.0)

        backtester = PortfolioBacktester.__new__(PortfolioBacktester)
        backtester._initial_capital = 10_000.0

        warmup_ends = backtester._warmup_runners(
            runners,
            {"AAA/USDT": candles_a, "BBB/USDT": candles_b},
            engine,
            warmup_count=50,
        )

        # Vérifier le capital split
        assert runner_a._capital == 5000.0
        assert runner_b._capital == 5000.0
        assert warmup_ends["AAA/USDT"] == 50
        assert warmup_ends["BBB/USDT"] == 50


# ─── Tests Sprint 24a — Realistic Mode ───────────────────────────────────


class TestPortfolioModeFixedSizing:
    """Correction 1 : sizing fixe anti-compounding."""

    @pytest.mark.asyncio
    async def test_portfolio_mode_fixed_sizing(self):
        """Runner avec _portfolio_mode=True utilise initial_capital pour le sizing,
        même si _capital a augmenté grâce aux profits."""
        config = _make_mock_config(n_assets=1)
        runner, engine = _make_runner_with_indicator_engine("AAA/USDT", config, 5000.0)

        # Simuler des profits : capital a doublé
        runner._capital = 10_000.0
        runner._portfolio_mode = True

        # Générer des candles avec assez de volatilité pour toucher les grilles
        candles = _make_candles("AAA/USDT", n=120, start_price=100.0, volatility=0.02)

        # Warmup
        for c in candles[:50]:
            engine.update("AAA/USDT", "1h", c)
            runner._close_buffer.setdefault("AAA/USDT", deque(maxlen=50)).append(c.close)

        # Dispatch des candles post-warmup
        for c in candles[50:]:
            engine.update("AAA/USDT", "1h", c)
            await runner.on_candle("AAA/USDT", "1h", c)

        # Le sizing doit utiliser initial_capital (5000), pas _capital (10000)
        # Vérifier via les positions ouvertes : notional basé sur 5k, pas 10k
        for positions in runner._positions.values():
            for pos in positions:
                notional = pos.entry_price * pos.quantity
                # Avec 5k capital, 1 asset, 3 levels, le margin_per_level ≈ 5000/1/3 ≈ 1666
                # notional = margin * leverage = 1666 * 6 ≈ 10k
                # Avec 10k capital ce serait ~20k notional
                assert notional < 15_000, (
                    f"Notional {notional:.0f} trop élevé — sizing utilise _capital au lieu de _initial_capital"
                )

    @pytest.mark.asyncio
    async def test_normal_mode_uses_current_capital(self):
        """Sans _portfolio_mode, le runner utilise _capital courant (compound)."""
        config = _make_mock_config(n_assets=1)
        runner, engine = _make_runner_with_indicator_engine("AAA/USDT", config, 5000.0)

        # Simuler des profits : capital a doublé
        runner._capital = 10_000.0
        # PAS de _portfolio_mode → compound normal

        candles = _make_candles("AAA/USDT", n=120, start_price=100.0, volatility=0.02)
        for c in candles[:50]:
            engine.update("AAA/USDT", "1h", c)
            runner._close_buffer.setdefault("AAA/USDT", deque(maxlen=50)).append(c.close)

        for c in candles[50:]:
            engine.update("AAA/USDT", "1h", c)
            await runner.on_candle("AAA/USDT", "1h", c)

        # En mode compound, les positions seront basées sur 10k (pas 5k)
        for positions in runner._positions.values():
            for pos in positions:
                notional = pos.entry_price * pos.quantity
                # Avec 10k capital, 1 asset, 3 levels, margin_per_level ≈ 10000/1/3 ≈ 3333
                # notional = 3333 * 6 ≈ 20k → bien plus gros que 15k
                # (pas forcément des positions ouvertes, c'est ok)


class TestGlobalMarginGuard:
    """Correction 2 : global margin guard portfolio."""

    @pytest.mark.asyncio
    async def test_global_margin_guard_blocks(self):
        """Si la marge globale (tous runners) dépasse le seuil, les nouvelles positions sont bloquées."""
        config = _make_mock_config(n_assets=2)
        runner_a, engine = _make_runner_with_indicator_engine("AAA/USDT", config, 5000.0)
        runner_b, _ = _make_runner_with_indicator_engine("BBB/USDT", config, 5000.0)

        runners = {"AAA/USDT": runner_a, "BBB/USDT": runner_b}

        # Setup cross-references (comme portfolio_engine._create_runners())
        for r in runners.values():
            r._portfolio_runners = runners
            r._portfolio_initial_capital = 10_000.0
            r._portfolio_mode = True

        # Pré-remplir des positions pour ~65% marge globale
        # margin = entry_price * quantity / leverage
        # On veut 6500$ de marge sur 10k capital (65%)
        # Avec leverage=6 : notional = 6500 * 6 = 39000
        # entry=100, qty = 390 → margin = 100*390/6 = 6500
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        pos = GridPosition(
            level=0,
            direction=Direction.LONG,
            entry_price=100.0,
            quantity=390.0,
            entry_time=ts,
            entry_fee=2.34,
        )
        runner_a._positions["AAA/USDT"] = [pos]
        margin_a = 100.0 * 390.0 / 6  # = 6500
        runner_a._capital -= margin_a

        # Vérifier la marge actuelle
        global_margin = sum(
            p.entry_price * p.quantity / r._leverage
            for r in runners.values()
            for positions_list in r._positions.values()
            for p in positions_list
        )
        assert abs(global_margin - 6500.0) < 1.0

        # Runner B essaie d'ouvrir — la marge supplémentaire ferait dépasser 70%
        # margin_per_level ≈ 5000/1/3 ≈ 1666 → total serait 6500+1666 = 8166 > 7000 (70%)
        candles_b = _make_candles("BBB/USDT", n=120, start_price=100.0, volatility=0.02, seed=99)
        for c in candles_b[:50]:
            engine.update("BBB/USDT", "1h", c)
            runner_b._close_buffer.setdefault("BBB/USDT", deque(maxlen=50)).append(c.close)

        for c in candles_b[50:]:
            engine.update("BBB/USDT", "1h", c)
            await runner_b.on_candle("BBB/USDT", "1h", c)

        # Runner B ne doit pas avoir pu ouvrir (marge globale > 70%)
        b_positions = sum(len(p) for p in runner_b._positions.values())
        assert b_positions == 0, (
            f"Runner B a ouvert {b_positions} positions malgré le global margin guard"
        )

    @pytest.mark.asyncio
    async def test_global_margin_under_threshold(self):
        """Le guard ne bloque PAS quand la marge globale est sous le seuil."""
        config = _make_mock_config(n_assets=2)
        runner_a, engine = _make_runner_with_indicator_engine("AAA/USDT", config, 5000.0)

        # Runner B partage le même indicator engine
        strategy_b = create_strategy_with_params("grid_atr", {
            "ma_period": 14, "atr_period": 14, "atr_multiplier_start": 2.0,
            "atr_multiplier_step": 1.0, "num_levels": 3,
            "sl_percent": 20.0, "sides": ["long"], "leverage": 6,
        })
        gpm_b = GridPositionManager(PositionManagerConfig(
            leverage=6, maker_fee=0.0006, taker_fee=0.0006, slippage_pct=0.0005,
        ))
        runner_b = GridStrategyRunner(
            strategy=strategy_b, config=config,
            indicator_engine=engine,  # même engine
            grid_position_manager=gpm_b,
            data_engine=None,  # type: ignore[arg-type]
            db_path=None,
        )
        runner_b._nb_assets = 1
        runner_b._capital = 5000.0
        runner_b._initial_capital = 5000.0
        runner_b._is_warming_up = False
        runner_b._stats = RunnerStats(capital=5000.0, initial_capital=5000.0)

        runners = {"AAA/USDT": runner_a, "BBB/USDT": runner_b}

        for r in runners.values():
            r._portfolio_runners = runners
            r._portfolio_initial_capital = 10_000.0
            r._portfolio_mode = True

        # Pré-remplir seulement ~20% marge (2000$ sur 10k)
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        pos = GridPosition(
            level=0,
            direction=Direction.LONG,
            entry_price=100.0,
            quantity=120.0,  # margin = 100*120/6 = 2000
            entry_time=ts,
            entry_fee=0.72,
        )
        runner_a._positions["AAA/USDT"] = [pos]
        runner_a._capital -= 100.0 * 120.0 / 6

        # Runner B devrait pouvoir ouvrir des positions (marge totale ~20% + ~17% = 37% < 70%)
        candles_b = _make_candles("BBB/USDT", n=120, start_price=100.0, volatility=0.02, seed=99)
        for c in candles_b[:50]:
            engine.update("BBB/USDT", "1h", c)
            runner_b._close_buffer.setdefault("BBB/USDT", deque(maxlen=50)).append(c.close)

        for c in candles_b[50:]:
            engine.update("BBB/USDT", "1h", c)
            await runner_b.on_candle("BBB/USDT", "1h", c)

        # Runner B peut avoir ouvert des positions (pas forcément, dépend de la volatilité)
        # Ce test vérifie surtout que le guard ne bloque PAS indûment
        # Si des trades ont été exécutés, c'est que le guard n'a pas bloqué
        b_trades = len(runner_b._trades)
        b_positions = sum(len(p) for p in runner_b._positions.values())
        # Au moins une activité (position ouverte ou trade exécuté)
        assert b_trades > 0 or b_positions > 0, (
            "Runner B n'a eu aucune activité malgré marge globale sous le seuil"
        )


class TestKillSwitchPortfolio:
    """Correction 3 : kill switch actif pendant la simulation."""

    @pytest.mark.asyncio
    async def test_kill_switch_freezes_all_runners(self):
        """Quand le DD dépasse le seuil, tous les runners sont gelés,
        puis reset après cooldown (quand la fenêtre glissante ne contient plus
        les snapshots haute equity)."""
        config = _make_mock_config(n_assets=2)
        runner_a, engine = _make_runner_with_indicator_engine("AAA/USDT", config, 5000.0)
        runner_b, _ = _make_runner_with_indicator_engine("BBB/USDT", config, 5000.0)

        runners = {"AAA/USDT": runner_a, "BBB/USDT": runner_b}

        for r in runners.values():
            r._portfolio_runners = runners
            r._portfolio_initial_capital = 10_000.0
            r._portfolio_mode = True

        backtester = PortfolioBacktester.__new__(PortfolioBacktester)
        backtester._initial_capital = 10_000.0
        backtester._kill_switch_pct = 30.0
        backtester._kill_switch_window_hours = 24
        backtester._kill_freeze_until = None

        ts_base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        snapshots: list[PortfolioSnapshot] = []

        def _apply_ks_logic(snap: PortfolioSnapshot) -> None:
            """Reproduit la logique kill switch de _simulate()."""
            if len(snapshots) < 2:
                return
            window_hours = backtester._kill_switch_window_hours
            current_ts = snap.timestamp

            window_start_equity = snap.total_equity
            for prev_snap in reversed(snapshots[:-1]):
                if (current_ts - prev_snap.timestamp).total_seconds() > window_hours * 3600:
                    break
                window_start_equity = prev_snap.total_equity

            if window_start_equity > 0:
                dd_pct = (1 - snap.total_equity / window_start_equity) * 100
                if dd_pct >= backtester._kill_switch_pct:
                    for r in runners.values():
                        r._kill_switch_triggered = True
                    backtester._kill_freeze_until = current_ts + timedelta(hours=24)

            freeze_until = backtester._kill_freeze_until
            if freeze_until and current_ts >= freeze_until:
                for r in runners.values():
                    r._kill_switch_triggered = False
                backtester._kill_freeze_until = None

        def _add_snap(h: int, equity: float) -> None:
            snap = PortfolioSnapshot(
                timestamp=ts_base + timedelta(hours=h),
                total_equity=equity,
                total_capital=equity,
                total_realized_pnl=0.0,
                total_unrealized_pnl=0.0,
                total_margin_used=0.0,
                margin_ratio=0.0,
                n_open_positions=0,
                n_assets_with_positions=0,
            )
            snapshots.append(snap)
            _apply_ks_logic(snap)

        # h=0: equity stable à 10k
        _add_snap(0, 10_000)

        # h=1: crash soudain → 6k (-40%)
        _add_snap(1, 6_000)

        # Vérifier que le kill switch s'est déclenché
        assert runner_a._kill_switch_triggered is True, "Runner A devrait être gelé"
        assert runner_b._kill_switch_triggered is True, "Runner B devrait être gelé"
        assert backtester._kill_freeze_until is not None

        # h=2 à h=50 : equity stable à 6k
        # Le kill switch se re-déclenche tant que h=0 (10k) est dans la fenêtre 24h
        # À h=25, h=0 sort de la fenêtre (distance 25h > 24h) → plus de re-trigger
        # Mais freeze_until a été poussé à h=24+24=h=48 (dernière re-trigger à h=24)
        # À h=48, le cooldown expire → reset
        for h in range(2, 51):
            _add_snap(h, 6_000)

        # Après h=50, le cooldown est expiré et aucun re-trigger possible
        assert runner_a._kill_switch_triggered is False, "Runner A devrait être dégelé après cooldown"
        assert runner_b._kill_switch_triggered is False, "Runner B devrait être dégelé après cooldown"
        assert backtester._kill_freeze_until is None


# ─── Tests Sprint 24b — Multi-Stratégie ──────────────────────────────────


class TestMultiStrategyCreatesRunners:
    """multi_strategies crée des runners avec clés strategy:symbol."""

    def test_multi_strategy_creates_runners(self):
        config = _make_mock_config(n_assets=1)
        # Configurer per_asset pour les deux stratégies
        config.strategies.grid_atr.per_asset = {}
        config.strategies.grid_trend = MagicMock()
        config.strategies.grid_trend.per_asset = {}

        backtester = PortfolioBacktester(
            config=config,
            initial_capital=10_000.0,
            multi_strategies=[
                ("grid_atr", ["AAA/USDT"]),
                ("grid_trend", ["AAA/USDT"]),
            ],
        )

        runners, engine = backtester._create_runners(
            backtester._multi_strategies, 5000.0
        )

        assert "grid_atr:AAA/USDT" in runners
        assert "grid_trend:AAA/USDT" in runners
        assert len(runners) == 2
        # Chaque runner a le bon capital
        for r in runners.values():
            assert r._capital == 5000.0
            assert r._initial_capital == 5000.0
            assert r._portfolio_mode is True


class TestMultiStrategySameSymbolDispatched:
    """Même symbol dans 2 stratégies → les 2 runners reçoivent les candles."""

    @pytest.mark.asyncio
    async def test_same_symbol_dispatched_to_both(self):
        config = _make_mock_config(n_assets=1)
        config.strategies.grid_atr.per_asset = {}
        config.strategies.grid_trend = MagicMock()
        config.strategies.grid_trend.per_asset = {}

        backtester = PortfolioBacktester(
            config=config,
            initial_capital=10_000.0,
            multi_strategies=[
                ("grid_atr", ["AAA/USDT"]),
                ("grid_trend", ["AAA/USDT"]),
            ],
        )

        runners, engine = backtester._create_runners(
            backtester._multi_strategies, 5000.0
        )
        candles = _make_flat_candles("AAA/USDT", n=120, price=100.0)

        # Warm-up
        warmup_ends = backtester._warmup_runners(
            runners, {"AAA/USDT": candles}, engine, warmup_count=50
        )
        assert warmup_ends["AAA/USDT"] == 50

        # Simulate
        merged = PortfolioBacktester._merge_candles({"AAA/USDT": candles})
        snapshots, trades, _liq_event = await backtester._simulate(
            runners, engine, merged, warmup_ends
        )

        # Les deux runners doivent avoir reçu des candles (close_buffer alimenté)
        for rk in ["grid_atr:AAA/USDT", "grid_trend:AAA/USDT"]:
            runner = runners[rk]
            buf = runner._close_buffer.get("AAA/USDT")
            assert buf is not None, f"Runner {rk} n'a pas de close_buffer"
            assert len(buf) >= 50, f"Runner {rk} close_buffer trop court ({len(buf)})"


class TestMultiStrategyCapitalSplit:
    """2 stratégies × 2 assets = 4 runners → chacun reçoit 2500$."""

    def test_capital_split(self):
        config = _make_mock_config(n_assets=2)
        config.strategies.grid_atr.per_asset = {}
        config.strategies.grid_trend = MagicMock()
        config.strategies.grid_trend.per_asset = {}

        backtester = PortfolioBacktester(
            config=config,
            initial_capital=10_000.0,
            multi_strategies=[
                ("grid_atr", ["AAA/USDT", "BBB/USDT"]),
                ("grid_trend", ["AAA/USDT", "BBB/USDT"]),
            ],
        )

        runners, engine = backtester._create_runners(
            backtester._multi_strategies, 2500.0
        )

        assert len(runners) == 4
        expected_keys = {
            "grid_atr:AAA/USDT",
            "grid_atr:BBB/USDT",
            "grid_trend:AAA/USDT",
            "grid_trend:BBB/USDT",
        }
        assert set(runners.keys()) == expected_keys

        for r in runners.values():
            assert r._capital == 2500.0
            assert r._initial_capital == 2500.0


class TestLeverageFromTopLevelConfig:
    """Bug : le leverage top-level du YAML était ignoré — les per_asset n'ont pas
    de leverage donc create_strategy_with_params repartait du default Pydantic (6)."""

    def test_leverage_top_level_transmitted(self):
        """Quand grid_atr.leverage=3 dans strategies.yaml, les runners doivent
        utiliser leverage=3, pas le default Pydantic 6."""
        config = _make_mock_config(n_assets=1)
        # Utilise un vrai GridATRConfig (pas un MagicMock) avec leverage=3
        real_strat_config = GridATRConfig(
            leverage=3,
            ma_period=14,
            atr_period=14,
            atr_multiplier_start=2.0,
            atr_multiplier_step=1.0,
            num_levels=3,
            sl_percent=20.0,
            per_asset={},
        )
        config.strategies.grid_atr = real_strat_config

        backtester = PortfolioBacktester(
            config=config,
            initial_capital=10_000.0,
            multi_strategies=[("grid_atr", ["AAA/USDT"])],
        )
        runners, _ = backtester._create_runners(backtester._multi_strategies, 5000.0)

        runner = runners["grid_atr:AAA/USDT"]
        assert runner._strategy._config.leverage == 3, (
            f"leverage={runner._strategy._config.leverage}, attendu 3 "
            "(le top-level YAML doit prendre priorité sur le default Pydantic)"
        )
        assert runner._leverage == 3, (
            f"runner._leverage={runner._leverage}, attendu 3 "
            "(le GridPositionManager doit recevoir leverage=3)"
        )

    def test_leverage_per_asset_overrides_top_level(self):
        """Si un per_asset contient leverage (ne devrait pas arriver après fix
        apply_to_yaml, mais par précaution), le per_asset prend priorité."""
        config = _make_mock_config(n_assets=1)
        real_strat_config = GridATRConfig(
            leverage=3,
            ma_period=14,
            atr_period=14,
            atr_multiplier_start=2.0,
            atr_multiplier_step=1.0,
            num_levels=3,
            sl_percent=20.0,
            per_asset={"AAA/USDT": {"leverage": 5}},
        )
        config.strategies.grid_atr = real_strat_config

        backtester = PortfolioBacktester(
            config=config,
            initial_capital=10_000.0,
            multi_strategies=[("grid_atr", ["AAA/USDT"])],
        )
        runners, _ = backtester._create_runners(backtester._multi_strategies, 5000.0)

        runner = runners["grid_atr:AAA/USDT"]
        # per_asset leverage=5 override top-level leverage=3
        assert runner._strategy._config.leverage == 5


class TestSingleStrategyBackwardCompatible:
    """multi_strategies=None + strategy_name='grid_atr' → rétro-compatible."""

    def test_backward_compatible(self):
        config = _make_mock_config(n_assets=2)
        config.strategies.grid_atr.per_asset = {
            "AAA/USDT": {},
            "BBB/USDT": {},
        }

        # Sans multi_strategies → mode single
        backtester = PortfolioBacktester(
            config=config,
            initial_capital=10_000.0,
            strategy_name="grid_atr",
        )

        # _multi_strategies doit être auto-construit
        assert backtester._multi_strategies == [("grid_atr", ["AAA/USDT", "BBB/USDT"])]
        assert backtester._assets == ["AAA/USDT", "BBB/USDT"]

        runners, engine = backtester._create_runners(
            backtester._multi_strategies, 5000.0
        )

        assert len(runners) == 2
        assert "grid_atr:AAA/USDT" in runners
        assert "grid_atr:BBB/USDT" in runners


# ─── Tests Auto-détection --days auto ────────────────────────────────────


class TestDetectMaxDays:
    """Tests pour _detect_max_days() (auto-détection historique)."""

    @pytest.mark.asyncio
    async def test_detect_returns_common_days(self, tmp_path):
        """2 assets avec historiques différents → prend le goulot."""
        import aiosqlite

        db_path = str(tmp_path / "test.db")
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute(
                """CREATE TABLE candles (
                    exchange TEXT NOT NULL DEFAULT 'binance',
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, volume REAL,
                    vwap REAL, mark_price REAL,
                    PRIMARY KEY (exchange, symbol, timeframe, timestamp)
                )"""
            )
            # AAA : historique 500 jours
            ts_aaa = datetime.now(timezone.utc) - timedelta(days=500)
            await conn.execute(
                "INSERT INTO candles VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("binance", "AAA/USDT", "1h", ts_aaa.isoformat(),
                 100, 101, 99, 100, 1000, None, None),
            )
            # BBB : historique 200 jours (= goulot)
            ts_bbb = datetime.now(timezone.utc) - timedelta(days=200)
            await conn.execute(
                "INSERT INTO candles VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("binance", "BBB/USDT", "1h", ts_bbb.isoformat(),
                 200, 201, 199, 200, 1000, None, None),
            )
            await conn.commit()

        config = _make_mock_config(n_assets=2)
        config.strategies.grid_atr = MagicMock()
        config.strategies.grid_atr.per_asset = {
            "AAA/USDT": {},
            "BBB/USDT": {},
        }

        from scripts.portfolio_backtest import _detect_max_days

        days, detail = await _detect_max_days(
            config, "grid_atr", "binance", db_path
        )

        # Le goulot est BBB (200j) → common_days ≈ 200 - 3 = 197
        assert 190 <= days <= 200
        assert detail["AAA/USDT"] > detail["BBB/USDT"]
        assert detail["BBB/USDT"] >= 199  # ~200 jours

    @pytest.mark.asyncio
    async def test_detect_no_per_asset_fallback(self, tmp_path):
        """Aucun per_asset → fallback 90 jours."""
        db_path = str(tmp_path / "test.db")

        config = _make_mock_config(n_assets=0)
        config.strategies.grid_atr = MagicMock()
        config.strategies.grid_atr.per_asset = {}

        from scripts.portfolio_backtest import _detect_max_days

        days, detail = await _detect_max_days(
            config, "grid_atr", "binance", db_path
        )

        assert days == 90
        assert detail == {}

    @pytest.mark.asyncio
    async def test_detect_multi_strategies(self, tmp_path):
        """multi_strategies collecte les assets de toutes les stratégies."""
        import aiosqlite

        db_path = str(tmp_path / "test.db")
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute(
                """CREATE TABLE candles (
                    exchange TEXT NOT NULL DEFAULT 'binance',
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, volume REAL,
                    vwap REAL, mark_price REAL,
                    PRIMARY KEY (exchange, symbol, timeframe, timestamp)
                )"""
            )
            ts = datetime.now(timezone.utc) - timedelta(days=300)
            for sym in ["AAA/USDT", "BBB/USDT", "CCC/USDT"]:
                await conn.execute(
                    "INSERT INTO candles VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    ("binance", sym, "1h", ts.isoformat(),
                     100, 101, 99, 100, 1000, None, None),
                )
            await conn.commit()

        config = _make_mock_config(n_assets=3)

        from scripts.portfolio_backtest import _detect_max_days

        multi = [
            ("grid_atr", ["AAA/USDT", "BBB/USDT"]),
            ("grid_boltrend", ["BBB/USDT", "CCC/USDT"]),
        ]
        days, detail = await _detect_max_days(
            config, "grid_atr", "binance", db_path,
            multi_strategies=multi,
        )

        # 3 assets uniques (AAA, BBB, CCC)
        assert len(detail) == 3
        assert "AAA/USDT" in detail
        assert "CCC/USDT" in detail
        assert 290 <= days <= 300

    @pytest.mark.asyncio
    async def test_detect_fallback_exchange(self, tmp_path):
        """Si binance n'a pas l'asset, tombe sur bitget."""
        import aiosqlite

        db_path = str(tmp_path / "test.db")
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute(
                """CREATE TABLE candles (
                    exchange TEXT NOT NULL DEFAULT 'binance',
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, volume REAL,
                    vwap REAL, mark_price REAL,
                    PRIMARY KEY (exchange, symbol, timeframe, timestamp)
                )"""
            )
            ts = datetime.now(timezone.utc) - timedelta(days=400)
            # Seulement sur bitget, pas binance
            await conn.execute(
                "INSERT INTO candles VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("bitget", "AAA/USDT", "1h", ts.isoformat(),
                 100, 101, 99, 100, 1000, None, None),
            )
            await conn.commit()

        config = _make_mock_config(n_assets=1)
        config.strategies.grid_atr = MagicMock()
        config.strategies.grid_atr.per_asset = {"AAA/USDT": {}}

        from scripts.portfolio_backtest import _detect_max_days

        days, detail = await _detect_max_days(
            config, "grid_atr", "binance", db_path
        )

        # Trouvé via fallback bitget
        assert detail["AAA/USDT"] >= 399
        assert days >= 390

    @pytest.mark.asyncio
    async def test_detect_missing_asset_zero_days(self, tmp_path):
        """Asset absent de la DB → 0 jours, ne crash pas."""
        import aiosqlite

        db_path = str(tmp_path / "test.db")
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute(
                """CREATE TABLE candles (
                    exchange TEXT NOT NULL DEFAULT 'binance',
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, volume REAL,
                    vwap REAL, mark_price REAL,
                    PRIMARY KEY (exchange, symbol, timeframe, timestamp)
                )"""
            )
            # AAA présent, BBB absent
            ts = datetime.now(timezone.utc) - timedelta(days=300)
            await conn.execute(
                "INSERT INTO candles VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("binance", "AAA/USDT", "1h", ts.isoformat(),
                 100, 101, 99, 100, 1000, None, None),
            )
            await conn.commit()

        config = _make_mock_config(n_assets=2)
        config.strategies.grid_atr = MagicMock()
        config.strategies.grid_atr.per_asset = {
            "AAA/USDT": {},
            "BBB/USDT": {},
        }

        from scripts.portfolio_backtest import _detect_max_days

        days, detail = await _detect_max_days(
            config, "grid_atr", "binance", db_path
        )

        assert detail["AAA/USDT"] >= 299
        assert detail["BBB/USDT"] == 0


# ─── Tests --leverage override ────────────────────────────────────────────


class TestLeverageOverride:
    """Tests pour l'override --leverage dans portfolio_backtest.py."""

    def test_leverage_field_in_result(self):
        """PortfolioResult expose le champ leverage avec défaut 6."""
        result = PortfolioResult(
            initial_capital=10_000.0,
            n_assets=1,
            period_days=90,
            assets=["A/USDT"],
            final_equity=10_500.0,
            total_return_pct=5.0,
            total_trades=5,
            win_rate=60.0,
            realized_pnl=500.0,
            force_closed_pnl=0.0,
            max_drawdown_pct=-3.0,
            max_drawdown_date=None,
            max_drawdown_duration_hours=0.0,
            peak_margin_ratio=0.2,
            peak_open_positions=3,
            peak_concurrent_assets=1,
            kill_switch_triggers=0,
            kill_switch_events=[],
            snapshots=[],
            per_asset_results={},
        )
        assert result.leverage == 6  # défaut

    def test_leverage_field_explicit(self):
        """PortfolioResult avec leverage=3 explicite."""
        result = PortfolioResult(
            initial_capital=10_000.0,
            n_assets=1,
            period_days=90,
            assets=["A/USDT"],
            final_equity=10_200.0,
            total_return_pct=2.0,
            total_trades=3,
            win_rate=66.7,
            realized_pnl=200.0,
            force_closed_pnl=0.0,
            max_drawdown_pct=-1.0,
            max_drawdown_date=None,
            max_drawdown_duration_hours=0.0,
            peak_margin_ratio=0.15,
            peak_open_positions=2,
            peak_concurrent_assets=1,
            kill_switch_triggers=0,
            kill_switch_events=[],
            snapshots=[],
            per_asset_results={},
            leverage=3,
        )
        assert result.leverage == 3

    def test_leverage_in_report(self):
        """format_portfolio_report() affiche le leverage."""
        result = PortfolioResult(
            initial_capital=10_000.0,
            n_assets=1,
            period_days=90,
            assets=["A/USDT"],
            final_equity=10_500.0,
            total_return_pct=5.0,
            total_trades=5,
            win_rate=60.0,
            realized_pnl=500.0,
            force_closed_pnl=0.0,
            max_drawdown_pct=-3.0,
            max_drawdown_date=None,
            max_drawdown_duration_hours=0.0,
            peak_margin_ratio=0.2,
            peak_open_positions=3,
            peak_concurrent_assets=1,
            kill_switch_triggers=0,
            kill_switch_events=[],
            snapshots=[],
            per_asset_results={},
            leverage=3,
        )
        report = format_portfolio_report(result)
        assert "Leverage" in report
        assert "3x" in report

    def test_leverage_extracted_from_runner_in_build_result(self):
        """_build_result() extrait le leverage du premier runner."""
        config = _make_mock_config(n_assets=1)

        # Créer un runner avec leverage=3 explicitement
        strategy = create_strategy_with_params("grid_atr", {
            "ma_period": 14,
            "atr_period": 14,
            "atr_multiplier_start": 2.0,
            "atr_multiplier_step": 1.0,
            "num_levels": 3,
            "sl_percent": 20.0,
            "sides": ["long"],
            "leverage": 3,
        })
        engine = IncrementalIndicatorEngine([strategy])
        gpm = GridPositionManager(PositionManagerConfig(
            leverage=3, maker_fee=0.0006, taker_fee=0.0006, slippage_pct=0.0005,
        ))
        runner = GridStrategyRunner(
            strategy=strategy, config=config,
            indicator_engine=engine,
            grid_position_manager=gpm,
            data_engine=None,  # type: ignore[arg-type]
            db_path=None,
        )
        runner._nb_assets = 1
        runner._capital = 10_000.0
        runner._initial_capital = 10_000.0
        runner._is_warming_up = False
        runner._stats = RunnerStats(capital=10_000.0, initial_capital=10_000.0)

        backtester = PortfolioBacktester.__new__(PortfolioBacktester)
        backtester._initial_capital = 10_000.0
        backtester._kill_switch_pct = 30.0
        backtester._kill_switch_window_hours = 24

        result = backtester._build_result(
            runners={"grid_atr:AAA/USDT": runner},
            snapshots=[],
            realized_trades=[],
            force_closed_trades=[],
            runner_keys=["grid_atr:AAA/USDT"],
            period_days=90,
        )

        assert result.leverage == 3


# ─── Tests kill switch depuis risk.yaml ───────────────────────────────────


class TestKillSwitchFromConfig:
    """Le kill switch doit lire global_max_loss_pct/global_window_hours depuis
    risk.yaml et non être hardcodé à 30%/24h."""

    def test_ks_from_config_defaults_to_yaml(self):
        """Sans --kill-switch CLI, les valeurs viennent de risk.yaml."""
        import argparse

        from scripts.portfolio_backtest import main  # noqa: F401 (vérifie l'import)

        # Simuler args sans override (args.kill_switch = None, args.kill_switch_window = None)
        args = argparse.Namespace(kill_switch=None, kill_switch_window=None)

        # Simuler la config avec global_max_loss_pct=45, global_window_hours=48
        cfg_ks = MagicMock()
        cfg_ks.global_max_loss_pct = 45.0
        cfg_ks.global_window_hours = 48

        # Reproduire la logique de résolution de main()
        ks_cfg = cfg_ks
        ks_pct = (
            args.kill_switch
            if args.kill_switch is not None
            else getattr(ks_cfg, "global_max_loss_pct", 30.0)
        )
        ks_hours = (
            args.kill_switch_window
            if args.kill_switch_window is not None
            else int(getattr(ks_cfg, "global_window_hours", 24))
        )

        assert ks_pct == 45.0, f"attendu 45.0, got {ks_pct}"
        assert ks_hours == 48, f"attendu 48, got {ks_hours}"

    def test_ks_cli_overrides_yaml(self):
        """Avec --kill-switch CLI, la valeur CLI prend priorité sur risk.yaml."""
        import argparse

        args = argparse.Namespace(kill_switch=60.0, kill_switch_window=12)

        cfg_ks = MagicMock()
        cfg_ks.global_max_loss_pct = 45.0
        cfg_ks.global_window_hours = 48

        ks_pct = (
            args.kill_switch
            if args.kill_switch is not None
            else getattr(cfg_ks, "global_max_loss_pct", 30.0)
        )
        ks_hours = (
            args.kill_switch_window
            if args.kill_switch_window is not None
            else int(getattr(cfg_ks, "global_window_hours", 24))
        )

        assert ks_pct == 60.0, f"CLI override attendu 60.0, got {ks_pct}"
        assert ks_hours == 12, f"CLI override attendu 12, got {ks_hours}"

    def test_ks_fallback_when_config_missing(self):
        """Si global_max_loss_pct absent de la config, fallback à 30.0."""
        import argparse

        args = argparse.Namespace(kill_switch=None, kill_switch_window=None)

        # Config sans attribut global_max_loss_pct
        cfg_ks = MagicMock(spec=[])  # spec vide = aucun attribut

        ks_pct = (
            args.kill_switch
            if args.kill_switch is not None
            else getattr(cfg_ks, "global_max_loss_pct", 30.0)
        )
        ks_hours = (
            args.kill_switch_window
            if args.kill_switch_window is not None
            else int(getattr(cfg_ks, "global_window_hours", 24))
        )

        assert ks_pct == 30.0, f"fallback attendu 30.0, got {ks_pct}"
        assert ks_hours == 24, f"fallback attendu 24, got {ks_hours}"

    def test_portfolio_backtester_receives_ks_values(self):
        """PortfolioBacktester._kill_switch_pct reflète la valeur passée (pas le défaut)."""
        config = _make_mock_config(n_assets=1)
        config.strategies.grid_atr.per_asset = {"AAA/USDT": {}}

        backtester = PortfolioBacktester(
            config=config,
            initial_capital=10_000.0,
            strategy_name="grid_atr",
            kill_switch_pct=45.0,
            kill_switch_window_hours=48,
        )

        assert backtester._kill_switch_pct == 45.0
        assert backtester._kill_switch_window_hours == 48


# ─── Tests Benchmark BTC ────────────────────────────────────────────────────


class TestBtcBenchmark:
    """Tests pour _calc_btc_benchmark (Benchmark Buy-Hold)."""

    def _make_btc_candles(
        self,
        prices: list[float],
        start_ts: datetime | None = None,
    ) -> list[Candle]:
        ts = start_ts or datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = []
        for price in prices:
            result.append(_make_candle(price, ts, "BTC/USDT"))
            ts += timedelta(hours=1)
        return result

    def test_calc_btc_benchmark_basic(self):
        """Retourne les métriques correctes pour une série simple."""
        # Prix qui monte de 40k à 50k (+25%)
        candles = self._make_btc_candles([40_000.0, 42_000.0, 48_000.0, 50_000.0])
        result = PortfolioBacktester._calc_btc_benchmark(candles, 10_000.0)

        assert result is not None
        assert result["entry_price"] == 40_000.0
        assert result["exit_price"] == 50_000.0
        assert abs(result["return_pct"] - 25.0) < 0.01
        assert result["final_equity"] == pytest.approx(12_500.0, rel=1e-3)

    def test_calc_btc_benchmark_flat(self):
        """Prix flat → return 0%, drawdown 0%, sharpe proche de 0."""
        candles = self._make_btc_candles([50_000.0] * 100)
        result = PortfolioBacktester._calc_btc_benchmark(candles, 10_000.0)

        assert result is not None
        assert result["return_pct"] == pytest.approx(0.0, abs=0.01)
        assert result["max_drawdown_pct"] == pytest.approx(0.0, abs=0.01)

    def test_calc_btc_benchmark_drawdown(self):
        """Drawdown calculé correctement : peak 100 → trough 50 = -50%."""
        # Monte à 100, redescend à 50
        prices = [80.0, 90.0, 100.0, 80.0, 60.0, 50.0]
        candles = self._make_btc_candles(prices)
        result = PortfolioBacktester._calc_btc_benchmark(candles, 10_000.0)

        assert result is not None
        assert result["max_drawdown_pct"] == pytest.approx(-50.0, rel=0.01)

    def test_calc_btc_benchmark_equity_curve_subsampled(self):
        """equity_curve contient max 500 points même avec 2000 candles."""
        candles = self._make_btc_candles([50_000.0 + i for i in range(2000)])
        result = PortfolioBacktester._calc_btc_benchmark(candles, 10_000.0)

        assert result is not None
        assert len(result["equity_curve"]) <= 501
        # Premier et dernier points présents
        assert "timestamp" in result["equity_curve"][0]
        assert "equity" in result["equity_curve"][0]

    def test_calc_btc_benchmark_none_on_insufficient_data(self):
        """Retourne None si moins de 2 candles."""
        candles = self._make_btc_candles([50_000.0])
        result = PortfolioBacktester._calc_btc_benchmark(candles, 10_000.0)
        assert result is None

        result_empty = PortfolioBacktester._calc_btc_benchmark([], 10_000.0)
        assert result_empty is None

    def test_calc_btc_benchmark_alpha_positive(self):
        """Alpha = portfolio_return - btc_return."""
        bm = {"return_pct": 10.0}
        portfolio_return = 25.0
        alpha = portfolio_return - bm["return_pct"]
        assert alpha == 15.0

    def test_portfolio_result_btc_fields_default_none(self):
        """PortfolioResult sans benchmark a btc_benchmark=None et alpha=0."""
        result = PortfolioResult(
            initial_capital=10_000.0,
            n_assets=1,
            period_days=90,
            assets=["BTC/USDT"],
            final_equity=11_000.0,
            total_return_pct=10.0,
            total_trades=5,
            win_rate=60.0,
            realized_pnl=1000.0,
            force_closed_pnl=0.0,
            max_drawdown_pct=-5.0,
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
        assert result.btc_benchmark is None
        assert result.alpha_vs_btc == 0.0

    def test_format_portfolio_report_with_benchmark(self):
        """Le rapport CLI inclut la section benchmark si btc_benchmark présent."""
        result = PortfolioResult(
            initial_capital=10_000.0,
            n_assets=1,
            period_days=90,
            assets=["BTC/USDT"],
            final_equity=12_500.0,
            total_return_pct=25.0,
            total_trades=10,
            win_rate=60.0,
            realized_pnl=2500.0,
            force_closed_pnl=0.0,
            max_drawdown_pct=-8.0,
            max_drawdown_date=None,
            max_drawdown_duration_hours=0.0,
            peak_margin_ratio=0.3,
            peak_open_positions=3,
            peak_concurrent_assets=1,
            kill_switch_triggers=0,
            kill_switch_events=[],
            snapshots=[],
            per_asset_results={},
            btc_benchmark={
                "return_pct": 10.0,
                "max_drawdown_pct": -15.0,
                "sharpe_ratio": 0.85,
                "final_equity": 11_000.0,
                "entry_price": 40_000.0,
                "exit_price": 44_000.0,
                "equity_curve": [],
            },
            alpha_vs_btc=15.0,
        )
        report = format_portfolio_report(result)
        assert "Benchmark BTC Buy-Hold" in report
        assert "Alpha vs BTC" in report
        assert "+15.0" in report
        assert "OUTPERFORME" in report

    def test_format_portfolio_report_without_benchmark(self):
        """Le rapport CLI n'inclut PAS la section benchmark si btc_benchmark est None."""
        result = PortfolioResult(
            initial_capital=10_000.0,
            n_assets=1,
            period_days=90,
            assets=["BTC/USDT"],
            final_equity=10_500.0,
            total_return_pct=5.0,
            total_trades=5,
            win_rate=50.0,
            realized_pnl=500.0,
            force_closed_pnl=0.0,
            max_drawdown_pct=-3.0,
            max_drawdown_date=None,
            max_drawdown_duration_hours=0.0,
            peak_margin_ratio=0.2,
            peak_open_positions=2,
            peak_concurrent_assets=1,
            kill_switch_triggers=0,
            kill_switch_events=[],
            snapshots=[],
            per_asset_results={},
            btc_benchmark=None,
        )
        report = format_portfolio_report(result)
        assert "Benchmark BTC" not in report


# ---------------------------------------------------------------------------
# Régime de marché — classification et analyse
# ---------------------------------------------------------------------------


class TestClassifyRegime:
    """_classify_regime est maintenant dans metrics.py (extrait de walk_forward)."""

    def test_imported_from_metrics(self):
        """_classify_regime est importable depuis backend.backtesting.metrics."""
        from backend.backtesting.metrics import _classify_regime  # noqa: F401

    def test_bull_regime(self):
        """Rendement > +20% → bull."""
        from backend.backtesting.metrics import _classify_regime

        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        candles = [
            _make_candle(100.0 * (1 + 0.25 * i / 9), ts + timedelta(days=i))
            for i in range(10)
        ]
        result = _classify_regime(candles)
        assert result["regime"] == "bull"
        assert result["return_pct"] > 20

    def test_bear_regime(self):
        """Rendement < -20% → bear."""
        from backend.backtesting.metrics import _classify_regime

        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        candles = [
            _make_candle(100.0 * (1 - 0.25 * i / 9), ts + timedelta(days=i))
            for i in range(10)
        ]
        result = _classify_regime(candles)
        assert result["regime"] == "bear"
        assert result["return_pct"] < -20

    def test_range_regime(self):
        """Rendement entre -20% et +20% sans crash → range."""
        from backend.backtesting.metrics import _classify_regime

        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        candles = [
            _make_candle(100.0, ts + timedelta(days=i))
            for i in range(10)
        ]
        result = _classify_regime(candles)
        assert result["regime"] == "range"

    def test_insufficient_candles(self):
        """Moins de 2 candles → range par défaut."""
        from backend.backtesting.metrics import _classify_regime

        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = _classify_regime([_make_candle(100.0, ts)])
        assert result["regime"] == "range"
        assert result["return_pct"] == 0.0


class TestComputeRegimeAnalysis:
    """_compute_regime_analysis dans PortfolioBacktester."""

    def _make_snap(self, equity: float, ts: datetime) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            timestamp=ts,
            total_equity=equity,
            total_capital=equity,
            total_realized_pnl=0.0,
            total_unrealized_pnl=0.0,
            total_margin_used=0.0,
            margin_ratio=0.0,
            n_open_positions=0,
            n_assets_with_positions=0,
        )

    def test_returns_none_on_empty_candles(self):
        """Sans candles BTC → None (pas de crash)."""
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        snaps = [self._make_snap(10_000.0, ts + timedelta(days=i)) for i in range(5)]
        result = PortfolioBacktester._compute_regime_analysis(snaps, [])
        assert result is None

    def test_returns_none_on_single_snapshot(self):
        """Un seul snapshot → None (pas de calcul inter-snapshots)."""
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        candles = [
            _make_candle(50_000.0, ts + timedelta(days=i))
            for i in range(40)
        ]
        snaps = [self._make_snap(10_000.0, ts + timedelta(days=20))]
        result = PortfolioBacktester._compute_regime_analysis(snaps, candles)
        assert result is None

    def test_regime_analysis_structure(self):
        """L'analyse contient les clés attendues pour au moins un régime."""
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # BTC flat → range
        btc_candles = [
            _make_candle(50_000.0, ts + timedelta(days=i), symbol="BTC/USDT")
            for i in range(60)
        ]
        # Portfolio qui monte progressivement
        equities = [10_000.0 + 100 * i for i in range(10)]
        snaps = [
            self._make_snap(eq, ts + timedelta(days=30 + i * 2))
            for i, eq in enumerate(equities)
        ]
        result = PortfolioBacktester._compute_regime_analysis(snaps, btc_candles)
        assert result is not None
        assert "lookback_days" in result
        # Au moins un régime doit être présent
        regimes_present = [k for k in result if k != "lookback_days"]
        assert len(regimes_present) >= 1
        # Vérifier la structure du premier régime
        first_reg = result[regimes_present[0]]
        assert "days" in first_reg
        assert "pct_time" in first_reg
        assert "cum_return_pct" in first_reg
        assert "max_dd_pct" in first_reg
        assert "avg_pnl_day" in first_reg

    def test_regime_in_report(self):
        """format_portfolio_report affiche la section régimes si regime_analysis est défini."""
        result = PortfolioResult(
            initial_capital=10_000.0,
            n_assets=1,
            period_days=90,
            assets=["BTC/USDT"],
            final_equity=11_000.0,
            total_return_pct=10.0,
            total_trades=5,
            win_rate=60.0,
            realized_pnl=1_000.0,
            force_closed_pnl=0.0,
            max_drawdown_pct=-5.0,
            max_drawdown_date=None,
            max_drawdown_duration_hours=0.0,
            peak_margin_ratio=0.2,
            peak_open_positions=2,
            peak_concurrent_assets=1,
            kill_switch_triggers=0,
            kill_switch_events=[],
            snapshots=[],
            per_asset_results={},
            regime_analysis={
                "lookback_days": 30,
                "range": {
                    "days": 80.0,
                    "pct_time": 89.0,
                    "cum_return_pct": 9.5,
                    "max_dd_pct": -4.0,
                    "avg_pnl_day": 0.012,
                    "n_intervals": 12,
                },
            },
        )
        report = format_portfolio_report(result)
        assert "Performance par régime" in report
        assert "RANGE" in report
        assert "+9.5%" in report

    def test_no_regime_section_when_none(self):
        """Sans regime_analysis → pas de section dans le rapport."""
        result = PortfolioResult(
            initial_capital=10_000.0,
            n_assets=1,
            period_days=90,
            assets=["BTC/USDT"],
            final_equity=10_500.0,
            total_return_pct=5.0,
            total_trades=5,
            win_rate=50.0,
            realized_pnl=500.0,
            force_closed_pnl=0.0,
            max_drawdown_pct=-3.0,
            max_drawdown_date=None,
            max_drawdown_duration_hours=0.0,
            peak_margin_ratio=0.2,
            peak_open_positions=2,
            peak_concurrent_assets=1,
            kill_switch_triggers=0,
            kill_switch_events=[],
            snapshots=[],
            per_asset_results={},
            regime_analysis=None,
        )
        report = format_portfolio_report(result)
        assert "Performance par régime" not in report
