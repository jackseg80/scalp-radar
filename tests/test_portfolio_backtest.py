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
from backend.core.config import GridATRConfig
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
            assets=["AAA/USDT", "BBB/USDT"],
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
        assert "Par Asset" in report


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

        warmup_ends = backtester._warmup_runners(
            runners, {"AAA/USDT": candles}, engine, warmup_count=50
        )

        merged = PortfolioBacktester._merge_candles({"AAA/USDT": candles})
        snapshots, realized = await backtester._simulate(
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
