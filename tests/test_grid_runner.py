"""Tests pour GridStrategyRunner (Sprint 11 — Paper Trading Envelope DCA)."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from backend.core.grid_position_manager import GridPositionManager
from backend.core.incremental_indicators import IncrementalIndicatorEngine
from backend.core.models import Candle, Direction, MarketRegime, TimeFrame
from backend.core.position_manager import PositionManagerConfig, TradeResult
from backend.backtesting.simulator import GridStrategyRunner, RunnerStats, Simulator
from backend.strategies.base_grid import BaseGridStrategy, GridLevel, GridPosition, GridState


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_candle(
    close: float = 100_000.0,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
    volume: float = 100.0,
    ts: datetime | None = None,
    symbol: str = "BTC/USDT",
    tf: TimeFrame = TimeFrame.H1,
) -> Candle:
    o = open_ or close
    h = high or max(close, o) * 1.001
    lo = low or min(close, o) * 0.999
    return Candle(
        timestamp=ts or datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        open=o,
        high=h,
        low=lo,
        close=close,
        volume=volume,
        symbol=symbol,
        timeframe=tf,
    )


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
    name: str = "envelope_dca",
    timeframe: str = "1h",
    ma_period: int = 7,
    max_positions: int = 2,
    grid_levels: list[GridLevel] | None = None,
    close_all_result: str | None = None,
) -> MagicMock:
    """Crée un mock de BaseGridStrategy."""
    strategy = MagicMock(spec=BaseGridStrategy)
    strategy.name = name

    config = MagicMock()
    config.timeframe = timeframe
    config.ma_period = ma_period
    config.leverage = 6
    strategy._config = config

    strategy.min_candles = {"1h": 50}
    strategy.max_positions = max_positions
    strategy.compute_grid.return_value = grid_levels or []
    strategy.should_close_all.return_value = close_all_result
    strategy.get_tp_price.return_value = float("nan")
    strategy.get_sl_price.return_value = float("nan")
    strategy.get_current_conditions.return_value = []

    return strategy


def _make_mock_config() -> MagicMock:
    config = MagicMock()
    config.risk.kill_switch.max_session_loss_percent = 5.0
    config.risk.kill_switch.max_daily_loss_percent = 10.0
    config.risk.position.max_risk_per_trade_percent = 2.0
    return config


def _make_grid_runner(
    strategy=None,
    config=None,
    gpm_config=None,
) -> GridStrategyRunner:
    """Crée un GridStrategyRunner avec des mocks par défaut."""
    if strategy is None:
        strategy = _make_mock_strategy()

    if config is None:
        config = _make_mock_config()

    indicator_engine = MagicMock(spec=IncrementalIndicatorEngine)
    indicator_engine.get_indicators.return_value = {}
    indicator_engine.update = MagicMock()

    if gpm_config is None:
        gpm_config = _make_gpm_config()
    gpm = GridPositionManager(gpm_config)

    data_engine = MagicMock()
    data_engine.get_funding_rate.return_value = None
    data_engine.get_open_interest.return_value = []

    return GridStrategyRunner(
        strategy=strategy,
        config=config,
        indicator_engine=indicator_engine,
        grid_position_manager=gpm,
        data_engine=data_engine,
    )


def _fill_buffer(runner: GridStrategyRunner, symbol: str = "BTC/USDT", n: int = 10, base_close: float = 100_000.0):
    """Remplit le buffer de closes pour dépasser la SMA requise."""
    runner._close_buffer[symbol] = deque(maxlen=50)
    for i in range(n):
        runner._close_buffer[symbol].append(base_close + i * 10)


# ─── Tests GridStrategyRunner Core ────────────────────────────────────────────


class TestGridRunnerInit:
    def test_initial_state(self):
        """Le runner démarre avec capital=10k, 0 positions, compat attrs."""
        runner = _make_grid_runner()
        status = runner.get_status()
        assert status["capital"] == 10_000.0
        assert status["net_pnl"] == 0.0
        assert status["total_trades"] == 0
        assert status["has_position"] is False
        assert status["kill_switch"] is False
        assert status["open_positions"] == 0
        assert status["max_positions"] == 2

    def test_compat_attrs(self):
        """Les attributs de compatibilité duck-typing sont présents."""
        runner = _make_grid_runner()
        assert runner._position is None
        assert runner._position_symbol is None
        assert isinstance(runner._pending_events, list)

    def test_name_property(self):
        runner = _make_grid_runner()
        assert runner.name == "envelope_dca"

    def test_strategy_property(self):
        runner = _make_grid_runner()
        assert runner.strategy is not None


class TestGridRunnerOnCandle:
    @pytest.mark.asyncio
    async def test_wrong_timeframe_ignored(self):
        """Les bougies d'un timeframe non-stratégie sont ignorées."""
        runner = _make_grid_runner()
        _fill_buffer(runner)
        candle = _make_candle(tf=TimeFrame.M5)  # Stratégie attend 1h

        await runner.on_candle("BTC/USDT", "5m", candle)

        # Aucune position ouverte
        assert len(runner._positions) == 0

    @pytest.mark.asyncio
    async def test_not_enough_data_skips(self):
        """Pas assez de closes dans le buffer → pas de signal."""
        runner = _make_grid_runner()
        # Buffer vide, stratégie demande ma_period=7
        candle = _make_candle()

        await runner.on_candle("BTC/USDT", "1h", candle)
        # Seulement 1 close dans le buffer, il faut 7
        assert len(runner._positions) == 0

    @pytest.mark.asyncio
    async def test_level_touched_opens_position(self):
        """Un niveau touché ouvre une position."""
        strategy = _make_mock_strategy()
        # Retourner un niveau LONG à 95000
        strategy.compute_grid.return_value = [
            GridLevel(index=0, entry_price=95_000.0, direction=Direction.LONG, size_fraction=0.5),
        ]
        runner = _make_grid_runner(strategy=strategy)
        _fill_buffer(runner, n=10, base_close=100_000.0)

        # Bougie dont le low touche le niveau
        candle = _make_candle(close=96_000.0, low=94_500.0, high=97_000.0)
        await runner.on_candle("BTC/USDT", "1h", candle)

        assert len(runner._positions) == 1
        assert runner._positions[0].entry_price == 95_000.0
        assert runner._positions[0].direction == Direction.LONG
        assert runner._grid_symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_two_levels_progressively(self):
        """Deux niveaux touchés sur deux bougies différentes → 2 positions."""
        strategy = _make_mock_strategy(max_positions=3)
        runner = _make_grid_runner(strategy=strategy)
        _fill_buffer(runner, n=10, base_close=100_000.0)

        # Première bougie : touche niveau 0
        strategy.compute_grid.return_value = [
            GridLevel(index=0, entry_price=95_000.0, direction=Direction.LONG, size_fraction=0.33),
            GridLevel(index=1, entry_price=92_000.0, direction=Direction.LONG, size_fraction=0.33),
        ]
        candle1 = _make_candle(
            close=96_000.0, low=94_500.0, high=97_000.0,
            ts=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        )
        await runner.on_candle("BTC/USDT", "1h", candle1)
        assert len(runner._positions) == 1

        # Deuxième bougie : touche niveau 1
        strategy.compute_grid.return_value = [
            GridLevel(index=1, entry_price=92_000.0, direction=Direction.LONG, size_fraction=0.33),
        ]
        candle2 = _make_candle(
            close=93_000.0, low=91_500.0, high=94_000.0,
            ts=datetime(2024, 6, 15, 13, 0, tzinfo=timezone.utc),
        )
        await runner.on_candle("BTC/USDT", "1h", candle2)
        assert len(runner._positions) == 2

    @pytest.mark.asyncio
    async def test_tp_global_closes_all(self):
        """TP global → fermeture de toutes les positions, trade enregistré."""
        strategy = _make_mock_strategy()
        runner = _make_grid_runner(strategy=strategy)
        _fill_buffer(runner, n=10, base_close=100_000.0)

        # Pré-remplir une position
        runner._positions = [
            GridPosition(
                level=0, direction=Direction.LONG, entry_price=95_000.0,
                quantity=0.01, entry_time=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
                entry_fee=0.57,
            ),
        ]
        runner._grid_symbol = "BTC/USDT"

        # TP = SMA ~ 100045, le high de la bougie dépasse
        strategy.get_tp_price.return_value = 100_000.0
        strategy.get_sl_price.return_value = 80_000.0

        candle = _make_candle(close=101_000.0, high=101_500.0, low=99_000.0)
        await runner.on_candle("BTC/USDT", "1h", candle)

        assert len(runner._positions) == 0
        assert runner._stats.total_trades == 1
        assert runner._stats.net_pnl != 0.0
        assert runner._grid_symbol is None

    @pytest.mark.asyncio
    async def test_sl_global_closes_all(self):
        """SL global → fermeture de toutes les positions."""
        strategy = _make_mock_strategy()
        runner = _make_grid_runner(strategy=strategy)
        _fill_buffer(runner, n=10, base_close=100_000.0)

        runner._positions = [
            GridPosition(
                level=0, direction=Direction.LONG, entry_price=95_000.0,
                quantity=0.01, entry_time=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
                entry_fee=0.57,
            ),
        ]
        runner._grid_symbol = "BTC/USDT"

        strategy.get_tp_price.return_value = 100_000.0
        strategy.get_sl_price.return_value = 90_000.0  # SL à 90k

        # Bougie dont le low touche le SL
        candle = _make_candle(close=89_000.0, low=88_000.0, high=92_000.0)
        await runner.on_candle("BTC/USDT", "1h", candle)

        assert len(runner._positions) == 0
        assert runner._stats.total_trades == 1

    @pytest.mark.asyncio
    async def test_kill_switch_blocks_new_positions(self):
        """Après kill switch, on_candle ne fait rien."""
        runner = _make_grid_runner()
        runner._kill_switch_triggered = True
        _fill_buffer(runner, n=10)

        candle = _make_candle()
        await runner.on_candle("BTC/USDT", "1h", candle)
        assert len(runner._positions) == 0

    @pytest.mark.asyncio
    async def test_kill_switch_triggered_on_big_loss(self):
        """Le kill switch se déclenche si la perte dépasse le seuil."""
        strategy = _make_mock_strategy()
        runner = _make_grid_runner(strategy=strategy)

        # Simuler une grosse perte (> 5% de 10k)
        trade = TradeResult(
            direction=Direction.LONG,
            entry_price=95_000.0,
            exit_price=85_000.0,
            quantity=0.01,
            entry_time=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
            exit_time=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
            gross_pnl=-600.0,  # Perte > 5% de 10k = 500
            fee_cost=5.0,
            slippage_cost=2.0,
            net_pnl=-607.0,
            exit_reason="sl_global",
            market_regime=MarketRegime.RANGING,
        )
        runner._record_trade(trade, "BTC/USDT")

        assert runner._kill_switch_triggered is True
        assert runner._stats.is_active is False

    @pytest.mark.asyncio
    async def test_no_fee_deduction_on_open(self):
        """L'ouverture d'une position ne déduit PAS les fees du capital
        (les fees sont incluses dans le net_pnl à la fermeture)."""
        strategy = _make_mock_strategy()
        strategy.compute_grid.return_value = [
            GridLevel(index=0, entry_price=95_000.0, direction=Direction.LONG, size_fraction=0.5),
        ]
        runner = _make_grid_runner(strategy=strategy)
        _fill_buffer(runner, n=10, base_close=100_000.0)

        capital_before = runner._capital
        candle = _make_candle(close=96_000.0, low=94_500.0, high=97_000.0)
        await runner.on_candle("BTC/USDT", "1h", candle)

        # Capital ne change PAS à l'ouverture
        assert runner._capital == capital_before

    @pytest.mark.asyncio
    async def test_pending_events_trade_event_format(self):
        """Les events émis sont des TradeEvent, pas des dicts."""
        strategy = _make_mock_strategy()
        strategy.compute_grid.return_value = [
            GridLevel(index=0, entry_price=95_000.0, direction=Direction.LONG, size_fraction=0.5),
        ]
        runner = _make_grid_runner(strategy=strategy)
        _fill_buffer(runner, n=10, base_close=100_000.0)

        candle = _make_candle(close=96_000.0, low=94_500.0, high=97_000.0)
        await runner.on_candle("BTC/USDT", "1h", candle)

        assert len(runner._pending_events) == 1
        event = runner._pending_events[0]
        # Vérifier que c'est un TradeEvent (pas un dict)
        from backend.execution.executor import TradeEvent
        assert isinstance(event, TradeEvent)
        assert event.event_type.value == "open"
        assert event.symbol == "BTC/USDT"


# ─── Tests State Persistence ─────────────────────────────────────────────────


class TestGridRunnerState:
    def test_restore_state_basic(self):
        """restore_state restaure capital, stats, kill_switch."""
        runner = _make_grid_runner()
        state = {
            "capital": 9_500.0,
            "net_pnl": -500.0,
            "total_trades": 3,
            "wins": 1,
            "losses": 2,
            "kill_switch": False,
            "is_active": True,
        }
        runner.restore_state(state)

        assert runner._capital == 9_500.0
        assert runner._stats.net_pnl == -500.0
        assert runner._stats.total_trades == 3
        assert runner._stats.wins == 1
        assert runner._stats.losses == 2

    def test_restore_state_with_grid_positions(self):
        """restore_state restaure les positions grid ouvertes."""
        runner = _make_grid_runner()
        state = {
            "capital": 9_800.0,
            "net_pnl": -200.0,
            "total_trades": 1,
            "wins": 0,
            "losses": 1,
            "kill_switch": False,
            "is_active": True,
            "grid_positions": [
                {
                    "level": 0,
                    "direction": "LONG",
                    "entry_price": 95_000.0,
                    "quantity": 0.01,
                    "entry_time": "2024-06-15T10:00:00+00:00",
                    "entry_fee": 0.57,
                },
                {
                    "level": 1,
                    "direction": "LONG",
                    "entry_price": 92_000.0,
                    "quantity": 0.01,
                    "entry_time": "2024-06-15T11:00:00+00:00",
                    "entry_fee": 0.55,
                },
            ],
            "grid_symbol": "BTC/USDT",
        }
        runner.restore_state(state)

        assert len(runner._positions) == 2
        assert runner._positions[0].level == 0
        assert runner._positions[0].direction == Direction.LONG
        assert runner._positions[0].entry_price == 95_000.0
        assert runner._positions[1].level == 1
        assert runner._grid_symbol == "BTC/USDT"

    def test_restore_state_no_grid_positions_backward_compat(self):
        """Sans grid_positions dans le state → positions vide."""
        runner = _make_grid_runner()
        state = {
            "capital": 10_000.0,
            "net_pnl": 0.0,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "kill_switch": False,
            "is_active": True,
        }
        runner.restore_state(state)
        assert len(runner._positions) == 0

    def test_get_status_format(self):
        """get_status retourne les champs grid spécifiques."""
        runner = _make_grid_runner()
        runner._positions = [
            GridPosition(
                level=0, direction=Direction.LONG, entry_price=95_000.0,
                quantity=0.01, entry_time=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
                entry_fee=0.57,
            ),
        ]
        status = runner.get_status()
        assert status["open_positions"] == 1
        assert status["max_positions"] == 2
        assert status["has_position"] is True
        assert status["avg_entry_price"] == 95_000.0

    def test_get_trades_returns_closed(self):
        """get_trades retourne les trades clôturés."""
        runner = _make_grid_runner()
        trade = TradeResult(
            direction=Direction.LONG,
            entry_price=95_000.0,
            exit_price=100_000.0,
            quantity=0.01,
            entry_time=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
            exit_time=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
            gross_pnl=50.0,
            fee_cost=2.0,
            slippage_cost=1.0,
            net_pnl=47.0,
            exit_reason="tp_global",
            market_regime=MarketRegime.RANGING,
        )
        runner._trades.append(("BTC/USDT", trade))

        trades = runner.get_trades()
        assert len(trades) == 1
        assert trades[0][0] == "BTC/USDT"
        assert trades[0][1].net_pnl == 47.0


# ─── Tests Dashboard ─────────────────────────────────────────────────────────


class TestGridRunnerDashboard:
    def test_build_context_with_sma(self):
        """build_context retourne un StrategyContext avec SMA mergée."""
        runner = _make_grid_runner()
        _fill_buffer(runner, n=10, base_close=100_000.0)

        ctx = runner.build_context("BTC/USDT")
        assert ctx is not None
        indicators_1h = ctx.indicators.get("1h", {})
        assert "sma" in indicators_1h
        assert "close" in indicators_1h
        assert indicators_1h["sma"] > 0

    def test_get_grid_positions_format(self):
        """get_grid_positions retourne le bon format avec symbole."""
        runner = _make_grid_runner()
        runner._grid_symbol = "BTC/USDT"
        runner._positions = [
            GridPosition(
                level=0, direction=Direction.LONG, entry_price=95_000.0,
                quantity=0.01, entry_time=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
                entry_fee=0.57,
            ),
        ]

        gp = runner.get_grid_positions()
        assert len(gp) == 1
        assert gp[0]["symbol"] == "BTC/USDT"
        assert gp[0]["strategy"] == "envelope_dca"
        assert gp[0]["direction"] == "LONG"
        assert gp[0]["level"] == 0
        assert gp[0]["type"] == "grid"


# ─── Tests Simulator Integration ─────────────────────────────────────────────


class TestSimulatorGridIntegration:
    @pytest.mark.asyncio
    async def test_simulator_creates_grid_runner_for_envelope_dca(self):
        """Le Simulator crée un GridStrategyRunner pour envelope_dca."""
        config = MagicMock()
        config.risk.position.default_leverage = 15
        config.risk.fees.maker_percent = 0.02
        config.risk.fees.taker_percent = 0.06
        config.risk.slippage.default_estimate_percent = 0.05
        config.risk.slippage.high_volatility_multiplier = 2.0
        config.risk.position.max_risk_per_trade_percent = 2.0

        engine = MagicMock()
        engine.on_candle = MagicMock()
        engine.get_all_symbols.return_value = ["BTC/USDT"]

        # Mock get_enabled_strategies pour retourner une grid strategy
        mock_strategy = _make_mock_strategy()

        with patch("backend.backtesting.simulator.get_enabled_strategies", return_value=[mock_strategy]):
            with patch("backend.backtesting.simulator.is_grid_strategy", return_value=True):
                sim = Simulator(data_engine=engine, config=config)
                await sim.start()

        assert len(sim._runners) == 1
        assert isinstance(sim._runners[0], GridStrategyRunner)

    @pytest.mark.asyncio
    async def test_simulator_creates_live_runner_for_normal_strategy(self):
        """Le Simulator crée un LiveStrategyRunner pour une stratégie normale."""
        from backend.backtesting.simulator import LiveStrategyRunner

        config = MagicMock()
        config.risk.position.default_leverage = 15
        config.risk.fees.maker_percent = 0.02
        config.risk.fees.taker_percent = 0.06
        config.risk.slippage.default_estimate_percent = 0.05
        config.risk.slippage.high_volatility_multiplier = 2.0
        config.risk.position.max_risk_per_trade_percent = 2.0

        engine = MagicMock()
        engine.on_candle = MagicMock()
        engine.get_all_symbols.return_value = ["BTC/USDT"]

        mock_strategy = MagicMock()
        mock_strategy.name = "vwap_rsi"
        mock_strategy.min_candles = {"5m": 50}

        with patch("backend.backtesting.simulator.get_enabled_strategies", return_value=[mock_strategy]):
            with patch("backend.backtesting.simulator.is_grid_strategy", return_value=False):
                sim = Simulator(data_engine=engine, config=config)
                await sim.start()

        assert len(sim._runners) == 1
        assert isinstance(sim._runners[0], LiveStrategyRunner)

    @pytest.mark.asyncio
    async def test_get_all_status_includes_grid_runner(self):
        """get_all_status inclut le runner grid."""
        config = MagicMock()
        config.risk.position.default_leverage = 15
        config.risk.fees.maker_percent = 0.02
        config.risk.fees.taker_percent = 0.06
        config.risk.slippage.default_estimate_percent = 0.05
        config.risk.slippage.high_volatility_multiplier = 2.0
        config.risk.position.max_risk_per_trade_percent = 2.0

        engine = MagicMock()
        engine.on_candle = MagicMock()
        engine.get_all_symbols.return_value = ["BTC/USDT"]

        mock_strategy = _make_mock_strategy()

        with patch("backend.backtesting.simulator.get_enabled_strategies", return_value=[mock_strategy]):
            with patch("backend.backtesting.simulator.is_grid_strategy", return_value=True):
                sim = Simulator(data_engine=engine, config=config)
                await sim.start()

        statuses = sim.get_all_status()
        assert "envelope_dca" in statuses
        assert statuses["envelope_dca"]["open_positions"] == 0

    @pytest.mark.asyncio
    async def test_get_open_positions_returns_grid_positions(self):
        """get_open_positions retourne les positions grid."""
        config = MagicMock()
        config.risk.position.default_leverage = 15
        config.risk.fees.maker_percent = 0.02
        config.risk.fees.taker_percent = 0.06
        config.risk.slippage.default_estimate_percent = 0.05
        config.risk.slippage.high_volatility_multiplier = 2.0
        config.risk.position.max_risk_per_trade_percent = 2.0

        engine = MagicMock()
        engine.on_candle = MagicMock()
        engine.get_all_symbols.return_value = ["BTC/USDT"]

        mock_strategy = _make_mock_strategy()

        with patch("backend.backtesting.simulator.get_enabled_strategies", return_value=[mock_strategy]):
            with patch("backend.backtesting.simulator.is_grid_strategy", return_value=True):
                sim = Simulator(data_engine=engine, config=config)
                await sim.start()

        # Ajouter manuellement une position grid
        grid_runner = sim._runners[0]
        assert isinstance(grid_runner, GridStrategyRunner)
        grid_runner._positions = [
            GridPosition(
                level=0, direction=Direction.LONG, entry_price=95_000.0,
                quantity=0.01, entry_time=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
                entry_fee=0.57,
            ),
        ]
        grid_runner._grid_symbol = "BTC/USDT"

        positions = sim.get_open_positions()
        assert len(positions) == 1
        assert positions[0]["type"] == "grid"
        assert positions[0]["symbol"] == "BTC/USDT"


# ─── Tests Warm-up ───────────────────────────────────────────────────────────


class TestGridRunnerWarmup:
    @pytest.mark.asyncio
    async def test_warmup_fills_buffer(self):
        """_warmup_from_db charge les bougies et remplit le buffer."""
        runner = _make_grid_runner()
        db = AsyncMock()
        candles = [
            _make_candle(
                close=95_000.0 + i * 100,
                ts=datetime(2024, 6, 15, i, 0, tzinfo=timezone.utc),
            )
            for i in range(20)
        ]
        db.get_recent_candles = AsyncMock(return_value=candles)

        await runner._warmup_from_db(db, "BTC/USDT")

        assert len(runner._close_buffer["BTC/USDT"]) == 20
        # Vérifier que l'indicator engine a été alimenté
        assert runner._indicator_engine.update.call_count == 20

    @pytest.mark.asyncio
    async def test_warmup_empty_db(self):
        """Pas de données en DB → buffer vide, pas de crash."""
        runner = _make_grid_runner()
        db = AsyncMock()
        db.get_recent_candles = AsyncMock(return_value=[])

        await runner._warmup_from_db(db, "BTC/USDT")

        assert "BTC/USDT" not in runner._close_buffer


# ─── Tests Database get_recent_candles ────────────────────────────────────────


class TestDatabaseGetRecentCandles:
    @pytest.mark.asyncio
    async def test_get_recent_candles(self, tmp_path):
        """get_recent_candles retourne les N dernières bougies en ASC."""
        from backend.core.database import Database
        from backend.core.models import TimeFrame

        db = Database(db_path=str(tmp_path / "test.db"))
        await db.init()

        # Insérer 20 bougies 1h
        candles = []
        for i in range(20):
            candles.append(Candle(
                timestamp=datetime(2024, 6, 15, i, 0, tzinfo=timezone.utc),
                open=95_000.0 + i * 100,
                high=95_100.0 + i * 100,
                low=94_900.0 + i * 100,
                close=95_050.0 + i * 100,
                volume=100.0,
                symbol="BTC/USDT",
                timeframe=TimeFrame.H1,
                exchange="binance",
            ))
        await db.insert_candles_batch(candles)

        # Récupérer les 10 dernières
        result = await db.get_recent_candles("BTC/USDT", "1h", limit=10)

        assert len(result) == 10
        # Vérifier l'ordre ASC
        assert result[0].timestamp < result[-1].timestamp
        # Vérifier que c'est bien les 10 dernières
        assert result[-1].close == candles[-1].close
        assert result[0].close == candles[10].close

        await db.close()
