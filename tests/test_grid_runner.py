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

    runner = GridStrategyRunner(
        strategy=strategy,
        config=config,
        indicator_engine=indicator_engine,
        grid_position_manager=gpm,
        data_engine=data_engine,
    )
    # Par défaut les tests sont en mode live (pas warm-up)
    runner._is_warming_up = False
    return runner


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
        assert len(runner._positions.get("BTC/USDT", [])) == 0

    @pytest.mark.asyncio
    async def test_not_enough_data_skips(self):
        """Pas assez de closes dans le buffer → pas de signal."""
        runner = _make_grid_runner()
        # Buffer vide, stratégie demande ma_period=7
        candle = _make_candle()

        await runner.on_candle("BTC/USDT", "1h", candle)
        # Seulement 1 close dans le buffer, il faut 7
        assert len(runner._positions.get("BTC/USDT", [])) == 0

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

        positions = runner._positions.get("BTC/USDT", [])
        assert len(positions) == 1
        assert positions[0].entry_price == 95_000.0
        assert positions[0].direction == Direction.LONG
        assert "BTC/USDT" in runner._positions

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
        assert len(runner._positions.get("BTC/USDT", [])) == 1

        # Deuxième bougie : touche niveau 1
        strategy.compute_grid.return_value = [
            GridLevel(index=1, entry_price=92_000.0, direction=Direction.LONG, size_fraction=0.33),
        ]
        candle2 = _make_candle(
            close=93_000.0, low=91_500.0, high=94_000.0,
            ts=datetime(2024, 6, 15, 13, 0, tzinfo=timezone.utc),
        )
        await runner.on_candle("BTC/USDT", "1h", candle2)
        assert len(runner._positions.get("BTC/USDT", [])) == 2

    @pytest.mark.asyncio
    async def test_tp_global_closes_all(self):
        """TP global → fermeture de toutes les positions, trade enregistré."""
        strategy = _make_mock_strategy()
        runner = _make_grid_runner(strategy=strategy)
        _fill_buffer(runner, n=10, base_close=100_000.0)

        # Pré-remplir une position
        runner._positions["BTC/USDT"] = [
            GridPosition(
                level=0, direction=Direction.LONG, entry_price=95_000.0,
                quantity=0.01, entry_time=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
                entry_fee=0.57,
            ),
        ]

        # TP = SMA ~ 100045, le high de la bougie dépasse
        strategy.get_tp_price.return_value = 100_000.0
        strategy.get_sl_price.return_value = 80_000.0

        candle = _make_candle(close=101_000.0, high=101_500.0, low=99_000.0)
        await runner.on_candle("BTC/USDT", "1h", candle)

        assert len(runner._positions.get("BTC/USDT", [])) == 0
        assert runner._stats.total_trades == 1
        assert runner._stats.net_pnl != 0.0

    @pytest.mark.asyncio
    async def test_sl_global_closes_all(self):
        """SL global → fermeture de toutes les positions."""
        strategy = _make_mock_strategy()
        runner = _make_grid_runner(strategy=strategy)
        _fill_buffer(runner, n=10, base_close=100_000.0)

        runner._positions["BTC/USDT"] = [
            GridPosition(
                level=0, direction=Direction.LONG, entry_price=95_000.0,
                quantity=0.01, entry_time=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
                entry_fee=0.57,
            ),
        ]

        strategy.get_tp_price.return_value = 100_000.0
        strategy.get_sl_price.return_value = 90_000.0  # SL à 90k

        # Bougie dont le low touche le SL
        candle = _make_candle(close=89_000.0, low=88_000.0, high=92_000.0)
        await runner.on_candle("BTC/USDT", "1h", candle)

        assert len(runner._positions.get("BTC/USDT", [])) == 0
        assert runner._stats.total_trades == 1

    @pytest.mark.asyncio
    async def test_kill_switch_blocks_new_positions(self):
        """Après kill switch, on_candle ne fait rien."""
        runner = _make_grid_runner()
        runner._kill_switch_triggered = True
        _fill_buffer(runner, n=10)

        candle = _make_candle()
        await runner.on_candle("BTC/USDT", "1h", candle)
        assert len(runner._positions.get("BTC/USDT", [])) == 0

    @pytest.mark.asyncio
    async def test_kill_switch_not_triggered_on_big_loss(self):
        """Le kill switch NE se déclenche PAS pour les grid/DCA (pertes temporaires normales)."""
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

        # Grid/DCA : pas de kill switch (protection par SL individuel)
        assert runner._kill_switch_triggered is False

    @pytest.mark.asyncio
    async def test_capital_decremented_on_open(self):
        """L'ouverture d'une position réserve la marge."""
        strategy = _make_mock_strategy()
        strategy.compute_grid.return_value = [
            GridLevel(index=0, entry_price=95_000.0, direction=Direction.LONG, size_fraction=0.5),
        ]
        runner = _make_grid_runner(strategy=strategy)
        _fill_buffer(runner, n=10, base_close=100_000.0)

        capital_before = runner._capital
        candle = _make_candle(close=96_000.0, low=94_500.0, high=97_000.0)
        await runner.on_candle("BTC/USDT", "1h", candle)

        # Capital doit DIMINUER après ouverture (marge réservée)
        assert runner._capital < capital_before
        positions = runner._positions.get("BTC/USDT", [])
        assert len(positions) == 1
        pos = positions[0]
        notional = pos.entry_price * pos.quantity
        margin = notional / 6  # leverage=6
        expected = capital_before - margin
        assert abs(runner._capital - expected) < 0.01

    @pytest.mark.asyncio
    async def test_grid_capital_restored_on_close(self):
        """Après fermeture, capital = initial + net_pnl (marge rendue)."""
        strategy = _make_mock_strategy()
        strategy.compute_grid.return_value = [
            GridLevel(index=0, entry_price=95_000.0, direction=Direction.LONG, size_fraction=0.5),
        ]
        runner = _make_grid_runner(strategy=strategy)
        _fill_buffer(runner, n=10, base_close=100_000.0)

        initial_capital = runner._capital

        # Ouvrir un niveau
        candle1 = _make_candle(
            close=96_000.0, low=94_500.0, high=97_000.0,
            ts=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        )
        await runner.on_candle("BTC/USDT", "1h", candle1)
        assert len(runner._positions.get("BTC/USDT", [])) == 1
        capital_after_open = runner._capital
        assert capital_after_open < initial_capital

        # Fermer via TP global
        strategy.compute_grid.return_value = []
        strategy.get_tp_price.return_value = 100_000.0
        strategy.get_sl_price.return_value = 80_000.0

        candle2 = _make_candle(
            close=101_000.0, high=101_500.0, low=99_000.0,
            ts=datetime(2024, 6, 15, 13, 0, tzinfo=timezone.utc),
        )
        await runner.on_candle("BTC/USDT", "1h", candle2)

        # Toutes les positions fermées
        assert len(runner._positions.get("BTC/USDT", [])) == 0
        assert runner._stats.total_trades == 1
        # Capital = initial + net_pnl (marge entièrement rendue)
        expected = initial_capital + runner._stats.net_pnl
        assert abs(runner._capital - expected) < 0.01

    @pytest.mark.asyncio
    async def test_grid_no_overflow_after_100_cycles(self):
        """100 cycles open/close near-breakeven, capital reste raisonnable."""
        strategy = _make_mock_strategy(max_positions=2)
        runner = _make_grid_runner(strategy=strategy)
        _fill_buffer(runner, n=10, base_close=100_000.0)

        initial_capital = runner._capital

        for cycle in range(100):
            ts_open = datetime(2024, 6, 15, cycle % 24, 0, tzinfo=timezone.utc)
            ts_close = datetime(2024, 6, 15, cycle % 24, 30, tzinfo=timezone.utc)

            # Ouvrir 2 niveaux
            strategy.compute_grid.return_value = [
                GridLevel(index=0, entry_price=95_000.0, direction=Direction.LONG, size_fraction=0.5),
                GridLevel(index=1, entry_price=93_000.0, direction=Direction.LONG, size_fraction=0.5),
            ]
            strategy.get_tp_price.return_value = float("nan")
            strategy.get_sl_price.return_value = float("nan")
            strategy.should_close_all.return_value = None

            candle_open = _make_candle(
                close=94_000.0, low=92_500.0, high=96_000.0, ts=ts_open,
            )
            await runner.on_candle("BTC/USDT", "1h", candle_open)

            # Fermer near-breakeven (exit ≈ avg_entry → fees seules)
            strategy.should_close_all.return_value = "tp_global"
            candle_close = _make_candle(
                close=94_500.0, low=94_000.0, high=95_000.0, ts=ts_close,
            )
            await runner.on_candle("BTC/USDT", "1h", candle_close)

        # Near-breakeven : capital ne doit pas exploser (< 2× initial)
        assert runner._capital < initial_capital * 2
        # Et ne doit pas être négatif
        assert runner._capital > 0

    @pytest.mark.asyncio
    async def test_grid_zero_capital_skips_level(self):
        """Avec capital=0, aucune position n'est ouverte."""
        strategy = _make_mock_strategy(max_positions=4)
        runner = _make_grid_runner(strategy=strategy)
        _fill_buffer(runner, n=10, base_close=100_000.0)

        runner._capital = 0.0

        strategy.compute_grid.return_value = [
            GridLevel(index=0, entry_price=95_000.0, direction=Direction.LONG, size_fraction=0.25),
        ]
        candle = _make_candle(close=96_000.0, low=94_500.0, high=97_000.0)
        await runner.on_candle("BTC/USDT", "1h", candle)

        # Pas de position ouverte (GPM rejette capital <= 0)
        assert len(runner._positions.get("BTC/USDT", [])) == 0
        assert runner._capital == 0.0

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
                    "symbol": "BTC/USDT",
                    "level": 0,
                    "direction": "LONG",
                    "entry_price": 95_000.0,
                    "quantity": 0.01,
                    "entry_time": "2024-06-15T10:00:00+00:00",
                    "entry_fee": 0.57,
                },
                {
                    "symbol": "BTC/USDT",
                    "level": 1,
                    "direction": "LONG",
                    "entry_price": 92_000.0,
                    "quantity": 0.01,
                    "entry_time": "2024-06-15T11:00:00+00:00",
                    "entry_fee": 0.55,
                },
            ],
        }
        runner.restore_state(state)

        positions = runner._positions.get("BTC/USDT", [])
        assert len(positions) == 2
        assert positions[0].level == 0
        assert positions[0].direction == Direction.LONG
        assert positions[0].entry_price == 95_000.0
        assert positions[1].level == 1
        assert "BTC/USDT" in runner._positions

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
        assert runner._positions == {}

    def test_get_status_format(self):
        """get_status retourne les champs grid spécifiques."""
        runner = _make_grid_runner()
        runner._positions["BTC/USDT"] = [
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
        runner._positions["BTC/USDT"] = [
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
        grid_runner._positions["BTC/USDT"] = [
            GridPosition(
                level=0, direction=Direction.LONG, entry_price=95_000.0,
                quantity=0.01, entry_time=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
                entry_fee=0.57,
            ),
        ]

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

    @pytest.mark.asyncio
    async def test_warmup_capped_at_max(self):
        """_warmup_from_db plafonne le nombre de candles à MAX_WARMUP_CANDLES."""
        runner = _make_grid_runner()
        runner._is_warming_up = True
        db = AsyncMock()
        db.get_recent_candles = AsyncMock(return_value=[])

        await runner._warmup_from_db(db, "BTC/USDT")

        # Vérifie que le limit passé à get_recent_candles <= MAX_WARMUP_CANDLES
        call_args = db.get_recent_candles.call_args
        limit_arg = call_args[0][2]  # 3ème argument positionnel = limit
        assert limit_arg <= GridStrategyRunner.MAX_WARMUP_CANDLES

    @pytest.mark.asyncio
    async def test_warmup_capital_not_modified(self):
        """Pendant le warm-up, le capital ne bouge pas malgré les trades."""
        strategy = _make_mock_strategy()
        strategy.compute_grid.return_value = [
            GridLevel(index=0, entry_price=95_000.0, direction=Direction.LONG, size_fraction=0.5),
        ]
        runner = _make_grid_runner(strategy=strategy)
        runner._is_warming_up = True
        _fill_buffer(runner, n=10, base_close=100_000.0)

        initial_capital = runner._capital

        # Candle ancienne (warm-up) qui touche le niveau
        candle = _make_candle(
            close=96_000.0, low=94_500.0, high=97_000.0,
            ts=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        )
        await runner.on_candle("BTC/USDT", "1h", candle)

        # Position ouverte mais capital inchangé
        assert len(runner._positions.get("BTC/USDT", [])) == 1
        assert runner._capital == initial_capital

    @pytest.mark.asyncio
    async def test_warmup_trade_recorded_without_stats(self):
        """Pendant le warm-up, les trades sont enregistrés mais les stats ne bougent pas."""
        strategy = _make_mock_strategy()
        runner = _make_grid_runner(strategy=strategy)
        runner._is_warming_up = True
        _fill_buffer(runner, n=10, base_close=100_000.0)

        # Pré-remplir une position warm-up
        runner._positions["BTC/USDT"] = [
            GridPosition(
                level=0, direction=Direction.LONG, entry_price=95_000.0,
                quantity=0.01, entry_time=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
                entry_fee=0.57,
            ),
        ]

        strategy.get_tp_price.return_value = 100_000.0
        strategy.get_sl_price.return_value = 80_000.0

        candle = _make_candle(
            close=101_000.0, high=101_500.0, low=99_000.0,
            ts=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        )
        await runner.on_candle("BTC/USDT", "1h", candle)

        # Trade en historique
        assert len(runner._trades) == 1
        # Stats non modifiées (warm-up)
        assert runner._stats.total_trades == 0
        assert runner._stats.net_pnl == 0.0

    @pytest.mark.asyncio
    async def test_warmup_ends_on_recent_candle(self):
        """Le warm-up se termine quand une candle récente arrive."""
        from datetime import timedelta as td

        strategy = _make_mock_strategy()
        runner = _make_grid_runner(strategy=strategy)
        runner._is_warming_up = True
        _fill_buffer(runner, n=10, base_close=100_000.0)

        assert runner._is_warming_up is True

        # Candle récente (< 2h) → fin du warm-up
        now = datetime.now(tz=timezone.utc)
        candle = _make_candle(
            close=100_000.0,
            ts=now - td(minutes=30),
        )
        await runner.on_candle("BTC/USDT", "1h", candle)

        assert runner._is_warming_up is False

    @pytest.mark.asyncio
    async def test_end_warmup_resets_capital_and_stats(self):
        """_end_warmup() reset capital, stats et positions mais garde les trades."""
        runner = _make_grid_runner()
        runner._is_warming_up = True

        # Simuler des trades et positions warm-up
        runner._capital = 50_000.0  # Capital gonflé par warm-up
        runner._realized_pnl = 40_000.0
        runner._stats.total_trades = 100
        runner._stats.wins = 60
        runner._stats.losses = 40
        runner._trades = [("BTC/USDT", MagicMock())] * 5
        runner._positions["BTC/USDT"] = [MagicMock()]

        runner._end_warmup()

        assert runner._is_warming_up is False
        assert runner._capital == 10_000.0
        assert runner._realized_pnl == 0.0
        assert runner._stats.total_trades == 0
        assert runner._stats.wins == 0
        assert runner._stats.losses == 0
        # Trades conservés en historique
        assert len(runner._trades) == 5
        # Positions fermées
        assert len(runner._positions) == 0

    @pytest.mark.asyncio
    async def test_warmup_no_executor_events(self):
        """Pendant le warm-up, aucun événement Executor n'est émis."""
        strategy = _make_mock_strategy()
        strategy.compute_grid.return_value = [
            GridLevel(index=0, entry_price=95_000.0, direction=Direction.LONG, size_fraction=0.5),
        ]
        runner = _make_grid_runner(strategy=strategy)
        runner._is_warming_up = True
        _fill_buffer(runner, n=10, base_close=100_000.0)

        candle = _make_candle(
            close=96_000.0, low=94_500.0, high=97_000.0,
            ts=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        )
        await runner.on_candle("BTC/USDT", "1h", candle)

        # Pas d'événements Executor pendant warm-up
        assert len(runner._pending_events) == 0

    @pytest.mark.asyncio
    async def test_restore_state_disables_warmup(self):
        """Restaurer un état sauvegardé désactive le warm-up."""
        runner = _make_grid_runner()
        runner._is_warming_up = True

        runner.restore_state({
            "capital": 12_000.0,
            "kill_switch": False,
            "realized_pnl": 2_000.0,
            "total_trades": 5,
            "wins": 3,
            "losses": 2,
        })

        assert runner._is_warming_up is False
        assert runner._capital == 12_000.0

    @pytest.mark.asyncio
    async def test_restore_state_ignores_kill_switch(self):
        """Même si le state sauvegardé a kill_switch=True, le grid l'ignore."""
        runner = _make_grid_runner()

        runner.restore_state({
            "capital": 8_000.0,
            "kill_switch": True,
            "realized_pnl": -2_000.0,
            "total_trades": 3,
            "wins": 0,
            "losses": 3,
        })

        assert runner._kill_switch_triggered is False
        assert runner._capital == 8_000.0


class TestGridKillSwitchDisabled:
    def test_grid_no_kill_switch_on_large_loss(self):
        """Un grid runner ne déclenche pas le kill switch après une grosse perte."""
        runner = _make_grid_runner()

        # Simuler un trade avec perte > 5% du capital (seuil = 5%)
        trade = TradeResult(
            direction=Direction.LONG,
            entry_price=100_000.0,
            exit_price=90_000.0,
            quantity=0.01,
            entry_time=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
            exit_time=datetime(2024, 6, 15, 13, 0, tzinfo=timezone.utc),
            gross_pnl=-1_000.0,
            fee_cost=5.0,
            slippage_cost=2.0,
            net_pnl=-1_007.0,  # 10.07% de 10k — dépasse le seuil de 5%
            exit_reason="sl",
            market_regime=MarketRegime.RANGING,
        )
        runner._record_trade(trade, "BTC/USDT")

        assert runner.is_kill_switch_triggered is False
        assert runner._stats.is_active is not False
        assert runner._stats.total_trades == 1
        assert runner._stats.net_pnl == pytest.approx(-1_007.0)

    def test_grid_multiple_losses_no_kill_switch(self):
        """Même après plusieurs pertes cumulées, pas de kill switch grid."""
        runner = _make_grid_runner()

        for i in range(5):
            trade = TradeResult(
                direction=Direction.LONG,
                entry_price=100_000.0,
                exit_price=98_000.0,
                quantity=0.01,
                entry_time=datetime(2024, 6, 15, i, 0, tzinfo=timezone.utc),
                exit_time=datetime(2024, 6, 15, i, 30, tzinfo=timezone.utc),
                gross_pnl=-200.0,
                fee_cost=3.0,
                slippage_cost=1.0,
                net_pnl=-204.0,
                exit_reason="sl",
                market_regime=MarketRegime.RANGING,
            )
            runner._record_trade(trade, "BTC/USDT")

        # 5 × -204 = -1020, soit ~10.2% de perte cumulée
        assert runner.is_kill_switch_triggered is False
        assert runner._stats.total_trades == 5


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
