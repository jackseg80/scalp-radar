"""Tests Phase 2 — Cooldown anti-churning après close grid.

Vérifie que le cooldown empêche les réouvertures immédiates
sur le même symbol, côté paper (GridStrategyRunner) et live (Executor).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.models import Candle, Direction, TimeFrame


# ─── Helpers ──────────────────────────────────────────────────────────────


def _make_candle(
    close: float = 50_000.0,
    low: float | None = None,
    high: float | None = None,
    ts: datetime | None = None,
) -> Candle:
    if ts is None:
        ts = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
    return Candle(
        timestamp=ts,
        open=close,
        high=high if high is not None else close * 1.01,
        low=low if low is not None else close * 0.99,
        close=close,
        volume=100.0,
        symbol="BTC/USDT",
        timeframe=TimeFrame.H1,
    )


# ─── Tests GridStrategyRunner (paper) ─────────────────────────────────────


class TestCooldownRunner:
    """Cooldown anti-churning côté paper (GridStrategyRunner)."""

    @pytest.mark.asyncio
    async def test_cooldown_blocks_reentry_after_close(self):
        """Close à T=14h → compute_grid non évalué à T+1h et T+2h, OK à T+3h."""
        from backend.backtesting.simulator import GridStrategyRunner

        runner = _make_grid_runner(cooldown_candles=3)
        runner._is_warming_up = False

        # Enregistrer un close à T=14:00
        t0 = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
        runner._record_close("BTC/USDT", t0)

        # T+1h (15:00) — en cooldown
        candle_1h = _make_candle(ts=t0 + timedelta(hours=1))
        await runner.on_candle("BTC/USDT", "1h", candle_1h)
        runner._strategy.compute_grid.assert_not_called()

        # T+2h (16:00) — encore en cooldown
        candle_2h = _make_candle(ts=t0 + timedelta(hours=2))
        await runner.on_candle("BTC/USDT", "1h", candle_2h)
        runner._strategy.compute_grid.assert_not_called()

        # T+3h (17:00) — cooldown expiré, compute_grid doit être appelé
        candle_3h = _make_candle(ts=t0 + timedelta(hours=3))
        await runner.on_candle("BTC/USDT", "1h", candle_3h)
        runner._strategy.compute_grid.assert_called()

    @pytest.mark.asyncio
    async def test_cooldown_zero_disables(self):
        """cooldown_candles=0 → compute_grid évalué immédiatement après close."""
        from backend.backtesting.simulator import GridStrategyRunner

        runner = _make_grid_runner(cooldown_candles=0)
        runner._is_warming_up = False

        t0 = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
        runner._record_close("BTC/USDT", t0)

        # T+1h — pas de cooldown
        candle = _make_candle(ts=t0 + timedelta(hours=1))
        await runner.on_candle("BTC/USDT", "1h", candle)
        runner._strategy.compute_grid.assert_called()

    @pytest.mark.asyncio
    async def test_cooldown_per_symbol(self):
        """Close sur BTC à T → ETH peut ouvrir à T+1h (indépendant)."""
        from collections import deque

        from backend.backtesting.simulator import GridStrategyRunner

        runner = _make_grid_runner(cooldown_candles=3)
        runner._is_warming_up = False
        # Ajouter ETH à la whitelist + pré-remplir buffer closes
        runner._per_asset_keys.add("ETH/USDT")
        eth_buf = deque(maxlen=50)
        for _ in range(20):
            eth_buf.append(3000.0)
        runner._close_buffer["ETH/USDT"] = eth_buf

        t0 = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
        runner._record_close("BTC/USDT", t0)

        # ETH à T+1h — PAS en cooldown (seul BTC est bloqué)
        candle_eth = Candle(
            timestamp=t0 + timedelta(hours=1),
            open=3000, high=3030, low=2970, close=3000,
            volume=50.0, symbol="ETH/USDT", timeframe=TimeFrame.H1,
        )
        await runner.on_candle("ETH/USDT", "1h", candle_eth)
        runner._strategy.compute_grid.assert_called()

    @pytest.mark.asyncio
    async def test_cooldown_does_not_block_exits(self):
        """Positions ouvertes BTC + cooldown actif → check exits toujours évalué."""
        from backend.backtesting.simulator import GridStrategyRunner
        from backend.strategies.base_grid import GridPosition

        runner = _make_grid_runner(cooldown_candles=3)
        runner._is_warming_up = False

        t0 = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
        runner._record_close("ETH/USDT", t0)  # cooldown sur ETH

        # BTC a une position ouverte
        runner._positions["BTC/USDT"] = [
            GridPosition(
                level=0, direction=Direction.LONG,
                entry_price=50000, quantity=0.01,
                entry_time=t0 - timedelta(hours=5),
                entry_fee=0.3,
            )
        ]

        # Simuler TP touché
        runner._strategy.get_tp_price.return_value = 51000.0
        runner._strategy.get_sl_price.return_value = 45000.0
        runner._gpm.check_global_tp_sl.return_value = ("tp", 51000.0)

        candle = _make_candle(close=51000, ts=t0 + timedelta(hours=1))
        await runner.on_candle("BTC/USDT", "1h", candle)

        # Le close a eu lieu (pas bloqué par cooldown ETH)
        runner._gpm.close_all_positions.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_close_called_on_tp(self):
        """Close par tp_global → _last_close_time[symbol] enregistré."""
        from backend.backtesting.simulator import GridStrategyRunner
        from backend.strategies.base_grid import GridPosition, GridState

        runner = _make_grid_runner(cooldown_candles=3)
        runner._is_warming_up = False

        t0 = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
        runner._positions["BTC/USDT"] = [
            GridPosition(
                level=0, direction=Direction.LONG,
                entry_price=50000, quantity=0.01,
                entry_time=t0 - timedelta(hours=5),
                entry_fee=0.3,
            )
        ]

        runner._strategy.get_tp_price.return_value = 51000.0
        runner._strategy.get_sl_price.return_value = 45000.0
        runner._gpm.check_global_tp_sl.return_value = ("tp", 51000.0)

        trade_mock = MagicMock()
        trade_mock.net_pnl = 10.0
        trade_mock.gross_pnl = 12.0
        trade_mock.fee_cost = 2.0
        trade_mock.exit_time = t0
        trade_mock.exit_price = 51000.0
        trade_mock.exit_reason = "tp"
        trade_mock.direction = Direction.LONG
        trade_mock.entry_price = 50000.0
        trade_mock.quantity = 0.01
        runner._gpm.close_all_positions.return_value = trade_mock

        candle = _make_candle(close=51000, ts=t0)
        await runner.on_candle("BTC/USDT", "1h", candle)

        assert "BTC/USDT" in runner._last_close_time
        assert runner._last_close_time["BTC/USDT"] == t0

    @pytest.mark.asyncio
    async def test_cooldown_not_recorded_during_warmup(self):
        """Close pendant warm-up → _last_close_time PAS enregistré."""
        from backend.backtesting.simulator import GridStrategyRunner
        from backend.strategies.base_grid import GridPosition

        runner = _make_grid_runner(cooldown_candles=3)
        runner._is_warming_up = True

        t0 = datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc)
        runner._positions["BTC/USDT"] = [
            GridPosition(
                level=0, direction=Direction.LONG,
                entry_price=50000, quantity=0.01,
                entry_time=t0 - timedelta(hours=5),
                entry_fee=0.3,
            )
        ]

        runner._strategy.get_tp_price.return_value = 51000.0
        runner._strategy.get_sl_price.return_value = 45000.0
        runner._gpm.check_global_tp_sl.return_value = ("tp", 51000.0)

        trade_mock = MagicMock()
        trade_mock.net_pnl = 10.0
        runner._gpm.close_all_positions.return_value = trade_mock

        # Bougie ancienne (warmup)
        candle = _make_candle(close=51000, ts=t0)
        await runner.on_candle("BTC/USDT", "1h", candle)

        assert "BTC/USDT" not in runner._last_close_time

    @pytest.mark.asyncio
    async def test_record_close_called_on_should_close_all(self):
        """Close par should_close_all → _last_close_time enregistré."""
        from backend.backtesting.simulator import GridStrategyRunner
        from backend.strategies.base_grid import GridPosition

        runner = _make_grid_runner(cooldown_candles=3)
        runner._is_warming_up = False

        t0 = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
        runner._positions["BTC/USDT"] = [
            GridPosition(
                level=0, direction=Direction.LONG,
                entry_price=50000, quantity=0.01,
                entry_time=t0 - timedelta(hours=5),
                entry_fee=0.3,
            )
        ]

        # TP/SL ne triggèrent pas, mais should_close_all oui
        runner._strategy.get_tp_price.return_value = float("nan")
        runner._strategy.get_sl_price.return_value = 45000.0
        runner._gpm.check_global_tp_sl.return_value = (None, None)
        runner._strategy.should_close_all.return_value = "sma_cross"

        trade_mock = MagicMock()
        trade_mock.net_pnl = -5.0
        trade_mock.gross_pnl = -3.0
        trade_mock.fee_cost = 2.0
        trade_mock.exit_time = t0
        trade_mock.exit_price = 49000.0
        trade_mock.exit_reason = "sma_cross"
        trade_mock.direction = Direction.LONG
        trade_mock.entry_price = 50000.0
        trade_mock.quantity = 0.01
        runner._gpm.close_all_positions.return_value = trade_mock

        candle = _make_candle(close=49000, ts=t0)
        await runner.on_candle("BTC/USDT", "1h", candle)

        assert "BTC/USDT" in runner._last_close_time


# ─── Tests Executor (live) ────────────────────────────────────────────────


class TestCooldownExecutor:
    """Cooldown anti-churning côté live (Executor)."""

    @pytest.mark.asyncio
    async def test_executor_cooldown_blocks_entry(self):
        """_last_close_time récent → _on_candle skip avec log debug."""
        executor = _make_executor_mock(cooldown_candles=3)

        t0 = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
        executor._last_close_time["BTC/USDT:USDT"] = t0

        candle = _make_candle(ts=t0 + timedelta(hours=1))
        await executor._on_candle("BTC/USDT", "1h", candle)

        # get_runner_context ne doit PAS être appelé (skip avant)
        executor._simulator.get_runner_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_executor_cooldown_expired_allows_entry(self):
        """_last_close_time vieux de 4h, cooldown=3, tf=1h → indicateurs lus."""
        executor = _make_executor_mock(cooldown_candles=3)

        t0 = datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc)
        executor._last_close_time["BTC/USDT:USDT"] = t0

        # Candle 4h après le close — cooldown expiré
        candle = _make_candle(ts=t0 + timedelta(hours=4))
        await executor._on_candle("BTC/USDT", "1h", candle)

        # get_runner_context DOIT être appelé
        executor._simulator.get_runner_context.assert_called()

    def test_executor_record_close_in_close_grid_cycle(self):
        """_close_grid_cycle() → _last_close_time mis à jour."""
        from backend.execution.executor import Executor

        executor = MagicMock(spec=Executor)
        executor._last_close_time = {}
        Executor._record_grid_close(executor, "BTC/USDT:USDT")

        assert "BTC/USDT:USDT" in executor._last_close_time

    def test_executor_cooldown_survives_restart(self):
        """get_state() contient last_close_times, restore le restaure."""
        from backend.execution.executor import Executor

        executor = _make_executor_for_state()

        t0 = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
        executor._last_close_time["BTC/USDT:USDT"] = t0

        state = executor.get_state_for_persistence()
        assert "last_close_times" in state
        assert "BTC/USDT:USDT" in state["last_close_times"]

        # Restauration
        executor2 = _make_executor_for_state()
        executor2.restore_positions(state)
        assert "BTC/USDT:USDT" in executor2._last_close_time
        assert executor2._last_close_time["BTC/USDT:USDT"] == t0


# ─── Tests Config ─────────────────────────────────────────────────────────


class TestCooldownConfig:

    def test_config_default_value_grid_atr(self):
        """GridATRConfig() sans cooldown_candles → default 3."""
        from backend.core.config import GridATRConfig
        config = GridATRConfig()
        assert config.cooldown_candles == 3

    def test_config_default_value_grid_boltrend(self):
        """GridBolTrendConfig() sans cooldown_candles → default 3."""
        from backend.core.config import GridBolTrendConfig
        config = GridBolTrendConfig()
        assert config.cooldown_candles == 3

    def test_config_zero_valid(self):
        """cooldown_candles=0 → accepté."""
        from backend.core.config import GridATRConfig
        config = GridATRConfig(cooldown_candles=0)
        assert config.cooldown_candles == 0

    def test_config_in_get_params_for_symbol(self):
        """cooldown_candles inclus dans get_params_for_symbol()."""
        from backend.core.config import GridATRConfig
        config = GridATRConfig(cooldown_candles=5)
        params = config.get_params_for_symbol("BTC/USDT")
        assert params["cooldown_candles"] == 5


# ─── Tests TF_SECONDS extraction ─────────────────────────────────────────


class TestTFSeconds:

    def test_tf_seconds_importable(self):
        """TF_SECONDS importable depuis base_grid."""
        from backend.strategies.base_grid import TF_SECONDS
        assert TF_SECONDS["1h"] == 3600
        assert TF_SECONDS["4h"] == 14400

    def test_grid_atr_uses_shared_constant(self):
        """grid_atr.py utilise TF_SECONDS depuis base_grid."""
        import backend.strategies.grid_atr as mod
        assert not hasattr(mod, "_TF_SECONDS")

    def test_grid_boltrend_uses_shared_constant(self):
        """grid_boltrend.py utilise TF_SECONDS depuis base_grid."""
        import backend.strategies.grid_boltrend as mod
        assert not hasattr(mod, "_TF_SECONDS")


# ─── Tests Persistence (StateManager) ────────────────────────────────────


class TestCooldownPersistence:

    def test_statemanager_saves_close_times(self):
        """StateManager sérialise _last_close_time."""
        from backend.backtesting.simulator import GridStrategyRunner

        runner = _make_grid_runner(cooldown_candles=3)
        t0 = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
        runner._last_close_time["BTC/USDT"] = t0

        # Simuler la sérialisation du StateManager
        close_times = getattr(runner, "_last_close_time", {})
        data = {sym: ts.isoformat() for sym, ts in close_times.items()}

        assert data["BTC/USDT"] == t0.isoformat()

    def test_runner_restores_close_times(self):
        """_apply_restored_state restaure _last_close_time."""
        from backend.backtesting.simulator import GridStrategyRunner

        runner = _make_grid_runner(cooldown_candles=3)
        t0 = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)

        state = {
            "capital": 10000,
            "last_close_times": {"BTC/USDT": t0.isoformat()},
        }
        runner._apply_restored_state(state)

        assert "BTC/USDT" in runner._last_close_time
        assert runner._last_close_time["BTC/USDT"] == t0


# ─── Factory helpers ──────────────────────────────────────────────────────


def _make_grid_runner(cooldown_candles: int = 3):
    """Crée un GridStrategyRunner mock pour les tests cooldown."""
    from collections import deque

    from backend.backtesting.simulator import GridStrategyRunner

    strategy = MagicMock()
    strategy.name = "grid_atr"
    strategy._config = MagicMock()
    strategy._config.timeframe = "1h"
    strategy._config.ma_period = 14
    strategy._config.cooldown_candles = cooldown_candles
    strategy._config.leverage = 6
    strategy._config.num_levels = 3
    strategy._config.sl_percent = 20.0
    strategy._config.per_asset = {"BTC/USDT": {}}
    strategy._config.sides = ["long"]
    strategy.max_positions = 3
    strategy.compute_grid = MagicMock(return_value=[])
    strategy.get_tp_price = MagicMock(return_value=float("nan"))
    strategy.get_sl_price = MagicMock(return_value=45000.0)
    strategy.should_close_all = MagicMock(return_value=None)
    strategy.min_candles = {"1h": 50}
    strategy.compute_live_indicators = MagicMock(return_value={})

    config = MagicMock()
    config.assets = [MagicMock(symbol="BTC/USDT")]
    config.risk = MagicMock()
    config.risk.initial_capital = 10000.0
    config.risk.kill_switch = MagicMock()
    config.risk.kill_switch.global_max_loss_pct = 45.0
    config.risk.kill_switch.runner_max_loss_pct = 25.0
    config.risk.max_margin_ratio = 0.70
    config.risk.regime_filter_enabled = False

    indicator_engine = MagicMock()
    indicator_engine.get_indicators = MagicMock(return_value={})
    gpm = MagicMock()
    gpm.check_global_tp_sl = MagicMock(return_value=(None, None))
    data_engine = MagicMock()
    data_engine.get_funding_rate = MagicMock(return_value=None)

    runner = GridStrategyRunner(
        strategy=strategy,
        config=config,
        indicator_engine=indicator_engine,
        grid_position_manager=gpm,
        data_engine=data_engine,
    )

    # Post-init : sortir du warm-up et ajuster
    runner._is_warming_up = False
    runner._warmup_ended_at = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
    runner._candles_since_warmup = 20

    # Buffer de closes pré-rempli (>= ma_period=14) pour passer le guard SMA
    buf = deque(maxlen=50)
    for _ in range(20):
        buf.append(50000.0)
    runner._close_buffer["BTC/USDT"] = buf

    return runner


def _make_executor_mock(cooldown_candles: int = 3):
    """Crée un Executor simplifié pour les tests cooldown."""
    from backend.execution.executor import Executor

    executor = MagicMock(spec=Executor)
    executor._running = True
    executor._connected = True
    executor._strategies = {
        "grid_atr": MagicMock(),
    }
    executor._strategies["grid_atr"]._config = MagicMock()
    executor._strategies["grid_atr"]._config.timeframe = "1h"
    executor._strategies["grid_atr"]._config.cooldown_candles = cooldown_candles
    executor._strategies["grid_atr"].max_positions = 3

    executor._grid_states = {}
    executor._last_close_time = {}
    executor._pending_levels = set()
    executor._pending_notional = 0.0
    executor._balance_bootstrapped = True
    executor._exchange_balance = 1000.0

    executor._simulator = MagicMock()
    executor._simulator.get_runner_context = MagicMock(return_value=None)

    # Bind la vraie méthode
    executor._on_candle = Executor._on_candle.__get__(executor, Executor)

    return executor


def _make_executor_for_state():
    """Crée un Executor minimal pour tester persistence."""
    from collections import deque

    from backend.execution.executor import Executor

    executor = MagicMock(spec=Executor)
    executor._positions = {}
    executor._grid_states = {}
    executor._last_close_time = {}
    executor._risk_manager = MagicMock()
    executor._risk_manager.get_state.return_value = {}
    executor._order_history = deque(maxlen=200)

    # Bind les vraies méthodes
    executor.get_state_for_persistence = Executor.get_state_for_persistence.__get__(
        executor, Executor
    )
    executor.restore_positions = Executor.restore_positions.__get__(
        executor, Executor
    )

    return executor
