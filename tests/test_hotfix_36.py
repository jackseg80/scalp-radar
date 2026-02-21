"""Tests Hotfix 36 — Warmup position tracking + DataEngine auto-recovery.

Couvre :
- Fix A : warmup position tracking (remplace le cooldown 3h par tracking ciblé par symbol)
- Fix B : DataEngine never give up (_watch_symbol sans max_attempts)
- Fix B3 : restart_dead_tasks relance les tâches mortes
- Fix B4 : Watchdog auto-recovery sur data_stale
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from collections import deque

import pytest

from backend.backtesting.simulator import GridStrategyRunner
from backend.core.grid_position_manager import GridPositionManager
from backend.core.incremental_indicators import IncrementalIndicatorEngine
from backend.core.models import Candle, Direction, TimeFrame
from backend.strategies.base_grid import BaseGridStrategy, GridLevel


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_candle(
    close: float = 96_000.0,
    high: float | None = None,
    low: float | None = None,
) -> Candle:
    h = high or close * 1.002
    lo = low or close * 0.998
    return Candle(
        timestamp=datetime.now(tz=timezone.utc),
        open=close,
        high=h,
        low=lo,
        close=close,
        volume=100.0,
        symbol="BTC/USDT",
        timeframe=TimeFrame.H1,
    )


def _make_mock_strategy(
    name: str = "grid_atr",
    grid_levels: list[GridLevel] | None = None,
) -> MagicMock:
    from backend.core.position_manager import PositionManagerConfig

    strategy = MagicMock(spec=BaseGridStrategy)
    strategy.name = name
    config = MagicMock()
    config.timeframe = "1h"
    config.ma_period = 7
    config.leverage = 6
    config.per_asset = {}
    strategy._config = config
    strategy.min_candles = {"1h": 50}
    strategy.max_positions = 3
    strategy.compute_grid.return_value = grid_levels or []
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
    config.risk.max_live_grids = 4
    config.risk.kill_switch.max_session_loss_percent = 5.0
    config.risk.kill_switch.max_daily_loss_percent = 10.0
    config.risk.kill_switch.grid_max_session_loss_percent = 25.0
    config.risk.kill_switch.grid_max_daily_loss_percent = 25.0
    config.risk.position.max_risk_per_trade_percent = 2.0
    config.assets = [MagicMock(symbol="BTC/USDT")]
    return config


def _make_grid_runner(strategy=None, config=None) -> GridStrategyRunner:
    from backend.core.position_manager import PositionManagerConfig

    if strategy is None:
        strategy = _make_mock_strategy()
    if config is None:
        config = _make_mock_config()

    indicator_engine = MagicMock(spec=IncrementalIndicatorEngine)
    indicator_engine.get_indicators.return_value = {}
    indicator_engine.update = MagicMock()
    indicator_engine._buffers = {}

    gpm_config = PositionManagerConfig(
        leverage=6,
        maker_fee=0.0002,
        taker_fee=0.0006,
        slippage_pct=0.0005,
        high_vol_slippage_mult=2.0,
        max_risk_per_trade=0.02,
    )
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
    runner._is_warming_up = False
    runner._warmup_ended_at = datetime.now(tz=timezone.utc)
    return runner


def _fill_buffer(runner: GridStrategyRunner, symbol: str = "BTC/USDT", n: int = 15) -> None:
    runner._close_buffer[symbol] = deque(maxlen=50)
    for i in range(n):
        runner._close_buffer[symbol].append(98_000.0 + i * 100)


# ─── Fix A : Warmup position tracking ────────────────────────────────────────


class TestWarmupPositionTracking:
    """Tests du warmup position tracking (remplace le cooldown 3h)."""

    def test_warmup_tracking_blocks_open_for_warmup_symbols(self):
        """Event OPEN supprimé si symbol dans _warmup_position_symbols."""
        runner = _make_grid_runner()
        runner._warmup_position_symbols = {"BTC/USDT"}

        pos = MagicMock()
        pos.direction.value = "LONG"
        pos.entry_price = 95_000.0
        pos.quantity = 0.01
        pos.entry_time = datetime.now(tz=timezone.utc)
        level = MagicMock()

        runner._emit_open_event("BTC/USDT", level, pos)

        assert len(runner._pending_events) == 0

    def test_warmup_tracking_allows_open_for_non_warmup_symbols(self):
        """Event OPEN passe si symbol PAS dans _warmup_position_symbols."""
        runner = _make_grid_runner()
        runner._warmup_position_symbols = {"ETH/USDT"}  # BTC pas dedans

        pos = MagicMock()
        pos.direction.value = "LONG"
        pos.entry_price = 95_000.0
        pos.quantity = 0.01
        pos.entry_time = datetime.now(tz=timezone.utc)
        level = MagicMock()

        with patch("backend.execution.executor.TradeEvent"):
            runner._emit_open_event("BTC/USDT", level, pos)

        assert len(runner._pending_events) == 1

    def test_warmup_tracking_blocks_close_and_removes_symbol(self):
        """Event CLOSE supprimé pour warmup symbol + symbol retiré du set."""
        runner = _make_grid_runner()
        runner._warmup_position_symbols = {"BTC/USDT", "ETH/USDT"}

        trade = MagicMock()
        trade.direction.value = "LONG"
        trade.entry_price = 95_000.0
        trade.quantity = 0.01
        trade.exit_time = datetime.now(tz=timezone.utc)
        trade.exit_reason = "tp"
        trade.exit_price = 100_000.0

        runner._emit_close_event("BTC/USDT", trade)

        assert len(runner._pending_events) == 0
        assert "BTC/USDT" not in runner._warmup_position_symbols
        assert "ETH/USDT" in runner._warmup_position_symbols

    def test_warmup_tracking_allows_close_after_removal(self):
        """Après retrait du set, CLOSE passe normalement."""
        runner = _make_grid_runner()
        runner._warmup_position_symbols = set()  # vide

        trade = MagicMock()
        trade.direction.value = "LONG"
        trade.entry_price = 95_000.0
        trade.quantity = 0.01
        trade.exit_time = datetime.now(tz=timezone.utc)
        trade.exit_reason = "sl_global"
        trade.exit_price = 90_000.0

        with patch("backend.execution.executor.TradeEvent"):
            runner._emit_close_event("BTC/USDT", trade)

        assert len(runner._pending_events) == 1

    def test_warmup_tracking_complete_cycle(self):
        """Cycle complet : open bloqué → close bloqué + retrait → prochain open passe."""
        runner = _make_grid_runner()
        runner._warmup_position_symbols = {"BTC/USDT"}

        pos = MagicMock()
        pos.direction.value = "LONG"
        pos.entry_price = 95_000.0
        pos.quantity = 0.01
        pos.entry_time = datetime.now(tz=timezone.utc)
        level = MagicMock()

        # 1. OPEN bloqué
        runner._emit_open_event("BTC/USDT", level, pos)
        assert len(runner._pending_events) == 0

        # 2. CLOSE bloqué + symbol retiré
        trade = MagicMock()
        trade.direction.value = "LONG"
        trade.entry_price = 95_000.0
        trade.quantity = 0.01
        trade.exit_time = datetime.now(tz=timezone.utc)
        trade.exit_reason = "tp"
        trade.exit_price = 100_000.0

        runner._emit_close_event("BTC/USDT", trade)
        assert len(runner._pending_events) == 0
        assert "BTC/USDT" not in runner._warmup_position_symbols

        # 3. Prochain OPEN passe
        with patch("backend.execution.executor.TradeEvent"):
            runner._emit_open_event("BTC/USDT", level, pos)
        assert len(runner._pending_events) == 1

    @pytest.mark.asyncio
    async def test_paper_positions_open_with_warmup_tracking(self):
        """Positions paper s'ouvrent, events Executor bloqués pour warmup symbols."""
        level = GridLevel(
            index=0,
            entry_price=95_000.0,
            direction=Direction.LONG,
            size_fraction=0.33,
        )
        strategy = _make_mock_strategy(grid_levels=[level])
        runner = _make_grid_runner(strategy=strategy)
        runner._warmup_position_symbols = {"BTC/USDT"}

        _fill_buffer(runner)

        candle = _make_candle(close=96_000.0, low=94_500.0, high=97_000.0)
        await runner.on_candle("BTC/USDT", "1h", candle)

        positions = runner._positions.get("BTC/USDT", [])
        assert len(positions) == 1
        assert len(runner._pending_events) == 0

    def test_warmup_tracking_empty_set_allows_all(self):
        """Si warmup set vide, tous les events passent (pas de warm-up positions)."""
        runner = _make_grid_runner()
        runner._warmup_position_symbols = set()

        pos = MagicMock()
        pos.direction.value = "LONG"
        pos.entry_price = 95_000.0
        pos.quantity = 0.01
        pos.entry_time = datetime.now(tz=timezone.utc)
        level = MagicMock()

        with patch("backend.execution.executor.TradeEvent"):
            runner._emit_open_event("BTC/USDT", level, pos)

        assert len(runner._pending_events) == 1

    def test_end_warmup_populates_warmup_set(self):
        """_end_warmup() peuple _warmup_position_symbols depuis les positions ouvertes."""
        runner = _make_grid_runner()
        runner._is_warming_up = True

        # Simuler des positions ouvertes pendant le warm-up
        pos_btc = MagicMock()
        pos_eth = MagicMock()
        runner._positions = {
            "BTC/USDT": [pos_btc],
            "ETH/USDT": [pos_eth],
            "SOL/USDT": [],  # pas de position → pas dans le set
        }

        runner._end_warmup()

        assert runner._warmup_position_symbols == {"BTC/USDT", "ETH/USDT"}
        assert runner._is_warming_up is False
        assert runner._warmup_ended_at is not None

    def test_end_warmup_excludes_restored_symbols(self):
        """Symbols restaurés par _pending_restore ne sont pas dans le warmup set."""
        runner = _make_grid_runner()
        runner._is_warming_up = True

        # Positions warm-up sur BTC et ETH
        pos_btc = MagicMock()
        pos_eth = MagicMock()
        runner._positions = {
            "BTC/USDT": [pos_btc],
            "ETH/USDT": [pos_eth],
        }

        # État sauvegardé : BTC a des positions live (restart scenario)
        # Format attendu par _apply_restored_state : grid_positions = liste plate
        runner._pending_restore = {
            "capital": 9500.0,
            "realized_pnl": -500.0,
            "grid_positions": [
                {
                    "symbol": "BTC/USDT",
                    "entry_price": 95000,
                    "quantity": 0.01,
                    "direction": "LONG",
                    "level": 0,
                    "entry_fee": 0.0,
                    "entry_time": datetime.now(tz=timezone.utc).isoformat(),
                },
            ],
        }

        runner._end_warmup()

        # BTC exclu (restauré), ETH reste dans le warmup set
        assert "BTC/USDT" not in runner._warmup_position_symbols
        assert "ETH/USDT" in runner._warmup_position_symbols


# ─── Fix B : DataEngine recovery ─────────────────────────────────────────────


class TestDataEngineRecovery:
    """Tests de restart_dead_tasks et du watch never-give-up."""

    @pytest.mark.asyncio
    async def test_restart_dead_tasks_relaunches(self):
        """Tâche terminée → relancée avec bon symbol."""
        from backend.core.data_engine import DataEngine

        config = MagicMock()
        asset = MagicMock()
        asset.symbol = "BTC/USDT"
        asset.timeframes = ["1h"]
        config.assets = [asset]
        config.exchange.websocket.reconnect_delay = 1

        db = MagicMock()
        engine = DataEngine(config, db)
        engine._running = True

        # Créer une tâche "morte" (terminée avec exception)
        async def _dead():
            raise RuntimeError("dead")

        dead_task = asyncio.create_task(_dead(), name="watch_BTC/USDT")
        await asyncio.sleep(0.05)  # Laisser la tâche se terminer
        assert dead_task.done()

        engine._tasks = [dead_task]

        # Mock _watch_symbol pour ne pas lancer une vraie boucle
        with patch.object(engine, "_watch_symbol", new=AsyncMock()) as mock_watch:
            restarted = await engine.restart_dead_tasks()

        assert restarted == 1
        assert len(engine._tasks) == 1
        assert not engine._tasks[0].done() or engine._tasks[0] != dead_task

    @pytest.mark.asyncio
    async def test_restart_dead_tasks_skips_alive(self):
        """Tâche vivante → pas touchée."""
        from backend.core.data_engine import DataEngine

        config = MagicMock()
        config.assets = []
        db = MagicMock()
        engine = DataEngine(config, db)
        engine._running = True

        # Tâche vivante (sleep indéfini)
        async def _alive():
            await asyncio.sleep(100)

        alive_task = asyncio.create_task(_alive(), name="watch_ETH/USDT")
        engine._tasks = [alive_task]

        restarted = await engine.restart_dead_tasks()

        assert restarted == 0
        assert len(engine._tasks) == 1
        assert engine._tasks[0] is alive_task

        alive_task.cancel()
        try:
            await alive_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_restart_dead_tasks_skips_cancelled(self):
        """Tâche cancelled → pas relancée."""
        from backend.core.data_engine import DataEngine

        config = MagicMock()
        config.assets = []
        db = MagicMock()
        engine = DataEngine(config, db)
        engine._running = True

        async def _wait():
            await asyncio.sleep(100)

        task = asyncio.create_task(_wait(), name="watch_BTC/USDT")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        engine._tasks = [task]
        restarted = await engine.restart_dead_tasks()

        assert restarted == 0

    @pytest.mark.asyncio
    async def test_watch_symbol_never_exits_on_rate_limit(self):
        """Après N rate limits, la boucle continue (pas de break)."""
        from backend.core.data_engine import DataEngine

        config = MagicMock()
        config.exchange.websocket.reconnect_delay = 0.01
        db = MagicMock()
        engine = DataEngine(config, db)
        engine._running = True

        call_count = 0

        async def _failing_subscribe(symbol, timeframes):
            nonlocal call_count
            call_count += 1
            if call_count >= 5:
                engine._running = False  # Stopper la boucle après 5 tentatives
            raise RuntimeError("rate limit 30006")

        with patch.object(engine, "_subscribe_klines", side_effect=_failing_subscribe):
            await engine._watch_symbol("BTC/USDT", ["1h"])

        # La boucle a fait 5 itérations (pas de break prématuré)
        assert call_count == 5


# ─── Fix B3 : Watchdog auto-recovery ─────────────────────────────────────────


class TestWatchdogAutoRecovery:
    """Tests de l'auto-recovery dans le Watchdog."""

    @pytest.mark.asyncio
    async def test_watchdog_triggers_restart_after_10min(self):
        """data_stale > 600s → appelle restart_dead_tasks."""
        from backend.monitoring.watchdog import Watchdog

        data_engine = MagicMock()
        data_engine.is_connected = True
        # Dernière update il y a 11 min
        data_engine.last_update = datetime.now(tz=timezone.utc) - timedelta(minutes=11)
        data_engine.restart_dead_tasks = AsyncMock(return_value=2)

        simulator = MagicMock()
        simulator.runners = []
        simulator.is_kill_switch_triggered.return_value = False

        notifier = MagicMock()
        notifier.notify_anomaly = AsyncMock()

        watchdog = Watchdog(data_engine, simulator, notifier)
        await watchdog._check()

        data_engine.restart_dead_tasks.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_watchdog_triggers_full_reconnect_after_30min(self):
        """data_stale > 1800s + 0 tasks restarted → full_reconnect."""
        from backend.monitoring.watchdog import Watchdog

        data_engine = MagicMock()
        data_engine.is_connected = True
        # Dernière update il y a 35 min
        data_engine.last_update = datetime.now(tz=timezone.utc) - timedelta(minutes=35)
        data_engine.restart_dead_tasks = AsyncMock(return_value=0)
        data_engine.full_reconnect = AsyncMock()

        simulator = MagicMock()
        simulator.runners = []
        simulator.is_kill_switch_triggered.return_value = False

        notifier = MagicMock()
        notifier.notify_anomaly = AsyncMock()

        watchdog = Watchdog(data_engine, simulator, notifier)
        await watchdog._check()

        data_engine.restart_dead_tasks.assert_awaited_once()
        data_engine.full_reconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_watchdog_no_recovery_under_10min(self):
        """data_stale > 5min mais < 10min → alerte seule, pas de recovery."""
        from backend.monitoring.watchdog import Watchdog

        data_engine = MagicMock()
        data_engine.is_connected = True
        # Dernière update il y a 7 min (> 300s mais < 600s)
        data_engine.last_update = datetime.now(tz=timezone.utc) - timedelta(minutes=7)
        data_engine.restart_dead_tasks = AsyncMock(return_value=0)

        simulator = MagicMock()
        simulator.runners = []
        simulator.is_kill_switch_triggered.return_value = False

        notifier = MagicMock()
        notifier.notify_anomaly = AsyncMock()

        watchdog = Watchdog(data_engine, simulator, notifier)
        await watchdog._check()

        # Alerte envoyée
        notifier.notify_anomaly.assert_awaited_once()
        # Mais PAS de restart (< 600s)
        data_engine.restart_dead_tasks.assert_not_awaited()
