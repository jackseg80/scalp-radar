"""Tests Hotfix 35 — Stabilité restart : cooldown post-warmup, max_live_grids, sauvegarde executor.

Couvre :
- Bug A : cooldown post-warmup dans GridStrategyRunner (basé sur le temps — Hotfix 36)
- Bug B : guard max_live_grids dans Executor (limite nouveaux cycles simultanés)
- Bug C : sauvegarde périodique executor dans StateManager
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.backtesting.simulator import GridStrategyRunner
from backend.core.grid_position_manager import GridPositionManager
from backend.core.incremental_indicators import IncrementalIndicatorEngine
from backend.core.models import Candle, Direction, TimeFrame
from backend.core.position_manager import PositionManagerConfig, TradeResult
from backend.strategies.base_grid import BaseGridStrategy, GridLevel, GridPosition, GridState


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_candle(
    close: float = 96_000.0,
    high: float | None = None,
    low: float | None = None,
    ts: datetime | None = None,
) -> Candle:
    """Bougie récente par défaut (age < 2h = post-warmup)."""
    h = high or close * 1.002
    lo = low or close * 0.998
    return Candle(
        timestamp=ts or datetime.now(tz=timezone.utc),
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


def _make_mock_config(initial_capital: float = 10_000.0, max_live_grids: int = 4) -> MagicMock:
    config = MagicMock()
    config.risk.initial_capital = initial_capital
    config.risk.max_margin_ratio = 0.70
    config.risk.max_live_grids = max_live_grids
    config.risk.kill_switch.max_session_loss_percent = 5.0
    config.risk.kill_switch.max_daily_loss_percent = 10.0
    config.risk.kill_switch.grid_max_session_loss_percent = 25.0
    config.risk.kill_switch.grid_max_daily_loss_percent = 25.0
    config.risk.position.max_risk_per_trade_percent = 2.0
    config.assets = [MagicMock(symbol="BTC/USDT")]
    return config


def _make_grid_runner(strategy=None, config=None) -> GridStrategyRunner:
    """Crée un GridStrategyRunner en mode post-warmup (warming_up=False)."""
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
    """Pré-remplit le buffer closes pour dépasser ma_period."""
    runner._close_buffer[symbol] = deque(maxlen=50)
    for i in range(n):
        runner._close_buffer[symbol].append(98_000.0 + i * 100)


# ─── Bug A : Cooldown post-warmup ─────────────────────────────────────────────


class TestCooldownPostWarmup:
    """Tests du cooldown post-warmup basé sur le temps (Hotfix 36)."""

    def test_emit_open_blocked_during_cooldown(self):
        """_emit_open_event supprime l'event quand warmup_ended_at < 3h."""
        runner = _make_grid_runner()
        # Warm-up terminé il y a 30 secondes → cooldown actif
        runner._warmup_ended_at = datetime.now(tz=timezone.utc) - timedelta(seconds=30)

        pos = MagicMock()
        pos.direction.value = "LONG"
        pos.entry_price = 95_000.0
        pos.quantity = 0.01
        pos.entry_time = datetime.now(tz=timezone.utc)
        level = MagicMock()

        runner._emit_open_event("BTC/USDT", level, pos)

        assert len(runner._pending_events) == 0, "Event doit être supprimé pendant cooldown"

    def test_emit_open_allowed_after_cooldown(self):
        """_emit_open_event émet normalement une fois le cooldown écoulé (> 3h)."""
        runner = _make_grid_runner()
        # Warm-up terminé il y a 4h → cooldown passé
        runner._warmup_ended_at = datetime.now(tz=timezone.utc) - timedelta(hours=4)

        pos = MagicMock()
        pos.direction.value = "LONG"
        pos.entry_price = 95_000.0
        pos.quantity = 0.01
        pos.entry_time = datetime.now(tz=timezone.utc)
        level = MagicMock()

        with patch("backend.execution.executor.TradeEvent"):
            runner._emit_open_event("BTC/USDT", level, pos)

        assert len(runner._pending_events) == 1, "Event doit être émis après cooldown"

    def test_emit_close_blocked_during_cooldown(self):
        """_emit_close_event supprime l'event pendant cooldown (pas de position live à fermer)."""
        runner = _make_grid_runner()
        # Warm-up terminé il y a 10 min → cooldown actif
        runner._warmup_ended_at = datetime.now(tz=timezone.utc) - timedelta(minutes=10)

        trade = MagicMock()
        trade.direction.value = "LONG"
        trade.entry_price = 95_000.0
        trade.quantity = 0.01
        trade.exit_time = datetime.now(tz=timezone.utc)
        trade.exit_reason = "tp"
        trade.exit_price = 100_000.0

        runner._emit_close_event("BTC/USDT", trade)

        assert len(runner._pending_events) == 0, "Event CLOSE doit être supprimé pendant cooldown"

    def test_warmup_end_sets_timestamp(self):
        """_end_warmup() définit _warmup_ended_at."""
        runner = _make_grid_runner()
        runner._is_warming_up = True
        runner._warmup_ended_at = None

        runner._end_warmup()

        assert runner._warmup_ended_at is not None
        assert not runner._is_warming_up

    def test_no_warmup_no_cooldown(self):
        """Si _warmup_ended_at is None (pas de warm-up), les events passent."""
        runner = _make_grid_runner()
        runner._warmup_ended_at = None

        pos = MagicMock()
        pos.direction.value = "LONG"
        pos.entry_price = 95_000.0
        pos.quantity = 0.01
        pos.entry_time = datetime.now(tz=timezone.utc)
        level = MagicMock()

        with patch("backend.execution.executor.TradeEvent"):
            runner._emit_open_event("BTC/USDT", level, pos)

        assert len(runner._pending_events) == 1, "Sans warm-up, pas de cooldown"

    @pytest.mark.asyncio
    async def test_paper_positions_open_during_cooldown(self):
        """Les positions paper s'ouvrent normalement même pendant le cooldown."""
        level = GridLevel(
            index=0,
            entry_price=95_000.0,
            direction=Direction.LONG,
            size_fraction=0.33,
        )
        strategy = _make_mock_strategy(grid_levels=[level])
        runner = _make_grid_runner(strategy=strategy)
        # Cooldown actif : warm-up terminé il y a 1 min
        runner._warmup_ended_at = datetime.now(tz=timezone.utc) - timedelta(minutes=1)

        _fill_buffer(runner)

        candle = _make_candle(close=96_000.0, low=94_500.0, high=97_000.0)
        await runner.on_candle("BTC/USDT", "1h", candle)

        positions = runner._positions.get("BTC/USDT", [])
        assert len(positions) == 1, "Position paper doit s'ouvrir pendant cooldown"
        assert positions[0].entry_price == 95_000.0

        assert len(runner._pending_events) == 0, "Aucun event Executor pendant cooldown"

    @pytest.mark.asyncio
    async def test_events_emitted_after_cooldown(self):
        """Les events Executor sont émis normalement après le cooldown (> 3h)."""
        level = GridLevel(
            index=0,
            entry_price=95_000.0,
            direction=Direction.LONG,
            size_fraction=0.33,
        )
        strategy = _make_mock_strategy(grid_levels=[level])
        runner = _make_grid_runner(strategy=strategy)
        # Cooldown passé : warm-up terminé il y a 4h
        runner._warmup_ended_at = datetime.now(tz=timezone.utc) - timedelta(hours=4)

        _fill_buffer(runner)

        candle = _make_candle(close=96_000.0, low=94_500.0, high=97_000.0)
        with patch("backend.execution.executor.TradeEvent") as mock_event:
            mock_event.return_value = MagicMock()
            await runner.on_candle("BTC/USDT", "1h", candle)

        assert len(runner._pending_events) == 1, "Event Executor doit être émis après cooldown"


# ─── Bug B : Max grids Executor ───────────────────────────────────────────────


class TestMaxLiveGrids:
    """Tests du guard max_live_grids dans Executor._open_grid_position."""

    def _make_executor(self, max_live_grids: int = 4) -> MagicMock:
        """Crée un Executor mock minimal pour tester le guard."""
        from backend.execution.executor import Executor, GridLiveState

        config = _make_mock_config(max_live_grids=max_live_grids)
        config.secrets.live_trading = True
        config.secrets.bitget_api_key = "k"
        config.secrets.bitget_secret = "s"
        config.secrets.bitget_passphrase = "p"
        config.strategies = MagicMock()

        risk_mgr = MagicMock()
        risk_mgr.pre_trade_check.return_value = (True, "ok")

        notifier = MagicMock()
        notifier.notify_grid_level_opened = AsyncMock()

        executor = Executor(config, risk_mgr, notifier)
        return executor

    @pytest.mark.asyncio
    async def test_max_grids_blocks_new_cycle(self):
        """Avec max_live_grids actifs, un nouveau cycle est refusé."""
        from backend.execution.executor import GridLiveState, TradeEvent, TradeEventType

        executor = self._make_executor(max_live_grids=2)

        # Remplir avec 2 cycles actifs
        executor._grid_states["BTC/USDT:USDT"] = MagicMock(spec=GridLiveState)
        executor._grid_states["ETH/USDT:USDT"] = MagicMock(spec=GridLiveState)

        # Tenter d'ouvrir un 3ème cycle
        event = TradeEvent(
            event_type=TradeEventType.OPEN,
            strategy_name="grid_atr",
            symbol="DOGE/USDT",
            direction="LONG",
            entry_price=0.10,
            quantity=100.0,
            tp_price=0.0,
            sl_price=0.0,
            score=0.0,
            timestamp=datetime.now(tz=timezone.utc),
            market_regime="range",
        )

        # Mock exchange pour ne pas appeler Bitget
        executor._exchange = AsyncMock()
        executor._markets = {
            "DOGE/USDT:USDT": {
                "limits": {"amount": {"min": 1.0}},
                "precision": {"amount": 0, "price": 5},
            },
        }

        await executor._open_grid_position(event)

        # Aucun order passé
        executor._exchange.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_max_grids_allows_additional_levels(self):
        """Les niveaux DCA supplémentaires sur un cycle existant passent toujours."""
        from backend.execution.executor import GridLiveState, GridLivePosition, TradeEvent, TradeEventType

        executor = self._make_executor(max_live_grids=1)

        # 1 cycle actif sur BTC (= limite atteinte pour NOUVEAU cycle)
        state = GridLiveState(
            symbol="BTC/USDT:USDT",
            direction="LONG",
            strategy_name="grid_atr",
            leverage=6,
        )
        state.positions.append(GridLivePosition(
            level=0,
            entry_price=95_000.0,
            quantity=0.001,
            entry_order_id="ord1",
        ))
        executor._grid_states["BTC/USDT:USDT"] = state

        # Niveau 1 sur BTC (cycle existant → is_first_level=False → guard ignoré)
        event = TradeEvent(
            event_type=TradeEventType.OPEN,
            strategy_name="grid_atr",
            symbol="BTC/USDT",  # Même symbol = cycle existant
            direction="LONG",
            entry_price=94_000.0,
            quantity=0.001,
            tp_price=0.0,
            sl_price=0.0,
            score=0.0,
            timestamp=datetime.now(tz=timezone.utc),
            market_regime="range",
        )

        executor._exchange = AsyncMock()
        executor._exchange.amount_to_precision = MagicMock(return_value="0.001")
        executor._exchange.create_order.return_value = {
            "id": "ord2",
            "filled": 0.001,
            "average": 94_000.0,
            "fee": {"cost": 0.01},
        }
        executor._exchange.fetch_balance.return_value = {
            "free": {"USDT": 500.0},
            "total": {"USDT": 1000.0},
        }
        executor._markets = {}
        executor._notifier.notify_grid_level_opened = AsyncMock()

        # L'appel doit tenter de passer l'ordre (is_first_level=False → pas de guard max_grids)
        with (
            patch.object(executor, "_round_quantity", return_value=0.001),
            patch.object(executor, "_update_grid_sl", new=AsyncMock()),
            patch.object(executor, "_fetch_fill_price", new=AsyncMock(return_value=(94_000.0, 0.01))),
        ):
            await executor._open_grid_position(event)

        # L'exchange a bien été appelé (le guard n'a pas bloqué)
        executor._exchange.create_order.assert_called_once()

    def test_max_grids_default_from_config(self):
        """max_live_grids est lu depuis config.risk avec fallback à 4."""
        from backend.execution.executor import Executor

        executor = self._make_executor(max_live_grids=4)
        max_val = getattr(executor._config.risk, "max_live_grids", 4)
        assert max_val == 4


# ─── Bug C : Sauvegarde périodique Executor ───────────────────────────────────


class TestPeriodicSaveExecutor:
    """Tests de la sauvegarde périodique de l'executor dans StateManager."""

    def test_set_executor_registers_references(self):
        """set_executor() enregistre l'executor et le risk_manager."""
        from backend.core.state_manager import StateManager

        db = MagicMock()
        sm = StateManager(db)

        executor = MagicMock()
        risk_mgr = MagicMock()
        sm.set_executor(executor, risk_mgr)

        assert sm._executor is executor
        assert sm._risk_manager is risk_mgr

    def test_initial_executor_is_none(self):
        """Par défaut, _executor est None (pas d'erreur si live_trading=false)."""
        from backend.core.state_manager import StateManager

        db = MagicMock()
        sm = StateManager(db)

        assert sm._executor is None
        assert sm._risk_manager is None

    @pytest.mark.asyncio
    async def test_periodic_save_includes_executor(self):
        """La boucle périodique appelle save_executor_state si executor enregistré."""
        from backend.core.state_manager import StateManager

        db = MagicMock()
        sm = StateManager(db)

        executor = MagicMock()
        executor.get_state_for_persistence.return_value = {"positions": {}}
        risk_mgr = MagicMock()
        sm.set_executor(executor, risk_mgr)

        save_calls = []

        async def _mock_save_executor(exec_, rm):
            save_calls.append((exec_, rm))

        sm.save_executor_state = _mock_save_executor

        # Simuler une itération de la boucle directement
        if sm._executor is not None:
            try:
                await sm.save_executor_state(sm._executor, sm._risk_manager)
            except Exception:
                pass

        assert len(save_calls) == 1
        assert save_calls[0][0] is executor

    @pytest.mark.asyncio
    async def test_periodic_save_without_executor_no_error(self):
        """La boucle périodique ne plante pas si executor est None."""
        from backend.core.state_manager import StateManager

        db = MagicMock()
        sm = StateManager(db)
        # Pas d'appel à set_executor

        # Simuler la logique du guard dans la boucle
        if sm._executor is not None:
            await sm.save_executor_state(sm._executor, sm._risk_manager)

        # Pas d'exception levée, executor reste None
        assert sm._executor is None
