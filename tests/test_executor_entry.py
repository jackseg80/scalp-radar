"""Tests Phase 1 — Entrées autonomes de l'Executor.

Vérifie que _on_candle() évalue les niveaux grid et ouvre des positions
indépendamment du Simulator paper.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.models import Candle, Direction, TimeFrame
from backend.execution.executor import (
    Executor,
    GridLivePosition,
    GridLiveState,
    TradeEvent,
    TradeEventType,
    to_futures_symbol,
)
from backend.strategies.base import StrategyContext
from backend.strategies.base_grid import GridLevel, GridState


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_config() -> MagicMock:
    config = MagicMock()
    config.secrets.live_trading = True
    config.secrets.bitget_api_key = "k"
    config.secrets.bitget_secret = "s"
    config.secrets.bitget_passphrase = "p"
    config.risk.position.default_leverage = 6
    config.risk.position.max_concurrent_positions = 5
    config.risk.fees.taker_percent = 0.06
    config.risk.fees.maker_percent = 0.02
    config.risk.slippage.default_estimate_percent = 0.05
    config.risk.max_live_grids = 4
    config.risk.initial_capital = 1000.0
    config.risk.max_margin_ratio = 0.70
    config.risk.kill_switch.max_session_loss_percent = 5.0
    config.risk.kill_switch.grid_max_session_loss_percent = 25.0
    config.assets = [MagicMock(symbol="BTC/USDT", min_order_size=0.001)]
    config.strategies.grid_atr.leverage = 6
    config.strategies.grid_atr.timeframe = "1h"
    config.strategies.grid_atr.sl_percent = 20.0
    return config


def _make_exchange() -> AsyncMock:
    exchange = AsyncMock()
    exchange.fetch_balance = AsyncMock(return_value={
        "free": {"USDT": 900},
        "total": {"USDT": 1000},
    })
    exchange.fetch_positions = AsyncMock(return_value=[])
    exchange.fetch_open_orders = AsyncMock(return_value=[])
    exchange.create_order = AsyncMock(return_value={
        "id": "order_1",
        "filled": 0.01,
        "average": 50_000.0,
        "status": "closed",
        "fee": {"cost": 0.03},
    })
    exchange.cancel_order = AsyncMock()
    exchange.set_leverage = AsyncMock()
    exchange.amount_to_precision = MagicMock(side_effect=lambda _sym, qty: f"{qty:.3f}")
    exchange.close = AsyncMock()
    return exchange


def _make_strategy(
    *, levels: list[GridLevel] | None = None, max_pos: int = 3,
) -> MagicMock:
    strategy = MagicMock()
    strategy._config = MagicMock()
    strategy._config.timeframe = "1h"
    strategy._config.leverage = 6
    strategy._config.sl_percent = 20.0
    strategy._config.per_asset = {}
    strategy.name = "grid_atr"
    strategy.max_positions = max_pos
    strategy.compute_grid = MagicMock(return_value=levels or [])
    strategy.should_close_all = MagicMock(return_value=None)
    strategy.get_tp_price = MagicMock(return_value=float("nan"))
    strategy.get_sl_price = MagicMock(return_value=float("nan"))
    return strategy


def _make_candle(
    close: float = 50_000.0,
    low: float | None = None,
    high: float | None = None,
    ts: datetime | None = None,
) -> Candle:
    return Candle(
        timestamp=ts or datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        open=close,
        high=high if high is not None else close * 1.01,
        low=low if low is not None else close * 0.99,
        close=close,
        volume=100.0,
        symbol="BTC/USDT",
        timeframe=TimeFrame.H1,
    )


def _make_ctx(
    *, close: float = 50_000.0, sma: float = 49_000.0,
) -> StrategyContext:
    return StrategyContext(
        symbol="BTC/USDT",
        timestamp=datetime.now(tz=timezone.utc),
        candles={},
        indicators={"1h": {"close": close, "sma": sma, "regime": "ranging"}},
        current_position=None,
        capital=10_000.0,
        config=MagicMock(),
    )


def _make_executor(
    *, strategy: MagicMock | None = None, ctx: StrategyContext | None = None,
) -> Executor:
    config = _make_config()
    rm = MagicMock()
    rm.pre_trade_check = MagicMock(return_value=(True, ""))
    rm.is_kill_switch_triggered = False
    rm.register_position = MagicMock()
    rm.record_balance_snapshot = MagicMock()
    notifier = AsyncMock()

    executor = Executor(config, rm, notifier)
    executor._exchange = _make_exchange()
    executor._markets = {
        "BTC/USDT:USDT": {
            "limits": {"amount": {"min": 0.001}},
            "precision": {"amount": 3},
        },
    }
    executor._running = True
    executor._connected = True
    executor._exchange_balance = 1000.0
    executor._balance_bootstrapped = True

    strat = strategy or _make_strategy()
    executor._strategies = {"grid_atr": strat}

    sim = MagicMock()
    sim.get_runner_context = MagicMock(return_value=ctx or _make_ctx())
    executor._simulator = sim

    return executor


# ── TestOnCandle ─────────────────────────────────────────────────────────


class TestOnCandle:
    """Tests du cœur de _on_candle() : triggers, skips, edge cases."""

    @pytest.mark.asyncio
    async def test_skips_when_not_running(self):
        executor = _make_executor()
        executor._running = False
        await executor._on_candle("BTC/USDT", "1h", _make_candle())
        executor._simulator.get_runner_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_wrong_timeframe(self):
        executor = _make_executor()
        await executor._on_candle("BTC/USDT", "5m", _make_candle())
        executor._simulator.get_runner_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_simulator_not_ready(self):
        executor = _make_executor()
        executor._simulator.get_runner_context.return_value = None
        await executor._on_candle("BTC/USDT", "1h", _make_candle())
        # Pas de crash, compute_grid jamais appelé
        executor._strategies["grid_atr"].compute_grid.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_indicators_incomplete(self):
        ctx = _make_ctx()
        ctx.indicators["1h"] = {}  # Pas de sma/close
        executor = _make_executor(ctx=ctx)
        await executor._on_candle("BTC/USDT", "1h", _make_candle())
        executor._strategies["grid_atr"].compute_grid.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_full_grid(self):
        strat = _make_strategy(max_pos=2)
        executor = _make_executor(strategy=strat)
        # Grille avec 2 positions (= max)
        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT", direction="LONG", strategy_name="grid_atr",
            leverage=6, positions=[
                GridLivePosition(level=0, entry_price=50000, quantity=0.01, entry_order_id="e0"),
                GridLivePosition(level=1, entry_price=49000, quantity=0.01, entry_order_id="e1"),
            ],
        )
        await executor._on_candle("BTC/USDT", "1h", _make_candle())
        strat.compute_grid.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_triggered(self):
        """candle.low <= entry_price → _open_grid_position appelé."""
        level = GridLevel(index=0, entry_price=49_600.0, direction=Direction.LONG, size_fraction=0.25)
        strat = _make_strategy(levels=[level])
        executor = _make_executor(strategy=strat)
        # candle.low = 50000*0.99 = 49500 < 49600 → triggered
        candle = _make_candle(close=50_000.0, low=49_500.0)

        with patch.object(executor, "_open_grid_position", new_callable=AsyncMock) as mock_open:
            await executor._on_candle("BTC/USDT", "1h", candle)
            mock_open.assert_called_once()
            event = mock_open.call_args[0][0]
            assert isinstance(event, TradeEvent)
            assert event.direction == "LONG"
            assert event.quantity > 0

    @pytest.mark.asyncio
    async def test_long_not_triggered(self):
        """candle.low > entry_price → pas d'appel."""
        level = GridLevel(index=0, entry_price=48_000.0, direction=Direction.LONG, size_fraction=0.25)
        strat = _make_strategy(levels=[level])
        executor = _make_executor(strategy=strat)
        candle = _make_candle(close=50_000.0, low=49_500.0)

        with patch.object(executor, "_open_grid_position", new_callable=AsyncMock) as mock_open:
            await executor._on_candle("BTC/USDT", "1h", candle)
            mock_open.assert_not_called()

    @pytest.mark.asyncio
    async def test_short_triggered(self):
        """candle.high >= entry_price → _open_grid_position appelé."""
        level = GridLevel(index=0, entry_price=50_400.0, direction=Direction.SHORT, size_fraction=0.25)
        strat = _make_strategy(levels=[level])
        executor = _make_executor(strategy=strat)
        candle = _make_candle(close=50_000.0, high=50_500.0)

        with patch.object(executor, "_open_grid_position", new_callable=AsyncMock) as mock_open:
            await executor._on_candle("BTC/USDT", "1h", candle)
            mock_open.assert_called_once()
            event = mock_open.call_args[0][0]
            assert event.direction == "SHORT"

    @pytest.mark.asyncio
    async def test_skips_filled_level(self):
        """Level 0 déjà rempli → seul level 1 est évalué."""
        levels = [
            GridLevel(index=0, entry_price=49_600.0, direction=Direction.LONG, size_fraction=0.25),
            GridLevel(index=1, entry_price=49_000.0, direction=Direction.LONG, size_fraction=0.25),
        ]
        strat = _make_strategy(levels=levels, max_pos=3)
        executor = _make_executor(strategy=strat)
        # Level 0 déjà rempli
        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT", direction="LONG", strategy_name="grid_atr",
            leverage=6, positions=[
                GridLivePosition(level=0, entry_price=49600, quantity=0.01, entry_order_id="e0"),
            ],
        )
        # candle.low = 48900 → les deux levels sont dans le range, mais level 0 est skip
        candle = _make_candle(close=50_000.0, low=48_900.0)

        with patch.object(executor, "_open_grid_position", new_callable=AsyncMock) as mock_open:
            await executor._on_candle("BTC/USDT", "1h", candle)
            mock_open.assert_called_once()
            event = mock_open.call_args[0][0]
            assert event.entry_price == 49_000.0

    @pytest.mark.asyncio
    async def test_pending_levels_prevents_double(self):
        """Pending key dans _pending_levels → skip."""
        level = GridLevel(index=0, entry_price=49_600.0, direction=Direction.LONG, size_fraction=0.25)
        strat = _make_strategy(levels=[level])
        executor = _make_executor(strategy=strat)
        executor._pending_levels.add("BTC/USDT:USDT:0")
        candle = _make_candle(close=50_000.0, low=49_500.0)

        with patch.object(executor, "_open_grid_position", new_callable=AsyncMock) as mock_open:
            await executor._on_candle("BTC/USDT", "1h", candle)
            mock_open.assert_not_called()

    @pytest.mark.asyncio
    async def test_zero_balance_skips(self):
        """Balance <= 0 → skip all entries."""
        level = GridLevel(index=0, entry_price=49_600.0, direction=Direction.LONG, size_fraction=0.25)
        strat = _make_strategy(levels=[level])
        executor = _make_executor(strategy=strat)
        executor._exchange_balance = 0.0
        candle = _make_candle(close=50_000.0, low=49_500.0)

        with patch.object(executor, "_open_grid_position", new_callable=AsyncMock) as mock_open:
            await executor._on_candle("BTC/USDT", "1h", candle)
            mock_open.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_live_balance_not_paper(self):
        """Le capital dans le StrategyContext doit être le solde Bitget, pas le paper."""
        level = GridLevel(index=0, entry_price=49_600.0, direction=Direction.LONG, size_fraction=0.25)
        strat = _make_strategy(levels=[level])
        executor = _make_executor(strategy=strat)
        executor._exchange_balance = 500.0  # Balance live
        candle = _make_candle(close=50_000.0, low=49_500.0)

        captured_ctx = {}
        original_compute = strat.compute_grid

        def capture_compute(ctx, gs):
            captured_ctx["capital"] = ctx.capital
            return original_compute(ctx, gs)

        strat.compute_grid = capture_compute

        with patch.object(executor, "_open_grid_position", new_callable=AsyncMock):
            await executor._on_candle("BTC/USDT", "1h", candle)

        assert captured_ctx.get("capital") == 500.0

    @pytest.mark.asyncio
    async def test_uses_live_grid_state_not_paper(self):
        """GridState construit depuis _grid_states (live), pas paper."""
        level = GridLevel(index=1, entry_price=49_000.0, direction=Direction.LONG, size_fraction=0.25)
        strat = _make_strategy(levels=[level], max_pos=3)
        executor = _make_executor(strategy=strat)
        # 1 position live
        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT", direction="LONG", strategy_name="grid_atr",
            leverage=6, positions=[
                GridLivePosition(level=0, entry_price=49600, quantity=0.01, entry_order_id="e0"),
            ],
        )
        candle = _make_candle(close=50_000.0, low=48_900.0)

        captured_gs = {}
        original_compute = strat.compute_grid

        def capture_compute(ctx, gs):
            captured_gs["state"] = gs
            return original_compute(ctx, gs)

        strat.compute_grid = capture_compute

        with patch.object(executor, "_open_grid_position", new_callable=AsyncMock):
            await executor._on_candle("BTC/USDT", "1h", candle)

        assert len(captured_gs["state"].positions) == 1
        assert captured_gs["state"].positions[0].entry_price == 49_600.0

    @pytest.mark.asyncio
    async def test_pending_levels_cleared_after_open(self):
        """Pending key nettoyée dans le finally, succès ou échec."""
        level = GridLevel(index=0, entry_price=49_600.0, direction=Direction.LONG, size_fraction=0.25)
        strat = _make_strategy(levels=[level])
        executor = _make_executor(strategy=strat)
        candle = _make_candle(close=50_000.0, low=49_500.0)

        with patch.object(
            executor, "_open_grid_position",
            new_callable=AsyncMock, side_effect=RuntimeError("test"),
        ):
            await executor._on_candle("BTC/USDT", "1h", candle)

        # Pending key nettoyée malgré l'erreur
        assert "BTC/USDT:USDT:0" not in executor._pending_levels

    @pytest.mark.asyncio
    async def test_quantity_calculated_correctly(self):
        """Quantity = size_fraction * balance * leverage / entry_price."""
        level = GridLevel(index=0, entry_price=50_000.0, direction=Direction.LONG, size_fraction=0.25)
        strat = _make_strategy(levels=[level])
        executor = _make_executor(strategy=strat)
        executor._exchange_balance = 1000.0
        candle = _make_candle(close=50_000.0, low=49_500.0)

        with patch.object(executor, "_open_grid_position", new_callable=AsyncMock) as mock_open:
            await executor._on_candle("BTC/USDT", "1h", candle)
            event = mock_open.call_args[0][0]
            # 0.25 * 1000 * 6 / 50000 = 0.03
            assert event.quantity == pytest.approx(0.03, abs=0.001)


# ── TestBuildGridState ───────────────────────────────────────────────────


class TestLiveStateToGridState:
    """Tests de _live_state_to_grid_state()."""

    def test_empty(self):
        executor = _make_executor()
        gs = executor._live_state_to_grid_state("BTC/USDT:USDT")
        assert gs.positions == []
        assert gs.avg_entry_price == 0
        assert gs.total_quantity == 0

    def test_from_live_positions(self):
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT", direction="LONG", strategy_name="grid_atr",
            leverage=6, positions=[
                GridLivePosition(level=0, entry_price=50_000.0, quantity=0.01, entry_order_id="e0"),
                GridLivePosition(level=1, entry_price=49_000.0, quantity=0.01, entry_order_id="e1"),
            ],
        )
        gs = executor._live_state_to_grid_state("BTC/USDT:USDT", current_price=51_000.0)
        assert len(gs.positions) == 2
        assert gs.avg_entry_price == pytest.approx(49_500.0)
        assert gs.total_quantity == pytest.approx(0.02)
        # LONG, prix monte → unrealized > 0
        assert gs.unrealized_pnl > 0

    def test_single_position(self):
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT", direction="LONG", strategy_name="grid_atr",
            leverage=6, positions=[
                GridLivePosition(level=0, entry_price=50_000.0, quantity=0.01, entry_order_id="e0"),
            ],
        )
        gs = executor._live_state_to_grid_state("BTC/USDT:USDT")
        assert gs.avg_entry_price == 50_000.0
        assert gs.unrealized_pnl == 0  # Pas de current_price → 0


# ── TestBalanceBootstrap ─────────────────────────────────────────────────


class TestBalanceBootstrap:
    """Tests de _ensure_balance()."""

    @pytest.mark.asyncio
    async def test_fetches_on_first_call(self):
        executor = _make_executor()
        executor._balance_bootstrapped = False
        executor._exchange_balance = None
        balance = await executor._ensure_balance()
        assert balance == 1000.0
        assert executor._balance_bootstrapped is True
        executor._exchange.fetch_balance.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_cache_after_bootstrap(self):
        executor = _make_executor()
        executor._balance_bootstrapped = True
        executor._exchange_balance = 800.0
        balance = await executor._ensure_balance()
        assert balance == 800.0
        executor._exchange.fetch_balance.assert_not_called()

    @pytest.mark.asyncio
    async def test_subtracts_pending_notional(self):
        executor = _make_executor()
        executor._exchange_balance = 1000.0
        executor._pending_notional = 200.0
        balance = await executor._ensure_balance()
        assert balance == 800.0


# ── TestPendingNotional ──────────────────────────────────────────────────


class TestPendingNotional:
    """Tests du mécanisme _pending_notional."""

    @pytest.mark.asyncio
    async def test_increments_on_trigger(self):
        level = GridLevel(index=0, entry_price=50_000.0, direction=Direction.LONG, size_fraction=0.25)
        strat = _make_strategy(levels=[level])
        executor = _make_executor(strategy=strat)
        executor._exchange_balance = 1000.0
        candle = _make_candle(close=50_000.0, low=49_500.0)

        with patch.object(executor, "_open_grid_position", new_callable=AsyncMock):
            await executor._on_candle("BTC/USDT", "1h", candle)

        assert executor._pending_notional > 0

    @pytest.mark.asyncio
    async def test_not_reset_while_orders_pending(self):
        executor = _make_executor()
        executor._pending_levels = {"BTC/USDT:USDT:0"}
        executor._pending_notional = 250.0
        executor._exchange_balance = 1000.0

        await executor.refresh_balance()

        # Pas de reset car pending_levels non vide
        assert executor._pending_notional == 250.0

    @pytest.mark.asyncio
    async def test_resets_when_no_pending_orders(self):
        executor = _make_executor()
        executor._pending_levels = set()
        executor._pending_notional = 250.0
        executor._exchange_balance = 1000.0

        await executor.refresh_balance()

        assert executor._pending_notional == 0.0
        assert executor._balance_bootstrapped is True
