"""Tests Executor Autonome — exit monitor + sync boot + hardening OPEN."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.execution.executor import (
    Executor,
    GridLivePosition,
    GridLiveState,
    TradeEvent,
    TradeEventType,
)


# ─── Helpers ──────────────────────────────────────────────────────────


def _make_config():
    config = MagicMock()
    config.secrets.live_trading = True
    config.secrets.bitget_api_key = "test"
    config.secrets.bitget_secret = "test"
    config.secrets.bitget_passphrase = "test"
    config.risk.position.default_leverage = 3
    config.risk.position.max_concurrent_positions = 5
    config.risk.fees.taker_percent = 0.06
    config.risk.fees.maker_percent = 0.02
    config.risk.slippage.default_estimate_percent = 0.05
    config.risk.margin.mode = "cross"
    config.risk.margin.min_free_margin_percent = 20
    config.risk.max_live_grids = 4
    config.risk.initial_capital = 1000.0
    config.assets = [MagicMock(symbol="BTC/USDT", min_order_size=0.001)]
    config.correlation_groups = {}
    # Strategy config
    config.strategies.grid_atr.leverage = 3
    config.strategies.grid_atr.sl_percent = 20.0
    config.strategies.grid_atr.timeframe = "1h"
    config.strategies.grid_atr.ma_period = 7
    config.strategies.grid_boltrend.leverage = 3
    config.strategies.grid_boltrend.sl_percent = 15.0
    config.strategies.grid_boltrend.timeframe = "1h"
    return config


def _make_exchange():
    exchange = AsyncMock()
    exchange.load_markets = AsyncMock(return_value={"BTC/USDT:USDT": {}})
    exchange.fetch_balance = AsyncMock(return_value={
        "free": {"USDT": 900},
        "total": {"USDT": 1000},
    })
    exchange.create_order = AsyncMock(return_value={
        "id": "order123",
        "filled": 0.01,
        "average": 50000.0,
        "status": "closed",
        "fee": {"cost": 0.03},
    })
    exchange.cancel_order = AsyncMock()
    exchange.fetch_order = AsyncMock(return_value={
        "id": "order123", "status": "open", "average": 50000.0,
    })
    exchange.fetch_positions = AsyncMock(return_value=[])
    exchange.fetch_open_orders = AsyncMock(return_value=[])
    exchange.set_leverage = AsyncMock()
    exchange.set_margin_mode = AsyncMock()
    exchange.set_position_mode = AsyncMock()
    exchange.amount_to_precision = MagicMock(side_effect=lambda sym, qty: str(qty))
    exchange.price_to_precision = MagicMock(side_effect=lambda sym, price: str(price))
    exchange.close = AsyncMock()
    return exchange


def _make_notifier():
    return AsyncMock()


def _make_risk_manager():
    rm = MagicMock()
    rm.pre_trade_check = MagicMock(return_value=(True, ""))
    rm.register_position = MagicMock()
    rm.unregister_position = MagicMock()
    rm.record_trade_result = MagicMock()
    rm.get_status = MagicMock(return_value={})
    rm.get_state = MagicMock(return_value={})
    return rm


def _make_executor(config=None, exchange=None, notifier=None, risk_manager=None):
    config = config or _make_config()
    exchange = exchange or _make_exchange()
    notifier = notifier or _make_notifier()
    risk_manager = risk_manager or _make_risk_manager()

    executor = Executor(config, risk_manager, notifier)
    executor._exchange = exchange
    executor._markets = {"BTC/USDT:USDT": {"limits": {"amount": {"min": 0.001}}}}
    executor._running = True
    executor._connected = True
    return executor


def _make_grid_state(
    symbol="BTC/USDT:USDT",
    direction="LONG",
    strategy_name="grid_atr",
    entry_price=50000.0,
    quantity=0.01,
    levels=1,
):
    """Crée un GridLiveState avec N niveaux."""
    positions = []
    for i in range(levels):
        positions.append(GridLivePosition(
            level=i,
            entry_price=entry_price * (1 - 0.02 * i),
            quantity=quantity,
            entry_order_id=f"entry_{i}",
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ))
    return GridLiveState(
        symbol=symbol,
        direction=direction,
        strategy_name=strategy_name,
        leverage=3,
        positions=positions,
        sl_order_id="sl_123",
        sl_price=entry_price * 0.8,
    )


@dataclass
class FakeCandle:
    """Candle minimale pour les tests."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 100.0


def _make_data_engine(symbol="BTC/USDT", timeframe="1h", closes=None):
    """Crée un DataEngine mock avec un buffer de candles."""
    if closes is None:
        closes = [50000.0 + i * 100 for i in range(50)]

    from datetime import timedelta

    base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = [
        FakeCandle(
            timestamp=base_ts + timedelta(hours=i),
            open=c - 10, high=c + 50, low=c - 50, close=c,
        )
        for i, c in enumerate(closes)
    ]

    buffers = defaultdict(lambda: defaultdict(list))
    buffers[symbol][timeframe] = candles

    engine = MagicMock()
    engine._buffers = buffers
    return engine


def _make_strategy(should_close_result=None, compute_indicators_result=None):
    """Crée une stratégie mock avec should_close_all configurable."""
    strategy = MagicMock()
    strategy.should_close_all = MagicMock(return_value=should_close_result)
    strategy._config = MagicMock()
    strategy._config.timeframe = "1h"
    strategy._config.ma_period = 7

    if compute_indicators_result is not None:
        strategy.compute_live_indicators = MagicMock(return_value=compute_indicators_result)
    else:
        strategy.compute_live_indicators = MagicMock(return_value={})

    return strategy


# ═══════════════════════════════════════════════════════════════════════
# Bloc 1 — Exit autonome via should_close_all
# ═══════════════════════════════════════════════════════════════════════


class TestExitAutonomous:

    @pytest.mark.asyncio
    async def test_exit_calls_should_close_all(self):
        """should_close_all retourne 'tp_global' → _close_grid_cycle appelé."""
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = _make_grid_state()
        executor._data_engine = _make_data_engine()
        executor._strategies = {"grid_atr": _make_strategy(should_close_result="tp_global")}

        # Mock _close_grid_cycle pour vérifier l'appel
        executor._close_grid_cycle = AsyncMock()

        await executor._check_grid_exit("BTC/USDT:USDT")

        executor._close_grid_cycle.assert_called_once()
        event = executor._close_grid_cycle.call_args[0][0]
        assert event.exit_reason == "tp_global"
        assert event.event_type == TradeEventType.CLOSE

    @pytest.mark.asyncio
    async def test_exit_no_close_when_should_close_returns_none(self):
        """should_close_all retourne None → pas de close."""
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = _make_grid_state()
        executor._data_engine = _make_data_engine()
        executor._strategies = {"grid_atr": _make_strategy(should_close_result=None)}

        executor._close_grid_cycle = AsyncMock()

        await executor._check_grid_exit("BTC/USDT:USDT")

        executor._close_grid_cycle.assert_not_called()

    @pytest.mark.asyncio
    async def test_exit_sl_global_via_should_close_all(self):
        """should_close_all retourne 'sl_global' → close."""
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = _make_grid_state()
        executor._data_engine = _make_data_engine()
        executor._strategies = {"grid_atr": _make_strategy(should_close_result="sl_global")}

        executor._close_grid_cycle = AsyncMock()

        await executor._check_grid_exit("BTC/USDT:USDT")

        executor._close_grid_cycle.assert_called_once()
        event = executor._close_grid_cycle.call_args[0][0]
        assert event.exit_reason == "sl_global"

    @pytest.mark.asyncio
    async def test_exit_no_data_engine(self):
        """data_engine=None → skip silencieux."""
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = _make_grid_state()
        executor._data_engine = None
        executor._strategies = {"grid_atr": _make_strategy(should_close_result="tp_global")}

        executor._close_grid_cycle = AsyncMock()

        await executor._check_grid_exit("BTC/USDT:USDT")

        executor._close_grid_cycle.assert_not_called()

    @pytest.mark.asyncio
    async def test_exit_no_strategy(self):
        """Stratégie introuvable → skip."""
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = _make_grid_state()
        executor._data_engine = _make_data_engine()
        executor._strategies = {}  # Pas de stratégie enregistrée

        executor._close_grid_cycle = AsyncMock()

        await executor._check_grid_exit("BTC/USDT:USDT")

        executor._close_grid_cycle.assert_not_called()

    @pytest.mark.asyncio
    async def test_exit_empty_buffer(self):
        """Pas de candles dans le buffer → skip."""
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = _make_grid_state()
        # Buffer vide
        engine = MagicMock()
        engine._buffers = defaultdict(lambda: defaultdict(list))
        executor._data_engine = engine
        executor._strategies = {"grid_atr": _make_strategy(should_close_result="tp_global")}

        executor._close_grid_cycle = AsyncMock()

        await executor._check_grid_exit("BTC/USDT:USDT")

        executor._close_grid_cycle.assert_not_called()

    @pytest.mark.asyncio
    async def test_exit_builds_correct_context(self):
        """Vérifier que ctx.indicators contient close et sma."""
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = _make_grid_state()

        # Buffer avec prix connus
        closes = [100.0] * 10
        executor._data_engine = _make_data_engine(closes=closes)

        # Capturer l'appel à should_close_all
        captured_args = {}

        def capture_should_close(ctx, grid_state):
            captured_args["ctx"] = ctx
            captured_args["grid_state"] = grid_state
            return None

        strategy = _make_strategy()
        strategy.should_close_all = capture_should_close
        executor._strategies = {"grid_atr": strategy}

        await executor._check_grid_exit("BTC/USDT:USDT")

        ctx = captured_args["ctx"]
        assert ctx.indicators["1h"]["close"] == 100.0
        assert ctx.indicators["1h"]["sma"] == pytest.approx(100.0, rel=1e-6)
        assert ctx.symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_exit_builds_correct_grid_state(self):
        """GridState.avg_entry_price correspond à l'état live."""
        executor = _make_executor()
        live_state = _make_grid_state(entry_price=50000.0, levels=2)
        executor._grid_states["BTC/USDT:USDT"] = live_state

        executor._data_engine = _make_data_engine()

        captured = {}

        def capture(ctx, grid_state):
            captured["grid_state"] = grid_state
            return None

        strategy = _make_strategy()
        strategy.should_close_all = capture
        executor._strategies = {"grid_atr": strategy}

        await executor._check_grid_exit("BTC/USDT:USDT")

        gs = captured["grid_state"]
        assert gs.avg_entry_price == live_state.avg_entry_price
        assert gs.total_quantity == live_state.total_quantity
        assert len(gs.positions) == 2

    @pytest.mark.asyncio
    async def test_exit_uses_per_asset_ma_period(self):
        """Exit monitor utilise le ma_period per_asset, pas le default."""
        config = _make_config()
        # Default ma_period=14 (top-level) → pas assez de candles (8)
        config.strategies.grid_atr.ma_period = 14
        # Per_asset DYDX/USDT → ma_period=7 → assez de candles (8)
        config.strategies.grid_atr.get_params_for_symbol = MagicMock(
            return_value={"ma_period": 7, "timeframe": "1h", "sl_percent": 20.0},
        )

        executor = _make_executor(config=config)
        executor._grid_states["DYDX/USDT:USDT"] = _make_grid_state(
            symbol="DYDX/USDT:USDT", entry_price=1.0,
        )
        # Seulement 8 candles — assez pour ma_period=7, pas pour ma_period=14
        closes = [1.0] * 8
        executor._data_engine = _make_data_engine(
            symbol="DYDX/USDT", closes=closes,
        )

        captured = {}

        def capture(ctx, grid_state):
            captured["ctx"] = ctx
            return None

        strategy = _make_strategy()
        # compute_live_indicators retourne {} → SMA fallback activé
        strategy.compute_live_indicators = MagicMock(return_value={})
        strategy.should_close_all = capture
        executor._strategies = {"grid_atr": strategy}

        await executor._check_grid_exit("DYDX/USDT:USDT")

        # SMA doit être calculée (per_asset ma_period=7 ≤ 8 candles)
        ctx = captured["ctx"]
        assert "sma" in ctx.indicators["1h"], (
            "SMA absente — ma_period per_asset ignoré, default trop grand"
        )
        assert ctx.indicators["1h"]["sma"] == pytest.approx(1.0)

        # Vérifier que get_params_for_symbol a été appelé avec le bon symbol
        config.strategies.grid_atr.get_params_for_symbol.assert_called_once_with(
            "DYDX/USDT",
        )

    @pytest.mark.asyncio
    async def test_exit_idempotent(self):
        """Double appel close → le 2ème est no-op (state supprimé)."""
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = _make_grid_state()
        executor._data_engine = _make_data_engine()
        executor._strategies = {"grid_atr": _make_strategy(should_close_result="tp_global")}

        call_count = 0

        async def mock_close(event):
            nonlocal call_count
            call_count += 1
            # Simuler ce que _close_grid_cycle fait : supprimer le state
            executor._grid_states.pop("BTC/USDT:USDT", None)

        executor._close_grid_cycle = mock_close

        # Premier appel → close
        await executor._check_grid_exit("BTC/USDT:USDT")
        assert call_count == 1

        # Deuxième appel → state supprimé → no-op
        await executor._check_grid_exit("BTC/USDT:USDT")
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_exit_monitor_loop_catches_errors(self):
        """Exception dans check → log error, boucle continue."""
        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = _make_grid_state()

        call_count = 0

        async def failing_check():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Test error")
            # Deuxième appel : stop la boucle
            executor._running = False

        executor._check_all_live_exits = failing_check
        executor._EXIT_CHECK_INTERVAL = 0.01  # Rapide pour le test

        await executor._exit_monitor_loop()

        assert call_count >= 2  # La boucle a survécu à l'erreur


# ═══════════════════════════════════════════════════════════════════════
# Bloc 2 — Sync boot
# ═══════════════════════════════════════════════════════════════════════


def _make_mock_runner(name="grid_atr", capital=1000.0, leverage=3, positions=None):
    """Crée un mock GridStrategyRunner."""
    from backend.backtesting.simulator import GridStrategyRunner

    runner = MagicMock(spec=GridStrategyRunner)
    runner.name = name
    runner._capital = capital
    runner._leverage = leverage
    runner._positions = positions if positions is not None else {}
    return runner


def _make_mock_simulator(runners=None):
    """Crée un mock Simulator avec des runners."""
    simulator = MagicMock()
    simulator.runners = runners or []
    return simulator


class TestSyncBoot:

    @pytest.mark.asyncio
    async def test_sync_injects_live_to_paper(self):
        """Position live sans paper → injectée."""
        from backend.execution.sync import sync_live_to_paper

        runner = _make_mock_runner(positions={})
        simulator = _make_mock_simulator(runners=[runner])

        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = _make_grid_state(
            entry_price=50000.0, quantity=0.01, levels=1,
        )

        await sync_live_to_paper(executor, simulator)

        assert "BTC/USDT" in runner._positions
        assert len(runner._positions["BTC/USDT"]) == 1
        pos = runner._positions["BTC/USDT"][0]
        assert pos.entry_price == 50000.0
        assert pos.quantity == 0.01

    @pytest.mark.asyncio
    async def test_sync_removes_paper_without_live(self):
        """Position paper sans live → supprimée, marge rendue."""
        from backend.execution.sync import sync_live_to_paper
        from backend.core.models import Direction
        from backend.strategies.base_grid import GridPosition

        paper_pos = GridPosition(
            level=0, direction=Direction.LONG,
            entry_price=50000.0, quantity=0.01,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            entry_fee=0.03,
        )
        runner = _make_mock_runner(
            capital=800.0, leverage=3,
            positions={"BTC/USDT": [paper_pos]},
        )
        simulator = _make_mock_simulator(runners=[runner])

        executor = _make_executor()
        # Pas de grid_states → pas de position live

        await sync_live_to_paper(executor, simulator)

        assert runner._positions["BTC/USDT"] == []
        # Marge rendue = 50000 * 0.01 / 3 ≈ 166.67
        expected_margin = 50000.0 * 0.01 / 3
        assert runner._capital == pytest.approx(800.0 + expected_margin, rel=1e-4)

    @pytest.mark.asyncio
    async def test_sync_keeps_matching_positions(self):
        """Position live + paper → pas touchée."""
        from backend.execution.sync import sync_live_to_paper
        from backend.core.models import Direction
        from backend.strategies.base_grid import GridPosition

        paper_pos = GridPosition(
            level=0, direction=Direction.LONG,
            entry_price=50000.0, quantity=0.01,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            entry_fee=0.03,
        )
        runner = _make_mock_runner(
            capital=800.0,
            positions={"BTC/USDT": [paper_pos]},
        )
        simulator = _make_mock_simulator(runners=[runner])

        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = _make_grid_state()

        await sync_live_to_paper(executor, simulator)

        # Paper a déjà la position → pas touchée
        assert len(runner._positions["BTC/USDT"]) == 1
        assert runner._capital == 800.0

    @pytest.mark.asyncio
    async def test_sync_capital_adjusted_on_inject(self):
        """Capital paper réduit de marge injectée."""
        from backend.execution.sync import sync_live_to_paper

        initial_capital = 1000.0
        runner = _make_mock_runner(capital=initial_capital, leverage=3, positions={})
        simulator = _make_mock_simulator(runners=[runner])

        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = _make_grid_state(
            entry_price=50000.0, quantity=0.01, levels=1,
        )

        await sync_live_to_paper(executor, simulator)

        margin = 50000.0 * 0.01 / 3
        assert runner._capital == pytest.approx(initial_capital - margin, rel=1e-4)

    @pytest.mark.asyncio
    async def test_sync_capital_returned_on_remove(self):
        """Capital paper augmenté de marge retirée."""
        from backend.execution.sync import sync_live_to_paper
        from backend.core.models import Direction
        from backend.strategies.base_grid import GridPosition

        paper_pos = GridPosition(
            level=0, direction=Direction.LONG,
            entry_price=60000.0, quantity=0.02,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            entry_fee=0.072,
        )
        initial_capital = 500.0
        runner = _make_mock_runner(
            capital=initial_capital, leverage=3,
            positions={"ETH/USDT": [paper_pos]},
        )
        simulator = _make_mock_simulator(runners=[runner])

        executor = _make_executor()
        # Pas de live sur ETH

        await sync_live_to_paper(executor, simulator)

        margin = 60000.0 * 0.02 / 3
        assert runner._capital == pytest.approx(initial_capital + margin, rel=1e-4)

    @pytest.mark.asyncio
    async def test_sync_multi_strategy(self):
        """grid_atr + grid_boltrend → bon runner chacun."""
        from backend.execution.sync import sync_live_to_paper

        runner_atr = _make_mock_runner(name="grid_atr", capital=1000.0, positions={})
        runner_bolt = _make_mock_runner(name="grid_boltrend", capital=1000.0, positions={})
        simulator = _make_mock_simulator(runners=[runner_atr, runner_bolt])

        executor = _make_executor()
        executor._grid_states["BTC/USDT:USDT"] = _make_grid_state(
            strategy_name="grid_atr",
        )
        executor._grid_states["ETH/USDT:USDT"] = _make_grid_state(
            symbol="ETH/USDT:USDT", strategy_name="grid_boltrend",
            entry_price=3000.0,
        )

        await sync_live_to_paper(executor, simulator)

        assert "BTC/USDT" in runner_atr._positions
        assert "ETH/USDT" in runner_bolt._positions

    @pytest.mark.asyncio
    async def test_sync_no_live_positions(self):
        """0 positions live → log info, rien fait."""
        from backend.execution.sync import sync_live_to_paper

        runner = _make_mock_runner(positions={"BTC/USDT": []})
        simulator = _make_mock_simulator(runners=[runner])

        executor = _make_executor()
        # Pas de grid_states

        await sync_live_to_paper(executor, simulator)

        # Rien ne change
        assert runner._positions == {"BTC/USDT": []}

    @pytest.mark.asyncio
    async def test_sync_creates_grid_states_from_exchange(self):
        """Si executor._grid_states est vide, fetch Bitget et crée les GridLiveState."""
        from backend.execution.sync import sync_live_to_paper

        runner = _make_mock_runner(name="grid_atr", capital=1000.0, positions={})
        simulator = _make_mock_simulator(runners=[runner])

        exchange = _make_exchange()
        exchange.fetch_positions = AsyncMock(return_value=[
            {
                "symbol": "FET/USDT:USDT",
                "contracts": 3812.0,
                "entryPrice": 0.1667,
                "side": "long",
            },
            {
                "symbol": "GALA/USDT:USDT",
                "contracts": 124476.0,
                "entryPrice": 0.003919,
                "side": "long",
            },
            {
                "symbol": "ZERO/USDT:USDT",
                "contracts": 0.0,
                "entryPrice": 1.0,
                "side": "long",
            },
        ])

        executor = _make_executor(exchange=exchange)
        assert len(executor._grid_states) == 0

        await sync_live_to_paper(executor, simulator)

        # 2 positions valides créées (la 3e avec contracts=0 ignorée)
        assert "FET/USDT:USDT" in executor._grid_states
        assert "GALA/USDT:USDT" in executor._grid_states
        assert "ZERO/USDT:USDT" not in executor._grid_states

        fet = executor._grid_states["FET/USDT:USDT"]
        assert fet.strategy_name == "grid_atr"
        assert fet.direction == "LONG"
        assert len(fet.positions) == 1
        assert fet.positions[0].quantity == pytest.approx(3812.0)
        assert fet.positions[0].entry_price == pytest.approx(0.1667)
        assert fet.positions[0].entry_order_id == "restored-from-sync"

        gala = executor._grid_states["GALA/USDT:USDT"]
        assert gala.positions[0].quantity == pytest.approx(124476.0)

    @pytest.mark.asyncio
    async def test_sync_grid_states_not_overwritten_when_populated(self):
        """Si executor._grid_states est déjà peuplé, fetch_positions n'est pas appelé."""
        from backend.execution.sync import sync_live_to_paper

        runner = _make_mock_runner(name="grid_atr", capital=1000.0, positions={})
        simulator = _make_mock_simulator(runners=[runner])

        exchange = _make_exchange()
        exchange.fetch_positions = AsyncMock(return_value=[])

        executor = _make_executor(exchange=exchange)
        # Pré-peupler _grid_states
        existing_state = _make_grid_state(symbol="BTC/USDT:USDT")
        executor._grid_states["BTC/USDT:USDT"] = existing_state

        await sync_live_to_paper(executor, simulator)

        # fetch_positions appelé 1x (pour _reconcile via la branche paper)
        # mais PAS pour _populate (car _grid_states n'était pas vide)
        # La position existante est toujours là
        assert "BTC/USDT:USDT" in executor._grid_states
        assert executor._grid_states["BTC/USDT:USDT"] is existing_state


# ═══════════════════════════════════════════════════════════════════════
# Bloc 3 — Hardening OPEN
# ═══════════════════════════════════════════════════════════════════════


class TestHardeningOpen:

    @pytest.mark.asyncio
    async def test_open_rejected_if_exchange_has_position(self):
        """Position Bitget existante → OPEN ignoré."""
        executor = _make_executor()
        executor._exchange.fetch_positions = AsyncMock(return_value=[
            {"contracts": 0.05, "symbol": "BTC/USDT:USDT"},
        ])

        event = TradeEvent(
            event_type=TradeEventType.OPEN,
            strategy_name="grid_atr",
            symbol="BTC/USDT",
            direction="LONG",
            entry_price=50000.0,
            quantity=0.01,
            tp_price=0.0,
            sl_price=40000.0,
            score=0.8,
            timestamp=datetime.now(tz=timezone.utc),
        )

        await executor._open_grid_position(event)

        # L'ordre d'entrée ne doit PAS être passé
        executor._exchange.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_open_allowed_if_no_exchange_position(self):
        """Pas de position exchange → OPEN normal."""
        executor = _make_executor()
        executor._exchange.fetch_positions = AsyncMock(return_value=[
            {"contracts": 0, "symbol": "BTC/USDT:USDT"},
        ])

        event = TradeEvent(
            event_type=TradeEventType.OPEN,
            strategy_name="grid_atr",
            symbol="BTC/USDT",
            direction="LONG",
            entry_price=50000.0,
            quantity=0.01,
            tp_price=0.0,
            sl_price=40000.0,
            score=0.8,
            timestamp=datetime.now(tz=timezone.utc),
        )

        await executor._open_grid_position(event)

        # L'ordre d'entrée doit être passé
        executor._exchange.create_order.assert_called()

    @pytest.mark.asyncio
    async def test_open_dca_level_on_existing_grid_ok(self):
        """DCA level sur grid existante → passe (pas de check exchange)."""
        executor = _make_executor()
        # Grid déjà active
        executor._grid_states["BTC/USDT:USDT"] = _make_grid_state()

        # fetch_positions ne doit PAS être appelé pour un DCA level
        executor._exchange.fetch_positions = AsyncMock(side_effect=AssertionError("Ne devrait pas être appelé"))

        event = TradeEvent(
            event_type=TradeEventType.OPEN,
            strategy_name="grid_atr",
            symbol="BTC/USDT",
            direction="LONG",
            entry_price=49000.0,
            quantity=0.01,
            tp_price=0.0,
            sl_price=40000.0,
            score=0.8,
            timestamp=datetime.now(tz=timezone.utc),
        )

        await executor._open_grid_position(event)

        # L'ordre d'entrée doit être passé (DCA level)
        executor._exchange.create_order.assert_called()
