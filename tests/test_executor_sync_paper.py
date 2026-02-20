"""Tests synchronisation fermetures live → paper.

Quand l'exit monitor ferme une position live, la position paper correspondante
doit être fermée au même prix et au même moment via Simulator.force_close_grid().
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.backtesting.simulator import GridStrategyRunner, Simulator
from backend.core.grid_position_manager import GridPositionManager
from backend.core.models import Direction
from backend.core.position_manager import PositionManagerConfig, TradeResult
from backend.execution.executor import Executor, TradeEvent, TradeEventType
from backend.execution.executor import GridLivePosition, GridLiveState
from backend.strategies.base_grid import GridPosition


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_gpm_config() -> PositionManagerConfig:
    return PositionManagerConfig(
        leverage=6,
        maker_fee=0.0002,
        taker_fee=0.0006,
        slippage_pct=0.0005,
        high_vol_slippage_mult=2.0,
        max_risk_per_trade=0.02,
    )


def _make_mock_strategy(name="grid_atr") -> MagicMock:
    strategy = MagicMock()
    strategy.name = name
    config = MagicMock()
    config.timeframe = "1h"
    config.ma_period = 7
    config.leverage = 6
    strategy._config = config
    strategy.min_candles = {"1h": 50}
    strategy.max_positions = 4
    strategy.compute_grid.return_value = []
    strategy.should_close_all.return_value = None
    strategy.get_tp_price.return_value = float("nan")
    strategy.get_sl_price.return_value = float("nan")
    strategy.get_current_conditions.return_value = []
    return strategy


def _make_grid_runner(
    strategy=None, initial_capital=10_000.0,
) -> GridStrategyRunner:
    if strategy is None:
        strategy = _make_mock_strategy()
    config = MagicMock()
    config.risk = MagicMock()
    config.risk.kill_switch = MagicMock()
    config.risk.kill_switch.global_max_loss_pct = 45.0
    config.risk.kill_switch.session_max_loss_pct = 25.0
    config.risk.initial_capital = initial_capital
    config.risk.fees = MagicMock()
    config.risk.fees.taker_percent = 0.06
    config.risk.fees.maker_percent = 0.02
    config.risk.fees.slippage_percent = 0.05
    indicator_engine = MagicMock()
    indicator_engine.get_indicators.return_value = {}
    indicator_engine.update = MagicMock()
    gpm = GridPositionManager(_make_gpm_config())
    data_engine = MagicMock()
    data_engine.get_funding_rate.return_value = None
    data_engine.get_open_interest.return_value = []
    runner = GridStrategyRunner(
        strategy=strategy, config=config,
        indicator_engine=indicator_engine,
        grid_position_manager=gpm,
        data_engine=data_engine,
    )
    runner._is_warming_up = False
    runner._warmup_ended_at = datetime.now(tz=timezone.utc)
    return runner


def _make_positions(
    symbol="BTC/USDT", entry_price=50_000.0, quantity=0.01, levels=2,
) -> list[GridPosition]:
    positions = []
    for i in range(levels):
        positions.append(GridPosition(
            level=i,
            direction=Direction.LONG,
            entry_price=entry_price * (1 - 0.02 * i),
            quantity=quantity,
            entry_time=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
            entry_fee=entry_price * quantity * 0.0006,
        ))
    return positions


def _make_mock_exchange() -> AsyncMock:
    exchange = AsyncMock()
    exchange.load_markets = AsyncMock(return_value={
        "BTC/USDT:USDT": {
            "limits": {"amount": {"min": 0.001}},
            "precision": {"amount": 3, "price": 1},
        },
    })
    exchange.fetch_balance = AsyncMock(return_value={
        "free": {"USDT": 5_000.0},
        "total": {"USDT": 10_000.0},
    })
    exchange.fetch_positions = AsyncMock(return_value=[])
    exchange.set_leverage = AsyncMock()
    exchange.set_margin_mode = AsyncMock()
    exchange.create_order = AsyncMock(return_value={
        "id": "close_1", "status": "closed",
        "filled": 0.02, "average": 51_000.0,
    })
    exchange.cancel_order = AsyncMock()
    exchange.fetch_order = AsyncMock(return_value={"status": "closed"})
    exchange.fetch_my_trades = AsyncMock(return_value=[
        {"price": 51_000.0, "amount": 0.02},
    ])
    exchange.fetch_open_orders = AsyncMock(return_value=[])
    exchange.amount_to_precision = MagicMock(side_effect=lambda s, q: str(q))
    exchange.price_to_precision = MagicMock(side_effect=lambda s, p: str(p))
    return exchange


def _make_executor(simulator=None) -> Executor:
    config = MagicMock()
    config.risk = MagicMock()
    config.risk.fees = MagicMock()
    config.risk.fees.taker_percent = 0.06
    config.risk.fees.maker_percent = 0.02
    config.risk.fees.slippage_percent = 0.05
    config.risk.max_concurrent_positions = 10
    config.risk.max_margin_ratio = 0.7
    config.risk.kill_switch = MagicMock()
    config.risk.kill_switch.global_max_loss_pct = 45.0
    rm = MagicMock()
    rm.pre_trade_check = MagicMock(return_value=(True, ""))
    rm.is_kill_switch_triggered = False
    rm.register_position = MagicMock()
    rm.unregister_position = MagicMock()
    rm.record_trade_result = MagicMock()
    rm.get_status = MagicMock(return_value={})
    rm.get_state = MagicMock(return_value={})
    notifier = AsyncMock()
    executor = Executor(config, rm, notifier, selector=MagicMock())
    exchange = _make_mock_exchange()
    executor._exchange = exchange
    executor._markets = exchange.load_markets.return_value
    executor._running = True
    executor._connected = True
    if simulator is not None:
        executor._simulator = simulator
    return executor


# ── Tests Simulator.force_close_grid ─────────────────────────────────────


class TestForceCloseGrid:
    """Tests unitaires pour Simulator.force_close_grid()."""

    def test_force_close_grid_updates_capital(self):
        """Après force_close_grid, capital du runner reflète le P&L + marge restituée."""
        runner = _make_grid_runner(initial_capital=10_000.0)
        positions = _make_positions(
            symbol="BTC/USDT", entry_price=50_000.0, quantity=0.01, levels=2,
        )
        runner._positions["BTC/USDT"] = positions

        # Déduire la marge (simule l'ouverture)
        total_notional = sum(p.entry_price * p.quantity for p in positions)
        margin = total_notional / runner._leverage
        runner._capital -= margin

        capital_before = runner._capital
        exit_price = 51_000.0

        # Créer un Simulator avec ce runner
        sim = MagicMock(spec=Simulator)
        sim._runners = [runner]
        # Appeler la vraie méthode
        Simulator.force_close_grid(sim, "grid_atr", "BTC/USDT", exit_price, "tp_global")

        assert runner._positions["BTC/USDT"] == []
        assert runner._capital > capital_before
        assert runner._stats.total_trades == 1
        assert len(runner._trades) == 1
        sym, trade = runner._trades[0]
        assert sym == "BTC/USDT"
        assert isinstance(trade, TradeResult)
        assert trade.exit_reason == "tp_global"

    def test_force_close_grid_no_positions(self):
        """force_close_grid sur symbol sans positions = no-op silencieux."""
        runner = _make_grid_runner()
        runner._positions["BTC/USDT"] = []

        sim = MagicMock(spec=Simulator)
        sim._runners = [runner]
        Simulator.force_close_grid(sim, "grid_atr", "BTC/USDT", 51_000.0, "tp_global")

        assert runner._stats.total_trades == 0
        assert len(runner._trades) == 0

    def test_force_close_grid_wrong_strategy(self):
        """force_close_grid sur mauvais strategy_name = no-op silencieux."""
        runner = _make_grid_runner()
        runner._positions["BTC/USDT"] = _make_positions()

        sim = MagicMock(spec=Simulator)
        sim._runners = [runner]
        Simulator.force_close_grid(sim, "wrong_strategy", "BTC/USDT", 51_000.0, "tp_global")

        # Positions toujours là
        assert len(runner._positions["BTC/USDT"]) == 2
        assert runner._stats.total_trades == 0

    def test_force_close_grid_wrong_symbol(self):
        """force_close_grid sur mauvais symbol = no-op silencieux."""
        runner = _make_grid_runner()
        runner._positions["BTC/USDT"] = _make_positions()

        sim = MagicMock(spec=Simulator)
        sim._runners = [runner]
        Simulator.force_close_grid(sim, "grid_atr", "ETH/USDT", 3_000.0, "tp_global")

        assert len(runner._positions["BTC/USDT"]) == 2
        assert runner._stats.total_trades == 0

    def test_force_close_grid_updates_realized_pnl(self):
        """realized_pnl mis à jour correctement après force_close."""
        runner = _make_grid_runner()
        positions = _make_positions(
            symbol="BTC/USDT", entry_price=50_000.0, quantity=0.01, levels=1,
        )
        runner._positions["BTC/USDT"] = positions
        total_notional = sum(p.entry_price * p.quantity for p in positions)
        runner._capital -= total_notional / runner._leverage
        assert runner._realized_pnl == 0.0

        sim = MagicMock(spec=Simulator)
        sim._runners = [runner]
        Simulator.force_close_grid(sim, "grid_atr", "BTC/USDT", 51_000.0, "tp_global")

        assert runner._realized_pnl != 0.0
        assert runner._stats.net_pnl == runner._capital - runner._initial_capital

    def test_force_close_grid_win_loss_tracking(self):
        """Wins/losses correctement incrémentés."""
        runner = _make_grid_runner()
        # Position à 50k, exit à 51k = profit
        runner._positions["BTC/USDT"] = _make_positions(
            entry_price=50_000.0, quantity=0.01, levels=1,
        )
        total_notional = 50_000.0 * 0.01
        runner._capital -= total_notional / runner._leverage

        sim = MagicMock(spec=Simulator)
        sim._runners = [runner]
        Simulator.force_close_grid(sim, "grid_atr", "BTC/USDT", 51_000.0, "tp_global")

        assert runner._stats.wins == 1
        assert runner._stats.losses == 0

    def test_force_close_grid_loss_tracking(self):
        """Loss correctement comptabilisé."""
        runner = _make_grid_runner()
        # Position à 50k, exit à 45k = perte
        runner._positions["BTC/USDT"] = _make_positions(
            entry_price=50_000.0, quantity=0.01, levels=1,
        )
        total_notional = 50_000.0 * 0.01
        runner._capital -= total_notional / runner._leverage

        sim = MagicMock(spec=Simulator)
        sim._runners = [runner]
        Simulator.force_close_grid(sim, "grid_atr", "BTC/USDT", 45_000.0, "sl_global")

        assert runner._stats.wins == 0
        assert runner._stats.losses == 1


# ── Tests Executor sync vers paper ───────────────────────────────────────


class TestExecutorSyncPaper:
    """Tests intégration : _close_grid_cycle appelle force_close_grid."""

    @pytest.mark.asyncio
    async def test_close_grid_syncs_to_paper(self):
        """Fermeture live → paper fermé au même prix."""
        simulator = MagicMock()
        simulator.force_close_grid = MagicMock()
        executor = _make_executor(simulator=simulator)

        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT",
            direction="LONG",
            strategy_name="grid_atr",
            leverage=6,
            positions=[GridLivePosition(
                level=0, entry_price=50_000.0, quantity=0.01,
                entry_order_id="entry_0",
            )],
            sl_order_id="sl_1",
            sl_price=40_000.0,
        )
        executor._risk_manager.register_position({
            "symbol": "BTC/USDT:USDT", "direction": "LONG",
        })

        event = TradeEvent(
            event_type=TradeEventType.CLOSE,
            strategy_name="grid_atr",
            symbol="BTC/USDT",
            direction="LONG",
            entry_price=50_000.0,
            exit_price=51_000.0,
            quantity=0.01,
            tp_price=0.0,
            sl_price=0.0,
            score=0.0,
            timestamp=datetime.now(tz=timezone.utc),
            market_regime="unknown",
            exit_reason="tp_global",
        )
        await executor._close_grid_cycle(event)

        # Vérifier que la sync a été appelée
        simulator.force_close_grid.assert_called_once()
        call_args = simulator.force_close_grid.call_args
        assert call_args.kwargs["strategy_name"] == "grid_atr"
        assert call_args.kwargs["symbol"] == "BTC/USDT"
        assert call_args.kwargs["exit_reason"] == "tp_global"
        # exit_price doit être le prix réel (from exchange, not event)
        assert call_args.kwargs["exit_price"] > 0

    @pytest.mark.asyncio
    async def test_close_grid_paper_sync_failure_doesnt_break_live(self):
        """Si sync paper échoue, la fermeture live n'est pas affectée."""
        simulator = MagicMock()
        simulator.force_close_grid = MagicMock(side_effect=RuntimeError("paper broke"))
        executor = _make_executor(simulator=simulator)

        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT",
            direction="LONG",
            strategy_name="grid_atr",
            leverage=6,
            positions=[GridLivePosition(
                level=0, entry_price=50_000.0, quantity=0.01,
                entry_order_id="entry_0",
            )],
            sl_order_id="sl_1",
            sl_price=40_000.0,
        )

        event = TradeEvent(
            event_type=TradeEventType.CLOSE,
            strategy_name="grid_atr",
            symbol="BTC/USDT",
            direction="LONG",
            entry_price=50_000.0,
            exit_price=51_000.0,
            quantity=0.01,
            tp_price=0.0,
            sl_price=0.0,
            score=0.0,
            timestamp=datetime.now(tz=timezone.utc),
            market_regime="unknown",
            exit_reason="tp_global",
        )
        await executor._close_grid_cycle(event)

        # Fermeture live ok malgré l'erreur paper
        assert "BTC/USDT:USDT" not in executor._grid_states
        simulator.force_close_grid.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_grid_no_simulator(self):
        """Sans simulator, pas de sync (et pas d'erreur)."""
        executor = _make_executor(simulator=None)

        executor._grid_states["BTC/USDT:USDT"] = GridLiveState(
            symbol="BTC/USDT:USDT",
            direction="LONG",
            strategy_name="grid_atr",
            leverage=6,
            positions=[GridLivePosition(
                level=0, entry_price=50_000.0, quantity=0.01,
                entry_order_id="entry_0",
            )],
            sl_order_id="sl_1",
            sl_price=40_000.0,
        )

        event = TradeEvent(
            event_type=TradeEventType.CLOSE,
            strategy_name="grid_atr",
            symbol="BTC/USDT",
            direction="LONG",
            entry_price=50_000.0,
            exit_price=51_000.0,
            quantity=0.01,
            tp_price=0.0,
            sl_price=0.0,
            score=0.0,
            timestamp=datetime.now(tz=timezone.utc),
            market_regime="unknown",
            exit_reason="tp_global",
        )
        await executor._close_grid_cycle(event)

        assert "BTC/USDT:USDT" not in executor._grid_states

    @pytest.mark.asyncio
    async def test_handle_grid_sl_syncs_to_paper(self):
        """SL exécuté par Bitget → paper aussi fermé."""
        simulator = MagicMock()
        simulator.force_close_grid = MagicMock()
        executor = _make_executor(simulator=simulator)

        state = GridLiveState(
            symbol="BTC/USDT:USDT",
            direction="LONG",
            strategy_name="grid_atr",
            leverage=6,
            positions=[GridLivePosition(
                level=0, entry_price=50_000.0, quantity=0.01,
                entry_order_id="entry_0",
            )],
            sl_order_id="sl_1",
            sl_price=40_000.0,
        )
        executor._grid_states["BTC/USDT:USDT"] = state

        await executor._handle_grid_sl_executed(
            "BTC/USDT:USDT", state, exit_price=40_000.0,
        )

        simulator.force_close_grid.assert_called_once()
        call_args = simulator.force_close_grid.call_args
        assert call_args.kwargs["strategy_name"] == "grid_atr"
        assert call_args.kwargs["symbol"] == "BTC/USDT"
        assert call_args.kwargs["exit_price"] == 40_000.0
        assert call_args.kwargs["exit_reason"] == "sl_global"
