"""Tests Audit 9 — Fixes P0 critiques de l'executor.

P0-1 : rollback _pending_notional sur échec d'entrée grid
P0-2 : emergency_close_grid conserve le state si le close échoue
P0-3 : guard contre double traitement dans _handle_grid_sl_executed
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.execution.executor import (
    Executor,
    GridLivePosition,
    GridLiveState,
)
from backend.strategies.base_grid import GridLevel


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
    config.risk.fees.slippage_percent = 0.05
    config.risk.slippage.default_estimate_percent = 0.05
    config.risk.max_live_grids = 4
    config.risk.initial_capital = 1000.0
    config.risk.max_margin_ratio = 0.70
    config.risk.kill_switch.max_session_loss_percent = 5.0
    config.risk.kill_switch.grid_max_session_loss_percent = 25.0
    config.risk.kill_switch.global_max_loss_pct = 45.0
    config.assets = [MagicMock(symbol="BTC/USDT", min_order_size=0.001)]
    config.strategies.grid_atr.leverage = 6
    config.strategies.grid_atr.timeframe = "1h"
    config.strategies.grid_atr.sl_percent = 20.0
    config.correlation_groups = {}
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
    exchange.fetch_my_trades = AsyncMock(return_value=[
        {"price": 50_000.0, "amount": 0.01},
    ])
    exchange.amount_to_precision = MagicMock(side_effect=lambda _sym, qty: f"{qty:.3f}")
    exchange.price_to_precision = MagicMock(side_effect=lambda _sym, p: str(p))
    exchange.close = AsyncMock()
    return exchange


def _make_executor(*, simulator=None) -> Executor:
    config = _make_config()
    rm = MagicMock()
    rm.pre_trade_check = MagicMock(return_value=(True, ""))
    rm.is_kill_switch_triggered = False
    rm.register_position = MagicMock()
    rm.unregister_position = MagicMock()
    rm.record_trade_result = MagicMock()
    rm.record_balance_snapshot = MagicMock()
    rm.get_status = MagicMock(return_value={})
    rm.get_state = MagicMock(return_value={})
    notifier = AsyncMock()

    executor = Executor(config, rm, notifier, selector=MagicMock())
    executor._exchange = _make_exchange()
    executor._markets = {
        "BTC/USDT:USDT": {
            "limits": {"amount": {"min": 0.001}},
            "precision": {"amount": 3, "price": 1},
        },
    }
    executor._running = True
    executor._connected = True
    executor._exchange_balance = 1000.0
    executor._balance_bootstrapped = True
    if simulator is not None:
        executor._simulator = simulator
    return executor


def _make_grid_state(
    symbol="BTC/USDT:USDT",
    direction="LONG",
    strategy_name="grid_atr",
    levels=1,
    entry_price=50_000.0,
    quantity=0.01,
    leverage=6,
    sl_price=40_000.0,
) -> GridLiveState:
    positions = [
        GridLivePosition(
            level=i,
            entry_price=entry_price * (1 - 0.02 * i),
            quantity=quantity,
            entry_order_id=f"entry_{i}",
        )
        for i in range(levels)
    ]
    return GridLiveState(
        symbol=symbol,
        direction=direction,
        strategy_name=strategy_name,
        leverage=leverage,
        positions=positions,
        sl_order_id="sl_1",
        sl_price=sl_price,
    )


# ── P0-1 : rollback _pending_notional sur échec ─────────────────────────


class TestPendingNotionalRollback:
    """P0-1 Audit 9 : _pending_notional doit revenir à 0 après échec d'entrée."""

    @pytest.mark.asyncio
    async def test_rollback_on_entry_failure(self):
        """Quand _open_grid_position() lève une exception,
        _pending_notional doit être décrémenté (pas de marge fantôme)."""
        from backend.core.models import Candle, Direction, TimeFrame
        from backend.strategies.base import StrategyContext

        level = GridLevel(
            index=0, entry_price=50_000.0,
            direction=Direction.LONG, size_fraction=0.25,
        )
        strategy = MagicMock()
        strategy._config = MagicMock()
        strategy._config.timeframe = "1h"
        strategy._config.leverage = 6
        strategy._config.sl_percent = 20.0
        strategy._config.per_asset = {}
        strategy.name = "grid_atr"
        strategy.max_positions = 3
        strategy.compute_grid = MagicMock(return_value=[level])
        strategy.should_close_all = MagicMock(return_value=None)
        strategy.get_tp_price = MagicMock(return_value=float("nan"))
        strategy.get_sl_price = MagicMock(return_value=float("nan"))

        executor = _make_executor()
        executor._strategies = {"grid_atr": strategy}

        ctx = StrategyContext(
            symbol="BTC/USDT",
            timestamp=datetime.now(tz=timezone.utc),
            candles={},
            indicators={"1h": {"close": 50_000.0, "sma": 49_000.0, "regime": "ranging"}},
            current_position=None,
            capital=10_000.0,
            config=MagicMock(),
        )
        sim = MagicMock()
        sim.get_runner_context = MagicMock(return_value=ctx)
        executor._simulator = sim

        # Simuler un échec réseau
        executor._exchange.create_order = AsyncMock(
            side_effect=Exception("Bitget timeout"),
        )

        candle = Candle(
            timestamp=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
            open=50_000.0, high=50_500.0, low=49_500.0, close=50_000.0,
            volume=100.0, symbol="BTC/USDT", timeframe=TimeFrame.H1,
        )

        initial_pending = executor._pending_notional
        await executor._on_candle("BTC/USDT", "1h", candle)

        # Après échec, la marge fantôme doit être nettoyée
        assert executor._pending_notional == initial_pending, (
            f"_pending_notional devrait revenir à {initial_pending}, "
            f"trouvé {executor._pending_notional} (marge fantôme)"
        )

    @pytest.mark.asyncio
    async def test_pending_notional_kept_on_success(self):
        """Quand l'entrée réussit, _pending_notional reste accumulé
        (nécessaire pour le margin guard des itérations suivantes)."""
        from backend.core.models import Candle, Direction, TimeFrame
        from backend.strategies.base import StrategyContext

        level = GridLevel(
            index=0, entry_price=50_000.0,
            direction=Direction.LONG, size_fraction=0.25,
        )
        strategy = MagicMock()
        strategy._config = MagicMock()
        strategy._config.timeframe = "1h"
        strategy._config.leverage = 6
        strategy._config.sl_percent = 20.0
        strategy._config.per_asset = {}
        strategy.name = "grid_atr"
        strategy.max_positions = 3
        strategy.compute_grid = MagicMock(return_value=[level])
        strategy.should_close_all = MagicMock(return_value=None)
        strategy.get_tp_price = MagicMock(return_value=float("nan"))
        strategy.get_sl_price = MagicMock(return_value=float("nan"))

        executor = _make_executor()
        executor._strategies = {"grid_atr": strategy}

        ctx = StrategyContext(
            symbol="BTC/USDT",
            timestamp=datetime.now(tz=timezone.utc),
            candles={},
            indicators={"1h": {"close": 50_000.0, "sma": 49_000.0, "regime": "ranging"}},
            current_position=None,
            capital=10_000.0,
            config=MagicMock(),
        )
        sim = MagicMock()
        sim.get_runner_context = MagicMock(return_value=ctx)
        executor._simulator = sim

        candle = Candle(
            timestamp=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
            open=50_000.0, high=50_500.0, low=49_500.0, close=50_000.0,
            volume=100.0, symbol="BTC/USDT", timeframe=TimeFrame.H1,
        )

        await executor._on_candle("BTC/USDT", "1h", candle)

        # Succès : _pending_notional doit rester accumulé (0.25 * 1000/1 = 250)
        assert executor._pending_notional > 0, (
            "_pending_notional devrait rester > 0 après succès"
        )


# ── P0-2 : emergency_close_grid conserve le state si le close échoue ────


class TestEmergencyCloseGridStatePreserved:
    """P0-2 Audit 9 : si le market close échoue, le state ne doit pas
    être supprimé — la position est toujours sur l'exchange."""

    @pytest.mark.asyncio
    async def test_state_preserved_on_close_failure(self):
        """Si create_order(market close) échoue, grid_states doit
        conserver le state pour retry par le polling."""
        executor = _make_executor()

        state = _make_grid_state(levels=2)
        executor._grid_states["BTC/USDT:USDT"] = state

        # Simuler échec du market close
        executor._exchange.create_order = AsyncMock(
            side_effect=Exception("Network timeout"),
        )

        await executor._emergency_close_grid("BTC/USDT:USDT", state)

        # State doit être conservé (position toujours ouverte sur Bitget)
        assert "BTC/USDT:USDT" in executor._grid_states, (
            "grid_states ne doit PAS supprimer le state si le close a échoué "
            "— position orpheline sans SL sinon"
        )
        # SL marqué absent pour forcer retry
        assert state.sl_order_id == "", (
            "sl_order_id doit être vidé pour signaler l'absence de SL"
        )

    @pytest.mark.asyncio
    async def test_state_removed_on_close_success(self):
        """Si le market close réussit, le state doit être nettoyé normalement."""
        executor = _make_executor()

        state = _make_grid_state(levels=2)
        executor._grid_states["BTC/USDT:USDT"] = state

        await executor._emergency_close_grid("BTC/USDT:USDT", state)

        # State supprimé après close réussi
        assert "BTC/USDT:USDT" not in executor._grid_states, (
            "grid_states doit être nettoyé après un close réussi"
        )

    @pytest.mark.asyncio
    async def test_risk_manager_not_unregistered_on_failure(self):
        """Si le close échoue, la position ne doit pas être désenregistrée
        du risk manager — elle est toujours active."""
        executor = _make_executor()

        state = _make_grid_state()
        executor._grid_states["BTC/USDT:USDT"] = state

        executor._exchange.create_order = AsyncMock(
            side_effect=Exception("Timeout"),
        )

        await executor._emergency_close_grid("BTC/USDT:USDT", state)

        executor._risk_manager.unregister_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_notifier_called_on_both_success_and_failure(self):
        """L'alerte SL failed doit être envoyée dans les deux cas."""
        executor = _make_executor()

        # Cas succès
        state_ok = _make_grid_state()
        executor._grid_states["BTC/USDT:USDT"] = state_ok
        await executor._emergency_close_grid("BTC/USDT:USDT", state_ok)

        # Cas échec
        state_fail = _make_grid_state()
        executor._grid_states["BTC/USDT:USDT"] = state_fail
        executor._exchange.create_order = AsyncMock(
            side_effect=Exception("Timeout"),
        )
        await executor._emergency_close_grid("BTC/USDT:USDT", state_fail)

        # notify_live_sl_failed appelé 2 fois (success + failure)
        assert executor._notifier.notify_live_sl_failed.call_count == 2


# ── P0-3 : guard contre double traitement _handle_grid_sl_executed ──────


class TestHandleGridSlDuplicateGuard:
    """P0-3 Audit 9 : si watchOrders et polling détectent le même SL,
    _handle_grid_sl_executed ne doit s'exécuter qu'une seule fois."""

    @pytest.mark.asyncio
    async def test_second_call_is_noop(self):
        """Deux appels consécutifs à _handle_grid_sl_executed :
        le second doit être un no-op (grid_states déjà nettoyé)."""
        executor = _make_executor()

        state = _make_grid_state()
        executor._grid_states["BTC/USDT:USDT"] = state

        # Premier appel : traitement normal
        await executor._handle_grid_sl_executed(
            "BTC/USDT:USDT", state, exit_price=40_000.0,
        )

        assert "BTC/USDT:USDT" not in executor._grid_states

        # record_trade_result appelé 1 fois
        assert executor._risk_manager.record_trade_result.call_count == 1

        # Second appel : le guard empêche le double traitement
        await executor._handle_grid_sl_executed(
            "BTC/USDT:USDT", state, exit_price=40_000.0,
        )

        # Toujours 1 seul appel (pas 2)
        assert executor._risk_manager.record_trade_result.call_count == 1

    @pytest.mark.asyncio
    async def test_unknown_symbol_is_noop(self):
        """Appel avec un symbol inconnu doit être un no-op."""
        executor = _make_executor()

        state = _make_grid_state(symbol="ETH/USDT:USDT")
        # NE PAS ajouter dans grid_states

        await executor._handle_grid_sl_executed(
            "ETH/USDT:USDT", state, exit_price=3_000.0,
        )

        executor._risk_manager.record_trade_result.assert_not_called()

    @pytest.mark.asyncio
    async def test_pop_instead_of_del(self):
        """grid_states.pop() ne doit jamais lever KeyError."""
        executor = _make_executor()

        state = _make_grid_state()
        executor._grid_states["BTC/USDT:USDT"] = state

        await executor._handle_grid_sl_executed(
            "BTC/USDT:USDT", state, exit_price=40_000.0,
        )

        # Pas de KeyError
        assert "BTC/USDT:USDT" not in executor._grid_states
