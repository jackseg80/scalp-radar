"""Tests pour l'Executor (Sprint 5a) — mock ccxt."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.execution.executor import (
    Executor,
    LivePosition,
    TradeEvent,
    TradeEventType,
    to_futures_symbol,
)
from backend.execution.risk_manager import LiveRiskManager


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_config() -> MagicMock:
    config = MagicMock()
    config.secrets.live_trading = True
    config.secrets.bitget_api_key = "test_key"
    config.secrets.bitget_secret = "test_secret"
    config.secrets.bitget_passphrase = "test_pass"
    config.secrets.bitget_sandbox = True
    config.risk.position.max_concurrent_positions = 3
    config.risk.position.default_leverage = 15
    config.risk.margin.min_free_margin_percent = 20
    config.risk.margin.mode = "cross"
    config.risk.fees.taker_percent = 0.06
    config.risk.fees.maker_percent = 0.02
    config.risk.kill_switch.max_session_loss_percent = 5.0
    config.assets = [
        MagicMock(symbol="BTC/USDT", min_order_size=0.001, tick_size=0.1),
    ]
    return config


def _make_mock_exchange() -> AsyncMock:
    exchange = AsyncMock()
    exchange.load_markets = AsyncMock(return_value={
        "BTC/USDT:USDT": {
            "limits": {"amount": {"min": 0.001}},
            "precision": {"amount": 3, "price": 1},
        },
    })
    exchange.fetch_balance = AsyncMock(return_value={
        "free": {"USDT": 5_000.0, "SUSDT": 5_000.0},
        "total": {"USDT": 10_000.0, "SUSDT": 10_000.0},
    })
    exchange.fetch_positions = AsyncMock(return_value=[])
    exchange.set_leverage = AsyncMock()
    exchange.set_margin_mode = AsyncMock()
    exchange.create_order = AsyncMock(return_value={
        "id": "order_123",
        "status": "closed",
        "filled": 0.001,
        "average": 100_000.0,
    })
    exchange.cancel_order = AsyncMock()
    exchange.fetch_order = AsyncMock(return_value={"status": "closed"})
    exchange.fetch_my_trades = AsyncMock(return_value=[
        {"price": 100_500.0, "amount": 0.001},
    ])
    exchange.watch_orders = AsyncMock(return_value=[])
    exchange.close = AsyncMock()
    return exchange


def _make_notifier() -> AsyncMock:
    notifier = AsyncMock()
    notifier.notify_live_order_opened = AsyncMock()
    notifier.notify_live_order_closed = AsyncMock()
    notifier.notify_live_sl_failed = AsyncMock()
    notifier.notify_reconciliation = AsyncMock()
    return notifier


def _make_executor(
    config=None, risk_manager=None, notifier=None, exchange=None,
) -> Executor:
    if config is None:
        config = _make_config()
    if risk_manager is None:
        risk_manager = LiveRiskManager(config)
        risk_manager.set_initial_capital(10_000.0)
    if notifier is None:
        notifier = _make_notifier()

    executor = Executor(config, risk_manager, notifier)
    if exchange is None:
        exchange = _make_mock_exchange()
    executor._exchange = exchange
    executor._markets = exchange.load_markets.return_value
    executor._running = True
    executor._connected = True
    return executor


def _make_open_event(
    symbol="BTC/USDT",
    strategy="vwap_rsi",
    direction="LONG",
    entry_price=100_000.0,
    quantity=0.001,
    tp_price=100_800.0,
    sl_price=99_700.0,
) -> TradeEvent:
    return TradeEvent(
        event_type=TradeEventType.OPEN,
        strategy_name=strategy,
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        quantity=quantity,
        tp_price=tp_price,
        sl_price=sl_price,
        score=0.75,
        timestamp=datetime.now(tz=timezone.utc),
        market_regime="RANGING",
    )


def _make_close_event(
    symbol="BTC/USDT",
    strategy="vwap_rsi",
    direction="LONG",
    exit_price=100_800.0,
    exit_reason="tp",
) -> TradeEvent:
    return TradeEvent(
        event_type=TradeEventType.CLOSE,
        strategy_name=strategy,
        symbol=symbol,
        direction=direction,
        entry_price=100_000.0,
        quantity=0.001,
        tp_price=100_800.0,
        sl_price=99_700.0,
        score=0.75,
        timestamp=datetime.now(tz=timezone.utc),
        market_regime="RANGING",
        exit_reason=exit_reason,
        exit_price=exit_price,
    )


# ─── Symbol mapping ───────────────────────────────────────────────────────


class TestSymbolMapping:
    def test_btc_spot_to_futures(self):
        assert to_futures_symbol("BTC/USDT") == "BTC/USDT:USDT"

    def test_eth_spot_to_futures(self):
        assert to_futures_symbol("ETH/USDT") == "ETH/USDT:USDT"

    def test_sol_spot_to_futures(self):
        assert to_futures_symbol("SOL/USDT") == "SOL/USDT:USDT"

    def test_unknown_symbol_raises(self):
        with pytest.raises(ValueError, match="non supporté"):
            to_futures_symbol("DOGE/USDT")


# ─── Event filtering ──────────────────────────────────────────────────────


class TestEventFiltering:
    @pytest.mark.asyncio
    async def test_ignores_non_vwap_rsi_strategy(self):
        executor = _make_executor()
        event = _make_open_event(strategy="momentum")
        await executor.handle_event(event)
        # Pas d'appel create_order
        executor._exchange.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_non_btc_symbol(self):
        executor = _make_executor()
        event = _make_open_event(symbol="ETH/USDT")
        await executor.handle_event(event)
        executor._exchange.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_when_not_running(self):
        executor = _make_executor()
        executor._running = False
        event = _make_open_event()
        await executor.handle_event(event)
        executor._exchange.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_accepts_vwap_rsi_btc(self):
        executor = _make_executor()
        event = _make_open_event()
        await executor.handle_event(event)
        # create_order appelé au moins pour l'entry
        assert executor._exchange.create_order.call_count >= 1


# ─── Open position ─────────────────────────────────────────────────────────


class TestOpenPosition:
    @pytest.mark.asyncio
    async def test_open_places_entry_market_order(self):
        executor = _make_executor()
        event = _make_open_event()
        await executor._open_position(event)

        # Premier appel = entry market
        first_call = executor._exchange.create_order.call_args_list[0]
        assert first_call[0][0] == "BTC/USDT:USDT"
        assert first_call[0][1] == "market"
        assert first_call[0][2] == "buy"  # LONG → buy

    @pytest.mark.asyncio
    async def test_open_places_sl_trigger_order(self):
        executor = _make_executor()
        # SL = deuxième appel create_order
        executor._exchange.create_order = AsyncMock(side_effect=[
            {"id": "entry_1", "status": "closed", "filled": 0.001, "average": 100_000.0},
            {"id": "sl_1", "status": "open"},
            {"id": "tp_1", "status": "open"},
        ])
        event = _make_open_event()
        await executor._open_position(event)

        sl_call = executor._exchange.create_order.call_args_list[1]
        assert sl_call[1]["params"]["triggerPrice"] == 99_700.0
        assert sl_call[1]["params"]["reduceOnly"] is True

    @pytest.mark.asyncio
    async def test_open_places_tp_trigger_order(self):
        executor = _make_executor()
        executor._exchange.create_order = AsyncMock(side_effect=[
            {"id": "entry_1", "status": "closed", "filled": 0.001, "average": 100_000.0},
            {"id": "sl_1", "status": "open"},
            {"id": "tp_1", "status": "open"},
        ])
        event = _make_open_event()
        await executor._open_position(event)

        tp_call = executor._exchange.create_order.call_args_list[2]
        assert tp_call[1]["params"]["triggerPrice"] == 100_800.0

    @pytest.mark.asyncio
    async def test_open_registers_live_position(self):
        executor = _make_executor()
        event = _make_open_event()
        await executor._open_position(event)

        assert executor._position is not None
        assert executor._position.symbol == "BTC/USDT:USDT"
        assert executor._position.direction == "LONG"

    @pytest.mark.asyncio
    async def test_open_notifies_telegram(self):
        notifier = _make_notifier()
        executor = _make_executor(notifier=notifier)
        event = _make_open_event()
        await executor._open_position(event)

        notifier.notify_live_order_opened.assert_called_once()

    @pytest.mark.asyncio
    async def test_open_rejected_by_risk_manager(self):
        executor = _make_executor()
        executor._risk_manager._kill_switch_triggered = True
        event = _make_open_event()
        await executor._open_position(event)

        # Aucun appel create_order (rejeté par pre_trade_check)
        executor._exchange.create_order.assert_not_called()
        assert executor._position is None

    @pytest.mark.asyncio
    async def test_open_skipped_if_already_has_position(self):
        executor = _make_executor()
        executor._position = LivePosition(
            symbol="BTC/USDT:USDT", direction="LONG",
            entry_price=99_000, quantity=0.001,
            entry_order_id="existing",
        )
        event = _make_open_event()
        await executor._open_position(event)
        executor._exchange.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_entry_failed_no_sl_tp_placed(self):
        executor = _make_executor()
        executor._exchange.create_order = AsyncMock(
            side_effect=Exception("Bitget API error"),
        )
        event = _make_open_event()
        await executor._open_position(event)

        assert executor._position is None
        assert executor._exchange.create_order.call_count == 1  # Seul l'entry


# ─── SL rollback ──────────────────────────────────────────────────────────


class TestSLRollback:
    @pytest.mark.asyncio
    async def test_sl_failed_triggers_close_market(self):
        """Si le SL échoue après retries → close market immédiat."""
        notifier = _make_notifier()
        executor = _make_executor(notifier=notifier)

        # Entry OK, SL échoue 3 fois, puis close market
        executor._exchange.create_order = AsyncMock(side_effect=[
            {"id": "entry_1", "status": "closed", "filled": 0.001, "average": 100_000.0},
            Exception("SL failed 1"),
            Exception("SL failed 2"),
            Exception("SL failed 3"),
            {"id": "close_1", "status": "closed"},  # close market urgence
        ])

        event = _make_open_event()
        await executor._open_position(event)

        # Position NON enregistrée (close market déclenché)
        assert executor._position is None
        # Notification urgente envoyée
        notifier.notify_live_sl_failed.assert_called_once()

    @pytest.mark.asyncio
    async def test_sl_succeeds_on_second_retry(self):
        executor = _make_executor()
        executor._exchange.create_order = AsyncMock(side_effect=[
            {"id": "entry_1", "status": "closed", "filled": 0.001, "average": 100_000.0},
            Exception("SL failed 1"),
            {"id": "sl_1", "status": "open"},  # Retry 2 OK
            {"id": "tp_1", "status": "open"},
        ])
        event = _make_open_event()
        await executor._open_position(event)

        assert executor._position is not None
        assert executor._position.sl_order_id == "sl_1"

    @pytest.mark.asyncio
    async def test_tp_failed_position_kept(self):
        """Si le TP échoue, la position est gardée (SL protège)."""
        executor = _make_executor()
        executor._exchange.create_order = AsyncMock(side_effect=[
            {"id": "entry_1", "status": "closed", "filled": 0.001, "average": 100_000.0},
            {"id": "sl_1", "status": "open"},
            Exception("TP failed"),
        ])
        event = _make_open_event()
        await executor._open_position(event)

        assert executor._position is not None
        assert executor._position.sl_order_id == "sl_1"
        assert executor._position.tp_order_id is None


# ─── Close position ────────────────────────────────────────────────────────


class TestClosePosition:
    @pytest.mark.asyncio
    async def test_close_cancels_sl_tp_and_market_close(self):
        executor = _make_executor()
        executor._position = LivePosition(
            symbol="BTC/USDT:USDT", direction="LONG",
            entry_price=100_000.0, quantity=0.001,
            entry_order_id="entry_1",
            sl_order_id="sl_1", tp_order_id="tp_1",
            strategy_name="vwap_rsi",
        )
        executor._risk_manager.register_position({
            "symbol": "BTC/USDT:USDT", "direction": "LONG",
        })

        executor._exchange.create_order = AsyncMock(return_value={
            "id": "close_1", "average": 100_500.0,
        })

        event = _make_close_event(exit_reason="regime_change")
        await executor._close_position(event)

        # SL + TP annulés
        assert executor._exchange.cancel_order.call_count == 2
        # Market close placé
        executor._exchange.create_order.assert_called_once()
        close_call = executor._exchange.create_order.call_args
        assert close_call[0][1] == "market"
        assert close_call[1]["params"]["reduceOnly"] is True
        # Position nettoyée
        assert executor._position is None

    @pytest.mark.asyncio
    async def test_close_no_position_noop(self):
        executor = _make_executor()
        event = _make_close_event()
        await executor._close_position(event)
        executor._exchange.create_order.assert_not_called()


# ─── Quantity rounding ─────────────────────────────────────────────────────


class TestQuantityRounding:
    def test_rounds_to_market_precision(self):
        executor = _make_executor()
        # precision.amount = 3 → 3 décimales
        result = executor._round_quantity(0.00156, "BTC/USDT:USDT")
        assert result == 0.001

    def test_respects_min_amount(self):
        executor = _make_executor()
        result = executor._round_quantity(0.0001, "BTC/USDT:USDT")
        assert result >= 0.001  # min_amount

    def test_fallback_to_config(self):
        executor = _make_executor()
        executor._markets = {}  # Pas de données markets
        result = executor._round_quantity(0.0015, "BTC/USDT:USDT")
        assert result >= 0.001


# ─── Reconciliation ───────────────────────────────────────────────────────


class TestReconciliation:
    @pytest.mark.asyncio
    async def test_no_positions_both_sides(self):
        """Cas 4 : aucune position → clean."""
        notifier = _make_notifier()
        executor = _make_executor(notifier=notifier)
        executor._exchange.fetch_positions = AsyncMock(return_value=[])
        await executor._reconcile_on_boot()
        assert executor._position is None

    @pytest.mark.asyncio
    async def test_position_on_exchange_no_local(self):
        """Cas 2 : position orpheline sur exchange."""
        notifier = _make_notifier()
        executor = _make_executor(notifier=notifier)
        executor._exchange.fetch_positions = AsyncMock(return_value=[
            {"contracts": 0.001, "side": "long", "entryPrice": 100_000.0,
             "symbol": "BTC/USDT:USDT"},
        ])
        await executor._reconcile_on_boot()
        notifier.notify_reconciliation.assert_called_once()
        call_arg = notifier.notify_reconciliation.call_args[0][0]
        assert "orpheline" in call_arg

    @pytest.mark.asyncio
    async def test_local_position_closed_during_downtime(self):
        """Cas 3 : position locale mais fermée sur exchange."""
        notifier = _make_notifier()
        executor = _make_executor(notifier=notifier)
        executor._position = LivePosition(
            symbol="BTC/USDT:USDT", direction="LONG",
            entry_price=100_000.0, quantity=0.001,
            entry_order_id="old_entry",
            strategy_name="vwap_rsi",
        )
        executor._risk_manager.register_position({
            "symbol": "BTC/USDT:USDT", "direction": "LONG",
        })
        executor._exchange.fetch_positions = AsyncMock(return_value=[])
        executor._exchange.fetch_my_trades = AsyncMock(return_value=[
            {"price": 100_500.0},
        ])

        await executor._reconcile_on_boot()

        assert executor._position is None
        notifier.notify_reconciliation.assert_called_once()
        call_arg = notifier.notify_reconciliation.call_args[0][0]
        assert "downtime" in call_arg


# ─── Leverage setup ────────────────────────────────────────────────────────


class TestLeverageSetup:
    @pytest.mark.asyncio
    async def test_sets_leverage_when_no_position(self):
        executor = _make_executor()
        executor._exchange.fetch_positions = AsyncMock(return_value=[])
        await executor._setup_leverage_and_margin("BTC/USDT:USDT")
        executor._exchange.set_leverage.assert_called_once_with(
            15, "BTC/USDT:USDT", params={"productType": "SUSDT-FUTURES"},
        )

    @pytest.mark.asyncio
    async def test_skips_leverage_when_position_open(self):
        executor = _make_executor()
        executor._exchange.fetch_positions = AsyncMock(return_value=[
            {"contracts": 0.001, "symbol": "BTC/USDT:USDT"},
        ])
        await executor._setup_leverage_and_margin("BTC/USDT:USDT")
        executor._exchange.set_leverage.assert_not_called()


# ─── Lifecycle ─────────────────────────────────────────────────────────────


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_stop_does_not_close_positions(self):
        executor = _make_executor()
        executor._position = LivePosition(
            symbol="BTC/USDT:USDT", direction="LONG",
            entry_price=100_000.0, quantity=0.001,
            entry_order_id="test",
        )
        await executor.stop()
        # Pas de create_order (pas de close market)
        executor._exchange.create_order.assert_not_called()
        assert executor._connected is False

    def test_get_status(self):
        executor = _make_executor()
        status = executor.get_status()
        assert status["enabled"] is True
        assert status["connected"] is True
        assert status["sandbox"] is True
        assert status["position"] is None

    def test_get_status_with_position(self):
        executor = _make_executor()
        executor._position = LivePosition(
            symbol="BTC/USDT:USDT", direction="LONG",
            entry_price=100_000.0, quantity=0.001,
            entry_order_id="test",
            sl_price=99_700.0, tp_price=100_800.0,
        )
        status = executor.get_status()
        assert status["position"]["symbol"] == "BTC/USDT:USDT"
        assert status["position"]["sl_price"] == 99_700.0
