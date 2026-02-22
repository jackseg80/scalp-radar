"""Tests pour l'Executor (Sprint 5b) — mock ccxt."""

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
    config.risk.position.max_concurrent_positions = 3
    config.risk.position.default_leverage = 15
    config.risk.margin.min_free_margin_percent = 20
    config.risk.margin.mode = "cross"
    config.risk.fees.taker_percent = 0.06
    config.risk.fees.maker_percent = 0.02
    config.risk.kill_switch.max_session_loss_percent = 5.0
    config.risk.kill_switch.grid_max_session_loss_percent = 25.0
    config.assets = [
        MagicMock(symbol="BTC/USDT", min_order_size=0.001, tick_size=0.1, correlation_group=None),
        MagicMock(symbol="ETH/USDT", min_order_size=0.01, tick_size=0.01, correlation_group=None),
        MagicMock(symbol="SOL/USDT", min_order_size=0.1, tick_size=0.001, correlation_group=None),
    ]
    config.correlation_groups = {}
    return config


def _make_mock_exchange() -> AsyncMock:
    exchange = AsyncMock()
    exchange.load_markets = AsyncMock(return_value={
        "BTC/USDT:USDT": {
            "limits": {"amount": {"min": 0.001}},
            "precision": {"amount": 3, "price": 1},
        },
        "ETH/USDT:USDT": {
            "limits": {"amount": {"min": 0.01}},
            "precision": {"amount": 2, "price": 2},
        },
        "SOL/USDT:USDT": {
            "limits": {"amount": {"min": 0.1}},
            "precision": {"amount": 1, "price": 3},
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
    exchange.fetch_open_orders = AsyncMock(return_value=[])
    # Méthodes de precision ccxt (simule DECIMAL_PLACES mode)
    exchange.amount_to_precision = MagicMock(
        side_effect=lambda sym, qty: f"{int(qty * 1000) / 1000:.3f}",
    )
    exchange.price_to_precision = MagicMock(
        side_effect=lambda sym, price: f"{int(price * 10) / 10:.1f}",
    )
    return exchange


def _make_notifier() -> AsyncMock:
    notifier = AsyncMock()
    notifier.notify_live_order_opened = AsyncMock()
    notifier.notify_live_order_closed = AsyncMock()
    notifier.notify_live_sl_failed = AsyncMock()
    notifier.notify_reconciliation = AsyncMock()
    return notifier


def _make_selector(allowed: bool = True) -> MagicMock:
    """Crée un mock AdaptiveSelector."""
    selector = MagicMock()
    selector.is_allowed = MagicMock(return_value=allowed)
    selector.set_active_symbols = MagicMock()
    selector.get_status = MagicMock(return_value={
        "allowed_strategies": ["vwap_rsi"],
        "active_symbols": ["BTC/USDT"],
    })
    return selector


def _make_executor(
    config=None, risk_manager=None, notifier=None, exchange=None,
    selector=None,
) -> Executor:
    if config is None:
        config = _make_config()
    if risk_manager is None:
        risk_manager = LiveRiskManager(config)
        risk_manager.set_initial_capital(10_000.0)
    if notifier is None:
        notifier = _make_notifier()

    executor = Executor(config, risk_manager, notifier, selector=selector)
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

    def test_any_usdt_pair_works(self):
        assert to_futures_symbol("SHIB/USDT") == "SHIB/USDT:USDT"
        assert to_futures_symbol("ICP/USDT") == "ICP/USDT:USDT"
        assert to_futures_symbol("GALA/USDT") == "GALA/USDT:USDT"

    def test_already_futures_format(self):
        assert to_futures_symbol("BTC/USDT:USDT") == "BTC/USDT:USDT"

    def test_unknown_quote_raises(self):
        with pytest.raises(ValueError, match="non supporté"):
            to_futures_symbol("BTC/EUR")


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
    async def test_open_rounds_sl_tp_to_market_precision(self):
        """SL/TP avec trop de décimales → arrondis à precision.price."""
        executor = _make_executor()
        executor._exchange.create_order = AsyncMock(side_effect=[
            {"id": "entry_1", "status": "closed", "filled": 0.001, "average": 67977.58},
            {"id": "sl_1", "status": "open"},
            {"id": "tp_1", "status": "open"},
        ])
        # Prix avec 2 décimales (Bitget BTC veut 1 décimale max)
        event = _make_open_event(
            entry_price=67977.58, sl_price=67773.65, tp_price=68521.42,
        )
        await executor._open_position(event)

        sl_call = executor._exchange.create_order.call_args_list[1]
        tp_call = executor._exchange.create_order.call_args_list[2]
        # precision.price = 1 → arrondi à 1 décimale (truncation)
        assert sl_call[1]["params"]["triggerPrice"] == 67773.6
        assert tp_call[1]["params"]["triggerPrice"] == 68521.4
        # LivePosition stocke les prix arrondis
        pos = executor._positions["BTC/USDT:USDT"]
        assert pos.sl_price == 67773.6
        assert pos.tp_price == 68521.4

    @pytest.mark.asyncio
    async def test_open_registers_live_position(self):
        executor = _make_executor()
        event = _make_open_event()
        await executor._open_position(event)

        assert "BTC/USDT:USDT" in executor._positions
        pos = executor._positions["BTC/USDT:USDT"]
        assert pos.symbol == "BTC/USDT:USDT"
        assert pos.direction == "LONG"

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
        assert "BTC/USDT:USDT" not in executor._positions

    @pytest.mark.asyncio
    async def test_open_skipped_if_already_has_position(self):
        executor = _make_executor()
        executor._positions["BTC/USDT:USDT"] = LivePosition(
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

        assert "BTC/USDT:USDT" not in executor._positions
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
        assert "BTC/USDT:USDT" not in executor._positions
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

        assert "BTC/USDT:USDT" in executor._positions
        assert executor._positions["BTC/USDT:USDT"].sl_order_id == "sl_1"

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

        assert "BTC/USDT:USDT" in executor._positions
        pos = executor._positions["BTC/USDT:USDT"]
        assert pos.sl_order_id == "sl_1"
        assert pos.tp_order_id is None


# ─── Close position ────────────────────────────────────────────────────────


class TestClosePosition:
    @pytest.mark.asyncio
    async def test_close_cancels_sl_tp_and_market_close(self):
        executor = _make_executor()
        executor._positions["BTC/USDT:USDT"] = LivePosition(
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
        assert "BTC/USDT:USDT" not in executor._positions

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
        # Si amount_to_precision lève une exception, fallback config
        executor._exchange.amount_to_precision = MagicMock(
            side_effect=Exception("no market"),
        )
        result = executor._round_quantity(0.0015, "BTC/USDT:USDT")
        assert result >= 0.001


class TestRoundPrice:
    def test_rounds_to_market_precision(self):
        executor = _make_executor()
        # precision.price = 1 → 1 décimale
        assert executor._round_price(67773.65, "BTC/USDT:USDT") == 67773.6
        assert executor._round_price(100_123.99, "BTC/USDT:USDT") == 100_123.9

    def test_already_rounded(self):
        executor = _make_executor()
        assert executor._round_price(67773.0, "BTC/USDT:USDT") == 67773.0

    def test_fallback_no_exchange(self):
        executor = _make_executor()
        # Si price_to_precision lève une exception, retourne le prix tel quel
        executor._exchange.price_to_precision = MagicMock(
            side_effect=Exception("no market"),
        )
        assert executor._round_price(67773.65, "BTC/USDT:USDT") == 67773.65


# ─── Reconciliation ───────────────────────────────────────────────────────


class TestReconciliation:
    @pytest.mark.asyncio
    async def test_no_positions_both_sides(self):
        """Cas 4 : aucune position → clean."""
        notifier = _make_notifier()
        executor = _make_executor(notifier=notifier)
        executor._exchange.fetch_positions = AsyncMock(return_value=[])
        await executor._reconcile_on_boot()
        assert not executor._positions

    @pytest.mark.asyncio
    async def test_position_on_exchange_no_local(self):
        """Cas 2 : position orpheline sur exchange."""
        notifier = _make_notifier()
        executor = _make_executor(notifier=notifier)

        async def _fake_fetch_positions(*args, **kwargs):
            params = kwargs.get("params", {})
            symbols = args[0] if args else None
            if isinstance(symbols, list) and "BTC/USDT:USDT" in symbols:
                return [
                    {"contracts": 0.001, "side": "long", "entryPrice": 100_000.0,
                     "symbol": "BTC/USDT:USDT"},
                ]
            return []

        executor._exchange.fetch_positions = AsyncMock(side_effect=_fake_fetch_positions)
        await executor._reconcile_on_boot()
        notifier.notify_reconciliation.assert_called()
        orphan_calls = [
            c for c in notifier.notify_reconciliation.call_args_list
            if "orpheline" in c[0][0]
        ]
        assert len(orphan_calls) >= 1

    @pytest.mark.asyncio
    async def test_local_position_closed_during_downtime(self):
        """Cas 3 : position locale mais fermée sur exchange."""
        notifier = _make_notifier()
        executor = _make_executor(notifier=notifier)
        executor._positions["BTC/USDT:USDT"] = LivePosition(
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

        assert "BTC/USDT:USDT" not in executor._positions
        downtime_calls = [
            c for c in notifier.notify_reconciliation.call_args_list
            if "downtime" in c[0][0]
        ]
        assert len(downtime_calls) >= 1


# ─── Orphan trigger orders ────────────────────────────────────────────────


class TestOrphanOrders:
    @pytest.mark.asyncio
    async def test_cancels_orphan_trigger_orders(self):
        """Ordres trigger sans position associée → annulés."""
        notifier = _make_notifier()
        executor = _make_executor(notifier=notifier)
        executor._positions.clear()
        executor._exchange.fetch_positions = AsyncMock(return_value=[])
        executor._exchange.fetch_open_orders = AsyncMock(return_value=[
            {"id": "orphan_sl_1", "symbol": "BTC/USDT:USDT", "type": "limit"},
            {"id": "orphan_tp_2", "symbol": "BTC/USDT:USDT", "type": "limit"},
        ])

        await executor._reconcile_on_boot()

        assert executor._exchange.cancel_order.call_count == 2
        notifier.notify_reconciliation.assert_called()
        last_call = notifier.notify_reconciliation.call_args_list[-1][0][0]
        assert "orphelin" in last_call.lower()
        assert "2" in last_call

    @pytest.mark.asyncio
    async def test_keeps_tracked_orders(self):
        """Ordres trackés par la position locale → pas annulés."""
        notifier = _make_notifier()
        executor = _make_executor(notifier=notifier)
        executor._positions["BTC/USDT:USDT"] = LivePosition(
            symbol="BTC/USDT:USDT", direction="LONG",
            entry_price=100_000.0, quantity=0.001,
            entry_order_id="entry_1",
            sl_order_id="sl_tracked", tp_order_id="tp_tracked",
            strategy_name="vwap_rsi",
        )
        executor._risk_manager.register_position({
            "symbol": "BTC/USDT:USDT", "direction": "LONG",
        })

        async def _fake_fetch_positions(*args, **kwargs):
            params = kwargs.get("params", {})
            if params:
                return [
                    {"contracts": 0.001, "side": "long", "symbol": "BTC/USDT:USDT"},
                ]
            return [
                {"contracts": 0.001, "side": "long", "symbol": "BTC/USDT:USDT"},
            ]

        executor._exchange.fetch_positions = AsyncMock(side_effect=_fake_fetch_positions)
        executor._exchange.fetch_open_orders = AsyncMock(return_value=[
            {"id": "sl_tracked", "symbol": "BTC/USDT:USDT"},
            {"id": "tp_tracked", "symbol": "BTC/USDT:USDT"},
        ])

        await executor._reconcile_on_boot()

        executor._exchange.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_tracked_and_orphan_orders(self):
        """Mix d'ordres trackés et orphelins → seuls les orphelins annulés."""
        notifier = _make_notifier()
        executor = _make_executor(notifier=notifier)
        executor._positions["BTC/USDT:USDT"] = LivePosition(
            symbol="BTC/USDT:USDT", direction="LONG",
            entry_price=100_000.0, quantity=0.001,
            entry_order_id="entry_1",
            sl_order_id="sl_tracked", tp_order_id="tp_tracked",
            strategy_name="vwap_rsi",
        )
        executor._risk_manager.register_position({
            "symbol": "BTC/USDT:USDT", "direction": "LONG",
        })

        async def _fake_fetch_positions(*args, **kwargs):
            params = kwargs.get("params", {})
            if params:
                return [
                    {"contracts": 0.001, "side": "long", "symbol": "BTC/USDT:USDT"},
                ]
            return [
                {"contracts": 0.001, "side": "long", "symbol": "BTC/USDT:USDT"},
            ]

        executor._exchange.fetch_positions = AsyncMock(side_effect=_fake_fetch_positions)
        executor._exchange.fetch_open_orders = AsyncMock(return_value=[
            {"id": "sl_tracked", "symbol": "BTC/USDT:USDT"},
            {"id": "tp_tracked", "symbol": "BTC/USDT:USDT"},
            {"id": "old_tp_from_crash", "symbol": "BTC/USDT:USDT"},
        ])

        await executor._reconcile_on_boot()

        executor._exchange.cancel_order.assert_called_once_with(
            "old_tp_from_crash", "BTC/USDT:USDT",
        )
        last_call = notifier.notify_reconciliation.call_args_list[-1][0][0]
        assert "orphelin" in last_call.lower()

    @pytest.mark.asyncio
    async def test_fetch_open_orders_failure_non_blocking(self):
        """Échec fetch_open_orders → log warning, pas de crash."""
        executor = _make_executor()
        executor._positions.clear()
        executor._exchange.fetch_positions = AsyncMock(return_value=[])
        executor._exchange.fetch_open_orders = AsyncMock(
            side_effect=Exception("API down"),
        )

        await executor._reconcile_on_boot()
        assert not executor._positions


# ─── Leverage setup ────────────────────────────────────────────────────────


class TestLeverageSetup:
    @pytest.mark.asyncio
    async def test_sets_leverage_when_no_position(self):
        executor = _make_executor()
        executor._exchange.fetch_positions = AsyncMock(return_value=[])
        await executor._setup_leverage_and_margin("BTC/USDT:USDT")
        executor._exchange.set_leverage.assert_called_once_with(
            15, "BTC/USDT:USDT",
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
        executor._positions["BTC/USDT:USDT"] = LivePosition(
            symbol="BTC/USDT:USDT", direction="LONG",
            entry_price=100_000.0, quantity=0.001,
            entry_order_id="test",
        )
        await executor.stop()
        executor._exchange.create_order.assert_not_called()
        assert executor._connected is False

    def test_get_status(self):
        executor = _make_executor()
        status = executor.get_status()
        assert status["enabled"] is True
        assert status["connected"] is True
        assert status["position"] is None
        assert status["positions"] == []

    def test_get_status_with_position(self):
        executor = _make_executor()
        executor._positions["BTC/USDT:USDT"] = LivePosition(
            symbol="BTC/USDT:USDT", direction="LONG",
            entry_price=100_000.0, quantity=0.001,
            entry_order_id="test",
            sl_price=99_700.0, tp_price=100_800.0,
        )
        status = executor.get_status()
        assert status["position"]["symbol"] == "BTC/USDT:USDT"
        assert status["position"]["sl_price"] == 99_700.0
        assert len(status["positions"]) == 1

    def test_get_status_with_selector(self):
        selector = _make_selector()
        executor = _make_executor(selector=selector)
        status = executor.get_status()
        assert "selector" in status
        assert "allowed_strategies" in status["selector"]


# ─── Persistence ──────────────────────────────────────────────────────────


class TestPersistence:
    def test_get_state_for_persistence(self):
        executor = _make_executor()
        executor._positions["BTC/USDT:USDT"] = LivePosition(
            symbol="BTC/USDT:USDT", direction="LONG",
            entry_price=100_000.0, quantity=0.001,
            entry_order_id="entry_1",
            sl_order_id="sl_1", tp_order_id="tp_1",
            strategy_name="vwap_rsi",
            sl_price=99_700.0, tp_price=100_800.0,
        )
        state = executor.get_state_for_persistence()
        assert "positions" in state
        assert "BTC/USDT:USDT" in state["positions"]
        pos_data = state["positions"]["BTC/USDT:USDT"]
        assert pos_data["direction"] == "LONG"
        assert pos_data["sl_order_id"] == "sl_1"

    def test_restore_positions_new_format(self):
        executor = _make_executor()
        state = {
            "positions": {
                "BTC/USDT:USDT": {
                    "symbol": "BTC/USDT:USDT",
                    "direction": "LONG",
                    "entry_price": 100_000.0,
                    "quantity": 0.001,
                    "entry_order_id": "entry_1",
                    "sl_order_id": "sl_1",
                    "tp_order_id": "tp_1",
                    "entry_time": "2024-01-01T00:00:00+00:00",
                    "strategy_name": "vwap_rsi",
                    "sl_price": 99_700.0,
                    "tp_price": 100_800.0,
                },
            },
        }
        executor.restore_positions(state)
        assert "BTC/USDT:USDT" in executor._positions
        pos = executor._positions["BTC/USDT:USDT"]
        assert pos.direction == "LONG"
        assert pos.sl_order_id == "sl_1"

    def test_restore_positions_old_format_backward_compat(self):
        """Ancien format single position → migré en dict."""
        executor = _make_executor()
        state = {
            "position": {
                "symbol": "BTC/USDT:USDT",
                "direction": "LONG",
                "entry_price": 100_000.0,
                "quantity": 0.001,
                "entry_order_id": "entry_1",
                "sl_order_id": "sl_1",
                "tp_order_id": "tp_1",
                "entry_time": "2024-01-01T00:00:00+00:00",
                "strategy_name": "vwap_rsi",
                "sl_price": 99_700.0,
                "tp_price": 100_800.0,
            },
        }
        executor.restore_positions(state)
        assert "BTC/USDT:USDT" in executor._positions
        pos = executor._positions["BTC/USDT:USDT"]
        assert pos.direction == "LONG"

    def test_restore_positions_empty(self):
        executor = _make_executor()
        executor.restore_positions({})
        assert not executor._positions

    def test_persistence_roundtrip(self):
        """save → restore → positions identiques."""
        executor = _make_executor()
        executor._positions["BTC/USDT:USDT"] = LivePosition(
            symbol="BTC/USDT:USDT", direction="LONG",
            entry_price=100_000.0, quantity=0.001,
            entry_order_id="entry_1",
            sl_order_id="sl_1", tp_order_id="tp_1",
            strategy_name="vwap_rsi",
            sl_price=99_700.0, tp_price=100_800.0,
        )
        state = executor.get_state_for_persistence()

        executor2 = _make_executor()
        executor2.restore_positions(state)
        assert "BTC/USDT:USDT" in executor2._positions
        pos = executor2._positions["BTC/USDT:USDT"]
        assert pos.entry_price == 100_000.0
        assert pos.sl_order_id == "sl_1"
        assert pos.strategy_name == "vwap_rsi"


# ─── Multi-position ──────────────────────────────────────────────────────


class TestMultiPosition:
    @pytest.mark.asyncio
    async def test_open_two_positions_different_symbols(self):
        """Ouvrir 2 positions sur des symboles différents."""
        executor = _make_executor()
        executor._exchange.create_order = AsyncMock(side_effect=[
            # BTC entry + SL + TP
            {"id": "btc_entry", "status": "closed", "filled": 0.001, "average": 100_000.0},
            {"id": "btc_sl", "status": "open"},
            {"id": "btc_tp", "status": "open"},
            # ETH entry + SL + TP
            {"id": "eth_entry", "status": "closed", "filled": 0.01, "average": 3_500.0},
            {"id": "eth_sl", "status": "open"},
            {"id": "eth_tp", "status": "open"},
        ])

        await executor._open_position(_make_open_event(symbol="BTC/USDT"))
        await executor._open_position(_make_open_event(
            symbol="ETH/USDT", entry_price=3_500.0, quantity=0.01,
            sl_price=3_400.0, tp_price=3_600.0,
        ))

        assert len(executor._positions) == 2
        assert "BTC/USDT:USDT" in executor._positions
        assert "ETH/USDT:USDT" in executor._positions

    @pytest.mark.asyncio
    async def test_same_symbol_rejected(self):
        """Même symbole déjà ouvert → rejeté."""
        executor = _make_executor()
        executor._positions["BTC/USDT:USDT"] = LivePosition(
            symbol="BTC/USDT:USDT", direction="LONG",
            entry_price=100_000.0, quantity=0.001,
            entry_order_id="existing",
        )
        await executor._open_position(_make_open_event())
        executor._exchange.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_one_keeps_other(self):
        """Fermer une position garde l'autre."""
        executor = _make_executor()
        executor._positions["BTC/USDT:USDT"] = LivePosition(
            symbol="BTC/USDT:USDT", direction="LONG",
            entry_price=100_000.0, quantity=0.001,
            entry_order_id="btc_entry",
            sl_order_id="btc_sl", tp_order_id="btc_tp",
            strategy_name="vwap_rsi",
        )
        executor._positions["ETH/USDT:USDT"] = LivePosition(
            symbol="ETH/USDT:USDT", direction="SHORT",
            entry_price=3_500.0, quantity=0.01,
            entry_order_id="eth_entry",
            sl_order_id="eth_sl", tp_order_id="eth_tp",
            strategy_name="momentum",
        )
        executor._risk_manager.register_position({"symbol": "BTC/USDT:USDT", "direction": "LONG"})
        executor._risk_manager.register_position({"symbol": "ETH/USDT:USDT", "direction": "SHORT"})

        executor._exchange.create_order = AsyncMock(return_value={
            "id": "close_btc", "average": 100_500.0,
        })

        event = _make_close_event(symbol="BTC/USDT", exit_reason="tp")
        await executor._close_position(event)

        assert "BTC/USDT:USDT" not in executor._positions
        assert "ETH/USDT:USDT" in executor._positions
        assert executor._positions["ETH/USDT:USDT"].direction == "SHORT"

    @pytest.mark.asyncio
    async def test_orphan_cleanup_multi_position(self):
        """Tracked IDs de TOUTES les positions sont préservées lors du cleanup orphelins."""
        notifier = _make_notifier()
        executor = _make_executor(notifier=notifier)
        executor._positions["BTC/USDT:USDT"] = LivePosition(
            symbol="BTC/USDT:USDT", direction="LONG",
            entry_price=100_000.0, quantity=0.001,
            entry_order_id="btc_entry",
            sl_order_id="btc_sl", tp_order_id="btc_tp",
        )
        executor._positions["ETH/USDT:USDT"] = LivePosition(
            symbol="ETH/USDT:USDT", direction="SHORT",
            entry_price=3_500.0, quantity=0.01,
            entry_order_id="eth_entry",
            sl_order_id="eth_sl", tp_order_id="eth_tp",
        )

        executor._exchange.fetch_open_orders = AsyncMock(return_value=[
            {"id": "btc_sl", "symbol": "BTC/USDT:USDT"},
            {"id": "btc_tp", "symbol": "BTC/USDT:USDT"},
            {"id": "eth_sl", "symbol": "ETH/USDT:USDT"},
            {"id": "eth_tp", "symbol": "ETH/USDT:USDT"},
            {"id": "old_orphan", "symbol": "SOL/USDT:USDT"},
        ])

        await executor._cancel_orphan_orders()

        # Seul l'orphelin SOL annulé
        executor._exchange.cancel_order.assert_called_once_with(
            "old_orphan", "SOL/USDT:USDT",
        )

    def test_persistence_multi_position_roundtrip(self):
        """Persistence multi-position: save + restore."""
        executor = _make_executor()
        executor._positions["BTC/USDT:USDT"] = LivePosition(
            symbol="BTC/USDT:USDT", direction="LONG",
            entry_price=100_000.0, quantity=0.001,
            entry_order_id="btc_entry",
            sl_order_id="btc_sl", tp_order_id="btc_tp",
            strategy_name="vwap_rsi",
        )
        executor._positions["ETH/USDT:USDT"] = LivePosition(
            symbol="ETH/USDT:USDT", direction="SHORT",
            entry_price=3_500.0, quantity=0.01,
            entry_order_id="eth_entry",
            strategy_name="momentum",
        )

        state = executor.get_state_for_persistence()

        executor2 = _make_executor()
        executor2.restore_positions(state)
        assert len(executor2._positions) == 2
        assert executor2._positions["BTC/USDT:USDT"].direction == "LONG"
        assert executor2._positions["ETH/USDT:USDT"].strategy_name == "momentum"

    def test_backward_compat_position_property(self):
        """Property 'position' retourne la première position ou None."""
        executor = _make_executor()
        assert executor.position is None
        assert executor.positions == []

        executor._positions["BTC/USDT:USDT"] = LivePosition(
            symbol="BTC/USDT:USDT", direction="LONG",
            entry_price=100_000.0, quantity=0.001,
            entry_order_id="test",
        )
        assert executor.position is not None
        assert executor.position.symbol == "BTC/USDT:USDT"
        assert len(executor.positions) == 1


# ─── Balance Refresh ──────────────────────────────────────────────────────


class TestBalanceRefresh:

    @pytest.mark.asyncio
    async def test_refresh_balance_updates_value(self):
        """refresh_balance() met à jour _exchange_balance."""
        executor = _make_executor()
        executor._exchange_balance = 10_000.0

        # Mock fetch_balance retourne un nouveau solde
        executor._exchange.fetch_balance = AsyncMock(return_value={
            "free": {"USDT": 11_000.0},
            "total": {"USDT": 12_000.0},
        })

        result = await executor.refresh_balance()
        assert result == 12_000.0
        assert executor._exchange_balance == 12_000.0

    @pytest.mark.asyncio
    async def test_refresh_balance_logs_large_change(self):
        """Log WARNING si le solde change de plus de 10%."""
        executor = _make_executor()
        executor._exchange_balance = 10_000.0

        # +50% change
        executor._exchange.fetch_balance = AsyncMock(return_value={
            "free": {"USDT": 15_000.0},
            "total": {"USDT": 15_000.0},
        })

        with patch("backend.execution.executor.logger") as mock_logger:
            await executor.refresh_balance()

        assert executor._exchange_balance == 15_000.0
        mock_logger.warning.assert_called_once()
        args = mock_logger.warning.call_args[0]
        assert "balance" in args[0]
        assert args[1] == pytest.approx(50.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_refresh_balance_no_log_small_change(self):
        """Pas de WARNING si le solde change de moins de 10%."""
        executor = _make_executor()
        executor._exchange_balance = 10_000.0

        # +5% change
        executor._exchange.fetch_balance = AsyncMock(return_value={
            "free": {"USDT": 10_500.0},
            "total": {"USDT": 10_500.0},
        })

        with patch("backend.execution.executor.logger") as mock_logger:
            await executor.refresh_balance()

        assert executor._exchange_balance == 10_500.0
        mock_logger.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_refresh_balance_exchange_error(self):
        """Si fetch_balance échoue, retourne None sans crasher."""
        executor = _make_executor()
        executor._exchange_balance = 10_000.0
        executor._exchange.fetch_balance = AsyncMock(side_effect=Exception("timeout"))

        result = await executor.refresh_balance()
        assert result is None
        # Le solde reste inchangé
        assert executor._exchange_balance == 10_000.0

    @pytest.mark.asyncio
    async def test_refresh_balance_no_exchange(self):
        """Sans exchange, retourne None."""
        executor = _make_executor()
        executor._exchange = None

        result = await executor.refresh_balance()
        assert result is None

    def test_get_status_includes_balance(self):
        """get_status() expose exchange_balance."""
        executor = _make_executor()
        executor._exchange_balance = 5_000.0
        status = executor.get_status()
        assert status["exchange_balance"] == 5_000.0

    def test_get_status_balance_none_when_unset(self):
        """get_status() retourne None si balance jamais fetchée."""
        executor = _make_executor()
        executor._exchange_balance = None
        status = executor.get_status()
        assert status["exchange_balance"] is None

    @pytest.mark.asyncio
    async def test_balance_refresh_loop_runs(self):
        """La boucle refresh appelle refresh_balance périodiquement."""
        executor = _make_executor()
        executor._exchange_balance = 10_000.0
        executor._balance_refresh_interval = 0.05  # 50ms pour le test

        executor._exchange.fetch_balance = AsyncMock(return_value={
            "free": {"USDT": 10_100.0},
            "total": {"USDT": 10_100.0},
        })

        # Lancer la boucle
        task = asyncio.create_task(executor._balance_refresh_loop())
        await asyncio.sleep(0.15)  # Attendre ~3 cycles
        executor._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # fetch_balance a été appelé au moins une fois
        assert executor._exchange.fetch_balance.call_count >= 1
        assert executor._exchange_balance == 10_100.0
