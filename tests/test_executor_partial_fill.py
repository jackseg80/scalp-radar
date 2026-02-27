"""Tests pour la protection contre les partial fills sur close grid.

5 scénarios :
1. Partial fill → 2ème market order envoyé pour le résidu
2. Partial fill + retry échoue → anomaly notifiée
3. Full fill → pas de retry
4. Floating point tolerance → résidu < min_qty ignoré
5. Post-close fetch_positions détecte position résiduelle → cleanup
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from backend.alerts.notifier import AnomalyType
from backend.execution.executor import (
    Executor,
    GridLivePosition,
    GridLiveState,
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
        MagicMock(
            symbol="ETH/USDT", min_order_size=0.01,
            tick_size=0.01, correlation_group=None,
        ),
    ]
    config.correlation_groups = {}
    config.strategies.grid_multi_tf.sl_percent = 20.0
    config.strategies.grid_multi_tf.leverage = 7
    config.strategies.grid_multi_tf.live_eligible = True
    return config


def _make_mock_exchange(
    *,
    close_filled: float = 0.07,
    close_amount: float = 0.07,
) -> AsyncMock:
    exchange = AsyncMock()
    exchange.load_markets = AsyncMock(return_value={
        "ETH/USDT:USDT": {
            "limits": {"amount": {"min": 0.01}},
            "precision": {"amount": 2, "price": 2},
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
        "id": "close_order_1",
        "status": "closed",
        "filled": close_filled,
        "average": 2500.0,
        "fee": {"cost": 0.10},
    })
    exchange.cancel_order = AsyncMock()
    exchange.fetch_order = AsyncMock(return_value={"status": "open"})
    exchange.fetch_my_trades = AsyncMock(return_value=[])
    exchange.watch_orders = AsyncMock(return_value=[])
    exchange.close = AsyncMock()
    exchange.fetch_open_orders = AsyncMock(return_value=[])
    exchange.amount_to_precision = MagicMock(
        side_effect=lambda sym, qty: f"{round(qty, 2):.2f}",
    )
    exchange.price_to_precision = MagicMock(
        side_effect=lambda sym, price: f"{round(price, 2):.2f}",
    )
    return exchange


def _make_executor(config=None, exchange=None) -> Executor:
    if config is None:
        config = _make_config()
    risk_manager = LiveRiskManager(config)
    risk_manager.set_initial_capital(10_000.0)
    notifier = AsyncMock()

    executor = Executor(config, risk_manager, notifier)
    if exchange is None:
        exchange = _make_mock_exchange()
    executor._exchange = exchange
    executor._markets = exchange.load_markets.return_value
    executor._running = True
    executor._connected = True
    return executor


def _make_grid_state(
    direction: str = "SHORT",
    strategy: str = "grid_multi_tf",
) -> GridLiveState:
    """3 entries : 0.03 + 0.03 + 0.01 = 0.07 total (scénario ETH réel)."""
    return GridLiveState(
        symbol="ETH/USDT",
        direction=direction,
        strategy_name=strategy,
        leverage=7,
        sl_order_id="sl_trigger_1",
        sl_price=2700.0,
        opened_at=datetime.now(tz=timezone.utc),
        positions=[
            GridLivePosition(
                level=0, entry_price=2500.0, quantity=0.03,
                entry_order_id="entry_1",
                entry_time=datetime.now(tz=timezone.utc),
            ),
            GridLivePosition(
                level=1, entry_price=2520.0, quantity=0.03,
                entry_order_id="entry_2",
                entry_time=datetime.now(tz=timezone.utc),
            ),
            GridLivePosition(
                level=2, entry_price=2540.0, quantity=0.01,
                entry_order_id="entry_3",
                entry_time=datetime.now(tz=timezone.utc),
            ),
        ],
    )


def _make_close_event(
    exit_reason: str = "tp_global",
    exit_price: float = 2450.0,
) -> TradeEvent:
    return TradeEvent(
        event_type=TradeEventType.CLOSE,
        strategy_name="grid_multi_tf",
        symbol="ETH/USDT",
        direction="SHORT",
        entry_price=2500.0,
        quantity=0.07,
        tp_price=0.0,
        sl_price=0.0,
        score=0.0,
        timestamp=datetime.now(tz=timezone.utc),
        market_regime="RANGING",
        exit_reason=exit_reason,
        exit_price=exit_price,
    )


# ─── Tests _handle_partial_close_fill ─────────────────────────────────────


class TestPartialCloseFillRetry:
    """Partial fill sur market close → 2ème market order pour le résidu."""

    @pytest.mark.asyncio
    async def test_partial_fill_sends_retry_order(self):
        """filled=0.06 sur 0.07 → retry pour 0.01 résiduel."""
        exchange = _make_mock_exchange(close_filled=0.06)
        executor = _make_executor(exchange=exchange)
        futures_sym = "ETH/USDT:USDT"

        await executor._handle_partial_close_fill(
            futures_sym, "buy", 0.07, 0.06, "grid_multi_tf",
        )

        # 2ème appel create_order pour le résidu
        assert exchange.create_order.call_count == 1
        exchange.create_order.assert_called_once_with(
            futures_sym, "market", "buy", 0.01,
            params={"reduceOnly": True},
        )

        # Anomaly notifiée
        executor._notifier.notify_anomaly.assert_called_once()
        call_args = executor._notifier.notify_anomaly.call_args
        assert call_args[0][0] == AnomalyType.PARTIAL_FILL

    @pytest.mark.asyncio
    async def test_partial_fill_retry_fails_still_notifies(self):
        """Si le 2ème market order échoue, on notifie quand même."""
        exchange = _make_mock_exchange(close_filled=0.06)
        exchange.create_order = AsyncMock(side_effect=Exception("insufficient balance"))
        executor = _make_executor(exchange=exchange)
        futures_sym = "ETH/USDT:USDT"

        await executor._handle_partial_close_fill(
            futures_sym, "buy", 0.07, 0.06, "grid_multi_tf",
        )

        # Anomaly toujours notifiée malgré l'échec
        executor._notifier.notify_anomaly.assert_called_once()
        call_args = executor._notifier.notify_anomaly.call_args
        assert call_args[0][0] == AnomalyType.PARTIAL_FILL


class TestFullFillNoRetry:
    """Full fill → aucune action supplémentaire."""

    @pytest.mark.asyncio
    async def test_full_fill_no_retry(self):
        """filled == requested → pas de retry."""
        exchange = _make_mock_exchange(close_filled=0.07)
        executor = _make_executor(exchange=exchange)

        await executor._handle_partial_close_fill(
            "ETH/USDT:USDT", "buy", 0.07, 0.07, "grid_multi_tf",
        )

        exchange.create_order.assert_not_called()
        executor._notifier.notify_anomaly.assert_not_called()


class TestFloatingPointTolerance:
    """Résidu < min_qty (floating point) → considéré comme full fill."""

    @pytest.mark.asyncio
    async def test_floating_point_residual_below_min_qty(self):
        """filled=0.06999... → résidu ~0.00001 < min_qty=0.01 → ignoré."""
        exchange = _make_mock_exchange()
        executor = _make_executor(exchange=exchange)

        # 0.07 - 0.06999 = 0.00001 qui une fois rounded < min_qty=0.01
        await executor._handle_partial_close_fill(
            "ETH/USDT:USDT", "buy", 0.07, 0.06999, "grid_multi_tf",
        )

        exchange.create_order.assert_not_called()
        executor._notifier.notify_anomaly.assert_not_called()

    @pytest.mark.asyncio
    async def test_residual_exactly_min_qty_triggers_retry(self):
        """Résidu == min_qty → tradeable, donc retry."""
        exchange = _make_mock_exchange()
        executor = _make_executor(exchange=exchange)

        # 0.07 - 0.06 = 0.01 == min_qty → tradeable → retry
        await executor._handle_partial_close_fill(
            "ETH/USDT:USDT", "buy", 0.07, 0.06, "grid_multi_tf",
        )

        exchange.create_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_residual_above_min_qty_triggers_retry(self):
        """Résidu > min_qty → retry."""
        exchange = _make_mock_exchange()
        executor = _make_executor(exchange=exchange)

        # 0.07 - 0.04 = 0.03 > min_qty=0.01 → retry
        await executor._handle_partial_close_fill(
            "ETH/USDT:USDT", "buy", 0.07, 0.04, "grid_multi_tf",
        )

        exchange.create_order.assert_called_once()


# ─── Tests _verify_no_residual_position ───────────────────────────────────


class TestPostClosePositionCheck:
    """Vérification post-close via fetch_positions."""

    @pytest.mark.asyncio
    async def test_no_residual_position_ok(self):
        """fetch_positions retourne [] → rien à faire."""
        exchange = _make_mock_exchange()
        exchange.fetch_positions = AsyncMock(return_value=[])
        executor = _make_executor(exchange=exchange)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await executor._verify_no_residual_position(
                "ETH/USDT:USDT", "buy", "grid_multi_tf",
            )

        # Pas de create_order supplémentaire
        exchange.create_order.assert_not_called()
        executor._notifier.notify_anomaly.assert_not_called()

    @pytest.mark.asyncio
    async def test_residual_position_triggers_cleanup(self):
        """fetch_positions retourne 0.01 contracts → market close envoyé."""
        exchange = _make_mock_exchange()
        exchange.fetch_positions = AsyncMock(return_value=[
            {"contracts": "0.01", "symbol": "ETH/USDT:USDT"},
        ])
        executor = _make_executor(exchange=exchange)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await executor._verify_no_residual_position(
                "ETH/USDT:USDT", "buy", "grid_multi_tf",
            )

        exchange.create_order.assert_called_once_with(
            "ETH/USDT:USDT", "market", "buy", 0.01,
            params={"reduceOnly": True},
        )
        executor._notifier.notify_anomaly.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_positions_failure_graceful(self):
        """fetch_positions échoue → log warning, pas de crash."""
        exchange = _make_mock_exchange()
        exchange.fetch_positions = AsyncMock(side_effect=Exception("network"))
        executor = _make_executor(exchange=exchange)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            # Ne doit pas lever d'exception
            await executor._verify_no_residual_position(
                "ETH/USDT:USDT", "buy", "grid_multi_tf",
            )

        exchange.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_zero_contracts_no_action(self):
        """fetch_positions retourne 0 contracts → rien à faire."""
        exchange = _make_mock_exchange()
        exchange.fetch_positions = AsyncMock(return_value=[
            {"contracts": "0", "symbol": "ETH/USDT:USDT"},
        ])
        executor = _make_executor(exchange=exchange)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await executor._verify_no_residual_position(
                "ETH/USDT:USDT", "buy", "grid_multi_tf",
            )

        exchange.create_order.assert_not_called()


# ─── Tests intégration _close_grid_cycle ──────────────────────────────────


class TestCloseGridCyclePartialFill:
    """Intégration : _close_grid_cycle détecte et gère partial fill."""

    @pytest.mark.asyncio
    async def test_close_grid_partial_fill_retries(self):
        """Market close filled=0.06 sur 0.07 → 2ème order pour résidu 0.01."""
        exchange = _make_mock_exchange(close_filled=0.06)
        # Premier appel = close (filled=0.06), deuxième = retry résidu
        close_return = {
            "id": "close_order_1", "status": "closed",
            "filled": 0.06, "average": 2450.0, "fee": {"cost": 0.08},
        }
        retry_return = {
            "id": "retry_order_1", "status": "closed",
            "filled": 0.01, "average": 2450.0, "fee": {"cost": 0.01},
        }
        exchange.create_order = AsyncMock(side_effect=[close_return, retry_return])

        executor = _make_executor(exchange=exchange)
        futures_sym = "ETH/USDT:USDT"
        state = _make_grid_state()
        executor._grid_states[futures_sym] = state
        executor._risk_manager.register_position({
            "symbol": futures_sym, "direction": "SHORT",
            "quantity": 0.07, "entry_price": 2500.0, "sl_price": 2700.0,
        })

        event = _make_close_event()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await executor._close_grid_cycle(event)

        # 2 create_order : close initial + retry résidu
        assert exchange.create_order.call_count == 2
        # Vérifier que le 2ème est bien pour le résidu
        retry_call = exchange.create_order.call_args_list[1]
        assert retry_call[0] == (futures_sym, "market", "buy", 0.01)
        assert retry_call[1] == {"params": {"reduceOnly": True}}

        # Grid state nettoyé
        assert futures_sym not in executor._grid_states

    @pytest.mark.asyncio
    async def test_close_grid_full_fill_no_retry(self):
        """Full fill → aucune action supplémentaire."""
        close_return = {
            "id": "close_order_1", "status": "closed",
            "filled": 0.07, "average": 2450.0, "fee": {"cost": 0.10},
        }
        exchange = _make_mock_exchange(close_filled=0.07)
        exchange.create_order = AsyncMock(return_value=close_return)
        executor = _make_executor(exchange=exchange)
        futures_sym = "ETH/USDT:USDT"
        state = _make_grid_state()
        executor._grid_states[futures_sym] = state
        executor._risk_manager.register_position({
            "symbol": futures_sym, "direction": "SHORT",
            "quantity": 0.07, "entry_price": 2500.0, "sl_price": 2700.0,
        })

        event = _make_close_event()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await executor._close_grid_cycle(event)

        # Un seul create_order (le close initial)
        assert exchange.create_order.call_count == 1
        assert futures_sym not in executor._grid_states


# ─── Tests intégration _handle_grid_sl_executed ───────────────────────────


class TestHandleGridSlPartialFill:
    """_handle_grid_sl_executed vérifie position résiduelle après cleanup."""

    @pytest.mark.asyncio
    async def test_sl_executed_with_residual_position(self):
        """SL exécuté mais 0.01 résiduel détecté → cleanup market order."""
        exchange = _make_mock_exchange()
        exchange.fetch_positions = AsyncMock(return_value=[
            {"contracts": "0.01", "symbol": "ETH/USDT:USDT"},
        ])
        cleanup_return = {
            "id": "cleanup_1", "status": "closed",
            "filled": 0.01, "average": 2700.0,
        }
        exchange.create_order = AsyncMock(return_value=cleanup_return)

        executor = _make_executor(exchange=exchange)
        futures_sym = "ETH/USDT:USDT"
        state = _make_grid_state()
        executor._grid_states[futures_sym] = state
        executor._risk_manager.register_position({
            "symbol": futures_sym, "direction": "SHORT",
            "quantity": 0.07, "entry_price": 2500.0, "sl_price": 2700.0,
        })

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await executor._handle_grid_sl_executed(
                futures_sym, state, 2700.0, exit_fee=0.12,
            )

        # cleanup order envoyé pour le résidu
        exchange.create_order.assert_called_once_with(
            futures_sym, "market", "buy", 0.01,
            params={"reduceOnly": True},
        )
        assert futures_sym not in executor._grid_states

    @pytest.mark.asyncio
    async def test_sl_executed_no_residual(self):
        """SL exécuté proprement → pas de cleanup."""
        exchange = _make_mock_exchange()
        exchange.fetch_positions = AsyncMock(return_value=[])
        executor = _make_executor(exchange=exchange)
        futures_sym = "ETH/USDT:USDT"
        state = _make_grid_state()
        executor._grid_states[futures_sym] = state
        executor._risk_manager.register_position({
            "symbol": futures_sym, "direction": "SHORT",
            "quantity": 0.07, "entry_price": 2500.0, "sl_price": 2700.0,
        })

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await executor._handle_grid_sl_executed(
                futures_sym, state, 2700.0, exit_fee=0.12,
            )

        exchange.create_order.assert_not_called()
        assert futures_sym not in executor._grid_states


# ─── Test _get_min_quantity ───────────────────────────────────────────────


class TestGetMinQuantity:
    """Vérification du helper _get_min_quantity."""

    def test_returns_min_from_markets(self):
        executor = _make_executor()
        assert executor._get_min_quantity("ETH/USDT:USDT") == 0.01

    def test_returns_zero_for_unknown_symbol(self):
        executor = _make_executor()
        assert executor._get_min_quantity("UNKNOWN/USDT:USDT") == 0.0
