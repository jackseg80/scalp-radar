"""Tests détection divergence leverage au boot (Sprint 36b-safety)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.execution.executor import Executor
from backend.execution.risk_manager import LiveRiskManager


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_config(leverage_atr: int = 6) -> MagicMock:
    """Config avec grid_atr sur BTC/USDT + DOGE/USDT (leverage configurable)."""
    config = MagicMock()
    config.secrets.live_trading = True
    config.secrets.bitget_api_key = "key"
    config.secrets.bitget_secret = "secret"
    config.secrets.bitget_passphrase = "pass"
    config.risk.position.default_leverage = 3
    config.risk.margin.mode = "cross"
    config.risk.fees.taker_percent = 0.06
    config.risk.fees.maker_percent = 0.02
    config.risk.kill_switch.max_session_loss_percent = 5.0
    config.risk.kill_switch.grid_max_session_loss_percent = 25.0

    config.assets = [
        MagicMock(symbol="BTC/USDT"),
        MagicMock(symbol="DOGE/USDT"),
    ]
    config.correlation_groups = {}

    grid_atr = MagicMock()
    grid_atr.enabled = True
    grid_atr.leverage = leverage_atr
    grid_atr.per_asset = {"BTC/USDT": {}, "DOGE/USDT": {}}
    grid_atr.sl_percent = 20.0
    grid_atr.live_eligible = True

    strategies = MagicMock()
    strategies.model_fields = {"grid_atr": None}
    strategies.grid_atr = grid_atr
    config.strategies = strategies

    return config


def _make_exchange(btc_position_leverage: int | None = None) -> AsyncMock:
    """Exchange mock. Si btc_position_leverage, BTC a une position ouverte à ce leverage."""
    exchange = AsyncMock()
    exchange.load_markets = AsyncMock(return_value={
        "BTC/USDT:USDT": {"limits": {"amount": {"min": 0.001}}},
        "DOGE/USDT:USDT": {"limits": {"amount": {"min": 1.0}}},
    })
    exchange.fetch_balance = AsyncMock(return_value={
        "free": {"USDT": 5_000.0},
        "total": {"USDT": 10_000.0},
    })
    exchange.set_leverage = AsyncMock()
    exchange.set_margin_mode = AsyncMock()
    exchange.set_position_mode = AsyncMock()
    exchange.close = AsyncMock()
    exchange.fetch_open_orders = AsyncMock(return_value=[])
    exchange.options = {}

    def _fetch_positions(symbols=None):
        if btc_position_leverage is not None and symbols and "BTC/USDT:USDT" in symbols:
            return [{"contracts": 0.05, "leverage": btc_position_leverage, "symbol": "BTC/USDT:USDT"}]
        return []

    exchange.fetch_positions = AsyncMock(side_effect=_fetch_positions)
    return exchange


def _make_executor(config: MagicMock, exchange: AsyncMock) -> Executor:
    risk_manager = LiveRiskManager(config)
    risk_manager.set_initial_capital(10_000.0)
    notifier = AsyncMock()
    selector = MagicMock()
    selector.set_active_symbols = MagicMock()

    executor = Executor(config, risk_manager, notifier, selector=selector,
                        strategy_name="grid_atr")
    executor._exchange = exchange
    executor._markets = {}
    return executor


def _patches(executor: Executor, exchange: AsyncMock):
    return (
        patch.object(executor, "_create_exchange", return_value=exchange),
        patch.object(executor, "_reconcile_on_boot", new_callable=AsyncMock),
        patch.object(executor, "_watch_orders_loop", new_callable=AsyncMock),
        patch.object(executor, "_poll_positions_loop", new_callable=AsyncMock),
        patch.object(executor, "_balance_refresh_loop", new_callable=AsyncMock),
    )


# ─── Tests ─────────────────────────────────────────────────────────────────


class TestLeverageDivergenceDetection:

    @pytest.mark.asyncio
    async def test_start_skips_leverage_on_open_positions_with_divergence(self):
        """set_leverage non appelé pour un symbol avec position ouverte et leverage divergent."""
        # Config: 6x, position ouverte: 3x → divergence
        config = _make_config(leverage_atr=6)
        exchange = _make_exchange(btc_position_leverage=3)
        executor = _make_executor(config, exchange)

        with _patches(executor, exchange)[0], _patches(executor, exchange)[1], \
             _patches(executor, exchange)[2], _patches(executor, exchange)[3], \
             _patches(executor, exchange)[4]:
            await executor.start()

        # set_leverage ne doit PAS être appelé pour BTC (position ouverte)
        btc_calls = [
            c for c in exchange.set_leverage.call_args_list
            if "BTC/USDT:USDT" in str(c)
        ]
        assert len(btc_calls) == 0

        # La divergence doit être enregistrée
        assert "BTC/USDT:USDT" in executor._leverage_divergent
        assert executor._leverage_divergent["BTC/USDT:USDT"] == (3, 6)

    @pytest.mark.asyncio
    async def test_start_sets_leverage_on_assets_without_positions(self):
        """set_leverage est bien appelé pour les assets sans position ouverte."""
        # Config: 6x, BTC a position ouverte (divergente), DOGE libre
        config = _make_config(leverage_atr=6)
        exchange = _make_exchange(btc_position_leverage=3)
        executor = _make_executor(config, exchange)

        with _patches(executor, exchange)[0], _patches(executor, exchange)[1], \
             _patches(executor, exchange)[2], _patches(executor, exchange)[3], \
             _patches(executor, exchange)[4]:
            await executor.start()

        # DOGE doit recevoir set_leverage (pas de position ouverte)
        doge_calls = [
            c for c in exchange.set_leverage.call_args_list
            if "DOGE/USDT:USDT" in str(c)
        ]
        assert len(doge_calls) == 1
        assert doge_calls[0].args == (6, "DOGE/USDT:USDT")

    @pytest.mark.asyncio
    async def test_start_no_divergence_proceeds_normally(self):
        """Position ouverte mais même leverage que config → aucune divergence."""
        # Config: 6x, position ouverte: 6x → pas de divergence
        config = _make_config(leverage_atr=6)
        exchange = _make_exchange(btc_position_leverage=6)
        executor = _make_executor(config, exchange)

        with _patches(executor, exchange)[0], _patches(executor, exchange)[1], \
             _patches(executor, exchange)[2], _patches(executor, exchange)[3], \
             _patches(executor, exchange)[4]:
            await executor.start()

        # Aucune divergence enregistrée
        assert executor._leverage_divergent == {}

        # Telegram notify_leverage_divergence non appelé
        executor._notifier.notify_leverage_divergence.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_divergence_telegram_alert_sent(self):
        """Une alerte Telegram est envoyée quand une divergence leverage est détectée."""
        # Config: 7x, position ouverte: 3x → divergence
        config = _make_config(leverage_atr=7)
        exchange = _make_exchange(btc_position_leverage=3)
        executor = _make_executor(config, exchange)

        with _patches(executor, exchange)[0], _patches(executor, exchange)[1], \
             _patches(executor, exchange)[2], _patches(executor, exchange)[3], \
             _patches(executor, exchange)[4]:
            await executor.start()

        # notify_leverage_divergence doit être appelé exactement une fois
        executor._notifier.notify_leverage_divergence.assert_awaited_once()

        # Le premier argument est le nom de la stratégie
        call_args = executor._notifier.notify_leverage_divergence.call_args
        assert call_args.args[0] == "grid_atr"
        # Les détails mentionnent BTC avec les leverages
        assert "BTC/USDT" in call_args.args[1]
        assert "3x" in call_args.args[1]
        assert "7x" in call_args.args[1]
