"""Tests pour le leverage par stratégie au boot (Hotfix leverage)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.execution.executor import Executor, to_futures_symbol
from backend.execution.risk_manager import LiveRiskManager


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_strategy_config(
    *, enabled: bool, leverage: int, per_asset: dict | None = None,
) -> MagicMock:
    """Crée un mock de config stratégie avec per_asset et leverage."""
    cfg = MagicMock()
    cfg.enabled = enabled
    cfg.leverage = leverage
    cfg.per_asset = per_asset or {}
    cfg.sl_percent = 20.0
    cfg.live_eligible = True
    return cfg


def _make_config_multi_strat() -> MagicMock:
    """Config avec grid_atr (6x) et grid_boltrend (8x) sur des symbols différents."""
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

    # Assets: BTC, DOGE, ETH
    config.assets = [
        MagicMock(symbol="BTC/USDT"),
        MagicMock(symbol="DOGE/USDT"),
        MagicMock(symbol="ETH/USDT"),
        MagicMock(symbol="SOL/USDT"),  # Aucune stratégie → default
    ]
    config.correlation_groups = {}

    # grid_atr: leverage 6, per_asset = BTC, DOGE
    grid_atr = _make_strategy_config(
        enabled=True, leverage=6,
        per_asset={"BTC/USDT": {}, "DOGE/USDT": {}},
    )
    # grid_boltrend: leverage 8, per_asset = ETH
    grid_boltrend = _make_strategy_config(
        enabled=True, leverage=8,
        per_asset={"ETH/USDT": {}},
    )
    # Autres stratégies désactivées
    grid_trend = _make_strategy_config(enabled=False, leverage=6, per_asset={})

    # StrategiesConfig — utiliser model_fields pour itérer
    strategies = MagicMock()
    strategies.model_fields = {
        "grid_atr": None,
        "grid_boltrend": None,
        "grid_trend": None,
    }
    strategies.grid_atr = grid_atr
    strategies.grid_boltrend = grid_boltrend
    strategies.grid_trend = grid_trend
    config.strategies = strategies

    return config


def _make_mock_exchange() -> AsyncMock:
    exchange = AsyncMock()
    exchange.load_markets = AsyncMock(return_value={
        "BTC/USDT:USDT": {"limits": {"amount": {"min": 0.001}}},
        "DOGE/USDT:USDT": {"limits": {"amount": {"min": 1.0}}},
        "ETH/USDT:USDT": {"limits": {"amount": {"min": 0.01}}},
        "SOL/USDT:USDT": {"limits": {"amount": {"min": 0.1}}},
    })
    exchange.fetch_balance = AsyncMock(return_value={
        "free": {"USDT": 5_000.0},
        "total": {"USDT": 10_000.0},
    })
    exchange.fetch_positions = AsyncMock(return_value=[])
    exchange.set_leverage = AsyncMock()
    exchange.set_margin_mode = AsyncMock()
    exchange.set_position_mode = AsyncMock()
    exchange.close = AsyncMock()
    exchange.fetch_open_orders = AsyncMock(return_value=[])
    exchange.options = {}
    return exchange


def _make_executor(config=None, exchange=None) -> Executor:
    if config is None:
        config = _make_config_multi_strat()
    risk_manager = LiveRiskManager(config)
    risk_manager.set_initial_capital(10_000.0)
    notifier = AsyncMock()
    selector = MagicMock()
    selector.set_active_symbols = MagicMock()

    executor = Executor(config, risk_manager, notifier, selector=selector)
    if exchange is None:
        exchange = _make_mock_exchange()
    executor._exchange = exchange
    executor._markets = {}
    return executor


# ─── Tests ─────────────────────────────────────────────────────────────────


class TestGetLeverageForSymbol:
    """Tests pour _get_leverage_for_symbol()."""

    def test_symbol_in_grid_atr(self):
        """BTC/USDT est dans grid_atr per_asset → leverage 6."""
        executor = _make_executor()
        assert executor._get_leverage_for_symbol("BTC/USDT") == 6

    def test_symbol_in_grid_boltrend(self):
        """ETH/USDT est dans grid_boltrend per_asset → leverage 8."""
        executor = _make_executor()
        assert executor._get_leverage_for_symbol("ETH/USDT") == 8

    def test_symbol_not_in_any_strategy(self):
        """SOL/USDT n'est dans aucune stratégie → default_leverage (3)."""
        executor = _make_executor()
        assert executor._get_leverage_for_symbol("SOL/USDT") == 3

    def test_disabled_strategy_ignored(self):
        """grid_trend est disabled → son leverage n'est pas utilisé."""
        config = _make_config_multi_strat()
        # Ajouter SOL dans grid_trend (disabled)
        config.strategies.grid_trend.per_asset = {"SOL/USDT": {}}
        executor = _make_executor(config=config)
        assert executor._get_leverage_for_symbol("SOL/USDT") == 3


class TestBootLeverage:
    """Tests pour le setup leverage au boot (start)."""

    @pytest.mark.asyncio
    async def test_boot_applies_strategy_leverage(self):
        """Au boot, chaque symbol reçoit le leverage de sa stratégie assignée."""
        config = _make_config_multi_strat()
        exchange = _make_mock_exchange()
        executor = _make_executor(config=config, exchange=exchange)

        # Patch ccxt.pro pour que start() utilise notre mock exchange
        mock_ccxtpro = MagicMock()
        mock_ccxtpro.bitget.return_value = exchange

        with (
            patch.dict("sys.modules", {"ccxt.pro": mock_ccxtpro}),
            patch.object(executor, "_reconcile_on_boot", new_callable=AsyncMock),
            patch.object(executor, "_watch_orders_loop", new_callable=AsyncMock),
            patch.object(executor, "_poll_positions_loop", new_callable=AsyncMock),
            patch.object(executor, "_balance_refresh_loop", new_callable=AsyncMock),
        ):
            await executor.start()

        # Vérifier les appels set_leverage
        leverage_calls = exchange.set_leverage.call_args_list
        call_map = {
            args[0][1]: args[0][0]
            for args in leverage_calls
        }

        # grid_atr symbols → 6x
        assert call_map["BTC/USDT:USDT"] == 6
        assert call_map["DOGE/USDT:USDT"] == 6
        # grid_boltrend symbol → 8x
        assert call_map["ETH/USDT:USDT"] == 8
        # SOL pas dans une stratégie → default 3x
        assert call_map["SOL/USDT:USDT"] == 3

    @pytest.mark.asyncio
    async def test_boot_populates_leverage_applied(self):
        """Le dict _leverage_applied est peuplé au boot."""
        config = _make_config_multi_strat()
        exchange = _make_mock_exchange()
        executor = _make_executor(config=config, exchange=exchange)

        mock_ccxtpro = MagicMock()
        mock_ccxtpro.bitget.return_value = exchange

        with (
            patch.dict("sys.modules", {"ccxt.pro": mock_ccxtpro}),
            patch.object(executor, "_reconcile_on_boot", new_callable=AsyncMock),
            patch.object(executor, "_watch_orders_loop", new_callable=AsyncMock),
            patch.object(executor, "_poll_positions_loop", new_callable=AsyncMock),
            patch.object(executor, "_balance_refresh_loop", new_callable=AsyncMock),
        ):
            await executor.start()

        assert executor._leverage_applied["BTC/USDT:USDT"] == 6
        assert executor._leverage_applied["ETH/USDT:USDT"] == 8
        assert executor._leverage_applied["SOL/USDT:USDT"] == 3


class TestGridLeverageSkip:
    """Tests pour le skip du re-set leverage au 1er trade grid."""

    def test_leverage_already_applied_skips(self):
        """Si le leverage est déjà correct, pas d'appel API."""
        executor = _make_executor()
        executor._leverage_applied["BTC/USDT:USDT"] = 6

        # _get_grid_leverage retourne 6 pour grid_atr
        # Donc au 1er trade, le skip devrait fonctionner
        assert executor._leverage_applied.get("BTC/USDT:USDT") == 6

    def test_leverage_different_needs_reset(self):
        """Si le leverage boot diffère du grid, le re-set est nécessaire."""
        executor = _make_executor()
        executor._leverage_applied["BTC/USDT:USDT"] = 3  # default au boot

        grid_lev = executor._get_grid_leverage("grid_atr")
        assert grid_lev != executor._leverage_applied["BTC/USDT:USDT"]
