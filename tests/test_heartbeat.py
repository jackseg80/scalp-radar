"""Tests pour le Heartbeat."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.alerts.heartbeat import Heartbeat


class TestHeartbeat:
    """Tests pour le Heartbeat."""

    def test_build_message_format_sim(self):
        """Vérifie le format du message heartbeat en mode Simulator."""
        telegram = MagicMock()
        simulator = MagicMock()
        
        # Mocker la config pour live_trading=False (données simulator utilisées pour trades)
        config = MagicMock()
        config.secrets.live_trading = False
        simulator.config = config

        simulator.get_all_status.return_value = {
            "vwap_rsi": {
                "net_pnl": 250.0,
                "total_trades": 10,
                "wins": 6,
                "is_active": True,
            },
            "momentum": {
                "net_pnl": -50.0,
                "total_trades": 5,
                "wins": 3,
                "is_active": True,
            },
        }

        hb = Heartbeat(telegram, simulator, interval_seconds=60)
        message = hb._build_message()

        assert "Heartbeat Scalp Radar" in message
        assert "PnL session: <b>+0.00$</b>" in message # Car live_pnl=0 par défaut
        assert "Trades session: 15" in message
        assert "Stratégies: momentum, vwap_rsi" in message

    def test_build_message_live_data(self):
        """Vérifie l'inclusion des données LIVE dans le heartbeat."""
        telegram = MagicMock()
        simulator = MagicMock()
        
        config = MagicMock()
        config.secrets.live_trading = True
        simulator.config = config
        simulator.get_all_status.return_value = {}

        # Mock ExecutorManager
        executor_mgr = MagicMock()
        executor = MagicMock()
        executor.is_enabled = True
        executor.is_connected = True
        executor.get_status.return_value = {
            "risk_manager": {"session_pnl": 25.08},
            "positions": [
                {"symbol": "ETH/USDT:USDT", "direction": "LONG", "unrealized_pnl": 1.2},
                {"symbol": "SOL/USDT:USDT", "direction": "LONG", "unrealized_pnl": 1.5},
            ]
        }
        executor_mgr.executors = {"grid_atr": executor}
        
        # Mock RiskManager for trade history count
        risk_mgr = MagicMock()
        risk_mgr._trade_history = [MagicMock()] * 49
        executor_mgr.risk_managers = {"grid_atr": risk_mgr}

        hb = Heartbeat(telegram, simulator, interval_seconds=60, executor_mgr=executor_mgr)
        message = hb._build_message()

        assert "PnL session: <b>+25.08$</b>" in message
        assert "Latent: <b>+2.70$</b>" in message
        assert "Positions: 2 ouvertes (ETH LONG, SOL LONG)" in message
        assert "Trades session: 49" in message
        assert "Stratégies: grid_atr" in message

    def test_build_message_filtering_live_eligible(self):
        """Vérifie le filtrage des stratégies non live_eligible dans le fallback sim."""
        telegram = MagicMock()
        simulator = MagicMock()
        
        config = MagicMock()
        config.secrets.live_trading = True
        # vwap_rsi est live_eligible, momentum ne l'est pas
        config.strategies.vwap_rsi.live_eligible = True
        config.strategies.momentum.live_eligible = False
        simulator.config = config

        simulator.get_all_status.return_value = {
            "vwap_rsi": {"is_active": True},
            "momentum": {"is_active": True},
        }

        hb = Heartbeat(telegram, simulator, interval_seconds=60)
        message = hb._build_message()

        assert "vwap_rsi" in message
        assert "momentum" not in message

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        """Vérifie que stop() annule la tâche."""
        telegram = MagicMock()
        simulator = MagicMock()
        simulator.get_all_status.return_value = {}

        hb = Heartbeat(telegram, simulator, interval_seconds=3600)
        await hb.start()

        assert hb._task is not None
        assert not hb._task.done()

        await hb.stop()
        assert hb._running is False
