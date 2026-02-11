"""Tests pour le Heartbeat."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.alerts.heartbeat import Heartbeat


class TestHeartbeat:
    """Tests pour le Heartbeat."""

    def test_build_message_format(self):
        """Vérifie le format du message heartbeat."""
        telegram = MagicMock()
        simulator = MagicMock()
        simulator.get_all_status.return_value = {
            "vwap_rsi": {
                "net_pnl": 250.0,
                "total_trades": 10,
                "wins": 7,
                "is_active": True,
            },
            "momentum": {
                "net_pnl": -50.0,
                "total_trades": 5,
                "wins": 2,
                "is_active": True,
            },
        }

        hb = Heartbeat(telegram, simulator, interval_seconds=60)
        message = hb._build_message()

        assert "Heartbeat" in message
        assert "+200.00$" in message  # 250 - 50
        assert "Trades: 15" in message
        assert "60%" in message  # 9/15
        assert "vwap_rsi" in message
        assert "momentum" in message

    def test_build_message_no_trades(self):
        """Message heartbeat sans aucun trade."""
        telegram = MagicMock()
        simulator = MagicMock()
        simulator.get_all_status.return_value = {
            "vwap_rsi": {
                "net_pnl": 0.0,
                "total_trades": 0,
                "wins": 0,
                "is_active": True,
            },
        }

        hb = Heartbeat(telegram, simulator, interval_seconds=60)
        message = hb._build_message()

        assert "Trades: 0" in message
        assert "0%" in message

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
