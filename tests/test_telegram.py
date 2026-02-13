"""Tests pour le client Telegram et le Notifier."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.alerts.notifier import AnomalyType, Notifier
from backend.alerts.telegram import TelegramClient


class TestTelegramClient:
    """Tests pour TelegramClient."""

    @pytest.mark.asyncio
    async def test_send_message_success(self):
        client = TelegramClient("fake_token", "fake_chat_id")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("backend.alerts.telegram.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await client.send_message("test message")

        assert result is True
        mock_instance.post.assert_called_once()
        call_kwargs = mock_instance.post.call_args
        assert call_kwargs[1]["json"]["text"] == "test message"
        assert call_kwargs[1]["json"]["chat_id"] == "fake_chat_id"

    @pytest.mark.asyncio
    async def test_send_trade_alert_format(self):
        client = TelegramClient("fake_token", "fake_chat_id")
        client.send_message = AsyncMock(return_value=True)

        trade = {
            "direction": "LONG",
            "entry_price": 98500.0,
            "exit_price": 98800.0,
            "net_pnl": 15.50,
            "exit_reason": "tp",
        }
        await client.send_trade_alert(trade, "vwap_rsi")

        client.send_message.assert_called_once()
        text = client.send_message.call_args[0][0]
        assert "vwap_rsi" in text
        assert "LONG" in text
        assert "98500.00" in text
        assert "+15.50$" in text
        assert "tp" in text

    @pytest.mark.asyncio
    async def test_send_kill_switch_alert_format(self):
        client = TelegramClient("fake_token", "fake_chat_id")
        client.send_message = AsyncMock(return_value=True)

        await client.send_kill_switch_alert("momentum", 5.2)

        text = client.send_message.call_args[0][0]
        assert "KILL SWITCH" in text
        assert "momentum" in text
        assert "5.2%" in text

    @pytest.mark.asyncio
    async def test_send_startup_message_format(self):
        client = TelegramClient("fake_token", "fake_chat_id")
        client.send_message = AsyncMock(return_value=True)

        await client.send_startup_message(["vwap_rsi", "momentum"])

        text = client.send_message.call_args[0][0]
        assert "démarré" in text
        assert "vwap_rsi" in text
        assert "momentum" in text


class TestNotifier:
    """Tests pour le Notifier."""

    @pytest.mark.asyncio
    async def test_notifier_without_telegram_logs_only(self):
        """Sans Telegram, les notifications sont juste loguées (pas d'erreur)."""
        notifier = Notifier(telegram=None)

        # Aucune exception
        await notifier.notify_trade({"net_pnl": 10.0}, "vwap_rsi")
        await notifier.notify_kill_switch("momentum", 5.0)
        await notifier.notify_anomaly(AnomalyType.WS_DISCONNECTED)
        await notifier.notify_startup(["vwap_rsi"])
        await notifier.notify_shutdown()

    @pytest.mark.asyncio
    async def test_notifier_dispatches_to_telegram(self):
        """Avec Telegram, les notifications sont envoyées."""
        telegram = MagicMock()
        telegram.send_trade_alert = AsyncMock(return_value=True)
        telegram.send_kill_switch_alert = AsyncMock(return_value=True)
        telegram.send_message = AsyncMock(return_value=True)
        telegram.send_startup_message = AsyncMock(return_value=True)
        telegram.send_shutdown_message = AsyncMock(return_value=True)

        notifier = Notifier(telegram=telegram)

        await notifier.notify_trade({"net_pnl": 10.0}, "vwap_rsi")
        telegram.send_trade_alert.assert_called_once()

        await notifier.notify_kill_switch("momentum", 5.0)
        telegram.send_kill_switch_alert.assert_called_once()

        await notifier.notify_startup(["vwap_rsi"])
        telegram.send_startup_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_notifier_anomaly_structured(self):
        """Les anomalies utilisent des messages structurés."""
        telegram = MagicMock()
        telegram.send_message = AsyncMock(return_value=True)

        notifier = Notifier(telegram=telegram)

        await notifier.notify_anomaly(AnomalyType.WS_DISCONNECTED, "depuis 45s")

        text = telegram.send_message.call_args[0][0]
        assert "Anomalie" in text
        assert "WebSocket déconnecté" in text
        assert "depuis 45s" in text


# ─── Sprint 12 : messages grid ──────────────────────────────────────────


class TestTelegramGrid:
    """Tests pour les messages Telegram grid DCA."""

    @pytest.mark.asyncio
    async def test_send_grid_level_opened_format(self):
        """Format message ouverture niveau grid."""
        client = TelegramClient("fake_token", "fake_chat_id")
        client.send_message = AsyncMock(return_value=True)

        await client.send_grid_level_opened(
            symbol="BTC/USDT",
            direction="LONG",
            level_num=2,
            quantity=0.001,
            entry_price=48_000.0,
            avg_price=49_000.0,
            sl_price=39_200.0,
            strategy="envelope_dca",
        )

        client.send_message.assert_called_once()
        text = client.send_message.call_args[0][0]
        assert "GRID ENTRY #2" in text
        assert "LONG" in text
        assert "BTC/USDT" in text
        assert "envelope_dca" in text
        assert "48000.00" in text
        assert "49000.00" in text
        assert "39200.00" in text

    @pytest.mark.asyncio
    async def test_send_grid_cycle_closed_format(self):
        """Format message fermeture cycle grid."""
        client = TelegramClient("fake_token", "fake_chat_id")
        client.send_message = AsyncMock(return_value=True)

        await client.send_grid_cycle_closed(
            symbol="BTC/USDT",
            direction="LONG",
            num_positions=3,
            avg_entry=49_000.0,
            exit_price=51_000.0,
            net_pnl=12.50,
            exit_reason="tp_global",
            strategy="envelope_dca",
        )

        client.send_message.assert_called_once()
        text = client.send_message.call_args[0][0]
        assert "GRID CLOSE" in text
        assert "WIN" in text
        assert "LONG" in text
        assert "3" in text
        assert "49000.00" in text
        assert "51000.00" in text
        assert "+12.50$" in text
        assert "tp_global" in text

    @pytest.mark.asyncio
    async def test_send_grid_cycle_closed_loss(self):
        """Format message fermeture cycle grid en perte."""
        client = TelegramClient("fake_token", "fake_chat_id")
        client.send_message = AsyncMock(return_value=True)

        await client.send_grid_cycle_closed(
            symbol="ETH/USDT",
            direction="SHORT",
            num_positions=2,
            avg_entry=3_000.0,
            exit_price=3_100.0,
            net_pnl=-8.50,
            exit_reason="sl_global",
            strategy="envelope_dca",
        )

        text = client.send_message.call_args[0][0]
        assert "LOSS" in text
        assert "SHORT" in text
        assert "-8.50$" in text
        assert "sl_global" in text
