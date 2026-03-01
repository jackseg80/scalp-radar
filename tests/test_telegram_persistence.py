"""Tests Sprint 63b — Persistence des alertes Telegram."""

import httpx
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from backend.core.database import Database
from backend.alerts.telegram import TelegramClient


@pytest_asyncio.fixture
async def db():
    """DB en mémoire initialisée avec toutes les tables."""
    database = Database(db_path=":memory:")
    await database.init()
    yield database
    await database.close()


# ─── Tests DB ────────────────────────────────────────────────────────────────


class TestTelegramAlertsDB:
    @pytest.mark.asyncio
    async def test_insert_and_get_alert(self, db):
        """Round-trip : insert + récupération."""
        await db.insert_telegram_alert(
            timestamp="2026-03-01T12:00:00+00:00",
            alert_type="trade",
            message="Test alert",
            strategy="grid_atr",
            success=True,
        )
        alerts = await db.get_telegram_alerts()
        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "trade"
        assert alerts[0]["strategy"] == "grid_atr"
        assert alerts[0]["success"] == 1

    @pytest.mark.asyncio
    async def test_filter_by_type(self, db):
        """Filtre par alert_type."""
        await db.insert_telegram_alert("2026-03-01T12:00:00+00:00", "trade", "t1")
        await db.insert_telegram_alert("2026-03-01T12:01:00+00:00", "heartbeat", "h1")
        await db.insert_telegram_alert("2026-03-01T12:02:00+00:00", "trade", "t2")

        trades = await db.get_telegram_alerts(alert_type="trade")
        assert len(trades) == 2

        hb = await db.get_telegram_alerts(alert_type="heartbeat")
        assert len(hb) == 1

    @pytest.mark.asyncio
    async def test_filter_by_strategy(self, db):
        """Filtre par stratégie."""
        await db.insert_telegram_alert(
            "2026-03-01T12:00:00+00:00", "trade", "t1", strategy="grid_atr"
        )
        await db.insert_telegram_alert(
            "2026-03-01T12:01:00+00:00", "trade", "t2", strategy="grid_multi_tf"
        )

        alerts = await db.get_telegram_alerts(strategy="grid_atr")
        assert len(alerts) == 1
        assert alerts[0]["message"] == "t1"

    @pytest.mark.asyncio
    async def test_filter_by_since(self, db):
        """Filtre par date since."""
        await db.insert_telegram_alert("2026-02-28T12:00:00+00:00", "trade", "old")
        await db.insert_telegram_alert("2026-03-01T12:00:00+00:00", "trade", "new")

        alerts = await db.get_telegram_alerts(since="2026-03-01T00:00:00+00:00")
        assert len(alerts) == 1
        assert alerts[0]["message"] == "new"

    @pytest.mark.asyncio
    async def test_limit(self, db):
        """Respect du limit."""
        for i in range(10):
            await db.insert_telegram_alert(
                f"2026-03-01T12:{i:02d}:00+00:00", "trade", f"m{i}"
            )

        alerts = await db.get_telegram_alerts(limit=5)
        assert len(alerts) == 5

    @pytest.mark.asyncio
    async def test_order_desc(self, db):
        """Les alertes sont retournées du plus récent au plus ancien."""
        await db.insert_telegram_alert("2026-03-01T10:00:00+00:00", "trade", "first")
        await db.insert_telegram_alert("2026-03-01T12:00:00+00:00", "trade", "second")

        alerts = await db.get_telegram_alerts()
        assert alerts[0]["message"] == "second"
        assert alerts[1]["message"] == "first"

    @pytest.mark.asyncio
    async def test_message_truncated(self, db):
        """Messages > 2000 chars sont tronqués."""
        long_msg = "x" * 3000
        await db.insert_telegram_alert("2026-03-01T12:00:00+00:00", "trade", long_msg)

        alerts = await db.get_telegram_alerts()
        assert len(alerts[0]["message"]) == 2000


# ─── Tests TelegramClient persistence ────────────────────────────────────────


class TestTelegramClientPersistence:
    @pytest.mark.asyncio
    async def test_send_message_persists_to_db(self, db):
        """send_message() persiste l'alerte en DB."""
        client = TelegramClient("fake_token", "fake_chat_id")
        client.set_db(db)

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("backend.alerts.telegram.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await client.send_message(
                "test msg", alert_type="trade", strategy="grid_atr"
            )

        assert result is True
        alerts = await db.get_telegram_alerts()
        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "trade"
        assert alerts[0]["strategy"] == "grid_atr"
        assert alerts[0]["success"] == 1

    @pytest.mark.asyncio
    async def test_send_message_without_db_no_crash(self):
        """Sans DB, send_message fonctionne normalement."""
        client = TelegramClient("fake_token", "fake_chat_id")
        # Pas de set_db

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("backend.alerts.telegram.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await client.send_message("test msg", alert_type="trade")

        assert result is True

    @pytest.mark.asyncio
    async def test_send_trade_alert_passes_alert_type(self):
        """send_trade_alert passe alert_type='trade' et strategy."""
        client = TelegramClient("fake_token", "fake_chat_id")
        client.send_message = AsyncMock(return_value=True)

        trade = {
            "net_pnl": 10.0,
            "direction": "LONG",
            "entry_price": 100,
            "exit_price": 110,
            "exit_reason": "tp",
        }
        await client.send_trade_alert(trade, "grid_atr")

        client.send_message.assert_called_once()
        _, kwargs = client.send_message.call_args
        assert kwargs.get("alert_type") == "trade"
        assert kwargs.get("strategy") == "grid_atr"

    @pytest.mark.asyncio
    async def test_send_kill_switch_passes_alert_type(self):
        """send_kill_switch_alert passe alert_type='kill_switch'."""
        client = TelegramClient("fake_token", "fake_chat_id")
        client.send_message = AsyncMock(return_value=True)

        await client.send_kill_switch_alert("grid_atr", 35.0)

        _, kwargs = client.send_message.call_args
        assert kwargs.get("alert_type") == "kill_switch"
        assert kwargs.get("strategy") == "grid_atr"

    @pytest.mark.asyncio
    async def test_send_startup_passes_system_type(self):
        """send_startup_message passe alert_type='system'."""
        client = TelegramClient("fake_token", "fake_chat_id")
        client.send_message = AsyncMock(return_value=True)

        await client.send_startup_message(["grid_atr", "grid_multi_tf"])

        _, kwargs = client.send_message.call_args
        assert kwargs.get("alert_type") == "system"

    @pytest.mark.asyncio
    async def test_failed_send_persists_with_success_false(self, db):
        """Un échec d'envoi est persisté avec success=0."""
        client = TelegramClient("fake_token", "fake_chat_id")
        client.set_db(db)

        with patch("backend.alerts.telegram.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.side_effect = httpx.HTTPError("Network error")
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await client.send_message("test msg", alert_type="anomaly")

        assert result is False
        alerts = await db.get_telegram_alerts()
        assert len(alerts) == 1
        assert alerts[0]["success"] == 0

    @pytest.mark.asyncio
    async def test_send_grid_cycle_closed_passes_trade_type(self):
        """send_grid_cycle_closed passe alert_type='trade'."""
        client = TelegramClient("fake_token", "fake_chat_id")
        client.send_message = AsyncMock(return_value=True)

        await client.send_grid_cycle_closed(
            "BTC/USDT", "LONG", 3, 95000.0, 96000.0, 150.0, "tp", "grid_atr"
        )

        _, kwargs = client.send_message.call_args
        assert kwargs.get("alert_type") == "trade"
        assert kwargs.get("strategy") == "grid_atr"

    @pytest.mark.asyncio
    async def test_send_live_sl_failed_passes_anomaly_type(self):
        """send_live_sl_failed passe alert_type='anomaly'."""
        client = TelegramClient("fake_token", "fake_chat_id")
        client.send_message = AsyncMock(return_value=True)

        await client.send_live_sl_failed("BTC/USDT", "grid_atr")

        _, kwargs = client.send_message.call_args
        assert kwargs.get("alert_type") == "anomaly"
        assert kwargs.get("strategy") == "grid_atr"
