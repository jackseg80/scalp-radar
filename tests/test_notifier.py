"""Tests Notifier — cooldown anti-spam Telegram."""

import pytest
from unittest.mock import AsyncMock

from backend.alerts.notifier import AnomalyType, Notifier


@pytest.mark.asyncio
async def test_anomaly_cooldown():
    """Même anomalie envoyée 3x rapidement → 1 seul message Telegram."""
    mock_telegram = AsyncMock()
    notifier = Notifier(telegram=mock_telegram)

    await notifier.notify_anomaly(AnomalyType.ALL_STRATEGIES_STOPPED)
    await notifier.notify_anomaly(AnomalyType.ALL_STRATEGIES_STOPPED)
    await notifier.notify_anomaly(AnomalyType.ALL_STRATEGIES_STOPPED)

    assert mock_telegram.send_message.call_count == 1


@pytest.mark.asyncio
async def test_different_anomalies_not_throttled():
    """Anomalies différentes → chacune envoyée."""
    mock_telegram = AsyncMock()
    notifier = Notifier(telegram=mock_telegram)

    await notifier.notify_anomaly(AnomalyType.ALL_STRATEGIES_STOPPED)
    await notifier.notify_anomaly(AnomalyType.DATA_STALE, "17536s")

    assert mock_telegram.send_message.call_count == 2


@pytest.mark.asyncio
async def test_cooldown_expired_resends():
    """Après expiration du cooldown → renvoi autorisé."""
    mock_telegram = AsyncMock()
    notifier = Notifier(telegram=mock_telegram)

    await notifier.notify_anomaly(AnomalyType.SL_PLACEMENT_FAILED)
    # Simuler expiration du cooldown (SL = 300s)
    notifier._last_anomaly_sent[AnomalyType.SL_PLACEMENT_FAILED] -= 301
    await notifier.notify_anomaly(AnomalyType.SL_PLACEMENT_FAILED)

    assert mock_telegram.send_message.call_count == 2


@pytest.mark.asyncio
async def test_no_telegram_no_crash():
    """Sans Telegram configuré → pas de crash, juste le log."""
    notifier = Notifier(telegram=None)
    await notifier.notify_anomaly(AnomalyType.WS_DISCONNECTED)
    # Pas d'exception = OK
