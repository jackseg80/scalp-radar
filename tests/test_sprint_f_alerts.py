"""Tests Sprint F — Alertes opérationnelles.

F1: Margin proximity alert (>90% marge utilisée)
F2: Funding rate extremes alert (|funding| > 0.1%)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ────────────────── Setup AnomalyType ──────────────────


def test_anomaly_type_margin_proximity_exists():
    """AnomalyType.MARGIN_PROXIMITY existe avec la bonne valeur."""
    from backend.alerts.notifier import AnomalyType

    assert hasattr(AnomalyType, "MARGIN_PROXIMITY")
    assert AnomalyType.MARGIN_PROXIMITY.value == "margin_proximity"


def test_anomaly_type_funding_alert_exists():
    """AnomalyType.FUNDING_ALERT existe avec la bonne valeur."""
    from backend.alerts.notifier import AnomalyType

    assert hasattr(AnomalyType, "FUNDING_ALERT")
    assert AnomalyType.FUNDING_ALERT.value == "funding_alert"


def test_anomaly_messages_margin_proximity():
    """Message formaté pour MARGIN_PROXIMITY."""
    from backend.alerts.notifier import AnomalyType, _ANOMALY_MESSAGES

    assert AnomalyType.MARGIN_PROXIMITY in _ANOMALY_MESSAGES
    assert "90" in _ANOMALY_MESSAGES[AnomalyType.MARGIN_PROXIMITY]


def test_anomaly_cooldown_funding_alert():
    """Cooldown FUNDING_ALERT < 1800s (plus court que les alertes état persistant)."""
    from backend.alerts.notifier import AnomalyType, _ANOMALY_COOLDOWNS

    assert AnomalyType.FUNDING_ALERT in _ANOMALY_COOLDOWNS
    assert _ANOMALY_COOLDOWNS[AnomalyType.FUNDING_ALERT] < 1800


# ────────────────── F1 : Margin Proximity ──────────────────


def _make_executor_for_margin():
    """Crée un executor minimal pour tester _check_margin_proximity."""
    from backend.execution.executor import Executor

    config = MagicMock()
    config.risk.position.default_leverage = 15
    config.assets = []
    notifier = AsyncMock()

    ex = MagicMock()
    ex._config = config
    ex._notifier = notifier
    ex._strategy_name = "grid_atr"
    ex._grid_states = {}
    ex._positions = {}
    ex._pending_notional = 0.0
    return ex, notifier, Executor


@pytest.mark.asyncio
async def test_margin_proximity_alert_triggered():
    """Alerte déclenchée quand marge > 90% du solde."""
    from backend.alerts.notifier import AnomalyType

    ex, notifier, Executor = _make_executor_for_margin()

    # Grid avec 950 USDT de marge sur un solde de 1000 USDT (95% > 90%)
    gs = MagicMock()
    gs.avg_entry_price = 100.0
    gs.total_quantity = 142.5  # notional = 14250, margin = 14250/15 = 950
    gs.leverage = 15
    ex._grid_states = {"BTC/USDT:USDT": gs}

    await Executor._check_margin_proximity(ex, 1000.0)

    notifier.notify_anomaly.assert_awaited_once()
    args = notifier.notify_anomaly.await_args
    assert args[0][0] == AnomalyType.MARGIN_PROXIMITY
    assert "950" in args[0][1] or "95" in args[0][1]


@pytest.mark.asyncio
async def test_margin_proximity_no_alert_below_threshold():
    """Pas d'alerte si marge < 90%."""
    ex, notifier, Executor = _make_executor_for_margin()

    # Marge de 500 USDT sur 1000 USDT (50%) — sous le seuil
    gs = MagicMock()
    gs.avg_entry_price = 100.0
    gs.total_quantity = 75.0  # notional = 7500, margin = 7500/15 = 500
    gs.leverage = 15
    ex._grid_states = {"BTC/USDT:USDT": gs}

    await Executor._check_margin_proximity(ex, 1000.0)

    notifier.notify_anomaly.assert_not_awaited()


@pytest.mark.asyncio
async def test_margin_proximity_zero_balance_skipped():
    """balance <= 0 → pas d'appel (évite la division par zéro)."""
    ex, notifier, Executor = _make_executor_for_margin()
    gs = MagicMock()
    gs.avg_entry_price = 100.0
    gs.total_quantity = 100.0
    gs.leverage = 15
    ex._grid_states = {"BTC/USDT:USDT": gs}

    await Executor._check_margin_proximity(ex, 0.0)

    notifier.notify_anomaly.assert_not_awaited()


@pytest.mark.asyncio
async def test_margin_proximity_includes_pending_notional():
    """_pending_notional est inclus dans le calcul."""
    from backend.alerts.notifier import AnomalyType

    ex, notifier, Executor = _make_executor_for_margin()
    ex._grid_states = {}
    ex._pending_notional = 950.0  # 95% du solde

    await Executor._check_margin_proximity(ex, 1000.0)

    notifier.notify_anomaly.assert_awaited_once()
    args = notifier.notify_anomaly.await_args
    assert args[0][0] == AnomalyType.MARGIN_PROXIMITY


# ────────────────── F2 : Funding Alert ──────────────────


def _make_executor_for_funding():
    """Crée un executor minimal pour tester _check_funding_rates."""
    config = MagicMock()
    notifier = AsyncMock()

    ex = MagicMock()
    ex._config = config
    ex._notifier = notifier
    ex._log_prefix = "Executor"
    ex._strategy_name = "grid_atr"
    ex._data_engine = MagicMock()
    # Rendre _check_funding_rates invocable directement
    from backend.execution.executor import Executor
    ex._check_funding_rates = Executor._check_funding_rates.__get__(ex)
    return ex, notifier


@pytest.mark.asyncio
async def test_funding_alert_triggered_positive():
    """Alerte déclenchée pour funding > 0.1%."""
    from backend.alerts.notifier import AnomalyType
    from backend.execution.executor import Executor

    config = MagicMock()
    asset = MagicMock()
    asset.symbol = "BTC/USDT"
    config.assets = [asset]

    notifier = AsyncMock()
    ex = MagicMock()
    ex._config = config
    ex._notifier = notifier
    ex._log_prefix = "Executor"
    ex._strategy_name = "grid_atr"
    ex._data_engine = MagicMock()
    ex._data_engine.get_funding_rate = MagicMock(return_value=0.15)

    await Executor._check_funding_rates(ex)

    notifier.notify_anomaly.assert_awaited_once()
    args = notifier.notify_anomaly.await_args
    assert args[0][0] == AnomalyType.FUNDING_ALERT
    assert "BTC/USDT" in args[0][1]


@pytest.mark.asyncio
async def test_funding_alert_triggered_negative():
    """Alerte déclenchée pour funding < -0.1%."""
    from backend.alerts.notifier import AnomalyType
    from backend.execution.executor import Executor

    config = MagicMock()
    asset = MagicMock()
    asset.symbol = "ETH/USDT"
    config.assets = [asset]

    notifier = AsyncMock()
    ex = MagicMock()
    ex._config = config
    ex._notifier = notifier
    ex._log_prefix = "Executor"
    ex._strategy_name = "grid_atr"
    ex._data_engine = MagicMock()
    ex._data_engine.get_funding_rate = MagicMock(return_value=-0.25)

    await Executor._check_funding_rates(ex)

    notifier.notify_anomaly.assert_awaited_once()
    args = notifier.notify_anomaly.await_args
    assert args[0][0] == AnomalyType.FUNDING_ALERT


@pytest.mark.asyncio
async def test_funding_alert_not_triggered_normal():
    """Pas d'alerte si |funding| <= 0.1%."""
    from backend.execution.executor import Executor

    config = MagicMock()
    asset = MagicMock()
    asset.symbol = "BTC/USDT"
    config.assets = [asset]

    notifier = AsyncMock()
    ex = MagicMock()
    ex._config = config
    ex._notifier = notifier
    ex._log_prefix = "Executor"
    ex._strategy_name = "grid_atr"
    ex._data_engine = MagicMock()
    ex._data_engine.get_funding_rate = MagicMock(return_value=0.05)

    await Executor._check_funding_rates(ex)

    notifier.notify_anomaly.assert_not_awaited()


@pytest.mark.asyncio
async def test_funding_alert_skips_none_rate():
    """Pas d'alerte si get_funding_rate() retourne None."""
    from backend.execution.executor import Executor

    config = MagicMock()
    asset = MagicMock()
    asset.symbol = "BTC/USDT"
    config.assets = [asset]

    notifier = AsyncMock()
    ex = MagicMock()
    ex._config = config
    ex._notifier = notifier
    ex._log_prefix = "Executor"
    ex._strategy_name = "grid_atr"
    ex._data_engine = MagicMock()
    ex._data_engine.get_funding_rate = MagicMock(return_value=None)

    await Executor._check_funding_rates(ex)

    notifier.notify_anomaly.assert_not_awaited()


@pytest.mark.asyncio
async def test_funding_check_skipped_when_no_data_engine():
    """Pas d'erreur si _data_engine est None."""
    from backend.execution.executor import Executor

    ex = MagicMock()
    ex._data_engine = None
    ex._config = MagicMock()
    ex._notifier = AsyncMock()
    ex._log_prefix = "Executor"
    ex._strategy_name = "grid_atr"

    # Ne doit pas lever d'exception
    await Executor._check_funding_rates(ex)
    ex._notifier.notify_anomaly.assert_not_awaited()
