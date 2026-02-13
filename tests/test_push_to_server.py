"""Tests pour push_to_server — sync local → serveur."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from backend.optimization.optimization_db import push_to_server
from backend.optimization.report import FinalReport, ValidationResult


def _make_report() -> FinalReport:
    """Crée un FinalReport de test."""
    validation = ValidationResult(
        bitget_sharpe=1.5, bitget_net_return_pct=8.0, bitget_trades=25,
        bitget_sharpe_ci_low=0.8, bitget_sharpe_ci_high=2.1,
        binance_oos_avg_sharpe=1.3, transfer_ratio=0.85,
        transfer_significant=True, volume_warning=False, volume_warning_detail="",
    )
    return FinalReport(
        strategy_name="vwap_rsi", symbol="BTC/USDT", timestamp=datetime(2026, 2, 13, 12, 0),
        grade="A", total_score=87, wfo_avg_is_sharpe=2.0, wfo_avg_oos_sharpe=1.7,
        wfo_consistency_rate=0.80, wfo_n_windows=20, recommended_params={"rsi_period": 14},
        mc_p_value=0.02, mc_significant=True, mc_underpowered=False, dsr=0.95,
        dsr_max_expected_sharpe=3.2, stability=0.88, cliff_params=[], convergence=0.75,
        divergent_params=[], validation=validation, oos_is_ratio=0.85, bitget_transfer=0.85,
        live_eligible=True, warnings=[], n_distinct_combos=600,
    )


def _make_config(sync_enabled=True, sync_server_url="http://192.168.1.200:8000", sync_api_key="secret"):
    """Crée un mock config pour les tests."""
    config = MagicMock()
    config.secrets.sync_enabled = sync_enabled
    config.secrets.sync_server_url = sync_server_url
    config.secrets.sync_api_key = sync_api_key
    return config


def test_push_disabled(caplog):
    """sync_enabled=False → pas d'appel HTTP."""
    with patch("backend.core.config.get_config", return_value=_make_config(sync_enabled=False)):
        with patch("httpx.Client") as mock_client:
            push_to_server(_make_report(), None, 60.0, "5m")
            mock_client.assert_not_called()


def test_push_success():
    """Push réussi → log info, pas d'exception."""
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"status": "created"}

    mock_client_instance = MagicMock()
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)
    mock_client_instance.post.return_value = mock_response

    with patch("backend.core.config.get_config", return_value=_make_config()):
        with patch("httpx.Client", return_value=mock_client_instance):
            push_to_server(_make_report(), None, 60.0, "5m")

    mock_client_instance.post.assert_called_once()
    call_kwargs = mock_client_instance.post.call_args
    assert "X-API-Key" in call_kwargs.kwargs["headers"]
    assert call_kwargs.kwargs["headers"]["X-API-Key"] == "secret"


def test_push_server_down():
    """Serveur injoignable → log warning, pas d'exception levée."""
    import httpx

    mock_client_instance = MagicMock()
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)
    mock_client_instance.post.side_effect = httpx.ConnectError("Connection refused")

    with patch("backend.core.config.get_config", return_value=_make_config()):
        with patch("httpx.Client", return_value=mock_client_instance):
            # Ne doit PAS lever d'exception
            push_to_server(_make_report(), None, 60.0, "5m")


def test_push_timeout():
    """Timeout → log warning, pas d'exception levée."""
    import httpx

    mock_client_instance = MagicMock()
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)
    mock_client_instance.post.side_effect = httpx.TimeoutException("Timeout")

    with patch("backend.core.config.get_config", return_value=_make_config()):
        with patch("httpx.Client", return_value=mock_client_instance):
            push_to_server(_make_report(), None, 60.0, "5m")


def test_push_empty_server_url():
    """sync_server_url vide → pas d'appel HTTP."""
    with patch("backend.core.config.get_config", return_value=_make_config(sync_server_url="")):
        with patch("httpx.Client") as mock_client:
            push_to_server(_make_report(), None, 60.0, "5m")
            mock_client.assert_not_called()
