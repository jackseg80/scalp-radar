"""Tests Sprint 36b — Multi-Executor (un Executor par stratégie live).

22 tests couvrant :
- Config (clés API per-strategy)
- Executor isolation (strategy_name, set_strategies filter)
- ExecutorManager (agrégation)
- StateManager (fichiers per-executor, migration legacy)
- API routes (agrégation, kill-switch per-strategy)
- Watchdog (multi-executor)
- Intégration (helper _get_live_eligible_strategies)
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.execution.executor_manager import ExecutorManager


# ─── Helpers ────────────────────────────────────────────────────────


def _make_mock_executor(
    strategy_name: str = "grid_atr",
    is_enabled: bool = True,
    is_connected: bool = True,
    exchange_balance: float = 1000.0,
    positions: list | None = None,
    order_history: list | None = None,
) -> MagicMock:
    """Crée un mock Executor avec les bons attributs."""
    ex = MagicMock()
    ex.strategy_name = strategy_name
    ex.is_enabled = is_enabled
    ex.is_connected = is_connected
    ex.exchange_balance = exchange_balance

    ex._risk_manager = MagicMock()
    ex._risk_manager.is_kill_switch_triggered = False
    ex._risk_manager._session_pnl = 0.0
    ex._risk_manager._total_orders = 5
    ex._risk_manager._initial_capital = 1000.0
    ex._risk_manager.open_positions_count = 2

    ex._order_history = order_history or []

    status = {
        "enabled": is_enabled,
        "connected": is_connected,
        "exchange_balance": exchange_balance,
        "strategy_name": strategy_name,
        "positions": positions or [],
        "executor_grid_state": None,
    }
    ex.get_status.return_value = status
    ex.get_state_for_persistence.return_value = {"positions": [], "grid_states": {}}
    ex.stop = AsyncMock()
    ex.refresh_balance = AsyncMock(return_value=exchange_balance)

    return ex


def _make_mock_risk_manager(
    kill_switch: bool = False,
    session_pnl: float = 0.0,
) -> MagicMock:
    rm = MagicMock()
    rm.is_kill_switch_triggered = kill_switch
    rm._session_pnl = session_pnl
    rm._total_orders = 3
    rm._initial_capital = 1000.0
    rm.open_positions_count = 1
    return rm


# ═══════════════════════════════════════════════════════════════════
# 1. Config : clés API per-strategy (3 tests)
# ═══════════════════════════════════════════════════════════════════


class TestConfigExecutorKeys:
    """Tests pour get_executor_keys() et has_dedicated_keys()."""

    def test_get_executor_keys_per_strategy(self, monkeypatch):
        """Env vars BITGET_API_KEY_GRID_ATR retournées."""
        monkeypatch.setenv("BITGET_API_KEY_GRID_ATR", "key_atr")
        monkeypatch.setenv("BITGET_SECRET_GRID_ATR", "secret_atr")
        monkeypatch.setenv("BITGET_PASSPHRASE_GRID_ATR", "pass_atr")

        config = MagicMock()
        config.secrets.bitget_api_key = "global_key"
        config.secrets.bitget_secret = "global_secret"
        config.secrets.bitget_passphrase = "global_pass"

        from backend.core.config import AppConfig

        keys = AppConfig.get_executor_keys(config, "grid_atr")
        assert keys == ("key_atr", "secret_atr", "pass_atr")

    def test_get_executor_keys_fallback_global(self, monkeypatch):
        """Fallback sur clés globales si per-strategy absentes."""
        # S'assurer que les vars per-strategy n'existent pas
        monkeypatch.delenv("BITGET_API_KEY_GRID_MULTI_TF", raising=False)
        monkeypatch.delenv("BITGET_SECRET_GRID_MULTI_TF", raising=False)
        monkeypatch.delenv("BITGET_PASSPHRASE_GRID_MULTI_TF", raising=False)

        config = MagicMock()
        config.secrets.bitget_api_key = "global_key"
        config.secrets.bitget_secret = "global_secret"
        config.secrets.bitget_passphrase = "global_pass"

        from backend.core.config import AppConfig

        keys = AppConfig.get_executor_keys(config, "grid_multi_tf")
        assert keys == ("global_key", "global_secret", "global_pass")

    def test_has_dedicated_keys(self, monkeypatch):
        """True si les 3 clés per-strategy sont présentes."""
        monkeypatch.setenv("BITGET_API_KEY_GRID_ATR", "k")
        monkeypatch.setenv("BITGET_SECRET_GRID_ATR", "s")
        monkeypatch.setenv("BITGET_PASSPHRASE_GRID_ATR", "p")
        monkeypatch.delenv("BITGET_API_KEY_GRID_MULTI_TF", raising=False)

        from backend.core.config import AppConfig

        config = MagicMock()
        assert AppConfig.has_dedicated_keys(config, "grid_atr") is True
        assert AppConfig.has_dedicated_keys(config, "grid_multi_tf") is False


# ═══════════════════════════════════════════════════════════════════
# 2. Executor isolation (4 tests)
# ═══════════════════════════════════════════════════════════════════


class TestExecutorIsolation:
    """Tests pour l'isolation par strategy_name dans Executor."""

    def test_executor_strategy_name_stored(self):
        """strategy_name accessible via property."""
        from backend.execution.executor import Executor

        config = MagicMock()
        config.secrets.live_trading = True
        config.secrets.bitget_sandbox = False
        rm = MagicMock()
        notifier = MagicMock()

        ex = Executor(config, rm, notifier, strategy_name="grid_atr")
        assert ex.strategy_name == "grid_atr"

    def test_executor_strategy_name_none_by_default(self):
        """strategy_name est None par défaut (backward compat)."""
        from backend.execution.executor import Executor

        config = MagicMock()
        config.secrets.live_trading = True
        config.secrets.bitget_sandbox = False
        rm = MagicMock()
        notifier = MagicMock()

        ex = Executor(config, rm, notifier)
        assert ex.strategy_name is None

    def test_executor_set_strategies_filters_to_own(self):
        """set_strategies() ne garde que sa stratégie."""
        from backend.execution.executor import Executor

        config = MagicMock()
        config.secrets.live_trading = True
        config.secrets.bitget_sandbox = False
        rm = MagicMock()
        notifier = MagicMock()

        ex = Executor(config, rm, notifier, strategy_name="grid_atr")

        strats = {"grid_atr": MagicMock(), "grid_boltrend": MagicMock()}
        ex.set_strategies(strats)

        assert list(ex._strategies.keys()) == ["grid_atr"]

    def test_executor_log_prefix(self):
        """Log prefix contient le nom de stratégie."""
        from backend.execution.executor import Executor

        config = MagicMock()
        config.secrets.live_trading = True
        config.secrets.bitget_sandbox = False
        rm = MagicMock()
        notifier = MagicMock()

        ex = Executor(config, rm, notifier, strategy_name="grid_atr")
        assert "grid_atr" in ex._log_prefix

        ex2 = Executor(config, rm, notifier)
        assert ex2._log_prefix == "Executor"


# ═══════════════════════════════════════════════════════════════════
# 3. ExecutorManager (5 tests)
# ═══════════════════════════════════════════════════════════════════


class TestExecutorManager:
    """Tests pour la couche d'agrégation ExecutorManager."""

    def test_manager_add_get(self):
        """Ajout et récupération par nom."""
        mgr = ExecutorManager()
        ex = _make_mock_executor("grid_atr")
        rm = _make_mock_risk_manager()

        mgr.add("grid_atr", ex, rm)

        assert mgr.get("grid_atr") is ex
        assert mgr.get("grid_boltrend") is None
        assert "grid_atr" in mgr.executors
        assert "grid_atr" in mgr.risk_managers

    def test_manager_get_status_aggregates(self):
        """get_status() merge positions et balances."""
        mgr = ExecutorManager()

        ex1 = _make_mock_executor(
            "grid_atr", exchange_balance=1000.0,
            positions=[{"symbol": "BTC/USDT", "direction": "LONG"}],
        )
        ex2 = _make_mock_executor(
            "grid_boltrend", exchange_balance=500.0,
            positions=[{"symbol": "ETH/USDT", "direction": "LONG"}],
        )

        mgr.add("grid_atr", ex1, _make_mock_risk_manager())
        mgr.add("grid_boltrend", ex2, _make_mock_risk_manager())

        status = mgr.get_status()

        assert status["enabled"] is True
        assert status["exchange_balance"] == 1500.0
        assert len(status["positions"]) == 2
        assert "per_strategy" in status
        assert "grid_atr" in status["per_strategy"]
        assert "grid_boltrend" in status["per_strategy"]

    def test_manager_is_enabled_any(self):
        """is_enabled = True si au moins un executor est enabled."""
        mgr = ExecutorManager()

        ex1 = _make_mock_executor("grid_atr", is_enabled=False)
        ex2 = _make_mock_executor("grid_boltrend", is_enabled=True)

        mgr.add("grid_atr", ex1, _make_mock_risk_manager())
        mgr.add("grid_boltrend", ex2, _make_mock_risk_manager())

        assert mgr.is_enabled is True

        # Tous désactivés
        mgr2 = ExecutorManager()
        mgr2.add("x", _make_mock_executor(is_enabled=False), _make_mock_risk_manager())
        assert mgr2.is_enabled is False

    def test_manager_get_all_order_history_sorted(self):
        """Merge + tri par timestamp décroissant."""
        mgr = ExecutorManager()

        ex1 = _make_mock_executor(order_history=[
            {"timestamp": "2024-01-15T12:00:00Z", "symbol": "BTC"},
            {"timestamp": "2024-01-15T14:00:00Z", "symbol": "BTC"},
        ])
        ex2 = _make_mock_executor(order_history=[
            {"timestamp": "2024-01-15T13:00:00Z", "symbol": "ETH"},
        ])

        mgr.add("grid_atr", ex1, _make_mock_risk_manager())
        mgr.add("grid_boltrend", ex2, _make_mock_risk_manager())

        orders = mgr.get_all_order_history(limit=10)
        assert len(orders) == 3
        # Tri décroissant
        assert orders[0]["timestamp"] == "2024-01-15T14:00:00Z"
        assert orders[1]["timestamp"] == "2024-01-15T13:00:00Z"
        assert orders[2]["timestamp"] == "2024-01-15T12:00:00Z"

    @pytest.mark.asyncio
    async def test_manager_stop_all(self):
        """Tous les stop() sont appelés."""
        mgr = ExecutorManager()

        ex1 = _make_mock_executor("grid_atr")
        ex2 = _make_mock_executor("grid_boltrend")

        mgr.add("grid_atr", ex1, _make_mock_risk_manager())
        mgr.add("grid_boltrend", ex2, _make_mock_risk_manager())

        await mgr.stop_all()

        ex1.stop.assert_awaited_once()
        ex2.stop.assert_awaited_once()


# ═══════════════════════════════════════════════════════════════════
# 4. StateManager (4 tests)
# ═══════════════════════════════════════════════════════════════════


class TestStateManagerMultiExecutor:
    """Tests pour la persistence per-executor."""

    def test_state_file_naming(self):
        """Nommage correct des fichiers per-strategy."""
        from backend.core.state_manager import StateManager

        db = MagicMock()
        sm = StateManager(db)

        assert sm._executor_state_path("grid_atr") == "data/executor_grid_atr_state.json"
        assert sm._executor_state_path("grid_multi_tf") == "data/executor_grid_multi_tf_state.json"
        assert sm._executor_state_path(None) == "data/executor_state.json"

    @pytest.mark.asyncio
    async def test_save_load_per_strategy_state(self, tmp_path):
        """Round-trip : save puis load per-strategy."""
        from backend.core.state_manager import StateManager

        db = MagicMock()
        sm = StateManager(db, executor_state_file=str(tmp_path / "exec.json"))

        # Override le path pour utiliser tmp_path
        original = sm._executor_state_path

        def _tmp_path(name=None):
            if name:
                return str(tmp_path / f"executor_{name}_state.json")
            return str(tmp_path / "exec.json")

        sm._executor_state_path = _tmp_path

        # Sauvegarder
        ex = MagicMock()
        ex.get_state_for_persistence.return_value = {
            "positions": [{"symbol": "BTC/USDT"}],
            "session_pnl": 42.5,
        }
        rm = MagicMock()

        await sm.save_executor_state(ex, rm, strategy_name="grid_atr")

        # Vérifier fichier créé
        assert (tmp_path / "executor_grid_atr_state.json").exists()

        # Charger
        loaded = await sm.load_executor_state(strategy_name="grid_atr")
        assert loaded is not None
        assert loaded["positions"][0]["symbol"] == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_legacy_migration_fallback(self, tmp_path):
        """Fichier legacy lu si per-strategy absent."""
        from backend.core.state_manager import StateManager

        db = MagicMock()
        legacy_path = str(tmp_path / "executor_state.json")
        sm = StateManager(db, executor_state_file=legacy_path)

        # Écrire un fichier legacy
        legacy_data = {
            "saved_at": "2024-01-15T12:00:00Z",
            "executor": {
                "positions": [{"symbol": "BTC/USDT", "legacy": True}],
            },
        }
        Path(legacy_path).write_text(json.dumps(legacy_data), encoding="utf-8")

        # Override path pour que per-strategy ne pointe nulle part
        original = sm._executor_state_path

        def _tmp_path(name=None):
            if name:
                return str(tmp_path / f"executor_{name}_state.json")
            return legacy_path

        sm._executor_state_path = _tmp_path

        # Charger per-strategy (fichier absent) → fallback legacy
        loaded = await sm.load_executor_state(strategy_name="grid_atr")
        assert loaded is not None
        assert loaded["positions"][0]["legacy"] is True

    def test_set_executors_disables_singleton(self):
        """set_executors() désactive le mode singleton legacy."""
        from backend.core.state_manager import StateManager

        db = MagicMock()
        sm = StateManager(db)

        # Mode legacy
        sm._executor = MagicMock()
        sm._risk_manager = MagicMock()

        # Switch to multi
        mgr = MagicMock()
        sm.set_executors(mgr)

        assert sm._executor_mgr is mgr
        assert sm._executor is None
        assert sm._risk_manager is None


# ═══════════════════════════════════════════════════════════════════
# 5. Watchdog (2 tests)
# ═══════════════════════════════════════════════════════════════════


class TestWatchdogMultiExecutor:
    """Tests pour le Watchdog avec multi-executor."""

    @pytest.mark.asyncio
    async def test_watchdog_multi_executor_checks_all(self):
        """Tous les executors sont vérifiés."""
        from backend.monitoring.watchdog import Watchdog

        engine = MagicMock()
        engine.is_connected = True
        engine.last_update = datetime.now(tz=timezone.utc)

        simulator = MagicMock()
        simulator.runners = []
        simulator.is_kill_switch_triggered.return_value = False

        notifier = AsyncMock()

        # Un executor déconnecté
        ex1 = _make_mock_executor("grid_atr", is_enabled=True, is_connected=False)
        ex2 = _make_mock_executor("grid_boltrend", is_enabled=True, is_connected=True)

        mgr = ExecutorManager()
        mgr.add("grid_atr", ex1, _make_mock_risk_manager())
        mgr.add("grid_boltrend", ex2, _make_mock_risk_manager())

        wd = Watchdog(
            data_engine=engine,
            simulator=simulator,
            notifier=notifier,
            executor_mgr=mgr,
        )

        await wd._check()

        # grid_atr déconnecté → issue détectée
        issues = wd._current_issues
        assert any("grid_atr" in i for i in issues)
        # grid_boltrend connecté → pas d'issue
        assert not any("grid_boltrend" in i and "déconnecté" in i for i in issues)

    @pytest.mark.asyncio
    async def test_watchdog_backward_compat_single(self):
        """Ancien param executor= fonctionne encore."""
        from backend.monitoring.watchdog import Watchdog

        engine = MagicMock()
        engine.is_connected = True
        engine.last_update = datetime.now(tz=timezone.utc)

        simulator = MagicMock()
        simulator.runners = []
        simulator.is_kill_switch_triggered.return_value = False

        notifier = AsyncMock()

        ex = MagicMock()
        ex.is_enabled = True
        ex.is_connected = False
        ex._risk_manager = MagicMock()
        ex._risk_manager.is_kill_switch_triggered = False

        wd = Watchdog(
            data_engine=engine,
            simulator=simulator,
            notifier=notifier,
            executor=ex,
        )

        await wd._check()

        assert any("déconnecté" in i for i in wd._current_issues)


# ═══════════════════════════════════════════════════════════════════
# 6. Integration (1 test)
# ═══════════════════════════════════════════════════════════════════


class TestLiveEligibleStrategies:
    """Test du helper _get_live_eligible_strategies."""

    def test_live_eligible_strategies_helper(self):
        """Retourne uniquement les stratégies enabled + live_eligible."""
        from backend.api.server import _get_live_eligible_strategies

        config = MagicMock()

        # Simuler 3 stratégies via model_fields
        grid_atr = MagicMock()
        grid_atr.enabled = True
        grid_atr.live_eligible = True

        grid_boltrend = MagicMock()
        grid_boltrend.enabled = True
        grid_boltrend.live_eligible = False

        grid_multi_tf = MagicMock()
        grid_multi_tf.enabled = False
        grid_multi_tf.live_eligible = False

        config.strategies.model_fields = {
            "grid_atr": None,
            "grid_boltrend": None,
            "grid_multi_tf": None,
            "custom_strategies": None,  # Doit être ignoré
        }

        def _getattr(name, default=None):
            mapping = {
                "grid_atr": grid_atr,
                "grid_boltrend": grid_boltrend,
                "grid_multi_tf": grid_multi_tf,
            }
            return mapping.get(name, default)

        config.strategies.__class__ = type(config.strategies)

        # Patch getattr pour retourner nos mocks
        original_getattr = config.strategies.__getattr__

        def patched_getattr(name, default=None):
            return _getattr(name, default)

        with patch.object(type(config.strategies), "__getattr__", side_effect=_getattr):
            result = _get_live_eligible_strategies(config)

        assert result == ["grid_atr"]
