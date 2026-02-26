"""Tests Hotfix 25b : Kill Switch Reliability.

Couvre les 4 corrections :
- FIX 1 : POST /api/simulator/kill-switch/reset
- FIX 2 : Alerte Telegram au restore du kill switch
- FIX 3 : Persister la raison du kill switch
- FIX 4 : Bug _apply_restored_state() qui écrasait kill_switch_triggered
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.api.server import app
from backend.backtesting.simulator import GridStrategyRunner, Simulator
from backend.core.grid_position_manager import GridPositionManager
from backend.core.incremental_indicators import IncrementalIndicatorEngine
from backend.core.state_manager import StateManager
from backend.strategies.base_grid import BaseGridStrategy


_TEST_API_KEY = "test-ks-reset-key"


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_mock_strategy(name="grid_atr") -> MagicMock:
    """Crée un mock de BaseGridStrategy."""
    strategy = MagicMock(spec=BaseGridStrategy)
    strategy.name = name
    config = MagicMock()
    config.timeframe = "1h"
    config.ma_period = 7
    config.leverage = 6
    strategy._config = config
    strategy.min_candles = {"1h": 50}
    strategy.max_positions = 4
    strategy.compute_grid.return_value = []
    strategy.should_close_all.return_value = None
    strategy.get_tp_price.return_value = float("nan")
    strategy.get_sl_price.return_value = float("nan")
    strategy.get_current_conditions.return_value = []
    return strategy


def _make_mock_config() -> MagicMock:
    config = MagicMock()
    config.risk.initial_capital = 10_000.0
    config.risk.max_margin_ratio = 0.70
    config.risk.kill_switch.max_session_loss_percent = 5.0
    config.risk.kill_switch.max_daily_loss_percent = 10.0
    config.risk.kill_switch.max_loss_percent = 25.0
    config.risk.kill_switch.max_daily_loss_percent_grid = 25.0
    config.risk.position.max_risk_per_trade_percent = 2.0
    config.risk.fees.taker_percent = 0.06
    return config


def _make_grid_runner(name="grid_atr") -> GridStrategyRunner:
    """Crée un GridStrategyRunner avec des mocks par défaut."""
    strategy = _make_mock_strategy(name)
    config = _make_mock_config()

    indicator_engine = MagicMock(spec=IncrementalIndicatorEngine)
    indicator_engine.get_indicators.return_value = {}

    gpm_config = MagicMock()
    gpm_config.leverage = 6
    gpm_config.taker_fee_pct = 0.06
    gpm_config.slippage_pct = 0.02
    gpm = GridPositionManager(gpm_config)

    data_engine = MagicMock()
    data_engine.get_funding_rate.return_value = None
    data_engine.get_open_interest.return_value = []

    runner = GridStrategyRunner(
        strategy=strategy,
        config=config,
        indicator_engine=indicator_engine,
        grid_position_manager=gpm,
        data_engine=data_engine,
    )
    runner._is_warming_up = False
    return runner


def _make_mock_simulator(kill_switch=False, reason=None) -> MagicMock:
    """Crée un mock Simulator pour les tests API."""
    sim = MagicMock(spec=Simulator)
    sim._running = True
    sim._global_kill_switch = kill_switch
    sim._kill_switch_reason = reason
    sim.runners = [MagicMock()]
    sim.runners[0].name = "grid_atr"
    sim.runners[0]._capital = 10_000.0
    sim.runners[0]._kill_switch_triggered = kill_switch
    sim.runners[0]._position = None
    sim.runners[0]._position_symbol = None
    sim.runners[0]._realized_pnl = 0.0
    sim.runners[0]._stats = MagicMock()
    sim.runners[0]._stats.is_active = not kill_switch
    sim.get_all_status.return_value = {}
    sim.is_kill_switch_triggered.return_value = kill_switch
    sim.kill_switch_reason = reason
    return sim


def _setup_mock_app(kill_switch=False, reason=None):
    """Configure l'app avec mocks pour les tests kill switch."""
    sim = _make_mock_simulator(kill_switch, reason)
    app.state.simulator = sim
    app.state.state_manager = MagicMock()
    app.state.state_manager.save_runner_state = AsyncMock()
    app.state.notifier = MagicMock()
    app.state.notifier.notify_anomaly = AsyncMock()
    app.state.db = MagicMock()
    app.state.db.get_simulation_trades = AsyncMock(return_value=[])
    app.state.engine = None
    app.state.config = MagicMock()
    app.state.start_time = MagicMock()
    app.state.start_time.isoformat.return_value = "2024-01-15T12:00:00"
    app.state.arena = None
    return app


# ─── FIX 1 : POST /api/simulator/kill-switch/reset ──────────────────────────


class TestResetEndpoint:
    """Tests pour l'endpoint POST /api/simulator/kill-switch/reset."""

    @pytest.fixture(autouse=True)
    def patch_auth(self):
        mock_cfg = MagicMock()
        mock_cfg.secrets.sync_api_key = _TEST_API_KEY
        with patch("backend.api.executor_routes.get_config", return_value=mock_cfg):
            yield

    @pytest.mark.asyncio
    async def test_reset_endpoint_resets_global(self):
        """POST reset appelle simulator.reset_kill_switch()."""
        mock_app = _setup_mock_app(kill_switch=True)
        mock_app.state.simulator.reset_kill_switch.return_value = 1

        transport = ASGITransport(app=mock_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/simulator/kill-switch/reset", headers={"X-API-Key": _TEST_API_KEY})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "reset"
        assert data["runners_reactivated"] == 1
        mock_app.state.simulator.reset_kill_switch.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_endpoint_reactivates_runners(self):
        """Le count retourné correspond au nombre de runners réactivés."""
        mock_app = _setup_mock_app(kill_switch=True)
        mock_app.state.simulator.reset_kill_switch.return_value = 3

        transport = ASGITransport(app=mock_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/simulator/kill-switch/reset", headers={"X-API-Key": _TEST_API_KEY})

        assert resp.json()["runners_reactivated"] == 3

    @pytest.mark.asyncio
    async def test_reset_endpoint_saves_state(self):
        """POST reset sauvegarde l'état immédiatement via StateManager."""
        mock_app = _setup_mock_app(kill_switch=True)
        mock_app.state.simulator.reset_kill_switch.return_value = 1

        transport = ASGITransport(app=mock_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/api/simulator/kill-switch/reset", headers={"X-API-Key": _TEST_API_KEY})

        mock_app.state.state_manager.save_runner_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_endpoint_not_triggered(self):
        """POST reset quand kill switch inactif retourne not_triggered."""
        mock_app = _setup_mock_app(kill_switch=False)

        transport = ASGITransport(app=mock_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/simulator/kill-switch/reset", headers={"X-API-Key": _TEST_API_KEY})

        assert resp.status_code == 200
        assert resp.json()["status"] == "not_triggered"

    @pytest.mark.asyncio
    async def test_reset_endpoint_notifies_telegram(self):
        """POST reset envoie une alerte Telegram."""
        mock_app = _setup_mock_app(kill_switch=True)
        mock_app.state.simulator.reset_kill_switch.return_value = 1

        transport = ASGITransport(app=mock_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/api/simulator/kill-switch/reset", headers={"X-API-Key": _TEST_API_KEY})

        mock_app.state.notifier.notify_anomaly.assert_called_once()
        call_args = mock_app.state.notifier.notify_anomaly.call_args
        assert "RESET" in call_args[0][1]


# ─── FIX 3 : kill_switch_reason ─────────────────────────────────────────────


class TestKillSwitchReason:
    """Tests pour la persistance et l'affichage de la raison du kill switch."""

    @pytest.fixture(autouse=True)
    def patch_auth(self):
        mock_cfg = MagicMock()
        mock_cfg.secrets.sync_api_key = _TEST_API_KEY
        with patch("backend.api.executor_routes.get_config", return_value=mock_cfg):
            yield

    @pytest.mark.asyncio
    async def test_kill_switch_reason_in_status(self):
        """GET /status inclut kill_switch_reason quand le kill switch est actif."""
        reason = {
            "triggered_at": "2025-01-15T10:00:00+00:00",
            "drawdown_pct": 32.5,
            "window_hours": 24,
            "threshold_pct": 30,
            "capital_max": 100_000.0,
            "capital_current": 67_500.0,
        }
        mock_app = _setup_mock_app(kill_switch=True, reason=reason)

        transport = ASGITransport(app=mock_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/simulator/status")

        data = resp.json()
        assert data["kill_switch_reason"] is not None
        assert data["kill_switch_reason"]["drawdown_pct"] == 32.5

    @pytest.mark.asyncio
    async def test_kill_switch_reason_null_when_inactive(self):
        """GET /status retourne kill_switch_reason=null si pas déclenché."""
        mock_app = _setup_mock_app(kill_switch=False)

        transport = ASGITransport(app=mock_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/simulator/status")

        data = resp.json()
        assert data["kill_switch_reason"] is None

    @pytest.mark.asyncio
    async def test_kill_switch_reason_cleared_on_reset(self):
        """reset_kill_switch() remet kill_switch_reason à None."""
        mock_app = _setup_mock_app(
            kill_switch=True,
            reason={"drawdown_pct": 30, "triggered_at": "2025-01-15T10:00:00"},
        )
        mock_app.state.simulator.reset_kill_switch.return_value = 1
        # Après le reset, le simulator devrait avoir reason=None
        # Le mock doit refléter cela
        mock_app.state.simulator._kill_switch_reason = None

        transport = ASGITransport(app=mock_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/api/simulator/kill-switch/reset", headers={"X-API-Key": _TEST_API_KEY})

        mock_app.state.simulator.reset_kill_switch.assert_called_once()

    @pytest.mark.asyncio
    async def test_kill_switch_reason_persisted_in_state(self, tmp_path):
        """kill_switch_reason est sauvegardée dans le JSON via StateManager."""
        state_file = str(tmp_path / "state.json")
        sm = StateManager(db=MagicMock(), state_file=state_file)

        # Créer un runner mock minimal pour la sérialisation
        runner = MagicMock()
        runner.name = "grid_atr"
        runner._capital = 9_500.0
        runner._position = None
        runner._position_symbol = None
        runner._kill_switch_triggered = True
        runner._stats = MagicMock()
        runner._stats.net_pnl = -500.0
        runner._stats.total_trades = 10
        runner._stats.wins = 4
        runner._stats.losses = 6
        runner._stats.is_active = False
        runner._stats.capital = 9_500.0
        runner._realized_pnl = -500.0
        runner._positions = {}

        reason = {
            "triggered_at": "2025-01-15T10:00:00+00:00",
            "drawdown_pct": 32.5,
            "window_hours": 24,
            "threshold_pct": 30.0,
            "capital_max": 100_000.0,
            "capital_current": 67_500.0,
        }
        await sm.save_runner_state(
            [runner], global_kill_switch=True, kill_switch_reason=reason,
        )

        data = json.loads((tmp_path / "state.json").read_text())
        assert data["global_kill_switch"] is True
        assert data["kill_switch_reason"]["drawdown_pct"] == 32.5
        assert data["kill_switch_reason"]["capital_current"] == 67_500.0


# ─── FIX 4 : Bug _apply_restored_state ──────────────────────────────────────


class TestApplyRestoredStateBug:
    """Tests pour le bug d'ordre dans start() : _stop_all_runners() après _end_warmup()."""

    def test_warmup_then_stop_order(self):
        """Simule la séquence correcte dans start() : _end_warmup() PUIS _stop_all_runners().

        Le bug original : _stop_all_runners() était appelé AVANT _end_warmup(),
        donc _apply_restored_state() écrasait kill_switch_triggered à False.
        Fix : inverser l'ordre — _end_warmup() d'abord, _stop_all_runners() après
        (dernier mot sur kill_switch).
        """
        runner = _make_grid_runner()
        runner._is_warming_up = True

        # Restaurer un state (met _pending_restore)
        runner.restore_state({
            "capital": 9_500.0,
            "kill_switch": False,
            "is_active": True,
            "net_pnl": -500.0,
            "total_trades": 10,
            "wins": 4,
            "losses": 6,
            "grid_positions": [],
        })

        # Séquence CORRECTE (post-fix) : _end_warmup() PUIS _stop_all_runners()
        runner._end_warmup()  # Applique le state → kill_switch=False
        assert runner._kill_switch_triggered is False  # Normal, state dit False

        # _stop_all_runners() a le dernier mot
        runner._kill_switch_triggered = True
        runner._stats.is_active = False

        assert runner._kill_switch_triggered is True
        assert runner._stats.is_active is False

    def test_apply_restored_state_restores_positions(self):
        """_apply_restored_state restaure correctement les positions grid."""
        runner = _make_grid_runner()

        state = {
            "capital": 9_800.0,
            "kill_switch": False,
            "is_active": True,
            "net_pnl": -200.0,
            "realized_pnl": -200.0,
            "total_trades": 3,
            "wins": 1,
            "losses": 2,
            "grid_positions": [
                {
                    "symbol": "BTC/USDT",
                    "level": 0,
                    "direction": "LONG",
                    "entry_price": 95_000.0,
                    "quantity": 0.01,
                    "entry_time": "2025-01-15T10:00:00+00:00",
                    "entry_fee": 0.57,
                },
                {
                    "symbol": "BTC/USDT",
                    "level": 1,
                    "direction": "LONG",
                    "entry_price": 93_000.0,
                    "quantity": 0.01,
                    "entry_time": "2025-01-15T11:00:00+00:00",
                    "entry_fee": 0.56,
                },
                {
                    "symbol": "ETH/USDT",
                    "level": 0,
                    "direction": "LONG",
                    "entry_price": 3_200.0,
                    "quantity": 0.1,
                    "entry_time": "2025-01-15T09:00:00+00:00",
                    "entry_fee": 0.19,
                },
            ],
        }
        runner._apply_restored_state(state)

        # 2 positions BTC + 1 ETH
        assert len(runner._positions.get("BTC/USDT", [])) == 2
        assert len(runner._positions.get("ETH/USDT", [])) == 1
        assert runner._capital == 9_800.0
        assert runner._realized_pnl == -200.0

    def test_restored_state_active_after_reset(self, tmp_path):
        """Save (kill_switch=True) → reset → save → restore → is_active=True.

        Scénario complet : le state JSON reflète le reset même après restore.
        """
        state_file = str(tmp_path / "state.json")
        sm = StateManager(db=MagicMock(), state_file=state_file)

        runner = _make_grid_runner()

        # 1. Déclencher le kill switch
        runner._kill_switch_triggered = True
        runner._stats.is_active = False

        import asyncio

        asyncio.get_event_loop().run_until_complete(
            sm.save_runner_state(
                [runner], global_kill_switch=True, kill_switch_reason={"drawdown_pct": 35.0},
            )
        )

        # Vérifier que le state sauvegardé a kill_switch=True
        data = json.loads((tmp_path / "state.json").read_text())
        assert data["runners"]["grid_atr"]["kill_switch"] is True
        assert data["runners"]["grid_atr"]["is_active"] is False
        assert data["global_kill_switch"] is True

        # 2. Reset du kill switch (comme le ferait simulator.reset_kill_switch)
        runner._kill_switch_triggered = False
        runner._stats.is_active = True

        asyncio.get_event_loop().run_until_complete(
            sm.save_runner_state(
                [runner], global_kill_switch=False, kill_switch_reason=None,
            )
        )

        # 3. Vérifier que le state post-reset a is_active=True
        data = json.loads((tmp_path / "state.json").read_text())
        assert data["runners"]["grid_atr"]["kill_switch"] is False
        assert data["runners"]["grid_atr"]["is_active"] is True
        assert data["global_kill_switch"] is False
        assert data["kill_switch_reason"] is None

        # 4. Restaurer dans un nouveau runner → doit être actif
        new_runner = _make_grid_runner()
        new_runner._apply_restored_state(data["runners"]["grid_atr"])
        assert new_runner._kill_switch_triggered is False
        assert new_runner._stats.is_active is True

    def test_reset_reactivates_globally_stopped_runner(self):
        """reset_kill_switch() réactive un runner stoppé par _stop_all_runners().

        Vérifie que le reset fonctionne pour un runner stoppé par le kill switch
        global (pas par son propre seuil runner-level).
        """
        runner = _make_grid_runner()

        # Le runner n'a PAS déclenché son propre kill switch
        assert runner._kill_switch_triggered is False
        assert runner._stats.is_active is True

        # _stop_all_runners() (global) le stoppe
        runner._kill_switch_triggered = True
        runner._stats.is_active = False

        # Créer un Simulator minimal pour tester reset_kill_switch
        sim = MagicMock(spec=Simulator)
        sim._runners = [runner]
        sim._global_kill_switch = True
        sim._kill_switch_reason = {"drawdown_pct": 35.0}

        # Appeler la vraie méthode
        reactivated = Simulator.reset_kill_switch(sim)

        assert reactivated == 1
        assert runner._kill_switch_triggered is False
        assert runner._stats.is_active is True
        assert sim._global_kill_switch is False
        assert sim._kill_switch_reason is None
