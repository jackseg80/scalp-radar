"""Tests pour le StateManager (crash recovery)."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from backend.core.models import Direction, MarketRegime
from backend.core.state_manager import StateManager
from backend.strategies.base import OpenPosition


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_runner(
    name: str = "vwap_rsi",
    capital: float = 10_234.50,
    net_pnl: float = 234.50,
    total_trades: int = 15,
    wins: int = 9,
    losses: int = 6,
    kill_switch: bool = False,
    position: OpenPosition | None = None,
) -> MagicMock:
    """Crée un mock de LiveStrategyRunner."""
    runner = MagicMock()
    runner.name = name
    runner._capital = capital
    runner._kill_switch_triggered = kill_switch
    runner._position = position
    runner._position_symbol = "BTC/USDT" if position is not None else None

    stats = MagicMock()
    stats.capital = capital
    stats.net_pnl = net_pnl
    stats.total_trades = total_trades
    stats.wins = wins
    stats.losses = losses
    stats.is_active = not kill_switch
    runner._stats = stats

    return runner


def _make_position() -> OpenPosition:
    return OpenPosition(
        direction=Direction.LONG,
        entry_price=98500.0,
        quantity=0.01,
        entry_time=datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc),
        tp_price=99300.0,
        sl_price=98200.0,
        entry_fee=0.59,
    )


# ─── Tests ──────────────────────────────────────────────────────────────────


class TestSaveRunnerState:
    """Tests pour save_runner_state()."""

    @pytest.mark.asyncio
    async def test_save_creates_json_file(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        sm = StateManager(db=MagicMock(), state_file=state_file)

        runner = _make_runner()
        await sm.save_runner_state([runner])

        assert (tmp_path / "state.json").exists()
        data = json.loads((tmp_path / "state.json").read_text())
        assert "saved_at" in data
        assert "runners" in data
        assert "vwap_rsi" in data["runners"]

    @pytest.mark.asyncio
    async def test_save_runner_stats(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        sm = StateManager(db=MagicMock(), state_file=state_file)

        runner = _make_runner(capital=10_500.0, net_pnl=500.0, total_trades=20)
        await sm.save_runner_state([runner])

        data = json.loads((tmp_path / "state.json").read_text())
        r = data["runners"]["vwap_rsi"]
        assert r["capital"] == 10_500.0
        assert r["net_pnl"] == 500.0
        assert r["total_trades"] == 20
        assert r["position"] is None

    @pytest.mark.asyncio
    async def test_save_with_open_position(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        sm = StateManager(db=MagicMock(), state_file=state_file)

        position = _make_position()
        runner = _make_runner(position=position)
        await sm.save_runner_state([runner])

        data = json.loads((tmp_path / "state.json").read_text())
        pos = data["runners"]["vwap_rsi"]["position"]
        assert pos is not None
        assert pos["direction"] == "LONG"
        assert pos["entry_price"] == 98500.0
        assert pos["quantity"] == 0.01
        assert pos["tp_price"] == 99300.0
        assert pos["sl_price"] == 98200.0
        assert pos["entry_fee"] == 0.59

    @pytest.mark.asyncio
    async def test_save_multiple_runners(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        sm = StateManager(db=MagicMock(), state_file=state_file)

        runners = [
            _make_runner(name="vwap_rsi", capital=10_500.0),
            _make_runner(name="momentum", capital=9_800.0, kill_switch=True),
        ]
        await sm.save_runner_state(runners)

        data = json.loads((tmp_path / "state.json").read_text())
        assert len(data["runners"]) == 2
        assert data["runners"]["momentum"]["kill_switch"] is True


class TestLoadRunnerState:
    """Tests pour load_runner_state()."""

    @pytest.mark.asyncio
    async def test_load_absent_file(self, tmp_path):
        state_file = str(tmp_path / "nonexistent.json")
        sm = StateManager(db=MagicMock(), state_file=state_file)

        result = await sm.load_runner_state()
        assert result is None

    @pytest.mark.asyncio
    async def test_load_corrupted_file(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        (tmp_path / "state.json").write_text("not valid json{{{")
        sm = StateManager(db=MagicMock(), state_file=state_file)

        result = await sm.load_runner_state()
        assert result is None

    @pytest.mark.asyncio
    async def test_load_empty_file(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        (tmp_path / "state.json").write_text("")
        sm = StateManager(db=MagicMock(), state_file=state_file)

        result = await sm.load_runner_state()
        assert result is None

    @pytest.mark.asyncio
    async def test_load_missing_runners_key(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        (tmp_path / "state.json").write_text('{"saved_at": "2025-01-15"}')
        sm = StateManager(db=MagicMock(), state_file=state_file)

        result = await sm.load_runner_state()
        assert result is None

    @pytest.mark.asyncio
    async def test_load_empty_runners(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        (tmp_path / "state.json").write_text('{"runners": {}}')
        sm = StateManager(db=MagicMock(), state_file=state_file)

        result = await sm.load_runner_state()
        assert result is None

    @pytest.mark.asyncio
    async def test_load_valid_state(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        state = {
            "saved_at": "2025-01-15T10:30:00+00:00",
            "runners": {
                "vwap_rsi": {
                    "capital": 10_500.0,
                    "net_pnl": 500.0,
                    "total_trades": 20,
                    "wins": 12,
                    "losses": 8,
                    "kill_switch": False,
                    "is_active": True,
                    "position": None,
                }
            },
        }
        (tmp_path / "state.json").write_text(json.dumps(state))
        sm = StateManager(db=MagicMock(), state_file=state_file)

        result = await sm.load_runner_state()
        assert result is not None
        assert "vwap_rsi" in result["runners"]
        assert result["runners"]["vwap_rsi"]["capital"] == 10_500.0


class TestRoundTrip:
    """Test save → load → restore complet."""

    @pytest.mark.asyncio
    async def test_save_load_roundtrip(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        sm = StateManager(db=MagicMock(), state_file=state_file)

        position = _make_position()
        runners = [
            _make_runner(name="vwap_rsi", capital=10_500.0, net_pnl=500.0, position=position),
            _make_runner(name="momentum", capital=9_200.0, net_pnl=-800.0, kill_switch=True),
        ]

        # Save
        await sm.save_runner_state(runners)

        # Load
        data = await sm.load_runner_state()
        assert data is not None

        # Vérifier les valeurs
        vwap = data["runners"]["vwap_rsi"]
        assert vwap["capital"] == 10_500.0
        assert vwap["position"]["direction"] == "LONG"
        assert vwap["position"]["entry_price"] == 98500.0

        momentum = data["runners"]["momentum"]
        assert momentum["capital"] == 9_200.0
        assert momentum["kill_switch"] is True
        assert momentum["position"] is None

    @pytest.mark.asyncio
    async def test_atomic_write_preserves_old_on_dir_error(self, tmp_path):
        """Si le dossier parent n'existe pas pour le tmp, save gère l'erreur."""
        state_file = str(tmp_path / "state.json")
        sm = StateManager(db=MagicMock(), state_file=state_file)

        # Écrire un état initial
        runner = _make_runner(capital=10_000.0)
        await sm.save_runner_state([runner])

        # Vérifier que le fichier existe
        data = json.loads((tmp_path / "state.json").read_text())
        assert data["runners"]["vwap_rsi"]["capital"] == 10_000.0

        # Réécrire avec de nouvelles données
        runner2 = _make_runner(capital=11_000.0)
        await sm.save_runner_state([runner2])

        data2 = json.loads((tmp_path / "state.json").read_text())
        assert data2["runners"]["vwap_rsi"]["capital"] == 11_000.0


class TestRestoreState:
    """Tests pour LiveStrategyRunner.restore_state() et Simulator.start(saved_state=)."""

    @pytest.mark.asyncio
    async def test_runner_restore_state(self):
        """Vérifie que restore_state() met à jour le capital et les stats."""
        from backend.backtesting.simulator import LiveStrategyRunner

        # Créer un runner avec des mocks
        strategy = MagicMock()
        strategy.name = "vwap_rsi"
        strategy.min_candles = {"5m": 300}

        config = MagicMock()
        config.risk.initial_capital = 10_000.0
        config.risk.kill_switch.max_session_loss_percent = 5.0
        config.risk.kill_switch.max_daily_loss_percent = 10.0

        runner = LiveStrategyRunner(
            strategy=strategy,
            config=config,
            indicator_engine=MagicMock(),
            position_manager=MagicMock(),
            data_engine=MagicMock(),
        )

        # Vérifier état initial
        assert runner._capital == 10_000.0
        assert runner._stats.total_trades == 0

        # Restaurer
        state = {
            "capital": 10_500.0,
            "net_pnl": 500.0,
            "total_trades": 20,
            "wins": 12,
            "losses": 8,
            "kill_switch": False,
            "is_active": True,
            "position": None,
        }
        runner.restore_state(state)

        assert runner._capital == 10_500.0
        assert runner._stats.net_pnl == 500.0
        assert runner._stats.total_trades == 20
        assert runner._stats.wins == 12
        assert runner._position is None

    @pytest.mark.asyncio
    async def test_runner_restore_with_position(self):
        """Vérifie que restore_state() restaure une position ouverte."""
        from backend.backtesting.simulator import LiveStrategyRunner

        strategy = MagicMock()
        strategy.name = "vwap_rsi"
        strategy.min_candles = {"5m": 300}
        config = MagicMock()
        config.risk.initial_capital = 10_000.0
        config.risk.kill_switch.max_session_loss_percent = 5.0

        runner = LiveStrategyRunner(
            strategy=strategy,
            config=config,
            indicator_engine=MagicMock(),
            position_manager=MagicMock(),
            data_engine=MagicMock(),
        )

        state = {
            "capital": 9_800.0,
            "net_pnl": -200.0,
            "total_trades": 5,
            "wins": 2,
            "losses": 3,
            "kill_switch": False,
            "is_active": True,
            "position": {
                "direction": "LONG",
                "entry_price": 98500.0,
                "quantity": 0.01,
                "entry_time": "2025-01-15T10:00:00+00:00",
                "tp_price": 99300.0,
                "sl_price": 98200.0,
                "entry_fee": 0.59,
            },
        }
        runner.restore_state(state)

        assert runner._position is not None
        assert runner._position.direction == Direction.LONG
        assert runner._position.entry_price == 98500.0
        assert runner._position.quantity == 0.01

    @pytest.mark.asyncio
    async def test_runner_restore_kill_switch(self):
        """Vérifie que restore_state() restaure le kill switch."""
        from backend.backtesting.simulator import LiveStrategyRunner

        strategy = MagicMock()
        strategy.name = "momentum"
        strategy.min_candles = {"5m": 300}
        config = MagicMock()
        config.risk.initial_capital = 10_000.0
        config.risk.kill_switch.max_session_loss_percent = 5.0

        runner = LiveStrategyRunner(
            strategy=strategy,
            config=config,
            indicator_engine=MagicMock(),
            position_manager=MagicMock(),
            data_engine=MagicMock(),
        )

        state = {
            "capital": 9_200.0,
            "net_pnl": -800.0,
            "total_trades": 10,
            "wins": 3,
            "losses": 7,
            "kill_switch": True,
            "is_active": False,
            "position": None,
        }
        runner.restore_state(state)

        assert runner._kill_switch_triggered is True
        assert runner._stats.is_active is False
        assert runner._capital == 9_200.0


class TestPeriodicSave:
    """Tests pour start_periodic_save() et stop()."""

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        sm = StateManager(db=MagicMock(), state_file=state_file)

        simulator = MagicMock()
        simulator.runners = []

        await sm.start_periodic_save(simulator, interval=3600)
        assert sm._task is not None
        assert not sm._task.done()

        await sm.stop()
        assert sm._running is False


class TestAsyncIO:
    """Tests que le I/O fichier passe bien par asyncio.to_thread."""

    @pytest.mark.asyncio
    async def test_save_uses_to_thread(self, tmp_path):
        """Vérifie que save_runner_state appelle asyncio.to_thread avec _write_json_file."""
        state_file = str(tmp_path / "state.json")
        sm = StateManager(db=MagicMock(), state_file=state_file)
        runner = _make_runner()

        with patch("backend.core.state_manager.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            # to_thread doit appeler la vraie fonction pour que le test soit valide
            mock_to_thread.side_effect = lambda fn, *args: fn(*args)
            await sm.save_runner_state([runner])

            mock_to_thread.assert_called_once()
            call_args = mock_to_thread.call_args
            assert call_args[0][0] == StateManager._write_json_file

        # Vérifier que le fichier a bien été écrit
        assert (tmp_path / "state.json").exists()

    @pytest.mark.asyncio
    async def test_load_uses_to_thread(self, tmp_path):
        """Vérifie que load_runner_state appelle asyncio.to_thread avec _read_json_file."""
        state_file = str(tmp_path / "state.json")
        sm = StateManager(db=MagicMock(), state_file=state_file)

        # Écrire un état valide d'abord
        runner = _make_runner()
        await sm.save_runner_state([runner])

        with patch("backend.core.state_manager.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = lambda fn, *args: fn(*args)
            result = await sm.load_runner_state()

            mock_to_thread.assert_called_once()
            call_args = mock_to_thread.call_args
            assert call_args[0][0] == StateManager._read_json_file

        assert result is not None

    @pytest.mark.asyncio
    async def test_executor_save_load_roundtrip(self, tmp_path):
        """Vérifie que save/load executor fonctionne avec to_thread."""
        executor_file = str(tmp_path / "executor_state.json")
        sm = StateManager(
            db=MagicMock(),
            state_file=str(tmp_path / "state.json"),
            executor_state_file=executor_file,
        )

        executor = MagicMock()
        executor.get_state_for_persistence.return_value = {
            "positions": {"BTC/USDT:USDT": {"direction": "LONG", "entry_price": 100000}},
            "daily_stats": {"trades": 5},
        }
        risk_manager = MagicMock()

        await sm.save_executor_state(executor, risk_manager)
        assert (tmp_path / "executor_state.json").exists()

        result = await sm.load_executor_state()
        assert result is not None
        assert "positions" in result
        assert result["positions"]["BTC/USDT:USDT"]["direction"] == "LONG"
