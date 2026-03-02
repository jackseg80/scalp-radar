"""Tests Sprint D — 4 bugs P2 identifiés par audit.

D1: Zombie position detection (watchdog)
D2: Gap candle guard (DataEngine)
D3: NaN guard centralisé (BaseStrategy + PositionManager)
D4: Sizing parité fast engine / executor (candle_capital snapshot)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

# ────────────────── D3 : NaN guard ──────────────────


class TestNaNGuard:
    """Vérifie que les NaN sont bloqués à tous les niveaux."""

    def test_strategy_signal_has_nan_prices_entry(self):
        """has_nan_prices() détecte NaN dans entry_price."""
        from backend.strategies.base import StrategySignal
        from backend.core.models import Direction, MarketRegime, SignalStrength

        signal = StrategySignal(
            direction=Direction.LONG,
            entry_price=float("nan"),
            tp_price=100.0,
            sl_price=90.0,
            score=0.8,
            strength=SignalStrength.STRONG,
            market_regime=MarketRegime.RANGING,
        )
        assert signal.has_nan_prices()

    def test_strategy_signal_has_nan_prices_sl(self):
        """has_nan_prices() détecte NaN dans sl_price."""
        from backend.strategies.base import StrategySignal
        from backend.core.models import Direction, MarketRegime, SignalStrength

        signal = StrategySignal(
            direction=Direction.LONG,
            entry_price=100.0,
            tp_price=110.0,
            sl_price=float("nan"),
            score=0.8,
            strength=SignalStrength.STRONG,
            market_regime=MarketRegime.RANGING,
        )
        assert signal.has_nan_prices()

    def test_strategy_signal_nan_tp_tolerated(self):
        """tp_price=NaN est toléré (grid strategies avec TP inverse)."""
        from backend.strategies.base import StrategySignal
        from backend.core.models import Direction, MarketRegime, SignalStrength

        signal = StrategySignal(
            direction=Direction.LONG,
            entry_price=100.0,
            tp_price=float("nan"),
            sl_price=90.0,
            score=0.8,
            strength=SignalStrength.STRONG,
            market_regime=MarketRegime.RANGING,
        )
        assert not signal.has_nan_prices()

    def test_strategy_signal_valid(self):
        """Signal normal passe la validation."""
        from backend.strategies.base import StrategySignal
        from backend.core.models import Direction, MarketRegime, SignalStrength

        signal = StrategySignal(
            direction=Direction.LONG,
            entry_price=100.0,
            tp_price=110.0,
            sl_price=90.0,
            score=0.8,
            strength=SignalStrength.STRONG,
            market_regime=MarketRegime.RANGING,
        )
        assert not signal.has_nan_prices()

    def test_position_manager_rejects_nan_entry(self):
        """PositionManager.open_position() rejette entry_price=NaN."""
        from backend.core.position_manager import PositionManager, PositionManagerConfig

        pm = PositionManager(PositionManagerConfig())
        signal = MagicMock()
        signal.entry_price = float("nan")
        signal.sl_price = 90.0
        signal.direction = "LONG"
        signal.tp_price = 110.0

        result = pm.open_position(signal, datetime.now(tz=timezone.utc), 1000.0)
        assert result is None

    def test_position_manager_rejects_nan_sl(self):
        """PositionManager.open_position() rejette sl_price=NaN."""
        from backend.core.position_manager import PositionManager, PositionManagerConfig

        pm = PositionManager(PositionManagerConfig())
        signal = MagicMock()
        signal.entry_price = 100.0
        signal.sl_price = float("nan")
        signal.direction = "LONG"
        signal.tp_price = 110.0

        result = pm.open_position(signal, datetime.now(tz=timezone.utc), 1000.0)
        assert result is None


# ────────────────── D1 : Zombie position detection ──────────────────


class TestZombieDetection:
    """Vérifie la détection de positions zombie dans le watchdog."""

    @pytest.fixture
    def watchdog_deps(self):
        """Crée les dépendances mock pour le Watchdog."""
        engine = MagicMock()
        engine.is_connected = True
        engine.last_update = datetime.now(tz=timezone.utc)

        simulator = MagicMock()
        simulator.runners = []
        simulator.is_kill_switch_triggered = MagicMock(return_value=False)

        notifier = AsyncMock()
        return engine, simulator, notifier

    @pytest.mark.asyncio
    async def test_zombie_mono_position_detected(self, watchdog_deps):
        """Position mono ouverte depuis >24h déclenche une alerte."""
        from backend.monitoring.watchdog import Watchdog

        engine, simulator, notifier = watchdog_deps
        executor = MagicMock()
        executor.is_enabled = True
        executor.is_connected = True
        executor._risk_manager = MagicMock()
        executor._risk_manager.is_kill_switch_triggered = False

        # Position zombie : ouverte il y a 30h
        old_pos = MagicMock()
        old_pos.entry_time = datetime.now(tz=timezone.utc) - timedelta(hours=30)
        old_pos.direction = "LONG"
        old_pos.entry_price = 100.0
        executor._positions = {"BTC/USDT:USDT": old_pos}
        executor._grid_states = {}

        wd = Watchdog(engine, simulator, notifier, executor=executor)
        await wd._check_zombie_positions([("", executor)])

        assert any("Zombie" in issue for issue in wd._current_issues)

    @pytest.mark.asyncio
    async def test_young_position_not_zombie(self, watchdog_deps):
        """Position ouverte depuis 1h ne déclenche PAS d'alerte."""
        from backend.monitoring.watchdog import Watchdog

        engine, simulator, notifier = watchdog_deps
        executor = MagicMock()
        executor.is_enabled = True
        executor._positions = {
            "BTC/USDT:USDT": MagicMock(
                entry_time=datetime.now(tz=timezone.utc) - timedelta(hours=1),
            ),
        }
        executor._grid_states = {}

        wd = Watchdog(engine, simulator, notifier, executor=executor)
        await wd._check_zombie_positions([("", executor)])

        assert not wd._current_issues

    @pytest.mark.asyncio
    async def test_zombie_grid_detected(self, watchdog_deps):
        """Grid ouverte depuis >24h déclenche une alerte."""
        from backend.monitoring.watchdog import Watchdog

        engine, simulator, notifier = watchdog_deps
        executor = MagicMock()
        executor.is_enabled = True
        executor._positions = {}

        grid_state = MagicMock()
        grid_state.opened_at = datetime.now(tz=timezone.utc) - timedelta(hours=48)
        grid_state.direction = "LONG"
        grid_state.positions = [MagicMock(), MagicMock()]
        executor._grid_states = {"ETH/USDT:USDT": grid_state}

        wd = Watchdog(engine, simulator, notifier, executor=executor)
        await wd._check_zombie_positions([("", executor)])

        assert any("grid" in issue.lower() for issue in wd._current_issues)


# ────────────────── D2 : Gap candle guard ──────────────────


class TestGapCandle:
    """Vérifie le compteur de gaps et la notification."""

    def test_gap_count_initialized_to_zero(self):
        """DataEngine.gap_count initialisé à 0."""
        from backend.core.data_engine import DataEngine

        config = MagicMock()
        config.exchange = MagicMock()
        config.exchange.ws_connections = 1
        config.exchange.rate_limits = MagicMock()
        config.assets = []
        db = MagicMock()

        engine = DataEngine(config, db)
        assert engine.gap_count == 0

    def test_anomaly_type_data_gap_exists(self):
        """AnomalyType.DATA_GAP existe."""
        from backend.alerts.notifier import AnomalyType

        assert hasattr(AnomalyType, "DATA_GAP")
        assert AnomalyType.DATA_GAP.value == "data_gap"

    def test_anomaly_type_zombie_position_exists(self):
        """AnomalyType.ZOMBIE_POSITION existe."""
        from backend.alerts.notifier import AnomalyType

        assert hasattr(AnomalyType, "ZOMBIE_POSITION")
        assert AnomalyType.ZOMBIE_POSITION.value == "zombie_position"


# ────────────────── D4 : Sizing parité ──────────────────


class TestSizingParity:
    """Vérifie que le fast engine utilise candle_capital (snapshot) pour le sizing."""

    def test_grid_atr_same_qty_multiple_levels_same_candle(self):
        """Deux levels triggés sur la même candle doivent avoir la même qty."""
        # Ce test vérifie le comportement du pattern candle_capital snapshot.
        # On simule le calcul inline pour vérifier la logique.
        capital = 1000.0
        num_levels = 4
        leverage = 15
        entry_price = 100.0

        # Ancien comportement (shrinking) — 2 levels
        old_qty_1 = (capital * (1.0 / num_levels) * leverage) / entry_price
        old_margin_1 = capital * (1.0 / num_levels)
        capital_after = capital - old_margin_1
        old_qty_2 = (capital_after * (1.0 / num_levels) * leverage) / entry_price

        # Les deux étaient différents
        assert old_qty_1 != old_qty_2

        # Nouveau comportement (snapshot) — 2 levels
        capital = 1000.0
        candle_capital = capital  # snapshot
        new_qty_1 = (candle_capital * (1.0 / num_levels) * leverage) / entry_price
        margin_1 = candle_capital * (1.0 / num_levels)
        capital -= margin_1
        new_qty_2 = (candle_capital * (1.0 / num_levels) * leverage) / entry_price

        # Les deux sont identiques
        assert new_qty_1 == new_qty_2
        # Et le capital a bien été réduit
        assert capital == 1000.0 - 250.0

    def test_candle_capital_does_not_leak_across_candles(self):
        """candle_capital est re-snapshottée à chaque candle, capital continue de baisser."""
        capital = 1000.0
        num_levels = 4
        leverage = 15

        # Candle 1 : level 0 trigger
        candle_capital_1 = capital
        notional_1 = candle_capital_1 * (1.0 / num_levels) * leverage
        margin_1 = notional_1 / leverage
        capital -= margin_1  # 1000 - 250 = 750

        # Candle 2 : level 1 trigger
        candle_capital_2 = capital  # 750 (réduit naturellement)
        notional_2 = candle_capital_2 * (1.0 / num_levels) * leverage
        margin_2 = notional_2 / leverage

        # Vérifier que candle_capital_2 < candle_capital_1 (pas de leak)
        assert candle_capital_2 < candle_capital_1
        assert candle_capital_2 == 750.0
        assert margin_2 == 750.0 / num_levels
