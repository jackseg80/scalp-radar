"""Tests pour backend/backtesting/simulator.py."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

import pytest

from backend.core.models import Candle, Direction, MarketRegime, TimeFrame
from backend.core.position_manager import PositionManager, PositionManagerConfig, TradeResult
from backend.core.incremental_indicators import IncrementalIndicatorEngine
from backend.strategies.base import OpenPosition, StrategyContext, StrategySignal
from backend.backtesting.simulator import LiveStrategyRunner, RunnerStats, Simulator


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_candle(
    close: float = 100_000.0,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
    volume: float = 100.0,
    ts: datetime | None = None,
) -> Candle:
    o = open_ or close
    h = high or max(close, o) * 1.001
    lo = low or min(close, o) * 0.999
    return Candle(
        timestamp=ts or datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
        open=o,
        high=h,
        low=lo,
        close=close,
        volume=volume,
        symbol="BTC/USDT",
        timeframe=TimeFrame.M5,
    )


def _make_pm_config() -> PositionManagerConfig:
    return PositionManagerConfig(
        leverage=15,
        maker_fee=0.0002,
        taker_fee=0.0006,
        slippage_pct=0.0005,
        high_vol_slippage_mult=2.0,
        max_risk_per_trade=0.02,
    )


def _make_runner(
    strategy=None,
    config=None,
    indicator_engine=None,
    position_manager=None,
    data_engine=None,
) -> LiveStrategyRunner:
    """Crée un runner avec des mocks par défaut."""
    if strategy is None:
        strategy = MagicMock()
        strategy.name = "test_strat"
        strategy.min_candles = {"5m": 50}
        strategy.evaluate.return_value = None
        strategy.check_exit.return_value = None

    if config is None:
        config = MagicMock()
        config.risk.kill_switch.max_session_loss_percent = 5.0
        config.risk.kill_switch.max_daily_loss_percent = 10.0
        config.risk.position.max_risk_per_trade_percent = 2.0

    if indicator_engine is None:
        indicator_engine = MagicMock(spec=IncrementalIndicatorEngine)
        indicator_engine.get_indicators.return_value = {
            "5m": {
                "rsi": 30.0,
                "vwap": 99_500.0,
                "adx": 15.0,
                "di_plus": 10.0,
                "di_minus": 12.0,
                "atr": 500.0,
                "atr_sma": 450.0,
                "close": 100_000.0,
            }
        }

    if position_manager is None:
        position_manager = PositionManager(_make_pm_config())

    if data_engine is None:
        data_engine = MagicMock()
        data_engine.get_funding_rate.return_value = None
        data_engine.get_open_interest.return_value = []

    return LiveStrategyRunner(
        strategy=strategy,
        config=config,
        indicator_engine=indicator_engine,
        position_manager=position_manager,
        data_engine=data_engine,
    )


# ─── Tests LiveStrategyRunner ────────────────────────────────────────────────


class TestRunnerBasics:
    def test_initial_state(self):
        """Le runner démarre avec capital=10k, pas de position, pas de trades."""
        runner = _make_runner()
        status = runner.get_status()
        assert status["capital"] == 10_000.0
        assert status["net_pnl"] == 0.0
        assert status["total_trades"] == 0
        assert status["has_position"] is False
        assert status["kill_switch"] is False
        assert status["is_active"] is True

    def test_name_from_strategy(self):
        runner = _make_runner()
        assert runner.name == "test_strat"


class TestRunnerOnCandle:
    @pytest.mark.asyncio
    async def test_no_indicators_returns_early(self):
        """Si pas d'indicateurs, on n'évalue pas."""
        ie = MagicMock(spec=IncrementalIndicatorEngine)
        ie.get_indicators.return_value = {}

        strategy = MagicMock()
        strategy.name = "test"
        strategy.min_candles = {"5m": 50}

        runner = _make_runner(strategy=strategy, indicator_engine=ie)
        candle = _make_candle()

        await runner.on_candle("BTC/USDT", "5m", candle)
        strategy.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_called_no_signal(self):
        """Si evaluate retourne None, pas de position ouverte."""
        strategy = MagicMock()
        strategy.name = "test"
        strategy.min_candles = {"5m": 50}
        strategy.evaluate.return_value = None

        runner = _make_runner(strategy=strategy)
        candle = _make_candle()

        await runner.on_candle("BTC/USDT", "5m", candle)
        strategy.evaluate.assert_called_once()
        assert runner.get_status()["has_position"] is False

    @pytest.mark.asyncio
    async def test_evaluate_signal_opens_position(self):
        """Si evaluate retourne un signal, une position est ouverte."""
        signal = StrategySignal(
            direction=Direction.LONG,
            entry_price=100_000.0,
            tp_price=100_800.0,
            sl_price=99_700.0,
            score=0.75,
            strength="MODERATE",
            market_regime=MarketRegime.RANGING,
        )

        strategy = MagicMock()
        strategy.name = "test"
        strategy.min_candles = {"5m": 50}
        strategy.evaluate.return_value = signal

        runner = _make_runner(strategy=strategy)
        candle = _make_candle()

        await runner.on_candle("BTC/USDT", "5m", candle)
        assert runner.get_status()["has_position"] is True
        assert runner.get_status()["capital"] < 10_000.0  # Fee déduite

    @pytest.mark.asyncio
    async def test_tp_hit_closes_position(self):
        """Si TP touché, la position est fermée avec gain."""
        signal = StrategySignal(
            direction=Direction.LONG,
            entry_price=100_000.0,
            tp_price=100_800.0,
            sl_price=99_700.0,
            score=0.75,
            strength="MODERATE",
            market_regime=MarketRegime.RANGING,
        )

        strategy = MagicMock()
        strategy.name = "test"
        strategy.min_candles = {"5m": 50}
        strategy.evaluate.return_value = signal
        strategy.check_exit.return_value = None

        runner = _make_runner(strategy=strategy)

        # Ouvrir
        candle1 = _make_candle(close=100_000.0, ts=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc))
        await runner.on_candle("BTC/USDT", "5m", candle1)
        assert runner.get_status()["has_position"] is True

        # Ne plus ouvrir de nouvelles positions
        strategy.evaluate.return_value = None

        # TP touché
        candle2 = _make_candle(
            close=100_900.0,
            high=101_000.0,
            low=100_500.0,
            open_=100_500.0,
            ts=datetime(2024, 1, 15, 12, 5, tzinfo=timezone.utc),
        )
        await runner.on_candle("BTC/USDT", "5m", candle2)
        assert runner.get_status()["has_position"] is False
        assert runner.get_status()["total_trades"] == 1
        assert runner.get_status()["wins"] == 1

    @pytest.mark.asyncio
    async def test_sl_hit_closes_position(self):
        """Si SL touché, la position est fermée avec perte."""
        signal = StrategySignal(
            direction=Direction.LONG,
            entry_price=100_000.0,
            tp_price=100_800.0,
            sl_price=99_700.0,
            score=0.75,
            strength="MODERATE",
            market_regime=MarketRegime.RANGING,
        )

        strategy = MagicMock()
        strategy.name = "test"
        strategy.min_candles = {"5m": 50}
        strategy.evaluate.return_value = signal
        strategy.check_exit.return_value = None

        runner = _make_runner(strategy=strategy)

        # Ouvrir
        candle1 = _make_candle(close=100_000.0, ts=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc))
        await runner.on_candle("BTC/USDT", "5m", candle1)
        assert runner.get_status()["has_position"] is True

        strategy.evaluate.return_value = None

        # SL touché
        candle2 = _make_candle(
            close=99_600.0,
            high=100_100.0,
            low=99_500.0,
            open_=100_000.0,
            ts=datetime(2024, 1, 15, 12, 5, tzinfo=timezone.utc),
        )
        await runner.on_candle("BTC/USDT", "5m", candle2)
        assert runner.get_status()["has_position"] is False
        assert runner.get_status()["total_trades"] == 1
        assert runner.get_status()["losses"] == 1


class TestKillSwitch:
    @pytest.mark.asyncio
    async def test_kill_switch_triggers_on_session_loss(self):
        """Kill switch si perte session >= max_session_loss_percent."""
        config = MagicMock()
        config.risk.kill_switch.max_session_loss_percent = 5.0
        config.risk.kill_switch.max_daily_loss_percent = 10.0
        config.risk.position.max_risk_per_trade_percent = 2.0

        runner = _make_runner(config=config)

        # Simuler un gros trade perdant (> 5% du capital initial)
        trade = TradeResult(
            direction=Direction.LONG,
            entry_price=100_000.0,
            exit_price=95_000.0,
            quantity=0.01,
            entry_time=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
            exit_time=datetime(2024, 1, 15, 12, 5, tzinfo=timezone.utc),
            gross_pnl=-500.0,
            fee_cost=5.0,
            slippage_cost=2.0,
            net_pnl=-507.0,  # 5.07% de 10k
            exit_reason="sl",
            market_regime=MarketRegime.RANGING,
        )
        runner._record_trade(trade)

        assert runner.is_kill_switch_triggered is True
        assert runner.get_status()["is_active"] is False

    @pytest.mark.asyncio
    async def test_kill_switch_prevents_new_positions(self):
        """Après kill switch, on_candle ne fait rien."""
        runner = _make_runner()
        runner._kill_switch_triggered = True

        strategy = runner._strategy
        candle = _make_candle()
        await runner.on_candle("BTC/USDT", "5m", candle)
        strategy.evaluate.assert_not_called()


class TestRegimeChangeExit:
    @pytest.mark.asyncio
    async def test_regime_change_closes_position(self):
        """Position coupée quand le régime passe RANGING → TRENDING."""
        signal = StrategySignal(
            direction=Direction.LONG,
            entry_price=100_000.0,
            tp_price=100_800.0,
            sl_price=99_700.0,
            score=0.75,
            strength="MODERATE",
            market_regime=MarketRegime.RANGING,
        )

        strategy = MagicMock()
        strategy.name = "test"
        strategy.min_candles = {"5m": 50}
        strategy.evaluate.return_value = signal
        strategy.check_exit.return_value = None

        ie = MagicMock(spec=IncrementalIndicatorEngine)
        # Première candle : régime RANGING (adx=15)
        ie.get_indicators.return_value = {
            "5m": {
                "rsi": 30.0, "vwap": 99_500.0,
                "adx": 15.0, "di_plus": 10.0, "di_minus": 12.0,
                "atr": 500.0, "atr_sma": 450.0, "close": 100_000.0,
            }
        }

        runner = _make_runner(strategy=strategy, indicator_engine=ie)

        # Ouvrir position
        candle1 = _make_candle(close=100_000.0, ts=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc))
        await runner.on_candle("BTC/USDT", "5m", candle1)
        assert runner.get_status()["has_position"] is True

        strategy.evaluate.return_value = None

        # Deuxième candle : régime change vers TRENDING_UP (adx=35, di_plus>di_minus)
        ie.get_indicators.return_value = {
            "5m": {
                "rsi": 55.0, "vwap": 100_000.0,
                "adx": 35.0, "di_plus": 30.0, "di_minus": 10.0,
                "atr": 600.0, "atr_sma": 450.0, "close": 100_200.0,
            }
        }

        candle2 = _make_candle(
            close=100_200.0,
            ts=datetime(2024, 1, 15, 12, 5, tzinfo=timezone.utc),
        )
        await runner.on_candle("BTC/USDT", "5m", candle2)
        assert runner.get_status()["has_position"] is False
        assert runner.get_status()["total_trades"] == 1


# ─── Tests Simulator ─────────────────────────────────────────────────────────


class TestSimulator:
    @pytest.mark.asyncio
    async def test_dispatch_candle_updates_indicators(self):
        """_dispatch_candle met à jour les indicateurs puis dispatch aux runners."""
        de = MagicMock()
        de.on_candle = MagicMock()
        de.get_funding_rate.return_value = None
        de.get_open_interest.return_value = []

        config = MagicMock()
        sim = Simulator(data_engine=de, config=config)

        ie = MagicMock(spec=IncrementalIndicatorEngine)
        ie.get_indicators.return_value = {}
        sim._indicator_engine = ie
        sim._running = True

        runner = MagicMock()
        runner.name = "mock"
        runner.on_candle = AsyncMock()
        sim._runners = [runner]

        candle = _make_candle()
        await sim._dispatch_candle("BTC/USDT", "5m", candle)

        ie.update.assert_called_once_with("BTC/USDT", "5m", candle)
        runner.on_candle.assert_called_once_with("BTC/USDT", "5m", candle)

    @pytest.mark.asyncio
    async def test_dispatch_not_running(self):
        """Si pas running, _dispatch_candle ne fait rien."""
        de = MagicMock()
        config = MagicMock()
        sim = Simulator(data_engine=de, config=config)
        sim._running = False

        ie = MagicMock(spec=IncrementalIndicatorEngine)
        sim._indicator_engine = ie

        candle = _make_candle()
        await sim._dispatch_candle("BTC/USDT", "5m", candle)
        ie.update.assert_not_called()

    def test_get_all_status_empty(self):
        """Status vide si pas de runners."""
        de = MagicMock()
        config = MagicMock()
        sim = Simulator(data_engine=de, config=config)
        assert sim.get_all_status() == {}


# ─── Tests Orphan Cleanup ──────────────────────────────────────────────────


class TestOrphanCleanup:
    @pytest.mark.asyncio
    async def test_orphan_cleanup_on_disable(self):
        """Runner désactivé avec position → orphan_closures enregistré."""
        from unittest.mock import patch

        config = MagicMock()
        config.risk.position.default_leverage = 15
        config.risk.fees.maker_percent = 0.02
        config.risk.fees.taker_percent = 0.06
        config.risk.slippage.default_estimate_percent = 0.05
        config.risk.slippage.high_volatility_multiplier = 2.0
        config.risk.position.max_risk_per_trade_percent = 2.0
        config.secrets.live_trading = False

        engine = MagicMock()
        engine.on_candle = MagicMock()
        engine.get_all_symbols.return_value = ["BTC/USDT"]

        # Seul strat_a est enabled
        strategy_a = MagicMock()
        strategy_a.name = "strat_a"
        strategy_a.min_candles = {"5m": 50}

        saved_state = {
            "runners": {
                "strat_a": {
                    "capital": 10000.0,
                    "net_pnl": 0.0,
                    "total_trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "kill_switch": False,
                    "is_active": True,
                    "position": None,
                    "position_symbol": None,
                },
                "strat_b": {
                    "capital": 9800.0,
                    "net_pnl": -200.0,
                    "total_trades": 1,
                    "wins": 0,
                    "losses": 1,
                    "kill_switch": False,
                    "is_active": True,
                    "position": {
                        "direction": "long",
                        "entry_price": 100000.0,
                        "quantity": 0.01,
                        "entry_time": "2024-06-15T10:00:00+00:00",
                        "tp_price": 101000.0,
                        "sl_price": 99000.0,
                        "entry_fee": 0.60,
                    },
                    "position_symbol": "BTC/USDT",
                },
            },
        }

        with patch(
            "backend.backtesting.simulator.get_enabled_strategies",
            return_value=[strategy_a],
        ), patch(
            "backend.backtesting.simulator.is_grid_strategy",
            return_value=False,
        ):
            sim = Simulator(data_engine=engine, config=config)
            await sim.start(saved_state=saved_state)

        # Seul strat_a a un runner
        assert len(sim._runners) == 1
        assert sim._runners[0].name == "strat_a"

        # Orphan closure enregistré pour strat_b
        assert len(sim.orphan_closures) == 1
        closure = sim.orphan_closures[0]
        assert closure.strategy_name == "strat_b"
        assert closure.symbol == "BTC/USDT"
        assert closure.direction == "long"
        assert closure.entry_price == 100000.0
        assert closure.estimated_fee_cost > 0
        assert closure.reason == "strategy_disabled"

    @pytest.mark.asyncio
    async def test_orphan_cleanup_no_positions(self):
        """Runner désactivé sans position → pas d'orphan closure."""
        from unittest.mock import patch

        config = MagicMock()
        config.risk.position.default_leverage = 15
        config.risk.fees.maker_percent = 0.02
        config.risk.fees.taker_percent = 0.06
        config.risk.slippage.default_estimate_percent = 0.05
        config.risk.slippage.high_volatility_multiplier = 2.0
        config.risk.position.max_risk_per_trade_percent = 2.0
        config.secrets.live_trading = False

        engine = MagicMock()
        engine.on_candle = MagicMock()
        engine.get_all_symbols.return_value = ["BTC/USDT"]

        strategy_a = MagicMock()
        strategy_a.name = "strat_a"
        strategy_a.min_candles = {"5m": 50}

        saved_state = {
            "runners": {
                "strat_a": {
                    "capital": 10000.0, "net_pnl": 0.0,
                    "total_trades": 0, "wins": 0, "losses": 0,
                    "kill_switch": False, "is_active": True,
                    "position": None, "position_symbol": None,
                },
                "strat_b": {
                    "capital": 10000.0, "net_pnl": 0.0,
                    "total_trades": 0, "wins": 0, "losses": 0,
                    "kill_switch": False, "is_active": True,
                    "position": None, "position_symbol": None,
                },
            },
        }

        with patch(
            "backend.backtesting.simulator.get_enabled_strategies",
            return_value=[strategy_a],
        ), patch(
            "backend.backtesting.simulator.is_grid_strategy",
            return_value=False,
        ):
            sim = Simulator(data_engine=engine, config=config)
            await sim.start(saved_state=saved_state)

        assert len(sim._runners) == 1
        assert len(sim.orphan_closures) == 0


# ─── Tests Collision Warning ───────────────────────────────────────────────


class TestCollisionWarning:
    @pytest.mark.asyncio
    async def test_collision_warning_same_symbol(self):
        """2 runners ouvrent sur le même symbol → warning enregistré."""
        de = MagicMock()
        de.on_candle = MagicMock()
        de.get_funding_rate.return_value = None
        de.get_open_interest.return_value = []

        config = MagicMock()
        sim = Simulator(data_engine=de, config=config)

        ie = MagicMock(spec=IncrementalIndicatorEngine)
        ie.get_indicators.return_value = {
            "5m": {
                "rsi": 30.0, "vwap": 99500.0,
                "adx": 15.0, "di_plus": 10.0, "di_minus": 12.0,
                "atr": 500.0, "atr_sma": 450.0, "close": 100000.0,
            },
        }
        sim._indicator_engine = ie
        sim._running = True

        # Runner A a déjà une position LONG BTC
        runner_a = _make_runner()
        runner_a._strategy.name = "strat_a"
        # name is a read-only property → set via _strategy.name
        runner_a._position = OpenPosition(
            direction=Direction.LONG, entry_price=100000.0,
            quantity=0.01,
            entry_time=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            tp_price=101000.0, sl_price=99000.0, entry_fee=0.6,
        )
        runner_a._position_symbol = "BTC/USDT"
        # Pas de nouveau signal → garde la position
        runner_a._strategy.evaluate.return_value = None
        runner_a._strategy.check_exit.return_value = None

        # Runner B va ouvrir SHORT sur BTC/USDT
        signal_b = StrategySignal(
            direction=Direction.SHORT, entry_price=100000.0,
            tp_price=99200.0, sl_price=100300.0,
            score=0.8, strength="MODERATE",
            market_regime=MarketRegime.RANGING,
        )
        runner_b = _make_runner()
        runner_b._strategy.name = "strat_b"
        # name is a read-only property → set via _strategy.name
        runner_b._strategy.evaluate.return_value = signal_b

        sim._runners = [runner_a, runner_b]

        candle = _make_candle()
        await sim._dispatch_candle("BTC/USDT", "5m", candle)

        # Runner B a ouvert
        assert runner_b._position is not None

        # Collision détectée
        assert len(sim.collision_warnings) == 1
        w = sim.collision_warnings[0]
        assert w["symbol"] == "BTC/USDT"
        assert w["runner_opening"] == "strat_b"
        assert w["runner_existing"] == "strat_a"

    @pytest.mark.asyncio
    async def test_no_collision_different_symbols(self):
        """Runners sur des symbols différents → pas de warning."""
        de = MagicMock()
        de.on_candle = MagicMock()
        de.get_funding_rate.return_value = None
        de.get_open_interest.return_value = []

        config = MagicMock()
        sim = Simulator(data_engine=de, config=config)

        ie = MagicMock(spec=IncrementalIndicatorEngine)
        ie.get_indicators.return_value = {
            "5m": {
                "rsi": 30.0, "vwap": 99500.0,
                "adx": 15.0, "di_plus": 10.0, "di_minus": 12.0,
                "atr": 500.0, "atr_sma": 450.0, "close": 100000.0,
            },
        }
        sim._indicator_engine = ie
        sim._running = True

        # Runner A a une position sur ETH
        runner_a = _make_runner()
        runner_a._strategy.name = "strat_a"
        # name is a read-only property → set via _strategy.name
        runner_a._position = OpenPosition(
            direction=Direction.LONG, entry_price=3000.0,
            quantity=0.1,
            entry_time=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            tp_price=3100.0, sl_price=2900.0, entry_fee=0.18,
        )
        runner_a._position_symbol = "ETH/USDT"
        runner_a._strategy.evaluate.return_value = None
        runner_a._strategy.check_exit.return_value = None

        # Runner B ouvre sur BTC (pas de collision)
        signal_b = StrategySignal(
            direction=Direction.LONG, entry_price=100000.0,
            tp_price=101000.0, sl_price=99500.0,
            score=0.7, strength="MODERATE",
            market_regime=MarketRegime.RANGING,
        )
        runner_b = _make_runner()
        runner_b._strategy.name = "strat_b"
        # name is a read-only property → set via _strategy.name
        runner_b._strategy.evaluate.return_value = signal_b

        sim._runners = [runner_a, runner_b]

        candle = _make_candle()
        await sim._dispatch_candle("BTC/USDT", "5m", candle)

        # Runner B a ouvert sur BTC
        assert runner_b._position is not None

        # Pas de collision (symbols différents)
        assert len(sim.collision_warnings) == 0

    def test_is_kill_switch_triggered_false(self):
        """Pas de kill switch par défaut."""
        de = MagicMock()
        config = MagicMock()
        sim = Simulator(data_engine=de, config=config)
        assert sim.is_kill_switch_triggered() is False
