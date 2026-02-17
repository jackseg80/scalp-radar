"""Tests pour Simulator.get_grid_state() (Sprint 16)."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from backend.core.grid_position_manager import GridPositionManager
from backend.core.incremental_indicators import IncrementalIndicatorEngine
from backend.core.models import Direction
from backend.core.position_manager import PositionManagerConfig
from backend.backtesting.simulator import GridStrategyRunner, Simulator
from backend.strategies.base_grid import GridPosition


def _make_gpm(leverage: int = 6) -> GridPositionManager:
    config = PositionManagerConfig(
        leverage=leverage,
        maker_fee=0.0002,
        taker_fee=0.0006,
        slippage_pct=0.0005,
        high_vol_slippage_mult=2.0,
        max_risk_per_trade=0.02,
    )
    return GridPositionManager(config)


def _make_data_engine(prices: dict[str, float]) -> MagicMock:
    """DataEngine mock qui retourne des prix pour chaque symbol."""
    engine = MagicMock()
    engine.config = MagicMock()
    engine.config.assets = []

    def _get_data(symbol):
        data = MagicMock()
        price = prices.get(symbol, 0)
        if price > 0:
            candle = MagicMock()
            candle.close = price
            data.candles = {"1m": [candle]}
        else:
            data.candles = {"1m": []}
        return data

    engine.get_data.side_effect = _get_data
    engine.get_all_symbols.return_value = list(prices.keys())
    return engine


def _make_grid_runner(
    strategy_name: str,
    config: MagicMock,
    data_engine: MagicMock,
    leverage: int = 6,
    max_positions: int = 4,
    tp_price: float = float("nan"),
    sl_price: float = float("nan"),
) -> GridStrategyRunner:
    """Crée un GridStrategyRunner avec mocks."""
    strategy = MagicMock()
    strategy.name = strategy_name
    strategy._config.timeframe = "1h"
    strategy._config.ma_period = 7
    strategy._config.leverage = leverage
    strategy._config.per_asset = {}
    strategy.max_positions = max_positions
    strategy.min_candles = {"1h": 50}
    strategy.get_tp_price.return_value = tp_price
    strategy.get_sl_price.return_value = sl_price
    strategy.get_current_conditions.return_value = []

    indicator_engine = MagicMock(spec=IncrementalIndicatorEngine)
    indicator_engine.get_indicators.return_value = {"1h": {"sma": 100.0, "close": 100.0}}

    runner = GridStrategyRunner(
        strategy=strategy,
        config=config,
        indicator_engine=indicator_engine,
        grid_position_manager=_make_gpm(leverage),
        data_engine=data_engine,
    )
    runner._is_warming_up = False
    return runner


def _make_position(
    level: int = 0,
    direction: Direction = Direction.LONG,
    entry_price: float = 100.0,
    quantity: float = 1.0,
) -> GridPosition:
    return GridPosition(
        level=level,
        direction=direction,
        entry_price=entry_price,
        quantity=quantity,
        entry_time=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
        entry_fee=entry_price * quantity * 0.0006,
    )


class TestGetGridStateEmpty:
    """Tests avec 0 positions ou 0 runners."""

    def test_no_runners(self):
        """Simulator sans runners → grids vide."""
        engine = _make_data_engine({})
        config = MagicMock()
        config.risk.initial_capital = 10_000
        sim = Simulator(data_engine=engine, config=config)
        result = sim.get_grid_state()
        assert result["grid_positions"] == {}
        assert result["summary"]["total_positions"] == 0
        assert result["summary"]["total_assets"] == 0

    def test_grid_runner_no_positions(self):
        """GridStrategyRunner sans positions → grids vide."""
        engine = _make_data_engine({"BTC/USDT": 96000.0})
        config = MagicMock()
        config.risk.initial_capital = 10_000

        sim = Simulator(data_engine=engine, config=config)
        runner = _make_grid_runner("envelope_dca", config, engine)
        sim._runners.append(runner)

        result = sim.get_grid_state()
        assert result["grid_positions"] == {}
        assert result["summary"]["total_positions"] == 0


class TestGetGridStateWithPositions:
    """Tests avec positions ouvertes."""

    def test_single_asset_two_levels(self):
        """2 positions LONG sur BTC → state correctement calculé."""
        engine = _make_data_engine({"BTC/USDT": 96000.0})
        config = MagicMock()
        config.risk.initial_capital = 10_000

        sim = Simulator(data_engine=engine, config=config)
        runner = _make_grid_runner(
            "envelope_dca", config, engine,
            tp_price=97000.0, sl_price=92000.0,
        )
        runner._positions["BTC/USDT"] = [
            _make_position(level=0, entry_price=96000.0, quantity=0.01),
            _make_position(level=1, entry_price=94000.0, quantity=0.01),
        ]
        runner._close_buffer["BTC/USDT"] = deque([96000.0] * 10, maxlen=50)
        sim._runners.append(runner)

        result = sim.get_grid_state()
        assert "BTC/USDT" in result["grid_positions"]
        g = result["grid_positions"]["BTC/USDT"]

        assert g["symbol"] == "BTC/USDT"
        assert g["strategy"] == "envelope_dca"
        assert g["direction"] == "LONG"
        assert g["levels_open"] == 2
        assert g["levels_max"] == 4
        assert g["current_price"] == 96000.0
        assert g["tp_price"] == 97000.0
        assert g["sl_price"] == 92000.0
        assert g["leverage"] == 6
        assert len(g["positions"]) == 2

        # Vérifier que le summary est cohérent
        assert result["summary"]["total_positions"] == 2
        assert result["summary"]["total_assets"] == 1

    def test_unrealized_pnl_long(self):
        """entry=100, current=110, qty=1, LONG → upnl=+10."""
        engine = _make_data_engine({"TEST/USDT": 110.0})
        config = MagicMock()
        config.risk.initial_capital = 10_000

        sim = Simulator(data_engine=engine, config=config)
        runner = _make_grid_runner("envelope_dca", config, engine)
        runner._positions["TEST/USDT"] = [
            _make_position(level=0, entry_price=100.0, quantity=1.0),
        ]
        runner._close_buffer["TEST/USDT"] = deque([110.0] * 10, maxlen=50)
        sim._runners.append(runner)

        result = sim.get_grid_state()
        g = result["grid_positions"]["TEST/USDT"]

        # P&L non réalisé : (110 - 100) × 1.0 = +10
        assert g["unrealized_pnl"] == 10.0
        assert g["unrealized_pnl_pct"] > 0

        # Summary reflète le total
        assert result["summary"]["total_unrealized_pnl"] == 10.0

    def test_unrealized_pnl_short(self):
        """entry=100, current=90, qty=1, SHORT → upnl=+10."""
        engine = _make_data_engine({"TEST/USDT": 90.0})
        config = MagicMock()
        config.risk.initial_capital = 10_000

        sim = Simulator(data_engine=engine, config=config)
        runner = _make_grid_runner("envelope_dca_short", config, engine)
        runner._positions["TEST/USDT"] = [
            _make_position(level=0, direction=Direction.SHORT, entry_price=100.0, quantity=1.0),
        ]
        runner._close_buffer["TEST/USDT"] = deque([90.0] * 10, maxlen=50)
        sim._runners.append(runner)

        result = sim.get_grid_state()
        g = result["grid_positions"]["TEST/USDT"]

        # P&L non réalisé : (100 - 90) × 1.0 = +10
        assert g["unrealized_pnl"] == 10.0
        assert g["direction"] == "SHORT"

    def test_multi_asset(self):
        """Positions sur 2 assets → 2 entrées dans grid_positions."""
        engine = _make_data_engine({"BTC/USDT": 96000.0, "ETH/USDT": 3200.0})
        config = MagicMock()
        config.risk.initial_capital = 10_000

        sim = Simulator(data_engine=engine, config=config)
        runner = _make_grid_runner("envelope_dca", config, engine)
        runner._positions["BTC/USDT"] = [
            _make_position(level=0, entry_price=95000.0, quantity=0.01),
        ]
        runner._positions["ETH/USDT"] = [
            _make_position(level=0, entry_price=3100.0, quantity=0.1),
        ]
        runner._close_buffer["BTC/USDT"] = deque([96000.0] * 10, maxlen=50)
        runner._close_buffer["ETH/USDT"] = deque([3200.0] * 10, maxlen=50)
        sim._runners.append(runner)

        result = sim.get_grid_state()
        assert len(result["grid_positions"]) == 2
        assert "BTC/USDT" in result["grid_positions"]
        assert "ETH/USDT" in result["grid_positions"]
        assert result["summary"]["total_assets"] == 2

    def test_price_fallback_no_1m(self):
        """Si pas de candles 1m mais 5m dispo → fallback prix."""
        engine = MagicMock()
        engine.config = MagicMock()
        engine.config.assets = []

        def _get_data(symbol):
            data = MagicMock()
            candle_5m = MagicMock()
            candle_5m.close = 100.0
            data.candles = {"1m": [], "5m": [candle_5m]}
            return data

        engine.get_data.side_effect = _get_data
        engine.get_all_symbols.return_value = ["TEST/USDT"]

        config = MagicMock()
        config.risk.initial_capital = 10_000
        sim = Simulator(data_engine=engine, config=config)
        runner = _make_grid_runner("envelope_dca", config, engine)
        runner._positions["TEST/USDT"] = [
            _make_position(level=0, entry_price=95.0, quantity=1.0),
        ]
        runner._close_buffer["TEST/USDT"] = deque([100.0] * 10, maxlen=50)
        sim._runners.append(runner)

        result = sim.get_grid_state()
        assert "TEST/USDT" in result["grid_positions"]
        assert result["grid_positions"]["TEST/USDT"]["current_price"] == 100.0

    def test_tp_sl_none_when_nan(self):
        """Si TP/SL retourne NaN → None dans le JSON."""
        engine = _make_data_engine({"TEST/USDT": 100.0})
        config = MagicMock()
        config.risk.initial_capital = 10_000

        sim = Simulator(data_engine=engine, config=config)
        # tp_price et sl_price = NaN par défaut dans _make_grid_runner
        runner = _make_grid_runner("envelope_dca", config, engine)
        runner._positions["TEST/USDT"] = [
            _make_position(level=0, entry_price=100.0, quantity=1.0),
        ]
        runner._close_buffer["TEST/USDT"] = deque([100.0] * 10, maxlen=50)
        sim._runners.append(runner)

        result = sim.get_grid_state()
        g = result["grid_positions"]["TEST/USDT"]
        assert g["tp_price"] is None
        assert g["sl_price"] is None
        assert g["tp_distance_pct"] is None
        assert g["sl_distance_pct"] is None


# ─── Tests unrealized_pnl / margin / equity (Hotfix 20f) ────────────────


class TestGridRunnerUnrealizedPnl:
    """Tests pour les champs unrealized_pnl, margin_used, equity dans get_status()."""

    def test_status_includes_unrealized_pnl(self):
        """Runner avec 2 positions LONG à des prix différents → unrealized_pnl correct."""
        config = MagicMock()
        config.risk.initial_capital = 10_000.0
        config.assets = []
        engine = _make_data_engine({"BTC/USDT": 105.0, "ETH/USDT": 55.0})
        runner = _make_grid_runner("grid_atr", config, engine, leverage=10)

        # 2 positions : BTC acheté à 100 (prix courant 105), ETH à 50 (prix courant 55)
        runner._positions["BTC/USDT"] = [
            _make_position(level=0, direction=Direction.LONG, entry_price=100.0, quantity=2.0),
        ]
        runner._positions["ETH/USDT"] = [
            _make_position(level=0, direction=Direction.LONG, entry_price=50.0, quantity=3.0),
        ]
        # Simuler les prix courants (normalement mis à jour par on_candle)
        runner._last_prices["BTC/USDT"] = 105.0
        runner._last_prices["ETH/USDT"] = 55.0

        status = runner.get_status()

        # BTC: (105 - 100) × 2 = 10, ETH: (55 - 50) × 3 = 15 → total = 25
        assert status["unrealized_pnl"] == 25.0

    def test_status_includes_margin_used(self):
        """margin_used = somme (entry_price × quantity / leverage) pour toutes les positions."""
        config = MagicMock()
        config.risk.initial_capital = 10_000.0
        config.assets = []
        engine = _make_data_engine({"BTC/USDT": 100.0})
        runner = _make_grid_runner("grid_atr", config, engine, leverage=10)

        runner._positions["BTC/USDT"] = [
            _make_position(level=0, entry_price=100.0, quantity=2.0),
            _make_position(level=1, entry_price=95.0, quantity=2.0),
        ]
        runner._last_prices["BTC/USDT"] = 100.0

        status = runner.get_status()

        # Marge: (100×2)/10 + (95×2)/10 = 20 + 19 = 39
        assert status["margin_used"] == 39.0

    def test_status_equity_equals_capital_plus_margin_plus_unrealized(self):
        """equity = capital + margin_used + unrealized_pnl."""
        config = MagicMock()
        config.risk.initial_capital = 10_000.0
        config.assets = []
        engine = _make_data_engine({"BTC/USDT": 90.0})
        runner = _make_grid_runner("grid_atr", config, engine, leverage=10)

        runner._positions["BTC/USDT"] = [
            _make_position(level=0, direction=Direction.LONG, entry_price=100.0, quantity=5.0),
        ]
        runner._last_prices["BTC/USDT"] = 90.0  # perte de 10 × 5 = -50

        status = runner.get_status()

        # margin = (100 × 5) / 10 = 50
        assert status["unrealized_pnl"] == -50.0
        assert status["margin_used"] == 50.0
        assert status["equity"] == status["capital"] + status["margin_used"] + status["unrealized_pnl"]
        assert status["equity"] == 10_000.0 + 50.0 - 50.0

    def test_status_no_positions_unrealized_zero(self):
        """Sans positions, unrealized_pnl = 0, equity = capital."""
        config = MagicMock()
        config.risk.initial_capital = 10_000.0
        config.assets = []
        engine = _make_data_engine({})
        runner = _make_grid_runner("grid_atr", config, engine, leverage=10)

        status = runner.get_status()

        assert status["unrealized_pnl"] == 0.0
        assert status["margin_used"] == 0.0
        assert status["equity"] == 10_000.0
        assert status["assets_with_positions"] == 0

    def test_status_short_position_unrealized(self):
        """Position SHORT : unrealized = (entry - current) × qty."""
        config = MagicMock()
        config.risk.initial_capital = 10_000.0
        config.assets = []
        engine = _make_data_engine({"BTC/USDT": 95.0})
        runner = _make_grid_runner("grid_atr", config, engine, leverage=10)

        runner._positions["BTC/USDT"] = [
            _make_position(level=0, direction=Direction.SHORT, entry_price=100.0, quantity=2.0),
        ]
        runner._last_prices["BTC/USDT"] = 95.0

        status = runner.get_status()

        # SHORT: (100 - 95) × 2 = 10
        assert status["unrealized_pnl"] == 10.0
        assert status["assets_with_positions"] == 1


# ─── Tests per_asset filtering (Hotfix 28b) ────────────────────────────


class TestGridRunnerPerAssetFilter:
    """Tests pour le filtre per_asset : seuls les symbols autorisés sont tradés."""

    @pytest.mark.asyncio
    async def test_skip_symbol_not_in_per_asset(self):
        """Si symbol pas dans per_asset → on_candle() skip sans créer de position."""
        config = MagicMock()
        config.risk.initial_capital = 10_000.0
        config.assets = []
        engine = _make_data_engine({"BTC/USDT": 100.0, "SOL/USDT": 50.0})

        # Créer un runner avec per_asset = {"BTC/USDT": {}}
        strategy = MagicMock()
        strategy.name = "grid_atr"
        strategy._config.timeframe = "1h"
        strategy._config.ma_period = 7
        strategy._config.leverage = 6
        strategy._config.per_asset = {"BTC/USDT": {}}  # Seul BTC autorisé
        strategy.max_positions = 4
        strategy.min_candles = {"1h": 50}
        strategy.get_tp_price.return_value = 110.0
        strategy.get_sl_price.return_value = 90.0
        strategy.get_current_conditions.return_value = []

        indicator_engine = MagicMock(spec=IncrementalIndicatorEngine)
        indicator_engine.get_indicators.return_value = {"1h": {"sma": 100.0, "close": 100.0}}

        runner = GridStrategyRunner(
            strategy=strategy,
            config=config,
            indicator_engine=indicator_engine,
            grid_position_manager=_make_gpm(),
            data_engine=engine,
        )
        runner._is_warming_up = False

        # Pré-remplir le buffer pour passer la condition min candles
        runner._close_buffer["BTC/USDT"] = deque([100.0] * 10, maxlen=50)
        runner._close_buffer["SOL/USDT"] = deque([50.0] * 10, maxlen=50)

        # Créer des candles test
        from backend.core.models import Candle
        btc_candle = Candle(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
            open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0,
        )
        sol_candle = Candle(
            symbol="SOL/USDT",
            timeframe="1h",
            timestamp=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
            open=50.0, high=51.0, low=49.0, close=50.0, volume=500.0,
        )

        # Envoyer une candle BTC (autorisé)
        await runner.on_candle("BTC/USDT", "1h", btc_candle)
        # Envoyer une candle SOL (non autorisé)
        await runner.on_candle("SOL/USDT", "1h", sol_candle)

        # Vérifier que SOL n'a AUCUNE position
        assert "SOL/USDT" not in runner._positions or len(runner._positions["SOL/USDT"]) == 0
        # BTC devrait être dans _last_prices (traité)
        assert "BTC/USDT" in runner._last_prices
        # SOL ne devrait PAS être dans _last_prices (skippé)
        assert "SOL/USDT" not in runner._last_prices

    @pytest.mark.asyncio
    async def test_empty_per_asset_allows_all(self):
        """Si per_asset est vide → backward compat, tous les symbols sont acceptés."""
        config = MagicMock()
        config.risk.initial_capital = 10_000.0
        config.assets = []
        engine = _make_data_engine({"BTC/USDT": 100.0, "SOL/USDT": 50.0})

        # Créer un runner avec per_asset vide
        strategy = MagicMock()
        strategy.name = "grid_atr"
        strategy._config.timeframe = "1h"
        strategy._config.ma_period = 7
        strategy._config.leverage = 6
        strategy._config.per_asset = {}  # Vide = tous acceptés
        strategy.max_positions = 4
        strategy.min_candles = {"1h": 50}
        strategy.get_tp_price.return_value = float("nan")
        strategy.get_sl_price.return_value = float("nan")
        strategy.get_current_conditions.return_value = []

        indicator_engine = MagicMock(spec=IncrementalIndicatorEngine)
        indicator_engine.get_indicators.return_value = {"1h": {"sma": 100.0, "close": 100.0}}

        runner = GridStrategyRunner(
            strategy=strategy,
            config=config,
            indicator_engine=indicator_engine,
            grid_position_manager=_make_gpm(),
            data_engine=engine,
        )
        runner._is_warming_up = False

        # Pré-remplir le buffer
        runner._close_buffer["BTC/USDT"] = deque([100.0] * 10, maxlen=50)
        runner._close_buffer["SOL/USDT"] = deque([50.0] * 10, maxlen=50)

        # Créer des candles
        from backend.core.models import Candle
        btc_candle = Candle(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
            open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0,
        )
        sol_candle = Candle(
            symbol="SOL/USDT",
            timeframe="1h",
            timestamp=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
            open=50.0, high=51.0, low=49.0, close=50.0, volume=500.0,
        )

        # Envoyer les deux candles
        await runner.on_candle("BTC/USDT", "1h", btc_candle)
        await runner.on_candle("SOL/USDT", "1h", sol_candle)

        # Les deux devraient être traités (présents dans _last_prices)
        assert "BTC/USDT" in runner._last_prices
        assert "SOL/USDT" in runner._last_prices
