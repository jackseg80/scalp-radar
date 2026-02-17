"""Tests Sprint 27 — Filtre Darwinien par Régime.

Couvre :
- REGIME_LIVE_TO_WFO mapping (constante)
- _should_allow_new_grid() logique de filtrage
- Intégration on_candle() (filter blocks new grid)
- Config regime_filter_enabled toggle
- Database.get_regime_profiles() requête
- PortfolioBacktester._create_runners() propagation
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from backend.backtesting.simulator import (
    REGIME_LIVE_TO_WFO,
    GridStrategyRunner,
    RunnerStats,
)
from backend.core.grid_position_manager import GridPositionManager
from backend.core.incremental_indicators import IncrementalIndicatorEngine
from backend.core.models import Candle, Direction, MarketRegime, TimeFrame
from backend.core.position_manager import PositionManagerConfig
from backend.strategies.base_grid import BaseGridStrategy, GridPosition


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_gpm_config(leverage: int = 6) -> PositionManagerConfig:
    return PositionManagerConfig(
        leverage=leverage,
        maker_fee=0.0002,
        taker_fee=0.0006,
        slippage_pct=0.0005,
        high_vol_slippage_mult=2.0,
        max_risk_per_trade=0.02,
    )


def _make_mock_strategy(
    name: str = "grid_atr",
    timeframe: str = "1h",
    ma_period: int = 7,
    max_positions: int = 3,
) -> MagicMock:
    strategy = MagicMock(spec=BaseGridStrategy)
    strategy.name = name
    config = MagicMock()
    config.timeframe = timeframe
    config.ma_period = ma_period
    config.leverage = 6
    config.per_asset = {}
    config.sl_percent = 20.0
    strategy._config = config
    strategy.min_candles = {"1h": 50}
    strategy.max_positions = max_positions
    strategy.compute_grid.return_value = []
    strategy.should_close_all.return_value = None
    strategy.get_tp_price.return_value = float("nan")
    strategy.get_sl_price.return_value = float("nan")
    strategy.get_current_conditions.return_value = []
    strategy.compute_live_indicators.return_value = {}
    return strategy


def _make_mock_config(
    n_assets: int = 1,
    regime_filter_enabled: bool = True,
) -> MagicMock:
    config = MagicMock()
    config.risk.initial_capital = 10_000.0
    config.risk.max_margin_ratio = 0.70
    config.risk.fees.maker_percent = 0.02
    config.risk.fees.taker_percent = 0.06
    config.risk.slippage.default_estimate_percent = 0.05
    config.risk.slippage.high_volatility_multiplier = 2.0
    config.risk.position.max_risk_per_trade_percent = 2.0
    config.risk.position.default_leverage = 15
    config.risk.kill_switch.max_session_loss_percent = 5.0
    config.risk.kill_switch.max_daily_loss_percent = 10.0
    config.risk.kill_switch.grid_max_session_loss_percent = 25.0
    config.risk.kill_switch.grid_max_daily_loss_percent = 25.0
    config.risk.kill_switch.global_max_loss_pct = 30.0
    config.risk.kill_switch.global_window_hours = 24
    config.risk.regime_filter_enabled = regime_filter_enabled
    config.assets = [MagicMock(symbol=f"ASSET{i}/USDT") for i in range(n_assets)]
    return config


def _make_grid_runner(
    strategy=None,
    config=None,
    regime_profile=None,
) -> GridStrategyRunner:
    if strategy is None:
        strategy = _make_mock_strategy()
    if config is None:
        config = _make_mock_config()

    indicator_engine = MagicMock(spec=IncrementalIndicatorEngine)
    indicator_engine.get_indicators.return_value = {}
    indicator_engine.update = MagicMock()

    gpm = GridPositionManager(_make_gpm_config())
    data_engine = MagicMock()
    data_engine.get_funding_rate.return_value = None
    data_engine.get_open_interest.return_value = []

    runner = GridStrategyRunner(
        strategy=strategy,
        config=config,
        indicator_engine=indicator_engine,
        grid_position_manager=gpm,
        data_engine=data_engine,
        regime_profile=regime_profile,
    )
    runner._is_warming_up = False
    return runner


def _fill_buffer(
    runner: GridStrategyRunner,
    symbol: str = "BTC/USDT",
    n: int = 10,
    base_close: float = 100_000.0,
) -> None:
    runner._close_buffer[symbol] = deque(maxlen=50)
    for i in range(n):
        runner._close_buffer[symbol].append(base_close + i * 10)


def _make_candle(
    close: float = 100_000.0,
    high: float | None = None,
    low: float | None = None,
    ts: datetime | None = None,
) -> Candle:
    h = high or close * 1.001
    lo = low or close * 0.999
    return Candle(
        timestamp=ts or datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        open=close,
        high=h,
        low=lo,
        close=close,
        volume=100.0,
        symbol="BTC/USDT",
        timeframe=TimeFrame.H1,
    )


# Bear profile: sharpe négatif
BEAR_NEGATIVE_PROFILE = {
    "BTC/USDT": {
        "bull": {"avg_oos_sharpe": 6.88, "consistency": 1.0, "n_windows": 3},
        "bear": {"avg_oos_sharpe": -2.4, "consistency": 0.0, "n_windows": 1},
        "range": {"avg_oos_sharpe": 5.56, "consistency": 1.0, "n_windows": 3},
    }
}

# All positive profile
ALL_POSITIVE_PROFILE = {
    "BTC/USDT": {
        "bull": {"avg_oos_sharpe": 6.88, "consistency": 1.0, "n_windows": 3},
        "bear": {"avg_oos_sharpe": 1.5, "consistency": 0.5, "n_windows": 2},
        "range": {"avg_oos_sharpe": 5.56, "consistency": 1.0, "n_windows": 3},
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# TestRegimeMapping
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegimeMapping:
    """Vérifie la constante REGIME_LIVE_TO_WFO."""

    def test_mapping_four_regimes(self):
        """4 régimes mappés : UP→bull, DOWN→bear, RANGING→range, HIGH_VOL→crash."""
        assert REGIME_LIVE_TO_WFO[MarketRegime.TRENDING_UP] == "bull"
        assert REGIME_LIVE_TO_WFO[MarketRegime.TRENDING_DOWN] == "bear"
        assert REGIME_LIVE_TO_WFO[MarketRegime.RANGING] == "range"
        assert REGIME_LIVE_TO_WFO[MarketRegime.HIGH_VOLATILITY] == "crash"
        assert len(REGIME_LIVE_TO_WFO) == 4

    def test_low_volatility_not_mapped(self):
        """LOW_VOLATILITY n'a pas de correspondance WFO → absent du dict."""
        assert MarketRegime.LOW_VOLATILITY not in REGIME_LIVE_TO_WFO


# ═══════════════════════════════════════════════════════════════════════════════
# TestShouldAllowNewGrid
# ═══════════════════════════════════════════════════════════════════════════════


class TestShouldAllowNewGrid:
    """Vérifie _should_allow_new_grid() en isolation."""

    def test_no_profile_allows(self):
        """Pas de regime_profile → toujours autorisé (backward compat)."""
        runner = _make_grid_runner(regime_profile=None)
        runner._current_regime = MarketRegime.TRENDING_DOWN
        assert runner._should_allow_new_grid("BTC/USDT") is True

    def test_negative_sharpe_blocks(self):
        """Bear sharpe < 0 + regime TRENDING_DOWN → bloqué."""
        runner = _make_grid_runner(regime_profile=BEAR_NEGATIVE_PROFILE)
        runner._current_regime = MarketRegime.TRENDING_DOWN
        assert runner._should_allow_new_grid("BTC/USDT") is False
        assert runner._regime_filter_blocks == 1

    def test_positive_sharpe_allows(self):
        """Bear sharpe > 0 + regime TRENDING_DOWN → autorisé."""
        runner = _make_grid_runner(regime_profile=ALL_POSITIVE_PROFILE)
        runner._current_regime = MarketRegime.TRENDING_DOWN
        assert runner._should_allow_new_grid("BTC/USDT") is True

    def test_uncovered_regime_allows(self):
        """Régime 'crash' absent du profil → bénéfice du doute → autorisé."""
        # BEAR_NEGATIVE_PROFILE n'a pas de clé "crash"
        runner = _make_grid_runner(regime_profile=BEAR_NEGATIVE_PROFILE)
        runner._current_regime = MarketRegime.HIGH_VOLATILITY
        assert runner._should_allow_new_grid("BTC/USDT") is True

    def test_zero_sharpe_allows(self):
        """Sharpe = 0.0 (>= 0) → autorisé."""
        profile = {
            "BTC/USDT": {
                "bear": {"avg_oos_sharpe": 0.0, "consistency": 0.5, "n_windows": 2},
            }
        }
        runner = _make_grid_runner(regime_profile=profile)
        runner._current_regime = MarketRegime.TRENDING_DOWN
        assert runner._should_allow_new_grid("BTC/USDT") is True

    def test_filter_disabled_via_config(self):
        """regime_filter_enabled=False → filtre ignoré, même sharpe négatif."""
        config = _make_mock_config(regime_filter_enabled=False)
        runner = _make_grid_runner(
            config=config,
            regime_profile=BEAR_NEGATIVE_PROFILE,
        )
        runner._current_regime = MarketRegime.TRENDING_DOWN
        assert runner._should_allow_new_grid("BTC/USDT") is True


# ═══════════════════════════════════════════════════════════════════════════════
# TestRegimeFilterIntegration
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegimeFilterIntegration:
    """Tests d'intégration : on_candle, DB, portfolio."""

    @pytest.mark.asyncio
    async def test_blocks_new_grid_on_candle(self):
        """Pas de positions + régime défavorable → compute_grid jamais appelé."""
        from unittest.mock import patch

        runner = _make_grid_runner(regime_profile=BEAR_NEGATIVE_PROFILE)
        _fill_buffer(runner, "BTC/USDT", n=10)

        candle = _make_candle(
            close=100_000.0,
            ts=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        )

        # Pas de positions ouvertes
        assert runner._positions.get("BTC/USDT", []) == []

        # Mocker detect_market_regime pour retourner TRENDING_DOWN
        with patch(
            "backend.backtesting.simulator.detect_market_regime",
            return_value=MarketRegime.TRENDING_DOWN,
        ):
            await runner.on_candle("BTC/USDT", "1h", candle)

        # compute_grid ne doit PAS avoir été appelé
        runner._strategy.compute_grid.assert_not_called()
        assert runner._regime_filter_blocks == 1

    @pytest.mark.asyncio
    async def test_does_not_block_existing_positions(self):
        """Positions existantes + régime défavorable → filtre bypasse (DCA continue)."""
        runner = _make_grid_runner(regime_profile=BEAR_NEGATIVE_PROFILE)
        runner._current_regime = MarketRegime.TRENDING_DOWN
        _fill_buffer(runner, "BTC/USDT", n=10)

        # Ajouter une position existante
        runner._positions["BTC/USDT"] = [
            GridPosition(
                level=0,
                direction=Direction.LONG,
                entry_price=95_000.0,
                quantity=0.01,
                entry_time=datetime(2024, 6, 14, 10, 0, tzinfo=timezone.utc),
                entry_fee=0.57,
            ),
        ]

        candle = _make_candle(
            close=100_000.0,
            ts=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        )

        await runner.on_candle("BTC/USDT", "1h", candle)

        # Le filtre est bypasse car positions existent → on arrive au compute_grid
        # (ou au check TP/SL, dans tous les cas le filtre ne bloque pas)
        # Vérifions que le compteur n'a PAS été incrémenté
        assert runner._regime_filter_blocks == 0

    @pytest.mark.asyncio
    async def test_get_regime_profiles_from_db(self, tmp_path):
        """get_regime_profiles() parse correctement le JSON depuis la DB."""
        from backend.core.database import Database

        db_path = str(tmp_path / "test.db")

        # Utiliser Database.init() pour créer le schéma complet
        db = Database(db_path)
        await db.init()

        # Insérer des données de test via SQL direct
        ra1 = json.dumps({
            "bull": {"avg_oos_sharpe": 6.88, "consistency": 1.0, "n_windows": 3},
            "bear": {"avg_oos_sharpe": -2.4, "consistency": 0.0, "n_windows": 1},
        })
        ra2 = json.dumps({
            "range": {"avg_oos_sharpe": 3.2, "consistency": 0.8, "n_windows": 4},
        })
        # INSERT avec toutes les colonnes NOT NULL du schéma
        INSERT_SQL = (
            "INSERT INTO optimization_results "
            "(strategy_name, asset, timeframe, created_at, grade, total_score, "
            "n_windows, best_params, is_latest, regime_analysis) "
            "VALUES (?, ?, '1h', '2024-01-01', 'B', 50, 10, '{}', ?, ?)"
        )
        assert db._conn is not None
        await db._conn.execute(INSERT_SQL, ("grid_atr", "BTC/USDT", 1, ra1))
        await db._conn.execute(INSERT_SQL, ("grid_atr", "ETH/USDT", 1, ra2))
        # Résultat non-latest → ignoré
        await db._conn.execute(
            INSERT_SQL, ("grid_atr", "SOL/USDT", 0, '{"bull": {"avg_oos_sharpe": 9.0}}'),
        )
        # Résultat sans regime_analysis → ignoré
        await db._conn.execute(INSERT_SQL, ("grid_atr", "DOGE/USDT", 1, None))
        await db._conn.commit()

        profiles = await db.get_regime_profiles("grid_atr")
        await db.close()

        assert len(profiles) == 2
        assert "BTC/USDT" in profiles
        assert "ETH/USDT" in profiles
        assert profiles["BTC/USDT"]["bear"]["avg_oos_sharpe"] == -2.4
        assert profiles["ETH/USDT"]["range"]["avg_oos_sharpe"] == 3.2
        # SOL/USDT (non-latest) et DOGE/USDT (NULL) absents
        assert "SOL/USDT" not in profiles
        assert "DOGE/USDT" not in profiles

    def test_portfolio_runners_receive_regime_profile(self):
        """_create_runners() propage correctement le regime_profile aux runners."""
        from backend.backtesting.portfolio_engine import PortfolioBacktester

        config = MagicMock()
        config.risk.initial_capital = 10_000.0
        config.risk.fees.maker_percent = 0.02
        config.risk.fees.taker_percent = 0.06
        config.risk.slippage.default_estimate_percent = 0.05
        config.risk.slippage.high_volatility_multiplier = 2.0
        config.risk.position.max_risk_per_trade_percent = 2.0
        config.risk.regime_filter_enabled = True
        config.strategies.grid_atr.per_asset = {}

        bt = PortfolioBacktester.__new__(PortfolioBacktester)
        bt._config = config
        bt._initial_capital = 10_000.0
        bt._multi_strategies = [("grid_atr", ["BTC/USDT", "ETH/USDT"])]
        bt._assets = ["BTC/USDT", "ETH/USDT"]
        bt._exchange = "binance"

        regime_profiles = {
            "grid_atr": {
                "BTC/USDT": {"bear": {"avg_oos_sharpe": -2.4}},
                "ETH/USDT": {"bull": {"avg_oos_sharpe": 5.0}},
            }
        }

        runners, _ = bt._create_runners(
            [("grid_atr", ["BTC/USDT", "ETH/USDT"])],
            5_000.0,
            regime_profiles,
        )

        # Chaque runner a reçu le profil de la stratégie
        for key, runner in runners.items():
            assert runner._regime_profile is not None
            assert "BTC/USDT" in runner._regime_profile
            assert "ETH/USDT" in runner._regime_profile
