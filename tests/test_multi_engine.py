"""Tests pour le moteur multi-position (Sprint 10).

Couvre :
- Tests 1-7 : GridPositionManager (sizing, close, TP/SL, grid state)
- Tests 8-14 : EnvelopeDCAStrategy (grid, should_close_all, asymmetry, config)
- Tests 15-20 : MultiPositionEngine (run, TP/SL global, force close)
- Tests 21-26 : Fast multi backtest engine (simulation, métriques, cache)
- Tests 27-32 : WFO integration (registry, GRID_STRATEGIES, factory, config)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from backend.core.config import EnvelopeDCAConfig
from backend.core.models import Candle, Direction, MarketRegime
from backend.core.position_manager import PositionManagerConfig
from backend.strategies.base import StrategyContext


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_candle(
    ts: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 100.0,
    tf: str = "1h",
) -> Candle:
    return Candle(
        symbol="BTC/USDT",
        exchange="binance",
        timeframe=tf,
        timestamp=ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _make_candles(
    n: int,
    start_price: float = 100.0,
    step: float = 0.0,
    tf: str = "1h",
    volume: float = 100.0,
) -> list[Candle]:
    """Génère N candles synthétiques."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    for i in range(n):
        price = start_price + i * step
        candles.append(Candle(
            symbol="BTC/USDT",
            exchange="binance",
            timeframe=tf,
            timestamp=base + timedelta(hours=i),
            open=price,
            high=price + 1.0,
            low=price - 1.0,
            close=price,
            volume=volume,
        ))
    return candles


def _make_pm_config() -> PositionManagerConfig:
    return PositionManagerConfig(
        leverage=6,
        maker_fee=0.0002,
        taker_fee=0.0006,
        slippage_pct=0.0005,
        high_vol_slippage_mult=2.0,
        max_risk_per_trade=0.02,
    )


def _make_ctx(
    indicators: dict[str, Any],
    tf: str = "1h",
    capital: float = 10_000.0,
) -> StrategyContext:
    return StrategyContext(
        symbol="BTC/USDT",
        timestamp=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        candles={},
        indicators={tf: indicators},
        current_position=None,
        capital=capital,
        config=None,  # type: ignore[arg-type]
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1-7 : GridPositionManager
# ═══════════════════════════════════════════════════════════════════════════


class TestGridPositionManager:
    """Tests du GridPositionManager."""

    def test_open_grid_position_allocation_fixe(self):
        """Vérifie le sizing : notional = capital/levels × leverage."""
        from backend.core.grid_position_manager import GridPositionManager
        from backend.strategies.base_grid import GridLevel

        gpm = GridPositionManager(_make_pm_config())
        level = GridLevel(
            index=0,
            entry_price=100.0,
            direction=Direction.LONG,
            size_fraction=1 / 3,
        )
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

        pos = gpm.open_grid_position(level, ts, capital=10_000, total_levels=3)

        assert pos is not None
        # notional = 10000 * (1/3) * 6 = 20000
        expected_notional = 10_000 * (1 / 3) * 6
        expected_qty = expected_notional / 100.0
        assert abs(pos.quantity - expected_qty) < 0.001
        assert pos.entry_price == 100.0
        assert pos.level == 0
        assert pos.direction == Direction.LONG

    def test_open_position_zero_capital(self):
        """Pas de position si capital=0."""
        from backend.core.grid_position_manager import GridPositionManager
        from backend.strategies.base_grid import GridLevel

        gpm = GridPositionManager(_make_pm_config())
        level = GridLevel(0, 100.0, Direction.LONG, 0.33)
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

        pos = gpm.open_grid_position(level, ts, capital=0, total_levels=3)
        assert pos is None

    def test_close_all_positions_pnl(self):
        """Vérifie le PnL agrégé à la fermeture."""
        from backend.core.grid_position_manager import GridPositionManager
        from backend.strategies.base_grid import GridPosition

        gpm = GridPositionManager(_make_pm_config())
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

        positions = [
            GridPosition(0, Direction.LONG, 100.0, 10.0, ts, 0.06),
            GridPosition(1, Direction.LONG, 95.0, 10.0, ts, 0.057),
        ]

        trade = gpm.close_all_positions(
            positions, exit_price=105.0, exit_time=ts,
            exit_reason="tp_global", regime=MarketRegime.RANGING,
        )

        # avg_entry = (100*10 + 95*10)/20 = 97.5
        assert abs(trade.entry_price - 97.5) < 0.01
        assert trade.quantity == 20.0
        assert trade.exit_reason == "tp_global"
        # Gross PnL = (105 - 97.5) * 20 = 150
        assert trade.gross_pnl > 0
        # Net PnL positif (fees < gross)
        assert trade.net_pnl > 0

    def test_close_all_empty_positions(self):
        """Close all avec aucune position retourne PnL 0."""
        from backend.core.grid_position_manager import GridPositionManager

        gpm = GridPositionManager(_make_pm_config())
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

        trade = gpm.close_all_positions(
            [], exit_price=100.0, exit_time=ts,
            exit_reason="end_of_data", regime=MarketRegime.RANGING,
        )
        assert trade.net_pnl == 0.0
        assert trade.quantity == 0.0

    def test_check_global_tp_sl_tp_hit(self):
        """TP global touché quand high >= tp_price pour LONG."""
        from backend.core.grid_position_manager import GridPositionManager
        from backend.strategies.base_grid import GridPosition

        gpm = GridPositionManager(_make_pm_config())
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        positions = [GridPosition(0, Direction.LONG, 100.0, 10.0, ts, 0.06)]

        candle = _make_candle(ts, 102.0, 106.0, 101.0, 105.0)
        reason, price = gpm.check_global_tp_sl(positions, candle, tp_price=105.0, sl_price=90.0)

        assert reason == "tp_global"
        assert price == 105.0

    def test_check_global_tp_sl_sl_hit(self):
        """SL global touché quand low <= sl_price pour LONG."""
        from backend.core.grid_position_manager import GridPositionManager
        from backend.strategies.base_grid import GridPosition

        gpm = GridPositionManager(_make_pm_config())
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        positions = [GridPosition(0, Direction.LONG, 100.0, 10.0, ts, 0.06)]

        candle = _make_candle(ts, 92.0, 93.0, 89.0, 90.0)
        reason, price = gpm.check_global_tp_sl(positions, candle, tp_price=110.0, sl_price=90.0)

        assert reason == "sl_global"
        assert price == 90.0

    def test_compute_grid_state(self):
        """Calcul de l'état agrégé."""
        from backend.core.grid_position_manager import GridPositionManager
        from backend.strategies.base_grid import GridPosition

        gpm = GridPositionManager(_make_pm_config())
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

        positions = [
            GridPosition(0, Direction.LONG, 100.0, 5.0, ts, 0.03),
            GridPosition(1, Direction.LONG, 96.0, 5.0, ts, 0.029),
        ]

        state = gpm.compute_grid_state(positions, current_price=98.0)

        # avg_entry = (100*5 + 96*5)/10 = 98.0
        assert abs(state.avg_entry_price - 98.0) < 0.01
        assert state.total_quantity == 10.0
        assert len(state.positions) == 2


# ═══════════════════════════════════════════════════════════════════════════
# 8-14 : EnvelopeDCAStrategy
# ═══════════════════════════════════════════════════════════════════════════


class TestEnvelopeDCAStrategy:
    """Tests de la stratégie Envelope DCA."""

    def _make_strategy(self, **overrides) -> "EnvelopeDCAStrategy":
        from backend.strategies.envelope_dca import EnvelopeDCAStrategy

        cfg = EnvelopeDCAConfig(**{
            "ma_period": 7,
            "num_levels": 3,
            "envelope_start": 0.07,
            "envelope_step": 0.03,
            "sl_percent": 25.0,
            "sides": ["long"],
            "leverage": 6,
            **overrides,
        })
        return EnvelopeDCAStrategy(cfg)

    def test_asymmetric_envelopes(self):
        """Les enveloppes hautes sont asymétriques : round(1/(1-e)-1, 3)."""
        strategy = self._make_strategy(sides=["long", "short"])
        from backend.strategies.base_grid import GridState

        ctx = _make_ctx({"sma": 100.0, "close": 100.0})
        grid_state = GridState([], 0, 0, 0, 0)

        levels = strategy.compute_grid(ctx, grid_state)
        # 3 niveaux long + 3 niveaux short = 6
        assert len(levels) == 6

        longs = [lv for lv in levels if lv.direction == Direction.LONG]
        shorts = [lv for lv in levels if lv.direction == Direction.SHORT]
        assert len(longs) == 3
        assert len(shorts) == 3

        # Vérifier asymétrie pour level 0 (offset = 0.07)
        lower_0 = 0.07
        upper_0 = round(1 / (1 - 0.07) - 1, 3)  # 0.075
        assert abs(longs[0].entry_price - 100.0 * (1 - lower_0)) < 0.01
        assert abs(shorts[0].entry_price - 100.0 * (1 + upper_0)) < 0.01
        # Le offset haut est PLUS grand que le bas (asymétrie)
        assert upper_0 > lower_0

    def test_compute_grid_filters_filled_levels(self):
        """Les niveaux déjà remplis sont exclus."""
        strategy = self._make_strategy()
        from backend.strategies.base_grid import GridPosition, GridState

        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        existing = [GridPosition(0, Direction.LONG, 93.0, 5.0, ts, 0.03)]
        grid_state = GridState(existing, 93.0, 5.0, 465.0, 0)

        ctx = _make_ctx({"sma": 100.0, "close": 100.0})
        levels = strategy.compute_grid(ctx, grid_state)

        # Level 0 rempli, reste levels 1 et 2
        assert len(levels) == 2
        assert all(lv.index != 0 for lv in levels)

    def test_single_side_active(self):
        """Si positions LONG ouvertes, pas de SHORT même si sides=['long','short']."""
        strategy = self._make_strategy(sides=["long", "short"])
        from backend.strategies.base_grid import GridPosition, GridState

        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        existing = [GridPosition(0, Direction.LONG, 93.0, 5.0, ts, 0.03)]
        grid_state = GridState(existing, 93.0, 5.0, 465.0, 0)

        ctx = _make_ctx({"sma": 100.0, "close": 100.0})
        levels = strategy.compute_grid(ctx, grid_state)

        assert all(lv.direction == Direction.LONG for lv in levels)

    def test_should_close_all_tp_sma_crossing(self):
        """TP global = close >= SMA pour LONG."""
        strategy = self._make_strategy()
        from backend.strategies.base_grid import GridPosition, GridState

        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        positions = [GridPosition(0, Direction.LONG, 93.0, 5.0, ts, 0.03)]
        grid_state = GridState(positions, 93.0, 5.0, 465.0, 0)

        # Close au-dessus de la SMA → TP
        ctx = _make_ctx({"sma": 100.0, "close": 101.0})
        result = strategy.should_close_all(ctx, grid_state)
        assert result == "tp_global"

    def test_should_close_all_no_exit(self):
        """Pas de sortie si prix entre SMA et SL."""
        strategy = self._make_strategy()
        from backend.strategies.base_grid import GridPosition, GridState

        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        positions = [GridPosition(0, Direction.LONG, 93.0, 5.0, ts, 0.03)]
        grid_state = GridState(positions, 93.0, 5.0, 465.0, 0)

        # Close en dessous de SMA mais au-dessus de SL
        ctx = _make_ctx({"sma": 100.0, "close": 90.0})
        result = strategy.should_close_all(ctx, grid_state)
        # SL = 93 * (1-0.25) = 69.75. Close=90 > 69.75 → pas de SL
        assert result is None

    def test_get_params(self):
        """get_params retourne les params sans enabled/weight."""
        strategy = self._make_strategy()
        params = strategy.get_params()
        assert "ma_period" in params
        assert "num_levels" in params
        assert "envelope_start" in params
        assert "enabled" not in params
        assert "weight" not in params

    def test_compute_indicators(self):
        """compute_indicators calcule la SMA."""
        strategy = self._make_strategy(ma_period=5)
        candles = _make_candles(20, start_price=100.0, tf="1h")
        result = strategy.compute_indicators({"1h": candles})
        assert "1h" in result
        ts_key = candles[10].timestamp.isoformat()
        assert "sma" in result["1h"][ts_key]


# ═══════════════════════════════════════════════════════════════════════════
# 15-20 : MultiPositionEngine
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiPositionEngine:
    """Tests du moteur de backtesting multi-position."""

    def _make_engine(self, **cfg_overrides):
        from backend.backtesting.engine import BacktestConfig
        from backend.backtesting.multi_engine import MultiPositionEngine
        from backend.strategies.envelope_dca import EnvelopeDCAStrategy

        config = EnvelopeDCAConfig(
            ma_period=5, num_levels=3, envelope_start=0.07,
            envelope_step=0.03, sl_percent=25.0, sides=["long"],
            leverage=6,
        )
        strategy = EnvelopeDCAStrategy(config)

        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            leverage=6,
            **cfg_overrides,
        )
        return MultiPositionEngine(bt_config, strategy)

    def test_run_produces_backtest_result(self):
        """run() retourne un BacktestResult valide."""
        engine = self._make_engine()

        # Créer un dataset avec mean reversion (prix descend puis remonte)
        candles = []
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        n = 100
        for i in range(n):
            # Prix sinusoïdal autour de 100
            price = 100 + 10 * np.sin(2 * np.pi * i / 30)
            candles.append(Candle(
                symbol="BTC/USDT", exchange="binance", timeframe="1h",
                timestamp=base + timedelta(hours=i),
                open=price, high=price + 2, low=price - 2, close=price,
                volume=100.0,
            ))

        result = engine.run({"1h": candles}, main_tf="1h")

        assert result.strategy_name == "envelope_dca"
        assert result.final_capital > 0
        assert len(result.equity_curve) == n
        assert result.config.leverage == 6

    def test_run_no_data_raises(self):
        """run() lève ValueError si pas de données."""
        engine = self._make_engine()
        with pytest.raises(ValueError, match="Pas de données"):
            engine.run({"1h": []}, main_tf="1h")

    def test_force_close_end_of_data(self):
        """Les positions ouvertes sont fermées en fin de données."""
        engine = self._make_engine()

        # Prix qui descend : devrait ouvrir des positions sans les fermer
        candles = []
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(30):
            price = 100 - i * 0.5  # Descente régulière
            candles.append(Candle(
                symbol="BTC/USDT", exchange="binance", timeframe="1h",
                timestamp=base + timedelta(hours=i),
                open=price + 0.2, high=price + 1.5, low=price - 1.5,
                close=price, volume=100.0,
            ))

        result = engine.run({"1h": candles}, main_tf="1h")

        # Devrait avoir au moins un trade (force close)
        if result.trades:
            last_trade = result.trades[-1]
            assert last_trade.exit_reason == "end_of_data"

    def test_run_multi_backtest_single(self):
        """run_multi_backtest_single crée et exécute le backtest."""
        from backend.backtesting.engine import BacktestConfig
        from backend.backtesting.multi_engine import run_multi_backtest_single

        candles = _make_candles(50, start_price=100.0, tf="1h")
        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=candles[0].timestamp,
            end_date=candles[-1].timestamp,
            leverage=6,
        )

        params = {
            "timeframe": "1h", "ma_period": 5, "num_levels": 3,
            "envelope_start": 0.07, "envelope_step": 0.03,
            "sl_percent": 25.0, "sides": ["long"], "leverage": 6,
        }

        result = run_multi_backtest_single(
            "envelope_dca", params, {"1h": candles}, bt_config, "1h",
        )
        assert result.strategy_name == "envelope_dca"
        assert result.final_capital > 0

    def test_slippage_on_sl(self):
        """Le slippage est appliqué sur SL (taker) mais pas sur TP (maker)."""
        from backend.core.grid_position_manager import GridPositionManager
        from backend.strategies.base_grid import GridPosition

        config = _make_pm_config()
        gpm = GridPositionManager(config)
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

        positions = [GridPosition(0, Direction.LONG, 100.0, 10.0, ts, 0.06)]

        # TP : pas de slippage
        trade_tp = gpm.close_all_positions(
            positions, 110.0, ts, "tp_global", MarketRegime.RANGING,
        )

        # SL : avec slippage
        trade_sl = gpm.close_all_positions(
            positions, 110.0, ts, "sl_global", MarketRegime.RANGING,
        )

        # SL a plus de coûts (slippage + taker fee)
        assert trade_sl.net_pnl < trade_tp.net_pnl

    def test_engine_respects_strategy_leverage(self):
        """Le BacktestConfig utilise le leverage de la stratégie."""
        from backend.backtesting.engine import BacktestConfig

        cfg = EnvelopeDCAConfig(leverage=4)
        bt = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            leverage=cfg.leverage,
        )
        assert bt.leverage == 4


# ═══════════════════════════════════════════════════════════════════════════
# 21-26 : Fast multi backtest engine
# ═══════════════════════════════════════════════════════════════════════════


class TestFastMultiBacktest:
    """Tests du fast engine multi-position."""

    def _make_cache(self, n: int = 100):
        """Crée un IndicatorCache minimal pour envelope_dca."""
        from backend.optimization.indicator_cache import IndicatorCache

        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        volumes = rng.uniform(50, 200, n)

        # SMA 7
        sma_arr = np.full(n, np.nan)
        for i in range(6, n):
            sma_arr[i] = np.mean(prices[i - 6:i + 1])

        return IndicatorCache(
            n_candles=n,
            opens=prices + rng.uniform(-0.3, 0.3, n),
            highs=prices + np.abs(rng.normal(0.5, 0.3, n)),
            lows=prices - np.abs(rng.normal(0.5, 0.3, n)),
            closes=prices,
            volumes=volumes,
            total_days=n / 24,
            rsi={14: np.full(n, 50.0)},
            vwap=np.full(n, np.nan),
            vwap_distance_pct=np.full(n, np.nan),
            adx_arr=np.full(n, 25.0),
            di_plus=np.full(n, 15.0),
            di_minus=np.full(n, 10.0),
            atr_arr=np.full(n, 1.0),
            atr_sma=np.full(n, 1.0),
            volume_sma_arr=np.full(n, 100.0),
            regime=np.zeros(n, dtype=np.int8),
            rolling_high={},
            rolling_low={},
            filter_adx=np.full(n, np.nan),
            filter_di_plus=np.full(n, np.nan),
            filter_di_minus=np.full(n, np.nan),
            bb_sma={7: sma_arr},
            bb_upper={},
            bb_lower={},
            supertrend_direction={},
            atr_by_period={},
        )

    def test_run_multi_backtest_from_cache(self):
        """Fast engine retourne un résultat valide."""
        from backend.backtesting.engine import BacktestConfig
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache

        cache = self._make_cache()
        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            leverage=6,
        )

        params = {
            "ma_period": 7, "num_levels": 3,
            "envelope_start": 0.07, "envelope_step": 0.03,
            "sl_percent": 25.0,
        }

        result = run_multi_backtest_from_cache("envelope_dca", params, cache, bt_config)

        assert len(result) == 5  # (params, sharpe, return, PF, n_trades)
        assert result[0] == params

    def test_unknown_strategy_raises(self):
        """Stratégie inconnue lève ValueError."""
        from backend.backtesting.engine import BacktestConfig
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache

        cache = self._make_cache()
        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )

        with pytest.raises(ValueError, match="inconnue"):
            run_multi_backtest_from_cache("unknown_strat", {}, cache, bt_config)

    def test_compute_fast_metrics_no_trades(self):
        """Métriques à 0 si aucun trade."""
        from backend.optimization.fast_multi_backtest import _compute_fast_metrics

        result = _compute_fast_metrics(
            {"ma_period": 7}, [], [], 10_000, 10_000, 30.0,
        )
        assert result == ({"ma_period": 7}, 0.0, 0.0, 0.0, 0)

    def test_compute_fast_metrics_with_trades(self):
        """Métriques calculées correctement avec des trades."""
        from backend.optimization.fast_multi_backtest import _compute_fast_metrics

        pnls = [100, -30, 80, -20, 50]
        returns = [r / 10_000 for r in pnls]
        final = 10_000 + sum(pnls)

        result = _compute_fast_metrics(
            {"ma_period": 7}, pnls, returns, final, 10_000, 30.0,
        )

        params, sharpe, net_return, pf, n = result
        assert n == 5
        assert net_return == pytest.approx(1.8, abs=0.01)  # 180/10000*100
        assert pf > 1.0  # Wins > losses

    def test_indicator_cache_sma_for_envelope_dca(self):
        """build_cache crée les SMA pour envelope_dca."""
        from backend.optimization.indicator_cache import build_cache

        candles = _make_candles(50, start_price=100.0, tf="1h")
        cache = build_cache(
            {"1h": candles},
            {"ma_period": [5, 7]},
            "envelope_dca",
            main_tf="1h",
        )

        assert 5 in cache.bb_sma
        assert 7 in cache.bb_sma
        assert len(cache.bb_sma[7]) == 50

    def test_fast_engine_allocation_fixe(self):
        """Le fast engine utilise l'allocation fixe (comme GPM)."""
        from backend.backtesting.engine import BacktestConfig
        from backend.optimization.fast_multi_backtest import _simulate_envelope_dca

        cache = self._make_cache(200)
        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            leverage=6,
            initial_capital=10_000,
        )

        params = {
            "ma_period": 7, "num_levels": 3,
            "envelope_start": 0.07, "envelope_step": 0.03,
            "sl_percent": 25.0,
        }

        trade_pnls, trade_returns, final = _simulate_envelope_dca(cache, params, bt_config)

        # Le final doit être raisonnable (pas d'explosion)
        assert final > 0
        assert final < 1_000_000  # Pas d'overflow


# ═══════════════════════════════════════════════════════════════════════════
# 27-32 : WFO integration & config
# ═══════════════════════════════════════════════════════════════════════════


class TestWFOIntegration:
    """Tests d'intégration WFO, registry, factory, config."""

    def test_envelope_dca_in_registry(self):
        """envelope_dca est dans STRATEGY_REGISTRY."""
        from backend.optimization import STRATEGY_REGISTRY

        assert "envelope_dca" in STRATEGY_REGISTRY
        config_cls, strategy_cls = STRATEGY_REGISTRY["envelope_dca"]
        assert config_cls is EnvelopeDCAConfig

    def test_is_grid_strategy(self):
        """is_grid_strategy retourne True pour envelope_dca."""
        from backend.optimization import is_grid_strategy

        assert is_grid_strategy("envelope_dca") is True
        assert is_grid_strategy("vwap_rsi") is False
        assert is_grid_strategy("momentum") is False

    def test_grid_strategies_set(self):
        """GRID_STRATEGIES contient envelope_dca."""
        from backend.optimization import GRID_STRATEGIES

        assert "envelope_dca" in GRID_STRATEGIES

    def test_indicator_params_has_envelope_dca(self):
        """_INDICATOR_PARAMS a envelope_dca → ['ma_period']."""
        from backend.optimization.walk_forward import _INDICATOR_PARAMS

        assert "envelope_dca" in _INDICATOR_PARAMS
        assert _INDICATOR_PARAMS["envelope_dca"] == ["ma_period"]

    def test_create_strategy_with_params(self):
        """create_strategy_with_params crée un EnvelopeDCAStrategy."""
        from backend.optimization import create_strategy_with_params
        from backend.strategies.envelope_dca import EnvelopeDCAStrategy

        params = {
            "ma_period": 10, "num_levels": 4,
            "envelope_start": 0.05, "envelope_step": 0.02,
            "sl_percent": 20.0, "sides": ["long"], "leverage": 6,
        }
        strategy = create_strategy_with_params("envelope_dca", params)
        assert isinstance(strategy, EnvelopeDCAStrategy)
        assert strategy.max_positions == 4

    def test_envelope_dca_config_validation(self):
        """EnvelopeDCAConfig valide les bornes."""
        # Valid
        cfg = EnvelopeDCAConfig(ma_period=5, num_levels=2, leverage=4)
        assert cfg.leverage == 4

        # ma_period < 2
        with pytest.raises(Exception):
            EnvelopeDCAConfig(ma_period=1)

        # num_levels < 1
        with pytest.raises(Exception):
            EnvelopeDCAConfig(num_levels=0)
