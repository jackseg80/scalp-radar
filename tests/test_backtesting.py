"""Tests pour backend/backtesting/engine.py et metrics.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from backend.backtesting.engine import BacktestConfig, BacktestEngine, TradeResult
from backend.backtesting.metrics import BacktestMetrics, calculate_metrics, format_metrics_table
from backend.core.models import Candle, Direction, MarketRegime, TimeFrame
from backend.strategies.base import BaseStrategy, OpenPosition, StrategyContext, StrategySignal
from backend.core.models import SignalStrength


# ─── Stratégie de test ───────────────────────────────────────────────────────


class AlwaysLongStrategy(BaseStrategy):
    """Stratégie de test : entre LONG à chaque opportunité."""

    name = "test_always_long"

    def __init__(self, tp_pct: float = 0.5, sl_pct: float = 0.25):
        self._tp_pct = tp_pct
        self._sl_pct = sl_pct
        self._first_entry = True

    def evaluate(self, ctx: StrategyContext) -> StrategySignal | None:
        if ctx.current_position is not None:
            return None
        if not self._first_entry:
            return None
        self._first_entry = False
        main_ind = ctx.indicators.get("5m", {})
        close = main_ind.get("close", 100.0)
        return StrategySignal(
            direction=Direction.LONG,
            entry_price=close,
            tp_price=close * (1 + self._tp_pct / 100),
            sl_price=close * (1 - self._sl_pct / 100),
            score=0.8,
            strength=SignalStrength.STRONG,
            market_regime=MarketRegime.RANGING,
        )

    def check_exit(self, ctx: StrategyContext, position: OpenPosition) -> str | None:
        return None

    def compute_indicators(self, candles_by_tf):
        result = {}
        for tf, candles in candles_by_tf.items():
            result[tf] = {}
            for c in candles:
                result[tf][c.timestamp.isoformat()] = {
                    "close": c.close,
                    "rsi": 50.0,
                    "vwap": c.close,
                    "adx": 15.0,
                    "di_plus": 12.0,
                    "di_minus": 12.0,
                    "atr": 1.0,
                    "atr_sma": 1.0,
                    "volume": c.volume,
                    "volume_sma": c.volume,
                }
        return result

    @property
    def min_candles(self):
        return {"5m": 1}

    def get_params(self):
        return {"tp_pct": self._tp_pct, "sl_pct": self._sl_pct}


def _make_candles(prices: list[tuple[float, float, float, float]], start_minutes: int = 0) -> list[Candle]:
    """Crée des bougies à partir de (open, high, low, close)."""
    candles = []
    base_time = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
    for i, (o, h, l, c) in enumerate(prices):
        candles.append(Candle(
            timestamp=base_time + timedelta(minutes=(start_minutes + i) * 5),
            open=o, high=h, low=l, close=c,
            volume=1000.0,
            symbol="BTC/USDT",
            timeframe=TimeFrame.M5,
        ))
    return candles


def _default_config() -> BacktestConfig:
    return BacktestConfig(
        symbol="BTC/USDT",
        start_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
        end_date=datetime(2024, 1, 16, tzinfo=timezone.utc),
        initial_capital=10_000.0,
        leverage=15,
        maker_fee=0.0002,
        taker_fee=0.0006,
        slippage_pct=0.0005,
        high_vol_slippage_mult=2.0,
        max_risk_per_trade=0.02,
    )


# ─── Tests du moteur ─────────────────────────────────────────────────────────


class TestBacktestEngine:
    def test_tp_hit(self):
        """Trade LONG : TP touché → profit avec maker fee."""
        # Entry à 100, TP à 100.5, les bougies montent
        candles = _make_candles([
            (100, 100.5, 99.5, 100),  # Bougie d'entrée
            (100, 100.6, 99.8, 100.3),  # TP touché (high > 100.5)
        ])
        config = _default_config()
        strategy = AlwaysLongStrategy(tp_pct=0.5, sl_pct=0.25)
        engine = BacktestEngine(config, strategy)
        result = engine.run({"5m": candles}, main_tf="5m")

        assert len(result.trades) >= 1
        trade = result.trades[0]
        assert trade.exit_reason == "tp"
        assert trade.exit_price == pytest.approx(100.5)
        # Fee TP = maker
        assert trade.slippage_cost == 0.0  # Pas de slippage sur TP

    def test_sl_hit(self):
        """Trade LONG : SL touché → perte avec taker fee + slippage."""
        candles = _make_candles([
            (100, 100.5, 99.5, 100),  # Entry
            (100, 100.2, 99.6, 99.7),  # SL touché (low < 99.75)
        ])
        config = _default_config()
        strategy = AlwaysLongStrategy(tp_pct=0.5, sl_pct=0.25)
        engine = BacktestEngine(config, strategy)
        result = engine.run({"5m": candles}, main_tf="5m")

        assert len(result.trades) >= 1
        trade = result.trades[0]
        assert trade.exit_reason == "sl"
        assert trade.slippage_cost > 0  # Slippage appliqué
        assert trade.net_pnl < 0

    def test_maker_vs_taker_fees(self):
        """TP = maker fee, SL = taker fee."""
        config = _default_config()

        # TP trade
        candles_tp = _make_candles([
            (100, 100.5, 99.5, 100),
            (100, 100.6, 99.8, 100.5),
        ])
        strategy_tp = AlwaysLongStrategy(tp_pct=0.5, sl_pct=0.25)
        result_tp = BacktestEngine(config, strategy_tp).run({"5m": candles_tp}, main_tf="5m")

        # SL trade
        candles_sl = _make_candles([
            (100, 100.5, 99.5, 100),
            (100, 100.2, 99.5, 99.7),
        ])
        strategy_sl = AlwaysLongStrategy(tp_pct=0.5, sl_pct=0.25)
        result_sl = BacktestEngine(config, strategy_sl).run({"5m": candles_sl}, main_tf="5m")

        if result_tp.trades and result_sl.trades:
            # SL trade devrait avoir des fees plus élevées (taker + taker vs taker + maker)
            tp_fees = result_tp.trades[0].fee_cost
            sl_fees = result_sl.trades[0].fee_cost
            assert sl_fees > tp_fees

    def test_ohlc_heuristic_green_candle_long(self):
        """Bougie verte + LONG : TP/SL les deux touchés → TP d'abord."""
        # Bougie verte (close > open) → high d'abord → LONG TP touché d'abord
        candles = _make_candles([
            (100, 100.5, 99.5, 100),  # Entry
            (100, 100.6, 99.5, 100.4),  # Verte, TP (100.5) et SL (99.75) tous deux touchés
        ])
        config = _default_config()
        strategy = AlwaysLongStrategy(tp_pct=0.5, sl_pct=0.25)
        engine = BacktestEngine(config, strategy)
        result = engine.run({"5m": candles}, main_tf="5m")

        if result.trades:
            # Bougie verte → TP d'abord pour LONG
            assert result.trades[0].exit_reason == "tp"

    def test_ohlc_heuristic_red_candle_long(self):
        """Bougie rouge + LONG : TP/SL les deux touchés → SL d'abord."""
        candles = _make_candles([
            (100, 100.5, 99.5, 100),  # Entry
            (100.2, 100.6, 99.5, 99.8),  # Rouge (close < open), TP et SL touchés
        ])
        config = _default_config()
        strategy = AlwaysLongStrategy(tp_pct=0.5, sl_pct=0.25)
        engine = BacktestEngine(config, strategy)
        result = engine.run({"5m": candles}, main_tf="5m")

        if result.trades:
            assert result.trades[0].exit_reason == "sl"

    def test_position_sizing_includes_sl_cost(self):
        """Le position sizing doit inclure taker_fee + slippage dans le coût SL."""
        config = _default_config()
        candles = _make_candles([
            (100, 100.5, 99.5, 100),
            (100, 100.5, 99.5, 100),  # Pas de TP/SL
            (100, 100.6, 99.5, 100.5),  # TP
        ])
        strategy = AlwaysLongStrategy(tp_pct=0.5, sl_pct=0.25)
        engine = BacktestEngine(config, strategy)
        result = engine.run({"5m": candles}, main_tf="5m")

        if result.trades:
            trade = result.trades[0]
            # Vérifier que la quantity est calculée avec le coût SL réel
            sl_distance_pct = 0.0025  # 0.25%
            sl_real_cost = sl_distance_pct + config.taker_fee + config.slippage_pct
            risk_amount = config.initial_capital * config.max_risk_per_trade
            expected_notional = risk_amount / sl_real_cost
            expected_qty = expected_notional / 100.0  # entry_price = 100
            assert trade.quantity == pytest.approx(expected_qty, rel=0.01)

    def test_equity_curve_per_candle(self):
        """L'equity curve doit avoir un point par bougie."""
        candles = _make_candles([
            (100, 100.5, 99.5, 100),
            (100, 100.2, 99.8, 100.1),
            (100, 100.6, 99.5, 100.5),  # TP
        ])
        config = _default_config()
        strategy = AlwaysLongStrategy(tp_pct=0.5, sl_pct=0.25)
        engine = BacktestEngine(config, strategy)
        result = engine.run({"5m": candles}, main_tf="5m")

        assert len(result.equity_curve) == len(candles)
        assert len(result.equity_timestamps) == len(candles)

    def test_strategy_params_in_result(self):
        """BacktestResult doit contenir les paramètres de la stratégie."""
        candles = _make_candles([(100, 101, 99, 100)])
        config = _default_config()
        strategy = AlwaysLongStrategy()
        engine = BacktestEngine(config, strategy)
        result = engine.run({"5m": candles}, main_tf="5m")

        assert "tp_pct" in result.strategy_params
        assert "sl_pct" in result.strategy_params

    def test_end_of_data_closure(self):
        """Les positions ouvertes sont fermées en fin de données."""
        candles = _make_candles([
            (100, 100.5, 99.5, 100),
            (100, 100.2, 99.8, 100.1),  # Pas de TP/SL
        ])
        config = _default_config()
        strategy = AlwaysLongStrategy(tp_pct=5.0, sl_pct=5.0)  # TP/SL lointains
        engine = BacktestEngine(config, strategy)
        result = engine.run({"5m": candles}, main_tf="5m")

        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "end_of_data"

    def test_no_data_raises(self):
        """Pas de données → ValueError."""
        config = _default_config()
        strategy = AlwaysLongStrategy()
        engine = BacktestEngine(config, strategy)
        with pytest.raises(ValueError, match="Pas de données"):
            engine.run({}, main_tf="5m")


# ─── Tests des métriques ─────────────────────────────────────────────────────


class TestMetrics:
    def _make_trades(self) -> list[TradeResult]:
        base_time = datetime(2024, 1, 15, tzinfo=timezone.utc)
        return [
            TradeResult(
                direction=Direction.LONG,
                entry_price=100, exit_price=100.5, quantity=1.0,
                entry_time=base_time, exit_time=base_time + timedelta(hours=1),
                gross_pnl=0.5, fee_cost=0.1, slippage_cost=0.02, net_pnl=0.38,
                exit_reason="tp", market_regime=MarketRegime.RANGING,
            ),
            TradeResult(
                direction=Direction.LONG,
                entry_price=100, exit_price=99.7, quantity=1.0,
                entry_time=base_time + timedelta(hours=2),
                exit_time=base_time + timedelta(hours=3),
                gross_pnl=-0.3, fee_cost=0.12, slippage_cost=0.03, net_pnl=-0.45,
                exit_reason="sl", market_regime=MarketRegime.RANGING,
            ),
            TradeResult(
                direction=Direction.LONG,
                entry_price=100, exit_price=100.3, quantity=1.0,
                entry_time=base_time + timedelta(hours=4),
                exit_time=base_time + timedelta(hours=5),
                gross_pnl=0.3, fee_cost=0.08, slippage_cost=0.0, net_pnl=0.22,
                exit_reason="signal_exit", market_regime=MarketRegime.TRENDING_UP,
            ),
        ]

    def test_win_rate(self):
        from backend.backtesting.engine import BacktestResult
        trades = self._make_trades()
        result = BacktestResult(
            config=_default_config(),
            strategy_name="test",
            strategy_params={},
            trades=trades,
            equity_curve=[10000, 10000.38, 9999.93, 10000.15],
            equity_timestamps=[
                datetime(2024, 1, 15, i, tzinfo=timezone.utc) for i in range(4)
            ],
            final_capital=10000.15,
        )
        metrics = calculate_metrics(result)
        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.win_rate == pytest.approx(66.67, rel=0.01)

    def test_fee_drag(self):
        from backend.backtesting.engine import BacktestResult
        trades = self._make_trades()
        result = BacktestResult(
            config=_default_config(),
            strategy_name="test",
            strategy_params={},
            trades=trades,
            equity_curve=[10000],
            equity_timestamps=[datetime(2024, 1, 15, tzinfo=timezone.utc)],
            final_capital=10000.15,
        )
        metrics = calculate_metrics(result)
        # Gross P&L = 0.5 + (-0.3) + 0.3 = 0.5
        # Total fees = 0.1 + 0.12 + 0.08 = 0.30
        # Gross wins = 0.5 + 0.3 = 0.8
        assert metrics.gross_pnl == pytest.approx(0.5)
        assert metrics.total_fees == pytest.approx(0.30)
        # Fee drag = fees / gross_wins = 0.30 / 0.8 * 100 = 37.5%
        assert metrics.fee_drag_pct == pytest.approx(37.5)

    def test_profit_factor_net_and_gross(self):
        from backend.backtesting.engine import BacktestResult
        trades = self._make_trades()
        result = BacktestResult(
            config=_default_config(),
            strategy_name="test",
            strategy_params={},
            trades=trades,
            equity_curve=[10000],
            equity_timestamps=[datetime(2024, 1, 15, tzinfo=timezone.utc)],
            final_capital=10000.15,
        )
        metrics = calculate_metrics(result)
        # Net wins = 0.38 + 0.22 = 0.60, net losses = 0.45
        assert metrics.profit_factor == pytest.approx(0.60 / 0.45, rel=0.01)
        # Gross wins = 0.5 + 0.3 = 0.8, gross losses = 0.3
        assert metrics.gross_profit_factor == pytest.approx(0.8 / 0.3, rel=0.01)

    def test_sortino_uses_downside_only(self):
        """Le Sortino ne compte que les rendements négatifs au dénominateur."""
        from backend.backtesting.engine import BacktestResult
        # Tous les trades gagnants → downside_std = 0 → Sortino = 0 (pas d'inf)
        trades = [
            TradeResult(
                direction=Direction.LONG,
                entry_price=100, exit_price=100.5, quantity=1.0,
                entry_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
                exit_time=datetime(2024, 1, 15, 1, tzinfo=timezone.utc),
                gross_pnl=0.5, fee_cost=0.05, slippage_cost=0.0, net_pnl=0.45,
                exit_reason="tp", market_regime=MarketRegime.RANGING,
            ),
        ]
        result = BacktestResult(
            config=_default_config(),
            strategy_name="test",
            strategy_params={},
            trades=trades,
            equity_curve=[10000, 10000.45],
            equity_timestamps=[
                datetime(2024, 1, 15, i, tzinfo=timezone.utc) for i in range(2)
            ],
            final_capital=10000.45,
        )
        metrics = calculate_metrics(result)
        # Avec un seul trade, pas assez pour Sharpe/Sortino
        # Mais au moins ça ne crashe pas
        assert not np.isinf(metrics.sortino_ratio)

    def test_regime_breakdown(self):
        from backend.backtesting.engine import BacktestResult
        trades = self._make_trades()
        result = BacktestResult(
            config=_default_config(),
            strategy_name="test",
            strategy_params={},
            trades=trades,
            equity_curve=[10000],
            equity_timestamps=[datetime(2024, 1, 15, tzinfo=timezone.utc)],
            final_capital=10000.15,
        )
        metrics = calculate_metrics(result)
        assert "RANGING" in metrics.regime_stats
        assert "TRENDING_UP" in metrics.regime_stats
        assert metrics.regime_stats["RANGING"]["trades"] == 2
        assert metrics.regime_stats["TRENDING_UP"]["trades"] == 1

    def test_format_table_contains_key_fields(self):
        from backend.backtesting.engine import BacktestResult
        trades = self._make_trades()
        result = BacktestResult(
            config=_default_config(),
            strategy_name="test",
            strategy_params={},
            trades=trades,
            equity_curve=[10000, 10000.38, 9999.93, 10000.15],
            equity_timestamps=[
                datetime(2024, 1, 15, i, tzinfo=timezone.utc) for i in range(4)
            ],
            final_capital=10000.15,
        )
        metrics = calculate_metrics(result)
        table = format_metrics_table(metrics, title="TEST")
        assert "Win rate" in table
        assert "Profit factor" in table
        assert "Fees" in table
        assert "Sharpe" in table
        assert "Sortino" in table
        assert "drawdown" in table.lower()

    def test_empty_trades(self):
        from backend.backtesting.engine import BacktestResult
        result = BacktestResult(
            config=_default_config(),
            strategy_name="test",
            strategy_params={},
            trades=[],
            equity_curve=[10000],
            equity_timestamps=[datetime(2024, 1, 15, tzinfo=timezone.utc)],
            final_capital=10000,
        )
        metrics = calculate_metrics(result)
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.net_pnl == 0.0
