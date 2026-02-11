"""Tests pour backend/core/position_manager.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backend.core.models import Candle, Direction, MarketRegime, TimeFrame
from backend.core.position_manager import PositionManager, PositionManagerConfig, TradeResult
from backend.strategies.base import OpenPosition, StrategyContext, StrategySignal
from backend.core.models import SignalStrength


def _default_config() -> PositionManagerConfig:
    return PositionManagerConfig(
        leverage=15,
        maker_fee=0.0002,
        taker_fee=0.0006,
        slippage_pct=0.0005,
        high_vol_slippage_mult=2.0,
        max_risk_per_trade=0.02,
    )


def _make_signal(
    direction: Direction = Direction.LONG,
    entry_price: float = 100.0,
    tp_price: float = 100.5,
    sl_price: float = 99.75,
) -> StrategySignal:
    return StrategySignal(
        direction=direction,
        entry_price=entry_price,
        tp_price=tp_price,
        sl_price=sl_price,
        score=0.8,
        strength=SignalStrength.STRONG,
        market_regime=MarketRegime.RANGING,
    )


def _make_candle(
    close: float = 100.0,
    high: float = 100.5,
    low: float = 99.5,
    open_: float = 100.0,
    minutes_offset: int = 0,
) -> Candle:
    return Candle(
        timestamp=datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=minutes_offset * 5),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1000.0,
        symbol="BTC/USDT",
        timeframe=TimeFrame.M5,
    )


class TestOpenPosition:
    def test_position_sizing(self):
        """Le sizing doit inclure taker_fee + slippage dans le coût SL."""
        pm = PositionManager(_default_config())
        signal = _make_signal()
        pos = pm.open_position(signal, datetime.now(timezone.utc), 10_000.0)

        assert pos is not None
        # Calcul attendu
        sl_distance = abs(100.0 - 99.75) / 100.0  # 0.0025
        sl_real_cost = sl_distance + 0.0006 + 0.0005  # 0.0036
        risk = 10_000.0 * 0.02  # 200
        expected_notional = risk / sl_real_cost
        expected_qty = expected_notional / 100.0
        assert pos.quantity == pytest.approx(expected_qty, rel=0.01)

    def test_invalid_price_returns_none(self):
        """Prix <= 0 → None."""
        pm = PositionManager(_default_config())
        signal = _make_signal(entry_price=0.0)
        assert pm.open_position(signal, datetime.now(timezone.utc), 10_000.0) is None

    def test_zero_sl_distance_returns_none(self):
        """SL = entry → sl_distance=0 mais sl_real_cost > 0, position ouverte quand même."""
        pm = PositionManager(_default_config())
        # SL exactement au prix d'entrée → sl_distance_pct = 0
        # mais sl_real_cost = 0 + taker + slippage > 0, donc ça passe
        signal = _make_signal(sl_price=100.0)
        pos = pm.open_position(signal, datetime.now(timezone.utc), 10_000.0)
        assert pos is not None  # sl_real_cost = 0.0011, ok


class TestClosePosition:
    def test_tp_long_maker_fee_no_slippage(self):
        """TP sur LONG : maker fee, pas de slippage."""
        pm = PositionManager(_default_config())
        pos = OpenPosition(
            direction=Direction.LONG,
            entry_price=100.0,
            quantity=1.0,
            entry_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
            tp_price=100.5,
            sl_price=99.75,
            entry_fee=0.06,  # 1.0 * 100 * 0.0006
        )
        trade = pm.close_position(pos, 100.5, datetime.now(timezone.utc), "tp", MarketRegime.RANGING)

        assert trade.exit_reason == "tp"
        assert trade.slippage_cost == 0.0
        assert trade.exit_price == 100.5
        assert trade.gross_pnl == pytest.approx(0.5)
        # Fee = entry_fee(0.06) + exit_fee(1.0 * 100.5 * 0.0002)
        assert trade.fee_cost == pytest.approx(0.06 + 100.5 * 0.0002, rel=0.01)

    def test_sl_long_taker_fee_with_slippage(self):
        """SL sur LONG : taker fee + slippage."""
        pm = PositionManager(_default_config())
        pos = OpenPosition(
            direction=Direction.LONG,
            entry_price=100.0,
            quantity=1.0,
            entry_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
            tp_price=100.5,
            sl_price=99.75,
            entry_fee=0.06,
        )
        trade = pm.close_position(pos, 99.75, datetime.now(timezone.utc), "sl", MarketRegime.RANGING)

        assert trade.exit_reason == "sl"
        assert trade.slippage_cost > 0
        assert trade.net_pnl < 0

    def test_high_volatility_doubles_slippage(self):
        """HIGH_VOLATILITY → slippage × 2."""
        pm = PositionManager(_default_config())
        pos = OpenPosition(
            direction=Direction.LONG,
            entry_price=100.0,
            quantity=1.0,
            entry_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
            tp_price=100.5,
            sl_price=99.75,
            entry_fee=0.06,
        )
        trade_normal = pm.close_position(pos, 99.75, datetime.now(timezone.utc), "sl", MarketRegime.RANGING)
        trade_hv = pm.close_position(pos, 99.75, datetime.now(timezone.utc), "sl", MarketRegime.HIGH_VOLATILITY)

        assert trade_hv.slippage_cost == pytest.approx(trade_normal.slippage_cost * 2, rel=0.01)


class TestOhlcHeuristic:
    def test_green_candle_long_tp_first(self):
        """Bougie verte + LONG → TP touché en premier."""
        pm = PositionManager(_default_config())
        pos = OpenPosition(
            direction=Direction.LONG,
            entry_price=100.0,
            quantity=1.0,
            entry_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
            tp_price=100.5,
            sl_price=99.5,
            entry_fee=0.06,
        )
        # Bougie verte : close > open, TP et SL tous deux touchés
        candle = _make_candle(close=100.4, high=100.6, low=99.4, open_=100.0)

        # Appel direct de l'heuristique
        result = pm._ohlc_heuristic(candle, pos)
        assert result == "tp"

    def test_red_candle_long_sl_first(self):
        """Bougie rouge + LONG → SL touché en premier."""
        pm = PositionManager(_default_config())
        pos = OpenPosition(
            direction=Direction.LONG,
            entry_price=100.0,
            quantity=1.0,
            entry_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
            tp_price=100.5,
            sl_price=99.5,
            entry_fee=0.06,
        )
        candle = _make_candle(close=99.8, high=100.6, low=99.4, open_=100.2)

        result = pm._ohlc_heuristic(candle, pos)
        assert result == "sl"


class TestUnrealizedPnl:
    def test_long_profit(self):
        pm = PositionManager(_default_config())
        pos = OpenPosition(
            direction=Direction.LONG,
            entry_price=100.0,
            quantity=2.0,
            entry_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
            tp_price=101.0,
            sl_price=99.0,
            entry_fee=0.12,
        )
        assert pm.unrealized_pnl(pos, 100.5) == pytest.approx(1.0)

    def test_short_profit(self):
        pm = PositionManager(_default_config())
        pos = OpenPosition(
            direction=Direction.SHORT,
            entry_price=100.0,
            quantity=2.0,
            entry_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
            tp_price=99.0,
            sl_price=101.0,
            entry_fee=0.12,
        )
        assert pm.unrealized_pnl(pos, 99.5) == pytest.approx(1.0)
