"""Tests des modèles Pydantic."""

from datetime import datetime, timezone

import pytest

from backend.core.models import (
    Candle,
    Direction,
    MarketRegime,
    MultiTimeframeData,
    OrderBookLevel,
    OrderBookSnapshot,
    SessionState,
    Signal,
    SignalStrength,
    TimeFrame,
    Trade,
)


class TestTimeFrame:
    def test_from_string(self):
        assert TimeFrame.from_string("1m") == TimeFrame.M1
        assert TimeFrame.from_string("5m") == TimeFrame.M5
        assert TimeFrame.from_string("15m") == TimeFrame.M15
        assert TimeFrame.from_string("1h") == TimeFrame.H1

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="TimeFrame inconnu"):
            TimeFrame.from_string("2h")

    def test_to_milliseconds(self):
        assert TimeFrame.M1.to_milliseconds() == 60_000
        assert TimeFrame.M5.to_milliseconds() == 300_000
        assert TimeFrame.M15.to_milliseconds() == 900_000
        assert TimeFrame.H1.to_milliseconds() == 3_600_000

    def test_value(self):
        assert TimeFrame.M1.value == "1m"
        assert TimeFrame.H1.value == "1h"


class TestCandle:
    def test_valid_candle(self):
        c = Candle(
            timestamp=datetime.now(tz=timezone.utc),
            open=100.0,
            high=105.0,
            low=98.0,
            close=103.0,
            volume=1000.0,
            symbol="BTC/USDT",
            timeframe=TimeFrame.M5,
        )
        assert c.symbol == "BTC/USDT"
        assert c.vwap is None
        assert c.mark_price is None

    def test_candle_with_optional_fields(self):
        c = Candle(
            timestamp=datetime.now(tz=timezone.utc),
            open=100.0,
            high=105.0,
            low=98.0,
            close=103.0,
            volume=1000.0,
            symbol="BTC/USDT",
            timeframe=TimeFrame.M5,
            vwap=101.5,
            mark_price=102.0,
        )
        assert c.vwap == 101.5
        assert c.mark_price == 102.0

    def test_candle_low_gt_high_fails(self):
        with pytest.raises(ValueError, match="low"):
            Candle(
                timestamp=datetime.now(tz=timezone.utc),
                open=100.0,
                high=95.0,
                low=105.0,
                close=100.0,
                volume=100.0,
                symbol="BTC/USDT",
                timeframe=TimeFrame.M1,
            )

    def test_candle_high_lt_open_fails(self):
        with pytest.raises(ValueError, match="high"):
            Candle(
                timestamp=datetime.now(tz=timezone.utc),
                open=100.0,
                high=99.0,
                low=95.0,
                close=98.0,
                volume=100.0,
                symbol="BTC/USDT",
                timeframe=TimeFrame.M1,
            )

    def test_candle_negative_volume_fails(self):
        with pytest.raises(ValueError):
            Candle(
                timestamp=datetime.now(tz=timezone.utc),
                open=100.0,
                high=105.0,
                low=98.0,
                close=103.0,
                volume=-10.0,
                symbol="BTC/USDT",
                timeframe=TimeFrame.M1,
            )


class TestOrderBook:
    def test_spread_and_mid_price(self):
        ob = OrderBookSnapshot(
            timestamp=datetime.now(tz=timezone.utc),
            symbol="BTC/USDT",
            bids=[OrderBookLevel(price=99.0, quantity=10.0)],
            asks=[OrderBookLevel(price=101.0, quantity=5.0)],
        )
        assert ob.spread == 2.0
        assert ob.mid_price == 100.0

    def test_empty_orderbook(self):
        ob = OrderBookSnapshot(
            timestamp=datetime.now(tz=timezone.utc),
            symbol="BTC/USDT",
        )
        assert ob.spread == 0.0
        assert ob.mid_price == 0.0


class TestTrade:
    def test_valid_trade(self):
        t = Trade(
            id="t1",
            symbol="BTC/USDT",
            direction=Direction.LONG,
            entry_price=100.0,
            exit_price=100.5,
            quantity=1.0,
            leverage=20,
            gross_pnl=10.0,
            fee_cost=1.2,
            slippage_cost=0.5,
            net_pnl=8.3,
            entry_time=datetime.now(tz=timezone.utc),
            exit_time=datetime.now(tz=timezone.utc),
            strategy_name="vwap_rsi",
        )
        assert t.net_pnl == 8.3

    def test_net_pnl_mismatch_fails(self):
        with pytest.raises(ValueError, match="net_pnl"):
            Trade(
                id="t2",
                symbol="BTC/USDT",
                direction=Direction.SHORT,
                entry_price=100.0,
                exit_price=99.5,
                quantity=1.0,
                leverage=20,
                gross_pnl=10.0,
                fee_cost=1.2,
                slippage_cost=0.5,
                net_pnl=5.0,  # Devrait être 8.3
                entry_time=datetime.now(tz=timezone.utc),
                exit_time=datetime.now(tz=timezone.utc),
                strategy_name="vwap_rsi",
            )


class TestSessionState:
    def test_win_rate(self):
        s = SessionState(
            start_time=datetime.now(tz=timezone.utc),
            total_trades=10,
            wins=6,
            losses=4,
        )
        assert s.win_rate == 0.6

    def test_win_rate_no_trades(self):
        s = SessionState(start_time=datetime.now(tz=timezone.utc))
        assert s.win_rate == 0.0


class TestSignal:
    def test_valid_signal(self):
        sig = Signal(
            timestamp=datetime.now(tz=timezone.utc),
            strategy_name="vwap_rsi",
            symbol="BTC/USDT",
            direction=Direction.LONG,
            strength=SignalStrength.STRONG,
            score=0.85,
            entry_price=97500.0,
            tp_price=97987.5,
            sl_price=97256.25,
            market_regime=MarketRegime.RANGING,
            signals_detail={"vwap": 0.8, "rsi": 0.9},
        )
        assert sig.score == 0.85

    def test_score_out_of_range(self):
        with pytest.raises(ValueError):
            Signal(
                timestamp=datetime.now(tz=timezone.utc),
                strategy_name="test",
                symbol="BTC/USDT",
                direction=Direction.LONG,
                strength=SignalStrength.WEAK,
                score=1.5,
                entry_price=100.0,
            )
