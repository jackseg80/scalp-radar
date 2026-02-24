"""Tests Sprint 45 — Live trades DB persistence."""

from __future__ import annotations

import pytest
import pytest_asyncio

from backend.core.database import Database


@pytest_asyncio.fixture
async def db():
    """DB en mémoire initialisée avec toutes les tables."""
    database = Database(db_path=":memory:")
    await database.init()
    yield database
    await database.close()


def _make_live_trade(**overrides) -> dict:
    """Helper pour créer un live trade."""
    base = {
        "timestamp": "2026-02-24T10:00:00+00:00",
        "strategy_name": "grid_atr",
        "symbol": "BTC/USDT:USDT",
        "direction": "LONG",
        "trade_type": "entry",
        "side": "buy",
        "quantity": 0.01,
        "price": 95000.0,
        "order_id": "order_123",
        "fee": 0.57,
        "pnl": None,
        "pnl_pct": None,
        "leverage": 7,
        "grid_level": 0,
        "context": "grid",
    }
    base.update(overrides)
    return base


# ─── Insert & Read ──────────────────────────────────────────────────────


class TestInsertLiveTrade:
    @pytest.mark.asyncio
    async def test_insert_and_read_roundtrip(self, db):
        """Insert un trade et le relire."""
        trade = _make_live_trade()
        trade_id = await db.insert_live_trade(trade)
        assert trade_id > 0

        trades = await db.get_live_trades(limit=10)
        assert len(trades) == 1
        assert trades[0]["symbol"] == "BTC/USDT:USDT"
        assert trades[0]["strategy_name"] == "grid_atr"
        assert trades[0]["price"] == 95000.0

    @pytest.mark.asyncio
    async def test_insert_close_with_pnl(self, db):
        """Insert un close avec P&L."""
        trade = _make_live_trade(
            trade_type="tp_close",
            side="sell",
            price=96000.0,
            pnl=8.5,
            pnl_pct=1.2,
        )
        await db.insert_live_trade(trade)

        trades = await db.get_live_trades()
        assert trades[0]["pnl"] == 8.5
        assert trades[0]["pnl_pct"] == 1.2

    @pytest.mark.asyncio
    async def test_trade_count(self, db):
        """get_live_trade_count retourne le bon nombre."""
        assert await db.get_live_trade_count() == 0
        await db.insert_live_trade(_make_live_trade())
        assert await db.get_live_trade_count() == 1
        await db.insert_live_trade(_make_live_trade(order_id="order_456"))
        assert await db.get_live_trade_count() == 2


# ─── Period Filter ──────────────────────────────────────────────────────


class TestLiveTradesPeriodFilter:
    @pytest.mark.asyncio
    async def test_filter_by_period_today(self, db):
        """Seuls les trades d'aujourd'hui sont retournés."""
        # Trade ancien
        await db.insert_live_trade(_make_live_trade(
            timestamp="2025-01-01T10:00:00+00:00",
        ))
        # Trade récent (sera dans "all" mais pas "today")
        await db.insert_live_trade(_make_live_trade(
            timestamp="2026-02-24T10:00:00+00:00",
            order_id="order_today",
        ))

        all_trades = await db.get_live_trades(period="all")
        assert len(all_trades) == 2

    @pytest.mark.asyncio
    async def test_filter_by_strategy(self, db):
        """Filtre par stratégie."""
        await db.insert_live_trade(_make_live_trade(strategy_name="grid_atr"))
        await db.insert_live_trade(_make_live_trade(
            strategy_name="grid_multi_tf", order_id="order_mt",
        ))

        atr = await db.get_live_trades(strategy="grid_atr")
        assert len(atr) == 1
        assert atr[0]["strategy_name"] == "grid_atr"

    @pytest.mark.asyncio
    async def test_filter_by_symbol(self, db):
        """Filtre par symbol."""
        await db.insert_live_trade(_make_live_trade(symbol="BTC/USDT:USDT"))
        await db.insert_live_trade(_make_live_trade(
            symbol="ETH/USDT:USDT", order_id="order_eth",
        ))

        btc = await db.get_live_trades(symbol="BTC/USDT:USDT")
        assert len(btc) == 1


# ─── Stats ──────────────────────────────────────────────────────────────


class TestLiveStats:
    @pytest.mark.asyncio
    async def test_stats_empty(self, db):
        """Stats vides retournent les bons defaults."""
        stats = await db.get_live_stats()
        assert stats["total_trades"] == 0
        assert stats["total_pnl"] == 0.0
        assert stats["win_rate"] == 0.0
        assert stats["profit_factor"] == 0.0
        assert stats["best_trade"] is None

    @pytest.mark.asyncio
    async def test_stats_pnl_and_winrate(self, db):
        """P&L total et win rate corrects."""
        # 2 wins + 1 loss
        await db.insert_live_trade(_make_live_trade(
            trade_type="tp_close", pnl=10.0, order_id="o1",
            timestamp="2026-02-24T10:00:00+00:00",
        ))
        await db.insert_live_trade(_make_live_trade(
            trade_type="tp_close", pnl=5.0, order_id="o2",
            timestamp="2026-02-24T11:00:00+00:00",
        ))
        await db.insert_live_trade(_make_live_trade(
            trade_type="sl_close", pnl=-3.0, order_id="o3",
            timestamp="2026-02-24T12:00:00+00:00",
        ))

        stats = await db.get_live_stats()
        assert stats["total_trades"] == 3
        assert stats["total_pnl"] == 12.0
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["win_rate"] == 66.7
        assert stats["gross_profit"] == 15.0
        assert stats["gross_loss"] == -3.0
        assert stats["profit_factor"] == 5.0

    @pytest.mark.asyncio
    async def test_stats_best_worst(self, db):
        """Best et worst trade."""
        await db.insert_live_trade(_make_live_trade(
            trade_type="tp_close", pnl=20.0, order_id="best",
            symbol="ETH/USDT:USDT",
            timestamp="2026-02-24T10:00:00+00:00",
        ))
        await db.insert_live_trade(_make_live_trade(
            trade_type="sl_close", pnl=-8.0, order_id="worst",
            symbol="BTC/USDT:USDT",
            timestamp="2026-02-24T11:00:00+00:00",
        ))

        stats = await db.get_live_stats()
        assert stats["best_trade"]["pnl"] == 20.0
        assert stats["best_trade"]["symbol"] == "ETH/USDT:USDT"
        assert stats["worst_trade"]["pnl"] == -8.0

    @pytest.mark.asyncio
    async def test_stats_streak(self, db):
        """Streak calculée correctement."""
        for i, pnl in enumerate([5.0, -2.0, -1.0, -3.0]):
            await db.insert_live_trade(_make_live_trade(
                trade_type="tp_close" if pnl > 0 else "sl_close",
                pnl=pnl, order_id=f"streak_{i}",
                timestamp=f"2026-02-24T{10+i:02d}:00:00+00:00",
            ))

        stats = await db.get_live_stats()
        assert stats["current_streak"]["type"] == "loss"
        assert stats["current_streak"]["count"] == 3

    @pytest.mark.asyncio
    async def test_stats_ignores_entries(self, db):
        """Les entries (pnl=None) ne sont pas comptées dans les stats."""
        await db.insert_live_trade(_make_live_trade(
            trade_type="entry", pnl=None, order_id="entry1",
        ))
        await db.insert_live_trade(_make_live_trade(
            trade_type="tp_close", pnl=5.0, order_id="close1",
            timestamp="2026-02-24T10:00:00+00:00",
        ))

        stats = await db.get_live_stats()
        assert stats["total_trades"] == 1  # Seulement le close


# ─── Daily PnL ──────────────────────────────────────────────────────────


class TestLiveDailyPnl:
    @pytest.mark.asyncio
    async def test_daily_aggregation(self, db):
        """P&L groupé par jour."""
        await db.insert_live_trade(_make_live_trade(
            trade_type="tp_close", pnl=10.0, order_id="d1",
            timestamp="2026-02-23T10:00:00+00:00",
        ))
        await db.insert_live_trade(_make_live_trade(
            trade_type="sl_close", pnl=-3.0, order_id="d2",
            timestamp="2026-02-23T14:00:00+00:00",
        ))
        await db.insert_live_trade(_make_live_trade(
            trade_type="tp_close", pnl=5.0, order_id="d3",
            timestamp="2026-02-24T10:00:00+00:00",
        ))

        daily = await db.get_live_daily_pnl(days=30)
        assert len(daily) == 2
        # Premier jour : 10 - 3 = 7
        assert daily[0]["pnl"] == 7.0
        assert daily[0]["trades"] == 2
        # Deuxième jour : 5
        assert daily[1]["pnl"] == 5.0


# ─── Per Asset Stats ────────────────────────────────────────────────────


class TestLivePerAssetStats:
    @pytest.mark.asyncio
    async def test_per_asset_stats(self, db):
        """Stats par asset."""
        await db.insert_live_trade(_make_live_trade(
            trade_type="tp_close", pnl=10.0, order_id="a1",
            symbol="BTC/USDT:USDT",
            timestamp="2026-02-24T10:00:00+00:00",
        ))
        await db.insert_live_trade(_make_live_trade(
            trade_type="sl_close", pnl=-2.0, order_id="a2",
            symbol="BTC/USDT:USDT",
            timestamp="2026-02-24T11:00:00+00:00",
        ))
        await db.insert_live_trade(_make_live_trade(
            trade_type="tp_close", pnl=5.0, order_id="a3",
            symbol="ETH/USDT:USDT",
            timestamp="2026-02-24T12:00:00+00:00",
        ))

        per_asset = await db.get_live_per_asset_stats()
        assert len(per_asset) == 2

        btc = next(a for a in per_asset if a["symbol"] == "BTC/USDT:USDT")
        assert btc["total_trades"] == 2
        assert btc["total_pnl"] == 8.0
        assert btc["wins"] == 1
        assert btc["win_rate"] == 50.0


# ─── Purge ─────────────────────────────────────────────────────────────


class TestPurgeLiveTrades:
    @pytest.mark.asyncio
    async def test_purge_all(self, db):
        """Purge tous les trades."""
        await db.insert_live_trade(_make_live_trade(order_id="p1"))
        await db.insert_live_trade(_make_live_trade(order_id="p2"))
        assert await db.get_live_trade_count() == 2

        deleted = await db.purge_live_trades()
        assert deleted == 2
        assert await db.get_live_trade_count() == 0

    @pytest.mark.asyncio
    async def test_purge_by_context(self, db):
        """Purge seulement les trades d'un contexte donné."""
        await db.insert_live_trade(_make_live_trade(
            order_id="sync1", context="sync_bitget_trades",
        ))
        await db.insert_live_trade(_make_live_trade(
            order_id="live1", context="grid",
        ))
        assert await db.get_live_trade_count() == 2

        deleted = await db.purge_live_trades(context="sync_bitget_trades")
        assert deleted == 1
        assert await db.get_live_trade_count() == 1

        remaining = await db.get_live_trades()
        assert remaining[0]["context"] == "grid"


# ─── Classify & P&L (sync script) ─────────────────────────────────────


class TestClassifyAndPnl:
    def test_classify_long_only_buy_is_entry(self):
        """grid_atr LONG-only : buy = entry."""
        from scripts.sync_bitget_trades import _classify_trade
        trade = {"side": "buy", "info": {}}
        direction, trade_type = _classify_trade(trade, "grid_atr")
        assert direction == "LONG"
        assert trade_type == "entry"

    def test_classify_long_only_sell_is_close(self):
        """grid_atr LONG-only : sell = close."""
        from scripts.sync_bitget_trades import _classify_trade
        trade = {"side": "sell", "info": {}}
        direction, trade_type = _classify_trade(trade, "grid_atr")
        assert direction == "LONG"
        assert trade_type == "close"

    def test_classify_tradeside_close(self):
        """tradeSide=close dans info → close."""
        from scripts.sync_bitget_trades import _classify_trade
        trade = {"side": "buy", "info": {"tradeSide": "close"}}
        direction, trade_type = _classify_trade(trade, "grid_multi_tf")
        assert direction == "SHORT"
        assert trade_type == "close"

    def test_classify_reduce_only(self):
        """reduceOnly=true → close."""
        from scripts.sync_bitget_trades import _classify_trade
        trade = {"side": "sell", "info": {"reduceOnly": True}}
        direction, trade_type = _classify_trade(trade, "grid_multi_tf")
        assert direction == "LONG"
        assert trade_type == "close"

    def test_classify_multi_tf_sell_entry_short(self):
        """grid_multi_tf : sell sans reduceOnly = entry SHORT."""
        from scripts.sync_bitget_trades import _classify_trade
        trade = {"side": "sell", "info": {}}
        direction, trade_type = _classify_trade(trade, "grid_multi_tf")
        assert direction == "SHORT"
        assert trade_type == "entry"

    def test_aggregate_fills_single_fill(self):
        """Un seul fill → retourné tel quel."""
        from scripts.sync_bitget_trades import _aggregate_fills_by_order
        fills = [{"order": "A", "amount": 0.01, "price": 100.0, "side": "buy", "info": {}, "fee": {"cost": 0.5}}]
        result = _aggregate_fills_by_order(fills)
        assert len(result) == 1
        assert result[0]["amount"] == 0.01
        assert result[0]["price"] == 100.0

    def test_aggregate_fills_multiple_fills_same_order(self):
        """3 fills du même ordre → 1 enregistrement agrégé (VWAP + qty totale)."""
        from scripts.sync_bitget_trades import _aggregate_fills_by_order
        fills = [
            {"order": "D", "amount": 0.01, "price": 98.0, "side": "sell", "info": {"tradeSide": "close"}, "fee": {"cost": 0.55}},
            {"order": "D", "amount": 0.01, "price": 98.0, "side": "sell", "info": {"tradeSide": "close"}, "fee": {"cost": 0.55}},
            {"order": "D", "amount": 0.01, "price": 98.0, "side": "sell", "info": {"tradeSide": "close"}, "fee": {"cost": 0.55}},
        ]
        result = _aggregate_fills_by_order(fills)
        assert len(result) == 1
        assert round(result[0]["amount"], 6) == 0.03
        assert round(result[0]["price"], 4) == 98.0
        assert round(result[0]["fee"]["cost"], 4) == 1.65

    def test_aggregate_fills_vwap_different_prices(self):
        """VWAP correct quand fills à prix différents."""
        from scripts.sync_bitget_trades import _aggregate_fills_by_order
        fills = [
            {"order": "D", "amount": 0.02, "price": 96.0, "side": "sell", "info": {}, "fee": {"cost": 1.0}},
            {"order": "D", "amount": 0.01, "price": 99.0, "side": "sell", "info": {}, "fee": {"cost": 0.5}},
        ]
        result = _aggregate_fills_by_order(fills)
        assert len(result) == 1
        # VWAP = (0.02*96 + 0.01*99) / 0.03 = (1.92 + 0.99) / 0.03 = 97.0
        assert round(result[0]["price"], 4) == 97.0
        assert round(result[0]["amount"], 6) == 0.03

    def test_aggregate_fills_different_orders(self):
        """Fills de 2 ordres différents → 2 enregistrements séparés."""
        from scripts.sync_bitget_trades import _aggregate_fills_by_order
        fills = [
            {"order": "A", "amount": 0.01, "price": 100.0, "side": "buy", "info": {}, "fee": {"cost": 0.5}},
            {"order": "B", "amount": 0.01, "price": 95.0, "side": "buy", "info": {}, "fee": {"cost": 0.5}},
        ]
        result = _aggregate_fills_by_order(fills)
        assert len(result) == 2

    def test_aggregate_fills_output_fields(self):
        """Après agrégation, les champs amount/price/fee sont corrects pour C1."""
        from scripts.sync_bitget_trades import _aggregate_fills_by_order
        fills = [
            {"order": "E1", "amount": 0.01, "price": 100.0, "side": "buy",  "info": {}, "fee": {"cost": 0}, "datetime": "2026-02-24T10:00:00+00:00"},
            {"order": "E2", "amount": 0.01, "price": 95.0,  "side": "buy",  "info": {}, "fee": {"cost": 0}, "datetime": "2026-02-24T10:10:00+00:00"},
            {"order": "E3", "amount": 0.01, "price": 90.0,  "side": "buy",  "info": {}, "fee": {"cost": 0}, "datetime": "2026-02-24T10:20:00+00:00"},
            {"order": "C1", "amount": 0.01, "price": 98.0,  "side": "sell", "info": {"tradeSide": "close"}, "fee": {"cost": 0.18}, "datetime": "2026-02-24T11:00:00+00:00"},
            {"order": "C1", "amount": 0.01, "price": 98.0,  "side": "sell", "info": {"tradeSide": "close"}, "fee": {"cost": 0.18}, "datetime": "2026-02-24T11:00:00+00:00"},
            {"order": "C1", "amount": 0.01, "price": 98.0,  "side": "sell", "info": {"tradeSide": "close"}, "fee": {"cost": 0.18}, "datetime": "2026-02-24T11:00:00+00:00"},
        ]
        aggregated = _aggregate_fills_by_order(fills)
        assert len(aggregated) == 4  # E1, E2, E3, C1

        # C1 doit avoir qty=0.03, vwap=98.0, fee=0.54
        c1 = next(a for a in aggregated if a["order"] == "C1")
        assert c1["amount"] == pytest.approx(0.03)
        assert c1["price"] == pytest.approx(98.0)
        assert c1["fee"]["cost"] == pytest.approx(0.54)

    def test_pnl_fifo_grid_cycle_wins(self):
        """Scénario grid complet : 3 entries + 1 close agrégé → 1 cycle_close WIN.

        avg_entry = (100+95+90)/3 = 95.0
        pnl = (98-95)*3 - 0.54 = 9 - 0.54 = 8.46
        """
        from scripts.sync_bitget_trades import _group_into_cycles
        records = [
            {"timestamp": "2026-02-24T10:00:00+00:00", "symbol": "BTC/USDT:USDT",
             "direction": "LONG", "trade_type": "entry",  "price": 100.0, "quantity": 1.0,
             "fee": 0, "leverage": 3, "strategy_name": "grid_atr", "order_id": "E1", "side": "buy"},
            {"timestamp": "2026-02-24T10:10:00+00:00", "symbol": "BTC/USDT:USDT",
             "direction": "LONG", "trade_type": "entry",  "price": 95.0,  "quantity": 1.0,
             "fee": 0, "leverage": 3, "strategy_name": "grid_atr", "order_id": "E2", "side": "buy"},
            {"timestamp": "2026-02-24T10:20:00+00:00", "symbol": "BTC/USDT:USDT",
             "direction": "LONG", "trade_type": "entry",  "price": 90.0,  "quantity": 1.0,
             "fee": 0, "leverage": 3, "strategy_name": "grid_atr", "order_id": "E3", "side": "buy"},
            # Close agrégé : qty=3.0, fee=0.54
            {"timestamp": "2026-02-24T11:00:00+00:00", "symbol": "BTC/USDT:USDT",
             "direction": "LONG", "trade_type": "close",  "price": 98.0,  "quantity": 3.0,
             "fee": 0.54, "leverage": 3, "strategy_name": "grid_atr", "order_id": "C1", "side": "sell"},
        ]
        result = _group_into_cycles(records, leverage=3)
        cycle_closes = [r for r in result if r["trade_type"] == "cycle_close"]
        entries = [r for r in result if r["trade_type"] == "entry"]

        assert len(cycle_closes) == 1
        assert len(entries) == 3

        cc = cycle_closes[0]
        # avg_entry = (100+95+90)/3 = 95.0, avg_close = 98.0
        # pnl = (98-95)*3 - 0.54 = 8.46
        assert cc["pnl"] == pytest.approx(8.46, abs=0.001)
        assert cc["pnl"] > 0
        assert cc["grid_level"] == 3  # 3 entries dans ce cycle


# ─── Group Into Cycles ──────────────────────────────────────────────────


def _make_record(**overrides) -> dict:
    """Helper pour créer un record classifié destiné à _group_into_cycles."""
    base = {
        "timestamp": "2026-02-24T10:00:00+00:00",
        "strategy_name": "grid_atr",
        "symbol": "BTC/USDT:USDT",
        "direction": "LONG",
        "trade_type": "entry",
        "side": "buy",
        "quantity": 1.0,
        "price": 100.0,
        "order_id": "E1",
        "fee": 0.0,
        "leverage": 3,
    }
    base.update(overrides)
    return base


class TestGroupIntoCycles:
    def test_simple_cycle_long(self):
        """1 entry + 1 close → 1 cycle_close + 1 entry dans le résultat."""
        from scripts.sync_bitget_trades import _group_into_cycles
        records = [
            _make_record(trade_type="entry", price=100.0, quantity=1.0, order_id="E1"),
            _make_record(trade_type="close", price=105.0, quantity=1.0, order_id="C1",
                         side="sell", timestamp="2026-02-24T11:00:00+00:00"),
        ]
        result = _group_into_cycles(records, leverage=3)
        cycle_closes = [r for r in result if r["trade_type"] == "cycle_close"]
        entries = [r for r in result if r["trade_type"] == "entry"]

        assert len(cycle_closes) == 1
        assert len(entries) == 1
        cc = cycle_closes[0]
        # pnl = (105 - 100) * 1 - 0 = 5.0
        assert cc["pnl"] == pytest.approx(5.0, abs=0.001)
        assert cc["grid_level"] == 1

    def test_simple_cycle_short(self):
        """1 entry SHORT + 1 close → cycle_close avec pnl correct."""
        from scripts.sync_bitget_trades import _group_into_cycles
        records = [
            _make_record(direction="SHORT", trade_type="entry", price=3000.0,
                         quantity=1.0, order_id="E1", side="sell"),
            _make_record(direction="SHORT", trade_type="close", price=2900.0,
                         quantity=1.0, order_id="C1", side="buy", fee=1.8,
                         timestamp="2026-02-24T11:00:00+00:00"),
        ]
        result = _group_into_cycles(records, leverage=3)
        cycle_closes = [r for r in result if r["trade_type"] == "cycle_close"]

        assert len(cycle_closes) == 1
        cc = cycle_closes[0]
        # pnl = (3000 - 2900) * 1 - 1.8 = 98.2
        assert cc["pnl"] == pytest.approx(98.2, abs=0.001)
        assert cc["direction"] == "SHORT"

    def test_multi_entry_single_close(self):
        """3 entries + 1 close global → 1 cycle_close avec avg pondéré."""
        from scripts.sync_bitget_trades import _group_into_cycles
        records = [
            _make_record(price=100.0, quantity=0.01, order_id="E1",
                         timestamp="2026-02-24T10:00:00+00:00"),
            _make_record(price=95.0, quantity=0.01, order_id="E2",
                         timestamp="2026-02-24T10:10:00+00:00"),
            _make_record(price=90.0, quantity=0.01, order_id="E3",
                         timestamp="2026-02-24T10:20:00+00:00"),
            _make_record(trade_type="close", price=97.0, quantity=0.03, order_id="C1",
                         side="sell", fee=0.2, timestamp="2026-02-24T11:00:00+00:00"),
        ]
        result = _group_into_cycles(records, leverage=3)
        cycle_closes = [r for r in result if r["trade_type"] == "cycle_close"]
        entries = [r for r in result if r["trade_type"] == "entry"]

        assert len(cycle_closes) == 1
        assert len(entries) == 3
        cc = cycle_closes[0]
        assert cc["grid_level"] == 3
        # avg_entry = (100+95+90)/3 = 95.0
        # pnl = (97 - 95) * 0.03 - 0.2 = 0.06 - 0.2 = -0.14
        assert cc["pnl"] == pytest.approx(-0.14, abs=0.001)

    def test_multi_close_until_position_zero(self):
        """2 entries + 2 closes parttiels → 1 seul cycle_close."""
        from scripts.sync_bitget_trades import _group_into_cycles
        records = [
            _make_record(price=100.0, quantity=1.0, order_id="E1",
                         timestamp="2026-02-24T10:00:00+00:00"),
            _make_record(price=98.0, quantity=1.0, order_id="E2",
                         timestamp="2026-02-24T10:10:00+00:00"),
            # Premier close : position passe de 2 à 1 (pas encore à 0)
            _make_record(trade_type="close", price=103.0, quantity=1.0, order_id="C1",
                         side="sell", timestamp="2026-02-24T11:00:00+00:00"),
            # Deuxième close : position passe à 0 → cycle complet
            _make_record(trade_type="close", price=105.0, quantity=1.0, order_id="C2",
                         side="sell", timestamp="2026-02-24T12:00:00+00:00"),
        ]
        result = _group_into_cycles(records, leverage=3)
        cycle_closes = [r for r in result if r["trade_type"] == "cycle_close"]
        assert len(cycle_closes) == 1

        cc = cycle_closes[0]
        # avg_entry = (100+98)/2 = 99.0, avg_close = (103+105)/2 = 104.0
        # pnl = (104 - 99) * 2 = 10.0
        assert cc["pnl"] == pytest.approx(10.0, abs=0.001)
        assert cc["grid_level"] == 2  # 2 entries

    def test_open_cycle_no_cycle_close(self):
        """Cycle non clôturé → aucun cycle_close, entries conservées."""
        from scripts.sync_bitget_trades import _group_into_cycles
        records = [
            _make_record(price=100.0, quantity=1.0, order_id="E1"),
            _make_record(price=95.0, quantity=1.0, order_id="E2",
                         timestamp="2026-02-24T10:10:00+00:00"),
        ]
        result = _group_into_cycles(records, leverage=3)
        cycle_closes = [r for r in result if r["trade_type"] == "cycle_close"]
        entries = [r for r in result if r["trade_type"] == "entry"]

        assert len(cycle_closes) == 0
        assert len(entries) == 2

    def test_orphan_close_ignored(self):
        """Close sans entry correspondante → ignoré."""
        from scripts.sync_bitget_trades import _group_into_cycles
        records = [
            _make_record(trade_type="close", price=105.0, quantity=1.0, order_id="C1",
                         side="sell"),
        ]
        result = _group_into_cycles(records, leverage=3)
        assert len(result) == 0

    def test_two_independent_cycles(self):
        """2 cycles successifs sur le même symbol → 2 cycle_close."""
        from scripts.sync_bitget_trades import _group_into_cycles
        records = [
            _make_record(price=100.0, quantity=1.0, order_id="E1",
                         timestamp="2026-02-24T10:00:00+00:00"),
            _make_record(trade_type="close", price=102.0, quantity=1.0, order_id="C1",
                         side="sell", timestamp="2026-02-24T11:00:00+00:00"),
            # Second cycle
            _make_record(price=101.0, quantity=1.0, order_id="E2",
                         timestamp="2026-02-24T12:00:00+00:00"),
            _make_record(trade_type="close", price=103.0, quantity=1.0, order_id="C2",
                         side="sell", timestamp="2026-02-24T13:00:00+00:00"),
        ]
        result = _group_into_cycles(records, leverage=3)
        cycle_closes = [r for r in result if r["trade_type"] == "cycle_close"]
        assert len(cycle_closes) == 2
        # Premier cycle : pnl = (102-100)*1 = 2.0
        assert cycle_closes[0]["pnl"] == pytest.approx(2.0, abs=0.001)
        # Second cycle : pnl = (103-101)*1 = 2.0
        assert cycle_closes[1]["pnl"] == pytest.approx(2.0, abs=0.001)

    def test_cycle_close_order_id_format(self):
        """order_id du cycle_close = cycle_{first_entry_order_id}."""
        from scripts.sync_bitget_trades import _group_into_cycles
        records = [
            _make_record(price=100.0, quantity=1.0, order_id="ENTRY_ABC"),
            _make_record(trade_type="close", price=102.0, quantity=1.0, order_id="C1",
                         side="sell", timestamp="2026-02-24T11:00:00+00:00"),
        ]
        result = _group_into_cycles(records, leverage=3)
        cc = next(r for r in result if r["trade_type"] == "cycle_close")
        assert cc["order_id"] == "cycle_ENTRY_ABC"

    def test_pnl_pct_calculation(self):
        """pnl_pct = pnl / margin * 100, margin = avg_entry * qty / leverage."""
        from scripts.sync_bitget_trades import _group_into_cycles
        records = [
            _make_record(price=90000.0, quantity=0.01, order_id="E1", leverage=7),
            _make_record(trade_type="close", price=91000.0, quantity=0.01, order_id="C1",
                         side="sell", fee=0.55, leverage=7,
                         timestamp="2026-02-24T11:00:00+00:00"),
        ]
        result = _group_into_cycles(records, leverage=7)
        cc = next(r for r in result if r["trade_type"] == "cycle_close")
        # pnl = (91000 - 90000) * 0.01 - 0.55 = 10 - 0.55 = 9.45
        assert cc["pnl"] == pytest.approx(9.45, abs=0.001)
        # margin = 90000 * 0.01 / 7 = 128.57
        # pnl_pct = 9.45 / 128.57 * 100 ≈ 7.35
        assert cc["pnl_pct"] == pytest.approx(7.35, abs=0.1)

    def test_notional_dust_tolerance(self):
        """Résidu < $0.50 en notionnel → cycle fermé (filtre dust micro-fills).

        Scénario : close légèrement en dessous de l'entry, résidu = 0.00001 unité
        × prix $10000 = $0.10 < $0.50 → traité comme position = 0.
        Sans filtre notionnel : résidu 0.00001 > tolérance 2%(0.0001)=0.000002
        → ok dans les 2 cas, mais avec dust plus extrême :
        Résidu 0.000008 × $10000 = $0.08 < $0.50 → cycle complet.
        """
        from scripts.sync_bitget_trades import _group_into_cycles
        records = [
            _make_record(price=10000.0, quantity=0.0001, order_id="E1"),
            # Close légèrement en dessous : résidu = 0.0001 - 0.000092 = 0.000008
            # Notionnel résidu = 0.000008 × 10000 = $0.08 < $0.50 → cycle fermé
            _make_record(trade_type="close", price=10000.0, quantity=0.000092, order_id="C1",
                         side="sell", timestamp="2026-02-24T11:00:00+00:00"),
        ]
        result = _group_into_cycles(records, leverage=3)
        cycle_closes = [r for r in result if r["trade_type"] == "cycle_close"]
        # Le cycle doit être considéré comme complet (résidu dust < $0.50)
        assert len(cycle_closes) == 1


# ─── Merge Close Bursts ─────────────────────────────────────────────────


def _make_close(order_id, timestamp, price=98.0, qty=0.1, fee=0.2, symbol="BTC/USDT:USDT", direction="LONG"):
    return {
        "timestamp": timestamp, "symbol": symbol, "direction": direction,
        "trade_type": "close", "side": "sell",
        "price": price, "quantity": qty, "fee": fee,
        "order_id": order_id, "strategy_name": "grid_atr", "leverage": 7,
    }


def _make_entry_rec(order_id, timestamp, price=100.0, qty=0.1, symbol="BTC/USDT:USDT", direction="LONG"):
    return {
        "timestamp": timestamp, "symbol": symbol, "direction": direction,
        "trade_type": "entry", "side": "buy",
        "price": price, "quantity": qty, "fee": 0.1,
        "order_id": order_id, "strategy_name": "grid_atr", "leverage": 7,
    }


class TestMergeCloseBursts:
    def test_single_close_unchanged(self):
        """1 seul close → retourné tel quel."""
        from scripts.sync_bitget_trades import _merge_close_bursts
        records = [_make_close("C1", "2026-02-24T10:00:00+00:00")]
        result = _merge_close_bursts(records)
        assert len(result) == 1
        assert result[0]["trade_type"] == "close"

    def test_three_closes_within_window_merged(self):
        """3 closes en 30s (SL sweep) → 1 seul close fusionné."""
        from scripts.sync_bitget_trades import _merge_close_bursts
        records = [
            _make_close("C1", "2026-02-24T10:00:00+00:00", price=98.0, qty=0.1, fee=0.2),
            _make_close("C2", "2026-02-24T10:00:10+00:00", price=97.5, qty=0.1, fee=0.2),
            _make_close("C3", "2026-02-24T10:00:20+00:00", price=97.0, qty=0.1, fee=0.2),
        ]
        result = _merge_close_bursts(records, window_minutes=5)
        closes = [r for r in result if r["trade_type"] == "close"]
        assert len(closes) == 1
        c = closes[0]
        assert c["quantity"] == pytest.approx(0.3, abs=1e-6)
        # VWAP = (98*0.1 + 97.5*0.1 + 97*0.1) / 0.3 = 29.25/0.3 = 97.5
        assert c["price"] == pytest.approx(97.5, abs=0.001)
        assert c["fee"] == pytest.approx(0.6, abs=1e-4)

    def test_closes_outside_window_not_merged(self):
        """2 closes espacés de 10 min (> window=5) → 2 closes séparés."""
        from scripts.sync_bitget_trades import _merge_close_bursts
        records = [
            _make_close("C1", "2026-02-24T10:00:00+00:00"),
            _make_close("C2", "2026-02-24T10:10:00+00:00"),  # +10 min
        ]
        result = _merge_close_bursts(records, window_minutes=5)
        closes = [r for r in result if r["trade_type"] == "close"]
        assert len(closes) == 2

    def test_entry_between_closes_breaks_merge(self):
        """close + entry + close → 2 closes séparés (entry reset le buffer)."""
        from scripts.sync_bitget_trades import _merge_close_bursts
        records = [
            _make_close("C1", "2026-02-24T10:00:00+00:00"),
            _make_entry_rec("E1", "2026-02-24T10:01:00+00:00"),  # entry entre les 2 closes
            _make_close("C2", "2026-02-24T10:02:00+00:00"),     # proche mais après une entry
        ]
        result = _merge_close_bursts(records, window_minutes=5)
        closes = [r for r in result if r["trade_type"] == "close"]
        entries = [r for r in result if r["trade_type"] == "entry"]
        assert len(closes) == 2
        assert len(entries) == 1

    def test_sl_sweep_three_levels(self):
        """Simulation SL grid 3 niveaux : 3 entries + 3 closes rapides → 1 cycle_close."""
        from scripts.sync_bitget_trades import _merge_close_bursts, _group_into_cycles
        records = [
            _make_entry_rec("E1", "2026-02-24T09:00:00+00:00", price=100.0, qty=1.0),
            _make_entry_rec("E2", "2026-02-24T09:30:00+00:00", price=95.0, qty=1.0),
            _make_entry_rec("E3", "2026-02-24T10:00:00+00:00", price=90.0, qty=1.0),
            # SL : 3 closes en quelques secondes
            _make_close("SL1", "2026-02-24T10:30:00+00:00", price=88.0, qty=1.0, fee=0.2),
            _make_close("SL2", "2026-02-24T10:30:05+00:00", price=88.0, qty=1.0, fee=0.2),
            _make_close("SL3", "2026-02-24T10:30:10+00:00", price=88.0, qty=1.0, fee=0.2),
        ]
        merged = _merge_close_bursts(records, window_minutes=5)
        result = _group_into_cycles(merged, leverage=7)

        cycle_closes = [r for r in result if r["trade_type"] == "cycle_close"]
        assert len(cycle_closes) == 1, "SL sweep = 1 seul cycle, pas 3"

        cc = cycle_closes[0]
        # avg_entry = (100+95+90)/3 = 95.0, avg_close = 88.0
        # entry_fees = 3 * 0.1 = 0.3, close_fees = 3 * 0.2 = 0.6
        # pnl = (88 - 95) * 3 - (0.3 + 0.6) = -21 - 0.9 = -21.9
        assert cc["pnl"] == pytest.approx(-21.9, abs=0.01)
        assert cc["grid_level"] == 3
