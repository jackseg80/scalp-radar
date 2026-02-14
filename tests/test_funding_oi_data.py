"""Tests Sprint 7b — Funding/OI data, extra_data_builder, config per_asset, WFO integration.

22 tests couvrant :
- Tables DB funding_rates et open_interest (CRUD)
- extra_data_builder (alignement, forward-fill)
- BacktestEngine avec extra_data
- Config per_asset pour FundingConfig et LiquidationConfig
- STRATEGY_REGISTRY étendu
- create_strategy_with_params mirroring
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import pytest_asyncio

from backend.core.database import Database
from backend.core.models import Candle, TimeFrame


# ─── Fixtures ─────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def db():
    """DB en mémoire pour chaque test."""
    database = Database(db_path=":memory:")
    await database.init()
    yield database
    await database.close()


def _make_candle(
    ts_offset_minutes: int = 0,
    price: float = 50000.0,
    symbol: str = "BTC/USDT",
    tf: TimeFrame = TimeFrame.M5,
) -> Candle:
    base = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts = base + timedelta(minutes=ts_offset_minutes)
    return Candle(
        timestamp=ts,
        open=price,
        high=price + 50,
        low=price - 50,
        close=price + 10,
        volume=1000.0,
        symbol=symbol,
        timeframe=tf,
    )


def _make_funding_rate(ts_offset_hours: int = 0, rate: float = 0.01) -> dict:
    """Crée un funding rate dict (epoch ms, rate en %)."""
    base = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts = base + timedelta(hours=ts_offset_hours)
    return {
        "symbol": "BTC/USDT",
        "exchange": "binance",
        "timestamp": int(ts.timestamp() * 1000),
        "funding_rate": rate,
    }


def _make_oi_record(
    ts_offset_minutes: int = 0,
    oi: float = 100000.0,
    oi_value: float = 5_000_000_000.0,
) -> dict:
    """Crée un OI record dict (epoch ms)."""
    base = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts = base + timedelta(minutes=ts_offset_minutes)
    return {
        "symbol": "BTC/USDT",
        "exchange": "binance",
        "timeframe": "5m",
        "timestamp": int(ts.timestamp() * 1000),
        "oi": oi,
        "oi_value": oi_value,
    }


# ─── Tests DB : funding_rates ────────────────────────────────────────────


@pytest.mark.asyncio
class TestFundingRatesCRUD:
    async def test_insert_and_get(self, db):
        rates = [_make_funding_rate(h, 0.01 * h) for h in range(3)]
        inserted = await db.insert_funding_rates_batch(rates)
        assert inserted == 3

        result = await db.get_funding_rates("BTC/USDT", "binance")
        assert len(result) == 3
        assert result[0]["funding_rate"] == 0.0
        assert result[2]["funding_rate"] == pytest.approx(0.02)

    async def test_no_duplicates(self, db):
        rate = _make_funding_rate(0, 0.01)
        await db.insert_funding_rates_batch([rate])
        inserted = await db.insert_funding_rates_batch([rate])
        assert inserted == 0

    async def test_get_with_time_filter(self, db):
        rates = [_make_funding_rate(h * 8, 0.01) for h in range(10)]
        await db.insert_funding_rates_batch(rates)

        # Filtrer les 5 premiers
        end_ts = rates[4]["timestamp"]
        result = await db.get_funding_rates("BTC/USDT", "binance", end_ts=end_ts)
        assert len(result) == 5

    async def test_get_latest_timestamp(self, db):
        rates = [_make_funding_rate(h * 8, 0.01) for h in range(5)]
        await db.insert_funding_rates_batch(rates)

        latest = await db.get_latest_funding_timestamp("BTC/USDT", "binance")
        assert latest == rates[-1]["timestamp"]

    async def test_get_latest_timestamp_empty(self, db):
        latest = await db.get_latest_funding_timestamp("BTC/USDT", "binance")
        assert latest is None

    async def test_empty_batch(self, db):
        inserted = await db.insert_funding_rates_batch([])
        assert inserted == 0


# ─── Tests DB : open_interest ────────────────────────────────────────────


@pytest.mark.asyncio
class TestOpenInterestCRUD:
    async def test_insert_and_get(self, db):
        records = [_make_oi_record(m * 5, oi=100000 + m * 1000) for m in range(5)]
        inserted = await db.insert_oi_batch(records)
        assert inserted == 5

        result = await db.get_open_interest("BTC/USDT", "5m", "binance")
        assert len(result) == 5
        assert result[0]["oi"] == 100000.0

    async def test_no_duplicates(self, db):
        record = _make_oi_record(0)
        await db.insert_oi_batch([record])
        inserted = await db.insert_oi_batch([record])
        assert inserted == 0

    async def test_get_latest_timestamp(self, db):
        records = [_make_oi_record(m * 5) for m in range(5)]
        await db.insert_oi_batch(records)

        latest = await db.get_latest_oi_timestamp("BTC/USDT", "5m", "binance")
        assert latest == records[-1]["timestamp"]

    async def test_get_latest_timestamp_empty(self, db):
        latest = await db.get_latest_oi_timestamp("BTC/USDT", "5m", "binance")
        assert latest is None

    async def test_empty_batch(self, db):
        inserted = await db.insert_oi_batch([])
        assert inserted == 0


# ─── Tests extra_data_builder ────────────────────────────────────────────


class TestExtraDataBuilder:
    def test_funding_forward_fill(self):
        """Le funding rate est forward-filled entre les publications (toutes les 8h)."""
        from backend.backtesting.extra_data_builder import build_extra_data_map
        from backend.strategies.base import EXTRA_FUNDING_RATE

        # 12 bougies 5m (1 heure)
        candles = [_make_candle(ts_offset_minutes=m * 5) for m in range(12)]

        # Un funding rate à t=0
        funding = [_make_funding_rate(0, 0.05)]

        result = build_extra_data_map(candles, funding_rates=funding)

        # Toutes les bougies doivent avoir le funding rate (forward-fill)
        for candle in candles:
            ts_iso = candle.timestamp.isoformat()
            assert ts_iso in result
            assert EXTRA_FUNDING_RATE in result[ts_iso]
            assert result[ts_iso][EXTRA_FUNDING_RATE] == 0.05

    def test_funding_updates_on_new_rate(self):
        """Le funding rate change quand un nouveau rate arrive."""
        from backend.backtesting.extra_data_builder import build_extra_data_map
        from backend.strategies.base import EXTRA_FUNDING_RATE

        candles = [_make_candle(ts_offset_minutes=m * 5) for m in range(24)]

        # Rate 0.01 à t=0, rate 0.05 à t=60min
        funding = [
            _make_funding_rate(0, 0.01),
            _make_funding_rate(1, 0.05),  # +1h
        ]

        result = build_extra_data_map(candles, funding_rates=funding)

        # Premières 12 bougies = 0.01, après = 0.05
        ts_first = candles[0].timestamp.isoformat()
        ts_last = candles[-1].timestamp.isoformat()
        assert result[ts_first][EXTRA_FUNDING_RATE] == 0.01
        assert result[ts_last][EXTRA_FUNDING_RATE] == 0.05

    def test_oi_change_pct(self):
        """L'OI change est calculé correctement (variation %)."""
        from backend.backtesting.extra_data_builder import build_extra_data_map
        from backend.strategies.base import EXTRA_OI_CHANGE_PCT

        candles = [_make_candle(ts_offset_minutes=m * 5) for m in range(3)]

        oi = [
            _make_oi_record(0, oi_value=1_000_000),
            _make_oi_record(5, oi_value=1_100_000),   # +10%
            _make_oi_record(10, oi_value=1_045_000),   # -5%
        ]

        result = build_extra_data_map(candles, oi_records=oi)

        ts0 = candles[0].timestamp.isoformat()
        ts1 = candles[1].timestamp.isoformat()
        ts2 = candles[2].timestamp.isoformat()

        # Premier OI : change = 0 (pas de précédent)
        assert result[ts0][EXTRA_OI_CHANGE_PCT] == 0.0
        # Deuxième : +10%
        assert result[ts1][EXTRA_OI_CHANGE_PCT] == pytest.approx(10.0)
        # Troisième : -5%
        assert result[ts2][EXTRA_OI_CHANGE_PCT] == pytest.approx(-5.0)

    def test_empty_data(self):
        """Pas de funding/OI = extra_data vides."""
        from backend.backtesting.extra_data_builder import build_extra_data_map

        candles = [_make_candle(ts_offset_minutes=m * 5) for m in range(3)]
        result = build_extra_data_map(candles)

        for candle in candles:
            ts_iso = candle.timestamp.isoformat()
            assert ts_iso in result
            assert result[ts_iso] == {}


# ─── Tests Config per_asset ──────────────────────────────────────────────


class TestConfigPerAsset:
    def test_funding_config_per_asset(self):
        from backend.core.config import FundingConfig

        cfg = FundingConfig(
            extreme_positive_threshold=0.03,
            per_asset={"BTC/USDT": {"extreme_positive_threshold": 0.05}},
        )
        params = cfg.get_params_for_symbol("BTC/USDT")
        assert params["extreme_positive_threshold"] == 0.05

        params_default = cfg.get_params_for_symbol("ETH/USDT")
        assert params_default["extreme_positive_threshold"] == 0.03

    def test_liquidation_config_per_asset(self):
        from backend.core.config import LiquidationConfig

        cfg = LiquidationConfig(
            oi_change_threshold=5.0,
            per_asset={"SOL/USDT": {"oi_change_threshold": 7.0}},
        )
        params = cfg.get_params_for_symbol("SOL/USDT")
        assert params["oi_change_threshold"] == 7.0


# ─── Registry tests → centralisés dans test_strategy_registry.py ─────────


# ─── Tests BacktestEngine avec extra_data ────────────────────────────────


class TestBacktestEngineExtraData:
    def test_run_without_extra_data_unchanged(self):
        """Sans extra_data, le comportement est identique à avant."""
        from backend.backtesting.engine import BacktestConfig, BacktestEngine
        from backend.strategies.vwap_rsi import VwapRsiStrategy
        from backend.core.config import VwapRsiConfig

        candles = [_make_candle(ts_offset_minutes=m * 5) for m in range(100)]
        config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=candles[0].timestamp,
            end_date=candles[-1].timestamp,
        )
        strategy = VwapRsiStrategy(VwapRsiConfig())
        engine = BacktestEngine(config, strategy)
        result = engine.run({"5m": candles})
        # Pas de crash, résultat cohérent
        assert result.strategy_name == "vwap_rsi"
        assert len(result.equity_curve) == 100

    def test_run_with_extra_data(self):
        """Avec extra_data, les données sont injectées dans le context."""
        from backend.backtesting.engine import BacktestConfig, BacktestEngine
        from backend.strategies.funding import FundingStrategy
        from backend.core.config import FundingConfig

        candles = [_make_candle(ts_offset_minutes=m * 15) for m in range(50)]
        config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=candles[0].timestamp,
            end_date=candles[-1].timestamp,
        )
        strategy = FundingStrategy(FundingConfig(timeframe="5m"))
        engine = BacktestEngine(config, strategy)

        # Construire extra_data_map
        extra_map: dict[str, dict[str, Any]] = {}
        for c in candles:
            extra_map[c.timestamp.isoformat()] = {
                "funding_rate": 0.05,
            }

        result = engine.run(
            {"5m": candles},
            extra_data_by_timestamp=extra_map,
        )
        assert result.strategy_name == "funding"
        assert len(result.equity_curve) == 50
