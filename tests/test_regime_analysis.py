"""Tests Sprint 15b — Analyse par régime de marché.

Couvre :
- _classify_regime() : classification Bull/Bear/Range/Crash
- Migration DB regime_analysis
- Stockage/récupération regime_analysis dans optimization_db
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from backend.core.database import Database
from backend.core.models import Candle, TimeFrame
from backend.optimization.optimization_db import (
    get_result_by_id_async,
    save_result_sync,
    save_result_from_payload_sync,
    build_push_payload,
)
from backend.optimization.report import FinalReport, ValidationResult
from backend.optimization.walk_forward import _classify_regime


# ─── Helpers ─────────────────────────────────────────────────────────────


def _make_candles(
    prices: list[float],
    start: datetime | None = None,
    interval_hours: int = 1,
) -> list[Candle]:
    """Génère une liste de candles 1h à partir de prix close."""
    if start is None:
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    candles = []
    for i, price in enumerate(prices):
        ts = start + timedelta(hours=i * interval_hours)
        candles.append(Candle(
            timestamp=ts,
            open=price,
            high=price * 1.005,
            low=price * 0.995,
            close=price,
            volume=1000.0,
            symbol="BTC/USDT",
            timeframe=TimeFrame.H1,
        ))
    return candles


def _make_validation() -> ValidationResult:
    return ValidationResult(
        bitget_sharpe=1.5, bitget_net_return_pct=8.0, bitget_trades=25,
        bitget_sharpe_ci_low=0.8, bitget_sharpe_ci_high=2.1,
        binance_oos_avg_sharpe=1.3, transfer_ratio=0.85,
        transfer_significant=True, volume_warning=False, volume_warning_detail="",
    )


def _make_report(**overrides) -> FinalReport:
    defaults = dict(
        strategy_name="vwap_rsi", symbol="BTC/USDT",
        timestamp=datetime(2026, 2, 14, 12, 0),
        grade="A", total_score=87, wfo_avg_is_sharpe=2.0, wfo_avg_oos_sharpe=1.7,
        wfo_consistency_rate=0.80, wfo_n_windows=20, recommended_params={"rsi_period": 14},
        mc_p_value=0.02, mc_significant=True, mc_underpowered=False, dsr=0.95,
        dsr_max_expected_sharpe=3.2, stability=0.88, cliff_params=[], convergence=0.75,
        divergent_params=[], validation=_make_validation(), oos_is_ratio=0.85,
        bitget_transfer=0.85, live_eligible=True, warnings=[], n_distinct_combos=600,
    )
    defaults.update(overrides)
    return FinalReport(**defaults)


@pytest.fixture
def temp_db(tmp_path):
    """DB temporaire avec table optimization_results + colonne regime_analysis."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE optimization_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            asset TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            created_at TEXT NOT NULL,
            duration_seconds REAL,
            grade TEXT NOT NULL,
            total_score REAL NOT NULL,
            oos_sharpe REAL,
            consistency REAL,
            oos_is_ratio REAL,
            dsr REAL,
            param_stability REAL,
            monte_carlo_pvalue REAL,
            mc_underpowered INTEGER DEFAULT 0,
            n_windows INTEGER NOT NULL,
            n_distinct_combos INTEGER,
            best_params TEXT NOT NULL,
            wfo_windows TEXT,
            monte_carlo_summary TEXT,
            validation_summary TEXT,
            warnings TEXT,
            is_latest INTEGER DEFAULT 1,
            source TEXT DEFAULT 'local',
            regime_analysis TEXT,
            win_rate_oos REAL,
            tail_risk_ratio REAL,
            UNIQUE(strategy_name, asset, timeframe, created_at)
        );
    """)
    conn.close()
    return db_path


# ─── Tests _classify_regime ──────────────────────────────────────────────


class TestClassifyRegime:
    """Tests unitaires pour _classify_regime."""

    def test_empty_candles(self):
        """< 2 candles → range par défaut."""
        result = _classify_regime([])
        assert result["regime"] == "range"
        assert result["return_pct"] == 0.0

    def test_single_candle(self):
        """1 candle → range par défaut."""
        candles = _make_candles([100.0])
        result = _classify_regime(candles)
        assert result["regime"] == "range"

    def test_range_regime(self):
        """Rendement entre -20% et +20% → Range."""
        # +10% sur la période
        prices = [100.0] * 50 + [110.0] * 10
        candles = _make_candles(prices)
        result = _classify_regime(candles)
        assert result["regime"] == "range"
        assert 9.0 < result["return_pct"] < 11.0

    def test_bull_regime(self):
        """Rendement > +20% → Bull."""
        # Hausse linéaire de 100 à 130 (+30%)
        prices = [100.0 + i * 0.5 for i in range(61)]
        candles = _make_candles(prices)
        result = _classify_regime(candles)
        assert result["regime"] == "bull"
        assert result["return_pct"] > 20.0

    def test_bear_regime(self):
        """Rendement < -20% → Bear."""
        # Baisse linéaire de 100 à 70 (-30%)
        prices = [100.0 - i * 0.5 for i in range(61)]
        candles = _make_candles(prices)
        result = _classify_regime(candles)
        assert result["regime"] == "bear"
        assert result["return_pct"] < -20.0

    def test_crash_regime(self):
        """Drawdown > 30% en < 14 jours → Crash (prioritaire)."""
        # Monte à 100, puis chute brutale à 65 en 10 jours (200h < 336h = 14j)
        prices = [100.0] * 100  # stable 100 jours
        # Crash en 200h (< 14 jours × 24h = 336h)
        prices += [100.0 - i * 0.2 for i in range(200)]
        # Rebond pour que le return global ne soit pas bear
        prices += [95.0] * 50
        candles = _make_candles(prices)

        # Le crash fait 100 → 60 = -40% en 200h (< 14j)
        result = _classify_regime(candles)
        assert result["regime"] == "crash"

    def test_crash_priority_over_bear(self):
        """Crash est prioritaire sur Bear (même si return < -20%)."""
        # Chute de 100 à 50 en 10 jours
        prices = [100.0 - i * 0.5 for i in range(101)]  # 100 → 50
        candles = _make_candles(prices)
        result = _classify_regime(candles)
        # return_pct = -50% → pourrait être bear
        # mais dd > 30% en < 14j → crash prioritaire
        assert result["regime"] == "crash"

    def test_slow_decline_not_crash(self):
        """Baisse lente (> 14 jours pour -30%) → Bear, pas Crash."""
        # Baisse de 100 à 65 sur 60 jours (1440h) — jamais -30% en 14j
        n = 60 * 24  # 1440 heures
        prices = [100.0 - (35.0 * i / n) for i in range(n + 1)]
        candles = _make_candles(prices)
        result = _classify_regime(candles)
        assert result["regime"] == "bear"
        assert result["return_pct"] < -20.0

    def test_return_pct_accuracy(self):
        """Vérifier la précision du calcul return_pct."""
        prices = [100.0, 115.0]
        candles = _make_candles(prices)
        result = _classify_regime(candles)
        assert result["return_pct"] == 15.0
        assert result["regime"] == "range"

    def test_max_dd_pct(self):
        """Vérifier le calcul du max drawdown."""
        # 100 → 120 → 90 → 100 : max dd = (90 - 120) / 120 = -25%
        prices = [100.0, 110.0, 120.0, 100.0, 90.0, 95.0, 100.0]
        candles = _make_candles(prices)
        result = _classify_regime(candles)
        assert result["max_dd_pct"] == pytest.approx(-25.0, abs=0.1)


# ─── Tests DB : migration ───────────────────────────────────────────────


class TestRegimeDBMigration:
    """Test que la migration ajoute la colonne regime_analysis."""

    @pytest.mark.asyncio
    async def test_migration_adds_column(self, tmp_path):
        """La migration ajoute regime_analysis à une table existante."""
        db = Database(db_path=str(tmp_path / "migrate.db"))
        await db.init()

        # Vérifier que la colonne existe après init
        conn = sqlite3.connect(str(tmp_path / "migrate.db"))
        cursor = conn.execute("PRAGMA table_info(optimization_results)")
        col_names = [row[1] for row in cursor.fetchall()]
        conn.close()
        await db.close()

        assert "regime_analysis" in col_names

    @pytest.mark.asyncio
    async def test_migration_idempotent(self, tmp_path):
        """La migration ne crashe pas si appelée deux fois."""
        db = Database(db_path=str(tmp_path / "idem.db"))
        await db.init()
        await db.close()

        # Deuxième init ne doit pas crasher
        db2 = Database(db_path=str(tmp_path / "idem.db"))
        await db2.init()
        await db2.close()


# ─── Tests DB : stockage regime_analysis ─────────────────────────────────


class TestRegimeStorage:
    """Tests de stockage/récupération regime_analysis."""

    def test_save_result_sync_with_regime(self, temp_db):
        """save_result_sync stocke regime_analysis en JSON."""
        regime_analysis = {
            "Bull": {"count": 3, "avg_oos_sharpe": 1.5, "consistency": 0.67, "avg_return_pct": 8.2},
            "Range": {"count": 5, "avg_oos_sharpe": 0.8, "consistency": 0.60, "avg_return_pct": 2.1},
        }
        report = _make_report()
        save_result_sync(
            temp_db, report, wfo_windows=None, duration=120.0,
            timeframe="5m", regime_analysis=regime_analysis,
        )

        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("SELECT regime_analysis FROM optimization_results")
        row = cursor.fetchone()
        conn.close()

        assert row[0] is not None
        parsed = json.loads(row[0])
        assert "Bull" in parsed
        assert parsed["Bull"]["count"] == 3
        assert parsed["Range"]["avg_oos_sharpe"] == 0.8

    def test_save_result_sync_without_regime(self, temp_db):
        """save_result_sync sans regime_analysis → NULL en DB."""
        report = _make_report()
        save_result_sync(temp_db, report, wfo_windows=None, duration=120.0, timeframe="5m")

        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("SELECT regime_analysis FROM optimization_results")
        row = cursor.fetchone()
        conn.close()

        assert row[0] is None

    @pytest.mark.asyncio
    async def test_get_result_by_id_parses_regime(self, temp_db):
        """get_result_by_id_async parse regime_analysis du JSON."""
        regime_data = {
            "Crash": {"count": 1, "avg_oos_sharpe": -0.5, "consistency": 0.0, "avg_return_pct": -15.0},
        }
        conn = sqlite3.connect(temp_db)
        conn.execute("""INSERT INTO optimization_results (
            strategy_name, asset, timeframe, created_at, duration_seconds,
            grade, total_score, oos_sharpe, consistency, oos_is_ratio, dsr,
            param_stability, monte_carlo_pvalue, mc_underpowered, n_windows, n_distinct_combos,
            best_params, wfo_windows, monte_carlo_summary, validation_summary, warnings,
            is_latest, source, regime_analysis
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            "vwap_rsi", "BTC/USDT", "5m", "2026-02-14T12:00:00", 120.0,
            "A", 87.0, 1.8, 0.85, 0.92, 0.95, 0.88, 0.02, 0, 20, 600,
            '{"rsi_period": 14}', None, '{}', '{}', '[]', 1, 'local',
            json.dumps(regime_data),
        ))
        conn.commit()
        conn.close()

        result = await get_result_by_id_async(temp_db, 1)

        assert result is not None
        assert result["regime_analysis"] is not None
        assert isinstance(result["regime_analysis"], dict)
        assert "Crash" in result["regime_analysis"]
        assert result["regime_analysis"]["Crash"]["count"] == 1

    @pytest.mark.asyncio
    async def test_get_result_by_id_regime_null(self, temp_db):
        """get_result_by_id_async retourne None si regime_analysis est NULL."""
        conn = sqlite3.connect(temp_db)
        conn.execute("""INSERT INTO optimization_results (
            strategy_name, asset, timeframe, created_at, duration_seconds,
            grade, total_score, oos_sharpe, consistency, oos_is_ratio, dsr,
            param_stability, monte_carlo_pvalue, mc_underpowered, n_windows, n_distinct_combos,
            best_params, wfo_windows, monte_carlo_summary, validation_summary, warnings,
            is_latest, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            "vwap_rsi", "BTC/USDT", "5m", "2026-02-14T12:00:00", 120.0,
            "A", 87.0, 1.8, 0.85, 0.92, 0.95, 0.88, 0.02, 0, 20, 600,
            '{"rsi_period": 14}', None, '{}', '{}', '[]', 1, 'local',
        ))
        conn.commit()
        conn.close()

        result = await get_result_by_id_async(temp_db, 1)
        assert result is not None
        assert result.get("regime_analysis") is None

    def test_build_push_payload_with_regime(self):
        """build_push_payload inclut regime_analysis."""
        report = _make_report()
        regime = {"Bull": {"count": 2, "avg_oos_sharpe": 1.0, "consistency": 0.5, "avg_return_pct": 5.0}}
        payload = build_push_payload(
            report, wfo_windows=None, duration=60.0, timeframe="5m",
            regime_analysis=regime,
        )

        assert "regime_analysis" in payload
        assert payload["regime_analysis"] is not None
        parsed = json.loads(payload["regime_analysis"])
        assert parsed["Bull"]["count"] == 2

    def test_build_push_payload_without_regime(self):
        """build_push_payload sans regime_analysis → None."""
        report = _make_report()
        payload = build_push_payload(report, wfo_windows=None, duration=60.0, timeframe="5m")

        assert payload.get("regime_analysis") is None

    def test_save_result_from_payload_with_regime(self, temp_db):
        """save_result_from_payload_sync stocke regime_analysis depuis payload."""
        regime = {"Range": {"count": 4, "avg_oos_sharpe": 0.6, "consistency": 0.75, "avg_return_pct": 1.5}}
        payload = {
            "strategy_name": "vwap_rsi",
            "asset": "BTC/USDT",
            "timeframe": "5m",
            "created_at": "2026-02-14T12:00:00",
            "duration_seconds": 120.0,
            "grade": "A",
            "total_score": 87.0,
            "oos_sharpe": 1.8,
            "consistency": 0.85,
            "oos_is_ratio": 0.92,
            "dsr": 0.95,
            "param_stability": 0.88,
            "monte_carlo_pvalue": 0.02,
            "mc_underpowered": 0,
            "n_windows": 20,
            "n_distinct_combos": 600,
            "best_params": '{"rsi_period": 14}',
            "wfo_windows": None,
            "monte_carlo_summary": '{}',
            "validation_summary": '{}',
            "warnings": '[]',
            "source": "local",
            "regime_analysis": json.dumps(regime),
        }

        status = save_result_from_payload_sync(temp_db, payload)
        assert status == "created"

        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("SELECT regime_analysis FROM optimization_results")
        row = cursor.fetchone()
        conn.close()

        parsed = json.loads(row[0])
        assert parsed["Range"]["count"] == 4
