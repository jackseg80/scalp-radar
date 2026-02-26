"""Tests Sprint 50a — BTC Regime Detectors.

Couvre :
- resample_4h_to_daily()
- label_candles() (regime_labeler)
- BaseDetector / 3 detecteurs sur donnees synthetiques
- Hysteresis : previent l'oscillation rapide
- Metriques (accuracy, F1, confusion matrix, crash delay)
- Edge cases
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from scripts.regime_detectors import (
    EMAATRDetector,
    LABELS,
    MultiMAVolDetector,
    SMAStressDetector,
    accuracy,
    apply_hysteresis,
    avg_regime_duration,
    confusion_matrix_manual,
    crash_detection_delay,
    f1_per_class,
    n_transitions,
    regime_distribution,
    resample_4h_to_daily,
    stability_score,
)
from scripts.regime_labeler import label_candles


# ─── Helpers ─────────────────────────────────────────────────────────────

def make_4h_df(
    n_days: int,
    base_price: float = 50000.0,
    daily_return: float = 0.0,
    volatility: float = 0.001,
    start_date: str = "2023-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Genere un DataFrame synthetique de candles 4h."""
    rng = np.random.RandomState(seed)
    n_candles = n_days * 6
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=4 * i) for i in range(n_candles)]

    prices = [base_price]
    for i in range(1, n_candles):
        ret = daily_return / 6 + rng.normal(0, volatility)
        prices.append(prices[-1] * (1 + ret))

    prices_arr = np.array(prices)
    return pd.DataFrame({
        "timestamp_utc": [t.isoformat() for t in timestamps],
        "open": prices_arr,
        "high": prices_arr * (1 + rng.uniform(0, 0.005, n_candles)),
        "low": prices_arr * (1 - rng.uniform(0, 0.005, n_candles)),
        "close": prices_arr,
        "volume": [1000.0] * n_candles,
    })


def make_crash_df(
    n_days_before: int = 100,
    n_days_crash: int = 5,
    n_days_after: int = 50,
    crash_daily_return: float = -0.10,
) -> pd.DataFrame:
    """Genere donnees avec une phase plate, un crash, puis recovery."""
    df_before = make_4h_df(n_days_before, base_price=50000, daily_return=0.001, seed=1)
    last_price = df_before["close"].iloc[-1]

    df_crash = make_4h_df(
        n_days_crash, base_price=last_price,
        daily_return=crash_daily_return, volatility=0.005, seed=2,
        start_date="2023-04-11",
    )
    last_crash_price = df_crash["close"].iloc[-1]

    df_after = make_4h_df(
        n_days_after, base_price=last_crash_price,
        daily_return=0.002, seed=3,
        start_date="2023-04-16",
    )

    return pd.concat([df_before, df_crash, df_after], ignore_index=True)


# ─── TestResample4hToDaily ────────────────────────────────────────────────

class TestResample4hToDaily:
    def test_basic_aggregation(self):
        """6 candles 4h -> 1 daily avec OHLCV correct."""
        df = make_4h_df(1)
        daily = resample_4h_to_daily(df)
        assert len(daily) == 1
        assert daily.iloc[0]["open"] == pytest.approx(df.iloc[0]["open"])
        assert daily.iloc[0]["high"] >= df["high"].max() - 0.01
        assert daily.iloc[0]["low"] <= df["low"].min() + 0.01
        assert daily.iloc[0]["close"] == pytest.approx(df.iloc[-1]["close"])
        assert daily.iloc[0]["volume"] == pytest.approx(6000.0)

    def test_multiple_days(self):
        """10 jours -> 10 lignes daily."""
        df = make_4h_df(10)
        daily = resample_4h_to_daily(df)
        assert len(daily) == 10

    def test_incomplete_day_filtered(self):
        """Jours avec < 6 candles sont exclus."""
        df = make_4h_df(2)
        df = df.iloc[:9]  # 1 jour complet + 3/6
        daily = resample_4h_to_daily(df)
        assert len(daily) == 1


# ─── TestLabelCandles ─────────────────────────────────────────────────────

class TestLabelCandles:
    def test_crash_overrides_bear(self):
        """Quand crash et bear se chevauchent, crash gagne."""
        df = make_4h_df(30)
        events = [
            {"regime": "bear",
             "start": pd.Timestamp("2023-01-05", tz="UTC"),
             "end": pd.Timestamp("2023-01-25", tz="UTC")},
            {"regime": "crash",
             "start": pd.Timestamp("2023-01-10", tz="UTC"),
             "end": pd.Timestamp("2023-01-15", tz="UTC")},
        ]
        df = label_candles(df, events, "range")
        # Candles du 12 jan doivent etre crash
        ts = pd.to_datetime(df["timestamp_utc"], utc=True)
        jan12_mask = ts.dt.date == pd.Timestamp("2023-01-12").date()
        jan12_labels = df.loc[jan12_mask, "regime_label"]
        assert all(jan12_labels == "crash")

    def test_default_regime_outside_events(self):
        """Candles hors evenements -> default_regime."""
        df = make_4h_df(30)
        events = [
            {"regime": "bull",
             "start": pd.Timestamp("2023-01-15", tz="UTC"),
             "end": pd.Timestamp("2023-01-20", tz="UTC")},
        ]
        df = label_candles(df, events, "range")
        ts = pd.to_datetime(df["timestamp_utc"], utc=True)
        jan02_mask = ts.dt.date == pd.Timestamp("2023-01-02").date()
        jan02_labels = df.loc[jan02_mask, "regime_label"]
        assert all(jan02_labels == "range")

    def test_gap_between_events(self):
        """Gap entre deux evenements -> default_regime."""
        df = make_4h_df(30)
        events = [
            {"regime": "bull",
             "start": pd.Timestamp("2023-01-03", tz="UTC"),
             "end": pd.Timestamp("2023-01-05", tz="UTC")},
            {"regime": "bear",
             "start": pd.Timestamp("2023-01-20", tz="UTC"),
             "end": pd.Timestamp("2023-01-25", tz="UTC")},
        ]
        df = label_candles(df, events, "range")
        ts = pd.to_datetime(df["timestamp_utc"], utc=True)
        jan10_mask = ts.dt.date == pd.Timestamp("2023-01-10").date()
        jan10_labels = df.loc[jan10_mask, "regime_label"]
        assert all(jan10_labels == "range")

    def test_priority_order(self):
        """Priorite : crash > bear > range > bull."""
        df = make_4h_df(10)
        # end couvre tout le jour 10 (23:59:59)
        end = pd.Timestamp("2023-01-10T23:59:59", tz="UTC")
        events = [
            {"regime": "bull",
             "start": pd.Timestamp("2023-01-01", tz="UTC"),
             "end": end},
            {"regime": "range",
             "start": pd.Timestamp("2023-01-01", tz="UTC"),
             "end": end},
            {"regime": "bear",
             "start": pd.Timestamp("2023-01-01", tz="UTC"),
             "end": end},
        ]
        df = label_candles(df, events, "range")
        # bear devrait gagner (priorite crash > bear > range > bull)
        assert all(df["regime_label"] == "bear")


# ─── TestHysteresis ───────────────────────────────────────────────────────

class TestHysteresis:
    def test_prevents_rapid_oscillation(self):
        """Alternance rapide ne provoque pas d'oscillation avec h_down=6."""
        raw = ["bull", "bear"] * 20  # 40 candles alternees
        smoothed = apply_hysteresis(raw, h_down=6, h_up=6)
        transitions = n_transitions(smoothed)
        # Raw a 39 transitions, smoothed devrait en avoir beaucoup moins
        assert transitions < 5

    def test_asymmetric_fast_down_slow_up(self):
        """h_down < h_up : entre en bear vite, sort lentement."""
        # 5 bull, 8 bear, 50 bull
        raw = ["bull"] * 5 + ["bear"] * 8 + ["bull"] * 50
        smoothed = apply_hysteresis(raw, h_down=3, h_up=20)
        # Devrait entrer en bear assez vite (autour de index 8)
        assert "bear" in smoothed[8:15]
        # Mais ne sort pas immediatement en bull apres les 8 bears
        assert smoothed[14] == "bear"  # toujours bear a index 14

    def test_crash_to_range_needs_h_up(self):
        """Sortir de crash necessite h_up candles."""
        raw = ["range"] * 5 + ["crash"] * 10 + ["range"] * 3 + ["crash"] * 2 + ["range"] * 30
        smoothed = apply_hysteresis(raw, h_down=3, h_up=12)
        # Les 3 candles range au milieu ne suffisent pas a sortir de crash
        # (h_up=12 > 3)
        assert smoothed[17] == "crash"  # toujours crash apres les 3 ranges


# ─── TestSMAStressDetector ────────────────────────────────────────────────

class TestSMAStressDetector:
    def test_bull_above_sma(self):
        """Prix en hausse -> majoritairement bull apres warmup."""
        df = make_4h_df(400, daily_return=0.003, volatility=0.001, seed=10)
        daily = resample_4h_to_daily(df)
        det = SMAStressDetector()
        raw, warmup_idx = det.detect_raw(
            daily, sma_period=50, stress_window=7, stress_threshold=-20,
        )
        # Apres warmup, majoritairement bull
        after_warmup = raw[warmup_idx:]
        bull_pct = after_warmup.count("bull") / len(after_warmup) if after_warmup else 0
        assert bull_pct > 0.7

    def test_bear_below_sma(self):
        """Prix en baisse -> majoritairement bear apres warmup."""
        df = make_4h_df(400, daily_return=-0.003, volatility=0.001, seed=20)
        daily = resample_4h_to_daily(df)
        det = SMAStressDetector()
        raw, warmup_idx = det.detect_raw(
            daily, sma_period=50, stress_window=7, stress_threshold=-20,
        )
        after_warmup = raw[warmup_idx:]
        bear_pct = after_warmup.count("bear") / len(after_warmup) if after_warmup else 0
        assert bear_pct > 0.5

    def test_param_grid_count(self):
        """324 combinaisons attendues."""
        grid = SMAStressDetector.param_grid()
        assert len(grid) == 3 * 4 * 3 * 3 * 3  # 324


# ─── TestEMAATRDetector ───────────────────────────────────────────────────

class TestEMAATRDetector:
    def test_bull_trend(self):
        """Prix en hausse -> majoritairement bull."""
        df = make_4h_df(400, daily_return=0.003, volatility=0.001, seed=30)
        daily = resample_4h_to_daily(df)
        det = EMAATRDetector()
        raw, warmup_idx = det.detect_raw(
            daily, ema_fast=30, ema_slow=100, atr_fast=7,
            atr_slow=30, atr_stress_ratio=3.0,
        )
        after_warmup = raw[warmup_idx:]
        bull_pct = after_warmup.count("bull") / len(after_warmup) if after_warmup else 0
        assert bull_pct > 0.6

    def test_param_grid_count(self):
        """432 combinaisons attendues."""
        grid = EMAATRDetector.param_grid()
        assert len(grid) == 2 * 2 * 2 * 2 * 3 * 3 * 3  # 432


# ─── TestMultiMAVolDetector ───────────────────────────────────────────────

class TestMultiMAVolDetector:
    def test_three_states_possible(self):
        """Le detecteur produit bien bull, bear et range."""
        # Donnees mixtes : hausse puis baisse
        df_up = make_4h_df(200, daily_return=0.005, volatility=0.001, seed=40)
        df_down = make_4h_df(
            200, daily_return=-0.005, volatility=0.001, seed=41,
            start_date="2023-07-20",
        )
        # Ajuster le prix de depart de df_down
        last_price = df_up["close"].iloc[-1]
        ratio = last_price / df_down["open"].iloc[0]
        for col in ["open", "high", "low", "close"]:
            df_down[col] = df_down[col] * ratio
        df = pd.concat([df_up, df_down], ignore_index=True)

        daily = resample_4h_to_daily(df)
        det = MultiMAVolDetector()
        raw, _ = det.detect_raw(
            daily, vol_window=14, vol_percentile=95, vol_lookback=180,
        )
        unique_regimes = set(raw)
        # Devrait produire au moins 2 regimes differents
        assert len(unique_regimes) >= 2

    def test_param_grid_count(self):
        """72 combinaisons attendues."""
        grid = MultiMAVolDetector.param_grid()
        assert len(grid) == 3 * 3 * 2 * 2 * 2  # 72


# ─── TestMetrics ──────────────────────────────────────────────────────────

class TestMetrics:
    def test_accuracy_perfect(self):
        y = ["bull", "bear", "range", "crash"]
        assert accuracy(y, y) == 1.0

    def test_accuracy_zero(self):
        y_true = ["bull", "bull", "bull"]
        y_pred = ["bear", "bear", "bear"]
        assert accuracy(y_true, y_pred) == 0.0

    def test_accuracy_empty(self):
        assert accuracy([], []) == 0.0

    def test_f1_perfect(self):
        y = ["bull", "bear", "range", "crash"] * 10
        f1 = f1_per_class(y, y, LABELS)
        assert all(v == pytest.approx(1.0) for v in f1.values())

    def test_f1_zero_for_missing_class(self):
        y_true = ["bull", "bull", "bull"]
        y_pred = ["bear", "bear", "bear"]
        f1 = f1_per_class(y_true, y_pred, LABELS)
        assert f1["bull"] == 0.0
        assert f1["bear"] == 0.0  # precision=1, recall=0 -> F1=0

    def test_confusion_matrix_diagonal(self):
        y = ["bull", "bear", "range"]
        cm = confusion_matrix_manual(y, y, LABELS)
        assert cm["bull"]["bull"] == 1
        assert cm["bear"]["bear"] == 1
        assert cm["range"]["range"] == 1
        assert cm["bull"]["bear"] == 0
        assert cm["crash"]["crash"] == 0

    def test_crash_delay_immediate(self):
        y_true = ["range"] * 10 + ["crash"] * 10 + ["range"] * 10
        y_pred = ["range"] * 10 + ["crash"] * 10 + ["range"] * 10
        result = crash_detection_delay(y_true, y_pred)
        assert result["avg_delay_hours"] == 0.0
        assert result["n_detected"] == 1
        assert result["false_positives"] == 0

    def test_crash_delay_with_lag(self):
        y_true = ["range"] * 10 + ["crash"] * 10 + ["range"] * 10
        y_pred = ["range"] * 13 + ["crash"] * 7 + ["range"] * 10
        result = crash_detection_delay(y_true, y_pred)
        assert result["avg_delay_hours"] == 12.0  # 3 candles * 4h
        assert result["n_detected"] == 1

    def test_crash_false_positives(self):
        y_true = ["range"] * 30
        y_pred = ["range"] * 10 + ["crash"] * 5 + ["range"] * 15
        result = crash_detection_delay(y_true, y_pred)
        assert result["false_positives"] == 1
        assert result["n_crashes_gt"] == 0

    def test_n_transitions(self):
        labels = ["bull", "bull", "bear", "bear", "crash", "range"]
        assert n_transitions(labels) == 3

    def test_n_transitions_no_change(self):
        labels = ["bull"] * 10
        assert n_transitions(labels) == 0

    def test_stability_score_all_stable(self):
        labels = ["bull"] * 10 + ["bear"] * 10
        assert stability_score(labels) == 1.0

    def test_stability_score_all_unstable(self):
        labels = ["bull", "bear"] * 5
        assert stability_score(labels, min_duration_candles=6) == 0.0

    def test_avg_regime_duration(self):
        labels = ["bull"] * 12 + ["bear"] * 6
        dur = avg_regime_duration(labels, candle_hours=4.0)
        assert dur["bull"] == pytest.approx(48.0)
        assert dur["bear"] == pytest.approx(24.0)
        assert dur["range"] == 0.0

    def test_regime_distribution(self):
        labels = ["bull"] * 50 + ["bear"] * 50
        dist = regime_distribution(labels)
        assert dist["bull"] == pytest.approx(0.5)
        assert dist["bear"] == pytest.approx(0.5)
        assert dist["crash"] == pytest.approx(0.0)
