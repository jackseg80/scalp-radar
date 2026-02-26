"""BTC Market Regime Detectors — Sprint 50a.

3 detecteurs de regime bases sur indicateurs daily (recalcules depuis les 4h).
Chaque detecteur classifie chaque candle 4h en: bull, bear, range, crash.

Module importe par regime_analysis.py et les tests.

Usage standalone (verification rapide):
    uv run python -m scripts.regime_detectors
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import groupby
from typing import Any

import numpy as np
import pandas as pd

# ─── Severite des regimes (plus bas = plus severe) ───────────────────────

SEVERITY = {"bull": 0, "range": 1, "bear": 2, "crash": 3}
LABELS = ["bull", "bear", "range", "crash"]


# ─── Resample 4h → Daily ─────────────────────────────────────────────────

def resample_4h_to_daily(df_4h: pd.DataFrame) -> pd.DataFrame:
    """Resample 4h candles vers daily OHLCV.

    Regles:
    - open  = open de la premiere candle 4h du jour (00:00 UTC)
    - high  = max(high) des 6 candles du jour
    - low   = min(low) des 6 candles du jour
    - close = close de la derniere candle 4h du jour (20:00 UTC)
    - volume = sum(volume)

    Filtre les jours incomplets (< 6 candles).
    """
    df = df_4h.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df["date"] = df["timestamp_utc"].dt.date

    daily = df.groupby("date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()

    # Filtrer les jours incomplets
    counts = df.groupby("date").size()
    complete_days = counts[counts >= 6].index
    n_before = len(daily)
    daily = daily[daily["date"].isin(complete_days)].reset_index(drop=True)
    n_filtered = n_before - len(daily)

    if n_filtered > 5:
        import warnings
        warnings.warn(
            f"resample_4h_to_daily: {n_filtered} jours incomplets filtres "
            f"(attendu <= 5 sur ~{n_before} jours)",
            stacklevel=2,
        )

    return daily


# ─── Helpers indicateurs (pandas-natifs) ──────────────────────────────────

def sma_series(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema_series(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def atr_series(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int,
) -> pd.Series:
    """Average True Range (Wilder)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()


def rolling_max_drawdown(close: pd.Series, window: int) -> pd.Series:
    """Drawdown glissant (%) sur une fenetre de N jours. Valeurs negatives."""
    result = pd.Series(0.0, index=close.index, dtype=float)
    close_vals = close.values
    for i in range(len(close_vals)):
        start = max(0, i - window)
        window_slice = close_vals[start:i + 1]
        if len(window_slice) < 2:
            continue
        peak = window_slice[0]
        max_dd = 0.0
        for val in window_slice:
            if val > peak:
                peak = val
            dd = (val - peak) / peak * 100 if peak > 0 else 0.0
            if dd < max_dd:
                max_dd = dd
        result.iloc[i] = max_dd
    return result


def realized_volatility(close: pd.Series, window: int) -> pd.Series:
    """Volatilite realisee annualisee (std des log-returns sur N jours)."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=window, min_periods=window).std() * math.sqrt(365)


# ─── Hysteresis (partagee) ────────────────────────────────────────────────

def apply_hysteresis(
    raw_labels: list[str],
    h_down: int,
    h_up: int,
) -> list[str]:
    """Hysteresis asymetrique sur les labels.

    h_down : candles pour transition vers etat PLUS severe (bull->bear, *->crash)
    h_up   : candles pour transition vers etat MOINS severe (bear->bull, crash->*)
    Severite : bull(0) < range(1) < bear(2) < crash(3)
    """
    if not raw_labels:
        return []

    result = [raw_labels[0]]
    current = raw_labels[0]
    counter = 0
    pending = None

    for i in range(1, len(raw_labels)):
        new = raw_labels[i]
        if new != current:
            if pending == new:
                counter += 1
            else:
                pending = new
                counter = 1

            new_sev = SEVERITY.get(new, 1)
            cur_sev = SEVERITY.get(current, 1)
            threshold = h_down if new_sev > cur_sev else h_up

            if counter >= threshold:
                current = new
                counter = 0
                pending = None
        else:
            counter = 0
            pending = None

        result.append(current)

    return result


# ─── DetectorResult ───────────────────────────────────────────────────────

@dataclass
class DetectorResult:
    """Resultat d'un run de detecteur."""
    labels_4h: list[str]
    labels_daily: list[str]
    raw_labels_daily: list[str]
    params: dict[str, Any]
    warmup_end_idx: int  # Index 4h de la premiere candle avec signal valide


# ─── Base Detector ────────────────────────────────────────────────────────

class BaseDetector(ABC):
    """Classe abstraite pour les detecteurs de regime."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def detect_raw(self, df_daily: pd.DataFrame, **params) -> tuple[list[str], int]:
        """Labels bruts sur daily (sans hysteresis).

        Retourne (labels, warmup_end_daily_idx).
        """
        ...

    @classmethod
    @abstractmethod
    def param_grid(cls) -> list[dict[str, Any]]:
        """Liste des combinaisons de parametres pour grid search."""
        ...

    def run(
        self,
        df_4h: pd.DataFrame,
        df_daily: pd.DataFrame,
        **params,
    ) -> DetectorResult:
        """Pipeline complet : detect_raw → remap 4h → hysteresis 4h."""
        h_down = params.pop("h_down", 12)
        h_up = params.pop("h_up", 36)

        # 1. Labels bruts sur daily
        raw_daily, warmup_daily_idx = self.detect_raw(df_daily, **params)

        # 2. Remap daily → 4h
        df_daily_copy = df_daily.copy()
        df_daily_copy["regime"] = raw_daily
        date_to_regime = dict(zip(df_daily_copy["date"], df_daily_copy["regime"]))

        df_4h_copy = df_4h.copy()
        df_4h_copy["date"] = pd.to_datetime(df_4h_copy["timestamp_utc"]).dt.date
        raw_4h = [date_to_regime.get(d, "range") for d in df_4h_copy["date"]]

        # 3. Hysteresis sur 4h
        smoothed_4h = apply_hysteresis(raw_4h, h_down, h_up)

        # 4. Remap smoothed daily (pour debug)
        smoothed_daily = apply_hysteresis(raw_daily, max(1, h_down // 6), max(1, h_up // 6))

        # 5. Warmup : convertir index daily → index 4h
        if warmup_daily_idx < len(df_daily):
            warmup_date = df_daily.iloc[warmup_daily_idx]["date"]
            warmup_4h_idx = 0
            for idx, d in enumerate(df_4h_copy["date"]):
                if d >= warmup_date:
                    warmup_4h_idx = idx
                    break
        else:
            warmup_4h_idx = len(df_4h)

        return DetectorResult(
            labels_4h=smoothed_4h,
            labels_daily=smoothed_daily,
            raw_labels_daily=raw_daily,
            params={"h_down": h_down, "h_up": h_up, **params},
            warmup_end_idx=warmup_4h_idx,
        )


# ─── Detecteur 1 : SMA + Stress rapide ───────────────────────────────────

class SMAStressDetector(BaseDetector):
    """SMA(N) tendance + drawdown glissant stress."""

    @property
    def name(self) -> str:
        return "sma_stress"

    def detect_raw(self, df_daily: pd.DataFrame, **params) -> tuple[list[str], int]:
        sma_period = params.get("sma_period", 200)
        stress_window = params.get("stress_window", 7)
        stress_threshold = params.get("stress_threshold", -20)

        close = df_daily["close"]
        sma = sma_series(close, sma_period)
        dd = rolling_max_drawdown(close, stress_window)

        labels = []
        warmup_idx = 0
        warmup_found = False

        for i in range(len(df_daily)):
            if pd.isna(sma.iloc[i]):
                labels.append("range")  # warmup
            else:
                if not warmup_found:
                    warmup_idx = i
                    warmup_found = True
                if dd.iloc[i] < stress_threshold:
                    labels.append("crash")
                elif close.iloc[i] > sma.iloc[i]:
                    labels.append("bull")
                else:
                    labels.append("bear")

        return labels, warmup_idx

    @classmethod
    def param_grid(cls) -> list[dict[str, Any]]:
        combos = itertools.product(
            [150, 200, 250],       # sma_period
            [5, 7, 10, 14],        # stress_window
            [-15, -20, -25],       # stress_threshold
            [6, 12, 18],           # h_down
            [24, 36, 48],          # h_up
        )
        return [
            dict(sma_period=c[0], stress_window=c[1], stress_threshold=c[2],
                 h_down=c[3], h_up=c[4])
            for c in combos
        ]


# ─── Detecteur 2 : EMA 50/200 + ATR Ratio ────────────────────────────────

class EMAATRDetector(BaseDetector):
    """EMA cross tendance + ATR ratio stress."""

    @property
    def name(self) -> str:
        return "ema_atr"

    def detect_raw(self, df_daily: pd.DataFrame, **params) -> tuple[list[str], int]:
        ema_fast = params.get("ema_fast", 50)
        ema_slow = params.get("ema_slow", 200)
        atr_fast_p = params.get("atr_fast", 7)
        atr_slow_p = params.get("atr_slow", 30)
        atr_stress_ratio = params.get("atr_stress_ratio", 2.0)

        close = df_daily["close"]
        high = df_daily["high"]
        low = df_daily["low"]

        ema_f = ema_series(close, ema_fast)
        ema_s = ema_series(close, ema_slow)
        atr_f = atr_series(high, low, close, atr_fast_p)
        atr_s = atr_series(high, low, close, atr_slow_p)

        labels = []
        warmup_idx = 0
        warmup_found = False

        for i in range(len(df_daily)):
            if pd.isna(ema_s.iloc[i]) or pd.isna(atr_s.iloc[i]):
                labels.append("range")
            else:
                if not warmup_found:
                    warmup_idx = i
                    warmup_found = True

                atr_ratio = atr_f.iloc[i] / atr_s.iloc[i] if atr_s.iloc[i] > 0 else 0
                if atr_ratio > atr_stress_ratio:
                    labels.append("crash")
                elif ema_f.iloc[i] > ema_s.iloc[i]:
                    labels.append("bull")
                else:
                    labels.append("bear")

        return labels, warmup_idx

    @classmethod
    def param_grid(cls) -> list[dict[str, Any]]:
        combos = itertools.product(
            [30, 50],               # ema_fast
            [150, 200],             # ema_slow
            [5, 7],                 # atr_fast
            [20, 30],               # atr_slow
            [1.5, 2.0, 2.5],       # atr_stress_ratio
            [6, 12, 18],            # h_down
            [24, 36, 48],           # h_up
        )
        return [
            dict(ema_fast=c[0], ema_slow=c[1], atr_fast=c[2], atr_slow=c[3],
                 atr_stress_ratio=c[4], h_down=c[5], h_up=c[6])
            for c in combos
        ]


# ─── Detecteur 3 : Multi-MA + Volatility Percentile ──────────────────────

class MultiMAVolDetector(BaseDetector):
    """3 etats tendance (bull/range/bear) + vol percentile stress.

    Limitation : sma_fast=50, sma_slow=200 fixes (pas explores dans la grille).
    A explorer en Sprint 50b si ce detecteur est retenu.
    """

    @property
    def name(self) -> str:
        return "multi_ma_vol"

    def detect_raw(self, df_daily: pd.DataFrame, **params) -> tuple[list[str], int]:
        sma_fast_p = 50   # fixe
        sma_slow_p = 200  # fixe
        vol_window = params.get("vol_window", 14)
        vol_percentile = params.get("vol_percentile", 95)
        vol_lookback = params.get("vol_lookback", 365)

        close = df_daily["close"]
        sma_f = sma_series(close, sma_fast_p)
        sma_s = sma_series(close, sma_slow_p)
        vol = realized_volatility(close, vol_window)

        labels = []
        warmup_idx = 0
        warmup_found = False

        for i in range(len(df_daily)):
            if pd.isna(sma_s.iloc[i]) or pd.isna(vol.iloc[i]):
                labels.append("range")
            else:
                if not warmup_found:
                    warmup_idx = i
                    warmup_found = True

                # Vol percentile historique (rolling lookback)
                lookback_start = max(0, i - vol_lookback)
                vol_history = vol.iloc[lookback_start:i + 1].dropna()
                if len(vol_history) > 10:
                    vol_thresh = np.percentile(vol_history.values, vol_percentile)
                    stress = vol.iloc[i] > vol_thresh
                else:
                    stress = False

                if stress:
                    labels.append("crash")
                elif close.iloc[i] > sma_f.iloc[i] and sma_f.iloc[i] > sma_s.iloc[i]:
                    labels.append("bull")
                elif close.iloc[i] < sma_f.iloc[i] and sma_f.iloc[i] < sma_s.iloc[i]:
                    labels.append("bear")
                else:
                    labels.append("range")

        return labels, warmup_idx

    @classmethod
    def param_grid(cls) -> list[dict[str, Any]]:
        combos = itertools.product(
            [7, 14, 21],            # vol_window
            [90, 95, 97],           # vol_percentile
            [180, 365],             # vol_lookback
            [6, 12],                # h_down
            [24, 36],               # h_up
        )
        return [
            dict(vol_window=c[0], vol_percentile=c[1], vol_lookback=c[2],
                 h_down=c[3], h_up=c[4])
            for c in combos
        ]


# ─── Metriques (pas de scikit-learn) ─────────────────────────────────────

def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Pourcentage de candles correctement classifiees."""
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def f1_per_class(
    y_true: list[str], y_pred: list[str], labels: list[str],
) -> dict[str, float]:
    """F1 score par classe (precision + recall)."""
    result = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)
        result[label] = f1
    return result


def confusion_matrix_manual(
    y_true: list[str], y_pred: list[str], labels: list[str],
) -> dict[str, dict[str, int]]:
    """Matrice de confusion comme dict de dicts."""
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
    return matrix


def crash_detection_delay(
    y_true: list[str],
    y_pred: list[str],
    candle_hours: float = 4.0,
) -> dict[str, Any]:
    """Mesure le delai de detection de chaque crash.

    Pour chaque debut de crash dans le ground truth, cherche la premiere
    candle crash/bear dans les predictions.
    """
    # Trouver les debuts de crash dans le ground truth
    crash_starts = []
    in_crash = False
    for i, label in enumerate(y_true):
        if label == "crash" and not in_crash:
            crash_starts.append(i)
            in_crash = True
        elif label != "crash":
            in_crash = False

    delays = []
    for start_idx in crash_starts:
        found = False
        for j in range(start_idx, min(start_idx + 150, len(y_pred))):
            if y_pred[j] == "crash":
                delays.append((j - start_idx) * candle_hours)
                found = True
                break
        if not found:
            delays.append(float("inf"))

    # Faux positifs : crashes predits sans crash ground truth
    false_positives = 0
    in_pred_crash = False
    for i, label in enumerate(y_pred):
        if label == "crash" and not in_pred_crash:
            in_pred_crash = True
            is_real = any(abs(i - cs) <= 30 for cs in crash_starts)
            if not is_real:
                false_positives += 1
        elif label != "crash":
            in_pred_crash = False

    finite_delays = [d for d in delays if d != float("inf")]
    return {
        "n_crashes_gt": len(crash_starts),
        "n_detected": len(finite_delays),
        "avg_delay_hours": (sum(finite_delays) / len(finite_delays)
                            if finite_delays else float("inf")),
        "max_delay_hours": max(finite_delays) if finite_delays else float("inf"),
        "delays_hours": delays,
        "false_positives": false_positives,
    }


def n_transitions(labels: list[str]) -> int:
    """Nombre de changements de regime."""
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])


def avg_regime_duration(
    labels: list[str], candle_hours: float = 4.0,
) -> dict[str, float]:
    """Duree moyenne de chaque regime en heures."""
    durations: dict[str, list[float]] = {r: [] for r in LABELS}
    for regime, group in groupby(labels):
        length = sum(1 for _ in group) * candle_hours
        if regime in durations:
            durations[regime].append(length)
    return {
        r: (sum(d) / len(d) if d else 0.0)
        for r, d in durations.items()
    }


def stability_score(labels: list[str], min_duration_candles: int = 6) -> float:
    """Fraction de segments qui durent >= min_duration_candles (1 jour a 4h)."""
    segments = [(regime, sum(1 for _ in group)) for regime, group in groupby(labels)]
    if not segments:
        return 0.0
    stable = sum(1 for _, length in segments if length >= min_duration_candles)
    return stable / len(segments)


def regime_distribution(labels: list[str]) -> dict[str, float]:
    """Pourcentage de temps dans chaque regime."""
    total = len(labels) if labels else 1
    return {r: labels.count(r) / total for r in LABELS}


# ─── Registre des detecteurs ─────────────────────────────────────────────

ALL_DETECTORS: list[BaseDetector] = [
    SMAStressDetector(),
    EMAATRDetector(),
    MultiMAVolDetector(),
]


# ─── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Regime Detectors — Sprint 50a ===")
    for det in ALL_DETECTORS:
        grid = det.param_grid()
        print(f"  {det.name}: {len(grid)} combinaisons")
    total = sum(len(d.param_grid()) for d in ALL_DETECTORS)
    print(f"  Total: {total} combinaisons")
