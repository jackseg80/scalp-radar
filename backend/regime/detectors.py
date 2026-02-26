"""Détecteurs de régime BTC — copie sélective depuis scripts/regime_detectors.py.

Seul EMAATRDetector + helpers binaires sont inclus ici pour éviter
l'import fragile scripts → backend.

Source: scripts/regime_detectors.py (Sprint 50a / 50a-bis)
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# ─── Sévérité des régimes ─────────────────────────────────────────────────

SEVERITY = {"bull": 0, "range": 1, "bear": 2, "crash": 3}
LABELS = ["bull", "bear", "range", "crash"]

# ─── Classification binaire ──────────────────────────────────────────────

BINARY_NORMAL = {"bull", "range"}
BINARY_DEFENSIVE = {"bear", "crash"}
BINARY_LABELS = ["normal", "defensive"]


def to_binary_labels(labels: list[str]) -> list[str]:
    """Regroupe 4 classes -> 2 classes binaires.

    bull + range -> "normal", bear + crash -> "defensive".
    """
    return [
        "defensive" if lbl in BINARY_DEFENSIVE else "normal"
        for lbl in labels
    ]


# ─── Hysteresis ──────────────────────────────────────────────────────────

def apply_hysteresis(
    raw_labels: list[str],
    h_down: int,
    h_up: int,
) -> list[str]:
    """Hysteresis asymétrique sur les labels.

    h_down : candles pour transition vers état PLUS sévère
    h_up   : candles pour transition vers état MOINS sévère
    Sévérité : bull(0) < range(1) < bear(2) < crash(3)
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


# ─── Helpers indicateurs ─────────────────────────────────────────────────

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


def resample_4h_to_daily(df_4h: pd.DataFrame) -> pd.DataFrame:
    """Resample 4h candles vers daily OHLCV.

    Règles:
    - open  = open de la première candle 4h du jour (00:00 UTC)
    - high  = max(high) des 6 candles du jour
    - low   = min(low) des 6 candles du jour
    - close = close de la dernière candle 4h du jour (20:00 UTC)
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
            f"resample_4h_to_daily: {n_filtered} jours incomplets filtrés "
            f"(attendu <= 5 sur ~{n_before} jours)",
            stacklevel=2,
        )

    return daily


# ─── DetectorResult ──────────────────────────────────────────────────────

@dataclass
class DetectorResult:
    """Résultat d'un run de détecteur."""
    labels_4h: list[str]
    labels_daily: list[str]
    raw_labels_daily: list[str]
    params: dict[str, Any]
    warmup_end_idx: int


# ─── Base Detector ───────────────────────────────────────────────────────

class BaseDetector(ABC):
    """Classe abstraite pour les détecteurs de régime."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def detect_raw(self, df_daily: pd.DataFrame, **params) -> tuple[list[str], int]:
        """Labels bruts sur daily (sans hysteresis)."""
        ...

    @classmethod
    @abstractmethod
    def param_grid(cls) -> list[dict[str, Any]]:
        """Liste des combinaisons de paramètres pour grid search."""
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


# ─── EMA/ATR Detector ───────────────────────────────────────────────────

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
            [20, 30],              # atr_slow
            [1.5, 2.0, 2.5],      # atr_stress_ratio
            [6, 12, 18],           # h_down
            [24, 36, 48],          # h_up
        )
        return [
            dict(ema_fast=c[0], ema_slow=c[1], atr_fast=c[2], atr_slow=c[3],
                 atr_stress_ratio=c[4], h_down=c[5], h_up=c[6])
            for c in combos
        ]
