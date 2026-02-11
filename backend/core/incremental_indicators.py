"""Indicateurs incrémentaux pour le Simulator (paper trading live).

Maintient des buffers numpy rolling par (symbol, timeframe).
Recalcule les indicateurs sur la fenêtre complète à chaque nouvelle candle.
Sur 300-500 floats, c'est < 1ms — pas besoin de variantes incrémentales.

Réutilise les mêmes fonctions que le backtest (backend/core/indicators.py).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from backend.core.indicators import (
    adx,
    atr,
    detect_market_regime,
    rsi,
    sma,
    volume_sma,
    vwap_rolling,
)
from backend.core.models import Candle
from backend.strategies.base import BaseStrategy


class IncrementalIndicatorEngine:
    """Buffers numpy rolling par (symbol, timeframe).

    Recalcule les indicateurs sur la fenêtre complète à chaque update.
    Utilisé par le Simulator pour le paper trading live.
    """

    def __init__(self, strategies: list[BaseStrategy], max_buffer: int = 500) -> None:
        # Déduire la taille max de buffer par TF depuis les stratégies
        self._max_per_tf: dict[str, int] = {}
        for strat in strategies:
            for tf, needed in strat.min_candles.items():
                current = self._max_per_tf.get(tf, 0)
                self._max_per_tf[tf] = max(current, needed, max_buffer)

        # Buffers : {(symbol, tf): list[Candle]}
        self._buffers: dict[tuple[str, str], list[Candle]] = defaultdict(list)

        # Timeframes connus (union de toutes les stratégies)
        self._timeframes: set[str] = set(self._max_per_tf.keys())

    @property
    def timeframes(self) -> set[str]:
        """Timeframes gérés par cet engine."""
        return self._timeframes

    def update(self, symbol: str, timeframe: str, candle: Candle) -> None:
        """Ajoute une bougie au buffer et trim si nécessaire."""
        key = (symbol, timeframe)
        buf = self._buffers[key]

        # Éviter les doublons (même timestamp)
        if buf and buf[-1].timestamp >= candle.timestamp:
            return

        buf.append(candle)

        # Trim au max
        max_size = self._max_per_tf.get(timeframe, 500)
        if len(buf) > max_size:
            self._buffers[key] = buf[-max_size:]

    def get_indicators(self, symbol: str) -> dict[str, dict[str, Any]]:
        """Recalcule et retourne les indicateurs pour tous les TF d'un symbol.

        Retourne {"5m": {"rsi": 23.5, "vwap": 98500, ...}, "15m": {"rsi": 45, ...}}
        Chaque valeur est le dernier point (la valeur la plus récente).
        """
        result: dict[str, dict[str, Any]] = {}

        for tf in self._timeframes:
            key = (symbol, tf)
            buf = self._buffers.get(key)
            if not buf or len(buf) < 2:
                continue

            indicators = self._compute_latest(buf, tf)
            if indicators:
                result[tf] = indicators

        return result

    def _compute_latest(
        self, candles: list[Candle], timeframe: str
    ) -> dict[str, Any] | None:
        """Calcule les indicateurs sur le buffer complet, retourne le dernier point."""
        n = len(candles)
        if n < 2:
            return None

        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)

        # RSI
        rsi_arr = rsi(closes)
        rsi_val = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else float("nan")

        # ADX + DI
        adx_arr, di_plus_arr, di_minus_arr = adx(highs, lows, closes)
        adx_val = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else float("nan")
        di_plus_val = float(di_plus_arr[-1]) if not np.isnan(di_plus_arr[-1]) else float("nan")
        di_minus_val = float(di_minus_arr[-1]) if not np.isnan(di_minus_arr[-1]) else float("nan")

        # ATR + ATR SMA
        atr_arr = atr(highs, lows, closes)
        atr_val = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else float("nan")

        # ATR SMA aligné
        valid_atr = atr_arr[~np.isnan(atr_arr)]
        atr_sma_val = float("nan")
        if len(valid_atr) >= 20:
            atr_sma_arr = sma(valid_atr, 20)
            if not np.isnan(atr_sma_arr[-1]):
                atr_sma_val = float(atr_sma_arr[-1])

        # VWAP rolling
        vwap_arr = vwap_rolling(highs, lows, closes, volumes)
        vwap_val = float(vwap_arr[-1]) if not np.isnan(vwap_arr[-1]) else float("nan")

        # Volume SMA
        vol_sma_arr = volume_sma(volumes)
        vol_sma_val = float(vol_sma_arr[-1]) if not np.isnan(vol_sma_arr[-1]) else float("nan")

        # Régime
        regime = detect_market_regime(adx_val, di_plus_val, di_minus_val, atr_val, atr_sma_val)

        return {
            "rsi": rsi_val,
            "vwap": vwap_val,
            "adx": adx_val,
            "di_plus": di_plus_val,
            "di_minus": di_minus_val,
            "atr": atr_val,
            "atr_sma": atr_sma_val,
            "volume": float(volumes[-1]),
            "volume_sma": vol_sma_val,
            "close": float(closes[-1]),
            "regime": regime,
        }

    def get_buffer_sizes(self) -> dict[tuple[str, str], int]:
        """Retourne la taille de chaque buffer (debug/monitoring)."""
        return {key: len(buf) for key, buf in self._buffers.items()}
