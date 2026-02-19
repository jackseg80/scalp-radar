"""Indicateurs incrémentaux pour le Simulator (paper trading live).

Maintient des buffers rolling par (symbol, timeframe).
Recalcule les indicateurs sur la fenêtre complète à chaque nouvelle candle.
Sur 300-500 floats, c'est < 1ms — pas besoin de variantes incrémentales.

IMPORTANT : Toutes les fonctions d'indicateurs sont en **pur Python** (zéro
allocation numpy) pour éviter les segfaults liés à la corruption heap sur
CPython 3.13 + numpy 2.3.5 + Windows. Sur des backtests longs (>100K appels),
les millions d'allocations/libérations numpy finissent par corrompre le heap
→ access violation aléatoire dans les boucles Wilder.
Pur Python sur 500 éléments ≈ même performance (<1ms) sans aucun risque.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from backend.core.models import Candle, MarketRegime
from backend.strategies.base import BaseStrategy


class IncrementalIndicatorEngine:
    """Buffers rolling par (symbol, timeframe).

    Recalcule les indicateurs sur la fenêtre complète à chaque update.
    Utilisé par le Simulator (paper trading live) et le Portfolio Backtest.
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

    # ------------------------------------------------------------------
    # Calcul des indicateurs — pur Python, zéro numpy
    # ------------------------------------------------------------------

    def _compute_latest(
        self,
        candles: list[Candle],
        timeframe: str = "",
        _key: tuple[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """Calcule les indicateurs sur le buffer complet, retourne le dernier point.

        Pur Python — aucune allocation numpy. Les paramètres ``timeframe``
        et ``_key`` sont conservés pour compatibilité API mais ignorés.
        """
        n = len(candles)
        if n < 2:
            return None

        rsi_val = self._rsi_last(candles)
        adx_val, di_plus_val, di_minus_val = self._adx_last(candles)
        atr_val, atr_sma_val = self._atr_last(candles)
        vwap_val = self._vwap_last(candles)
        vol_sma_val = self._vol_sma_last(candles)

        regime = _detect_regime(
            adx_val, di_plus_val, di_minus_val, atr_val, atr_sma_val
        )

        return {
            "rsi": rsi_val,
            "vwap": vwap_val,
            "adx": adx_val,
            "di_plus": di_plus_val,
            "di_minus": di_minus_val,
            "atr": atr_val,
            "atr_sma": atr_sma_val,
            "volume": candles[-1].volume,
            "volume_sma": vol_sma_val,
            "close": candles[-1].close,
            "regime": regime,
        }

    # ── RSI (Wilder smoothing) ────────────────────────────────────────

    @staticmethod
    def _rsi_last(candles: list[Candle], period: int = 14) -> float:
        """RSI avec lissage de Wilder — retourne uniquement la dernière valeur."""
        n = len(candles)
        if n < period + 2:
            return float("nan")

        # Deltas, gains, losses
        avg_gain = 0.0
        avg_loss = 0.0
        for i in range(1, period + 1):
            delta = candles[i].close - candles[i - 1].close
            if delta > 0:
                avg_gain += delta
            else:
                avg_loss -= delta
        avg_gain /= period
        avg_loss /= period

        # Wilder smoothing
        for i in range(period + 1, n):
            delta = candles[i].close - candles[i - 1].close
            avg_gain = (avg_gain * (period - 1) + max(delta, 0.0)) / period
            avg_loss = (avg_loss * (period - 1) + max(-delta, 0.0)) / period

        if avg_loss == 0.0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    # ── ATR (Wilder smoothing) + ATR SMA ──────────────────────────────

    @staticmethod
    def _atr_last(
        candles: list[Candle], period: int = 14, sma_period: int = 20
    ) -> tuple[float, float]:
        """ATR Wilder + SMA de l'ATR — retourne (atr, atr_sma)."""
        n = len(candles)
        if n < period + 2:
            return float("nan"), float("nan")

        # True Range (depuis index 1)
        tr_list: list[float] = []
        for i in range(1, n):
            h = candles[i].high
            l_val = candles[i].low
            pc = candles[i - 1].close
            tr = max(h - l_val, abs(h - pc), abs(l_val - pc))
            tr_list.append(tr)

        m = len(tr_list)
        if m < period:
            return float("nan"), float("nan")

        # Seed : moyenne simple des period premiers TRs
        atr_val = sum(tr_list[:period]) / period
        atr_history: list[float] = [atr_val]

        # Wilder smoothing
        for i in range(period, m):
            atr_val = (atr_val * (period - 1) + tr_list[i]) / period
            atr_history.append(atr_val)

        # ATR SMA sur les dernières valeurs ATR valides
        atr_sma_val = float("nan")
        if len(atr_history) >= sma_period:
            atr_sma_val = sum(atr_history[-sma_period:]) / sma_period

        return atr_val, atr_sma_val

    # ── ADX + DI+/DI- (Wilder smoothing) ─────────────────────────────

    @staticmethod
    def _adx_last(candles: list[Candle], period: int = 14) -> tuple[float, float, float]:
        """ADX + DI+/DI- Wilder — retourne (adx, di_plus, di_minus)."""
        n = len(candles)
        if n < 2 * period + 2:
            return float("nan"), float("nan"), float("nan")

        # DM+, DM-, TR (index 0 = placeholder)
        dm_plus: list[float] = [0.0]
        dm_minus: list[float] = [0.0]
        tr_list: list[float] = [candles[0].high - candles[0].low]

        for i in range(1, n):
            h = candles[i].high
            l_val = candles[i].low
            ph = candles[i - 1].high
            pl = candles[i - 1].low
            pc = candles[i - 1].close

            up = h - ph
            down = pl - l_val

            dm_plus.append(up if (up > down and up > 0) else 0.0)
            dm_minus.append(down if (down > up and down > 0) else 0.0)
            tr_list.append(max(h - l_val, abs(h - pc), abs(l_val - pc)))

        # Seeds : somme des period premières valeurs (indices 1..period)
        sm_tr = 0.0
        sm_plus = 0.0
        sm_minus = 0.0
        for j in range(1, period + 1):
            sm_tr += tr_list[j]
            sm_plus += dm_plus[j]
            sm_minus += dm_minus[j]

        # DI+/DI- au premier index
        di_plus = 100.0 * sm_plus / sm_tr if sm_tr > 0 else 0.0
        di_minus = 100.0 * sm_minus / sm_tr if sm_tr > 0 else 0.0
        di_sum = di_plus + di_minus
        dx_values: list[float] = [
            100.0 * abs(di_plus - di_minus) / di_sum if di_sum > 0 else 0.0
        ]

        # Wilder smoothing DI+/DI- et accumulation DX
        for i in range(period + 1, n):
            sm_tr = sm_tr - sm_tr / period + tr_list[i]
            sm_plus = sm_plus - sm_plus / period + dm_plus[i]
            sm_minus = sm_minus - sm_minus / period + dm_minus[i]

            di_plus = 100.0 * sm_plus / sm_tr if sm_tr > 0 else 0.0
            di_minus = 100.0 * sm_minus / sm_tr if sm_tr > 0 else 0.0

            di_sum = di_plus + di_minus
            dx = 100.0 * abs(di_plus - di_minus) / di_sum if di_sum > 0 else 0.0
            dx_values.append(dx)

        # ADX = Wilder smoothed DX
        if len(dx_values) < period:
            return float("nan"), di_plus, di_minus

        adx_val = sum(dx_values[:period]) / period
        for i in range(period, len(dx_values)):
            adx_val = (adx_val * (period - 1) + dx_values[i]) / period

        return adx_val, di_plus, di_minus

    # ── VWAP Rolling ──────────────────────────────────────────────────

    @staticmethod
    def _vwap_last(candles: list[Candle], window: int = 288) -> float:
        """VWAP rolling — retourne uniquement la dernière valeur."""
        n = len(candles)
        if n < window:
            return float("nan")

        tp_vol_sum = 0.0
        vol_sum = 0.0
        for i in range(n - window, n):
            c = candles[i]
            tp = (c.high + c.low + c.close) / 3.0
            tp_vol_sum += tp * c.volume
            vol_sum += c.volume

        if vol_sum > 0:
            return tp_vol_sum / vol_sum
        return (candles[-1].high + candles[-1].low + candles[-1].close) / 3.0

    # ── Volume SMA ────────────────────────────────────────────────────

    @staticmethod
    def _vol_sma_last(candles: list[Candle], period: int = 20) -> float:
        """Volume SMA — retourne uniquement la dernière valeur."""
        n = len(candles)
        if n < period:
            return float("nan")
        return sum(candles[i].volume for i in range(n - period, n)) / period

    # ── Utilitaires ───────────────────────────────────────────────────

    def get_buffer_sizes(self) -> dict[tuple[str, str], int]:
        """Retourne la taille de chaque buffer (debug/monitoring)."""
        return {key: len(buf) for key, buf in self._buffers.items()}


# ─── Détection de régime de marché (pur Python) ──────────────────────────


def _detect_regime(
    adx_val: float,
    di_plus: float,
    di_minus: float,
    atr_val: float,
    atr_sma: float,
) -> MarketRegime:
    """Détecte le régime de marché à partir des valeurs scalaires.

    Pur Python — utilise math.isnan au lieu de np.isnan.
    Même logique que indicators.detect_market_regime().
    """
    for v in (adx_val, di_plus, di_minus, atr_val, atr_sma):
        if math.isnan(v):
            return MarketRegime.RANGING

    if atr_sma > 0:
        if atr_val > 2.0 * atr_sma:
            return MarketRegime.HIGH_VOLATILITY
        if atr_val < 0.5 * atr_sma:
            return MarketRegime.LOW_VOLATILITY

    if adx_val > 25:
        if di_plus > di_minus:
            return MarketRegime.TRENDING_UP
        return MarketRegime.TRENDING_DOWN

    return MarketRegime.RANGING
