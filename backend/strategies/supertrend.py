"""Stratégie SuperTrend pour Scalp Radar.

Trade les retournements de tendance détectés par l'indicateur SuperTrend.
Entrée sur flip de direction. TP et SL en % fixe.
Timeframe : 1h.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from backend.core.config import SuperTrendConfig
from backend.core.indicators import (
    adx,
    atr,
    detect_market_regime,
    sma,
    supertrend,
)
from backend.core.models import Candle, Direction, MarketRegime, SignalStrength
from backend.strategies.base import BaseStrategy, OpenPosition, StrategyContext, StrategySignal


class SuperTrendStrategy(BaseStrategy):
    """SuperTrend Flip.

    Entry LONG : direction flip de DOWN (-1) à UP (1).
    Entry SHORT : direction flip de UP (1) à DOWN (-1).
    TP : % fixe.
    SL : % fixe.
    """

    name = "supertrend"

    def __init__(self, config: SuperTrendConfig) -> None:
        self._config = config

    @property
    def min_candles(self) -> dict[str, int]:
        return {
            self._config.timeframe: max(self._config.atr_period + 20, 50),
        }

    def compute_indicators(
        self, candles_by_tf: dict[str, list[Candle]]
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Pré-calcule ATR, SuperTrend, ADX."""
        result: dict[str, dict[str, dict[str, float]]] = {}

        main_tf = self._config.timeframe
        if main_tf in candles_by_tf and candles_by_tf[main_tf]:
            result[main_tf] = self._compute_main(candles_by_tf[main_tf])

        return result

    def _compute_main(self, candles: list[Candle]) -> dict[str, dict[str, float]]:
        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)

        atr_period = self._config.atr_period
        multiplier = self._config.atr_multiplier

        atr_arr = atr(highs, lows, closes, atr_period)
        st_values, st_direction = supertrend(highs, lows, closes, atr_arr, multiplier)

        # ADX pour le régime
        adx_arr, di_plus_arr, di_minus_arr = adx(highs, lows, closes)

        # ATR SMA aligné
        atr_sma_full = np.full_like(atr_arr, np.nan)
        valid_atr = ~np.isnan(atr_arr)
        valid_indices = np.where(valid_atr)[0]
        if len(valid_indices) >= 20:
            atr_valid = atr_arr[valid_atr]
            atr_sma_valid = sma(atr_valid, 20)
            for j, idx in enumerate(valid_indices):
                if not np.isnan(atr_sma_valid[j]):
                    atr_sma_full[idx] = atr_sma_valid[j]

        indicators: dict[str, dict[str, float]] = {}
        for i, candle in enumerate(candles):
            ts = candle.timestamp.isoformat()
            # Direction précédente pour détecter les flips
            prev_dir = float(st_direction[i - 1]) if i > 0 else float("nan")
            indicators[ts] = {
                "close": float(closes[i]),
                "st_value": float(st_values[i]),
                "st_direction": float(st_direction[i]),
                "st_prev_direction": prev_dir,
                "atr": float(atr_arr[i]),
                "atr_sma": float(atr_sma_full[i]),
                "adx": float(adx_arr[i]),
                "di_plus": float(di_plus_arr[i]),
                "di_minus": float(di_minus_arr[i]),
            }

        return indicators

    def evaluate(self, ctx: StrategyContext) -> StrategySignal | None:
        """Évalue les conditions d'entrée SuperTrend."""
        main_tf = self._config.timeframe
        main_ind = ctx.indicators.get(main_tf)
        if not main_ind:
            return None

        close = main_ind.get("close", float("nan"))
        st_direction = main_ind.get("st_direction", float("nan"))
        prev_direction = main_ind.get("st_prev_direction", float("nan"))
        st_value = main_ind.get("st_value", float("nan"))
        atr_val = main_ind.get("atr", float("nan"))
        atr_sma_val = main_ind.get("atr_sma", float("nan"))
        adx_val = main_ind.get("adx", float("nan"))
        di_plus = main_ind.get("di_plus", float("nan"))
        di_minus = main_ind.get("di_minus", float("nan"))

        if any(np.isnan(v) for v in [close, st_direction, prev_direction]):
            return None

        # Flip detection
        long_flip = prev_direction == -1.0 and st_direction == 1.0
        short_flip = prev_direction == 1.0 and st_direction == -1.0

        if not long_flip and not short_flip:
            return None

        direction = Direction.LONG if long_flip else Direction.SHORT

        # Régime de marché
        regime = detect_market_regime(adx_val, di_plus, di_minus, atr_val, atr_sma_val)

        # TP/SL % fixe
        tp_pct = self._resolve_param("tp_percent", ctx.symbol)
        sl_pct = self._resolve_param("sl_percent", ctx.symbol)

        if direction == Direction.LONG:
            tp_price = close * (1 + tp_pct / 100)
            sl_price = close * (1 - sl_pct / 100)
        else:
            tp_price = close * (1 - tp_pct / 100)
            sl_price = close * (1 + sl_pct / 100)

        # Score basé sur distance au SuperTrend + ATR
        if not np.isnan(st_value) and st_value > 0:
            distance_to_st = abs(close - st_value) / close * 100
            distance_score = min(1.0, distance_to_st / 3.0)
        else:
            distance_score = 0.5

        score = max(0.1, min(1.0, 0.5 + distance_score * 0.5))

        if score >= 0.7:
            strength = SignalStrength.STRONG
        elif score >= 0.4:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        return StrategySignal(
            direction=direction,
            entry_price=close,
            tp_price=tp_price,
            sl_price=sl_price,
            score=score,
            strength=strength,
            market_regime=regime,
            signals_detail={
                "distance_score": distance_score,
                "st_value": st_value if not np.isnan(st_value) else 0.0,
                "st_direction": st_direction,
                "prev_direction": prev_direction,
                "atr": atr_val if not np.isnan(atr_val) else 0.0,
            },
        )

    def get_current_conditions(self, ctx: StrategyContext) -> list[dict]:
        """Conditions d'entrée SuperTrend pour le dashboard."""
        main_tf = self._config.timeframe
        main_ind = ctx.indicators.get(main_tf, {})

        st_direction = main_ind.get("st_direction", float("nan"))
        prev_direction = main_ind.get("st_prev_direction", float("nan"))
        close = main_ind.get("close", float("nan"))
        st_value = main_ind.get("st_value", float("nan"))

        is_flip = False
        dir_str = "unknown"
        if not np.isnan(st_direction):
            dir_str = "UP" if st_direction == 1.0 else "DOWN"
            if not np.isnan(prev_direction) and prev_direction != st_direction:
                is_flip = True

        distance_pct = float("nan")
        if not any(np.isnan(v) for v in [close, st_value]) and close > 0:
            distance_pct = abs(close - st_value) / close * 100

        return [
            {
                "name": "st_direction",
                "met": is_flip,
                "value": dir_str,
                "threshold": "flip",
            },
            {
                "name": "distance_to_st",
                "met": not np.isnan(distance_pct),
                "value": round(distance_pct, 2) if not np.isnan(distance_pct) else None,
                "threshold": "any",
            },
        ]

    def check_exit(self, ctx: StrategyContext, position: OpenPosition) -> str | None:
        """Pas de sortie anticipée — TP/SL fixes gèrent tout."""
        return None

    def get_params(self) -> dict:
        return {
            "timeframe": self._config.timeframe,
            "atr_period": self._config.atr_period,
            "atr_multiplier": self._config.atr_multiplier,
            "tp_percent": self._config.tp_percent,
            "sl_percent": self._config.sl_percent,
        }
