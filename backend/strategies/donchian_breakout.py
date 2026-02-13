"""Stratégie Donchian Breakout pour Scalp Radar.

Trade les cassures du canal Donchian (plus haut/bas des N dernières bougies).
TP et SL basés sur des multiples d'ATR.
Timeframe : 1h.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from backend.core.config import DonchianBreakoutConfig
from backend.core.indicators import (
    adx,
    atr,
    detect_market_regime,
    sma,
)
from backend.core.models import Candle, Direction, MarketRegime, SignalStrength
from backend.strategies.base import BaseStrategy, OpenPosition, StrategyContext, StrategySignal


class DonchianBreakoutStrategy(BaseStrategy):
    """Donchian Channel Breakout.

    Entry LONG : close > plus haut des N dernières bougies.
    Entry SHORT : close < plus bas des N dernières bougies.
    TP : ATR × atr_tp_multiple.
    SL : ATR × atr_sl_multiple.
    """

    name = "donchian_breakout"

    def __init__(self, config: DonchianBreakoutConfig) -> None:
        self._config = config

    @property
    def min_candles(self) -> dict[str, int]:
        return {
            self._config.timeframe: max(self._config.entry_lookback + 20, 50),
        }

    def compute_indicators(
        self, candles_by_tf: dict[str, list[Candle]]
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Pré-calcule rolling high/low, ATR, ADX."""
        result: dict[str, dict[str, dict[str, float]]] = {}

        main_tf = self._config.timeframe
        if main_tf in candles_by_tf and candles_by_tf[main_tf]:
            result[main_tf] = self._compute_main(candles_by_tf[main_tf])

        return result

    def _compute_main(self, candles: list[Candle]) -> dict[str, dict[str, float]]:
        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)

        lookback = self._config.entry_lookback
        atr_period = self._config.atr_period

        atr_arr = atr(highs, lows, closes, atr_period)
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

        # Rolling high/low (exclut le candle courant)
        rolling_high = np.full(len(closes), np.nan)
        rolling_low = np.full(len(closes), np.nan)
        for i in range(lookback, len(closes)):
            rolling_high[i] = np.max(highs[i - lookback:i])
            rolling_low[i] = np.min(lows[i - lookback:i])

        indicators: dict[str, dict[str, float]] = {}
        for i, candle in enumerate(candles):
            ts = candle.timestamp.isoformat()
            indicators[ts] = {
                "close": float(closes[i]),
                "atr": float(atr_arr[i]),
                "atr_sma": float(atr_sma_full[i]),
                "adx": float(adx_arr[i]),
                "di_plus": float(di_plus_arr[i]),
                "di_minus": float(di_minus_arr[i]),
                "rolling_high": float(rolling_high[i]),
                "rolling_low": float(rolling_low[i]),
            }

        return indicators

    def evaluate(self, ctx: StrategyContext) -> StrategySignal | None:
        """Évalue les conditions d'entrée Donchian Breakout."""
        main_tf = self._config.timeframe
        main_ind = ctx.indicators.get(main_tf)
        if not main_ind:
            return None

        close = main_ind.get("close", float("nan"))
        atr_val = main_ind.get("atr", float("nan"))
        atr_sma_val = main_ind.get("atr_sma", float("nan"))
        adx_val = main_ind.get("adx", float("nan"))
        di_plus = main_ind.get("di_plus", float("nan"))
        di_minus = main_ind.get("di_minus", float("nan"))
        rolling_high = main_ind.get("rolling_high", float("nan"))
        rolling_low = main_ind.get("rolling_low", float("nan"))

        if any(np.isnan(v) for v in [close, atr_val, rolling_high, rolling_low]):
            return None

        # Conditions d'entrée
        long_breakout = close > rolling_high
        short_breakout = close < rolling_low

        if not long_breakout and not short_breakout:
            return None

        direction = Direction.LONG if long_breakout else Direction.SHORT

        # Régime de marché
        regime = detect_market_regime(adx_val, di_plus, di_minus, atr_val, atr_sma_val)

        # TP/SL basé sur ATR multiples
        atr_tp_mult = self._resolve_param("atr_tp_multiple", ctx.symbol)
        atr_sl_mult = self._resolve_param("atr_sl_multiple", ctx.symbol)

        tp_distance = atr_val * atr_tp_mult
        sl_distance = atr_val * atr_sl_mult

        if direction == Direction.LONG:
            tp_price = close + tp_distance
            sl_price = close - sl_distance
        else:
            tp_price = close - tp_distance
            sl_price = close + sl_distance

        # Score basé sur force du breakout
        channel_width = rolling_high - rolling_low
        if channel_width > 0:
            if long_breakout:
                breakout_pct = (close - rolling_high) / channel_width
            else:
                breakout_pct = (rolling_low - close) / channel_width
            breakout_score = min(1.0, breakout_pct * 5)
        else:
            breakout_score = 0.5

        score = max(0.1, min(1.0, 0.5 + breakout_score * 0.5))

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
                "breakout_score": breakout_score,
                "channel_width": channel_width,
                "atr": atr_val,
                "rolling_high": rolling_high,
                "rolling_low": rolling_low,
            },
        )

    def get_current_conditions(self, ctx: StrategyContext) -> list[dict]:
        """Conditions d'entrée Donchian pour le dashboard."""
        main_tf = self._config.timeframe
        main_ind = ctx.indicators.get(main_tf, {})

        close = main_ind.get("close", float("nan"))
        rolling_high = main_ind.get("rolling_high", float("nan"))
        rolling_low = main_ind.get("rolling_low", float("nan"))
        atr_val = main_ind.get("atr", float("nan"))

        is_breakout = False
        position_str = "inside"
        if not any(np.isnan(v) for v in [close, rolling_high, rolling_low]):
            if close > rolling_high:
                is_breakout = True
                position_str = "above"
            elif close < rolling_low:
                is_breakout = True
                position_str = "below"

        return [
            {
                "name": "channel_position",
                "met": is_breakout,
                "value": position_str,
                "threshold": "outside",
            },
            {
                "name": "atr_available",
                "met": not np.isnan(atr_val) and atr_val > 0,
                "value": round(atr_val, 2) if not np.isnan(atr_val) else None,
                "threshold": "> 0",
            },
        ]

    def check_exit(self, ctx: StrategyContext, position: OpenPosition) -> str | None:
        """Pas de sortie anticipée — TP/SL ATR gèrent tout."""
        return None

    def get_params(self) -> dict:
        return {
            "timeframe": self._config.timeframe,
            "entry_lookback": self._config.entry_lookback,
            "atr_period": self._config.atr_period,
            "atr_tp_multiple": self._config.atr_tp_multiple,
            "atr_sl_multiple": self._config.atr_sl_multiple,
        }
