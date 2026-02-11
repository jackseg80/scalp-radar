"""Stratégie Momentum Breakout pour Scalp Radar.

Trade AVEC la tendance (complémentaire à VWAP+RSI qui trade CONTRE).
Logique : prix casse le max/min des N dernières bougies avec volume et ADX 15m en tendance.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from backend.core.config import MomentumConfig
from backend.core.indicators import (
    adx,
    atr,
    detect_market_regime,
    rsi,
    sma,
    volume_sma,
)
from backend.core.models import Candle, Direction, MarketRegime, SignalStrength
from backend.strategies.base import BaseStrategy, OpenPosition, StrategyContext, StrategySignal


class MomentumStrategy(BaseStrategy):
    """Momentum Breakout.

    Entry LONG : prix casse le high des N dernières bougies + volume spike + ADX 15m trending bullish.
    Entry SHORT : symétrique (casse le low + ADX 15m trending bearish).
    Exit anticipée : ADX chute sous 20 → momentum essoufflé.
    """

    name = "momentum"

    def __init__(self, config: MomentumConfig) -> None:
        self._config = config

    @property
    def min_candles(self) -> dict[str, int]:
        return {
            self._config.timeframe: max(self._config.breakout_lookback + 50, 100),
            self._config.trend_filter_timeframe: 50,
        }

    def compute_indicators(
        self, candles_by_tf: dict[str, list[Candle]]
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Pré-calcule ATR, ADX+DI, volume SMA, rolling high/low."""
        result: dict[str, dict[str, dict[str, float]]] = {}

        main_tf = self._config.timeframe
        if main_tf in candles_by_tf and candles_by_tf[main_tf]:
            result[main_tf] = self._compute_main(candles_by_tf[main_tf])

        filter_tf = self._config.trend_filter_timeframe
        if filter_tf in candles_by_tf and candles_by_tf[filter_tf]:
            result[filter_tf] = self._compute_filter(candles_by_tf[filter_tf])

        return result

    def _compute_main(self, candles: list[Candle]) -> dict[str, dict[str, float]]:
        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)

        atr_arr = atr(highs, lows, closes)
        adx_arr, di_plus_arr, di_minus_arr = adx(highs, lows, closes)
        vol_sma_arr = volume_sma(volumes)

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

        # Rolling high/low sur breakout_lookback
        lookback = self._config.breakout_lookback
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
                "high": float(highs[i]),
                "low": float(lows[i]),
                "atr": float(atr_arr[i]),
                "atr_sma": float(atr_sma_full[i]),
                "adx": float(adx_arr[i]),
                "di_plus": float(di_plus_arr[i]),
                "di_minus": float(di_minus_arr[i]),
                "volume": float(volumes[i]),
                "volume_sma": float(vol_sma_arr[i]),
                "rolling_high": float(rolling_high[i]),
                "rolling_low": float(rolling_low[i]),
            }

        return indicators

    def _compute_filter(self, candles: list[Candle]) -> dict[str, dict[str, float]]:
        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)

        adx_arr, di_plus_arr, di_minus_arr = adx(highs, lows, closes)

        indicators: dict[str, dict[str, float]] = {}
        for i, candle in enumerate(candles):
            ts = candle.timestamp.isoformat()
            indicators[ts] = {
                "adx": float(adx_arr[i]),
                "di_plus": float(di_plus_arr[i]),
                "di_minus": float(di_minus_arr[i]),
            }
        return indicators

    def evaluate(self, ctx: StrategyContext) -> StrategySignal | None:
        """Évalue les conditions d'entrée Momentum Breakout."""
        main_tf = self._config.timeframe
        filter_tf = self._config.trend_filter_timeframe

        main_ind = ctx.indicators.get(main_tf)
        filter_ind = ctx.indicators.get(filter_tf)
        if not main_ind or not filter_ind:
            return None

        close = main_ind.get("close", float("nan"))
        atr_val = main_ind.get("atr", float("nan"))
        atr_sma_val = main_ind.get("atr_sma", float("nan"))
        adx_val = main_ind.get("adx", float("nan"))
        di_plus = main_ind.get("di_plus", float("nan"))
        di_minus = main_ind.get("di_minus", float("nan"))
        vol = main_ind.get("volume", 0.0)
        vol_sma = main_ind.get("volume_sma", float("nan"))
        rolling_high = main_ind.get("rolling_high", float("nan"))
        rolling_low = main_ind.get("rolling_low", float("nan"))

        if any(np.isnan(v) for v in [close, atr_val, rolling_high, rolling_low]):
            return None

        # Filtre 15m : ADX > 25 = tendance confirmée (inverse de VWAP+RSI)
        filter_adx = filter_ind.get("adx", float("nan"))
        filter_di_plus = filter_ind.get("di_plus", float("nan"))
        filter_di_minus = filter_ind.get("di_minus", float("nan"))

        if np.isnan(filter_adx) or filter_adx < 25:
            return None  # Pas de tendance 15m → pas de momentum trade

        is_15m_bullish = filter_di_plus > filter_di_minus
        is_15m_bearish = filter_di_minus > filter_di_plus

        # Volume spike
        has_volume = (
            not np.isnan(vol_sma)
            and vol_sma > 0
            and vol > vol_sma * self._config.volume_confirmation_multiplier
        )

        # Breakout conditions
        long_breakout = close > rolling_high and is_15m_bullish and has_volume
        short_breakout = close < rolling_low and is_15m_bearish and has_volume

        if not long_breakout and not short_breakout:
            return None

        direction = Direction.LONG if long_breakout else Direction.SHORT

        # Régime de marché
        regime = detect_market_regime(adx_val, di_plus, di_minus, atr_val, atr_sma_val)

        # TP/SL basé sur ATR
        if not np.isnan(atr_val) and atr_val > 0:
            atr_tp = atr_val * self._config.atr_multiplier_tp
            atr_sl = atr_val * self._config.atr_multiplier_sl
        else:
            atr_tp = close * self._config.tp_percent / 100
            atr_sl = close * self._config.sl_percent / 100

        # Cap par les % config
        max_tp = close * self._config.tp_percent / 100
        max_sl = close * self._config.sl_percent / 100
        tp_distance = min(atr_tp, max_tp)
        sl_distance = min(atr_sl, max_sl)

        if direction == Direction.LONG:
            tp_price = close + tp_distance
            sl_price = close - sl_distance
        else:
            tp_price = close - tp_distance
            sl_price = close + sl_distance

        # Score
        breakout_score = 0.6  # Base : breakout confirmé
        volume_score = min(1.0, (vol / vol_sma - 1) / 3) if vol_sma > 0 else 0.0
        trend_score = min(1.0, filter_adx / 40)  # Plus ADX est haut, plus la tendance est forte

        score = breakout_score * 0.4 + volume_score * 0.3 + trend_score * 0.3

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
                "volume_score": volume_score,
                "trend_score": trend_score,
                "atr": atr_val,
                "filter_adx": filter_adx,
            },
        )

    def get_current_conditions(self, ctx: StrategyContext) -> list[dict]:
        """Conditions d'entrée Momentum pour le dashboard."""
        main_tf = self._config.timeframe
        filter_tf = self._config.trend_filter_timeframe
        main_ind = ctx.indicators.get(main_tf, {})
        filter_ind = ctx.indicators.get(filter_tf, {})

        close = main_ind.get("close", float("nan"))
        vol = main_ind.get("volume", 0.0)
        vol_sma = main_ind.get("volume_sma", float("nan"))
        rolling_high = main_ind.get("rolling_high", float("nan"))
        rolling_low = main_ind.get("rolling_low", float("nan"))
        filter_adx = filter_ind.get("adx", float("nan"))

        # Breakout distance (% au-delà du range)
        breakout_dist = 0.0
        if not any(np.isnan(v) for v in [close, rolling_high, rolling_low]):
            range_size = rolling_high - rolling_low
            if range_size > 0:
                if close > rolling_high:
                    breakout_dist = (close - rolling_high) / range_size * 100
                elif close < rolling_low:
                    breakout_dist = (rolling_low - close) / range_size * 100

        # Volume ratio
        vol_ratio = vol / vol_sma if (
            not np.isnan(vol_sma) and vol_sma > 0
        ) else float("nan")

        return [
            {
                "name": "adx_strong",
                "met": not np.isnan(filter_adx) and filter_adx >= 25,
                "value": round(filter_adx, 1) if not np.isnan(filter_adx) else None,
                "threshold": 25,
            },
            {
                "name": "breakout",
                "met": not np.isnan(close) and not np.isnan(rolling_high) and (
                    close > rolling_high or close < rolling_low
                ),
                "value": round(breakout_dist, 1),
                "threshold": 0,
            },
            {
                "name": "volume_confirm",
                "met": not np.isnan(vol_ratio) and vol_ratio >= self._config.volume_confirmation_multiplier,
                "value": round(vol_ratio, 1) if not np.isnan(vol_ratio) else None,
                "threshold": self._config.volume_confirmation_multiplier,
            },
        ]

    def check_exit(self, ctx: StrategyContext, position: OpenPosition) -> str | None:
        """ADX chute sous 20 → momentum essoufflé."""
        main_tf = self._config.timeframe
        main_ind = ctx.indicators.get(main_tf)
        if not main_ind:
            return None

        adx_val = main_ind.get("adx", float("nan"))
        if np.isnan(adx_val):
            return None

        if adx_val < 20:
            logger.debug(
                "Signal exit Momentum : ADX={:.1f} < 20, momentum essoufflé",
                adx_val,
            )
            return "signal_exit"

        return None

    def get_params(self) -> dict:
        return {
            "timeframe": self._config.timeframe,
            "trend_filter_timeframe": self._config.trend_filter_timeframe,
            "breakout_lookback": self._config.breakout_lookback,
            "volume_confirmation_multiplier": self._config.volume_confirmation_multiplier,
            "atr_multiplier_tp": self._config.atr_multiplier_tp,
            "atr_multiplier_sl": self._config.atr_multiplier_sl,
            "tp_percent": self._config.tp_percent,
            "sl_percent": self._config.sl_percent,
        }
