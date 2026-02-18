"""Stratégie Bollinger Trend Following pour Scalp Radar.

Trade les breakouts des bandes de Bollinger filtrés par tendance long terme.
Entrée quand le close franchit la bande (breakout), avec confirm prev_close
dans les bandes + spread suffisant + filtre SMA long terme.
Sortie dynamique au retour à la SMA de Bollinger.
Timeframe : 1h (défaut).
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from backend.core.config import BolTrendConfig
from backend.core.indicators import (
    adx,
    atr,
    bollinger_bands,
    detect_market_regime,
    sma,
)
from backend.core.models import Candle, Direction, MarketRegime, SignalStrength
from backend.strategies.base import BaseStrategy, OpenPosition, StrategyContext, StrategySignal


class BolTrendStrategy(BaseStrategy):
    """Bollinger Trend Following.

    Entry LONG : prev_close < prev_upper AND close > upper
                 AND spread > min_bol_spread AND close > long_ma.
    Entry SHORT : prev_close > prev_lower AND close < lower
                  AND spread > min_bol_spread AND close < long_ma.
    Exit : close croise la SMA de Bollinger (retour au centre) — TP dynamique.
    SL : % fixe depuis l'entrée (filet de sécurité).
    """

    name = "boltrend"

    def __init__(self, config: BolTrendConfig) -> None:
        self._config = config

    @property
    def min_candles(self) -> dict[str, int]:
        return {
            self._config.timeframe: max(self._config.bol_window, self._config.long_ma_window) + 20,
        }

    def compute_indicators(
        self, candles_by_tf: dict[str, list[Candle]]
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Pré-calcule BB(bol_window, bol_std), SMA(long_ma_window), ATR, ADX."""
        result: dict[str, dict[str, dict[str, float]]] = {}

        main_tf = self._config.timeframe
        if main_tf in candles_by_tf and candles_by_tf[main_tf]:
            result[main_tf] = self._compute_main(candles_by_tf[main_tf])

        return result

    def _compute_main(self, candles: list[Candle]) -> dict[str, dict[str, float]]:
        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)

        bol_window = self._config.bol_window
        bol_std = self._config.bol_std
        long_ma_window = self._config.long_ma_window

        bb_sma_arr, bb_upper, bb_lower = bollinger_bands(closes, bol_window, bol_std)
        long_ma = sma(closes, long_ma_window)

        # ATR + ADX pour le régime de marché
        atr_arr = atr(highs, lows, closes)
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
            prev_close = float(closes[i - 1]) if i > 0 else float("nan")
            prev_upper = float(bb_upper[i - 1]) if i > 0 else float("nan")
            prev_lower = float(bb_lower[i - 1]) if i > 0 else float("nan")

            # Spread des bandes précédentes
            if i > 0 and not np.isnan(prev_upper) and not np.isnan(prev_lower) and prev_lower > 0:
                prev_spread = (prev_upper - prev_lower) / prev_lower
            else:
                prev_spread = float("nan")

            indicators[ts] = {
                "close": float(closes[i]),
                "bb_sma": float(bb_sma_arr[i]),
                "bb_upper": float(bb_upper[i]),
                "bb_lower": float(bb_lower[i]),
                "long_ma": float(long_ma[i]),
                "prev_close": prev_close,
                "prev_upper": prev_upper,
                "prev_lower": prev_lower,
                "prev_spread": prev_spread,
                "atr": float(atr_arr[i]),
                "atr_sma": float(atr_sma_full[i]),
                "adx": float(adx_arr[i]),
                "di_plus": float(di_plus_arr[i]),
                "di_minus": float(di_minus_arr[i]),
            }

        return indicators

    def evaluate(self, ctx: StrategyContext) -> StrategySignal | None:
        """Évalue les conditions d'entrée BolTrend."""
        main_tf = self._config.timeframe
        main_ind = ctx.indicators.get(main_tf)
        if not main_ind:
            return None

        close = main_ind.get("close", float("nan"))
        bb_upper = main_ind.get("bb_upper", float("nan"))
        bb_lower = main_ind.get("bb_lower", float("nan"))
        bb_sma_val = main_ind.get("bb_sma", float("nan"))
        long_ma = main_ind.get("long_ma", float("nan"))
        prev_close = main_ind.get("prev_close", float("nan"))
        prev_upper = main_ind.get("prev_upper", float("nan"))
        prev_lower = main_ind.get("prev_lower", float("nan"))
        prev_spread = main_ind.get("prev_spread", float("nan"))
        atr_val = main_ind.get("atr", float("nan"))
        atr_sma_val = main_ind.get("atr_sma", float("nan"))
        adx_val = main_ind.get("adx", float("nan"))
        di_plus = main_ind.get("di_plus", float("nan"))
        di_minus = main_ind.get("di_minus", float("nan"))

        if any(np.isnan(v) for v in [close, bb_upper, bb_lower, long_ma, prev_close, prev_upper, prev_lower]):
            return None

        min_bol_spread = self._resolve_param("min_bol_spread", ctx.symbol)

        # Spread check
        spread_ok = not np.isnan(prev_spread) and prev_spread > min_bol_spread

        # Signal Long : breakout haussier + trend filter
        long_signal = (
            prev_close < prev_upper
            and close > bb_upper
            and spread_ok
            and close > long_ma
        )

        # Signal Short : breakout baissier + trend filter
        short_signal = (
            prev_close > prev_lower
            and close < bb_lower
            and spread_ok
            and close < long_ma
        )

        if not long_signal and not short_signal:
            return None

        direction = Direction.LONG if long_signal else Direction.SHORT

        # Régime de marché
        regime = detect_market_regime(adx_val, di_plus, di_minus, atr_val, atr_sma_val)

        # TP très éloigné (check_exit gère le vrai TP via SMA crossing)
        sl_pct = self._resolve_param("sl_percent", ctx.symbol)

        if direction == Direction.LONG:
            tp_price = close * 2.0
            sl_price = close * (1 - sl_pct / 100)
        else:
            tp_price = close * 0.5
            sl_price = close * (1 + sl_pct / 100)

        # Score basé sur la distance de breakout
        band_width = bb_upper - bb_lower
        if band_width > 0:
            if long_signal:
                distance_score = min(1.0, (close - bb_upper) / band_width * 2)
            else:
                distance_score = min(1.0, (bb_lower - close) / band_width * 2)
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
                "bb_sma": bb_sma_val,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "long_ma": long_ma,
                "prev_spread": prev_spread if not np.isnan(prev_spread) else 0.0,
            },
        )

    def get_current_conditions(self, ctx: StrategyContext) -> list[dict]:
        """Conditions d'entrée BolTrend pour le dashboard."""
        main_tf = self._config.timeframe
        main_ind = ctx.indicators.get(main_tf, {})

        close = main_ind.get("close", float("nan"))
        bb_upper = main_ind.get("bb_upper", float("nan"))
        bb_lower = main_ind.get("bb_lower", float("nan"))
        long_ma = main_ind.get("long_ma", float("nan"))
        prev_close = main_ind.get("prev_close", float("nan"))
        prev_upper = main_ind.get("prev_upper", float("nan"))
        prev_spread = main_ind.get("prev_spread", float("nan"))

        # Breakout detection
        breakout = "none"
        breakout_met = False
        if not any(np.isnan(v) for v in [close, bb_upper, bb_lower, prev_close, prev_upper]):
            if prev_close < prev_upper and close > bb_upper:
                breakout = "long"
                breakout_met = True
            elif prev_close > main_ind.get("prev_lower", float("nan")) and close < bb_lower:
                breakout = "short"
                breakout_met = True

        # Trend filter
        trend_ok = False
        trend_str = "unknown"
        if not any(np.isnan(v) for v in [close, long_ma]):
            trend_ok = True
            trend_str = "bullish" if close > long_ma else "bearish"

        return [
            {
                "name": "bb_breakout",
                "met": breakout_met,
                "value": breakout,
                "threshold": "breakout",
            },
            {
                "name": "trend_filter",
                "met": trend_ok,
                "value": trend_str,
                "threshold": "aligned",
            },
            {
                "name": "bb_spread",
                "met": not np.isnan(prev_spread) and prev_spread > self._config.min_bol_spread,
                "value": round(prev_spread, 4) if not np.isnan(prev_spread) else None,
                "threshold": self._config.min_bol_spread,
            },
        ]

    def check_exit(self, ctx: StrategyContext, position: OpenPosition) -> str | None:
        """TP dynamique : close croise la SMA de Bollinger → sortie."""
        main_tf = self._config.timeframe
        main_ind = ctx.indicators.get(main_tf)
        if not main_ind:
            return None

        close = main_ind.get("close", float("nan"))
        bb_sma_val = main_ind.get("bb_sma", float("nan"))

        if np.isnan(close) or np.isnan(bb_sma_val):
            return None

        # LONG exit: breakout s'essouffle, close revient sous la SMA
        if position.direction == Direction.LONG and close < bb_sma_val:
            logger.debug(
                "Signal exit BolTrend : close={:.2f} < SMA={:.2f}",
                close, bb_sma_val,
            )
            return "signal_exit"

        # SHORT exit: breakout s'essouffle, close revient au-dessus de la SMA
        if position.direction == Direction.SHORT and close > bb_sma_val:
            logger.debug(
                "Signal exit BolTrend : close={:.2f} > SMA={:.2f}",
                close, bb_sma_val,
            )
            return "signal_exit"

        return None

    def get_params(self) -> dict:
        return {
            "timeframe": self._config.timeframe,
            "bol_window": self._config.bol_window,
            "bol_std": self._config.bol_std,
            "min_bol_spread": self._config.min_bol_spread,
            "long_ma_window": self._config.long_ma_window,
            "sl_percent": self._config.sl_percent,
            "leverage": self._config.leverage,
        }
