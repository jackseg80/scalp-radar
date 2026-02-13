"""Stratégie Bollinger Band Mean Reversion pour Scalp Radar.

Trade CONTRE la tendance : entrée aux extrêmes des bandes de Bollinger,
sortie au retour à la SMA (TP dynamique).
Timeframe : 1h.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from backend.core.config import BollingerMRConfig
from backend.core.indicators import (
    adx,
    atr,
    bollinger_bands,
    detect_market_regime,
    sma,
)
from backend.core.models import Candle, Direction, MarketRegime, SignalStrength
from backend.strategies.base import BaseStrategy, OpenPosition, StrategyContext, StrategySignal


class BollingerMRStrategy(BaseStrategy):
    """Bollinger Band Mean Reversion.

    Entry LONG : close < bande basse (prix sous-évalué).
    Entry SHORT : close > bande haute (prix sur-évalué).
    Exit : close croise la SMA (retour à la moyenne) — TP dynamique.
    SL : % fixe depuis l'entrée.
    """

    name = "bollinger_mr"

    def __init__(self, config: BollingerMRConfig) -> None:
        self._config = config

    @property
    def min_candles(self) -> dict[str, int]:
        return {
            self._config.timeframe: max(self._config.bb_period + 20, 50),
        }

    def compute_indicators(
        self, candles_by_tf: dict[str, list[Candle]]
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Pré-calcule SMA, Bollinger Bands, ATR, ADX."""
        result: dict[str, dict[str, dict[str, float]]] = {}

        main_tf = self._config.timeframe
        if main_tf in candles_by_tf and candles_by_tf[main_tf]:
            result[main_tf] = self._compute_main(candles_by_tf[main_tf])

        return result

    def _compute_main(self, candles: list[Candle]) -> dict[str, dict[str, float]]:
        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)

        period = self._config.bb_period
        std_dev = self._config.bb_std

        bb_sma, bb_upper, bb_lower = bollinger_bands(closes, period, std_dev)

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
            indicators[ts] = {
                "close": float(closes[i]),
                "bb_sma": float(bb_sma[i]),
                "bb_upper": float(bb_upper[i]),
                "bb_lower": float(bb_lower[i]),
                "atr": float(atr_arr[i]),
                "atr_sma": float(atr_sma_full[i]),
                "adx": float(adx_arr[i]),
                "di_plus": float(di_plus_arr[i]),
                "di_minus": float(di_minus_arr[i]),
            }

        return indicators

    def evaluate(self, ctx: StrategyContext) -> StrategySignal | None:
        """Évalue les conditions d'entrée Bollinger MR."""
        main_tf = self._config.timeframe
        main_ind = ctx.indicators.get(main_tf)
        if not main_ind:
            return None

        close = main_ind.get("close", float("nan"))
        bb_sma_val = main_ind.get("bb_sma", float("nan"))
        bb_upper = main_ind.get("bb_upper", float("nan"))
        bb_lower = main_ind.get("bb_lower", float("nan"))
        atr_val = main_ind.get("atr", float("nan"))
        atr_sma_val = main_ind.get("atr_sma", float("nan"))
        adx_val = main_ind.get("adx", float("nan"))
        di_plus = main_ind.get("di_plus", float("nan"))
        di_minus = main_ind.get("di_minus", float("nan"))

        if any(np.isnan(v) for v in [close, bb_sma_val, bb_upper, bb_lower]):
            return None

        # Conditions d'entrée
        long_signal = close < bb_lower
        short_signal = close > bb_upper

        if not long_signal and not short_signal:
            return None

        direction = Direction.LONG if long_signal else Direction.SHORT

        # Régime de marché
        regime = detect_market_regime(adx_val, di_plus, di_minus, atr_val, atr_sma_val)

        # TP très éloigné (désactive le TP fixe — check_exit gère le vrai TP)
        # SL = % fixe
        sl_pct = self._resolve_param("sl_percent", ctx.symbol)

        if direction == Direction.LONG:
            tp_price = close * 2.0  # Très éloigné, jamais touché
            sl_price = close * (1 - sl_pct / 100)
        else:
            tp_price = close * 0.5  # Très éloigné, jamais touché
            sl_price = close * (1 + sl_pct / 100)

        # Score basé sur la distance aux bandes
        band_width = bb_upper - bb_lower
        if band_width > 0:
            if long_signal:
                distance_score = min(1.0, (bb_lower - close) / band_width * 2)
            else:
                distance_score = min(1.0, (close - bb_upper) / band_width * 2)
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
                "band_width": band_width,
            },
        )

    def get_current_conditions(self, ctx: StrategyContext) -> list[dict]:
        """Conditions d'entrée Bollinger MR pour le dashboard."""
        main_tf = self._config.timeframe
        main_ind = ctx.indicators.get(main_tf, {})

        close = main_ind.get("close", float("nan"))
        bb_upper = main_ind.get("bb_upper", float("nan"))
        bb_lower = main_ind.get("bb_lower", float("nan"))

        outside_bands = False
        position_str = "inside"
        if not any(np.isnan(v) for v in [close, bb_upper, bb_lower]):
            if close < bb_lower:
                outside_bands = True
                position_str = "below"
            elif close > bb_upper:
                outside_bands = True
                position_str = "above"

        return [
            {
                "name": "bb_position",
                "met": outside_bands,
                "value": position_str,
                "threshold": "outside",
            },
            {
                "name": "sl_percent",
                "met": True,
                "value": self._config.sl_percent,
                "threshold": self._config.sl_percent,
            },
        ]

    def check_exit(self, ctx: StrategyContext, position: OpenPosition) -> str | None:
        """TP dynamique : close croise la SMA → sortie."""
        main_tf = self._config.timeframe
        main_ind = ctx.indicators.get(main_tf)
        if not main_ind:
            return None

        close = main_ind.get("close", float("nan"))
        bb_sma_val = main_ind.get("bb_sma", float("nan"))

        if np.isnan(close) or np.isnan(bb_sma_val):
            return None

        if position.direction == Direction.LONG and close >= bb_sma_val:
            logger.debug(
                "Signal exit Bollinger MR : close={:.2f} >= SMA={:.2f}",
                close, bb_sma_val,
            )
            return "signal_exit"

        if position.direction == Direction.SHORT and close <= bb_sma_val:
            logger.debug(
                "Signal exit Bollinger MR : close={:.2f} <= SMA={:.2f}",
                close, bb_sma_val,
            )
            return "signal_exit"

        return None

    def get_params(self) -> dict:
        return {
            "timeframe": self._config.timeframe,
            "bb_period": self._config.bb_period,
            "bb_std": self._config.bb_std,
            "sl_percent": self._config.sl_percent,
        }
