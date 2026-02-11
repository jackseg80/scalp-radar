"""Stratégie Funding Rate Arbitrage pour Scalp Radar.

Scalp lent sur taux de financement extrêmes.
Funding rate négatif extrême → LONG (shorts paient, pression short excessive).
Funding rate positif extrême → SHORT (longs paient, pression long excessive).

NOTE : cette stratégie ne peut PAS être backtestée sur données historiques
(pas d'historique funding rates en DB). Validation en paper trading uniquement.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from loguru import logger

from backend.core.config import FundingConfig
from backend.core.models import Candle, Direction, MarketRegime, SignalStrength
from backend.strategies.base import (
    EXTRA_FUNDING_RATE,
    BaseStrategy,
    OpenPosition,
    StrategyContext,
    StrategySignal,
)


class FundingStrategy(BaseStrategy):
    """Funding Rate Arbitrage.

    Entry LONG : funding_rate < extreme_negative_threshold (-0.03%)
    Entry SHORT : funding_rate > extreme_positive_threshold (0.03%)
    Exit : funding revient à neutre (< |0.01%|)
    """

    name = "funding"

    def __init__(self, config: FundingConfig) -> None:
        self._config = config
        self._signal_detected_at: dict[str, datetime] = {}

    @property
    def min_candles(self) -> dict[str, int]:
        return {self._config.timeframe: 10}

    def compute_indicators(
        self, candles_by_tf: dict[str, list[Candle]]
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Pas d'indicateurs classiques — le funding rate vient via extra_data."""
        result: dict[str, dict[str, dict[str, float]]] = {}
        tf = self._config.timeframe
        if tf in candles_by_tf:
            result[tf] = {}
            for candle in candles_by_tf[tf]:
                ts = candle.timestamp.isoformat()
                result[tf][ts] = {"close": candle.close}
        return result

    def evaluate(self, ctx: StrategyContext) -> StrategySignal | None:
        """Évalue les conditions d'entrée basées sur le funding rate."""
        funding_rate = ctx.extra_data.get(EXTRA_FUNDING_RATE)
        if funding_rate is None:
            return None

        main_tf = self._config.timeframe
        main_ind = ctx.indicators.get(main_tf, {})
        close = main_ind.get("close", float("nan"))
        if np.isnan(close) or close <= 0:
            return None

        # Conditions d'entrée
        is_extreme_neg = funding_rate < self._config.extreme_negative_threshold
        is_extreme_pos = funding_rate > self._config.extreme_positive_threshold

        if not is_extreme_neg and not is_extreme_pos:
            self._signal_detected_at.pop(ctx.symbol, None)
            return None

        # Entry delay : attendre N minutes après la première détection
        delay = timedelta(minutes=self._config.entry_delay_minutes)
        if ctx.symbol not in self._signal_detected_at:
            self._signal_detected_at[ctx.symbol] = ctx.timestamp
            return None

        if ctx.timestamp - self._signal_detected_at[ctx.symbol] < delay:
            return None

        # Signal confirmé
        self._signal_detected_at.pop(ctx.symbol, None)

        direction = Direction.LONG if is_extreme_neg else Direction.SHORT

        # TP/SL
        if direction == Direction.LONG:
            tp_price = close * (1 + self._config.tp_percent / 100)
            sl_price = close * (1 - self._config.sl_percent / 100)
        else:
            tp_price = close * (1 - self._config.tp_percent / 100)
            sl_price = close * (1 + self._config.sl_percent / 100)

        # Score basé sur l'intensité du funding
        intensity = abs(funding_rate) / max(
            abs(self._config.extreme_positive_threshold),
            abs(self._config.extreme_negative_threshold),
        )
        score = min(1.0, intensity * 0.7 + 0.3)

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
            market_regime=MarketRegime.RANGING,
            signals_detail={
                "funding_rate": funding_rate,
                "intensity": intensity,
            },
        )

    def get_current_conditions(self, ctx: StrategyContext) -> list[dict]:
        """Conditions d'entrée Funding pour le dashboard."""
        funding_rate = ctx.extra_data.get(EXTRA_FUNDING_RATE)

        # Delay tracking
        delay_minutes = self._config.entry_delay_minutes
        elapsed = 0.0
        if ctx.symbol in self._signal_detected_at:
            elapsed = (ctx.timestamp - self._signal_detected_at[ctx.symbol]).total_seconds() / 60

        is_extreme = funding_rate is not None and (
            funding_rate < self._config.extreme_negative_threshold
            or funding_rate > self._config.extreme_positive_threshold
        )

        return [
            {
                "name": "funding_extreme",
                "met": is_extreme,
                "value": round(funding_rate, 4) if funding_rate is not None else None,
                "threshold": f"{self._config.extreme_negative_threshold}/{self._config.extreme_positive_threshold}",
            },
            {
                "name": "delay_ok",
                "met": is_extreme and elapsed >= delay_minutes,
                "value": round(elapsed, 1) if is_extreme else 0,
                "threshold": delay_minutes,
            },
        ]

    def check_exit(self, ctx: StrategyContext, position: OpenPosition) -> str | None:
        """Funding revient à neutre → sortie."""
        funding_rate = ctx.extra_data.get(EXTRA_FUNDING_RATE)
        if funding_rate is None:
            return None

        if abs(funding_rate) < 0.01:
            logger.debug(
                "Signal exit Funding : rate={:.4f}% revenu à neutre",
                funding_rate,
            )
            return "signal_exit"

        return None

    def get_params(self) -> dict:
        return {
            "timeframe": self._config.timeframe,
            "extreme_positive_threshold": self._config.extreme_positive_threshold,
            "extreme_negative_threshold": self._config.extreme_negative_threshold,
            "entry_delay_minutes": self._config.entry_delay_minutes,
            "tp_percent": self._config.tp_percent,
            "sl_percent": self._config.sl_percent,
        }
