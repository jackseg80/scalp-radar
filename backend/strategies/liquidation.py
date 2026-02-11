"""Stratégie Liquidation Zone Hunting pour Scalp Radar.

Estime les zones de liquidation via open interest + levier moyen estimé.
Trade la cascade quand le prix approche ces zones.

NOTE : validation en paper trading uniquement (pas d'historique OI pour backtest).
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from backend.core.config import LiquidationConfig
from backend.core.models import Candle, Direction, MarketRegime, SignalStrength
from backend.strategies.base import (
    EXTRA_OI_CHANGE_PCT,
    EXTRA_OPEN_INTEREST,
    BaseStrategy,
    OpenPosition,
    StrategyContext,
    StrategySignal,
)


class LiquidationStrategy(BaseStrategy):
    """Liquidation Zone Hunting.

    1. Estimer les zones de liquidation :
       - liq_long_zone = price × (1 - 1/leverage) — où les longs se font liquider
       - liq_short_zone = price × (1 + 1/leverage) — où les shorts se font liquider
    2. Si OI a augmenté > seuil → leviers chargés
    3. Si prix approche une zone (< zone_buffer_percent) :
       - LONG si approche zone shorts (short squeeze anticipé)
       - SHORT si approche zone longs (cascade de liquidation)
    """

    name = "liquidation"

    def __init__(self, config: LiquidationConfig) -> None:
        self._config = config

    @property
    def min_candles(self) -> dict[str, int]:
        return {self._config.timeframe: 50}

    def compute_indicators(
        self, candles_by_tf: dict[str, list[Candle]]
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Pas d'indicateurs classiques — OI vient via extra_data."""
        result: dict[str, dict[str, dict[str, float]]] = {}
        tf = self._config.timeframe
        if tf in candles_by_tf:
            result[tf] = {}
            for candle in candles_by_tf[tf]:
                ts = candle.timestamp.isoformat()
                result[tf][ts] = {"close": candle.close}
        return result

    def evaluate(self, ctx: StrategyContext) -> StrategySignal | None:
        """Évalue les conditions d'entrée basées sur les zones de liquidation."""
        main_tf = self._config.timeframe
        main_ind = ctx.indicators.get(main_tf, {})
        close = main_ind.get("close", float("nan"))
        if np.isnan(close) or close <= 0:
            return None

        # OI data depuis extra_data
        oi_change = ctx.extra_data.get(EXTRA_OI_CHANGE_PCT, 0.0)
        oi_snapshots = ctx.extra_data.get(EXTRA_OPEN_INTEREST, [])

        # Condition 1 : OI doit avoir augmenté significativement
        if oi_change < self._config.oi_change_threshold:
            return None

        # Estimer les zones de liquidation
        leverage = self._config.leverage_estimate
        liq_long_zone = close * (1 - 1 / leverage)  # Longs liquidés ici
        liq_short_zone = close * (1 + 1 / leverage)  # Shorts liquidés ici

        buffer_pct = self._config.zone_buffer_percent / 100

        # Distance du prix aux zones
        dist_to_long_liq = (close - liq_long_zone) / close
        dist_to_short_liq = (liq_short_zone - close) / close

        # Check si le prix approche une zone
        near_long_liq = dist_to_long_liq < buffer_pct  # Proche de la zone longs
        near_short_liq = dist_to_short_liq < buffer_pct  # Proche de la zone shorts

        if not near_long_liq and not near_short_liq:
            return None

        if near_short_liq:
            # Prix approche la zone de liq des shorts → short squeeze → LONG
            direction = Direction.LONG
        else:
            # Prix approche la zone de liq des longs → cascade → SHORT
            direction = Direction.SHORT

        # TP/SL
        if direction == Direction.LONG:
            tp_price = close * (1 + self._config.tp_percent / 100)
            sl_price = close * (1 - self._config.sl_percent / 100)
        else:
            tp_price = close * (1 - self._config.tp_percent / 100)
            sl_price = close * (1 + self._config.sl_percent / 100)

        # Score basé sur l'intensité de l'OI change
        oi_score = min(1.0, oi_change / (self._config.oi_change_threshold * 3))
        proximity_score = 1.0 - min(dist_to_long_liq, dist_to_short_liq) / buffer_pct
        proximity_score = max(0.0, min(1.0, proximity_score))

        score = oi_score * 0.5 + proximity_score * 0.5

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
                "oi_change": oi_change,
                "oi_score": oi_score,
                "proximity_score": proximity_score,
                "liq_long_zone": liq_long_zone,
                "liq_short_zone": liq_short_zone,
            },
        )

    def get_current_conditions(self, ctx: StrategyContext) -> list[dict]:
        """Conditions d'entrée Liquidation pour le dashboard."""
        main_tf = self._config.timeframe
        main_ind = ctx.indicators.get(main_tf, {})
        close = main_ind.get("close", float("nan"))

        oi_change = ctx.extra_data.get(EXTRA_OI_CHANGE_PCT, 0.0)

        # Zone proximity
        leverage = self._config.leverage_estimate
        buffer_pct = self._config.zone_buffer_percent / 100
        zone_dist = None
        near_zone = False
        if not np.isnan(close) and close > 0:
            liq_long_zone = close * (1 - 1 / leverage)
            liq_short_zone = close * (1 + 1 / leverage)
            dist_to_long = (close - liq_long_zone) / close
            dist_to_short = (liq_short_zone - close) / close
            zone_dist = round(min(dist_to_long, dist_to_short) * 100, 2)
            near_zone = min(dist_to_long, dist_to_short) < buffer_pct

        # Volume — on n'a pas volume_sma pour liquidation, on utilise juste OI
        return [
            {
                "name": "zone_proximity",
                "met": near_zone,
                "value": zone_dist,
                "threshold": round(buffer_pct * 100, 2),
            },
            {
                "name": "oi_threshold",
                "met": oi_change >= self._config.oi_change_threshold,
                "value": round(oi_change, 2),
                "threshold": self._config.oi_change_threshold,
            },
        ]

    def check_exit(self, ctx: StrategyContext, position: OpenPosition) -> str | None:
        """OI chute brutalement → cascade terminée → sortie."""
        oi_change = ctx.extra_data.get(EXTRA_OI_CHANGE_PCT, 0.0)

        # OI chute de plus de 3% → la cascade est terminée
        if oi_change < -3.0:
            logger.debug(
                "Signal exit Liquidation : OI change={:.1f}% (cascade terminée)",
                oi_change,
            )
            return "signal_exit"

        return None

    def get_params(self) -> dict:
        return {
            "timeframe": self._config.timeframe,
            "oi_change_threshold": self._config.oi_change_threshold,
            "leverage_estimate": self._config.leverage_estimate,
            "zone_buffer_percent": self._config.zone_buffer_percent,
            "tp_percent": self._config.tp_percent,
            "sl_percent": self._config.sl_percent,
        }
