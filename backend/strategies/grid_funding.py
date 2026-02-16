"""Stratégie Grid Funding (DCA sur Funding Rate Négatif).

Entre LONG quand le funding rate est très négatif — signal structurel indépendant du prix.
L'edge : les shorts paient les longs tant que le funding est négatif.
TP = funding redevient positif OU prix > SMA. SL = % classique.
Timeframe : 1h. LONG-only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.core.config import GridFundingConfig
from backend.core.indicators import sma
from backend.core.models import Candle, Direction
from backend.strategies.base import StrategyContext
from backend.strategies.base_grid import BaseGridStrategy, GridLevel, GridState


class GridFundingStrategy(BaseGridStrategy):
    """Grid Funding — DCA sur funding rate négatif.

    Signal d'entrée : funding_rate < -threshold (structurel, pas prix).
    Multi-niveaux : chaque level a un seuil plus négatif.
    TP : funding > 0 OU prix >= SMA (selon tp_mode).
    SL : % fixe depuis prix moyen.
    """

    name = "grid_funding"

    def __init__(self, config: GridFundingConfig) -> None:
        self._config = config

    @property
    def max_positions(self) -> int:
        return self._config.num_levels

    @property
    def min_candles(self) -> dict[str, int]:
        return {
            self._config.timeframe: max(self._config.ma_period + 10, 50),
        }

    def compute_indicators(
        self, candles_by_tf: dict[str, list[Candle]]
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Calcule SMA sur 1h (pour TP). Le funding est dans extra_data."""
        result: dict[str, dict[str, dict[str, Any]]] = {}
        tf = self._config.timeframe
        if tf in candles_by_tf and candles_by_tf[tf]:
            candles = candles_by_tf[tf]
            closes = np.array([c.close for c in candles], dtype=float)
            sma_arr = sma(closes, self._config.ma_period)

            indicators: dict[str, dict[str, Any]] = {}
            for i, c in enumerate(candles):
                ts = c.timestamp.isoformat()
                indicators[ts] = {
                    "sma": float(sma_arr[i]),
                    "close": c.close,
                }
            result[tf] = indicators
        return result

    def compute_grid(
        self, ctx: StrategyContext, grid_state: GridState
    ) -> list[GridLevel]:
        """Signal basé sur le funding rate, pas le prix.

        Lit ctx.extra_data["funding_rate"] (en %, depuis build_extra_data_map),
        convertit en raw decimal (/100) pour comparer aux seuils.
        """
        # Funding rate : DB stocke en %, on convertit en raw decimal
        funding_rate_pct = ctx.extra_data.get("funding_rate")
        if funding_rate_pct is None:
            return []
        funding_rate = funding_rate_pct / 100  # % → raw decimal

        if np.isnan(funding_rate):
            return []

        indicators = ctx.indicators.get(self._config.timeframe, {})
        close = indicators.get("close", float("nan"))
        if np.isnan(close):
            return []

        filled_levels = {p.level for p in grid_state.positions}

        levels: list[GridLevel] = []
        for i in range(self._config.num_levels):
            if i in filled_levels:
                continue
            threshold = -(
                self._config.funding_threshold_start
                + i * self._config.funding_threshold_step
            )
            if funding_rate <= threshold:
                levels.append(GridLevel(
                    index=i,
                    entry_price=close,  # entre au prix courant
                    direction=Direction.LONG,
                    size_fraction=1.0 / self._config.num_levels,
                ))

        return levels

    def should_close_all(
        self, ctx: StrategyContext, grid_state: GridState
    ) -> str | None:
        """TP: funding > 0 OU prix >= SMA. SL: % classique."""
        if not grid_state.positions:
            return None

        indicators = ctx.indicators.get(self._config.timeframe, {})
        sma_val = indicators.get("sma", float("nan"))
        close = indicators.get("close", float("nan"))

        if np.isnan(sma_val) or np.isnan(close):
            return None

        # Funding rate (% → raw decimal)
        funding_rate_pct = ctx.extra_data.get("funding_rate", 0)
        funding_rate = funding_rate_pct / 100

        # SL global (toujours actif, même pendant min_hold)
        sl_pct = self._config.sl_percent / 100
        if close <= grid_state.avg_entry_price * (1 - sl_pct):
            return "sl_global"

        # Min hold check — bloque le TP mais pas le SL
        # Compter les candles depuis la première position
        if grid_state.positions:
            first_entry = min(p.entry_time for p in grid_state.positions)
            candles_held = 0
            if hasattr(ctx, "timestamp") and ctx.timestamp and first_entry:
                delta = ctx.timestamp - first_entry
                candles_held = int(delta.total_seconds() / 3600)  # 1h candles
            if candles_held < self._config.min_hold_candles:
                return None

        # TP modes
        tp_mode = self._config.tp_mode
        if tp_mode in ("funding_positive", "funding_or_sma"):
            if funding_rate > 0:
                return "tp_funding"
        if tp_mode in ("sma_cross", "funding_or_sma"):
            if close >= sma_val:
                return "tp_sma"

        return None

    def get_tp_price(
        self, grid_state: GridState, current_indicators: dict
    ) -> float:
        """TP = SMA actuelle (dynamique)."""
        return current_indicators.get("sma", float("nan"))

    def get_sl_price(
        self, grid_state: GridState, current_indicators: dict
    ) -> float:
        """SL = prix moyen - sl_percent%."""
        if not grid_state.positions:
            return float("nan")
        sl_pct = self._config.sl_percent / 100
        return grid_state.avg_entry_price * (1 - sl_pct)

    def get_params(self) -> dict[str, Any]:
        return {
            "funding_threshold_start": self._config.funding_threshold_start,
            "funding_threshold_step": self._config.funding_threshold_step,
            "num_levels": self._config.num_levels,
            "tp_mode": self._config.tp_mode,
            "ma_period": self._config.ma_period,
            "sl_percent": self._config.sl_percent,
            "min_hold_candles": self._config.min_hold_candles,
            "sides": self._config.sides,
            "leverage": self._config.leverage,
        }
