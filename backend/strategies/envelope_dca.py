"""Stratégie Envelope DCA (Mean Reversion Multi-Niveaux).

SMA sur le close + N enveloppes asymétriques (LONG en dessous, SHORT au-dessus).
Entrée à chaque niveau touché (DCA). TP = retour à la SMA. SL = % depuis prix moyen.
Timeframe : 1h.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.core.config import EnvelopeDCAConfig
from backend.core.indicators import sma
from backend.core.models import Candle, Direction
from backend.strategies.base import OpenPosition, StrategyContext
from backend.strategies.base_grid import BaseGridStrategy, GridLevel, GridState


class EnvelopeDCAStrategy(BaseGridStrategy):
    """Envelope DCA — Mean Reversion multi-niveaux.

    Bandes basses (LONG) : sma × (1 - offset)
    Bandes hautes (SHORT) : sma × (1 + round(1/(1-offset) - 1, 3))
    Les bandes ne sont PAS symétriques (asymétrie log-return).
    """

    name = "envelope_dca"

    def __init__(self, config: EnvelopeDCAConfig) -> None:
        self._config = config

    @property
    def max_positions(self) -> int:
        return self._config.num_levels

    @property
    def min_candles(self) -> dict[str, int]:
        return {
            self._config.timeframe: max(self._config.ma_period + 20, 50),
        }

    def compute_indicators(
        self, candles_by_tf: dict[str, list[Candle]]
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Calcule SMA uniquement. Enveloppes calculées à la volée."""
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
        """Retourne les niveaux non encore remplis.

        Un seul côté actif : si positions LONG ouvertes, pas de SHORT.
        """
        indicators = ctx.indicators.get(self._config.timeframe, {})
        sma_val = indicators.get("sma", float("nan"))

        if np.isnan(sma_val):
            return []

        filled_levels = {p.level for p in grid_state.positions}

        # Déterminer le côté actif
        if grid_state.positions:
            active_direction = grid_state.positions[0].direction
            active_sides = (
                ["long"] if active_direction == Direction.LONG else ["short"]
            )
        else:
            active_sides = self._config.sides

        # Calcul des enveloppes
        lower_offsets = [
            self._config.envelope_start + i * self._config.envelope_step
            for i in range(self._config.num_levels)
        ]
        high_offsets = [round(1 / (1 - e) - 1, 3) for e in lower_offsets]

        levels: list[GridLevel] = []
        for i in range(self._config.num_levels):
            if i in filled_levels:
                continue

            if "long" in active_sides:
                levels.append(GridLevel(
                    index=i,
                    entry_price=sma_val * (1 - lower_offsets[i]),
                    direction=Direction.LONG,
                    size_fraction=1.0 / self._config.num_levels,
                ))

            if "short" in active_sides:
                levels.append(GridLevel(
                    index=i,
                    entry_price=sma_val * (1 + high_offsets[i]),
                    direction=Direction.SHORT,
                    size_fraction=1.0 / self._config.num_levels,
                ))

        return levels

    def should_close_all(
        self, ctx: StrategyContext, grid_state: GridState
    ) -> str | None:
        """Fermer si le prix revient à la SMA (TP) ou dépasse le SL."""
        if not grid_state.positions:
            return None

        indicators = ctx.indicators.get(self._config.timeframe, {})
        sma_val = indicators.get("sma", float("nan"))
        close = indicators.get("close", float("nan"))

        if np.isnan(sma_val) or np.isnan(close):
            return None

        direction = grid_state.positions[0].direction

        # TP : retour à la SMA
        if direction == Direction.LONG and close >= sma_val:
            return "tp_global"
        if direction == Direction.SHORT and close <= sma_val:
            return "tp_global"

        # SL : prix s'éloigne trop du prix moyen
        sl_pct = self._config.sl_percent / 100
        if direction == Direction.LONG:
            if close <= grid_state.avg_entry_price * (1 - sl_pct):
                return "sl_global"
        else:
            if close >= grid_state.avg_entry_price * (1 + sl_pct):
                return "sl_global"

        return None

    def get_tp_price(
        self, grid_state: GridState, current_indicators: dict
    ) -> float:
        """TP = SMA actuelle (dynamique)."""
        return current_indicators.get("sma", float("nan"))

    def get_sl_price(
        self, grid_state: GridState, current_indicators: dict
    ) -> float:
        """SL = % depuis prix moyen."""
        if not grid_state.positions:
            return float("nan")
        avg = grid_state.avg_entry_price
        sl_pct = self._config.sl_percent / 100
        direction = grid_state.positions[0].direction
        if direction == Direction.LONG:
            return avg * (1 - sl_pct)
        return avg * (1 + sl_pct)

    def get_params(self) -> dict[str, Any]:
        return self._config.model_dump(
            exclude={"enabled", "live_eligible", "weight", "per_asset"}
        )
