"""Stratégie Grid Range ATR (Range Trading Bidirectionnel).

SMA centre + N niveaux LONG et SHORT avec espacement = ATR × atr_spacing_mult.
Chaque position a son propre TP (retour SMA) et SL (% depuis entry).
Timeframe : 1h.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.core.config import GridRangeATRConfig
from backend.core.indicators import atr, sma
from backend.core.models import Candle, Direction
from backend.strategies.base import StrategyContext
from backend.strategies.base_grid import BaseGridStrategy, GridLevel, GridState


class GridRangeATRStrategy(BaseGridStrategy):
    """Grid Range ATR — range trading bidirectionnel.

    Niveaux LONG : SMA - (i+1) × ATR × spacing_mult
    Niveaux SHORT : SMA + (i+1) × ATR × spacing_mult
    TP individuel : retour à la SMA (dynamique ou fixe selon tp_mode).
    SL individuel : sl_percent depuis le prix d'entrée.
    """

    name = "grid_range_atr"

    def __init__(self, config: GridRangeATRConfig) -> None:
        self._config = config

    @property
    def max_positions(self) -> int:
        count = 0
        if "long" in self._config.sides:
            count += self._config.num_levels
        if "short" in self._config.sides:
            count += self._config.num_levels
        return count

    @property
    def min_candles(self) -> dict[str, int]:
        min_needed = max(self._config.ma_period, self._config.atr_period) + 20
        return {
            self._config.timeframe: max(min_needed, 50),
        }

    def compute_indicators(
        self, candles_by_tf: dict[str, list[Candle]]
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Calcule SMA + ATR."""
        result: dict[str, dict[str, dict[str, Any]]] = {}
        tf = self._config.timeframe
        if tf in candles_by_tf and candles_by_tf[tf]:
            candles = candles_by_tf[tf]
            closes = np.array([c.close for c in candles], dtype=float)
            highs = np.array([c.high for c in candles], dtype=float)
            lows = np.array([c.low for c in candles], dtype=float)

            sma_arr = sma(closes, self._config.ma_period)
            atr_arr = atr(highs, lows, closes, self._config.atr_period)

            indicators: dict[str, dict[str, Any]] = {}
            for i, c in enumerate(candles):
                ts = c.timestamp.isoformat()
                indicators[ts] = {
                    "sma": float(sma_arr[i]),
                    "atr": float(atr_arr[i]),
                    "close": c.close,
                }
            result[tf] = indicators
        return result

    def compute_grid(
        self, ctx: StrategyContext, grid_state: GridState
    ) -> list[GridLevel]:
        """Retourne les niveaux non remplis des DEUX côtés (LONG et SHORT).

        Level encoding : LONG = 0..N-1, SHORT = N..2N-1.
        Contrairement à grid_atr, les deux côtés sont actifs simultanément.
        """
        indicators = ctx.indicators.get(self._config.timeframe, {})
        sma_val = indicators.get("sma", float("nan"))
        atr_val = indicators.get("atr", float("nan"))

        if np.isnan(sma_val) or np.isnan(atr_val) or atr_val <= 0:
            return []

        filled_levels = {p.level for p in grid_state.positions}

        levels: list[GridLevel] = []
        spacing = atr_val * self._config.atr_spacing_mult

        for i in range(self._config.num_levels):
            # LONG levels : index 0..N-1
            if "long" in self._config.sides and i not in filled_levels:
                entry_price = sma_val - (i + 1) * spacing
                if entry_price > 0:
                    levels.append(GridLevel(
                        index=i,
                        entry_price=entry_price,
                        direction=Direction.LONG,
                        size_fraction=1.0 / self.max_positions,
                    ))

            # SHORT levels : index N..2N-1
            short_idx = self._config.num_levels + i
            if "short" in self._config.sides and short_idx not in filled_levels:
                entry_price = sma_val + (i + 1) * spacing
                levels.append(GridLevel(
                    index=short_idx,
                    entry_price=entry_price,
                    direction=Direction.SHORT,
                    size_fraction=1.0 / self.max_positions,
                ))

        return levels

    def should_close_all(
        self, ctx: StrategyContext, grid_state: GridState
    ) -> str | None:
        """Jamais de close-all global — TP/SL individuels par position.

        Le kill switch framework (Simulator) prend le relais pour
        la protection globale du capital.
        """
        return None

    def get_tp_price(
        self, grid_state: GridState, current_indicators: dict
    ) -> float:
        """TP global non utilisé (NaN). TP individuel géré par le fast engine.

        Retourne NaN car GridStrategyRunner n'est pas utilisé pour
        grid_range — pas d'appel à check_global_tp_sl.
        """
        return float("nan")

    def get_sl_price(
        self, grid_state: GridState, current_indicators: dict
    ) -> float:
        """SL global non utilisé (NaN). SL individuel géré par le fast engine."""
        return float("nan")

    def get_params(self) -> dict[str, Any]:
        return {
            "ma_period": self._config.ma_period,
            "atr_period": self._config.atr_period,
            "atr_spacing_mult": self._config.atr_spacing_mult,
            "num_levels": self._config.num_levels,
            "sl_percent": self._config.sl_percent,
            "tp_mode": self._config.tp_mode,
            "sides": self._config.sides,
            "leverage": self._config.leverage,
        }
