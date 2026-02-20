"""Stratégie Grid ATR (Mean Reversion Adaptative à la Volatilité).

SMA sur le close + N enveloppes basées sur l'ATR (au lieu de % fixes).
Entrée DCA à chaque niveau touché. TP = retour à la SMA. SL = % depuis prix moyen.
Timeframe : 1h.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.core.config import GridATRConfig
from backend.core.indicators import atr, sma
from backend.core.models import Candle, Direction
from backend.strategies.base import StrategyContext
from backend.strategies.base_grid import BaseGridStrategy, GridLevel, GridState

_TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


class GridATRStrategy(BaseGridStrategy):
    """Grid ATR — Mean Reversion adaptative à la volatilité.

    Enveloppes LONG : SMA - ATR × (start + i × step)
    Enveloppes SHORT : SMA + ATR × (start + i × step)
    Symétrie naturelle (pas besoin de la formule 1/(1-e)-1 d'envelope_dca).
    """

    name = "grid_atr"

    def __init__(self, config: GridATRConfig) -> None:
        self._config = config

    @property
    def max_positions(self) -> int:
        return self._config.num_levels

    @property
    def min_candles(self) -> dict[str, int]:
        min_needed = max(self._config.ma_period, self._config.atr_period) + 20
        return {
            self._config.timeframe: max(min_needed, 50),
        }

    def compute_indicators(
        self, candles_by_tf: dict[str, list[Candle]]
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Calcule SMA + ATR. Enveloppes calculées à la volée dans compute_grid."""
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
        """Retourne les niveaux non encore remplis.

        Enveloppes = SMA ± ATR × multiplier (adaptatives à la volatilité).
        Un seul côté actif : si positions LONG ouvertes, pas de SHORT.
        """
        indicators = ctx.indicators.get(self._config.timeframe, {})
        sma_val = indicators.get("sma", float("nan"))
        atr_val = indicators.get("atr", float("nan"))

        if np.isnan(sma_val) or np.isnan(atr_val) or atr_val <= 0:
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

        levels: list[GridLevel] = []
        for i in range(self._config.num_levels):
            if i in filled_levels:
                continue

            multiplier = (
                self._config.atr_multiplier_start + i * self._config.atr_multiplier_step
            )

            if "long" in active_sides:
                entry_price = sma_val - atr_val * multiplier
                if entry_price > 0:
                    levels.append(GridLevel(
                        index=i,
                        entry_price=entry_price,
                        direction=Direction.LONG,
                        size_fraction=1.0 / self._config.num_levels,
                    ))

            if "short" in active_sides:
                entry_price = sma_val + atr_val * multiplier
                levels.append(GridLevel(
                    index=i,
                    entry_price=entry_price,
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

        # SL : % fixe depuis prix moyen
        sl_pct = self._config.sl_percent / 100
        if direction == Direction.LONG:
            if close <= grid_state.avg_entry_price * (1 - sl_pct):
                return "sl_global"
        else:
            if close >= grid_state.avg_entry_price * (1 + sl_pct):
                return "sl_global"

        # Time-based stop loss
        if self._config.max_hold_candles > 0:
            first_entry = min(p.entry_time for p in grid_state.positions)
            tf_secs = _TF_SECONDS.get(self._config.timeframe, 3600)
            candles_held = int((ctx.timestamp - first_entry).total_seconds() / tf_secs)
            if candles_held >= self._config.max_hold_candles:
                avg_e = grid_state.avg_entry_price
                total_qty = grid_state.total_quantity
                if direction == Direction.LONG:
                    unrealized = (close - avg_e) * total_qty
                else:
                    unrealized = (avg_e - close) * total_qty
                if unrealized < 0:
                    return "time_stop"

        return None

    def get_tp_price(
        self, grid_state: GridState, current_indicators: dict
    ) -> float:
        """TP = SMA actuelle (dynamique)."""
        return current_indicators.get("sma", float("nan"))

    def get_sl_price(
        self, grid_state: GridState, current_indicators: dict
    ) -> float:
        """SL = prix moyen ± sl_percent%."""
        if not grid_state.positions:
            return float("nan")

        sl_pct = self._config.sl_percent / 100
        direction = grid_state.positions[0].direction

        if direction == Direction.LONG:
            return grid_state.avg_entry_price * (1 - sl_pct)
        return grid_state.avg_entry_price * (1 + sl_pct)

    def get_params(self) -> dict[str, Any]:
        return {
            "ma_period": self._config.ma_period,
            "atr_period": self._config.atr_period,
            "atr_multiplier_start": self._config.atr_multiplier_start,
            "atr_multiplier_step": self._config.atr_multiplier_step,
            "num_levels": self._config.num_levels,
            "sl_percent": self._config.sl_percent,
            "sides": self._config.sides,
            "leverage": self._config.leverage,
            "max_hold_candles": self._config.max_hold_candles,
        }
