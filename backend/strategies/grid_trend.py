"""Stratégie Grid Trend (Trend Following DCA).

Filtre directionnel EMA cross + ADX (force du trend) :
  - EMA fast > slow ET ADX > seuil → DCA LONG (pullbacks sous l'EMA fast)
  - EMA fast < slow ET ADX > seuil → DCA SHORT (pullbacks au-dessus de l'EMA fast)
  - ADX < seuil → zone neutre (pas de nouveaux trades)
TP = trailing stop ATR (pas de retour à la SMA).
SL = % fixe depuis prix moyen.
Force close au flip de direction (EMA cross).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from backend.core.config import GridTrendConfig
from backend.core.indicators import adx, atr, ema
from backend.core.models import Candle, Direction
from backend.strategies.base import StrategyContext
from backend.strategies.base_grid import BaseGridStrategy, GridLevel, GridState


class GridTrendStrategy(BaseGridStrategy):
    """Grid Trend — trend following DCA avec trailing stop ATR.

    Enveloppes LONG : EMA_fast - ATR × (pull_start + i × pull_step) quand trend UP
    Enveloppes SHORT : EMA_fast + ATR × (pull_start + i × pull_step) quand trend DOWN
    """

    name = "grid_trend"

    def __init__(self, config: GridTrendConfig) -> None:
        self._config = config

    @property
    def max_positions(self) -> int:
        return self._config.num_levels

    @property
    def min_candles(self) -> dict[str, int]:
        min_needed = max(
            self._config.ema_slow,
            self._config.adx_period * 2 + 1,  # ADX a besoin de ~2×period
            self._config.atr_period,
        ) + 20
        return {
            self._config.timeframe: max(min_needed, 50),
        }

    def compute_indicators(
        self, candles_by_tf: dict[str, list[Candle]]
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Calcule EMA fast/slow + ADX + ATR sur le timeframe principal."""
        result: dict[str, dict[str, dict[str, Any]]] = {}
        tf = self._config.timeframe
        if tf not in candles_by_tf or not candles_by_tf[tf]:
            return result

        candles = candles_by_tf[tf]
        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)

        ema_fast_arr = ema(closes, self._config.ema_fast)
        ema_slow_arr = ema(closes, self._config.ema_slow)
        adx_arr, _, _ = adx(highs, lows, closes, self._config.adx_period)
        atr_arr = atr(highs, lows, closes, self._config.atr_period)

        indicators: dict[str, dict[str, Any]] = {}
        for i, c in enumerate(candles):
            ts = c.timestamp.isoformat()
            indicators[ts] = {
                "ema_fast": float(ema_fast_arr[i]),
                "ema_slow": float(ema_slow_arr[i]),
                "adx": float(adx_arr[i]),
                "atr": float(atr_arr[i]),
                "close": c.close,
            }
        result[tf] = indicators
        return result

    def compute_grid(
        self, ctx: StrategyContext, grid_state: GridState
    ) -> list[GridLevel]:
        """Niveaux pullback si trend confirmé, [] sinon."""
        indicators = ctx.indicators.get(self._config.timeframe, {})
        ema_fast_val = indicators.get("ema_fast", float("nan"))
        ema_slow_val = indicators.get("ema_slow", float("nan"))
        adx_val = indicators.get("adx", float("nan"))
        atr_val = indicators.get("atr", float("nan"))

        if any(math.isnan(v) for v in (ema_fast_val, ema_slow_val, adx_val, atr_val)):
            return []
        if atr_val <= 0:
            return []

        # Zone neutre : ADX trop faible
        if adx_val < self._config.adx_threshold:
            return []

        # Déterminer direction
        if ema_fast_val > ema_slow_val and "long" in self._config.sides:
            allowed_direction = "long"
        elif ema_fast_val < ema_slow_val and "short" in self._config.sides:
            allowed_direction = "short"
        else:
            return []

        # Direction lock : positions ouvertes dans l'autre sens → pas de nouveaux niveaux
        if grid_state.positions:
            active_dir = grid_state.positions[0].direction
            expected_dir = Direction.LONG if allowed_direction == "long" else Direction.SHORT
            if active_dir != expected_dir:
                return []

        filled_levels = {p.level for p in grid_state.positions}

        levels: list[GridLevel] = []
        for i in range(self._config.num_levels):
            if i in filled_levels:
                continue

            offset = atr_val * (self._config.pull_start + i * self._config.pull_step)

            if allowed_direction == "long":
                entry_price = ema_fast_val - offset
                direction = Direction.LONG
            else:
                entry_price = ema_fast_val + offset
                direction = Direction.SHORT

            if entry_price <= 0:
                continue

            levels.append(GridLevel(
                index=i,
                entry_price=entry_price,
                direction=direction,
                size_fraction=1.0 / self._config.num_levels,
            ))

        return levels

    def should_close_all(
        self, ctx: StrategyContext, grid_state: GridState
    ) -> str | None:
        """Fermer si direction flip (EMA cross) ou SL. Pas de TP SMA."""
        if not grid_state.positions:
            return None

        indicators = ctx.indicators.get(self._config.timeframe, {})
        ema_fast_val = indicators.get("ema_fast", float("nan"))
        ema_slow_val = indicators.get("ema_slow", float("nan"))
        close = indicators.get("close", float("nan"))

        if any(math.isnan(v) for v in (ema_fast_val, ema_slow_val, close)):
            return None

        direction = grid_state.positions[0].direction

        # Force close sur direction flip
        if direction == Direction.LONG and ema_fast_val < ema_slow_val:
            return "direction_flip"
        if direction == Direction.SHORT and ema_fast_val > ema_slow_val:
            return "direction_flip"

        # SL classique
        sl_pct = self._config.sl_percent / 100
        if direction == Direction.LONG:
            if close <= grid_state.avg_entry_price * (1 - sl_pct):
                return "sl_global"
        else:
            if close >= grid_state.avg_entry_price * (1 + sl_pct):
                return "sl_global"

        # Note: trailing stop géré par le fast engine / GridStrategyRunner
        return None

    def get_tp_price(
        self, grid_state: GridState, current_indicators: dict
    ) -> float:
        """Pas de TP fixe — trailing stop géré par le fast engine."""
        return float("nan")

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

    def compute_live_indicators(
        self, candles: list[Candle],
    ) -> dict[str, dict[str, Any]]:
        """Calcule EMA fast/slow + ADX depuis le buffer de candles 1h (mode live/portfolio).

        Retourne {timeframe: {"ema_fast": ..., "ema_slow": ..., "adx": ...}}.
        Nécessite suffisamment de candles pour que ADX soit stable.
        """
        min_needed = max(
            self._config.ema_slow,
            self._config.adx_period * 2 + 1,
        ) + 20
        if len(candles) < min_needed:
            return {}

        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)

        ema_fast_arr = ema(closes, self._config.ema_fast)
        ema_slow_arr = ema(closes, self._config.ema_slow)
        adx_arr, _, _ = adx(highs, lows, closes, self._config.adx_period)

        return {
            self._config.timeframe: {
                "ema_fast": float(ema_fast_arr[-1]),
                "ema_slow": float(ema_slow_arr[-1]),
                "adx": float(adx_arr[-1]),
            }
        }

    def get_params(self) -> dict[str, Any]:
        return {
            "ema_fast": self._config.ema_fast,
            "ema_slow": self._config.ema_slow,
            "adx_period": self._config.adx_period,
            "adx_threshold": self._config.adx_threshold,
            "atr_period": self._config.atr_period,
            "pull_start": self._config.pull_start,
            "pull_step": self._config.pull_step,
            "num_levels": self._config.num_levels,
            "trail_mult": self._config.trail_mult,
            "sl_percent": self._config.sl_percent,
            "sides": self._config.sides,
            "leverage": self._config.leverage,
        }
