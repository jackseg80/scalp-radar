"""Stratégie Grid Momentum — DCA pullback sur breakout Donchian + trailing stop ATR.

Signal d'activation : breakout Donchian (close > donchian_high) + filtre volume + ADX optionnel.
Exécution : grid DCA pullback depuis le prix du breakout.
Sortie : trailing stop ATR (profil convexe) ou SL global %.
Force close sur direction flip (breakout inverse détecté).
Timeframe : 1h (défaut).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from backend.core.config import GridMomentumConfig
from backend.core.indicators import adx as compute_adx, atr, sma
from backend.core.models import Candle, Direction
from backend.strategies.base import StrategyContext
from backend.strategies.base_grid import BaseGridStrategy, GridLevel, GridState


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling max excluant la bougie courante : result[i] = max(arr[i-window:i])."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window, n):
        result[i] = float(np.max(arr[i - window : i]))
    return result


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling min excluant la bougie courante : result[i] = min(arr[i-window:i])."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window, n):
        result[i] = float(np.min(arr[i - window : i]))
    return result


class GridMomentumStrategy(BaseGridStrategy):
    """Grid Momentum — DCA pullback sur breakout Donchian + trailing stop ATR.

    Activation : close > donchian_high (LONG) ou close < donchian_low (SHORT)
                 + volume > volume_sma * vol_multiplier
                 + ADX > adx_threshold (si threshold > 0).
    Niveaux DCA : pullback sous le breakout price (LONG) / au-dessus (SHORT).
    TP : trailing stop ATR (pas de TP fixe).
    SL : avg_entry ± sl_percent%.
    Force close : direction flip (breakout inverse sans filtre volume/ADX).
    """

    name = "grid_momentum"

    def __init__(self, config: GridMomentumConfig) -> None:
        self._config = config

    @property
    def max_positions(self) -> int:
        return self._config.num_levels

    @property
    def min_candles(self) -> dict[str, int]:
        min_needed = max(
            self._config.donchian_period,
            self._config.adx_period * 2 + 1,
            self._config.atr_period,
            self._config.vol_sma_period,
        ) + 20
        return {self._config.timeframe: max(min_needed, 50)}

    def compute_indicators(
        self, candles_by_tf: dict[str, list[Candle]]
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Calcule Donchian high/low, ATR, ADX, volume SMA."""
        result: dict[str, dict[str, dict[str, Any]]] = {}
        tf = self._config.timeframe
        if tf not in candles_by_tf or not candles_by_tf[tf]:
            return result

        candles = candles_by_tf[tf]
        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)

        donchian_high = _rolling_max(highs, self._config.donchian_period)
        donchian_low = _rolling_min(lows, self._config.donchian_period)
        atr_arr = atr(highs, lows, closes, self._config.atr_period)
        vol_sma = sma(volumes, self._config.vol_sma_period)

        adx_arr = np.full(len(closes), np.nan)
        if self._config.adx_threshold > 0:
            adx_arr, _, _ = compute_adx(highs, lows, closes, self._config.adx_period)

        indicators: dict[str, dict[str, Any]] = {}
        for i, c in enumerate(candles):
            ts = c.timestamp.isoformat()
            indicators[ts] = {
                "close": c.close,
                "high": c.high,
                "volume": c.volume,
                "donchian_high": float(donchian_high[i]),
                "donchian_low": float(donchian_low[i]),
                "atr": float(atr_arr[i]),
                "adx": float(adx_arr[i]),
                "volume_sma": float(vol_sma[i]),
            }
        result[tf] = indicators
        return result

    def compute_grid(
        self, ctx: StrategyContext, grid_state: GridState
    ) -> list[GridLevel]:
        """Retourne les niveaux de la grille.

        Si positions ouvertes : niveaux DCA pullback depuis anchor (positions[0].entry_price).
        Sinon : détecte breakout Donchian + volume → niveaux depuis close si breakout.
        Pas de breakout → [].
        """
        indicators = ctx.indicators.get(self._config.timeframe, {})
        close = indicators.get("close", float("nan"))
        atr_val = indicators.get("atr", float("nan"))
        donchian_high = indicators.get("donchian_high", float("nan"))
        donchian_low = indicators.get("donchian_low", float("nan"))
        volume = indicators.get("volume", float("nan"))
        volume_sma = indicators.get("volume_sma", float("nan"))
        adx_val = indicators.get("adx", float("nan"))

        if any(math.isnan(v) for v in (close, atr_val)):
            return []
        if atr_val <= 0:
            return []

        num_levels = self._config.num_levels
        sides = self._config.sides

        # === Positions existantes : niveaux DCA pullback depuis anchor ===
        if grid_state.positions:
            anchor = grid_state.positions[0].entry_price
            direction = grid_state.positions[0].direction
            filled_levels = {p.level for p in grid_state.positions}
            levels: list[GridLevel] = []
            for k in range(num_levels):
                if k in filled_levels:
                    continue
                if k == 0:
                    entry_price = anchor
                else:
                    offset = atr_val * (self._config.pullback_start + (k - 1) * self._config.pullback_step)
                    if direction == Direction.LONG:
                        entry_price = anchor - offset
                    else:
                        entry_price = anchor + offset
                if entry_price > 0:
                    levels.append(GridLevel(
                        index=k,
                        entry_price=entry_price,
                        direction=direction,
                        size_fraction=1.0 / num_levels,
                    ))
            return levels

        # === Pas de positions : détecter breakout ===
        if any(math.isnan(v) for v in (donchian_high, donchian_low, volume, volume_sma)):
            return []

        # Filtre volume
        volume_ok = volume > volume_sma * self._config.vol_multiplier

        # Filtre ADX (optionnel : 0 = désactivé)
        adx_ok = True
        if self._config.adx_threshold > 0:
            if math.isnan(adx_val) or adx_val < self._config.adx_threshold:
                adx_ok = False

        if not volume_ok or not adx_ok:
            return []

        # Breakout LONG
        long_breakout = "long" in sides and close > donchian_high
        # Breakout SHORT
        short_breakout = "short" in sides and close < donchian_low

        if not long_breakout and not short_breakout:
            return []

        # Si les deux breakouts sont détectés (très rare), LONG a priorité
        direction = Direction.LONG if long_breakout else Direction.SHORT

        levels = []
        for k in range(num_levels):
            if k == 0:
                entry_price = close
            else:
                offset = atr_val * (self._config.pullback_start + (k - 1) * self._config.pullback_step)
                if direction == Direction.LONG:
                    entry_price = close - offset
                else:
                    entry_price = close + offset
            if entry_price > 0:
                levels.append(GridLevel(
                    index=k,
                    entry_price=entry_price,
                    direction=direction,
                    size_fraction=1.0 / num_levels,
                ))
        return levels

    def should_close_all(
        self, ctx: StrategyContext, grid_state: GridState
    ) -> str | None:
        """Direction flip + trailing stop. PAS de SL ici (géré par get_sl_price).

        Direction flip : breakout inverse détecté SANS filtre volume/ADX (sortie de protection).
        Trailing stop : close < HWM - trailing_atr_mult * ATR (LONG).
        """
        if not grid_state.positions:
            return None

        indicators = ctx.indicators.get(self._config.timeframe, {})
        close = indicators.get("close", float("nan"))
        donchian_high = indicators.get("donchian_high", float("nan"))
        donchian_low = indicators.get("donchian_low", float("nan"))
        atr_val = indicators.get("atr", float("nan"))
        hwm = indicators.get("hwm", float("nan"))

        if math.isnan(close):
            return None

        direction = grid_state.positions[0].direction

        # Direction flip (sans filtre volume/ADX — sortie de protection)
        if not math.isnan(donchian_low) and direction == Direction.LONG and close < donchian_low:
            return "direction_flip"
        if not math.isnan(donchian_high) and direction == Direction.SHORT and close > donchian_high:
            return "direction_flip"

        # Trailing stop (si HWM disponible via le runner)
        if not math.isnan(hwm) and not math.isnan(atr_val) and atr_val > 0:
            trail_distance = atr_val * self._config.trailing_atr_mult
            if direction == Direction.LONG:
                if close < hwm - trail_distance:
                    return "trail_stop"
            else:
                if close > hwm + trail_distance:
                    return "trail_stop"

        return None

    def get_tp_price(
        self, grid_state: GridState, current_indicators: dict
    ) -> float:
        """Pas de TP fixe — trailing stop géré par should_close_all() / fast engine."""
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
        """Calcule Donchian + ATR + volume SMA + ADX depuis le buffer 1h (mode live/portfolio).

        HWM n'est PAS calculé ici (pas accès aux positions) — géré par le runner.
        """
        min_needed = max(
            self._config.donchian_period,
            self._config.adx_period * 2 + 1,
            self._config.atr_period,
            self._config.vol_sma_period,
        ) + 20
        if len(candles) < min_needed:
            return {}

        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)

        donchian_high = _rolling_max(highs, self._config.donchian_period)
        donchian_low = _rolling_min(lows, self._config.donchian_period)
        atr_arr = atr(highs, lows, closes, self._config.atr_period)
        vol_sma = sma(volumes, self._config.vol_sma_period)

        i = len(candles) - 1

        result: dict[str, Any] = {
            "donchian_high": float(donchian_high[i]),
            "donchian_low": float(donchian_low[i]),
            "atr": float(atr_arr[i]),
            "volume_sma": float(vol_sma[i]),
        }

        if self._config.adx_threshold > 0:
            adx_arr, _, _ = compute_adx(highs, lows, closes, self._config.adx_period)
            result["adx"] = float(adx_arr[i])

        return {self._config.timeframe: result}

    def get_params(self) -> dict[str, Any]:
        return {
            "timeframe": self._config.timeframe,
            "donchian_period": self._config.donchian_period,
            "vol_sma_period": self._config.vol_sma_period,
            "vol_multiplier": self._config.vol_multiplier,
            "adx_period": self._config.adx_period,
            "adx_threshold": self._config.adx_threshold,
            "atr_period": self._config.atr_period,
            "pullback_start": self._config.pullback_start,
            "pullback_step": self._config.pullback_step,
            "num_levels": self._config.num_levels,
            "trailing_atr_mult": self._config.trailing_atr_mult,
            "sl_percent": self._config.sl_percent,
            "sides": self._config.sides,
            "leverage": self._config.leverage,
            "cooldown_candles": self._config.cooldown_candles,
        }
