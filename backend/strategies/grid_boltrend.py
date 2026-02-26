"""Stratégie Grid BolTrend — DCA event-driven sur breakout Bollinger.

Signal d'activation : breakout Bollinger (même logique que boltrend.py)
Exécution : grid DCA multi-niveaux dans la direction du breakout
TP inverse : close < SMA (LONG) ou close > SMA (SHORT) — breakout épuisé
SL global : % fixe depuis prix moyen d'entrée
Timeframe : 1h (défaut).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from backend.core.config import GridBolTrendConfig
from backend.core.indicators import atr, bollinger_bands, sma
from backend.core.models import Candle, Direction
from backend.strategies.base import StrategyContext
from backend.strategies.base_grid import BaseGridStrategy, GridLevel, GridState, TF_SECONDS


class GridBolTrendStrategy(BaseGridStrategy):
    """Grid BolTrend — DCA event-driven sur breakout Bollinger + filtre SMA.

    Activation : breakout Bollinger (prev_close dans les bandes, close hors bandes)
                 + spread suffisant + filtre SMA long terme.
    Niveaux DCA : close ∓ k × ATR × atr_spacing_mult (fixés au breakout).
    TP inverse : close < bb_sma (LONG) ou close > bb_sma (SHORT).
    SL global : avg_entry ± sl_percent%.
    """

    name = "grid_boltrend"

    def __init__(self, config: GridBolTrendConfig) -> None:
        self._config = config

    @property
    def max_positions(self) -> int:
        return self._config.num_levels

    @property
    def min_candles(self) -> dict[str, int]:
        # Prendre le max sur tous les per_asset overrides (ex: DOGE long_ma_window=400)
        max_bol = self._config.bol_window
        max_ma = self._config.long_ma_window
        for overrides in getattr(self._config, "per_asset", {}).values():
            max_bol = max(max_bol, overrides.get("bol_window", 0))
            max_ma = max(max_ma, overrides.get("long_ma_window", 0))
        min_needed = max(max_bol, max_ma) + 20
        return {self._config.timeframe: max(min_needed, 50)}

    def compute_indicators(
        self, candles_by_tf: dict[str, list[Candle]]
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Pré-calcule BB(bol_window, bol_std), SMA(long_ma_window), ATR."""
        result: dict[str, dict[str, dict[str, Any]]] = {}
        tf = self._config.timeframe
        if tf in candles_by_tf and candles_by_tf[tf]:
            candles = candles_by_tf[tf]
            closes = np.array([c.close for c in candles], dtype=float)
            highs_arr = np.array([c.high for c in candles], dtype=float)
            lows_arr = np.array([c.low for c in candles], dtype=float)

            bb_sma_arr, bb_upper, bb_lower = bollinger_bands(
                closes, self._config.bol_window, self._config.bol_std,
            )
            long_ma = sma(closes, self._config.long_ma_window)
            atr_arr = atr(highs_arr, lows_arr, closes, self._config.atr_period)

            indicators: dict[str, dict[str, Any]] = {}
            for i, c in enumerate(candles):
                ts = c.timestamp.isoformat()
                prev_close = float(closes[i - 1]) if i > 0 else float("nan")
                prev_upper = float(bb_upper[i - 1]) if i > 0 else float("nan")
                prev_lower = float(bb_lower[i - 1]) if i > 0 else float("nan")

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
                    "atr": float(atr_arr[i]),
                    "prev_close": prev_close,
                    "prev_upper": prev_upper,
                    "prev_lower": prev_lower,
                    "prev_spread": prev_spread,
                }
            result[tf] = indicators
        return result

    def get_current_conditions(self, ctx: StrategyContext) -> list[dict]:
        """Niveaux enrichis + gate position Bollinger pour le dashboard."""
        conditions = super().get_current_conditions(ctx)

        main_tf = getattr(self._config, "timeframe", "1h")
        ind = ctx.indicators.get(main_tf, {}) if ctx.indicators else {}
        close = ind.get("close")
        bb_upper = ind.get("bb_upper")
        bb_lower = ind.get("bb_lower")

        if close and bb_upper and bb_lower:
            if close < bb_lower:
                bb_label = "SOUS lower"
                bb_met = True
            elif close > bb_upper:
                bb_label = "AU-DESSUS upper"
                bb_met = True
            else:
                bb_label = "DANS les bandes"
                bb_met = False
        else:
            bb_label = "INDEFINI"
            bb_met = False

        conditions.insert(0, {
            "name": "Position Bollinger",
            "met": bb_met,
            "value": bb_label,
            "threshold": "breakout",
            "gate": True,
        })
        return conditions

    def compute_grid(
        self, ctx: StrategyContext, grid_state: GridState
    ) -> list[GridLevel]:
        """Retourne les niveaux de la grille.

        Si positions ouvertes : niveaux DCA depuis positions[0].entry_price (anchor).
        Sinon : détecte breakout → niveaux depuis close si breakout.
        Pas de breakout → [].
        """
        indicators = ctx.indicators.get(self._config.timeframe, {})
        close = indicators.get("close", float("nan"))
        bb_upper = indicators.get("bb_upper", float("nan"))
        bb_lower = indicators.get("bb_lower", float("nan"))
        long_ma = indicators.get("long_ma", float("nan"))
        atr_val = indicators.get("atr", float("nan"))
        prev_close = indicators.get("prev_close", float("nan"))
        prev_upper = indicators.get("prev_upper", float("nan"))
        prev_lower = indicators.get("prev_lower", float("nan"))
        prev_spread = indicators.get("prev_spread", float("nan"))

        if any(np.isnan(v) for v in [close, atr_val]):
            return []

        num_levels = self._config.num_levels
        spacing = atr_val * self._config.atr_spacing_mult
        sides = self._config.sides

        # === Positions existantes : niveaux DCA depuis l'anchor ===
        if grid_state.positions:
            anchor = grid_state.positions[0].entry_price
            direction = grid_state.positions[0].direction
            filled_levels = {p.level for p in grid_state.positions}
            levels: list[GridLevel] = []
            for k in range(num_levels):
                if k in filled_levels:
                    continue
                if direction == Direction.LONG:
                    entry_price = anchor - k * spacing
                else:
                    entry_price = anchor + k * spacing
                if entry_price > 0:
                    levels.append(GridLevel(
                        index=k,
                        entry_price=entry_price,
                        direction=direction,
                        size_fraction=1.0 / num_levels,
                    ))
            return levels

        # === Pas de positions : détecter breakout ===
        if any(np.isnan(v) for v in [bb_upper, bb_lower, long_ma, prev_close, prev_upper, prev_lower]):
            return []

        min_spread = self._config.min_bol_spread
        spread_ok = not np.isnan(prev_spread) and prev_spread > min_spread

        # Breakout LONG
        long_breakout = (
            "long" in sides
            and prev_close < prev_upper
            and close > bb_upper
            and spread_ok
            and close > long_ma
        )

        # Breakout SHORT
        short_breakout = (
            "short" in sides
            and prev_close > prev_lower
            and close < bb_lower
            and spread_ok
            and close < long_ma
        )

        if not long_breakout and not short_breakout:
            return []

        direction = Direction.LONG if long_breakout else Direction.SHORT
        levels = []
        for k in range(num_levels):
            if direction == Direction.LONG:
                entry_price = close - k * spacing
            else:
                entry_price = close + k * spacing
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
        """TP inverse + SL global. Ferme TOUTES les positions d'un coup.

        TP LONG : close < bb_sma (breakout épuisé, retour sous la SMA).
        TP SHORT : close > bb_sma (breakout épuisé, retour au-dessus de la SMA).
        SL : avg_entry ± sl_percent%.
        """
        if not grid_state.positions:
            return None

        indicators = ctx.indicators.get(self._config.timeframe, {})
        close = indicators.get("close", float("nan"))
        bb_sma = indicators.get("bb_sma", float("nan"))

        if np.isnan(close) or np.isnan(bb_sma):
            return None

        direction = grid_state.positions[0].direction

        # SL global
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
            tf_secs = TF_SECONDS.get(self._config.timeframe, 3600)
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

        # TP inverse : close croise la SMA (breakout épuisé)
        if direction == Direction.LONG and close < bb_sma:
            return "signal_exit"
        if direction == Direction.SHORT and close > bb_sma:
            return "signal_exit"

        return None

    def get_tp_price(
        self, grid_state: GridState, current_indicators: dict
    ) -> float:
        """TP inverse géré par should_close_all() (close < bb_sma LONG).

        check_global_tp_sl() utilise la convention standard high >= tp_price,
        incompatible avec un TP inverse. On retourne NaN pour le désactiver.
        """
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
        """Calcule les indicateurs Bollinger Bands pour le mode live/portfolio.

        IncrementalIndicatorEngine ne calcule pas les BB ni la long_ma.
        Cette méthode comble le manque depuis le buffer de candles brutes.

        Nécessite au moins max(bol_window, long_ma_window) + 1 candles pour
        avoir à la fois la valeur courante (i) et la valeur précédente (i-1).
        Retourne {} si le buffer est insuffisant.
        """
        min_needed = max(self._config.bol_window, self._config.long_ma_window) + 1
        if len(candles) < min_needed:
            return {}

        closes = np.array([c.close for c in candles], dtype=float)
        highs_arr = np.array([c.high for c in candles], dtype=float)
        lows_arr = np.array([c.low for c in candles], dtype=float)

        bb_sma_arr, bb_upper_arr, bb_lower_arr = bollinger_bands(
            closes, self._config.bol_window, self._config.bol_std,
        )
        long_ma_arr = sma(closes, self._config.long_ma_window)

        i = len(candles) - 1  # index de la candle courante (la plus récente)

        prev_close = float(closes[i - 1])
        prev_upper = float(bb_upper_arr[i - 1])
        prev_lower = float(bb_lower_arr[i - 1])

        if not np.isnan(prev_upper) and not np.isnan(prev_lower) and prev_lower > 0:
            prev_spread = (prev_upper - prev_lower) / prev_lower
        else:
            prev_spread = float("nan")

        tf = self._config.timeframe
        return {
            tf: {
                "bb_sma": float(bb_sma_arr[i]),
                "bb_upper": float(bb_upper_arr[i]),
                "bb_lower": float(bb_lower_arr[i]),
                "long_ma": float(long_ma_arr[i]),
                "prev_close": prev_close,
                "prev_upper": prev_upper,
                "prev_lower": prev_lower,
                "prev_spread": prev_spread,
            }
        }

    def get_params(self) -> dict[str, Any]:
        return {
            "timeframe": self._config.timeframe,
            "bol_window": self._config.bol_window,
            "bol_std": self._config.bol_std,
            "long_ma_window": self._config.long_ma_window,
            "min_bol_spread": self._config.min_bol_spread,
            "atr_period": self._config.atr_period,
            "atr_spacing_mult": self._config.atr_spacing_mult,
            "num_levels": self._config.num_levels,
            "sl_percent": self._config.sl_percent,
            "sides": self._config.sides,
            "leverage": self._config.leverage,
            "max_hold_candles": self._config.max_hold_candles,
        }
