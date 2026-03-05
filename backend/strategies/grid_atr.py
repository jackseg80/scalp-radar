"""Stratégie Grid ATR (Mean Reversion Adaptative à la Volatilité).

SMA sur le close + N enveloppes basées sur l'ATR (au lieu de % fixes).
Entrée DCA à chaque niveau touché. TP = retour à la SMA. SL = % depuis prix moyen.
Timeframe : 1h.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from backend.core.config import GridATRConfig
from backend.core.indicators import atr, sma
from backend.core.models import Candle, Direction
from backend.strategies.base import StrategyContext
from backend.strategies.base_grid import BaseGridStrategy, GridLevel, GridState, TF_SECONDS


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

    def _calculate_effective_atr(self, indicators: dict[str, Any]) -> tuple[float, bool]:
        """Calcule l'ATR effectif (avec plancher spacing) et indique si le plancher est actif."""
        atr_val = indicators.get("atr", float("nan"))
        close_val = indicators.get("close", float("nan"))
        min_spacing = self._config.min_grid_spacing_pct

        if np.isnan(atr_val) or atr_val <= 0:
            return float("nan"), False

        if min_spacing > 0 and close_val > 0:
            floor_atr = close_val * min_spacing / 100
            if floor_atr > atr_val:
                return floor_atr, True

        return atr_val, False

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
        close_val = indicators.get("close", float("nan"))

        if np.isnan(sma_val) or np.isnan(atr_val) or atr_val <= 0:
            return []

        # Filtre volatilité minimum (Skip ouverture cycle uniquement)
        if self._config.min_atr_pct > 0 and close_val > 0 and not grid_state.positions:
            atr_pct = atr_val / close_val * 100
            if atr_pct < self._config.min_atr_pct:
                logger.debug(
                    "grid_atr — min_atr_pct bloquant pour {} : {:.2f}% < seuil {:.2f}%",
                    ctx.symbol, atr_pct, self._config.min_atr_pct
                )
                return []

        effective_atr, floor_active = self._calculate_effective_atr(indicators)

        # Log une fois par nouveau cycle si le plancher est actif
        if floor_active and not grid_state.positions:
             logger.info(
                "grid_atr — plancher ATR actif pour {} : spacing={:.1f}%",
                ctx.symbol, self._config.min_grid_spacing_pct
            )

        filled_levels = {p.level for p in grid_state.positions}

        # Déterminer le côté actif
        if grid_state.positions:
            active_direction = grid_state.positions[0].direction
            active_sides = ["long"] if active_direction == Direction.LONG else ["short"]
        else:
            active_sides = self._config.sides

        levels: list[GridLevel] = []
        for i in range(self._config.num_levels):
            if i in filled_levels:
                continue

            multiplier = self._config.atr_multiplier_start + i * self._config.atr_multiplier_step

            if "long" in active_sides:
                entry_price = sma_val - effective_atr * multiplier
                if entry_price > 0:
                    levels.append(GridLevel(
                        index=i,
                        entry_price=entry_price,
                        direction=Direction.LONG,
                        size_fraction=1.0 / self._config.num_levels,
                    ))

            if "short" in active_sides:
                entry_price = sma_val + effective_atr * multiplier
                levels.append(GridLevel(
                    index=i,
                    entry_price=entry_price,
                    direction=Direction.SHORT,
                    size_fraction=1.0 / self._config.num_levels,
                ))

        return levels

    def get_current_conditions(self, ctx: StrategyContext) -> list[dict]:
        """Niveaux enrichis + gates volatilité/spacing pour le dashboard."""
        indicators = ctx.indicators.get(self._config.timeframe, {})
        close_val = indicators.get("close", float("nan"))
        atr_val = indicators.get("atr", float("nan"))

        # Calculer les niveaux même si bloqué par min_atr_pct pour affichage dashboard
        grid_state = GridState(
            positions=[], avg_entry_price=0, total_quantity=0,
            total_notional=0, unrealized_pnl=0,
        )

        # Désactiver temporairement le filtre pour forcer le calcul des niveaux
        old_min = self._config.min_atr_pct
        self._config.min_atr_pct = 0.0
        try:
            levels = self.compute_grid(ctx, grid_state)
        finally:
            self._config.min_atr_pct = old_min

        # 1. Niveaux Grid
        conditions = []
        current_price = close_val or 0
        for lvl in levels:
            entry = lvl.entry_price
            dist_pct = (entry - current_price) / current_price * 100 if current_price > 0 else 0
            abs_dist = abs(dist_pct)
            proximity = "imminent" if abs_dist < 1 else "close" if abs_dist < 3 else "medium" if abs_dist < 6 else "far"
            conditions.append({
                "name": f"Level {lvl.index + 1} ({lvl.direction.value})",
                "met": False,
                "value": entry,
                "threshold": "touch",
                "distance_pct": round(dist_pct, 2),
                "proximity": proximity,
            })

        # 2. Gate Volatilité (ATR%)
        if not np.isnan(close_val) and close_val > 0:
            atr_pct = atr_val / close_val * 100 if not np.isnan(atr_val) else 0
            met_vol = atr_pct >= old_min

            diff = old_min - atr_pct
            val_str = f"{atr_pct:.2f}%"
            if not met_vol and diff > 0:
                val_str += f" (manque {diff:.2f}%)"

            conditions.insert(0, {
                "name": "Volatilité (ATR%)",
                "met": met_vol,
                "value": val_str,
                "threshold": f"min {old_min:.2f}%",
                "gate": True
            })

            # 3. Gate Plancher Spacing
            _, floor_active = self._calculate_effective_atr(indicators)
            conditions.insert(1, {
                "name": "Plancher Spacing",
                "met": True,
                "value": "ACTIF" if floor_active else "Inactif",
                "threshold": f"{self._config.min_grid_spacing_pct:.1f}%",
                "gate": True,
                "spacing_pct_active": floor_active
            })

        return conditions

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

        # TP : retour à la SMA + contrainte profit minimum
        min_profit = self._config.min_profit_pct
        if direction == Direction.LONG and close >= sma_val:
            if min_profit <= 0 or close >= grid_state.avg_entry_price * (1 + min_profit / 100):
                return "tp_global"
        if direction == Direction.SHORT and close <= sma_val:
            if min_profit <= 0 or close <= grid_state.avg_entry_price * (1 - min_profit / 100):
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
            "min_grid_spacing_pct": self._config.min_grid_spacing_pct,
            "min_profit_pct": self._config.min_profit_pct,
            "min_atr_pct": self._config.min_atr_pct,
        }
