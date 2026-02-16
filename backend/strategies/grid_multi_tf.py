"""Stratégie Grid Multi-TF (Supertrend 4h + Grid ATR 1h).

Filtre directionnel Supertrend sur le 4h :
  - ST UP → DCA LONG sous la SMA (enveloppes ATR)
  - ST DOWN → DCA SHORT au-dessus de la SMA
Exécution identique à grid_atr sur le 1h.
TP = retour à la SMA. SL = % depuis prix moyen.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from backend.core.config import GridMultiTFConfig
from backend.core.indicators import atr, sma, supertrend
from backend.core.models import Candle, Direction
from backend.strategies.base import StrategyContext
from backend.strategies.base_grid import BaseGridStrategy, GridLevel, GridState


class GridMultiTFStrategy(BaseGridStrategy):
    """Grid Multi-TF — filtre Supertrend 4h + exécution Grid ATR 1h.

    Enveloppes LONG : SMA - ATR × (start + i × step) quand ST 4h = UP
    Enveloppes SHORT : SMA + ATR × (start + i × step) quand ST 4h = DOWN
    """

    name = "grid_multi_tf"

    def __init__(self, config: GridMultiTFConfig) -> None:
        self._config = config

    @property
    def max_positions(self) -> int:
        return self._config.num_levels

    @property
    def min_candles(self) -> dict[str, int]:
        min_needed = max(
            self._config.ma_period,
            self._config.atr_period,
            self._config.st_atr_period,
        ) + 20
        return {
            self._config.timeframe: max(min_needed, 50),
        }

    def compute_indicators(
        self, candles_by_tf: dict[str, list[Candle]]
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Calcule SMA + ATR sur 1h ET Supertrend 4h (resampleé depuis 1h)."""
        result: dict[str, dict[str, dict[str, Any]]] = {}
        tf = self._config.timeframe
        if tf not in candles_by_tf or not candles_by_tf[tf]:
            return result

        candles = candles_by_tf[tf]
        n = len(candles)
        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)

        sma_arr = sma(closes, self._config.ma_period)
        atr_arr = atr(highs, lows, closes, self._config.atr_period)

        # --- Resampling 1h → 4h + Supertrend (anti-lookahead) ---
        st_dir_1h = _compute_st_4h_mapped_to_1h(
            candles, highs, lows, closes,
            self._config.st_atr_period, self._config.st_atr_multiplier,
        )

        # Indicateurs 1h
        indicators_1h: dict[str, dict[str, Any]] = {}
        for i, c in enumerate(candles):
            ts = c.timestamp.isoformat()
            indicators_1h[ts] = {
                "sma": float(sma_arr[i]),
                "atr": float(atr_arr[i]),
                "close": c.close,
            }
        result[tf] = indicators_1h

        # Indicateurs 4h mappés sur timestamps 1h (pour StrategyContext)
        indicators_4h: dict[str, dict[str, Any]] = {}
        for i, c in enumerate(candles):
            ts = c.timestamp.isoformat()
            indicators_4h[ts] = {"st_direction": float(st_dir_1h[i])}
        result["4h"] = indicators_4h

        return result

    def compute_grid(
        self, ctx: StrategyContext, grid_state: GridState
    ) -> list[GridLevel]:
        """Retourne les niveaux non encore remplis, filtrés par Supertrend 4h.

        Direction déterminée par Supertrend 4h :
          - UP (1) → LONG : enveloppes sous SMA
          - DOWN (-1) → SHORT : enveloppes au-dessus SMA
          - NaN/None → pas de trading (safety)
        """
        indicators = ctx.indicators.get(self._config.timeframe, {})
        sma_val = indicators.get("sma", float("nan"))
        atr_val = indicators.get("atr", float("nan"))

        if np.isnan(sma_val) or np.isnan(atr_val) or atr_val <= 0:
            return []

        # Lire la direction Supertrend 4h
        ind_4h = ctx.indicators.get("4h", {})
        st_direction = ind_4h.get("st_direction")
        if st_direction is None or (isinstance(st_direction, float) and math.isnan(st_direction)):
            return []

        # Déterminer le côté autorisé
        if st_direction == 1:
            allowed_direction = "long"
        elif st_direction == -1:
            allowed_direction = "short"
        else:
            return []

        # Vérifier la whitelist sides
        if allowed_direction not in self._config.sides:
            return []

        filled_levels = {p.level for p in grid_state.positions}

        # Si positions ouvertes dans l'autre direction, ne pas ajouter
        if grid_state.positions:
            active_dir = grid_state.positions[0].direction
            expected_dir = Direction.LONG if allowed_direction == "long" else Direction.SHORT
            if active_dir != expected_dir:
                return []

        levels: list[GridLevel] = []
        for i in range(self._config.num_levels):
            if i in filled_levels:
                continue

            multiplier = (
                self._config.atr_multiplier_start + i * self._config.atr_multiplier_step
            )

            if allowed_direction == "long":
                entry_price = sma_val - atr_val * multiplier
                direction = Direction.LONG
            else:
                entry_price = sma_val + atr_val * multiplier
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
        """Fermer si TP/SL classique OU si Supertrend 4h a flippé."""
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

        # Direction flip : Supertrend 4h a changé
        ind_4h = ctx.indicators.get("4h", {})
        st_direction = ind_4h.get("st_direction")
        if st_direction is not None and not (isinstance(st_direction, float) and math.isnan(st_direction)):
            if direction == Direction.LONG and st_direction == -1:
                return "direction_flip"
            if direction == Direction.SHORT and st_direction == 1:
                return "direction_flip"

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
            "st_atr_period": self._config.st_atr_period,
            "st_atr_multiplier": self._config.st_atr_multiplier,
            "ma_period": self._config.ma_period,
            "atr_period": self._config.atr_period,
            "atr_multiplier_start": self._config.atr_multiplier_start,
            "atr_multiplier_step": self._config.atr_multiplier_step,
            "num_levels": self._config.num_levels,
            "sl_percent": self._config.sl_percent,
            "sides": self._config.sides,
            "leverage": self._config.leverage,
        }


# ─── Helper : Supertrend 4h resampleé sur indices 1h ──────────────────


def _compute_st_4h_mapped_to_1h(
    candles_1h: list[Candle],
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    st_atr_period: int,
    st_atr_multiplier: float,
) -> np.ndarray:
    """Resample 1h → 4h, calcule Supertrend, mappe sur indices 1h.

    Anti-lookahead : chaque candle 1h utilise la direction du bucket 4h
    PRÉCÉDENT (pas le courant, qui n'est pas encore clôturé).

    Returns:
        Array (n,) de directions 1/-1/NaN mappées sur les candles 1h.
    """
    n = len(candles_1h)
    if n == 0:
        return np.array([], dtype=float)

    # Bucket 4h = timestamp // 14400 (frontières UTC 00h/04h/08h/12h/16h/20h)
    bucket_size = 14400
    timestamps = np.array([c.timestamp.timestamp() for c in candles_1h])
    buckets = (timestamps // bucket_size).astype(np.int64)

    # Buckets uniques dans l'ordre
    unique_buckets: list[int] = []
    prev_bucket = -1
    for b in buckets:
        if b != prev_bucket:
            unique_buckets.append(int(b))
            prev_bucket = b
    unique_buckets_arr = np.array(unique_buckets)

    # OHLC 4h
    h4_highs_list: list[float] = []
    h4_lows_list: list[float] = []
    h4_closes_list: list[float] = []

    for bucket_id in unique_buckets_arr:
        mask = buckets == bucket_id
        h4_highs_list.append(float(np.max(highs[mask])))
        h4_lows_list.append(float(np.min(lows[mask])))
        indices = np.where(mask)[0]
        h4_closes_list.append(float(closes[indices[-1]]))

    h4_highs = np.array(h4_highs_list, dtype=float)
    h4_lows = np.array(h4_lows_list, dtype=float)
    h4_closes = np.array(h4_closes_list, dtype=float)

    if len(h4_closes) == 0:
        return np.full(n, np.nan)

    # Supertrend 4h
    atr_4h = atr(h4_highs, h4_lows, h4_closes, st_atr_period)
    _, st_dir_4h = supertrend(h4_highs, h4_lows, h4_closes, atr_4h, st_atr_multiplier)

    # Mapping anti-lookahead : candle 1h[i] → bucket PRÉCÉDENT
    st_dir_1h = np.full(n, np.nan)
    for i in range(n):
        current_bucket = buckets[i]
        idx = np.searchsorted(unique_buckets_arr, current_bucket, side="left") - 1
        if idx >= 0 and not np.isnan(st_dir_4h[idx]):
            st_dir_1h[i] = st_dir_4h[idx]

    return st_dir_1h
