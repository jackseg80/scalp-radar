"""Stratégie VWAP + RSI Mean Reversion pour Scalp Radar.

Logique : entre en position quand le prix s'éloigne du VWAP avec un RSI
extrême et un spike de volume, filtré par la tendance 15m.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from backend.core.config import VwapRsiConfig
from backend.core.indicators import (
    adx,
    atr,
    detect_market_regime,
    rsi,
    sma,
    volume_sma,
    vwap_rolling,
)
from backend.core.models import Candle, Direction, MarketRegime, SignalStrength
from backend.strategies.base import BaseStrategy, OpenPosition, StrategyContext, StrategySignal


class VwapRsiStrategy(BaseStrategy):
    """VWAP + RSI Mean Reversion.

    Entry LONG : prix < VWAP - deviation, RSI < 25, volume spike, 15m pas bearish.
    Entry SHORT : symétrique.
    Exit anticipée : RSI revient > 50 (LONG) ou < 50 (SHORT) en profit.
    """

    name = "vwap_rsi"

    def __init__(self, config: VwapRsiConfig) -> None:
        self._config = config

    @property
    def min_candles(self) -> dict[str, int]:
        return {
            self._config.timeframe: 300,  # 288 pour VWAP 24h + marge
            self._config.trend_filter_timeframe: 50,
        }

    def compute_indicators(
        self, candles_by_tf: dict[str, list[Candle]]
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Pré-calcule RSI, VWAP, ADX+DI, ATR, volume SMA par timeframe."""
        result: dict[str, dict[str, dict[str, float]]] = {}

        # TF principal (ex: 5m)
        main_tf = self._config.timeframe
        if main_tf in candles_by_tf and candles_by_tf[main_tf]:
            result[main_tf] = self._compute_tf_indicators(
                candles_by_tf[main_tf], is_main=True
            )

        # TF filtre (ex: 15m)
        filter_tf = self._config.trend_filter_timeframe
        if filter_tf in candles_by_tf and candles_by_tf[filter_tf]:
            result[filter_tf] = self._compute_tf_indicators(
                candles_by_tf[filter_tf], is_main=False
            )

        return result

    def _compute_tf_indicators(
        self, candles: list[Candle], is_main: bool
    ) -> dict[str, dict[str, float]]:
        """Calcule les indicateurs pour un timeframe donné."""
        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)

        rsi_arr = rsi(closes, self._config.rsi_period)
        adx_arr, di_plus_arr, di_minus_arr = adx(highs, lows, closes)
        atr_arr = atr(highs, lows, closes)
        atr_sma_arr = sma(atr_arr[~np.isnan(atr_arr)], 20) if np.any(~np.isnan(atr_arr)) else np.array([])

        indicators: dict[str, dict[str, float]] = {}

        if is_main:
            vwap_arr = vwap_rolling(highs, lows, closes, volumes)
            vol_sma_arr = volume_sma(volumes)

            # Reconstruire atr_sma aligné avec atr_arr
            atr_sma_full = np.full_like(atr_arr, np.nan)
            valid_atr = ~np.isnan(atr_arr)
            valid_indices = np.where(valid_atr)[0]
            if len(valid_indices) >= 20:
                atr_valid = atr_arr[valid_atr]
                atr_sma_valid = sma(atr_valid, 20)
                for j, idx in enumerate(valid_indices):
                    if not np.isnan(atr_sma_valid[j]):
                        atr_sma_full[idx] = atr_sma_valid[j]

            for i, candle in enumerate(candles):
                ts = candle.timestamp.isoformat()
                indicators[ts] = {
                    "rsi": float(rsi_arr[i]),
                    "vwap": float(vwap_arr[i]),
                    "adx": float(adx_arr[i]),
                    "di_plus": float(di_plus_arr[i]),
                    "di_minus": float(di_minus_arr[i]),
                    "atr": float(atr_arr[i]),
                    "atr_sma": float(atr_sma_full[i]),
                    "volume_sma": float(vol_sma_arr[i]),
                    "volume": float(volumes[i]),
                    "close": float(closes[i]),
                }
        else:
            for i, candle in enumerate(candles):
                ts = candle.timestamp.isoformat()
                indicators[ts] = {
                    "rsi": float(rsi_arr[i]),
                    "adx": float(adx_arr[i]),
                    "di_plus": float(di_plus_arr[i]),
                    "di_minus": float(di_minus_arr[i]),
                }

        return indicators

    def evaluate(self, ctx: StrategyContext) -> StrategySignal | None:
        """Évalue les conditions d'entrée VWAP+RSI."""
        main_tf = self._config.timeframe
        filter_tf = self._config.trend_filter_timeframe

        main_ind = ctx.indicators.get(main_tf)
        filter_ind = ctx.indicators.get(filter_tf)

        if not main_ind or not filter_ind:
            return None

        # Vérifier que les indicateurs sont disponibles (pas NaN)
        rsi_val = main_ind.get("rsi", float("nan"))
        vwap_val = main_ind.get("vwap", float("nan"))
        vol_val = main_ind.get("volume", 0.0)
        vol_sma_val = main_ind.get("volume_sma", float("nan"))
        close_val = main_ind.get("close", float("nan"))
        adx_val = main_ind.get("adx", float("nan"))
        di_plus = main_ind.get("di_plus", float("nan"))
        di_minus = main_ind.get("di_minus", float("nan"))
        atr_val = main_ind.get("atr", float("nan"))
        atr_sma_val = main_ind.get("atr_sma", float("nan"))

        if any(np.isnan(v) for v in [rsi_val, vwap_val, vol_sma_val, close_val]):
            return None

        # Filtrage 15m
        filter_di_plus = filter_ind.get("di_plus", float("nan"))
        filter_di_minus = filter_ind.get("di_minus", float("nan"))
        filter_adx = filter_ind.get("adx", float("nan"))

        # Filtre principal : ADX 15m > seuil = marché en tendance → pas de mean reversion
        if not np.isnan(filter_adx) and filter_adx > self._config.trend_adx_threshold:
            return None

        is_15m_bearish = (
            not np.isnan(filter_adx)
            and filter_adx > 20
            and not np.isnan(filter_di_minus)
            and not np.isnan(filter_di_plus)
            and filter_di_minus > filter_di_plus
        )
        is_15m_bullish = (
            not np.isnan(filter_adx)
            and filter_adx > 20
            and not np.isnan(filter_di_plus)
            and not np.isnan(filter_di_minus)
            and filter_di_plus > filter_di_minus
        )

        # Volume spike
        has_volume_spike = (
            vol_sma_val > 0
            and vol_val > vol_sma_val * self._config.volume_spike_multiplier
        )

        # VWAP deviation
        vwap_dev = (close_val - vwap_val) / vwap_val * 100 if vwap_val > 0 else 0

        # Régime de marché (5m)
        regime = detect_market_regime(adx_val, di_plus, di_minus, atr_val, atr_sma_val)

        # Filtre régime 5m : mean reversion = range ou low vol uniquement
        if regime not in (MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY):
            return None

        # Conditions LONG
        long_signal = (
            rsi_val < self._config.rsi_long_threshold
            and vwap_dev < -self._config.vwap_deviation_entry
            and has_volume_spike
            and not is_15m_bearish
        )

        # Conditions SHORT
        short_signal = (
            rsi_val > self._config.rsi_short_threshold
            and vwap_dev > self._config.vwap_deviation_entry
            and has_volume_spike
            and not is_15m_bullish
        )

        if not long_signal and not short_signal:
            return None

        direction = Direction.LONG if long_signal else Direction.SHORT

        # Score composé
        if direction == Direction.LONG:
            rsi_score = max(0.0, min(1.0, (self._config.rsi_long_threshold - rsi_val) / self._config.rsi_long_threshold))
            vwap_score = max(0.0, min(1.0, abs(vwap_dev) / (self._config.vwap_deviation_entry * 3)))
        else:
            rsi_score = max(0.0, min(1.0, (rsi_val - self._config.rsi_short_threshold) / (100 - self._config.rsi_short_threshold)))
            vwap_score = max(0.0, min(1.0, abs(vwap_dev) / (self._config.vwap_deviation_entry * 3)))

        volume_score = max(0.0, min(1.0, (vol_val / vol_sma_val - 1) / (self._config.volume_spike_multiplier * 2))) if vol_sma_val > 0 else 0.0

        # Trend alignment score
        trend_score = 0.5  # Neutre par défaut
        if direction == Direction.LONG and is_15m_bullish:
            trend_score = 1.0
        elif direction == Direction.SHORT and is_15m_bearish:
            trend_score = 1.0

        score = rsi_score * 0.35 + vwap_score * 0.25 + volume_score * 0.2 + trend_score * 0.2

        # Strength
        if score >= 0.7:
            strength = SignalStrength.STRONG
        elif score >= 0.4:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        # TP / SL
        entry_price = close_val
        if direction == Direction.LONG:
            tp_price = entry_price * (1 + self._config.tp_percent / 100)
            sl_price = entry_price * (1 - self._config.sl_percent / 100)
        else:
            tp_price = entry_price * (1 - self._config.tp_percent / 100)
            sl_price = entry_price * (1 + self._config.sl_percent / 100)

        return StrategySignal(
            direction=direction,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            score=score,
            strength=strength,
            market_regime=regime,
            signals_detail={
                "rsi_score": rsi_score,
                "vwap_score": vwap_score,
                "volume_score": volume_score,
                "trend_score": trend_score,
                "rsi": rsi_val,
                "vwap_deviation": vwap_dev,
            },
        )

    def get_current_conditions(self, ctx: StrategyContext) -> list[dict]:
        """Conditions d'entrée VWAP+RSI pour le dashboard."""
        main_tf = self._config.timeframe
        main_ind = ctx.indicators.get(main_tf, {})

        rsi_val = main_ind.get("rsi", float("nan"))
        vwap_val = main_ind.get("vwap", float("nan"))
        close_val = main_ind.get("close", float("nan"))
        vol_val = main_ind.get("volume", 0.0)
        vol_sma_val = main_ind.get("volume_sma", float("nan"))
        adx_val = main_ind.get("adx", float("nan"))
        di_plus = main_ind.get("di_plus", float("nan"))
        di_minus = main_ind.get("di_minus", float("nan"))
        atr_val = main_ind.get("atr", float("nan"))
        atr_sma_val = main_ind.get("atr_sma", float("nan"))

        # VWAP deviation
        vwap_dev = abs((close_val - vwap_val) / vwap_val * 100) if (
            not np.isnan(close_val) and not np.isnan(vwap_val) and vwap_val > 0
        ) else float("nan")

        # Volume ratio
        vol_ratio = vol_val / vol_sma_val if (
            not np.isnan(vol_sma_val) and vol_sma_val > 0
        ) else float("nan")

        # Regime
        regime = detect_market_regime(adx_val, di_plus, di_minus, atr_val, atr_sma_val)

        return [
            {
                "name": "rsi_extreme",
                "met": not np.isnan(rsi_val) and (
                    rsi_val < self._config.rsi_long_threshold
                    or rsi_val > self._config.rsi_short_threshold
                ),
                "value": round(rsi_val, 1) if not np.isnan(rsi_val) else None,
                "threshold": f"{self._config.rsi_long_threshold}/{self._config.rsi_short_threshold}",
            },
            {
                "name": "vwap_proximity",
                "met": not np.isnan(vwap_dev) and vwap_dev >= self._config.vwap_deviation_entry,
                "value": round(vwap_dev, 2) if not np.isnan(vwap_dev) else None,
                "threshold": self._config.vwap_deviation_entry,
            },
            {
                "name": "volume_spike",
                "met": not np.isnan(vol_ratio) and vol_ratio >= self._config.volume_spike_multiplier,
                "value": round(vol_ratio, 1) if not np.isnan(vol_ratio) else None,
                "threshold": self._config.volume_spike_multiplier,
            },
            {
                "name": "regime_ok",
                "met": regime in (MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY),
                "value": regime.value,
                "threshold": "RANGING",
            },
        ]

    def check_exit(self, ctx: StrategyContext, position: OpenPosition) -> str | None:
        """Sortie anticipée si RSI revient à 50 et trade en profit."""
        main_tf = self._config.timeframe
        main_ind = ctx.indicators.get(main_tf)
        if not main_ind:
            return None

        rsi_val = main_ind.get("rsi", float("nan"))
        close_val = main_ind.get("close", float("nan"))
        if np.isnan(rsi_val) or np.isnan(close_val):
            return None

        # Vérifier si en profit
        if position.direction == Direction.LONG:
            in_profit = close_val > position.entry_price
            rsi_normalized = rsi_val > 50
        else:
            in_profit = close_val < position.entry_price
            rsi_normalized = rsi_val < 50

        if in_profit and rsi_normalized:
            logger.debug(
                "Signal exit VWAP+RSI : RSI={:.1f}, direction={}, profit=oui",
                rsi_val,
                position.direction.value,
            )
            return "signal_exit"

        return None

    def get_params(self) -> dict:
        """Retourne les paramètres de la stratégie pour BacktestResult."""
        return {
            "timeframe": self._config.timeframe,
            "trend_filter_timeframe": self._config.trend_filter_timeframe,
            "rsi_period": self._config.rsi_period,
            "rsi_long_threshold": self._config.rsi_long_threshold,
            "rsi_short_threshold": self._config.rsi_short_threshold,
            "volume_spike_multiplier": self._config.volume_spike_multiplier,
            "vwap_deviation_entry": self._config.vwap_deviation_entry,
            "trend_adx_threshold": self._config.trend_adx_threshold,
            "tp_percent": self._config.tp_percent,
            "sl_percent": self._config.sl_percent,
        }
