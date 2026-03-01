"""Chargeur de configuration YAML avec validation Pydantic.

Charge assets.yaml, strategies.yaml, risk.yaml, exchanges.yaml
et valide avec des modèles Pydantic stricts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings


# ─── MODÈLES DE CONFIGURATION ──────────────────────────────────────────────


class AssetConfig(BaseModel):
    symbol: str
    exchange: str
    type: str  # futures, spot
    timeframes: list[str]
    max_leverage: int = Field(ge=1)
    min_order_size: float = Field(gt=0)
    tick_size: float = Field(gt=0)
    correlation_group: Optional[str] = None


class CorrelationGroupConfig(BaseModel):
    max_concurrent_same_direction: int = Field(ge=1)
    max_exposure_percent: float = Field(gt=0, le=100)


# ─── Strategy configs ───────────────────────────────────────────────────────


class VwapRsiConfig(BaseModel):
    enabled: bool = True
    live_eligible: bool = True
    timeframe: str = "5m"
    trend_filter_timeframe: str = "15m"
    rsi_period: int = Field(default=14, ge=2)
    rsi_long_threshold: float = Field(default=30, ge=0, le=100)
    rsi_short_threshold: float = Field(default=70, ge=0, le=100)
    volume_spike_multiplier: float = Field(default=2.0, gt=0)
    vwap_deviation_entry: float = Field(default=0.3, gt=0)
    trend_adx_threshold: float = Field(default=25.0, ge=0)
    tp_percent: float = Field(default=0.8, gt=0)
    sl_percent: float = Field(default=0.3, gt=0)
    weight: float = Field(default=0.25, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Retourne les paramètres avec overrides per_asset appliqués."""
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}


class LiquidationConfig(BaseModel):
    enabled: bool = False
    live_eligible: bool = False
    timeframe: str = "5m"
    oi_change_threshold: float = Field(default=5.0, gt=0)
    leverage_estimate: int = Field(default=15, ge=1)
    zone_buffer_percent: float = Field(default=1.5, gt=0)
    tp_percent: float = Field(default=0.8, gt=0)
    sl_percent: float = Field(default=0.4, gt=0)
    weight: float = Field(default=0.20, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Retourne les paramètres avec overrides per_asset appliqués."""
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}


class MomentumConfig(BaseModel):
    enabled: bool = False
    live_eligible: bool = True
    timeframe: str = "5m"
    trend_filter_timeframe: str = "15m"
    breakout_lookback: int = Field(default=20, ge=2)
    volume_confirmation_multiplier: float = Field(default=2.0, gt=0)
    atr_multiplier_tp: float = Field(default=2.0, gt=0)
    atr_multiplier_sl: float = Field(default=1.0, gt=0)
    tp_percent: float = Field(default=0.6, gt=0)
    sl_percent: float = Field(default=0.3, gt=0)
    weight: float = Field(default=0.20, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Retourne les paramètres avec overrides per_asset appliqués."""
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}


class FundingConfig(BaseModel):
    enabled: bool = False
    live_eligible: bool = False
    timeframe: str = "15m"
    extreme_positive_threshold: float = Field(default=0.03)
    extreme_negative_threshold: float = Field(default=-0.03)
    entry_delay_minutes: int = Field(default=5, ge=0)
    tp_percent: float = Field(default=0.4, gt=0)
    sl_percent: float = Field(default=0.2, gt=0)
    weight: float = Field(default=0.15, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Retourne les paramètres avec overrides per_asset appliqués."""
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}


class BollingerMRConfig(BaseModel):
    enabled: bool = True
    live_eligible: bool = False
    timeframe: str = "1h"
    bb_period: int = Field(default=20, ge=2)
    bb_std: float = Field(default=2.0, gt=0)
    sl_percent: float = Field(default=5.0, gt=0)
    weight: float = Field(default=0.15, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Retourne les paramètres avec overrides per_asset appliqués."""
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}


class DonchianBreakoutConfig(BaseModel):
    enabled: bool = True
    live_eligible: bool = False
    timeframe: str = "1h"
    entry_lookback: int = Field(default=20, ge=2)
    atr_period: int = Field(default=14, ge=2)
    atr_tp_multiple: float = Field(default=3.0, gt=0)
    atr_sl_multiple: float = Field(default=1.5, gt=0)
    weight: float = Field(default=0.15, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Retourne les paramètres avec overrides per_asset appliqués."""
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}


class SuperTrendConfig(BaseModel):
    enabled: bool = True
    live_eligible: bool = False
    timeframe: str = "1h"
    atr_period: int = Field(default=10, ge=2)
    atr_multiplier: float = Field(default=3.0, gt=0)
    tp_percent: float = Field(default=4.0, gt=0)
    sl_percent: float = Field(default=2.0, gt=0)
    weight: float = Field(default=0.15, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Retourne les paramètres avec overrides per_asset appliqués."""
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}


class BolTrendConfig(BaseModel):
    enabled: bool = False
    live_eligible: bool = False
    timeframe: str = "1h"
    bol_window: int = Field(default=100, ge=2)
    bol_std: float = Field(default=2.2, gt=0)
    min_bol_spread: float = Field(default=0.0, ge=0)
    long_ma_window: int = Field(default=550, ge=2)
    sl_percent: float = Field(default=15.0, gt=0)
    leverage: int = Field(default=2, ge=1)
    weight: float = Field(default=0.15, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Retourne les paramètres avec overrides per_asset appliqués."""
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}


class EnvelopeDCAConfig(BaseModel):
    enabled: bool = True
    live_eligible: bool = False
    timeframe: str = "1h"
    ma_period: int = Field(default=5, ge=2, le=50)
    num_levels: int = Field(default=4, ge=1, le=6)
    envelope_start: float = Field(default=0.05, gt=0)
    envelope_step: float = Field(default=0.05, gt=0)
    sl_percent: float = Field(default=25.0, gt=0)
    sides: list[str] = Field(default=["long"])
    leverage: int = Field(default=6, ge=1, le=20)
    weight: float = Field(default=0.20, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Retourne les paramètres avec overrides per_asset appliqués."""
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}


class EnvelopeDCAShortConfig(BaseModel):
    enabled: bool = False
    live_eligible: bool = False
    timeframe: str = "1h"
    ma_period: int = Field(default=7, ge=2, le=50)
    num_levels: int = Field(default=2, ge=1, le=6)
    envelope_start: float = Field(default=0.05, gt=0)
    envelope_step: float = Field(default=0.02, gt=0)
    sl_percent: float = Field(default=20.0, gt=0)
    sides: list[str] = Field(default=["short"])
    leverage: int = Field(default=6, ge=1, le=20)
    weight: float = Field(default=0.20, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Retourne les paramètres avec overrides per_asset appliqués."""
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}


class GridATRConfig(BaseModel):
    enabled: bool = False
    live_eligible: bool = False
    timeframe: str = "1h"
    ma_period: int = Field(default=14, ge=2, le=50)
    atr_period: int = Field(default=14, ge=2, le=50)
    atr_multiplier_start: float = Field(default=2.0, gt=0)
    atr_multiplier_step: float = Field(default=1.0, gt=0)
    num_levels: int = Field(default=3, ge=1, le=6)
    sl_percent: float = Field(default=20.0, gt=0)
    sides: list[str] = Field(default=["long"])
    leverage: int = Field(default=6, ge=1, le=20)
    weight: float = Field(default=0.20, ge=0, le=1)
    max_hold_candles: int = Field(default=0, ge=0)
    cooldown_candles: int = Field(default=3, ge=0)
    min_grid_spacing_pct: float = Field(default=0.0, ge=0, le=10.0)
    min_profit_pct: float = Field(default=0.0, ge=0, le=10.0)
    min_atr_pct: float = Field(default=0.0, ge=0, le=10.0)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Retourne les paramètres avec overrides per_asset appliqués."""
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}


class GridRangeATRConfig(BaseModel):
    """Grid Range ATR : range trading bidirectionnel avec TP/SL individuels."""

    enabled: bool = False
    live_eligible: bool = False
    timeframe: str = "1h"
    ma_period: int = Field(default=20, ge=2, le=50)
    atr_period: int = Field(default=14, ge=2, le=50)
    atr_spacing_mult: float = Field(default=0.3, gt=0)
    num_levels: int = Field(default=2, ge=1, le=6)
    sl_percent: float = Field(default=10.0, gt=0)
    tp_mode: str = Field(default="dynamic_sma")
    sides: list[str] = Field(default=["long", "short"])
    leverage: int = Field(default=6, ge=1, le=20)
    weight: float = Field(default=0.20, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Retourne les paramètres avec overrides per_asset appliqués."""
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}


class GridMultiTFConfig(BaseModel):
    """Grid Multi-TF : filtre Supertrend 4h + exécution Grid ATR 1h."""

    enabled: bool = False
    live_eligible: bool = False
    timeframe: str = "1h"
    # Filtre trend (4h Supertrend)
    st_atr_period: int = Field(default=10, ge=2, le=50)
    st_atr_multiplier: float = Field(default=3.0, gt=0)
    # Exécution grid (1h) — mêmes params que grid_atr
    ma_period: int = Field(default=14, ge=2, le=50)
    atr_period: int = Field(default=14, ge=2, le=50)
    atr_multiplier_start: float = Field(default=2.0, gt=0)
    atr_multiplier_step: float = Field(default=1.0, gt=0)
    num_levels: int = Field(default=3, ge=1, le=6)
    sl_percent: float = Field(default=20.0, gt=0)
    sides: list[str] = Field(default=["long", "short"])
    leverage: int = Field(default=6, ge=1, le=20)
    weight: float = Field(default=0.20, ge=0, le=1)
    max_hold_candles: int = Field(default=0, ge=0)
    cooldown_candles: int = Field(default=3, ge=0)
    min_grid_spacing_pct: float = Field(default=0.0, ge=0, le=10.0)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Retourne les paramètres avec overrides per_asset appliqués."""
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}


class GridFundingConfig(BaseModel):
    """Grid Funding : DCA sur funding rate négatif (LONG-only)."""

    enabled: bool = False
    live_eligible: bool = False
    timeframe: str = "1h"
    funding_threshold_start: float = Field(default=0.0005, gt=0)  # raw decimal, -0.05%
    funding_threshold_step: float = Field(default=0.0005, gt=0)
    num_levels: int = Field(default=3, ge=1, le=6)
    tp_mode: str = Field(default="funding_or_sma")  # funding_positive, sma_cross, funding_or_sma
    ma_period: int = Field(default=14, ge=2, le=50)
    sl_percent: float = Field(default=15.0, gt=0)
    min_hold_candles: int = Field(default=8, ge=0)  # minimum 8h (1 période funding)
    sides: list[str] = Field(default=["long"])
    leverage: int = Field(default=6, ge=1, le=20)
    weight: float = Field(default=0.20, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Retourne les paramètres avec overrides per_asset appliqués."""
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}


class GridTrendConfig(BaseModel):
    """Grid Trend : DCA trend following (EMA cross + ADX + trailing stop ATR)."""

    enabled: bool = False
    live_eligible: bool = False
    timeframe: str = "1h"
    ema_fast: int = Field(default=20, ge=5, le=50)
    ema_slow: int = Field(default=50, ge=20, le=200)
    adx_period: int = Field(default=14, ge=7, le=30)
    adx_threshold: float = Field(default=20.0, ge=10, le=40)
    atr_period: int = Field(default=14, ge=5, le=30)
    pull_start: float = Field(default=1.0, gt=0)
    pull_step: float = Field(default=0.5, gt=0)
    num_levels: int = Field(default=3, ge=1, le=6)
    trail_mult: float = Field(default=2.0, gt=0)
    sl_percent: float = Field(default=15.0, gt=0)
    sides: list[str] = Field(default=["long", "short"])
    leverage: int = Field(default=6, ge=1, le=20)
    weight: float = Field(default=0.20, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Retourne les paramètres avec overrides per_asset appliqués."""
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}


class GridBolTrendConfig(BaseModel):
    """Grid BolTrend : DCA event-driven sur breakout Bollinger + filtre SMA."""

    enabled: bool = False
    live_eligible: bool = False
    timeframe: str = "1h"
    bol_window: int = Field(default=100, ge=2)
    bol_std: float = Field(default=2.0, gt=0)
    long_ma_window: int = Field(default=200, ge=2)
    min_bol_spread: float = Field(default=0.0, ge=0)
    atr_period: int = Field(default=14, ge=2, le=50)
    atr_spacing_mult: float = Field(default=1.0, gt=0)
    num_levels: int = Field(default=3, ge=1, le=6)
    sl_percent: float = Field(default=15.0, gt=0)
    sides: list[str] = Field(default=["long", "short"])
    leverage: int = Field(default=6, ge=1, le=20)
    weight: float = Field(default=0.20, ge=0, le=1)
    max_hold_candles: int = Field(default=0, ge=0)
    cooldown_candles: int = Field(default=3, ge=0)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Retourne les paramètres avec overrides per_asset appliqués."""
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}


class GridMomentumConfig(BaseModel):
    """Grid Momentum : DCA pullback sur breakout Donchian + trailing stop ATR."""

    enabled: bool = False
    live_eligible: bool = False
    timeframe: str = "1h"
    donchian_period: int = Field(default=30, ge=10, le=100)
    vol_sma_period: int = Field(default=20, ge=5, le=50)
    vol_multiplier: float = Field(default=1.5, gt=0)
    adx_period: int = Field(default=14, ge=5, le=30)
    adx_threshold: float = Field(default=0.0, ge=0)
    atr_period: int = Field(default=14, ge=5, le=30)
    pullback_start: float = Field(default=1.0, gt=0)
    pullback_step: float = Field(default=0.5, gt=0)
    num_levels: int = Field(default=3, ge=1, le=6)
    trailing_atr_mult: float = Field(default=3.0, gt=0)
    sl_percent: float = Field(default=15.0, gt=0)
    sides: list[str] = Field(default=["long", "short"])
    leverage: int = Field(default=6, ge=1, le=20)
    weight: float = Field(default=0.15, ge=0, le=1)
    cooldown_candles: int = Field(default=3, ge=0, le=10)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Retourne les paramètres avec overrides per_asset appliqués."""
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}


class CustomStrategyConfig(BaseModel):
    enabled: bool = False
    timeframe: str = "1h"
    description: str = ""


class StrategiesConfig(BaseModel):
    vwap_rsi: VwapRsiConfig = Field(default_factory=VwapRsiConfig)
    liquidation: LiquidationConfig = Field(default_factory=LiquidationConfig)
    momentum: MomentumConfig = Field(default_factory=MomentumConfig)
    funding: FundingConfig = Field(default_factory=FundingConfig)
    bollinger_mr: BollingerMRConfig = Field(default_factory=BollingerMRConfig)
    donchian_breakout: DonchianBreakoutConfig = Field(default_factory=DonchianBreakoutConfig)
    supertrend: SuperTrendConfig = Field(default_factory=SuperTrendConfig)
    boltrend: BolTrendConfig = Field(default_factory=BolTrendConfig)
    envelope_dca: EnvelopeDCAConfig = Field(default_factory=EnvelopeDCAConfig)
    envelope_dca_short: EnvelopeDCAShortConfig = Field(default_factory=EnvelopeDCAShortConfig)
    grid_atr: GridATRConfig = Field(default_factory=GridATRConfig)
    grid_range_atr: GridRangeATRConfig = Field(default_factory=GridRangeATRConfig)
    grid_multi_tf: GridMultiTFConfig = Field(default_factory=GridMultiTFConfig)
    grid_funding: GridFundingConfig = Field(default_factory=GridFundingConfig)
    grid_trend: GridTrendConfig = Field(default_factory=GridTrendConfig)
    grid_boltrend: GridBolTrendConfig = Field(default_factory=GridBolTrendConfig)
    grid_momentum: GridMomentumConfig = Field(default_factory=GridMomentumConfig)
    custom_strategies: dict[str, CustomStrategyConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_weights(self) -> StrategiesConfig:
        enabled = [
            s for s in [
                self.vwap_rsi, self.liquidation,
                self.momentum, self.funding,
                self.bollinger_mr, self.donchian_breakout, self.supertrend,
                self.boltrend,
                self.envelope_dca, self.envelope_dca_short, self.grid_atr,
                self.grid_range_atr, self.grid_multi_tf, self.grid_funding,
                self.grid_trend,
                self.grid_boltrend,
                self.grid_momentum,
            ]
            if s.enabled
        ]
        total = sum(s.weight for s in enabled)
        if enabled and abs(total - 1.0) > 0.05:
            logger.warning(
                "Poids des stratégies actives = {:.2f} (attendu ~1.0)", total
            )
        return self


# ─── Risk configs ───────────────────────────────────────────────────────────


class KillSwitchConfig(BaseModel):
    max_session_loss_percent: float = Field(default=5.0, gt=0)
    max_daily_loss_percent: float = Field(default=10.0, gt=0)
    grid_max_session_loss_percent: Optional[float] = None
    grid_max_daily_loss_percent: Optional[float] = None
    global_max_loss_pct: float = Field(default=45.0, gt=0)
    global_window_hours: int = Field(default=24, ge=1)


class PositionConfig(BaseModel):
    max_risk_per_trade_percent: float = Field(default=2.0, gt=0)
    max_concurrent_positions: int = Field(default=3, ge=1)
    default_leverage: int = Field(default=15, ge=1)
    max_leverage: int = Field(default=30, ge=1)


class FeesConfig(BaseModel):
    maker_percent: float = Field(default=0.02, ge=0)
    taker_percent: float = Field(default=0.06, ge=0)
    use_bgb_discount: bool = False


class SlippageConfig(BaseModel):
    default_estimate_percent: float = Field(default=0.05, ge=0)
    high_volatility_multiplier: float = Field(default=2.0, ge=1)


class MarginConfig(BaseModel):
    mode: str = "cross"
    min_free_margin_percent: float = Field(default=20, ge=0, le=100)


class SlTpConfig(BaseModel):
    mode: str = "server_side"
    sl_type: str = "market"
    sl_real_cost_includes: list[str] = Field(
        default_factory=lambda: ["distance", "taker_fee", "slippage"]
    )


class AdaptiveSelectorConfig(BaseModel):
    min_trades: int = Field(default=3, ge=1)
    min_profit_factor: float = Field(default=1.0, ge=0)
    eval_interval_seconds: int = Field(default=300, ge=30)
    force_strategies: list[str] = Field(default_factory=list)


class RiskConfig(BaseModel):
    initial_capital: float = Field(default=10_000.0, gt=0)
    max_margin_ratio: float = Field(default=0.70, ge=0.1, le=1.0)
    kill_switch: KillSwitchConfig = Field(default_factory=KillSwitchConfig)
    position: PositionConfig = Field(default_factory=PositionConfig)
    fees: FeesConfig = Field(default_factory=FeesConfig)
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    margin: MarginConfig = Field(default_factory=MarginConfig)
    sl_tp: SlTpConfig = Field(default_factory=SlTpConfig)
    adaptive_selector: AdaptiveSelectorConfig = Field(
        default_factory=AdaptiveSelectorConfig,
    )
    regime_filter_enabled: bool = Field(default=True)  # Sprint 27
    selector_bypass_at_boot: bool = Field(default=False)  # Hotfix 28a
    max_live_grids: int = Field(default=4, ge=1)  # Hotfix 35 : max cycles grid simultanés

    @model_validator(mode="after")
    def validate_leverage(self) -> RiskConfig:
        if self.position.default_leverage > self.position.max_leverage:
            raise ValueError(
                f"default_leverage ({self.position.default_leverage}) "
                f"> max_leverage ({self.position.max_leverage})"
            )
        return self


# ─── Exchange configs ───────────────────────────────────────────────────────


class WebSocketConfig(BaseModel):
    url: str
    ping_interval: int = 25
    reconnect_delay: int = 5
    max_reconnect_attempts: int = 10


class RateLimitCategoryConfig(BaseModel):
    requests_per_second: int = Field(ge=1)


class RateLimitsConfig(BaseModel):
    market_data: RateLimitCategoryConfig = Field(
        default_factory=lambda: RateLimitCategoryConfig(requests_per_second=20)
    )
    trade: RateLimitCategoryConfig = Field(
        default_factory=lambda: RateLimitCategoryConfig(requests_per_second=10)
    )
    account: RateLimitCategoryConfig = Field(
        default_factory=lambda: RateLimitCategoryConfig(requests_per_second=10)
    )
    position: RateLimitCategoryConfig = Field(
        default_factory=lambda: RateLimitCategoryConfig(requests_per_second=10)
    )


class ApiConfig(BaseModel):
    base_url: str
    futures_type: str = "USDT-M"
    price_type: str = "mark_price"


class ExchangeConfig(BaseModel):
    name: str
    websocket: WebSocketConfig
    rate_limits: RateLimitsConfig = Field(default_factory=RateLimitsConfig)
    api: ApiConfig


# ─── Secrets (.env) ─────────────────────────────────────────────────────────


class SecretsConfig(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    bitget_api_key: str = ""
    bitget_secret: str = ""
    bitget_passphrase: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    database_url: str = "sqlite:///data/scalp_radar.db"
    log_level: str = "DEBUG"
    enable_websocket: bool = True
    heartbeat_interval: int = 3600  # secondes (1h par défaut, overridable via .env)
    live_trading: bool = False  # LIVE_TRADING env var, défaut false = simulation only
    # Sandbox Bitget supprimé (cassé, ccxt #25523) — mainnet only
    selector_bypass_at_boot: bool | None = None  # Override risk.yaml si défini dans .env
    force_strategies: str | None = None  # FORCE_STRATEGIES, comma-separated (ex: "grid_atr,grid_trend")
    active_strategies: str | None = None  # ACTIVE_STRATEGIES, comma-separated (ex: "grid_atr,grid_boltrend")

    # Sync WFO local → serveur
    sync_server_url: str = ""  # SYNC_SERVER_URL, ex: "http://192.168.1.200:8000"
    sync_api_key: str = ""  # SYNC_API_KEY, secret partagé local ↔ serveur
    sync_enabled: bool = False  # SYNC_ENABLED, désactivé par défaut

    # Backfill automatique quotidien
    backfill_enabled: bool = True  # BACKFILL_ENABLED, cron 03:00 UTC


# ─── APP CONFIG (agrégation) ────────────────────────────────────────────────


def _load_yaml(filepath: Path) -> dict[str, Any]:
    with open(filepath, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class AppConfig:
    """Configuration globale de l'application.

    Charge et valide les 4 fichiers YAML + les secrets .env.
    """

    def __init__(
        self,
        config_dir: Path | str = "config",
        env_file: Path | str | None = ".env",
    ) -> None:
        config_dir = Path(config_dir)

        # Charger les YAML
        assets_raw = _load_yaml(config_dir / "assets.yaml")
        strategies_raw = _load_yaml(config_dir / "strategies.yaml")
        risk_raw = _load_yaml(config_dir / "risk.yaml")
        exchanges_raw = _load_yaml(config_dir / "exchanges.yaml")

        # Assets
        self.assets: list[AssetConfig] = [
            AssetConfig(**a) for a in assets_raw.get("assets", [])
        ]
        self.correlation_groups: dict[str, CorrelationGroupConfig] = {
            name: CorrelationGroupConfig(**cfg)
            for name, cfg in assets_raw.get("correlation_groups", {}).items()
        }

        # Stratégies
        custom_raw = strategies_raw.pop("custom_strategies", {})
        self.strategies = StrategiesConfig(**strategies_raw)
        self.strategies.custom_strategies = {
            name: CustomStrategyConfig(**cfg)
            for name, cfg in custom_raw.items()
        }

        # Risque
        self.risk = RiskConfig(**risk_raw)

        # Exchange (Bitget par défaut)
        self.exchange = ExchangeConfig(**exchanges_raw.get("bitget", {}))

        # Secrets
        kwargs = {}
        if env_file and Path(env_file).exists():
            kwargs["_env_file"] = str(env_file)
        self.secrets = SecretsConfig(**kwargs)

        # Override YAML par env var (.env)
        if self.secrets.selector_bypass_at_boot is not None:
            self.risk.selector_bypass_at_boot = self.secrets.selector_bypass_at_boot

        if self.secrets.force_strategies is not None:
            strategies = [
                s.strip()
                for s in self.secrets.force_strategies.split(",")
                if s.strip()
            ]
            self.risk.adaptive_selector.force_strategies = strategies
            logger.info(
                "Config: force_strategies overridé par .env: {}",
                strategies,
            )

        # ACTIVE_STRATEGIES : filtre les stratégies activées au runtime
        self._active_strategies: list[str] = []
        if self.secrets.active_strategies is not None:
            self._active_strategies = [
                s.strip()
                for s in self.secrets.active_strategies.split(",")
                if s.strip()
            ]
            if self._active_strategies:
                logger.info(
                    "Config: ACTIVE_STRATEGIES filtré par .env: {}",
                    self._active_strategies,
                )

        # Validations croisées
        self._validate_cross_config()

        logger.info(
            "Config chargée : {} assets, exchange={}, websocket={}",
            len(self.assets),
            self.exchange.name,
            self.secrets.enable_websocket,
        )

    @property
    def active_strategies(self) -> list[str]:
        """Liste des stratégies autorisées. Vide = toutes les enabled."""
        return self._active_strategies

    def _validate_cross_config(self) -> None:
        """Validations qui croisent plusieurs fichiers de config."""
        # Vérifier que les correlation_groups référencés existent
        for asset in self.assets:
            if asset.correlation_group and asset.correlation_group not in self.correlation_groups:
                raise ValueError(
                    f"Asset {asset.symbol} référence le groupe de corrélation "
                    f"'{asset.correlation_group}' qui n'existe pas"
                )

        # Vérifier cohérence leverage par asset
        for asset in self.assets:
            if asset.max_leverage > self.risk.position.max_leverage:
                logger.warning(
                    "Asset {} max_leverage ({}) > risk max_leverage ({})",
                    asset.symbol,
                    asset.max_leverage,
                    self.risk.position.max_leverage,
                )

        # Vérifier cohérence positions vs correlation_groups
        for group_name, group_cfg in self.correlation_groups.items():
            assets_in_group = [
                a for a in self.assets if a.correlation_group == group_name
            ]
            if (
                len(assets_in_group) > 0
                and group_cfg.max_concurrent_same_direction
                > self.risk.position.max_concurrent_positions
            ):
                logger.warning(
                    "Groupe '{}' : max_concurrent_same_direction ({}) > "
                    "max_concurrent_positions global ({})",
                    group_name,
                    group_cfg.max_concurrent_same_direction,
                    self.risk.position.max_concurrent_positions,
                )

    # ─── Multi-Executor (Sprint 36b) ───────────────────────────────────

    def get_executor_keys(self, strategy_name: str) -> tuple[str, str, str]:
        """Retourne (api_key, secret, passphrase) pour un executor.

        Cherche BITGET_API_KEY_{STRATEGY_UPPER}, etc. dans les env vars.
        Fallback sur les clés globales si absentes ou incomplètes.
        """
        import os

        suffix = strategy_name.upper()
        api_key = os.environ.get(f"BITGET_API_KEY_{suffix}", "")
        secret = os.environ.get(f"BITGET_SECRET_{suffix}", "")
        passphrase = os.environ.get(f"BITGET_PASSPHRASE_{suffix}", "")

        if api_key and secret and passphrase:
            return api_key, secret, passphrase

        return (
            self.secrets.bitget_api_key,
            self.secrets.bitget_secret,
            self.secrets.bitget_passphrase,
        )

    def has_dedicated_keys(self, strategy_name: str) -> bool:
        """True si la stratégie a ses propres clés API (sous-compte dédié)."""
        import os

        suffix = strategy_name.upper()
        return bool(
            os.environ.get(f"BITGET_API_KEY_{suffix}", "")
            and os.environ.get(f"BITGET_SECRET_{suffix}", "")
            and os.environ.get(f"BITGET_PASSPHRASE_{suffix}", "")
        )

    def reload(self, config_dir: Path | str = "config") -> None:
        """Recharge les fichiers YAML (hot-reload en dev)."""
        new = AppConfig(config_dir=config_dir)
        self.assets = new.assets
        self.correlation_groups = new.correlation_groups
        self.strategies = new.strategies
        self.risk = new.risk
        self.exchange = new.exchange
        logger.info("Config rechargée")


# ─── SINGLETON ──────────────────────────────────────────────────────────────

_config: Optional[AppConfig] = None


def get_config(
    config_dir: Path | str = "config",
    env_file: Path | str | None = ".env",
    force_reload: bool = False,
) -> AppConfig:
    """Retourne la config globale (singleton)."""
    global _config
    if _config is None or force_reload:
        _config = AppConfig(config_dir=config_dir, env_file=env_file)
    return _config
