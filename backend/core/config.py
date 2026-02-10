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


class LiquidationConfig(BaseModel):
    enabled: bool = False
    timeframe: str = "5m"
    oi_change_threshold: float = Field(default=5.0, gt=0)
    leverage_estimate: int = Field(default=15, ge=1)
    zone_buffer_percent: float = Field(default=0.5, gt=0)
    tp_percent: float = Field(default=0.8, gt=0)
    sl_percent: float = Field(default=0.4, gt=0)
    weight: float = Field(default=0.20, ge=0, le=1)


class OrderflowConfig(BaseModel):
    enabled: bool = False
    timeframe: str = "1m"
    imbalance_threshold: float = Field(default=2.0, gt=0)
    large_order_multiplier: float = Field(default=5.0, gt=0)
    absorption_threshold: float = Field(default=0.7, ge=0, le=1)
    confirmation_only: bool = True
    weight: float = Field(default=0.20, ge=0, le=1)


class MomentumConfig(BaseModel):
    enabled: bool = False
    timeframe: str = "5m"
    trend_filter_timeframe: str = "15m"
    breakout_lookback: int = Field(default=20, ge=2)
    volume_confirmation_multiplier: float = Field(default=2.0, gt=0)
    atr_multiplier_tp: float = Field(default=2.0, gt=0)
    atr_multiplier_sl: float = Field(default=1.0, gt=0)
    tp_percent: float = Field(default=0.6, gt=0)
    sl_percent: float = Field(default=0.3, gt=0)
    weight: float = Field(default=0.20, ge=0, le=1)


class FundingConfig(BaseModel):
    enabled: bool = False
    timeframe: str = "15m"
    extreme_positive_threshold: float = Field(default=0.03)
    extreme_negative_threshold: float = Field(default=-0.03)
    entry_delay_minutes: int = Field(default=5, ge=0)
    tp_percent: float = Field(default=0.4, gt=0)
    sl_percent: float = Field(default=0.2, gt=0)
    weight: float = Field(default=0.15, ge=0, le=1)


class CustomStrategyConfig(BaseModel):
    enabled: bool = False
    timeframe: str = "1h"
    description: str = ""


class StrategiesConfig(BaseModel):
    vwap_rsi: VwapRsiConfig = Field(default_factory=VwapRsiConfig)
    liquidation: LiquidationConfig = Field(default_factory=LiquidationConfig)
    orderflow: OrderflowConfig = Field(default_factory=OrderflowConfig)
    momentum: MomentumConfig = Field(default_factory=MomentumConfig)
    funding: FundingConfig = Field(default_factory=FundingConfig)
    custom_strategies: dict[str, CustomStrategyConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_weights(self) -> StrategiesConfig:
        enabled = [
            s for s in [
                self.vwap_rsi, self.liquidation, self.orderflow,
                self.momentum, self.funding,
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


class RiskConfig(BaseModel):
    kill_switch: KillSwitchConfig = Field(default_factory=KillSwitchConfig)
    position: PositionConfig = Field(default_factory=PositionConfig)
    fees: FeesConfig = Field(default_factory=FeesConfig)
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    margin: MarginConfig = Field(default_factory=MarginConfig)
    sl_tp: SlTpConfig = Field(default_factory=SlTpConfig)

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
    sandbox: bool = False
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

        # Validations croisées
        self._validate_cross_config()

        logger.info(
            "Config chargée : {} assets, exchange={}, websocket={}",
            len(self.assets),
            self.exchange.name,
            self.secrets.enable_websocket,
        )

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
