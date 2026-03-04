"""Factory pour l'instanciation des stratégies — Scalp Radar."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from backend.strategies.base import BaseStrategy
from backend.strategies.bollinger_mr import BollingerMRStrategy
from backend.strategies.boltrend import BolTrendStrategy
from backend.strategies.donchian_breakout import DonchianBreakoutStrategy
from backend.strategies.envelope_dca import EnvelopeDCAStrategy
from backend.strategies.envelope_dca_short import EnvelopeDCAShortStrategy
from backend.strategies.funding import FundingStrategy
from backend.strategies.grid_atr import GridATRStrategy
from backend.strategies.grid_boltrend import GridBolTrendStrategy
from backend.strategies.grid_funding import GridFundingStrategy
from backend.strategies.grid_momentum import GridMomentumStrategy
from backend.strategies.grid_multi_tf import GridMultiTFStrategy
from backend.strategies.grid_range_atr import GridRangeATRStrategy
from backend.strategies.grid_trend import GridTrendStrategy
from backend.strategies.liquidation import LiquidationStrategy
from backend.strategies.momentum import MomentumStrategy
from backend.strategies.supertrend import SuperTrendStrategy
from backend.strategies.vwap_rsi import VwapRsiStrategy

if TYPE_CHECKING:
    from backend.core.config import AppConfig, StrategyConfig

logger = logging.getLogger(__name__)

# Mapping nom -> classe
STRATEGY_MAPPING: dict[str, type[BaseStrategy]] = {
    "vwap_rsi": VwapRsiStrategy,
    "momentum": MomentumStrategy,
    "funding": FundingStrategy,
    "liquidation": LiquidationStrategy,
    "bollinger_mr": BollingerMRStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
    "supertrend": SuperTrendStrategy,
    "boltrend": BolTrendStrategy,
    # Multi-position (Grid/DCA)
    "envelope_dca": EnvelopeDCAStrategy,
    "envelope_dca_short": EnvelopeDCAShortStrategy,
    "grid_atr": GridATRStrategy,
    "grid_multi_tf": GridMultiTFStrategy,
    "grid_funding": GridFundingStrategy,
    "grid_trend": GridTrendStrategy,
    "grid_range_atr": GridRangeATRStrategy,
    "grid_boltrend": GridBolTrendStrategy,
    "grid_momentum": GridMomentumStrategy,
}


def get_strategy_class(name: str) -> type[BaseStrategy] | None:
    """Retourne la classe de stratégie par son nom technique."""
    return STRATEGY_MAPPING.get(name)


def create_strategy(name: str, config: StrategyConfig) -> BaseStrategy | None:
    """Instancie une stratégie par son nom et sa config."""
    strat_cls = get_strategy_class(name)
    if not strat_cls:
        return None
    return strat_cls(config)


def get_enabled_strategies(config: AppConfig) -> list[BaseStrategy]:
    """Instancie toutes les stratégies activées dans la config YAML."""
    enabled_instances = []

    # config.strategies est un objet Pydantic (StrategiesConfig)
    # On itère sur les champs définis
    for name in config.strategies.model_fields:
        strat_config = getattr(config.strategies, name)
        if not strat_config or not getattr(strat_config, "enabled", False):
            continue

        instance = create_strategy(name, strat_config)
        if instance:
            enabled_instances.append(instance)
        else:
            logger.warning("Classe de stratégie inconnue: %s", name)

    return enabled_instances
