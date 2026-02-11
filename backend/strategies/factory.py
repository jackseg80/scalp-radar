"""Factory pour créer les stratégies depuis la config."""

from __future__ import annotations

from backend.core.config import AppConfig
from backend.strategies.base import BaseStrategy
from backend.strategies.funding import FundingStrategy
from backend.strategies.liquidation import LiquidationStrategy
from backend.strategies.momentum import MomentumStrategy
from backend.strategies.vwap_rsi import VwapRsiStrategy


def create_strategy(name: str, config: AppConfig) -> BaseStrategy:
    """Crée une stratégie par nom depuis la config."""
    strategies_config = config.strategies
    mapping: dict[str, tuple] = {
        "vwap_rsi": (VwapRsiStrategy, strategies_config.vwap_rsi),
        "momentum": (MomentumStrategy, strategies_config.momentum),
        "funding": (FundingStrategy, strategies_config.funding),
        "liquidation": (LiquidationStrategy, strategies_config.liquidation),
    }
    if name not in mapping:
        raise ValueError(f"Stratégie inconnue : {name}")

    cls, strat_config = mapping[name]
    return cls(strat_config)


def get_enabled_strategies(config: AppConfig) -> list[BaseStrategy]:
    """Retourne la liste des stratégies activées dans la config."""
    strategies: list[BaseStrategy] = []
    strats = config.strategies

    if strats.vwap_rsi.enabled:
        strategies.append(VwapRsiStrategy(strats.vwap_rsi))
    if strats.momentum.enabled:
        strategies.append(MomentumStrategy(strats.momentum))
    if strats.funding.enabled:
        strategies.append(FundingStrategy(strats.funding))
    if strats.liquidation.enabled:
        strategies.append(LiquidationStrategy(strats.liquidation))

    return strategies
