"""Package d'optimisation des paramètres de stratégies.

Registre central des stratégies optimisables et factory pour créer
des instances avec paramètres custom (grid search / WFO).
"""

from __future__ import annotations

from typing import Any

from backend.strategies.base import BaseStrategy
from backend.strategies.bollinger_mr import BollingerMRStrategy
from backend.strategies.donchian_breakout import DonchianBreakoutStrategy
from backend.strategies.envelope_dca import EnvelopeDCAStrategy
from backend.strategies.envelope_dca_short import EnvelopeDCAShortStrategy
from backend.strategies.funding import FundingStrategy
from backend.strategies.liquidation import LiquidationStrategy
from backend.strategies.momentum import MomentumStrategy
from backend.strategies.supertrend import SuperTrendStrategy
from backend.strategies.vwap_rsi import VwapRsiStrategy

from backend.core.config import (
    BollingerMRConfig,
    DonchianBreakoutConfig,
    EnvelopeDCAConfig,
    EnvelopeDCAShortConfig,
    FundingConfig,
    LiquidationConfig,
    MomentumConfig,
    SuperTrendConfig,
    VwapRsiConfig,
)

# Registre central — pas de switch/case, extensible par ajout de ligne
STRATEGY_REGISTRY: dict[str, tuple[type, type]] = {
    "vwap_rsi": (VwapRsiConfig, VwapRsiStrategy),
    "momentum": (MomentumConfig, MomentumStrategy),
    "funding": (FundingConfig, FundingStrategy),
    "liquidation": (LiquidationConfig, LiquidationStrategy),
    "bollinger_mr": (BollingerMRConfig, BollingerMRStrategy),
    "donchian_breakout": (DonchianBreakoutConfig, DonchianBreakoutStrategy),
    "supertrend": (SuperTrendConfig, SuperTrendStrategy),
    "envelope_dca": (EnvelopeDCAConfig, EnvelopeDCAStrategy),
    "envelope_dca_short": (EnvelopeDCAShortConfig, EnvelopeDCAShortStrategy),
}

# Stratégies qui nécessitent extra_data (funding rates, OI) pour le backtest
STRATEGIES_NEED_EXTRA_DATA: set[str] = {"funding", "liquidation"}

# Stratégies grid/DCA (utilisent MultiPositionEngine au lieu de BacktestEngine)
GRID_STRATEGIES: set[str] = {"envelope_dca", "envelope_dca_short"}


def is_grid_strategy(name: str) -> bool:
    """Retourne True si la stratégie utilise le moteur multi-position."""
    return name in GRID_STRATEGIES


def create_strategy_with_params(
    strategy_name: str, params: dict[str, Any]
) -> BaseStrategy:
    """Crée une stratégie avec paramètres custom depuis le registre.

    Utilisé par le walk-forward optimizer et run_backtest_single.
    Les params fournis écrasent les valeurs par défaut de la config.
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Stratégie '{strategy_name}' non optimisable. "
            f"Disponibles : {list(STRATEGY_REGISTRY.keys())}"
        )
    config_cls, strategy_cls = STRATEGY_REGISTRY[strategy_name]

    # Mirroring : extreme_negative_threshold = -extreme_positive_threshold
    if strategy_name == "funding" and "extreme_positive_threshold" in params:
        params.setdefault(
            "extreme_negative_threshold",
            -abs(params["extreme_positive_threshold"]),
        )

    # Merge defaults + custom params
    defaults = config_cls().model_dump()
    merged = {**defaults, **params}
    cfg = config_cls(**merged)
    return strategy_cls(cfg)
