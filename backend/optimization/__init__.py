"""Package d'optimisation des paramètres de stratégies.

Registre central des stratégies optimisables et factory pour créer
des instances avec paramètres custom (grid search / WFO).
"""

from __future__ import annotations

from typing import Any

from backend.strategies.base import BaseStrategy
from backend.strategies.bollinger_mr import BollingerMRStrategy
from backend.strategies.boltrend import BolTrendStrategy
from backend.strategies.donchian_breakout import DonchianBreakoutStrategy
from backend.strategies.envelope_dca import EnvelopeDCAStrategy
from backend.strategies.envelope_dca_short import EnvelopeDCAShortStrategy
from backend.strategies.funding import FundingStrategy
from backend.strategies.grid_atr import GridATRStrategy
from backend.strategies.grid_funding import GridFundingStrategy
from backend.strategies.grid_multi_tf import GridMultiTFStrategy
from backend.strategies.grid_range_atr import GridRangeATRStrategy
from backend.strategies.grid_trend import GridTrendStrategy
from backend.strategies.liquidation import LiquidationStrategy
from backend.strategies.momentum import MomentumStrategy
from backend.strategies.supertrend import SuperTrendStrategy
from backend.strategies.vwap_rsi import VwapRsiStrategy

from backend.core.config import (
    BollingerMRConfig,
    BolTrendConfig,
    DonchianBreakoutConfig,
    EnvelopeDCAConfig,
    EnvelopeDCAShortConfig,
    FundingConfig,
    GridATRConfig,
    GridFundingConfig,
    GridMultiTFConfig,
    GridRangeATRConfig,
    GridTrendConfig,
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
    "boltrend": (BolTrendConfig, BolTrendStrategy),
    "envelope_dca": (EnvelopeDCAConfig, EnvelopeDCAStrategy),
    "envelope_dca_short": (EnvelopeDCAShortConfig, EnvelopeDCAShortStrategy),
    "grid_atr": (GridATRConfig, GridATRStrategy),
    "grid_range_atr": (GridRangeATRConfig, GridRangeATRStrategy),
    "grid_multi_tf": (GridMultiTFConfig, GridMultiTFStrategy),
    "grid_funding": (GridFundingConfig, GridFundingStrategy),
    "grid_trend": (GridTrendConfig, GridTrendStrategy),
}

# Stratégies qui nécessitent extra_data (funding rates, OI) pour le backtest
STRATEGIES_NEED_EXTRA_DATA: set[str] = {
    "funding", "liquidation",
    "grid_funding", "grid_atr", "grid_range_atr", "envelope_dca", "envelope_dca_short",
    "grid_multi_tf", "grid_trend",
}

# Stratégies grid/DCA (utilisent MultiPositionEngine au lieu de BacktestEngine)
GRID_STRATEGIES: set[str] = {"envelope_dca", "envelope_dca_short", "grid_atr", "grid_range_atr", "grid_multi_tf", "grid_funding", "grid_trend"}

# Stratégies SANS fast engine (pas d'implémentation numpy rapide)
_NO_FAST_ENGINE: set[str] = {"funding", "liquidation"}

# Stratégies avec fast engine (WFO accéléré) — découplé de STRATEGIES_NEED_EXTRA_DATA
FAST_ENGINE_STRATEGIES: set[str] = set(STRATEGY_REGISTRY.keys()) - _NO_FAST_ENGINE


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
