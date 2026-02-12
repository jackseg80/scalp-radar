"""Package d'optimisation des paramètres de stratégies.

Registre central des stratégies optimisables et factory pour créer
des instances avec paramètres custom (grid search / WFO).
"""

from __future__ import annotations

from typing import Any

from backend.strategies.base import BaseStrategy
from backend.strategies.momentum import MomentumStrategy
from backend.strategies.vwap_rsi import VwapRsiStrategy

# Lazy imports des configs pour éviter les dépendances circulaires
from backend.core.config import MomentumConfig, VwapRsiConfig

# Registre central — pas de switch/case, extensible par ajout de ligne
# Funding et Liquidation exclus : pas de données historiques OI/funding
STRATEGY_REGISTRY: dict[str, tuple[type, type]] = {
    "vwap_rsi": (VwapRsiConfig, VwapRsiStrategy),
    "momentum": (MomentumConfig, MomentumStrategy),
}


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
    # Merge defaults + custom params
    defaults = config_cls().model_dump()
    merged = {**defaults, **params}
    cfg = config_cls(**merged)
    return strategy_cls(cfg)
