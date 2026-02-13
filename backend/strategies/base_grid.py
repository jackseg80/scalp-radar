"""Classes de base pour les stratégies grid/DCA multi-position.

BaseGridStrategy hérite de BaseStrategy pour la compatibilité avec
l'écosystème existant (Arena, Simulator, Dashboard, factory).
Les méthodes mono-position (evaluate, check_exit) retournent None.
Le MultiPositionEngine utilise les méthodes grid à la place.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from backend.core.models import Candle, Direction
from backend.strategies.base import BaseStrategy, OpenPosition, StrategyContext, StrategySignal


@dataclass
class GridLevel:
    """Un niveau d'entrée dans la grille."""

    index: int  # 0, 1, 2, 3 (0 = plus proche de la MA)
    entry_price: float
    direction: Direction
    size_fraction: float  # Fraction du capital alloué (ex: 0.25 pour 4 niveaux)


@dataclass
class GridPosition:
    """Une position individuelle dans la grille."""

    level: int
    direction: Direction
    entry_price: float
    quantity: float
    entry_time: datetime
    entry_fee: float


@dataclass
class GridState:
    """État complet de toutes les positions de la grille."""

    positions: list[GridPosition]
    avg_entry_price: float
    total_quantity: float
    total_notional: float
    unrealized_pnl: float


class BaseGridStrategy(BaseStrategy):
    """Stratégie multi-position avec grille de niveaux.

    Hérite de BaseStrategy pour la compatibilité Arena/Simulator/Dashboard.
    Les méthodes mono-position (evaluate, check_exit) retournent None.
    Le MultiPositionEngine utilise les méthodes grid-spécifiques :
    - compute_grid() : niveaux d'entrée à chaque bougie
    - should_close_all() : condition de sortie globale
    - get_sl_price() / get_tp_price() : SL/TP global dynamique
    """

    name: str = "base_grid"

    # --- Implémentations par défaut BaseStrategy (mono-position) ---

    def evaluate(self, ctx: StrategyContext) -> StrategySignal | None:
        """Non utilisé par MultiPositionEngine."""
        return None

    def check_exit(self, ctx: StrategyContext, position: OpenPosition) -> str | None:
        """Non utilisé par MultiPositionEngine."""
        return None

    def get_current_conditions(self, ctx: StrategyContext) -> list[dict]:
        """Retourne les niveaux de la grille pour le dashboard."""
        grid_state = GridState(
            positions=[], avg_entry_price=0, total_quantity=0,
            total_notional=0, unrealized_pnl=0,
        )
        levels = self.compute_grid(ctx, grid_state)
        conditions = []
        for lvl in levels:
            conditions.append({
                "name": f"Level {lvl.index + 1} ({lvl.direction.value})",
                "met": False,
                "value": f"{lvl.entry_price:.2f}",
                "threshold": "touch",
            })
        return conditions

    # --- Interface grid (abstraite) ---

    @abstractmethod
    def compute_grid(
        self, ctx: StrategyContext, grid_state: GridState
    ) -> list[GridLevel]:
        """Calcule les niveaux de la grille pour la bougie courante.

        Un seul côté actif à la fois : si des positions LONG sont ouvertes,
        ne PAS retourner de niveaux SHORT (et inversement).
        """

    @abstractmethod
    def should_close_all(
        self, ctx: StrategyContext, grid_state: GridState
    ) -> str | None:
        """Vérifie si toutes les positions doivent être fermées.

        Returns:
            "tp_global", "sl_global", ou None.
        """

    @abstractmethod
    def get_sl_price(
        self, grid_state: GridState, current_indicators: dict
    ) -> float:
        """SL global basé sur le prix moyen d'entrée."""

    @abstractmethod
    def get_tp_price(
        self, grid_state: GridState, current_indicators: dict
    ) -> float:
        """TP global dynamique (ex: retour à la SMA)."""

    @property
    @abstractmethod
    def max_positions(self) -> int:
        """Nombre maximum de positions simultanées (= nombre de niveaux)."""

    def get_params(self) -> dict[str, Any]:
        """Retourne les paramètres pour le reporting."""
        return {}
