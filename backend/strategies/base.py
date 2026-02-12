"""Classes de base pour les stratégies de trading Scalp Radar.

Définit StrategyContext, StrategySignal et BaseStrategy (ABC).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from backend.core.config import AppConfig
from backend.core.models import Candle, Direction, MarketRegime, SignalStrength

# Constantes pour les clés extra_data (pas de magic strings)
EXTRA_FUNDING_RATE = "funding_rate"
EXTRA_OPEN_INTEREST = "open_interest"
EXTRA_OI_CHANGE_PCT = "oi_change_pct"
EXTRA_ORDERBOOK = "orderbook"


@dataclass
class OpenPosition:
    """Position ouverte dans le backtester."""

    direction: Direction
    entry_price: float
    quantity: float
    entry_time: datetime
    tp_price: float
    sl_price: float
    entry_fee: float


@dataclass
class StrategyContext:
    """Contexte passé à la stratégie à chaque bougie.

    Le moteur de backtest construit ce contexte en injectant les indicateurs
    pré-calculés. Pour le 15m, c'est toujours la dernière bougie clôturée
    (l'alignement multi-TF est géré par le moteur).
    """

    symbol: str
    timestamp: datetime
    candles: dict[str, list[Candle]]
    indicators: dict[str, dict[str, Any]]
    current_position: OpenPosition | None
    capital: float
    config: AppConfig
    extra_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategySignal:
    """Signal de trading émis par une stratégie."""

    direction: Direction
    entry_price: float
    tp_price: float
    sl_price: float
    score: float  # 0-1
    strength: SignalStrength
    market_regime: MarketRegime
    signals_detail: dict[str, float] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Classe abstraite pour les stratégies de trading.

    Chaque stratégie doit implémenter :
    - evaluate() : conditions d'entrée
    - check_exit() : sortie anticipée (appelée seulement si ni TP ni SL touchés)
    - compute_indicators() : pré-calcul des indicateurs sur tout le dataset
    - min_candles : nombre minimum de bougies requises par timeframe

    Deux chemins de résolution des paramètres :
    - **Optimisation** : params injectés explicitement via create_strategy_with_params() → _config
      contient déjà les valeurs optimisées pour cet asset, _resolve_param() n'est pas appelé.
    - **Production** (Simulator/Executor) : _config contient les defaults YAML, _resolve_param()
      résout les overrides per_asset au runtime via le symbole du contexte.
    """

    name: str = "base"

    def _resolve_param(self, param_name: str, symbol: str) -> Any:
        """Résout un paramètre avec override per_asset (chemin production).

        Cherche dans _config.per_asset[symbol][param_name], sinon retourne
        la valeur par défaut de _config.<param_name>.
        En optimisation, _config n'a pas de per_asset (ou il est vide),
        donc cette méthode retourne simplement la valeur injectée.
        """
        config = getattr(self, "_config", None)
        if config is None:
            raise AttributeError(f"{self.__class__.__name__} n'a pas de _config")
        per_asset = getattr(config, "per_asset", {})
        overrides = per_asset.get(symbol, {})
        if param_name in overrides:
            return overrides[param_name]
        return getattr(config, param_name)

    @abstractmethod
    def evaluate(self, ctx: StrategyContext) -> StrategySignal | None:
        """Évalue les conditions d'entrée. Retourne un signal ou None."""

    @abstractmethod
    def check_exit(self, ctx: StrategyContext, position: OpenPosition) -> str | None:
        """Vérifie les conditions de sortie anticipée.

        Retourne "signal_exit" ou None.
        Appelé UNIQUEMENT si ni TP ni SL n'ont été touchés sur cette bougie.
        """

    @abstractmethod
    def compute_indicators(
        self, candles_by_tf: dict[str, list[Candle]]
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Pré-calcule tous les indicateurs sur le dataset complet.

        Appelé une fois au début du backtest.
        Retourne {tf: {timestamp_iso: {indicator_name: value}}}.
        Le moteur gère l'alignement multi-TF (last_available_before).
        """

    @abstractmethod
    def get_current_conditions(self, ctx: StrategyContext) -> list[dict]:
        """Retourne les conditions d'entrée avec leur état actuel.

        Chaque condition : {"name": str, "met": bool, "value": float|str, "threshold": float|str}
        Ne modifie PAS la logique de trading (check_entry reste le point d'entrée).
        Méthode read-only pour le dashboard.
        """

    @property
    @abstractmethod
    def min_candles(self) -> dict[str, int]:
        """Nombre minimum de bougies par timeframe.

        Ex: {"5m": 300, "15m": 50}
        """
