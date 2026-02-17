"""AdaptiveSelector — contrôle quelles stratégies peuvent trader en live.

Évalue périodiquement les performances Arena et autorise/bloque les stratégies
en fonction de critères configurables (min_trades, profit_factor, live_eligible).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from backend.backtesting.arena import StrategyArena
    from backend.core.config import AppConfig


# Mapping nom stratégie → attribut config pour accéder à live_eligible
_STRATEGY_CONFIG_ATTR = {
    "vwap_rsi": "vwap_rsi",
    "momentum": "momentum",
    "funding": "funding",
    "liquidation": "liquidation",
    "envelope_dca": "envelope_dca",
    "envelope_dca_short": "envelope_dca_short",
    "bollinger_mr": "bollinger_mr",
    "donchian_breakout": "donchian_breakout",
    "supertrend": "supertrend",
    "grid_atr": "grid_atr",
    "grid_multi_tf": "grid_multi_tf",
    "grid_funding": "grid_funding",
    "grid_trend": "grid_trend",
}


class AdaptiveSelector:
    """Sélecteur adaptatif : gate les OPEN events vers l'Executor.

    - Seuls les OPEN sont gatés (les CLOSE passent toujours)
    - Évalue depuis Arena.get_ranking() : live_eligible, is_active,
      min_trades, net_return > 0, profit_factor >= seuil
    - Réévaluation périodique (configurable, défaut 5 min)
    """

    def __init__(self, arena: StrategyArena, config: AppConfig) -> None:
        self._arena = arena
        self._config = config
        self._selector_config = config.risk.adaptive_selector

        self._allowed_strategies: set[str] = set()
        self._active_symbols: set[str] = set()
        self._task: asyncio.Task | None = None

    # ─── Public API ──────────────────────────────────────────────────

    def is_allowed(self, strategy_name: str, symbol: str) -> bool:
        """Vérifie si une stratégie peut ouvrir un trade sur ce symbole."""
        if symbol not in self._active_symbols:
            return False
        return strategy_name in self._allowed_strategies

    def evaluate(self) -> None:
        """Réévalue quelles stratégies sont autorisées en live."""
        ranking = self._arena.get_ranking()
        new_allowed: set[str] = set()

        for perf in ranking:
            # 1. live_eligible dans strategies.yaml
            if not self._is_live_eligible(perf.name):
                continue

            # 2. Stratégie active (kill switch simulation non déclenché)
            if not perf.is_active:
                continue

            # 3. Assez de trades en simulation
            if perf.total_trades < self._selector_config.min_trades:
                continue

            # 4. Rentable (net return > 0)
            if perf.net_return_pct <= 0:
                continue

            # 5. Profit factor suffisant
            if perf.profit_factor < self._selector_config.min_profit_factor:
                continue

            new_allowed.add(perf.name)

        # Log les changements
        added = new_allowed - self._allowed_strategies
        removed = self._allowed_strategies - new_allowed

        if added:
            logger.info(
                "AdaptiveSelector: stratégies autorisées: +{}",
                ", ".join(sorted(added)),
            )
        if removed:
            logger.info(
                "AdaptiveSelector: stratégies retirées: -{}",
                ", ".join(sorted(removed)),
            )

        self._allowed_strategies = new_allowed

    def set_active_symbols(self, symbols: set[str]) -> None:
        """Définit les symboles actifs (ceux dont le leverage setup a réussi)."""
        self._active_symbols = set(symbols)
        logger.info(
            "AdaptiveSelector: symboles actifs = {}",
            ", ".join(sorted(self._active_symbols)) or "(aucun)",
        )

    # ─── Lifecycle ───────────────────────────────────────────────────

    async def start(self) -> None:
        """Évaluation initiale + lance la boucle périodique."""
        self.evaluate()
        self._task = asyncio.create_task(self._eval_loop())
        logger.info(
            "AdaptiveSelector démarré (intervalle={}s, min_trades={}, min_pf={:.1f})",
            self._selector_config.eval_interval_seconds,
            self._selector_config.min_trades,
            self._selector_config.min_profit_factor,
        )

    async def stop(self) -> None:
        """Arrête la boucle périodique."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("AdaptiveSelector arrêté")

    # ─── Status ──────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Statut pour le dashboard."""
        return {
            "allowed_strategies": sorted(self._allowed_strategies),
            "active_symbols": sorted(self._active_symbols),
            "min_trades": self._selector_config.min_trades,
            "min_profit_factor": self._selector_config.min_profit_factor,
            "eval_interval_seconds": self._selector_config.eval_interval_seconds,
        }

    # ─── Internals ───────────────────────────────────────────────────

    def _is_live_eligible(self, strategy_name: str) -> bool:
        """Vérifie le flag live_eligible dans strategies.yaml."""
        attr_name = _STRATEGY_CONFIG_ATTR.get(strategy_name)
        if attr_name is None:
            return False
        strategy_cfg = getattr(self._config.strategies, attr_name, None)
        if strategy_cfg is None:
            return False
        return getattr(strategy_cfg, "live_eligible", False)

    async def _eval_loop(self) -> None:
        """Boucle d'évaluation périodique."""
        while True:
            await asyncio.sleep(self._selector_config.eval_interval_seconds)
            try:
                self.evaluate()
            except Exception:
                logger.exception("AdaptiveSelector: erreur lors de l'évaluation")
