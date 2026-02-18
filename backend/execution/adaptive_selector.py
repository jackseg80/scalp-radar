"""AdaptiveSelector — contrôle quelles stratégies peuvent trader en live.

Évalue périodiquement les performances Arena et autorise/bloque les stratégies
en fonction de critères configurables (min_trades, profit_factor, live_eligible).

Hotfix 28a :
- FIX 1 : charge les trades historiques depuis simulation_trades DB au boot
- FIX 2 : bypass configurable au boot (autorise tout si DB vide + LIVE_TRADING)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from backend.backtesting.arena import StrategyArena
    from backend.core.config import AppConfig
    from backend.core.database import Database


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
    "grid_range_atr": "grid_range_atr",
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

    def __init__(
        self,
        arena: StrategyArena,
        config: AppConfig,
        db: Database | None = None,
    ) -> None:
        self._arena = arena
        self._config = config
        self._selector_config = config.risk.adaptive_selector
        self._db = db

        self._allowed_strategies: set[str] = set()
        self._active_symbols: set[str] = set()
        self._task: asyncio.Task | None = None

        # FIX 1 : compteurs trades DB (chargés une fois au start)
        self._db_trade_counts: dict[str, int] = {}

        # FIX 2 : bypass au boot (autorise tout sans min_trades/return/PF)
        self._bypass_active: bool = (
            getattr(config.risk, "selector_bypass_at_boot", False)
            and getattr(config.secrets, "live_trading", False)
        )

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
        min_trades = self._selector_config.min_trades
        force_set = set(self._selector_config.force_strategies)

        # Bypass : collecter les eligible pour vérifier si TOUTES sont prêtes
        bypass_eligible_counts: list[int] = []

        for perf in ranking:
            # 1. live_eligible dans strategies.yaml
            if not self._is_live_eligible(perf.name):
                continue

            # 2. Stratégie active (kill switch simulation non déclenché)
            if not perf.is_active:
                continue

            # FIX 1 : trades effectifs = max(mémoire, DB)
            effective_trades = max(
                perf.total_trades,
                self._db_trade_counts.get(perf.name, 0),
            )

            # Bypass mode : skip min_trades/net_return/profit_factor
            if self._bypass_active:
                new_allowed.add(perf.name)
                bypass_eligible_counts.append(effective_trades)
                continue

            # 3. Assez de trades en simulation
            if effective_trades < min_trades:
                continue

            # FIX deadlock : session vierge (0 trades Arena) → skip net_return/PF
            # On ne peut pas évaluer une performance qui n'existe pas
            if perf.total_trades == 0:
                new_allowed.add(perf.name)
                continue

            # 4. Rentable (net return > 0)
            if perf.net_return_pct <= 0:
                # force_strategies : bypass net_return/PF si forcée
                if perf.name in force_set:
                    logger.warning(
                        "AdaptiveSelector: {} forcée malgré net_return={:.1f}%",
                        perf.name,
                        perf.net_return_pct,
                    )
                    new_allowed.add(perf.name)
                continue

            # 5. Profit factor suffisant
            if perf.profit_factor < self._selector_config.min_profit_factor:
                # force_strategies : bypass PF si forcée
                if perf.name in force_set:
                    logger.warning(
                        "AdaptiveSelector: {} forcée malgré PF={:.2f}",
                        perf.name,
                        perf.profit_factor,
                    )
                    new_allowed.add(perf.name)
                continue

            new_allowed.add(perf.name)

        # Auto-désactivation bypass : TOUTES les eligible doivent atteindre min_trades
        if self._bypass_active and bypass_eligible_counts:
            if all(count >= min_trades for count in bypass_eligible_counts):
                logger.info(
                    "AdaptiveSelector: bypass désactivé (toutes les stratégies "
                    "ont >= {} trades)",
                    min_trades,
                )
                self._bypass_active = False

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
        await self._load_trade_counts_from_db()
        self.evaluate()
        self._task = asyncio.create_task(self._eval_loop())
        bypass_msg = " (BYPASS ACTIF)" if self._bypass_active else ""
        logger.info(
            "AdaptiveSelector démarré (intervalle={}s, min_trades={}, min_pf={:.1f}){}",
            self._selector_config.eval_interval_seconds,
            self._selector_config.min_trades,
            self._selector_config.min_profit_factor,
            bypass_msg,
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
            "bypass_active": self._bypass_active,
            "db_trade_counts": self._db_trade_counts,
        }

    # ─── Internals ───────────────────────────────────────────────────

    async def _load_trade_counts_from_db(self) -> None:
        """Charge le nombre de trades historiques depuis simulation_trades.

        Résiste aux erreurs : si DB indisponible ou table absente,
        garde un dict vide (pas de crash).
        """
        if self._db is None:
            return
        try:
            self._db_trade_counts = await self._db.get_trade_counts_by_strategy()
            if self._db_trade_counts:
                logger.info(
                    "AdaptiveSelector: trades DB chargés — {}",
                    ", ".join(
                        f"{k}={v}" for k, v in sorted(self._db_trade_counts.items())
                    ),
                )
        except Exception:
            logger.warning("AdaptiveSelector: impossible de charger les trades DB")
            self._db_trade_counts = {}

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
