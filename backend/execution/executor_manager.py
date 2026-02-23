"""ExecutorManager : agrège N Executor instances (Sprint 36b).

Duck-type l'ancien Executor singleton pour que les consommateurs
(routes API, WebSocket, health, watchdog) fonctionnent sans changement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from backend.execution.executor import Executor
    from backend.execution.risk_manager import LiveRiskManager


class ExecutorManager:
    """Gère N Executor instances (un par stratégie live).

    Expose get_status(), is_enabled, is_connected, exchange_balance
    avec la même interface que l'ancien Executor singleton.
    """

    def __init__(self) -> None:
        self._executors: dict[str, Executor] = {}
        self._risk_managers: dict[str, LiveRiskManager] = {}

    def add(
        self,
        strategy_name: str,
        executor: Executor,
        risk_manager: LiveRiskManager,
    ) -> None:
        self._executors[strategy_name] = executor
        self._risk_managers[strategy_name] = risk_manager

    def get(self, strategy_name: str) -> Executor | None:
        return self._executors.get(strategy_name)

    @property
    def executors(self) -> dict[str, Executor]:
        return self._executors

    @property
    def risk_managers(self) -> dict[str, LiveRiskManager]:
        return self._risk_managers

    @property
    def is_enabled(self) -> bool:
        return any(e.is_enabled for e in self._executors.values())

    @property
    def is_connected(self) -> bool:
        if not self._executors:
            return False
        return all(
            e.is_connected for e in self._executors.values() if e.is_enabled
        )

    @property
    def exchange_balance(self) -> float | None:
        if not self._executors:
            return None
        total = 0.0
        for e in self._executors.values():
            bal = e.exchange_balance
            if bal is not None:
                total += bal
        return total

    def get_status(self) -> dict[str, Any]:
        """Statut agrégé — même format que l'ancien Executor.get_status()."""
        if not self._executors:
            return {"enabled": False, "message": "Aucun executor actif"}

        all_positions: list[dict[str, Any]] = []
        executor_grids: dict[str, dict[str, Any]] = {}
        per_strategy: dict[str, dict[str, Any]] = {}

        for name, executor in self._executors.items():
            status = executor.get_status()
            per_strategy[name] = status

            all_positions.extend(status.get("positions", []))

            egs = status.get("executor_grid_state")
            if egs and egs.get("grid_positions"):
                executor_grids.update(egs["grid_positions"])

        result: dict[str, Any] = {
            "enabled": self.is_enabled,
            "connected": self.is_connected,
            "exchange_balance": self.exchange_balance,
            "position": all_positions[0] if all_positions else None,
            "positions": all_positions,
            "risk_manager": self._aggregate_risk_status(),
            "per_strategy": per_strategy,
        }

        if executor_grids:
            result["executor_grid_state"] = {
                "grid_positions": executor_grids,
                "summary": {
                    "total_positions": sum(
                        g.get("levels", 0) for g in executor_grids.values()
                    ),
                    "total_assets": len(executor_grids),
                    "total_margin_used": round(
                        sum(g.get("margin_used", 0) for g in executor_grids.values()), 2,
                    ),
                    "total_unrealized_pnl": round(
                        sum(g.get("unrealized_pnl", 0) for g in executor_grids.values()), 2,
                    ),
                },
            }

        # Selector (partagé entre executors)
        for e in self._executors.values():
            sel = getattr(e, "_selector", None)
            if sel:
                result["selector"] = sel.get_status()
                break

        return result

    def _aggregate_risk_status(self) -> dict[str, Any]:
        """Agrège les risk managers de tous les executors."""
        total_pnl = 0.0
        total_orders = 0
        any_ks = False
        total_capital = 0.0
        open_count = 0

        for rm in self._risk_managers.values():
            total_pnl += getattr(rm, "_session_pnl", 0.0)
            total_orders += getattr(rm, "_total_orders", 0)
            if getattr(rm, "is_kill_switch_triggered", False):
                any_ks = True
            total_capital += getattr(rm, "_initial_capital", 0.0)
            open_count += getattr(rm, "open_positions_count", 0)

        return {
            "session_pnl": total_pnl,
            "kill_switch": any_ks,
            "open_positions_count": open_count,
            "total_orders": total_orders,
            "initial_capital": total_capital,
        }

    def get_all_order_history(self, limit: int = 50) -> list[dict]:
        """Merge les historiques d'ordres de tous les executors, triés par timestamp."""
        all_orders: list[dict] = []
        for e in self._executors.values():
            all_orders.extend(e._order_history)
        all_orders.sort(key=lambda o: o.get("timestamp", ""), reverse=True)
        return all_orders[:limit]

    async def refresh_all_balances(self) -> dict[str, float | None]:
        results: dict[str, float | None] = {}
        for name, executor in self._executors.items():
            try:
                results[name] = await executor.refresh_balance()
            except Exception as e:
                logger.warning("ExecutorManager: refresh balance {} échoué: {}", name, e)
                results[name] = None
        return results

    async def stop_all(self) -> None:
        for name, executor in self._executors.items():
            try:
                await executor.stop()
                logger.info("ExecutorManager: {} arrêté", name)
            except Exception as e:
                logger.error("ExecutorManager: erreur arrêt {}: {}", name, e)
