"""StrategyArena — comparaison parallèle des stratégies.

Maintient un classement live basé sur les performances des LiveStrategyRunners
du Simulator. Capital isolé par stratégie.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math
import sqlite3

from backend.backtesting.simulator import GridStrategyRunner, LiveStrategyRunner, Simulator
from backend.core.database import Database
from backend.core.position_manager import TradeResult


@dataclass
class StrategyPerformance:
    """Performance d'une stratégie pour le classement Arena."""

    name: str
    capital: float
    net_pnl: float
    net_return_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    is_active: bool


class StrategyArena:
    """Comparaison parallèle des stratégies.

    Lit les stats des LiveStrategyRunners du Simulator
    et produit un classement.
    """

    def __init__(self, simulator: Simulator) -> None:
        self._simulator = simulator

    def get_ranking(self) -> list[StrategyPerformance]:
        """Retourne les stratégies classées par net_return_pct décroissant."""
        perfs = [
            self._compute_performance(runner)
            for runner in self._simulator.runners
        ]
        perfs.sort(key=lambda p: p.net_return_pct, reverse=True)
        return perfs

    def get_strategy_detail(self, name: str) -> dict | None:
        """Retourne le détail d'une stratégie (status + trades)."""
        for runner in self._simulator.runners:
            if runner.name == name:
                return {
                    "status": runner.get_status(),
                    "trades": [
                        {
                            "symbol": sym,
                            "direction": t.direction.value,
                            "entry_price": t.entry_price,
                            "exit_price": t.exit_price,
                            "quantity": t.quantity,
                            "entry_time": t.entry_time.isoformat(),
                            "exit_time": t.exit_time.isoformat(),
                            "net_pnl": t.net_pnl,
                            "exit_reason": t.exit_reason,
                        }
                        for sym, t in runner.get_trades()
                    ],
                    "performance": self._perf_to_dict(
                        self._compute_performance(runner)
                    ),
                }
        return None

    async def get_reporting_ranking(self, db: Database | None) -> list[dict]:
        """Persisted, reconciled reporting; never used by automatic selection."""
        ranking = [
            (await self._reporting_detail(runner, db))["performance"]
            for runner in self._simulator.runners
        ]
        return sorted(ranking, key=lambda p: p["net_return_pct"], reverse=True)

    async def get_reporting_detail(self, name: str, db: Database | None) -> dict | None:
        for runner in self._simulator.runners:
            if runner.name == name:
                return await self._reporting_detail(runner, db)
        return None

    async def _reporting_detail(
        self, runner: LiveStrategyRunner | GridStrategyRunner, db: Database | None,
    ) -> dict:
        # Freeze before the first await: persistence can lag or a new trade can
        # arrive during the query. Any mismatch fails closed for this response.
        stats = replace(runner.get_stats())
        status = runner.get_status()
        funding = getattr(runner, "_total_funding_cost", 0.0)
        memory_keys = Counter(
            (symbol, t.entry_time.isoformat(), t.exit_time.isoformat(), t.net_pnl)
            for symbol, t in runner.get_trades()
        )
        perf = self._perf_to_dict(self._compute_performance(runner))
        perf.update(
            profit_factor=None,
            profit_factor_unbounded=False,
            max_drawdown_pct=None,
            history_status="unavailable",
            history_reason="database_unavailable",
            history_trade_count=0,
            history_first_entry=None,
            history_first_exit=None,
            history_last_exit=None,
            history_window="latest_count_reconciled_not_session_id",
            metrics_basis="closed_trades_net_of_trading_costs_excluding_funding_and_unrealized",
            funding_cost=funding,
        )
        result = {"status": status, "trades": [], "performance": perf}
        if db is None:
            return result
        try:
            rows = await db.get_simulation_trades(
                strategy_name=runner.name, limit=stats.total_trades,
            ) if stats.total_trades else []
        except (sqlite3.Error, OSError):
            perf["history_reason"] = "database_read_failed"
            return result

        perf["history_trade_count"] = len(rows)
        if (runner.get_stats() != stats
                or getattr(runner, "_total_funding_cost", 0.0) != funding):
            perf["history_reason"] = "runner_changed_during_read"
            return result
        if len(rows) != stats.total_trades:
            perf["history_reason"] = "trade_count_mismatch"
            return result
        # Database returns descending (exit_time, id); reverse to preserve ties.
        rows = list(reversed(rows))
        if any(row["strategy"] != runner.name for row in rows):
            perf["history_reason"] = "strategy_mismatch"
            return result
        persisted_keys = Counter(
            (r["symbol"], r["entry_time"], r["exit_time"], r["net_pnl"])
            for r in rows
        )
        if memory_keys - persisted_keys:
            perf["history_reason"] = "in_memory_trades_mismatch"
            return result
        pnls = [row["net_pnl"] for row in rows]
        if not all(math.isfinite(p) for p in [*pnls, funding, stats.net_pnl]):
            perf["history_reason"] = "non_finite_pnl"
            return result
        if (sum(p > 0 for p in pnls) != stats.wins
                or sum(p <= 0 for p in pnls) != stats.losses):
            perf["history_reason"] = "win_loss_mismatch"
            return result
        if not math.isclose(
            math.fsum(pnls) - funding, stats.net_pnl, rel_tol=1e-9, abs_tol=1e-6,
        ):
            perf["history_reason"] = "pnl_funding_mismatch"
            return result
        pf = self._profit_factor_from_pnls(pnls)
        perf.update(
            # JSON cannot encode infinity; explicitly describe an all-win sample.
            profit_factor=pf if math.isfinite(pf) else None,
            profit_factor_unbounded=not math.isfinite(pf),
            max_drawdown_pct=self._drawdown_from_pnls(pnls, stats.initial_capital),
            history_status="reconciled", history_reason=None,
            history_first_entry=min((r["entry_time"] for r in rows), default=None),
            history_first_exit=rows[0]["exit_time"] if rows else None,
            history_last_exit=rows[-1]["exit_time"] if rows else None,
        )
        result["trades"] = rows
        return result

    def _compute_performance(self, runner: LiveStrategyRunner) -> StrategyPerformance:
        """Calcule les métriques de performance d'un runner."""
        stats = runner.get_stats()
        trades = [t for _, t in runner.get_trades()]

        # Win rate
        win_rate = 0.0
        if stats.total_trades > 0:
            win_rate = stats.wins / stats.total_trades * 100

        # Profit factor
        profit_factor = self._calc_profit_factor(trades)

        # Max drawdown
        max_dd_pct = self._calc_max_drawdown_pct(trades, stats.initial_capital)

        # Net return
        net_return_pct = 0.0
        if stats.initial_capital > 0:
            net_return_pct = stats.net_pnl / stats.initial_capital * 100

        return StrategyPerformance(
            name=runner.name,
            capital=stats.capital,
            net_pnl=stats.net_pnl,
            net_return_pct=net_return_pct,
            total_trades=stats.total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown_pct=max_dd_pct,
            is_active=stats.is_active,
        )

    @staticmethod
    def _calc_profit_factor(trades: list[TradeResult]) -> float:
        """Profit factor = gross_wins / gross_losses. 0.0 si pas de pertes."""
        return StrategyArena._profit_factor_from_pnls([t.net_pnl for t in trades])

    @staticmethod
    def _profit_factor_from_pnls(pnls: list[float]) -> float:
        gross_wins = sum(p for p in pnls if p > 0)
        gross_losses = abs(sum(p for p in pnls if p <= 0))
        if gross_losses == 0:
            return 0.0 if gross_wins == 0 else float("inf")
        return gross_wins / gross_losses

    @staticmethod
    def _calc_max_drawdown_pct(
        trades: list[TradeResult], initial_capital: float
    ) -> float:
        """Closed-trade drawdown as a percentage of the preceding peak."""
        return StrategyArena._drawdown_from_pnls(
            [t.net_pnl for t in trades], initial_capital,
        )

    @staticmethod
    def _drawdown_from_pnls(pnls: list[float], initial_capital: float) -> float:
        equity = initial_capital
        peak = equity
        max_dd = 0.0

        for pnl in pnls:
            equity += pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        return max_dd

    @staticmethod
    def _perf_to_dict(perf: StrategyPerformance) -> dict:
        return {
            "name": perf.name,
            "capital": perf.capital,
            "net_pnl": perf.net_pnl,
            "net_return_pct": perf.net_return_pct,
            "total_trades": perf.total_trades,
            "win_rate": perf.win_rate,
            "profit_factor": perf.profit_factor,
            "max_drawdown_pct": perf.max_drawdown_pct,
            "is_active": perf.is_active,
        }
