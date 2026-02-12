"""StrategyArena — comparaison parallèle des stratégies.

Maintient un classement live basé sur les performances des LiveStrategyRunners
du Simulator. Capital isolé par stratégie.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.backtesting.simulator import LiveStrategyRunner, Simulator
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
        gross_wins = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        gross_losses = abs(sum(t.net_pnl for t in trades if t.net_pnl <= 0))
        if gross_losses == 0:
            return 0.0 if gross_wins == 0 else float("inf")
        return gross_wins / gross_losses

    @staticmethod
    def _calc_max_drawdown_pct(
        trades: list[TradeResult], initial_capital: float
    ) -> float:
        """Max drawdown en % du capital initial."""
        if not trades:
            return 0.0

        equity = initial_capital
        peak = equity
        max_dd = 0.0

        for trade in trades:
            equity += trade.net_pnl
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
