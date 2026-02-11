"""Tests pour backend/backtesting/arena.py."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from backend.core.models import Direction, MarketRegime
from backend.core.position_manager import TradeResult
from backend.backtesting.simulator import LiveStrategyRunner, RunnerStats, Simulator
from backend.backtesting.arena import StrategyArena, StrategyPerformance


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_mock_runner(
    name: str = "strat_a",
    capital: float = 10_000.0,
    initial_capital: float = 10_000.0,
    net_pnl: float = 0.0,
    total_trades: int = 0,
    wins: int = 0,
    losses: int = 0,
    is_active: bool = True,
    trades: list[TradeResult] | None = None,
) -> MagicMock:
    """Crée un mock LiveStrategyRunner avec stats configurables."""
    runner = MagicMock(spec=LiveStrategyRunner)
    runner.name = name
    runner.is_kill_switch_triggered = not is_active

    stats = RunnerStats(
        capital=capital,
        initial_capital=initial_capital,
        net_pnl=net_pnl,
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        is_active=is_active,
    )
    runner.get_stats.return_value = stats
    runner.get_trades.return_value = trades or []
    runner.get_status.return_value = {
        "name": name,
        "capital": capital,
        "net_pnl": net_pnl,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / total_trades * 100 if total_trades > 0 else 0.0,
        "is_active": is_active,
        "kill_switch": not is_active,
        "has_position": False,
    }
    return runner


def _make_trade(net_pnl: float, direction: Direction = Direction.LONG) -> TradeResult:
    return TradeResult(
        direction=direction,
        entry_price=100_000.0,
        exit_price=100_100.0 if net_pnl > 0 else 99_900.0,
        quantity=0.01,
        entry_time=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
        exit_time=datetime(2024, 1, 15, 12, 5, tzinfo=timezone.utc),
        gross_pnl=net_pnl + 2.0 if net_pnl > 0 else net_pnl + 1.0,
        fee_cost=1.5,
        slippage_cost=0.5,
        net_pnl=net_pnl,
        exit_reason="tp" if net_pnl > 0 else "sl",
        market_regime=MarketRegime.RANGING,
    )


def _make_arena(runners: list[MagicMock]) -> StrategyArena:
    sim = MagicMock(spec=Simulator)
    sim.runners = runners
    return StrategyArena(sim)


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestGetRanking:
    def test_empty_ranking(self):
        """Pas de runners → ranking vide."""
        arena = _make_arena([])
        assert arena.get_ranking() == []

    def test_single_runner_no_trades(self):
        """Un runner sans trades → perf à zéro."""
        runner = _make_mock_runner(name="strat_a")
        arena = _make_arena([runner])
        ranking = arena.get_ranking()
        assert len(ranking) == 1
        assert ranking[0].name == "strat_a"
        assert ranking[0].net_pnl == 0.0
        assert ranking[0].profit_factor == 0.0
        assert ranking[0].max_drawdown_pct == 0.0

    def test_ranking_sorted_by_return(self):
        """Classement trié par net_return_pct décroissant."""
        runner_a = _make_mock_runner(
            name="strat_a",
            capital=10_500.0,
            net_pnl=500.0,
            total_trades=10,
            wins=7,
            losses=3,
            trades=[_make_trade(80.0) for _ in range(7)] + [_make_trade(-10.0) for _ in range(3)],
        )
        runner_b = _make_mock_runner(
            name="strat_b",
            capital=10_200.0,
            net_pnl=200.0,
            total_trades=5,
            wins=3,
            losses=2,
            trades=[_make_trade(100.0) for _ in range(3)] + [_make_trade(-50.0) for _ in range(2)],
        )
        runner_c = _make_mock_runner(
            name="strat_c",
            capital=9_800.0,
            net_pnl=-200.0,
            total_trades=4,
            wins=1,
            losses=3,
            trades=[_make_trade(50.0)] + [_make_trade(-83.33) for _ in range(3)],
        )

        arena = _make_arena([runner_c, runner_a, runner_b])
        ranking = arena.get_ranking()
        assert [r.name for r in ranking] == ["strat_a", "strat_b", "strat_c"]

    def test_profit_factor_calculation(self):
        """Profit factor = gains / pertes."""
        trades = [_make_trade(100.0), _make_trade(50.0), _make_trade(-30.0)]
        runner = _make_mock_runner(
            name="strat_pf",
            net_pnl=120.0,
            total_trades=3,
            wins=2,
            losses=1,
            trades=trades,
        )
        arena = _make_arena([runner])
        ranking = arena.get_ranking()
        pf = ranking[0].profit_factor
        assert pf == pytest.approx(150.0 / 30.0, abs=0.01)

    def test_profit_factor_no_losses(self):
        """Profit factor = inf si que des gains."""
        trades = [_make_trade(100.0), _make_trade(50.0)]
        runner = _make_mock_runner(
            name="strat_win",
            net_pnl=150.0,
            total_trades=2,
            wins=2,
            losses=0,
            trades=trades,
        )
        arena = _make_arena([runner])
        ranking = arena.get_ranking()
        assert ranking[0].profit_factor == float("inf")

    def test_max_drawdown_calculation(self):
        """Max drawdown calculé correctement sur l'equity curve."""
        trades = [
            _make_trade(100.0),   # equity: 10100
            _make_trade(200.0),   # equity: 10300 (peak)
            _make_trade(-150.0),  # equity: 10150
            _make_trade(-100.0),  # equity: 10050 (trough : dd = 250/10300 = 2.43%)
            _make_trade(50.0),    # equity: 10100
        ]
        runner = _make_mock_runner(
            name="strat_dd",
            capital=10_100.0,
            net_pnl=100.0,
            total_trades=5,
            wins=3,
            losses=2,
            trades=trades,
        )
        arena = _make_arena([runner])
        ranking = arena.get_ranking()
        expected_dd = (10300 - 10050) / 10300 * 100
        assert ranking[0].max_drawdown_pct == pytest.approx(expected_dd, abs=0.01)

    def test_win_rate_calculation(self):
        """Win rate calculé correctement."""
        runner = _make_mock_runner(
            name="strat_wr",
            total_trades=10,
            wins=6,
            losses=4,
        )
        arena = _make_arena([runner])
        ranking = arena.get_ranking()
        assert ranking[0].win_rate == pytest.approx(60.0, abs=0.01)


class TestGetStrategyDetail:
    def test_existing_strategy(self):
        """Détail d'une stratégie existante."""
        trades = [_make_trade(50.0)]
        runner = _make_mock_runner(name="my_strat", trades=trades, total_trades=1, wins=1)
        arena = _make_arena([runner])
        detail = arena.get_strategy_detail("my_strat")
        assert detail is not None
        assert "status" in detail
        assert "trades" in detail
        assert "performance" in detail
        assert len(detail["trades"]) == 1

    def test_unknown_strategy(self):
        """Stratégie inconnue → None."""
        arena = _make_arena([])
        assert arena.get_strategy_detail("unknown") is None
