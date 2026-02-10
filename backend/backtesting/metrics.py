"""Calcul des métriques de backtesting pour Scalp Radar.

Inclut : win rate, profit factor (net + gross), Sharpe, Sortino,
max drawdown (% et duree), fee drag, breakdown par regime de marche.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from backend.backtesting.engine import BacktestResult, TradeResult


@dataclass
class BacktestMetrics:
    """Métriques complètes d'un backtest."""

    # Performance
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L
    gross_pnl: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    net_pnl: float = 0.0
    net_return_pct: float = 0.0

    # Ratios
    profit_factor: float = 0.0  # net wins / net losses
    gross_profit_factor: float = 0.0  # gross wins / gross losses
    avg_win: float = 0.0
    avg_loss: float = 0.0
    risk_reward_ratio: float = 0.0
    expectancy: float = 0.0

    # Risk
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: timedelta = field(default_factory=lambda: timedelta())
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Fee impact
    fee_drag_pct: float = 0.0

    # Breakdown par régime
    regime_stats: dict[str, dict] = field(default_factory=dict)


def calculate_metrics(result: BacktestResult) -> BacktestMetrics:
    """Calcule toutes les métriques depuis un BacktestResult."""
    metrics = BacktestMetrics()
    trades = result.trades

    if not trades:
        return metrics

    # --- Performance -----------------------------------------------------
    metrics.total_trades = len(trades)
    metrics.winning_trades = sum(1 for t in trades if t.net_pnl > 0)
    metrics.losing_trades = sum(1 for t in trades if t.net_pnl <= 0)
    metrics.win_rate = metrics.winning_trades / metrics.total_trades * 100

    # --- P&L -------------------------------------------------------------
    metrics.gross_pnl = sum(t.gross_pnl for t in trades)
    metrics.total_fees = sum(t.fee_cost for t in trades)
    metrics.total_slippage = sum(t.slippage_cost for t in trades)
    metrics.net_pnl = sum(t.net_pnl for t in trades)
    metrics.net_return_pct = metrics.net_pnl / result.config.initial_capital * 100

    # --- Profit Factor (net et gross) ------------------------------------
    net_wins = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    net_losses = abs(sum(t.net_pnl for t in trades if t.net_pnl <= 0))
    gross_wins = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
    gross_losses = abs(sum(t.gross_pnl for t in trades if t.gross_pnl <= 0))

    metrics.profit_factor = net_wins / net_losses if net_losses > 0 else float("inf")
    metrics.gross_profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # --- Avg win / loss --------------------------------------------------
    winning = [t.net_pnl for t in trades if t.net_pnl > 0]
    losing = [t.net_pnl for t in trades if t.net_pnl <= 0]

    metrics.avg_win = np.mean(winning) if winning else 0.0
    metrics.avg_loss = np.mean(losing) if losing else 0.0
    metrics.risk_reward_ratio = (
        abs(metrics.avg_win / metrics.avg_loss) if metrics.avg_loss != 0 else float("inf")
    )

    win_rate_frac = metrics.win_rate / 100
    loss_rate_frac = 1 - win_rate_frac
    metrics.expectancy = (
        win_rate_frac * metrics.avg_win + loss_rate_frac * metrics.avg_loss
    )

    # --- Fee drag --------------------------------------------------------
    # Ratio fees / volume brut des gains (si positifs) ou fees / total turnover
    if gross_wins > 0:
        metrics.fee_drag_pct = metrics.total_fees / gross_wins * 100
    else:
        metrics.fee_drag_pct = 100.0 if metrics.total_fees > 0 else 0.0

    # --- Drawdown (sur l'equity curve, point par bougie) -----------------
    if result.equity_curve:
        equity = np.array(result.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100

        metrics.max_drawdown_pct = abs(float(np.min(drawdown)))

        # Durée du max drawdown
        if result.equity_timestamps and len(result.equity_timestamps) == len(equity):
            metrics.max_drawdown_duration = _calculate_max_dd_duration(
                equity, result.equity_timestamps
            )

    # --- Sharpe & Sortino ------------------------------------------------
    returns = _calculate_trade_returns(trades, result.config.initial_capital)
    if len(returns) >= 2:
        # Annualisation basée sur la fréquence réelle des trades
        if len(result.equity_timestamps) >= 2:
            total_days = (result.equity_timestamps[-1] - result.equity_timestamps[0]).total_seconds() / 86400
            trades_per_year = len(trades) / max(total_days, 1) * 365
        else:
            trades_per_year = 365  # fallback
        metrics.sharpe_ratio = _sharpe(returns, periods_per_year=trades_per_year)
        metrics.sortino_ratio = _sortino(returns, periods_per_year=trades_per_year)

    # --- Breakdown par régime --------------------------------------------
    metrics.regime_stats = _regime_breakdown(trades)

    return metrics


def _calculate_trade_returns(trades: list[TradeResult], initial_capital: float) -> list[float]:
    """Calcule les rendements par trade (net_pnl / capital avant trade)."""
    returns = []
    capital = initial_capital
    for trade in trades:
        if capital > 0:
            returns.append(trade.net_pnl / capital)
        capital += trade.net_pnl
    return returns


def _sharpe(returns: list[float], periods_per_year: float) -> float:
    """Sharpe ratio annualisé, risk-free rate = 0."""
    arr = np.array(returns)
    if np.std(arr) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr) * np.sqrt(periods_per_year))


def _sortino(returns: list[float], periods_per_year: float) -> float:
    """Sortino ratio annualisé. Dénominateur = downside deviation.

    downside_std = sqrt(mean(min(returns, 0)²))
    """
    arr = np.array(returns)
    downside = np.minimum(arr, 0.0)
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std == 0:
        return 0.0
    return float(np.mean(arr) / downside_std * np.sqrt(periods_per_year))


def _calculate_max_dd_duration(
    equity: np.ndarray, timestamps: list[datetime]
) -> timedelta:
    """Calcule la duree du plus long drawdown (peak → recovery ou fin)."""
    peak = np.maximum.accumulate(equity)
    in_drawdown = equity < peak

    max_duration = timedelta()
    dd_start = None

    for i in range(len(equity)):
        if in_drawdown[i]:
            if dd_start is None:
                dd_start = timestamps[i]
        else:
            if dd_start is not None:
                duration = timestamps[i] - dd_start
                if duration > max_duration:
                    max_duration = duration
                dd_start = None

    # Si on termine en drawdown
    if dd_start is not None:
        duration = timestamps[-1] - dd_start
        if duration > max_duration:
            max_duration = duration

    return max_duration


def _regime_breakdown(trades: list[TradeResult]) -> dict[str, dict]:
    """Breakdown des performances par regime de marche."""
    regimes: dict[str, list[TradeResult]] = {}
    for trade in trades:
        regime_name = trade.market_regime.value
        if regime_name not in regimes:
            regimes[regime_name] = []
        regimes[regime_name].append(trade)

    stats = {}
    for regime_name, regime_trades in regimes.items():
        wins = sum(1 for t in regime_trades if t.net_pnl > 0)
        total = len(regime_trades)
        stats[regime_name] = {
            "trades": total,
            "win_rate": wins / total * 100 if total > 0 else 0.0,
            "net_pnl": sum(t.net_pnl for t in regime_trades),
        }

    return stats


def format_metrics_table(metrics: BacktestMetrics, title: str = "") -> str:
    """Formate les métriques en tableau lisible pour la console."""
    lines = []

    if title:
        border = "=" * 50
        lines.append(f"\n  {border}")
        lines.append(f"  {title}")
        lines.append(f"  {border}")

    lines.append("")
    lines.append("  Performance")
    lines.append("  -----------")
    lines.append(f"  Trades         : {metrics.total_trades}")
    lines.append(f"  Win rate       : {metrics.win_rate:.1f}%")
    lines.append(f"  Net P&L        : {metrics.net_pnl:+.2f} ({metrics.net_return_pct:+.1f}%)")
    lines.append(f"  Profit factor  : {metrics.profit_factor:.2f} (net) / {metrics.gross_profit_factor:.2f} (gross)")

    lines.append("")
    lines.append("  Frais & Slippage")
    lines.append("  ----------------")
    lines.append(f"  Gross P&L      : {metrics.gross_pnl:+.2f}")
    lines.append(f"  Fees           : -{metrics.total_fees:.2f} ({metrics.fee_drag_pct:.1f}% des gains bruts)")
    lines.append(f"  Slippage       : -{metrics.total_slippage:.2f}")
    lines.append(f"  Net P&L        : {metrics.net_pnl:+.2f}")

    lines.append("")
    lines.append("  Risque")
    lines.append("  ------")
    dd_hours = metrics.max_drawdown_duration.total_seconds() / 3600
    if dd_hours >= 24:
        dd_str = f"{dd_hours / 24:.0f}j {dd_hours % 24:.0f}h"
    else:
        dd_str = f"{dd_hours:.0f}h"
    lines.append(f"  Max drawdown   : -{metrics.max_drawdown_pct:.1f}% (duree: {dd_str})")
    lines.append(f"  Sharpe ratio   : {metrics.sharpe_ratio:.2f}")
    lines.append(f"  Sortino ratio  : {metrics.sortino_ratio:.2f}")

    lines.append("")
    lines.append("  Details")
    lines.append("  -------")
    lines.append(f"  Avg win        : {metrics.avg_win:+.2f}")
    lines.append(f"  Avg loss       : {metrics.avg_loss:+.2f}")
    lines.append(f"  Risk/Reward    : {metrics.risk_reward_ratio:.2f}")
    lines.append(f"  Expectancy     : {metrics.expectancy:+.2f}")

    if metrics.regime_stats:
        lines.append("")
        lines.append("  Par regime de marche")
        lines.append("  --------------------")
        for regime, stats in sorted(metrics.regime_stats.items()):
            lines.append(
                f"  {regime:<18s}: {stats['trades']} trades, "
                f"{stats['win_rate']:.0f}% win, {stats['net_pnl']:+.2f}"
            )

    if title:
        lines.append(f"  {'=' * 50}")

    return "\n".join(lines)
