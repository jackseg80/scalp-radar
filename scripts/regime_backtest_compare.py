"""Comparaison A/B/C : impact du leverage dynamique (régime BTC).

Run A : leverage fixe 7x (baseline)
Run B : leverage fixe 4x (borne conservative)
Run C : leverage dynamique 7x/4x piloté par ema_atr

Usage :
    uv run python -m scripts.regime_backtest_compare --strategy grid_atr
    uv run python -m scripts.regime_backtest_compare --strategy grid_atr --days 365

Sprint 50b.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from loguru import logger

from backend.backtesting.portfolio_engine import (
    PortfolioBacktester,
    PortfolioResult,
    PortfolioSnapshot,
    format_portfolio_report,
)
from backend.core.config import get_config
from backend.core.logging_setup import setup_logging
from backend.regime.btc_regime_signal import RegimeSignal, compute_regime_signal


# ─── Métriques ───────────────────────────────────────────────────────────


def _calc_sharpe(snapshots: list[PortfolioSnapshot]) -> float:
    """Sharpe annualisé depuis equity snapshots 1h."""
    equities = [s.total_equity for s in snapshots]
    if len(equities) < 2:
        return 0.0
    returns = [
        (equities[i + 1] - equities[i]) / equities[i]
        for i in range(len(equities) - 1)
        if equities[i] > 0
    ]
    if not returns:
        return 0.0
    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    std_r = var**0.5
    return (mean_r / std_r * (24 * 365) ** 0.5) if std_r > 0 else 0.0


def _calc_calmar(total_return_pct: float, max_dd_pct: float, n_days: int = 365) -> float:
    """Calmar annualisé = (return_pct / n_years) / |max_dd_pct|."""
    if max_dd_pct == 0:
        return float("inf") if total_return_pct > 0 else 0.0
    n_years = max(n_days / 365.25, 0.1)
    annual_return_pct = total_return_pct / n_years
    return round(annual_return_pct / abs(max_dd_pct), 2)


def _compute_verdict(
    return_a: float,
    return_c: float,
    dd_a: float,
    dd_c: float,
    sharpe_a: float,
    sharpe_c: float,
) -> tuple[str, dict[str, bool]]:
    """Calcule le verdict Go/No-Go.

    Critères :
    - Return C >= 80% de Return A
    - DD C < DD A (amélioration)
    - Sharpe C >= Sharpe A

    2/3 → GO, 1/3 → BORDERLINE, 0/3 → NO-GO
    """
    criteria = {
        "return_ok": return_c >= 0.8 * return_a,
        "dd_ok": abs(dd_c) < abs(dd_a),
        "sharpe_ok": sharpe_c >= sharpe_a,
    }
    score = sum(criteria.values())
    if score >= 2:
        verdict = "GO"
    elif score == 0:
        verdict = "NO-GO"
    else:
        verdict = "BORDERLINE"
    return verdict, criteria


# ─── Helpers rapport ────────────────────────────────────────────────────


def _regime_breakdown(
    snapshots: list[PortfolioSnapshot],
    regime_signal: RegimeSignal,
) -> dict[str, dict]:
    """Breakdown des performances par régime (normal / defensive)."""
    if not snapshots:
        return {}

    blocks: dict[str, list[PortfolioSnapshot]] = {"normal": [], "defensive": []}
    for snap in snapshots:
        regime = regime_signal.get_regime_at(snap.timestamp)
        blocks[regime].append(snap)

    result: dict[str, dict] = {}
    for regime, snaps in blocks.items():
        if len(snaps) < 2:
            result[regime] = {
                "hours": len(snaps),
                "return_pct": 0.0,
                "max_dd_pct": 0.0,
            }
            continue

        # Return sur ce bloc
        first_eq = snaps[0].total_equity
        last_eq = snaps[-1].total_equity
        ret_pct = (last_eq / first_eq - 1) * 100 if first_eq > 0 else 0.0

        # Max DD sur ce bloc
        peak = snaps[0].total_equity
        max_dd = 0.0
        for s in snaps:
            if s.total_equity > peak:
                peak = s.total_equity
            dd = (s.total_equity / peak - 1) * 100 if peak > 0 else 0.0
            if dd < max_dd:
                max_dd = dd

        result[regime] = {
            "hours": len(snaps),
            "return_pct": round(ret_pct, 2),
            "max_dd_pct": round(max_dd, 2),
        }

    return result


def _format_table_row(label: str, result: PortfolioResult, sharpe: float, calmar: float) -> str:
    """Formate une ligne du tableau comparatif."""
    return (
        f"| {label:20s} "
        f"| {result.total_return_pct:>8.1f}% "
        f"| {result.max_drawdown_pct:>7.2f}% "
        f"| {sharpe:>6.2f} "
        f"| {calmar:>6.2f} "
        f"| {result.total_trades:>6d} "
        f"| {result.win_rate:>5.1f}% |"
    )


def _generate_report(
    result_a: PortfolioResult,
    result_b: PortfolioResult,
    result_c: PortfolioResult,
    regime_signal: RegimeSignal,
    sharpe_a: float,
    sharpe_b: float,
    sharpe_c: float,
    calmar_a: float,
    calmar_b: float,
    calmar_c: float,
    verdict: str,
    criteria: dict[str, bool],
    breakdown: dict[str, dict],
) -> str:
    """Génère le rapport Markdown."""
    lines: list[str] = []
    lines.append("# Regime Impact Report — Sprint 50b\n")
    lines.append(f"Date : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")

    # Résumé exécutif
    lines.append("## Résumé exécutif\n")
    lines.append(f"- **Verdict** : **{verdict}**")
    for k, v in criteria.items():
        status = "OK" if v else "NOK"
        lines.append(f"  - {k} : {status}")
    lines.append(f"- Transitions détectées : {len(regime_signal.transitions)}")
    lines.append(f"- Leverage changes (Run C) : {len(result_c.leverage_changes)}\n")

    # Tableau comparatif
    lines.append("## Tableau comparatif\n")
    header = (
        "| Run                  | Return   | Max DD  | Sharpe | Calmar | Trades | WinR   |"
    )
    sep = "|" + "-" * 22 + "|" + "-" * 10 + "|" + "-" * 9 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 8 + "|"
    lines.append(header)
    lines.append(sep)
    lines.append(_format_table_row("A: Fixed 7x", result_a, sharpe_a, calmar_a))
    lines.append(_format_table_row("B: Fixed 4x", result_b, sharpe_b, calmar_b))
    lines.append(_format_table_row("C: Dynamic 7x/4x", result_c, sharpe_c, calmar_c))
    lines.append("")

    # Transitions
    lines.append("## Transitions de régime\n")
    if regime_signal.transitions:
        lines.append("| # | Timestamp | From | To |")
        lines.append("|---|-----------|------|----|")
        for i, t in enumerate(regime_signal.transitions, 1):
            lines.append(f"| {i} | {t['timestamp']} | {t['from']} | {t['to']} |")
    else:
        lines.append("Aucune transition détectée.\n")
    lines.append("")

    # Leverage changes
    if result_c.leverage_changes:
        lines.append("## Changements de leverage (Run C)\n")
        lines.append("| Timestamp | Runner | Old | New | Regime |")
        lines.append("|-----------|--------|-----|-----|--------|")
        for lc in result_c.leverage_changes[:50]:  # limiter l'affichage
            lines.append(
                f"| {lc['timestamp']} | {lc['runner']} | {lc['old']}x | {lc['new']}x | {lc['regime']} |"
            )
        if len(result_c.leverage_changes) > 50:
            lines.append(f"\n... et {len(result_c.leverage_changes) - 50} autres changements.\n")
        lines.append("")

    # Breakdown par régime
    lines.append("## Breakdown par régime (Run C)\n")
    if breakdown:
        lines.append("| Régime | Heures | Return | Max DD |")
        lines.append("|--------|--------|--------|--------|")
        for regime, data in breakdown.items():
            lines.append(
                f"| {regime} | {data['hours']} | {data['return_pct']:+.2f}% | {data['max_dd_pct']:.2f}% |"
            )
    lines.append("")

    return "\n".join(lines)


def _generate_plot(
    result_a: PortfolioResult,
    result_b: PortfolioResult,
    result_c: PortfolioResult,
    regime_signal: RegimeSignal,
    output_path: str,
) -> None:
    """Génère le plot equity curves + drawdown avec bandes régime."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib non disponible, plot ignoré")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # --- Equity curves ---
    for result, label, color in [
        (result_a, "A: Fixed 7x", "#2196F3"),
        (result_b, "B: Fixed 4x", "#9E9E9E"),
        (result_c, "C: Dynamic 7x/4x", "#4CAF50"),
    ]:
        ts = [s.timestamp for s in result.snapshots]
        eq = [s.total_equity for s in result.snapshots]
        ax1.plot(ts, eq, label=label, color=color, linewidth=1)

    # Bandes régime
    if regime_signal.timestamps and result_c.snapshots:
        ts_range = [s.timestamp for s in result_c.snapshots]
        for i in range(len(ts_range) - 1):
            regime = regime_signal.get_regime_at(ts_range[i])
            if regime == "defensive":
                ax1.axvspan(ts_range[i], ts_range[i + 1], alpha=0.08, color="red")

    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left")
    ax1.set_title("Equity Curves — Regime Impact Comparison")
    ax1.grid(True, alpha=0.3)

    # --- Drawdown curves ---
    for result, label, color in [
        (result_a, "A: Fixed 7x", "#2196F3"),
        (result_b, "B: Fixed 4x", "#9E9E9E"),
        (result_c, "C: Dynamic 7x/4x", "#4CAF50"),
    ]:
        ts = [s.timestamp for s in result.snapshots]
        eq = [s.total_equity for s in result.snapshots]
        peak = eq[0] if eq else 1
        dd = []
        for e in eq:
            if e > peak:
                peak = e
            dd.append((e / peak - 1) * 100 if peak > 0 else 0)
        ax2.plot(ts, dd, label=label, color=color, linewidth=1)

    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot sauvegardé : {output_path}")


# ─── Main ───────────────────────────────────────────────────────────────


async def main(args: argparse.Namespace) -> None:
    """Orchestre les 3 runs A/B/C et génère le rapport."""
    setup_logging(level="INFO")
    config = get_config()

    strategy = args.strategy
    exchange = args.exchange
    db_path = args.db

    # Kill switch
    ks_cfg = getattr(config.risk, "kill_switch", None)
    ks_pct: float = getattr(ks_cfg, "global_max_loss_pct", 45.0)
    ks_hours: int = int(getattr(ks_cfg, "global_window_hours", 24))

    # Dates
    end = datetime.now(timezone.utc)
    if args.days == "auto":
        from scripts.portfolio_backtest import _detect_max_days

        common_days, _ = await _detect_max_days(
            config, strategy, exchange, db_path
        )
        days = common_days
    else:
        days = int(args.days)
    start = end - timedelta(days=days)

    print(f"\n{'=' * 65}")
    print(f"  REGIME IMPACT COMPARISON — Sprint 50b")
    print(f"{'=' * 65}")
    print(f"  Stratégie           : {strategy}")
    print(f"  Période             : {days} jours")
    print(f"  Capital             : ${args.capital:,.0f}")
    print(f"  Exchange            : {exchange}")

    # 1. Calculer le regime signal
    print(f"\n  [1/4] Calcul du regime signal BTC...")
    regime_signal = await compute_regime_signal(
        db_path=db_path, start=start, end=end, exchange=exchange
    )
    n_trans = len(regime_signal.transitions)
    print(f"         {n_trans} transitions détectées")

    # 2. Run A : Fixed 7x
    print(f"\n  [2/4] Run A — Fixed {args.lev_normal}x...")
    t0 = time.monotonic()
    backtester_a = PortfolioBacktester(
        config=config,
        initial_capital=args.capital,
        strategy_name=strategy,
        exchange=exchange,
        kill_switch_pct=ks_pct,
        kill_switch_window_hours=ks_hours,
        leverage=args.lev_normal,
    )
    result_a = await backtester_a.run(start, end, db_path=db_path)
    print(f"         {result_a.total_return_pct:+.1f}% | DD {result_a.max_drawdown_pct:.2f}% | {time.monotonic() - t0:.0f}s")

    # 3. Run B : Fixed 4x
    print(f"\n  [3/4] Run B — Fixed {args.lev_defensive}x...")
    t0 = time.monotonic()
    backtester_b = PortfolioBacktester(
        config=config,
        initial_capital=args.capital,
        strategy_name=strategy,
        exchange=exchange,
        kill_switch_pct=ks_pct,
        kill_switch_window_hours=ks_hours,
        leverage=args.lev_defensive,
    )
    result_b = await backtester_b.run(start, end, db_path=db_path)
    print(f"         {result_b.total_return_pct:+.1f}% | DD {result_b.max_drawdown_pct:.2f}% | {time.monotonic() - t0:.0f}s")

    # 4. Run C : Dynamic leverage
    print(f"\n  [4/4] Run C — Dynamic {args.lev_normal}x/{args.lev_defensive}x...")
    t0 = time.monotonic()
    backtester_c = PortfolioBacktester(
        config=config,
        initial_capital=args.capital,
        strategy_name=strategy,
        exchange=exchange,
        kill_switch_pct=ks_pct,
        kill_switch_window_hours=ks_hours,
        regime_signal=regime_signal,
    )
    result_c = await backtester_c.run(start, end, db_path=db_path)
    print(f"         {result_c.total_return_pct:+.1f}% | DD {result_c.max_drawdown_pct:.2f}% | {time.monotonic() - t0:.0f}s")
    print(f"         {len(result_c.leverage_changes)} leverage changes")

    # 5. Métriques
    sharpe_a = _calc_sharpe(result_a.snapshots)
    sharpe_b = _calc_sharpe(result_b.snapshots)
    sharpe_c = _calc_sharpe(result_c.snapshots)
    calmar_a = _calc_calmar(result_a.total_return_pct, result_a.max_drawdown_pct, days)
    calmar_b = _calc_calmar(result_b.total_return_pct, result_b.max_drawdown_pct, days)
    calmar_c = _calc_calmar(result_c.total_return_pct, result_c.max_drawdown_pct, days)

    verdict, criteria = _compute_verdict(
        return_a=result_a.total_return_pct,
        return_c=result_c.total_return_pct,
        dd_a=result_a.max_drawdown_pct,
        dd_c=result_c.max_drawdown_pct,
        sharpe_a=sharpe_a,
        sharpe_c=sharpe_c,
    )

    # 6. Breakdown par régime
    breakdown = _regime_breakdown(result_c.snapshots, regime_signal)

    # 7. Affichage console
    print(f"\n{'=' * 65}")
    print(f"  VERDICT : {verdict}")
    print(f"{'=' * 65}")
    for k, v in criteria.items():
        status = "OK" if v else "NOK"
        print(f"    {k:15s} : {status}")
    print()

    header = f"  {'Run':20s} {'Return':>9s} {'MaxDD':>8s} {'Sharpe':>7s} {'Calmar':>7s}"
    print(header)
    print("  " + "-" * 55)
    for label, r, sh, ca in [
        ("A: Fixed 7x", result_a, sharpe_a, calmar_a),
        ("B: Fixed 4x", result_b, sharpe_b, calmar_b),
        ("C: Dynamic 7x/4x", result_c, sharpe_c, calmar_c),
    ]:
        print(
            f"  {label:20s} {r.total_return_pct:>+8.1f}% {r.max_drawdown_pct:>7.2f}% "
            f"{sh:>7.2f} {ca:>7.2f}"
        )
    print()

    # 8. Rapport Markdown
    report = _generate_report(
        result_a, result_b, result_c, regime_signal,
        sharpe_a, sharpe_b, sharpe_c,
        calmar_a, calmar_b, calmar_c,
        verdict, criteria, breakdown,
    )
    report_path = args.output or "docs/regime_impact_report.md"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(report, encoding="utf-8")
    print(f"  Rapport : {report_path}")

    # 9. Plot
    plot_path = args.plot or "docs/images/regime_equity_curves.png"
    _generate_plot(result_a, result_b, result_c, regime_signal, plot_path)
    print(f"  Plot    : {plot_path}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comparaison A/B/C — impact du leverage dynamique (régime BTC)"
    )
    parser.add_argument(
        "--strategy", type=str, default="grid_atr", help="Nom de la stratégie"
    )
    parser.add_argument(
        "--days",
        type=str,
        default="auto",
        help="Période de backtest en jours (défaut: 'auto')",
    )
    parser.add_argument(
        "--capital", type=float, default=10_000, help="Capital initial ($)"
    )
    parser.add_argument(
        "--exchange", type=str, default="binance", help="Source des candles"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/scalp_radar.db",
        help="Chemin de la base de données",
    )
    parser.add_argument(
        "--lev-normal", type=int, default=7, help="Leverage mode normal (défaut: 7)"
    )
    parser.add_argument(
        "--lev-defensive",
        type=int,
        default=4,
        help="Leverage mode defensive (défaut: 4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Chemin du rapport Markdown (défaut: docs/regime_impact_report.md)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Chemin du plot PNG (défaut: docs/images/regime_equity_curves.png)",
    )

    parsed_args = parser.parse_args()
    asyncio.run(main(parsed_args))
