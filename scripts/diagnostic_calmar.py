"""Diagnostic Calmar anormal — grid_multi_tf vs grid_atr.

Analyse l'equity curve, l'activité, les fees, et le crash d'août 2024
pour vérifier le réalisme du Calmar 22.87 affiché pour grid_multi_tf.

Usage :
    uv run python -m scripts.diagnostic_calmar
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from backend.backtesting.portfolio_engine import PortfolioBacktester, PortfolioResult, PortfolioSnapshot
from backend.core.config import get_config
from backend.core.logging_setup import setup_logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def daily_returns(snapshots: list[PortfolioSnapshot]) -> list[dict]:
    """Agrège les snapshots en returns journaliers."""
    if len(snapshots) < 2:
        return []

    # Grouper par jour
    daily: dict[str, list[PortfolioSnapshot]] = defaultdict(list)
    for s in snapshots:
        day = s.timestamp.strftime("%Y-%m-%d")
        daily[day].append(s)

    days_sorted = sorted(daily.keys())
    results = []
    for i in range(1, len(days_sorted)):
        prev_day = days_sorted[i - 1]
        curr_day = days_sorted[i]
        prev_eq = daily[prev_day][-1].total_equity
        curr_eq = daily[curr_day][-1].total_equity
        if prev_eq > 0:
            ret = (curr_eq / prev_eq - 1) * 100
        else:
            ret = 0.0
        results.append({
            "date": curr_day,
            "equity": curr_eq,
            "return_pct": ret,
            "n_positions": daily[curr_day][-1].n_open_positions,
            "n_assets": daily[curr_day][-1].n_assets_with_positions,
            "margin_ratio": daily[curr_day][-1].margin_ratio,
        })
    return results


def equity_stats(daily_rets: list[dict]) -> dict:
    """Calcule les stats de l'equity curve journalière."""
    if not daily_rets:
        return {}

    returns = [d["return_pct"] for d in daily_rets]
    n = len(returns)
    loss_days = sum(1 for r in returns if r < 0)
    gain_days = sum(1 for r in returns if r > 0)
    flat_days = n - loss_days - gain_days

    # Plus grosse perte journalière
    worst_day = min(returns)
    worst_date = daily_rets[returns.index(worst_day)]["date"]

    # Plus gros gain
    best_day = max(returns)
    best_date = daily_rets[returns.index(best_day)]["date"]

    # Plus longue série de jours consécutifs en perte
    max_losing_streak = 0
    current_streak = 0
    for r in returns:
        if r < 0:
            current_streak += 1
            max_losing_streak = max(max_losing_streak, current_streak)
        else:
            current_streak = 0

    # Écart-type des returns
    mean_r = sum(returns) / n
    variance = sum((r - mean_r) ** 2 for r in returns) / n
    std_r = math.sqrt(variance) if variance > 0 else 0.0

    # Max drawdown journalier
    peak = daily_rets[0]["equity"]
    max_dd = 0.0
    max_dd_date = ""
    for d in daily_rets:
        if d["equity"] > peak:
            peak = d["equity"]
        dd = (d["equity"] / peak - 1) * 100 if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
            max_dd_date = d["date"]

    return {
        "total_days": n,
        "loss_days": loss_days,
        "gain_days": gain_days,
        "flat_days": flat_days,
        "worst_day_pct": round(worst_day, 3),
        "worst_day_date": worst_date,
        "best_day_pct": round(best_day, 3),
        "best_day_date": best_date,
        "max_losing_streak": max_losing_streak,
        "std_daily_return": round(std_r, 4),
        "mean_daily_return": round(mean_r, 4),
        "max_drawdown_pct": round(max_dd, 2),
        "max_drawdown_date": max_dd_date,
    }


def trades_by_month(trades: list[tuple[str, Any]]) -> dict[str, dict]:
    """Agrège les trades par mois."""
    monthly: dict[str, list] = defaultdict(list)
    for runner_key, trade in trades:
        month = trade.exit_time.strftime("%Y-%m")
        monthly[month].append((runner_key, trade))

    result = {}
    for month in sorted(monthly.keys()):
        month_trades = monthly[month]
        pnl = sum(t.net_pnl for _, t in month_trades)
        wins = sum(1 for _, t in month_trades if t.net_pnl > 0)
        losses = sum(1 for _, t in month_trades if t.net_pnl <= 0)
        assets = set(rk.split(":")[-1] for rk, _ in month_trades)
        result[month] = {
            "trades": len(month_trades),
            "pnl": round(pnl, 2),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / len(month_trades) * 100, 1) if month_trades else 0,
            "active_assets": sorted(assets),
            "n_assets": len(assets),
        }
    return result


def crash_aug_2024(snapshots: list[PortfolioSnapshot], trades: list[tuple[str, Any]]) -> dict:
    """Analyse la période du crash d'août 2024."""
    start = datetime(2024, 8, 1, tzinfo=timezone.utc)
    end = datetime(2024, 8, 15, tzinfo=timezone.utc)

    # Snapshots pendant le crash
    crash_snaps = [s for s in snapshots if start <= s.timestamp <= end]
    if not crash_snaps:
        return {"error": "Pas de snapshots entre 2024-08-01 et 2024-08-15"}

    # Max DD pendant la période
    peak = crash_snaps[0].total_equity
    max_dd = 0.0
    max_dd_ts = ""
    peak_positions = 0
    for s in crash_snaps:
        if s.total_equity > peak:
            peak = s.total_equity
        dd = (s.total_equity / peak - 1) * 100 if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
            max_dd_ts = s.timestamp.isoformat()
            peak_positions = s.n_open_positions

    # Positions au début et au pic du crash
    first_snap = crash_snaps[0]
    # Le crash de BTC était autour du 5 août
    crash_peak = datetime(2024, 8, 5, tzinfo=timezone.utc)
    closest = min(crash_snaps, key=lambda s: abs((s.timestamp - crash_peak).total_seconds()))

    # Trades pendant le crash
    crash_trades = [
        (rk, t) for rk, t in trades
        if start <= t.exit_time <= end
    ]
    crash_pnl = sum(t.net_pnl for _, t in crash_trades)

    # Trades AVANT le crash (supertrend flip pre-emptif?)
    pre_crash_start = datetime(2024, 7, 28, tzinfo=timezone.utc)
    pre_crash_trades = [
        (rk, t) for rk, t in trades
        if pre_crash_start <= t.exit_time < start
    ]

    return {
        "period": f"{start.date()} -> {end.date()}",
        "n_snapshots": len(crash_snaps),
        "max_dd_pct": round(max_dd, 2),
        "max_dd_timestamp": max_dd_ts,
        "positions_at_max_dd": peak_positions,
        "equity_start": round(first_snap.total_equity, 2),
        "equity_at_crash_peak": round(closest.total_equity, 2),
        "positions_start": first_snap.n_open_positions,
        "positions_at_crash_peak": closest.n_open_positions,
        "assets_at_crash_peak": closest.n_assets_with_positions,
        "crash_trades": len(crash_trades),
        "crash_pnl": round(crash_pnl, 2),
        "pre_crash_trades_jul28_aug01": len(pre_crash_trades),
        "pre_crash_pnl": round(sum(t.net_pnl for _, t in pre_crash_trades), 2),
    }


def concurrent_positions_stats(snapshots: list[PortfolioSnapshot]) -> dict:
    """Stats sur les positions simultanées."""
    if not snapshots:
        return {}

    positions = [s.n_open_positions for s in snapshots]
    assets = [s.n_assets_with_positions for s in snapshots]

    return {
        "avg_positions": round(sum(positions) / len(positions), 1),
        "max_positions": max(positions),
        "avg_active_assets": round(sum(assets) / len(assets), 1),
        "max_active_assets": max(assets),
        "pct_time_no_positions": round(sum(1 for p in positions if p == 0) / len(positions) * 100, 1),
    }


def calmar_annualized(total_return_pct: float, max_dd_pct: float, n_days: int) -> float:
    """Calmar annualisé standard."""
    if max_dd_pct == 0 or n_days == 0:
        return float("inf") if total_return_pct > 0 else 0.0
    n_years = n_days / 365.25
    annual_return = total_return_pct / n_years
    return round(annual_return / abs(max_dd_pct), 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_backtest(strategy_name: str, days: int, capital: float, exchange: str, db_path: str) -> PortfolioResult:
    """Lance un portfolio backtest pour une stratégie."""
    config = get_config()

    # Override KS à 99% pour voir le vrai DD
    ks_cfg = getattr(config.risk, "kill_switch", None)
    if ks_cfg is not None:
        if hasattr(ks_cfg, "grid_max_session_loss_percent"):
            ks_cfg.grid_max_session_loss_percent = 99.0
        if hasattr(ks_cfg, "max_session_loss_percent"):
            ks_cfg.max_session_loss_percent = 99.0

    backtester = PortfolioBacktester(
        config=config,
        initial_capital=capital,
        strategy_name=strategy_name,
        exchange=exchange,
        kill_switch_pct=99.0,
        kill_switch_window_hours=24,
    )

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    return await backtester.run(start=start_dt, end=end_dt, db_path=db_path)


async def run_single_asset_backtest(
    strategy_name: str, symbol: str, days: int, capital: float,
    exchange: str, db_path: str,
) -> PortfolioResult:
    """Lance un portfolio backtest sur un seul asset."""
    config = get_config()

    ks_cfg = getattr(config.risk, "kill_switch", None)
    if ks_cfg is not None:
        if hasattr(ks_cfg, "grid_max_session_loss_percent"):
            ks_cfg.grid_max_session_loss_percent = 99.0
        if hasattr(ks_cfg, "max_session_loss_percent"):
            ks_cfg.max_session_loss_percent = 99.0

    backtester = PortfolioBacktester(
        config=config,
        initial_capital=capital,
        strategy_name=strategy_name,
        assets=[symbol],
        exchange=exchange,
        kill_switch_pct=99.0,
        kill_switch_window_hours=24,
    )

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    return await backtester.run(start=start_dt, end=end_dt, db_path=db_path)


def print_separator(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


async def main() -> None:
    setup_logging(level="WARNING")

    DAYS = 730
    CAPITAL = 1000.0
    EXCHANGE = "binance"
    DB_PATH = "data/scalp_radar.db"
    SINGLE_ASSET = "DYDX/USDT"

    strategies = ["grid_multi_tf", "grid_atr"]
    results: dict[str, PortfolioResult] = {}
    single_results: dict[str, PortfolioResult] = {}

    # ══════════════════════════════════════════════════════════════
    # 1. Lancer les portfolio backtests
    # ══════════════════════════════════════════════════════════════
    for strat in strategies:
        print(f"\n>>> Lancement portfolio backtest : {strat} ({DAYS}j, {CAPITAL}$)...")
        t0 = time.monotonic()
        try:
            results[strat] = await run_backtest(strat, DAYS, CAPITAL, EXCHANGE, DB_PATH)
            elapsed = time.monotonic() - t0
            print(f"    OK en {elapsed:.0f}s — {results[strat].total_trades} trades")
        except Exception as e:
            print(f"    ERREUR : {e}")

    # ══════════════════════════════════════════════════════════════
    # 2. Single asset comparison (DYDX/USDT)
    # ══════════════════════════════════════════════════════════════
    for strat in strategies:
        if strat not in results:
            continue
        # Vérifier que l'asset est dans les per_asset
        if SINGLE_ASSET.replace("/", "\\") not in [a.replace("/", "\\") for a in results[strat].assets]:
            # Tenter quand même avec le format brut
            pass
        print(f"\n>>> Single asset {SINGLE_ASSET} : {strat}...")
        t0 = time.monotonic()
        try:
            single_results[strat] = await run_single_asset_backtest(
                strat, SINGLE_ASSET, DAYS, CAPITAL, EXCHANGE, DB_PATH,
            )
            elapsed = time.monotonic() - t0
            print(f"    OK en {elapsed:.0f}s — {single_results[strat].total_trades} trades")
        except Exception as e:
            print(f"    ERREUR : {e}")

    # ══════════════════════════════════════════════════════════════
    # RAPPORT DIAGNOSTIC
    # ══════════════════════════════════════════════════════════════

    print_separator("DIAGNOSTIC CALMAR — grid_multi_tf vs grid_atr")
    print(f"  Période : {DAYS} jours | Capital : {CAPITAL}$ | Exchange : {EXCHANGE}")

    # ── POINT 1 : Formule Calmar ──────────────────────────────────
    print_separator("POINT 1 — Formule Calmar")
    for strat in strategies:
        if strat not in results:
            continue
        r = results[strat]
        brut = r.total_return_pct / abs(r.max_drawdown_pct) if r.max_drawdown_pct != 0 else float("inf")
        annualized = calmar_annualized(r.total_return_pct, r.max_drawdown_pct, r.period_days)
        n_years = r.period_days / 365.25
        annual_ret = r.total_return_pct / n_years
        print(f"\n  {strat}:")
        print(f"    Total return     : {r.total_return_pct:+.1f}%")
        print(f"    Max DD           : {r.max_drawdown_pct:.1f}%")
        print(f"    Période          : {r.period_days}j ({n_years:.2f} ans)")
        print(f"    Calmar BRUT      : {brut:.2f}  (return_total / |DD|) <- ACTUEL")
        print(f"    Return annualisé : {annual_ret:+.1f}%")
        print(f"    Calmar ANNUALISÉ : {annualized:.2f}  (return_annuel / |DD|) <- STANDARD")

    # ── POINT 2 : Equity curve ────────────────────────────────────
    print_separator("POINT 2 — Equity Curve (stats journalières)")
    for strat in strategies:
        if strat not in results:
            continue
        r = results[strat]
        dr = daily_returns(r.snapshots)
        stats = equity_stats(dr)
        print(f"\n  {strat} ({r.total_trades} trades, {len(r.assets)} assets):")
        for k, v in stats.items():
            print(f"    {k:25s} : {v}")

    # ── POINT 3 : Activité réelle ─────────────────────────────────
    print_separator("POINT 3 — Activité réelle (trades/mois, assets simultanés)")
    for strat in strategies:
        if strat not in results:
            continue
        r = results[strat]
        monthly = trades_by_month(r.all_trades)
        conc = concurrent_positions_stats(r.snapshots)

        print(f"\n  {strat}:")
        print(f"    Positions simultanées:")
        for k, v in conc.items():
            print(f"      {k:25s} : {v}")

        print(f"\n    Trades par mois:")
        print(f"    {'Mois':10s} {'Trades':>7s} {'PnL':>10s} {'WR':>6s} {'Assets':>7s}  Assets actifs")
        months_with_zero = 0
        for month, data in monthly.items():
            print(
                f"    {month:10s} {data['trades']:7d} {data['pnl']:+10.2f} "
                f"{data['win_rate']:5.1f}% {data['n_assets']:7d}  {', '.join(data['active_assets'][:5])}"
            )
        # Vérifier les mois manquants
        if monthly:
            first_month = min(monthly.keys())
            last_month = max(monthly.keys())
            # Générer tous les mois entre first et last
            y, m = int(first_month[:4]), int(first_month[5:7])
            ey, em = int(last_month[:4]), int(last_month[5:7])
            all_months = []
            while (y, m) <= (ey, em):
                all_months.append(f"{y:04d}-{m:02d}")
                m += 1
                if m > 12:
                    m = 1
                    y += 1
            missing = [mo for mo in all_months if mo not in monthly]
            if missing:
                print(f"\n    MOIS SANS TRADES : {', '.join(missing)}")
                months_with_zero = len(missing)
            else:
                print(f"\n    Tous les mois ont des trades (aucun mois à 0)")

    # ── POINT 4 : Crash août 2024 ────────────────────────────────
    print_separator("POINT 4 — Performance pendant le crash d'août 2024")
    for strat in strategies:
        if strat not in results:
            continue
        r = results[strat]
        crash = crash_aug_2024(r.snapshots, r.all_trades)
        print(f"\n  {strat}:")
        for k, v in crash.items():
            print(f"    {k:35s} : {v}")

    # ── POINT 5 : Fees ────────────────────────────────────────────
    print_separator("POINT 5 — Comparaison Fees")
    config = get_config()
    print(f"\n  Config risk.yaml (partagée) :")
    print(f"    taker_fee   : {config.risk.fees.taker_percent}% ({config.risk.fees.taker_percent/100:.4f})")
    print(f"    maker_fee   : {config.risk.fees.maker_percent}% ({config.risk.fees.maker_percent/100:.4f})")
    print(f"    slippage    : {config.risk.slippage.default_estimate_percent}% ({config.risk.slippage.default_estimate_percent/100:.5f})")
    print(f"    high_vol_x  : {config.risk.slippage.high_volatility_multiplier}x")
    print(f"\n  -> Les fees sont IDENTIQUES entre grid_atr et grid_multi_tf (même config)")

    for strat in strategies:
        if strat not in results:
            continue
        r = results[strat]
        # Estimer le coût total des fees à partir des trades
        total_gross = 0.0
        total_net = 0.0
        total_funding = r.funding_paid_total
        for _, t in r.all_trades:
            total_gross += t.gross_pnl if hasattr(t, "gross_pnl") else t.net_pnl
            total_net += t.net_pnl
        fee_cost = total_gross - total_net
        print(f"\n  {strat}:")
        print(f"    Total trades       : {r.total_trades}")
        print(f"    Gross PnL          : {total_gross:+.2f}$")
        print(f"    Net PnL            : {total_net:+.2f}$")
        print(f"    Fee+slippage cost  : {fee_cost:.2f}$")
        print(f"    Funding cost       : {total_funding:.2f}$")
        if r.total_trades > 0:
            print(f"    Avg fee/trade      : {fee_cost/r.total_trades:.4f}$")

    # ── POINT 6 : Single asset comparison ─────────────────────────
    print_separator("POINT 6 — Comparaison single asset (DYDX/USDT)")

    for strat in strategies:
        if strat not in single_results:
            continue
        r = single_results[strat]
        n_years = r.period_days / 365.25
        print(f"\n  {strat} — {SINGLE_ASSET} seul ({r.period_days}j):")
        print(f"    Trades         : {r.total_trades}")
        print(f"    Return         : {r.total_return_pct:+.1f}%")
        print(f"    Max DD         : {r.max_drawdown_pct:.1f}%")
        calmar_a = calmar_annualized(r.total_return_pct, r.max_drawdown_pct, r.period_days)
        print(f"    Calmar annuel  : {calmar_a:.2f}")
        print(f"    Win rate       : {r.win_rate:.1f}%")
        print(f"    Final equity   : {r.final_equity:.2f}$")

    # Note sur le WFO engine
    print_separator("POINT 6b — Note architecture moteurs")
    print("""
  Le WFO (walk_forward.py) utilise BacktestEngine (engine.py) pour les grid strategies.
  Le portfolio backtest utilise GridStrategyRunner (simulator.py).
  Ce sont DEUX MOTEURS DIFFÉRENTS :
    - BacktestEngine : event-driven, compute_indicators(), evaluate(), check_exit()
    - GridStrategyRunner : simulateur prod, on_candle(), grid_position_manager

  IMPLICATION : les résultats WFO (Calmar, Sharpe) peuvent diverger du portfolio backtest
  si les deux moteurs n'ont pas exactement le même comportement de TP/SL/fees.
  Le single-asset comparison ci-dessus compare seulement le portfolio engine
  entre stratégies (même moteur). Pour comparer les deux moteurs, il faudrait
  un WFO backtest sur le même asset avec les mêmes params.
""")

    # ── RÉSUMÉ ────────────────────────────────────────────────────
    print_separator("RÉSUMÉ")
    if "grid_multi_tf" in results and "grid_atr" in results:
        mtf = results["grid_multi_tf"]
        atr = results["grid_atr"]
        mtf_calmar = calmar_annualized(mtf.total_return_pct, mtf.max_drawdown_pct, mtf.period_days)
        atr_calmar = calmar_annualized(atr.total_return_pct, atr.max_drawdown_pct, atr.period_days)

        print(f"\n  {'Métrique':30s} {'grid_multi_tf':>15s} {'grid_atr':>15s}")
        print(f"  {'-'*60}")
        print(f"  {'Return total':30s} {mtf.total_return_pct:+14.1f}% {atr.total_return_pct:+14.1f}%")
        print(f"  {'Max DD':30s} {mtf.max_drawdown_pct:14.1f}% {atr.max_drawdown_pct:14.1f}%")
        print(f"  {'Calmar brut':30s} {mtf.total_return_pct/abs(mtf.max_drawdown_pct) if mtf.max_drawdown_pct else 0:14.2f} {atr.total_return_pct/abs(atr.max_drawdown_pct) if atr.max_drawdown_pct else 0:14.2f}")
        print(f"  {'Calmar ANNUALISÉ':30s} {mtf_calmar:14.2f} {atr_calmar:14.2f}")
        print(f"  {'Trades':30s} {mtf.total_trades:14d} {atr.total_trades:14d}")
        print(f"  {'Win rate':30s} {mtf.win_rate:13.1f}% {atr.win_rate:13.1f}%")
        print(f"  {'Assets':30s} {mtf.n_assets:14d} {atr.n_assets:14d}")
        print(f"  {'Peak positions':30s} {mtf.peak_open_positions:14d} {atr.peak_open_positions:14d}")
        print(f"  {'Leverage':30s} {mtf.leverage:14d} {atr.leverage:14d}")
        print(f"  {'Funding cost':30s} {mtf.funding_paid_total:+13.2f}$ {atr.funding_paid_total:+13.2f}$")

    print("\n  FIN DU DIAGNOSTIC\n")


if __name__ == "__main__":
    asyncio.run(main())
