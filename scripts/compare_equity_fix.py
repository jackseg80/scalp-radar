"""Compare l'impact du fix equity = capital + margin + unrealized.

Lance UN SEUL backtest et calcule les deux formules en post-processing
depuis les snapshots (qui contiennent capital, margin et unrealized séparément).

Usage :
    uv run python -m scripts.compare_equity_fix
    uv run python -m scripts.compare_equity_fix --leverage 7 --days 365
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone

from backend.backtesting.portfolio_engine import PortfolioBacktester, PortfolioSnapshot
from backend.core.config import get_config
from backend.core.logging_setup import setup_logging


def _dd_from_equity(equity_list: list[float]) -> tuple[float, int]:
    """Max DD% et index du trough depuis une courbe equity."""
    if not equity_list:
        return 0.0, 0
    peak = equity_list[0]
    max_dd = 0.0
    trough_idx = 0
    for i, e in enumerate(equity_list):
        if e > peak:
            peak = e
        dd = (e / peak - 1) * 100 if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
            trough_idx = i
    return max_dd, trough_idx


async def _run(strategy: str, leverage: int, days: int, db_path: str) -> None:
    config = get_config()

    # Override leverage
    strat_cfg = getattr(config.strategies, strategy, None)
    if strat_cfg and hasattr(strat_cfg, "leverage"):
        strat_cfg.leverage = leverage

    # Désactiver KS pour voir le vrai max DD
    ks_cfg = getattr(config.risk, "kill_switch", None)
    if ks_cfg:
        for attr in ("grid_max_session_loss_percent", "max_session_loss_percent"):
            if hasattr(ks_cfg, attr):
                setattr(ks_cfg, attr, 99.0)

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    print(f"\n{'='*65}")
    print(f"  Compare equity fix — {strategy} {leverage}x — {days}j")
    print(f"  Periode : {start_dt.strftime('%Y-%m-%d')} -> {end_dt.strftime('%Y-%m-%d')}")
    print(f"{'='*65}")

    backtester = PortfolioBacktester(
        config=config,
        strategy_name=strategy,
        exchange="binance",
        kill_switch_pct=99.0,
        leverage=leverage,
    )

    result = await backtester.run(start=start_dt, end=end_dt, db_path=db_path)
    snaps: list[PortfolioSnapshot] = result.snapshots
    initial = backtester._initial_capital

    if not snaps:
        print("Aucun snapshot — données insuffisantes.")
        return

    print(f"  Snapshots : {len(snaps)}  |  Trades : {result.total_trades}")
    print(f"  Return total (après force-close) : {result.total_return_pct:+.1f}%")
    print()

    # ── Reconstruire les deux courbes d'equity ───────────────────────────
    # ANCIENNE formule : equity = capital + unrealized (sans margin)
    old_equity = [
        s.total_capital + s.total_unrealized_pnl
        for s in snaps
    ]
    # NOUVELLE formule : equity = capital + margin + unrealized (FIX)
    # = total_equity tel que maintenant calculé dans _take_snapshot
    new_equity = [s.total_equity for s in snaps]

    # Vérification : new_equity devrait == total_capital + total_margin + total_unrealized
    # On vérifie aussi : equity_correct = initial + realized + unrealized
    check_equity = [
        initial + s.total_realized_pnl + s.total_unrealized_pnl
        for s in snaps
    ]

    # ── Métriques ────────────────────────────────────────────────────────
    old_dd, old_trough_idx = _dd_from_equity(old_equity)
    new_dd, new_trough_idx = _dd_from_equity(new_equity)
    chk_dd, _ = _dd_from_equity(check_equity)

    old_trough_snap = snaps[old_trough_idx]
    new_trough_snap = snaps[new_trough_idx]

    print(f"  {'Métrique':<35}  {'AVANT fix':>12}  {'APRÈS fix':>12}  {'Vérif.':>10}")
    print(f"  {'-'*35}  {'-'*12}  {'-'*12}  {'-'*10}")

    def pct(v: float) -> str:
        return f"{v:+.1f}%"

    print(f"  {'Max Drawdown':<35}  {pct(old_dd):>12}  {pct(new_dd):>12}  {pct(chk_dd):>10}")

    old_peak = max(old_equity)
    new_peak = max(new_equity)
    print(f"  {'Peak equity ($)':<35}  {old_peak:>12,.0f}  {new_peak:>12,.0f}")
    print(f"  {'Trough equity ($)':<35}  {old_equity[old_trough_idx]:>12,.0f}  {new_equity[new_trough_idx]:>12,.0f}")

    print()
    print(f"  Details au trough (AVANT fix -- {old_trough_snap.timestamp.strftime('%Y-%m-%d')}):")
    m1 = old_trough_snap.total_capital + old_trough_snap.total_unrealized_pnl
    print(f"    Capital libre    : {old_trough_snap.total_capital:,.0f}$")
    print(f"    Marge verrouilee : {old_trough_snap.total_margin_used:,.0f}$  <-- MANQUAIT dans l'equity")
    print(f"    Unrealized PnL   : {old_trough_snap.total_unrealized_pnl:+,.0f}$")
    print(f"    Equity AVANT     : {m1:,.0f}$  (capital + unrealized)")
    print(f"    Equity APRES     : {old_trough_snap.total_equity:,.0f}$  (+ margin)")
    print(f"    Positions        : {old_trough_snap.n_open_positions}")
    print(f"    Margin ratio     : {old_trough_snap.margin_ratio*100:.1f}%")

    if new_trough_idx != old_trough_idx:
        print()
        print(f"  Details au trough (APRES fix -- {new_trough_snap.timestamp.strftime('%Y-%m-%d')}):")
        print(f"    Capital libre    : {new_trough_snap.total_capital:,.0f}$")
        print(f"    Marge verrouilee : {new_trough_snap.total_margin_used:,.0f}$")
        print(f"    Unrealized PnL   : {new_trough_snap.total_unrealized_pnl:+,.0f}$")
        print(f"    Equity APRES     : {new_trough_snap.total_equity:,.0f}$")
        print(f"    Positions        : {new_trough_snap.n_open_positions}")

    # -- Invariant verification -----------------------------------------------
    print()
    print("  Verification invariant equity = initial + realized + unrealized :")
    max_delta = max(abs(new_equity[i] - check_equity[i]) for i in range(len(snaps)))
    if max_delta < 0.01:
        print(f"    OK  Max ecart = {max_delta:.6f}$ (invariant respecte)")
    else:
        print(f"    FAIL Max ecart = {max_delta:.2f}$ (BUG residuel!)")

    # -- Resume de l'impact ---------------------------------------------------
    print()
    print("  Resume de l'impact du fix :")
    dd_reduction = old_dd - new_dd
    print(f"    DD reduit de {abs(dd_reduction):.1f}pp ({old_dd:.1f}% -> {new_dd:.1f}%)")
    calmar_old = result.total_return_pct / abs(old_dd) if old_dd != 0 else 999
    calmar_new = result.total_return_pct / abs(new_dd) if new_dd != 0 else 999
    print(f"    Calmar : {calmar_old:.2f} -> {calmar_new:.2f}")

    # Margin au trough (quantifie l'erreur)
    margin_at_old_trough = old_trough_snap.total_margin_used
    print(f"    Marge verrouilee au trough : {margin_at_old_trough:,.0f}$ "
          f"({margin_at_old_trough/backtester._initial_capital*100:.0f}% du capital initial)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="grid_atr")
    parser.add_argument("--leverage", type=int, default=7)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--db", default="data/candles.db")
    args = parser.parse_args()

    setup_logging(level="WARNING")
    asyncio.run(_run(args.strategy, args.leverage, args.days, args.db))
