"""Diagnostic : analyse du Max Drawdown à différents leverages.

Répond aux questions :
1. À quelle date le max DD se produit-il ? Même date pour 3x et 8x ?
2. Combien de positions étaient ouvertes au moment du pic DD ?
3. Quel était le margin ratio à ce moment ?
4. Le DD vient-il de SL hits ou d'unrealized losses ?

Usage :
    uv run python -m scripts.diagnose_dd_leverage
    uv run python -m scripts.diagnose_dd_leverage --strategy grid_boltrend
    uv run python -m scripts.diagnose_dd_leverage --leverages 3,6,8 --days 365
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

from backend.backtesting.portfolio_engine import PortfolioBacktester, PortfolioSnapshot
from backend.core.config import get_config
from backend.core.database import Database
from backend.core.logging_setup import setup_logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TroughDetail(NamedTuple):
    leverage: int
    total_return_pct: float
    max_dd_pct: float
    peak_date: datetime
    peak_equity: float
    trough_date: datetime
    trough_equity: float
    trough_capital: float       # cash libre à ce moment
    trough_unrealized: float    # P&L non réalisé à ce moment
    trough_realized: float      # P&L cumulé réalisé à ce moment
    trough_positions: int       # nb positions ouvertes
    trough_margin_ratio: float  # margin / initial_capital
    window_24h_realized_loss: float  # P&L réalisé des 24h précédant le trough (SL hits)


def _find_trough_details(
    snapshots: list[PortfolioSnapshot],
    initial_capital: float,
    leverage: int,
) -> TroughDetail | None:
    """Trouve le snapshot au max DD et extrait les métriques clés."""
    if not snapshots:
        return None

    peak = snapshots[0].total_equity
    peak_idx = 0
    max_dd = 0.0
    max_dd_idx = 0

    for i, snap in enumerate(snapshots):
        if snap.total_equity > peak:
            peak = snap.total_equity
            peak_idx = i
        dd = (snap.total_equity / peak - 1) * 100 if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
            max_dd_idx = i

    if max_dd == 0.0:
        print(f"  [leverage={leverage}x] Aucun drawdown détecté.")
        return None

    peak_snap = snapshots[peak_idx]
    trough_snap = snapshots[max_dd_idx]

    # P&L réalisé dans les 24h précédant le trough (proxy pour SL hits)
    trough_time = trough_snap.timestamp
    window_start = trough_time - timedelta(hours=24)
    snaps_24h = [s for s in snapshots if window_start <= s.timestamp <= trough_time]
    if len(snaps_24h) >= 2:
        realized_24h_delta = (
            snaps_24h[-1].total_realized_pnl - snaps_24h[0].total_realized_pnl
        )
    else:
        realized_24h_delta = 0.0

    total_return_pct = (
        (snapshots[-1].total_equity / snapshots[0].total_equity - 1) * 100
        if snapshots[0].total_equity > 0 else 0.0
    )

    return TroughDetail(
        leverage=leverage,
        total_return_pct=total_return_pct,
        max_dd_pct=max_dd,
        peak_date=peak_snap.timestamp,
        peak_equity=peak_snap.total_equity,
        trough_date=trough_snap.timestamp,
        trough_equity=trough_snap.total_equity,
        trough_capital=trough_snap.total_capital,
        trough_unrealized=trough_snap.total_unrealized_pnl,
        trough_realized=trough_snap.total_realized_pnl,
        trough_positions=trough_snap.n_open_positions,
        trough_margin_ratio=trough_snap.margin_ratio,
        window_24h_realized_loss=realized_24h_delta,
    )


def _print_detail(d: TroughDetail, initial_capital: float) -> None:
    """Affiche les détails du max DD pour un leverage donné."""
    print(f"\n{'='*60}")
    print(f"  LEVERAGE : {d.leverage}x")
    print(f"{'='*60}")
    print(f"  Return total         : {d.total_return_pct:+.1f}%")
    print(f"  Max DD               : {d.max_dd_pct:.1f}%")
    print(f"  Calmar               : {d.total_return_pct / abs(d.max_dd_pct):.2f}")
    print()
    print(f"  Peak equity          : {d.peak_equity:,.0f}$ ({d.peak_date.strftime('%Y-%m-%d')})")
    print(f"  Trough equity        : {d.trough_equity:,.0f}$ ({d.trough_date.strftime('%Y-%m-%d')})")
    print(f"  ↘ Durée peak→trough  : {(d.trough_date - d.peak_date).days}j")
    print()
    print(f"  Au moment du trough :")
    print(f"    Capital libre      : {d.trough_capital:,.0f}$")
    print(f"    Unrealized P&L     : {d.trough_unrealized:+,.0f}$")
    print(f"    Realized P&L (cum) : {d.trough_realized:+,.0f}$")
    equity_check = d.trough_capital + d.trough_unrealized
    missing_margin = d.trough_equity - equity_check
    if abs(missing_margin) > 1:
        print(f"    ⚠ Equity ≠ cap+unreal : delta={missing_margin:+,.0f}$ (marge manquante?)")
    print(f"    Positions ouvertes : {d.trough_positions}")
    print(f"    Margin ratio       : {d.trough_margin_ratio*100:.1f}%")
    print()
    # Diagnostic source du DD
    abs_dd = d.peak_equity - d.trough_equity
    unrealized_contrib = abs(d.trough_unrealized)
    capital_contrib = d.peak_equity - d.trough_capital - abs(d.trough_unrealized)
    print(f"  Source du DD (abs={abs_dd:,.0f}$) :")
    print(f"    Unrealized losses  : {unrealized_contrib:,.0f}$ ({unrealized_contrib/abs_dd*100:.0f}% du DD)")
    print(f"    Capital consommé   : {capital_contrib:,.0f}$ ({capital_contrib/abs_dd*100:.0f}% du DD)")
    print(f"    P&L réalisé 24h    : {d.window_24h_realized_loss:+,.0f}$ (proxy SL hits vs TP)")
    if d.window_24h_realized_loss < -abs_dd * 0.3:
        print(f"    → Probable SL hits massifs (réalisé 24h = {d.window_24h_realized_loss/abs_dd*100:.0f}% du DD)")
    else:
        print(f"    → Mainly unrealized losses (positions DCA toujours ouvertes)")


def _compare(details: list[TroughDetail]) -> None:
    """Compare les dates et métriques clés entre leverages."""
    if len(details) < 2:
        return

    print(f"\n{'='*60}")
    print("  COMPARAISON")
    print(f"{'='*60}")

    # Même date de trough ?
    dates = [d.trough_date.strftime('%Y-%m-%d') for d in details]
    if len(set(dates)) == 1:
        print(f"  ✓ Même date trough pour tous les leverages : {dates[0]}")
        print("    → Même événement de marché, comportement attendu")
    else:
        print("  ✗ Dates trough différentes :")
        for d in details:
            print(f"    {d.leverage}x → {d.trough_date.strftime('%Y-%m-%d')}")

    # DD% et return% vs leverage
    print()
    print(f"  {'Lev':>4}  {'Return%':>9}  {'DD%':>7}  {'Calmar':>7}  {'Positions':>9}  {'MarginR%':>9}")
    print(f"  {'-'*4}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*9}")
    for d in details:
        calmar = d.total_return_pct / abs(d.max_dd_pct) if d.max_dd_pct != 0 else 999
        print(
            f"  {d.leverage:>3}x  {d.total_return_pct:>+8.1f}%  {d.max_dd_pct:>6.1f}%"
            f"  {calmar:>7.2f}  {d.trough_positions:>9d}  {d.trough_margin_ratio*100:>8.1f}%"
        )

    # Scaling théorique
    base = details[0]
    print()
    print(f"  Scaling théorique vs {base.leverage}x :")
    for d in details[1:]:
        expected_return = base.total_return_pct * d.leverage / base.leverage
        expected_dd = base.max_dd_pct * d.leverage / base.leverage
        print(f"    {d.leverage}x return prédit={expected_return:+.0f}% réel={d.total_return_pct:+.0f}%  "
              f"DD prédit={expected_dd:.1f}% réel={d.max_dd_pct:.1f}%")

    print()
    print("  NOTE : Si DD% est quasi-identique à tous les leverages,")
    print("  c'est mathématiquement attendu (invariance % si P&L ∝ leverage).")
    print("  Le metric pertinent est le Calmar (return/DD).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _main(
    strategy: str,
    leverages: list[int],
    days: int,
    db_path: str,
    capital: float,
    exchange: str,
) -> None:
    config = get_config()
    db = Database(db_path)
    await db.connect()

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    print(f"\nDiagnostic DD leverage — {strategy}")
    print(f"Période : {start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')} ({days}j)")
    print(f"Capital initial : {capital:,.0f}$")
    print(f"Leverages testés : {leverages}")

    results: list[TroughDetail] = []

    for lev in leverages:
        print(f"\n[{lev}x] Lancement backtest…", flush=True)
        t0 = asyncio.get_event_loop().time()

        # Override leverage dans la config
        strat_cfg = getattr(config.strategies, strategy, None)
        if strat_cfg is not None and hasattr(strat_cfg, "leverage"):
            strat_cfg.leverage = lev

        # Override KS à 99% pour voir le vrai max DD
        ks_cfg = getattr(config.risk, "kill_switch", None)
        if ks_cfg is not None:
            for attr in ("grid_max_session_loss_percent", "max_session_loss_percent"):
                if hasattr(ks_cfg, attr):
                    setattr(ks_cfg, attr, 99.0)

        backtester = PortfolioBacktester(
            config=config,
            initial_capital=capital,
            strategy_name=strategy,
            exchange=exchange,
            kill_switch_pct=99.0,
            leverage=lev,
        )

        result = await backtester.run(start=start_dt, end=end_dt, db_path=db_path)
        elapsed = asyncio.get_event_loop().time() - t0

        print(f"  Done en {elapsed:.0f}s — {result.total_trades} trades, return={result.total_return_pct:+.1f}%")

        detail = _find_trough_details(result.snapshots, capital, lev)
        if detail:
            results.append(detail)
            _print_detail(detail, capital)

    if len(results) >= 2:
        _compare(results)

    await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="grid_atr", help="Nom de la stratégie")
    parser.add_argument("--leverages", default="3,6,8", help="Leverages à tester (ex: 3,6,8)")
    parser.add_argument("--days", type=int, default=365, help="Nb de jours de backtest")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Capital initial ($)")
    parser.add_argument("--exchange", default="binance", help="Exchange source des données")
    parser.add_argument("--db", default="data/candles.db", help="Chemin vers la base de données")
    parser.add_argument("--verbose", action="store_true", help="Logs détaillés")
    args = parser.parse_args()

    setup_logging(level="DEBUG" if args.verbose else "WARNING")

    try:
        leverages = [int(x.strip()) for x in args.leverages.split(",") if x.strip()]
    except ValueError:
        print(f"Format leverages invalide : '{args.leverages}'")
        sys.exit(1)

    asyncio.run(_main(
        strategy=args.strategy,
        leverages=leverages,
        days=args.days,
        db_path=args.db,
        capital=args.capital,
        exchange=args.exchange,
    ))
