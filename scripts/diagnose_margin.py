"""Diagnostic de marge — portfolio backtest multi-asset.

Analyse la starvation de marge : combien de fois le margin guard (70%) bloque
des ouvertures de positions, par asset et par type de guard (local vs global).

Usage:
  uv run python -m scripts.diagnose_margin
  uv run python -m scripts.diagnose_margin --strategy grid_atr --days 365 --leverage 7

Rapport produit :
  1. Trades skippés par margin guard par asset (local vs global)
  2. Margin utilization timeline (échantillon 24h)
  3. Comparaison 6x vs levier cible (métriques marge + skip counts)
  4. Ordre d'allocation des runners (avantage premier itéré)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.backtesting.portfolio_engine import (
    PortfolioBacktester,
    PortfolioResult,
    PortfolioSnapshot,
)
from backend.backtesting.simulator import GridStrategyRunner
from backend.core.config import AppConfig, get_config

DB_PATH = str(ROOT / "data" / "scalp_radar.db")


# ---------------------------------------------------------------------------
# Compteurs de diagnostic
# ---------------------------------------------------------------------------


class DiagCounters:
    """Compteurs de rejet du margin guard et d'ouverture réussie, par asset."""

    def __init__(self) -> None:
        self.skip_local: dict[str, int] = defaultdict(int)
        self.skip_global: dict[str, int] = defaultdict(int)
        self.success: dict[str, int] = defaultdict(int)
        self.runner_order: list[str] = []

    def reset(self) -> None:
        self.skip_local.clear()
        self.skip_global.clear()
        self.success.clear()
        self.runner_order.clear()

    def attach(self, runners: dict[str, GridStrategyRunner]) -> None:
        """Configure les hooks de diagnostic sur chaque runner.

        Chaque runner reçoit 3 callables :
          _on_skip_local(symbol)    — appelé quand le local margin guard bloque
          _on_skip_global(symbol)   — appelé quand le global margin guard bloque
          _on_position_opened(symbol) — appelé quand une position est ouverte
        """
        self.runner_order = list(runners.keys())
        sl = self.skip_local
        sg = self.skip_global
        sc = self.success

        for runner_key, runner in runners.items():
            sym = runner_key.split(":", 1)[1] if ":" in runner_key else runner_key
            # Capturer sym via argument par défaut (évite le piège closure Python)
            runner._on_skip_local = lambda _s, _sym=sym: sl.__setitem__(
                _sym, sl[_sym] + 1
            )
            runner._on_skip_global = lambda _s, _sym=sym: sg.__setitem__(
                _sym, sg[_sym] + 1
            )
            runner._on_position_opened = lambda _s, _sym=sym: sc.__setitem__(
                _sym, sc[_sym] + 1
            )


# ---------------------------------------------------------------------------
# Sous-classe PortfolioBacktester instrumentée
# ---------------------------------------------------------------------------


class DiagBacktester(PortfolioBacktester):
    """PortfolioBacktester avec injection des hooks de diagnostic."""

    def __init__(self, *args: Any, diag: DiagCounters, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._diag = diag
        self._diag_snapshots: list[PortfolioSnapshot] = []

    def _create_runners(self, *args: Any, **kwargs: Any) -> Any:
        runners, engine = super()._create_runners(*args, **kwargs)
        self._diag.attach(runners)
        return runners, engine

    async def _simulate(self, *args: Any, **kwargs: Any) -> Any:
        snapshots, trades, liq = await super()._simulate(*args, **kwargs)
        self._diag_snapshots = list(snapshots)
        return snapshots, trades, liq


# ---------------------------------------------------------------------------
# Lancement d'un backtest
# ---------------------------------------------------------------------------


async def run_backtest(
    config: Any,
    strategy_name: str,
    assets: list[str],
    days: int,
    leverage: int,
    diag: DiagCounters,
    no_guard: bool = False,
) -> tuple[PortfolioResult, list[PortfolioSnapshot]]:
    """Lance un portfolio backtest et retourne le résultat + snapshots bruts.

    Si no_guard=True, le max_margin_ratio est temporairement porté à 10.0
    pour désactiver les guards et obtenir une référence « trades possibles ».
    """
    diag.reset()

    original_ratio: float | None = None
    if no_guard:
        try:
            original_ratio = config.risk.max_margin_ratio
            # Pydantic v2 frozen → contournement via object.__setattr__
            try:
                config.risk.max_margin_ratio = 10.0
            except (AttributeError, TypeError, ValueError):
                object.__setattr__(config.risk, "max_margin_ratio", 10.0)
        except Exception:
            pass  # Si l'override échoue, le backtest tourne quand même

    end = datetime.now()
    start = end - timedelta(days=days)

    backtester = DiagBacktester(
        config=config,
        initial_capital=10_000.0,
        strategy_name=strategy_name,
        assets=assets,
        kill_switch_pct=45.0,
        leverage=leverage,
        diag=diag,
    )

    result = await backtester.run(start, end, db_path=DB_PATH)
    snapshots = backtester._diag_snapshots

    if no_guard and original_ratio is not None:
        try:
            config.risk.max_margin_ratio = original_ratio
        except (AttributeError, TypeError, ValueError):
            object.__setattr__(config.risk, "max_margin_ratio", original_ratio)

    return result, snapshots


# ---------------------------------------------------------------------------
# Récupération des assets
# ---------------------------------------------------------------------------


def get_assets(config: Any, strategy_name: str) -> list[str]:
    """Retourne les assets per_asset configurés pour la stratégie."""
    strat_cfg = getattr(config.strategies, strategy_name, None)
    if strat_cfg is None:
        return []
    pa = getattr(strat_cfg, "per_asset", {})
    if isinstance(pa, dict) and pa:
        return sorted(pa.keys())
    return []


# ---------------------------------------------------------------------------
# Helpers d'affichage
# ---------------------------------------------------------------------------


def _bar(ratio: float, width: int = 22) -> str:
    """Barre de progression ASCII."""
    filled = max(0, min(width, int(round(ratio * width))))
    return "█" * filled + "░" * (width - filled)


def _section(title: str) -> None:
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}")


# ---------------------------------------------------------------------------
# Rapport 1 — Trades skippés par margin guard
# ---------------------------------------------------------------------------


def print_skip_table(
    diag: DiagCounters,
    diag_ng: DiagCounters,
) -> None:
    _section("1. TRADES SKIPPÉS PAR MARGIN GUARD (par asset)")
    print(
        "  Skip-L = bloqué par guard local (runner)\n"
        "  Skip-G = bloqué par guard global (portfolio)\n"
        "  Ref-NG = ouvertures réussies sans guard (référence)\n"
    )

    all_assets = sorted(
        set(
            list(diag.success)
            + list(diag.skip_local)
            + list(diag.skip_global)
            + list(diag_ng.success)
        )
    )

    header = (
        f"  {'Asset':<14} {'OK':>6} {'Skip-L':>7} {'Skip-G':>7}"
        f" {'Ref-NG':>7} {'Skip%':>7}"
    )
    print(header)
    print(f"  {'-' * 54}")

    rows: list[tuple] = []
    for sym in all_assets:
        ok = diag.success.get(sym, 0)
        sl_ = diag.skip_local.get(sym, 0)
        sg_ = diag.skip_global.get(sym, 0)
        ng = diag_ng.success.get(sym, 0)
        skip_total = sl_ + sg_
        skip_pct = skip_total / ng * 100 if ng > 0 else 0.0
        rows.append((sym, ok, sl_, sg_, ng, skip_total, skip_pct))

    rows.sort(key=lambda r: r[6], reverse=True)

    for sym, ok, sl_, sg_, ng, skip_total, skip_pct in rows:
        base = sym.replace("/USDT", "")
        flag = " ◄ STARVED" if skip_pct > 60 else (" ·" if skip_pct > 30 else "")
        print(
            f"  {base:<14} {ok:>6} {sl_:>7} {sg_:>7}"
            f" {ng:>7} {skip_pct:>6.1f}%{flag}"
        )

    total_ok = sum(r[1] for r in rows)
    total_sl = sum(r[2] for r in rows)
    total_sg = sum(r[3] for r in rows)
    total_ng = sum(r[4] for r in rows)
    total_skip_pct = (total_sl + total_sg) / total_ng * 100 if total_ng > 0 else 0.0

    print(f"  {'-' * 54}")
    print(
        f"  {'TOTAL':<14} {total_ok:>6} {total_sl:>7} {total_sg:>7}"
        f" {total_ng:>7} {total_skip_pct:>6.1f}%"
    )


# ---------------------------------------------------------------------------
# Rapport 2 — Margin utilization timeline
# ---------------------------------------------------------------------------


def print_margin_timeline(
    snapshots: list[PortfolioSnapshot],
    leverage: int,
) -> None:
    _section(f"2. MARGIN UTILIZATION TIMELINE (levier {leverage}x, échantillon ~24h)")

    if not snapshots:
        print("  Aucun snapshot disponible.")
        return

    # Un snapshot par jour (premier de chaque date)
    daily: list[PortfolioSnapshot] = []
    seen: set = set()
    for snap in snapshots:
        d = snap.timestamp.date()
        if d not in seen:
            seen.add(d)
            daily.append(snap)

    print(
        f"  {'Date':<12} {'Margin%':>8} {'Positions':>10} {'Assets':>8}  Barre"
    )
    print(f"  {'-' * 66}")

    for snap in daily:
        pct = snap.margin_ratio * 100
        bar = _bar(snap.margin_ratio, 22)
        if pct > 65:
            flag = " !! proche guard"
        elif pct > 55:
            flag = " !  tendu"
        else:
            flag = ""
        print(
            f"  {snap.timestamp.strftime('%Y-%m-%d'):<12}"
            f" {pct:>7.1f}%"
            f" {snap.n_open_positions:>10}"
            f" {snap.n_assets_with_positions:>8}  {bar}{flag}"
        )

    ratios = [s.margin_ratio * 100 for s in snapshots]
    avg = sum(ratios) / len(ratios) if ratios else 0.0
    above60 = sum(1 for r in ratios if r > 60) / len(ratios) * 100 if ratios else 0.0
    print(f"\n  Moyenne : {avg:.1f}%  |  % temps > 60% : {above60:.1f}%")


# ---------------------------------------------------------------------------
# Rapport 3 — Comparaison 6x vs levier cible
# ---------------------------------------------------------------------------


def _margin_stats(
    snaps: list[PortfolioSnapshot],
) -> tuple[float, float, float, float]:
    if not snaps:
        return 0.0, 0.0, 0.0, 0.0
    ratios = [s.margin_ratio * 100 for s in snaps]
    avg = sum(ratios) / len(ratios)
    med = sorted(ratios)[len(ratios) // 2]
    above50 = sum(1 for r in ratios if r > 50) / len(ratios) * 100
    above60 = sum(1 for r in ratios if r > 60) / len(ratios) * 100
    return avg, med, above50, above60


def print_comparison(
    lev_base: int,
    result_base: PortfolioResult,
    snaps_base: list[PortfolioSnapshot],
    diag_base: DiagCounters,
    lev_target: int,
    result_target: PortfolioResult,
    snaps_target: list[PortfolioSnapshot],
    diag_target: DiagCounters,
) -> None:
    _section(f"3. COMPARAISON {lev_base}x vs {lev_target}x")

    avg_b, med_b, a50_b, a60_b = _margin_stats(snaps_base)
    avg_t, med_t, a50_t, a60_t = _margin_stats(snaps_target)

    skip_b = sum(diag_base.skip_local.values()) + sum(diag_base.skip_global.values())
    skip_t = sum(diag_target.skip_local.values()) + sum(diag_target.skip_global.values())

    w = 10
    print(f"\n  {'Métrique':<38} {f'{lev_base}x':>{w}} {f'{lev_target}x':>{w}}")
    print(f"  {'-' * (40 + w * 2)}")

    def row(label: str, v1: Any, v2: Any, fmt: str = "") -> None:
        f1 = f"{v1:{fmt}}" if fmt else str(v1)
        f2 = f"{v2:{fmt}}" if fmt else str(v2)
        print(f"  {label:<38} {f1:>{w}} {f2:>{w}}")

    row("Trades total", result_base.total_trades, result_target.total_trades)
    row("Trades skippés (margin guard)", skip_b, skip_t)
    row("Return total (%)", f"{result_base.total_return_pct:+.1f}%", f"{result_target.total_return_pct:+.1f}%")
    row("Max drawdown (%)", f"{result_base.max_drawdown_pct:.1f}%", f"{result_target.max_drawdown_pct:.1f}%")
    row("Kill switch triggers", result_base.kill_switch_triggers, result_target.kill_switch_triggers)
    row("Peak margin (%)", f"{result_base.peak_margin_ratio * 100:.1f}%", f"{result_target.peak_margin_ratio * 100:.1f}%")
    row("Margin moyen (%)", f"{avg_b:.1f}%", f"{avg_t:.1f}%")
    row("Margin médian (%)", f"{med_b:.1f}%", f"{med_t:.1f}%")
    row("% temps > 50% margin", f"{a50_b:.1f}%", f"{a50_t:.1f}%")
    row("% temps > 60% margin", f"{a60_b:.1f}%", f"{a60_t:.1f}%")

    # Assets les plus impactés par le levier
    print(f"\n  Assets les plus impactés par le changement de levier :")
    print(f"  {'Asset':<16} {f'Skip({lev_base}x)':>12} {f'Skip({lev_target}x)':>12} {'Delta':>8}")
    print(f"  {'-' * 52}")

    all_assets = sorted(
        set(
            list(diag_base.skip_local)
            + list(diag_base.skip_global)
            + list(diag_target.skip_local)
            + list(diag_target.skip_global)
        )
    )
    impact_rows: list[tuple] = []
    for sym in all_assets:
        s_b = diag_base.skip_local.get(sym, 0) + diag_base.skip_global.get(sym, 0)
        s_t = diag_target.skip_local.get(sym, 0) + diag_target.skip_global.get(sym, 0)
        impact_rows.append((sym, s_b, s_t, s_t - s_b))

    impact_rows.sort(key=lambda r: abs(r[3]), reverse=True)

    for sym, s_b, s_t, delta in impact_rows[:12]:
        base = sym.replace("/USDT", "")
        flag = " ▲ plus bloqué" if delta > 50 else (" ▼" if delta < -20 else "")
        print(
            f"  {base:<16} {s_b:>12} {s_t:>12} {delta:>+8}{flag}"
        )


# ---------------------------------------------------------------------------
# Rapport 4 — Ordre d'allocation
# ---------------------------------------------------------------------------


def print_allocation_order(runner_order: list[str]) -> None:
    _section("4. ORDRE D'ALLOCATION DES RUNNERS")
    print(
        "\n  Cet ordre est fixe pour toute la simulation (insertion dans le dict).\n"
        "  Quand la marge est tendue, les premiers runners ont priorité\n"
        "  sur les suivants pour l'ouverture de nouvelles positions.\n"
    )
    print(f"  {'#':>4}  {'Stratégie':<22}  {'Asset'}")
    print(f"  {'-' * 50}")

    for i, key in enumerate(runner_order, 1):
        if ":" in key:
            strat, sym = key.split(":", 1)
        else:
            strat, sym = "", key
        base = sym.replace("/USDT", "")
        print(f"  {i:>4}  {strat:<22}  {base}")

    print(f"\n  Total : {len(runner_order)} runners")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _main(args: argparse.Namespace) -> None:
    config = get_config(config_dir=ROOT / "config")

    strategy_name: str = args.strategy
    leverage_target: int = args.leverage
    days: int = args.days
    leverage_base: int = 6

    assets = get_assets(config, strategy_name)
    if not assets:
        print(
            f"  Aucun asset per_asset trouvé pour «{strategy_name}» dans config/strategies.yaml."
        )
        sys.exit(1)

    print(
        f"\n{'═' * 72}\n"
        f"  DIAGNOSTIC MARGIN STARVATION\n"
        f"  Stratégie : {strategy_name.upper()}  |  Période : {days}j  "
        f"|  Levier cible : {leverage_target}x\n"
        f"  Assets ({len(assets)}) : "
        + ", ".join(a.replace("/USDT", "") for a in assets)
        + f"\n{'═' * 72}"
    )

    # ------------------------------------------------------------------
    # Backtest 1 : levier cible, avec margin guard
    # ------------------------------------------------------------------
    diag_target = DiagCounters()
    print(f"\n[1/3] Backtest {leverage_target}x avec margin guard (levier cible)…")
    result_target, snaps_target = await run_backtest(
        config, strategy_name, assets, days, leverage_target, diag_target, no_guard=False
    )
    print(
        f"      → {result_target.total_trades} trades, "
        f"return={result_target.total_return_pct:+.1f}%"
    )

    # ------------------------------------------------------------------
    # Backtest 2 : levier base (6x), avec margin guard
    # ------------------------------------------------------------------
    diag_base = DiagCounters()
    if leverage_target != leverage_base:
        print(f"\n[2/3] Backtest {leverage_base}x (référence)…")
        result_base, snaps_base = await run_backtest(
            config, strategy_name, assets, days, leverage_base, diag_base, no_guard=False
        )
        print(
            f"      → {result_base.total_trades} trades, "
            f"return={result_base.total_return_pct:+.1f}%"
        )
    else:
        result_base = result_target
        snaps_base = snaps_target
        diag_base = diag_target
        print(f"\n[2/3] Levier cible = {leverage_base}x (pas de backtest de référence séparé).")

    # ------------------------------------------------------------------
    # Backtest 3 : levier cible, SANS margin guard (référence skip counts)
    # ------------------------------------------------------------------
    diag_ng = DiagCounters()
    print(f"\n[3/3] Backtest {leverage_target}x SANS margin guard (référence pour skip counts)…")
    result_ng, _ = await run_backtest(
        config, strategy_name, assets, days, leverage_target, diag_ng, no_guard=True
    )
    print(f"      → {result_ng.total_trades} trades sans guard (base de comparaison)")

    # ------------------------------------------------------------------
    # Rapports
    # ------------------------------------------------------------------
    print_skip_table(diag_target, diag_ng)
    print_margin_timeline(snaps_target, leverage_target)

    if leverage_target != leverage_base:
        print_comparison(
            leverage_base, result_base, snaps_base, diag_base,
            leverage_target, result_target, snaps_target, diag_target,
        )
    else:
        _section("3. COMPARAISON 6x vs 7x")
        print(
            f"  (Relancez avec --leverage != {leverage_base} pour la comparaison)"
        )

    print_allocation_order(diag_target.runner_order)

    print(f"\n{'═' * 72}")
    print("  FIN DU DIAGNOSTIC")
    print(f"{'═' * 72}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic margin starvation — portfolio backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--strategy", default="grid_atr", help="Nom de la stratégie (défaut: grid_atr)"
    )
    parser.add_argument(
        "--days", type=int, default=365, help="Période en jours (défaut: 365)"
    )
    parser.add_argument(
        "--leverage",
        type=int,
        default=7,
        help="Levier cible à diagnostiquer (défaut: 7, comparé à 6x)",
    )
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
