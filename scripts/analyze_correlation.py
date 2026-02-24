"""
Analyse corrélation drawdowns entre stratégies.

Usage:
    uv run python -m scripts.analyze_correlation                          # ATR vs MTF (défaut)
    uv run python -m scripts.analyze_correlation --list                   # labels disponibles
    uv run python -m scripts.analyze_correlation --labels "l1,l2"         # paire
    uv run python -m scripts.analyze_correlation --labels "l1,l2,l3"      # trio
"""
import sys
import io
import json
import math
import argparse
import sqlite3
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

# Force UTF-8 sur Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DB_PATH = Path(__file__).parent.parent / "data" / "scalp_radar.db"

# Labels par défaut (comportement original)
LABEL_ATR = "grid_atr_14assets_7x_post40a"
LABEL_MTF = "grid_multi_tf_14assets_6x_post40a"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_equity_curve(conn: sqlite3.Connection, label: str):
    row = conn.execute(
        "SELECT id, label, initial_capital, equity_curve FROM portfolio_backtests WHERE label = ?",
        (label,)
    ).fetchone()
    if row is None:
        print(f"[ERREUR] Label '{label}' introuvable en DB.")
        print("  Labels disponibles :")
        for r in conn.execute("SELECT id, label FROM portfolio_backtests ORDER BY id").fetchall():
            print(f"    id={r[0]}  {r[1]}")
        sys.exit(1)
    _id, _lbl, initial_capital, ec_json = row
    print(f"  OK '{label}'  id={_id}  capital={initial_capital:,.0f}$")
    return json.loads(ec_json), initial_capital


def parse_ts(s: str) -> datetime:
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def equity_to_daily(curve: list[dict]) -> dict[str, float]:
    """Dernier point de chaque jour → dict date_str → equity."""
    daily: dict[str, float] = {}
    for pt in curve:
        d = parse_ts(pt["timestamp"]).strftime("%Y-%m-%d")
        daily[d] = pt["equity"]
    return daily


def daily_drawdown(daily_equity: dict[str, float]) -> dict[str, float]:
    """Dict date → drawdown% par rapport au peak courant (valeur <= 0)."""
    dd: dict[str, float] = {}
    peak = 0.0
    for d in sorted(daily_equity):
        eq = daily_equity[d]
        if eq > peak:
            peak = eq
        dd[d] = (eq - peak) / peak * 100.0 if peak > 0 else 0.0
    return dd


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return num / (sx * sy)


def worst_n(dd: dict[str, float], n: int = 5) -> list[tuple[str, float]]:
    return sorted(dd.items(), key=lambda x: x[1])[:n]


def combined_return_n(daily_list: list[dict], weights: list[float]) -> dict[str, float]:
    """Return cumulé combiné (%) pour N stratégies depuis le premier jour commun."""
    common = sorted(set.intersection(*[set(d) for d in daily_list]))
    if not common:
        return {}
    d0 = common[0]
    eq0 = [d[d0] for d in daily_list]
    result: dict[str, float] = {}
    for d in common:
        result[d] = sum(
            w * (daily_list[i][d] / eq0[i] - 1.0) * 100.0
            for i, w in enumerate(weights)
        )
    return result


def max_dd_from_returns(combined_ret: dict[str, float]) -> float:
    peak = 0.0
    max_dd = 0.0
    for d in sorted(combined_ret):
        val = 100.0 + combined_ret[d]
        if val > peak:
            peak = val
        dd = (val - peak) / peak * 100.0
        if dd < max_dd:
            max_dd = dd
    return max_dd


def final_return(combined_ret: dict[str, float]) -> float:
    if not combined_ret:
        return 0.0
    return combined_ret[max(combined_ret)]


def short_name(label: str) -> str:
    """grid_atr_14assets_7x → grid_atr  |  grid_boltrend_6x → grid_boltrend"""
    parts = label.split("_")
    # Prend les parties jusqu'à la première qui commence par un chiffre
    name_parts = []
    for p in parts:
        if p and p[0].isdigit():
            break
        name_parts.append(p)
    return "_".join(name_parts) if name_parts else parts[0]


def r_interp(r: float) -> str:
    if math.isnan(r):
        return "N/A"
    if abs(r) < 0.3:
        return "faible -> bonne diversification"
    elif abs(r) < 0.6:
        return "moderee -> diversification partielle"
    else:
        return "forte -> peu de diversification"


# ── Sections d'analyse ────────────────────────────────────────────────────────

def section_correlations(labels, dd_list):
    sep = "-" * 70
    print(sep)
    print("  1. CORRELATIONS DE PEARSON DES DRAWDOWNS JOURNALIERS")
    print(sep)
    names = [short_name(l) for l in labels]
    for i, j in combinations(range(len(labels)), 2):
        common = sorted(set(dd_list[i]) & set(dd_list[j]))
        xs = [dd_list[i][d] for d in common]
        ys = [dd_list[j][d] for d in common]
        r = pearson(xs, ys)
        print(f"  {names[i]:<22} vs {names[j]:<22} : r = {r:+.4f}  [{r_interp(r)}]")
    print()


def section_worst_dd(labels, dd_list):
    sep = "-" * 70
    print(sep)
    print("  2. TOP 5 PIRES DRAWDOWNS PAR STRATEGIE")
    print(sep)
    names = [short_name(l) for l in labels]
    col_w = 13

    for idx in range(len(labels)):
        worst = worst_n(dd_list[idx], 5)
        print(f"\n  {names[idx]} -- 5 pires :")
        hdr = f"  {'Date':<12}" + "".join(f"  {n:>{col_w}}" for n in names)
        print(hdr)
        print(f"  {'-'*12}" + f"  {'-'*col_w}" * len(names))
        for d, _ in worst:
            row = f"  {d:<12}"
            for dd in dd_list:
                val = dd.get(d, float("nan"))
                cell = f"{val:.2f}%" if not math.isnan(val) else "N/A"
                row += f"  {cell:>{col_w}}"
            print(row)
    print()


def _pair_allocs_grid():
    return [
        (1.00, 0.00), (0.80, 0.20), (0.70, 0.30), (0.60, 0.40),
        (0.50, 0.50), (0.40, 0.60), (0.30, 0.70), (0.20, 0.80), (0.00, 1.00),
    ]


def section_pair_allocations(labels, daily_list):
    """Section 3 : allocations pairwise."""
    sep = "-" * 70
    print(sep)
    print("  3. ALLOCATIONS PAIRWISE (return total / max DD / ratio R/DD)")
    print(sep)
    names = [short_name(l) for l in labels]
    all_best = []

    for i, j in combinations(range(len(labels)), 2):
        na, nb = names[i], names[j]
        print(f"\n  --- Paire {na} / {nb} ---")
        print(f"  {'Alloc A/B':<16}  {'Return%':>9}  {'Max DD%':>9}  {'Ratio R/DD':>10}")
        print(f"  {'-'*16}  {'-'*9}  {'-'*9}  {'-'*10}")

        best_ratio = float("-inf")
        best = None
        row_data = []

        for wa, wb in _pair_allocs_grid():
            comb = combined_return_n([daily_list[i], daily_list[j]], [wa, wb])
            ret  = final_return(comb)
            mdd  = max_dd_from_returns(comb)
            ratio = abs(ret / mdd) if mdd < -0.1 else float("inf")
            row_data.append((wa, wb, ret, mdd, ratio))
            if ratio > best_ratio and mdd < -0.1:
                best_ratio = ratio
                best = (wa, wb, ret, mdd, ratio)

        for wa, wb, ret, mdd, ratio in row_data:
            lbl = f"{int(wa*100)}/{int(wb*100)}"
            marker = " <-- OPTIMAL" if (best and wa == best[0]) else ""
            ratio_str = f"{ratio:.2f}" if ratio != float("inf") else "    inf"
            print(f"  {lbl:<16}  {ret:>+9.2f}%  {mdd:>9.2f}%  {ratio_str:>10}{marker}")

        if best:
            all_best.append((na, nb, best))

    print()
    return all_best


def section_trio_allocation(labels, daily_list):
    """Section 4 : allocation 3 voies, 10% steps, chaque strat >= 10%."""
    sep = "-" * 70
    print(sep)
    print("  4. ALLOCATION 3 VOIES (10% steps, chaque strategie >= 10%)")
    print(sep)
    names = [short_name(l) for l in labels]

    combos = [
        (w1 / 100, w2 / 100, (100 - w1 - w2) / 100)
        for w1 in range(10, 81, 10)
        for w2 in range(10, 81, 10)
        if 10 <= 100 - w1 - w2 <= 80
    ]

    results = []
    for w1, w2, w3 in combos:
        comb  = combined_return_n(daily_list, [w1, w2, w3])
        ret   = final_return(comb)
        mdd   = max_dd_from_returns(comb)
        ratio = abs(ret / mdd) if mdd < -0.1 else float("inf")
        results.append((w1, w2, w3, ret, mdd, ratio))

    results.sort(key=lambda x: -x[5] if x[5] != float("inf") else 1e9)

    col = 16
    print(f"\n  {names[0]:<{col}} {names[1]:<{col}} {names[2]:<{col}}  {'Return%':>9}  {'Max DD%':>9}  {'Ratio':>8}")
    print(f"  {'-'*col} {'-'*col} {'-'*col}  {'-'*9}  {'-'*9}  {'-'*8}")

    best = results[0] if results else None
    for k, (w1, w2, w3, ret, mdd, ratio) in enumerate(results):
        marker = " <-- OPTIMAL" if (best and w1 == best[0] and w2 == best[1]) else ""
        ratio_str = f"{ratio:.2f}" if ratio != float("inf") else "    inf"
        p1, p2, p3 = f"{int(w1*100)}%", f"{int(w2*100)}%", f"{int(w3*100)}%"
        print(f"  {p1:<{col}} {p2:<{col}} {p3:<{col}}  {ret:>+9.2f}%  {mdd:>9.2f}%  {ratio_str:>8}{marker}")
        if k == 9:
            print(f"  ... ({len(results) - 10} autres combinaisons)")
            break

    print()
    return best, names


def section_summary(labels, daily_list, dd_list, all_best_pairs, trio_best):
    """Section finale : métriques solo + co-drawdown + recommandation."""
    sep = "-" * 70
    SEP = "=" * 70
    names = [short_name(l) for l in labels]

    print(sep)
    print("  METRIQUES SOLO PAR STRATEGIE (periode complete de chaque backtest)")
    print(sep)
    for idx in range(len(labels)):
        dates = sorted(daily_list[idx])
        if not dates:
            continue
        d0, dn = dates[0], dates[-1]
        ret  = (daily_list[idx][dn] / daily_list[idx][d0] - 1.0) * 100.0
        mdd  = min(dd_list[idx].values()) if dd_list[idx] else 0.0
        vals = list(dd_list[idx].values())
        avg  = sum(abs(v) for v in vals) / len(vals) if vals else 0.0
        print(f"  {names[idx]:<26} : return {ret:>+8.2f}%  max_dd {mdd:>7.2f}%  avg_dd_abs {avg:.2f}%")
    print()

    # Co-drawdowns par paire
    print(f"  Jours simultanement en DD < -5% :")
    for i, j in combinations(range(len(labels)), 2):
        common = sorted(set(dd_list[i]) & set(dd_list[j]))
        both_bad = [(d, dd_list[i][d], dd_list[j][d])
                    for d in common if dd_list[i][d] < -5 and dd_list[j][d] < -5]
        line = f"    {names[i]} + {names[j]} : {len(both_bad)} jours"
        if both_bad:
            worst_co = min(both_bad, key=lambda x: x[1] + x[2])
            line += f"  (pire: {worst_co[0]} {names[i]}={worst_co[1]:.2f}% {names[j]}={worst_co[2]:.2f}%)"
        print(line)
    print()

    # Récap allocations optimales
    if all_best_pairs:
        print(f"  Allocations optimales (paires) :")
        for na, nb, (wa, wb, ret, mdd, ratio) in all_best_pairs:
            print(f"    {na}/{nb} : {int(wa*100)}%/{int(wb*100)}%"
                  f"  return {ret:>+.2f}%  max_dd {mdd:.2f}%  ratio {ratio:.2f}")

    if trio_best:
        w1, w2, w3, ret, mdd, ratio, trio_names = trio_best
        print(f"\n  Allocation optimale 3 voies :")
        print(f"    {trio_names[0]} {int(w1*100)}%"
              f" / {trio_names[1]} {int(w2*100)}%"
              f" / {trio_names[2]} {int(w3*100)}%")
        print(f"    return {ret:>+.2f}%  max_dd {mdd:.2f}%  ratio {ratio:.2f}")

    print(f"\n{SEP}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse corrélation drawdowns entre backtests sauvés en DB"
    )
    parser.add_argument(
        "--labels", metavar="L1,L2[,L3]",
        help="Labels des backtests à comparer (virgule-séparés, 2 ou 3 max)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Lister les labels disponibles en DB"
    )
    args = parser.parse_args()

    conn = sqlite3.connect(str(DB_PATH))

    if args.list:
        rows = conn.execute(
            "SELECT id, label, initial_capital FROM portfolio_backtests ORDER BY id"
        ).fetchall()
        conn.close()
        if not rows:
            print("  Aucun backtest sauvé en DB.")
            return
        print(f"\n  {'ID':>4}  {'Label':<55}  {'Capital':>10}")
        print(f"  {'-'*4}  {'-'*55}  {'-'*10}")
        for r in rows:
            print(f"  {r[0]:>4}  {r[1]:<55}  {r[2]:>9,.0f}$")
        print()
        return

    # Résoudre les labels
    if args.labels:
        labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    else:
        labels = [LABEL_ATR, LABEL_MTF]

    if len(labels) < 2:
        print("[ERREUR] Au moins 2 labels requis.")
        sys.exit(1)
    if len(labels) > 3:
        print("[ERREUR] Maximum 3 labels supportés.")
        sys.exit(1)

    SEP = "=" * 70
    names = [short_name(l) for l in labels]

    print(f"\n{SEP}")
    print(f"  ANALYSE CORRELATION DRAWDOWNS : {' / '.join(names)}")
    print(f"{SEP}\n")

    print("Chargement des backtests depuis DB :")
    curves = []
    for label in labels:
        curve, _ = load_equity_curve(conn, label)
        curves.append(curve)
    conn.close()
    print()

    daily_list = [equity_to_daily(c) for c in curves]
    dd_list    = [daily_drawdown(d) for d in daily_list]

    # Période commune (pour info)
    common_global = sorted(set.intersection(*[set(d) for d in daily_list]))
    if not common_global:
        print("[ERREUR] Aucune date commune entre toutes les stratégies.")
        sys.exit(1)
    print(f"Periode commune (toutes) : {common_global[0]} -> {common_global[-1]}  ({len(common_global)} jours)")
    if len(labels) == 3:
        for i, j in combinations(range(len(labels)), 2):
            cpair = sorted(set(daily_list[i]) & set(daily_list[j]))
            print(f"  Paire {names[i]}/{names[j]} : {cpair[0]} -> {cpair[-1]}  ({len(cpair)} jours)")
    print()

    # Sections
    section_correlations(labels, dd_list)
    section_worst_dd(labels, dd_list)
    all_best_pairs = section_pair_allocations(labels, daily_list)

    trio_best = None
    if len(labels) == 3:
        best, trio_names = section_trio_allocation(labels, daily_list)
        if best:
            w1, w2, w3, ret, mdd, ratio = best
            trio_best = (w1, w2, w3, ret, mdd, ratio, trio_names)

    section_summary(labels, daily_list, dd_list, all_best_pairs, trio_best)


if __name__ == "__main__":
    main()
