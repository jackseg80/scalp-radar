"""
compare_wfo.py -- Comparaison des resultats WFO avant/apres Sprint 36.

Pour chaque (strategy_name, asset) ayant un resultat is_latest=1 ET is_latest=0,
affiche un tableau comparatif Grade / Score / OOS Sharpe / Consistency.

Usage : uv run python scripts/compare_wfo.py
"""

import sqlite3
import sys
from pathlib import Path

# Force UTF-8 sur Windows pour les emojis et caracteres speciaux
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB_PATH = Path("data/scalp_radar.db")
STRATEGIES = ("grid_atr", "grid_boltrend")

GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}


def grade_delta(before: str, after: str) -> int:
    """Retourne la différence de rang (positif = régression)."""
    b = GRADE_ORDER.get(before, 99)
    a = GRADE_ORDER.get(after, 99)
    return a - b  # positif = régression, négatif = amélioration


def fmt_delta(value: float, decimals: int = 1, invert: bool = False) -> str:
    """Formate un delta avec signe et couleur ANSI."""
    if value is None:
        return "  n/a "
    better = value > 0 if not invert else value < 0
    sign = "+" if value >= 0 else ""
    formatted = f"{sign}{value:.{decimals}f}"
    if better:
        return f"\033[32m{formatted:>7}\033[0m"  # vert
    elif value == 0:
        return f"{formatted:>7}"
    else:
        return f"\033[31m{formatted:>7}\033[0m"  # rouge


def grade_display(grade: str) -> str:
    return grade if grade else "?"


def load_data() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # is_latest=1 pour chaque (strategy, asset)
    c.execute(
        """
        SELECT strategy_name, asset, grade, total_score, oos_sharpe, consistency, created_at
        FROM optimization_results
        WHERE strategy_name IN ('grid_atr', 'grid_boltrend')
          AND is_latest = 1
        """,
    )
    latest = {(r["strategy_name"], r["asset"]): dict(r) for r in c.fetchall()}

    # Pour is_latest=0, prendre le run le plus récent par (strategy, asset)
    c.execute(
        """
        SELECT strategy_name, asset, grade, total_score, oos_sharpe, consistency,
               MAX(created_at) as created_at
        FROM optimization_results
        WHERE strategy_name IN ('grid_atr', 'grid_boltrend')
          AND is_latest = 0
        GROUP BY strategy_name, asset
        """,
    )
    previous = {(r["strategy_name"], r["asset"]): dict(r) for r in c.fetchall()}

    conn.close()

    # Combiner : seulement les paires qui ont les deux
    rows = []
    for key, new in latest.items():
        if key not in previous:
            continue
        old = previous[key]
        strat, asset = key

        d_score = (
            (new["total_score"] or 0) - (old["total_score"] or 0)
            if new["total_score"] is not None and old["total_score"] is not None
            else None
        )
        d_sharpe = (
            (new["oos_sharpe"] or 0) - (old["oos_sharpe"] or 0)
            if new["oos_sharpe"] is not None and old["oos_sharpe"] is not None
            else None
        )
        g_delta = grade_delta(
            old["grade"] or "F",
            new["grade"] or "F",
        )

        rows.append(
            {
                "strategy": strat,
                "asset": asset,
                "grade_before": old["grade"] or "?",
                "grade_after": new["grade"] or "?",
                "score_before": old["total_score"],
                "score_after": new["total_score"],
                "d_score": d_score,
                "sharpe_before": old["oos_sharpe"],
                "sharpe_after": new["oos_sharpe"],
                "d_sharpe": d_sharpe,
                "consist_before": old["consistency"],
                "consist_after": new["consistency"],
                "g_delta": g_delta,
                "date_before": (old["created_at"] or "")[:10],
                "date_after": (new["created_at"] or "")[:10],
            }
        )

    # Tri : stratégie, puis delta score (les améliorations en dernier)
    rows.sort(key=lambda r: (r["strategy"], -(r["d_score"] or 0)))
    return rows


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'Asset':<14} "
        f"{'Grade bef→aft':^14} "
        f"{'Score bef':>9} "
        f"{'Score aft':>9} "
        f"{'ΔScore':>7} "
        f"{'OOS Sharpe bef':>14} "
        f"{'OOS Sharpe aft':>14} "
        f"{'ΔSharpe':>8} "
        f"{'Consist bef':>11} "
        f"{'Consist aft':>11}"
    )
    sep = "-" * len(header)

    current_strategy = None

    for row in rows:
        if row["strategy"] != current_strategy:
            current_strategy = row["strategy"]
            print(f"\n\033[1;36m{'=' * 120}\033[0m")
            print(f"\033[1;36m  Strategie : {current_strategy.upper()}\033[0m")
            print(f"\033[1;36m{'=' * 120}\033[0m")
            print(f"  {header}")
            print(f"  {sep}")

        g_before = grade_display(row["grade_before"])
        g_after = grade_display(row["grade_after"])
        g_delta = row["g_delta"]

        # Marqueurs
        marker = "  "
        if g_delta >= 2:
            marker = "🔴"
        elif g_delta != 0:
            marker = "⚠️ "

        # Grade coloré
        if g_delta < 0:
            grade_str = f"\033[32m{g_before} → {g_after}\033[0m"  # amélioration
        elif g_delta > 0:
            grade_str = f"\033[31m{g_before} → {g_after}\033[0m"  # régression
        else:
            grade_str = f"{g_before} → {g_after}"

        asset_col = f"{marker}{row['asset']:<13}"

        score_b = f"{row['score_before']:>9.1f}" if row["score_before"] is not None else f"{'n/a':>9}"
        score_a = f"{row['score_after']:>9.1f}" if row["score_after"] is not None else f"{'n/a':>9}"
        d_score_col = fmt_delta(row["d_score"], decimals=1)

        sharpe_b = f"{row['sharpe_before']:>14.3f}" if row["sharpe_before"] is not None else f"{'n/a':>14}"
        sharpe_a = f"{row['sharpe_after']:>14.3f}" if row["sharpe_after"] is not None else f"{'n/a':>14}"
        d_sharpe_col = fmt_delta(row["d_sharpe"], decimals=3)

        consist_b = f"{row['consist_before']:>11.2f}" if row["consist_before"] is not None else f"{'n/a':>11}"
        consist_a = f"{row['consist_after']:>11.2f}" if row["consist_after"] is not None else f"{'n/a':>11}"

        print(
            f"  {asset_col} "
            f"{grade_str:^14} "
            f"{score_b} "
            f"{score_a} "
            f"{d_score_col} "
            f"{sharpe_b} "
            f"{sharpe_a} "
            f"{d_sharpe_col} "
            f"{consist_b} "
            f"{consist_a}"
        )


def print_summary(rows: list[dict]) -> None:
    print(f"\n{'-' * 60}")
    print("RESUME")
    print(f"{'-' * 60}")
    for strat in STRATEGIES:
        strat_rows = [r for r in rows if r["strategy"] == strat]
        if not strat_rows:
            continue
        improved = sum(1 for r in strat_rows if (r["d_score"] or 0) > 0)
        unchanged = sum(1 for r in strat_rows if (r["d_score"] or 0) == 0)
        degraded = sum(1 for r in strat_rows if (r["d_score"] or 0) < 0)
        grade_changes = sum(1 for r in strat_rows if r["g_delta"] != 0)
        avg_delta = sum(r["d_score"] or 0 for r in strat_rows) / len(strat_rows)
        print(
            f"  {strat:<16}  {len(strat_rows):>2} paires comparées  "
            f"↑{improved} = {unchanged} ↓{degraded}  "
            f"Δ score moy: {avg_delta:+.1f}  "
            f"changements de grade: {grade_changes}"
        )


def main() -> None:
    if not DB_PATH.exists():
        print(f"Erreur : base de données introuvable : {DB_PATH}")
        return

    rows = load_data()
    if not rows:
        print("Aucun résultat à comparer (aucune paire avec is_latest=1 ET is_latest=0).")
        return

    print_table(rows)
    print_summary(rows)
    print()


if __name__ == "__main__":
    main()
