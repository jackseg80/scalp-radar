"""Script one-shot pour nettoyer les runs invalides de l'Explorer.

Supprime les runs lancés AVANT le fix UX sliders qui ont 1-5 combos
au lieu de 324 (params_override pré-rempli automatiquement).

Les runs CLI anciens (0 combo) sont préservés.
"""

import sqlite3
import sys
from pathlib import Path


def main(auto_confirm=False):
    db_path = Path("data/scalp_radar.db")

    if not db_path.exists():
        print(f"[X] DB introuvable : {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Identifier les runs invalides (1 à 5 combos)
    cursor.execute("""
        SELECT r.id, r.strategy_name, r.asset, r.created_at, COUNT(c.id) as combo_count
        FROM optimization_results r
        LEFT JOIN wfo_combo_results c ON c.optimization_result_id = r.id
        GROUP BY r.id
        HAVING combo_count BETWEEN 1 AND 5
        ORDER BY r.created_at DESC
    """)

    invalid_runs = cursor.fetchall()

    if not invalid_runs:
        print("[OK] Aucun run invalide trouve")
        conn.close()
        return

    print(f"\n[!] {len(invalid_runs)} run(s) invalide(s) trouve(s) :\n")
    for run_id, strategy, asset, created_at, combo_count in invalid_runs:
        print(f"  ID={run_id}, {strategy} x {asset}, {created_at[:16]}, {combo_count} combo(s)")

    # Confirmation utilisateur
    if not auto_confirm:
        print(f"\n[ATTENTION] Ces runs et leurs combo_results vont etre SUPPRIMES.")
        confirm = input("Confirmer ? (oui/non) : ").strip().lower()

        if confirm not in ("oui", "o", "y", "yes"):
            print("[X] Annule par l'utilisateur")
            conn.close()
            return
    else:
        print("\n[AUTO] Confirmation automatique (--yes)")

    # 2. Supprimer les combo_results des runs invalides
    invalid_ids = [row[0] for row in invalid_runs]
    placeholders = ",".join("?" * len(invalid_ids))

    cursor.execute(f"""
        DELETE FROM wfo_combo_results
        WHERE optimization_result_id IN ({placeholders})
    """, invalid_ids)

    combo_deleted = cursor.rowcount
    print(f"\n[DELETE] {combo_deleted} combo_results supprime(s)")

    # 3. Supprimer les runs invalides eux-mêmes
    cursor.execute(f"""
        DELETE FROM optimization_results
        WHERE id IN ({placeholders})
    """, invalid_ids)

    runs_deleted = cursor.rowcount
    print(f"[DELETE] {runs_deleted} run(s) supprime(s)")

    conn.commit()
    conn.close()

    print("\n[OK] Nettoyage termine avec succes")
    print("\n[INFO] Vous pouvez maintenant lancer un nouveau WFO avec le fix UX")
    print("       -> Tous les sliders decoches = 324 combos testees")


if __name__ == "__main__":
    auto_confirm = "--yes" in sys.argv or "-y" in sys.argv
    main(auto_confirm=auto_confirm)
