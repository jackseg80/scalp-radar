"""Reset du simulateur paper trading.

Supprime l'état corrompu (capital overflow) et repart à zéro :
- Supprime data/simulator_state.json (capital, positions, stats)
- Vide la table simulation_trades dans la DB
- Idempotent : safe à relancer plusieurs fois
- Ne touche PAS aux données d'optimisation ni à la DB
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

STATE_FILE = "data/simulator_state.json"
EXECUTOR_STATE_FILE = "data/executor_state.json"
DB_FILE = "data/scalp_radar.db"


def reset_simulator(reset_executor: bool = False) -> None:
    """Reset le simulateur paper trading."""
    # 1. Lire et afficher l'état actuel avant suppression
    if Path(STATE_FILE).exists():
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                state = json.load(f)

            saved_at = state.get("saved_at", "inconnu")
            runners = state.get("runners", {})
            print(f"État simulateur trouvé (sauvegardé à {saved_at})")
            print(f"  {len(runners)} runner(s) :")

            for name, rs in runners.items():
                capital = rs.get("capital", 0)
                net_pnl = rs.get("net_pnl", 0)
                trades = rs.get("total_trades", 0)
                grid_pos = len(rs.get("grid_positions", []))
                print(
                    f"    {name}: capital={capital:,.2f}$, "
                    f"pnl={net_pnl:+,.2f}$, trades={trades}, "
                    f"positions_grid={grid_pos}"
                )

                # Détecter l'overflow
                if abs(capital) > 1_000_000:
                    print(f"    ⚠ OVERFLOW DÉTECTÉ (capital > 1M$)")

        except (json.JSONDecodeError, OSError) as e:
            print(f"Fichier état corrompu : {e}")

        # Supprimer
        os.remove(STATE_FILE)
        print(f"\n✓ {STATE_FILE} supprimé")
    else:
        print(f"  {STATE_FILE} absent (déjà clean)")

    # 2. Vider la table simulation_trades dans la DB
    if Path(DB_FILE).exists():
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.execute("SELECT COUNT(*) FROM simulation_trades")
            count = cursor.fetchone()[0]
            print(f"\nTable simulation_trades : {count} trade(s) en DB")

            cursor = conn.execute("DELETE FROM simulation_trades")
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            print(f"✓ {deleted} trade(s) supprimé(s) de la DB")
        except sqlite3.OperationalError as e:
            if "no such table" in str(e).lower():
                print("\n  Table simulation_trades absente (DB ancienne)")
            else:
                print(f"\n  Erreur DB : {e}")
        except Exception as e:
            print(f"\n  Erreur inattendue DB : {e}")
    else:
        print(f"\n  {DB_FILE} absent (pas de DB)")

    # 3. Reset executor state si demandé
    if reset_executor:
        if Path(EXECUTOR_STATE_FILE).exists():
            try:
                with open(EXECUTOR_STATE_FILE, encoding="utf-8") as f:
                    data = json.load(f)
                print(f"\nÉtat executor trouvé (sauvegardé à {data.get('saved_at', 'inconnu')})")
            except (json.JSONDecodeError, OSError):
                print("\nFichier état executor corrompu")
            os.remove(EXECUTOR_STATE_FILE)
            print(f"✓ {EXECUTOR_STATE_FILE} supprimé")
        else:
            print(f"\n  {EXECUTOR_STATE_FILE} absent (déjà clean)")

    print("\n✓ Reset terminé.")
    print("  Au prochain démarrage, chaque stratégie repartira avec :")
    print("  - Capital initial = 10 000$")
    print("  - 0 position ouverte")
    print("  - 0 trade historique (en mémoire ET en DB)")


if __name__ == "__main__":
    reset_executor = "--executor" in sys.argv or "-e" in sys.argv
    reset_simulator(reset_executor=reset_executor)
