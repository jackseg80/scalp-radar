"""Reset du simulateur paper trading.

Supprime l'état corrompu (capital overflow) et repart à zéro :
- Supprime data/simulator_state.json (capital, positions, stats)
- Idempotent : safe à relancer plusieurs fois
- Ne touche PAS aux données d'optimisation ni à la DB
"""

import json
import os
import sys
from pathlib import Path

STATE_FILE = "data/simulator_state.json"
EXECUTOR_STATE_FILE = "data/executor_state.json"


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

    # 2. Reset executor state si demandé
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
    print("  - 0 trade historique")


if __name__ == "__main__":
    reset_executor = "--executor" in sys.argv or "-e" in sys.argv
    reset_simulator(reset_executor=reset_executor)
