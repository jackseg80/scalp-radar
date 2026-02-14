import sqlite3

conn = sqlite3.connect("data/scalp_radar.db")

# Vérifie
row = conn.execute(
    "SELECT id, strategy_name, asset, grade, total_score, created_at "
    "FROM optimization_results "
    "WHERE strategy_name='vwap_rsi' AND asset='BTC/USDT' AND total_score=87.0"
).fetchone()

if row:
    print(f"Trouvé: id={row[0]}, grade={row[3]}, score={row[4]}, date={row[5]}")
    conn.execute("DELETE FROM optimization_results WHERE id=?", (row[0],))
    # Remet is_latest sur le plus récent
    conn.execute(
        "UPDATE optimization_results SET is_latest=1 "
        "WHERE id = ("
        "  SELECT id FROM optimization_results "
        "  WHERE strategy_name='vwap_rsi' AND asset='BTC/USDT' "
        "  ORDER BY created_at DESC LIMIT 1"
        ")"
    )
    conn.commit()
    print("Supprimé + is_latest corrigé")
else:
    print("Pas trouvé, rien à faire")

conn.close()