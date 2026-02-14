import sqlite3

conn = sqlite3.connect('data/scalp_radar.db')
cursor = conn.execute("""
    SELECT r.id, r.created_at, r.grade, COUNT(c.id) as n_combos
    FROM optimization_results r
    LEFT JOIN wfo_combo_results c ON c.optimization_result_id = r.id
    WHERE r.strategy_name='envelope_dca' AND r.asset='BTC/USDT'
    GROUP BY r.id
    ORDER BY r.created_at DESC
    LIMIT 3
""")
rows = cursor.fetchall()
for row in rows:
    print(f"ID={row[0]}, Date={row[1][:16]}, Grade={row[2]}, Combos={row[3]}")
conn.close()
