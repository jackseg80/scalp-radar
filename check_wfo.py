import sqlite3
conn = sqlite3.connect('data/scalp_radar.db')
rows = conn.execute(
    "SELECT asset, timeframe, grade, total_score, created_at "
    "FROM optimization_results "
    "WHERE strategy_name='grid_atr' AND is_latest=1 "
    "ORDER BY created_at DESC"
).fetchall()
for r in rows:
    print(f"{r[0]:12} {r[1]:4} Grade:{r[2]}  Score:{r[3]:.1f}  Date:{r[4][:10]}")
conn.close()