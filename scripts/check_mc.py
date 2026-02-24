import sqlite3
conn = sqlite3.connect('data/scalp_radar.db')
cur = conn.execute("SELECT asset, monte_carlo_pvalue, mc_underpowered, n_windows, created_at FROM optimization_results WHERE strategy_name = 'grid_atr' ORDER BY asset, created_at DESC LIMIT 42")
for r in cur.fetchall():
    print(r[0], r[1], r[2], r[3], r[4])
conn.close()
