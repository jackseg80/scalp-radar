import sqlite3, json
conn = sqlite3.connect('data/scalp_radar.db')
cur = conn.execute("SELECT asset, monte_carlo_pvalue, monte_carlo_summary FROM optimization_results WHERE strategy_name = 'grid_atr' AND is_latest = 1 AND asset = 'BTC/USDT'")
r = cur.fetchone()
print("Column mc_pvalue:", r[1])
print("JSON blob:", r[2][:500] if r[2] else "NULL")
conn.close()
