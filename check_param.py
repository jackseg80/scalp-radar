# check_params.py
import sqlite3, json
conn = sqlite3.connect('data/scalp_radar.db')
assets = ['ADA/USDT', 'AVAX/USDT', 'BTC/USDT', 'SOL/USDT', 'ATOM/USDT', 'NEAR/USDT', 'ETH/USDT']
for a in assets:
    r = conn.execute(
        "SELECT best_params FROM optimization_results "
        "WHERE strategy_name='grid_atr' AND asset=? AND is_latest=1", (a,)
    ).fetchone()
    if r:
        p = json.loads(r[0])
        print(f"{a}: min_atr_pct={p.get('min_atr_pct','N/A'):4}  min_grid_spacing_pct={p.get('min_grid_spacing_pct','N/A')}")
conn.close()