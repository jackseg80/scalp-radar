import sqlite3, json
conn = sqlite3.connect('data/scalp_radar.db')
rows = conn.execute("""
    SELECT r.asset, c.params FROM wfo_combo_results c
    JOIN optimization_results r ON c.optimization_result_id = r.id
    WHERE r.strategy_name = 'grid_atr' AND r.is_latest = 1 AND c.is_best = 1
""").fetchall()
print(f"{'Asset':15s} {'SL':>5s} {'Levels':>6s} {'ATR_start':>9s}")
for asset, params_json in rows:
    p = json.loads(params_json)
    sl = p.get('sl_percent', '?')
    nl = p.get('num_levels', '?')
    atr_s = p.get('atr_multiplier_start', '?')
    print(f"{asset:15s} {sl:>5} {nl:>6} {atr_s:>9}")
conn.close()
