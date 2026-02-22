import sqlite3, json
conn = sqlite3.connect('data/scalp_radar.db')
rows = conn.execute("""
    SELECT c.oos_sharpe, c.n_windows_evaluated, c.per_window_sharpes 
    FROM wfo_combo_results c
    JOIN optimization_results r ON c.optimization_result_id = r.id
    WHERE r.strategy_name = 'grid_atr' AND r.asset = 'BTC/USDT'
    AND r.is_latest = 1
    ORDER BY c.is_best DESC
    LIMIT 5
""").fetchall()
for r in rows:
    pws = json.loads(r[2]) if r[2] else []
    print(f"sharpe={r[0]}, n_win={r[1]}, per_window_len={len(pws)}")
conn.close()
