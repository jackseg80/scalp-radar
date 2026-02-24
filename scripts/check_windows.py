import sqlite3, json
conn = sqlite3.connect('data/scalp_radar.db')
rows = conn.execute("""
    SELECT r.strategy_name, r.asset, r.n_windows, c.oos_sharpe, c.n_windows_evaluated,
           c.consistency, c.oos_trades
    FROM wfo_combo_results c
    JOIN optimization_results r ON c.optimization_result_id = r.id
    WHERE r.is_latest = 1 AND c.is_best = 1
    ORDER BY r.strategy_name, r.asset
""").fetchall()
print(f"{'Strategy':15s} {'Asset':15s} {'WFO_win':>7s} {'Best_win':>8s} {'Ratio':>6s} {'Sharpe':>7s} {'Cons':>5s} {'Trades':>7s}")
print("-" * 80)
for r in rows:
    ratio = f"{r[4]}/{r[2]}"
    flag = " <<<" if r[4] < r[2] * 0.5 else ""
    print(f"{r[0]:15s} {r[1]:15s} {r[2]:>7d} {r[4]:>8d} {ratio:>6s} {r[3]:>7.2f} {r[5]:>5.0%} {r[6]:>7d}{flag}")
conn.close()
