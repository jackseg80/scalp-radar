# scripts/show_results.py
import sqlite3, json
conn = sqlite3.connect("data/scalp_radar.db")
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT asset,grade,total_score,oos_sharpe,consistency,n_windows,best_params "
    "FROM optimization_results WHERE strategy_name='grid_atr' AND is_latest=1 "
    "ORDER BY total_score DESC"
).fetchall()
for r in rows:
    p = json.loads(r["best_params"])
    print(
        f"{r['asset']:12s} Grade {r['grade']}  Score {int(r['total_score'] or 0):3d}  "
        f"OOS Sharpe {float(r['oos_sharpe'] or 0):+.2f}  "
        f"Consistency {float(r['consistency'] or 0) * 100:.0f}%  "
        f"Windows {r['n_windows']}  "
        f"ma={p.get('ma_period')} atr={p.get('atr_period')} "
        f"start={p.get('atr_multiplier_start')} step={p.get('atr_multiplier_step')} "
        f"lvl={p.get('num_levels')} sl={p.get('sl_percent')}"
    )
conn.close()