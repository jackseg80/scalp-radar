import sqlite3
import json

conn = sqlite3.connect('data/scalp_radar.db')
cursor = conn.execute("""
    SELECT j.id, j.created_at, j.strategy_name, j.asset, j.status, j.params_override, r.id as result_id
    FROM optimization_jobs j
    LEFT JOIN optimization_results r ON r.id = j.result_id
    WHERE j.strategy_name='envelope_dca' AND j.asset='BTC/USDT'
    ORDER BY j.created_at DESC
    LIMIT 5
""")
rows = cursor.fetchall()
for row in rows:
    job_id, created, strat, asset, status, params_json, result_id = row
    params = json.loads(params_json) if params_json else None
    print(f"\nJob ID: {job_id[:8]}")
    print(f"Date: {created[:16]}")
    print(f"Status: {status}")
    print(f"Result ID: {result_id}")
    print(f"params_override JSON: {repr(params_json)}")
    print(f"params_override parsed: {params}")
conn.close()
