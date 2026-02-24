import sqlite3, json
conn = sqlite3.connect('data/scalp_radar.db')

# Top 5 assets ? investiguer : ceux qui ont le plus chang? ou les plus gros contributeurs
assets = ['CRV/USDT', 'BTC/USDT', 'AVAX/USDT', 'DOGE/USDT', 'GALA/USDT']

for asset in assets:
    rows = conn.execute('''
        SELECT best_params, total_score, grade, created_at, is_latest
        FROM optimization_results
        WHERE strategy_name = 'grid_atr' AND asset = ?
        ORDER BY created_at DESC LIMIT 3
    ''', (asset,)).fetchall()
    
    print(f"\n{'='*60}")
    print(f"  {asset}")
    print(f"{'='*60}")
    for r in rows:
        params = json.loads(r[0])
        tag = ">>> NEW" if r[4] == 1 else "    old"
        print(f"  {tag} | {r[2]} ({int(r[1])}) | {r[3][:16]}")
        for k, v in sorted(params.items()):
            print(f"       {k:20s}: {v}")

conn.close()
