import sqlite3
db = sqlite3.connect("data/scalp_radar.db")
r = db.execute(
    "SELECT MIN(timestamp), MAX(timestamp), typeof(timestamp) "
    "FROM candles WHERE symbol='BTC/USDT' AND timeframe='1h' AND exchange='binance'"
).fetchone()
print(f"Min: {r[0]}")
print(f"Max: {r[1]}")
print(f"Type: {r[2]}")
if isinstance(r[0], (int, float)):
    diff_days = (r[1] - r[0]) / 86400000.0
    print(f"Diff ms: {r[1] - r[0]}")
    print(f"Diff days: {diff_days:.0f}")
else:
    print(f"String format - not numeric")
db.close()

# Ajouter à la fin de check_ts.py
new_assets = ['DOT/USDT','ATOM/USDT','LTC/USDT','FIL/USDT','MATIC/USDT','ETC/USDT','TRX/USDT','XLM/USDT']
placeholders = ','.join(f"'{s}'" for s in new_assets)
rows = db.execute(
    f"SELECT symbol, COUNT(*), MIN(timestamp), MAX(timestamp) "
    f"FROM candles WHERE timeframe='1h' AND exchange='binance' AND symbol IN ({placeholders}) "
    f"GROUP BY symbol ORDER BY symbol"
).fetchall()
print("\n--- Nouveaux assets ---")
for r in rows:
    print(f"{r[0]:15s} {r[1]:6d} candles  {r[2][:10]} -> {r[3][:10]}")