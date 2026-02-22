"""Check historical data availability for new candidate assets."""
import ccxt
import datetime

ex = ccxt.binance()
candidates = ["XRP/USDT", "SUI/USDT", "BCH/USDT", "BNB/USDT", "AAVE/USDT", "ARB/USDT", "OP/USDT"]

print(f"SYMBOL        1ERE BOUGIE 1H      JOURS DISPO")
print("-" * 48)
for sym in candidates:
    try:
        candles = ex.fetch_ohlcv(sym, "1h", limit=1, since=ex.parse8601("2019-01-01T00:00:00Z"))
        if candles:
            first = datetime.datetime.fromtimestamp(candles[0][0] / 1000, tz=datetime.timezone.utc)
            days = (datetime.datetime.now(tz=datetime.timezone.utc) - first).days
            print(f"  {sym:<12} {first.strftime('%Y-%m-%d'):<20} {days}j")
        else:
            print(f"  {sym:<12} PAS DE DONNEES")
    except Exception as e:
        print(f"  {sym:<12} ERREUR: {e}")