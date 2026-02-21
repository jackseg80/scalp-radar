"""Vérification des minimums Bitget avant passage LIVE.

Se connecte à Bitget, load_markets(), et vérifie pour chaque asset du Top 10
que le sizing respecte les minimums Bitget. Le leverage est lu depuis strategies.yaml.

Usage :
    uv run python scripts/check_live_sizing.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import yaml

# Ajouter la racine du projet au path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── Configuration ─────────────────────────────────────────────────────────

_STRATS_PATH = Path(__file__).resolve().parent.parent / "config" / "strategies.yaml"
with open(_STRATS_PATH) as f:
    _strats = yaml.safe_load(f)

TOTAL_CAPITAL = 1000.0  # USDT
TOP_10_ASSETS = [
    "BTC/USDT", "CRV/USDT", "DOGE/USDT", "DYDX/USDT",
    "FET/USDT", "GALA/USDT", "ICP/USDT", "NEAR/USDT", "AVAX/USDT",
]
LEVERAGE = _strats.get("grid_atr", {}).get("leverage", 6)
NB_ASSETS = len(TOP_10_ASSETS)

# Nombre de levels par asset (depuis strategies.yaml per_asset grid_atr)
LEVELS_PER_ASSET = {
    "BTC/USDT": 2,
    "CRV/USDT": 4,
    "DOGE/USDT": 4,
    "DYDX/USDT": 2,
    "FET/USDT": 4,
    "GALA/USDT": 4,
    "ICP/USDT": 2,
    "NEAR/USDT": 4,
    "AVAX/USDT": 3,
}


async def main() -> None:
    import ccxt.pro as ccxtpro

    api_key = os.getenv("BITGET_API_KEY", "")
    secret = os.getenv("BITGET_SECRET", "")
    passphrase = os.getenv("BITGET_PASSPHRASE", "")

    if not api_key:
        print("ERREUR: BITGET_API_KEY non défini dans .env")
        print("Conseil: définir les variables ou exécuter depuis le répertoire racine du projet")
        sys.exit(1)

    exchange = ccxtpro.bitget({
        "apiKey": api_key,
        "secret": secret,
        "password": passphrase,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

    try:
        markets = await exchange.load_markets()

        # Fetch balance
        balance = await exchange.fetch_balance({"type": "swap"})
        free_usdt = float(balance.get("free", {}).get("USDT", 0))
        total_usdt = float(balance.get("total", {}).get("USDT", 0))

        print("=" * 80)
        print("  VÉRIFICATION SIZING LIVE — Grid ATR Top 10")
        print("=" * 80)
        print(f"  Capital total configuré : {TOTAL_CAPITAL:.0f} USDT")
        print(f"  Balance Bitget          : {total_usdt:.2f} USDT (libre: {free_usdt:.2f})")
        print(f"  Leverage                : {LEVERAGE}x")
        print(f"  Nb assets               : {NB_ASSETS}")
        print(f"  Capital par asset       : {TOTAL_CAPITAL / NB_ASSETS:.0f} USDT")
        print("=" * 80)

        if total_usdt < TOTAL_CAPITAL:
            print(f"\n  ⚠ ATTENTION: Balance ({total_usdt:.2f}) < Capital configuré ({TOTAL_CAPITAL:.0f})")
            print(f"  Il faut transférer au moins {TOTAL_CAPITAL - total_usdt:.2f} USDT supplémentaires")

        # Fetch tickers pour prix actuels
        tickers = {}
        for symbol in TOP_10_ASSETS:
            futures_sym = f"{symbol}:USDT"
            try:
                ticker = await exchange.fetch_ticker(futures_sym)
                tickers[symbol] = float(ticker.get("last", 0))
            except Exception as e:
                print(f"  ERREUR ticker {futures_sym}: {e}")
                tickers[symbol] = 0.0

        print(f"\n{'Symbol':<14} {'Prix':>10} {'Levels':>6} {'Notional/Lvl':>14} "
              f"{'Min Qty':>10} {'Qty/Lvl':>10} {'Min Not.':>10} {'Status':>8}")
        print("-" * 96)

        all_ok = True
        total_margin_worst_case = 0.0

        for symbol in TOP_10_ASSETS:
            futures_sym = f"{symbol}:USDT"
            market = markets.get(futures_sym)
            price = tickers.get(symbol, 0)

            if market is None:
                print(f"{symbol:<14} {'MARCHÉ NON TROUVÉ':>70} {'NOK':>8}")
                all_ok = False
                continue

            if price <= 0:
                print(f"{symbol:<14} {'PRIX NON DISPONIBLE':>70} {'NOK':>8}")
                all_ok = False
                continue

            # Minimums du marché
            min_qty = market.get("limits", {}).get("amount", {}).get("min", 0) or 0
            min_notional = market.get("limits", {}).get("cost", {}).get("min", 0) or 0
            qty_precision = market.get("precision", {}).get("amount", 8)

            # Calcul sizing
            num_levels = LEVELS_PER_ASSET.get(symbol, 3)
            capital_per_asset = TOTAL_CAPITAL / NB_ASSETS
            margin_per_level = capital_per_asset / num_levels
            notional_per_level = margin_per_level * LEVERAGE
            qty_per_level = notional_per_level / price

            # Arrondir la quantité
            if isinstance(qty_precision, int):
                qty_rounded = round(qty_per_level, qty_precision)
            else:
                # qty_precision pourrait être un step (ex: 0.001)
                step = qty_precision
                qty_rounded = round(qty_per_level / step) * step

            # Vérifications
            issues = []
            if qty_rounded < min_qty:
                issues.append(f"qty {qty_rounded} < min {min_qty}")
            if min_notional > 0 and notional_per_level < min_notional:
                issues.append(f"notional {notional_per_level:.2f} < min {min_notional}")

            status = "OK" if not issues else "NOK"
            if issues:
                all_ok = False

            # Margin worst case (tous les levels ouverts)
            total_margin_worst_case += capital_per_asset

            print(f"{symbol:<14} {price:>10.4f} {num_levels:>6} "
                  f"${notional_per_level:>12.2f} "
                  f"{min_qty:>10g} {qty_rounded:>10g} "
                  f"{'$' + str(min_notional) if min_notional else 'N/A':>10} "
                  f"{status:>8}")

            if issues:
                for issue in issues:
                    print(f"{'':>14} └─ {issue}")

        print("-" * 96)
        print(f"\n  Margin worst case (tous levels ouverts) : "
              f"{total_margin_worst_case:.0f} USDT "
              f"({total_margin_worst_case / TOTAL_CAPITAL * 100:.0f}% du capital)")
        print(f"  Max margin ratio configuré : 70%")

        if total_margin_worst_case / TOTAL_CAPITAL > 0.70:
            print(f"  ⚠ Margin worst case ({total_margin_worst_case / TOTAL_CAPITAL * 100:.0f}%) "
                  f"> max_margin_ratio (70%)")
            print(f"    → En pratique ce n'est jamais atteint : toutes les grilles "
                  f"n'ouvrent pas tous les levels en même temps")

        print(f"\n  RÉSULTAT GLOBAL : {'✅ TOUS OK' if all_ok else '❌ CERTAINS NOK — voir détails'}")

        # Vérification du mapping to_futures_symbol()
        print("\n" + "=" * 80)
        print("  VÉRIFICATION MAPPING to_futures_symbol()")
        print("=" * 80)

        from backend.execution.executor import to_futures_symbol

        mapping_ok = True
        for symbol in TOP_10_ASSETS:
            try:
                mapped = to_futures_symbol(symbol)
                in_markets = mapped in markets
                status_sym = "✅" if in_markets else "⚠ pas dans markets"
                print(f"  {status_sym} {symbol} → {mapped}")
                if not in_markets:
                    mapping_ok = False
            except ValueError as e:
                print(f"  ❌ {symbol} → ERREUR: {e}")
                mapping_ok = False

        print(f"\n  MAPPING : {'✅ COMPLET' if mapping_ok else '❌ PROBLÈMES — voir détails'}")

    finally:
        await exchange.close()


if __name__ == "__main__":
    # Charger .env si présent
    from dotenv import load_dotenv
    load_dotenv()

    asyncio.run(main())
