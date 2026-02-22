"""Audit #4 --- Comparaison fees reelles Bitget vs modele backtest.

Usage:
    uv run python -m scripts.audit_fees
    uv run python -m scripts.audit_fees --days 7 -v
    uv run python -m scripts.audit_fees --debug       # dump 3 raw trades JSON
"""

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median

import ccxt
import yaml
from loguru import logger


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TradeAudit:
    symbol: str
    order_id: str
    trade_id: str
    side: str           # buy / sell
    order_type: str     # market / limit / unknown (from ccxt type field)
    taker_or_maker: str # taker / maker / unknown (from ccxt takerOrMaker)
    notional: float     # price x quantity (USDT)
    fee_paid: float     # fee absolue USDT
    fee_rate: float     # fee_paid / notional
    fill_price: float
    timestamp: int      # ms epoch
    datetime_str: str
    fee_currency: str   # USDT normally


# ---------------------------------------------------------------------------
# Bitget connection
# ---------------------------------------------------------------------------

def create_exchange(config) -> ccxt.bitget:
    """Connexion read-only Bitget swap (sync)."""
    exchange = ccxt.bitget({
        "apiKey": config.secrets.bitget_api_key,
        "secret": config.secrets.bitget_secret,
        "password": config.secrets.bitget_passphrase,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    exchange.load_markets()
    return exchange


def _resolve_swap_symbol(exchange: ccxt.bitget, symbol: str) -> str | None:
    """BTC/USDT -> BTC/USDT:USDT for Bitget swap."""
    swap = f"{symbol}:USDT"
    if swap in exchange.markets:
        return swap
    if symbol in exchange.markets:
        return symbol
    return None


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def get_active_symbols() -> list[str]:
    """Assets actifs depuis strategies.yaml (per_asset des strategies enabled)."""
    path = Path("config/strategies.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)

    symbols: set[str] = set()
    for strat_name in ("grid_atr", "grid_boltrend"):
        strat = data.get(strat_name, {})
        if strat.get("enabled", False):
            symbols.update(strat.get("per_asset", {}).keys())
    return sorted(symbols)


def fetch_trades_for_symbol(
    exchange: ccxt.bitget,
    swap_symbol: str,
    since_ms: int,
    limit: int = 100,
) -> list[dict]:
    """Fetch avec pagination (Bitget retourne max 100/page)."""
    all_trades: list[dict] = []
    cursor = since_ms
    while True:
        batch = exchange.fetch_my_trades(swap_symbol, since=cursor, limit=limit)
        if not batch:
            break
        all_trades.extend(batch)
        if len(batch) < limit:
            break
        cursor = batch[-1]["timestamp"] + 1
        time.sleep(0.3)
    return all_trades


def _detect_taker_or_maker(trade: dict) -> str:
    """Detect taker/maker from ccxt unified trade structure.

    Priority:
    1. ccxt ``takerOrMaker`` field (most reliable)
    2. Bitget raw ``info.side`` containing 'Taker'/'Maker'
    3. Fallback to 'taker' (conservative)
    """
    # 1. Standard ccxt field
    tom = trade.get("takerOrMaker")
    if tom in ("taker", "maker"):
        return tom

    # 2. Bitget raw info
    info = trade.get("info") or {}
    raw_side = str(info.get("side", ""))
    if "maker" in raw_side.lower():
        return "maker"
    if "taker" in raw_side.lower():
        return "taker"

    # 3. Conservative fallback
    return "taker"


def fetch_all_trades(
    exchange: ccxt.bitget,
    symbols: list[str],
    since_ms: int,
    verbose: bool = False,
    debug: bool = False,
) -> list[TradeAudit]:
    """Fetch tous les trades pour les symbols actifs."""
    result: list[TradeAudit] = []
    debug_dumped = 0

    for symbol in symbols:
        swap = _resolve_swap_symbol(exchange, symbol)
        if not swap:
            if verbose:
                logger.warning(f"{symbol}: introuvable sur Bitget swap, ignore")
            continue

        try:
            raw = fetch_trades_for_symbol(exchange, swap, since_ms)
            if verbose:
                logger.info(f"{symbol}: {len(raw)} fills")

            for t in raw:
                # --debug : dump first 3 raw trades
                if debug and debug_dumped < 3:
                    print(f"\n  === RAW TRADE #{debug_dumped + 1} ({symbol}) ===")
                    print(json.dumps(t, indent=2, default=str))
                    debug_dumped += 1

                fee_obj = t.get("fee") or {}
                fee_cost = float(fee_obj.get("cost", 0) or 0)
                fee_cur = fee_obj.get("currency", "USDT") or "USDT"
                price = float(t["price"])
                amount = float(t["amount"])
                notional = float(t.get("cost", 0) or 0) or price * amount
                taker_or_maker = _detect_taker_or_maker(t)

                result.append(TradeAudit(
                    symbol=symbol,
                    order_id=t.get("order", "") or "",
                    trade_id=str(t["id"]),
                    side=t["side"],
                    order_type=(t.get("type") or "unknown").lower(),
                    taker_or_maker=taker_or_maker,
                    notional=notional,
                    fee_paid=fee_cost,
                    fee_rate=fee_cost / notional if notional > 0 else 0.0,
                    fill_price=price,
                    timestamp=t["timestamp"],
                    datetime_str=t.get("datetime", ""),
                    fee_currency=fee_cur,
                ))
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"{symbol}: erreur fetch -- {e}")

    return result


def _model_rate(t: TradeAudit, taker_rate: float, maker_rate: float) -> float:
    """Return the model fee rate for a trade based on taker/maker detection."""
    if t.taker_or_maker == "maker":
        return maker_rate
    # taker or unknown -> conservative (taker)
    return taker_rate


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    """Format a rate as percentage string."""
    return f"{v * 100:.4f}%"


def print_report(trades: list[TradeAudit], config) -> None:
    W = 65

    # Model values (already in %, e.g. 0.02 means 0.02%)
    maker_pct = config.risk.fees.maker_percent        # 0.02
    taker_pct = config.risk.fees.taker_percent         # 0.06
    slip_pct = config.risk.slippage.default_estimate_percent  # 0.05

    maker_rate = maker_pct / 100   # 0.0002
    taker_rate = taker_pct / 100   # 0.0006

    # --- Header ---
    ts_range = [t.timestamp for t in trades]
    date_from = datetime.fromtimestamp(min(ts_range) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    date_to = datetime.fromtimestamp(max(ts_range) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    symbols_seen = sorted({t.symbol for t in trades})

    takers = [t for t in trades if t.taker_or_maker == "taker"]
    makers = [t for t in trades if t.taker_or_maker == "maker"]
    unknowns = [t for t in trades if t.taker_or_maker not in ("taker", "maker")]

    # Check for non-USDT fee currencies
    non_usdt = {t.fee_currency for t in trades if t.fee_currency != "USDT"}

    by_symbol: dict[str, list[TradeAudit]] = {}
    for t in trades:
        by_symbol.setdefault(t.symbol, []).append(t)

    print()
    print("=" * W)
    print("  AUDIT FEES -- Modele backtest vs realite Bitget")
    print("=" * W)
    print()
    print(f"  Periode analysee      : {date_from} -> {date_to}")
    print(f"  Trades (fills) total  : {len(trades)}")
    print(f"  Assets couverts       : {len(symbols_seen)}")
    print(f"    - Taker fills       : {len(takers)}")
    print(f"    - Maker fills       : {len(makers)}")
    if unknowns:
        print(f"    - Inconnu (->taker) : {len(unknowns)}")
    if non_usdt:
        print(f"  !! Fees en devise non-USDT detectees : {non_usdt}")

    # === FEES ===
    print()
    print(f"  --- FEES {'-' * (W - 12)}")
    print()

    header = f"  {'Type':<15}| {'Modele':>8} | {'Reel moy':>10} | {'Reel med':>10} | {'Ecart':>7}"
    sep = f"  {'':->14}|{'':->10}|{'':->12}|{'':->12}|{'':->9}"
    print(header)
    print(sep)

    if takers:
        rates = [t.fee_rate * 100 for t in takers]
        avg = mean(rates)
        med = median(rates)
        ecart = ((avg - taker_pct) / taker_pct) * 100
        print(f"  {'Taker':<15}| {taker_pct:>7.3f}% | {avg:>9.4f}% | {med:>9.4f}% | {ecart:>+6.0f}%")
    else:
        print(f"  {'Taker':<15}| {taker_pct:>7.3f}% | {'N/A':>10} | {'N/A':>10} | {'N/A':>7}")

    if makers:
        rates = [t.fee_rate * 100 for t in makers]
        avg = mean(rates)
        med = median(rates)
        ecart = ((avg - maker_pct) / maker_pct) * 100
        print(f"  {'Maker':<15}| {maker_pct:>7.3f}% | {avg:>9.4f}% | {med:>9.4f}% | {ecart:>+6.0f}%")
    else:
        print(f"  {'Maker':<15}| {maker_pct:>7.3f}% | {'N/A':>10} | {'N/A':>10} | {'N/A':>7}")

    print()

    # Global fee totals — model uses taker_rate or maker_rate based on detection
    total_fee_real = sum(t.fee_paid for t in trades)
    total_fee_model = sum(
        t.notional * _model_rate(t, taker_rate, maker_rate)
        for t in trades
    )
    total_notional = sum(t.notional for t in trades)
    fee_ecart_global = (
        (total_fee_real - total_fee_model) / total_fee_model * 100
        if total_fee_model > 0
        else 0
    )

    print(f"  Fee totale payee      : {total_fee_real:.4f} USDT")
    print(f"  Fee modele estimee    : {total_fee_model:.4f} USDT  (sur memes trades)")
    print(f"  Notional total        : {total_notional:,.2f} USDT")
    delta_label = (
        "(modele surevalue)" if fee_ecart_global < 0
        else "(modele sous-evalue)" if fee_ecart_global > 0
        else "(aligne)"
    )
    print(f"  Ecart global          : {fee_ecart_global:+.1f}% {delta_label}")

    # === PAR ASSET ===
    print()
    print(f"  --- PAR ASSET {'-' * (W - 17)}")
    print()
    print(f"  {'Asset':<14}| {'Fills':>5} | {'Fee moy':>10} | {'Notional':>12} | {'Ecart $':>9}")
    print(f"  {'':->13}|{'':->7}|{'':->12}|{'':->14}|{'':->11}")

    for sym in sorted(by_symbol):
        st = by_symbol[sym]
        avg_fee = mean(t.fee_rate * 100 for t in st)
        notional = sum(t.notional for t in st)
        fee_real = sum(t.fee_paid for t in st)
        fee_model = sum(
            t.notional * _model_rate(t, taker_rate, maker_rate)
            for t in st
        )
        short = sym.replace("/USDT", "")
        print(f"  {short:<14}| {len(st):>5} | {avg_fee:>9.4f}% | {notional:>11,.2f} $ | {fee_real - fee_model:>+8.4f} $")

    # === SLIPPAGE ===
    print()
    print(f"  --- SLIPPAGE {'-' * (W - 16)}")
    print()
    print(f"  Modele backtest       : {slip_pct:.2f}%")
    print("  Non mesurable avec les donnees actuelles.")
    print("  Methode recommandee : comparer les logs Executor")
    print("  (lignes 'slippage detected') sur 2-4 semaines.")

    # === VERDICT ===
    print()
    print(f"  --- VERDICT {'-' * (W - 15)}")
    print()

    if abs(fee_ecart_global) < 10:
        print(f"  OK  Le modele backtest est ALIGNE avec les fees reelles (ecart {fee_ecart_global:+.1f}%).")
        print("      Aucune correction necessaire.")
    elif fee_ecart_global < -10:
        print(f"  OK  Le modele backtest SUREVALUE les fees de {abs(fee_ecart_global):.0f}%.")
        print("      Les resultats reels sont MEILLEURS que le backtest.")
        print("      Approche conservatrice, aucune correction necessaire.")
    else:
        print(f"  !!  Le modele backtest SOUS-EVALUE les fees de {fee_ecart_global:.0f}%.")
        estimated_impact = total_fee_real - total_fee_model
        if estimated_impact > 0:
            print(f"      Impact estime (sur {len(trades)} fills) : {estimated_impact:+.4f} USDT")
        if takers:
            rec = mean(t.fee_rate * 100 for t in takers)
            print(f"      Recommandation : augmenter taker_fee a {rec:.3f}% dans risk.yaml")
        if makers:
            rec = mean(t.fee_rate * 100 for t in makers)
            print(f"      Recommandation : ajuster maker_fee a {rec:.3f}% dans risk.yaml")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Audit #4 -- Comparaison fees reelles Bitget vs modele backtest"
    )
    parser.add_argument("--days", type=int, default=30, help="Jours d'historique (defaut 30)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Details par symbol")
    parser.add_argument("--debug", action="store_true", help="Dump 3 premiers trades bruts (JSON)")
    args = parser.parse_args()

    from backend.core.config import get_config

    config = get_config()

    if not config.secrets.bitget_api_key:
        logger.error("BITGET_API_KEY manquant dans .env")
        return

    # 1. Connect
    logger.info("Connexion Bitget (swap)...")
    exchange = create_exchange(config)
    logger.info(f"Connecte. {len(exchange.markets)} marches charges.")

    # 2. Active symbols
    symbols = get_active_symbols()
    logger.info(
        f"{len(symbols)} assets actifs : "
        + ", ".join(s.replace("/USDT", "") for s in symbols)
    )

    # 3. Fetch trades
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    since_ms = int(since.timestamp() * 1000)
    logger.info(f"Fetch trades depuis {since.strftime('%Y-%m-%d')} ({args.days}j)...")

    trades = fetch_all_trades(exchange, symbols, since_ms, args.verbose, args.debug)
    logger.info(f"{len(trades)} fills recuperes")

    if args.debug:
        # Distribution taker/maker
        from collections import Counter
        tom_counts = Counter(t.taker_or_maker for t in trades)
        type_counts = Counter(t.order_type for t in trades)
        logger.info(f"takerOrMaker: {dict(tom_counts)}")
        logger.info(f"order_type:   {dict(type_counts)}")

    # 4. Minimum sample check
    if len(trades) < 10:
        print(f"\n  Seulement {len(trades)} fills trouves sur {args.days} jours.")
        print("  Echantillon insuffisant pour un audit fiable.")
        print("  Reprendre l'audit quand > 50 trades accumules.\n")
        if trades:
            print("  Fills trouves :")
            for t in trades:
                short = t.symbol.replace("/USDT", "")
                print(
                    f"    {t.datetime_str}  {short:<6} {t.side:<4} "
                    f"{t.taker_or_maker:<6} "
                    f"notional={t.notional:.2f}$  fee={t.fee_paid:.4f}$  "
                    f"rate={t.fee_rate * 100:.4f}%"
                )
        print()
        return

    # 5. Report
    print_report(trades, config)


if __name__ == "__main__":
    main()
