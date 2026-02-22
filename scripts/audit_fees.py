"""Audit #4 --- Comparaison fees reelles Bitget vs modele backtest.

Usage:
    uv run python -m scripts.audit_fees
    uv run python -m scripts.audit_fees --days 7 -v
"""

import argparse
import sqlite3
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
    order_type: str     # market / limit / unknown
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


def fetch_all_trades(
    exchange: ccxt.bitget,
    symbols: list[str],
    since_ms: int,
    verbose: bool = False,
) -> list[TradeAudit]:
    """Fetch tous les trades pour les symbols actifs."""
    result: list[TradeAudit] = []

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
                fee_obj = t.get("fee") or {}
                fee_cost = float(fee_obj.get("cost", 0) or 0)
                fee_cur = fee_obj.get("currency", "USDT") or "USDT"
                price = float(t["price"])
                amount = float(t["amount"])
                notional = float(t.get("cost", 0) or 0) or price * amount

                result.append(TradeAudit(
                    symbol=symbol,
                    order_id=t.get("order", "") or "",
                    trade_id=str(t["id"]),
                    side=t["side"],
                    order_type=(t.get("type") or "unknown").lower(),
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


def fetch_simulation_trades(days: int) -> list[dict]:
    """Trades simulation depuis la DB locale (pour comparaison slippage)."""
    db_path = Path("data/scalp_radar.db")
    if not db_path.exists():
        return []

    since = datetime.now(timezone.utc) - timedelta(days=days)
    since_str = since.strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT symbol, direction, entry_price, exit_price,
                      entry_time, exit_time, strategy_name, exit_reason
               FROM simulation_trades
               WHERE entry_time >= ?
               ORDER BY entry_time""",
            (since_str,),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"DB simulation_trades: {e}")
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    """Format a rate as percentage string."""
    return f"{v * 100:.4f}%"


def print_report(
    trades: list[TradeAudit],
    config,
    sim_trades: list[dict],
    days: int,
) -> None:
    W = 65

    # Model values (already in %, e.g. 0.02 means 0.02%)
    maker_pct = config.risk.fees.maker_percent        # 0.02
    taker_pct = config.risk.fees.taker_percent         # 0.06
    slip_pct = config.risk.slippage.default_estimate_percent  # 0.05

    maker_rate = maker_pct / 100   # 0.0002
    taker_rate = taker_pct / 100   # 0.0006
    slip_rate = slip_pct / 100     # 0.0005

    # --- Header ---
    ts_range = [t.timestamp for t in trades]
    date_from = datetime.fromtimestamp(min(ts_range) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    date_to = datetime.fromtimestamp(max(ts_range) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    symbols_seen = sorted({t.symbol for t in trades})

    market = [t for t in trades if t.order_type == "market"]
    limit_ = [t for t in trades if t.order_type == "limit"]
    other = [t for t in trades if t.order_type not in ("market", "limit")]

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
    print(f"    - Market orders     : {len(market)}")
    print(f"    - Limit orders      : {len(limit_)}")
    if other:
        print(f"    - Autres            : {len(other)} ({', '.join({t.order_type for t in other})})")
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

    if market:
        rates = [t.fee_rate * 100 for t in market]
        avg = mean(rates)
        med = median(rates)
        ecart = ((avg - taker_pct) / taker_pct) * 100
        print(f"  {'Market':<15}| {taker_pct:>7.3f}% | {avg:>9.4f}% | {med:>9.4f}% | {ecart:>+6.0f}%")
    else:
        print(f"  {'Market':<15}| {taker_pct:>7.3f}% | {'N/A':>10} | {'N/A':>10} | {'N/A':>7}")

    if limit_:
        rates = [t.fee_rate * 100 for t in limit_]
        avg = mean(rates)
        med = median(rates)
        ecart = ((avg - maker_pct) / maker_pct) * 100
        print(f"  {'Limit':<15}| {maker_pct:>7.3f}% | {avg:>9.4f}% | {med:>9.4f}% | {ecart:>+6.0f}%")
    else:
        print(f"  {'Limit':<15}| {maker_pct:>7.3f}% | {'N/A':>10} | {'N/A':>10} | {'N/A':>7}")

    print()

    # Global fee totals
    total_fee_real = sum(t.fee_paid for t in trades)
    total_fee_model = sum(
        t.notional * (taker_rate if t.order_type == "market" else maker_rate)
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
            t.notional * (taker_rate if t.order_type == "market" else maker_rate)
            for t in st
        )
        short = sym.replace("/USDT", "")
        print(f"  {short:<14}| {len(st):>5} | {avg_fee:>9.4f}% | {notional:>11,.2f} $ | {fee_real - fee_model:>+8.4f} $")

    # === SLIPPAGE ===
    print()
    print(f"  --- SLIPPAGE {'-' * (W - 16)}")
    print()

    if not sim_trades:
        print("  Aucun trade simulation en DB pour comparaison slippage.")
        print(f"  Modele backtest : {slip_pct:.2f}%")
    else:
        # Match Bitget fills <-> simulation trades by symbol + time window (5 min)
        matched_slippages: dict[str, list[float]] = {}
        used_sim: set[int] = set()  # indices already matched

        for bt in trades:
            bt_time = datetime.fromtimestamp(bt.timestamp / 1000, tz=timezone.utc)

            for idx, st in enumerate(sim_trades):
                if idx in used_sim:
                    continue
                if st["symbol"] != bt.symbol:
                    continue

                try:
                    raw_t = st["entry_time"]
                    st_time = datetime.fromisoformat(raw_t.replace("Z", "+00:00"))
                    if st_time.tzinfo is None:
                        st_time = st_time.replace(tzinfo=timezone.utc)
                except Exception:
                    continue

                if abs((bt_time - st_time).total_seconds()) < 300:
                    signal_price = st["entry_price"]
                    if signal_price and signal_price > 0:
                        slip = abs(bt.fill_price - signal_price) / signal_price * 100
                        matched_slippages.setdefault(bt.symbol, []).append(slip)
                        used_sim.add(idx)
                    break

        total_matched = sum(len(v) for v in matched_slippages.values())

        if total_matched > 0:
            print(f"  Trades matches sim<->Bitget : {total_matched}")
            print()
            print(f"  {'Asset':<14}| {'Match':>5} | {'Slip. moy':>10} | {'Slip. max':>10} | {'Modele':>8}")
            print(f"  {'':->13}|{'':->7}|{'':->12}|{'':->12}|{'':->10}")

            all_slips: list[float] = []
            for sym in sorted(matched_slippages):
                slips = matched_slippages[sym]
                all_slips.extend(slips)
                short = sym.replace("/USDT", "")
                print(
                    f"  {short:<14}| {len(slips):>5} | "
                    f"{mean(slips):>9.3f}% | {max(slips):>9.3f}% | "
                    f"{slip_pct:>7.2f}%"
                )

            print()
            avg_slip = mean(all_slips)
            slip_ecart = ((avg_slip / 100 - slip_rate) / slip_rate * 100) if slip_rate else 0
            print(f"  Slippage moyen global  : {avg_slip:.3f}%")
            print(f"  Modele backtest        : {slip_pct:.2f}%")
            print(f"  Ecart                  : {slip_ecart:+.0f}%")
        else:
            print("  Aucun match trouve entre trades sim et Bitget (fenetre 5 min).")
            print(f"  Modele backtest slippage : {slip_pct:.2f}%")

    # === DETAIL FILLS (verbose) ===
    # skipped here, verbose details logged during fetch

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
        if market:
            rec = mean(t.fee_rate * 100 for t in market)
            print(f"      Recommandation : augmenter taker_fee a {rec:.3f}% dans risk.yaml")
        if limit_:
            rec = mean(t.fee_rate * 100 for t in limit_)
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

    trades = fetch_all_trades(exchange, symbols, since_ms, args.verbose)
    logger.info(f"{len(trades)} fills recuperes")

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
                    f"    {t.datetime_str}  {short:<6} {t.side:<4} {t.order_type:<7} "
                    f"notional={t.notional:.2f}$  fee={t.fee_paid:.4f}$  "
                    f"rate={t.fee_rate * 100:.4f}%"
                )
        print()
        return

    # 5. Simulation trades for slippage
    sim_trades = fetch_simulation_trades(args.days)
    if sim_trades:
        logger.info(f"{len(sim_trades)} trades simulation en DB pour slippage")

    # 6. Report
    print_report(trades, config, sim_trades, args.days)


if __name__ == "__main__":
    main()
