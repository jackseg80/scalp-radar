"""Synchronise les trades Bitget historiques vers live_trades en DB.

Rattrapage pour les trades executés avant le Sprint 45.
Utilise ccxt Bitget fetchMyTrades() pour récupérer les fills,
classifie entry/close, et groupe en cycles de grille.

Lancement :
    uv run python -m scripts.sync_bitget_trades --purge --strategy grid_atr --days 30
    uv run python -m scripts.sync_bitget_trades --dry-run --strategy grid_atr --days 7

NOTE : utiliser --purge pour un re-sync propre. Sans --purge, les cycles
       dont les entries sont déjà en DB sont ignorés (sync incrémental).
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import groupby

import ccxt.pro as ccxtpro
from loguru import logger

from backend.core.config import get_config
from backend.core.database import Database
from backend.core.logging_setup import setup_logging


# Stratégies LONG-only : sell = toujours close
LONG_ONLY_STRATEGIES = {"grid_atr", "grid_boltrend", "grid_momentum"}

# Fenêtre de fusion des closes consécutifs (SL grid ferme N niveaux en N ordres rapides)
MERGE_WINDOW_MINUTES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync Bitget trades to live_trades DB")
    parser.add_argument("--strategy", type=str, default=None, help="Stratégie spécifique")
    parser.add_argument("--days", type=int, default=30, help="Nombre de jours à récupérer")
    parser.add_argument("--dry-run", action="store_true", help="Afficher sans insérer")
    parser.add_argument(
        "--purge", action="store_true",
        help="Purger les trades sync existants avant re-sync (recommandé)",
    )
    return parser.parse_args()


def _classify_trade(
    trade: dict,
    strategy: str | None,
) -> tuple[str, str]:
    """Classifie un fill Bitget en (direction, trade_type).

    Retourne (direction, trade_type) où :
    - direction: "LONG" ou "SHORT"
    - trade_type: "entry" ou "close"
    """
    side = trade.get("side", "buy")
    info = trade.get("info", {})

    # 1) Champs Bitget fiables
    trade_side = info.get("tradeSide", "")       # "open" ou "close"
    reduce_only = info.get("reduceOnly", False)
    if isinstance(reduce_only, str):
        reduce_only = reduce_only.lower() == "true"

    # 2) Déterminer si c'est un close
    is_close = trade_side == "close" or reduce_only

    # 3) Heuristique par stratégie pour LONG-only
    if not is_close and strategy in LONG_ONLY_STRATEGIES:
        # Pour LONG-only: sell = forcément un close
        if side == "sell":
            is_close = True

    # 4) Direction
    if is_close:
        # Close d'un LONG = sell, close d'un SHORT = buy
        direction = "LONG" if side == "sell" else "SHORT"
    else:
        # Entry LONG = buy, entry SHORT = sell
        direction = "LONG" if side == "buy" else "SHORT"

    # 5) Trade type
    trade_type = "close" if is_close else "entry"
    return direction, trade_type


def _aggregate_fills_by_order(fills: list[dict]) -> list[dict]:
    """Agrège les fills Bitget par order_id.

    Bitget retourne un fill par niveau de prix pour un seul ordre
    (ex: un close de 0.03 peut générer 3 fills à 0.01 chacun).
    On agrège : qty totale + prix VWAP + fees totales.
    Conserve le premier fill comme référence pour les métadonnées.
    """
    orders: dict[str, dict] = {}  # order_id → aggregated

    for fill in fills:
        oid = fill.get("order", "") or fill.get("id", "")
        qty = float(fill.get("amount", 0) or 0)
        price = float(fill.get("price", 0) or 0)
        fee = float((fill.get("fee") or {}).get("cost", 0) or 0)

        if oid not in orders:
            orders[oid] = {
                "_ref_fill": fill,
                "_total_qty": qty,
                "_total_cost": price * qty,
                "_total_fee": fee,
            }
        else:
            orders[oid]["_total_qty"] += qty
            orders[oid]["_total_cost"] += price * qty
            orders[oid]["_total_fee"] += fee

    result = []
    for oid, agg in orders.items():
        ref = agg["_ref_fill"]
        total_qty = agg["_total_qty"]
        total_cost = agg["_total_cost"]
        vwap = total_cost / total_qty if total_qty > 0 else 0.0

        aggregated = dict(ref)
        aggregated["amount"] = total_qty
        aggregated["price"] = vwap
        if aggregated.get("fee") is not None:
            aggregated["fee"] = {"cost": agg["_total_fee"]}
        aggregated["order"] = oid
        result.append(aggregated)

    return result


def _parse_ts(ts: str) -> datetime:
    """Parse un timestamp ISO 8601 en datetime UTC."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _flush_close_buffer(closes: list[dict]) -> dict:
    """Fusionne une liste de closes en un seul (VWAP + somme qty + somme fees)."""
    if len(closes) == 1:
        return closes[0]
    total_qty = sum(c["quantity"] for c in closes)
    total_cost = sum(c["price"] * c["quantity"] for c in closes)
    total_fee = sum(c.get("fee", 0) or 0 for c in closes)
    vwap = total_cost / total_qty if total_qty > 0 else 0.0
    merged = dict(closes[-1])  # timestamp le plus récent comme base
    merged["quantity"] = round(total_qty, 8)
    merged["price"] = round(vwap, 6)
    merged["fee"] = round(total_fee, 6)
    return merged


def _merge_close_bursts(
    records: list[dict],
    window_minutes: int = MERGE_WINDOW_MINUTES,
) -> list[dict]:
    """Fusionne les closes consécutifs proches dans le temps par (symbol, direction).

    Un SL grid peut fermer N niveaux en N ordres séparés en quelques secondes.
    Sans fusion : chaque close partiel ramène la position à 0 → N faux cycles.
    Avec fusion : N closes consécutifs sans entry entre eux, espacés de moins
    de window_minutes → 1 close agrégé (VWAP, somme qty, somme fees).
    """
    sorted_records = sorted(
        records,
        key=lambda r: (r["symbol"], r["direction"], r["timestamp"]),
    )

    result: list[dict] = []

    for _key, group in groupby(
        sorted_records, key=lambda r: (r["symbol"], r["direction"]),
    ):
        close_buffer: list[dict] = []

        for rec in group:
            if rec["trade_type"] == "entry":
                # Flush les closes en attente avant de traiter l'entry
                if close_buffer:
                    result.append(_flush_close_buffer(close_buffer))
                    close_buffer = []
                result.append(rec)
            else:
                # close : vérifier si proche du dernier dans le buffer
                if not close_buffer:
                    close_buffer.append(rec)
                else:
                    last_ts = _parse_ts(close_buffer[-1]["timestamp"])
                    cur_ts = _parse_ts(rec["timestamp"])
                    delta_min = (cur_ts - last_ts).total_seconds() / 60
                    if delta_min <= window_minutes:
                        close_buffer.append(rec)
                    else:
                        result.append(_flush_close_buffer(close_buffer))
                        close_buffer = [rec]

        # Flush restant
        if close_buffer:
            result.append(_flush_close_buffer(close_buffer))

    return result


def _group_into_cycles(
    records: list[dict],
    leverage: int = 3,
) -> list[dict]:
    """Groupe les entries et closes en cycles de grille.

    Un cycle commence au premier entry et se termine quand la position
    revient à ~0 après un ou plusieurs closes.

    Retourne :
    - Les entries individuelles (trade_type='entry', pnl=None) pour audit
    - UN record par cycle complet (trade_type='cycle_close', pnl=réel)
    - Les closes individuels sont absorbés dans le cycle_close (non stockés)
    """
    # Trier par (symbol, direction, timestamp) pour traitement séquentiel
    records.sort(key=lambda r: (r["symbol"], r["direction"], r["timestamp"]))

    result: list[dict] = []

    # État par (symbol, direction)
    state: dict[tuple[str, str], dict] = {}

    for rec in records:
        key = (rec["symbol"], rec["direction"])
        if key not in state:
            state[key] = {"qty": 0.0, "entries": [], "closes": [], "max_qty": 0.0}
        st = state[key]

        if rec["trade_type"] == "entry":
            st["qty"] += rec["quantity"]
            st["max_qty"] = max(st["max_qty"], st["qty"])
            st["entries"].append(rec)
            result.append(rec)  # Garder les entries pour audit

        else:  # close
            if st["qty"] <= 0:
                # Close sans entry correspondante → orphelin, ignorer
                logger.debug("Close orphelin ignoré : {} {} @ {}", rec["symbol"], rec["direction"], rec["price"])
                continue

            st["qty"] -= rec["quantity"]
            st["closes"].append(rec)

            # Tolérance : 2% de la position max OU résiduel < $0.50 en notionnel
            # (cas : micro-fills BTC de 0.0001 laissent un résidu dust)
            tolerance = max(1e-6, st["max_qty"] * 0.02)
            cur_price = rec.get("price", 0) or 0
            notional_residual = abs(st["qty"]) * cur_price if cur_price > 0 else float("inf")

            if abs(st["qty"]) <= tolerance or notional_residual < 0.50:
                # ── Cycle complet ─────────────────────────────────────────
                entries = st["entries"]
                closes = st["closes"]

                if entries and closes:
                    # Avg entry pondérée
                    total_entry_qty = sum(e["quantity"] for e in entries)
                    avg_entry = (
                        sum(e["price"] * e["quantity"] for e in entries) / total_entry_qty
                        if total_entry_qty > 0 else 0.0
                    )

                    # Avg close pondérée
                    total_close_qty = sum(c["quantity"] for c in closes)
                    avg_close = (
                        sum(c["price"] * c["quantity"] for c in closes) / total_close_qty
                        if total_close_qty > 0 else 0.0
                    )

                    # P&L brut
                    direction = rec["direction"]
                    if direction == "LONG":
                        pnl = (avg_close - avg_entry) * total_close_qty
                    else:
                        pnl = (avg_entry - avg_close) * total_close_qty

                    # Fees totales (entries + closes)
                    total_fees = sum(
                        r.get("fee", 0) or 0 for r in entries + closes
                    )
                    pnl = round(pnl - total_fees, 4)

                    # pnl_pct
                    lev = rec.get("leverage") or leverage
                    margin = avg_entry * total_close_qty / lev
                    pnl_pct = round(pnl / margin * 100, 2) if margin > 0 else None

                    first_entry = entries[0]
                    last_close = closes[-1]

                    cycle_close = {
                        "timestamp": last_close["timestamp"],
                        "strategy_name": rec["strategy_name"],
                        "symbol": rec["symbol"],
                        "direction": direction,
                        "trade_type": "cycle_close",
                        "side": "sell" if direction == "LONG" else "buy",
                        "quantity": round(total_close_qty, 8),
                        "price": round(avg_close, 4),
                        "order_id": f"cycle_{first_entry['order_id']}",
                        "fee": round(total_fees, 6),
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "leverage": lev,
                        "grid_level": len(entries),   # Nb d'entries dans ce cycle
                        "context": "sync_bitget_trades",
                    }
                    result.append(cycle_close)

                # Réinitialiser pour le prochain cycle
                state[key] = {"qty": 0.0, "entries": [], "closes": [], "max_qty": 0.0}

    # Cycles non fermés : logger uniquement
    for (sym, direction), st in state.items():
        if st["entries"] or st["closes"]:
            logger.info(
                "Cycle ouvert non clôturé : {} {} — {} entries, qty={:.6f}",
                sym, direction, len(st["entries"]), st["qty"],
            )

    return result


async def _fetch_all_trades(
    exchange: ccxtpro.Exchange,
    symbol: str,
    since: int,
) -> list[dict]:
    """Fetch tous les trades avec pagination (Bitget limite à 500/requête)."""
    all_trades: list[dict] = []
    current_since = since

    while True:
        try:
            batch = await exchange.fetch_my_trades(
                symbol, since=current_since, limit=500,
            )
        except Exception as e:
            logger.warning("Erreur fetch trades {}: {}", symbol, e)
            break

        if not batch:
            break

        all_trades.extend(batch)

        last_ts = batch[-1].get("timestamp", 0)
        if last_ts <= current_since:
            break
        current_since = last_ts + 1

        if len(batch) < 500:
            break

    return all_trades


async def main() -> None:
    args = parse_args()
    config = get_config()
    setup_logging(level="INFO")

    db = Database()
    await db.init()

    # Créer l'exchange Bitget
    if args.strategy:
        api_key, secret, passphrase = config.get_executor_keys(args.strategy)
    else:
        api_key = config.secrets.bitget_api_key
        secret = config.secrets.bitget_secret
        passphrase = config.secrets.bitget_passphrase

    exchange = ccxtpro.bitget({
        "apiKey": api_key,
        "secret": secret,
        "password": passphrase,
        "options": {"defaultType": "swap"},
    })

    try:
        await exchange.load_markets()

        # Purge si demandé
        if args.purge:
            count = await db.purge_live_trades(context="sync_bitget_trades")
            logger.info("Purgé {} trades sync existants", count)
        else:
            logger.warning(
                "Sans --purge, les cycles dont les entries sont déjà en DB seront ignorés.",
            )

        since = int(
            (datetime.now(tz=timezone.utc) - timedelta(days=args.days)).timestamp() * 1000,
        )

        # order_ids déjà en DB (pour dédup des entries)
        existing = await db.get_live_trades(period="all", limit=10000)
        existing_order_ids = {t["order_id"] for t in existing if t.get("order_id")}

        # Symbols des assets configurés
        symbols = []
        for asset in config.assets:
            futures_sym = f"{asset.symbol}:USDT"
            if futures_sym in exchange.markets:
                symbols.append(futures_sym)

        # Leverage par stratégie
        strategy_name = args.strategy or "unknown"
        strategy_config = getattr(config.strategies, strategy_name, None)
        leverage = getattr(strategy_config, "leverage", None) or 3

        # ── Phase 1 : Fetch + Aggregate + Classify ───────────────────────
        raw_records: list[dict] = []

        for symbol in symbols:
            logger.info("Fetching trades for {} since {} days...", symbol, args.days)
            fills = await _fetch_all_trades(exchange, symbol, since)
            orders = _aggregate_fills_by_order(fills)
            logger.info("  {} fills → {} ordres pour {}", len(fills), len(orders), symbol)

            for trade in orders:
                order_id = trade.get("order", "") or trade.get("id", "")
                # Dédup : si cet order est déjà en DB (entry existante), on ignore
                if order_id in existing_order_ids:
                    continue

                direction, trade_type = _classify_trade(trade, args.strategy)
                side = trade.get("side", "buy")
                fee_cost = float((trade.get("fee") or {}).get("cost", 0) or 0)

                raw_records.append({
                    "timestamp": trade.get(
                        "datetime", datetime.now(tz=timezone.utc).isoformat(),
                    ),
                    "strategy_name": strategy_name,
                    "symbol": symbol,
                    "direction": direction,
                    "trade_type": trade_type,
                    "side": side,
                    "quantity": float(trade.get("amount", 0)),
                    "price": float(trade.get("price", 0)),
                    "order_id": order_id,
                    "fee": fee_cost,
                    "pnl": None,
                    "pnl_pct": None,
                    "leverage": leverage,
                    "grid_level": None,
                    "context": "sync_bitget_trades",
                })

        n_raw_entries = sum(1 for r in raw_records if r["trade_type"] == "entry")
        n_raw_closes = sum(1 for r in raw_records if r["trade_type"] != "entry")
        logger.info(
            "Bruts : {} entries + {} closes = {} ordres",
            n_raw_entries, n_raw_closes, len(raw_records),
        )

        # ── Phase 1b : Fusionner les closes en rafale (SL multi-niveaux) ──
        n_raw_closes_before = sum(1 for r in raw_records if r["trade_type"] != "entry")
        merged_records = _merge_close_bursts(raw_records, window_minutes=MERGE_WINDOW_MINUTES)
        n_merged_closes = sum(1 for r in merged_records if r["trade_type"] != "entry")
        if n_raw_closes_before != n_merged_closes:
            logger.info(
                "Merge bursts : {} closes → {} (fusionnés en fenêtre {}min)",
                n_raw_closes_before, n_merged_closes, MERGE_WINDOW_MINUTES,
            )

        # ── Phase 2 : Grouper en cycles ───────────────────────────────────
        result_records = _group_into_cycles(merged_records, leverage=leverage)

        n_entries = sum(1 for r in result_records if r["trade_type"] == "entry")
        n_cycles = sum(1 for r in result_records if r["trade_type"] == "cycle_close")
        logger.info(
            "Cycles : {} entries (audit) + {} cycle_close",
            n_entries, n_cycles,
        )

        # ── Phase 3 : Insérer en DB ──────────────────────────────────────
        total_inserted = 0
        total_pnl = 0.0

        for record in result_records:
            # Dédup au niveau cycle_close aussi
            if record["order_id"] in existing_order_ids:
                continue

            if args.dry_run:
                pnl_str = f"  P&L={record['pnl']:.2f}$" if record["pnl"] is not None else ""
                logger.info(
                    "[DRY-RUN] {} {} {} @ {:.4f}{} ({})",
                    record["trade_type"], record["direction"],
                    record["symbol"], record["price"],
                    pnl_str, record["order_id"],
                )
            else:
                await db.insert_live_trade(record)
                existing_order_ids.add(record["order_id"])

            total_inserted += 1
            if record["trade_type"] == "cycle_close" and record["pnl"] is not None:
                total_pnl += record["pnl"]

        logger.info("─" * 60)
        logger.info(
            "Sync terminée : {} records insérés ({} entries + {} cycles)",
            total_inserted,
            sum(1 for r in result_records if r["trade_type"] == "entry"),
            n_cycles,
        )
        logger.info("P&L total : {:.2f} USDT", total_pnl)

        # Résumé par symbol (cycles seulement)
        pnl_by_sym: dict[str, float] = defaultdict(float)
        wins_by_sym: dict[str, int] = defaultdict(int)
        cycles_by_sym: dict[str, int] = defaultdict(int)
        for r in result_records:
            if r["trade_type"] == "cycle_close" and r["pnl"] is not None:
                sym = r["symbol"]
                pnl_by_sym[sym] += r["pnl"]
                cycles_by_sym[sym] += 1
                if r["pnl"] > 0:
                    wins_by_sym[sym] += 1

        for sym in sorted(pnl_by_sym):
            total_c = cycles_by_sym[sym]
            wr = round(wins_by_sym[sym] / total_c * 100) if total_c else 0
            logger.info(
                "  {} : {} cycles, WR {}%, P&L {:.2f}",
                sym, total_c, wr, pnl_by_sym[sym],
            )

    finally:
        await exchange.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
