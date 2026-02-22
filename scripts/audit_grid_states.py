"""Audit #5 --- Grid states vs Bitget : coherence en temps reel.

Compare l'etat local (executor_state.json ou API) avec les positions
reelles sur Bitget (fetch_positions) et reporte toute divergence.

Usage:
    uv run python -m scripts.audit_grid_states
    uv run python -m scripts.audit_grid_states -v
    uv run python -m scripts.audit_grid_states --mode api
    uv run python -m scripts.audit_grid_states --state-file /path/to/executor_state.json
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import ccxt
from loguru import logger

# ---------------------------------------------------------------------------
# Config loader (minimal, same pattern as audit_fees.py)
# ---------------------------------------------------------------------------

def _load_config():
    """Load config via backend config loader."""
    from backend.core.config import get_config
    return get_config()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LocalPosition:
    symbol: str
    direction: str
    strategy: str
    num_levels: int
    total_quantity: float
    avg_entry: float
    sl_price: float
    sl_order_id: str | None
    leverage: int
    opened_at: str

@dataclass
class BitgetPosition:
    symbol: str
    side: str
    contracts: float
    entry_price: float
    unrealized_pnl: float
    leverage: int
    margin_mode: str
    liquidation_price: float | None

@dataclass
class SLOrder:
    order_id: str
    symbol: str
    trigger_price: float
    side: str
    order_type: str

@dataclass
class Divergence:
    severity: str       # "FANTOME", "ORPHELINE", "DESYNC", "SL_MANQUANT", "SL_ORPHELIN"
    symbol: str
    message: str
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 1 --- Lire l'etat local
# ---------------------------------------------------------------------------

def read_local_from_file(state_file: str) -> tuple[dict, list[LocalPosition]]:
    """Lit grid_states depuis executor_state.json."""
    path = Path(state_file)
    if not path.exists():
        logger.error(f"Fichier introuvable : {path}")
        return {}, []

    with open(path) as f:
        state = json.load(f)

    meta = {
        "mode": "file",
        "file": str(path),
        "last_modified": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
    }

    grid_states = state.get("grid_states", {})
    positions = []
    for _sym, gs in grid_states.items():
        levels = gs.get("positions", [])
        total_qty = sum(p.get("quantity", 0) for p in levels)
        if total_qty == 0 and not levels:
            continue
        total_notional = sum(
            p.get("quantity", 0) * p.get("entry_price", 0) for p in levels
        )
        avg_entry = total_notional / total_qty if total_qty > 0 else 0.0
        positions.append(LocalPosition(
            symbol=gs["symbol"],
            direction=gs.get("direction", "?"),
            strategy=gs.get("strategy_name", "?"),
            num_levels=len(levels),
            total_quantity=total_qty,
            avg_entry=avg_entry,
            sl_price=gs.get("sl_price", 0.0),
            sl_order_id=gs.get("sl_order_id"),
            leverage=gs.get("leverage", 6),
            opened_at=gs.get("opened_at", "?"),
        ))

    return meta, positions


def read_local_from_api(base_url: str = "http://localhost:8000") -> tuple[dict, list[LocalPosition]]:
    """Lit les positions grid depuis l'API executor."""
    import httpx

    url = f"{base_url}/api/executor/status"
    try:
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"API inaccessible ({url}): {e}")
        return {"mode": "api", "url": url, "error": str(e)}, []

    data = resp.json()
    meta = {"mode": "api", "url": url}

    positions = []
    for pos in data.get("positions", []):
        if pos.get("type") != "grid":
            continue
        sub_positions = pos.get("positions", [])
        total_qty = sum(p.get("quantity", 0) for p in sub_positions)
        total_notional = sum(
            p.get("quantity", 0) * p.get("entry_price", 0) for p in sub_positions
        )
        avg_entry = total_notional / total_qty if total_qty > 0 else 0.0
        positions.append(LocalPosition(
            symbol=pos["symbol"],
            direction=pos.get("direction", "?"),
            strategy=pos.get("strategy_name", "?"),
            num_levels=pos.get("levels", len(sub_positions)),
            total_quantity=total_qty,
            avg_entry=avg_entry,
            sl_price=pos.get("sl_price", 0.0),
            sl_order_id=pos.get("sl_order_id"),
            leverage=pos.get("leverage", 6),
            opened_at=pos.get("entry_time", "?"),
        ))

    return meta, positions


# ---------------------------------------------------------------------------
# Phase 2 --- Lire l'etat Bitget
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


def fetch_bitget_positions(exchange: ccxt.bitget) -> list[BitgetPosition]:
    """Recupere toutes les positions ouvertes sur Bitget."""
    raw = exchange.fetch_positions()
    positions = []
    for p in raw:
        contracts = float(p.get("contracts") or 0)
        entry_price = float(p.get("entryPrice") or 0)
        if contracts <= 0 or entry_price <= 0:
            continue
        positions.append(BitgetPosition(
            symbol=p.get("symbol", "?"),
            side=p.get("side", "?"),
            contracts=contracts,
            entry_price=entry_price,
            unrealized_pnl=float(p.get("unrealizedPnl") or 0),
            leverage=int(float(p.get("leverage") or 1)),
            margin_mode=p.get("marginMode", "?"),
            liquidation_price=float(lp) if (lp := p.get("liquidationPrice")) else None,
        ))
    return positions


# ---------------------------------------------------------------------------
# Phase 3 --- Ordres SL ouverts sur Bitget
# ---------------------------------------------------------------------------

def fetch_sl_orders(exchange: ccxt.bitget, symbols: list[str], verbose: bool = False) -> list[SLOrder]:
    """Recupere les ordres stop-loss ouverts pour les symbols donnes."""
    all_orders: list[SLOrder] = []
    for sym in symbols:
        try:
            orders = exchange.fetch_open_orders(sym)
            for o in orders:
                trigger = o.get("triggerPrice") or o.get("stopPrice") or 0
                if float(trigger) > 0:
                    all_orders.append(SLOrder(
                        order_id=o.get("id", "?"),
                        symbol=sym,
                        trigger_price=float(trigger),
                        side=o.get("side", "?"),
                        order_type=o.get("type", "?"),
                    ))
            if verbose:
                logger.debug(f"  {sym}: {len(orders)} ordres ouverts")
        except Exception as e:
            logger.warning(f"Erreur fetch_open_orders({sym}): {e}")
        time.sleep(0.3)
    return all_orders


# ---------------------------------------------------------------------------
# Phase 4 --- Comparaison
# ---------------------------------------------------------------------------

PRICE_TOLERANCE = 0.005   # 0.5%
QTY_TOLERANCE = 0.001     # 0.1%


def _normalize_direction(d: str) -> str:
    return d.upper().strip()


def compare(
    local: list[LocalPosition],
    bitget: list[BitgetPosition],
    sl_orders: list[SLOrder],
) -> list[Divergence]:
    """Compare local vs Bitget et retourne les divergences."""
    divergences: list[Divergence] = []

    local_by_sym = {p.symbol: p for p in local}
    bitget_by_sym = {p.symbol: p for p in bitget}
    sl_by_sym: dict[str, list[SLOrder]] = {}
    for sl in sl_orders:
        sl_by_sym.setdefault(sl.symbol, []).append(sl)

    all_symbols = set(local_by_sym) | set(bitget_by_sym)

    for sym in sorted(all_symbols):
        loc = local_by_sym.get(sym)
        bit = bitget_by_sym.get(sym)

        # TYPE A --- Fantome (local only)
        if loc and not bit:
            divergences.append(Divergence(
                severity="FANTOME",
                symbol=sym,
                message=(
                    f"Position dans grid_states ({loc.direction}, "
                    f"{loc.total_quantity} @ {loc.avg_entry:.4f}) "
                    f"mais AUCUNE position sur Bitget"
                ),
                details={"strategy": loc.strategy, "levels": loc.num_levels},
            ))
            continue

        # TYPE B --- Orpheline (Bitget only)
        if bit and not loc:
            divergences.append(Divergence(
                severity="ORPHELINE",
                symbol=sym,
                message=(
                    f"Position sur Bitget ({bit.side}, "
                    f"{bit.contracts} @ {bit.entry_price:.4f}) "
                    f"mais ABSENTE de grid_states"
                ),
                details={
                    "leverage": bit.leverage,
                    "margin_mode": bit.margin_mode,
                    "unrealized_pnl": bit.unrealized_pnl,
                },
            ))
            continue

        # Both exist --- check consistency
        assert loc and bit

        # Direction check
        loc_dir = _normalize_direction(loc.direction)
        bit_dir = _normalize_direction(bit.side)
        if loc_dir != bit_dir:
            divergences.append(Divergence(
                severity="DESYNC",
                symbol=sym,
                message=(
                    f"Direction differente : local={loc_dir}, Bitget={bit_dir}"
                ),
            ))
            continue

        # Quantity check
        qty_diff = abs(loc.total_quantity - bit.contracts)
        qty_pct = qty_diff / max(loc.total_quantity, 1e-12) * 100
        if qty_pct > QTY_TOLERANCE * 100:
            divergences.append(Divergence(
                severity="DESYNC",
                symbol=sym,
                message=(
                    f"Quantite : local {loc.total_quantity} vs Bitget {bit.contracts} "
                    f"(delta {qty_diff:+.6f}, {qty_pct:+.2f}%)"
                ),
                details={
                    "local_qty": loc.total_quantity,
                    "bitget_qty": bit.contracts,
                    "delta_pct": qty_pct,
                },
            ))

        # Price check
        price_diff = abs(loc.avg_entry - bit.entry_price)
        price_pct = price_diff / max(loc.avg_entry, 1e-12) * 100
        if price_pct > PRICE_TOLERANCE * 100:
            divergences.append(Divergence(
                severity="DESYNC",
                symbol=sym,
                message=(
                    f"Prix moyen : local {loc.avg_entry:.6f} vs Bitget {bit.entry_price:.6f} "
                    f"(delta {price_pct:+.2f}%)"
                ),
                details={
                    "local_price": loc.avg_entry,
                    "bitget_price": bit.entry_price,
                    "delta_pct": price_pct,
                },
            ))

        # TYPE D --- SL manquant
        if loc.sl_order_id:
            sym_sls = sl_by_sym.get(sym, [])
            found = any(sl.order_id == loc.sl_order_id for sl in sym_sls)
            if not found:
                divergences.append(Divergence(
                    severity="SL_MANQUANT",
                    symbol=sym,
                    message=(
                        f"SL local sl_order_id={loc.sl_order_id} "
                        f"absent des ordres ouverts Bitget"
                    ),
                    details={"sl_price": loc.sl_price},
                ))
        elif loc.sl_price > 0:
            divergences.append(Divergence(
                severity="SL_MANQUANT",
                symbol=sym,
                message=(
                    f"Position active mais sl_order_id=None "
                    f"(sl_price local={loc.sl_price})"
                ),
            ))

    # TYPE E --- SL orphelins
    active_symbols = set(bitget_by_sym)
    for sl in sl_orders:
        if sl.symbol not in active_symbols:
            divergences.append(Divergence(
                severity="SL_ORPHELIN",
                symbol=sl.symbol,
                message=(
                    f"Ordre SL ouvert (id={sl.order_id}, trigger={sl.trigger_price}) "
                    f"mais aucune position active"
                ),
            ))

    return divergences


# ---------------------------------------------------------------------------
# Phase 5 --- Rapport
# ---------------------------------------------------------------------------

def _supports_unicode() -> bool:
    """Detecte si le terminal supporte les emojis Unicode."""
    if os.environ.get("PYTHONIOENCODING", "").startswith("utf"):
        return True
    try:
        encoding = sys.stdout.encoding or ""
        return encoding.lower().startswith("utf")
    except Exception:
        return False


_UNICODE = _supports_unicode()

# Icons with ASCII fallback for Windows cp1252
_OK = "[OK]" if not _UNICODE else "\u2705"
_WARN = "[!!]" if not _UNICODE else "\u26a0\ufe0f"
_RED = "[XX]" if not _UNICODE else "\U0001f534"
_ORANGE = "[~~]" if not _UNICODE else "\U0001f7e0"
_GREEN = "[OK]" if not _UNICODE else "\U0001f7e2"

SEVERITY_ICONS = {
    "FANTOME": _WARN,
    "ORPHELINE": _RED,
    "DESYNC": _ORANGE,
    "SL_MANQUANT": _RED,
    "SL_ORPHELIN": _ORANGE,
}


def _fmt_pos(direction: str, qty: float, price: float) -> str:
    d = "L" if direction.upper().startswith("L") else "S"
    if price > 100:
        return f"{d} {qty}@{price:.1f}"
    elif price > 1:
        return f"{d} {qty}@{price:.4f}"
    else:
        return f"{d} {qty}@{price:.6f}"


def print_report(
    meta: dict,
    local: list[LocalPosition],
    bitget: list[BitgetPosition],
    sl_orders: list[SLOrder],
    divergences: list[Divergence],
):
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    local_by_sym = {p.symbol: p for p in local}
    bitget_by_sym = {p.symbol: p for p in bitget}
    sl_by_sym: dict[str, list[SLOrder]] = {}
    for sl in sl_orders:
        sl_by_sym.setdefault(sl.symbol, []).append(sl)

    matched = len(set(local_by_sym) & set(bitget_by_sym))
    fantomes = sum(1 for d in divergences if d.severity == "FANTOME")
    orphelines = sum(1 for d in divergences if d.severity == "ORPHELINE")
    desyncs = sum(1 for d in divergences if d.severity == "DESYNC")
    sl_manquants = sum(1 for d in divergences if d.severity == "SL_MANQUANT")
    sl_orphelins = sum(1 for d in divergences if d.severity == "SL_ORPHELIN")

    print()
    print("=" * 65)
    print("  AUDIT GRID STATES vs BITGET")
    print("=" * 65)
    print()
    print(f"  Timestamp             : {now}")
    print(f"  Mode                  : {meta.get('mode', '?')}")
    if meta.get("file"):
        print(f"  Fichier               : {meta['file']}")
        print(f"  Derniere sauvegarde   : {meta.get('last_modified', '?')}")
    if meta.get("url"):
        print(f"  URL                   : {meta['url']}")
    print()

    # --- Resume ---
    print("  --- RESUME " + "-" * 50)
    print()
    print(f"  Positions locales     : {len(local)}")
    print(f"  Positions Bitget      : {len(bitget)}")
    print(f"  Matchees              : {matched}")
    print(f"  Fantomes (local only) : {fantomes}")
    print(f"  Orphelines (Bitget)   : {orphelines}")
    print(f"  Desynchronisees       : {desyncs}")
    print(f"  SL manquants          : {sl_manquants}")
    print(f"  SL orphelins          : {sl_orphelins}")
    print()

    # --- Detail par symbol ---
    all_symbols = sorted(set(local_by_sym) | set(bitget_by_sym))
    if all_symbols:
        print("  --- DETAIL PAR SYMBOL " + "-" * 40)
        print()
        header = f"  {'Symbol':<22}| {'Local':<18}| {'Bitget':<18}| Status"
        print(header)
        print("  " + "-" * 22 + "|" + "-" * 18 + "|" + "-" * 18 + "|" + "-" * 12)

        div_by_sym: dict[str, list[Divergence]] = {}
        for d in divergences:
            div_by_sym.setdefault(d.symbol, []).append(d)

        for sym in all_symbols:
            loc = local_by_sym.get(sym)
            bit = bitget_by_sym.get(sym)
            sym_short = sym.replace("/USDT:USDT", "").replace("/USDT", "")

            loc_str = _fmt_pos(loc.direction, loc.total_quantity, loc.avg_entry) if loc else "-"
            bit_str = _fmt_pos(bit.side, bit.contracts, bit.entry_price) if bit else "-"

            sym_divs = div_by_sym.get(sym, [])
            if not sym_divs:
                status = f"{_OK} OK"
            else:
                worst = sym_divs[0]
                icon = SEVERITY_ICONS.get(worst.severity, "?")
                status = f"{icon} {worst.severity}"

            print(f"  {sym_short:<22}| {loc_str:<18}| {bit_str:<18}| {status}")
        print()

    # --- SL Status ---
    symbols_with_positions = sorted(set(local_by_sym) & set(bitget_by_sym))
    if symbols_with_positions:
        print("  --- SL STATUS " + "-" * 47)
        print()
        header = f"  {'Symbol':<22}| {'SL local':<18}| {'SL exchange':<18}| Status"
        print(header)
        print("  " + "-" * 22 + "|" + "-" * 18 + "|" + "-" * 18 + "|" + "-" * 12)

        for sym in symbols_with_positions:
            loc = local_by_sym[sym]
            sym_short = sym.replace("/USDT:USDT", "").replace("/USDT", "")
            sym_sls = sl_by_sym.get(sym, [])

            sl_local_str = f"{loc.sl_price:.4f}" if loc.sl_price > 0 else "aucun"
            if loc.sl_order_id:
                sl_local_str = f"{loc.sl_order_id[:8]}.. ({loc.sl_price:.4f})"

            has_sl_issue = any(
                d.symbol == sym and d.severity == "SL_MANQUANT"
                for d in divergences
            )
            if sym_sls and not has_sl_issue:
                sl_exch_str = f"{sym_sls[0].order_id[:8]}.. ouvert"
                status = f"{_OK} OK"
            elif sym_sls and has_sl_issue:
                sl_exch_str = f"{len(sym_sls)} ordre(s)"
                status = f"{_ORANGE} ID diff"
            elif not sym_sls and not has_sl_issue:
                sl_exch_str = "aucun"
                status = f"{_OK} OK" if loc.sl_price == 0 else f"{_RED} ABSENT"
            else:
                sl_exch_str = "aucun"
                status = f"{_RED} ABSENT"

            print(f"  {sym_short:<22}| {sl_local_str:<18}| {sl_exch_str:<18}| {status}")
        print()

    # --- Divergences detail ---
    if divergences:
        print("  --- DIVERGENCES " + "-" * 45)
        print()
        for d in divergences:
            icon = SEVERITY_ICONS.get(d.severity, "?")
            print(f"  {icon} {d.severity}: {d.symbol}")
            print(f"     {d.message}")
            if d.details:
                for k, v in d.details.items():
                    print(f"     {k}: {v}")
            print()

    # --- Verdict ---
    print("  --- VERDICT " + "-" * 49)
    print()
    criticals = fantomes + orphelines + sl_manquants
    if not divergences:
        print(f"  {_GREEN} Systeme coherent -- {len(local)} positions parfaitement synchronisees.")
    elif criticals > 0:
        print(f"  {_RED} INCOHERENCE CRITIQUE -- {criticals} divergence(s) critique(s) !")
        print("     Action immediate requise.")
    else:
        ok_count = matched - desyncs
        print(f"  {_GREEN} Systeme globalement coherent -- {ok_count}/{matched} positions OK.")
        print(f"     {desyncs + sl_orphelins} divergence(s) mineure(s) a surveiller.")
    print()
    print("=" * 65)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Audit grid_states vs positions Bitget"
    )
    parser.add_argument(
        "--mode", choices=["file", "api"], default="file",
        help="Source des donnees locales (default: file)",
    )
    parser.add_argument(
        "--state-file", default="data/executor_state.json",
        help="Chemin vers executor_state.json (mode file)",
    )
    parser.add_argument(
        "--api-url", default="http://localhost:8000",
        help="URL du serveur (mode api)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Afficher les details de debug",
    )
    args = parser.parse_args()

    if not args.verbose:
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="WARNING")

    # Phase 1 --- Etat local
    logger.info("Phase 1: Lecture etat local...")
    if args.mode == "file":
        meta, local_positions = read_local_from_file(args.state_file)
    else:
        meta, local_positions = read_local_from_api(args.api_url)

    logger.info(f"  {len(local_positions)} position(s) locale(s)")
    if args.verbose:
        for p in local_positions:
            logger.debug(
                f"  Local: {p.symbol} {p.direction} {p.total_quantity}@{p.avg_entry:.4f} "
                f"({p.num_levels} niveaux, SL={p.sl_price})"
            )

    # Phase 2 --- Etat Bitget
    logger.info("Phase 2: Connexion Bitget...")
    config = _load_config()
    exchange = create_exchange(config)
    logger.info("  Connecte. Fetch positions...")
    bitget_positions = fetch_bitget_positions(exchange)
    logger.info(f"  {len(bitget_positions)} position(s) Bitget")
    if args.verbose:
        for p in bitget_positions:
            logger.debug(
                f"  Bitget: {p.symbol} {p.side} {p.contracts}@{p.entry_price:.4f} "
                f"(lev={p.leverage}, uPnL={p.unrealized_pnl:.2f})"
            )

    # Phase 3 --- Ordres SL
    logger.info("Phase 3: Fetch ordres SL...")
    all_symbols = sorted(
        {p.symbol for p in local_positions} | {p.symbol for p in bitget_positions}
    )
    sl_orders = fetch_sl_orders(exchange, all_symbols, verbose=args.verbose)
    logger.info(f"  {len(sl_orders)} ordre(s) SL ouvert(s)")

    # Phase 4 --- Comparaison
    logger.info("Phase 4: Comparaison...")
    divergences = compare(local_positions, bitget_positions, sl_orders)

    # Phase 5 --- Rapport
    print_report(meta, local_positions, bitget_positions, sl_orders, divergences)


if __name__ == "__main__":
    main()
