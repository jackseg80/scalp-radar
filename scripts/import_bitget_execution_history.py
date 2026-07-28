"""Import a read-only Bitget private-order export as calibration evidence."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.core.database import Database


def _timestamp(value: Any) -> datetime:
    return datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc)


def _observation(order: dict[str, Any], strategy_name: str) -> dict[str, Any] | None:
    """Map one raw Bitget limit order without inventing absent fields."""
    status = str(order.get("status") or "").lower()
    requested = float(order.get("amount") or 0)
    filled = float(order.get("filled") or 0)
    price = float(order.get("price") or 0)
    average = float(order.get("average") or price)
    created_ms = order.get("timestamp")
    completed_ms = order.get("lastTradeTimestamp") or created_ms
    if (
        order.get("type") != "limit"
        or status not in {"closed", "canceled", "cancelled", "expired", "rejected"}
        or requested <= 0
        or price <= 0
        or not created_ms
        or not completed_ms
        or not order.get("id")
    ):
        return None
    created_at = _timestamp(created_ms)
    completed_at = _timestamp(completed_ms)
    latency_ms = max(0, int((completed_at - created_at).total_seconds() * 1000))
    slippage_pct = abs((average - price) / price * 100) if filled > 0 else 0.0
    side = str(order.get("side") or "buy").lower()
    return {
        "timestamp": completed_at.isoformat(),
        "strategy_name": strategy_name,
        "symbol": str(order.get("symbol") or ""),
        "direction": "LONG" if side == "buy" else "SHORT",
        "trade_type": "entry" if filled > 0 else "entry_unfilled",
        "side": side,
        "quantity": filled if filled > 0 else requested,
        "price": average if filled > 0 else price,
        "order_id": f"bitget:{order['id']}",
        "context": "bitget_private_order_history_v1",
        "intent_timestamp": created_at.isoformat(),
        "intent_price": price,
        "requested_quantity": requested,
        "filled_quantity": filled,
        "fill_timestamp": completed_at.isoformat(),
        "latency_ms": latency_ms,
        "slippage_pct": slippage_pct,
        "fill_ratio": filled / requested,
        "order_status": status,
    }


async def main(args: argparse.Namespace) -> int:
    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Bitget order export must be a JSON array")
    observations = [
        result for order in raw
        if (result := _observation(order, args.strategy)) is not None
    ]
    if not observations:
        raise ValueError("No eligible Bitget limit orders in export")
    db = Database(args.db)
    await db.init()
    try:
        assert db._conn is not None
        written = 0
        for observation in observations:
            cursor = await db._conn.execute(
                "SELECT 1 FROM live_trades WHERE order_id=?",
                (observation["order_id"],),
            )
            if await cursor.fetchone():
                continue
            await db.insert_live_trade(observation)
            written += 1
    finally:
        await db.close()
    print(json.dumps({"eligible": len(observations), "written": written}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import read-only Bitget private limit-order history",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--strategy", default="grid_atr")
    parser.add_argument("--db", default="data/scalp_radar.db")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
