"""Construit les extra_data alignés par timestamp pour le backtest.

Les données funding/OI sont à des fréquences différentes des candles.
Ce module aligne et forward-fill pour que chaque bougie ait des extra_data.

Usage :
    extra_map = build_extra_data_map(candles, funding_rates, oi_records)
    result = engine.run(candles_by_tf, extra_data_by_timestamp=extra_map)
"""

from __future__ import annotations

from typing import Any

from backend.core.models import Candle
from backend.strategies.base import (
    EXTRA_FUNDING_RATE,
    EXTRA_OI_CHANGE_PCT,
    EXTRA_OPEN_INTEREST,
)


def build_extra_data_map(
    candles: list[Candle],
    funding_rates: list[dict] | None = None,
    oi_records: list[dict] | None = None,
) -> dict[str, dict[str, Any]]:
    """Construit un dict {timestamp_iso: {extra_data}} pour chaque bougie.

    Funding rates : forward-fill (le taux est valide jusqu'au prochain).
    OI : oi_change_pct calculé vs précédent (même logique que DataEngine).

    Args:
        candles: Bougies du TF principal (triées par timestamp).
        funding_rates: Liste de dicts {"timestamp": epoch_ms, "funding_rate": float}.
        oi_records: Liste de dicts {"timestamp": epoch_ms, "oi_value": float}.

    Returns:
        Dict clé = timestamp ISO de la bougie → valeurs extra_data.
    """
    result: dict[str, dict[str, Any]] = {}

    funding_sorted = sorted(funding_rates or [], key=lambda r: r["timestamp"])
    oi_sorted = sorted(oi_records or [], key=lambda r: r["timestamp"])

    f_idx = 0
    current_funding_rate: float | None = None
    o_idx = 0
    prev_oi_value: float | None = None

    for candle in candles:
        candle_ts_ms = int(candle.timestamp.timestamp() * 1000)
        ts_iso = candle.timestamp.isoformat()
        extra: dict[str, Any] = {}

        # --- Funding rate (forward-fill) ---
        while f_idx < len(funding_sorted) and funding_sorted[f_idx]["timestamp"] <= candle_ts_ms:
            current_funding_rate = funding_sorted[f_idx]["funding_rate"]
            f_idx += 1

        if current_funding_rate is not None:
            extra[EXTRA_FUNDING_RATE] = current_funding_rate

        # --- Open Interest ---
        current_oi_value: float | None = None
        while o_idx < len(oi_sorted) and oi_sorted[o_idx]["timestamp"] <= candle_ts_ms:
            current_oi_value = oi_sorted[o_idx]["oi_value"]
            o_idx += 1

        if current_oi_value is not None:
            oi_change = 0.0
            if prev_oi_value is not None and prev_oi_value > 0:
                oi_change = (current_oi_value - prev_oi_value) / prev_oi_value * 100
            extra[EXTRA_OI_CHANGE_PCT] = oi_change
            extra[EXTRA_OPEN_INTEREST] = [current_oi_value]
            prev_oi_value = current_oi_value

        result[ts_iso] = extra

    return result
