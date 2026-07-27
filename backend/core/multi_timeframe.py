"""Shared, closed-bar multi-timeframe transformations."""

from __future__ import annotations

import numpy as np

from backend.core.indicators import atr, supertrend
from backend.core.models import Candle


def resample_complete_1h_to_4h(
    main_candles: list[Candle],
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build complete UTC 4h buckets and their closed-bar visibility mapping."""
    n = len(main_candles)
    if n == 0:
        empty = np.array([], dtype=float)
        return (
            empty,
            empty,
            empty,
            np.full(0, -1, dtype=np.int32),
            np.full(0, -1, dtype=np.int32),
        )

    bucket_size = 14_400
    hour_size = 3_600
    timestamps = np.array(
        [int(c.timestamp.timestamp()) for c in main_candles],
        dtype=np.int64,
    )
    buckets = timestamps // bucket_size
    grouped: dict[int, list[int]] = {}
    for index, bucket_id in enumerate(buckets):
        grouped.setdefault(int(bucket_id), []).append(index)

    h4_highs: list[float] = []
    h4_lows: list[float] = []
    h4_closes: list[float] = []
    complete_bucket_ids: list[int] = []
    segment_ids: list[int] = []
    segment = -1
    previous_complete: int | None = None

    for bucket_id in sorted(grouped):
        indices = grouped[bucket_id]
        bucket_open = bucket_id * bucket_size
        expected = [bucket_open + offset * hour_size for offset in range(4)]
        actual = [int(timestamps[index]) for index in indices]
        if actual != expected:
            continue
        if previous_complete is None or bucket_id != previous_complete + 1:
            segment += 1
        previous_complete = bucket_id
        complete_bucket_ids.append(bucket_id)
        segment_ids.append(segment)
        h4_highs.append(float(np.max(highs[indices])))
        h4_lows.append(float(np.min(lows[indices])))
        h4_closes.append(float(closes[indices[-1]]))

    bucket_to_index = {
        bucket_id: index for index, bucket_id in enumerate(complete_bucket_ids)
    }
    mapping = np.full(n, -1, dtype=np.int32)
    for index, current_bucket in enumerate(buckets):
        completed_index = bucket_to_index.get(int(current_bucket) - 1)
        if completed_index is not None:
            mapping[index] = completed_index

    return (
        np.asarray(h4_highs, dtype=float),
        np.asarray(h4_lows, dtype=float),
        np.asarray(h4_closes, dtype=float),
        mapping,
        np.asarray(segment_ids, dtype=np.int32),
    )


def resample_1h_to_4h(
    main_candles: list[Candle],
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return complete 4h OHLC and the closed-bar 1h mapping."""
    h4_highs, h4_lows, h4_closes, mapping, _segments = (
        resample_complete_1h_to_4h(main_candles, closes, highs, lows)
    )
    return h4_highs, h4_lows, h4_closes, mapping


def compute_supertrend_4h_mapped_to_1h(
    candles_1h: list[Candle],
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    st_atr_period: int,
    st_atr_multiplier: float,
) -> np.ndarray:
    """Return a fail-closed Supertrend 4h timeline for 1h signal candles."""
    n = len(candles_1h)
    if n == 0:
        return np.array([], dtype=float)

    h4_highs, h4_lows, h4_closes, mapping, segments = (
        resample_complete_1h_to_4h(candles_1h, closes, highs, lows)
    )
    st_direction_4h = np.full(len(h4_closes), np.nan, dtype=float)
    for segment_id in np.unique(segments):
        indices = np.where(segments == segment_id)[0]
        if len(indices) == 0:
            continue
        segment_atr = atr(
            h4_highs[indices],
            h4_lows[indices],
            h4_closes[indices],
            st_atr_period,
        )
        _, segment_direction = supertrend(
            h4_highs[indices],
            h4_lows[indices],
            h4_closes[indices],
            segment_atr,
            st_atr_multiplier,
        )
        st_direction_4h[indices] = segment_direction

    mapped = np.full(n, np.nan, dtype=float)
    valid = mapping >= 0
    if np.any(valid):
        mapped[valid] = st_direction_4h[mapping[valid]]
    return mapped
