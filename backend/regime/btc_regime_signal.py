"""Signal de régime BTC pré-calculé pour le portfolio backtest.

Charge les candles BTC 4h depuis la DB, applique le détecteur ema_atr
en mode binaire (normal/defensive), et retourne un signal utilisable
par PortfolioBacktester pour le leverage dynamique.

Sprint 50b.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from backend.core.database import Database
from backend.regime.detectors import (
    EMAATRDetector,
    resample_4h_to_daily,
    to_binary_labels,
)

# Params optimaux Sprint 50a-bis (ema_atr best F1 defensive = 0.668)
DEFAULT_PARAMS: dict[str, Any] = {
    "h_down": 6,
    "h_up": 24,
    "ema_fast": 50,
    "ema_slow": 200,
    "atr_fast": 7,
    "atr_slow": 30,
    "atr_stress_ratio": 2.0,
}


@dataclass
class RegimeSignal:
    """Signal de régime pré-calculé pour une période."""

    timestamps: list[datetime]  # timestamps 4h alignés
    regimes: list[str]  # "normal" ou "defensive"
    transitions: list[dict] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def get_regime_at(self, dt: datetime) -> str:
        """Retourne le régime actif à un timestamp donné.

        Lookup binaire sur self.timestamps.
        Retourne "normal" si dt < premier timestamp (warmup).
        """
        if not self.timestamps or dt < self.timestamps[0]:
            return "normal"
        idx = bisect.bisect_right(self.timestamps, dt) - 1
        return self.regimes[idx]

    def get_leverage(
        self,
        dt: datetime,
        lev_normal: int = 7,
        lev_defensive: int = 4,
    ) -> int:
        """Retourne le leverage recommandé à un timestamp donné."""
        regime = self.get_regime_at(dt)
        return lev_normal if regime == "normal" else lev_defensive


def _candles_to_dataframe(candles: list) -> pd.DataFrame:
    """Convertit une liste de Candle en DataFrame compatible avec les détecteurs."""
    rows = []
    for c in candles:
        rows.append({
            "timestamp_utc": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        })
    return pd.DataFrame(rows)


def _build_transitions(
    timestamps: list[datetime],
    regimes: list[str],
) -> list[dict]:
    """Construit la liste des transitions de régime."""
    transitions: list[dict] = []
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i - 1]:
            transitions.append({
                "timestamp": timestamps[i].isoformat(),
                "from": regimes[i - 1],
                "to": regimes[i],
            })
    return transitions


async def compute_regime_signal(
    db_path: str = "data/scalp_radar.db",
    start: datetime | None = None,
    end: datetime | None = None,
    exchange: str = "binance",
    detector_params: dict | None = None,
) -> RegimeSignal:
    """Charge les candles BTC 4h, applique ema_atr binaire, retourne le signal.

    Pipeline :
    1. Charger BTC/USDT 4h depuis la DB
    2. Convertir en DataFrame pandas
    3. Resampler 4h → daily
    4. Appliquer EMAATRDetector.run()
    5. Convertir labels 4-classes → binaire
    6. Construire et retourner RegimeSignal
    """
    params = detector_params if detector_params is not None else DEFAULT_PARAMS.copy()

    # 1. Charger BTC/USDT 4h
    db = Database(db_path)
    await db.init()
    candles = await db.get_candles(
        "BTC/USDT", "4h",
        start=start, end=end,
        limit=1_000_000,
        exchange=exchange,
    )
    await db.close()

    if not candles:
        raise ValueError(
            "BTC/USDT 4h not found in DB. "
            "Run: uv run python -m scripts.backfill_candles "
            "--symbol BTC/USDT --timeframe 4h --since 2017-01-01"
        )

    # 2. Convertir en DataFrame
    df_4h = _candles_to_dataframe(candles)

    # 3. Resampler 4h → daily
    df_daily = resample_4h_to_daily(df_4h)

    # 4. Appliquer le détecteur
    detector = EMAATRDetector()
    result = detector.run(df_4h, df_daily, **params)

    # 5. Convertir en binaire
    binary_labels = to_binary_labels(result.labels_4h)

    # 6. Construire RegimeSignal (ne garder que post-warmup)
    warmup_idx = result.warmup_end_idx
    ts_list = [c.timestamp for c in candles]

    # Garder tous les timestamps mais marquer le warmup comme "normal"
    final_regimes: list[str] = []
    for i, regime in enumerate(binary_labels):
        if i < warmup_idx:
            final_regimes.append("normal")
        else:
            final_regimes.append(regime)

    transitions = _build_transitions(ts_list, final_regimes)

    return RegimeSignal(
        timestamps=ts_list,
        regimes=final_regimes,
        transitions=transitions,
        params=params,
    )
