"""Regime Monitor — Sprint 61.

Calcule un snapshot quotidien du régime de marché BTC (bull/bear/range/crash)
et envoie une alerte Telegram à 00:05 UTC.  Expose un snapshot + historique
pour le widget frontend.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from backend.backtesting.metrics import _classify_regime
from backend.core.indicators import atr as compute_atr

if TYPE_CHECKING:
    from backend.alerts.telegram import TelegramClient
    from backend.core.database import Database
    from backend.core.models import Candle


# ─── DATACLASSES ─────────────────────────────────────────────────────────────


@dataclass
class RegimeSnapshot:
    """Snapshot du régime de marché BTC."""

    regime: str  # "BULL", "BEAR", "RANGE", "CRASH"
    regime_days: int  # Jours consécutifs dans ce régime
    btc_atr_14d_pct: float  # ATR(14) daily en % du prix
    btc_change_30d_pct: float  # Return BTC sur 30j
    volatility_level: str  # "LOW", "MEDIUM", "MEDIUM-HIGH", "HIGH"
    suggested_leverage: int  # 3, 4, 5 ou 6
    timestamp: datetime


@dataclass
class _DailyBar:
    """Barre daily agrégée depuis des candles 1h. Candle-like pour _classify_regime."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


# ─── HELPERS ─────────────────────────────────────────────────────────────────

_MIN_HOURLY_PER_DAY = 20  # Ignorer les jours avec < 20 candles sur 24


def _classify_volatility(atr_pct: float) -> tuple[str, int]:
    """Retourne (volatility_level, suggested_leverage) selon l'ATR daily en %."""
    if atr_pct < 2.0:
        return ("LOW", 3)
    if atr_pct < 3.0:
        return ("MEDIUM", 4)
    if atr_pct < 4.0:
        return ("MEDIUM-HIGH", 5)
    return ("HIGH", 6)


def _resample_1h_to_daily(candles: list[Candle]) -> list[_DailyBar]:
    """Agrège des candles 1h en barres daily.

    Ignore les jours avec moins de ``_MIN_HOURLY_PER_DAY`` candles.
    """
    by_date: dict[str, list[Candle]] = {}
    for c in candles:
        day_key = c.timestamp.strftime("%Y-%m-%d")
        by_date.setdefault(day_key, []).append(c)

    bars: list[_DailyBar] = []
    for day_key in sorted(by_date):
        group = by_date[day_key]
        if len(group) < _MIN_HOURLY_PER_DAY:
            continue
        group.sort(key=lambda c: c.timestamp)
        bars.append(
            _DailyBar(
                timestamp=group[0].timestamp.replace(hour=0, minute=0, second=0, microsecond=0),
                open=group[0].open,
                high=max(c.high for c in group),
                low=min(c.low for c in group),
                close=group[-1].close,
                volume=sum(c.volume for c in group),
            )
        )
    return bars


# ─── COMPUTE ─────────────────────────────────────────────────────────────────


async def compute_regime_snapshot(
    db: Database,
    exchange: str = "binance",
) -> RegimeSnapshot:
    """Calcule le snapshot régime actuel depuis les candles BTC 1h en base."""
    now = datetime.now(tz=timezone.utc)
    start = now - timedelta(days=60)
    candles = await db.get_candles(
        "BTC/USDT", "1h", start=start, limit=2000, exchange=exchange,
    )
    if not candles:
        raise ValueError("Pas de candles BTC/USDT 1h disponibles")

    daily = _resample_1h_to_daily(candles)
    if len(daily) < 15:
        raise ValueError(f"Pas assez de barres daily ({len(daily)}, min 15)")

    # Régime actuel (30 derniers jours)
    window_30 = daily[-30:]
    result = _classify_regime(window_30)
    regime = result["regime"].upper()

    # ATR(14) daily
    highs = np.array([b.high for b in daily], dtype=np.float64)
    lows = np.array([b.low for b in daily], dtype=np.float64)
    closes = np.array([b.close for b in daily], dtype=np.float64)
    atr_arr = compute_atr(highs, lows, closes, period=14)
    # Dernier ATR valide
    valid_mask = ~np.isnan(atr_arr)
    if not valid_mask.any():
        raise ValueError("ATR non calculable (pas assez de données)")
    last_atr = float(atr_arr[valid_mask][-1])
    current_close = float(closes[-1])
    btc_atr_14d_pct = (last_atr / current_close) * 100

    # Return 30j
    if len(closes) >= 30:
        close_30d_ago = float(closes[-30])
    else:
        close_30d_ago = float(closes[0])
    btc_change_30d_pct = ((current_close - close_30d_ago) / close_30d_ago) * 100

    # Volatilité + leverage suggéré
    volatility_level, suggested_leverage = _classify_volatility(btc_atr_14d_pct)

    # regime_days : calculé à l'initialisation via fenêtres glissantes
    regime_days = _compute_regime_days(daily, regime)

    return RegimeSnapshot(
        regime=regime,
        regime_days=regime_days,
        btc_atr_14d_pct=round(btc_atr_14d_pct, 2),
        btc_change_30d_pct=round(btc_change_30d_pct, 2),
        volatility_level=volatility_level,
        suggested_leverage=suggested_leverage,
        timestamp=now,
    )


def _compute_regime_days(daily: list[_DailyBar], current_regime: str) -> int:
    """Compte les jours consécutifs dans le régime actuel via fenêtres glissantes."""
    if len(daily) < 30:
        return 1
    count = 0
    # Remonter depuis la fin : classifier fenêtre [d-30..d] pour chaque jour d
    for end_idx in range(len(daily) - 1, 29, -1):
        window = daily[end_idx - 29 : end_idx + 1]  # 30 barres
        result = _classify_regime(window)
        if result["regime"].upper() == current_regime:
            count += 1
        else:
            break
    return max(count, 1)


# ─── MONITOR (scheduler) ────────────────────────────────────────────────────


class RegimeMonitor:
    """Calcule et envoie le snapshot régime quotidien à 00:05 UTC.

    Pattern identique à ``WeeklyReporter``.
    """

    def __init__(self, telegram: TelegramClient | None, db: Database) -> None:
        self._telegram = telegram
        self._db = db
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._latest: RegimeSnapshot | None = None
        self._history: list[dict[str, Any]] = []
        # Compteur incrémental : initialisé au premier compute, +1 ou reset ensuite
        self._last_regime: str | None = None

    # TODO: persist snapshots to DB

    async def start(self) -> None:
        self._running = True
        try:
            self._latest = await compute_regime_snapshot(self._db)
            self._last_regime = self._latest.regime
            self._history.append(self._snapshot_to_dict(self._latest))
            logger.info("RegimeMonitor: snapshot initial ({})", self._latest.regime)
        except Exception as e:
            logger.warning("RegimeMonitor: snapshot initial échoué: {}", e)
        self._task = asyncio.create_task(self._loop())
        logger.info("RegimeMonitor: activé (envoi quotidien 00:05 UTC)")

    async def _loop(self) -> None:
        while self._running:
            try:
                wait = self._seconds_until_next_0005_utc()
                logger.info("RegimeMonitor: prochain snapshot dans {:.1f}h", wait / 3600)
                await asyncio.sleep(wait)
                if not self._running:
                    break
                snapshot = await compute_regime_snapshot(self._db)
                # Compteur incrémental
                if self._last_regime and snapshot.regime == self._last_regime:
                    if self._latest:
                        snapshot = RegimeSnapshot(
                            regime=snapshot.regime,
                            regime_days=self._latest.regime_days + 1,
                            btc_atr_14d_pct=snapshot.btc_atr_14d_pct,
                            btc_change_30d_pct=snapshot.btc_change_30d_pct,
                            volatility_level=snapshot.volatility_level,
                            suggested_leverage=snapshot.suggested_leverage,
                            timestamp=snapshot.timestamp,
                        )
                else:
                    # Régime changé → reset à 1
                    snapshot = RegimeSnapshot(
                        regime=snapshot.regime,
                        regime_days=1,
                        btc_atr_14d_pct=snapshot.btc_atr_14d_pct,
                        btc_change_30d_pct=snapshot.btc_change_30d_pct,
                        volatility_level=snapshot.volatility_level,
                        suggested_leverage=snapshot.suggested_leverage,
                        timestamp=snapshot.timestamp,
                    )
                self._last_regime = snapshot.regime
                self._latest = snapshot
                self._history.append(self._snapshot_to_dict(snapshot))
                if len(self._history) > 30:
                    self._history = self._history[-30:]
                if self._telegram:
                    msg = self._format_telegram(snapshot)
                    await self._telegram.send_message(msg)
                logger.info("RegimeMonitor: snapshot envoyé ({})", snapshot.regime)
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("RegimeMonitor: erreur: {}", e)
                await asyncio.sleep(3600)

    @staticmethod
    def _seconds_until_next_0005_utc() -> float:
        """Secondes jusqu'au prochain 00:05 UTC."""
        now = datetime.now(tz=timezone.utc)
        target = now.replace(hour=0, minute=5, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return (target - now).total_seconds()

    @property
    def latest(self) -> RegimeSnapshot | None:
        return self._latest

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def _snapshot_to_dict(self, s: RegimeSnapshot) -> dict[str, Any]:
        return {
            "regime": s.regime,
            "regime_days": s.regime_days,
            "btc_atr_14d_pct": round(s.btc_atr_14d_pct, 2),
            "btc_change_30d_pct": round(s.btc_change_30d_pct, 2),
            "volatility_level": s.volatility_level,
            "suggested_leverage": s.suggested_leverage,
            "timestamp": s.timestamp.isoformat(),
        }

    def _format_telegram(self, s: RegimeSnapshot) -> str:
        emoji_map = {
            "BULL": "\U0001f7e2",
            "BEAR": "\U0001f534",
            "RANGE": "\U0001f7e1",
            "CRASH": "\u26a0\ufe0f",
        }
        vol_emoji = {
            "LOW": "\U0001f7e2",
            "MEDIUM": "\U0001f7e1",
            "MEDIUM-HIGH": "\U0001f7e0",
            "HIGH": "\U0001f534",
        }
        regime_emoji = emoji_map.get(s.regime, "\u2754")
        v_emoji = vol_emoji.get(s.volatility_level, "")
        date_str = s.timestamp.strftime("%Y-%m-%d")
        return (
            f"\U0001f4ca <b>REGIME MONITOR — {date_str}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Régime : {regime_emoji} <b>{s.regime}</b> (jour {s.regime_days})\n"
            f"Volatilité : {v_emoji} {s.volatility_level} (ATR {s.btc_atr_14d_pct:.1f}%)\n"
            f"BTC 30j : {s.btc_change_30d_pct:+.1f}%\n"
            f"\n"
            f"\U0001f4a1 Leverage suggéré : <b>{s.suggested_leverage}x</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("RegimeMonitor: arrêté")
