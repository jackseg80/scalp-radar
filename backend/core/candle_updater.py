"""CandleUpdater — mise à jour automatique quotidienne des candles historiques.

Boucle quotidienne à 03:00 UTC + endpoint API pour déclenchement manuel.
Réutilise fetch_symbol_timeframe() de scripts/fetch_history.py.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Coroutine

from loguru import logger

from backend.core.config import AppConfig
from backend.core.database import Database


class CandleUpdater:
    """Met à jour les candles historiques pour tous les assets."""

    def __init__(
        self,
        config: AppConfig,
        db: Database,
        ws_broadcast: Callable[[dict], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self._config = config
        self._db = db
        self._ws_broadcast = ws_broadcast
        self._running = False
        self._daily_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Démarre la boucle quotidienne."""
        self._daily_task = asyncio.create_task(self._daily_loop())
        logger.info("CandleUpdater démarré (daily à 03:00 UTC)")

    async def stop(self) -> None:
        """Arrête la boucle."""
        if self._daily_task:
            self._daily_task.cancel()
            try:
                await self._daily_task
            except asyncio.CancelledError:
                pass
            self._daily_task = None
        logger.info("CandleUpdater arrêté")

    async def _daily_loop(self) -> None:
        """Boucle qui tourne une fois par jour à 03:00 UTC."""
        # Guard : si BACKFILL_ENABLED=false, ne pas lancer le cron
        if not getattr(self._config.secrets, "backfill_enabled", True):
            logger.info("CandleUpdater: cron désactivé (BACKFILL_ENABLED=false)")
            return

        while True:
            now = datetime.now(tz=timezone.utc)
            next_run = now.replace(hour=3, minute=0, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            wait_seconds = (next_run - now).total_seconds()
            logger.info(
                "CandleUpdater: prochain run dans {:.1f}h",
                wait_seconds / 3600,
            )
            await asyncio.sleep(wait_seconds)
            try:
                result = await self.run_backfill()
                total = sum(
                    v for v in result.values() if isinstance(v, int)
                )
                logger.info(
                    "CandleUpdater daily terminé: {} candles insérées", total
                )
            except Exception as e:
                logger.error("CandleUpdater daily erreur: {}", e)

    @property
    def is_running(self) -> bool:
        return self._running

    async def run_backfill(
        self,
        exchanges: list[str] | None = None,
        timeframes: list[str] | None = None,
    ) -> dict[str, int | str]:
        """Lance le backfill pour tous les assets.

        Args:
            exchanges: Liste d'exchanges à fetch. Défaut: ["binance", "bitget"]
            timeframes: Override global des timeframes. Si None, utilise
                        les timeframes configurés par asset dans assets.yaml.

        Returns:
            Dict {symbol:exchange:tf: nb_candles_ou_erreur}
        """
        if self._running:
            raise RuntimeError("Un backfill est déjà en cours")

        self._running = True
        exchanges = exchanges or ["binance", "bitget"]

        try:
            from scripts.fetch_history import (
                create_exchange,
                fetch_symbol_timeframe,
            )

            results: dict[str, int | str] = {}

            # Construire la liste (asset, timeframes_per_asset)
            asset_tf_pairs: list[tuple[str, list[str]]] = []
            for asset in self._config.assets:
                tfs = timeframes or asset.timeframes
                asset_tf_pairs.append((asset.symbol, tfs))

            # Compter le total pour la progression
            total_pairs = sum(
                len(tfs) for _, tfs in asset_tf_pairs
            ) * len(exchanges)
            done = 0

            for exchange_name in exchanges:
                exchange = create_exchange(exchange_name)
                # Binance: max historique, Bitget: 90 jours
                days = 2700 if exchange_name == "binance" else 90
                start_date = datetime.now(tz=timezone.utc) - timedelta(days=days)
                end_date = datetime.now(tz=timezone.utc)

                for symbol, tfs in asset_tf_pairs:
                    for tf in tfs:
                        key = f"{symbol}:{exchange_name}:{tf}"
                        try:
                            count = await fetch_symbol_timeframe(
                                exchange,
                                self._db,
                                symbol,
                                tf,
                                start_date,
                                end_date,
                                exchange_name=exchange_name,
                            )
                            results[key] = count
                        except Exception as e:
                            logger.error(
                                "CandleUpdater: {} {} {} erreur: {}",
                                symbol,
                                exchange_name,
                                tf,
                                e,
                            )
                            results[key] = f"ERROR: {e}"

                        done += 1
                        await self._broadcast_progress(
                            done, total_pairs, symbol, exchange_name
                        )

                # Fermer proprement l'exchange ccxt
                if hasattr(exchange, "close"):
                    await exchange.close()

            return results
        finally:
            self._running = False

    async def _broadcast_progress(
        self, done: int, total: int, current_symbol: str, current_exchange: str
    ) -> None:
        """Envoie la progression via WebSocket."""
        if self._ws_broadcast:
            pct = round(done / total * 100, 1) if total > 0 else 0.0
            try:
                await self._ws_broadcast(
                    {
                        "type": "backfill_progress",
                        "data": {
                            "progress_pct": pct,
                            "done": done,
                            "total": total,
                            "current": f"{current_symbol} ({current_exchange})",
                            "running": True,
                        },
                    }
                )
            except Exception:
                pass

    async def get_status(self) -> dict:
        """Retourne l'état des données par asset/exchange.

        Pour chaque asset x exchange x timeframe, retourne :
        - first_candle: date de la première candle
        - last_candle: date de la dernière candle
        - candle_count: nombre total
        - days_available: nombre de jours couverts
        - is_stale: True si la dernière candle date de plus de 2h
        """
        symbols = [a.symbol for a in self._config.assets]
        status: dict[str, dict] = {}

        for symbol in symbols:
            status[symbol] = {}
            for exchange_name in ["binance", "bitget"]:
                stats = await self._db.get_candle_stats(
                    symbol, "1h", exchange=exchange_name
                )
                status[symbol][exchange_name] = stats

        return {
            "running": self._running,
            "assets": status,
        }
