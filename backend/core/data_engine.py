"""Data Engine : connexion WebSocket Bitget via ccxt, agrégation multi-timeframe.

Gère les souscriptions klines, la validation des données, le stockage mémoire
(buffer rolling borné) et la persistance en base.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Optional

import ccxt.pro as ccxtpro
from loguru import logger

from backend.core.config import AppConfig
from backend.core.database import Database
from backend.core.models import Candle, MultiTimeframeData, OISnapshot, TimeFrame

# Taille max du buffer rolling par (symbol, timeframe)
MAX_BUFFER_SIZE = 500


# ─── DATA VALIDATOR ─────────────────────────────────────────────────────────


class DataValidator:
    """Valide les candles entrantes avant injection dans le système."""

    @staticmethod
    def validate_candle(candle: Candle) -> bool:
        """Vérifie la cohérence OHLC, volume et timestamp."""
        if candle.low > candle.high:
            logger.warning(
                "Candle invalide {}/{}: low ({}) > high ({})",
                candle.symbol,
                candle.timeframe.value,
                candle.low,
                candle.high,
            )
            return False
        if candle.volume < 0:
            logger.warning(
                "Candle invalide {}/{}: volume négatif ({})",
                candle.symbol,
                candle.timeframe.value,
                candle.volume,
            )
            return False
        if candle.open <= 0 or candle.close <= 0:
            logger.warning(
                "Candle invalide {}/{}: prix ≤ 0",
                candle.symbol,
                candle.timeframe.value,
            )
            return False
        return True

    @staticmethod
    def check_gap(
        prev: Candle, curr: Candle, timeframe: TimeFrame
    ) -> bool:
        """Détecte un gap entre deux candles consécutives."""
        expected_delta_ms = timeframe.to_milliseconds()
        actual_delta = (
            curr.timestamp.timestamp() - prev.timestamp.timestamp()
        ) * 1000
        # Tolérance de 50% pour les micro-décalages
        return actual_delta > expected_delta_ms * 1.5

    @staticmethod
    def is_duplicate(
        candle: Candle, buffer: list[Candle]
    ) -> bool:
        """Vérifie si la candle existe déjà dans le buffer."""
        if not buffer:
            return False
        # Ne vérifie que les 5 dernières pour la performance
        for existing in buffer[-5:]:
            if existing.timestamp == candle.timestamp:
                return True
        return False


# ─── DATA ENGINE ────────────────────────────────────────────────────────────


class DataEngine:
    """Moteur de données temps réel via ccxt WebSocket.

    S'abonne aux klines multi-symbol multi-timeframe,
    valide, stocke en mémoire et persiste en base.
    """

    def __init__(
        self,
        config: AppConfig,
        database: Database,
    ) -> None:
        self.config = config
        self.db = database
        self.validator = DataValidator()

        # Buffer rolling : {symbol: {timeframe: [Candle, ...]}}
        self._buffers: dict[str, dict[str, list[Candle]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # État
        self._exchange: Optional[ccxtpro.bitget] = None
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._last_update: Optional[datetime] = None
        self._connected = False

        # Callbacks pour les consommateurs
        self._callbacks: list[Callable] = []

        # Données additionnelles (funding, OI)
        self._funding_rates: dict[str, float] = {}
        self._open_interest: dict[str, list[OISnapshot]] = {}
        self._oi_max_snapshots = 60  # 60 snapshots × 60s = 1h d'historique

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def last_update(self) -> Optional[datetime]:
        return self._last_update

    def get_data(self, symbol: str) -> MultiTimeframeData:
        """Retourne les données multi-timeframe pour un symbol."""
        candles_dict: dict[str, list[Candle]] = {}
        if symbol in self._buffers:
            for tf, buf in self._buffers[symbol].items():
                candles_dict[tf] = list(buf)
        return MultiTimeframeData(
            symbol=symbol,
            candles=candles_dict,
            last_update=self._last_update,
        )

    def get_all_symbols(self) -> list[str]:
        """Retourne la liste des symbols suivis."""
        return [a.symbol for a in self.config.assets]

    def on_candle(self, callback: Callable) -> None:
        """Enregistre un callback appelé à chaque nouvelle candle validée."""
        self._callbacks.append(callback)

    def get_funding_rate(self, symbol: str) -> float | None:
        """Retourne le dernier funding rate connu pour un symbol."""
        return self._funding_rates.get(symbol)

    def get_open_interest(self, symbol: str) -> list[OISnapshot]:
        """Retourne les snapshots d'OI pour un symbol."""
        return list(self._open_interest.get(symbol, []))

    # ─── LIFECYCLE ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Démarre les connexions WebSocket."""
        logger.info("DataEngine: démarrage...")

        self._exchange = ccxtpro.bitget({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })

        self._running = True

        # Lancer une tâche de watch par symbol
        for asset in self.config.assets:
            task = asyncio.create_task(
                self._watch_symbol(asset.symbol, asset.timeframes),
                name=f"watch_{asset.symbol}",
            )
            self._tasks.append(task)

        # Tâches de polling pour funding rate et OI
        self._tasks.append(
            asyncio.create_task(self._poll_funding_rates(), name="poll_funding")
        )
        self._tasks.append(
            asyncio.create_task(self._poll_open_interest(), name="poll_oi")
        )

        self._connected = True
        logger.info(
            "DataEngine: connecté, {} symbols × {} timeframes",
            len(self.config.assets),
            len(self.config.assets[0].timeframes) if self.config.assets else 0,
        )

    async def stop(self) -> None:
        """Arrête proprement les connexions."""
        logger.info("DataEngine: arrêt en cours...")
        self._running = False
        self._connected = False

        # Annuler les tâches
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Fermer l'exchange
        if self._exchange:
            try:
                await self._exchange.close()
            except Exception as e:
                logger.warning("Erreur fermeture exchange: {}", e)
            self._exchange = None

        logger.info("DataEngine: arrêté")

    # ─── WATCH LOOP ─────────────────────────────────────────────────────────

    async def _watch_symbol(
        self, symbol: str, timeframes: list[str]
    ) -> None:
        """Boucle de watch pour un symbol sur tous ses timeframes."""
        reconnect_delay = self.config.exchange.websocket.reconnect_delay
        max_attempts = self.config.exchange.websocket.max_reconnect_attempts
        attempt = 0

        while self._running:
            try:
                attempt += 1
                if attempt > 1:
                    logger.info(
                        "DataEngine: reconnexion {} (tentative {}/{})",
                        symbol,
                        attempt,
                        max_attempts,
                    )

                await self._subscribe_klines(symbol, timeframes)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._running:
                    break
                logger.error(
                    "DataEngine: erreur watch {} : {} (tentative {}/{})",
                    symbol,
                    e,
                    attempt,
                    max_attempts,
                )
                if attempt >= max_attempts:
                    logger.error(
                        "DataEngine: max reconnexions atteint pour {}",
                        symbol,
                    )
                    break
                # Backoff exponentiel
                delay = reconnect_delay * min(2 ** (attempt - 1), 60)
                await asyncio.sleep(delay)

    async def _subscribe_klines(
        self, symbol: str, timeframes: list[str]
    ) -> None:
        """S'abonne aux klines via ccxt watch_ohlcv."""
        assert self._exchange is not None

        while self._running:
            for tf in timeframes:
                if not self._running:
                    return
                try:
                    ohlcv_list = await self._exchange.watch_ohlcv(symbol, tf)
                    for ohlcv in ohlcv_list:
                        await self._on_candle_received(symbol, tf, ohlcv)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(
                        "DataEngine: erreur kline {}/{}: {}",
                        symbol,
                        tf,
                        e,
                    )

    async def _on_candle_received(
        self,
        symbol: str,
        timeframe_str: str,
        ohlcv: list,
    ) -> None:
        """Traite une candle reçue du WebSocket."""
        try:
            tf = TimeFrame.from_string(timeframe_str)
            candle = Candle(
                timestamp=datetime.fromtimestamp(
                    ohlcv[0] / 1000, tz=timezone.utc
                ),
                open=float(ohlcv[1]),
                high=float(ohlcv[2]),
                low=float(ohlcv[3]),
                close=float(ohlcv[4]),
                volume=float(ohlcv[5]) if len(ohlcv) > 5 else 0.0,
                symbol=symbol,
                timeframe=tf,
            )
        except (ValueError, IndexError) as e:
            logger.warning("DataEngine: candle malformée: {}", e)
            return

        # Validation
        if not self.validator.validate_candle(candle):
            return

        buffer = self._buffers[symbol][timeframe_str]

        # Doublon ?
        if self.validator.is_duplicate(candle, buffer):
            return

        # Gap ?
        if buffer and self.validator.check_gap(buffer[-1], candle, tf):
            logger.warning(
                "DataEngine: gap détecté {}/{} entre {} et {}",
                symbol,
                timeframe_str,
                buffer[-1].timestamp,
                candle.timestamp,
            )

        # Ajouter au buffer (borné)
        buffer.append(candle)
        if len(buffer) > MAX_BUFFER_SIZE:
            del buffer[: len(buffer) - MAX_BUFFER_SIZE]

        self._last_update = datetime.now(tz=timezone.utc)

        # Persister en base
        try:
            await self.db.insert_candles_batch([candle])
        except Exception as e:
            logger.error("DataEngine: erreur persistance: {}", e)

        # Notifier les callbacks
        for callback in self._callbacks:
            try:
                result = callback(symbol, timeframe_str, candle)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("DataEngine: erreur callback: {}", e)

    # ─── POLLING FUNDING & OI ──────────────────────────────────────────────

    async def _poll_funding_rates(self) -> None:
        """Polling des funding rates toutes les 5 minutes."""
        while self._running:
            try:
                await self._fetch_funding_rates()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("DataEngine: erreur polling funding: {}", e)
            await asyncio.sleep(300)  # 5 min

    async def _poll_open_interest(self) -> None:
        """Polling de l'open interest toutes les 60 secondes."""
        while self._running:
            try:
                await self._fetch_open_interest()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("DataEngine: erreur polling OI: {}", e)
            await asyncio.sleep(60)  # 60s

    async def _fetch_funding_rates(self) -> None:
        """Récupère les funding rates via ccxt."""
        if not self._exchange:
            return
        for asset in self.config.assets:
            try:
                result = await self._exchange.fetch_funding_rate(asset.symbol)
                if result and "fundingRate" in result:
                    rate = result["fundingRate"]
                    if rate is not None:
                        self._funding_rates[asset.symbol] = float(rate) * 100  # en %
            except Exception as e:
                logger.debug("DataEngine: funding rate non dispo pour {}: {}", asset.symbol, e)

    async def _fetch_open_interest(self) -> None:
        """Récupère l'open interest via ccxt."""
        if not self._exchange:
            return
        now = datetime.now(tz=timezone.utc)
        for asset in self.config.assets:
            try:
                result = await self._exchange.fetch_open_interest(asset.symbol)
                if result and "openInterestAmount" in result:
                    oi_value = float(result["openInterestAmount"])
                    snapshots = self._open_interest.setdefault(asset.symbol, [])

                    # Calculer le changement vs snapshot précédent
                    change_pct = 0.0
                    if snapshots:
                        prev = snapshots[-1].value
                        if prev > 0:
                            change_pct = (oi_value - prev) / prev * 100

                    snapshots.append(OISnapshot(
                        timestamp=now,
                        symbol=asset.symbol,
                        value=oi_value,
                        change_pct=change_pct,
                    ))

                    # Borner l'historique
                    if len(snapshots) > self._oi_max_snapshots:
                        self._open_interest[asset.symbol] = snapshots[-self._oi_max_snapshots:]
            except Exception as e:
                logger.debug("DataEngine: OI non dispo pour {}: {}", asset.symbol, e)
