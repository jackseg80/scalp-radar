"""Data Engine : connexion WebSocket Bitget via ccxt, agrégation multi-timeframe.

Gère les souscriptions klines, la validation des données, le stockage mémoire
(buffer rolling borné) et la persistance en base.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Callable, Optional

import ccxt.pro as ccxtpro
from loguru import logger

from backend.core.config import AppConfig
from backend.core.database import Database
from backend.core.models import Candle, MultiTimeframeData, OISnapshot, TimeFrame

if TYPE_CHECKING:
    from backend.alerts.notifier import Notifier

# Import runtime pour AnomalyType (utilisé dans _heartbeat_loop)
from backend.alerts.notifier import AnomalyType

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
        notifier: "Notifier | None" = None,
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

        # Buffer d'écriture DB (flush toutes les 5s au lieu de 1 INSERT par candle)
        self._write_buffer: list[Candle] = []
        self._flush_task: asyncio.Task | None = None

        # Données additionnelles (funding, OI)
        self._funding_rates: dict[str, float] = {}
        self._open_interest: dict[str, list[OISnapshot]] = {}
        self._oi_max_snapshots = 60  # 60 snapshots × 60s = 1h d'historique

        # Heartbeat : détecte silences WS > 5 min → full_reconnect
        self._notifier = notifier
        self._last_candle_received: float = time.time()
        self._heartbeat_interval: int = 300  # 5 minutes
        self._heartbeat_task: asyncio.Task | None = None
        self._heartbeat_tick: int = 0  # incrémenté à chaque cycle 60s

        # Monitoring per-symbol : timestamp de la dernière candle reçue
        self._last_update_per_symbol: dict[str, datetime] = {}

        # Backoff restart stale : compteur de tentatives et symbols abandonnés
        self._stale_restart_count: dict[str, int] = {}
        self._stale_abandoned: set[str] = set()

        # Mission 2026-03-06 : Agrégation native
        # source_tf: le plus petit TF pour lequel on s'abonne réellement au WS
        self._source_tfs: dict[str, str] = {}
        # target_tfs: les TFs supérieurs calculés par agrégation
        self._target_tfs: dict[str, list[str]] = {}

        # Mission 2026-03-06 : Mode Hybride Polling
        # Symbols en mode polling REST au lieu de WebSocket
        self._polling_tasks: dict[str, asyncio.Task] = {}
        self._polling_modes: set[str] = set()

        # Compteur de gaps détectés (exposé pour /health et tests)
        self.gap_count: int = 0

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

    def get_last_update(self, symbol: str) -> datetime | None:
        """Retourne le timestamp de la dernière bougie reçue pour un symbol."""
        return self._last_update_per_symbol.get(symbol)

    def get_open_interest(self, symbol: str) -> list[OISnapshot]:
        """Retourne les snapshots d'OI pour un symbol."""
        return list(self._open_interest.get(symbol, []))

    # ─── LIFECYCLE ──────────────────────────────────────────────────────────

    # Batching pour éviter le rate limit Bitget (code 30006) lors des souscriptions
    _SUBSCRIBE_BATCH_SIZE = 3   # symbols par batch (réduit pour éviter rate limit)
    _SUBSCRIBE_BATCH_DELAY = 3.0  # secondes entre les batchs (augmenté pour Bitget)

    async def start(self) -> None:
        """Démarre les connexions WebSocket avec staggering anti-rate-limit."""
        logger.info("DataEngine: démarrage...")

        self._exchange = ccxtpro.bitget({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })

        self._running = True

        # Lancer les tâches par batch pour éviter le rate limit Bitget
        assets = self.config.assets
        for i, asset in enumerate(assets):
            task = asyncio.create_task(
                self._watch_symbol(asset.symbol, asset.timeframes),
                name=f"watch_{asset.symbol}",
            )
            self._tasks.append(task)

            # Pause entre les batchs de souscriptions
            if (i + 1) % self._SUBSCRIBE_BATCH_SIZE == 0 and i + 1 < len(assets):
                logger.info(
                    "DataEngine: batch {}/{} souscrit, pause {}s...",
                    i + 1,
                    len(assets),
                    self._SUBSCRIBE_BATCH_DELAY,
                )
                await asyncio.sleep(self._SUBSCRIBE_BATCH_DELAY)

        # Tâche de flush buffer candles
        self._flush_task = asyncio.create_task(
            self._flush_candle_buffer(), name="flush_candles"
        )

        # Tâches de polling pour funding rate et OI
        self._tasks.append(
            asyncio.create_task(self._poll_funding_rates(), name="poll_funding")
        )
        self._tasks.append(
            asyncio.create_task(self._poll_open_interest(), name="poll_oi")
        )

        # Heartbeat : détecte les silences WS et déclenche full_reconnect si besoin
        self._last_candle_received = time.time()  # reset au démarrage
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name="heartbeat"
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

        # Annuler le heartbeat
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        self._heartbeat_task = None

        # Annuler la tâche de flush et flush final du buffer restant
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        self._flush_task = None

        # Flush final des candles en attente avant fermeture DB
        if self._write_buffer:
            try:
                batch = self._write_buffer.copy()
                self._write_buffer.clear()
                await self.db.insert_candles_batch(batch)
                logger.info("DataEngine: flush final {} candles", len(batch))
            except Exception as e:
                logger.error("DataEngine: erreur flush final: {}", e)

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

    # ─── AUTO-RECOVERY ─────────────────────────────────────────────────────

    async def restart_dead_tasks(self) -> int:
        """Relance les tâches de watch terminées (mortes).

        Retourne le nombre de tâches relancées.
        """
        restarted = 0
        new_tasks: list[asyncio.Task] = []

        for task in self._tasks:
            if task.done() and not task.cancelled():
                task_name = task.get_name()
                if task_name.startswith("watch_"):
                    symbol = task_name[6:]  # Retire "watch_"
                    asset = next(
                        (a for a in self.config.assets if a.symbol == symbol),
                        None,
                    )
                    if asset:
                        new_task = asyncio.create_task(
                            self._watch_symbol(symbol, asset.timeframes),
                            name=task_name,
                        )
                        new_tasks.append(new_task)
                        restarted += 1
                        logger.warning(
                            "DataEngine: tâche {} relancée (était morte)",
                            task_name,
                        )
                    # asset not found → skip (symbol retiré de la config)
                else:
                    new_tasks.append(task)
            else:
                new_tasks.append(task)

        self._tasks = new_tasks
        return restarted

    async def restart_stale_symbol(self, symbol: str) -> bool:
        """Kill et relance la task watch_ d'un symbol spécifique.

        Utilisé quand la task est vivante mais ne reçoit plus de données.
        Retourne True si la task a été relancée.
        """
        task_name = f"watch_{symbol}"

        for i, task in enumerate(self._tasks):
            if task.get_name() == task_name:
                # Cancel la task existante
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(task), timeout=5.0
                        )
                    except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                        pass

                # Trouver les timeframes pour ce symbol
                asset = next(
                    (a for a in self.config.assets if a.symbol == symbol),
                    None,
                )
                if asset is None:
                    logger.error(
                        "DataEngine: symbol {} non trouvé dans la config", symbol
                    )
                    return False

                # Relancer
                new_task = asyncio.create_task(
                    self._watch_symbol(symbol, asset.timeframes),
                    name=task_name,
                )
                self._tasks[i] = new_task
                logger.warning(
                    "DataEngine: task {} relancée (symbol stale)", task_name
                )
                return True

        logger.warning("DataEngine: task {} non trouvée", task_name)
        return False

    async def full_reconnect(self) -> None:
        """Recrée l'instance exchange et relance toutes les souscriptions.

        À utiliser quand restart_dead_tasks ne suffit pas (exchange cassé).
        """
        logger.warning("DataEngine: full reconnect — recréation exchange")

        # Fermer l'ancien exchange
        if self._exchange:
            try:
                await self._exchange.close()
            except Exception:
                pass

        # Recréer
        self._exchange = ccxtpro.bitget({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })

        # Annuler toutes les tâches
        for task in self._tasks:
            if not task.cancelled():
                task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Relancer toutes les tâches watch
        assets = self.config.assets
        for i, asset in enumerate(assets):
            task = asyncio.create_task(
                self._watch_symbol(asset.symbol, asset.timeframes),
                name=f"watch_{asset.symbol}",
            )
            self._tasks.append(task)

            if (i + 1) % self._SUBSCRIBE_BATCH_SIZE == 0 and i + 1 < len(assets):
                await asyncio.sleep(self._SUBSCRIBE_BATCH_DELAY)

        # Re-ajouter les tâches de polling
        self._tasks.append(
            asyncio.create_task(self._poll_funding_rates(), name="poll_funding")
        )
        self._tasks.append(
            asyncio.create_task(self._poll_open_interest(), name="poll_oi")
        )

        self._connected = True
        logger.warning(
            "DataEngine: full reconnect terminé ({} tâches)", len(self._tasks)
        )

    # ─── HEARTBEAT ──────────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Vérifie l'état du DataEngine toutes les minutes."""
        while self._running:
            try:
                await asyncio.sleep(60)
                self._heartbeat_tick += 1
                elapsed = time.time() - self._last_candle_received
                now_dt = datetime.now(tz=timezone.utc)

                # ── 1. Silence global → full_reconnect ──
                if elapsed > self._heartbeat_interval:
                    logger.warning(
                        "DataEngine: aucune candle depuis {:.0f}s — lancement full_reconnect",
                        elapsed,
                    )
                    if self._notifier:
                        try:
                            await self._notifier.notify_anomaly(
                                AnomalyType.DATA_STALE,
                                f"Aucune candle WS depuis {elapsed:.0f}s, reconnexion en cours",
                            )
                        except Exception as notif_err:
                            logger.warning(
                                "DataEngine: erreur alerte Telegram heartbeat: {}", notif_err
                            )
                    try:
                        await self.full_reconnect()
                        self._last_candle_received = time.time()
                        logger.info("DataEngine: heartbeat — reconnexion OK")
                    except Exception as e:
                        logger.error("DataEngine: full_reconnect échoué: {}", e)

                # ── 2. Relance des tasks watch_ mortes (chaque minute) ──
                try:
                    restarted = await self.restart_dead_tasks()
                    if restarted:
                        logger.warning(
                            "DataEngine: heartbeat a relancé {} task(s) morte(s)", restarted
                        )
                except Exception as e:
                    logger.warning("DataEngine: erreur restart_dead_tasks: {}", e)

                # ── 3. Stale per-symbol (toutes les 5 min) ──
                if self._heartbeat_tick % 5 == 0:
                    stale: list[tuple[str, float | None]] = []
                    for sym in self.get_all_symbols():
                        if sym in self._polling_modes:
                            continue
                        last = self._last_update_per_symbol.get(sym)
                        if last is None:
                            stale.append((sym, None))
                        else:
                            age = (now_dt - last).total_seconds()
                            if age > 300:
                                stale.append((sym, age))

                    if stale:
                        stale_names = [s[0] for s in stale]
                        logger.warning(
                            "DataEngine: {} symbol(s) sans données depuis 5min: {}",
                            len(stale), ", ".join(stale_names),
                        )

                        for sym, age in stale:
                            if age is not None and age <= 300:
                                continue
                            if sym in self._stale_abandoned:
                                continue

                            count = self._stale_restart_count.get(sym, 0)
                            if count >= 3:
                                # Mission 2026-03-06 : Basculer en mode POLLING
                                await self._start_polling(sym)
                                if self._notifier:
                                    try:
                                        await self._notifier.notify_anomaly(
                                            AnomalyType.DATA_STALE,
                                            f"⚠️ {sym} : passage en mode POLLING REST suite à {count} échecs WebSocket.",
                                        )
                                    except Exception:
                                        pass
                                continue

                            try:
                                restarted_sym = await self.restart_stale_symbol(sym)
                                if restarted_sym:
                                    self._stale_restart_count[sym] = count + 1
                                    logger.warning(
                                        "DataEngine: {} relancé (tentative {}/3)",
                                        sym, count + 1,
                                    )
                            except Exception as e:
                                logger.error("DataEngine: erreur restart {}: {}", sym, e)

                        # Escalade : si > 50% symbols stale > 15 min → full_reconnect
                        all_count = len(self.get_all_symbols())
                        if len(stale) > all_count // 2:
                            long_stale = [s for s, a in stale if a is not None and a > 900]
                            if len(long_stale) > 5:
                                logger.critical("DataEngine: {} symbols stale > 15min — full_reconnect", len(long_stale))
                                try:
                                    await self.full_reconnect()
                                    self._last_candle_received = time.time()
                                except Exception as e:
                                    logger.error("DataEngine: full_reconnect échoué: {}", e)

                        if self._notifier and len(stale) > 3:
                            try:
                                await self._notifier.notify_anomaly(
                                    AnomalyType.DATA_STALE,
                                    f"{len(stale)} symbols sans données: {', '.join(stale_names[:5])}",
                                )
                            except Exception as notif_err:
                                logger.warning("DataEngine: erreur alerte Telegram: {}", notif_err)

                # ── 4. Recovery Polling -> WebSocket (toutes les heures) ──
                if self._heartbeat_tick % 60 == 0 and self._polling_modes:
                    logger.info("DataEngine: tentative de récupération WebSocket pour : {}", ", ".join(self._polling_modes))
                    for sym in list(self._polling_modes):
                        await self._stop_polling(sym)
                        await self.restart_stale_symbol(sym)
                        if sym in self._stale_restart_count:
                            del self._stale_restart_count[sym]

                # ── 5. Log résumé toutes les 15 min ──
                if self._heartbeat_tick % 15 == 0:
                    all_syms = self.get_all_symbols()
                    active = sum(1 for sym in all_syms if sym in self._last_update_per_symbol 
                                 and (now_dt - self._last_update_per_symbol[sym]).total_seconds() < 120)
                    total_candles = sum(len(bufs) for sym_bufs in self._buffers.values() for bufs in sym_bufs.values())
                    logger.info("DataEngine: {}/{} symbols actifs, {} candles en buffer", active, len(all_syms), total_candles)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("DataEngine: erreur heartbeat loop: {}", e)

    # ─── WATCH LOOP ─────────────────────────────────────────────────────────

    async def _watch_symbol(
        self, symbol: str, timeframes: list[str]
    ) -> None:
        """Boucle de watch pour un symbol — ne s'arrête JAMAIS sauf cancel ou symbol invalide."""
        reconnect_delay = self.config.exchange.websocket.reconnect_delay
        attempt = 0

        if not timeframes:
            logger.error("DataEngine: aucun timeframe configuré pour {}", symbol)
            return

        source_tf = min(timeframes, key=lambda t: TimeFrame.from_string(t).to_minutes())
        target_tfs = [t for t in timeframes if t != source_tf]

        self._source_tfs[symbol] = source_tf
        self._target_tfs[symbol] = target_tfs

        logger.info("DataEngine: {} source_tf={}, targets={}", symbol, source_tf, target_tfs)

        while self._running:
            try:
                attempt += 1
                await self._subscribe_klines(symbol, [source_tf])
                attempt = 0
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._running: break
                err_str = str(e)
                if "does not have market symbol" in err_str:
                    logger.warning("DataEngine: {} retiré (non dispo)", symbol)
                    break
                logger.error("DataEngine: erreur watch {} : {} (tentative {})", symbol, e, attempt)
                delay = reconnect_delay * min(2 ** (attempt - 1), 300)
                await asyncio.sleep(delay)
                if attempt > 20: attempt = 10

    async def _subscribe_klines(
        self, symbol: str, timeframes: list[str]
    ) -> None:
        """S'abonne aux klines via ccxt watch_ohlcv."""
        assert self._exchange is not None
        consecutive_errors = 0

        while self._running:
            for tf in timeframes:
                if not self._running: return
                try:
                    ohlcv_list = await self._exchange.watch_ohlcv(symbol, tf)
                    consecutive_errors = 0
                    for ohlcv in ohlcv_list:
                        await self._on_candle_received(symbol, tf, ohlcv)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = any(code in err_str for code in ["30006", "429"])
                    if is_rate_limit:
                        consecutive_errors += 1
                        if consecutive_errors <= 3:
                            await asyncio.sleep(2.0 * consecutive_errors)
                        else: raise
                    else:
                        consecutive_errors += 1
                        if consecutive_errors <= 3:
                            logger.warning("DataEngine: erreur kline {}/{}: {}", symbol, tf, e)
                        if "does not have market symbol" in err_str: raise
                        await asyncio.sleep(1.0)

    async def _heal_gap(
        self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> None:
        """Récupère les candles manquantes via REST (Bitget) et les insère dans le buffer."""
        if self._exchange is None: return
        try:
            since_ms = int(start_dt.timestamp() * 1000) + 1
            tf = TimeFrame.from_string(timeframe)
            expected_delta = tf.to_milliseconds()
            actual_delta_ms = (end_dt - start_dt).total_seconds() * 1000
            missing_count = int(actual_delta_ms / expected_delta)

            if missing_count <= 1: return

            logger.info("DataEngine: guérison {} bougies pour {}/{}", missing_count - 1, symbol, timeframe)
            healed_ohlcv = await self._exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=missing_count)

            if healed_ohlcv:
                healed_candles = []
                for o in healed_ohlcv:
                    ts = datetime.fromtimestamp(o[0] / 1000, tz=timezone.utc)
                    if ts >= end_dt: continue
                    c = Candle(timestamp=ts, open=o[1], high=o[2], low=o[3], close=o[4], volume=o[5], symbol=symbol, timeframe=tf)
                    if self.validator.validate_candle(c):
                        healed_candles.append(c)

                if healed_candles:
                    buffer = self._buffers[symbol][timeframe]
                    existing_ts = {c.timestamp for c in buffer[-10:]}
                    to_add = [c for c in healed_candles if c.timestamp not in existing_ts]
                    if to_add:
                        buffer.extend(to_add)
                        buffer.sort(key=lambda x: x.timestamp)
                        self._write_buffer.extend(to_add)
                        logger.success("DataEngine: {} bougies guéries pour {}/{}", len(to_add), symbol, timeframe)
        except Exception as e:
            logger.error("DataEngine: échec guérison {}/{}: {}", symbol, timeframe, e)

    async def _aggregate_to_target_tf(
        self, symbol: str, source_tf_str: str, target_tf_str: str
    ) -> None:
        """Agrège les bougies du source_tf pour mettre à jour le target_tf."""
        target_tf = TimeFrame.from_string(target_tf_str)
        source_buffer = self._buffers[symbol][source_tf_str]
        if not source_buffer: return

        last_source = source_buffer[-1]
        period_start = target_tf.floor_timestamp(last_source.timestamp)
        constituents = [c for c in source_buffer if period_start <= c.timestamp < period_start + timedelta(minutes=target_tf.to_minutes())]
        if not constituents: return

        agg_candle = Candle(
            timestamp=period_start,
            open=constituents[0].open,
            high=max(c.high for c in constituents),
            low=min(c.low for c in constituents),
            close=constituents[-1].close,
            volume=sum(c.volume for c in constituents),
            symbol=symbol,
            timeframe=target_tf,
        )
        await self._store_and_dispatch(symbol, target_tf_str, agg_candle)

    async def _store_and_dispatch(
        self, symbol: str, timeframe_str: str, candle: Candle
    ) -> None:
        """Stocke une bougie en mémoire et DB, puis notifie les callbacks."""
        buffer = self._buffers[symbol][timeframe_str]
        if buffer and buffer[-1].timestamp == candle.timestamp:
            buffer[-1] = candle
        else:
            buffer.append(candle)
            if len(buffer) > MAX_BUFFER_SIZE: del buffer[: len(buffer) - MAX_BUFFER_SIZE]

        self._write_buffer.append(candle)
        for callback in self._callbacks:
            try:
                res = callback(symbol, timeframe_str, candle)
                if asyncio.iscoroutine(res): await res
            except Exception as e: logger.error("DataEngine: erreur callback: {}", e)

    async def _on_candle_received(
        self, symbol: str, timeframe_str: str, ohlcv: list
    ) -> None:
        """Traite une candle reçue du WebSocket (source TF)."""
        await self._process_ohlcv_item(symbol, timeframe_str, ohlcv, is_ws=True)

    async def _process_ohlcv_item(
        self, symbol: str, timeframe_str: str, ohlcv: list, is_ws: bool = True
    ) -> None:
        """Parse, valide et dispatch une bougie (WS ou Polling)."""
        try:
            tf = TimeFrame.from_string(timeframe_str)
            candle = Candle(
                timestamp=datetime.fromtimestamp(ohlcv[0] / 1000, tz=timezone.utc),
                open=float(ohlcv[1]), high=float(ohlcv[2]), low=float(ohlcv[3]), close=float(ohlcv[4]),
                volume=float(ohlcv[5]) if len(ohlcv) > 5 else 0.0,
                symbol=symbol, timeframe=tf
            )
        except (ValueError, IndexError) as e:
            logger.warning("DataEngine: candle malformée: {}", e)
            return

        if not self.validator.validate_candle(candle): return

        now_dt = datetime.now(tz=timezone.utc)
        self._last_update = now_dt
        self._last_candle_received = time.time()
        self._last_update_per_symbol[symbol] = now_dt

        if is_ws:
            if symbol in self._stale_restart_count: del self._stale_restart_count[symbol]
            if symbol in self._stale_abandoned: self._stale_abandoned.discard(symbol)

        buffer = self._buffers[symbol][timeframe_str]
        if buffer and buffer[-1].timestamp == candle.timestamp:
            await self._store_and_dispatch(symbol, timeframe_str, candle)
            for target_tf in self._target_tfs.get(symbol, []):
                await self._aggregate_to_target_tf(symbol, timeframe_str, target_tf)
            return

        if self.validator.is_duplicate(candle, buffer): return

        if is_ws and buffer and self.validator.check_gap(buffer[-1], candle, tf):
            self.gap_count += 1
            gap_from, gap_to = buffer[-1].timestamp, candle.timestamp
            logger.warning("DataEngine: gap détecté {}/{} entre {} et {}", symbol, timeframe_str, gap_from, gap_to)
            if self._notifier:
                try: await self._notifier.notify_anomaly(AnomalyType.DATA_GAP, f"{symbol}/{timeframe_str} gap")
                except Exception: pass
            await self._heal_gap(symbol, timeframe_str, gap_from, gap_to)

        await self._store_and_dispatch(symbol, timeframe_str, candle)
        for target_tf in self._target_tfs.get(symbol, []):
            await self._aggregate_to_target_tf(symbol, timeframe_str, target_tf)

    # ─── HYBRID POLLING ───────────────────────────────────────────────────

    async def _start_polling(self, symbol: str) -> None:
        """Bascule un symbole en mode Polling REST."""
        if symbol in self._polling_tasks: return
        source_tf = self._source_tfs.get(symbol)
        if not source_tf: return

        await self._stop_watch_task(symbol)
        self._polling_modes.add(symbol)
        task = asyncio.create_task(self._poll_symbol_rest(symbol, source_tf), name=f"poll_{symbol}")
        self._polling_tasks[symbol] = task
        logger.warning("DataEngine: passage en mode POLLING pour {} suite à instabilité WebSocket", symbol)

    async def _stop_polling(self, symbol: str) -> None:
        """Arrête le mode polling pour un symbole."""
        task = self._polling_tasks.pop(symbol, None)
        if task:
            task.cancel()
            try: await task
            except asyncio.CancelledError: pass
        self._polling_modes.discard(symbol)

    async def _stop_watch_task(self, symbol: str) -> None:
        """Arrête la tâche watch_ d'un symbole."""
        task_name = f"watch_{symbol}"
        new_tasks = []
        for task in self._tasks:
            if task.get_name() == task_name:
                if not task.done():
                    task.cancel()
                    try: await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
                    except Exception: pass
            else: new_tasks.append(task)
        self._tasks = new_tasks

    async def _poll_symbol_rest(self, symbol: str, timeframe: str) -> None:
        """Tâche de polling REST pour un symbole instable."""
        interval = 30
        while self._running and symbol in self._polling_modes:
            try:
                if not self._exchange:
                    await asyncio.sleep(1)
                    continue
                ohlcv_list = await self._exchange.fetch_ohlcv(symbol, timeframe, limit=5)
                if ohlcv_list:
                    for ohlcv in ohlcv_list:
                        await self._process_ohlcv_item(symbol, timeframe, ohlcv, is_ws=False)
            except asyncio.CancelledError: break
            except Exception as e: logger.error("DataEngine: erreur polling REST {}: {}", symbol, e)
            await asyncio.sleep(interval)

    # ─── FLUSH BUFFER ──────────────────────────────────────────────────────

    _FLUSH_INTERVAL = 30

    async def _flush_candle_buffer(self) -> None:
        """Flush périodique du buffer de candles vers la DB."""
        while self._running:
            try:
                await asyncio.sleep(self._FLUSH_INTERVAL)
                if self._write_buffer:
                    batch, self._write_buffer = self._write_buffer.copy(), []
                    try: await self.db.insert_candles_batch(batch)
                    except Exception as e: logger.error("DataEngine: erreur flush candles: {}", e)
            except asyncio.CancelledError: break

    # ─── POLLING FUNDING & OI ──────────────────────────────────────────────

    async def _poll_funding_rates(self) -> None:
        """Polling des funding rates toutes les 5 minutes."""
        while self._running:
            try: await self._fetch_funding_rates()
            except asyncio.CancelledError: break
            except Exception as e: logger.warning("DataEngine: erreur polling funding: {}", e)
            await asyncio.sleep(300)

    async def _poll_open_interest(self) -> None:
        """Polling de l'open interest toutes les 60 secondes."""
        while self._running:
            try: await self._fetch_open_interest()
            except asyncio.CancelledError: break
            except Exception as e: logger.warning("DataEngine: erreur polling OI: {}", e)
            await asyncio.sleep(60)

    @staticmethod
    def _to_swap_symbol(spot_symbol: str) -> str:
        if ":USDT" not in spot_symbol: return f"{spot_symbol}:USDT"
        return spot_symbol

    async def _fetch_funding_rates(self) -> None:
        if not self._exchange: return
        for asset in self.config.assets:
            try:
                swap_sym = self._to_swap_symbol(asset.symbol)
                result = await self._exchange.fetch_funding_rate(swap_sym)
                if result and "fundingRate" in result:
                    rate = result["fundingRate"]
                    if rate is not None: self._funding_rates[asset.symbol] = float(rate) * 100
            except Exception as e: logger.debug("DataEngine: funding rate non dispo pour {}: {}", asset.symbol, e)

    async def _fetch_open_interest(self) -> None:
        if not self._exchange: return
        now = datetime.now(tz=timezone.utc)
        for asset in self.config.assets:
            try:
                swap_sym = self._to_swap_symbol(asset.symbol)
                result = await self._exchange.fetch_open_interest(swap_sym)
                if result and "openInterestAmount" in result:
                    oi_value = float(result["openInterestAmount"])
                    snapshots = self._open_interest.setdefault(asset.symbol, [])
                    change_pct = 0.0
                    if snapshots:
                        prev = snapshots[-1].value
                        if prev > 0: change_pct = (oi_value - prev) / prev * 100
                    snapshots.append(OISnapshot(timestamp=now, symbol=asset.symbol, value=oi_value, change_pct=change_pct))
                    if len(snapshots) > self._oi_max_snapshots: self._open_interest[asset.symbol] = snapshots[-self._oi_max_snapshots:]
            except Exception as e: logger.debug("DataEngine: OI non dispo pour {}: {}", asset.symbol, e)
