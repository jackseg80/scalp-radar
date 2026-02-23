"""Executor : exécution d'ordres réels sur Bitget via ccxt Pro.

Sprint 12 : support grid DCA multi-niveaux (envelope_dca).
Sprint 5b : multi-stratégie, multi-paire, adaptive selector.
Pattern observer : reçoit les TradeEvent du Simulator via callback,
réplique en ordres réels sur Bitget.

Règle de sécurité #1 : JAMAIS de position sans SL.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from loguru import logger

from backend.core.models import Direction

if TYPE_CHECKING:
    from backend.alerts.notifier import Notifier
    from backend.core.config import AppConfig
    from backend.core.data_engine import DataEngine
    from backend.core.models import Candle
    from backend.execution.adaptive_selector import AdaptiveSelector
    from backend.execution.risk_manager import LiveRiskManager
    from backend.strategies.base_grid import BaseGridStrategy


# ─── TYPES ─────────────────────────────────────────────────────────────────


class TradeEventType(str, Enum):
    OPEN = "open"
    CLOSE = "close"


@dataclass
class TradeEvent:
    """Événement émis par le LiveStrategyRunner."""

    event_type: TradeEventType
    strategy_name: str
    symbol: str  # format spot "BTC/USDT"
    direction: str  # "LONG" | "SHORT"
    entry_price: float
    quantity: float
    tp_price: float
    sl_price: float
    score: float
    timestamp: datetime
    market_regime: str = ""
    exit_reason: str | None = None  # pour CLOSE
    exit_price: float | None = None  # pour CLOSE


@dataclass
class LivePosition:
    """Position live ouverte sur l'exchange."""

    symbol: str  # format futures "BTC/USDT:USDT"
    direction: str
    entry_price: float
    quantity: float
    entry_order_id: str
    sl_order_id: str | None = None
    tp_order_id: str | None = None
    entry_time: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    strategy_name: str = ""
    sl_price: float = 0.0
    tp_price: float = 0.0
    entry_fee: float = 0.0  # Fee réelle Bitget en USDT (Hotfix 34)


@dataclass
class GridLivePosition:
    """Position individuelle dans un cycle grid live."""

    level: int
    entry_price: float
    quantity: float
    entry_order_id: str
    entry_time: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    entry_fee: float = 0.0  # Fee réelle Bitget en USDT (Hotfix 34)


@dataclass
class GridLiveState:
    """État complet d'un cycle grid live sur un symbole.

    Un cycle grid = N positions ouvertes séquentiellement sur le même symbol,
    avec un SL global server-side et un TP client-side (SMA dynamique).
    Compté comme 1 "position" pour le RiskManager (max_concurrent, corrélation).
    """

    symbol: str  # format futures "BTC/USDT:USDT"
    direction: str  # "LONG" | "SHORT"
    strategy_name: str
    leverage: int
    positions: list[GridLivePosition] = field(default_factory=list)
    sl_order_id: str | None = None
    sl_price: float = 0.0
    opened_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    @property
    def total_quantity(self) -> float:
        return sum(p.quantity for p in self.positions)

    @property
    def avg_entry_price(self) -> float:
        if not self.positions:
            return 0.0
        total_notional = sum(p.quantity * p.entry_price for p in self.positions)
        return total_notional / self.total_quantity

    @property
    def total_entry_fees(self) -> float:
        """Somme des fees d'entrée réelles de tous les niveaux (Hotfix 34)."""
        return sum(p.entry_fee for p in self.positions)


# ─── MAPPING SYMBOLES ─────────────────────────────────────────────────────


def to_futures_symbol(spot_symbol: str) -> str:
    """Convertit un symbole spot en symbole futures swap Bitget (ccxt).

    Convention ccxt : BASE/QUOTE:SETTLE → ex. BTC/USDT:USDT
    Tous les perpetuels USDT Bitget suivent ce pattern.
    """
    if ":" in spot_symbol:
        return spot_symbol  # Déjà en format futures
    if spot_symbol.endswith("/USDT"):
        return f"{spot_symbol}:USDT"
    raise ValueError(f"Symbole non supporté pour futures: {spot_symbol}")


# ─── CONSTANTES ────────────────────────────────────────────────────────────

_SL_MAX_RETRIES = 3  # Nombre de tentatives pour placer le SL
_SL_RETRY_DELAY = 0.2  # Délai entre retries SL
_ORDER_DELAY = 0.1  # Délai entre ordres séquentiels (rate limiting)
_POLL_INTERVAL = 5  # Polling fallback en secondes
_ENTRY_TIMEOUT = 30  # Timeout pour le fill de l'ordre d'entrée


# ─── EXECUTOR ──────────────────────────────────────────────────────────────


class Executor:
    """Exécute les ordres réels sur Bitget.

    Responsabilités :
    1. Recevoir les TradeEvent du Simulator (via callback)
    2. Vérifier les pré-conditions (RiskManager + AdaptiveSelector)
    3. Passer les ordres via ccxt (entry market + SL/TP server-side)
    4. Surveiller les positions ouvertes (watchOrders + polling fallback)
    5. Détecter les fills TP/SL et synchroniser l'état
    """

    def __init__(
        self,
        config: AppConfig,
        risk_manager: LiveRiskManager,
        notifier: Notifier,
        selector: AdaptiveSelector | None = None,
    ) -> None:
        self._config = config
        self._risk_manager = risk_manager
        self._notifier = notifier
        self._selector = selector

        self._exchange: Any = None  # ccxt.pro.bitget (créé dans start)
        self._positions: dict[str, LivePosition] = {}
        self._grid_states: dict[str, GridLiveState] = {}  # {futures_sym: state}
        self._running = False
        self._connected = False
        self._watch_task: asyncio.Task[None] | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._balance_task: asyncio.Task[None] | None = None
        self._markets: dict[str, Any] = {}  # Cache load_markets()
        self._exchange_balance: float | None = None  # Hotfix 28a
        self._balance_refresh_interval: int = 300  # 5 minutes
        self._order_history: deque[dict] = deque(maxlen=200)  # Sprint 32
        self._leverage_applied: dict[str, int] = {}  # futures_sym → leverage au boot

        # Exit monitor autonome (Sprint Executor Autonome)
        self._data_engine: DataEngine | None = None
        self._simulator: Any = None  # Simulator — source unique indicateurs
        self._strategies: dict[str, BaseGridStrategy] = {}
        self._exit_check_task: asyncio.Task[None] | None = None

        # Phase 1 : entrées autonomes
        self._pending_levels: set[str] = set()  # "{futures_sym}:{level}" anti double-trigger
        self._pending_notional: float = 0.0  # Marge engagée pas encore reconciliée
        self._balance_bootstrapped: bool = False  # True après 1er fetch_balance réussi

        # Phase 2 : cooldown anti-churning
        self._last_close_time: dict[str, datetime] = {}  # {futures_sym: timestamp}

    # ─── Properties ────────────────────────────────────────────────────

    @property
    def is_enabled(self) -> bool:
        return self._config.secrets.live_trading

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def risk_manager(self) -> LiveRiskManager:
        return self._risk_manager

    @property
    def position(self) -> LivePosition | None:
        """Backward compat : première position ou None."""
        if not self._positions:
            return None
        return next(iter(self._positions.values()))

    @property
    def positions(self) -> list[LivePosition]:
        """Toutes les positions ouvertes."""
        return list(self._positions.values())

    @property
    def exchange_balance(self) -> float | None:
        """Dernier solde connu sur l'exchange (USDT total)."""
        return self._exchange_balance

    # Sandbox Bitget supprimé (cassé, ccxt #25523) — mainnet only

    # ─── Order History (Sprint 32) ────────────────────────────────────

    def _record_order(
        self,
        order_type: str,
        symbol: str,
        side: str,
        quantity: float,
        order_result: dict,
        strategy_name: str = "",
        context: str = "",
        paper_price: float = 0.0,
    ) -> None:
        """Enregistre un ordre reussi dans l'historique (FIFO, max 200)."""
        self._order_history.appendleft({
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "order_type": order_type,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "filled": float(order_result.get("filled") or 0),
            "average_price": float(order_result.get("average") or 0),
            "order_id": order_result.get("id", ""),
            "status": order_result.get("status", ""),
            "strategy_name": strategy_name,
            "context": context,
            "paper_price": paper_price,
        })

    def _update_order_price(
        self,
        order_id: str,
        real_price: float,
        fee: float | None = None,
    ) -> None:
        """Patche le prix reel et la fee dans l'historique pour un order_id donne.

        Appele apres _fetch_fill_price() pour retropropa le prix de fill Bitget
        dans l'enregistrement cree par _record_order() (qui a souvent average=0).
        """
        if not order_id or real_price <= 0:
            return
        for record in self._order_history:
            if record.get("order_id") == order_id:
                record["average_price"] = real_price
                if fee is not None:
                    record["fee"] = fee
                return

    # ─── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialise l'exchange ccxt Pro, réconcilie, lance la surveillance."""
        import ccxt.pro as ccxtpro

        self._exchange = ccxtpro.bitget({
            "apiKey": self._config.secrets.bitget_api_key,
            "secret": self._config.secrets.bitget_secret,
            "password": self._config.secrets.bitget_passphrase,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
            "sandbox": False,  # Sandbox Bitget cassé (ccxt #25523) — mainnet only
        })

        try:
            # 1. Charger les marchés (min_order_size, tick_size réels)
            self._markets = await self._exchange.load_markets()
            logger.info("Executor: {} marchés chargés", len(self._markets))

            # 2. Fetch balance pour le capital initial
            balance = await self._exchange.fetch_balance({
                "type": "swap",
            })
            coin = "USDT"
            free = float(balance.get("free", {}).get(coin, 0))
            total = float(balance.get("total", {}).get(coin, 0))
            self._risk_manager.set_initial_capital(total)
            self._exchange_balance = total
            logger.info(
                "Executor: balance USDT — libre={:.2f}, total={:.2f}", free, total,
            )

            # 3. Setup leverage pour tous les symboles configurés
            active_symbols: set[str] = set()
            for asset in self._config.assets:
                try:
                    futures_sym = to_futures_symbol(asset.symbol)
                    leverage = self._get_leverage_for_symbol(asset.symbol)
                    await self._setup_leverage_and_margin(futures_sym, leverage=leverage)
                    active_symbols.add(asset.symbol)
                except Exception as e:
                    logger.warning(
                        "Executor: setup échoué pour {} — désactivé: {}",
                        asset.symbol, e,
                    )

            if self._selector:
                self._selector.set_active_symbols(active_symbols)

            # 4. Réconciliation au boot
            await self._reconcile_on_boot()

            # 5. Lancer la surveillance
            self._running = True
            self._connected = True
            self._watch_task = asyncio.create_task(self._watch_orders_loop())
            self._poll_task = asyncio.create_task(self._poll_positions_loop())
            self._balance_task = asyncio.create_task(self._balance_refresh_loop())

            logger.info(
                "Executor: démarré en mode MAINNET ({} symboles actifs)",
                len(active_symbols),
            )

        except Exception as e:
            logger.error("Executor: échec démarrage: {}", e)
            self._connected = False
            raise

    async def stop(self) -> None:
        """Arrête la surveillance. NE ferme PAS les positions (TP/SL restent)."""
        self._running = False

        for task in (self._watch_task, self._poll_task, self._balance_task, self._exit_check_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._exchange:
            try:
                await self._exchange.close()
            except Exception as e:
                logger.warning("Executor: erreur fermeture exchange: {}", e)

        self._connected = False
        logger.info("Executor: arrêté (positions TP/SL restent sur exchange)")

    # ─── Balance refresh ────────────────────────────────────────────────

    async def refresh_balance(self) -> float | None:
        """Fetch le solde exchange et met à jour _exchange_balance.

        Log un WARNING si le solde change de plus de 10%.
        Retourne le nouveau solde ou None si échec.
        """
        if not self._exchange:
            return None
        try:
            balance = await self._exchange.fetch_balance({"type": "swap"})
            new_total = float(balance.get("total", {}).get("USDT", 0))
            old_total = self._exchange_balance

            if old_total is not None and old_total > 0:
                change_pct = abs(new_total - old_total) / old_total * 100
                if change_pct > 10:
                    logger.warning(
                        "Executor: balance changé de {:.1f}% ({:.2f} → {:.2f} USDT)",
                        change_pct, old_total, new_total,
                    )

            self._exchange_balance = new_total
            # Phase 1 : reset pending tracking après reconciliation balance
            self._balance_bootstrapped = True
            if not self._pending_levels:
                self._pending_notional = 0.0
            return new_total
        except Exception as e:
            logger.warning("Executor: échec refresh balance: {}", e)
            return None

    async def _balance_refresh_loop(self) -> None:
        """Boucle périodique de refresh du solde (toutes les 5 min)."""
        while self._running:
            await asyncio.sleep(self._balance_refresh_interval)
            if not self._running:
                break
            new_balance = await self.refresh_balance()
            # P1 Audit : snapshot pour kill switch global live (drawdown 24h)
            if new_balance is not None:
                self._risk_manager.record_balance_snapshot(new_balance)

    # ─── Balance bootstrap ──────────────────────────────────────────────

    async def _ensure_balance(self) -> float:
        """Retourne le solde Bitget disponible, avec fetch si pas encore initialisé."""
        if not self._balance_bootstrapped or self._exchange_balance is None:
            try:
                balance_info = await self._exchange.fetch_balance({"type": "swap"})
                self._exchange_balance = float(
                    balance_info.get("total", {}).get("USDT", 0),
                )
                self._balance_bootstrapped = True
                logger.info(
                    "Executor: balance bootstrappée: {:.2f} USDT", self._exchange_balance,
                )
            except Exception as e:
                logger.warning("Executor: échec bootstrap balance: {}", e)
                return 0.0
        return max((self._exchange_balance or 0.0) - self._pending_notional, 0.0)

    def _get_strategy_nb_assets(self, strategy_name: str) -> int:
        """Retourne le nombre d'assets configurés (per_asset) pour une stratégie.

        Utilisé pour diviser le capital disponible par asset, afin qu'un seul
        asset ne consomme pas toute la marge avant que les autres ne triggent.
        Retourne 1 si pas de per_asset (comportement conservateur).
        """
        strategy = self._strategies.get(strategy_name)
        if not strategy:
            return 1
        per_asset = getattr(strategy._config, "per_asset", None)
        if per_asset and isinstance(per_asset, dict) and len(per_asset) > 0:
            return len(per_asset)
        return 1

    # ─── Setup ─────────────────────────────────────────────────────────

    async def _setup_leverage_and_margin(
        self, futures_symbol: str, leverage: int | None = None,
    ) -> None:
        """Set leverage et margin mode, seulement s'il n'y a pas de position ouverte."""
        positions = await self._fetch_positions_safe(futures_symbol)
        has_open = any(
            float(p.get("contracts", 0)) > 0 for p in positions
        )

        if leverage is None:
            leverage = self._config.risk.position.default_leverage
        margin_mode = self._config.risk.margin.mode  # "cross" ou "isolated"

        if has_open:
            logger.warning(
                "Executor: position ouverte détectée — leverage/margin inchangés",
            )
            return

        # Position mode : one-way (pas de hedge long/short séparé)
        # Forcer l'option ccxt pour que create_order n'envoie pas tradeSide
        self._exchange.options["hedged"] = False
        try:
            await self._exchange.set_position_mode(
                False, futures_symbol,
            )
            logger.info("Executor: position mode set à 'one-way' pour {}", futures_symbol)
        except Exception as e:
            # Bitget renvoie une erreur si le mode est déjà celui demandé
            logger.debug("Executor: set_position_mode: {}", e)

        try:
            await self._exchange.set_leverage(
                leverage, futures_symbol,
            )
            self._leverage_applied[futures_symbol] = leverage
            logger.info("Executor: leverage set à {}x pour {}", leverage, futures_symbol)
        except Exception as e:
            logger.warning("Executor: impossible de set leverage: {}", e)

        try:
            await self._exchange.set_margin_mode(
                margin_mode, futures_symbol,
            )
            logger.info("Executor: margin mode set à '{}' pour {}", margin_mode, futures_symbol)
        except Exception as e:
            # Bitget renvoie une erreur si le mode est déjà celui demandé
            logger.debug("Executor: set_margin_mode: {}", e)

    # ─── Exit monitor autonome ─────────────────────────────────────────

    _EXIT_CHECK_INTERVAL = 60  # secondes

    def set_data_engine(self, data_engine: DataEngine) -> None:
        """Enregistre le DataEngine pour les checks TP/SL autonomes."""
        self._data_engine = data_engine

    def set_strategies(
        self,
        strategies: dict[str, BaseGridStrategy],
        simulator: Any = None,
    ) -> None:
        """Enregistre les instances de stratégie et le Simulator (source indicateurs)."""
        self._strategies = strategies
        self._simulator = simulator
        logger.info("Executor: {} stratégies enregistrées pour exit autonome", len(strategies))

    async def start_exit_monitor(self) -> None:
        """Démarre la boucle de vérification autonome des TP/SL."""
        if self._data_engine is None:
            logger.warning("Executor: pas de DataEngine, exit monitor désactivé")
            return
        if not self._strategies:
            logger.warning("Executor: pas de stratégies, exit monitor désactivé")
            return
        self._exit_check_task = asyncio.create_task(
            self._exit_monitor_loop(), name="executor_exit_monitor",
        )
        logger.info("Executor: exit monitor autonome démarré")

    async def _exit_monitor_loop(self) -> None:
        """Vérifie périodiquement si les positions live doivent être fermées."""
        while self._running:
            try:
                await asyncio.sleep(self._EXIT_CHECK_INTERVAL)
                if not self._running or not self._connected:
                    continue
                if self._grid_states:
                    logger.debug(
                        "Exit monitor: check {} positions ({})",
                        len(self._grid_states),
                        list(self._grid_states.keys()),
                    )
                await self._check_all_live_exits()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Executor: erreur exit monitor: {}", e)

    async def _check_all_live_exits(self) -> None:
        """Vérifie les conditions de sortie pour TOUTES les positions grid live."""
        for futures_sym in list(self._grid_states.keys()):
            try:
                await self._check_grid_exit(futures_sym)
            except Exception as e:
                logger.error("Executor: erreur check exit {}: {}", futures_sym, e)

    # ─── Entrées autonomes (Phase 1) ─────────────────────────────────────

    async def _on_candle(self, symbol: str, timeframe: str, candle: Candle) -> None:
        """Évalue les entrées grid sur chaque candle (autonome, sans dépendre du paper)."""
        if not self._running or not self._connected:
            return

        for strategy_name, strategy in self._strategies.items():
            strat_tf = getattr(strategy._config, "timeframe", "1h")
            if timeframe != strat_tf:
                continue

            futures_sym = to_futures_symbol(symbol)

            # Phase 2 : cooldown anti-churning
            raw_cd = getattr(strategy._config, "cooldown_candles", 0)
            cooldown = raw_cd if isinstance(raw_cd, (int, float)) else 0
            if cooldown > 0 and futures_sym not in self._grid_states:
                if futures_sym in self._last_close_time:
                    from backend.strategies.base_grid import TF_SECONDS
                    tf_seconds = TF_SECONDS.get(strat_tf, 3600)
                    elapsed = (candle.timestamp - self._last_close_time[futures_sym]).total_seconds()
                    if elapsed < cooldown * tf_seconds:
                        logger.debug(
                            "Executor entry: {} en cooldown ({:.0f}s/{:.0f}s), skip",
                            futures_sym, elapsed, cooldown * tf_seconds,
                        )
                        continue

            # Skip si grille pleine
            existing_state = self._grid_states.get(futures_sym)
            filled_levels: set[int] = set()
            if existing_state:
                filled_levels = {p.level for p in existing_state.positions}
                if len(existing_state.positions) >= strategy.max_positions:
                    continue

            # SOURCE UNIQUE : lire les indicateurs depuis le runner paper
            ctx = (
                self._simulator.get_runner_context(strategy_name, symbol)
                if self._simulator
                else None
            )
            if ctx is None:
                logger.debug(
                    "Executor entry: pas de contexte pour {} {} — skip",
                    strategy_name, symbol,
                )
                continue

            # Vérifier que les indicateurs essentiels sont présents
            tf_indicators = ctx.indicators.get(strat_tf, {})
            if not tf_indicators.get("sma") or not tf_indicators.get("close"):
                logger.debug(
                    "Executor entry: indicateurs incomplets pour {} {}, skip",
                    strategy_name, symbol,
                )
                continue

            # Remplacer le capital par le solde Bitget réel
            available_balance = await self._ensure_balance()
            if available_balance <= 0:
                logger.debug("Executor entry: balance <= 0, skip {}", symbol)
                continue

            # Diviser le capital par le nombre d'assets configurés.
            # Sans ça, le premier asset à trigger ses N levels peut consommer
            # toute la marge (ex: SOL prend 90% du compte sur 9 assets).
            nb_assets = self._get_strategy_nb_assets(strategy_name)
            allocated_balance = available_balance / nb_assets

            from backend.strategies.base import StrategyContext

            ctx = StrategyContext(
                symbol=ctx.symbol,
                timestamp=ctx.timestamp,
                candles=ctx.candles,
                indicators=ctx.indicators,
                current_position=ctx.current_position,
                capital=allocated_balance,
                config=ctx.config,
                extra_data=ctx.extra_data,
            )

            # Construire GridState depuis les positions LIVE
            grid_state = self._live_state_to_grid_state(futures_sym)

            # Évaluer la grille
            try:
                levels = strategy.compute_grid(ctx, grid_state)
            except Exception as e:
                logger.error(
                    "Executor entry: erreur compute_grid {} {}: {}",
                    strategy_name, symbol, e,
                )
                continue

            if not levels:
                continue

            grid_leverage = self._get_grid_leverage(strategy_name)

            # Vérifier quels niveaux sont touchés par cette candle
            for level in levels:
                if level.index in filled_levels:
                    continue

                # Anti double-trigger
                pending_key = f"{futures_sym}:{level.index}"
                if pending_key in self._pending_levels:
                    continue

                triggered = False
                if level.direction == Direction.LONG:
                    triggered = candle.low <= level.entry_price
                else:
                    triggered = candle.high >= level.entry_price

                if not triggered:
                    continue

                # Calculer la quantity sur le capital alloué par asset
                quantity = (
                    level.size_fraction * allocated_balance * grid_leverage
                ) / level.entry_price
                quantity = self._round_quantity(quantity, futures_sym)
                if quantity <= 0:
                    continue

                # Marquer comme pending AVANT l'appel async
                self._pending_levels.add(pending_key)
                level_margin = level.size_fraction * allocated_balance
                self._pending_notional += level_margin

                event = TradeEvent(
                    event_type=TradeEventType.OPEN,
                    strategy_name=strategy_name,
                    symbol=symbol,
                    direction=level.direction,
                    entry_price=level.entry_price,
                    quantity=quantity,
                    tp_price=0,
                    sl_price=0,
                    score=0,
                    timestamp=candle.timestamp,
                    market_regime=tf_indicators.get("regime", "unknown"),
                )

                try:
                    await self._open_grid_position(event)
                except Exception as e:
                    logger.error(
                        "Executor entry: erreur {} {} level {}: {}",
                        level.direction, futures_sym, level.index, e,
                    )
                finally:
                    self._pending_levels.discard(pending_key)

    # ─── GridState factory ────────────────────────────────────────────────

    def _live_state_to_grid_state(
        self, futures_sym: str, current_price: float | None = None,
    ) -> GridState:
        """Construit un GridState depuis les positions live.

        Factorisé pour être partagé entre _check_grid_exit() et _on_candle().
        """
        from backend.strategies.base_grid import GridPosition, GridState

        state = self._grid_states.get(futures_sym)
        if not state or not state.positions:
            return GridState(
                positions=[], avg_entry_price=0, total_quantity=0,
                total_notional=0, unrealized_pnl=0,
            )

        grid_positions = [
            GridPosition(
                level=p.level,
                direction=Direction(state.direction),
                entry_price=p.entry_price,
                quantity=p.quantity,
                entry_time=p.entry_time,
                entry_fee=getattr(p, "entry_fee", 0.0),
            )
            for p in state.positions
        ]

        total_qty = state.total_quantity
        avg_entry = state.avg_entry_price
        total_notional = avg_entry * total_qty

        unrealized = 0.0
        if current_price is not None and total_qty > 0:
            direction = Direction(state.direction)
            if direction == Direction.LONG:
                unrealized = (current_price - avg_entry) * total_qty
            else:
                unrealized = (avg_entry - current_price) * total_qty

        return GridState(
            positions=grid_positions,
            avg_entry_price=avg_entry,
            total_quantity=total_qty,
            total_notional=total_notional,
            unrealized_pnl=unrealized,
        )

    def _record_grid_close(self, futures_sym: str) -> None:
        """Enregistre le close pour le cooldown anti-churning."""
        self._last_close_time[futures_sym] = datetime.now(tz=timezone.utc)
        logger.debug("Executor: cooldown enregistré pour {}", futures_sym)

    async def _check_grid_exit(self, futures_sym: str) -> None:
        """Vérifie si un cycle grid doit être fermé.

        Lit les indicateurs depuis le runner paper (source unique de vérité pour
        la SMA/TP), puis vérifie le TP/SL avec le prix temps réel du DataEngine
        (réplique check_global_tp_sl du backtest). Fallback sur should_close_all()
        pour les cas spéciaux (TP inverse, signal BolTrend, etc.).
        """
        import math

        state = self._grid_states.get(futures_sym)
        if state is None or not state.positions:
            return

        strategy = self._strategies.get(state.strategy_name)
        if strategy is None:
            return

        # Convertir futures_sym ("BTC/USDT:USDT") → spot_sym ("BTC/USDT")
        spot_sym = futures_sym.split(":")[0] if ":" in futures_sym else futures_sym

        # ── SOURCE UNIQUE : lire les indicateurs du runner paper ──
        ctx = None
        if self._simulator is not None:
            ctx = self._simulator.get_runner_context(state.strategy_name, spot_sym)

        if ctx is None:
            logger.debug(
                "Exit monitor: pas de contexte runner pour {} {}, skip",
                state.strategy_name, spot_sym,
            )
            return

        strategy_tf = getattr(strategy._config, "timeframe", "1h")
        tf_indicators = ctx.indicators.get(strategy_tf, {})
        if "sma" not in tf_indicators or "close" not in tf_indicators:
            logger.debug(
                "Exit monitor: indicateurs incomplets pour {} (sma={}, close={}), skip",
                futures_sym, tf_indicators.get("sma"), tf_indicators.get("close"),
            )
            return

        current_close = tf_indicators["close"]

        # Construire le GridState (méthode factorisée Phase 1)
        grid_state = self._live_state_to_grid_state(futures_sym, current_close)
        avg_entry = grid_state.avg_entry_price
        total_qty = grid_state.total_quantity
        direction = Direction(state.direction)

        # ── PRIX TEMPS RÉEL : candle en cours du DataEngine ──
        # Permet de détecter un TP touché intra-candle (comme check_global_tp_sl
        # du backtest qui compare le HIGH). Le close de la candle en cours reflète
        # le dernier tick reçu via WebSocket.
        current_price = current_close  # fallback : close bougie fermée
        if self._data_engine is not None:
            buffers = getattr(self._data_engine, "_buffers", {})
            symbol_buffers = buffers.get(spot_sym, {})
            # Lire le timeframe le plus court disponible (prix le plus récent)
            for tf in ("1m", "5m", "15m", "1h"):
                tf_candles = symbol_buffers.get(tf, [])
                if tf_candles:
                    current_price = tf_candles[-1].close
                    break

        # ── CHECK TP/SL INTRA-CANDLE ──
        tp_price = strategy.get_tp_price(grid_state, tf_indicators)
        sl_price = strategy.get_sl_price(grid_state, tf_indicators)

        exit_reason: str | None = None
        if not math.isnan(tp_price):
            if direction == Direction.LONG and current_price >= tp_price:
                exit_reason = "tp_global"
            elif direction == Direction.SHORT and current_price <= tp_price:
                exit_reason = "tp_global"

        if exit_reason is None and not math.isnan(sl_price):
            if direction == Direction.LONG and current_price <= sl_price:
                exit_reason = "sl_global"
            elif direction == Direction.SHORT and current_price >= sl_price:
                exit_reason = "sl_global"

        # ── FALLBACK : should_close_all() pour les cas spéciaux ──
        # (TP inverse BolTrend : get_tp_price()=NaN, exit géré ici via signal SMA)
        if exit_reason is None:
            exit_reason = strategy.should_close_all(ctx, grid_state)

        if exit_reason is None:
            sma_val = tf_indicators.get("sma", 0.0)
            logger.debug(
                "Exit monitor: {} — price={:.6f}, sma={:.6f}, tp={:.6f}, no exit",
                futures_sym, current_price, sma_val, tp_price,
            )
            return

        logger.info(
            "Executor: EXIT AUTONOME {} {} — raison={}, price={:.6f}, tp={:.6f}, sl={:.6f}, avg_entry={:.6f}",
            state.direction, futures_sym, exit_reason,
            current_price, tp_price, sl_price, avg_entry,
        )

        # Réutiliser _close_grid_cycle avec un TradeEvent synthétique
        synthetic_event = TradeEvent(
            event_type=TradeEventType.CLOSE,
            strategy_name=state.strategy_name,
            symbol=spot_sym,
            direction=state.direction,
            entry_price=avg_entry,
            quantity=total_qty,
            tp_price=0.0,
            sl_price=0.0,
            score=0.0,
            timestamp=datetime.now(tz=timezone.utc),
            market_regime="unknown",
            exit_reason=exit_reason,
            exit_price=current_price,
        )
        await self._close_grid_cycle(synthetic_event)

    # ─── Ouverture de position ─────────────────────────────────────────

    async def _open_position(self, event: TradeEvent) -> None:
        """Ouvre une position réelle avec SL/TP server-side."""
        futures_sym = to_futures_symbol(event.symbol)

        # Déjà une position ouverte sur ce symbole ?
        if futures_sym in self._positions:
            logger.warning(
                "Executor: position déjà ouverte sur {}, ignore OPEN", futures_sym,
            )
            return

        # Exclusion mutuelle : pas de mono si grid active sur même symbol
        if futures_sym in self._grid_states:
            logger.warning(
                "Executor: cycle grid actif sur {}, ignore OPEN mono", futures_sym,
            )
            return

        # Pre-trade check
        balance = await self._exchange.fetch_balance({
            "type": "swap",
        })
        coin = "USDT"
        free = float(balance.get("free", {}).get(coin, 0))
        total = float(balance.get("total", {}).get(coin, 0))

        quantity = self._round_quantity(event.quantity, futures_sym)
        if quantity <= 0:
            logger.warning("Executor: quantité arrondie à 0, trade ignoré")
            return

        ok, reason = self._risk_manager.pre_trade_check(
            futures_sym, event.direction, quantity,
            event.entry_price, free, total,
        )
        if not ok:
            logger.warning("Executor: trade rejeté — {}", reason)
            return

        # 1. Ordre d'entrée (market)
        side = "buy" if event.direction == "LONG" else "sell"
        try:
            entry_order = await self._exchange.create_order(
                futures_sym, "market", side, quantity,
            )
            self._record_order("entry", futures_sym, side, quantity, entry_order, event.strategy_name, "mono", paper_price=event.entry_price)
        except Exception as e:
            logger.error("Executor: échec ordre d'entrée: {}", e)
            return

        entry_order_id = entry_order.get("id", "")
        filled_qty = float(entry_order.get("filled") or quantity)

        # Hotfix 34 : prix de fill réel Bitget (pas paper)
        avg_price_raw = entry_order.get("average")
        entry_fee = 0.0
        if avg_price_raw and float(avg_price_raw) > 0:
            avg_price = float(avg_price_raw)
            fee_info = entry_order.get("fee") or {}
            entry_fee = float(fee_info.get("cost") or 0) if fee_info.get("cost") is not None else 0.0
        else:
            avg_price, fetched_fee = await self._fetch_fill_price(
                entry_order_id, futures_sym, event.entry_price,
            )
            entry_fee = fetched_fee if fetched_fee is not None else 0.0

        # Sprint Journal V2 : retropropager le prix reel dans l'historique
        self._update_order_price(entry_order_id, avg_price, entry_fee if entry_fee > 0 else None)

        if filled_qty <= 0:
            logger.error("Executor: ordre d'entrée non rempli")
            return

        if avg_price != event.entry_price and event.entry_price > 0:
            slippage = (avg_price - event.entry_price) / event.entry_price * 100
            logger.info(
                "Executor: entry slippage {} {:.4f}% (paper={:.6f}, real={:.6f})",
                futures_sym, slippage, event.entry_price, avg_price,
            )

        logger.info(
            "Executor: ENTRY {} {} {:.6f} @ {:.2f} (order={})",
            event.direction, futures_sym, filled_qty, avg_price, entry_order_id,
        )

        await asyncio.sleep(_ORDER_DELAY)

        # 2. Placer SL (market trigger) — CRITIQUE, retry si échec
        close_side = "sell" if event.direction == "LONG" else "buy"
        sl_price = self._round_price(event.sl_price, futures_sym)
        tp_price = self._round_price(event.tp_price, futures_sym)

        sl_order_id = await self._place_sl_with_retry(
            futures_sym, close_side, filled_qty, sl_price,
            event.strategy_name,
        )

        if sl_order_id is None:
            # SL impossible → close market immédiat (règle #1)
            logger.critical(
                "Executor: SL IMPOSSIBLE — close market immédiat pour {}",
                futures_sym,
            )
            try:
                emg_order = await self._exchange.create_order(
                    futures_sym, "market", close_side, filled_qty,
                    params={"reduceOnly": True},
                )
                self._record_order("emergency_close", futures_sym, close_side, filled_qty, emg_order, event.strategy_name, "sl_failed")
            except Exception as e:
                logger.critical("Executor: ÉCHEC close market urgence: {}", e)

            await self._notifier.notify_live_sl_failed(
                futures_sym, event.strategy_name,
            )
            return

        await asyncio.sleep(_ORDER_DELAY)

        # 3. Placer TP (limit trigger) — moins critique
        tp_order_id = await self._place_tp(
            futures_sym, close_side, filled_qty, tp_price,
        )

        # 4. Enregistrer la position
        self._positions[futures_sym] = LivePosition(
            symbol=futures_sym,
            direction=event.direction,
            entry_price=avg_price,
            quantity=filled_qty,
            entry_order_id=entry_order_id,
            sl_order_id=sl_order_id,
            tp_order_id=tp_order_id,
            strategy_name=event.strategy_name,
            sl_price=sl_price,
            tp_price=tp_price,
            entry_fee=entry_fee,
        )

        self._risk_manager.register_position({
            "symbol": futures_sym,
            "direction": event.direction,
            "entry_price": avg_price,
            "quantity": filled_qty,
        })

        await self._notifier.notify_live_order_opened(
            futures_sym, event.direction, filled_qty, avg_price,
            sl_price, tp_price,
            event.strategy_name, entry_order_id,
        )

    async def _place_sl_with_retry(
        self,
        symbol: str,
        side: str,
        quantity: float,
        sl_price: float,
        strategy_name: str,
    ) -> str | None:
        """Place le SL avec retries. Retourne l'order_id ou None si échec total."""
        for attempt in range(1, _SL_MAX_RETRIES + 1):
            try:
                sl_order = await self._exchange.create_order(
                    symbol, "market", side, quantity,
                    params={
                        "triggerPrice": sl_price,
                        "triggerType": "mark_price",
                        "reduceOnly": True,
                    },
                )
                sl_id = sl_order.get("id", "")
                self._record_order("sl", symbol, side, quantity, sl_order, strategy_name, f"retry_{attempt}")
                logger.info(
                    "Executor: SL placé @ {:.2f} (order={}, tentative {})",
                    sl_price, sl_id, attempt,
                )
                return sl_id
            except Exception as e:
                logger.warning(
                    "Executor: échec SL tentative {}/{}: {}",
                    attempt, _SL_MAX_RETRIES, e,
                )
                if attempt < _SL_MAX_RETRIES:
                    await asyncio.sleep(_SL_RETRY_DELAY)

        return None

    async def _place_tp(
        self,
        symbol: str,
        side: str,
        quantity: float,
        tp_price: float,
    ) -> str | None:
        """Place le TP (limit trigger). Retourne l'order_id ou None."""
        try:
            tp_order = await self._exchange.create_order(
                symbol, "limit", side, quantity, tp_price,
                params={
                    "triggerPrice": tp_price,
                    "triggerType": "mark_price",
                    "reduceOnly": True,
                },
            )
            tp_id = tp_order.get("id", "")
            self._record_order("tp", symbol, side, quantity, tp_order)
            logger.info("Executor: TP placé @ {:.2f} (order={})", tp_price, tp_id)
            return tp_id
        except Exception as e:
            logger.warning("Executor: échec placement TP: {}", e)
            return None

    # ─── Grid DCA : ouverture / fermeture / SL ─────────────────────────

    async def _open_grid_position(self, event: TradeEvent) -> None:
        """Ouvre un niveau de la grille DCA.

        Différences avec _open_position :
        - Autorise PLUSIEURS positions sur le même symbol (niveaux DCA)
        - Pre-trade check uniquement au 1er niveau (un cycle = 1 slot)
        - Met à jour le SL global après chaque nouvelle entrée
        - PAS de TP trigger (TP dynamique = SMA, géré par le runner)
        """
        futures_sym = to_futures_symbol(event.symbol)

        # Exclusion mutuelle mono/grid
        if futures_sym in self._positions:
            logger.warning(
                "Executor: position mono active sur {}, ignore OPEN grid", futures_sym,
            )
            return

        state = self._grid_states.get(futures_sym)
        is_first_level = state is None

        # Guard : si c'est un nouveau cycle, vérifier qu'il n'y a pas déjà
        # une position sur Bitget (évite les doublons après desync)
        if is_first_level:
            try:
                positions = await self._fetch_positions_safe(futures_sym)
                has_exchange_position = any(
                    float(p.get("contracts", 0)) > 0 for p in positions
                )
                if has_exchange_position:
                    logger.warning(
                        "Executor: position {} déjà sur exchange mais pas dans grid_states — OPEN ignoré",
                        futures_sym,
                    )
                    return
            except Exception as e:
                logger.warning("Executor: échec vérification position exchange {}: {}", futures_sym, e)

        # Hotfix 35 : limiter le nombre de NOUVEAUX cycles grid simultanés
        if is_first_level:
            max_live_grids_raw = getattr(self._config.risk, "max_live_grids", 4)
            max_live_grids = max_live_grids_raw if isinstance(max_live_grids_raw, int) else 4
            active_grids = len(self._grid_states)
            if active_grids >= max_live_grids:
                logger.warning(
                    "Executor: max grids live atteint ({}/{}), nouveau cycle {} ignoré",
                    active_grids,
                    max_live_grids,
                    event.symbol,
                )
                return

        # Pre-trade check UNIQUEMENT au 1er niveau (Bug 2 fix)
        if is_first_level:
            grid_leverage = self._get_grid_leverage(event.strategy_name)

            # Setup leverage au 1er trade grid — skip si déjà correct au boot
            if self._leverage_applied.get(futures_sym) == grid_leverage:
                logger.debug(
                    "Executor: leverage {}x déjà appliqué au boot pour {}",
                    grid_leverage, futures_sym,
                )
            else:
                try:
                    await self._exchange.set_leverage(
                        grid_leverage, futures_sym,
                    )
                    self._leverage_applied[futures_sym] = grid_leverage
                    logger.info(
                        "Executor: leverage grid set a {}x pour {}",
                        grid_leverage, futures_sym,
                    )
                except Exception as e:
                    logger.warning("Executor: set leverage grid: {}", e)

            balance = await self._exchange.fetch_balance(
                {"type": "swap"},
            )
            coin = "USDT"
            free = float(balance.get("free", {}).get(coin, 0))
            total = float(balance.get("total", {}).get(coin, 0))

            quantity = self._round_quantity(event.quantity, futures_sym)
            if quantity <= 0:
                logger.warning("Executor: grid quantité arrondie à 0, trade ignoré")
                return

            ok, reason = self._risk_manager.pre_trade_check(
                futures_sym, event.direction, quantity,
                event.entry_price, free, total,
                leverage_override=grid_leverage,
            )
            if not ok:
                logger.warning("Executor: grid trade rejeté — {}", reason)
                return
        else:
            # P0 Audit : vérifier le kill switch même aux niveaux 2+
            if self._risk_manager.is_kill_switch_triggered:
                logger.warning(
                    "Executor: kill switch live actif, niveau grid {} ignoré pour {}",
                    len(state.positions) + 1, futures_sym,
                )
                return
            quantity = self._round_quantity(event.quantity, futures_sym)
            if quantity <= 0:
                logger.warning("Executor: grid quantité arrondie à 0, trade ignoré")
                return

        # Market entry
        side = "buy" if event.direction == "LONG" else "sell"
        try:
            entry_order = await self._exchange.create_order(
                futures_sym, "market", side, quantity,
            )
            self._record_order("entry", futures_sym, side, quantity, entry_order, event.strategy_name, "grid", paper_price=event.entry_price)
        except Exception as e:
            logger.error("Executor: échec grid entry: {}", e)
            return

        filled_qty = float(entry_order.get("filled") or quantity)
        order_id = entry_order.get("id", "")

        # Hotfix 34 : prix de fill réel Bitget (pas paper)
        avg_price_raw = entry_order.get("average")
        entry_fee = 0.0
        if avg_price_raw and float(avg_price_raw) > 0:
            avg_price = float(avg_price_raw)
            fee_info = entry_order.get("fee") or {}
            entry_fee = float(fee_info.get("cost") or 0) if fee_info.get("cost") is not None else 0.0
        else:
            avg_price, fetched_fee = await self._fetch_fill_price(
                order_id, futures_sym, event.entry_price,
            )
            entry_fee = fetched_fee if fetched_fee is not None else 0.0

        # Sprint Journal V2 : retropropager le prix reel dans l'historique
        self._update_order_price(order_id, avg_price, entry_fee if entry_fee > 0 else None)

        if filled_qty <= 0:
            logger.error("Executor: grid entry non remplie")
            return

        if avg_price != event.entry_price and event.entry_price > 0:
            slippage = (avg_price - event.entry_price) / event.entry_price * 100
            logger.info(
                "Executor: grid entry slippage {} {:.4f}% (paper={:.6f}, real={:.6f})",
                futures_sym, slippage, event.entry_price, avg_price,
            )

        # Créer ou mettre à jour GridLiveState
        if is_first_level:
            state = GridLiveState(
                symbol=futures_sym,
                direction=event.direction,
                strategy_name=event.strategy_name,
                leverage=self._get_grid_leverage(event.strategy_name),
            )
            self._grid_states[futures_sym] = state

            # Register dans RiskManager (1 cycle = 1 position)
            # NOTE: la quantité trackée par le RM ne sera pas mise à jour
            # aux niveaux suivants — dette technique acceptable.
            self._risk_manager.register_position({
                "symbol": futures_sym,
                "direction": event.direction,
                "entry_price": avg_price,
                "quantity": filled_qty,
            })

        level_num = len(state.positions)
        state.positions.append(GridLivePosition(
            level=level_num,
            entry_price=avg_price,
            quantity=filled_qty,
            entry_order_id=order_id,
            entry_fee=entry_fee,
        ))

        # Rate limiting (comme _open_position)
        await asyncio.sleep(_ORDER_DELAY)

        # Recalculer et replacer le SL global
        await self._update_grid_sl(futures_sym, state)

        # Telegram
        await self._notifier.notify_grid_level_opened(
            event.symbol, event.direction, level_num,
            filled_qty, avg_price,
            state.avg_entry_price, state.sl_price,
            event.strategy_name,
        )

        logger.info(
            "Executor: GRID {} level {} {} {:.6f} @ {:.2f} (avg={:.2f}, SL={:.2f})",
            event.direction, level_num, futures_sym,
            filled_qty, avg_price, state.avg_entry_price, state.sl_price,
        )

    async def _update_grid_sl(
        self, futures_sym: str, state: GridLiveState,
    ) -> None:
        """Annule l'ancien SL et place un nouveau basé sur le prix moyen.

        Appelée à chaque ouverture de niveau (le prix moyen change → le SL change).
        Quantité du SL = total_quantity (toutes les positions agrégées).
        """
        # 1. Annuler l'ancien SL s'il existe
        if state.sl_order_id:
            try:
                await self._exchange.cancel_order(
                    state.sl_order_id, futures_sym,
                )
                logger.info(
                    "Executor: ancien SL {} annulé pour {}", state.sl_order_id, futures_sym,
                )
            except Exception as e:
                logger.warning(
                    "Executor: échec cancel ancien SL {} pour {}: {} — fallback cancel_all",
                    state.sl_order_id, futures_sym, e,
                )
                await self._cancel_all_open_orders(futures_sym)

        # 2. Calculer nouveau SL
        sl_pct = self._get_grid_sl_percent(state.strategy_name)
        if state.direction == "LONG":
            new_sl = state.avg_entry_price * (1 - sl_pct / 100)
        else:
            new_sl = state.avg_entry_price * (1 + sl_pct / 100)
        new_sl = self._round_price(new_sl, futures_sym)

        # 3. Placer SL (retry 3x) — quantité = total agrégé
        close_side = "sell" if state.direction == "LONG" else "buy"
        sl_order_id = await self._place_sl_with_retry(
            futures_sym, close_side, state.total_quantity, new_sl,
            state.strategy_name,
        )

        if sl_order_id is None:
            # Règle #1 : JAMAIS de position sans SL
            logger.critical(
                "Executor: SL GRID IMPOSSIBLE — close urgence {}", futures_sym,
            )
            await self._emergency_close_grid(futures_sym, state)
            return

        state.sl_order_id = sl_order_id
        state.sl_price = new_sl

    async def _emergency_close_grid(
        self, futures_sym: str, state: GridLiveState,
    ) -> None:
        """Fermeture d'urgence si SL impossible. JAMAIS de position sans SL."""
        close_side = "sell" if state.direction == "LONG" else "buy"
        try:
            emg_order = await self._exchange.create_order(
                futures_sym, "market", close_side, state.total_quantity,
                params={"reduceOnly": True},
            )
            self._record_order("emergency_close", futures_sym, close_side, state.total_quantity, emg_order, state.strategy_name, "grid_sl_failed")
        except Exception as e:
            logger.critical("Executor: ÉCHEC close urgence grid: {}", e)

        self._risk_manager.unregister_position(futures_sym)
        self._record_grid_close(futures_sym)
        del self._grid_states[futures_sym]
        await self._notifier.notify_live_sl_failed(futures_sym, state.strategy_name)

    async def _cancel_all_open_orders(self, futures_sym: str) -> int:
        """Annule TOUS les ordres ouverts (limit + trigger) pour un symbol.

        Utilisé à la fermeture d'un cycle grid pour nettoyer les éventuels
        ordres trigger orphelins (anciens SL dont le cancel a échoué).
        Sur Bitget, les trigger orders (SL) requièrent un endpoint séparé.
        """
        cancelled = 0

        # 1. Ordres normaux (limit, market)
        try:
            open_orders = await self._exchange.fetch_open_orders(
                futures_sym, params={"type": "swap"},
            )
            for order in open_orders:
                try:
                    await self._exchange.cancel_order(order["id"], futures_sym)
                    cancelled += 1
                except Exception as e:
                    logger.warning(
                        "Executor: échec cancel ordre {} sur {}: {}",
                        order.get("id"), futures_sym, e,
                    )
        except Exception as e:
            logger.error("Executor: échec fetch_open_orders {}: {}", futures_sym, e)

        # 2. Trigger orders (SL trigger) — endpoint séparé sur Bitget
        try:
            trigger_orders = await self._exchange.fetch_open_orders(
                futures_sym, params={"type": "swap", "stop": True},
            )
            for order in trigger_orders:
                try:
                    await self._exchange.cancel_order(
                        order["id"], futures_sym,
                        params={"stop": True},
                    )
                    cancelled += 1
                except Exception as e:
                    logger.warning(
                        "Executor: échec cancel trigger {} sur {}: {}",
                        order.get("id"), futures_sym, e,
                    )
        except Exception as e:
            logger.error("Executor: échec fetch trigger orders {}: {}", futures_sym, e)

        if cancelled:
            logger.info(
                "Executor: {} ordre(s) annulé(s) pour {}", cancelled, futures_sym,
            )
        return cancelled

    async def _close_grid_cycle(self, event: TradeEvent) -> None:
        """Ferme toutes les positions d'un cycle DCA.

        Déclenché par :
        - tp_global (retour à la SMA, détecté par le runner)
        - sl_global (SL trigger exécuté sur exchange, détecté par watchOrders)
        - autre (signal_exit, etc.)
        """
        futures_sym = to_futures_symbol(event.symbol)
        state = self._grid_states.get(futures_sym)
        if state is None:
            return

        close_side = "sell" if state.direction == "LONG" else "buy"

        # 1. Annuler TOUS les ordres ouverts (pas juste sl_order_id)
        #    Nettoie les SL orphelins accumulés si des cancels ont échoué
        if event.exit_reason != "sl_global":
            await self._cancel_all_open_orders(futures_sym)

        # 2. Market close (sauf si SL déjà exécuté sur exchange)
        if event.exit_reason != "sl_global":
            try:
                close_order = await self._exchange.create_order(
                    futures_sym, "market", close_side, state.total_quantity,
                    params={"reduceOnly": True},
                )
                self._record_order("close", futures_sym, close_side, state.total_quantity, close_order, state.strategy_name, "grid_cycle", paper_price=event.exit_price or 0.0)
                # Hotfix 34 : prix de fill réel + fees
                exit_price_raw = close_order.get("average")
                exit_fee: float | None = None
                if exit_price_raw and float(exit_price_raw) > 0:
                    exit_price = float(exit_price_raw)
                    fee_info = close_order.get("fee") or {}
                    exit_fee = float(fee_info.get("cost") or 0) if fee_info.get("cost") is not None else None
                else:
                    exit_price, exit_fee = await self._fetch_fill_price(
                        close_order.get("id", ""), futures_sym, event.exit_price or 0,
                    )
                # Sprint Journal V2 : retropropager le prix reel dans l'historique
                self._update_order_price(close_order.get("id", ""), exit_price, exit_fee)
            except Exception as e:
                logger.error("Executor: échec close grid {}: {}", futures_sym, e)
                self._risk_manager.unregister_position(futures_sym)
                self._record_grid_close(futures_sym)
                del self._grid_states[futures_sym]
                return
        else:
            exit_price = event.exit_price or state.sl_price
            exit_fee = None  # SL exécuté par exchange, fee gérée par handler

        # 3. P&L net — fees réelles si disponibles (Hotfix 34)
        if exit_fee is not None:
            net_pnl = self._calculate_real_pnl(
                state.direction, state.avg_entry_price, exit_price,
                state.total_quantity, state.total_entry_fees, exit_fee,
            )
        else:
            net_pnl = self._calculate_pnl(
                state.direction, state.avg_entry_price, exit_price, state.total_quantity,
            )

        # 4. RiskManager
        from backend.execution.risk_manager import LiveTradeResult

        self._risk_manager.record_trade_result(LiveTradeResult(
            net_pnl=net_pnl,
            timestamp=datetime.now(tz=timezone.utc),
            symbol=futures_sym,
            direction=state.direction,
            exit_reason=event.exit_reason or "unknown",
            strategy_name=state.strategy_name,
        ))
        self._risk_manager.unregister_position(futures_sym)

        # 5. Telegram
        await self._notifier.notify_grid_cycle_closed(
            event.symbol, state.direction,
            len(state.positions), state.avg_entry_price, exit_price,
            net_pnl, event.exit_reason or "unknown",
            state.strategy_name,
        )

        logger.info(
            "Executor: GRID CLOSE {} {} — {} niveaux, avg={:.2f} -> {:.2f}, "
            "net={:+.2f} ({})",
            state.direction, futures_sym, len(state.positions),
            state.avg_entry_price, exit_price, net_pnl, event.exit_reason,
        )

        # 6. Cleanup
        self._record_grid_close(futures_sym)
        del self._grid_states[futures_sym]

        # 7. Sync fermeture vers le runner paper (même prix, même moment)
        if self._simulator is not None:
            try:
                spot_sym = futures_sym.split(":")[0] if ":" in futures_sym else futures_sym
                self._simulator.force_close_grid(
                    strategy_name=state.strategy_name,
                    symbol=spot_sym,
                    exit_price=exit_price,
                    exit_reason=event.exit_reason or "tp_global",
                )
            except Exception as e:
                logger.warning(
                    "Executor: sync close vers paper échoué pour {}: {}", futures_sym, e
                )

    async def _handle_grid_sl_executed(
        self, futures_sym: str, state: GridLiveState, exit_price: float,
        exit_fee: float | None = None,
    ) -> None:
        """Traite l'exécution du SL grid par Bitget (watchOrders ou polling)."""
        # Hotfix 34 : fees réelles si disponibles
        if exit_fee is not None:
            net_pnl = self._calculate_real_pnl(
                state.direction, state.avg_entry_price, exit_price,
                state.total_quantity, state.total_entry_fees, exit_fee,
            )
        else:
            net_pnl = self._calculate_pnl(
                state.direction, state.avg_entry_price, exit_price, state.total_quantity,
            )

        from backend.execution.risk_manager import LiveTradeResult

        self._risk_manager.record_trade_result(LiveTradeResult(
            net_pnl=net_pnl,
            timestamp=datetime.now(tz=timezone.utc),
            symbol=futures_sym,
            direction=state.direction,
            exit_reason="sl_global",
            strategy_name=state.strategy_name,
        ))
        self._risk_manager.unregister_position(futures_sym)

        # Convertir futures → spot pour le notifier
        spot_sym = futures_sym.split(":")[0] if ":" in futures_sym else futures_sym
        await self._notifier.notify_grid_cycle_closed(
            spot_sym, state.direction,
            len(state.positions), state.avg_entry_price, exit_price,
            net_pnl, "sl_global", state.strategy_name,
        )

        logger.info(
            "Executor: SL grid exécuté {} — net={:+.2f}", futures_sym, net_pnl,
        )

        # Nettoyer les éventuels ordres trigger orphelins restants
        await self._cancel_all_open_orders(futures_sym)

        self._record_grid_close(futures_sym)
        del self._grid_states[futures_sym]

        # Sync SL vers le runner paper
        if self._simulator is not None:
            try:
                self._simulator.force_close_grid(
                    strategy_name=state.strategy_name,
                    symbol=spot_sym,
                    exit_price=exit_price,
                    exit_reason="sl_global",
                )
            except Exception as e:
                logger.warning(
                    "Executor: sync SL vers paper échoué pour {}: {}", futures_sym, e
                )

    # ─── Fermeture de position ─────────────────────────────────────────

    async def _close_position(self, event: TradeEvent) -> None:
        """Ferme la position réelle (signal_exit ou regime_change)."""
        futures_sym = to_futures_symbol(event.symbol)
        pos = self._positions.get(futures_sym)
        if pos is None:
            return

        close_side = "sell" if pos.direction == "LONG" else "buy"

        # 1. Annuler SL/TP en attente
        await self._cancel_pending_orders(futures_sym)

        # 2. Market close
        try:
            close_order = await self._exchange.create_order(
                futures_sym, "market", close_side, pos.quantity,
                params={"reduceOnly": True},
            )
            self._record_order("close", futures_sym, close_side, pos.quantity, close_order, event.strategy_name, f"mono_{event.exit_reason or 'unknown'}", paper_price=event.exit_price or 0.0)
            # Hotfix 34 : prix de fill réel + fees
            exit_price_raw = close_order.get("average")
            exit_fee: float | None = None
            if exit_price_raw and float(exit_price_raw) > 0:
                exit_price = float(exit_price_raw)
                fee_info = close_order.get("fee") or {}
                exit_fee = float(fee_info.get("cost") or 0) if fee_info.get("cost") is not None else None
            else:
                exit_price, exit_fee = await self._fetch_fill_price(
                    close_order.get("id", ""), futures_sym, event.exit_price or 0,
                )
            # Sprint Journal V2 : retropropager le prix reel dans l'historique
            self._update_order_price(close_order.get("id", ""), exit_price, exit_fee)
        except Exception as e:
            logger.error("Executor: échec close market: {}", e)
            return

        # 3. Calculer P&L — fees réelles si disponibles (Hotfix 34)
        if exit_fee is not None:
            net_pnl = self._calculate_real_pnl(
                pos.direction, pos.entry_price, exit_price,
                pos.quantity, pos.entry_fee, exit_fee,
            )
        else:
            net_pnl = self._calculate_pnl(
                pos.direction, pos.entry_price, exit_price, pos.quantity,
            )

        logger.info(
            "Executor: CLOSE {} {} @ {:.2f} net={:+.2f} ({})",
            pos.direction, futures_sym, exit_price,
            net_pnl, event.exit_reason,
        )

        # 4. Enregistrer et notifier
        from backend.execution.risk_manager import LiveTradeResult

        self._risk_manager.record_trade_result(LiveTradeResult(
            net_pnl=net_pnl,
            timestamp=datetime.now(tz=timezone.utc),
            symbol=futures_sym,
            direction=pos.direction,
            exit_reason=event.exit_reason or "signal_exit",
            strategy_name=pos.strategy_name,
        ))
        self._risk_manager.unregister_position(futures_sym)

        await self._notifier.notify_live_order_closed(
            futures_sym, pos.direction,
            pos.entry_price, exit_price,
            net_pnl, event.exit_reason or "signal_exit",
            pos.strategy_name,
        )

        del self._positions[futures_sym]

    async def _cancel_pending_orders(self, symbol: str) -> None:
        """Annule les ordres SL/TP en attente pour un symbole."""
        pos = self._positions.get(symbol)
        if pos is None:
            return

        for order_id, label in [
            (pos.sl_order_id, "SL"),
            (pos.tp_order_id, "TP"),
        ]:
            if order_id:
                try:
                    await self._exchange.cancel_order(
                        order_id, pos.symbol,
                    )
                    logger.debug("Executor: {} annulé ({})", label, order_id)
                except Exception as e:
                    # L'ordre peut déjà être exécuté ou annulé
                    logger.debug("Executor: cancel {} ignoré: {}", label, e)

    # ─── Surveillance positions ────────────────────────────────────────

    async def _watch_orders_loop(self) -> None:
        """Mécanisme principal : watchOrders via ccxt Pro."""
        while self._running:
            try:
                if not self._positions and not self._grid_states:
                    await asyncio.sleep(1)
                    continue

                orders = await self._exchange.watch_orders()
                for order in orders:
                    await self._process_watched_order(order)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Executor: erreur watchOrders: {}", e)
                await asyncio.sleep(2)

    async def _process_watched_order(self, order: dict) -> None:
        """Traite un ordre détecté par watchOrders."""
        order_id = order.get("id", "")
        status = order.get("status", "")

        if status not in ("closed", "filled"):
            return

        # Hotfix 34 : extraire fee du WS push
        fee_info = order.get("fee") or {}
        exit_fee: float | None = (
            float(fee_info.get("cost") or 0)
            if fee_info.get("cost") is not None
            else None
        )

        # Scanner toutes les positions pour matcher l'order_id
        for symbol, pos in list(self._positions.items()):
            exit_reason = ""
            if order_id == pos.sl_order_id:
                exit_reason = "sl"
            elif order_id == pos.tp_order_id:
                exit_reason = "tp"
            else:
                continue

            exit_price = float(order.get("average") or order.get("price") or 0)

            # Sprint Journal V2 : enregistrer le SL/TP fill dans l'historique
            close_side = "sell" if pos.direction == "LONG" else "buy"
            self._record_order(
                exit_reason, symbol, close_side, pos.quantity,
                order, pos.strategy_name, f"watched_{exit_reason}",
            )

            # Si fee absente du WS push (fréquent pour trigger orders Bitget), fetch le fill
            if exit_fee is None or exit_fee == 0:
                _, fetched_fee = await self._fetch_fill_price(
                    order_id, symbol, exit_price,
                )
                if fetched_fee is not None:
                    exit_fee = fetched_fee

            await self._handle_exchange_close(symbol, exit_price, exit_reason, exit_fee)
            return

        # Scanner les grid states pour SL match
        for futures_sym, grid_state in list(self._grid_states.items()):
            if order_id == grid_state.sl_order_id:
                exit_price = float(
                    order.get("average") or order.get("price") or grid_state.sl_price,
                )

                # Sprint Journal V2 : enregistrer le SL grid fill dans l'historique
                close_side = "sell" if grid_state.direction == "LONG" else "buy"
                self._record_order(
                    "sl", futures_sym, close_side, grid_state.total_quantity,
                    order, grid_state.strategy_name, "watched_grid_sl",
                )

                # Si fee absente du WS push, fetch le fill
                if exit_fee is None or exit_fee == 0:
                    _, fetched_fee = await self._fetch_fill_price(
                        order_id, futures_sym, exit_price,
                    )
                    if fetched_fee is not None:
                        exit_fee = fetched_fee

                await self._handle_grid_sl_executed(
                    futures_sym, grid_state, exit_price, exit_fee,
                )
                return

        # Ordre non matché — log debug (ordres d'autres systèmes sur le sous-compte)
        logger.debug(
            "Executor: ordre non matché ignoré: id={}, symbol={}, status={}",
            order_id, order.get("symbol", "?"), status,
        )

    async def _poll_positions_loop(self) -> None:
        """Fallback : vérifie périodiquement l'état des positions."""
        while self._running:
            try:
                await asyncio.sleep(_POLL_INTERVAL)
                if not self._running:
                    continue

                for symbol in list(self._positions.keys()):
                    await self._check_position_still_open(symbol)
                for symbol in list(self._grid_states.keys()):
                    await self._check_grid_still_open(symbol)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Executor: erreur polling: {}", e)

    async def _check_position_still_open(self, symbol: str) -> None:
        """Vérifie si une position est toujours ouverte sur l'exchange.

        TODO Hotfix 34 : extraire fees depuis fetch_my_trades dans le polling.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return

        positions = await self._fetch_positions_safe(symbol)
        has_open = any(
            float(p.get("contracts", 0)) > 0 for p in positions
        )

        if not has_open:
            # Position fermée côté exchange (TP/SL exécuté)
            logger.info(
                "Executor: position {} fermée côté exchange (détectée par polling)",
                symbol,
            )
            # Tenter de récupérer le prix de sortie réel
            exit_price = await self._fetch_exit_price(symbol)
            exit_reason = await self._determine_exit_reason(symbol)
            await self._handle_exchange_close(symbol, exit_price, exit_reason)

    async def _check_grid_still_open(self, symbol: str) -> None:
        """Vérifie si la position grid est toujours ouverte sur l'exchange.

        TODO Hotfix 34 : extraire fees depuis fetch_my_trades dans le polling.
        """
        state = self._grid_states.get(symbol)
        if state is None:
            return

        positions = await self._fetch_positions_safe(symbol)
        has_open = any(float(p.get("contracts", 0)) > 0 for p in positions)

        if not has_open:
            logger.info(
                "Executor: grid {} fermée côté exchange (détectée par polling)", symbol,
            )
            exit_price = await self._fetch_exit_price(symbol)
            await self._handle_grid_sl_executed(symbol, state, exit_price)

    async def _handle_exchange_close(
        self, symbol: str, exit_price: float, exit_reason: str,
        exit_fee: float | None = None,
    ) -> None:
        """Traite la fermeture d'une position par l'exchange (TP/SL hit)."""
        pos = self._positions.get(symbol)
        if pos is None:
            return

        # Annuler l'autre ordre (si SL hit, annuler TP et vice-versa)
        await self._cancel_pending_orders(symbol)

        # Hotfix 34 : fees réelles si disponibles
        if exit_fee is not None:
            net_pnl = self._calculate_real_pnl(
                pos.direction, pos.entry_price, exit_price,
                pos.quantity, pos.entry_fee, exit_fee,
            )
        else:
            net_pnl = self._calculate_pnl(
                pos.direction, pos.entry_price, exit_price, pos.quantity,
            )

        logger.info(
            "Executor: EXCHANGE CLOSE {} {} @ {:.2f} net={:+.2f} ({})",
            pos.direction, symbol, exit_price, net_pnl, exit_reason,
        )

        from backend.execution.risk_manager import LiveTradeResult

        self._risk_manager.record_trade_result(LiveTradeResult(
            net_pnl=net_pnl,
            timestamp=datetime.now(tz=timezone.utc),
            symbol=symbol,
            direction=pos.direction,
            exit_reason=exit_reason,
            strategy_name=pos.strategy_name,
        ))
        self._risk_manager.unregister_position(symbol)

        await self._notifier.notify_live_order_closed(
            symbol, pos.direction,
            pos.entry_price, exit_price,
            net_pnl, exit_reason,
            pos.strategy_name,
        )

        del self._positions[symbol]

    # ─── Réconciliation au boot ────────────────────────────────────────

    async def _reconcile_on_boot(self) -> None:
        """Synchronise l'état local avec les positions Bitget réelles."""
        configured_symbols = []
        for a in self._config.assets:
            try:
                configured_symbols.append(to_futures_symbol(a.symbol))
            except ValueError:
                pass  # symbole non supporté en futures, ignoré

        for futures_sym in configured_symbols:
            await self._reconcile_symbol(futures_sym)

        # Réconcilier les cycles grid restaurés
        for futures_sym in list(self._grid_states.keys()):
            await self._reconcile_grid_symbol(futures_sym)

        # Nettoyage ordres trigger orphelins (TP/SL restés après fermeture)
        await self._cancel_orphan_orders()

    async def _reconcile_symbol(self, futures_sym: str) -> None:
        """Réconcilie une paire spécifique."""
        positions = await self._fetch_positions_safe(futures_sym)

        exchange_has_position = any(
            float(p.get("contracts", 0)) > 0 for p in positions
        )
        local_has_position = futures_sym in self._positions

        # Cas 1 : les deux côtés ont une position → reprendre le suivi
        if exchange_has_position and local_has_position:
            await self._notifier.notify_reconciliation(
                f"Position {futures_sym} trouvée sur exchange et en local — reprise."
            )
            logger.info(
                "Executor: réconciliation {} OK — position reprise", futures_sym,
            )

        # Cas 2 : exchange a une position, pas le local → orpheline
        elif exchange_has_position and not local_has_position:
            pos_data = next(
                p for p in positions if float(p.get("contracts", 0)) > 0
            )
            await self._notifier.notify_reconciliation(
                f"Position orpheline {futures_sym} détectée sur exchange "
                f"(contracts={pos_data.get('contracts')}). Non touchée."
            )
            logger.warning(
                "Executor: position orpheline sur exchange {} — non touchée",
                futures_sym,
            )

        # Cas 3 : local a une position, pas l'exchange → fermée pendant downtime
        elif not exchange_has_position and local_has_position:
            pos = self._positions[futures_sym]
            exit_price = await self._fetch_exit_price(futures_sym)
            net_pnl = self._calculate_pnl(
                pos.direction, pos.entry_price, exit_price, pos.quantity,
            )

            from backend.execution.risk_manager import LiveTradeResult

            self._risk_manager.record_trade_result(LiveTradeResult(
                net_pnl=net_pnl,
                timestamp=datetime.now(tz=timezone.utc),
                symbol=pos.symbol,
                direction=pos.direction,
                exit_reason="closed_during_downtime",
                strategy_name=pos.strategy_name,
            ))
            self._risk_manager.unregister_position(pos.symbol)

            await self._notifier.notify_reconciliation(
                f"Position {futures_sym} fermée pendant downtime. "
                f"P&L estimé: {net_pnl:+.2f}$"
            )
            logger.info(
                "Executor: position {} fermée pendant downtime, P&L={:+.2f}",
                futures_sym, net_pnl,
            )
            del self._positions[futures_sym]

        # Cas 4 : aucune position → clean
        else:
            logger.debug("Executor: réconciliation {} — aucune position", futures_sym)

    async def _reconcile_grid_symbol(self, futures_sym: str) -> None:
        """Réconcilie un cycle grid restauré avec l'exchange."""
        state = self._grid_states.get(futures_sym)
        if state is None:
            return

        positions = await self._fetch_positions_safe(futures_sym)
        has_position = any(
            float(p.get("contracts", 0)) > 0 for p in positions
        )

        if has_position:
            # Vérifier si le SL est toujours actif
            if state.sl_order_id:
                try:
                    sl_order = await self._exchange.fetch_order(
                        state.sl_order_id, futures_sym,
                    )
                    if sl_order.get("status") in ("closed", "filled"):
                        exit_price = float(
                            sl_order.get("average") or state.sl_price,
                        )
                        logger.info(
                            "Executor: SL grid exécuté pendant downtime {}",
                            futures_sym,
                        )
                        await self._handle_grid_sl_executed(
                            futures_sym, state, exit_price,
                        )
                        return
                except Exception:
                    pass

            # Rétablir le leverage
            try:
                await self._exchange.set_leverage(
                    state.leverage, futures_sym,
                )
            except Exception:
                pass

            logger.info(
                "Executor: cycle grid restauré {} ({} niveaux, SL={})",
                futures_sym, len(state.positions), state.sl_order_id,
            )
        else:
            # Position fermée pendant downtime (SL exécuté ou liquidation)
            exit_price = await self._fetch_exit_price(futures_sym)
            net_pnl = self._calculate_pnl(
                state.direction, state.avg_entry_price,
                exit_price, state.total_quantity,
            )
            from backend.execution.risk_manager import LiveTradeResult

            self._risk_manager.record_trade_result(LiveTradeResult(
                net_pnl=net_pnl,
                timestamp=datetime.now(tz=timezone.utc),
                symbol=futures_sym,
                direction=state.direction,
                exit_reason="closed_during_downtime",
                strategy_name=state.strategy_name,
            ))
            await self._notifier.notify_reconciliation(
                f"Cycle grid {futures_sym} fermé pendant downtime. "
                f"P&L estimé: {net_pnl:+.2f}$"
            )
            logger.info(
                "Executor: grid {} fermée pendant downtime, net={:+.2f}",
                futures_sym, net_pnl,
            )
            self._record_grid_close(futures_sym)
            del self._grid_states[futures_sym]

    async def _cancel_orphan_orders(self) -> None:
        """Annule les ordres trigger orphelins qui n'ont plus de position associée.

        Après un crash ou un arrêt, un SL/TP trigger order peut rester pendant
        sur l'exchange alors que la position a été fermée (par l'autre trigger).
        Ces ordres orphelins sont dangereux : ils pourraient ouvrir une nouvelle
        position involontaire si le prix revient sur le niveau.
        """
        try:
            open_orders = await self._exchange.fetch_open_orders(
                params={"type": "swap"},
            )
        except Exception as e:
            logger.warning("Executor: impossible de fetch open orders: {}", e)
            return

        if not open_orders:
            return

        # IDs des ordres trackés localement (positions mono + grid)
        tracked_ids: set[str] = set()
        for pos in self._positions.values():
            if pos.sl_order_id:
                tracked_ids.add(pos.sl_order_id)
            if pos.tp_order_id:
                tracked_ids.add(pos.tp_order_id)
            if pos.entry_order_id:
                tracked_ids.add(pos.entry_order_id)
        for grid_state in self._grid_states.values():
            if grid_state.sl_order_id:
                tracked_ids.add(grid_state.sl_order_id)
            for gp in grid_state.positions:
                if gp.entry_order_id:
                    tracked_ids.add(gp.entry_order_id)

        # Filtrer les ordres orphelins (trigger orders non trackés)
        orphans = [
            o for o in open_orders
            if o.get("id") and o["id"] not in tracked_ids
        ]

        if not orphans:
            return

        cancelled: list[str] = []
        for order in orphans:
            order_id = order["id"]
            symbol = order.get("symbol", "unknown")
            try:
                await self._exchange.cancel_order(
                    order_id, symbol,
                )
                cancelled.append(f"{order_id} ({symbol})")
                logger.info("Executor: ordre orphelin annulé: {} ({})", order_id, symbol)
            except Exception as e:
                logger.warning(
                    "Executor: échec annulation ordre orphelin {}: {}", order_id, e,
                )

        if cancelled:
            await self._notifier.notify_reconciliation(
                f"Ordres trigger orphelins annulés ({len(cancelled)}): "
                + ", ".join(cancelled)
            )

    async def _fetch_exit_price(self, symbol: str) -> float:
        """Récupère le prix de sortie réel depuis l'historique Bitget."""
        try:
            trades = await self._exchange.fetch_my_trades(
                symbol, limit=5,
            )
            if trades:
                # Dernier trade = le plus récent
                return float(trades[-1].get("price", 0))
        except Exception as e:
            logger.warning("Executor: impossible de fetch exit price: {}", e)

        return 0.0

    async def _fetch_fill_price(
        self,
        order_id: str,
        symbol: str,
        fallback_price: float,
    ) -> tuple[float, float | None]:
        """Récupère le prix de fill réel et la fee depuis Bitget (Hotfix 34).

        Stratégie :
        1. fetch_order(order_id) → order.average + order.fee.cost
        2. Fallback : fetch_my_trades(symbol) filtré par order_id
        3. Dernier recours : (fallback_price, None)

        Returns:
            (avg_price, total_fee) — fee=float si données réelles, None si fallback.
        """
        # 1. fetch_order (souvent suffisant après quelques ms)
        if order_id:
            try:
                order = await self._exchange.fetch_order(order_id, symbol)
                avg = order.get("average")
                if avg and float(avg) > 0:
                    fee_info = order.get("fee") or {}
                    fee_cost = float(fee_info.get("cost") or 0) if fee_info.get("cost") is not None else None
                    return float(avg), fee_cost
            except Exception as e:
                logger.debug("Executor: fetch_order {} échoué: {}", order_id, e)

        # 2. fetch_my_trades (fallback, laisser le fill se propager)
        await asyncio.sleep(0.3)
        if order_id:
            try:
                trades = await self._exchange.fetch_my_trades(symbol, limit=10)
                matched = [t for t in trades if t.get("order") == order_id]
                if matched:
                    total_qty = sum(float(t.get("amount", 0)) for t in matched)
                    total_cost = sum(
                        float(t.get("price", 0)) * float(t.get("amount", 0))
                        for t in matched
                    )
                    total_fee = sum(
                        float((t.get("fee") or {}).get("cost") or 0)
                        for t in matched
                    )
                    if total_qty > 0:
                        return total_cost / total_qty, total_fee
            except Exception as e:
                logger.debug("Executor: fetch_my_trades {} échoué: {}", symbol, e)

        # 3. Fallback paper price — fee=None signale l'absence de données réelles
        logger.warning(
            "Executor: FALLBACK prix paper pour {} order={} (fill Bitget indisponible)",
            symbol, order_id,
        )
        return fallback_price, None

    async def _determine_exit_reason(self, symbol: str) -> str:
        """Détermine la raison de fermeture (SL ou TP) depuis les ordres."""
        pos = self._positions.get(symbol)
        if pos is None:
            return "unknown"

        for order_id, reason in [
            (pos.sl_order_id, "sl"),
            (pos.tp_order_id, "tp"),
        ]:
            if order_id:
                try:
                    order = await self._exchange.fetch_order(
                        order_id, symbol,
                    )
                    if order.get("status") in ("closed", "filled"):
                        return reason
                except Exception:
                    pass

        return "unknown"

    # ─── Helpers ───────────────────────────────────────────────────────

    async def _fetch_positions_safe(self, symbol: str | None = None) -> list[dict]:
        """Fetch positions depuis Bitget (mainnet only)."""
        if symbol:
            return await self._exchange.fetch_positions([symbol])
        return await self._exchange.fetch_positions()

    def _round_quantity(self, quantity: float, futures_symbol: str) -> float:
        """Arrondit la quantité via ccxt (gère tick_size et decimal_places)."""
        try:
            rounded = float(
                self._exchange.amount_to_precision(futures_symbol, quantity),
            )
        except Exception:
            # Fallback sur config
            for asset in self._config.assets:
                if to_futures_symbol(asset.symbol) == futures_symbol:
                    step = asset.min_order_size
                    return max(step, round(quantity / step) * step)
            return quantity

        # Respecter le min_amount
        market = self._markets.get(futures_symbol)
        if market:
            min_amount = (
                market.get("limits", {}).get("amount", {}).get("min") or 0
            )
            return max(min_amount, rounded)
        return rounded

    def _round_price(self, price: float, futures_symbol: str) -> float:
        """Arrondit un prix via ccxt (gère tick_size et decimal_places)."""
        try:
            return float(
                self._exchange.price_to_precision(futures_symbol, price),
            )
        except Exception:
            return price

    def _calculate_pnl(
        self,
        direction: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
    ) -> float:
        """Calcule le P&L net (approximatif, fees réelles via l'exchange)."""
        if direction == "LONG":
            gross = (exit_price - entry_price) * quantity
        else:
            gross = (entry_price - exit_price) * quantity

        # Fees approximatives (taker entry + taker/maker exit)
        taker_fee = self._config.risk.fees.taker_percent / 100
        entry_fee = quantity * entry_price * taker_fee
        exit_fee = quantity * exit_price * taker_fee
        return gross - entry_fee - exit_fee

    def _calculate_real_pnl(
        self,
        direction: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        entry_fees: float,
        exit_fees: float,
    ) -> float:
        """P&L net avec fees réelles Bitget (absolues en USDT, Hotfix 34).

        Utilise les fees retournées par fetch_order/fetch_my_trades.
        Ne remplace PAS _calculate_pnl() qui reste le fallback pour la réconciliation.
        """
        if direction == "LONG":
            gross = (exit_price - entry_price) * quantity
        else:
            gross = (entry_price - exit_price) * quantity
        return gross - entry_fees - exit_fees

    # ─── Status ────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Retourne le statut de l'executor pour /health et le dashboard."""
        # Backward compat: "position" (première) + "positions" (liste complète)
        pos_info = None
        positions_list: list[dict[str, Any]] = []

        default_leverage = self._config.risk.position.default_leverage

        for pos in self._positions.values():
            notional = pos.entry_price * pos.quantity
            info = {
                "symbol": pos.symbol,
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "quantity": pos.quantity,
                "sl_price": pos.sl_price,
                "tp_price": pos.tp_price,
                "strategy_name": pos.strategy_name,
                "leverage": default_leverage,
                "notional": notional,
                "entry_time": pos.entry_time.isoformat(),
            }
            positions_list.append(info)
            if pos_info is None:
                pos_info = info

        # Grids enrichies (Sprint 39)
        executor_grids: dict[str, dict[str, Any]] = {}
        for futures_sym, gs in self._grid_states.items():
            info = self._enrich_grid_position(futures_sym, gs)
            positions_list.append(info)
            if pos_info is None:
                pos_info = info
            # Clé format paper : strategy:spot_symbol
            spot_sym = futures_sym.split(":")[0] if ":" in futures_sym else futures_sym
            executor_grids[f"{gs.strategy_name}:{spot_sym}"] = info

        result: dict[str, Any] = {
            "enabled": self.is_enabled,
            "connected": self.is_connected,
            "exchange_balance": self._exchange_balance,
            "position": pos_info,
            "positions": positions_list,
            "risk_manager": self._risk_manager.get_status(),
        }

        # Grid state au format paper-compatible (Sprint 39)
        if executor_grids:
            result["executor_grid_state"] = {
                "grid_positions": executor_grids,
                "summary": {
                    "total_positions": sum(
                        g.get("levels", 0) for g in executor_grids.values()
                    ),
                    "total_assets": len(executor_grids),
                    "total_margin_used": round(
                        sum(g.get("margin_used", 0) for g in executor_grids.values()), 2,
                    ),
                    "total_unrealized_pnl": round(
                        sum(g.get("unrealized_pnl", 0) for g in executor_grids.values()), 2,
                    ),
                },
            }

        if self._selector:
            result["selector"] = self._selector.get_status()

        return result

    def get_state_for_persistence(self) -> dict[str, Any]:
        """Retourne l'état à persister par le StateManager."""
        positions_data: dict[str, dict[str, Any]] = {}
        for symbol, pos in self._positions.items():
            positions_data[symbol] = {
                "symbol": pos.symbol,
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "quantity": pos.quantity,
                "entry_order_id": pos.entry_order_id,
                "sl_order_id": pos.sl_order_id,
                "tp_order_id": pos.tp_order_id,
                "entry_time": pos.entry_time.isoformat(),
                "strategy_name": pos.strategy_name,
                "sl_price": pos.sl_price,
                "tp_price": pos.tp_price,
                "entry_fee": pos.entry_fee,
            }

        grid_states_data: dict[str, dict[str, Any]] = {}
        for sym, gs in self._grid_states.items():
            grid_states_data[sym] = {
                "symbol": gs.symbol,
                "direction": gs.direction,
                "strategy_name": gs.strategy_name,
                "leverage": gs.leverage,
                "sl_order_id": gs.sl_order_id,
                "sl_price": gs.sl_price,
                "opened_at": gs.opened_at.isoformat(),
                "positions": [
                    {
                        "level": p.level,
                        "entry_price": p.entry_price,
                        "quantity": p.quantity,
                        "entry_order_id": p.entry_order_id,
                        "entry_time": p.entry_time.isoformat(),
                        "entry_fee": p.entry_fee,
                    }
                    for p in gs.positions
                ],
            }

        # Phase 2 : cooldown anti-churning
        last_close_times_data = {
            sym: ts.isoformat() for sym, ts in self._last_close_time.items()
        }

        return {
            "positions": positions_data,
            "grid_states": grid_states_data,
            "risk_manager": self._risk_manager.get_state(),
            "order_history": list(self._order_history),
            "last_close_times": last_close_times_data,
        }

    def restore_positions(self, state: dict[str, Any]) -> None:
        """Restaure les positions depuis l'état sauvegardé.

        Backward compat : accepte l'ancien format {"position": {...}}
        et le nouveau format {"positions": {"BTC/USDT:USDT": {...}, ...}}.
        """
        # Nouveau format : {"positions": {"BTC/USDT:USDT": {...}, ...}}
        positions_data = state.get("positions")

        if positions_data and isinstance(positions_data, dict):
            for symbol, pos_data in positions_data.items():
                self._positions[symbol] = self._restore_single_position(pos_data)
                logger.info(
                    "Executor: position restaurée — {} {} @ {:.2f}",
                    self._positions[symbol].direction,
                    symbol,
                    self._positions[symbol].entry_price,
                )
            return

        # Ancien format : {"position": {...}} (single position)
        pos_data = state.get("position")
        if pos_data is not None:
            symbol = pos_data["symbol"]
            self._positions[symbol] = self._restore_single_position(pos_data)
            logger.info(
                "Executor: position restaurée (ancien format) — {} {} @ {:.2f}",
                self._positions[symbol].direction,
                symbol,
                self._positions[symbol].entry_price,
            )

        # Grid states (Sprint 12)
        for sym, gs_data in state.get("grid_states", {}).items():
            self._grid_states[sym] = GridLiveState(
                symbol=gs_data["symbol"],
                direction=gs_data["direction"],
                strategy_name=gs_data["strategy_name"],
                leverage=gs_data.get("leverage", 6),
                sl_order_id=gs_data.get("sl_order_id"),
                sl_price=gs_data.get("sl_price", 0.0),
                opened_at=datetime.fromisoformat(gs_data["opened_at"]),
                positions=[
                    GridLivePosition(
                        level=p["level"],
                        entry_price=p["entry_price"],
                        quantity=p["quantity"],
                        entry_order_id=p["entry_order_id"],
                        entry_time=datetime.fromisoformat(p["entry_time"]),
                        entry_fee=p.get("entry_fee", 0.0),
                    )
                    for p in gs_data.get("positions", [])
                ],
            )
            logger.info(
                "Executor: grid restaurée — {} {} niveaux",
                sym, len(self._grid_states[sym].positions),
            )

        # Order history (Sprint 32)
        order_history = state.get("order_history", [])
        self._order_history = deque(order_history, maxlen=200)
        if order_history:
            logger.info("Executor: {} ordres restaurés dans l'historique", len(order_history))

        # Phase 2 : cooldown anti-churning
        for sym, ts_str in state.get("last_close_times", {}).items():
            try:
                self._last_close_time[sym] = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                pass

    # ─── Grid helpers ───────────────────────────────────────────────────

    @staticmethod
    def _is_grid_strategy(strategy_name: str) -> bool:
        """Vérifie si la stratégie est de type grid/DCA."""
        from backend.optimization import is_grid_strategy

        return is_grid_strategy(strategy_name)

    def _get_grid_sl_percent(self, strategy_name: str) -> float:
        """Récupère le sl_percent depuis la config stratégie."""
        strat_config = getattr(self._config.strategies, strategy_name, None)
        if strat_config and hasattr(strat_config, "sl_percent"):
            return strat_config.sl_percent
        return 20.0

    def _get_leverage_for_symbol(self, symbol: str) -> int:
        """Détermine le leverage de la stratégie assignée au symbol (boot)."""
        strategies = self._config.strategies
        for name in strategies.model_fields:
            strat_cfg = getattr(strategies, name, None)
            if strat_cfg is None:
                continue
            per_asset = getattr(strat_cfg, "per_asset", None)
            if not isinstance(per_asset, dict) or symbol not in per_asset:
                continue
            enabled = getattr(strat_cfg, "enabled", False)
            if not enabled:
                continue
            lev = getattr(strat_cfg, "leverage", None)
            if isinstance(lev, int) and lev > 0:
                return lev
        return self._config.risk.position.default_leverage

    def _get_grid_leverage(self, strategy_name: str) -> int:
        """Récupère le leverage depuis la config stratégie."""
        strat_config = getattr(self._config.strategies, strategy_name, None)
        if strat_config and hasattr(strat_config, "leverage"):
            return strat_config.leverage
        return 6

    # ─── Enrichissement métriques live (Sprint 39) ───────────────────────

    def _enrich_grid_position(
        self, futures_sym: str, gs: GridLiveState,
    ) -> dict[str, Any]:
        """Enrichit un GridLiveState avec métriques temps réel (prix, P&L, TP/SL).

        Suit le même pattern que _check_grid_exit() pour les indicateurs.
        Matche le format de Simulator.get_grid_state() pour compatibilité frontend.
        Graceful degradation : si DataEngine/Simulator absents, champs enrichis = None/0.
        """
        import math

        spot_sym = futures_sym.split(":")[0] if ":" in futures_sym else futures_sym
        strategy = self._strategies.get(gs.strategy_name)

        avg_entry = gs.avg_entry_price
        total_qty = gs.total_quantity
        notional = avg_entry * total_qty
        leverage = gs.leverage if gs.leverage > 0 else 6
        margin = notional / leverage if leverage > 0 else 0.0
        levels_max = self._get_grid_num_levels(gs.strategy_name)

        # Durée
        now = datetime.now(tz=timezone.utc)
        duration_hours = round((now - gs.opened_at).total_seconds() / 3600, 1)

        # Prix courant depuis DataEngine (fallback multi-TF)
        current_price: float | None = None
        if self._data_engine is not None:
            buffers = getattr(self._data_engine, "_buffers", {})
            symbol_buffers = buffers.get(spot_sym, {})
            for tf in ("1m", "5m", "15m", "1h"):
                tf_candles = symbol_buffers.get(tf, [])
                if tf_candles:
                    current_price = tf_candles[-1].close
                    break

        # P&L non réalisé via GridState existant
        grid_state = self._live_state_to_grid_state(futures_sym, current_price)
        unrealized_pnl = round(grid_state.unrealized_pnl, 2)
        unrealized_pnl_pct = round(
            unrealized_pnl / margin * 100 if margin > 0 else 0.0, 2,
        )

        # TP/SL depuis stratégie + contexte runner paper
        tp_price: float | None = None
        sl_price: float | None = None

        if strategy is not None and self._simulator is not None:
            ctx = self._simulator.get_runner_context(gs.strategy_name, spot_sym)
            if ctx is not None:
                strategy_tf = getattr(strategy._config, "timeframe", "1h")
                tf_indicators = ctx.indicators.get(strategy_tf, {})

                raw_tp = strategy.get_tp_price(grid_state, tf_indicators)
                raw_sl = strategy.get_sl_price(grid_state, tf_indicators)

                if not math.isnan(raw_tp):
                    tp_price = round(raw_tp, 6)
                if not math.isnan(raw_sl):
                    sl_price = round(raw_sl, 6)

        # Fallback SL depuis l'ordre exchange
        if sl_price is None and gs.sl_price > 0:
            sl_price = gs.sl_price

        # Distances
        tp_distance_pct: float | None = None
        sl_distance_pct: float | None = None
        if current_price and current_price > 0:
            if tp_price is not None:
                tp_distance_pct = round(
                    (tp_price - current_price) / current_price * 100, 2,
                )
            if sl_price is not None:
                sl_distance_pct = round(
                    (sl_price - current_price) / current_price * 100, 2,
                )

        # P&L par niveau
        direction = Direction(gs.direction)
        per_level: list[dict[str, Any]] = []
        for p in gs.positions:
            level_pnl: float | None = None
            level_pnl_pct: float | None = None
            if current_price is not None and current_price > 0:
                if direction == Direction.LONG:
                    level_pnl = (current_price - p.entry_price) * p.quantity
                else:
                    level_pnl = (p.entry_price - current_price) * p.quantity
                level_margin = (p.entry_price * p.quantity) / leverage if leverage > 0 else 0.0
                level_pnl_pct = (
                    round(level_pnl / level_margin * 100, 2) if level_margin > 0 else 0.0
                )
                level_pnl = round(level_pnl, 2)

            per_level.append({
                "level": p.level,
                "entry_price": p.entry_price,
                "quantity": p.quantity,
                "entry_time": p.entry_time.isoformat(),
                "direction": gs.direction,
                "pnl_usd": level_pnl,
                "pnl_pct": level_pnl_pct,
            })

        return {
            "symbol": gs.symbol,
            "direction": gs.direction,
            "entry_price": avg_entry,
            "quantity": total_qty,
            "sl_price": sl_price if sl_price is not None else gs.sl_price,
            "tp_price": tp_price if tp_price is not None else 0.0,
            "strategy_name": gs.strategy_name,
            "type": "grid",
            "levels": len(gs.positions),
            "levels_max": levels_max,
            "leverage": leverage,
            "notional": round(notional, 2),
            "entry_time": gs.opened_at.isoformat(),
            # Champs enrichis (Sprint 39)
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "tp_distance_pct": tp_distance_pct,
            "sl_distance_pct": sl_distance_pct,
            "margin_used": round(margin, 2),
            "duration_hours": duration_hours,
            "positions": per_level,
        }

    def _get_grid_num_levels(self, strategy_name: str) -> int:
        """Récupère le num_levels depuis la config stratégie."""
        strat_config = getattr(self._config.strategies, strategy_name, None)
        if strat_config and hasattr(strat_config, "num_levels"):
            return strat_config.num_levels
        return 4

    @staticmethod
    def _restore_single_position(pos_data: dict[str, Any]) -> LivePosition:
        """Crée une LivePosition depuis un dict sérialisé."""
        return LivePosition(
            symbol=pos_data["symbol"],
            direction=pos_data["direction"],
            entry_price=pos_data["entry_price"],
            quantity=pos_data["quantity"],
            entry_order_id=pos_data["entry_order_id"],
            sl_order_id=pos_data.get("sl_order_id"),
            tp_order_id=pos_data.get("tp_order_id"),
            entry_time=datetime.fromisoformat(pos_data["entry_time"]),
            strategy_name=pos_data.get("strategy_name", ""),
            sl_price=pos_data.get("sl_price", 0),
            tp_price=pos_data.get("tp_price", 0),
            entry_fee=pos_data.get("entry_fee", 0.0),
        )
