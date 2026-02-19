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

        # Exit monitor autonome (Sprint Executor Autonome)
        self._data_engine: DataEngine | None = None
        self._db: Any = None  # Database — warm-up SMA depuis historique DB
        self._strategies: dict[str, BaseGridStrategy] = {}
        self._exit_check_task: asyncio.Task[None] | None = None

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
                    await self._setup_leverage_and_margin(futures_sym)
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
            await self.refresh_balance()

    # ─── Setup ─────────────────────────────────────────────────────────

    async def _setup_leverage_and_margin(self, futures_symbol: str) -> None:
        """Set leverage et margin mode, seulement s'il n'y a pas de position ouverte."""
        positions = await self._fetch_positions_safe(futures_symbol)
        has_open = any(
            float(p.get("contracts", 0)) > 0 for p in positions
        )

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

    def set_db(self, db: Any) -> None:
        """Enregistre la DB pour le warm-up SMA quand le buffer WS est trop court."""
        self._db = db

    def set_strategies(self, strategies: dict[str, BaseGridStrategy]) -> None:
        """Enregistre les instances de stratégie pour should_close_all()."""
        self._strategies = strategies
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

    async def _check_grid_exit(self, futures_sym: str) -> None:
        """Vérifie si un cycle grid doit être fermé.

        Construit un StrategyContext depuis le DataEngine et appelle
        strategy.should_close_all() — même logique que le paper.
        """
        state = self._grid_states.get(futures_sym)
        if state is None or not state.positions:
            return

        strategy = self._strategies.get(state.strategy_name)
        if strategy is None:
            return

        if self._data_engine is None:
            return

        # Convertir futures_sym ("BTC/USDT:USDT") → spot_sym ("BTC/USDT")
        spot_sym = futures_sym.split(":")[0] if ":" in futures_sym else futures_sym

        # Récupérer les candles depuis le DataEngine
        buffers = self._data_engine._buffers.get(spot_sym, {})
        strategy_tf = getattr(strategy._config, "timeframe", "1h")
        candles_tf = list(buffers.get(strategy_tf, []))

        if not candles_tf:
            return

        current_close = candles_tf[-1].close

        # ── Résoudre ma_period per_asset avant tout (Hotfix ma_period per_asset) ──
        ma_period = 7  # fallback ultime
        strat_cfg = getattr(self._config.strategies, state.strategy_name, None)
        if strat_cfg is not None:
            try:
                asset_params = strat_cfg.get_params_for_symbol(spot_sym)
                raw = asset_params.get("ma_period", 7)
                if isinstance(raw, (int, float)):
                    ma_period = int(raw)
            except (AttributeError, TypeError):
                raw = getattr(strategy._config, "ma_period", 7)
                if isinstance(raw, (int, float)):
                    ma_period = int(raw)

        # ── Warm-up depuis la DB si buffer WS insuffisant (Hotfix buffer incomplet) ──
        if len(candles_tf) < ma_period and self._db is not None:
            try:
                db_candles = await self._db.get_candles(
                    symbol=spot_sym,
                    timeframe=strategy_tf,
                    limit=ma_period + 10,
                )
                if len(db_candles) >= ma_period:
                    live_timestamps = {c.timestamp for c in candles_tf}
                    live_only = [c for c in candles_tf if c.timestamp not in live_timestamps]
                    candles_tf = db_candles + live_only
                    logger.debug(
                        "Exit monitor: warm-up DB pour {} — {} candles DB + {} live = {} total",
                        futures_sym, len(db_candles), len(live_only), len(candles_tf),
                    )
            except Exception as e:
                logger.warning("Exit monitor: warm-up DB échoué pour {}: {}", futures_sym, e)

        # Si toujours pas assez de candles → skip (mieux vaut ne pas décider que faux TP)
        if len(candles_tf) < ma_period:
            logger.debug(
                "Exit monitor: buffer insuffisant pour {} ({}/{} candles), skip",
                futures_sym, len(candles_tf), ma_period,
            )
            return

        # Calculer les indicateurs via la stratégie elle-même
        indicators: dict = {}
        try:
            indicators = strategy.compute_live_indicators(list(candles_tf))
        except Exception as e:
            logger.debug("Executor: compute_live_indicators échoué pour {}: {}", futures_sym, e)

        # S'assurer que le timeframe principal a les données de base
        tf_indicators = indicators.get(strategy_tf, {})
        tf_indicators["close"] = current_close

        # SMA fallback si pas fourni par compute_live_indicators
        if "sma" not in tf_indicators:
            closes = [c.close for c in candles_tf[-ma_period:]]
            tf_indicators["sma"] = sum(closes) / len(closes)

        indicators[strategy_tf] = tf_indicators

        # Construire le GridState
        from backend.strategies.base_grid import GridPosition, GridState

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
        # Unrealized P&L approximatif
        direction = Direction(state.direction)
        if direction == Direction.LONG:
            unrealized = (current_close - avg_entry) * total_qty
        else:
            unrealized = (avg_entry - current_close) * total_qty

        grid_state = GridState(
            positions=grid_positions,
            avg_entry_price=avg_entry,
            total_quantity=total_qty,
            total_notional=total_notional,
            unrealized_pnl=unrealized,
        )

        # Construire le StrategyContext
        from backend.strategies.base import StrategyContext

        ctx = StrategyContext(
            symbol=spot_sym,
            timestamp=candles_tf[-1].timestamp,
            candles={},
            indicators=indicators,
            current_position=None,
            capital=0.0,
            config=None,
        )

        # APPELER LA STRATÉGIE — même logique que le paper
        exit_reason = strategy.should_close_all(ctx, grid_state)

        if exit_reason is None:
            return

        logger.info(
            "Executor: EXIT AUTONOME {} {} — raison={}, close={:.6f}, avg_entry={:.6f}",
            state.direction, futures_sym, exit_reason, current_close, avg_entry,
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
            exit_price=current_close,
        )
        await self._close_grid_cycle(synthetic_event)

    # ─── Event Handler (callback du Simulator) ─────────────────────────

    async def handle_event(self, event: TradeEvent) -> None:
        """Appelé par le Simulator via callback quand le runner trade."""
        if not self._running or not self._connected:
            return

        is_grid = self._is_grid_strategy(event.strategy_name)

        if event.event_type == TradeEventType.OPEN:
            # Gate OPEN via AdaptiveSelector (si configuré)
            if self._selector and not self._selector.is_allowed(
                event.strategy_name, event.symbol,
            ):
                return
            if is_grid:
                await self._open_grid_position(event)
            else:
                await self._open_position(event)
        elif event.event_type == TradeEventType.CLOSE:
            # CLOSE passe toujours (on doit pouvoir fermer)
            if is_grid:
                await self._close_grid_cycle(event)
            else:
                await self._close_position(event)

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

            # Setup leverage au 1er trade grid (pas au start)
            try:
                await self._exchange.set_leverage(
                    grid_leverage, futures_sym,
                )
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
            except Exception as e:
                logger.warning("Executor: échec cancel ancien SL grid: {}", e)

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
        del self._grid_states[futures_sym]
        await self._notifier.notify_live_sl_failed(futures_sym, state.strategy_name)

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

        # 1. Annuler SL (sauf si c'est le SL qui a déclenché)
        if event.exit_reason != "sl_global" and state.sl_order_id:
            try:
                await self._exchange.cancel_order(
                    state.sl_order_id, futures_sym,
                )
            except Exception as e:
                logger.debug("Executor: annulation SL grid échouée (probablement déjà exécuté): {}", e)

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
        del self._grid_states[futures_sym]

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
        del self._grid_states[futures_sym]

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
            ))
            await self._notifier.notify_reconciliation(
                f"Cycle grid {futures_sym} fermé pendant downtime. "
                f"P&L estimé: {net_pnl:+.2f}$"
            )
            logger.info(
                "Executor: grid {} fermée pendant downtime, net={:+.2f}",
                futures_sym, net_pnl,
            )
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

        for gs in self._grid_states.values():
            notional = gs.avg_entry_price * gs.total_quantity
            info = {
                "symbol": gs.symbol,
                "direction": gs.direction,
                "entry_price": gs.avg_entry_price,
                "quantity": gs.total_quantity,
                "sl_price": gs.sl_price,
                "tp_price": 0.0,
                "strategy_name": gs.strategy_name,
                "type": "grid",
                "levels": len(gs.positions),
                "levels_max": self._get_grid_num_levels(gs.strategy_name),
                "leverage": gs.leverage,
                "notional": notional,
                "entry_time": gs.opened_at.isoformat(),
                "positions": [
                    {
                        "level": p.level,
                        "entry_price": p.entry_price,
                        "quantity": p.quantity,
                        "entry_time": p.entry_time.isoformat(),
                    }
                    for p in gs.positions
                ],
            }
            positions_list.append(info)
            if pos_info is None:
                pos_info = info

        result: dict[str, Any] = {
            "enabled": self.is_enabled,
            "connected": self.is_connected,
            "exchange_balance": self._exchange_balance,
            "position": pos_info,
            "positions": positions_list,
            "risk_manager": self._risk_manager.get_status(),
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

        return {
            "positions": positions_data,
            "grid_states": grid_states_data,
            "risk_manager": self._risk_manager.get_state(),
            "order_history": list(self._order_history),
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

    def _get_grid_leverage(self, strategy_name: str) -> int:
        """Récupère le leverage depuis la config stratégie."""
        strat_config = getattr(self._config.strategies, strategy_name, None)
        if strat_config and hasattr(strat_config, "leverage"):
            return strat_config.leverage
        return 6

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
