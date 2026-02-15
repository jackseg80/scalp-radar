"""Executor : exécution d'ordres réels sur Bitget via ccxt Pro.

Sprint 12 : support grid DCA multi-niveaux (envelope_dca).
Sprint 5b : multi-stratégie, multi-paire, adaptive selector.
Pattern observer : reçoit les TradeEvent du Simulator via callback,
réplique en ordres réels sur Bitget.

Règle de sécurité #1 : JAMAIS de position sans SL.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from backend.alerts.notifier import Notifier
    from backend.core.config import AppConfig
    from backend.execution.adaptive_selector import AdaptiveSelector
    from backend.execution.risk_manager import LiveRiskManager


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


@dataclass
class GridLivePosition:
    """Position individuelle dans un cycle grid live."""

    level: int
    entry_price: float
    quantity: float
    entry_order_id: str
    entry_time: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


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


# ─── MAPPING SYMBOLES ─────────────────────────────────────────────────────

# Mapping symbole spot → futures swap Bitget
SYMBOL_SPOT_TO_FUTURES: dict[str, str] = {
    "BTC/USDT": "BTC/USDT:USDT",
    "ETH/USDT": "ETH/USDT:USDT",
    "SOL/USDT": "SOL/USDT:USDT",
    "DOGE/USDT": "DOGE/USDT:USDT",
    "LINK/USDT": "LINK/USDT:USDT",
}


def to_futures_symbol(spot_symbol: str) -> str:
    """Convertit un symbole spot en symbole futures ccxt."""
    result = SYMBOL_SPOT_TO_FUTURES.get(spot_symbol)
    if result is None:
        raise ValueError(f"Symbole non supporté pour futures: {spot_symbol}")
    return result


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
        self._markets: dict[str, Any] = {}  # Cache load_markets()

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
    def _sandbox_params(self) -> dict[str, str]:
        """Params productType + marginCoin pour le demo trading Bitget."""
        if self._config.secrets.bitget_sandbox:
            return {"productType": "SUSDT-FUTURES", "marginCoin": "SUSDT"}
        return {}

    @property
    def _margin_coin(self) -> str:
        """Devise de marge : SUSDT en sandbox, USDT en mainnet."""
        return "SUSDT" if self._config.secrets.bitget_sandbox else "USDT"

    # ─── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialise l'exchange ccxt Pro, réconcilie, lance la surveillance."""
        import ccxt.pro as ccxtpro

        sandbox = self._config.secrets.bitget_sandbox
        self._exchange = ccxtpro.bitget({
            "apiKey": self._config.secrets.bitget_api_key,
            "secret": self._config.secrets.bitget_secret,
            "password": self._config.secrets.bitget_passphrase,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
            "sandbox": sandbox,
        })

        try:
            # 1. Charger les marchés (min_order_size, tick_size réels)
            self._markets = await self._exchange.load_markets()
            logger.info("Executor: {} marchés chargés", len(self._markets))

            # 2. Fetch balance pour le capital initial
            balance = await self._exchange.fetch_balance({
                "type": "swap", **self._sandbox_params,
            })
            coin = self._margin_coin
            free = float(balance.get("free", {}).get(coin, 0))
            total = float(balance.get("total", {}).get(coin, 0))
            self._risk_manager.set_initial_capital(total)
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

            mode = "SANDBOX" if sandbox else "MAINNET"
            logger.info(
                "Executor: démarré en mode {} ({} symboles actifs)",
                mode, len(active_symbols),
            )

        except Exception as e:
            logger.error("Executor: échec démarrage: {}", e)
            self._connected = False
            raise

    async def stop(self) -> None:
        """Arrête la surveillance. NE ferme PAS les positions (TP/SL restent)."""
        self._running = False

        for task in (self._watch_task, self._poll_task):
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
                False, futures_symbol, params=self._sandbox_params,
            )
            logger.info("Executor: position mode set à 'one-way' pour {}", futures_symbol)
        except Exception as e:
            # Bitget renvoie une erreur si le mode est déjà celui demandé
            logger.debug("Executor: set_position_mode: {}", e)

        try:
            await self._exchange.set_leverage(
                leverage, futures_symbol, params=self._sandbox_params,
            )
            logger.info("Executor: leverage set à {}x pour {}", leverage, futures_symbol)
        except Exception as e:
            logger.warning("Executor: impossible de set leverage: {}", e)

        try:
            await self._exchange.set_margin_mode(
                margin_mode, futures_symbol, params=self._sandbox_params,
            )
            logger.info("Executor: margin mode set à '{}' pour {}", margin_mode, futures_symbol)
        except Exception as e:
            # Bitget renvoie une erreur si le mode est déjà celui demandé
            logger.debug("Executor: set_margin_mode: {}", e)

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
            "type": "swap", **self._sandbox_params,
        })
        coin = self._margin_coin
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
                params=self._sandbox_params,
            )
        except Exception as e:
            logger.error("Executor: échec ordre d'entrée: {}", e)
            return

        entry_order_id = entry_order.get("id", "")
        filled_qty = float(entry_order.get("filled") or quantity)
        avg_price = float(entry_order.get("average") or event.entry_price)

        if filled_qty <= 0:
            logger.error("Executor: ordre d'entrée non rempli")
            return

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
                await self._exchange.create_order(
                    futures_sym, "market", close_side, filled_qty,
                    params={"reduceOnly": True, **self._sandbox_params},
                )
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
                        **self._sandbox_params,
                    },
                )
                sl_id = sl_order.get("id", "")
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
                    **self._sandbox_params,
                },
            )
            tp_id = tp_order.get("id", "")
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

        # Pre-trade check UNIQUEMENT au 1er niveau (Bug 2 fix)
        if is_first_level:
            grid_leverage = self._get_grid_leverage(event.strategy_name)

            # Setup leverage au 1er trade grid (pas au start)
            try:
                await self._exchange.set_leverage(
                    grid_leverage, futures_sym, params=self._sandbox_params,
                )
                logger.info(
                    "Executor: leverage grid set a {}x pour {}",
                    grid_leverage, futures_sym,
                )
            except Exception as e:
                logger.warning("Executor: set leverage grid: {}", e)

            balance = await self._exchange.fetch_balance(
                {"type": "swap", **self._sandbox_params},
            )
            coin = self._margin_coin
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
                params=self._sandbox_params,
            )
        except Exception as e:
            logger.error("Executor: échec grid entry: {}", e)
            return

        filled_qty = float(entry_order.get("filled") or quantity)
        avg_price = float(entry_order.get("average") or event.entry_price)
        order_id = entry_order.get("id", "")

        if filled_qty <= 0:
            logger.error("Executor: grid entry non remplie")
            return

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
                    state.sl_order_id, futures_sym, params=self._sandbox_params,
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
            await self._exchange.create_order(
                futures_sym, "market", close_side, state.total_quantity,
                params={"reduceOnly": True, **self._sandbox_params},
            )
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
                    state.sl_order_id, futures_sym, params=self._sandbox_params,
                )
            except Exception as e:
                logger.debug("Executor: annulation SL grid échouée (probablement déjà exécuté): {}", e)

        # 2. Market close (sauf si SL déjà exécuté sur exchange)
        if event.exit_reason != "sl_global":
            try:
                close_order = await self._exchange.create_order(
                    futures_sym, "market", close_side, state.total_quantity,
                    params={"reduceOnly": True, **self._sandbox_params},
                )
                exit_price = float(
                    close_order.get("average") or event.exit_price or 0,
                )
            except Exception as e:
                logger.error("Executor: échec close grid {}: {}", futures_sym, e)
                self._risk_manager.unregister_position(futures_sym)
                del self._grid_states[futures_sym]
                return
        else:
            exit_price = event.exit_price or state.sl_price

        # 3. P&L net (via méthode existante, inclut fees)
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
    ) -> None:
        """Traite l'exécution du SL grid par Bitget (watchOrders ou polling)."""
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
                params={"reduceOnly": True, **self._sandbox_params},
            )
            exit_price = float(close_order.get("average") or event.exit_price or 0)
        except Exception as e:
            logger.error("Executor: échec close market: {}", e)
            return

        # 3. Calculer P&L
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
                        params=self._sandbox_params,
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

                orders = await self._exchange.watch_orders(
                    params=self._sandbox_params,
                )
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
            await self._handle_exchange_close(symbol, exit_price, exit_reason)
            return

        # Scanner les grid states pour SL match
        for futures_sym, grid_state in list(self._grid_states.items()):
            if order_id == grid_state.sl_order_id:
                exit_price = float(
                    order.get("average") or order.get("price") or grid_state.sl_price,
                )
                await self._handle_grid_sl_executed(futures_sym, grid_state, exit_price)
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
        """Vérifie si une position est toujours ouverte sur l'exchange."""
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
        """Vérifie si la position grid est toujours ouverte sur l'exchange."""
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
    ) -> None:
        """Traite la fermeture d'une position par l'exchange (TP/SL hit)."""
        pos = self._positions.get(symbol)
        if pos is None:
            return

        # Annuler l'autre ordre (si SL hit, annuler TP et vice-versa)
        await self._cancel_pending_orders(symbol)

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
                        params=self._sandbox_params,
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
                    state.leverage, futures_sym, params=self._sandbox_params,
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
                params={"type": "swap", **self._sandbox_params},
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
                    order_id, symbol, params=self._sandbox_params,
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
                params=self._sandbox_params,
            )
            if trades:
                # Dernier trade = le plus récent
                return float(trades[-1].get("price", 0))
        except Exception as e:
            logger.warning("Executor: impossible de fetch exit price: {}", e)

        return 0.0

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
                        params=self._sandbox_params,
                    )
                    if order.get("status") in ("closed", "filled"):
                        return reason
                except Exception:
                    pass

        return "unknown"

    # ─── Helpers ───────────────────────────────────────────────────────

    async def _fetch_positions_safe(self, symbol: str | None = None) -> list[dict]:
        """Fetch positions, gère le sandbox (pas de symbole pour éviter erreur marginCoin)."""
        if self._config.secrets.bitget_sandbox:
            positions = await self._exchange.fetch_positions(params=self._sandbox_params)
            if symbol:
                positions = [p for p in positions if p.get("symbol") == symbol]
            return positions
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

    # ─── Status ────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Retourne le statut de l'executor pour /health et le dashboard."""
        # Backward compat: "position" (première) + "positions" (liste complète)
        pos_info = None
        positions_list: list[dict[str, Any]] = []

        for pos in self._positions.values():
            info = {
                "symbol": pos.symbol,
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "quantity": pos.quantity,
                "sl_price": pos.sl_price,
                "tp_price": pos.tp_price,
                "strategy_name": pos.strategy_name,
            }
            positions_list.append(info)
            if pos_info is None:
                pos_info = info

        for gs in self._grid_states.values():
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
            }
            positions_list.append(info)
            if pos_info is None:
                pos_info = info

        result: dict[str, Any] = {
            "enabled": self.is_enabled,
            "connected": self.is_connected,
            "sandbox": self._config.secrets.bitget_sandbox,
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
                    }
                    for p in gs.positions
                ],
            }

        return {
            "positions": positions_data,
            "grid_states": grid_states_data,
            "risk_manager": self._risk_manager.get_state(),
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
                    )
                    for p in gs_data.get("positions", [])
                ],
            )
            logger.info(
                "Executor: grid restaurée — {} {} niveaux",
                sym, len(self._grid_states[sym].positions),
            )

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
        )
