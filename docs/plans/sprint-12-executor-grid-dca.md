# Sprint 12 — Executor Grid DCA + Alertes Telegram

## Contexte

L'Executor gère l'exécution live sur Bitget (Sprint 5a/5b). Il ne supporte que les positions mono (1 entry + SL/TP server-side). Sprint 12 l'adapte pour les cycles grid/DCA multi-niveaux (envelope_dca), où :
- Chaque niveau = 1 market order → mise à jour SL global
- TP = dynamique (SMA courante, détecté par GridStrategyRunner) → client-side
- SL = fixe (% depuis prix moyen) → server-side trigger Bitget

envelope_dca est la seule stratégie enabled (paper trading). Ce sprint la passe en live.

## 8 bugs corrigés vs plan original

| # | Bug | Fix |
|---|-----|-----|
| 1 | AdaptiveSelector bloque envelope_dca | Ajouter mapping + `live_eligible: true` |
| 2 | RiskManager rejette niveau 2 (`position_already_open`) | `pre_trade_check` uniquement au 1er niveau |
| 3 | `record_pnl()` n'existe pas | Utiliser `record_trade_result(LiveTradeResult(...))` |
| 4 | `_watch_orders_loop` dort sans positions mono | Condition inclut `_grid_states` |
| 5 | `_poll_positions_loop` ignore grids | Itérer aussi `_grid_states` |
| 6 | `_cancel_orphan_orders` supprime SL grid | Inclure grid SL dans `tracked_ids` |
| 7 | Leverage 15 au lieu de 6 dans margin check | `leverage_override` param dans `pre_trade_check` |
| 8 | Conflit mono/grid même symbol (Bitget agrège) | Exclusion mutuelle dans Executor |

---

## Fichiers modifiés

| Fichier | Type | Changements |
|---------|------|-------------|
| [strategies.yaml](config/strategies.yaml) | CONFIG | `live_eligible: true` |
| [adaptive_selector.py](backend/execution/adaptive_selector.py) | MODIFIÉ | +1 ligne `_STRATEGY_CONFIG_ATTR` |
| [risk_manager.py](backend/execution/risk_manager.py) | MODIFIÉ | +param `leverage_override` |
| [executor.py](backend/execution/executor.py) | MODIFIÉ | +2 dataclasses, ~10 méthodes, ~350 lignes |
| [notifier.py](backend/alerts/notifier.py) | MODIFIÉ | +2 méthodes grid |
| [telegram.py](backend/alerts/telegram.py) | MODIFIÉ | +2 méthodes format messages grid |
| [test_executor_grid.py](tests/test_executor_grid.py) | NOUVEAU | ~25 tests |
| [test_risk_manager.py](tests/test_risk_manager.py) | MODIFIÉ | +2 tests leverage_override |
| [test_telegram.py](tests/test_telegram.py) | MODIFIÉ | +2 tests format grid |

---

## Étape 1 — Config + AdaptiveSelector + RiskManager

### 1a. `config/strategies.yaml`
```yaml
envelope_dca:
  live_eligible: true   # false → true
```

### 1b. `backend/execution/adaptive_selector.py`
Ajouter dans `_STRATEGY_CONFIG_ATTR` (ligne 20-25) :
```python
"envelope_dca": "envelope_dca",
```

### 1c. `backend/execution/risk_manager.py`

**Modifier signature `pre_trade_check`** — ajouter `leverage_override: int | None = None` :
```python
def pre_trade_check(
    self, symbol, direction, quantity, entry_price,
    free_margin, total_balance,
    leverage_override: int | None = None,  # NOUVEAU
) -> tuple[bool, str]:
```

**Modifier check #5** (ligne 94) — utiliser leverage_override si fourni :
```python
leverage = leverage_override or self._config.risk.position.default_leverage
```

Pas d'autre changement. Le check `position_already_open` reste — l'Executor le contourne en n'appelant `pre_trade_check` qu'au 1er niveau grid. L'exclusion mutuelle mono/grid est gérée par l'Executor.

---

## Étape 2 — Executor : dataclasses + init

### Nouvelles dataclasses (après `LivePosition`, ~ligne 69)

```python
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
    """État complet d'un cycle grid live sur un symbole."""
    symbol: str              # futures "BTC/USDT:USDT"
    direction: str           # "LONG" | "SHORT"
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
```

### Modifier `__init__` — ajouter `_grid_states`
```python
self._grid_states: dict[str, GridLiveState] = {}  # {futures_sym: state}
```

### Helpers
```python
@staticmethod
def _is_grid_strategy(strategy_name: str) -> bool:
    from backend.optimization import is_grid_strategy
    return is_grid_strategy(strategy_name)

def _get_grid_sl_percent(self, strategy_name: str) -> float:
    strat_config = getattr(self._config.strategies, strategy_name, None)
    if strat_config and hasattr(strat_config, "sl_percent"):
        return strat_config.sl_percent
    return 20.0

def _get_grid_leverage(self, strategy_name: str) -> int:
    strat_config = getattr(self._config.strategies, strategy_name, None)
    if strat_config and hasattr(strat_config, "leverage"):
        return strat_config.leverage
    return 6
```

---

## Étape 3 — Executor : dispatch + exclusion mutuelle

### Modifier `handle_event` (lignes 311-326)

```python
async def handle_event(self, event: TradeEvent) -> None:
    if not self._running or not self._connected:
        return

    is_grid = self._is_grid_strategy(event.strategy_name)

    if event.event_type == TradeEventType.OPEN:
        if self._selector and not self._selector.is_allowed(
            event.strategy_name, event.symbol,
        ):
            return
        if is_grid:
            await self._open_grid_position(event)
        else:
            await self._open_position(event)
    elif event.event_type == TradeEventType.CLOSE:
        if is_grid:
            await self._close_grid_cycle(event)
        else:
            await self._close_position(event)
```

### Modifier `_open_position` — exclusion mutuelle (Bug 8)

Ajouter après le check `if futures_sym in self._positions:` (ligne 334) :
```python
# Exclusion mutuelle : pas de mono si grid active sur même symbol
if futures_sym in self._grid_states:
    logger.warning("Executor: cycle grid actif sur {}, ignore OPEN mono", futures_sym)
    return
```

---

## Étape 4 — Executor : ouverture grid

```python
async def _open_grid_position(self, event: TradeEvent) -> None:
    """Ouvre un niveau de la grille DCA."""
    futures_sym = to_futures_symbol(event.symbol)

    # Exclusion mutuelle mono/grid (Bug 8)
    if futures_sym in self._positions:
        logger.warning("Executor: position mono active sur {}, ignore OPEN grid", futures_sym)
        return

    state = self._grid_states.get(futures_sym)
    is_first_level = state is None

    # Pre-trade check UNIQUEMENT au 1er niveau (Bug 2)
    if is_first_level:
        grid_leverage = self._get_grid_leverage(event.strategy_name)

        # Setup leverage au 1er trade grid
        try:
            await self._exchange.set_leverage(
                grid_leverage, futures_sym, params=self._sandbox_params,
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
            return

        ok, reason = self._risk_manager.pre_trade_check(
            futures_sym, event.direction, quantity,
            event.entry_price, free, total,
            leverage_override=grid_leverage,  # Bug 7
        )
        if not ok:
            logger.warning("Executor: grid trade rejeté — {}", reason)
            return
    else:
        quantity = self._round_quantity(event.quantity, futures_sym)
        if quantity <= 0:
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
```

---

## Étape 5 — Executor : SL global + emergency close

```python
async def _update_grid_sl(self, futures_sym: str, state: GridLiveState) -> None:
    """Annule l'ancien SL et place un nouveau basé sur le prix moyen."""
    # 1. Annuler l'ancien SL
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
        logger.critical("Executor: SL GRID IMPOSSIBLE — close urgence {}", futures_sym)
        await self._emergency_close_grid(futures_sym, state)
        return

    state.sl_order_id = sl_order_id
    state.sl_price = new_sl

async def _emergency_close_grid(self, futures_sym: str, state: GridLiveState) -> None:
    """Fermeture d'urgence si SL impossible."""
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
```

---

## Étape 6 — Executor : fermeture grid

```python
async def _close_grid_cycle(self, event: TradeEvent) -> None:
    """Ferme toutes les positions d'un cycle DCA."""
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
        except Exception:
            pass

    # 2. Market close (sauf si SL déjà exécuté sur exchange)
    if event.exit_reason != "sl_global":
        try:
            close_order = await self._exchange.create_order(
                futures_sym, "market", close_side, state.total_quantity,
                params={"reduceOnly": True, **self._sandbox_params},
            )
            exit_price = float(close_order.get("average") or event.exit_price or 0)
        except Exception as e:
            logger.error("Executor: échec close grid: {}", e)
            return
    else:
        exit_price = event.exit_price or state.sl_price

    # 3. P&L net (via méthode existante)
    net_pnl = self._calculate_pnl(
        state.direction, state.avg_entry_price, exit_price, state.total_quantity,
    )

    # 4. RiskManager (Bug 3 fix : record_trade_result, pas record_pnl)
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
        "Executor: GRID CLOSE {} {} — {} niveaux, avg={:.2f} -> {:.2f}, net={:+.2f} ({})",
        state.direction, futures_sym, len(state.positions),
        state.avg_entry_price, exit_price, net_pnl, event.exit_reason,
    )

    # 6. Cleanup
    del self._grid_states[futures_sym]
```

---

## Étape 7 — Surveillance (watchOrders + polling)

### 7a. `_watch_orders_loop` — Bug 4 fix

Changer condition (ligne 595) :
```python
# AVANT: if not self._positions:
if not self._positions and not self._grid_states:
```

### 7b. `_process_watched_order` — scanner grid states

Ajouter après le scan de `self._positions` (avant le log debug ligne 633) :
```python
# Scanner les grid states pour SL match
for futures_sym, grid_state in list(self._grid_states.items()):
    if order_id == grid_state.sl_order_id:
        exit_price = float(order.get("average") or order.get("price") or grid_state.sl_price)
        await self._handle_grid_sl_executed(futures_sym, grid_state, exit_price)
        return
```

Nouvelle méthode :
```python
async def _handle_grid_sl_executed(
    self, futures_sym: str, state: GridLiveState, exit_price: float,
) -> None:
    """Traite l'exécution du SL grid par Bitget."""
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

    spot_sym = state.symbol.split(":")[0] if ":" in state.symbol else state.symbol
    await self._notifier.notify_grid_cycle_closed(
        spot_sym, state.direction,
        len(state.positions), state.avg_entry_price, exit_price,
        net_pnl, "sl_global", state.strategy_name,
    )
    logger.info(
        "Executor: SL grid exécuté {} — net={:+.2f}", futures_sym, net_pnl,
    )
    del self._grid_states[futures_sym]
```

### 7c. `_poll_positions_loop` — Bug 5 fix

Ajouter après le scan de `self._positions` (ligne 648) :
```python
for symbol in list(self._grid_states.keys()):
    await self._check_grid_still_open(symbol)
```

Nouvelle méthode :
```python
async def _check_grid_still_open(self, symbol: str) -> None:
    """Vérifie si la position grid est toujours ouverte sur l'exchange."""
    state = self._grid_states.get(symbol)
    if state is None:
        return
    positions = await self._fetch_positions_safe(symbol)
    has_open = any(float(p.get("contracts", 0)) > 0 for p in positions)
    if not has_open:
        logger.info("Executor: grid {} fermée côté exchange (polling)", symbol)
        exit_price = await self._fetch_exit_price(symbol)
        await self._handle_grid_sl_executed(symbol, state, exit_price)
```

---

## Étape 8 — Réconciliation + orphan orders

### 8a. `_cancel_orphan_orders` — Bug 6 fix

Ajouter dans la construction de `tracked_ids` (après le scan `self._positions`) :
```python
for grid_state in self._grid_states.values():
    if grid_state.sl_order_id:
        tracked_ids.add(grid_state.sl_order_id)
    for gp in grid_state.positions:
        if gp.entry_order_id:
            tracked_ids.add(gp.entry_order_id)
```

### 8b. `_reconcile_on_boot` — ajouter réconciliation grid

Ajouter après `_reconcile_symbol()` loop, avant `_cancel_orphan_orders()` :
```python
# Réconcilier les cycles grid restaurés
for futures_sym in list(self._grid_states.keys()):
    await self._reconcile_grid_symbol(futures_sym)
```

Nouvelle méthode :
```python
async def _reconcile_grid_symbol(self, futures_sym: str) -> None:
    """Réconcilie un cycle grid restauré avec l'exchange."""
    state = self._grid_states.get(futures_sym)
    if state is None:
        return

    positions = await self._fetch_positions_safe(futures_sym)
    has_position = any(float(p.get("contracts", 0)) > 0 for p in positions)

    if has_position:
        # Vérifier SL toujours actif
        if state.sl_order_id:
            try:
                sl_order = await self._exchange.fetch_order(
                    state.sl_order_id, futures_sym,
                    params=self._sandbox_params,
                )
                if sl_order.get("status") in ("closed", "filled"):
                    exit_price = float(sl_order.get("average") or state.sl_price)
                    logger.info("Executor: SL grid exécuté pendant downtime {}", futures_sym)
                    await self._handle_grid_sl_executed(futures_sym, state, exit_price)
                    return
            except Exception:
                pass
        # Leverage
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
        # Fermé pendant downtime
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
        logger.info("Executor: grid {} fermée pendant downtime, net={:+.2f}", futures_sym, net_pnl)
        del self._grid_states[futures_sym]
```

---

## Étape 9 — State persistence

### 9a. `get_state_for_persistence` — ajouter grid_states

Ajouter dans le dict retourné :
```python
"grid_states": {
    sym: {
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
    for sym, gs in self._grid_states.items()
},
```

### 9b. `restore_positions` — restaurer grid_states

Ajouter après la restauration des positions mono :
```python
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
    logger.info("Executor: grid restaurée — {} {} niveaux", sym, len(self._grid_states[sym].positions))
```

### 9c. `get_status` — inclure grid dans le dashboard

Ajouter dans la boucle positions (après l'itération `self._positions`) :
```python
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
```

---

## Étape 10 — Alertes Telegram

### 10a. `backend/alerts/notifier.py` — 2 méthodes

```python
async def notify_grid_level_opened(
    self, symbol, direction, level_num, quantity, entry_price,
    avg_price, sl_price, strategy,
) -> None:
    if self._telegram:
        await self._telegram.send_grid_level_opened(
            symbol, direction, level_num, quantity, entry_price,
            avg_price, sl_price, strategy,
        )

async def notify_grid_cycle_closed(
    self, symbol, direction, num_positions, avg_entry, exit_price,
    net_pnl, exit_reason, strategy,
) -> None:
    if self._telegram:
        await self._telegram.send_grid_cycle_closed(
            symbol, direction, num_positions, avg_entry, exit_price,
            net_pnl, exit_reason, strategy,
        )
```

### 10b. `backend/alerts/telegram.py` — 2 méthodes

```python
async def send_grid_level_opened(
    self, symbol, direction, level_num, quantity, entry_price,
    avg_price, sl_price, strategy,
) -> bool:
    text = (
        f"<b>GRID ENTRY #{level_num}</b>\n"
        f"{direction} {symbol}\n"
        f"Strategie: {strategy}\n"
        f"Entry: {entry_price:.2f} (qty: {quantity:.6f})\n"
        f"Prix moyen: {avg_price:.2f}\n"
        f"SL global: {sl_price:.2f}"
    )
    return await self.send_message(text)

async def send_grid_cycle_closed(
    self, symbol, direction, num_positions, avg_entry, exit_price,
    net_pnl, exit_reason, strategy,
) -> bool:
    status = "WIN" if net_pnl >= 0 else "LOSS"
    text = (
        f"<b>GRID CLOSE — {status}</b>\n"
        f"{direction} {symbol}\n"
        f"Positions: {num_positions}\n"
        f"{avg_entry:.2f} -> {exit_price:.2f}\n"
        f"P&L net: <b>{net_pnl:+.2f}$</b>\n"
        f"Raison: {exit_reason}\n"
        f"Strategie: {strategy}"
    )
    return await self.send_message(text)
```

---

## Étape 11 — Tests

### `tests/test_executor_grid.py` (nouveau, ~25 tests)

**Ouverture** (7 tests) :
1. Open 1er niveau — pre_trade_check + leverage setup + market order + SL placé + register_position
2. Open 2ème niveau — pas de pre_trade_check, SL recalculé (ancien annulé + nouveau)
3. Reject si position mono active sur même symbol (Bug 8)
4. Reject OPEN mono si grid active (Bug 8)
5. SL recalcul vérifie avg_entry_price correct
6. Emergency close si SL échoue (règle #1)
7. Reject si AdaptiveSelector bloque

**Fermeture** (4 tests) :
8. Close TP global (cancel SL + market close + P&L + unregister + Telegram)
9. Close SL global (pas de market close — Bitget l'a fait, pas de cancel SL)
10. Close avec 1 seul niveau
11. Close avec 2 niveaux — P&L sur avg_entry

**Surveillance** (4 tests) :
12. watchOrders tourne avec grid only (Bug 4)
13. watchOrders détecte SL grid exécuté
14. Polling détecte grid fermée (Bug 5)
15. Orphan orders inclut grid SL (Bug 6)

**State** (5 tests) :
16. get_state_for_persistence inclut grid_states
17. restore_positions restaure grid_states
18. Round-trip sérialisation/désérialisation
19. get_status inclut grid positions (type="grid", levels=N)
20. Réconciliation grid — position toujours ouverte

**RiskManager** (3 tests dans `test_risk_manager.py`) :
21. pre_trade_check avec leverage_override=6 → marge correcte
22. pre_trade_check sans override → default_leverage
23. Grid cycle compte comme 1 position pour max_concurrent

**Telegram** (2 tests dans `test_telegram.py`) :
24. Format message grid level opened
25. Format message grid cycle closed

---

## Vérification

```bash
# Tests unitaires
uv run python -m pytest tests/test_executor_grid.py -x -q
uv run python -m pytest tests/test_risk_manager.py -x -q
uv run python -m pytest tests/test_telegram.py -x -q

# Régression complète
uv run python -m pytest -x -q

# Test manuel (LIVE_TRADING=true, capital minimal ~9 USDT)
# 1. Vérifier /health → executor.connected: true
# 2. Attendre signal envelope_dca → log "GRID ENTRY level 0"
# 3. Vérifier Bitget : position + SL trigger
# 4. Vérifier Telegram : message reçu
```
