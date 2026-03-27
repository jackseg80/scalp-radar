# Plan : Grid Entry — Market Orders → Limit Orders

## Contexte

L'investigation du 27/03 a révélé que l'écart sim/live de grid_atr (89.7% vs 16.7% win rate) vient d'un **biais d'anticipation** : la simulation entre au prix du niveau de grille (touch théorique via `candle.low`), tandis que l'executor entre en market order au prix courant — souvent 3-4% plus haut après un mouvement.

La correction : placer des **limit orders aux prix des niveaux de grille**, comme la simulation l'assume implicitement. L'exchange se charge du fill quand le prix atteint le niveau.

## Fichiers à modifier

1. **`backend/execution/executor.py`** — changements principaux
2. **`backend/execution/order_monitor.py`** — détection des fills via watchOrders
3. **`backend/execution/boot_reconciler.py`** — réconciliation au boot
4. **Tests** — à ajouter/adapter

## Changements détaillés

### 1. Nouveau dataclass `PendingEntryOrder` (executor.py, après ligne 94)

```python
@dataclass
class PendingEntryOrder:
    """Limit order en attente de fill pour un niveau de grille."""
    order_id: str
    futures_sym: str
    level_index: int
    entry_price: float
    quantity: float
    direction: str  # "LONG" | "SHORT"
    strategy_name: str
    side: str  # "buy" | "sell"
    level_margin: float  # marge engagée pour _pending_notional
    placed_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
```

### 2. Nouvel attribut et constante (executor.py)

- Ligne ~213 : `self._pending_entry_orders: dict[str, dict[int, PendingEntryOrder]] = {}`
  - Clé externe : `futures_sym`, clé interne : `level_index`
- Ligne ~154 : `_LIMIT_PRICE_DRIFT_PCT = 0.002  # 0.2% — seuil pour cancel/replace`
- Ligne ~154 : `_LIMIT_ORDER_MAX_AGE_S = 7200  # 2h — expiration des ordres non remplis`

### 3. Refactorer `_on_candle` (executor.py, lignes 972-1053)

**Supprimer** la boucle actuelle qui fait :
- Touch detection (`candle.low <= level.entry_price`)
- Création de `TradeEvent`
- Appel `_open_grid_position(event)`

**Remplacer par** un appel à `_sync_entry_limits(...)` qui :

```
Pour chaque level dans levels (excluant filled_levels) :
  pending_key = futures_sym, level.index

  SI un pending order existe pour ce level :
    SI le prix a dérivé > 0.2% :
      → cancel l'ancien ordre (avec check fill-race)
      → placer un nouveau limit order
    SINON :
      → rien (l'ordre est toujours valide)

  SINON (pas de pending order) :
    → vérifier margin guard (code existant lignes 998-1018)
    → placer un limit order via _place_grid_limit_order()
    → tracker dans _pending_entry_orders

Pour chaque pending order existant dont le level n'est plus dans levels :
  → cancel l'ordre (avec check fill-race)
  → cleanup tracking + _pending_notional
```

**Conserver tel quel** : tout le code avant la boucle (cooldown, balance, per-asset, compute_grid).

### 4. Nouvelle méthode `_place_grid_limit_order` (executor.py)

Extraite de `_open_grid_position_unlocked` lignes 1554-1644, adaptée pour limit :

```python
async def _place_grid_limit_order(
    self, futures_sym: str, strategy_name: str, level: GridLevel,
    quantity: float, grid_leverage: int, level_margin: float,
) -> PendingEntryOrder | None:
```

Logique :
1. Déterminer `is_first_level` = pas de grid_state ET pas de pending orders pour ce symbol
2. Si first level : exchange position guard, max_live_grids guard, leverage setup, pre-trade check (même code que l'existant)
3. Si pas first level : kill switch check
4. **Limit order** : `create_order(futures_sym, "limit", side, quantity, level.entry_price)`
5. Vérifier si rempli immédiatement (`status == "closed"`) — si oui, appeler directement `_process_entry_fill`
6. Sinon : créer `PendingEntryOrder`, stocker dans `_pending_entry_orders`, incrémenter `_pending_notional`
7. Return le PendingEntryOrder

### 5. Nouvelle méthode `_process_entry_fill` (executor.py)

Extraite de `_open_grid_position_unlocked` lignes 1650-1743 (tout ce qui suit le placement d'ordre) :

```python
async def _process_entry_fill(
    self, futures_sym: str, pending: PendingEntryOrder,
    avg_price: float, filled_qty: float, entry_fee: float,
) -> None:
```

Logique (identique au post-fill actuel) :
1. Supprimer de `_pending_entry_orders`, décrémenter `_pending_notional`
2. Déterminer `is_first_level` = `futures_sym not in self._grid_states`
3. Si first level : créer `GridLiveState`, `risk_manager.register_position()`
4. Append `GridLivePosition` au state
5. `_save_state_now()`
6. `_update_grid_sl(futures_sym, state)` — Rule #1
7. Telegram `notify_grid_level_opened`
8. Log slippage (devrait être ~0 avec limit)
9. `_persist_live_trade(...)`

### 6. Détection des fills via watchOrders (order_monitor.py, ligne 117)

Après le scan des grid SL (ligne 92-117), ajouter un 3e scan :

```python
# Scanner les pending entry orders pour fill match
for futures_sym, level_orders in list(ex._pending_entry_orders.items()):
    for level_idx, pending in list(level_orders.items()):
        if order_id == pending.order_id:
            fill_price = float(order.get("average") or order.get("price") or pending.entry_price)
            filled_qty = float(order.get("filled") or pending.quantity)
            # Fee extraction (même pattern que lignes 54-60)
            await ex._process_entry_fill(futures_sym, pending, fill_price, filled_qty, exit_fee or 0.0)
            return
```

**Aussi ajouter** dans la condition initiale (ligne 25) : `or ex._pending_entry_orders` pour que le watchOrders tourne même sans grid_states (cas où on a des limit orders mais pas encore de position).

### 7. Polling fallback — `_check_pending_entry_fills` (executor.py)

Appelée depuis `_exit_monitor_loop` (ligne 826), après `_check_missing_sl()` :

```python
async def _check_pending_entry_fills(self) -> None:
    """Polling fallback pour détecter les fills des limit orders d'entrée."""
    async with self._state_lock:
        for futures_sym, level_orders in list(self._pending_entry_orders.items()):
            for level_idx, pending in list(level_orders.items()):
                try:
                    order = await self._exchange.fetch_order(pending.order_id, futures_sym)
                    if order.get("status") in ("closed", "filled"):
                        avg_price = float(order.get("average") or pending.entry_price)
                        filled_qty = float(order.get("filled") or pending.quantity)
                        fee_info = order.get("fee") or {}
                        fee = float(fee_info.get("cost") or 0) if fee_info.get("cost") is not None else 0.0
                        await self._process_entry_fill(futures_sym, pending, avg_price, filled_qty, fee)
                    elif order.get("status") in ("canceled", "cancelled", "expired", "rejected"):
                        # Ordre annulé par l'exchange
                        self._pending_entry_orders[futures_sym].pop(level_idx, None)
                        self._pending_notional = max(0.0, self._pending_notional - pending.level_margin)
                    elif (datetime.now(tz=timezone.utc) - pending.placed_at).total_seconds() > _LIMIT_ORDER_MAX_AGE_S:
                        # Expiration locale
                        await self._exchange.cancel_order(pending.order_id, futures_sym)
                        self._pending_entry_orders[futures_sym].pop(level_idx, None)
                        self._pending_notional = max(0.0, self._pending_notional - pending.level_margin)
                except Exception as e:
                    logger.warning("Executor: check pending fill {} lv{}: {}", futures_sym, level_idx, e)
                await asyncio.sleep(_ORDER_DELAY)  # rate limiting
            # Cleanup dict vide
            if not self._pending_entry_orders.get(futures_sym):
                self._pending_entry_orders.pop(futures_sym, None)
```

### 8. Cleanup dans `_close_grid_cycle` (executor.py, ligne ~2006)

Après `_cancel_all_open_orders(futures_sym)`, ajouter :

```python
# Cancel pending entry limit orders pour ce symbole
pending_for_sym = self._pending_entry_orders.pop(futures_sym, {})
for p in pending_for_sym.values():
    self._pending_notional = max(0.0, self._pending_notional - p.level_margin)
```

Même ajout dans `_handle_grid_sl_executed` et `_emergency_close_grid`.

### 9. Persistance d'état (executor.py)

**`get_state_for_persistence`** (ligne 2779) : ajouter `pending_entry_orders` au dict retourné :

```python
"pending_entry_orders": {
    sym: {
        str(idx): {
            "order_id": p.order_id, "futures_sym": p.futures_sym,
            "level_index": p.level_index, "entry_price": p.entry_price,
            "quantity": p.quantity, "direction": p.direction,
            "strategy_name": p.strategy_name, "side": p.side,
            "level_margin": p.level_margin,
            "placed_at": p.placed_at.isoformat(),
        }
        for idx, p in level_orders.items()
    }
    for sym, level_orders in self._pending_entry_orders.items()
},
```

**`restore_positions`** (ligne 2852) : restaurer les pending orders.

### 10. Boot reconciler (boot_reconciler.py)

**`cancel_orphan_orders`** (ligne 323-337) : ajouter les order_ids des pending entry orders aux `tracked_ids` :

```python
for level_orders in ex._pending_entry_orders.values():
    for p in level_orders.values():
        tracked_ids.add(p.order_id)
```

### 11. Suppression de l'ancien code

- `_open_grid_position` et `_open_grid_position_unlocked` : **conservés** comme fallback si besoin, mais plus appelés par `_on_candle`
- La touch detection (`candle.low <= level.entry_price`) dans `_on_candle` : **supprimée**
- `_pending_levels` (set) : **remplacé** par `_pending_entry_orders` (plus complet)

## Race condition : cancel + fill simultané

Quand `_sync_entry_limits` cancel un ordre dont le prix a dérivé :
1. Fetch order status
2. Si `closed`/`filled` → traiter comme fill, ne PAS cancel
3. Si `open` → cancel
4. Si le cancel échoue avec "OrderNotFound" → re-fetch, le traiter comme fill si rempli

## Vérification

1. **Tests unitaires** : mock exchange, vérifier que `create_order("limit", ...)` est appelé avec le bon prix
2. **Test d'intégration** : vérifier le cycle complet place → fill → state update → SL
3. **Sur le serveur** : déployer, observer les logs `Executor: grid LIMIT order placed @ ...`
4. **Validation** : comparer les prix d'entrée live vs sim — l'écart devrait être quasi-nul
5. Commande : `uv run pytest tests/ -x -q -k "executor or grid"` pour les tests existants
