# Hotfix 34 — Executor P&L basé sur les fills réels Bitget

## Contexte

Premier jour de trading live (18 fév 2026) : l'Executor surestime le P&L de +147% (affiché +7.37$ vs Bitget réel +2.98$). Le kill switch est aveugle car basé sur ce P&L faux. Trois bugs identifiés dans `executor.py` :

1. **Exit/Entry price fallback paper** — quand Bitget retourne `average=None` (fréquent sur market orders), fallback sur `event.exit_price`/`event.entry_price` = prix du Simulator
2. **Fees estimées** — `_calculate_pnl()` utilise `taker_percent` config (0.06%), jamais les fees réelles
3. **Pas de tracking fees entry** — `LivePosition`/`GridLivePosition` n'ont pas de champ `entry_fee`

## Fichier principal

[executor.py](backend/execution/executor.py) — seul fichier backend modifié

## Modifications (séquencées)

### 1. Dataclasses — ajouter `entry_fee`

**`LivePosition` (ligne 57)** : ajouter `entry_fee: float = 0.0` après `tp_price`

**`GridLivePosition` (ligne 74)** : ajouter `entry_fee: float = 0.0` après `entry_time`

**`GridLiveState` (ligne 85)** : ajouter property :
```python
@property
def total_entry_fees(self) -> float:
    return sum(p.entry_fee for p in self.positions)
```

### 2. Nouvelle méthode `_calculate_real_pnl()` (après ligne 1478)

P&L avec fees absolues réelles (USDT). **NE PAS modifier `_calculate_pnl()`** (utilisée dans 7 endroits pour la réconciliation).

```python
def _calculate_real_pnl(self, direction, entry_price, exit_price, quantity, entry_fees, exit_fees) -> float:
    gross = (exit_price - entry_price) * quantity if direction == "LONG" else (entry_price - exit_price) * quantity
    return gross - entry_fees - exit_fees
```

### 3. Nouvelle méthode `_fetch_fill_price()` (après `_fetch_exit_price` ligne 1397)

Quand `average=None` dans un order result, fetch le vrai fill :
1. `fetch_order(order_id)` → `order.average` + `order.fee.cost`
2. Fallback : sleep 0.3s + `fetch_my_trades(symbol, limit=10)` filtré par `order_id`
3. Dernier recours : `(fallback_price, None)` + WARNING log

**Signature retour : `tuple[float, float | None]`** = `(avg_price, total_fee)`.
- `fee = float` → données fees réelles (même si 0.0 = VIP/promo Bitget)
- `fee = None` → impossible de récupérer les fees → l'appelant utilise `_calculate_pnl()` estimé

### 4. Modifier les 2 entries

**`_open_position()` (ligne 492)** et **`_open_grid_position()` (ligne 707)** :

Remplacer `avg_price = float(entry_order.get("average") or event.entry_price)` par :
- Si `average` non-None et > 0 → utiliser directement + extraire `fee.cost`
- Sinon → `await self._fetch_fill_price(order_id, futures_sym, event.entry_price)`
- Log slippage si prix diffère

Passer `entry_fee` au constructeur `LivePosition` (ligne 543) et `GridLivePosition` (ligne 735).

**Convention `entry_fee`** : `float` si fee réelle connue (même 0.0), stockée dans la dataclass. Pour les states restaurés sans fees (backward compat), `entry_fee = 0.0` mais on n'a pas l'info "c'était du real" vs "c'était absent" — acceptable car la persistence ne dure que le temps du cycle grid.

### 5. Modifier les 2 exits

**`_close_grid_cycle()` (ligne 855)** et **`_close_position()` (ligne 954)** :

Remplacer `exit_price = float(close_order.get("average") or event.exit_price or 0)` par :
- Si `average` non-None et > 0 → utiliser directement + extraire `fee.cost` (→ `exit_fee: float`)
- Sinon → `await self._fetch_fill_price(order_id, futures_sym, event.exit_price or 0)`

**Calcul P&L — condition de branchement basée sur `has_real_fees`, pas `fee > 0` :**
```python
# exit_fee vient de _fetch_fill_price → float si données réelles, None si fallback
has_real_fees = exit_fee is not None
if has_real_fees:
    # Fees réelles (même si 0.0 = VIP Bitget)
    entry_fees = state.total_entry_fees  # ou pos.entry_fee pour mono
    net_pnl = self._calculate_real_pnl(
        direction, avg_entry, exit_price, qty, entry_fees, exit_fee,
    )
else:
    # Aucune donnée fee → fallback estimation
    net_pnl = self._calculate_pnl(direction, avg_entry, exit_price, qty)
```

### 6. Modifier les handlers exchange close

**`_handle_grid_sl_executed()` (ligne 901)** : ajouter param `exit_fee: float | None = None`, utiliser `_calculate_real_pnl()` si `exit_fee is not None`.

**`_handle_exchange_close()` (ligne 1123)** : ajouter param `exit_fee: float | None = None`, utiliser `pos.entry_fee` pour les entry fees.

**`_process_watched_order()` (lignes 1040-1060)** — extraction fees enrichie :
```python
# Pour chaque ordre watched (SL/TP exécuté par Bitget)
exit_price = float(order.get("average") or order.get("price") or 0)
fee_info = order.get("fee") or {}
exit_fee = float(fee_info.get("cost") or 0) if fee_info.get("cost") is not None else None

# Si fee absent du WS push (fréquent pour trigger orders Bitget), fetch le fill
if exit_fee is None or exit_fee == 0:
    order_id = order.get("id", "")
    if order_id:
        _, fetched_fee = await self._fetch_fill_price(order_id, symbol, exit_price)
        if fetched_fee is not None:
            exit_fee = fetched_fee

# Passer aux handlers
await self._handle_exchange_close(symbol, exit_price, exit_reason, exit_fee)
# ou pour grid:
await self._handle_grid_sl_executed(futures_sym, grid_state, exit_price, exit_fee)
```

### 7. Persistence

**`get_state_for_persistence()`** :
- Ligne 1556 (LivePosition) : ajouter `"entry_fee": pos.entry_fee`
- Ligne 1568-1576 (GridLivePosition) : ajouter `"entry_fee": p.entry_fee`

**`restore_positions()`** :
- `_restore_single_position()` ligne 1677 : ajouter `entry_fee=pos_data.get("entry_fee", 0.0)`
- GridLivePosition loop ligne 1630 : ajouter `entry_fee=p.get("entry_fee", 0.0)`

Backward compat : `.get("entry_fee", 0.0)` → les states existants restaurés avec fee=0.

### 8. Paths NON modifiés (fallback `_calculate_pnl()` conservé)

- `_reconcile_symbol()` (ligne 1220) — P&L estimé OK (pas de fees entry disponibles)
- `_reconcile_grid_symbol()` (ligne 1297) — idem
- `_check_position_still_open()` / `_check_grid_still_open()` (polling) — exit_price vient de `_fetch_exit_price()` existante, fees non disponibles
  - **TODO** à ajouter dans le code : `# TODO Hotfix 34 : extraire fees depuis fetch_my_trades dans le polling`

## Tests — nouveau fichier `tests/test_executor_real_pnl.py`

~14 tests, réutilisant les helpers de `test_executor_grid.py` (`_make_config`, `_make_mock_exchange`, `_make_executor`) :

| Classe | Test | Vérification |
|--------|------|-------------|
| `TestFetchFillPrice` | `test_from_fetch_order` | fetch_order retourne average+fee → `(price, fee)` |
| | `test_fallback_fetch_my_trades` | fetch_order échoue → fetch_my_trades → prix moyen pondéré + somme fees |
| | `test_fallback_returns_none_fee` | tout échoue → `(fallback_price, None)` + WARNING log |
| `TestCalculateRealPnl` | `test_long` | gross - entry_fees - exit_fees |
| | `test_short` | idem direction SHORT |
| `TestEntryFillPrice` | `test_open_grid_average_none` | create_order average=None → _fetch_fill_price appelé, entry_fee peuplé |
| | `test_open_mono_average_none` | idem pour _open_position |
| `TestExitFillPrice` | `test_close_grid_real_pnl` | close avec fees réelles → _calculate_real_pnl utilisé |
| | `test_close_mono_real_pnl` | idem pour _close_position |
| | `test_close_grid_none_fee_fallback` | exit_fee=None → fallback `_calculate_pnl()` |
| `TestWatchedOrderFees` | `test_watched_sl_fetches_fees` | SL watched avec fee=null → `_fetch_fill_price` appelé pour récupérer les fees |
| `TestPersistence` | `test_entry_fee_round_trip` | save → restore → entry_fee préservé |
| | `test_backward_compat` | state JSON sans entry_fee → restaure avec 0.0 |
| `TestDataclasses` | `test_grid_state_total_entry_fees` | somme des entry_fee de tous les niveaux |

**Tests existants non impactés** — les mocks retournent `average: 50000.0` (non-None), `entry_fee` default 0.0.

## Vérification

```bash
# 1. Tests nouveaux
uv run python -m pytest tests/test_executor_real_pnl.py -x -v

# 2. Non-régression executor
uv run python -m pytest tests/test_executor.py tests/test_executor_grid.py tests/test_executor_orders.py -x -q

# 3. Suite complète
uv run python -m pytest -x -q
```
