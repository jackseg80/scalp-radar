# Audit Plan Phase 1 — Entrées Autonomes Executor

## Objectif
Revue du plan Phase 1 (entrées autonomes Executor, indépendantes du paper) pour détecter
erreurs et contradictions avant implémentation.

---

## BUGS CRITIQUES (blocants — le code ne marchera pas)

### BUG 1 : `quantity=0` dans TradeEvent → rejeté systématiquement

**Où dans le plan** : `_on_candle()`, création du TradeEvent
**Code prévu** : `quantity=0,  # calculé par _open_grid_position`

**Réalité** : `_open_grid_position()` lit `event.quantity` directement (l. 962) :
```python
quantity = self._round_quantity(event.quantity, futures_sym)
if quantity <= 0:
    logger.warning("Executor: grid quantité arrondie à 0, trade ignoré")
    return  # ← SYSTÉMATIQUEMENT REJETÉ
```
La méthode ne recalcule PAS la quantity depuis `grid_size_fraction`. Elle attend une quantity > 0.

**Contradiction** : Le plan reconnaît ce point (#12 "points d'attention critiques") mais le code
fourni laisse `quantity=0`. C'est une contradiction directe.

**Fix** : Calculer avant de créer le TradeEvent :
```python
grid_leverage = self._get_grid_leverage(strategy_name)
quantity = (level.size_fraction * available_balance * grid_leverage) / level.entry_price
quantity = self._round_quantity(quantity, futures_sym)
if quantity <= 0:
    continue
```

---

### BUG 2 : Champs inexistants dans TradeEvent → TypeError à runtime

**Où dans le plan** : `_on_candle()`, création du TradeEvent

**Code prévu** :
```python
event = TradeEvent(
    ...
    grid_level=level.index,                        # ← N'EXISTE PAS
    grid_levels_total=strategy._config.num_levels, # ← N'EXISTE PAS
    grid_size_fraction=level.size_fraction,        # ← N'EXISTE PAS
)
```

**Réalité** — Définition complète de TradeEvent (executor.py l. 41-58) :
```python
@dataclass
class TradeEvent:
    event_type: TradeEventType
    strategy_name: str
    symbol: str
    direction: str
    entry_price: float
    quantity: float
    tp_price: float
    sl_price: float
    score: float
    timestamp: datetime
    market_regime: str = ""
    exit_reason: str | None = None
    exit_price: float | None = None
```
Pas de `grid_level`, `grid_levels_total`, `grid_size_fraction`. → `TypeError: __init__() got
unexpected keyword argument 'grid_level'` à chaque candle.

**Fix** : Supprimer ces 3 champs du TradeEvent. `_open_grid_position()` détermine lui-même
`level_num = len(state.positions)` et n'a pas besoin de ces champs depuis l'event.

---

### BUG 3 : Mauvais nom d'attribut balance (`self._balance` vs `self._exchange_balance`)

**Où dans le plan** : `_ensure_balance()` et reset dans `_reconcile()`

**Code prévu** :
```python
if not self._balance_bootstrapped or not self._balance:
    ...
    self._balance = new_balance
return max((self._balance or 0.0) - self._pending_notional, 0.0)
```

**Réalité** — Attribut réel (executor.py l. 185) :
```python
self._exchange_balance: float | None = None  # Hotfix 28a
```
`self._balance` n'existe pas dans `__init__`. → `AttributeError`.

**Fix** : Remplacer toutes les occurrences `self._balance` par `self._exchange_balance` dans
`_ensure_balance()`.

---

### BUG 4 : `_reconcile()` n'existe pas

**Où dans le plan** : section 1.e "Reset du `_pending_notional` dans `_reconcile()`"

**Réalité** : Il n'y a pas de méthode `_reconcile()` pour le balance.
La mise à jour périodique du balance se fait via :
- `_balance_refresh_loop()` → appelle `refresh_balance()` toutes les 5 min
- `refresh_balance()` fait `self._exchange_balance = new_total`

**Fix** : Ajouter le reset `_pending_notional` à la fin de `refresh_balance()` :
```python
# À ajouter dans refresh_balance(), après self._exchange_balance = new_total :
self._balance_bootstrapped = True
if not self._pending_levels:
    self._pending_notional = 0.0
```

---

## BUGS DE LOGIQUE (le code tourne mais ne fait pas ce qu'on veut)

### BUG 5 : Staleness check ne peut jamais se déclencher

**Où dans le plan** : `_on_candle()`, check fraîcheur ctx

**Code prévu** :
```python
staleness = (candle.timestamp - ctx.timestamp).total_seconds()
if staleness > 7200:  # >2h = stale
    logger.warning(...)
    continue
```

**Réalité** : `build_context()` set `timestamp=datetime.now(tz=timezone.utc)` à l'instant
de l'appel (GridStrategyRunner l. 1258, LiveStrategyRunner l. 467).
`ctx.timestamp` = now ≈ moment de `_on_candle()`.
`candle.timestamp` = timestamp de clôture de la bougie (ex: 14:00:00 UTC).

Pour une bougie 1h récente :
- `candle.timestamp = 14:00:00`
- `ctx.timestamp = 14:05:xx` (now)
- `staleness = (14:00:00 - 14:05:xx).total_seconds() = -300s` → JAMAIS > 7200

Ce check ne détectera jamais un indicateur gelé, même si le kill switch paper est actif depuis
2h. La protection principale reste le fix `_dispatch_candle()` (BUG confirmé en section 2 du plan).

**Fix** : Soit supprimer ce check (il ne sert à rien), soit changer la logique :
- Tracker dans un dict `_indicator_last_update[symbol]` le timestamp de la dernière
  `indicator_engine.update()`, et comparer avec `candle.timestamp`.
- Ou simplement vérifier que `ctx.indicators.get(strat_tf, {}).get("sma")` est non-None
  (déjà fait avec le check existant sur les indicateurs).

---

### BUG 6 : `extra_data` perdu lors de la reconstruction du StrategyContext

**Où dans le plan** : `_on_candle()`, reconstruction StrategyContext avec le bon capital

**Code prévu** :
```python
ctx = StrategyContext(
    symbol=ctx.symbol,
    timestamp=ctx.timestamp,
    candles=ctx.candles,
    indicators=ctx.indicators,
    current_position=ctx.current_position,
    capital=available_balance,
    config=ctx.config,
    # extra_data oublié ici
)
```

**Réalité** : `StrategyContext` a un champ `extra_data: dict[str, Any]` (défaut `{}`).
Certaines stratégies grid (grid_funding, grid_multi_tf) lisent `extra_data[EXTRA_FUNDING_RATE]`
dans `compute_grid()`. Sans ce champ, les conditions d'entrée seraient faussées.

**Fix** : Ajouter `extra_data=ctx.extra_data` dans la reconstruction.

---

## POINTS DE CONCEPTION À VÉRIFIER (pas des bugs, mais à confirmer)

### POINT 7 : `Direction` est StrEnum → `level.direction` dans TradeEvent est OK

`class Direction(str, Enum)` confirmé dans models.py. Donc `Direction.LONG == "LONG"` est True
et `isinstance(Direction.LONG, str)` est True.

→ `direction=level.direction` dans TradeEvent fonctionne (pas besoin de `.value`). ✓

---

### POINT 8 : `strategy._config.num_levels` → utiliser `strategy.max_positions` (API publique)

Le plan utilise `strategy._config.num_levels` pour vérifier si la grille est pleine.
La property publique `strategy.max_positions` (définie dans BaseGridStrategy, retourne
`self._config.num_levels`) est préférable pour respecter l'interface.

**Fix** : Remplacer `strategy._config.num_levels` par `strategy.max_positions`.

---

### POINT 9 : `_build_grid_state()` duplique du code de `_check_grid_exit()`

L'exact même construction de `GridState` existe déjà dans `_check_grid_exit()` (l. 562-592) :
```python
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
```

Le plan propose `_build_grid_state()` mais avec `Direction[state.direction]` (by name) au lieu
de `Direction(state.direction)` (by value). Les deux fonctionnent pour StrEnum, mais garder la
même convention que `_check_grid_exit()`.

Recommandation : Extraire en méthode privée partagée `_live_state_to_grid_state()` qui
calcule aussi l'`unrealized_pnl` à partir d'un prix courant optionnel.

---

### POINT 10 : `_on_candle()` dans `_dispatch_candle()` kill switch — CONFIRMÉ

Le problème est RÉEL. Dans `_dispatch_candle()` (l. 1874-1878) :
```python
if self._global_kill_switch:
    return  # ← indicator_engine.update() PAS appelé
# PUIS :
self._indicator_engine.update(symbol, timeframe, candle)
```

La fix proposée (déplacer `indicator_engine.update()` avant le check kill switch +
`update_indicators_only()` sur les runners pour les `_close_buffer`) est correcte.

Confirmation : `GridStrategyRunner` a bien `self._close_buffer: dict[str, deque]` (l. 581) et
`self._ma_period` (l. 580). `build_context()` utilise ces deux attributs (l. 1248-1250).

Pour `LiveStrategyRunner`, `build_context()` n'utilise PAS de `_close_buffer` → `update_indicators_only()` peut être un stub vide si on l'ajoute sur LiveStrategyRunner.

---

### POINT 11 : `TradeEventType.OPEN` — vérifier que la valeur existe

Le plan utilise `TradeEventType.OPEN`. Non vérifié dans cette analyse.
**Vérifier** que cet enum value existe bien dans `TradeEventType`.

---

## VALIDATIONS POSITIVES (plan correct sur ces points)

| Point | Statut |
|-------|--------|
| `to_futures_symbol()` déjà dans executor.py (l. 128-138) | ✓ pas besoin d'import |
| `_get_grid_leverage()` existe (l. 2261-2266) | ✓ |
| `self._strategies: dict[str, BaseGridStrategy]` existe | ✓ |
| `self._simulator` existe (l. 191) | ✓ |
| `get_runner_context()` → `build_context()` fonctionne comme décrit | ✓ |
| `StrategyContext.timestamp` field existe | ✓ |
| `StrategyContext.indicators` a bien le format `{tf: {indicator: val}}` | ✓ |
| Boot order : Simulator callback[0], Executor callback[1] confirmé | ✓ |
| DataEngine supporte multi-callbacks (`self._callbacks` liste) | ✓ |
| `max_positions` property existe et retourne `_config.num_levels` | ✓ |
| `_check_grid_exit()` construit déjà GridState depuis positions live | ✓ factoriser |
| `_exchange_balance` initialisé à `None` → `_ensure_balance()` bootstrap OK | ✓ |
| `Direction` est StrEnum → `level.direction` dans TradeEvent fonctionne | ✓ |

---

## RÉCAPITULATIF : Corrections nécessaires avant implémentation

### Fichier `backend/execution/executor.py`

**`__init__`** : Ajouter `_pending_levels`, `_pending_notional`, `_balance_bootstrapped` (plan OK)

**`_ensure_balance()`** :
- Remplacer `self._balance` → `self._exchange_balance` (BUG 3)

**`_on_candle()`** :
- Calculer `quantity` avant TradeEvent (BUG 1) :
  `quantity = level.size_fraction * available_balance * grid_leverage / level.entry_price`
- Supprimer `grid_level`, `grid_levels_total`, `grid_size_fraction` du TradeEvent (BUG 2)
- Remplacer `strategy._config.num_levels` → `strategy.max_positions` (POINT 8)
- Ajouter `extra_data=ctx.extra_data` dans la reconstruction StrategyContext (BUG 6)
- Supprimer le staleness check (BUG 5, il ne fonctionne pas) ou le remplacer

**`_build_grid_state()`** :
- Utiliser `Direction(state.direction)` au lieu de `Direction[state.direction]` (cohérence POINT 9)
- Peut être factorisé avec `_check_grid_exit()` (POINT 9)

**`refresh_balance()`** :
- Ajouter reset `_pending_notional` + `_balance_bootstrapped = True` ici (BUG 4)

### Fichier `backend/backtesting/simulator.py`

**`_dispatch_candle()`** :
- Déplacer `indicator_engine.update()` AVANT le check `_global_kill_switch` (POINT 10, confirmé)
- Appeler `runner.update_indicators_only()` sur tous les runners quand kill switch actif

**`GridStrategyRunner`** :
- Ajouter méthode `update_indicators_only()` (met à jour `_close_buffer` seulement)

**`LiveStrategyRunner`** :
- Ajouter stub `update_indicators_only()` pour duck-typing (no-op)

### Fichier `backend/api/server.py`

- Câblage `engine.on_candle(executor._on_candle)` après `start_exit_monitor()` (plan OK, séquence correcte)

---

## Tests — Corrections

**Test 4 (indicateurs incomplets)** : Correct ✓
**Test 6 (long triggered)** : Corriger pour fournir une `quantity` > 0 calculée
**Tests 19-21 (balance bootstrap)** : Utiliser `_exchange_balance` (pas `_balance`)
**Tests 26-28 (kill switch indicateurs)** : Confirmer `TradeEventType.OPEN` existe
**Test 15 (stale indicators)** : Supprimer ou réécrire (staleness check ne marche pas)
