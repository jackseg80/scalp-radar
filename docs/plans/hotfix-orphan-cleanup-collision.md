# Plan — Orphan Cleanup + Collision Warning

## Contexte

Quand une stratégie est désactivée (`enabled: false`) et le serveur redémarré, ses positions ouvertes disparaissent silencieusement — aucun log, aucun cleanup. En live, les positions restent sur Bitget sans suivi. En paper, le P&L est perdu.

De plus, en paper trading, 2 runners peuvent ouvrir sur le même symbol sans avertissement, alors qu'en live l'Executor applique l'exclusion mutuelle.

## Fichiers modifiés

| Fichier | Modification |
|---------|-------------|
| `backend/backtesting/simulator.py` | Fix 1 + Fix 2 (~80 lignes ajoutées) |
| `tests/test_simulator.py` | 4 nouveaux tests (~120 lignes) |

---

## Fix 1 — Cleanup positions orphelines au boot

### 1a. Nouveau dataclass `OrphanClosure` (après ligne 57)

```python
@dataclass
class OrphanClosure:
    """Position orpheline fermée au boot (stratégie désactivée)."""
    strategy_name: str
    symbol: str
    direction: str
    entry_price: float
    quantity: float
    estimated_fee_cost: float
    reason: str  # "strategy_disabled"
```

### 1b. Nouveaux attributs `Simulator.__init__` (après ligne 850)

```python
self._orphan_closures: list[OrphanClosure] = []
```

### 1c. Nouvelle méthode `Simulator._cleanup_orphan_runners()`

Signature : `def _cleanup_orphan_runners(self, saved_state: dict, enabled_names: set[str]) -> None`

Logique :
- Itère `saved_state["runners"]`, filtre ceux PAS dans `enabled_names`
- Si pas de position (mono ni grid) → log info, skip
- Si `config.secrets.live_trading` → log CRITICAL "VÉRIFIER BITGET MANUELLEMENT"
- Pour chaque position (mono ou grid) : calcul fee estimé (`entry_fee + qty * price * taker_rate`), créer `OrphanClosure`, log WARNING
- Fee rate depuis `self._config.risk.fees.taker_percent / 100`

### 1d. Appel dans `Simulator.start()` (entre ligne 920 et 921)

Insérer AVANT le bloc `if saved_state is not None` existant (restauration) :

```python
# Cleanup orphans (stratégies désactivées avec positions)
if saved_state is not None:
    enabled_names = {r.name for r in self._runners}
    self._cleanup_orphan_runners(saved_state, enabled_names)
```

L'ordre final dans `start()` :
1. `get_enabled_strategies()` → créer runners (lignes 868-919)
2. **NEW: `_cleanup_orphan_runners()`** — détecte et log les orphelins
3. Restaurer état des runners actifs (lignes 922-926, existant)
4. Warm-up grid (lignes 928-934, existant)
5. Câblage DataEngine (ligne 937, existant)

Note : pas besoin de sauvegarder l'état nettoyé immédiatement — la periodic save (60s) ne sauvegarde que les runners actifs dans `self._runners`, donc les orphelins disparaissent naturellement.

### 1e. Property `orphan_closures` (fin de la classe Simulator)

```python
@property
def orphan_closures(self) -> list[OrphanClosure]:
    return list(self._orphan_closures)
```

---

## Fix 2 — Warning collision paper trading

### 2a. Nouvel attribut `Simulator.__init__` (après ligne 850)

```python
self._collision_warnings: list[dict] = []
```

### 2b. Nouvelle méthode helper `Simulator._get_position_symbols()`

```python
def _get_position_symbols(self, runner) -> set[str]:
    if isinstance(runner, GridStrategyRunner):
        return {s for s, p in runner._positions.items() if p}
    if runner._position is not None and runner._position_symbol:
        return {runner._position_symbol}
    return set()
```

### 2c. Modifier `_dispatch_candle()` (lignes 946-978)

Ajouter snapshot avant la boucle et détection collision après chaque `on_candle()` :

- **Avant la boucle** (après ligne 957) : snapshot `positions_before = {r.name: self._get_position_symbols(r) for r in self._runners}`
- **Après les events de chaque runner** (après ligne 978) : détecter les nouveaux symbols, vérifier si un autre runner a déjà une position dessus

### 2d. Modifier `get_all_status()` (ligne 985-987)

Ajouter les `collision_warnings` dans le status de chaque runner concerné.

### 2e. Property `collision_warnings`

```python
@property
def collision_warnings(self) -> list[dict]:
    return list(self._collision_warnings)
```

---

## Tests (dans `tests/test_simulator.py`)

### `class TestOrphanCleanup` (2 tests)

1. **`test_orphan_cleanup_on_disable`** — saved_state avec 2 runners (strat_a enabled, strat_b disabled avec position mono LONG BTC). Boot → 1 seul runner créé (strat_a). `orphan_closures` contient 1 entrée pour strat_b avec symbol, direction, fee > 0.

2. **`test_orphan_cleanup_no_positions`** — saved_state avec runner disabled sans position → `orphan_closures` vide, pas d'erreur.

### `class TestCollisionWarning` (2 tests)

3. **`test_collision_warning`** — 2 runners dans un Simulator, runner_a a déjà une position LONG BTC, runner_b ouvre SHORT BTC via `on_candle`. `collision_warnings` contient 1 entrée avec symbol=BTC, runner_opening=strat_b, runner_existing=strat_a. Les deux positions existent.

4. **`test_no_collision_different_symbols`** — runner_a sur ETH, runner_b sur BTC, candle BTC dispatché. `collision_warnings` vide.

Pattern : utiliser les helpers existants `_make_candle()`, `_make_runner()`, `_make_pm_config()`. Pour les tests orphan, mocker `get_enabled_strategies` et `is_grid_strategy`. Pour les tests collision, manipuler `sim._runners` directement.

---

## Vérification

1. `uv run pytest tests/test_simulator.py -x -v` — 4 nouveaux tests passent
2. `uv run pytest -x` — 632 tests, 0 régression
3. Vérifier les logs : `ORPHAN PAPER` et `COLLISION` apparaissent dans les cas attendus
