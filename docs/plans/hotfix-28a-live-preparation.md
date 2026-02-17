# Hotfix 28a — Préparation déploiement live

## Contexte

Le bot est prêt à passer en live (mainnet, capital minimal). Trois garde-fous manquent :
1. Après `deploy.sh --clean`, le Selector repart à 0 trades et bloque le live
2. Premier déploiement avec DB vide → le Selector bloque tout sans recours
3. Aucun avertissement si le capital configuré diverge du solde réel Bitget

## Fichiers à modifier (7)

| # | Fichier | Fix |
|---|---------|-----|
| 1 | `backend/core/config.py` | FIX 2 — champ `selector_bypass_at_boot` |
| 2 | `config/risk.yaml` | FIX 2 — clé YAML |
| 3 | `backend/core/database.py` | FIX 1 — méthode `get_trade_counts_by_strategy()` |
| 4 | `backend/execution/executor.py` | FIX 3 — propriété `exchange_balance` |
| 5 | `backend/execution/adaptive_selector.py` | FIX 1 + FIX 2 — DB trades + bypass |
| 6 | `backend/api/server.py` | FIX 1 (wire db) + FIX 3 (capital check) |
| 7 | `tests/test_adaptive_selector.py` | 7 nouveaux tests |

---

## FIX 1 — Selector compte les trades DB

### database.py (ligne ~803, après `get_simulation_trades`)

Nouvelle méthode async :
```python
async def get_trade_counts_by_strategy(self) -> dict[str, int]:
    """COUNT(*) par strategy_name dans simulation_trades."""
```
L'index `idx_sim_trades_strategy` existe déjà → requête rapide.

### adaptive_selector.py

**Constructor** : ajouter param optionnel `db: Database | None = None`, stocker dans `self._db`.
Nouvel attribut `_db_trade_counts: dict[str, int] = {}`.

**Nouvelle méthode** `async _load_trade_counts_from_db()` :
- Appelle `self._db.get_trade_counts_by_strategy()`
- try/except → si DB absente ou erreur, garde `{}`
- Log INFO les compteurs chargés

**`start()`** : appeler `await self._load_trade_counts_from_db()` AVANT `self.evaluate()`.

**`evaluate()`** : remplacer le check min_trades par :
```python
effective_trades = max(perf.total_trades, self._db_trade_counts.get(perf.name, 0))
if effective_trades < self._selector_config.min_trades:
    continue
```

### server.py (ligne 122)

Passer la DB au Selector :
```python
selector = AdaptiveSelector(arena, config, db=db)
```

---

## FIX 2 — Flag bypass Selector au boot

### config.py (RiskConfig, après `regime_filter_enabled` ligne 434)

```python
selector_bypass_at_boot: bool = Field(default=False)
```

### risk.yaml (après `regime_filter_enabled`)

```yaml
selector_bypass_at_boot: false
```

### adaptive_selector.py

**Constructor** : initialiser `_bypass_active` :
```python
self._bypass_active = (
    getattr(config.risk, "selector_bypass_at_boot", False)
    and config.secrets.live_trading
)
```

**`evaluate()`** : quand `_bypass_active` est True, skip min_trades/net_return/profit_factor.
Auto-désactivation quand TOUTES les stratégies live_eligible+active atteignent `min_trades`
(via DB ou mémoire). Si une seule est encore en dessous, le bypass reste actif.
Logique : collecter les eligible dans une liste, vérifier `all(effective >= min for each)`.

Note : `_db_trade_counts` chargés une seule fois au `start()`. Pas de refresh périodique.
Pas grave : une fois que les runners en mémoire dépassent min_trades, le compteur DB
n'est plus utile.

**`get_status()`** : ajouter `bypass_active` et `db_trade_counts` au dict.

### Piège MagicMock dans les tests existants

`_make_config()` retourne un MagicMock → `config.risk.selector_bypass_at_boot` serait truthy par défaut.
**Ajouter explicitement** dans `_make_config()` :
```python
config.risk.selector_bypass_at_boot = False
config.secrets.live_trading = False
```
Cela protège les 12 tests existants de la classe `TestAdaptiveSelector`.

---

## FIX 3 — Warning écart capital

### executor.py

Nouvel attribut `_exchange_balance: float | None = None` dans `__init__`.
Affecter `self._exchange_balance = total` dans `start()` (ligne 238, après fetch_balance).
Nouvelle propriété read-only :
```python
@property
def exchange_balance(self) -> float | None:
    return self._exchange_balance
```

### server.py (après `await executor.start()`, avant `await selector.start()`)

```python
if executor.exchange_balance is not None:
    config_capital = config.risk.initial_capital
    real_balance = executor.exchange_balance
    if config_capital > 0:
        diff_pct = abs(real_balance - config_capital) / config_capital * 100
        if diff_pct > 20:
            msg = (
                f"Capital mismatch: risk.yaml={config_capital:.0f}$ "
                f"vs Bitget={real_balance:.0f}$ (écart {diff_pct:.0f}%)"
            )
            logger.warning(msg)
            await notifier.notify_reconciliation(msg)
```

Utilise `notify_reconciliation` existant (Notifier ligne 216) → Telegram + log.

---

## Tests (7 nouveaux dans test_adaptive_selector.py)

### Tests FIX 1

1. **`test_selector_loads_trades_from_db`** (async)
   - Créer DB temp avec 5 trades `grid_atr` dans `simulation_trades`
   - Boot Selector avec cette DB → `_db_trade_counts["grid_atr"] == 5`
   - Runner a 0 trades mais DB en a 5 ≥ min_trades=3 → autorisé

2. **`test_selector_empty_db`** (async)
   - DB vide → `_db_trade_counts == {}`
   - Runner a 0 trades → rejeté (comportement inchangé)

### Tests FIX 2

3. **`test_selector_bypass_with_live_trading`**
   - `selector_bypass_at_boot=True`, `live_trading=True`
   - Runner a 0 trades, return négatif, PF < 1 → autorisé quand même

4. **`test_selector_bypass_ignored_paper_mode`**
   - `selector_bypass_at_boot=True`, `live_trading=False`
   - `_bypass_active == False` → min_trades appliqué normalement

5. **`test_selector_bypass_auto_deactivates`**
   - Bypass actif, TOUTES les stratégies eligible ont ≥ min_trades
   - Après evaluate(), `_bypass_active == False`

5b. **`test_selector_bypass_stays_if_not_all_ready`**
   - Bypass actif, grid_atr a 5 trades mais vwap_rsi en a 0
   - Après evaluate(), `_bypass_active == True` (pas toutes prêtes)

### Test FIX 3

6. **`test_capital_mismatch_warning`**
   - Tester la propriété `exchange_balance` de l'Executor
   - Valeur None avant start, 8500.0 après affectation

### Helper DB

Utiliser `Database` réelle sur fichier temp (pas de mock) :
```python
async def _create_test_db(trades=None) -> Database:
    # tmpdir + Database(path) + await db.init()
    # INSERT INTO simulation_trades si trades fournis
```

---

## Ordre d'implémentation

1. `config.py` + `risk.yaml` (FIX 2 config, zéro dépendance)
2. `database.py` (FIX 1 query, zéro dépendance)
3. `executor.py` (FIX 3 propriété, zéro dépendance)
4. `adaptive_selector.py` (FIX 1 + FIX 2, dépend de 1-2)
5. `server.py` (wiring, dépend de 3-4)
6. `test_adaptive_selector.py` (6 tests, dépend de tout)

## Vérification

```powershell
# Tous les tests (1094 existants + 6 nouveaux = ~1100)
uv run python -m pytest --tb=short -q

# Tests spécifiques hotfix
uv run python -m pytest tests/test_adaptive_selector.py -v --tb=short
```
