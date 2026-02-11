# Sprint 5b — Scaling (3 paires × 4 stratégies + Adaptive Selector)

## Contexte

Sprint 5a a validé le pipeline executor sur 1 stratégie (VWAP+RSI) × 1 paire (BTC). Le fix des ordres trigger orphelins (commit `2bb30de`) sécurise la réconciliation. Sprint 5b déverrouille le scaling : multi-position, multi-stratégie, sélection adaptative.

**Ce qui est DEJA prêt** (pas besoin de modifier) :
- **RiskManager** : `_open_positions: list[dict]`, `pre_trade_check()` gère duplicats + max_concurrent
- **Simulator** : multi-stratégie multi-symbole, dispatch candles à tous les runners
- **Config** : 3 assets dans assets.yaml, 4 stratégies dans strategies.yaml
- **Symbol mapping** : `SYMBOL_SPOT_TO_FUTURES` a déjà BTC, ETH, SOL

**Goulot d'étranglement** : l'Executor avec `self._position: LivePosition | None` (single position) et les filtres hardcodés `_ALLOWED_STRATEGIES = {"vwap_rsi"}`, `_ALLOWED_SYMBOLS = {"BTC/USDT"}`.

---

## Décisions post-review

1. **`min_trades: 3`** (pas 10) — évite de bloquer l'executor pendant des jours au démarrage. Ajustable dans risk.yaml sans redéployer.
2. **`live_eligible: bool`** par stratégie dans strategies.yaml — funding et liquidation utilisent des données OI approximatives, `live_eligible: false`. Le selector vérifie ce flag EN PLUS des critères perf. Seules vwap_rsi et momentum sont `live_eligible: true`.
3. **try/except par symbole** au setup leverage — si SOL échoue, on continue avec BTC+ETH. Set `_active_symbols` pour filtrer.
4. **Log debug ordres non matchés** dans `_process_watched_order()` — aide au debugging si d'autres ordres passent sur le sous-compte.
5. **Refactoring en 2 passes** — Passe 1 : `_position` → `_positions` + migration 40 tests → valider. Passe 2 : multi-symbole + selector + features.
6. **Backward compat persistence** — `restore_positions()` détecte l'ancien format `"position": {...}` vs nouveau `"positions": {...}` et migre automatiquement.
7. **`max_concurrent_same_direction: 2`** — lu depuis `assets.yaml` (déjà configuré dans correlation_groups.crypto_major). Le risk_manager le lit via config.

---

## Fichiers à CRÉER

### 1. `backend/execution/adaptive_selector.py` (~120 lignes)

Classe `AdaptiveSelector` — contrôle quelles stratégies peuvent trader en live :

- **`__init__(arena, config)`** — stocke Arena + config, charge seuils depuis `risk.yaml`, construit `_allowed_symbols` depuis config.assets
- **`is_allowed(strategy_name, symbol) → bool`** — vérifie :
  1. Symbole dans `_active_symbols` (ceux dont le leverage setup a réussi)
  2. Stratégie dans `_allowed_strategies`
- **`evaluate()`** — réévalue depuis Arena.get_ranking() :
  - `live_eligible` = True dans strategies.yaml (funding/liquidation = False)
  - `is_active` = True (kill switch simulation non déclenché)
  - `total_trades >= min_trades` (défaut 3)
  - `net_return_pct > 0`
  - `profit_factor >= min_profit_factor` (défaut 1.0)
  - Log les changements (stratégies ajoutées/retirées)
- **`set_active_symbols(symbols: set[str])`** — appelé par executor après leverage setup
- **`start()`** — évaluation initiale + lance boucle périodique (5 min)
- **`stop()`** — arrête la boucle
- **`get_status() → dict`** — pour dashboard (strategies autorisées, symboles actifs, dernière éval)

**Design** : CLOSE events ne sont PAS filtrés (on doit toujours pouvoir fermer). Seuls les OPEN sont gatés par le selector.

### 2. `tests/test_adaptive_selector.py` (~12 tests)

- Arena vide → aucune stratégie autorisée
- Stratégie `live_eligible: true` avec assez de trades + PF positif → autorisée
- Stratégie `live_eligible: false` (funding) → rejetée même si performante
- Sous le seuil min_trades → rejetée
- Net return négatif → rejeté
- PF trop bas → rejeté
- Kill switch simulation → rejeté
- Évaluation dynamique (ajout/retrait + log)
- Symbole hors `_active_symbols` → rejeté
- `set_active_symbols` met à jour le filtre
- get_status() format correct
- CLOSE events ne sont pas concernés (testé dans test_executor)

---

## Fichiers à MODIFIER

### 3. `backend/execution/executor.py` — REFACTORING MAJEUR (2 passes)

#### PASSE 1 : Structure de données (sans nouvelles features)

**3a.** `self._position: LivePosition | None` → `self._positions: dict[str, LivePosition] = {}`

Propriétés :
- `position` (backward compat) → première position ou None
- `positions` → copie du dict

**3b.** Toutes les méthodes qui référencent `self._position` → lookup par symbole :
- `_open_position()` : `if futures_sym in self._positions` + `self._positions[futures_sym] = LivePosition(...)`
- `_close_position()` : `pos = self._positions.get(futures_sym)` + `del self._positions[futures_sym]`
- `_cancel_pending_orders(symbol)` : nouveau param symbole, lookup `pos = self._positions.get(symbol)`
- `_handle_exchange_close(symbol, ...)` : nouveau param symbole
- `_check_position_still_open(symbol)` : nouveau param symbole
- `_fetch_exit_price(symbol)` : nouveau param symbole
- `_determine_exit_reason(symbol)` : nouveau param symbole
- `get_status()` : retourne `positions` (liste) + `position` (compat)
- `get_state_for_persistence()` : sérialise dict de positions
- `restore_positions()` : désérialise dict + backward compat ancien format single

**→ Valider les 40 tests existants (migrés) avant passe 2.**

#### PASSE 2 : Multi-symbole + selector

**3c.** Supprimer `_ALLOWED_STRATEGIES` et `_ALLOWED_SYMBOLS` (lignes 90-92)

**3d.** `__init__` : nouveau param `selector: AdaptiveSelector | None = None`

**3e.** `handle_event()` : remplacer filtres hardcodés par selector
```python
if event.event_type == TradeEventType.OPEN:
    if self._selector and not self._selector.is_allowed(event.strategy_name, event.symbol):
        return
    await self._open_position(event)
elif event.event_type == TradeEventType.CLOSE:
    await self._close_position(event)  # CLOSE passe toujours
```

**3f.** `start()` : leverage pour tous les symboles avec try/except par symbole
```python
active_symbols: set[str] = set()
for asset in self._config.assets:
    futures_sym = to_futures_symbol(asset.symbol)
    try:
        await self._setup_leverage_and_margin(futures_sym)
        active_symbols.add(asset.symbol)
    except Exception as e:
        logger.warning("Executor: setup échoué pour {} — désactivé: {}", futures_sym, e)
if self._selector:
    self._selector.set_active_symbols(active_symbols)
```

**3g.** `_watch_orders_loop()` : watch TOUS les ordres (sans filtre symbole)
```python
orders = await self._exchange.watch_orders(params=self._sandbox_params)
```

**3h.** `_process_watched_order()` : scanner toutes les positions pour matcher + log debug ordres non matchés

**3i.** `_poll_positions_loop()` : itère `list(self._positions.keys())`

**3j.** `_reconcile_on_boot()` : itérer tous les symboles configurés
```python
configured_symbols = [to_futures_symbol(a.symbol) for a in self._config.assets]
for futures_sym in configured_symbols:
    await self._reconcile_symbol(futures_sym)
await self._cancel_orphan_orders()
```

**3k.** `_cancel_orphan_orders()` : tracked_ids depuis TOUTES les positions

### 4. `backend/execution/risk_manager.py` — Corrélation groups

Ajouter dans `pre_trade_check()` (après max_concurrent, ~ligne 79) :

```python
# Limite direction dans le groupe de corrélation (assets.yaml)
group = self._get_correlation_group(symbol)
if group:
    max_same_dir = self._get_max_same_direction(group)
    same_dir_count = sum(
        1 for pos in self._open_positions
        if pos["direction"] == direction
        and self._get_correlation_group(pos["symbol"]) == group
    )
    if same_dir_count >= max_same_dir:
        return False, f"correlation_group_limit ({group})"
```

Helpers : `_get_correlation_group(symbol)`, `_get_max_same_direction(group)` — mappent via config.assets et config.correlation_groups.

### 5. `backend/core/config.py` — Config adaptive selector + live_eligible

```python
class AdaptiveSelectorConfig(BaseModel):
    min_trades: int = Field(default=3, ge=1)
    min_profit_factor: float = Field(default=1.0, ge=0)
    eval_interval_seconds: int = Field(default=300, ge=30)
```

Ajouter dans `RiskConfig` : `adaptive_selector: AdaptiveSelectorConfig = AdaptiveSelectorConfig()`

Ajouter dans la config de chaque stratégie : `live_eligible: bool = True` (défaut True pour backward compat)

### 6. `config/risk.yaml` — Section adaptive_selector

```yaml
adaptive_selector:
  min_trades: 3
  min_profit_factor: 1.0
  eval_interval_seconds: 300
```

### 7. `config/strategies.yaml` — Activer les stratégies + live_eligible

```yaml
vwap_rsi:
  enabled: true
  live_eligible: true

momentum:
  enabled: true       # était false
  live_eligible: true

funding:
  enabled: true       # était false
  live_eligible: false # données OI approximatives, paper trading only

liquidation:
  enabled: true       # était false
  live_eligible: false # données OI approximatives, paper trading only
```

### 8. `backend/api/server.py` — Lifespan

- Créer `AdaptiveSelector(arena, config)` après l'Arena
- Passer `selector` au constructeur Executor
- `await selector.start()` après `executor.start()`
- `await selector.stop()` en shutdown (avant executor)
- `executor.restore_positions(state)` (pluriel)

### 9. `backend/api/executor_routes.py` — Param symbole

- `POST /test-trade` : `symbol: str = "BTC/USDT"` en query param
- `POST /test-close` : `symbol: str = "BTC/USDT"` en query param
- Utiliser `executor.positions` (dict) au lieu de `executor.position`

### 10. `frontend/src/components/ExecutorPanel.jsx` — Multi-positions

- Lire `positions = executor.positions || []` (liste)
- `.map()` pour afficher chaque position avec symbole + stratégie
- Afficher statut adaptive selector (stratégies actives)
- Backward compat : `executor.position` si `positions` absent

### 11. `tests/test_executor.py` — Mise à jour 40 tests + ~8 nouveaux

**Migration passe 1** (tous les `executor._position` → `executor._positions["BTC/USDT:USDT"]`) :
- Tests event filtering : remplacer tests `_ALLOWED_*` par tests selector
- Tests open/close : `assert "BTC/USDT:USDT" in executor._positions`
- Tests reconciliation : `assert not executor._positions`
- Helper `_make_config` : ajouter ETH et SOL dans config.assets
- Helper `_make_executor` : param `selector=None`

**Nouveaux tests passe 2** (~8 tests) :
- Ouvrir 2 positions (BTC + ETH)
- Même symbole rejeté
- Fermer une garde l'autre
- watchOrders matche la bonne position + log debug non matchés
- Réconciliation multi-symbole (3 symboles, cas variés)
- Persistence multi-position round-trip + backward compat single→multi
- Orphan cleanup multi-position
- Leverage setup failure → symbole désactivé, autres OK

### 12. `tests/test_risk_manager.py` — Tests corrélation

3 tests :
- 2 LONG même groupe → 3ème LONG rejeté (`correlation_group_limit`)
- 2 LONG + 1 SHORT → OK (direction différente)
- Sans groupe de corrélation → pas de limite

---

## Ordre d'implémentation

| Étape | Fichiers | Dépend de |
|-------|----------|-----------|
| 1. Config | `config.py`, `risk.yaml`, `strategies.yaml` | — |
| 2. Adaptive Selector + tests | `adaptive_selector.py`, `test_adaptive_selector.py` | 1 |
| 3. RiskManager corrélation + tests | `risk_manager.py`, `test_risk_manager.py` | 1 |
| 4a. Executor passe 1 (structure) + migration tests | `executor.py`, `test_executor.py` | — |
| 4b. Executor passe 2 (multi-sym + selector) | `executor.py` | 2, 3, 4a |
| 5. Tests executor nouveaux (~8) | `test_executor.py` | 4b |
| 6. Lifespan + API routes | `server.py`, `executor_routes.py` | 4b |
| 7. Frontend ExecutorPanel | `ExecutorPanel.jsx` | 4b |
| 8. Tests complets + validation | tous | 7 |
| 9. Documentation | `docs/plans/sprint-5b-scaling.md` | 8 |

Étapes 1, puis 2+3+4a en parallèle, puis 4b, puis 5+6+7 en parallèle, puis 8, puis 9.

---

## Vérification

1. `uv run pytest` — tous les tests passent (~277)
2. `LIVE_TRADING=false` — 4 stratégies en simulation, selector logge "aucune autorisée" (pas assez de trades encore)
3. `LIVE_TRADING=true BITGET_SANDBOX=true` — executor connecté, leverage setup 3 paires (ou 2 si une échoue)
4. Après ~3 trades simulation → selector autorise la première stratégie `live_eligible`
5. `/api/executor/test-trade?symbol=BTC/USDT` puis `?symbol=ETH/USDT` → 2 positions ouvertes
6. Dashboard affiche 2 positions + stratégies actives du selector
7. Corrélation : 2 LONG crypto_major → 3ème LONG rejeté
8. Restart : positions multi restaurées (test aussi migration ancien format single)
9. Funding/liquidation : `live_eligible: false` → jamais répliquées par l'executor même si performantes
