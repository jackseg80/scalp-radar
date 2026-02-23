# Sprint 36b — Multi-Executor (un Executor par stratégie live)

## Contexte

Un seul `Executor` gère actuellement TOUTES les stratégies live avec les mêmes clés API Bitget. Pour isoler financièrement chaque stratégie (sous-comptes Bitget séparés), il faut créer un Executor par stratégie live, chacun avec ses propres clés API.

**État actuel** :
- grid_atr : `enabled: true, live_eligible: true` — 14 assets, lever 7x, LIVE
- grid_multi_tf : `enabled: false, live_eligible: false` — 17 assets (config prête, à activer)
- grid_boltrend : `enabled: true, live_eligible: false` — PAPER ONLY, pas d'executor

**Pré-requis** : Sprint 36a (ACTIVE_STRATEGIES + Circuit Breaker) déployé et stable.

**Bénéfices** : isolation financière, sync au boot simplifiée, kill switch indépendants.

**Résultat attendu** : ~22 tests nouveaux, 0 régression.

---

## Étape 1 — Config : clés API par stratégie

**Fichier : `backend/core/config.py`**

L'approche Pydantic individuelle (48 champs pour 16 stratégies) est trop lourde. Utiliser `os.environ.get()` dynamique.

### 1a. Méthode `get_executor_keys()` dans AppConfig

```python
def get_executor_keys(self, strategy_name: str) -> tuple[str, str, str]:
    """Retourne (api_key, secret, passphrase) pour un executor.

    Cherche BITGET_API_KEY_{STRATEGY_UPPER}, etc.
    Fallback sur les clés globales si absentes.
    """
    import os
    suffix = strategy_name.upper()

    api_key = os.environ.get(f"BITGET_API_KEY_{suffix}", "")
    secret = os.environ.get(f"BITGET_SECRET_{suffix}", "")
    passphrase = os.environ.get(f"BITGET_PASSPHRASE_{suffix}", "")

    if api_key and secret and passphrase:
        return api_key, secret, passphrase

    # Fallback global
    return (
        self.secrets.bitget_api_key,
        self.secrets.bitget_secret,
        self.secrets.bitget_passphrase,
    )
```

### 1b. Méthode `has_dedicated_keys()` dans AppConfig

```python
def has_dedicated_keys(self, strategy_name: str) -> bool:
    """True si la stratégie a ses propres clés API (sous-compte dédié)."""
    import os
    suffix = strategy_name.upper()
    return bool(
        os.environ.get(f"BITGET_API_KEY_{suffix}", "")
        and os.environ.get(f"BITGET_SECRET_{suffix}", "")
        and os.environ.get(f"BITGET_PASSPHRASE_{suffix}", "")
    )
```

**Convention .env** : `BITGET_API_KEY_GRID_ATR`, `BITGET_SECRET_GRID_ATR`, `BITGET_PASSPHRASE_GRID_ATR`.

---

## Étape 2 — Executor : accepte `strategy_name` + clés custom

**Fichier : `backend/execution/executor.py`**

### 2a. Constructor — ajouter `strategy_name`

```python
def __init__(
    self,
    config: AppConfig,
    risk_manager: LiveRiskManager,
    notifier: Notifier,
    selector: AdaptiveSelector | None = None,
    strategy_name: str | None = None,  # NOUVEAU
) -> None:
    self._strategy_name = strategy_name
    # ... reste inchangé ...
```

Property + log prefix :
```python
@property
def strategy_name(self) -> str | None:
    return self._strategy_name

@property
def _log_prefix(self) -> str:
    return f"Executor[{self._strategy_name}]" if self._strategy_name else "Executor"
```

### 2b. `_create_exchange()` — utiliser les clés par stratégie (ligne 289)

```python
def _create_exchange(self) -> Any:
    import ccxt.pro as ccxtpro
    if self._strategy_name:
        api_key, secret, passphrase = self._config.get_executor_keys(self._strategy_name)
    else:
        api_key = self._config.secrets.bitget_api_key
        secret = self._config.secrets.bitget_secret
        passphrase = self._config.secrets.bitget_passphrase
    return ccxtpro.bitget({
        "apiKey": api_key,
        "secret": secret,
        "password": passphrase,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
        "sandbox": False,
    })
```

### 2c. `set_strategies()` — filtrer à SA stratégie (ligne 518)

```python
def set_strategies(self, strategies, simulator=None):
    if self._strategy_name:
        self._strategies = {k: v for k, v in strategies.items() if k == self._strategy_name}
    else:
        self._strategies = strategies
    self._simulator = simulator
    logger.info("{}: {} stratégies enregistrées", self._log_prefix, len(self._strategies))
```

**Note** : `_on_candle()` itère `self._strategies.items()` → déjà filtré, aucun changement nécessaire.

### 2d. `start()` — setup leverage uniquement sur SES assets (ligne 301+)

Dans la boucle `for asset in self._config.assets`, ajouter un filtre :
```python
if self._strategy_name:
    strat_cfg = getattr(self._config.strategies, self._strategy_name, None)
    per_asset = getattr(strat_cfg, "per_asset", {}) if strat_cfg else {}
    if asset.symbol not in per_asset:
        continue
```

### 2e. `get_status()` — ajouter `strategy_name` dans le retour

```python
result["strategy_name"] = self._strategy_name
```

### 2f. Log prefix — mise à jour mécanique

Remplacer les `"Executor: "` par `f"{self._log_prefix}: "` dans les logs critiques (open/close/SL/TP/start/stop, ~15 endroits). Les logs debug restent inchangés.

---

## Étape 3 — ExecutorManager : couche d'agrégation

**Nouveau fichier : `backend/execution/executor_manager.py`**

Classe mince qui agrège N executors et expose la même interface que l'ancien executor singleton. Le frontend ne change PAS — duck typing.

```python
class ExecutorManager:
    def __init__(self):
        self._executors: dict[str, Executor] = {}
        self._risk_managers: dict[str, LiveRiskManager] = {}

    def add(self, name, executor, risk_mgr): ...
    def get(self, name) -> Executor | None: ...

    @property
    def executors(self) -> dict[str, Executor]: ...
    @property
    def is_enabled(self) -> bool: ...  # any enabled
    @property
    def is_connected(self) -> bool: ...  # all connected
    @property
    def exchange_balance(self) -> float | None: ...  # sum

    def get_status(self) -> dict:
        """Agrège tous les executors dans le format actuel."""
        # Merge positions, grid_states, balances, risk_manager status
        # Ajoute "per_strategy": {name: status} pour le détail

    def get_all_order_history(self, limit=50) -> list[dict]:
        """Merge + tri par timestamp de tous les order_history."""

    async def refresh_all_balances(self) -> dict[str, float | None]: ...
    async def stop_all(self): ...
```

**Clé** : `app.state.executor = executor_mgr` — duck typing, les consommateurs (WS, health, routes) appellent `.get_status()` sans changement.

---

## Étape 4 — State persistence : un fichier par executor

**Fichier : `backend/core/state_manager.py`**

### 4a. Nommage des fichiers

```python
def _executor_state_path(self, strategy_name: str | None = None) -> str:
    if strategy_name:
        return f"data/executor_{strategy_name}_state.json"
    return self._executor_state_file  # "data/executor_state.json" legacy
```

### 4b. `save_executor_state()` — ajouter param `strategy_name`

Signature : `save_executor_state(executor, risk_manager, strategy_name=None)`
Écrit dans `_executor_state_path(strategy_name)`.

### 4c. `load_executor_state()` — migration legacy

```python
async def load_executor_state(self, strategy_name: str | None = None):
    path = self._executor_state_path(strategy_name)
    data = await asyncio.to_thread(self._read_json_file, path)

    # Migration : si fichier per-strategy absent, essayer le legacy
    if data is None and strategy_name:
        legacy = await asyncio.to_thread(self._read_json_file, self._executor_state_file)
        if legacy and isinstance(legacy, dict) and "executor" in legacy:
            logger.info("Migration legacy executor_state.json → {}", path)
            return legacy.get("executor")

    if data is None or not isinstance(data, dict) or "executor" not in data:
        return None
    return data.get("executor")
```

### 4d. `set_executors()` — remplace `set_executor()`

```python
def set_executors(self, executor_mgr) -> None:
    """Enregistre l'ExecutorManager pour la sauvegarde périodique."""
    self._executor_mgr = executor_mgr
    # Backward compat
    self._executor = None
    self._risk_manager = None
```

### 4e. Periodic save loop (ligne 228-233)

Remplacer le bloc `if self._executor is not None:` par :
```python
if hasattr(self, '_executor_mgr') and self._executor_mgr:
    for name, ex in self._executor_mgr.executors.items():
        rm = self._executor_mgr._risk_managers.get(name)
        if ex and rm:
            try:
                await self.save_executor_state(ex, rm, strategy_name=name)
            except Exception as e:
                logger.warning("StateManager: erreur sauvegarde executor {}: {}", name, e)
elif self._executor is not None:
    # Legacy single executor
    await self.save_executor_state(self._executor, self._risk_manager)
```

---

## Étape 5 — Server.py : orchestration multi-executor

**Fichier : `backend/api/server.py`** (lignes 125-182)

### 5a. Helper : stratégies live éligibles

```python
def _get_live_eligible_strategies(config: AppConfig) -> list[str]:
    result = []
    for name in config.strategies.model_fields:
        if name == "custom_strategies":
            continue
        cfg = getattr(config.strategies, name, None)
        if cfg and getattr(cfg, "enabled", False) and getattr(cfg, "live_eligible", False):
            result.append(name)
    return result
```

### 5b. Remplacement du bloc executor (lignes 126-182)

```python
from backend.execution.executor_manager import ExecutorManager

executor_mgr = ExecutorManager()
selector: AdaptiveSelector | None = None

if config.secrets.live_trading and engine and simulator:
    arena = app.state.arena
    selector = AdaptiveSelector(arena, config, db=db)
    live_strategies = _get_live_eligible_strategies(config)

    for strat_name in live_strategies:
        risk_mgr = LiveRiskManager(config, notifier=notifier)
        executor = Executor(config, risk_mgr, notifier, selector=selector, strategy_name=strat_name)

        # Restore état per-strategy
        strat_state = await state_manager.load_executor_state(strategy_name=strat_name)
        if strat_state:
            risk_mgr.restore_state(strat_state.get("risk_manager", {}))
            executor.restore_positions(strat_state)

        await executor.start()

        dedicated = config.has_dedicated_keys(strat_name)
        logger.info("Executor[{}] démarré (sous-compte: {})", strat_name,
                     "dédié" if dedicated else "global/partagé")
        executor_mgr.add(strat_name, executor, risk_mgr)

    # Warning si clés globales partagées entre >1 executor
    if len(live_strategies) > 1 and not all(config.has_dedicated_keys(s) for s in live_strategies):
        logger.warning("Multi-Executor: certains executors partagent les mêmes clés API — "
                       "risque de rate limit. Recommandé : sous-comptes dédiés.")

    await selector.start()

    # Câblage exit monitor + entrées autonomes PAR EXECUTOR
    strategy_instances = simulator.get_strategy_instances()
    for strat_name, executor in executor_mgr.executors.items():
        executor.set_data_engine(engine)
        executor.set_strategies(strategy_instances, simulator=simulator)

        from backend.execution.sync import sync_live_to_paper
        await sync_live_to_paper(executor, simulator)
        await executor.start_exit_monitor()

        engine.on_candle(executor._on_candle)  # DataEngine supporte N callbacks

    if state_manager is not None:
        state_manager.set_executors(executor_mgr)

    logger.info("Multi-Executor: {} executors démarrés ({})",
                len(executor_mgr.executors), list(executor_mgr.executors.keys()))

# app.state — backward compat via duck typing
app.state.executor = executor_mgr if executor_mgr.executors else None
app.state.executor_mgr = executor_mgr
app.state.risk_mgr = None  # Plus de singleton, utiliser executor_mgr._risk_managers
app.state.selector = selector
```

### 5c. Shutdown (lignes 219-225)

```python
if selector:
    await selector.stop()
if executor_mgr and executor_mgr.executors and state_manager:
    for name, ex in executor_mgr.executors.items():
        rm = executor_mgr._risk_managers.get(name)
        if rm:
            await state_manager.save_executor_state(ex, rm, strategy_name=name)
    await executor_mgr.stop_all()
    logger.info("Multi-Executor: tous les executors arrêtés")
```

### 5d. Watchdog — passer executor_mgr

```python
watchdog = Watchdog(
    data_engine=engine, simulator=simulator, notifier=notifier,
    executor_mgr=executor_mgr if executor_mgr.executors else None,
)
```

---

## Étape 6 — API routes : agrégation transparente

**Fichier : `backend/api/executor_routes.py`**

### 6a. GET /status — param optionnel `strategy`

```python
@router.get("/status", dependencies=[Depends(verify_executor_key)])
async def executor_status(
    request: Request,
    strategy: str | None = Query(default=None),
):
    executor_mgr = getattr(request.app.state, "executor", None)
    if executor_mgr is None:
        return {"enabled": False, "message": "Executor non actif"}
    if strategy and hasattr(executor_mgr, 'get'):
        ex = executor_mgr.get(strategy)
        if not ex:
            raise HTTPException(404, f"Executor '{strategy}' non trouvé")
        return ex.get_status()
    return executor_mgr.get_status()  # Agrégé
```

### 6b. POST /refresh-balance

Utiliser `executor_mgr.refresh_all_balances()`.

### 6c. POST /kill-switch/reset — param optionnel `strategy`

Accéder à `executor_mgr._risk_managers[strategy]` au lieu de `app.state.risk_mgr`.

### 6d. GET /orders

Utiliser `executor_mgr.get_all_order_history(limit)`.

### 6e. POST /test-trade et /test-close

Code mort (`handle_event()` n'existe pas). Marquer `deprecated` ou supprimer.

---

## Étape 7 — WebSocket : aucun changement nécessaire

**Fichier : `backend/api/websocket_routes.py`**

`_build_update_data(simulator, arena, executor, engine)` appelle `executor.get_status()`.
`ExecutorManager.get_status()` retourne le même format → duck typing, zéro changement.
`_merge_live_grids_into_state()` reçoit `exec_status` agrégé → fonctionne.

---

## Étape 8 — Watchdog : support multi-executor

**Fichier : `backend/monitoring/watchdog.py`**

### 8a. Constructor — ajouter `executor_mgr`

```python
def __init__(self, ..., executor=None, executor_mgr=None):
    self._executor = executor  # Legacy
    self._executor_mgr = executor_mgr
```

### 8b. `_check()` — itérer les executors

```python
executors_to_check = []
if self._executor_mgr:
    executors_to_check = list(self._executor_mgr.executors.values())
elif self._executor:
    executors_to_check = [self._executor]

for ex in executors_to_check:
    prefix = f"[{ex.strategy_name}]" if ex.strategy_name else ""
    if ex.is_enabled and not ex.is_connected:
        self._current_issues.append(f"Executor {prefix} déconnecté")
    if ex.is_enabled and ex._risk_manager.is_kill_switch_triggered:
        self._current_issues.append(f"Kill switch {prefix} déclenché")
```

---

## Étape 9 — Tests (~22 tests)

**Nouveau fichier : `tests/test_executor_multi.py`**

### Config (3 tests)
1. `test_get_executor_keys_per_strategy` — env vars `BITGET_API_KEY_GRID_ATR` retournées
2. `test_get_executor_keys_fallback_global` — fallback clés globales si absentes
3. `test_has_dedicated_keys` — True/False selon présence des 3 clés

### Executor isolation (4 tests)
4. `test_executor_strategy_name_stored` — property accessible
5. `test_executor_set_strategies_filters_to_own` — `set_strategies()` ne garde que sa stratégie
6. `test_executor_create_exchange_uses_per_strategy_keys` — mock `_create_exchange()`, vérifie les clés
7. `test_executor_log_prefix` — `[grid_atr]` présent

### ExecutorManager (5 tests)
8. `test_manager_add_get` — ajout et récupération
9. `test_manager_get_status_aggregates` — positions mergées, balances sommées
10. `test_manager_is_enabled_any` — True si au moins un enabled
11. `test_manager_get_all_order_history_sorted` — merge + tri par timestamp
12. `test_manager_stop_all` — tous les stop() appelés

### StateManager (4 tests)
13. `test_save_load_per_strategy_state` — round-trip fichier par stratégie
14. `test_legacy_migration_fallback` — fichier legacy lu si per-strategy absent
15. `test_state_file_naming` — `data/executor_grid_atr_state.json` correct
16. `test_set_executors_periodic_save` — sauvegarde tous les executors

### API routes (3 tests)
17. `test_status_aggregated` — GET /status retourne données mergées
18. `test_status_per_strategy` — GET /status?strategy=grid_atr retourne individuel
19. `test_kill_switch_reset_per_strategy` — reset un seul risk_manager

### Watchdog (2 tests)
20. `test_watchdog_multi_executor_checks_all` — tous les executors vérifiés
21. `test_watchdog_backward_compat_single` — ancien param executor= fonctionne encore

### Integration (1 test)
22. `test_live_eligible_strategies_helper` — `_get_live_eligible_strategies()` correct

---

## Étape 10 — Documentation

### `.env.example` — section Multi-Executor

```bash
# === Multi-Executor (Sprint 36b) ===
# Clés par sous-compte (optionnel — fallback sur clés globales)
# Convention : BITGET_API_KEY_{STRATEGY_UPPER}
# BITGET_API_KEY_GRID_ATR=
# BITGET_SECRET_GRID_ATR=
# BITGET_PASSPHRASE_GRID_ATR=
# BITGET_API_KEY_GRID_MULTI_TF=
# BITGET_SECRET_GRID_MULTI_TF=
# BITGET_PASSPHRASE_GRID_MULTI_TF=
```

### `COMMANDS.md` — section 18 Multi-Executor

Ajouter une section décrivant comment créer un sous-compte Bitget et configurer les clés.

---

## Pièges identifiés

1. **Rate limit clés partagées** : Si >1 executor utilise les clés globales, le rate limit Bitget est par-clé → risque de 30006. Warning au boot si détecté.

2. **`_reconcile_on_boot()`** : Avec clés globales partagées, `fetch_positions()` retourne TOUTES les positions. Ajouter un filtre par `strategy_name` dans la réconciliation quand `self._strategy_name` est set.

3. **`app.state.risk_mgr = None`** : Les routes qui accèdent directement à `app.state.risk_mgr` (kill-switch/reset) doivent être mises à jour pour utiliser `executor_mgr._risk_managers`.

4. **Legacy migration** : Premier boot → lit `executor_state.json` → sauvegarde dans `executor_grid_atr_state.json`. Le fichier legacy reste mais devient périmé.

5. **grid_boltrend paper only** : `live_eligible: false` → aucun executor créé, tourne normalement en paper dans le Simulator.

---

## Fichiers à modifier

| Fichier | Changement |
|---------|-----------|
| `backend/core/config.py` | `get_executor_keys()`, `has_dedicated_keys()` |
| `backend/execution/executor.py` | `strategy_name` param, `_create_exchange()`, `set_strategies()` filtre, `start()` filtre assets, log prefix |
| `backend/execution/executor_manager.py` | **NOUVEAU** — classe d'agrégation |
| `backend/core/state_manager.py` | Fichiers per-executor, migration legacy, `set_executors()` |
| `backend/api/server.py` | Boucle multi-executor, câblage N callbacks, shutdown |
| `backend/api/executor_routes.py` | Param `strategy`, agrégation, kill-switch multi |
| `backend/monitoring/watchdog.py` | `executor_mgr` param, itération |
| `tests/test_executor_multi.py` | **NOUVEAU** — 22 tests |
| `.env.example` | Section clés par sous-compte |
| `COMMANDS.md` | Section 18 Multi-Executor |

## Vérification

1. `uv run pytest tests/ -x -q` — 0 régression sur les ~1753 tests existants
2. `uv run pytest tests/test_executor_multi.py -v` — 22 tests passent
3. Boot local avec `LIVE_TRADING=false` — le code multi-executor ne crée aucun executor (pas de crash)
4. Boot local avec `LIVE_TRADING=true` + clés globales uniquement → un seul executor grid_atr créé (backward compat)
