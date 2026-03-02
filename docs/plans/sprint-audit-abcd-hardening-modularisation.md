# Sprint Audit A-B-C-D — Hardening, CI/CD, Modularisation, Bugs P2

**Date** : 2 mars 2026
**Commits** : 64e7248 (A-C), HEAD (D)

---

## Contexte

Deux audits successifs (Phase 1 surface + Phase 2 deep dive) ont identifié les vrais problèmes
du projet. Ce sprint les adresse par ordre de priorité.

---

## Sprint A — Lifespan hardening

**Problème** : 263 lignes de lifespan, 13 composants initialisés inline, ZERO try-except.
Si Database ou DataEngine échoue, l'app est en état zombie (HTTP répond, trading mort).

**Solution** :
- 9 helpers d'init extraits avec try-except individuel
- `_safe_stop()` avec `asyncio.wait_for(timeout=30)` — shutdown ne bloque plus
- `app.state.startup_components` exposé sur `/health` (status `degraded` si erreur partielle)

**Fichiers** : `backend/api/server.py`, `backend/api/health.py`
**Tests** : `tests/test_lifespan_hardening.py` (8 tests)

---

## Sprint B — CI/CD GitHub Actions

**Problème** : Zéro pipeline. 2196 tests exécutés manuellement. Un push peut casser la prod.

**Solution** : `.github/workflows/test.yml` — pytest + ruff sur push/PR vers main.

**Fichiers** : `.github/workflows/test.yml`

---

## Sprint C — Modularisation executor Phase 1

**Problème** : executor.py 3276 lignes, 3 bugs P0 en 2 semaines.

**Solution** : Extraction de 2 modules :

| Module | Lignes | Fonctions extraites |
|--------|--------|---------------------|
| `boot_reconciler.py` | ~280 | `reconcile_on_boot`, `_reconcile_symbol`, `_reconcile_grid_symbol`, `cancel_orphan_orders` |
| `order_monitor.py` | ~271 | `watch_orders_loop`, `process_watched_order`, `poll_positions_loop`, `check_position_still_open`, `check_grid_still_open`, `handle_exchange_close` |

executor.py : 3276 → 2791 lignes (-485).

**Fichiers** : `backend/execution/executor.py`, `backend/execution/boot_reconciler.py` (nouveau), `backend/execution/order_monitor.py` (nouveau)

**Bugs rencontrés** :
- Import circulaire `boot_reconciler → executor → boot_reconciler` : fix par copie locale `_to_futures_symbol()`
- Tests mockent `executor._handle_exchange_close` mais order_monitor appelait la fonction module directement → fix : route via `ex._handle_exchange_close()`

---

## Sprint D — 4 Bugs P2

### D1 — Zombie position detection

**Problème** : Positions ouvertes >24h sans mouvement bloquent la marge silencieusement.

**Fix** : `Watchdog._check_zombie_positions()` — check #7 dans `_check()`.
Compare `LivePosition.entry_time` et `GridLiveState.opened_at` à 24h.
Alerte `AnomalyType.ZOMBIE_POSITION` (cooldown 1h).

**Fichiers** : `backend/monitoring/watchdog.py`, `backend/alerts/notifier.py`

### D2 — Guard gaps candles

**Problème** : `DataEngine.check_gap()` existe mais ne fait que logger. Indicateurs potentiellement corrompus.

**Fix** :
- `DataEngine.gap_count` : compteur public incrémenté à chaque gap
- Alerte Telegram `AnomalyType.DATA_GAP` (cooldown 5 min)

**Fichiers** : `backend/core/data_engine.py`, `backend/alerts/notifier.py`

### D3 — NaN guard centralisé

**Problème** : Chaque stratégie vérifie les NaN individuellement, mais aucun guard post-signal.
Un `StrategySignal` avec `entry_price=NaN` ou `sl_price=NaN` passait jusqu'à `PositionManager`.
`PositionManager.open_position()` vérifiait `<= 0` mais NaN passe ce test.

**Fix** :
- `StrategySignal.has_nan_prices()` — méthode de validation (tp_price=NaN toléré pour grids inversées)
- Guard dans `Simulator._on_candle()` : `if signal is not None and not signal.has_nan_prices()`
- Guard dans `PositionManager.open_position()` : `math.isnan(entry_price) or math.isnan(sl_price) → return None`

**Fichiers** : `backend/strategies/base.py`, `backend/backtesting/simulator.py`, `backend/core/position_manager.py`

### D4 — Sizing parité fast engine / executor

**Problème** : Fast engine utilise `capital * (1/N)` où capital shrink entre levels sur la même candle.
Executor utilise `allocated_balance * size_fraction` fixe sur toute la candle.
→ Backtest sous-estime les résultats live (~6% selon audit précédent).

**Fix** : `candle_capital = capital` snapshot avant chaque boucle inner DCA — 5 fonctions :
- `_simulate_grid_common()` (grid_atr, envelope_dca, grid_multi_tf)
- `_simulate_grid_range()` (bidirectionnel)
- `_simulate_grid_funding()` (funding grid)
- `_simulate_grid_boltrend()` (boltrend)
- `_simulate_grid_multi_tf()` (multi-tf)

Les levels sur **différentes candles** gardent un capital naturellement réduit (correct).
Valeurs de référence `test_fast_engine_refactor.py` et `test_backtest_audit.py` mises à jour.

**Fichiers** : `backend/optimization/fast_multi_backtest.py`, `tests/test_fast_engine_refactor.py`, `tests/test_backtest_audit.py`

---

## Résultats

| Sprint | Tests ajoutés | Total | Régressions |
|--------|--------------|-------|-------------|
| A | 8 | 2207 | 0 |
| B | 0 | 2207 | 0 |
| C | 0 | 2207 | 0 |
| D | 14 | 2213 | 0 |

5 pré-existants non liés (SUI/XTZ/JUP/param_grids/resample_gaps).

---

## Ce qui reste

| Sprint | Contenu |
|--------|---------|
| E | Frontend tests (Vitest + React Testing Library, 9 composants critiques) |
| F | Alertes margin proximity, funding extrêmes, persist regime snapshots |
| C suite | `entry_handler.py` — fusion `_open_position` + `_open_grid_position` |
