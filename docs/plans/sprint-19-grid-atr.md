# Sprint 19 — Plan : Stratégie Grid ATR

## Contexte

Ajout d'une 10e stratégie **grid_atr** au système scalp-radar. C'est une grid/DCA adaptative où les enveloppes d'entrée sont basées sur l'ATR (volatilité) au lieu de pourcentages fixes (envelope_dca). Le brief utilisateur est très détaillé ; ce plan identifie les ajustements nécessaires après vérification du code existant.

**Résultat attendu :** stratégie complète, fast engine intégré, pipeline WFO opérationnel, ~35 tests, tous les tests passent.

---

## Divergences brief vs code existant (vérifiées)

| # | Point | Brief dit | Code réel | Action |
|---|-------|-----------|-----------|--------|
| 1 | `_calc_grid_pnl_atr` | Nouvelle fonction | `_calc_grid_pnl` (L207-236) est déjà générique (prend `direction`) | **Réutiliser** `_calc_grid_pnl`, pas de duplication |
| 2 | Heuristique TP/SL | Nested if/else par candle color | Booléens `tp_hit`/`sl_hit` + heuristic si both hit (L114-160) | **Suivre le pattern existant** (plus propre) |
| 3 | `_run_fast` liste | Brief ne mentionne que `_INDICATOR_PARAMS` | Ligne 971 : tuple hardcodé de noms de stratégies | **Ajouter `"grid_atr"`** au tuple L971 |
| 4 | `capital <= 0` guard | Brief l'ajoute | Absent du code existant envelope_dca | **Ajouter** (bon safety check) |
| 5 | `build_cache` `atr_by_period` | Brief dit "le cache a DÉJÀ atr_by_period" | Init vide L211, peuplé seulement pour donchian/supertrend (L212) | Ajouter `"grid_atr"` à la condition L212 OU bloc séparé |

---

## Fichiers à modifier/créer (11 fichiers, ~930 lignes)

### 1. `backend/core/config.py` — GridATRConfig (+25 lignes)
- Ajouter `GridATRConfig(BaseModel)` avant `StrategiesConfig` (après `EnvelopeDCAShortConfig` L224)
- Champs : `enabled`, `live_eligible`, `timeframe`, `ma_period`, `atr_period`, `atr_multiplier_start`, `atr_multiplier_step`, `num_levels`, `sl_percent`, `sides`, `leverage`, `weight`, `per_asset`
- Méthode `get_params_for_symbol()` (copie d'EnvelopeDCAConfig)
- **Modifier `StrategiesConfig`** L233 : ajouter `grid_atr: GridATRConfig = Field(default_factory=GridATRConfig)`
- **Modifier `validate_weights`** L248-254 : ajouter `self.grid_atr` dans la liste

### 2. `backend/strategies/grid_atr.py` — NOUVEAU (~180 lignes)
- Classe `GridATRStrategy(BaseGridStrategy)` — pattern exact d'envelope_dca.py
- `compute_indicators()` : calcule SMA + ATR (2 arrays au lieu de 1 seul pour envelope_dca)
- `compute_grid()` : `entry_price = sma ± atr × (start + i × step)` — symétrie naturelle SHORT
- `should_close_all()` : identique à envelope_dca (TP=SMA, SL=% avg_entry)
- `get_tp_price()`, `get_sl_price()`, `get_params()` : standard

### 3. `backend/strategies/factory.py` — (+5 lignes)
- Import `GridATRStrategy`
- Ajouter dans le mapping `create_strategy()` L22-30
- Ajouter dans `get_enabled_strategies()`

### 4. `backend/optimization/__init__.py` — (+5 lignes)
- Import `GridATRConfig`, `GridATRStrategy`
- Ajouter dans `STRATEGY_REGISTRY` L35-45
- Ajouter `"grid_atr"` dans `GRID_STRATEGIES` L51

### 5. `backend/optimization/indicator_cache.py` — (+15 lignes)
- Dans `build_cache()`, ajouter bloc `grid_atr` après L189 :
  - Peupler `bb_sma_dict` avec SMA pour chaque `ma_period` du grid
  - Étendre la condition L212 pour peupler `atr_by_period_dict` avec ATR pour chaque `atr_period`
- Les `if period not in` évitent la double computation si d'autres stratégies ont déjà calculé les mêmes périodes

### 6. `backend/optimization/fast_multi_backtest.py` — (+150 lignes)
- Ajouter branche `elif strategy_name == "grid_atr":` dans `run_multi_backtest_from_cache()` L39
- Nouvelle fonction `_simulate_grid_atr(cache, params, bt_config, direction)` :
  - Copie structurelle de `_simulate_envelope_dca` (L48-204) — ~140-150 lignes
  - **Différences** : accès `cache.atr_by_period[atr_period]` en plus de `cache.bb_sma[ma_period]`, calcul enveloppe `sma ± atr × multiplier`, guard `atr <= 0` et `capital <= 0`
  - **Réutilise** `_calc_grid_pnl` existant (L207-236) — pas de duplication
  - Suit le pattern TP/SL avec booléens `tp_hit`/`sl_hit` (pas les if/else imbriqués du brief)
- **NOTE** : `capital <= 0` guard ajouté seulement dans `_simulate_grid_atr`, pas retrofit dans `_simulate_envelope_dca` (éviter régressions)

### 7. `backend/optimization/walk_forward.py` — (+2 lignes)
- Ajouter `"grid_atr": ["ma_period", "atr_period"]` dans `_INDICATOR_PARAMS` L394-404
- **CRITIQUE** : Ajouter `"grid_atr"` dans le tuple hardcodé L971 pour activer le fast engine

### 8. `config/strategies.yaml` — (+15 lignes)
- Section `grid_atr:` après `envelope_dca_short:` avec `enabled: false`

### 9. `config/param_grids.yaml` — (+12 lignes)
- Section `grid_atr:` avec wfo config (180/60/60) + grille 3240 combos

### 10. `tests/test_grid_atr.py` — NOUVEAU (~550 lignes, ~38 tests)
- **Section 1** — Signaux (~12 tests) : name, max_positions, min_candles, compute_indicators, compute_grid (niveaux LONG, adaptivité ATR, SHORT symétrique, filtrage filled, direction lock, NaN/zero ATR), should_close_all (TP/SL/none)
- **Section 2** — TP/SL prices (~3 tests) : get_tp_price=SMA, get_sl_price LONG/SHORT
- **Section 3** — Fast engine (~8 tests) : run valide, unknown raises, adaptive levels, allocation fixe, no entry NaN, force close, TP at SMA, SL at percent
- **Section 4** — Parité fast/normal (~2 tests) :
  - **Données sinusoïdales** : `close = 100 + 8*sin(2π*i/48)` sur 500 bougies 1h (~20 jours), ATR réaliste (~5-10), `assert n_trades >= 3` pour garantir un test non-trivial
  - Comparer fast engine vs MultiPositionEngine : n_trades identique, net_return ±1%, sharpe ±1%
  - Speed test : fast engine au moins 10× plus rapide
- **Section 5** — Registry & integration (~5 tests) : in registry, in grid_strategies, is_grid_strategy, create_with_params, indicator_params, build_cache
- **Section 6** — Executor helpers (~2 tests) :
  - `test_executor_get_grid_sl_percent_grid_atr` : `executor._get_grid_sl_percent("grid_atr") == 20.0` (vérifie que `getattr(config.strategies, "grid_atr")` fonctionne)
  - `test_executor_get_grid_leverage_grid_atr` : `executor._get_grid_leverage("grid_atr") == 6`

### 11. `tests/test_strategy_registry.py` — (+10 lignes)
- Ajouter `"grid_atr"` dans `ALL_STRATEGIES` L24-34
- Ajouter `"grid_atr"` dans `GRID_STRATEGY_NAMES` L67
- Ajouter `"grid_atr": ["ma_period", "atr_period"]` dans `INDICATOR_PARAMS_EXPECTED` L91-94
- Ajouter test `test_create_grid_atr_with_params()`

---

## Ordre d'exécution

1. **Config** (étape 1) — `config.py` : GridATRConfig + StrategiesConfig + validate_weights
2. **Stratégie** (étape 2) — `grid_atr.py` : la logique métier
3. **Factory + Registry** (étapes 3-4) — intégration dans le pipeline
4. **YAML** (étapes 8-9) — strategies.yaml + param_grids.yaml
5. **Tests signaux** (section 1-2 de test_grid_atr.py) — valider la logique métier
6. **Indicator cache** (étape 5) — build_cache pour grid_atr
7. **Fast engine** (étape 6) — _simulate_grid_atr + branche dans run_multi_backtest_from_cache
8. **Walk forward** (étape 7) — _INDICATOR_PARAMS + _run_fast liste
9. **Tests fast engine + parité + integration** (sections 3-5) + test_strategy_registry.py
10. **`uv run python -m pytest --tb=short -q`** — validation finale

---

## Dette technique à tracer

- **TODO Sprint 20** : factoriser `_simulate_grid_common()` dans `fast_multi_backtest.py`. `_simulate_envelope_dca` et `_simulate_grid_atr` partagent ~80% du code (boucle TP/SL, ouverture positions, force close). Seul le calcul de `entry_price` diffère. Un callback ou un paramètre `entry_fn(sma, atr, lvl) -> float` permettrait d'unifier, mais seulement si c'est propre. Pour ce sprint, la duplication est acceptée.

---

## Vérification

```bash
# Tous les tests doivent passer (actuellement 727 + ~38 nouveaux)
uv run python -m pytest --tb=short -q

# Test ciblé grid_atr uniquement
uv run python -m pytest tests/test_grid_atr.py -v

# Test registry mis à jour
uv run python -m pytest tests/test_strategy_registry.py -v
```
