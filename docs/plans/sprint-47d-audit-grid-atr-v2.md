# Sprint 47d — Audit & déploiement grid_atr V2 adaptatif

## Contexte

Audit de bout en bout de la chaîne grid_atr V2 (Sprint 47 : `min_grid_spacing_pct`, `min_profit_pct`).
Objectif : vérifier que les paramètres per_asset du WFO sont bien propagés à tous les niveaux du runtime.

---

## Résultats de l'audit

### Config → Stratégie ✅
- `GridATRConfig.get_params_for_symbol(symbol)` fusionne top-level + per_asset sans whitelist
- Pas de filtrage de noms de paramètres (seules les contraintes Pydantic Field s'appliquent)

### Executor → Stratégie live — GAP CORRIGÉ ❌→✅
- L'executor appelait `strategy.compute_grid()` et `strategy.should_close_all()` sans patcher les per_asset
- Fix : patch `min_grid_spacing_pct` avant `compute_grid()`, `min_profit_pct` avant `should_close_all()`

### Paper Engine (Simulator) — GAP CORRIGÉ ❌→✅
- Le `GridStrategyRunner` patchait seulement `num_levels` avant `compute_grid()`
- `min_grid_spacing_pct` et `min_profit_pct` utilisaient la valeur top-level (0.0) pour TOUS les assets
- Fix : même patches dans les blocs correspondants de `_on_candle_inner()`

### Dashboard ✅
- `get_params()` retourne `min_grid_spacing_pct` et `min_profit_pct` (Sprint 47, déjà fait)

### Logs ✅
- Ajout log "plancher ATR actif" dans `compute_grid()` — une fois par nouveau cycle (quand `not grid_state.positions`)
- Log avec effective_atr, raw_atr, pourcentages et min_grid_spacing_pct

### WFO ✅
- `create_strategy_with_params()` fusionne via `{**defaults, **params}` → tous params propagés
- `portfolio_engine.get_strategy_runners()` utilise `get_params_for_symbol()` → correct

---

## Fichiers modifiés

- `backend/backtesting/simulator.py` : helper `_get_per_asset_float` + 2 patches
- `backend/execution/executor.py` : helper statique `_get_per_asset_float` + 2 patches
- `backend/strategies/grid_atr.py` : import loguru + log plancher ATR
- `docs/WORKFLOW_WFO.md` : note grid_atr V2 + commandes purge
- `tests/test_grid_atr.py` : 3 nouveaux tests (Section 8 : TestGridATRPerAssetChain)

## Résultats

- **1930 tests, 1930 passants**, 0 régression
