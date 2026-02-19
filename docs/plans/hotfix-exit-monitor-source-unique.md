# Hotfix Exit Monitor — Source Unique de Vérité

## Date
2026-02-19

## Problème
L'exit monitor (`_check_grid_exit()` dans `backend/execution/executor.py`) recalculait
la SMA indépendamment du Simulator paper. Après un restart :

- Buffer WS : ~7 candles récentes uniquement
- Paper (IncrementalIndicatorEngine + warmup DB) : 50+ candles → SMA correcte

Résultat : après restart avec marché baissier, SMA artificiellement basse → `close >= sma`
→ faux `tp_global` → 4 positions fermées à perte (-83.51$).

Deux hotfixes incrémentaux avaient été appliqués (per_asset ma_period + DB warmup)
mais ne traitaient pas la cause racine : deux sources d'indicateurs divergentes.

## Solution Architecturale

Source unique de vérité : l'exit monitor lit les indicateurs depuis le runner paper via
`Simulator.get_runner_context(strategy_name, symbol)` → `GridStrategyRunner.build_context(symbol)`.

`build_context()` utilise `_indicator_engine.get_indicators(symbol)` + SMA depuis
`_close_buffer` (warmed par DB au boot) — exactement la même source que le paper.

## Fichiers modifiés

### backend/backtesting/simulator.py
- +`get_runner_context(strategy_name, symbol) -> StrategyContext | None`

### backend/execution/executor.py
- `__init__` : `self._db` → `self._simulator`
- `set_strategies(strategies, simulator=None)` : accepte référence Simulator
- Suppression de `set_db()`
- Réécriture complète de `_check_grid_exit()` : plus aucun calcul SMA/indicateurs

### backend/api/server.py
- `executor.set_strategies(..., simulator=simulator)`
- Suppression `executor.set_db(db)`

### tests/test_executor_autonomous.py
- Helper `_make_simulator_ctx()` remplace `_make_data_engine()`
- Bloc 1 adapté (mock simulator)
- Bloc 4 supprimé (DB warmup, plus pertinent)
- +`test_exit_skips_when_sma_missing`
- +`test_exit_uses_simulator_context_not_buffer`

## Résultats
- 24/24 tests exit monitor passent
- 1447/1447 tests total, 0 régression

## Comportement nouveau
- Si runner paper en warm-up (pas encore d'indicateurs) → skip exit → pas de faux TP
- Log debug à chaque check (même si exit_reason is None)
- SL server-side Bitget protège pendant les périodes de skip
