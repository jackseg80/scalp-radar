# Sprint 60 — Exclusion force-close des métriques WFO OOS

**Date** : 28 février 2026
**Durée** : 1 session

## Contexte

La force-close fin de données (fermeture artificielle des positions ouvertes à la fin d'une fenêtre OOS) était ajoutée à `trade_pnls`/`trade_returns` dans les 5 moteurs fast backtest. Ce comportement polluait les métriques WFO :
- Sharpe biaisé par un seul trade "artificiel"
- n_trades gonflé
- profit_factor faussé
- `grid_range_atr` obtenait des Sharpe 13-21 depuis un unique trade force-close avec des positions à peine ouvertes

## Objectif

La force-close fin de fenêtre doit :
1. **Mettre à jour le capital** (impact réaliste — les positions se ferment)
2. **NE PAS être comptabilisée** dans `trade_pnls`/`trade_returns` → exclue de Sharpe, n_trades, profit_factor

## Ce qui n'est PAS touché

- **Force-close direction flip** (~ligne 229, `_simulate_grid_common`) : exit sur signal réel (flip de direction EMA/Supertrend). Reste dans les métriques.

## Divergences identifiées avant implémentation

1. `_compute_fast_metrics()` utilise `sum(trade_pnls)` pour `net_return_pct` (PAS `final_capital`) — variable `force_close_pnl=0.0` inutile
2. `test_force_close_end_of_data` : données ET assertions à modifier (la SMA rattrapait les prix après 50 bougies à 80)
3. `_simulate_grid_range()` : force-close en boucle par position (pas `_calc_grid_pnl` global)

## Modifications

### `backend/optimization/fast_multi_backtest.py`

5 blocs force-close modifiés :
- `_simulate_grid_common()` (~ligne 472) : supprimé `trade_pnls.append(pnl)` et `trade_returns.append(...)`
- `_simulate_grid_range()` (~ligne 772) : supprimé `trade_pnls.append(net)` et `trade_returns.append(...)` dans la boucle per-position
- `_simulate_grid_funding()` (~ligne 979) : supprimé `trade_pnls.append(pnl)` et `trade_returns.append(...)`
- `_simulate_grid_boltrend()` (~ligne 1300) : supprimé `trade_pnls.append(pnl)` et `trade_returns.append(...)`
- `_simulate_grid_momentum()` (~ligne 1591) : supprimé `trade_pnls.append(pnl)` et `trade_returns.append(...)`

### Tests modifiés/créés

- `tests/test_force_close_exclusion.py` : **7 nouveaux tests** couvrant le fix
- `tests/test_grid_atr.py::test_force_close_end_of_data` : n=24 (14 warmup + 10 à 80), assertions `n_trades==0, sharpe==0.0, ret==0.0`
- `tests/test_grid_range_atr.py` : correction look-ahead (SL à candle 2 pas 1)
- `tests/test_grid_funding.py::test_entry_on_negative_funding` : `len(trade_pnls)==0` + `final_capital!=10000`
- `tests/test_grid_boltrend.py::test_sides_long_only` : ajout rebond `prices[380:]=101` pour TP SHORT réel
- `tests/test_grid_boltrend_parity.py::_make_controlled_cache` : `closes[bi-1]=93.0` pour déclencher LONG breakout réel
- `tests/test_grid_trend.py` : 2 tests zone neutre → `len(trade_pnls)==0` (NaN skip bug documenté)
- `tests/test_fast_engine_refactor.py` : `_EXPECTED_ENVELOPE_DCA` → `(55.25, 25667.36, 35.06, 47)` (n_trades 48→47, force-close exclu)

## Résultats

- **2114 tests collectés, 2110 passants**
- 4 échecs pré-existants (non liés au fix) : leverage_sl message, config_assets YAML, multi_timeframe sides, param_grids sides
- 0 régression introduite
