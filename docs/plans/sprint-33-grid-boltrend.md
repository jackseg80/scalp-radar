# Sprint 33 — Grid BolTrend (16e strategie)

## Contexte

boltrend (Sprint 30b) obtient Grade C sur 4/5 assets mais seulement 4-6 trades/fenetre OOS (DSR 0.00 = plafond Grade C). Probleme : mono-position = pas assez de trades. Solution : gridifier boltrend pour multiplier les trades par 3-4x.

**Logique hybride** : signal d'activation boltrend (breakout Bollinger) + execution grid DCA (comme grid_atr) dans la direction du breakout. TP inverse : close < SMA (LONG) ou close > SMA (SHORT).

---

## Reponses aux 4 points critiques

### 1. `_build_entry_prices()` — NE PAS UTILISER
Les niveaux sont FIXES au moment du breakout (event-driven), pas recalcules a chaque candle. Precedent : `_simulate_grid_range()` et `_simulate_grid_funding()` de fast_multi_backtest.py n'utilisent pas `_build_entry_prices()`.

### 2. Activation/desactivation dans `_simulate_grid_common()` — IMPOSSIBLE
`_simulate_grid_common()` n'a pas de concept d'activation/desactivation event-driven. Le mecanisme `directions` gere les flips, pas un etat "grid OFF -> breakout -> grid ON". Solution : fonction dediee `_simulate_grid_boltrend()`.

### 3. TP inverse — INCOMPATIBLE avec `_simulate_grid_common()`
TP LONG dans `_simulate_grid_common()` : `highs[i] >= sma_arr[i]` (prix monte vers SMA).
TP LONG dans grid_boltrend : `closes[i] < sma_arr[i]` (prix redescend vers SMA apres breakout).
Conditions diametralement opposees. La fonction dediee gere nativement le TP inverse.

### 4. Zero modification de `_simulate_grid_common()` — REALISABLE
On ecrit `_simulate_grid_boltrend()` dediee, suivant le pattern etabli de `_simulate_grid_range()` et `_simulate_grid_funding()`.

---

## Fichiers modifies/crees

| Fichier | Action |
|---------|--------|
| `backend/core/config.py` | Ajouter `GridBolTrendConfig` + champ dans `StrategiesConfig` |
| `backend/strategies/grid_boltrend.py` | **NOUVEAU** — `GridBolTrendStrategy(BaseGridStrategy)` |
| `backend/strategies/factory.py` | Import + mapping + get_enabled |
| `backend/optimization/__init__.py` | STRATEGY_REGISTRY + GRID_STRATEGIES + STRATEGIES_NEED_EXTRA_DATA |
| `backend/optimization/indicator_cache.py` | Section grid_boltrend (BB + long_ma + ATR) |
| `backend/optimization/walk_forward.py` | `_INDICATOR_PARAMS` |
| `backend/optimization/fast_multi_backtest.py` | `_simulate_grid_boltrend()` + `run_multi_backtest_from_cache()` |
| `backend/execution/adaptive_selector.py` | `_STRATEGY_CONFIG_ATTR` |
| `backend/backtesting/multi_engine.py` | `_GRID_STRATEGIES_WITH_FUNDING` |
| `config/strategies.yaml` | Section grid_boltrend |
| `config/param_grids.yaml` | Grille WFO (1296 combos) |
| `tests/test_grid_boltrend.py` | **NOUVEAU** — 32 tests |
| `tests/test_fast_engine_refactor.py` | +1 strategie dans expected set |

---

## Resultats

- **32 tests grid_boltrend** passent
- **1309 tests total**, zero regression
- 1296 combos WFO (2x3x2x2x2x3x3x3)
- Fast engine dedie `_simulate_grid_boltrend()` (~180 lignes)
- TP inverse sur `closes[i]` (coherent avec boltrend check_exit)
- Zero modification de `_simulate_grid_common()`

---

## Pieges resolus

1. **`cache.bb_sma[long_ma_window]` doit exister** : la section grid_boltrend dans indicator_cache ajoute les SMA long terme dans `bb_sma_dict`
2. **TP inverse sur `closes[i]` pas `highs[i]`** : coherent avec boltrend check_exit
3. **Level 0 trigger immediat** : au breakout, `entry_levels[0] = close` et `lows[i] <= close` toujours vrai → entre immediatement
4. **Heuristique SL + signal_exit** : si les deux sur meme candle, `is_green` decide (bougie verte LONG → signal_exit, rouge → SL)
5. **`boltrend` n'est PAS dans `_INDICATOR_PARAMS`** : normal (mono-position via fast_backtest.py). grid_boltrend DOIT y etre (grid via fast_multi_backtest.py)
6. **`start_idx = max(bol_window, long_ma_window) + 1`** : tests doivent generer breakouts APRES cet index (sinon breakout invisible)
7. **BB bands sensibles au bruit** : tests fast engine utilisent des bands pre-calculees controlees (pas computees depuis prix bruites)
8. **SL check AVANT TP dans should_close_all()** : si entry=88 SHORT + sl=15% → SL a 101.2, close=102 declenche SL avant TP
