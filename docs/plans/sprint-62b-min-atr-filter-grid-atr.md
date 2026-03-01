# Sprint 62b — Filtre ATR minimum pour `grid_atr`

**Date** : 1 mars 2026

## Objectif

Ajouter un filtre de volatilité minimum à `grid_atr` : si l'ATR est trop faible, ne pas ouvrir de nouveaux niveaux.
Les sorties (TP, SL, time-stop) des positions existantes restent actives.

## Paramètre

`min_atr_pct` (float, défaut `0.0` = désactivé)

**Logique** : si `ATR(atr_period) / close × 100 < min_atr_pct` → skip ouverture de nouveaux niveaux (exits non bloqués)

## Grille WFO

`min_atr_pct: [0.0, 0.5, 1.0, 1.5]` → 864 → **3456 combos** grid_atr

## Implémentation

### Config
- `backend/core/config.py` : `GridATRConfig.min_atr_pct: float = Field(default=0.0, ge=0, le=10.0)`

### Live path
- `backend/strategies/grid_atr.py`, `compute_grid()` : filtre après `close_val` assignment :
  ```python
  if self._config.min_atr_pct > 0 and close_val > 0:
      if atr_val / close_val * 100 < self._config.min_atr_pct:
          return []
  ```
- `get_params()` : ajout de `"min_atr_pct": self._config.min_atr_pct`

### Backtest path (fast engine)
- `backend/optimization/fast_multi_backtest.py`, `_simulate_grid_common()` :
  - Nouveaux paramètres : `min_atr_pct: float = 0.0`, `atr_arr_for_filter: np.ndarray | None = None`
  - Section 4b (après section 4 zone neutre, avant section 5 ouvertures) :
    ```python
    if min_atr_pct > 0.0 and atr_arr_for_filter is not None:
        atr_i = atr_arr_for_filter[i]
        if close_i > 0 and not math.isnan(atr_i) and atr_i / close_i * 100 < min_atr_pct:
            continue
    ```
- `_simulate_grid_atr()` : extrait `min_atr_pct` des params, passe `atr_arr_for_filter = cache.atr_by_period.get(atr_period)` si `min_atr_pct > 0`

### Param grid
- `config/param_grids.yaml`, section `grid_atr.default` : `min_atr_pct: [0.0, 0.5, 1.0, 1.5]`

## Note architecture

`simulator.py` n'a pas besoin d'être modifié : le live path passe par `GridATRStrategy.compute_grid()` qui contient déjà le filtre.

## Tests (`tests/test_grid_atr_min_atr.py`)

**TestMinAtrFastEngine** (4 tests via fixture `make_indicator_cache`) :
- `test_min_atr_zero_no_filter` : min_atr_pct=0.0 → pas de filtrage (n_trades ≥ 0)
- `test_min_atr_filters_low_vol` : ATR=1.0 (1% de close=100), min_atr_pct=2.0 → 0 trades
- `test_min_atr_exits_still_active` : position ouverte phase 1 (ATR=3%), filtre actif phase 2 (ATR=0.5%) → TP se déclenche quand même
- `test_min_atr_in_param_grid` : `min_atr_pct` présent dans `param_grids.yaml`, liste avec 0.0

**TestMinAtrLiveStrategy** (3 tests via `GridATRStrategy`) :
- `test_compute_grid_filters_when_atr_too_low` : ATR/close=1% < min_atr_pct=2% → `[]`
- `test_compute_grid_passes_when_atr_sufficient` : ATR/close=3% ≥ 2% → niveaux générés
- `test_compute_grid_disabled_when_min_atr_zero` : min_atr_pct=0.0 → toujours des niveaux

## Résultats

- **7 nouveaux tests** → **2172 tests, 2166 passants** (6 pré-existants), 0 régression
