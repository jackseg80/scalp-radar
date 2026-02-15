# Sprint 20c — Factorisation fast engine + auto-dispatch WFO

## Contexte

`_simulate_envelope_dca()` et `_simulate_grid_atr()` dans `fast_multi_backtest.py` partagent ~80% du code (boucle OHLC, TP/SL, allocation, force close). Seul le calcul de `entry_price` diffère. Chaque nouvelle stratégie grid nécessite de copier-coller ~150 lignes. De plus, `walk_forward.py` contient deux tuples hardcodés listant les stratégies fast engine — à mettre à jour manuellement à chaque ajout.

**Objectif** : factoriser en `_build_entry_prices()` + `_simulate_grid_common()`, auto-dériver les listes WFO depuis une constante centralisée. Zéro changement de comportement (parité bit-à-bit).

---

## Fichiers à modifier

| Fichier | Action |
|---------|--------|
| [\_\_init\_\_.py](backend/optimization/__init__.py) | Ajouter `FAST_ENGINE_STRATEGIES` |
| [fast_multi_backtest.py](backend/optimization/fast_multi_backtest.py) | Ajouter `_build_entry_prices` + `_simulate_grid_common`, convertir les 2 fonctions en wrappers |
| [walk_forward.py](backend/optimization/walk_forward.py) | Remplacer 2 tuples hardcodés par import `FAST_ENGINE_STRATEGIES` |
| [tests/test_fast_engine_refactor.py](tests/test_fast_engine_refactor.py) | Nouveau fichier — ~13 tests |

---

## Étape 1 : `backend/optimization/__init__.py`

Après la ligne 54 (`GRID_STRATEGIES`), ajouter :

```python
# Stratégies avec fast engine (WFO accéléré) = toutes sauf celles nécessitant extra_data
FAST_ENGINE_STRATEGIES: set[str] = set(STRATEGY_REGISTRY.keys()) - STRATEGIES_NEED_EXTRA_DATA
```

Résultat : `{"vwap_rsi", "momentum", "bollinger_mr", "donchian_breakout", "supertrend", "envelope_dca", "envelope_dca_short", "grid_atr"}` — identique aux tuples hardcodés actuels.

---

## Étape 2 : `backend/optimization/fast_multi_backtest.py`

### 2a. Nouvelle fonction `_build_entry_prices` (après L18)

```python
def _build_entry_prices(
    strategy_name: str,
    cache: IndicatorCache,
    params: dict[str, Any],
    num_levels: int,
    direction: int,
) -> np.ndarray:
    """Factory retournant un array 2D (n_candles, num_levels) de prix d'entrée.

    NaN propagé pour les candles invalides (SMA NaN, ATR NaN ou <= 0).
    """
    n = cache.n_candles
    sma_arr = cache.bb_sma[params["ma_period"]]
    entry_prices = np.full((n, num_levels), np.nan)

    if strategy_name in ("envelope_dca", "envelope_dca_short"):
        lower_offsets = [
            params["envelope_start"] + lvl * params["envelope_step"]
            for lvl in range(num_levels)
        ]
        if direction == -1:
            envelope_offsets = [round(1 / (1 - e) - 1, 3) for e in lower_offsets]
        else:
            envelope_offsets = lower_offsets

        for lvl in range(num_levels):
            if direction == 1:
                entry_prices[:, lvl] = sma_arr * (1 - envelope_offsets[lvl])
            else:
                entry_prices[:, lvl] = sma_arr * (1 + envelope_offsets[lvl])

    elif strategy_name == "grid_atr":
        atr_arr = cache.atr_by_period[params["atr_period"]]
        multipliers = [
            params["atr_multiplier_start"] + lvl * params["atr_multiplier_step"]
            for lvl in range(num_levels)
        ]
        for lvl in range(num_levels):
            if direction == 1:
                entry_prices[:, lvl] = sma_arr - atr_arr * multipliers[lvl]
            else:
                entry_prices[:, lvl] = sma_arr + atr_arr * multipliers[lvl]
        # ATR NaN ou <= 0 : forcer NaN (SMA NaN déjà propagé naturellement)
        invalid = np.isnan(atr_arr) | (atr_arr <= 0)
        entry_prices[invalid, :] = np.nan

    else:
        raise ValueError(f"Stratégie grid inconnue pour _build_entry_prices: {strategy_name}")

    return entry_prices
```

### 2b. Nouvelle fonction `_simulate_grid_common` (après `_build_entry_prices`)

```python
def _simulate_grid_common(
    entry_prices: np.ndarray,
    sma_arr: np.ndarray,
    cache: IndicatorCache,
    bt_config: BacktestConfig,
    num_levels: int,
    sl_pct: float,
    direction: int,
) -> tuple[list[float], list[float], float]:
    """Boucle chaude unifiée pour toutes les stratégies grid/DCA.

    entry_prices : (n_candles, num_levels) pré-calculé par _build_entry_prices.
    sma_arr : SMA pour TP dynamique (retour vers la SMA).
    sl_pct : déjà divisé par 100.
    """
```

Logique identique à `_simulate_grid_atr` actuelle (L211-360), avec :
- NaN check : `if math.isnan(entry_prices[i, 0]): continue` (remplace le triple check)
- TP = `sma_arr[i]`
- Entry price = `float(entry_prices[i, lvl])` + check `math.isnan(ep) or ep <= 0`
- Guard `capital <= 0` après TP/SL, avant ouverture positions
- Tout le reste identique (TP/SL heuristic, allocation fixe, force close, `_calc_grid_pnl`)

### 2c. Wrappers backward-compat

Remplacer le corps de `_simulate_envelope_dca` (L52-208) par :

```python
def _simulate_envelope_dca(
    cache: IndicatorCache,
    params: dict[str, Any],
    bt_config: BacktestConfig,
    direction: int = 1,
) -> tuple[list[float], list[float], float]:
    """Wrapper backward-compat — délègue à _build_entry_prices + _simulate_grid_common."""
    strategy_name = "envelope_dca_short" if direction == -1 else "envelope_dca"
    num_levels = params["num_levels"]
    sl_pct = params["sl_percent"] / 100
    sma_arr = cache.bb_sma[params["ma_period"]]
    entry_prices = _build_entry_prices(strategy_name, cache, params, num_levels, direction)
    return _simulate_grid_common(
        entry_prices, sma_arr, cache, bt_config, num_levels, sl_pct, direction,
    )
```

Remplacer le corps de `_simulate_grid_atr` (L211-360) par :

```python
def _simulate_grid_atr(
    cache: IndicatorCache,
    params: dict[str, Any],
    bt_config: BacktestConfig,
    direction: int = 1,
) -> tuple[list[float], list[float], float]:
    """Wrapper backward-compat — délègue à _build_entry_prices + _simulate_grid_common."""
    num_levels = params["num_levels"]
    sl_pct = params["sl_percent"] / 100
    sma_arr = cache.bb_sma[params["ma_period"]]
    entry_prices = _build_entry_prices("grid_atr", cache, params, num_levels, direction)
    return _simulate_grid_common(
        entry_prices, sma_arr, cache, bt_config, num_levels, sl_pct, direction,
    )
```

### 2d. `run_multi_backtest_from_cache` — INCHANGÉ

Le dispatcher if/elif existant (L21-49) reste tel quel. Il appelle les wrappers qui délèguent au code commun.

### 2e. `_calc_grid_pnl` et `_compute_fast_metrics` — INCHANGÉS

---

## Étape 3 : `backend/optimization/walk_forward.py`

### Ligne 595 — `collect_combo_results`

```python
# AVANT :
collect_combo_results = strategy_name in ("vwap_rsi", "momentum", "bollinger_mr", "donchian_breakout", "supertrend", "envelope_dca", "envelope_dca_short", "grid_atr")

# APRÈS :
from backend.optimization import FAST_ENGINE_STRATEGIES
collect_combo_results = strategy_name in FAST_ENGINE_STRATEGIES
```

### Ligne 972 — `_parallel_backtest → _run_fast`

```python
# AVANT :
if strategy_name in ("vwap_rsi", "momentum", "bollinger_mr", "donchian_breakout", "supertrend", "envelope_dca", "envelope_dca_short", "grid_atr"):

# APRÈS :
from backend.optimization import FAST_ENGINE_STRATEGIES
if strategy_name in FAST_ENGINE_STRATEGIES:
```

Import lazy (dans la méthode) = même pattern que les imports existants dans walk_forward.py.

---

## Étape 4 : `tests/test_fast_engine_refactor.py` (nouveau)

~13 tests organisés en 3 classes :

### `TestBuildEntryPrices` (7 tests)
1. `test_envelope_dca_long` — shape (n, levels), valeurs = `sma * (1 - offset)`
2. `test_envelope_dca_short` — offsets asymétriques `round(1/(1-e)-1, 3)`
3. `test_grid_atr_long` — valeurs = `sma - atr * mult`
4. `test_grid_atr_nan_atr` — ATR NaN ou <= 0 → entry_prices NaN
5. `test_sma_nan_propagation` — SMA NaN → entry_prices NaN (les 2 stratégies)
6. `test_entry_price_lte_zero` — prix négatifs possibles mais gérés en aval par `_simulate_grid_common`
7. `test_unknown_strategy_raises` — `ValueError`

### `TestFastEngineStrategies` (3 tests)
8. `test_content` — contient les 8 stratégies attendues
9. `test_excludes_extra_data` — funding et liquidation absents
10. `test_derived_from_registry` — `== set(STRATEGY_REGISTRY.keys()) - STRATEGIES_NEED_EXTRA_DATA`

### `TestParityBitwise` (3 tests)
11. `test_parity_envelope_dca_long` — données synthétiques (seed fixe, n=200), résultat `run_multi_backtest_from_cache("envelope_dca", ...)` identique aux valeurs capturées pré-refactoring
12. `test_parity_grid_atr` — idem pour grid_atr
13. `test_parity_envelope_dca_short` — idem pour direction=-1

**Méthode de capture pré-refactoring** : avant le refactoring, exécuter un script de capture sur les mêmes données synthétiques, stocker les valeurs attendues (sharpe, return_pct, profit_factor, n_trades) comme constantes dans le test.

---

## Analyse de parité bit-à-bit

| Aspect | envelope_dca | grid_atr |
|--------|-------------|----------|
| Entry prices | `sma * (1-offset)` vectorisé = identique au scalaire | `sma - atr * mult` vectorisé = identique au scalaire |
| NaN skip | `isnan(entry_prices[i,0])` ⟺ `isnan(sma_arr[i])` | `isnan(entry_prices[i,0])` ⟺ `isnan(sma) or isnan(atr) or atr<=0` |
| Capital guard | Ajouté mais fonctionnellement neutre (capital<0 → qty<0 → skip) | Déjà présent |
| TP/SL heuristic | Branches elif/else cosmétiques supprimées, même résultat | Identique |

**Conclusion** : résultats bit-identiques pour les 3 variantes.

---

## Hors scope (dette technique restante)

- **`_INDICATOR_PARAMS`** (walk_forward.py L394-405) : dict hardcodé mappant chaque stratégie à ses params qui affectent `compute_indicators()`. Utilisé uniquement par `_run_sequential` (fallback si fast engine échoue). Pas un tuple fast engine, mais un point de maintenance manuelle. Factorisation possible via introspection des configs, mais complexité élevée pour un chemin rarement emprunté → hors scope Sprint 20c.

---

## Séquence d'implémentation

1. Capturer valeurs de référence pré-refactoring (script temporaire)
2. `__init__.py` — ajouter `FAST_ENGINE_STRATEGIES`
3. `fast_multi_backtest.py` — ajouter `_build_entry_prices` + `_simulate_grid_common`
4. `fast_multi_backtest.py` — convertir les 2 fonctions en wrappers
5. `walk_forward.py` — remplacer les 2 tuples
6. Créer `tests/test_fast_engine_refactor.py`
7. `pytest --tb=short -q` → 774+ tests passants

---

## Vérification

```powershell
# Tous les tests
uv run python -m pytest --tb=short -q

# Tests ciblés
uv run python -m pytest tests/test_fast_engine_refactor.py -v
uv run python -m pytest tests/test_envelope_dca_short.py tests/test_multi_engine.py tests/test_grid_atr.py -v

# Parité WFO end-to-end (3 assets)
uv run python -m scripts.optimize --strategy grid_atr --symbol BTC/USDT -v
uv run python -m scripts.optimize --strategy grid_atr --symbol DOGE/USDT -v
uv run python -m scripts.optimize --strategy grid_atr --symbol ENJ/USDT -v
# → Comparer grade, score, OOS Sharpe avec les résultats actuels en DB
```
