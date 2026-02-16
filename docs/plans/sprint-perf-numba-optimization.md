# Sprint Perf — Optimisation Backtest (Numba + Numpy vectorisé)

## Contexte

Le WFO exécute des milliers de backtests par run. Profiling :
- **60-70%** du temps : boucle Python scalaire `_simulate_trades()` (fast_backtest.py)
- **15-25%** : boucles Wilder des indicateurs (RSI, ATR, ADX, EMA, SuperTrend)
- Le reste : signaux vectorisés numpy (déjà rapide) + overhead

Ces boucles sont séquentielles (`result[i] = f(result[i-1])`) — impossibles à vectoriser, mais Numba `@njit` les compile en code machine natif (speedup 10-50x). Certaines opérations *peuvent* être vectorisées en pur numpy (True Range, rolling std). Enfin, `pandas>=2.2` est déclaré mais jamais importé.

**Speedup global attendu : 3-10x sur un run WFO complet.**

> Note : `_run_fast()` tourne dans le **process principal** (pas dans les workers ProcessPoolExecutor), donc l'import numba (~100MB) n'a pas le même problème que scipy (qui coûtait ~200MB *par worker*).

## Fichiers à modifier

| Fichier | Changements |
|---------|-------------|
| `pyproject.toml` | Retirer pandas, ajouter numba (groupe optional) |
| `backend/core/indicators.py` | Vectoriser TR + rolling std, Numba sur Wilder loops |
| `backend/optimization/fast_backtest.py` | Numba sur `_simulate_trades()` (1 fonction par stratégie) |
| `backend/optimization/indicator_cache.py` | Vectoriser `_rolling_max()` / `_rolling_min()` |

## Pré-requis : vérifier compatibilité Numba

Avant toute implémentation, valider :

```bash
uv pip install numba>=0.61 --dry-run
python -c "import numba; print(numba.__version__)"
```

Si Python 3.13 n'est pas supporté par numba stable → options :
- Pin Python local à 3.12 (Docker est déjà 3.12)
- Utiliser numba RC/dev si dispo
- Se limiter aux Phases 0-1 (numpy pur) en attendant

---

## Phase 0 — Retirer pandas (5 min, risque zéro)

**`pyproject.toml:10`** : supprimer `"pandas>=2.2"`.

Confirmé : 0 imports pandas dans tout le codebase. Gain : image Docker plus légère.

Vérification : `uv lock && uv sync && uv run pytest` → 902 tests OK.

---

## Phase 1 — Vectorisation numpy pure (pas de nouvelle dépendance)

### 1a. True Range dans `atr()` — `indicators.py:140-147`

Remplacer la boucle Python par numpy vectorisé :

```python
# Avant (boucle)
for i in range(1, len(closes)):
    hl = highs[i] - lows[i]
    hc = abs(highs[i] - closes[i - 1])
    lc = abs(lows[i] - closes[i - 1])
    tr[i] = max(hl, hc, lc)

# Après (vectorisé)
tr[1:] = np.maximum(
    highs[1:] - lows[1:],
    np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
)
```

### 1b. True Range + DM dans `adx()` — `indicators.py:188-201`

Même vectorisation pour TR + `+DM` / `-DM` via `np.where`.

### 1c. Rolling std dans `bollinger_bands()` — `indicators.py:325-328`

Remplacer la boucle par `sliding_window_view` (numpy ≥ 1.20, on a ≥ 2.0) :

```python
from numpy.lib.stride_tricks import sliding_window_view
std_arr = np.full(n, np.nan, dtype=float)
if n >= period:
    windows = sliding_window_view(closes, period)
    std_arr[period - 1:] = np.std(windows, axis=1, ddof=0)
```

> Note mémoire : `sliding_window_view` crée une **vue** (pas de copie), mais `np.std(axis=1)` itère sur toute la matrice. Pour les données actuelles (2000-8000 candles, period 20-50), c'est largement OK. Pour 100k+ candles avec gros window, un algo en deque monotone serait mieux — hors scope pour l'instant.

### 1d. Rolling max/min dans `indicator_cache.py:381-401`

Même approche `sliding_window_view` pour `_rolling_max()` et `_rolling_min()`.

```python
def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    result = np.full_like(arr, np.nan, dtype=float)
    n = len(arr)
    if n > window:
        from numpy.lib.stride_tricks import sliding_window_view
        views = sliding_window_view(arr, window)
        result[window:] = np.max(views[:-1], axis=1)
    return result
```

(`views[:-1]` car l'original exclut l'élément courant : `max(arr[i-window:i])`)

**Vérification Phase 1** : `uv run pytest` → 902 tests OK. Tests critiques : `test_indicators.py`, `test_fast_backtest.py` (parity tests avec `rtol=1e-10` — déjà tolérant).

---

## Phase 2 — Numba JIT sur indicateurs (incrémental)

### 2a. Setup + `ema()` seul (valider l'approche)

**`pyproject.toml`** — ajouter numba dans un groupe optionnel :

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]
optimization = [
    "numba>=0.61",
]
```

Installation locale : `uv sync --group optimization`

**`indicators.py`** — import avec fallback :

```python
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda func: func
```

Puis extraire la boucle `ema()` seule dans un `@njit(cache=True)` :

```python
@njit(cache=True)
def _ema_loop(values, result, period, multiplier):
    for i in range(period, len(values)):
        result[i] = values[i] * multiplier + result[i - 1] * (1 - multiplier)
    return result
```

**But** : valider que numba s'installe, compile, et que les tests passent avant de toucher aux autres fonctions.

**Vérification** : `uv run pytest tests/test_indicators.py` → OK.

### 2b. Numba sur les autres indicateurs

Même pattern `@njit(cache=True)` pour :

- **`rsi()`** : extraire `_rsi_wilder_loop(gains, losses, result, period, avg_gain, avg_loss)`
- **`atr()`** : extraire `_wilder_smooth(data, result, period, seed_val, start_idx)` (réutilisable)
- **`adx()`** : extraire `_adx_wilder_loop(tr, plus_dm, minus_dm, period, n)` — remplace `dx_values.append()` par array pré-alloué + compteur (numba ne gère pas les listes dynamiques efficacement)
- **`supertrend()`** : extraire `_supertrend_loop(highs, lows, closes, atr_arr, multiplier, st_values, direction, first_valid)`

**Design** : les wrappers publics gardent exactement la même signature. Les callers (indicator_cache.py, strategies, etc.) ne changent pas.

Tous les `@njit` utilisent `cache=True` → compilation LLVM sauvée sur disque (~1-2s au premier appel, instantané ensuite).

**Vérification** : `uv run pytest` → 902 tests OK. Les parity tests (`assert_allclose rtol=1e-10`) valident la parité numérique.

---

## Phase 3 — Numba sur `_simulate_trades()` (incrémental, 1 stratégie à la fois)

**Problème** : la boucle accède à des dicts Python (`cache.rsi[period]`) et branche sur des strings (`strategy_name`). Numba ne gère ni l'un ni l'autre.

**Scope** : les 5 stratégies **mono-position** (vwap_rsi, momentum, bollinger_mr, donchian, supertrend) passent par `_simulate_trades()`. Les stratégies **grid/DCA** (envelope_dca, envelope_dca_short, grid_atr, grid_multi_tf) passent par `_simulate_grid_common()` dans fast_multi_backtest.py — c'est un chemin séparé, gardé pour un sprint ultérieur (décision utilisateur). Le speedup Phase 3 s'applique donc uniquement au WFO des stratégies mono. Le WFO grid bénéficie quand même des Phases 1-2 (indicateurs plus rapides dans `build_cache()`).

**Architecture retenue** : **1 fonction `@njit` par stratégie** (pas un monolithe).

Le dispatch reste en Python (1 seul appel par backtest = overhead négligeable). Chaque stratégie a sa propre fonction compilée → plus facile à débugger, tester, et étendre.

### 3a. Une seule stratégie d'abord (vwap_rsi = la plus simple)

1. **Wrapper Python** `_simulate_trades()` (même signature qu'avant) :
   - Pré-extrait les arrays du cache dans des variables locales
   - Pré-extrait les params TP/SL en floats scalaires
   - Dispatche vers `_simulate_vwap_rsi_numba(...)` si numba dispo, sinon code Python original

2. **Fonction `@njit`** dédiée :

```python
@njit(cache=True)
def _simulate_vwap_rsi_numba(
    longs, shorts,
    opens, highs, lows, closes, regime,
    rsi_arr,
    tp_pct, sl_pct,
    initial_capital, taker_fee, maker_fee,
    slippage_pct, high_vol_slippage_mult,
    max_risk_per_trade,
):
    # ... boucle avec check_exit vwap_rsi inline
    return trade_pnls, trade_returns, n_trades, final_capital
```

**Retourne des arrays pré-alloués** (pas de listes Python) — le wrapper slice `[:n_trades]` et convertit en list.

3. **Conserver les fonctions Python existantes** (`_check_tp_sl`, `_open_trade`, `_close_trade`, `_check_exit`) — pas de suppression, utilisées en fallback et par les tests unitaires.

4. **Fallback** :

```python
def _simulate_trades(...):
    if NUMBA_AVAILABLE and strategy_name == "vwap_rsi":
        return _run_simulate_vwap_rsi_numba(...)
    # ... code Python original inchangé pour les autres
```

**Vérification** : `test_fast_vs_normal_parity_vwap_rsi` (fast_backtest.py:632) — doit passer avec numba ET sans.

### 3b. Étendre aux 4 autres stratégies

Même pattern pour :
- `_simulate_momentum_numba(...)` — params: atr_mult_tp, atr_mult_sl, tp_pct, sl_pct + adx_arr pour check_exit
- `_simulate_bollinger_numba(...)` — params: sl_pct + bb_sma_arr pour check_exit (SMA crossing)
- `_simulate_donchian_numba(...)` — params: atr_tp_multiple, atr_sl_multiple (pas de check_exit)
- `_simulate_supertrend_numba(...)` — params: tp_pct, sl_pct (pas de check_exit)

Le wrapper dispatche sur `strategy_name` vers la bonne fonction numba.

**Avantages vs monolithe** :
- Chaque fonction est ~80-120 lignes (pas 200+)
- Compilation rapide (~1-2s chacune, pas 5-15s pour un monolithe)
- Ajouter la 6e stratégie = ajouter 1 fichier, pas toucher un monolithe
- Tests de parité indépendants par stratégie
- Erreurs de typage numba localisées

**Code commun factorisé** : les helpers TP/SL check et close_trade sont identiques entre stratégies → extraire en `@njit` partagés :

```python
@njit(cache=True)
def _check_tp_sl_numba(high, low, direction, tp, sl):
    ...

@njit(cache=True)
def _close_trade_numba(direction, entry_price, exit_price, quantity, entry_fee, ...):
    ...
```

**Vérification Phase 3** : tests de parité existants :
- `test_fast_vs_normal_parity_vwap_rsi` (fast_backtest.py:632)
- `test_fast_vs_normal_parity_momentum` (fast_backtest.py:675)
- `test_fast_engine_speedup` (fast_backtest.py:717)
- `test_fast_vs_normal_parity` (grid_multi_tf.py:584)

---

## Benchmark

Ajouter un script `scripts/benchmark_fast_engine.py` :

```python
# 3 runs, exclut le premier (compilation numba), reporte mean ± std
# Stratégie : vwap_rsi sur données synthétiques (5000 candles, 200 combos)
# Compare : temps total Phase 1 seul vs Phase 1+2 vs Phase 1+2+3
# Machine : locale (mentionnée dans le rapport)
```

---

## Ordre d'exécution

```
Phase 0 (pandas)        →  5 min    →  tests
Phase 1 (numpy pur)     →  30 min   →  tests
Phase 2a (numba + ema)  →  30 min   →  tests   ← valide le setup
Phase 2b (autres indic) →  1h       →  tests
Phase 3a (vwap_rsi)     →  1h30     →  tests   ← valide l'approche
Phase 3b (4 stratégies) →  2h       →  tests
Benchmark               →  30 min
```

**Total estimé : ~6h** (vs 3h05 initial). Chaque phase indépendamment revertible.

## Stratégie de rollback

- **Phase 0** : re-ajouter pandas (trivial)
- **Phase 1** : git revert vectorisations (pur numpy)
- **Phase 2-3** : `NUMBA_AVAILABLE = False` → retour automatique au Python pur. Supprimer numba de pyproject.toml suffit.
- **Rollback nucléaire** : supprimer numba, garder Phase 0+1 (numpy vectorisé = free lunch sans risque)

## Vérification finale

1. `uv run pytest` → 902+ tests OK
2. `scripts/benchmark_fast_engine.py` → mesurer speedup réel (3 runs, exclure compilation)
3. Vérifier que le serveur Docker (sans numba) démarre normalement grâce au fallback
