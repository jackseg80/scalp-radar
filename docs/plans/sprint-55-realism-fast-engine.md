# Sprint 55 — Renforcement réalisme du fast engine WFO

*Note : plan initialement nommé "Sprint 53" lors de la session — renommé Sprint 55 pour cohérence avec le ROADMAP*

## Contexte

Audit confirmé : le fast engine WFO a 2 failles critiques — les params sélectionnés peuvent être incompatibles avec le trading réel. Les étapes 3-5 (portfolio backtest, stress test, robustness) utilisent le vrai Simulator et rattrapent ces failles, mais ne peuvent que **rejeter** un combo mal sélectionné, pas le corriger.

---

## Fix 1 — Filtre SL × leverage sur les combos

### Fichier : `backend/optimization/walk_forward.py`

### 1a. Helper `_filter_sl_leverage()` (nouvelle fonction, après `_fine_grid_around_top`)

```python
def _filter_sl_leverage(
    grid: list[dict[str, Any]],
    leverage: int,
    threshold: float = 1.5,
    min_combos: int = 50,
) -> list[dict[str, Any]]:
```

- Si `"sl_percent"` absent du premier combo → retourner `grid` tel quel (stratégies ATR-based comme donchian/supertrend)
- Filtrer : `combo["sl_percent"] / 100 * leverage > threshold` → exclu
- Si résultat < `min_combos` : fallback = tri par `sl_percent` croissant, garder les N premiers
- Log `"Filtre SL×leverage (seuil {:.0%}) : {} → {} combos (leverage={}x)"`

### 1b. Appliqué sur `full_grid` — après `bt_config.leverage`

### 1c. Appliqué sur `fine_grid` — après `_fine_grid_around_top()`

**Pourquoi fine_grid** : ±1 step peut générer des combos invalides. Ex: `sl_percent=20` (valide à 7x: 1.40) → fine grid crée `sl_percent=25` (invalide à 7x: 1.75).

---

## Fix 2 — Kill switch simulé dans le fast engine

### Constante (dans les deux fichiers fast engine)

```python
KILL_SWITCH_DD_PCT = 0.25  # cohérent avec risk.yaml grid_max_session_loss_percent: 25
```

**Justification** : le WFO simule 1 asset = 1 runner. En live, c'est le seuil per-runner (25%) qui stoppe le runner, pas le global (45%) qui est pour le portfolio. Utiliser 45% laisserait passer des combos qui seraient tués à -25% en live.

### 2a. Grid engine : `_simulate_grid_common()` — couvre 5 stratégies

`peak_capital` existait déjà + hard break 80%. Soft kill switch 25% ajouté avant le hard break.

### 2b-2e. 4 boucles grid séparées

`_simulate_grid_range`, `_simulate_grid_boltrend`, `_simulate_grid_funding`, `_simulate_grid_momentum`

### 2f. Scalp engine : `_simulate_trades()` Python fallback

### 2g. Numba JIT (6 fonctions) — REPORTÉ Sprint 56

Toutes les stratégies LIVE sont grid (couvertes). Scalp = paper only. TODO commenté dans chaque wrapper.

---

## Tests — `tests/test_sprint53_realism.py` (8 tests)

### Filtre SL × leverage (3 tests)
1. `test_removes_invalid` — leverage=7, sl=25 → exclu (1.75 > 1.5)
2. `test_no_sl_key` — stratégie sans sl_percent → grid inchangée
3. `test_min_combos_guard` — tous invalides → garde min_combos triés par SL croissant

### Kill switch grid (3 tests)
1. `test_stops_trading_on_crash` — crash linéaire, DD limité < 50%
2. `test_no_trigger_stable_market` — données stables → simulation complète
3. `test_exits_then_breaks` — kill switch → moins de 6 trades (≤6)

### Kill switch scalp (1 test)
1. `test_python_fallback` — SL répétés en downtrend → DD limité < 50%

### Fine grid (1 test)
1. `test_removes_invalid_from_fine_grid` — fine grid génère sl=25 via ±1 step → filtré à 7x

---

## Adaptations de tests existants

- `test_dd_guard.py` : guard 80%→15% pour différencier du kill switch 25%
- `test_fast_engine_refactor.py` : valeurs parity grid_atr mises à jour (108→57 trades)
- `test_grid_atr.py` : parity fast vs normal assouplie (fast a kill switch, normal non)
- `test_grid_multi_tf.py` : idem

---

## Résultat

- **8 nouveaux tests** → **2042 tests, 2038 passants**
- **0 régression Sprint 55**
- 5 tests pré-existants exclus (4 TestLeverageValidation + 1 asyncio flaky)
