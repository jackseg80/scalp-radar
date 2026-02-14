# Sprint 15 — Stratégie Envelope DCA SHORT

## Contexte

La stratégie `envelope_dca` (LONG) est la seule stratégie active en production. Elle ouvre des positions LONG DCA quand le prix descend sous des enveloppes SMA. On veut son miroir SHORT : ouvrir des shorts DCA quand le prix monte au-dessus des enveloppes SMA.

## Décision : Option B (stratégie séparée)

**Pourquoi pas l'Option A (paramétrer l'existant) :**
La classe `EnvelopeDCAStrategy` supporte **déjà** la direction SHORT via le champ `sides` dans la config. Cependant :
1. Le **fast engine** (`fast_multi_backtest.py`) est codé en dur pour LONG uniquement (enveloppes basses, direction=1)
2. Le **WFO optimise par nom de stratégie** — on ne peut pas avoir deux grilles de paramètres pour le même nom
3. Le dashboard, l'executor et l'adaptive selector identifient les stratégies par nom
4. Risque de régression : toucher à la logique LONG qui tourne en prod

**Approche retenue : sous-classe minimale + réutilisation maximale**

La classe `EnvelopeDCAStrategy` gère déjà les deux directions correctement dans `compute_grid()` et `should_close_all()`. On crée une sous-classe triviale qui ne change que le `name` et le `sides` par défaut :

```python
class EnvelopeDCAShortStrategy(EnvelopeDCAStrategy):
    name = "envelope_dca_short"
```

La config `EnvelopeDCAShortConfig` a `sides: ["short"]` par défaut. Zéro duplication de logique.

---

## Fichiers à modifier/créer

### 1. `backend/strategies/envelope_dca_short.py` (NOUVEAU)

Sous-classe minimale (~10 lignes) :
```python
class EnvelopeDCAShortStrategy(EnvelopeDCAStrategy):
    name = "envelope_dca_short"
    def __init__(self, config: EnvelopeDCAShortConfig) -> None:
        self._config = config
```

### 2. `backend/core/config.py`

- Ajouter `EnvelopeDCAShortConfig` (copie de `EnvelopeDCAConfig` avec `sides: ["short"]` par défaut)
- Ajouter `envelope_dca_short: EnvelopeDCAShortConfig` dans `StrategiesConfig`
- Ajouter dans le validateur `validate_weights`

### 3. `config/strategies.yaml`

Nouvelle entrée :
```yaml
envelope_dca_short:
  enabled: false          # Validation WFO d'abord
  live_eligible: false
  timeframe: "1h"
  ma_period: 7
  num_levels: 2
  envelope_start: 0.05
  envelope_step: 0.02
  sl_percent: 20.0
  sides: ["short"]
  leverage: 6
  weight: 0.20
  per_asset: {}
```

### 4. `config/param_grids.yaml`

Nouvelle entrée (grille identique à LONG pour commencer) :
```yaml
envelope_dca_short:
  wfo:
    is_days: 180
    oos_days: 60
    step_days: 60
  default:
    ma_period: [5, 7, 10]
    num_levels: [2, 3, 4]
    envelope_start: [0.05, 0.07, 0.10]
    envelope_step: [0.02, 0.03, 0.05]
    sl_percent: [15.0, 20.0, 25.0, 30.0]
```

### 5. `backend/optimization/__init__.py`

- Import `EnvelopeDCAShortConfig` et `EnvelopeDCAShortStrategy`
- Ajouter dans `STRATEGY_REGISTRY` : `"envelope_dca_short": (EnvelopeDCAShortConfig, EnvelopeDCAShortStrategy)`
- Ajouter dans `GRID_STRATEGIES` : `{"envelope_dca", "envelope_dca_short"}`

### 6. `backend/strategies/factory.py`

- Import `EnvelopeDCAShortStrategy`
- Ajouter dans `create_strategy()` mapping
- Ajouter dans `get_enabled_strategies()` le bloc if

### 7. `backend/optimization/fast_multi_backtest.py`

Changement principal — ajouter le support direction SHORT :
- Ajouter un paramètre `direction: int = 1` à `_simulate_envelope_dca()`
- Calculer les offsets d'enveloppe selon la direction :
  - LONG : `sma × (1 - offset)`, trigger `low <= entry_price`
  - SHORT : `sma × (1 + upper_offset)`, trigger `high >= entry_price`
    - `upper_offset = round(1/(1-lower_offset) - 1, 3)` (asymétrie, comme dans la classe)
- Inverser SL/TP checks :
  - LONG : `sl_hit = low <= sl_price`, `tp_hit = high >= tp_price`
  - SHORT : `sl_hit = high >= sl_price`, `tp_hit = low <= tp_price`
- Inverser SL price : LONG `avg*(1-sl_pct)`, SHORT `avg*(1+sl_pct)`
- Inverser OHLC heuristic pour SHORT
- Passer `direction` à `_calc_grid_pnl()` (déjà paramétré)
- Dispatch dans `run_multi_backtest_from_cache()` :
  ```python
  elif strategy_name == "envelope_dca_short":
      ..._simulate_envelope_dca(cache, params, bt_config, direction=-1)
  ```

### 8. `backend/optimization/indicator_cache.py`

Étendre la condition de build du cache SMA :
```python
if strategy_name in ("envelope_dca", "envelope_dca_short"):
```
(même logique, seule la SMA est pré-calculée, les enveloppes sont à la volée)

### 9. `backend/optimization/walk_forward.py`

3 modifications :
- `_INDICATOR_PARAMS` : ajouter `"envelope_dca_short": ["ma_period"]`
- Ligne ~500 `collect_combo_results` : ajouter `"envelope_dca_short"` dans la whitelist
- Ligne ~801 fast engine whitelist : ajouter `"envelope_dca_short"`

### 10. `backend/execution/adaptive_selector.py`

Ajouter dans le mapping :
```python
"envelope_dca_short": "envelope_dca_short",
```

### 11. `tests/test_envelope_dca_short.py` (NOUVEAU)

Tests unitaires (~15-20 tests) :
- **Signal generation** : vérifier que `compute_grid()` retourne des niveaux SHORT au-dessus de la SMA
- **Enveloppes asymétriques** : `upper_offset = round(1/(1-lower) - 1, 3)` produit les bons prix
- **TP global** : close <= SMA déclenche `should_close_all() == "tp_global"`
- **SL global** : close >= avg_entry × (1 + sl_pct) déclenche `should_close_all() == "sl_global"`
- **Direction lock** : si SHORT ouvert, pas de LONG proposé
- **Fast engine SHORT** : `_simulate_envelope_dca(direction=-1)` produit des trades cohérents
- **Registry** : `envelope_dca_short` dans STRATEGY_REGISTRY et GRID_STRATEGIES
- **Config** : `EnvelopeDCAShortConfig` a `sides: ["short"]` par défaut
- **Params** : `get_params()` retourne les bons champs
- **WFO intégration** : `create_strategy_with_params("envelope_dca_short", {...})` fonctionne

---

## Points d'attention / Risques

| Risque | Mitigation |
|--------|-----------|
| Régression envelope_dca LONG | Zéro modification de la classe LONG. Le fast engine ajoute un paramètre `direction` avec défaut `1` (backward compatible) |
| Fast engine SHORT incorrect | Tests dédiés comparant fast engine vs BacktestEngine (MultiPositionEngine) sur données synthétiques |
| WFO ne reconnaît pas la stratégie | Whitelist explicite dans 3 endroits du walk_forward.py |
| Grille de paramètres sous-optimale | Grille identique à LONG pour commencer, ajustable après premiers résultats WFO |
| `enabled: false` en prod | Intentionnel — validation WFO d'abord |

## Vérification

1. `pytest tests/` — les 597+ tests existants passent (zéro régression)
2. `pytest tests/test_envelope_dca_short.py` — nouveaux tests passent
3. Vérifier manuellement dans le CLI WFO : `python scripts/optimize.py envelope_dca_short BTC/USDT`
4. Vérifier dans l'Explorer frontend que `envelope_dca_short` apparaît dans la liste des stratégies

## Estimation effort

~2-3h de travail. La réutilisation de la classe existante réduit considérablement la complexité.

---

## RÉSULTATS (Sprint Complété)

**Tests** : 613 passants (+16 depuis Sprint 14b)
- 22 nouveaux tests dans `test_envelope_dca_short.py`
- 0 régression sur les 603 tests existants

**Implémentation finale** :
- `EnvelopeDCAShortStrategy` : 26 lignes (sous-classe minimale)
- Fast engine : paramètre `direction` backward compatible (défaut=1)
- OHLC heuristic inversée : bougie rouge (close < open) favorable pour SHORT → TP
- Enveloppes asymétriques : `upper_offset = round(1/(1-lower_offset) - 1, 3)`
- Frontend dynamique : endpoint `/api/optimization/strategies` (plus de hardcoding)

**Bonus** :
- Les stratégies futures apparaîtront automatiquement dans l'Explorer (découverte API dynamique)

**Prochaine étape** :
- Sprint 16 : Lancer WFO envelope_dca_short via l'Explorateur dashboard
- Comparer résultats LONG vs SHORT dans la page Recherche
- Décision activation si Grade >= C
