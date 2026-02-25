# Sprint 47 — Grid ATR Adaptatif (v2)

## Contexte

Grid_atr en production souffre en basse volatilité : l'ATR s'écrase → grilles microscopiques → les cycles TP ne couvrent plus les fees. Exemple réel du 25/02 : 7 assets ouverts simultanément, tous fermés en perte (-2.32$) car profit brut < fees.

Ajout de 2 paramètres adaptatifs à `grid_atr` uniquement. Default 0.0 = désactivé → backward compatible, WFO peut A/B tester contre la version classique.

Scope : **grid_atr seulement** — grid_multi_tf a le filtre Supertrend 4h qui réduit déjà l'exposition en basse vol. On valide d'abord sur une stratégie.

---

## Fichiers à modifier (4 fichiers prod + tests)

### 1. [config.py:237-252](backend/core/config.py#L237-L252) — GridATRConfig

Ajouter 2 champs Pydantic après `cooldown_candles` (L251), avant `per_asset` (L252) :

```python
min_grid_spacing_pct: float = Field(default=0.0, ge=0, le=10.0)
min_profit_pct: float = Field(default=0.0, ge=0, le=10.0)
```

### 2. [grid_atr.py](backend/strategies/grid_atr.py) — GridATRStrategy

#### 2a. `compute_grid()` (L79-125) — plancher ATR

Après la validation L83 et avant le for L98, calculer `effective_atr` :

```python
# Plancher ATR adaptatif
close_val = indicators.get("close", float("nan"))
min_spacing = self._config.min_grid_spacing_pct
if min_spacing > 0 and close_val > 0:
    effective_atr = max(atr_val, close_val * min_spacing / 100)
else:
    effective_atr = atr_val
```

Remplacer `atr_val` par `effective_atr` dans les calculs :
- L107 : `entry_price = sma_val - effective_atr * multiplier`
- L117 : `entry_price = sma_val + effective_atr * multiplier`

#### 2b. `should_close_all()` (L143-147) — profit minimum au TP

Remplacer le bloc TP (L143-147) par :

```python
# TP : retour à la SMA + contrainte profit minimum
min_profit = self._config.min_profit_pct
if direction == Direction.LONG and close >= sma_val:
    if min_profit <= 0 or close >= grid_state.avg_entry_price * (1 + min_profit / 100):
        return "tp_global"
if direction == Direction.SHORT and close <= sma_val:
    if min_profit <= 0 or close <= grid_state.avg_entry_price * (1 - min_profit / 100):
        return "tp_global"
```

Logique : TP fire seulement si DEUX conditions remplies :
1. Prix revenu à la SMA (classique)
2. Profit ≥ seuil min (nouveau, skip si 0.0)

SHORT : le profit vient quand le prix baisse → `close <= avg_entry * (1 - pct)`.

#### 2c. `get_params()` (L196-206) — exposer les nouveaux params

Ajouter dans le dict :
```python
"min_grid_spacing_pct": self._config.min_grid_spacing_pct,
"min_profit_pct": self._config.min_profit_pct,
```

### 3. [fast_multi_backtest.py](backend/optimization/fast_multi_backtest.py) — Fast WFO Engine

#### 3a. `_build_entry_prices()` — branche grid_atr (L62-76)

Après `atr_arr` (L64), ajouter le plancher :

```python
min_spacing = params.get("min_grid_spacing_pct", 0.0)
if min_spacing > 0:
    effective_atr = np.maximum(atr_arr, cache.closes * min_spacing / 100)
else:
    effective_atr = atr_arr
```

Remplacer `atr_arr` par `effective_atr` dans les calculs L71-73.
Le mask `invalid` (L75) reste sur `atr_arr` brut — si ATR raw est NaN, rien à faire même avec plancher.

#### 3b. `_simulate_grid_common()` — signature (L131-142)

Ajouter un paramètre :

```python
min_profit_pct: float = 0.0,
```

Backward compatible — tous les autres appelants (grid_boltrend, grid_range, etc.) ne passent pas ce param → default 0.0.

#### 3c. `_simulate_grid_common()` — TP classique (L278-306)

Dans le bloc `# --- MODE TP CLASSIQUE (SMA) ---`, modifier le check `tp_hit` :

```python
tp_price = sma_arr[i]

if direction == 1:
    tp_hit = cache.highs[i] >= tp_price
    if tp_hit and min_profit_pct > 0:
        tp_hit = cache.highs[i] >= avg_entry * (1 + min_profit_pct / 100)
else:
    tp_hit = cache.lows[i] <= tp_price
    if tp_hit and min_profit_pct > 0:
        tp_hit = cache.lows[i] <= avg_entry * (1 - min_profit_pct / 100)
```

Le `tp_price` (prix d'exit) reste `sma_arr[i]` — on modifie seulement la **condition de déclenchement**.

#### 3d. `_simulate_grid_atr()` (L1503-1506) — passer le param

```python
return _simulate_grid_common(
    entry_prices, sma_arr, cache, bt_config, num_levels, sl_pct, direction,
    max_hold_candles=params.get("max_hold_candles", 0),
    min_profit_pct=params.get("min_profit_pct", 0.0),
)
```

### 4. [param_grids.yaml:155-169](config/param_grids.yaml#L155-L169) — grille WFO grid_atr

Remplacer la section `default` de grid_atr par la grille réduite + nouveaux params :

```yaml
grid_atr:
  wfo:
    is_days: 180
    oos_days: 60
    step_days: 60
    embargo_days: 7
  default:
    timeframe: ["1h", "4h", "1d"]
    ma_period: [10, 14, 20]
    atr_period: [10, 14, 20]
    atr_multiplier_start: [1.5, 2.0, 2.5]
    atr_multiplier_step: [0.5, 1.0, 1.5]
    num_levels: [3, 4]
    sl_percent: [15.0, 20.0, 25.0]
    min_grid_spacing_pct: [0.0, 0.8, 1.2, 1.8]
    min_profit_pct: [0.0, 0.2, 0.4]
```

Réductions justifiées :
- `ma_period` : 7 supprimé (trop nerveux)
- `atr_multiplier_start` : 1.0 et 3.0 supprimés (extrêmes rarement sélectionnés)
- `num_levels` : 2 supprimé (jamais sélectionné par WFO)

**Total : 3 × 3 × 3 × 3 × 3 × 2 × 3 × 4 × 3 = 5 832 combos/asset** (~9 min/asset estimé)

Mettre à jour le commentaire L155 : `# Grid ATR : 5832 combos (3×3×3×3×3×2×3×4×3), ~9 min/asset`

---

## Tests

Fichier : [tests/test_grid_atr.py](tests/test_grid_atr.py)

### Tests min_grid_spacing_pct (4 tests)
1. `test_compute_grid_min_spacing_clamps_atr` : ATR=1.0, close=100, min_spacing=2.0% → effective_atr=2.0, vérifier les entry prices
2. `test_compute_grid_min_spacing_no_effect_high_atr` : ATR=5.0 > plancher → pas de clamping
3. `test_compute_grid_min_spacing_zero_disabled` : 0.0 = comportement classique identique
4. `test_compute_grid_min_spacing_short` : direction SHORT avec clamping

### Tests min_profit_pct (5 tests)
5. `test_tp_blocked_by_min_profit` : close ≥ SMA mais profit < seuil → pas de TP
6. `test_tp_allowed_by_min_profit` : close ≥ SMA ET profit ≥ seuil → TP fire
7. `test_tp_min_profit_zero_classic` : 0.0 = TP classique inchangé
8. `test_tp_min_profit_short_blocked` : SHORT, close ≤ SMA mais pas assez de profit
9. `test_tp_min_profit_short_hit` : SHORT, les 2 conditions OK → TP

### Tests intégration (2 tests)
10. `test_sl_ignores_min_profit` : SL fonctionne normalement avec min_profit > 0
11. `test_get_params_includes_adaptive_fields` : vérifie les 2 clés présentes dans get_params()

---

## Cas limites

| Cas | Comportement |
|-----|-------------|
| Params = 0.0 | Désactivé, comportement 100% identique à l'actuel |
| ATR NaN + plancher | `max(NaN, plancher)` = NaN en numpy → pas de trade (correct) |
| close NaN | Plancher non calculé → `effective_atr = atr_val` |
| min_profit trop élevé | TP jamais atteint → SL ou time_stop (WFO pénalise via sharpe négatif) |
| Trailing stop (grid_trend) | Branch séparée dans fast engine → min_profit non appliqué (correct) |
| Autres stratégies grid | `min_profit_pct=0.0` par défaut → aucun impact |

---

## Vérification

1. `PYTHON_JIT=0 uv run pytest tests/test_grid_atr.py -x -q` — nouveaux tests + régression
2. `PYTHON_JIT=0 uv run pytest tests/ -x -q` — suite complète (~1840 tests)
3. Vérifier que `get_params()` retourne les 2 nouvelles clés
