# Sprint 21a — Stratégie Grid Multi-TF (Backtest + WFO)

## Contexte

Grid ATR est une stratégie mean-reversion LONG-only qui achète les dips sous la SMA. Son défaut principal : elle entre LONG même en bear market (-46% drawdown en paper trading). Grid Multi-TF corrige ce problème en ajoutant un **filtre directionnel Supertrend 4h** — on ne trade que dans le sens du trend.

**Scope sprint 21a** : stratégie + fast engine + WFO uniquement. Le support live (Simulator/GridStrategyRunner + TimeFrame.H4) est reporté au sprint 21b.

---

## Architecture

```
Candles 1h ──→ Resampling 4h (UTC-aligned) ──→ Supertrend ──→ Direction (UP/DOWN)
                                                                    │
Candles 1h ──→ SMA + ATR ──→ Niveaux grid ATR ──→ Exécution filtrée par direction
```

- ST 4h = UP → LONG (enveloppes sous SMA, comme grid_atr)
- ST 4h = DOWN → SHORT (enveloppes au-dessus SMA)
- Flip de direction → force-close toutes les positions

---

## Étape 1 — Config

**Fichier : `backend/core/config.py`** (après `GridATRConfig` L247)

Ajouter `GridMultiTFConfig` — même structure que `GridATRConfig` + params Supertrend :

```python
class GridMultiTFConfig(BaseModel):
    enabled: bool = False
    live_eligible: bool = False
    timeframe: str = "1h"
    # Filtre trend (4h)
    st_atr_period: int = Field(default=10, ge=2, le=50)
    st_atr_multiplier: float = Field(default=3.0, gt=0)
    # Exécution grid (1h) — mêmes params que grid_atr
    ma_period: int = Field(default=14, ge=2, le=50)
    atr_period: int = Field(default=14, ge=2, le=50)
    atr_multiplier_start: float = Field(default=2.0, gt=0)
    atr_multiplier_step: float = Field(default=1.0, gt=0)
    num_levels: int = Field(default=3, ge=1, le=6)
    sl_percent: float = Field(default=20.0, gt=0)
    sides: list[str] = Field(default=["long", "short"])  # whitelist direction
    leverage: int = Field(default=6, ge=1, le=20)
    weight: float = Field(default=0.20, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}
```

Dans `StrategiesConfig` (L266) : ajouter `grid_multi_tf: GridMultiTFConfig = Field(default_factory=GridMultiTFConfig)`.
Dans `validate_weights` : ajouter `self.grid_multi_tf` dans la liste.

---

## Étape 2 — Stratégie

**Fichier : `backend/strategies/grid_multi_tf.py`** — NOUVEAU (~130 lignes)

Calqué sur `grid_atr.py`. Hérite `BaseGridStrategy`.

Différences clés vs grid_atr :
1. `min_candles` retourne `{"1h": max(ma_period, atr_period) + 20}` (pas de 4h — le cache fait le resampling)
2. `compute_grid()` lit la direction Supertrend depuis `ctx.indicators.get("4h", {}).get("st_direction")` :
   - `st_direction == 1` → `allowed = "long"`, enveloppes sous SMA
   - `st_direction == -1` → `allowed = "short"`, enveloppes au-dessus SMA
   - `st_direction is None/NaN` → retourne `[]` (pas de trading)
   - Vérifie `allowed in self._config.sides` (whitelist)
3. `should_close_all()` : en plus de TP/SL classique (identique grid_atr), si la direction Supertrend a changé depuis l'ouverture des positions → retourne `"direction_flip"`
4. `get_params()` inclut `st_atr_period` et `st_atr_multiplier`

**Réutiliser** : `backend/core/indicators.sma`, `backend/core/indicators.atr`, `backend/core/indicators.supertrend`

**Signature confirmée** (L339 indicators.py) : `supertrend(highs, lows, closes, atr_arr, multiplier)` — prend un `atr_arr` **pré-calculé** (pas un `atr_period`). Pattern identique à `indicator_cache.py` L243. Retourne `(supertrend_values, direction)` — on utilise `direction` (2ème élément).

---

## Étape 3 — Factory + Registry

**Fichier : `backend/strategies/factory.py`**
- Import `GridMultiTFStrategy` + `GridMultiTFConfig`
- Ajouter dans `mapping` de `create_strategy` et dans `get_enabled_strategies`

**Fichier : `backend/optimization/__init__.py`**
- Import + ajouter dans `STRATEGY_REGISTRY`, `GRID_STRATEGIES`
- `FAST_ENGINE_STRATEGIES` se met à jour automatiquement (dérivé du registry)

---

## Étape 4 — Indicator Cache (resampling 4h)

**Fichier : `backend/optimization/indicator_cache.py`**

### 4a. Nouveau champ IndicatorCache (L87)

```python
# Grid Multi-TF : Supertrend 4h mappé sur indices 1h
supertrend_dir_4h: dict[tuple[int, float], np.ndarray]  # {(st_atr_period, st_mult): dir_1h}
```

### 4b. Helper de resampling `_resample_1h_to_4h()`

```python
def _resample_1h_to_4h(
    main_candles: list[Candle],
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample 1h → 4h aligné aux frontières UTC (00h, 04h, 08h, 12h, 16h, 20h).

    Returns:
        (highs_4h, lows_4h, closes_4h, mapping_1h_to_4h)
        mapping[i] = index du dernier 4h COMPLÉTÉ avant candle 1h[i], ou -1
    """
```

**Logique** :
1. Calculer `bucket = timestamp_epoch // 14400` pour chaque candle 1h
2. Grouper par bucket : high=max, low=min, close=dernier close du bucket
3. Ne garder que les buckets **complets** (4 candles)
4. Mapping anti-lookahead : `mapping[i] = index du bucket PRÉCÉDENT le bucket de candle[i]`
   - Si `candle[i]` est dans le bucket `b`, son dernier 4h complété est le bucket `b-1`
   - Les candles dans le premier bucket → mapping = -1 (pas de direction)

### 4c. Branche `grid_multi_tf` dans `build_cache()` (L191-274)

```python
if strategy_name == "grid_multi_tf":
    # SMA + ATR multi-period (identique à grid_atr)
    ma_periods_atr = set(param_grid_values.get("ma_period", [14]))
    for period in ma_periods_atr:
        if period not in bb_sma_dict:
            bb_sma_dict[period] = sma(closes, period)

    atr_periods = set(param_grid_values.get("atr_period", [14]))
    for p in atr_periods:
        if p not in atr_by_period_dict:
            atr_by_period_dict[p] = atr(highs, lows, closes, p)

    # Resampling 1h → 4h
    h4_highs, h4_lows, h4_closes, mapping = _resample_1h_to_4h(
        main_candles, closes, highs, lows,
    )

    # Supertrend 4h pour chaque combo (st_atr_period, st_atr_multiplier)
    st_dir_4h_dict = {}
    for st_period in param_grid_values.get("st_atr_period", [10]):
        atr_4h = atr(h4_highs, h4_lows, h4_closes, st_period)
        for st_mult in param_grid_values.get("st_atr_multiplier", [3.0]):
            _, st_dir = supertrend(h4_highs, h4_lows, h4_closes, atr_4h, st_mult)
            # Mapper sur les indices 1h via le mapping anti-lookahead
            st_dir_1h = np.full(n, np.nan)
            for i in range(n):
                idx_4h = mapping[i]
                if idx_4h >= 0 and not np.isnan(st_dir[idx_4h]):
                    st_dir_1h[i] = st_dir[idx_4h]
            st_dir_4h_dict[(st_period, st_mult)] = st_dir_1h
```

### 4d. Mettre à jour le constructeur `IndicatorCache()`

Ajouter `supertrend_dir_4h=st_dir_4h_dict` (ou `{}` pour les autres stratégies).

---

## Étape 5 — Fast Engine

**Fichier : `backend/optimization/fast_multi_backtest.py`**

### 5a. `_build_entry_prices()` — nouvelle branche (après L76)

```python
elif strategy_name == "grid_multi_tf":
    atr_arr = cache.atr_by_period[params["atr_period"]]
    st_key = (params["st_atr_period"], params["st_atr_multiplier"])
    st_dir = cache.supertrend_dir_4h[st_key]

    multipliers = [
        params["atr_multiplier_start"] + lvl * params["atr_multiplier_step"]
        for lvl in range(num_levels)
    ]
    long_mask = st_dir == 1
    short_mask = st_dir == -1
    for lvl in range(num_levels):
        entry_prices[long_mask, lvl] = sma_arr[long_mask] - atr_arr[long_mask] * multipliers[lvl]
        entry_prices[short_mask, lvl] = sma_arr[short_mask] + atr_arr[short_mask] * multipliers[lvl]
    # NaN propagation
    invalid = np.isnan(atr_arr) | (atr_arr <= 0) | np.isnan(st_dir)
    entry_prices[invalid, :] = np.nan
```

Note : le paramètre `direction` (scalar) est ignoré pour grid_multi_tf — la direction vient de `st_dir`.

### 5b. `_simulate_grid_common()` — ajout directions dynamiques (L86)

Nouveau paramètre : `directions: np.ndarray | None = None`

Si `directions` est fourni :
```python
# Au début de la boucle for i in range(n):
if directions is not None:
    cur_dir = directions[i]
    if np.isnan(cur_dir):
        continue  # Pas de direction → skip
    cur_dir_int = int(cur_dir)
    if positions and cur_dir_int != last_dir:
        # Direction flip → force-close toutes les positions
        pnl = _calc_grid_pnl(positions, cache.closes[i], taker_fee, slippage_pct, last_dir)
        trade_pnls.append(pnl)
        if capital > 0:
            trade_returns.append(pnl / capital)
        capital += pnl
        positions = []
    last_dir = cur_dir_int
    direction = cur_dir_int  # Override le scalar pour TP/SL et entry
```

Les callers existants (`envelope_dca`, `grid_atr`) passent `directions=None` → aucun changement de comportement.

### 5c. Wrapper `_simulate_grid_multi_tf()` (L280-297)

```python
def _simulate_grid_multi_tf(
    cache: IndicatorCache,
    params: dict[str, Any],
    bt_config: BacktestConfig,
) -> tuple[list[float], list[float], float]:
    num_levels = params["num_levels"]
    sl_pct = params["sl_percent"] / 100
    sma_arr = cache.bb_sma[params["ma_period"]]
    st_key = (params["st_atr_period"], params["st_atr_multiplier"])
    directions = cache.supertrend_dir_4h[st_key]
    entry_prices = _build_entry_prices("grid_multi_tf", cache, params, num_levels, direction=1)
    return _simulate_grid_common(
        entry_prices, sma_arr, cache, bt_config, num_levels, sl_pct,
        direction=1,  # dummy initial — overridden par directions[i] dans la boucle
        directions=directions,
    )
```

**Clarification sur `direction=1` (dummy)** : Ce scalar n'est jamais utilisé quand `directions` est fourni. Dans la boucle `_simulate_grid_common`, dès qu'on entre `if directions is not None:`, le `direction` local est immédiatement écrasé par `direction = cur_dir_int` (soit 1=LONG soit -1=SHORT selon le Supertrend). Le scalar `direction=1` sert uniquement de placeholder pour la signature existante. L'array `entry_prices` contient déjà les bons prix (LONG ou SHORT) selon la direction à chaque candle.

### 5d. `run_multi_backtest_from_cache()` (L249)

Ajouter :
```python
elif strategy_name == "grid_multi_tf":
    trade_pnls, trade_returns, final_capital = _simulate_grid_multi_tf(
        cache, params, bt_config,
    )
```

---

## Étape 6 — WFO config

**Fichier : `backend/optimization/walk_forward.py`** (L394-405)

Ajouter dans `_INDICATOR_PARAMS` :
```python
"grid_multi_tf": ["ma_period", "atr_period", "st_atr_period", "st_atr_multiplier"],
```

Les 4 paramètres affectent le cache : `ma_period` → SMA, `atr_period` → ATR 1h, `st_atr_period + st_atr_multiplier` → clé Supertrend 4h.

---

## Étape 7 — Configs YAML

**Fichier : `config/strategies.yaml`**

```yaml
grid_multi_tf:
  enabled: false
  timeframe: 1h
  st_atr_period: 10
  st_atr_multiplier: 3.0
  ma_period: 14
  atr_period: 14
  atr_multiplier_start: 2.0
  atr_multiplier_step: 1.0
  num_levels: 3
  sl_percent: 20.0
  sides: ["long", "short"]
  leverage: 6
  per_asset: {}
```

**Fichier : `config/param_grids.yaml`**

```yaml
grid_multi_tf:
  wfo:
    is_days: 180
    oos_days: 60
    step_days: 60
  default:
    st_atr_period: [10, 14]
    st_atr_multiplier: [2.0, 3.0]
    ma_period: [7, 10]
    atr_period: [10, 14]
    atr_multiplier_start: [1.5, 2.0]
    atr_multiplier_step: [0.5, 1.0]
    num_levels: [2, 3]
    sl_percent: [15.0, 20.0, 25.0]
```

= 2×2×2×2×2×2×2×3 = **384 combos** (raisonnable, ~2min/asset).

---

## Étape 8 — Tests

**Fichier : `tests/conftest.py`** — Mettre à jour `make_indicator_cache` (L151)

Ajouter paramètre `supertrend_dir_4h: dict | None = None` → default `{}`.

**Fichier : `tests/test_grid_multi_tf.py`** — NOUVEAU (~350 lignes, ~30 tests)

### Section 1 — Resampling 4h (~5 tests)
1. `test_resample_alignment_utc` — candles groupées par frontières 00h/04h/08h/12h/16h/20h
2. `test_resample_no_lookahead` — mapping[i] pointe vers le 4h PRÉCÉDENT
3. `test_resample_incomplete_bucket` — bucket < 4 candles exclu
4. `test_resample_ohlc_correct` — high=max, low=min, close=dernier
5. `test_resample_first_period_nan` — premier bucket → mapping = -1

### Section 2 — Signaux stratégie (~8 tests)
6. `test_compute_grid_long_when_st_up` — ST=1 → niveaux LONG sous SMA
7. `test_compute_grid_short_when_st_down` — ST=-1 → niveaux SHORT au-dessus SMA
8. `test_compute_grid_empty_when_no_st` — pas de ST → `[]`
9. `test_sides_whitelist_filters` — `sides: ["long"]` + ST=DOWN → `[]`
10. `test_entry_prices_atr_based` — entry = SMA ± ATR × multiplier
11. `test_num_levels_respected` — N levels → N GridLevel retournés
12. `test_nan_atr_returns_empty` — ATR NaN/≤0 → `[]`
13. `test_should_close_all_direction_flip` — positions LONG + ST flip → `"direction_flip"`

### Section 3 — TP/SL (~3 tests)
14. `test_tp_at_sma_long` — LONG : TP quand close ≥ SMA
15. `test_tp_at_sma_short` — SHORT : TP quand close ≤ SMA
16. `test_sl_percent` — SL = avg_entry × (1 ± sl_pct)

### Section 4 — Fast engine (~7 tests)
17. `test_fast_engine_runs_without_crash`
18. `test_fast_engine_respects_st_filter` — pas de LONG quand ST=-1
19. `test_fast_engine_direction_flip_closes` — positions fermées au flip
20. `test_fast_engine_nan_direction_skipped` — candles sans direction = pas de trade
21. `test_fast_engine_force_close_end` — force-close à la fin des données
22. `test_fast_engine_pnl_positive_trend` — données synthétiques trending = P&L > 0
23. `test_fast_vs_normal_parity` — parité fast engine vs MultiPositionEngine (n_trades ±1, net_return ±2%)

### Section 5 — Registry + intégration (~6 tests)
24. `test_in_strategy_registry`
25. `test_in_grid_strategies`
26. `test_in_fast_engine_strategies`
27. `test_create_with_params`
28. `test_indicator_params_4_keys`
29. `test_config_defaults`

### Section 6 — Cache build (~3 tests)
30. `test_build_cache_creates_supertrend_dir_4h`
31. `test_build_cache_grid_atr_unchanged` — regression : grid_atr cache inchangé
32. `test_cache_multiple_st_combos` — 2 periods × 2 mults = 4 clés

---

## Ce qu'on NE touche PAS

- `grid_atr.py` — inchangé
- `envelope_dca.py`, `envelope_dca_short.py` — inchangés
- `_simulate_grid_common()` callers existants — `directions=None` par défaut
- `backend/core/models.py` — pas de TimeFrame.H4 (sprint 21b)
- `backend/backtesting/simulator.py` — pas de support live (sprint 21b)
- `backend/core/incremental_indicators.py` — pas modifié (sprint 21b)
- `backend/core/data_engine.py` — pas modifié (sprint 21b)
- Les 852 tests existants doivent tous passer

---

## Fichiers modifiés (résumé)

| Fichier | Action | ~Lignes |
|---------|--------|---------|
| `backend/core/config.py` | MODIFIER — `GridMultiTFConfig` + `StrategiesConfig` | +25 |
| `backend/strategies/grid_multi_tf.py` | NOUVEAU — stratégie | ~130 |
| `backend/strategies/factory.py` | MODIFIER — mapping + enabled | +5 |
| `backend/optimization/__init__.py` | MODIFIER — registry + GRID_STRATEGIES | +4 |
| `backend/optimization/indicator_cache.py` | MODIFIER — champ + resampling 4h + build_cache | +60 |
| `backend/optimization/fast_multi_backtest.py` | MODIFIER — entry_prices + directions + wrapper | +50 |
| `backend/optimization/walk_forward.py` | MODIFIER — `_INDICATOR_PARAMS` | +1 |
| `config/strategies.yaml` | MODIFIER — section grid_multi_tf | +12 |
| `config/param_grids.yaml` | MODIFIER — grille WFO 384 combos | +15 |
| `tests/conftest.py` | MODIFIER — `make_indicator_cache` + field | +3 |
| `tests/test_grid_multi_tf.py` | NOUVEAU — ~31 tests | ~350 |
| **Total** | | **~655** |

---

## Vérification

```bash
# 1. Tests du nouveau module
uv run python -m pytest tests/test_grid_multi_tf.py -v

# 2. Régression : tous les tests existants passent
uv run python -m pytest --tb=short -q

# 3. Régression spécifique grid_atr (parité)
uv run python -m pytest tests/test_grid_atr.py -v

# 4. Régression fast engine refactor
uv run python -m pytest tests/test_fast_engine_refactor.py -v

# 5. Test WFO rapide (si données BTC disponibles)
uv run python -m scripts.optimize --strategy grid_multi_tf --symbol BTC/USDT --exchange binance -v
```
