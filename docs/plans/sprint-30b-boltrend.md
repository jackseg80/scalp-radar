# Sprint 30b — Stratégie BolTrend (Bollinger Trend Following)

## Contexte

Intégration d'une stratégie live existante (Bollinger Breakout + filtre trend long terme) dans le framework scalp-radar : stratégie event-driven, fast engine WFO, registre, config, tests.

C'est une stratégie **MONO-POSITION** (comme supertrend / bollinger_mr), pas grid/DCA.

**Logique :** Breakout Bollinger (close sort des bandes avec prev_close dedans) filtré par SMA long terme. Sortie dynamique par retour à la SMA de Bollinger (pas de TP fixe). SL % fixe comme filet de sécurité.

---

## Fichiers à modifier/créer (ordre d'implémentation)

### 1. `backend/core/config.py` — BolTrendConfig

Ajouter après SuperTrendConfig (~ligne 183) :

```python
class BolTrendConfig(BaseModel):
    enabled: bool = False
    live_eligible: bool = False
    timeframe: str = "1h"
    bol_window: int = Field(default=100, ge=2)
    bol_std: float = Field(default=2.2, gt=0)
    min_bol_spread: float = Field(default=0.0, ge=0)
    long_ma_window: int = Field(default=550, ge=2)
    sl_percent: float = Field(default=15.0, gt=0)
    leverage: int = Field(default=2, ge=1)
    weight: float = Field(default=0.15, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}
```

Dans `StrategiesConfig` (ligne 360) :
- Ajouter champ `boltrend: BolTrendConfig = Field(default_factory=BolTrendConfig)`
- Ajouter `self.boltrend` dans la liste `validate_weights`

---

### 2. `backend/strategies/boltrend.py` — NOUVEAU FICHIER

Pattern calqué sur `bollinger_mr.py`. Différences clés :
- **evaluate()** : Breakout (pas mean reversion) → `prev_close < prev_upper AND close > upper` (LONG), + spread filter, + trend filter `close > long_ma`
- **check_exit()** : `close < ma_band` (LONG), `close > ma_band` (SHORT) — INVERSÉ vs bollinger_mr
- **compute_indicators()** : BB(bol_window, bol_std) + SMA(long_ma_window), stocker prev values
- **min_candles** : `max(bol_window, long_ma_window) + 20`
- **TP très éloigné** : `entry * 2` (LONG) / `entry * 0.5` (SHORT) — identique à bollinger_mr

Réutiliser : `bollinger_bands()`, `sma()`, `atr()`, `adx()`, `detect_market_regime()` depuis `backend/core/indicators.py`

---

### 3. `backend/strategies/factory.py`

- Import `BolTrendStrategy`
- Ajouter dans `mapping` de `create_strategy()` : `"boltrend": (BolTrendStrategy, strategies_config.boltrend)`
- Ajouter dans `get_enabled_strategies()` : bloc `if strats.boltrend.enabled:`

---

### 4. `backend/optimization/__init__.py`

- Import `BolTrendConfig`, `BolTrendStrategy`
- Ajouter `"boltrend": (BolTrendConfig, BolTrendStrategy)` dans `STRATEGY_REGISTRY`
- NE PAS ajouter dans `GRID_STRATEGIES` (mono-position)
- NE PAS ajouter dans `_NO_FAST_ENGINE` → auto-inclus dans `FAST_ENGINE_STRATEGIES`
- NE PAS ajouter dans `STRATEGIES_NEED_EXTRA_DATA` (pas de funding/OI)

---

### 5. `backend/optimization/indicator_cache.py` — build_cache()

Ajouter un bloc dans `build_cache()` (après le bloc `if strategy_name == "bollinger_mr":`, ~ligne 347) :

```python
if strategy_name == "boltrend":
    bol_windows: set[int] = set()
    bol_stds: set[float] = set()
    long_ma_windows: set[int] = set()
    if "bol_window" in param_grid_values:
        bol_windows.update(param_grid_values["bol_window"])
    if "bol_std" in param_grid_values:
        bol_stds.update(param_grid_values["bol_std"])
    if "long_ma_window" in param_grid_values:
        long_ma_windows.update(param_grid_values["long_ma_window"])
    if not bol_windows: bol_windows.add(100)
    if not bol_stds: bol_stds.add(2.2)
    if not long_ma_windows: long_ma_windows.add(550)

    for period in bol_windows:
        bb_sma_arr, _, _ = bollinger_bands(closes, period, 1.0)
        bb_sma_dict[period] = bb_sma_arr
        for std_dev in bol_stds:
            _, upper, lower = bollinger_bands(closes, period, std_dev)
            bb_upper_dict[(period, std_dev)] = upper
            bb_lower_dict[(period, std_dev)] = lower

    # SMA long terme (réutilise bb_sma_dict)
    for period in long_ma_windows:
        if period not in bb_sma_dict:
            bb_sma_dict[period] = sma(closes, period)
```

Pas de nouveau champ dans le dataclass `IndicatorCache` — on réutilise `bb_sma`, `bb_upper`, `bb_lower`.

---

### 6. `backend/optimization/fast_backtest.py` — Fast engine

**6 points de dispatch modifiés + 3 nouvelles fonctions :**

#### a) `run_backtest_from_cache()` — signal dispatch
```python
elif strategy_name == "boltrend":
    longs, shorts = _boltrend_signals(params, cache)
```

#### b) `_boltrend_signals()` — NOUVELLE FONCTION
Vectorized signal generation avec np.roll pour prev values, valid[0]=False pour éviter le wraparound.

#### c) `_simulate_boltrend_numba()` + `_run_simulate_boltrend()` — NOUVELLES FONCTIONS
Pattern copié de `_simulate_bollinger_numba()`. Le check_exit est INVERSÉ : `close < bb_sma[i]` LONG exit, `close > bb_sma[i]` SHORT exit. TP très éloigné (`tp_dist = entry_price`).

#### d) `_simulate_trades()` — Numba dispatch
```python
if strategy_name == "boltrend":
    return _run_simulate_boltrend(longs, shorts, cache, params, bt_config)
```

#### e) `_check_exit()` — fallback Python
check_exit INVERSÉ vs bollinger_mr (close < sma pour LONG, close > sma pour SHORT).

#### f) `_open_trade()` — TP/SL calculation
```python
elif strategy_name == "boltrend":
    sl_dist = entry_price * params["sl_percent"] / 100
    tp_dist = entry_price  # TP très éloigné
```

---

### 7. `backend/execution/adaptive_selector.py`

Ajouter dans `_STRATEGY_CONFIG_ATTR` : `"boltrend": "boltrend"`

---

### 8. `config/strategies.yaml`

```yaml
boltrend:
  enabled: false
  live_eligible: false
  timeframe: 1h
  bol_window: 100
  bol_std: 2.2
  min_bol_spread: 0.0
  long_ma_window: 550
  sl_percent: 15.0
  leverage: 2
  weight: 0.15
  per_asset: {}
```

---

### 9. `config/param_grids.yaml`

```yaml
boltrend:
  wfo:
    is_days: 180
    oos_days: 60
    step_days: 60
  default:
    bol_window: [50, 100, 150]
    bol_std: [1.5, 2.0, 2.5]
    long_ma_window: [200, 400, 550]
    sl_percent: [10.0, 15.0, 20.0]
    min_bol_spread: [0.0, 0.01, 0.02]
    timeframe: ["1h", "4h"]
```
486 combos par asset (3×3×3×3×3×2).

---

### 10. `tests/test_boltrend.py` — NOUVEAU FICHIER

25 tests couvrant :
- Config (BolTrendConfig defaults, get_params_for_symbol, per_asset override)
- Indicateurs (compute_indicators retourne BB + long_ma + prev values)
- Signaux (LONG breakout, SHORT breakout, trend filter, spread filter, NaN handling)
- Exits (LONG exit close < SMA, SHORT exit close > SMA, pas d'exit si favorable)
- Conditions dashboard (get_current_conditions)
- Fast engine (signals vectorisés, valid[0]=False, backtest 5-tuple)
- Registry + Factory + AdaptiveSelector mapping

---

## Points critiques

1. **check_exit INVERSÉ vs bollinger_mr** : BolTrend LONG sort quand `close < sma` (breakout s'essouffle), bollinger_mr LONG sort quand `close >= sma` (retour à la moyenne)
2. **np.roll wraparound** : position 0 invalide (closes[-1] se retrouve en position 0)
3. **SMA 550 sur 4h** : 550 candles × 4h = 2200h = 91 jours. Fenêtre IS 180 jours → 89 jours utiles (serré mais OK)
4. **TP très éloigné** : même pattern que bollinger_mr (`tp_dist = entry_price`)
5. **Pas dans GRID_STRATEGIES** : mono-position, utilise `run_backtest_from_cache` pas `run_multi_backtest_from_cache`
6. **bb_sma collision safe** : `bollinger_bands()` appelle `sma()` en interne — les deux retournent la même chose
7. **Timeframe groupement OK** : `_run_fast()` groupe par timeframe pour TOUTES les stratégies (pas seulement grid)

---

## Résultat

- **1261 tests**, 0 régression
- 10 fichiers modifiés/créés + 1 test modifié (test_fast_engine_refactor.py)
- 25 nouveaux tests + 1 test existant mis à jour
- 15e stratégie intégrée dans le framework
