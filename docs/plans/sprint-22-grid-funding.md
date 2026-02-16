# Sprint 22 — Grid Funding (DCA sur Funding Rate Negatif)

## Contexte

Grid_atr entre LONG quand le prix est bas (sous SMA). Grid Funding entre LONG quand le funding rate est tres negatif — signal structurel independant du prix. L'edge : les shorts paient les longs tant que le funding est negatif, meme si le prix ne bouge pas. Scope : Strategie + fast engine + WFO. LONG-only. Pas de support live.

## Corrections appliquees (review pre-plan)

1. **Unites funding** : DB stocke en % (`×100`). Le cache loader divise par 100 → raw decimal partout dans le fast engine. La strategie class divise aussi par 100 quand elle lit `ctx.extra_data`.
2. **Anti-lookahead** : pas de decalage +8h. `searchsorted(side='right') - 1` direct = dernier funding settle au moment de la candle.
3. **Decoupler FAST_ENGINE / NEED_EXTRA_DATA** : `_NO_FAST_ENGINE = {"funding", "liquidation"}`, grid_funding dans les deux sets (`STRATEGIES_NEED_EXTRA_DATA` + `FAST_ENGINE_STRATEGIES`).
4. **db_path** a `build_cache()` et `_run_fast()` : parametre optionnel, passe par l'optimizer.
5. **candle_timestamps / funding_rates_1h** : defaut `None` dans IndicatorCache + fixture.
6. **MultiPositionEngine** : ajouter `extra_data_by_timestamp` (manquait, bloque l'OOS evaluation).

---

## Plan d'implementation

### Etape 1 — Config (`backend/core/config.py`)

Ajouter `GridFundingConfig` apres `GridMultiTFConfig` (~25 lignes) :

```python
class GridFundingConfig(BaseModel):
    enabled: bool = False
    live_eligible: bool = False
    timeframe: str = "1h"
    funding_threshold_start: float = Field(default=0.0005, gt=0)  # raw decimal, -0.05%
    funding_threshold_step: float = Field(default=0.0005, gt=0)
    num_levels: int = Field(default=3, ge=1, le=6)
    tp_mode: str = Field(default="funding_or_sma")
    ma_period: int = Field(default=14, ge=2, le=50)
    sl_percent: float = Field(default=15.0, gt=0)
    min_hold_candles: int = Field(default=8, ge=0)
    sides: list[str] = Field(default=["long"])
    leverage: int = Field(default=6, ge=1, le=20)
    weight: float = Field(default=0.20, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}
```

Dans `StrategiesConfig` :
- Ajouter champ `grid_funding: GridFundingConfig = Field(default_factory=GridFundingConfig)`
- Ajouter `self.grid_funding` dans `validate_weights()` liste des strategies

---

### Etape 2 — Strategie (`backend/strategies/grid_funding.py`) — NOUVEAU

~120 lignes. Herite `BaseGridStrategy`. Pattern identique a `grid_atr.py`.

Points cles :
- `compute_grid()` : lit `ctx.extra_data.get("funding_rate", 0)`, **divise par 100** (DB=percent → raw), compare aux seuils raw decimal. Retourne des `GridLevel` avec `entry_price=close` (pas calcule a l'avance).
- `should_close_all()` : TP = funding > 0 OU prix >= SMA (selon `tp_mode`). SL = % classique. Min hold bloque le TP mais pas le SL.
- `compute_indicators()` : calcule SMA sur 1h (pour TP). Pas d'ATR.
- `get_tp_price()` / `get_sl_price()` : SMA / avg_entry ± sl%.
- `name = "grid_funding"`

---

### Etape 3 — Factory + Registry

**`backend/strategies/factory.py`** :
- Import `GridFundingStrategy` + `GridFundingConfig`
- Ajouter `"grid_funding": (GridFundingStrategy, strategies_config.grid_funding)` dans mapping
- Ajouter `if strats.grid_funding.enabled:` dans `get_enabled_strategies()`

**`backend/optimization/__init__.py`** :
- Import `GridFundingConfig`, `GridFundingStrategy`
- Ajouter `"grid_funding": (GridFundingConfig, GridFundingStrategy)` dans `STRATEGY_REGISTRY`
- Ajouter `"grid_funding"` dans `GRID_STRATEGIES`
- Ajouter `"grid_funding"` dans `STRATEGIES_NEED_EXTRA_DATA`
- **Decoupler** FAST_ENGINE : remplacer `set(...) - STRATEGIES_NEED_EXTRA_DATA` par `set(...) - _NO_FAST_ENGINE` ou `_NO_FAST_ENGINE = {"funding", "liquidation"}`

---

### Etape 4 — Indicator Cache (`backend/optimization/indicator_cache.py`)

**4a. Nouveaux champs `IndicatorCache`** :
```python
funding_rates_1h: np.ndarray | None = None    # shape (n,), raw decimal, forward-filled
candle_timestamps: np.ndarray | None = None   # epoch ms, shape (n,)
```

Note : utiliser `field(default=None)` dans le dataclass. Les tests existants qui creent des caches manuellement ne cassent pas grace au defaut.

**4b. Nouveau parametre `build_cache()`** :
```python
def build_cache(
    candles_by_tf, param_grid_values, strategy_name,
    main_tf="5m", filter_tf="15m",
    db_path: str | None = None,        # NOUVEAU
    symbol: str | None = None,         # NOUVEAU (pour query funding DB)
    exchange: str | None = None,       # NOUVEAU
) -> IndicatorCache:
```

**4c. Nouvelle fonction `_load_funding_rates_aligned()`** (~25 lignes) :
- Ouvre sqlite3 sync, query `SELECT timestamp, funding_rate FROM funding_rates WHERE symbol=? AND exchange=? ORDER BY timestamp`
- Divise `funding_rate` par 100 → raw decimal
- Pas de decalage : `np.searchsorted(fr_timestamps, candle_timestamps, side='right') - 1` direct
- Forward-fill via l'index searchsorted
- Retourne `np.ndarray` shape (n,), NaN avant le premier funding connu

**4d. Branche `grid_funding` dans `build_cache()`** :
```python
if strategy_name == "grid_funding":
    if db_path is None:
        raise ValueError("grid_funding requires db_path for funding rates")
    # SMA pour TP
    ma_periods = set(param_grid_values.get("ma_period", [14]))
    for period in ma_periods:
        if period not in bb_sma_dict:
            bb_sma_dict[period] = sma(closes, period)
    # Timestamps et funding rates
    candle_ts = np.array([c.timestamp.timestamp() * 1000 for c in main_candles])
    funding_1h = _load_funding_rates_aligned(symbol, exchange, candle_ts, db_path)
```

Passer `funding_rates_1h` et `candle_timestamps` au constructeur `IndicatorCache(...)`.

---

### Etape 5 — Fast Engine (`backend/optimization/fast_multi_backtest.py`)

**5a. `_build_entry_signals()`** (~20 lignes) :
- Retourne `np.ndarray` shape (n, num_levels) dtype bool
- `signals[:, lvl] = funding <= -(threshold_start + lvl * threshold_step)`
- NaN funding → False

**5b. `_calc_grid_pnl_with_funding()`** (~35 lignes) :
- Fonction separee (NE MODIFIE PAS `_calc_grid_pnl()`)
- Positions = list of `(entry_price, quantity, entry_idx)`
- PnL prix classique + fees + slippage
- Funding payments : pour chaque candle entre entry et exit, si `(epoch_ms // 3600000) % 8 == 0` → c'est une frontiere 8h → `funding_pnl -= fr * notional` (LONG + FR negatif = on recoit)
- Note : le `-=` avec un FR negatif donne un bonus positif. C'est correct.

**5c. `_simulate_grid_funding()`** (~80 lignes) :
- Boucle separee (NE REUTILISE PAS `_simulate_grid_common()`)
- Logique :
  1. Check exit : SL toujours actif, TP seulement apres min_hold
  2. TP modes : `funding_positive` (fr > 0), `sma_cross` (close >= sma), `funding_or_sma` (l'un ou l'autre)
  3. Check entry : parcourir levels, si signal actif ET level non rempli → ouvrir position au prix courant
  4. Force close derniere bougie
- PnL via `_calc_grid_pnl_with_funding()`

**5d. Branche dans `run_multi_backtest_from_cache()`** :
```python
elif strategy_name == "grid_funding":
    trade_pnls, trade_returns, final_capital = _simulate_grid_funding(cache, params, bt_config)
```

---

### Etape 6 — MultiPositionEngine (`backend/backtesting/multi_engine.py`)

Modification minimale (~10 lignes) :

**6a. `run_multi_backtest_single()`** : ajouter parametre `extra_data_by_timestamp: dict[str, dict[str, Any]] | None = None`, le passer a `MultiPositionEngine`.

**6b. `MultiPositionEngine.__init__()`** : stocker `self._extra_data = extra_data_by_timestamp or {}`.

**6c. `MultiPositionEngine.run()`** : dans la construction du `StrategyContext`, ajouter :
```python
ts_iso = candle.timestamp.isoformat()
extra = self._extra_data.get(ts_iso, {})
ctx = StrategyContext(..., extra_data=extra)
```

---

### Etape 7 — Walk Forward (`backend/optimization/walk_forward.py`)

**7a. `_INDICATOR_PARAMS`** : ajouter `"grid_funding": ["ma_period"]`

**7b. `_run_fast()`** : ajouter parametre `db_path: str | None = None`, `symbol: str | None = None`, `exchange: str | None = None`. Les passer a `build_cache()`.

**7c. `_parallel_backtest()`** : ajouter les memes parametres, les passer a `_run_fast()`.

**7d. `optimize()`** :
- Sauvegarder `db_path = db.db_path` avant `await db.close()`
- Passer `db_path`, `symbol`, `exchange` a `_parallel_backtest()` (qui les forward a `_run_fast()`)

**7e. OOS evaluation** : passer `extra_data_by_timestamp=oos_extra_data_map` a `run_multi_backtest_single()` dans la branche grid (lignes 736-740).

**7f. Workers** : passer `extra_data_by_timestamp=_worker_extra_data` dans `_run_single_backtest_worker()` et `_run_single_backtest_sequential()` quand `is_grid_strategy()`.

---

### Etape 8 — Config YAML

**`config/strategies.yaml`** — ajouter avant `custom_strategies:` :
```yaml
grid_funding:
  enabled: false
  live_eligible: false
  timeframe: 1h
  funding_threshold_start: 0.0005
  funding_threshold_step: 0.0005
  num_levels: 3
  tp_mode: funding_or_sma
  ma_period: 14
  sl_percent: 15.0
  min_hold_candles: 8
  sides:
  - long
  leverage: 6
  weight: 0.2
  per_asset: {}
```

**`config/param_grids.yaml`** — ajouter a la fin :
```yaml
grid_funding:
  wfo:
    is_days: 360
    oos_days: 90
    step_days: 90
  default:
    funding_threshold_start: [0.0003, 0.0005, 0.0008, 0.001]
    funding_threshold_step: [0.0003, 0.0005, 0.001]
    num_levels: [2, 3]
    ma_period: [7, 14, 21]
    sl_percent: [10.0, 15.0, 20.0, 25.0]
    tp_mode: ["funding_positive", "sma_cross", "funding_or_sma"]
    min_hold_candles: [4, 8, 16]
```
= 4×3×2×3×4×3×3 = **2592 combos**

---

### Etape 9 — Fixture test (`tests/conftest.py`)

Ajouter dans `make_indicator_cache._make()` :
- Parametres `funding_rates_1h: np.ndarray | None = None` et `candle_timestamps: np.ndarray | None = None`
- Les passer au constructeur `IndicatorCache(...)`

---

### Etape 10 — Tests (`tests/test_grid_funding.py`) — NOUVEAU

~400 lignes, ~36 tests organises en 7 sections :

**Section 1 — Funding rate alignment** (~5 tests) :
- Anti-lookahead correct (pas de decalage, searchsorted direct)
- Forward-fill sur candles 1h
- NaN avant premier funding
- searchsorted == naive loop
- Conversion DB percent → raw decimal

**Section 2 — Entry signals** (~5 tests) :
- Signal single/multi level selon funding threshold
- Pas de signal si funding positif ou NaN
- Seuils parametriques

**Section 3 — TP/SL** (~6 tests) :
- TP funding_positive, sma_cross, funding_or_sma
- SL percent
- min_hold bloque TP mais pas SL

**Section 4 — Funding payments PnL** (~5 tests) :
- Bonus PnL avec funding negatif
- Cout avec funding positif
- Zero funding = PnL classique
- Multi-positions accumulent
- Detection frontieres 8h

**Section 5 — Fast engine simulation** (~6 tests) :
- Run sans crash
- Entree sur funding negatif / pas d'entree si positif
- Sortie funding positif / SL / force close fin

**Section 6 — Registry + config** (~5 tests) :
- In STRATEGY_REGISTRY, GRID_STRATEGIES, STRATEGIES_NEED_EXTRA_DATA, FAST_ENGINE_STRATEGIES
- create_with_params fonctionne
- Config defaults corrects (LONG-only)

**Section 7 — Cache + DB** (~4 tests) :
- build_cache charge funding_rates_1h (non None)
- grid_atr cache inchange (pas de funding)
- ValueError si db_path manquant
- candle_timestamps rempli

---

## Fichiers modifies (resume)

| Fichier | Action | ~Lignes |
|---------|--------|---------|
| `backend/core/config.py` | MODIFIER — GridFundingConfig + StrategiesConfig | +30 |
| `backend/strategies/grid_funding.py` | NOUVEAU — strategie | ~120 |
| `backend/strategies/factory.py` | MODIFIER — mapping | +8 |
| `backend/optimization/__init__.py` | MODIFIER — registry + sets | +8 |
| `backend/optimization/indicator_cache.py` | MODIFIER — 2 champs + loader + build_cache | +70 |
| `backend/optimization/fast_multi_backtest.py` | MODIFIER — signals + pnl + simulate + branch | +150 |
| `backend/optimization/walk_forward.py` | MODIFIER — indicator_params + db_path + extra_data | +20 |
| `backend/backtesting/multi_engine.py` | MODIFIER — extra_data_by_timestamp | +10 |
| `config/strategies.yaml` | MODIFIER — section grid_funding | +14 |
| `config/param_grids.yaml` | MODIFIER — grille WFO | +15 |
| `tests/conftest.py` | MODIFIER — 2 champs fixture | +5 |
| `tests/test_grid_funding.py` | NOUVEAU — ~36 tests | ~400 |
| **Total** | **12 fichiers** | **~850** |

---

## Ce qu'on NE touche PAS

- `_calc_grid_pnl()` — inchange
- `_simulate_grid_common()` — inchange
- `_build_entry_prices()` — inchange (grid_funding n'utilise pas de prix pre-calcules)
- `grid_atr.py`, `envelope_dca.py`, `grid_multi_tf.py` — inchanges
- `build_extra_data_map()` — inchange (fournit deja funding_rate en percent)
- Les 902 tests existants doivent tous passer

---

## Verification

```bash
# 1. Tests du nouveau module
uv run python -m pytest tests/test_grid_funding.py -v

# 2. Regression complete
uv run python -m pytest --tb=short -q

# 3. Regression grid_atr (rien ne doit casser)
uv run python -m pytest tests/test_grid_atr.py tests/test_grid_multi_tf.py -v

# 4. WFO rapide sur un asset avec funding data
uv run python -m scripts.optimize --strategy grid_funding --symbol IMX/USDT --exchange binance -v
```

## Ordre d'implementation

1. Config (etape 1)
2. Strategie (etape 2)
3. Factory + Registry (etape 3)
4. MultiPositionEngine extra_data (etape 6) — prerequis pour les tests OOS
5. IndicatorCache (etape 4)
6. Fast engine (etape 5)
7. Walk forward (etape 7)
8. Config YAML (etape 8)
9. Fixtures (etape 9)
10. Tests (etape 10)
11. Run tests + fix
