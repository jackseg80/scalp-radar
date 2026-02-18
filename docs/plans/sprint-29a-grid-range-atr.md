# Sprint 29a — Grid Range ATR (Stratégie Bidirectionnelle)

## Contexte

grid_atr tourne en live mais dort en marché calme (prix proche de la SMA, niveaux ATR larges jamais touchés). `grid_range_atr` monétise les micro-oscillations avec des niveaux serrés, bidirectionnels, et un TP individuel par position.

**Scope Sprint 29a** : Strategy + Config + Fast engine + Registry + WFO pipeline + Tests.
**Hors scope** : Event-driven engine, runner live, Executor Bitget (Sprint 29b).

---

## Fichiers à modifier/créer

### 1. `backend/core/config.py` — Ajouter `GridRangeATRConfig`

Insérer après `GridATRConfig` (ligne 247) :

```python
class GridRangeATRConfig(BaseModel):
    enabled: bool = False
    live_eligible: bool = False
    timeframe: str = "1h"
    ma_period: int = Field(default=20, ge=2, le=50)
    atr_period: int = Field(default=14, ge=2, le=50)
    atr_spacing_mult: float = Field(default=0.3, gt=0)
    num_levels: int = Field(default=2, ge=1, le=6)
    sl_percent: float = Field(default=10.0, gt=0)
    tp_mode: str = Field(default="dynamic_sma")
    sides: list[str] = Field(default=["long", "short"])
    leverage: int = Field(default=6, ge=1, le=20)
    weight: float = Field(default=0.20, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}
```

Modifier `StrategiesConfig` (ligne 336) :
- Ajouter champ `grid_range_atr: GridRangeATRConfig = Field(default_factory=GridRangeATRConfig)`
- Ajouter `self.grid_range_atr` dans la liste `validate_weights` (ligne 361)

### 2. `backend/strategies/grid_range_atr.py` — NOUVEAU (~130 lignes)

`GridRangeATRStrategy(BaseGridStrategy)` — calqué sur `grid_atr.py` :

- `compute_indicators()` : SMA + ATR (identique à grid_atr)
- `compute_grid()` : retourne LONG **et** SHORT simultanément
  - Level encoding : LONG = `0..N-1`, SHORT = `N..2N-1`
  - Entrées = `SMA ± (i+1) × ATR × spacing_mult`
  - PAS de direction lock (positions LONG ne bloquent pas SHORT)
- `should_close_all()` → toujours `None` (TP/SL individuels)
  - **TODO Sprint 29b** : le SL global sera géré par le runner (`RangeGridRunner`), pas par la stratégie. Le kill switch framework (Simulator) prend le relais.
- `get_tp_price()` / `get_sl_price()` → `float("nan")` (pas de global)
  - **TODO Sprint 29b** : NaN safe car `GridStrategyRunner` n'est pas utilisé pour grid_range. Le futur `RangeGridRunner` n'appellera pas `check_global_tp_sl`.
- `max_positions` = `num_levels × len(sides)` (2 côtés = 2N)

### 3. `backend/strategies/factory.py`

- Import `GridRangeATRStrategy`
- Ajouter dans `create_strategy()` mapping (ligne 38)
- Ajouter dans `get_enabled_strategies()` (ligne 77)

### 4. `backend/optimization/__init__.py`

- Import `GridRangeATRConfig` + `GridRangeATRStrategy`
- Ajouter dans `STRATEGY_REGISTRY` (ligne 56)
- Ajouter `"grid_range_atr"` dans `GRID_STRATEGIES` (ligne 67)
- Ajouter `"grid_range_atr"` dans `STRATEGIES_NEED_EXTRA_DATA` (ligne 63)
- `FAST_ENGINE_STRATEGIES` : auto-calculé, rien à faire

### 5. `backend/optimization/indicator_cache.py` — 3 modifications

- Ligne 251 : ajouter `"grid_range_atr"` dans la condition SMA (avec grid_atr/grid_multi_tf)
- Ligne 282 : ajouter `"grid_range_atr"` dans la condition ATR multi-period
- Ligne 307 : ajouter `"grid_range_atr"` dans `_GRID_STRATEGIES_WITH_FUNDING`

### 6. `backend/optimization/fast_multi_backtest.py` — AJOUTS (pas de modif)

**ZÉRO modification à `_simulate_grid_common()` ni `_build_entry_prices()`.**

**Nouvelle fonction `_simulate_grid_range()`** (~120 lignes) :

Position tuple : `(slot_idx, direction, entry_price, qty, entry_fee, entry_sma)`

Boucle :
1. **Check exit individuel** pour chaque position :
   - TP : `high >= tp_price` (LONG) ou `low <= tp_price` (SHORT)
   - `tp_price` = SMA courante si `dynamic_sma`, SMA à l'ouverture si `fixed_center`
   - SL : `low <= entry × (1-sl_pct)` (LONG) ou `high >= entry × (1+sl_pct)` (SHORT)
   - Heuristique OHLC si TP+SL même candle
   - **Fees** : TP = `maker_fee` (limit, 0 slippage), SL = `taker_fee` + slippage (comme `_simulate_grid_common`)
   - PnL individuel → `trade_pnls.append(net)`
2. **Funding** : settlement 8h par position, `capital += -fr × notional × direction` (direction per-position, pas global)
3. **Guard capital** ≤ 0
4. **Ouverture** : entry prices inline `SMA ± (i+1) × ATR × spacing_mult`
   - Slots 0..N-1 = LONG (trigger si `low <= entry`), N..2N-1 = SHORT (trigger si `high >= entry`)
   - Allocation : `capital_courant / total_slots × leverage`
   - Entry fee : `qty × ep × taker_fee`
5. **Force close** fin de données : taker + slippage (comme SL)

Convention fees identique à `_simulate_grid_common()` :
- `_calc_grid_pnl()` pattern : `actual_exit = exit_price × (1 ∓ slippage)`, `gross = direction × (exit - entry) × qty`, `net = gross - entry_fee - exit_fee - slippage_cost`

**Dispatch dans `run_multi_backtest_from_cache()`** : ajouter le elif `grid_range_atr` (ligne 635)

### 7. `backend/optimization/walk_forward.py`

Ajouter dans `_INDICATOR_PARAMS` (ligne 409) :
```python
"grid_range_atr": ["ma_period", "atr_period"],
```

### 8. `backend/backtesting/multi_engine.py`

Ajouter `"grid_range_atr"` dans `_GRID_STRATEGIES_WITH_FUNDING` (ligne 22)

### 9. `config/strategies.yaml`

Insérer après `grid_atr` :
```yaml
grid_range_atr:
  enabled: false
  live_eligible: false
  timeframe: 1h
  ma_period: 20
  atr_period: 14
  atr_spacing_mult: 0.3
  num_levels: 2
  sl_percent: 10.0
  tp_mode: dynamic_sma
  sides:
  - long
  - short
  leverage: 6
  weight: 0.2
  per_asset: {}
```

### 10. `config/param_grids.yaml`

Insérer après `grid_atr` :
```yaml
# Grid Range ATR : 2160 combos (4×3×5×2×3×3×2)
grid_range_atr:
  wfo:
    is_days: 180
    oos_days: 60
    step_days: 60
  default:
    ma_period: [14, 20, 30, 50]
    atr_period: [10, 14, 20]
    atr_spacing_mult: [0.2, 0.3, 0.4, 0.5, 0.7]
    num_levels: [2, 3]
    sl_percent: [8.0, 12.0, 18.0]
    sides:
      - ["long", "short"]
      - ["long"]
      - ["short"]
    tp_mode: ["dynamic_sma", "fixed_center"]
```

### 11. `tests/test_grid_range_atr.py` — NOUVEAU (~42 tests)

**Section 1 — Signaux compute_grid** (~8 tests) :
Niveaux LONG sous SMA, SHORT au-dessus, spacing numérique correct, levels remplis exclus, pas de direction lock, sides filtré, NaN ATR/SMA → vide, `should_close_all` → None

**Section 2 — TP/SL individuels** (~4 tests) :
TP LONG `high >= SMA`, TP SHORT `low <= SMA`, dynamic_sma vs fixed_center, `get_tp_price` et `get_sl_price` → NaN

**Section 3 — SL individuel** (~3 tests) :
SL LONG `low <= entry × (1-sl_pct)`, SL SHORT `high >= entry × (1+sl_pct)`, SL identique tous niveaux

**Section 4 — Fast engine** (~12 tests) :
Résultat 5-tuple valide, pas de trades si ATR=0, positions bidirectionnelles simultanées, TP individuel ferme une sans affecter les autres, réouverture niveau libéré, dynamic_sma vs fixed_center différents, sides=["long"] → pas de SHORT, fees correctes (taker entry + maker TP exit), funding settlement par position, LONG+SHORT fermées sur même candle, capital épuisé entre deux positions

**Section 5 — Viabilité fees** (~2 tests) :
```python
def test_profitability_requires_sufficient_spacing():
    """Spacing trop serré (0.1× ATR) + taker fees → résultat négatif."""

def test_maker_vs_taker_fee_impact():
    """Même config, maker fees (0.02%) vs taker (0.06%) → résultat significativement différent."""
```
Protège contre le risque principal : fees qui mangent le edge des micro-trades.

**Section 6 — Registry et config** (~6 tests) :
STRATEGY_REGISTRY, GRID_STRATEGIES, FAST_ENGINE_STRATEGIES, create_strategy_with_params, GridRangeATRConfig defaults + validation

```python
def test_sides_list_and_tp_mode_string_in_per_asset():
    """sides (list) et tp_mode (string) survivent au pipeline create_strategy_with_params."""
```
Vérifie que les types non-numériques passent correctement dans le pipeline WFO → `--apply` → per_asset → YAML.

**Section 7 — PARITÉ** (~5 tests, LES PLUS IMPORTANTS) :
```python
# Vérifier que les stratégies existantes donnent EXACTEMENT les mêmes résultats
def test_grid_atr_parity(make_indicator_cache): ...
def test_envelope_dca_parity(make_indicator_cache): ...
def test_grid_trend_parity(make_indicator_cache): ...
def test_grid_multi_tf_parity(make_indicator_cache): ...
def test_grid_funding_parity(make_indicator_cache): ...
```
Mêmes seed, mêmes params → mêmes (trade_pnls, trade_returns, final_capital). Si un seul échoue = régression.

---

## Contraintes respectées

1. **ZÉRO modif** `_simulate_grid_common()` — boucle chaude de 5 stratégies en prod
2. **ZÉRO modif** `BaseGridStrategy` — héritage pur, override uniquement
3. **ZÉRO modif** `GridStrategyRunner` — le runner live de grid_atr
4. Entry prices calculés inline dans `_simulate_grid_range()` (pas de modif à `_build_entry_prices`)
5. Fast engine implémenté (pas de "on fera plus tard")
6. Sizing dynamique `capital_courant / total_slots`
7. SL fixe par position (pas de cascading, Sprint 29b)

---

## Ordre d'implémentation

1. Config (`GridRangeATRConfig` + `StrategiesConfig`)
2. Strategy class (`grid_range_atr.py`)
3. Factory + Registry (factory.py, `__init__.py`)
4. Indicator cache (3 sets à étendre)
5. Fast engine (`_simulate_grid_range` + dispatch)
6. Walk forward (`_INDICATOR_PARAMS`)
7. Multi engine (`_GRID_STRATEGIES_WITH_FUNDING`)
8. Configs YAML (strategies.yaml + param_grids.yaml)
9. Tests (~42 tests dont 5 parité, 2 viabilité fees)

---

## Vérification

```bash
# 1. Tests du nouveau module
uv run python -m pytest tests/test_grid_range_atr.py -v

# 2. Régression complète (1129 existants + ~42 nouveaux)
uv run python -m pytest --tb=short -q

# 3. Parité stratégies existantes (CRITIQUE)
uv run python -m pytest tests/test_fast_engine_refactor.py tests/test_grid_atr.py tests/test_grid_trend.py -v

# 4. Backtest exploratoire
uv run python -m scripts.run_backtest --strategy grid_range_atr --symbol BTC/USDT --days 365

# 5. WFO si viable
uv run python -m scripts.optimize --strategy grid_range_atr --symbol BTC/USDT --exchange binance -v
```
