# Sprint XX — Stratégie `grid_momentum` (17e stratégie)

## Context

grid_atr et grid_multi_tf sont des stratégies **mean-reversion** qui excellent en RANGE (84% du temps) mais sous-performent en régime BULL directionnel. Les tentatives trend-following précédentes ont échoué : grid_boltrend (DSR=0/15, trop peu de trades), grid_trend (Grade F, EMA trop lent).

`grid_momentum` est une stratégie **breakout/trend-following** utilisant un profil de payoff **convexe** (petites pertes sur faux breakouts, gros gains sur vrais trends), intentionnellement décorrélée de grid_atr (profil concave).

**Scope** : Implémentation complète + tests. PAS de WFO dans ce sprint.

---

## Corrections apportées au design document initial

| # | Erreur/Oubli | Correction |
|---|-------------|------------|
| 1 | `get_tp_price()` suggère `float('inf')` | → `float("nan")` (convention grid_boltrend L244, grid_trend L182) |
| 2 | "grid_atr = 3,240 combos" | → **9,720** combos (3×4×3×5×3×3×4 réel) |
| 3 | `cooldown_candles` absent de la config | → Ajouté : `cooldown_candles: int = Field(default=3, ge=0, le=10)` + dans WFO grid |
| 4 | `compute_live_indicators()` absent du design | → Ajouté : requis pour live/portfolio |
| 5 | Anti-lookahead Donchian "exclure bougie courante" | → Déjà géré par `_rolling_max()` : `rolling_high[i] = max(highs[i-window:i])` |
| 6 | Direction flip → re-entry same candle ? | → Non : close tout, `compute_grid()` évalue normalement à la bougie suivante |
| 7 | `exit_reason` trailing stop | → String = `"trail_stop"` (pas "trailing_stop") |
| 8 | Registrations manquantes | → Ajouter dans `GRID_STRATEGIES` et `FAST_ENGINE_STRATEGIES` (PAS `STRATEGIES_NEED_EXTRA_DATA` : grid_momentum n'utilise ni funding rates ni OI) |
| 9 | Combos WFO avec cooldown_candles | → ~31K combos avec réductions (voir section WFO) |

---

## Décisions architecturales

### 1. HWM (High Water Mark) → Option B enrichie

**Choix** : PAS d'ajout à `GridState`. Variable locale dans le fast engine (pattern grid_trend L245-275). Pour le live, HWM tracké dans `GridStrategyRunner` et injecté via le dict indicators.

**Justification** :
- grid_trend a exactement le même design (trailing stop uniquement dans le fast engine, SL+direction_flip en live)
- Ajouter un champ optionnel à `GridState` impacte 8 stratégies existantes → risque de régression
- `compute_live_indicators()` n'a PAS accès aux positions (signature: `candles: list[Candle]` uniquement) → impossible d'y calculer le HWM

**HWM en live — mécanisme** :
- `GridStrategyRunner` maintient `_hwm: dict[str, float]` (par symbol)
- Initialisé au `high` de la bougie du breakout (quand première position s'ouvre)
- Mis à jour à chaque `on_candle()` : `_hwm[symbol] = max(_hwm[symbol], candle.high)` (LONG)
- Injecté dans les indicators avant `should_close_all()` : `indicators["hwm"] = self._hwm.get(symbol, float("nan"))`
- Reset à `float("nan")` quand toutes les positions sont fermées
- Note : nécessite une modification minimale de `_on_candle_inner()` dans `simulator.py` (ajouter ~10 lignes)

### 2. Fast engine → Fonction dédiée `_simulate_grid_momentum()`

**Choix** : Fonction standalone (pattern grid_boltrend), PAS wrapper de `_simulate_grid_common()`.

**Justification** :
- Breakout = state machine (inactif → détection → DCA → exit → inactif), incompatible avec `_simulate_grid_common()` qui suppose des entry levels toujours actifs
- `_simulate_grid_common()` pré-calcule les entry prices via `_build_entry_prices()` — grid_momentum les fixe au moment du breakout (prix dynamique)
- Le trailing stop sera codé directement dans la fonction dédiée (trivial, ~15 lignes, copie du pattern L245-275)

### 3. TP → `float("nan")`

Pas de TP fixe. `get_tp_price()` retourne `float("nan")`. Le trailing stop gère la sortie en profit via `should_close_all()` retournant `"trail_stop"`.

### 4. Réutilisation du cache d'indicateurs

| Indicateur | Champ cache existant | Action |
|------------|---------------------|--------|
| Donchian high/low | `rolling_high[period]`, `rolling_low[period]` | Réutiliser, ajouter computation dans `build_cache()` pour grid_momentum |
| ATR | `atr_by_period[period]` | Réutiliser, déjà multi-period |
| Volume SMA | `volume_sma_arr` (period=20 fixe) | Réutiliser tel quel (vol_sma_period=20 non optimisé) |
| ADX | `adx_arr` (period=14 fixe) | Réutiliser tel quel (adx_period=14 non optimisé) |

Aucun nouveau champ à ajouter à `IndicatorCache`. Juste calculer `rolling_high`/`rolling_low` pour les `donchian_period` values dans `build_cache()`.

---

## Plan fichier par fichier

### 1. `backend/core/config.py` — GridMomentumConfig

Ajouter après `GridBolTrendConfig` (~L391) :

```python
class GridMomentumConfig(BaseModel):
    """Grid Momentum : DCA pullback sur breakout Donchian + trailing stop ATR."""
    enabled: bool = False
    live_eligible: bool = False
    timeframe: str = "1h"
    donchian_period: int = Field(default=30, ge=10, le=100)
    vol_sma_period: int = Field(default=20, ge=5, le=50)
    vol_multiplier: float = Field(default=1.5, gt=0)
    adx_period: int = Field(default=14, ge=5, le=30)
    adx_threshold: float = Field(default=0.0, ge=0)
    atr_period: int = Field(default=14, ge=5, le=30)
    pullback_start: float = Field(default=1.0, gt=0)
    pullback_step: float = Field(default=0.5, gt=0)
    num_levels: int = Field(default=3, ge=1, le=6)
    trailing_atr_mult: float = Field(default=3.0, gt=0)
    sl_percent: float = Field(default=15.0, gt=0)
    sides: list[str] = Field(default=["long", "short"])
    leverage: int = Field(default=6, ge=1, le=20)
    weight: float = Field(default=0.15, ge=0, le=1)
    cooldown_candles: int = Field(default=3, ge=0, le=10)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}
```

Ajouter dans `StrategiesConfig` (~L416) :
```python
grid_momentum: GridMomentumConfig = Field(default_factory=GridMomentumConfig)
```

---

### 2. `backend/strategies/grid_momentum.py` — Stratégie (NOUVEAU)

Pattern : hybride grid_boltrend (breakout event-driven) + grid_trend (trailing stop, direction flip).

```python
class GridMomentumStrategy(BaseGridStrategy):
    name = "grid_momentum"
```

**Méthodes :**

#### `compute_indicators(candles, params)` (~L50-100)
- Calculer : Donchian high/low (rolling max/min excluant bougie courante), ATR, close, high, volume, volume_sma, ADX
- Stocker `prev_close`, `prev_donchian_high`, `prev_donchian_low` (pour breakout: prev < threshold, current > threshold)
- Pattern identique à grid_boltrend L53-96

#### `compute_grid(ctx, grid_state)` → `list[GridLevel]` (~L102-180)
- **Si positions ouvertes** : anchor = first position entry_price, niveaux pullback depuis anchor (pattern grid_boltrend L126-145)
  - LONG : `anchor - k * atr * pullback_step` (k=0..num_levels-1, level 0 = anchor - atr * pullback_start)
  - Wait — correction : level 0 est le breakout price (déjà rempli). Levels 1+ sont les pullbacks.
  - Si `last_close_candle_idx` existe et `i - last_close_candle_idx < cooldown_candles` → return []
- **Si aucune position** :
  - Vérifier cooldown : si dernière fermeture < cooldown_candles bougies → return []
  - Breakout LONG : `close > donchian_high` ET `volume > volume_sma * vol_multiplier` ET (`adx > adx_threshold` ou threshold=0) ET `"long" in sides`
  - Breakout SHORT : `close < donchian_low` ET mêmes filtres ET `"short" in sides`
  - Si breakout → calculer niveaux depuis close :
    - Level 0 : close (trigger immédiat)
    - Level i : `close ∓ atr * (pullback_start + (i-1) * pullback_step)` (i≥1)

#### `should_close_all(ctx, grid_state)` → `str | None` (~L182-240)

**PAS de SL ici** — le SL est géré par `get_sl_price()` + `check_global_tp_sl()` dans le runner (OHLC heuristic, plus précis). `should_close_all()` ne gère que :

- **Direction flip** (sortie de protection, SANS filtre volume/ADX) :
  - LONG ouvert et `close < donchian_low` → `"direction_flip"`
  - SHORT ouvert et `close > donchian_high` → `"direction_flip"`
  - Le filtre volume/ADX est pour l'ENTRÉE uniquement, pas la sortie. On ferme dès que le breakout inverse est détecté.
- **Trailing stop** : `close < hwm - atr * trailing_atr_mult` (LONG) → `"trail_stop"`
  - HWM lu depuis `indicators["hwm"]` (injecté par le runner, voir décision archi #1)
  - Si `hwm` est NaN (pas de tracking disponible) → trailing stop désactivé (fallback SL uniquement)
  - En fast engine, le trailing stop est géré directement dans `_simulate_grid_momentum()`

#### `get_tp_price(grid_state, indicators)` → `float("nan")` (~L242-246)

#### `get_sl_price(grid_state, indicators)` → `float` (~L248-260)
- `avg_entry * (1 - sl_percent/100)` LONG, `* (1 + sl_percent/100)` SHORT

#### `compute_live_indicators(candles)` → `dict` (~L262-320)

Pattern GridTrendStrategy L202-231. Depuis le buffer 1h :

- Donchian high/low sur `donchian_period` dernières bougies (excluant la courante)
- ATR sur `atr_period`
- Volume SMA sur `vol_sma_period`
- ADX sur `adx_period`
- PAS de HWM ici (pas accès aux positions) — géré par le runner (voir décision archi #1)
- Retourner dict avec toutes les valeurs courantes

---

### 3. `backend/strategies/factory.py` — Registry

Ajouter l'import et l'entrée dans la map de `create_strategy()` :
```python
"grid_momentum": (GridMomentumStrategy, config.strategies.grid_momentum)
```

---

### 4. `backend/optimization/__init__.py` — Registrations

```python
# STRATEGY_REGISTRY : ajouter
"grid_momentum": (GridMomentumConfig, GridMomentumStrategy)

# GRID_STRATEGIES : ajouter
"grid_momentum"

# FAST_ENGINE_STRATEGIES : vérifier si auto-inclus, sinon ajouter

# NE PAS ajouter à STRATEGIES_NEED_EXTRA_DATA (pas de funding rates ni OI)
```

---

### 5. `backend/optimization/indicator_cache.py` — build_cache()

Ajouter une section pour `grid_momentum` dans `build_cache()` (~après grid_boltrend L424) :

```python
if "grid_momentum" in strategy_names:
    # Donchian channels (rolling_high/rolling_low)
    for p in param_grid_values.get("donchian_period", [30]):
        if p not in cache.rolling_high:
            cache.rolling_high[p] = _rolling_max(cache.highs, p)
            cache.rolling_low[p] = _rolling_min(cache.lows, p)
    # ATR multi-period (réutilise le pattern existant)
    for p in param_grid_values.get("atr_period", [14]):
        if p not in cache.atr_by_period:
            cache.atr_by_period[p] = _atr(cache.highs, cache.lows, cache.closes, p)
```

`volume_sma_arr` (period=20) et `adx_arr` (period=14) sont déjà calculés par défaut pour toutes les stratégies.

---

### 6. `backend/optimization/fast_multi_backtest.py` — `_simulate_grid_momentum()`

Fonction dédiée (~200 lignes), placée après `_simulate_grid_boltrend()` (~L1140).

**State machine** (3 états, pattern grid_boltrend) :

```
INACTIVE (direction=0) → BREAKOUT détecté → ACTIVE (direction=±1, DCA filling) → EXIT → INACTIVE
```

**Variables d'état** :
```python
direction = 0           # 0=inactif, 1=LONG, -1=SHORT
positions = []          # list[tuple[level, entry_price, qty, entry_fee]]
entry_levels = []       # list[float] — fixés au breakout
hwm = 0.0              # High/Low Water Mark
breakout_candle_idx = -1
last_exit_candle_idx = -1  # pour cooldown
capital = initial_capital
```

**Boucle principale (i = warmup..n_candles)** :

1. **CHECK EXITS** (si positions ouvertes) :
   - SL global : `avg_entry * (1 ± sl_pct)`, check `lows[i] <= sl_price` (LONG)
   - Trailing stop : `hwm = max(hwm, highs[i])`, `trail_price = hwm - atr[i] * trailing_atr_mult`, check `lows[i] <= trail_price`
   - Direction flip : breakout inverse détecté (`close > donchian_high` si SHORT, `close < donchian_low` si LONG) — **SANS filtre volume/ADX** (sortie de protection)
   - Résolution si multiple : SL > trail > flip (priorité au plus défavorable)
   - Exit : `_calc_grid_pnl()`, restituer margin, enregistrer trade, `last_exit_candle_idx = i`, reset state

2. **CHECK NEW ENTRIES** (si inactif, direction=0) :
   - Cooldown check : `i - last_exit_candle_idx < cooldown_candles` → skip
   - Breakout LONG : `close[i] > rolling_high[donchian_period][i]` ET `volumes[i] > volume_sma_arr[i] * vol_multiplier` ET (`adx_arr[i] > adx_threshold` ou threshold=0)
   - Breakout SHORT : `close[i] < rolling_low[donchian_period][i]` ET mêmes filtres
   - Si breakout :
     - `direction = ±1`, `breakout_candle_idx = i`
     - Calculer entry_levels depuis `close[i]`
     - Level 0 = `close[i]` → trigger immédiat, ouvrir position
     - HWM init = `highs[i]`

3. **FILL DCA LEVELS** (si actif, positions < num_levels) :
   - Pour chaque level non rempli : check si price touched
   - LONG : `lows[i] <= entry_level` → fill
   - SHORT : `highs[i] >= entry_level` → fill
   - Margin accounting : `margin = (entry_price * qty) / leverage`, déduire du capital

4. **HWM UPDATE** (si actif) :
   - LONG : `hwm = max(hwm, highs[i])`
   - SHORT : `hwm = min(hwm, lows[i])`

**Fees** : taker_fee + slippage pour toutes les sorties (market orders). Pattern `_calc_grid_pnl()` existant.

**Pas de funding** dans cette implémentation (cohérent avec grid_boltrend).

---

### 7. `backend/optimization/fast_multi_backtest.py` — Dispatcher

Ajouter dans `run_multi_backtest_from_cache()` (~L1200) :

```python
elif strategy_name == "grid_momentum":
    trade_pnls, trade_returns, final_capital = _simulate_grid_momentum(
        cache, params, bt_config
    )
```

---

### 8. `backend/optimization/walk_forward.py` — `_INDICATOR_PARAMS`

Ajouter :
```python
"grid_momentum": ["donchian_period", "atr_period"],
```

Seuls ces 2 params affectent la computation d'indicateurs dans `build_cache()`. `adx_period` (fixe=14) et `vol_sma_period` (fixe=20) utilisent les arrays par défaut du cache.

---

### 9. `config/strategies.yaml`

Ajouter après `grid_boltrend:` :

```yaml
grid_momentum:
  enabled: false
  live_eligible: false
  timeframe: 1h
  donchian_period: 30
  vol_sma_period: 20
  vol_multiplier: 1.5
  adx_period: 14
  adx_threshold: 0
  atr_period: 14
  pullback_start: 1.0
  pullback_step: 0.5
  num_levels: 3
  trailing_atr_mult: 3.0
  sl_percent: 15.0
  sides:
  - long
  - short
  leverage: 6
  cooldown_candles: 3
  weight: 0.15
  per_asset: {}
```

---

### 10. `config/param_grids.yaml`

```yaml
grid_momentum:
  wfo:
    is_days: 180
    oos_days: 60
    step_days: 60
  default:
    donchian_period: [20, 30, 48, 72]
    vol_multiplier: [1.2, 1.5, 2.0, 2.5]
    atr_period: [10, 14, 20]
    adx_threshold: [0, 20, 25]
    pullback_start: [0.5, 1.0, 1.5]
    pullback_step: [0.5, 1.0]
    num_levels: [2, 3, 4]
    trailing_atr_mult: [2.0, 3.0, 4.0]
    sl_percent: [15, 20, 25]
    cooldown_candles: [0, 3, 5]
```

**Combos** : 4×4×3×3×3×2×3×3×3×3 = **69,984** — trop élevé.

**Réduction recommandée** pour le premier WFO :
- `adx_threshold: [0, 20]` (2 au lieu de 3)
- `sl_percent: [15, 20]` (2 au lieu de 3)
- `cooldown_candles: [0, 3]` (2 au lieu de 3)

→ 4×4×3×2×3×2×3×3×2×2 = **20,736** combos. Le 2-pass WFO (coarse LHS ~500 → fine) gérera ça.

---

### 11. `tests/test_grid_momentum.py` — ~33 tests (NOUVEAU)

Structure calquée sur `tests/test_grid_boltrend.py`.

**Helpers** :
- `_make_strategy(**overrides)` → `GridMomentumStrategy`
- `_make_ctx(close, donchian_high, donchian_low, atr, volume, volume_sma, adx, prev_close, prev_donchian_high, prev_donchian_low)` → `StrategyContext`
- `_make_grid_state(positions)` → `GridState`
- `_make_breakout_cache(make_indicator_cache, direction="long", n=500)` → `IndicatorCache` avec données synthétiques

**Section 1 : Breakout detection + compute_grid (~8 tests)**
1. Breakout LONG : `close > donchian_high` + volume OK → niveaux LONG
2. Breakout SHORT : `close < donchian_low` + volume OK → niveaux SHORT
3. Pas de breakout si volume insuffisant
4. Pas de breakout si ADX < threshold (quand threshold > 0)
5. Breakout ignoré si côté pas dans `sides`
6. Niveaux pullback DCA calculés correctement (pullback_start, pullback_step)
7. Niveaux DCA en SHORT (au-dessus du prix)
8. Cooldown : pas de breakout si dernière fermeture < cooldown_candles bougies

**Section 2 : Trailing stop + SL + direction flip (~6 tests)**
9. Trailing stop LONG : `close < HWM - trailing_atr_mult * ATR` → `"trail_stop"`
10. Trailing stop SHORT : `close > LWM + trailing_atr_mult * ATR` → `"trail_stop"`
11. SL global se déclenche avant trailing si prix chute directement → `"sl_global"`
12. Direction flip LONG→SHORT : breakout SHORT pendant position LONG → `"direction_flip"`
13. Direction flip SHORT→LONG : breakout LONG pendant position SHORT → `"direction_flip"`
14. Pas de fermeture si ni trailing ni SL ni flip → `None`

**Section 3 : Fast engine (~10 tests)**
15. `_simulate_grid_momentum()` produit des trades sur données breakout synthétiques
16. Pas de trades si aucun breakout (données plates, prix constant)
17. Trailing stop ferme la position après pump puis reversal
18. SL global touché sur données bear directes
19. Direction flip gère correctement le changement de direction (sans filtre volume/ADX)
20. Capital tracking correct (pas d'overflow, pas de capital négatif)
21. HWM mis à jour correctement candle par candle
22. Cooldown respecté entre trades
23. Breakout SHORT fonctionne (symétrie)
24. Multi-level fill sur même bougie : breakout + long wick → Level 0 et Level 1 remplis simultanément

**Section 4 : Registry et config (~5 tests)**
25. `grid_momentum` dans `STRATEGY_REGISTRY`
26. `grid_momentum` dans `GRID_STRATEGIES`
27. `create_strategy_with_params("grid_momentum", {...})` retourne `GridMomentumStrategy`
28. `_INDICATOR_PARAMS["grid_momentum"]` correctement configuré
29. Config YAML parseable sans erreur

**Section 5 : Edge cases (~4 tests)**
30. ATR = 0 ou NaN → pas de crash, pas de niveaux
31. Prix constant → pas de breakout Donchian (rolling_high = rolling_low = prix)
32. Breakout immédiat sur la première bougie utilisable → géré proprement
33. Volume SMA = 0 → pas de division par zéro (le check est `volume > sma * mult`, pas ratio)

---

### 12. `docs/STRATEGIES.md` — Documentation

Ajouter une section grid_momentum dans la documentation existante :
- Description courte (breakout Donchian + DCA pullback + trailing ATR)
- Paramètres avec descriptions
- Profil de payoff (convexe vs concave)
- Décorrélation intentionnelle avec grid_atr/grid_multi_tf

---

## Ordre d'implémentation

1. `backend/core/config.py` — GridMomentumConfig + StrategiesConfig
2. `backend/strategies/grid_momentum.py` — Stratégie complète
3. `backend/strategies/factory.py` — Registry
4. `backend/optimization/__init__.py` — Registrations (3 sets : STRATEGY_REGISTRY, GRID_STRATEGIES, FAST_ENGINE_STRATEGIES)
5. `backend/optimization/indicator_cache.py` — build_cache() section
6. `backend/optimization/fast_multi_backtest.py` — `_simulate_grid_momentum()` + dispatcher
7. `backend/optimization/walk_forward.py` — `_INDICATOR_PARAMS`
8. `backend/backtesting/simulator.py` — HWM tracking dans `GridStrategyRunner._on_candle_inner()` (~10 lignes)
9. `config/strategies.yaml` + `config/param_grids.yaml`
10. `tests/test_grid_momentum.py` — 33 tests
11. `docs/STRATEGIES.md` — Documentation

---

## Vérification

1. **Tests unitaires** : `PYTHON_JIT=0 uv run pytest tests/test_grid_momentum.py -x -v`
2. **Zéro régression** : `PYTHON_JIT=0 uv run pytest tests/ -x -q` (les ~1807 tests existants passent)
3. **Config parse** : `uv run python -c "from backend.core.config import get_config; c = get_config(); print(c.strategies.grid_momentum)"`
4. **Fast engine smoke** : `uv run python -c "from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache; print('OK')"` (pas d'import error)
5. **Commit** : `feat(strategy): grid_momentum breakout DCA (Sprint XX)`
