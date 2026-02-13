# Sprint 9 — 3 nouvelles stratégies 1h

## Contexte
Les 4 stratégies 5m (VWAP+RSI, Momentum, Funding, Liquidation) ont toutes Grade F à l'optimisation WFO. On pivote vers des stratégies 1h éprouvées. On dispose de ~48k candles 1h Binance (2020-2026) pour 5 assets (BTC, ETH, SOL, DOGE, LINK).

3 stratégies à implémenter : Bollinger Band Mean Reversion, Donchian Breakout, SuperTrend.

---

## Étape 1 — Indicateurs numpy purs (`backend/core/indicators.py`)

Ajouter 2 fonctions pures réutilisables :

**`bollinger_bands(closes, period=20, std_dev=2.0) → (sma, upper, lower)`**
- SMA existante (`sma()`) + `np.std` rolling manuel
- Retourne 3 arrays (n,), NaN avant `period`

**`supertrend(highs, lows, closes, atr_arr, multiplier=3.0) → (st_values, direction)`**
- Calcul itératif (boucle Python, ~5ms pour 48k points)
- `direction[i]` = 1 (UP) ou -1 (DOWN)
- Signal = flip de direction entre `i-1` et `i`

Note : Donchian channels = `_rolling_max` / `_rolling_min` déjà dans `indicator_cache.py`, pas besoin de nouvelle fonction dans indicators.py.

---

## Étape 2 — Configs Pydantic (`backend/core/config.py`)

3 nouvelles classes + intégration dans `StrategiesConfig` :

```python
class BollingerMRConfig(BaseModel):
    enabled: bool = True
    live_eligible: bool = False
    timeframe: str = "1h"
    bb_period: int = Field(default=20, ge=2)
    bb_std: float = Field(default=2.0, gt=0)
    sl_percent: float = Field(default=5.0, gt=0)
    weight: float = Field(default=0.15, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)
    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]: ...

class DonchianBreakoutConfig(BaseModel):
    enabled: bool = True
    live_eligible: bool = False
    timeframe: str = "1h"
    entry_lookback: int = Field(default=20, ge=2)
    atr_period: int = Field(default=14, ge=2)
    atr_tp_multiple: float = Field(default=3.0, gt=0)
    atr_sl_multiple: float = Field(default=1.5, gt=0)
    weight: float = Field(default=0.15, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)
    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]: ...

class SuperTrendConfig(BaseModel):
    enabled: bool = True
    live_eligible: bool = False
    timeframe: str = "1h"
    atr_period: int = Field(default=10, ge=2)
    atr_multiplier: float = Field(default=3.0, gt=0)
    tp_percent: float = Field(default=4.0, gt=0)
    sl_percent: float = Field(default=2.0, gt=0)
    weight: float = Field(default=0.15, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)
    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]: ...
```

Ajouter dans `StrategiesConfig` :
```python
bollinger_mr: BollingerMRConfig = Field(default_factory=BollingerMRConfig)
donchian_breakout: DonchianBreakoutConfig = Field(default_factory=DonchianBreakoutConfig)
supertrend: SuperTrendConfig = Field(default_factory=SuperTrendConfig)
```
+ Mettre à jour `validate_weights` pour inclure les 3 nouvelles.

---

## Étape 3 — 3 fichiers stratégie

### `backend/strategies/bollinger_mr.py` — BollingerMRStrategy
- `name = "bollinger_mr"`
- `min_candles = {"1h": max(bb_period + 20, 50)}`
- `compute_indicators` : SMA + Bollinger bands via `indicators.bollinger_bands()`
- `evaluate` :
  - LONG si `close < lower_band`, SHORT si `close > upper_band`
  - `tp_price` = très éloigné (entry×2 LONG, entry×0.5 SHORT) — **désactive le TP fixe du BacktestEngine**
  - `sl_price` = entry ± sl_percent%
  - Score basé sur distance aux bandes (plus on est loin = plus fort)
- `check_exit` : close a croisé la SMA → `"signal_exit"` (TP dynamique). **C'est check_exit qui gère le vrai TP, pas tp_price.**
- `get_current_conditions` : 2 conditions (bb_position, sl_distance)
- **Point clé TP dynamique** : `evaluate()` ET `_open_trade` (fast engine) mettent TOUS LES DEUX un tp_price très éloigné pour que `_check_tp_sl` / `PositionManager.check_position_exit()` ne se déclenchent jamais sur le TP. La sortie réelle est gérée par `check_exit()` qui vérifie le crossing SMA à chaque bougie. Parité garantie entre les deux moteurs.

### `backend/strategies/donchian_breakout.py` — DonchianBreakoutStrategy
- `name = "donchian_breakout"`
- `min_candles = {"1h": max(entry_lookback + 20, 50)}`
- `compute_indicators` : rolling max(highs) / rolling min(lows) sur N bougies + ATR
- `evaluate` :
  - LONG si `close > rolling_high[lookback]`, SHORT si `close < rolling_low[lookback]`
  - `tp_price` = entry ± ATR × atr_tp_multiple
  - `sl_price` = entry ∓ ATR × atr_sl_multiple
  - Score basé sur force du breakout (distance au canal)
- `check_exit` : pas de sortie anticipée (TP/SL fixes)
- `get_current_conditions` : 2 conditions (channel_position, atr_ratio)

### `backend/strategies/supertrend.py` — SuperTrendStrategy
- `name = "supertrend"`
- `min_candles = {"1h": max(atr_period + 20, 50)}`
- `compute_indicators` : ATR + SuperTrend via `indicators.supertrend()`
- `evaluate` :
  - LONG si direction flip DOWN→UP, SHORT si flip UP→DOWN
  - `tp_price` = entry ± tp_percent%
  - `sl_price` = entry ∓ sl_percent%
  - Score basé sur ATR normalized + flip strength
- `check_exit` : pas de sortie anticipée (TP/SL fixes)
- `get_current_conditions` : 2 conditions (st_direction, distance_to_st)

---

## Étape 4 — Factory + Registry

### `backend/strategies/factory.py`
Ajouter les 3 stratégies dans `create_strategy()` et `get_enabled_strategies()`.

### `backend/optimization/__init__.py`
```python
STRATEGY_REGISTRY["bollinger_mr"] = (BollingerMRConfig, BollingerMRStrategy)
STRATEGY_REGISTRY["donchian_breakout"] = (DonchianBreakoutConfig, DonchianBreakoutStrategy)
STRATEGY_REGISTRY["supertrend"] = (SuperTrendConfig, SuperTrendStrategy)
```

---

## Étape 5 — IndicatorCache + build_cache (`backend/optimization/indicator_cache.py`)

### Extension IndicatorCache
```python
@dataclass
class IndicatorCache:
    # ... existant ...

    # Bollinger MR
    bb_sma: dict[int, np.ndarray]                          # {period: sma_array}
    bb_upper: dict[tuple[int, float], np.ndarray]           # {(period, std): upper_band}
    bb_lower: dict[tuple[int, float], np.ndarray]           # {(period, std): lower_band}

    # Donchian — réutilise rolling_high/rolling_low existants
    # (ajouter entry_lookback aux lookbacks lors du build)

    # SuperTrend
    supertrend_direction: dict[tuple[int, float], np.ndarray]  # {(atr_period, multiplier): direction}

    # ATR multi-period (pour Donchian/SuperTrend qui utilisent atr_period variable)
    atr_by_period: dict[int, np.ndarray]                     # {period: atr_array}
```

### Extension build_cache
- Si `strategy_name == "bollinger_mr"` : pré-calculer `bb_sma`, `bb_upper`, `bb_lower` pour chaque combo `(bb_period, bb_std)` du grid
- Si `strategy_name == "donchian_breakout"` : ajouter `entry_lookback` values aux rolling_high/rolling_low + pré-calculer ATR pour chaque `atr_period`
- Si `strategy_name == "supertrend"` : pré-calculer `supertrend_direction` pour chaque combo `(atr_period, atr_multiplier)` + ATR par period
- Initialiser les champs manquants à `{}` pour les stratégies qui n'en ont pas besoin

---

## Étape 6 — Fast engine (`backend/optimization/fast_backtest.py`)

### 3 nouvelles fonctions de signaux

**`_bollinger_mr_signals(params, cache)`** :
```python
bb_sma = cache.bb_sma[params["bb_period"]]
bb_lower = cache.bb_lower[(params["bb_period"], params["bb_std"])]
bb_upper = cache.bb_upper[(params["bb_period"], params["bb_std"])]
valid = ~np.isnan(bb_sma) & ~np.isnan(bb_lower) & ~np.isnan(bb_upper)
longs = valid & (cache.closes < bb_lower)
shorts = valid & (cache.closes > bb_upper)
```

**`_donchian_signals(params, cache)`** :
```python
lookback = params["entry_lookback"]
rolling_high = cache.rolling_high[lookback]
rolling_low = cache.rolling_low[lookback]
valid = ~np.isnan(rolling_high) & ~np.isnan(rolling_low)
longs = valid & (cache.closes > rolling_high)
shorts = valid & (cache.closes < rolling_low)
```

**`_supertrend_signals(params, cache)`** :
```python
direction = cache.supertrend_direction[(params["atr_period"], params["atr_multiplier"])]
valid = ~np.isnan(direction[:-1])  # Need previous direction
# Flip detection
prev_dir = np.roll(direction, 1)
longs = (prev_dir == -1) & (direction == 1)  # DOWN→UP
shorts = (prev_dir == 1) & (direction == -1)  # UP→DOWN
longs[0] = False  # Pas de signal sur la première bougie
shorts[0] = False
```

### Extension `_open_trade`
Ajouter 3 branches :
- **bollinger_mr** : `tp_price` très éloigné (entry×2 LONG, entry×0.5 SHORT), `sl_price` = entry ± sl_percent%
- **donchian_breakout** : TP/SL = ATR × multiples (comme momentum mais avec atr_period variable depuis `cache.atr_by_period`)
- **supertrend** : TP/SL = entry ± tp_percent% / sl_percent%

### Extension `_check_exit`
Ajouter branche **bollinger_mr** : `close >= bb_sma[period]` (LONG) ou `close <= bb_sma[period]` (SHORT) → return True

### Extension `run_backtest_from_cache`
Ajouter les 3 strategy_name dans le switch.

---

## Étape 7 — WFO updates (`backend/optimization/walk_forward.py`)

### `_INDICATOR_PARAMS`
```python
_INDICATOR_PARAMS["bollinger_mr"] = ["bb_period", "bb_std"]
_INDICATOR_PARAMS["donchian_breakout"] = ["entry_lookback", "atr_period"]
_INDICATOR_PARAMS["supertrend"] = ["atr_period", "atr_multiplier"]
```

### Per-strategy WFO config
Dans `optimize()`, après lecture de `opt_config`, ajouter :
```python
strategy_wfo = strategy_grids.get("wfo", {})
is_window_days = strategy_wfo.get("is_days", is_window_days)
oos_window_days = strategy_wfo.get("oos_days", oos_window_days)
step_days = strategy_wfo.get("step_days", step_days)
```

### Fast engine condition
Changer le check de `_parallel_backtest` :
```python
if strategy_name in ("vwap_rsi", "momentum", "bollinger_mr", "donchian_breakout", "supertrend"):
```

---

## Étape 8 — Config YAML

### `config/strategies.yaml` — Ajouter après `funding:`
```yaml
bollinger_mr:
  enabled: true
  live_eligible: false
  timeframe: "1h"
  bb_period: 20
  bb_std: 2.0
  sl_percent: 5.0
  weight: 0.15
  per_asset: {}

donchian_breakout:
  enabled: true
  live_eligible: false
  timeframe: "1h"
  entry_lookback: 20
  atr_period: 14
  atr_tp_multiple: 3.0
  atr_sl_multiple: 1.5
  weight: 0.15
  per_asset: {}

supertrend:
  enabled: true
  live_eligible: false
  timeframe: "1h"
  atr_period: 10
  atr_multiplier: 3.0
  tp_percent: 4.0
  sl_percent: 2.0
  weight: 0.15
  per_asset: {}
```

### `config/param_grids.yaml` — Ajouter 3 sections
```yaml
bollinger_mr:
  wfo:
    is_days: 180
    oos_days: 60
    step_days: 60
  default:
    bb_period: [15, 20, 25, 30]
    bb_std: [1.5, 2.0, 2.5]
    sl_percent: [3.0, 5.0, 7.0, 10.0]

donchian_breakout:
  wfo:
    is_days: 180
    oos_days: 60
    step_days: 60
  default:
    entry_lookback: [20, 30, 40, 55]
    atr_period: [10, 14, 20]
    atr_tp_multiple: [2.0, 3.0, 4.0]
    atr_sl_multiple: [1.0, 1.5, 2.0]

supertrend:
  wfo:
    is_days: 180
    oos_days: 60
    step_days: 60
  default:
    atr_period: [7, 10, 14, 20]
    atr_multiplier: [2.0, 2.5, 3.0, 4.0]
    tp_percent: [3.0, 4.0, 6.0, 8.0]
    sl_percent: [1.5, 2.0, 3.0, 4.0]
```

---

## Étape 9 — Tests (`tests/test_new_strategies.py`)

~45 tests couvrant :

**Bollinger MR (12 tests)** :
- Calcul Bollinger Bands correct (SMA ± std)
- Signal LONG quand close < lower band
- Signal SHORT quand close > upper band
- Pas de signal quand close entre les bandes
- check_exit : close croise SMA → "signal_exit"
- check_exit : close n'a pas croisé → None
- Score proportionnel à la distance aux bandes
- min_candles correct
- compute_indicators retourne structure valide
- get_current_conditions format correct
- get_params retourne les bons params
- Parité fast engine vs normal

**Donchian Breakout (10 tests)** :
- Rolling high/low corrects
- Signal LONG quand close > canal haut
- Signal SHORT quand close < canal bas
- Pas de signal dans le canal
- TP/SL = ATR multiples corrects
- min_candles correct
- compute_indicators structure valide
- get_current_conditions format correct
- get_params retourne les bons params
- Parité fast engine vs normal

**SuperTrend (12 tests)** :
- Calcul SuperTrend correct (direction flips)
- Signal LONG sur flip DOWN→UP
- Signal SHORT sur flip UP→DOWN
- Pas de signal sans flip
- TP/SL % fixes corrects
- Direction stable quand pas de flip
- min_candles correct
- compute_indicators structure valide
- get_current_conditions format correct
- get_params retourne les bons params
- Parité fast engine vs normal

**Registry/Config (6 tests)** :
- 3 stratégies dans STRATEGY_REGISTRY
- create_strategy_with_params fonctionne pour les 3
- Config Pydantic validation (bounds)
- get_params_for_symbol avec per_asset
- factory create_strategy / get_enabled_strategies
- Grids chargés correctement depuis param_grids.yaml

---

## Vérification

1. `uv run python -m pytest tests/ -x -q` — tous les tests passent (330 existants + ~50 nouveaux)
2. `uv run python -m scripts.optimize --strategy bollinger_mr --symbol BTC/USDT -v`
3. `uv run python -m scripts.optimize --strategy donchian_breakout --symbol BTC/USDT -v`
4. `uv run python -m scripts.optimize --strategy supertrend --symbol BTC/USDT -v`

---

## Points d'attention

- **Pas de trend_filter_timeframe** pour les 3 stratégies 1h → pas de filtre 15m
- **Bollinger TP dynamique** : seule stratégie où le TP n'est pas un % fixe. `evaluate()` met tp_price très éloigné (pas la SMA), et `check_exit()` gère la sortie réelle sur SMA crossing. Parité entre BacktestEngine et fast engine.
- **Donchian rolling_high/low** : `_rolling_max` utilise `arr[i-window:i]` qui **exclut l'élément courant** (vérifié ligne 264 indicator_cache.py). Donc `close[i] > rolling_high[i]` = "close dépasse le plus haut des N bougies **précédentes**" — signal valide, pas de shift nécessaire.
- **SuperTrend = itératif** : boucle Python dans indicators.py, mais rapide (~5ms/48k points). Pré-calculé dans le cache pour chaque variante (atr_period, multiplier)
- **ATR multi-period** : Donchian et SuperTrend utilisent `atr_period` variable (pas fixe à 14 comme les stratégies 5m). Le cache doit stocker un ATR par period.
- **WFO timeframe 1h** : `walk_forward.py:337-339` lit `default_cfg.timeframe` depuis le STRATEGY_REGISTRY — charge automatiquement les candles 1h. Pas de hardcode 5m.
- **_check_exit index i** : l'interface existante `_check_exit(strategy_name, cache, i, direction, entry_price, params)` passe déjà `cache` et `i` — accès `cache.bb_sma[period][i]` trivial.
- **WFO fenêtres 1h** : IS=180j, OOS=60j, step=60j → ~27 fenêtres sur 5.5 ans. Plus de données = meilleures estimations.
- **live_eligible: false** pour les 3 → paper trading uniquement jusqu'à validation Grade A/B
