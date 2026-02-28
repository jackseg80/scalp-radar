# Sprint 61 — trend_follow_daily : Trend Following EMA Cross Daily

## Contexte

Scalp-radar n'a qu'une stratégie viable (grid_atr, mean-reversion DCA 1h). On ajoute `trend_follow_daily`, une stratégie trend following EMA cross sur daily avec position unique et trailing stop ATR. **Fast engine only** — pas de live runner tant que le WFO n'a pas prouvé la viabilité.

Deux corrections critiques intégrées dans le moteur :
1. **Day 0 Bug Fix** : structure PHASE 1 (entrée) → PHASE 2 (sortie) par candle — le SL est vérifié le jour même de l'entrée
2. **Trailing init look-ahead fix** : trailing initialisé à partir de `entry_price` (pas `highs[i]`/`lows[i]` inconnus à l'ouverture)

---

## Fichiers à modifier/créer (9 fichiers)

| Fichier | Action |
|---------|--------|
| `backend/strategies/trend_follow_daily.py` | **Créer** — Config `@dataclass` |
| `backend/optimization/indicator_cache.py` | Modifier — Étendre `build_cache()` |
| `backend/optimization/fast_multi_backtest.py` | Modifier — `_simulate_trend_follow()` + dispatch |
| `backend/optimization/__init__.py` | Modifier — Registry + `MULTI_BACKTEST_STRATEGIES` |
| `backend/optimization/walk_forward.py` | Modifier — `_INDICATOR_PARAMS` + routage |
| `config/param_grids.yaml` | Modifier — Nouvelle section |
| `config/strategies.yaml` | Modifier — Nouvelle entrée |
| `tests/test_trend_follow_daily.py` | **Créer** — 17 tests |
| `backend/core/config.py` | **Pas touché** — config dans strategies/ |

---

## Étape 1 : Config — `backend/strategies/trend_follow_daily.py` (nouveau)

`@dataclass` avec les paramètres de la spec. Le WFO n'utilise que `config_cls().timeframe` (ligne 494 de walk_forward.py) — pas besoin de BaseModel.

```python
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class TrendFollowDailyConfig:
    name: str = "trend_follow_daily"
    enabled: bool = False
    live_eligible: bool = False
    timeframe: str = "1d"
    ema_fast: int = 9
    ema_slow: int = 50
    adx_period: int = 14
    adx_threshold: float = 20.0
    atr_period: int = 14
    trailing_atr_mult: float = 4.0
    exit_mode: str = "trailing"
    sl_percent: float = 10.0
    cooldown_candles: int = 3
    sides: list[str] = field(default_factory=lambda: ["long"])
    leverage: int = 6
```

---

## Étape 2 : Registry — `backend/optimization/__init__.py`

### 2a. Problème de routage résolu : `MULTI_BACKTEST_STRATEGIES`

Le WFO (`_run_fast`) utilise `is_grid_strategy()` pour choisir entre `run_multi_backtest_from_cache` et `run_backtest_from_cache`. `trend_follow_daily` n'est PAS une grid strategy, mais son moteur est dans `fast_multi_backtest.py`.

**Solution** : séparer routage et sémantique. Ajouter :

```python
# Stratégies routées vers run_multi_backtest_from_cache (grid + moteurs autonomes)
MULTI_BACKTEST_STRATEGIES: set[str] = GRID_STRATEGIES | {"trend_follow_daily"}

def uses_multi_backtest(name: str) -> bool:
    """True si la stratégie utilise run_multi_backtest_from_cache."""
    return name in MULTI_BACKTEST_STRATEGIES
```

### 2b. Registry entry

```python
from backend.strategies.trend_follow_daily import TrendFollowDailyConfig
"trend_follow_daily": (TrendFollowDailyConfig, None),  # Fast engine only
```

Type annotation ajustée : `dict[str, tuple[type, type | None]]`

### 2c. Guard dans `create_strategy_with_params`

Avant `config_cls().model_dump()`, ajouter :
```python
if strategy_cls is None:
    raise ValueError(
        f"Stratégie '{strategy_name}' n'a pas de runner live "
        f"(fast engine uniquement)."
    )
```

### 2d. PAS dans GRID_STRATEGIES, PAS dans STRATEGIES_NEED_EXTRA_DATA

---

## Étape 3 : IndicatorCache — `backend/optimization/indicator_cache.py`

### 3a. ATR multi-period (ligne ~430)
Ajouter `"trend_follow_daily"` à la condition existante :
```python
if strategy_name in ("donchian_breakout", "supertrend", "grid_atr", "grid_multi_tf",
                      "grid_trend", "grid_range_atr", "grid_boltrend", "grid_momentum",
                      "trend_follow_daily"):
```

### 3b. EMA + ADX multi-period (ligne ~512)
Élargir la condition :
```python
if strategy_name in ("grid_trend", "trend_follow_daily"):
```

---

## Étape 4 : Fast Engine — `backend/optimization/fast_multi_backtest.py`

### 4a. Dispatch dans `run_multi_backtest_from_cache()` (ligne ~1635)

Ajouter avant le `else: raise ValueError` :
```python
elif strategy_name == "trend_follow_daily":
    trade_pnls, trade_returns, final_capital = _simulate_trend_follow(
        cache, params, bt_config,
    )
```

### 4b. Nouveau moteur `_simulate_trend_follow()` + `_close_trend_position()`

Moteur indépendant (PAS `_simulate_grid_common`). Code complet fourni dans la spec utilisateur avec les deux corrections Day 0 et trailing init.

Points clés du moteur :
- **Structure boucle** : PHASE 1 (entrée) → PHASE 2 (sortie) — pas de `continue` après entrée, le SL est vérifié le jour même
- **Trailing init** : basé sur `entry_price ± atr×mult` (pas `highs[i]`/`lows[i]`)
- **Déduplication** : `if exit_mode == "signal": trailing_atr_mult = 0.0`
- **Ordre exits** : SL fixe → trailing → signal inverse. SL TOUJOURS vérifié.
- **Force-close fin** : exclu de `trade_pnls` (Sprint 60 convention)
- **Sizing** : position unique, `notional = capital × leverage`, `margin_locked = capital`

---

## Étape 5 : Walk-Forward — `backend/optimization/walk_forward.py`

### 5a. `_INDICATOR_PARAMS` (ligne ~425)
```python
"trend_follow_daily": ["ema_fast", "ema_slow", "adx_period", "atr_period"],
```

### 5b. Routage `_run_fast()` (ligne ~1158)
Remplacer :
```python
from backend.optimization import is_grid_strategy
is_grid = is_grid_strategy(strategy_name)
# ...
if is_grid:
```
Par :
```python
from backend.optimization import uses_multi_backtest
use_multi = uses_multi_backtest(strategy_name)
# ...
if use_multi:
```

Note : `is_grid_strategy()` reste intact pour ses autres usages (embargo, MultiPositionEngine, etc.).

---

## Étape 6 : Config YAML

### `config/param_grids.yaml`
```yaml
trend_follow_daily:
  wfo:
    is_days: 365
    oos_days: 120
    step_days: 60
    embargo_days: 1
  default:
    timeframe: ["1d"]
    ema_fast: [5, 9, 12]
    ema_slow: [20, 50]
    adx_period: [14]
    adx_threshold: [0, 15, 20]
    atr_period: [14]
    trailing_atr_mult: [3.0, 4.0, 5.0]
    exit_mode: ["trailing", "signal"]
    sl_percent: [10.0]
    cooldown_candles: [3]
    sides: [["long"], ["long", "short"]]
```

### `config/strategies.yaml`
```yaml
trend_follow_daily:
  enabled: false
  live_eligible: false
  leverage: 6
```

---

## Étape 7 : Tests — `tests/test_trend_follow_daily.py` (nouveau, 17 tests)

| # | Test | Vérifie |
|---|------|---------|
| 1 | Entrée LONG sur EMA cross haussier | direction, entry_price avec slippage, 1 trade |
| 2 | Entrée SHORT sur EMA cross baissier | symétrique, slippage inversé |
| 3 | ADX filter bloque l'entrée | ADX < threshold → 0 trades, capital inchangé |
| 4 | ADX threshold = 0 (désactivé) | ADX < 20 mais threshold=0 → entrée |
| 5 | Trailing stop sort en profit | HWM tracking, exit ~trailing_level |
| 6 | SL fixe sort en perte | exit à sl_price |
| 7 | **Day 0 Bug** — SL touché le jour de l'entrée | Flash crash jour 0 → SL déclenché |
| 8 | Exit mode "signal" — sortie sur cross inverse | exit sur opens[i] |
| 9 | Cooldown empêche ré-entrée | signal ignoré pendant cooldown |
| 10 | Force-close exclu des métriques | trade_pnls=[], capital != initial |
| 11 | Sides = ["long"] bloque SHORT | bear_cross ignoré |
| 12 | DD guard stoppe la simulation | final_capital > 0 |
| 13 | Registry et config | STRATEGY_REGISTRY, pas dans GRID_STRATEGIES |
| 14 | IndicatorCache EMA/ADX | build_cache avec strategy_name="trend_follow_daily" |
| 15 | Déduplication exit_mode/trailing | signal+3.0 == signal+5.0 |
| 16 | **Trailing init = entry_price** | Pas highs[i], look-ahead fix |
| 17 | **Pas de look-ahead signaux** | ema[i-1], entrée open[i] |

---

## Vérification

```bash
# 1. Tests unitaires
uv run pytest tests/test_trend_follow_daily.py -v

# 2. Pas de régression
uv run pytest tests/ -x -q

# 3. Test WFO rapide (si données daily disponibles)
uv run python -m scripts.optimize --strategy trend_follow_daily --symbol BTC/USDT --subprocess --force-timeframe 1d
```

---

## Ce qu'on NE TOUCHE PAS

- `backend/execution/executor.py` — pas de live
- `backend/core/grid_position_manager.py` — pas de grid
- `backend/strategies/base_grid.py` — pas de grid
- `backend/backtesting/multi_engine.py` — event-driven, pas utilisé
- `backend/core/config.py` — config dans strategies/
- `report.py` / grading V2 — fonctionne déjà
