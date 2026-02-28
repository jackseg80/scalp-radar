# Sprint 62 — Donchian Breakout : Extension de trend_follow_daily

**Date** : 28 février 2026

## Contexte

L'EMA crossover (Sprint 61) a échoué en WFO (0% consistency OOS). On ajoute un mode d'entrée Donchian breakout au moteur `_simulate_trend_follow()` existant. Pas de nouvelle stratégie — c'est un `entry_mode` supplémentaire. Le Donchian breakout (système Turtle Traders) entre quand le prix casse le plus haut/bas des N derniers jours — plus réactif que l'EMA cross.

## Fichiers modifiés (5) + tests

### 1. `backend/strategies/trend_follow_daily.py` — Config dataclass

Ajout de 3 champs à `TrendFollowDailyConfig` :

```python
# Entry mode
entry_mode: str = "donchian"          # "ema_cross" ou "donchian"
# Donchian params (ignorés si entry_mode == "ema_cross")
donchian_entry_period: int = 50       # Breakout N-day high/low
donchian_exit_period: int = 20        # Canal de sortie (plus court)
```

`exit_mode` gagne une 3e valeur valide : `"channel"` (en plus de `"trailing"` et `"signal"`).

### 2. `backend/optimization/indicator_cache.py:308-311` — Lookbacks trigger

Ajout de 2 conditions au bloc lookbacks existant :

```python
if "donchian_entry_period" in param_grid_values:
    lookbacks.update(param_grid_values["donchian_entry_period"])
if "donchian_exit_period" in param_grid_values:
    lookbacks.update(param_grid_values["donchian_exit_period"])
```

Les helpers `_rolling_max()`/`_rolling_min()` existaient déjà (lignes 630-662). Le bloc EMA utilise `.get("ema_fast", [])` — fonctionne déjà sans les clés EMA.

### 3. `backend/optimization/walk_forward.py:436` — Indicator group keys

```python
"trend_follow_daily": ["ema_fast", "ema_slow", "donchian_entry_period", "donchian_exit_period", "adx_period", "atr_period"],
```

### 4. `backend/optimization/fast_multi_backtest.py` — Moteur `_simulate_trend_follow()`

**Params** : `ema_fast`/`ema_slow` rendus optionnels (`.get()`), ajout `entry_mode`, `donchian_entry_period`, `donchian_exit_period`

**Déduplication** : `entry_mode=donchian` → EMA params normalisés (sans effet) ; `exit_mode=channel + entry_mode=ema_cross` → fallback trailing

**Arrays conditionnels** : EMA ou rolling_high/low chargés selon `entry_mode`

**Warmup** : adapté pour Donchian (`donchian_entry_period` au lieu de `ema_slow_period`)

**PHASE 1** : refactorisé — ADX check partagé, puis `if entry_mode == "ema_cross"` vs `elif entry_mode == "donchian"` (breakout sur `closes[prev] > rolling_high_entry[prev]`)

**PHASE 2** : nouveau bloc `exit_mode == "channel"` (priorité 3, après trailing, avant signal) :
```python
if exit_mode == "channel" and i > entry_candle:
    ch = rolling_low_exit[prev]  # LONG
    if not math.isnan(ch) and lows[i] <= ch:
        exit_price = ch * (1 - slippage_pct)
        # close position...
```

**Signal exit guard** : `and entry_mode == "ema_cross"` ajouté (pas d'EMA inverse en mode Donchian)

Anti-look-ahead confirmé : `_rolling_max()` exclut candle courante → `rolling_high[i] = max(highs[i-N:i])`

### 5. `config/param_grids.yaml` — Grille Donchian

```yaml
trend_follow_daily:
  wfo:
    is_days: 365
    oos_days: 120
    step_days: 60
    embargo_days: 1
  default:
    timeframe: ["1d"]
    entry_mode: ["donchian"]
    donchian_entry_period: [20, 50, 100]
    donchian_exit_period: [10, 20]
    adx_period: [14]
    adx_threshold: [0, 20]
    atr_period: [14]
    trailing_atr_mult: [3.0, 4.0, 5.0]
    exit_mode: ["trailing", "channel"]
    sl_percent: [10.0]
    position_fraction: [0.2, 0.3, 0.5]
    cooldown_candles: [3]
    sides: [["long"], ["long", "short"]]
```

**432 combos** : 3×2×2×3×2×1×3×1×2

### 6. `tests/test_trend_follow_daily.py` — 7 nouveaux tests

- `_make_cache_for_trend()` : ajout `rolling_high`/`rolling_low` params
- `_DEFAULT_PARAMS` : ajout `"entry_mode": "ema_cross"` explicite
- `_donchian_params()` : nouveau helper Donchian
- `_setup_donchian_breakout_long()` : setup helper

| # | Classe | Vérifie |
|---|--------|---------|
| 18 | `TestDonchianEntryLong` | `close > rolling_high` → entrée LONG |
| 19 | `TestDonchianEntryShort` | `close < rolling_low` → entrée SHORT |
| 20 | `TestChannelExit` | LONG sort quand `lows < rolling_low_exit` |
| 21 | `TestDonchianADXFilter` | Breakout bloqué si `ADX < threshold` |
| 22 | `TestDonchianNoLookAhead` | Canal inatteignable → 0 trade |
| 23 | `TestDonchianDeduplication` | `entry_mode=donchian` ignore ema_fast/ema_slow |
| 24 | `TestDonchianConfigDefaults` | Config a les nouveaux champs, defaults corrects |

## Résultats

- **37/37 tests** `test_trend_follow_daily.py` — 0 régression
- **2151 tests, 2144 passants** (7 pré-existants non liés)

## Prochaine étape

```bash
uv run python -m scripts.optimize --strategy trend_follow_daily --symbol BTC/USDT --subprocess --force-timeframe 1d
```
