# Sprint 58 — Parité Sprint 56 : 4 fixes réalisme sur 3 fast engines

**Date** : 28 février 2026
**Durée** : 1 session
**Status** : ✅ TERMINÉ

---

## Contexte

Sprint 56 avait introduit 4 fixes réalisme dans `_simulate_grid_common()` (moteur unifié utilisé par
grid_atr, grid_multi_tf, envelope_dca, grid_trend). Les 3 moteurs dédiés (`_simulate_grid_boltrend`,
`_simulate_grid_range`, `_simulate_grid_momentum`) n'avaient pas reçu ces fixes.

`_simulate_grid_trend` délègue entièrement à `_build_entry_prices()` + `_simulate_grid_common()` → déjà couvert, aucun changement nécessaire.

---

## Les 4 fixes Sprint 56

| Fix | Description | Implémentation |
|-----|-------------|---------------|
| 1 — Look-ahead bias | Indicateurs au candle `[i-1]`, pas `[i]` | `sma[i-1]`, `atr[i-1]`, `rolling_high[i-1]`, etc. + guard `if i == 0: continue` |
| 2 — Margin guard 70% | Bloquer nouvelles positions si marge utilisée ≥ 70% | `used_margin / total_equity >= max_margin_ratio` avant chaque entrée |
| 3 — Entry slippage | Slippage appliqué au prix d'entrée | LONG: `actual_ep = ep × (1 + slippage_pct)` / SHORT: `ep × (1 - slippage_pct)` |
| 4 — SL gap slippage | Fill midway si prix gap au-delà du SL | LONG: `exit_price -= 0.5 × max(0, exit_price - low[i])` |

---

## Récapitulatif par fonction

| Fonction | Fix 1 | Fix 2 | Fix 3 | Fix 4 | Tests |
|----------|-------|-------|-------|-------|-------|
| `_simulate_grid_trend` | ✅ déjà (via common) | ✅ déjà | ✅ déjà | ✅ déjà | — |
| `_simulate_grid_boltrend` | ✅ ajouté | ✅ ajouté | ✅ ajouté | ✅ ajouté | 13/13 ✅ |
| `_simulate_grid_range` | ✅ ajouté | ✅ ajouté | ✅ ajouté | ✅ ajouté | 17/17 ✅ |
| `_simulate_grid_momentum` | ✅ ajouté | ✅ ajouté | ✅ ajouté | ✅ ajouté | — |

---

## Détail des modifications

### `_simulate_grid_boltrend()`

**Init** :
```python
max_margin_ratio = bt_config.max_margin_ratio  # Sprint 56: margin guard
used_margin = 0.0  # Sprint 56: margin guard tracking
```

**Fix 1** : `long_ma[i]` → `long_ma[i-1]`, `atr_arr[i]` → `atr_arr[i-1]`

**Fix 2+3 — Level 0 (breakout entry)** :
```python
total_equity = capital + used_margin
if total_equity <= 0 or used_margin / total_equity < max_margin_ratio:
    actual_ep0 = ep0 * (1 + slippage_pct) if direction == 1 else ep0 * (1 - slippage_pct)
    ...
    used_margin += margin
```

**Fix 2+3 — DCA filling** :
```python
total_equity = capital + used_margin
if total_equity > 0 and used_margin / total_equity >= max_margin_ratio:
    break
actual_ep = ep * (1 + slippage_pct) if direction == 1 else ep * (1 - slippage_pct)
...
used_margin += margin
```

**Fix 4 — Exit SL** :
```python
if exit_reason == "sl_global":
    gap = max(0.0, exit_price - lows[i]) if direction == 1 else max(0.0, highs[i] - exit_price)
    exit_price -= 0.5 * gap if direction == 1 else -0.5 * gap
used_margin = 0.0  # reset
```

---

### `_simulate_grid_range()`

Stratégie bidirectionnelle (LONG + SHORT indépendants). Particularités :
- Positions fermées individuellement (pas en bloc) → `used_margin -= margin_to_return` par position
- Fix 4 appliqué individuellement avant `_calc_grid_pnl()`

**Init** :
```python
max_margin_ratio = bt_config.max_margin_ratio
used_margin = 0.0
```

**Fix 4 — SL individuel** :
```python
if exit_reason == "sl":
    if direction == 1:
        gap = max(0.0, exit_price - cache.lows[i])
        exit_price -= 0.5 * gap
    else:
        gap = max(0.0, cache.highs[i] - exit_price)
        exit_price += 0.5 * gap
margin_to_return = ep * qty / leverage
capital += margin_to_return
used_margin -= margin_to_return  # Sprint 56
```

**Fix 1+2+3 — Entry** :
```python
if i == 0:
    continue  # guard look-ahead
prev_sma = sma_arr[i - 1]
prev_atr = atr_arr[i - 1]
spacing = prev_atr * spacing_mult
# LONG:
ep = prev_sma - (lvl + 1) * spacing
if ep > 0 and cache.lows[i] <= ep:
    total_equity = capital + used_margin
    if total_equity > 0 and used_margin / total_equity >= max_margin_ratio:
        break
    actual_ep = ep * (1 + slippage_pct)
    ...
    used_margin += margin
# SHORT: actual_ep = ep * (1 - slippage_pct)
```

---

### `_simulate_grid_momentum()`

Stratégie Donchian breakout + DCA pullback. Particularités :
- `start_idx = max(donchian_period, atr_period) + 1 ≥ 31` → pas de guard `i==0` nécessaire
- Deux types de sortie : `sl_global` et `trail_stop` → Fix 4 appliqué aux deux

**Fix 1** : `rolling_high[i]` → `rolling_high[i-1]`, `rolling_low[i]` → `rolling_low[i-1]`, `atr_arr[i]` → `atr_arr[i-1]`, `vol_sma[i]` → `vol_sma[i-1]`

**Fix 4** :
```python
if exit_reason in ("sl_global", "trail_stop"):
    if direction == 1:
        gap = max(0.0, exit_price - lows[i])
        exit_price -= 0.5 * gap
    else:
        gap = max(0.0, highs[i] - exit_price)
        exit_price += 0.5 * gap
used_margin = 0.0  # reset
```

---

## Tests mis à jour

5 tests dans `tests/test_grid_range_atr.py` décalés candle 0 → candle 1 (conséquence du guard `if i == 0: continue`) :

| Test | Changement |
|------|------------|
| `test_individual_tp_long` | `lows[0]=96.0` → `lows[1]=96.0` |
| `test_individual_tp_short` | `highs[0]=104.0` → `highs[1]=104.0` |
| `test_level_reopen_after_close` | `lows[0]=96.0` → `lows[1]=96.0`, `lows[5]=96.0` → `lows[6]=96.0` |
| `test_fees_tp_maker_sl_taker` | `lows_tp[0]=96.0` → `lows_tp[1]=96.0` |
| `test_long_short_same_candle_close` | `lows[0]=95.0,highs[0]=104.0` → `lows[1]=95.0,highs[1]=104.0` |

---

## Résultat

- **0 nouveaux tests, 5 tests mis à jour**
- **2081 tests, 2081 passants**, 0 régression
- Tous les fast engines backtesting ont maintenant les 4 fixes réalisme Sprint 56
