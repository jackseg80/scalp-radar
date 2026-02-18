# Hotfix 33b — Audit de parité grid_boltrend

**Date** : 18 février 2026
**Status** : ✅ Terminé
**Tests** : 13 nouveaux, 1322 total (0 régression)

---

## Contexte

Après l'implémentation de la 16e stratégie (grid_boltrend, Sprint 33) et le hotfix TP inverse
(Hotfix 33a), un audit de fiabilité a été conduit pour valider la cohérence entre le fast engine
(`_simulate_grid_boltrend`) et le moteur event-driven (`MultiPositionEngine.run`).

L'audit a détecté une divergence PnL de **+31.73%** entre les deux moteurs — signal de 3 bugs.

---

## 5 vérifications d'audit

### Test 1 — Parité fast engine vs event-driven (CRITIQUE)

Objectif : nombre de trades ±1, PnL ±5%, directions concordantes.

Résultat avant fix : **divergence 31.73%** (échec intentionnel détectant les bugs).
Résultat après fix : divergence 2.62% (résiduel structurel acceptable).

### Test 2 — Look-ahead bias

Objectif : les N premiers trades avec N candles = les N premiers trades avec N+100 candles.
Résultat : **PASS** — aucun look-ahead bias.

### Test 3 — Frais correctement appliqués

Objectif : PnL avec frais < PnL sans frais, magnitude cohérente (~0.12% par trade à 6× levier).
Résultat : **PASS** après correction des fees signal_exit.

### Test 4 — Remplissage multi-niveaux réaliste

Objectif : candle étroite → 1 niveau, candle large → plusieurs niveaux.
Résultat : **PASS**.

### Test 5 — Script diagnostic trade log

Objectif : `scripts/grid_boltrend_diagnostic.py` produit un trade log détaillé (10 premiers trades).
Résultat : **PASS** après correction exit_price.

---

## Bugs identifiés et corrigés

### Bug 1 — exit_price signal_exit incorrect

**Fichier** : `backend/optimization/fast_multi_backtest.py`

Le fast engine utilisait `sma_val` (valeur de la SMA au moment de la sortie) comme prix de sortie
pour les exits `signal_exit`. Or le TP inverse grid_boltrend se déclenche quand `close < bb_sma`
(LONG) — mais le prix d'exécution réel est `close`, pas `sma_val`.

Conséquence : pour un LONG, `sma_val < close` toujours après le breakout → sortie optimiste →
PnL WFO surestimé.

```python
# AVANT (bugué)
exit_price = sma_val if is_green else sl_price

# APRÈS (corrigé)
exit_price = close_i if is_green else sl_price
```

### Bug 2 — Fees signal_exit incorrectes

**Fichier** : `backend/optimization/fast_multi_backtest.py`

Le fast engine appliquait `maker_fee + 0 slippage` pour les exits `signal_exit`. Or une sortie
au signal de marché est exécutée en taker (market order), donc `taker_fee + slippage_pct`.

```python
# AVANT (bugué)
if exit_reason == "signal_exit":
    fee = maker_fee
    slip = 0.0

# APRÈS (corrigé)
# signal_exit = clôture marché (taker fee + slippage)
fee = taker_fee
slip = slippage_pct
```

### Bug 3 — Double-comptage entry_fee (impact toutes stratégies grid)

**Fichier** : `backend/backtesting/multi_engine.py`

Le moteur event-driven déduisait `capital -= pos.entry_fee` immédiatement après l'ouverture
d'une position. Or `pos.entry_fee` est déjà inclus dans `trade.net_pnl` via
`GridPositionManager.close_all_positions()` (qui calcule `fee_cost = total_entry_fees + exit_fee`).

Résultat : double-comptage → sous-estimation systématique du capital → trades sous-dimensionnés.
**Affectait toutes les stratégies grid** (grid_atr, grid_range_atr, envelope_dca, etc.).

```python
# AVANT (bugué)
if pos is not None:
    positions.append(pos)
    capital -= pos.entry_fee  # double-comptage !

# APRÈS (corrigé)
if pos is not None:
    positions.append(pos)
    # entry_fee déjà incluse dans trade.net_pnl via close_all_positions()
```

---

## Fichiers créés

- `tests/test_grid_boltrend_parity.py` — 13 tests organisés en 5 classes :
  - `TestParity` (4 tests) : trade count, directions, PnL ±5%, short
  - `TestLookAheadBias` (2 tests) : fast engine + event-driven
  - `TestFees` (3 tests) : réduction, magnitude, slippage
  - `TestMultiLevelFilling` (3 tests) : candle étroite, large, count exact
  - `TestDiagnosticTradeLog` (1 test) : trade log print

- `scripts/grid_boltrend_diagnostic.py` — version instrumentée de `_simulate_grid_boltrend`
  collectant des `TradeLog` dataclasses (trade_idx, direction, n_positions, entry_candle_idx,
  exit_candle_idx, avg_entry_price, exit_price, exit_reason, gross_pnl, total_entry_fees,
  exit_fee, slippage_cost, net_pnl)

## Fichiers modifiés

- `backend/optimization/fast_multi_backtest.py` — Bug 1 + Bug 2 corrigés
- `backend/backtesting/multi_engine.py` — Bug 3 corrigé

---

## Résultat

| Métrique | Avant fix | Après fix |
|----------|-----------|-----------|
| Divergence PnL | +31.73% | 2.62% |
| Tests parité | 12/13 PASS | 13/13 PASS |
| Tests totaux | — | 1322 (0 régression) |
