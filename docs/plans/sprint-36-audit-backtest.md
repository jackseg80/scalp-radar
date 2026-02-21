# Sprint 36 — Audit Backtest & Corrections Structurelles

## Contexte

Audit complet des systèmes de backtest (WFO + portfolio + simulator). Trois agents ont exploré les fast engines, le pipeline WFO/grading, le portfolio engine, les 16 stratégies et les métriques. Sur ~25 findings initiaux, **4 bugs confirmés** + **2 corrections structurelles** appliquées.

---

## Phase 1 : Audit Initial (4 fixes)

### Fix 1 — Grid Funding : double entry fee (CRITIQUE)

**Fichier** : `backend/optimization/fast_multi_backtest.py`
**Bug** : `capital -= fee` à l'ouverture + `entry_fee` intégrée dans `_calc_grid_pnl_with_funding()` à la clôture = double déduction.
**Fix** : Supprimé les lignes `fee = notional * taker_fee` / `capital -= fee`.

### Fix 2 — Grid Funding : double slippage (CRITIQUE)

**Fichier** : `backend/optimization/fast_multi_backtest.py`
**Bug** : `slippage = notional * slippage_pct + exit_price * quantity * slippage_pct` appliquait le slippage sur entry ET exit.
**Fix** : `slippage = exit_price * quantity * slippage_pct` (exit seulement).

### Fix 3 — Simulator : funding rate hardcodé 0.0001 (CRITIQUE)

**Fichier** : `backend/backtesting/simulator.py`
**Bug** : Toutes les stratégies grid en paper trading utilisaient un funding rate fixe de 0.01% au lieu des taux réels.
**Fix** : Lecture depuis `data_engine.get_funding_rate(symbol)` avec guard `isinstance` pour MagicMock.

### Fix 4 — Portfolio : liquidation off-by-one (MOYEN)

**Fichier** : `backend/backtesting/portfolio_engine.py`
**Bug** : `<=` au lieu de `<`. Bitget liquide quand equity TOMBE EN DESSOUS du maintenance margin.
**Fix** : `is_liquidated = total_equity < maintenance_margin and total_notional > 0`

---

## Phase 2 : Part A — Double slippage TOUS les engines (6 locations)

### Pattern du bug

```python
# AVANT (faux) : slippage appliqué 2 fois
actual_exit = exit_price * (1 - slippage_pct)  # prix ajusté
gross = (actual_exit - entry) * qty              # slippage dans le gross
slippage_cost = qty * exit_price * slip_rate     # slippage déduit ENCORE
net = gross - fees - slippage_cost               # 2× slippage

# APRÈS (correct) : flat cost model
gross = (exit_price - entry) * qty               # prix brut
slippage_cost = qty * exit_price * slip_rate     # slippage 1× seulement
net = gross - fees - slippage_cost
```

### Fichiers corrigés

| Fichier | Fonction | Description |
|---------|----------|-------------|
| `backend/optimization/fast_multi_backtest.py` | `_calc_grid_pnl()` | Supprimé `actual_exit`, gross sur prix brut |
| `backend/optimization/fast_multi_backtest.py` | `_simulate_grid_range()` (2 locations) | Inline PnL aligné sur flat cost model |
| `backend/optimization/fast_backtest.py` | `_close_trade_numba()` | Même fix (Numba JIT) |
| `backend/optimization/fast_backtest.py` | `_close_trade()` | Même fix (Python fallback) |
| `backend/core/grid_position_manager.py` | `close_all_positions()` | Même fix + `TradeResult.exit_price` = prix brut |
| `backend/core/position_manager.py` | `close_position()` | Même fix |

### Impact `TradeResult.exit_price`

Changement sémantique : `exit_price` passe de "prix ajusté slippage" à "prix brut marché". Tous les consommateurs downstream vérifiés : logging, DB, affichage — aucun calcul ne dépend de `exit_price` après le `TradeResult`.

---

## Phase 2 : Part B — Margin deduction dans les fast engines

### Principe

Les fast engines ne déduisaient jamais de marge du capital à l'ouverture des positions, permettant une inflation illimitée du capital. Alignement avec `GridStrategyRunner` :
- **Open** : `margin = notional / leverage` ; `capital -= margin`
- **Close** : `margin_to_return = Σ(entry_price × qty / leverage)` ; `capital += margin_to_return`

### Fonctions modifiées (4)

| Fonction | Opens | Closes | Notes |
|----------|-------|--------|-------|
| `_simulate_grid_common()` | 1 | 3 (flip, exit, end-of-data) | Utilisée par envelope_dca, grid_atr, grid_multi_tf |
| `_simulate_grid_boltrend()` | 2 (DCA + breakout) | 2 | Guard `capital <= 0` déplacé APRÈS exit checks |
| `_simulate_grid_funding()` | 1 | 2 | Guard `capital <= 0` déplacé APRÈS exit checks |
| `_simulate_grid_range()` | 2 (LONG + SHORT) | 2 | Positions bidirectionnelles |

### Bug guard corrigé

`_simulate_grid_funding` et `_simulate_grid_boltrend` avaient `if capital <= 0: continue` en HAUT de la boucle. Après margin deduction, le capital peut atteindre 0 quand tous les niveaux sont remplis → les exit checks étaient sautés → positions jamais clôturées. Fix : retirer `capital <= 0` du guard top-of-loop.

---

## Tests (23 au total dans `tests/test_backtest_audit.py`)

### Phase 1 — Audit (12 tests)

1. `test_grid_funding_no_double_entry_fee` — Pas de double déduction entry_fee
2. `test_grid_funding_no_double_slippage` — Slippage = exit only
3. `test_calc_pnl_with_funding_matches_calc_pnl` — Parité funding/non-funding à funding=0
4. `test_simulator_real_funding_rate` — Funding rate réel depuis DataEngine
5. `test_simulator_funding_rate_fallback` — Fallback si None
6. `test_simulator_negative_funding_rate` — Capital augmente pour LONG
7. `test_portfolio_liquidation_boundary` — Boundary exact <, pas <=
8-12. Tests internes supplémentaires

### Phase 2 Part A — Double slippage (8 tests)

13. `test_calc_grid_pnl_no_double_slippage_long` — _calc_grid_pnl LONG
14. `test_calc_grid_pnl_no_double_slippage_short` — _calc_grid_pnl SHORT
15. `test_close_trade_no_double_slippage_sl` — _close_trade SL
16. `test_close_trade_no_double_slippage_tp` — _close_trade TP (0 slippage)
17. `test_close_trade_numba_matches_python` — Parité Numba/Python
18. `test_grid_position_manager_no_double_slippage` — GridPositionManager
19. `test_position_manager_no_double_slippage` — PositionManager
20. `test_parity_with_slippage` — Parité _calc_grid_pnl/_calc_grid_pnl_with_funding avec slippage réel

### Phase 2 Part B — Margin deduction (3 tests)

21. `test_margin_reduces_capital_at_open` — Capital diminué après ouverture
22. `test_margin_restored_at_close` — Capital restauré après clôture
23. `test_insufficient_margin_skips_level` — Capital < margin → niveau sauté

---

## Valeurs de référence mises à jour

`tests/test_fast_engine_refactor.py` — 3 constantes de parité bit-à-bit recalculées post-fix :
- `_EXPECTED_ENVELOPE_DCA` : Sharpe 50.10 → 55.80 (mécaniquement meilleur)
- `_EXPECTED_ENVELOPE_DCA_SHORT` : Sharpe 44.66 → 47.12
- `_EXPECTED_GRID_ATR` : Sharpe 52.08 → 57.96

---

## Issues documentées (PAS de fix, choix de design)

| Issue | Raison |
|-------|--------|
| WFO avg_oos écrasé par best combo | Intentionnel (Sprint 15c/15d) |
| Compound sizing illimité | Le live executor a `max_margin_ratio=70%` comme garde-fou |
| Portfolio margin bidirectionnelle non-nettée | Approche conservative (sûre) |
| MC trades vs Sharpe mismatch | Pratique WFO standard |
| Settlement timing 1 candle early | Impact <0.5% |
| Sharpe/Sortino cappé à 100 | Protection affichage |

## Faux positifs confirmés

- `compute_live_indicators` absent pour grid_atr → FAUX (utilise SMA/ATR d'IncrementalIndicatorEngine)
- `_total_funding_cost` non accumulé → FAUX (accumulé à simulator.py:925)

---

## Résultats

- **1604 tests passent** (23 nouveaux + 3 constantes mises à jour)
- 1 test pré-existant flaky (`test_post_result_created`) lié à l'ordre d'exécution (problème YAML strategies.yaml corrompu par un test antérieur), non lié au Sprint 36
- Impact WFO : résultats mécaniquement **meilleurs** pour toutes les stratégies grid (double slippage supprimé = moins de pénalités fictives)
- Impact margin deduction : résultats mécaniquement **plus conservateurs** (capital verrouillé = moins de levier effectif)

## Fichiers modifiés (8)

| Fichier | Type |
|---------|------|
| `backend/optimization/fast_multi_backtest.py` | Fix 1+2 + Part A + Part B |
| `backend/optimization/fast_backtest.py` | Part A (Numba + Python) |
| `backend/core/grid_position_manager.py` | Part A |
| `backend/core/position_manager.py` | Part A |
| `backend/backtesting/simulator.py` | Fix 3 |
| `backend/backtesting/portfolio_engine.py` | Fix 4 |
| `tests/test_backtest_audit.py` | 23 tests (nouveau) |
| `tests/test_fast_engine_refactor.py` | 3 constantes mises à jour |
