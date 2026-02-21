# Sprint Multi-Timeframe WFO — 21 février 2026

## Objectif

Permettre au WFO d'optimiser le timeframe pour toutes les stratégies (10 ajoutées aux 6 existantes).

## Contexte

6 stratégies avaient déjà `timeframe` dans leur grille WFO : boltrend, grid_boltrend, envelope_dca, envelope_dca_short, grid_atr, grid_trend.

10 stratégies avaient un timeframe fixe (non optimisable) :
- 4 scalp 5m : vwap_rsi, momentum, funding, liquidation
- 3 swing 1h : bollinger_mr, donchian_breakout, supertrend
- 3 grid 1h : grid_range_atr, grid_funding, grid_boltrend (étendu de ["1h"] à ["1h", "4h"])

## Changements

### 1. param_grids.yaml

Ajout de `timeframe` à 10 stratégies :

| Stratégie | Timeframes | Impact combos |
|-----------|-----------|---------------|
| vwap_rsi | ["5m", "15m"] | ×2 |
| momentum | ["5m", "15m"] | ×2 |
| funding | ["5m", "15m"] | ×2 |
| liquidation | ["5m", "15m"] | ×2 |
| bollinger_mr | ["1h", "4h"] | ×2 |
| donchian_breakout | ["1h", "4h"] | ×2 |
| supertrend | ["1h", "4h"] | ×2 |
| grid_range_atr | ["1h", "4h"] | ×2 |
| grid_funding | ["1h", "4h"] | ×2 |
| grid_boltrend | ["1h", "4h"] | ×2 (était ["1h"]) |

**Exclusion** : grid_multi_tf (filtre 4h interne baked-in, incompatible).

### 2. walk_forward.py — 4 fixes

1. **Chargement dynamique TFs** : extrait les valeurs `timeframe` du param grid avant de charger les candles depuis la DB. Les TFs présents en DB (5m, 15m, 1h) sont chargés directement.

2. **Resampling intelligent (fast engine)** : avant, la condition `if tf != main_tf and tf != "1h"` cassait pour les TFs déjà en mémoire (ex: 15m pour scalp). Maintenant : `if tf in candles_by_tf` → skip resampling.

3. **OOS evaluation** : même logique pour les candles OOS.

4. **Workers pool/séquentiel** : corrige un bug pré-existant où `_worker_main_tf` fixe était utilisé au lieu du `timeframe` du combo. Fix : `effective_tf = params.get("timeframe", _worker_main_tf)`.

### 3. tests/test_multi_timeframe.py

Test `test_grid_funding_no_timeframe` → `test_grid_funding_has_timeframe`.

## Architecture — Comment ça marche

- **5m/15m** : données déjà en DB (assets.yaml : [1m, 5m, 15m, 1h]), chargées directement
- **4h/1d** : pas en DB, resampleés depuis 1h via `resample_candles()` (existant)
- Le fast engine groupe les combos par TF et construit un `IndicatorCache` par groupe
- Chaque combo est évalué avec les bons candles pour son timeframe

## Tests

- 1559 passants, 0 régression
- Le segfault numpy (Python 3.13 + build_cache) est un problème pré-existant non lié

## Prochaine étape

Relancer les WFO sur les stratégies avec le nouveau paramètre timeframe pour voir si certains assets performent mieux en 15m ou 4h.
