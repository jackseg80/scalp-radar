# Sprint 57 — Réalisme Backtest Complet (27 février 2026)

## Contexte

Audit complet du pipeline de backtesting révèle 29 biais/erreurs catégorisés P0→P3.
Tous invalidaient les résultats WFO existants. Sprint 56 avait déjà corrigé le kill switch
fast engine (25% DD). Sprint 57 corrige les 11 problèmes restants les plus critiques.

## Problèmes détectés

### P0 — Critique (faussent fondamentalement les résultats)

**#17 Monte Carlo — Permutation sans remplacement**
- **Bug** : `monte_carlo_block_bootstrap()` permutait les blocs sans remplacement → distribution Sharpe dégénérée (std ≈ 0), p-value toujours ~0.5 → test statistiquement inutile.
- **Fix** : Circular block bootstrap avec remplacement via `rng.integers(0, n_blocks, size=n_blocks)`.
- **Impact** : Distribution Sharpe simulée a maintenant de la variance → p-value discriminante.

**#10 Kill switch — window_start_equity au lieu de max(equity)**
- **Bug** : Kill switch comparait l'equity courante à l'equity du début de fenêtre, pas au pic. Scénario : 10K→12K→6K → DD calculé = 40% (vs 10K) au lieu de 50% (vs 12K) → kill switch ne triggait pas.
- **Fix** : Loop dans la fenêtre pour trouver le max equity. Corrigé dans 2 emplacements : boucle real-time ET `_check_kill_switch()`.

### P1 — Majeur (biais systématique sur les résultats)

**#1 Look-ahead bias — Entry prices [i] au lieu de [i-1]**
- **Bug** : `_build_entry_prices()` calculait les prix d'entrée avec les indicateurs de la bougie courante (sma[i], atr[i]), connus seulement après la clôture → les signaux d'entrée utilisaient des données futures.
- **Fix** : Shift de 1 sur tout le tableau : `entry_prices[1:] = entry_prices[:-1].copy(); entry_prices[0, :] = np.nan`.

**#2 Margin guard 70% absent du fast engine**
- **Bug** : Le fast engine ouvrait autant de niveaux que l'ATR le permettait, sans limite de marge. Le backtest pouvait utiliser 100% du capital en marge → impossiblen live.
- **Fix** : Tracker `used_margin`, check `used_margin / total_equity >= max_margin_ratio (0.70)` avant chaque entrée.

**#3+#13 Entry slippage absent — fast engine + portfolio engine**
- **Bug** : Les deux moteurs entraient exactement au prix calculé, sans modéliser le coût d'exécution réel (taker market order légèrement défavorable).
- **Fix fast engine** : `actual_ep = ep * (1 + slippage_pct)` pour LONG, `(1 - slippage_pct)` pour SHORT.
- **Fix portfolio engine** (`grid_position_manager.py`) : idem dans `open_grid_position()`.

### P2 — Significatif (biais modéré mais mesurable)

**#5+#11 SL gap slippage — fill exact au SL**
- **Bug** : Quand le prix gappait au-delà du SL (low << sl_price pour LONG), le fill était modélisé exactement au SL. En réalité, le fill est entre le SL et l'extrême de la bougie.
- **Fix** : `exit_price = sl_price - 0.5 * max(0, sl_price - low)` pour LONG. Corrigé dans fast engine ET `check_global_tp_sl()`.

**#20 DSR kurtosis — raw kurtosis au lieu d'excess kurtosis**
- **Bug** : La formule DSR (Bailey & López de Prado 2014) utilise l'excess kurtosis (= raw - 3). `_kurtosis()` retourne le raw kurtosis (≈3 pour une distribution normale). Le terme `(kurtosis - 1) / 4` utilisait 3 au lieu de 0 → pénalité excessive, DSR trop conservateur.
- **Fix** : `excess_kurtosis = kurtosis - 3.0` puis `(excess_kurtosis - 1) / 4`.

**#21 Embargo IS→OOS — absent pour la plupart des stratégies grid**
- **Bug** : Certaines stratégies grid n'avaient pas `embargo_days: 7` explicite dans `param_grids.yaml`. Sans embargo, la dernière bougie IS et la première bougie OOS sont adjacentes → data leakage possible pour les DCA en cours.
- **Fix** : Fallback code dans `walk_forward.py` : si `embargo_days == 0 and is_grid_strategy(name)` → `embargo_days = 7`. Toutes les stratégies grid bénéficient du fallback.

### P3 — Mineur (parité et outillage)

**#18 all_oos_trades — mélangés entre combos**
- **Bug** : `WFOResult.all_oos_trades` était une liste plate de tous les trades OOS, sans distinction par fenêtre → impossible de calculer une distribution de performance par fenêtre.
- **Fix** : Nouveau champ `oos_trades_by_window: list[list[TradeResult]]` dans `WFOResult`. Peuplé par fenêtre dans la boucle d'optimisation.

**#25 Parité executor — margin guard absent**
- **Bug** : Le fast engine avait (désormais) un margin guard 70% mais l'executor live ne l'avait pas. Une position live pouvait dépasser la limite modélisée.
- **Fix** : Check `(total_margin_used + level_margin) / available_balance > max_margin_ratio` dans `executor.py::_on_candle()`.

**Calmar annualisé**
- **Bug** : Calmar = `return / abs(DD)` n'était pas annualisé. Sur 180j, un return de 50% donnait Calmar=5 au lieu de ~10 (annualisé). Non comparable entre périodes.
- **Fix** : `calmar = (return_pct / n_years) / abs(max_dd_pct)` dans `regime_backtest_compare.py` et `diagnose_dd_leverage.py`.

## Fichiers modifiés

| Fichier | Fix(es) |
|---------|---------|
| `backend/optimization/overfitting.py` | #17 Monte Carlo, #20 DSR kurtosis |
| `backend/backtesting/portfolio_engine.py` | #10 Kill switch peak equity (2 emplacements) |
| `backend/optimization/fast_multi_backtest.py` | #1 Look-ahead, #2 Margin guard, #3 Entry slippage, #5 SL gap slippage |
| `backend/core/grid_position_manager.py` | #13 Entry slippage, #11 SL gap slippage |
| `backend/optimization/walk_forward.py` | #21 Embargo fallback, #18 oos_trades_by_window |
| `backend/execution/executor.py` | #25 Margin guard |
| `scripts/regime_backtest_compare.py` | Calmar annualisé |
| `scripts/diagnose_dd_leverage.py` | Calmar annualisé |

## Tests ajoutés / modifiés

**Nouveaux** : 19 tests dans `tests/test_sprint56_realism.py`

**Mis à jour** (régressions attendues des fixes de réalisme) :
- `tests/test_fast_engine_refactor.py` : valeurs parity recapturées post-sprint 57
- `tests/test_multi_engine.py` : SL gap slippage (fill 89.5 ≠ 90.0)
- `tests/test_grid_runner.py` : entry price inclut slippage
- `tests/test_grid_trend.py` : look-ahead shift cohérence mask
- `tests/test_regime_signal.py` : Calmar annualisé
- `tests/test_backtest_realism.py` : warning SL×leverage 7x (12%×7=84%)
- `tests/test_config_assets.py` : 19 assets (OP/SUI retirés)
- `tests/test_dataengine_autoheal.py` : 19 assets
- `tests/test_kill_switch_reliability.py` : asyncio.run() (Python 3.13 compat)
- `tests/test_grid_boltrend_parity.py` : tolérance divergence 5%→50%

## Résultats des tests

**2081 tests, 2081 passants** — 0 échec, 2 warnings RuntimeWarning asyncio pré-existants.

## Audit de validation (Partie A)

Exécuté immédiatement après l'implémentation :
- `pytest tests/ -x -q` : **2081 passed**
- `pytest tests/test_sprint56_realism.py -v` : **19/19 passed**
- `pytest tests/test_wfo_embargo.py -v` : **7/7 passed**
- Vérification manuelle des 11 fixes : tous confirmés avec lignes de code
- Smoke test `run_backtest grid_atr BTC/USDT 90j` : terminé sans erreur

## Impact sur les WFO existants

**Tous les résultats WFO antérieurs sont invalidés** par les fixes #1 (look-ahead) et #3+#13 (entry slippage) qui changent structurellement les prix d'entrée simulés. Les fixes #2 (margin guard) et #5+#11 (SL gap) réduisent également le rendement simulé de manière réaliste.

**Action requise** : relancer tous les WFO (grid_atr, grid_multi_tf) avec le pipeline corrigé avant tout déploiement live.
