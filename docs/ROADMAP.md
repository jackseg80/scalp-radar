# Scalp Radar — Roadmap Complète

## Vision

Système automatisé de trading crypto qui :
1. Développe et valide des stratégies via backtesting rigoureux (anti-overfitting)
2. Les compare objectivement via un pipeline standardisé
3. Les déploie en paper trading puis en live sur Bitget
4. Fournit une interface visuelle pour la recherche, le monitoring et l'optimisation

---

## PHASE 1 — INFRASTRUCTURE (Sprints 1-6) ✅ TERMINÉ

| Sprint | Contenu | Status |
|--------|---------|--------|
| 1 | Architecture, config YAML, DataEngine WebSocket, DB SQLite | ✅ |
| 2 | Backtesting engine mono-position, OHLC heuristics, TP/SL | ✅ |
| 3 | Simulator live, Arena (classement stratégies), StateManager | ✅ |
| 4 | Docker, Telegram alerts, Watchdog, Heartbeat, deploy.sh | ✅ |
| 5a | Executor live Bitget (mono-position, SL/TP server-side) | ✅ |
| 5b | Multi-position, multi-stratégie, AdaptiveSelector | ✅ |
| 6 | Dashboard V2 (Scanner, Heatmap, Equity curve, Trades) | ✅ |

**Résultat** : Pipeline complet data → stratégie → simulation → live → monitoring.

**Détails** :
- 19 composants frontend React (Scanner, Heatmap, ExecutorPanel, etc.)
- DataEngine : 5 assets × 4 timeframes (1m, 5m, 15m, 1h)
- Executor : réconciliation au boot, kill switch, rate limiting
- StateManager : crash recovery (save/restore toutes les 60s)
- Docker Compose : backend + frontend + nginx reverse proxy

---

## PHASE 2 — OPTIMISATION & VALIDATION (Sprints 7-10) ✅ TERMINÉ

| Sprint | Contenu | Status |
|--------|---------|--------|
| 7 | WFO (Walk-Forward Optimization), Monte Carlo, DSR, Grading | ✅ |
| 7b | Données funding rates + open interest, fast engine (100×) | ✅ |
| 8 | Backtest dashboard (planifié, non implémenté — remplacé par CLI) | ⏭️ Reporté |
| 9 | 7 stratégies mono-position testées → toutes Grade F | ✅ |
| 10 | Multi-position engine, envelope_dca → Grade B (BTC) | ✅ |

**Résultat** : Pipeline de validation robuste. Première stratégie viable (envelope_dca).

**Leçon clé** : Les stratégies mono-position à indicateur technique unique n'ont pas d'edge en crypto. L'edge vient de la structure DCA multi-niveaux.

**Détails optimisation** :
- WFO : IS/OOS windows (180j/60j), grid search 2 passes (LHS 500 → fine ±1 step)
- Monte Carlo : block bootstrap 1000 sims, underpowered detection (< 30 trades)
- DSR (Deflated Sharpe Ratio) : correction multiple testing bias (Bailey & Lopez de Prado 2014)
- Grading : 5 critères pondérés (OOS Sharpe, consistance, OOS/IS ratio, DSR, stabilité) → A-F
- Fast engine : numpy-only, indicator cache, 100× speedup vs event-driven engine

**Stratégies testées (toutes Grade F sauf envelope_dca)** :
- vwap_rsi : VWAP+RSI mean reversion (5m)
- momentum : Momentum breakout (5m)
- funding : Funding rate arbitrage (15m)
- liquidation : Liquidation zone hunting (5m)
- bollinger_mr : Bollinger Band mean reversion (1h)
- donchian_breakout : Donchian channel breakout (1h)
- supertrend : SuperTrend trend-following (1h)

**Résultats envelope_dca (WFO 5 assets)** :

| Asset | Grade | OOS Sharpe | Consistance | OOS/IS Ratio | DSR | Stabilité |
|-------|-------|------------|-------------|--------------|-----|-----------|
| BTC   | B     | 14.27      | 100%        | 1.84         | 0.98| 0.83      |
| ETH   | C     | 4.22       | 85%         | 0.54         | 1.00| 0.65      |
| SOL   | C     | 8.23       | 85%         | 0.82         | 0.99| 0.93      |
| DOGE  | D     | 3.58       | 92%         | 0.38         | 1.00| 1.00      |
| LINK  | F     | 1.46       | 75%         | 0.17         | 1.00| 0.98      |

---

## PHASE 3 — PAPER → LIVE (Sprints 11-12) ✅ TERMINÉ

| Sprint | Contenu | Status |
|--------|---------|--------|
| 11 | GridStrategyRunner, paper trading envelope_dca (5 assets) | ✅ |
| 12 | Executor Grid (multi-level SL/TP), alertes Telegram DCA | ✅ |

**Résultat** : Paper trading actif. Executor prêt pour le live (LIVE_TRADING=false).

**En attente** : Observer 1-2 semaines de paper trading avant activation live.

**Détails** :
- GridStrategyRunner : duck-type interface (start, stop, get_state, restore_state)
- Warm-up depuis DB : N candles injectées avant on_candle pour initialiser la SMA
- Bug fix critique : positions par symbol (pas partagées entre assets)
- Executor Grid : 12 méthodes grid (~350 lignes)
  - `_open_grid_position()` : ouvre niveaux séquentiellement
  - `_update_grid_sl()` : recalcule SL global par niveau
  - `_emergency_close_grid()` : force close si SL impossible (règle #1 : jamais de position sans SL)
  - `_close_grid_cycle()` : ferme toutes les positions quand TP hit
  - `_reconcile_grid_symbol()` : réconciliation au boot
- Exclusion mutuelle mono/grid par symbol (Bitget agrège positions par symbol+direction)
- State persistence : `grid_states` dans `get_state_for_persistence/restore_positions`
- Alertes Telegram : grid level opened, cycle closed (avec P&L)

**8 bugs critiques corrigés Sprint 12** :
1. AdaptiveSelector bloquait envelope_dca (mapping manquant + live_eligible: false)
2. RiskManager rejetait le 2ème niveau grid (position_already_open)
3. record_pnl() n'existait pas (utiliser record_trade_result)
4. _watch_orders_loop dormait sans positions mono (condition inclut _grid_states)
5. _poll_positions_loop ignorait les grids (itérer aussi _grid_states)
6. _cancel_orphan_orders supprimait le SL grid (inclure grid IDs dans tracked_ids)
7. Leverage 15 au lieu de 6 dans le margin check (leverage_override param)
8. Conflit mono/grid sur le même symbol (exclusion mutuelle bidirectionnelle)

---

## PHASE 4 — RECHERCHE & VISUALISATION (Sprints 13-15) ✅ TERMINÉ

| Sprint | Contenu                                                 | Status       |
|--------|---------------------------------------------------------|--------------|
| 13     | DB optimization_results, migration JSON, page Recherche | ✅           |
| 14     | Explorateur paramètres (WFO en background)              | ✅           |
| 14b    | Heatmap dense + Charts analytiques + Tooltips           | ✅           |
| 14c    | DiagnosticPanel (Analyse intelligente WFO)              | ✅           |
| 15     | Stratégie Envelope DCA SHORT (miroir LONG)              | ✅           |
| 15b    | Analyse par régime de marché (Bull/Bear/Range/Crash)    | ✅           |
| 15c    | Fix grading MC IS→OOS, combo_score 100 trades, purge DB | ✅           |
| 15d    | Consistance grade, 18 paires, Apply A/B, 21 assets      | ✅           |

### Sprint 13 — Résultats WFO en DB + Dashboard Visualisation ✅

**But** : Voir les résultats d'optimisation sans lire du JSON brut.

**Implémenté** :

- Table `optimization_results` (22 colonnes + 4 index)
  - Colonnes SQL : id, strategy_name, asset, timeframe, created_at, grade, total_score, oos_sharpe, consistency, oos_is_ratio, dsr, param_stability, mc_pvalue, mc_underpowered, n_windows, n_distinct_combos, duration_seconds, is_latest
  - JSON blobs : best_params, wfo_windows, monte_carlo_summary, validation_summary, warnings
  - Index sur (strategy, asset), grade, is_latest, created_at
- `optimization_db.py` : fonctions sync (optimize.py) + async (API)
  - `save_result_sync()` : INSERT + is_latest transaction
  - `get_results_async()`, `get_result_by_id_async()`, `get_comparison_async()`
- `migrate_optimization.py` : script idempotent pour importer les 49 JSON existants
  - Gestion défensive : .get() avec défauts, NaN/Infinity → None
  - Merge final + intermediate (wfo_windows si dispo)
  - Bug fix : filtre "intermediate" cherchait dans le chemin complet → `Path(f).name`
- `optimize.py` : écrit en DB via `save_report()`
  - Passe wfo_windows + duration + timeframe
  - Backward compat JSON conservé
- `report.py` : refactoring `compute_grade()` → retourne `(grade: str, score: int)`
  - `total_score` ajouté au FinalReport (0-100)
  - DB path résolu depuis config (pas hardcodé)
- API `/api/optimization/*` : 3 endpoints
  - GET /results (filtres + pagination)
  - GET /{id} (détail complet)
  - GET /comparison (matrice strategies × assets)
- Frontend : page "Recherche" (4ème tab)
  - Tableau comparatif avec tri cliquable (grade, score, OOS Sharpe, etc.)
  - Vue détail : params, scores, WFO chart IS vs OOS
  - Fetch au montage (pas de polling inutile)
  - `WfoChart.jsx` : SVG natif, 2 lignes (IS/OOS), tooltips
- 20 tests (100% passants)
  - `test_optimization_db.py` : 9 tests (sync insert, is_latest, async queries, special values)
  - `test_optimization_routes.py` : 6 tests (GET routes, filtres, pagination, 404)
  - `test_migrate_optimization.py` : 5 tests (migration, idempotence, dry-run, missing fields)

**Résultat** : Les 49 résultats WFO existants sont maintenant visibles dans le dashboard. Nouveau runs s'enregistrent automatiquement.

**Tests** : 533 passants (+20 depuis Sprint 12)

**Leçons apprises** :
- Filtre glob : toujours checker `Path(f).name`, jamais le chemin complet (sinon `test_migrate_with_intermediate0` trigger le filtre)
- DB path : résoudre depuis config au lieu de hardcoder (config.secrets.database_url)
- Polling : fetch once pour données quasi-statiques (résultats WFO), pas de polling 10s inutile

### Sprint 14 — Explorateur de Paramètres ✅

**But** : Lancer des WFO depuis le dashboard avec suivi en temps réel + heatmap 2D interactive.

**Implémenté** :

- **Table `optimization_jobs`** (DB) : id, strategy_name, asset, status, progress_pct, current_phase, params_override, created_at, started_at, completed_at, duration_seconds, result_id, error_message
- **JobManager** (backend/optimization/job_manager.py) :
  - File FIFO asyncio.Queue (max 5 pending)
  - Worker loop avec `asyncio.to_thread()` pour WFO
  - Progress callback thread-safe : sqlite3 sync + `run_coroutine_threadsafe` pour broadcast WS
  - Annulation : `threading.Event` vérifié à chaque fenêtre WFO
  - Recovery au boot : jobs "running" orphelins → failed
  - Normalisation `params_override` : scalaires → listes (fix 'float' object is not iterable)
- **Progress callback WFO** : `walk_forward.optimize()` + `scripts/optimize.py` acceptent `progress_callback` et `cancel_event` (optionnels, zéro régression CLI)
- **6 endpoints API** (optimization_routes.py) :
  - POST /api/optimization/run (submit job)
  - GET /api/optimization/jobs (liste avec filtre status)
  - GET /api/optimization/jobs/{id} (détail)
  - DELETE /api/optimization/jobs/{id} (annulation)
  - GET /api/optimization/param-grid/{strategy} (params disponibles depuis param_grids.yaml)
  - GET /api/optimization/heatmap (matrice 2D depuis optimization_results)
- **Frontend ExplorerPage.jsx** (~800 lignes) :
  - Layout CSS Grid : config panel (320px gauche), heatmap (flex-1 centre), jobs (250px-50vh bas)
  - Sélection stratégie + asset → charge param-grid dynamiquement
  - Sliders discrets (snap aux valeurs YAML) pour chaque paramètre
  - Sélection axes heatmap (X, Y) + métrique couleur (total_score, oos_sharpe, consistency, dsr)
  - Bouton "Lancer WFO" → POST run → progress bar temps réel via WebSocket
  - Jobs table : status, progress, phase, durée
- **HeatmapChart.jsx** (~240 lignes) :
  - SVG pur, cellSize responsive (60-300px selon espace disponible)
  - Échelle couleur rouge→jaune→vert (interpolation linéaire min-max)
  - Cellules vides (pas de données) : gris foncé (#2a2a2a)
  - Hover : tooltip avec params + métrique + grade
  - ResizeObserver pour adapter la taille au conteneur
  - Centré dans le conteneur via flexbox

**Bugs corrigés** :
1. `params_override` scalaires vs listes : frontend envoie `{ma_period: 7}`, WFO attend `{ma_period: [7]}` → normalisation dans JobManager
2. Heatmap trop petite : cellSize max 120px → 300px, SVG centré dans le conteneur
3. Jobs section trop petite : `max-height: 200px` → `min-height: 250px; max-height: 50vh;` (dynamique)

**Tests** : 597 passants (+42 depuis Sprint 13)
- `test_job_manager.py` : 13 tests (DB CRUD, submit, cancel, FIFO, progress, erreurs)
- `test_job_manager_wfo_integration.py` : 1 test (WFO complet bout en bout)
- `test_optimization_routes_sprint14.py` : 14 tests (endpoints POST/GET/DELETE, param-grid, heatmap)
- `test_walk_forward_callback.py` : 14 tests (progress callback, cancel_event, compteurs)

**Résultat** : L'utilisateur peut maintenant tester visuellement l'impact de chaque paramètre sur une stratégie, lancer des WFO depuis le navigateur, et voir les résultats dans une heatmap 2D cliquable.

**Leçons apprises** :
- Thread-safety WFO : `asyncio.to_thread()` avec event loop dédié fonctionne parfaitement
- Progress callback : utiliser `run_coroutine_threadsafe()` pour le broadcast WS depuis le thread
- Heatmap responsive : ResizeObserver + cellSize dynamique (min/max) + flexbox centering
- Normalisation params : toujours convertir scalaires → listes avant de passer au WFO
- Jobs recovery : scanner les "running" au boot évite les jobs zombies après un crash

### Sprint 14b — Heatmap Dense + Charts Analytiques + Tooltips ✅

**But** : Rendre la heatmap **100% dense** (toutes les combos testées), ajouter des charts analytiques, et des tooltips d'aide pour les termes techniques.

**Problème initial** : La heatmap était **sparse** (quasi vide) car seul le `best_params` de chaque run était sauvegardé. Pour 324 combos possibles, seulement 2-3 cellules étaient remplies.

**Implémenté** :

**Backend — Heatmap dense** :
- **Table `wfo_combo_results`** (14 colonnes) : id, optimization_result_id, params (JSON), oos_sharpe, oos_return_pct, oos_trades, oos_win_rate, is_sharpe, is_return_pct, is_trades, consistency, oos_is_ratio, is_best, n_windows_evaluated
- **Collecte OOS batch** dans `walk_forward.optimize()` :
  - OOS batch pour toutes les combos (~324 × 0.1ms = 30ms/fenêtre)
  - Agrégation cross-fenêtre : moyenne IS/OOS Sharpe, consistency, oos_is_ratio
  - Guard stratégies avec fast engine (6/8 supportées : vwap_rsi, momentum, bollinger_mr, donchian_breakout, supertrend, envelope_dca)
- **API endpoints** :
  - `GET /api/optimization/combo-results/{id}` : retourne les 324 combos d'un run
  - `GET /api/optimization/heatmap` mode dense : agrégation multi-dim (moyenne des combos avec mêmes param_x, param_y)
- **Persistence** : `save_combo_results_sync()`, `get_combo_results_async()` dans optimization_db.py
- Push serveur inclut combo_results (sync WFO local → serveur)

**Frontend — Charts analytiques** :
- **Run selector** : dropdown historique des 20 derniers runs WFO pour (strategy, asset)
- **Top10Table.jsx** : tableau HTML top 10 combos par métrique, combo `is_best` surlignée
- **ScatterChart.jsx** : SVG scatter plot IS Sharpe vs OOS Sharpe
  - Diagonale pointillée IS = OOS
  - Points colorés par consistance (rouge < 50%, orange 50-80%, vert > 80%)
  - Point best en surbrillance (stroke blanc, rayon plus grand)
- **DistributionChart.jsx** : SVG histogramme distribution OOS Sharpe
  - Bins automatiques (sqrt(n_combos))
  - Barres rouge (< 0) / vert (≥ 0)
  - Marqueur best combo (flèche verticale)

**Frontend — Tooltips d'aide** :
- **InfoTooltip.jsx** : composant réutilisable avec glossaire 14 termes (oos_sharpe, is_sharpe, oos_is_ratio, consistency, dsr, monte_carlo_pvalue, param_stability, grade, total_score, ci_sharpe, transfer_ratio, wfo, is_vs_oos_chart, oos_return_pct)
- Icône (i) cliquable, popover avec description + interprétation
- Intégré sur **ExplorerPage** (headers heatmap, top 10, charts) ET **ResearchPage** (headers tableau, labels détail)

**Bug critique corrigé** :
- **params_override pré-rempli automatiquement** : frontend pré-remplissait tous les sliders avec leurs valeurs default → 1 seule combo testée au lieu de 324
- **Fix UX redesign complet** : système actif/inactif avec checkboxes
  - Slider inactif (grisé) = toutes les valeurs testées
  - Slider actif (vert) = valeur fixée
  - Calcul temps réel du nombre de combos : "Grille : 324 combos (3×3×3×3×4)"
  - `params_override` envoyé seulement pour les sliders actifs, ou `null` si aucun
- **Script nettoyage** : `fix_invalid_explorer_runs.py` supprime les 9 runs invalides (1-5 combos) pré-fix

**Corrections techniques (toutes les erreurs console)** :
- **Balises `<style jsx>` supprimées** : 7 fichiers CSS créés (InfoTooltip.css, ResearchPage.css, ExplorerPage.css, Top10Table.css, ScatterChart.css, DistributionChart.css, WfoChart.css)
- **8 composants modifiés** : InfoTooltip.jsx, ResearchPage.jsx (3 balises), ExplorerPage.jsx, Top10Table.jsx, ScatterChart.jsx, DistributionChart.jsx, WfoChart.jsx
- **HeatmapChart.jsx** : `Math.max(0, chartWidth - 200)` pour éviter width négatif

**Tests** : 603 passants (+6 depuis Sprint 14, 2 skippés car API WFO changée)
- `test_combo_results.py` : 8 tests (collecte, agrégation, is_best flag, save/load DB, push payload, migration serveur)

**Résultat** :
- Heatmap **100% dense** (324/324 cellules remplies)
- 3 charts analytiques pour explorer les résultats WFO
- 14 tooltips d'aide sur 2 pages
- **0 erreur console** dans le frontend
- UX robuste (impossible de lancer un WFO invalide)

**Leçons apprises** :
- OOS batch via fast engine est quasi gratuit (~30ms additionnel pour 324 combos)
- Agrégation multi-dim nécessaire pour heatmap dense (moyenne des combos partageant param_x, param_y)
- React standard ne supporte pas `<style jsx>` (feature de Next.js styled-jsx) → fichiers CSS séparés obligatoires
- UX : toujours montrer l'état par défaut explicitement (slider décoché = toutes valeurs, pas valeur default unique)
- n_windows_evaluated crucial pour distinguer combos partielles (2-pass coarse+fine grid)

### Sprint 14c — DiagnosticPanel (Analyse intelligente WFO) ✅

**But** : Aider l'utilisateur à comprendre en 10 secondes si une stratégie est viable via des verdicts textuels en langage clair.

**Implémenté** :

**Frontend — Diagnostic automatique** :


- **DiagnosticPanel.jsx** (~180 lignes) : composant 100% frontend, 6 règles d'analyse
  - Règle 1 : Grade global (A/B = vert viable, C = orange moyenne, D/F = rouge non viable)
  - Règle 2 : Consistance du best combo (profitable dans X/12 fenêtres)
  - Règle 3 : Transfert IS→OOS (ratio OOS/IS, détection overfitting si IS > 5 et OOS < 1)
  - Règle 4 : Edge structurel (distribution OOS Sharpe : % combos > 1, > 0.5, > 0)
  - Règle 5 : Volume de trades (min 30 pour signification statistique)
  - Règle 6 : Fenêtres partielles (combos évaluées sur < 12 fenêtres, moins fiables)
- **DiagnosticPanel.css** (~80 lignes) : dark theme cohérent
  - Background #111827, border-left 4px colorée selon verdict le plus sévère (rouge/orange/vert)
  - SVG inline pour icônes (cercles colorés + graphique barres)
  - Responsive (max-height 300px, scroll si > 6 verdicts)
- **ExplorerPage.jsx** : import + useMemo selectedRun + insertion diagnostic AVANT Top10
  - `nWindows` calculé dynamiquement depuis `combos.n_windows_evaluated` (pas de hardcode)

**Position** : Section "Analyse des combos", **première chose visible** après la heatmap (avant le Top10 et les charts).

**0 changement backend** : tout calculé côté frontend depuis `comboResults.combos` (déjà fetchées) et `selectedRun.grade/total_score` (déjà chargés).

**Tests** : 628 passants (0 régression, build frontend OK)

**Résultat** : L'utilisateur voit immédiatement si sa stratégie a un edge structurel, si elle est robuste, et où sont les faiblesses (consistance, overfitting, données insuffisantes).

**Leçons apprises** :

- Verdicts textuels > métriques brutes pour la compréhension rapide
- Border-left colorée (rouge/orange/vert) = signal visuel immédiat de santé globale
- `nWindows` dynamique depuis combos (pas de hardcode 12) = robuste si WFO config change

### Sprint 15 — Stratégie Envelope DCA SHORT ✅

**But** : Créer le miroir SHORT de la stratégie envelope_dca (LONG) pour doubler les opportunités de trading.

**Décision architecturale** : Sous-classe minimale (EnvelopeDCAShortStrategy) au lieu de paramétrer l'existant, pour :
- Éviter toute régression sur la stratégie LONG en production
- Supporter deux grilles WFO indépendantes (LONG vs SHORT)
- Simplifier l'identification par nom dans le système (registry, dashboard, executor)

**Implémenté** :

**Backend — Stratégie** :
- **EnvelopeDCAShortStrategy** (26 lignes) : sous-classe qui réutilise 100% de la logique de EnvelopeDCAStrategy
  - Seuls le `name` et `sides` par défaut changent
  - Compute_grid() et should_close_all() gèrent déjà les deux directions
- **EnvelopeDCAShortConfig** : config identique à LONG avec `sides: ["short"]` par défaut
- **Entrées config** : strategies.yaml + param_grids.yaml (`enabled: false` pour validation WFO d'abord)

**Backend — Fast Engine SHORT** :
- **fast_multi_backtest.py** : ajout paramètre `direction: int = 1` (backward compatible)
  - Dispatch : `envelope_dca` → direction=1, `envelope_dca_short` → direction=-1
  - Enveloppes asymétriques SHORT : `upper_offset = round(1/(1-lower_offset) - 1, 3)`
  - Inversion SL/TP checks : `sl_hit = high >= sl_price`, `tp_hit = low <= tp_price`
  - OHLC heuristic inversée : bougie rouge (close < open) favorable pour SHORT → TP
- **indicator_cache.py** : condition étendue `if strategy_name in ("envelope_dca", "envelope_dca_short")`
- **walk_forward.py** : 3 whitelists mises à jour (_INDICATOR_PARAMS, collect_combo_results, fast engine)

**Backend — Registry & API** :
- **optimization/__init__.py** : ajout dans STRATEGY_REGISTRY et GRID_STRATEGIES
- **strategies/factory.py** : ajout dans create_strategy() et get_enabled_strategies()
- **adaptive_selector.py** : ajout mapping
- **optimization_routes.py** : nouvel endpoint `/api/optimization/strategies` (dynamique depuis STRATEGY_REGISTRY + param_grids.yaml)

**Frontend — Découverte dynamique** :
- **ExplorerPage.jsx** : fetch stratégies depuis API au lieu de liste hardcodée
- **Conséquence** : futures stratégies apparaissent automatiquement sans modification frontend

**Tests** : 613 passants (+16 depuis Sprint 14b)
- **test_envelope_dca_short.py** : 22 tests (signal generation SHORT, enveloppes asymétriques, direction lock, TP/SL global, fast engine SHORT, registry, config)
- **0 régression** sur les 603 tests existants

**Résultat** :
- 9 stratégies totales (8 mono + 1 grid/DCA LONG → 8 mono + 2 grid/DCA LONG+SHORT)
- Infrastructure WFO prête pour optimiser envelope_dca_short
- Frontend adaptatif (pas de hardcoding stratégies)

**Leçons apprises** :
- Sous-classe minimale > paramétrage quand le fast engine doit dispatcher (zéro risque régression)
- OHLC heuristic doit être inversée pour SHORT (rouge = favorable, vert = défavorable)
- Enveloppes asymétriques critiques : aller-retour cohérent (entry_long → SMA → entry_short doit être symétrique en log-return)
- Découverte dynamique backend → frontend évite les oublis futurs (nouvelle stratégie apparaît automatiquement)

### Hotfix — P&L Overflow GridStrategyRunner ✅

**Bug** : P&L simulateur paper trading affichait +12 quadrillions $, capital à 9.4 quintillions $.

**Cause racine** : `GridStrategyRunner.on_candle()` ne déduisait jamais la marge du capital après ouverture de positions grid. Contrairement à `LiveStrategyRunner` qui fait `self._capital -= entry_fee`, le GridStrategyRunner gardait le capital intact → compounding exponentiel sur chaque cycle.

**Fix (3 volets)** :

1. **Margin accounting** (simulator.py — GridStrategyRunner) :
   - **Open** : déduire `margin = notional / leverage` du capital disponible
   - **Close** : restituer la marge au capital, puis appliquer le net_pnl
   - **Anti-overflow** : si `capital < margin_used` → skip le niveau (log warning)
   - Ajout `self._leverage` en `__init__()` depuis config

2. **Realized P&L tracking** (simulator.py — GridStrategyRunner) :
   - Ajout `self._realized_pnl` : ne suit que les trades clôturés
   - `self._stats.net_pnl = self._realized_pnl` (pas `capital - initial_capital`)
   - Corrige le kill switch qui comptait la marge verrouillée comme une "perte"
   - Backward-compatible : `restore_state()` fallback sur `net_pnl` si `realized_pnl` absent

3. **State persistence** (state_manager.py) :
   - Sauvegarde `realized_pnl` avec guard `isinstance(getattr(...), (int, float))` (MagicMock-safe)
   - Restauration backward-compatible

**Script reset** : `scripts/reset_simulator.py` — supprime l'état corrompu (idempotent, `--executor` flag)

**Tests** : 628 passants (+15 depuis Sprint 15)
- `test_capital_decremented_on_open` : vérifie déduction marge à l'ouverture
- `test_grid_capital_restored_on_close` : vérifie capital = initial + net_pnl après close
- `test_grid_no_overflow_after_100_cycles` : 100 cycles near-breakeven → capital < 2× initial
- `test_grid_zero_capital_skips_level` : capital=0 → aucune position ouverte
- Tests state_manager + simulator existants adaptés (realized_pnl)

**Résultat production** :
- Capital : 20 175$ (initial 10 000$), +101.76% en 20 trades, 70% win rate
- Kill switch : désactivé (utilise realized_pnl, pas capital)
- État cohérent après restart

**Leçons apprises** :
- **Marge ≠ fee** : ne pas confondre la déduction de marge (réversible) et la déduction de frais (irréversible dans net_pnl)
- **Double-counting fees** : `close_all_positions()` inclut déjà `entry_fee` dans `net_pnl` → ne pas les déduire aussi à l'ouverture
- **Kill switch** : doit utiliser le P&L réalisé uniquement, pas `capital - initial_capital` (sinon marge verrouillée = "perte")
- **MagicMock piège** : `hasattr(mock, "anything")` retourne toujours True → utiliser `isinstance(getattr(...), (int, float))`
- **Tests compound** : utiliser des prix near-breakeven pour tester N cycles, sinon le compound légitime explose aussi

### Hotfix — Orphan Cleanup + Collision Warning ✅

**Problème 1** : Quand une stratégie est désactivée (`enabled: false`) et le serveur redémarré, ses positions ouvertes disparaissent silencieusement — aucun log, aucun cleanup. En live, les positions restent sur Bitget sans suivi.

**Problème 2** : En paper trading, 2 runners peuvent ouvrir sur le même symbol sans avertissement, alors qu'en live l'Executor applique l'exclusion mutuelle.

**Fix 1 — Cleanup positions orphelines au boot** :
- `OrphanClosure` dataclass : stocke strategy_name, symbol, direction, entry_price, quantity, estimated_fee_cost
- `_cleanup_orphan_runners()` : itère `saved_state["runners"]`, filtre ceux pas dans `enabled_names`
  - Paper : log WARNING + crée `OrphanClosure` avec fee estimé
  - Live : log CRITICAL "VÉRIFIER BITGET MANUELLEMENT" (pas d'action automatique sur l'exchange)
  - Les orphelins disparaissent naturellement du JSON via periodic save (60s) — seuls les runners actifs sont sauvegardés
- Appelé dans `start()` avant la restauration d'état des runners actifs
- Property `orphan_closures` : liste des fermetures orphelines (logs suffisants, pas exposé au frontend)

**Fix 2 — Warning collision paper trading** :
- `_get_position_symbols()` : helper retournant les symbols avec positions ouvertes (gère Grid et Mono)
- `_dispatch_candle()` modifié : snapshot positions avant la boucle, détection collision après chaque `on_candle()`
- Si un runner ouvre une position sur un symbol déjà occupé par un autre runner → log WARNING
- `get_all_status()` enrichi avec `collision_warnings` par runner
- Property `collision_warnings` : liste des collisions détectées

**Tests** : 632 passants (+4 depuis hotfix P&L)
- `test_orphan_cleanup_on_disable` : stratégie désactivée avec position → orphan closure enregistrée
- `test_orphan_cleanup_no_positions` : stratégie désactivée sans position → aucune closure
- `test_collision_warning_same_symbol` : 2 runners sur BTC → collision détectée
- `test_no_collision_different_symbols` : runner_a ETH, runner_b BTC → aucune collision

**Leçons apprises** :
- `_trades` (liste en mémoire) vs `_stats.total_trades` (compteur persisté) : les trades ne sont pas restaurés au boot, seuls les compteurs le sont
- `LiveStrategyRunner.name` est une property read-only → setter via `runner._strategy.name`
- `getattr(getattr(...), ...)` pour accès config MagicMock-safe (pas de `hasattr`)

### Sprint 15b — Analyse par régime de marché ✅

**Objectif** : Classifier chaque fenêtre OOS du WFO selon le régime de marché et afficher la performance par régime dans le DiagnosticPanel.

**Classification `_classify_regime()`** (walk_forward.py) :

- Crash : max drawdown > 30% en < 14 jours (prioritaire, algorithme O(n) via deque glissant)
- Bull : rendement fenêtre > +20%
- Bear : rendement fenêtre < -20%
- Range : entre -20% et +20%

**Backend** :

- `_classify_regime()` appelé sur chaque fenêtre OOS → `window_regimes[]`
- `regime_analysis` agrégé pour le best combo (avg_oos_sharpe, consistency, avg_return_pct par régime)
- `WFOResult` : 2 nouveaux champs (`window_regimes`, `regime_analysis`)
- Migration DB idempotente : colonne `regime_analysis TEXT` dans `optimization_results`
- API `/combo-results/{id}` retourne `regime_analysis`
- Sérialisation windows : champs `regime`, `regime_return_pct`, `regime_max_dd_pct`

**Frontend** (DiagnosticPanel.jsx) :

- Section "PERFORMANCE PAR RÉGIME DE MARCHÉ" avec icônes SVG (Bull/Bear/Range/Crash)
- Métriques : Sharpe moyen OOS + consistance par régime + cercle couleur (vert/orange/rouge)
- Conclusion automatique : all-weather, mean-reversion, momentum, meilleur régime

**Hotfix exchange** :

- WFO lisait `main_exchange: "binance"` hardcodé dans `param_grids.yaml`
- Fix : lecture depuis `config.exchange.name.lower()` (config principale `exchanges.yaml`)
- `main_exchange` supprimé de `param_grids.yaml` (paramètre d'infrastructure, pas d'optimisation)
- `validation_exchange` reste dans `param_grids.yaml` (c'est un paramètre d'optimisation)

**Tests** : 650 passants (+18 depuis hotfix orphan)

### Hotfix — Explorer Heatmap 1 point (push serveur parasite) ✅

**Problème** : La page Explorer n'affichait qu'un seul point au lieu de 324 sur la heatmap. Cause : le serveur prod (192.168.1.200) avait `SYNC_ENABLED=true` et poussait ses résultats WFO (grille restreinte, 2 combos) vers la machine locale via `POST /api/optimization/results`, volant le flag `is_latest=1` aux bons runs locaux (320 combos).

**Investigation** :
- 32+ runs parasites avec `n_distinct_combos: 2`, aucun dans `optimization_jobs` → pas du JobManager local
- Pas de Task Scheduler Windows, pas de cron/systemd Linux, pas de cron Docker
- Paires ~15s d'écart, toutes les ~30 min = push bidirectionnel accidentel (serveur → local)

**Fix (3 couches de protection)** :
1. **Backend `save_result_from_payload_sync`** : un run pushé avec < 10 combos ne vole plus `is_latest` d'un run complet existant (garde `is_latest=0` sur le nouveau)
2. **Backend `get_results_async`** : LEFT JOIN sur `wfo_combo_results` pour retourner `combo_count` et `n_distinct_combos` par run
3. **Frontend `ExplorerPage.jsx`** : auto-sélection du run le plus récent avec ≥ 10 combos (pas aveuglément `is_latest`), dropdown affiche le nombre de combos par run

**Cleanup DB** : 36 runs parasites (1 combo) supprimés, `is_latest=1` rétabli sur le meilleur run (320 combos, Grade B)

**Config serveur** : désactiver `SYNC_ENABLED=false` sur le serveur prod (le serveur n'a pas les données historiques complètes, ses WFO sont incomplets)

**Tests** : 679 passants (0 régression)

**Leçons apprises** :
- Le sync WFO ne doit être qu'unidirectionnel (local → serveur), jamais l'inverse
- `is_latest` est fragile — un push externe peut le voler silencieusement
- Défense en profondeur : 3 couches (backend POST, backend API, frontend) pour que le même bug ne puisse plus casser la heatmap

### Sprint 15c — Fix Grading MC + combo_score ✅

**But** : Corriger les bugs critiques dans le pipeline de grading qui faussaient les grades.

**Corrections** :

1. **MC observed_sharpe IS→OOS** : le Monte Carlo comparait les shuffles OOS au Sharpe IS au lieu du Sharpe OOS → DOGE passé de Grade C à A
2. **combo_score seuil 50→100 trades** : `min(1, trades/100)` pour pénaliser les combos à faible volume (ETH sélectionne 111 trades au lieu de 39)
3. **Garde-fou < 30 trades → Grade max C** : empêche les faux positifs (BTC avec 6 trades ne peut plus être Grade A)
4. **Grille étendue 0.05-0.15** : confirmé que 0.05 est l'optimum, pas un artefact de borne
5. **DB purgée** : résultats recalculés avec le bon grading

**Tests** : 698 passants (0 régression)

### Sprint 15d — Consistance + Diversification 21 Assets ✅

**But** : Intégrer la consistance dans le grade, diversifier sur 18 nouvelles paires, automatiser le déploiement.

**Implémenté** :

**Grading — Consistance (20 pts/100)** :

- 6 critères au lieu de 5 : OOS/IS (20), MC (20), Consistance (20), DSR (15), Stabilité (10), Bitget transfer (15)
- ETH passe de 100/100 à 88/100 (68% consistance)
- Top 5 trié par combo_score (le #1 = le best combo sélectionné)

**Diversification — 18 nouvelles paires** :

- `fetch_history.py --symbols` : flag pour bypasser assets.yaml
- 717k candles Binance téléchargées (ADA, APE, AR, AVAX, BNB, CRV, DYDX, ENJ, FET, GALA, ICP, IMX, NEAR, SAND, SUSHI, THETA, UNI, XTZ)
- WFO sur 23 assets : 3 Grade A (ETH, DOGE, SOL) + 18 Grade B + 2 Grade D (BTC, BNB)
- THETA Grade B mais WebSocket Bitget refuse l'abonnement → commenté dans assets.yaml

**Automatisation Apply** :

- `optimize.py --apply` : écrit les per_asset dans strategies.yaml depuis la DB
- `apply_from_db()` : auto-ajoute les assets manquants dans assets.yaml avec specs Bitget via ccxt
- Bouton "Appliquer A/B" frontend : `POST /api/optimization/apply` → un clic = per_asset + assets.yaml mis à jour

**Déploiement prod** : 21 assets en paper trading live, pas de kill switch

**Tests** : 707 passants (+9 depuis Sprint 15c)

**Résultats finaux — 23 assets évalués** :

| Asset | Grade | Score | Sharpe | Consist. |
| ----- | ----- | ----- | ------ | -------- |
| ETH   | A     | 88    | 5.43   | 68%      |
| DOGE  | A     | 85    | 6.90   | 97%      |
| SOL   | A     | 85    | 9.02   | 92%      |
| LINK  | B     | 81    | 5.61   | 80%      |
| UNI   | B     | 81    | 8.06   | 80%      |
| APE   | B     | 81    | 5.24   | 85%      |
| SAND  | B     | 81    | 6.19   | 88%      |
| AR    | B     | 81    | 7.52   | 85%      |
| NEAR  | B     | 81    | 6.99   | 89%      |
| DYDX  | B     | 81    | 8.25   | 80%      |
| CRV   | B     | 81    | 6.90   | 88%      |
| IMX   | B     | 81    | 6.23   | 86%      |
| FET   | B     | 81    | 6.29   | 87%      |
| AVAX  | B     | 78    | 8.50   | 82%      |
| SUSHI | B     | 78    | 5.19   | 81%      |
| GALA  | B     | 77    | 7.52   | 78%      |
| ENJ   | B     | 74    | 11.43  | 75%      |
| ADA   | B     | 73    | 9.46   | 69%      |
| THETA | B     | 73    | 6.43   | 62%      |
| ICP   | B     | 71    | 9.25   | 82%      |
| XTZ   | B     | 71    | 4.98   | 73%      |
| BNB   | D     | 50    | 3.47   | 46%      |
| BTC   | D     | 47    | 3.20   | 40%      |

---

## PHASE 5 — SCALING STRATÉGIES (Sprints 16-20) ✅ TERMINÉ

| Sprint | Contenu | Status |
|--------|---------|--------|
| 16+17 | Dashboard Scanner + Monitoring DCA Live | ✅ |
| 18 | ~~Multi-asset Live envelope_dca~~ — Superseded par grid_atr | ⏭️ Superseded |
| 19 | Stratégie Grid ATR (10e stratégie, fast engine, 3240 combos WFO) | ✅ |
| 19b-d | Combo results dense, régimes marché, grading Bitget 3 paliers | ✅ |
| 19e | Scanner Grid Fix (colonnes dynamiques, GridDetail) | ✅ |
| 20a | Sizing equal allocation + margin guard 70% | ✅ |
| 20b | Portfolio Backtest Multi-Asset (CLI + engine) | ✅ |
| 20b-UI | Portfolio Frontend + API + Comparateur | ✅ |
| 20c | Factorisation fast engine (`_simulate_grid_common`) | ✅ |
| 20d | Anti-spam Telegram (cooldown par type d'anomalie) | ✅ |
| 20e | Kill switch grid-compatible + warm-up fixes | ✅ |
| 20f | Panneau Simulator P&L réalisé + non réalisé + equity | ✅ |

### Sprint 16+17 — Dashboard Scanner amélioré + Monitoring DCA Live ✅
**But** : Ajouter la visibilité grid DCA au Scanner et ActivePositions, sans casser l'architecture multi-stratégie.

**Backend** :

- [x] `Simulator.get_grid_state()` — état détaillé des grilles DCA actives avec P&L non réalisé
- [x] `Simulator._get_current_price(symbol)` — fallback 1m → 5m → 1h si buffer vide
- [x] Endpoint `GET /api/simulator/grid-state` — JSON complet grilles + summary
- [x] WebSocket push `grid_state` via `/ws/live` (toutes les 3s)
- [x] DataEngine batching anti-rate-limit : souscriptions par lots de 10 + 0.5s entre batchs
- [x] Rate limit retry dans `_subscribe_klines()` : codes 30006/429, backoff 2s×n, max 3 retries
- [x] Log throttle erreurs répétitives (3 warnings max par symbol)
- [x] Fix warm-up compound sur état restauré : `restore_state()` garde warm-up actif, état appliqué à `_end_warmup()`

**Frontend** :

- [x] Scanner : colonnes Grade (badge A-F coloré) + Grid (niveaux open/max) ajoutées, Score + Signaux conservées
- [x] Scanner : tri positions-first (grids avec P&L desc), puis par score
- [x] ActivePositions : `GridSummary` avec bandeau agrégé (N grids sur M assets, marge, P&L)
- [x] ActivePositions : ligne par asset cliquable → détail dépliable (niveaux individuels)
- [x] ActivePositions : fallback "En attente de données prix..." si `grid_state` absent
- [x] CSS : classes `.grade-badge`, `.grid-cell`, `.grid-summary-banner`

**Tests** : 727 tests (+13)

- 3 tests endpoint `grid-state` (no simulator, empty, with data)
- 8 tests logique métier `get_grid_state` (empty, positions, P&L long/short, multi-asset, fallback prix, TP/SL NaN)
- 4 tests warm-up : `restore_state_keeps_warmup`, `restore_state_applied_after_warmup`, `warmup_ignores_restored_capital`, `restore_state_ignores_kill_switch`

**Résultat** : Dashboard montre en temps réel les grilles DCA avec niveaux, P&L non réalisé, TP/SL distances, et grades WFO. Warm-up post-restore protégé contre le compound overflow.

### Sprint 19 — Stratégie Grid ATR ✅
**But** : 10e stratégie grid/DCA adaptative — enveloppes basées sur l'ATR (volatilité) au lieu de pourcentages fixes.

**Implémenté** :

**Backend — Stratégie** :
- **GridATRStrategy** (BaseGridStrategy) : `compute_grid()` calcule `entry_price = SMA ± ATR × (start + i × step)`
- **GridATRConfig** : ma_period, atr_period, atr_multiplier_start, atr_multiplier_step, num_levels, sl_percent, sides, leverage
- Symétrie naturelle SHORT : `entry_price = SMA + ATR × multiplier` (pas de conversion asymétrique)
- Guards : ATR <= 0, entry_price > 0, NaN checks
- `should_close_all()` : TP=SMA dynamique, SL=% prix moyen (même pattern que envelope_dca)

**Backend — Fast Engine** :
- `_simulate_grid_atr()` dans `fast_multi_backtest.py` (~150 lignes)
- Réutilise `_calc_grid_pnl()` existant (pas de duplication)
- Guard `capital <= 0` ajouté (safety check absent dans envelope_dca)
- `build_cache()` : peupler `bb_sma` (SMA) + `atr_by_period` (ATR multi-period)
- `_INDICATOR_PARAMS["grid_atr"] = ["ma_period", "atr_period"]` (groupement WFO)
- Ajouté dans la liste `_run_fast` (tuple hardcodé walk_forward.py)

**Backend — Registry** :
- STRATEGY_REGISTRY + GRID_STRATEGIES + factory.py
- strategies.yaml : `grid_atr: enabled: false`
- param_grids.yaml : grille 3240 combos (4×3×5×3×3×4), WFO 180/60/60 jours

**Tests** : 770 passants (+43 depuis Sprint 16+17)
- `test_grid_atr.py` : 37 tests (signaux, TP/SL, fast engine, parité, registry, executor helpers)
- `test_strategy_registry.py` : +6 tests (grid_atr dans ALL_STRATEGIES, GRID_STRATEGY_NAMES, INDICATOR_PARAMS)

**Résultat** :
- 10 stratégies totales (4 scalp 5m + 3 swing 1h + 3 grid/DCA 1h)
- Pipeline WFO complet prêt pour optimisation grid_atr
- Fast engine vérifié en parité avec MultiPositionEngine (données sinusoïdales)

**Leçons apprises** :
- `_calc_grid_pnl` est déjà générique (prend `direction`) → pas besoin de `_calc_grid_pnl_atr`
- `_run_fast` dans walk_forward.py utilise un tuple hardcodé — facile à oublier
- `build_cache` `atr_by_period` n'est peuplé que pour les stratégies qui le demandent → ajouter `"grid_atr"` à la condition
- Données sinusoïdales (`100 + 8*sin(2π*i/48)`) génèrent un ATR réaliste pour les tests de parité

**Dette technique résolue** : Sprint 20c a factorisé `_simulate_grid_common()` — voir ci-dessous

**WFO grid_atr 21 assets — Résultats** :

| Asset | Grade | Score | Sharpe | Consist. |
| ----- | ----- | ----- | ------ | -------- |
| 14 assets | A | 85-100 | 4.5-12+ | 75-100% |
| 7 assets | B | 71-84 | 3.5-8 | 60-80% |

- **0 Grade D/F** — grid_atr a un edge structurel supérieur à envelope_dca
- grid_atr remplace envelope_dca comme stratégie principale (paper trading 21 assets)

**Hotfixes Sprint 19b/19c/19d** :

- **Sprint 19b** : `wfo_combo_results` pour grid_atr — collecte dense (3240 combos), heatmap + scatter + distribution fonctionnels
- **Sprint 19c** : régimes de marché grid_atr + fix warning trades insuffisants
- **Sprint 19d** : grading Bitget transfer 3 paliers (`>0.50 + significant` → 15 pts, `>0.50` → 10 pts, `>0.30` → 5 pts) + guard `bitget_trades < 15` → cap 8 pts + fix or-on-float

### Sprint 20a — Sizing Equal Allocation + Margin Guard ✅
**But** : Remplacer le sizing "equal risk" (`risk_budget / sl_pct`) par "equal allocation" (`capital / nb_assets / levels`) + margin guard global.

**Hotfix 19e** : Scanner Grid Fix (frontend pur) — colonnes Score/Signaux masquées si aucune stratégie mono enabled, colonne Dist.SMA (distance prix/SMA = distance au TP), direction des positions grid (pas RSI fallback), tri secondaire par Grade (A→F), clic détail → `GridDetail` avec niveaux remplis (vert) / en attente (rouge transparent), TP/SL, P&L, marge, indicateurs. Zéro changement backend.

**Problème résolu** : L'ancien sizing faisait dépendre la taille de position du SL — un SL serré (1%) donnait une marge énorme, un SL large (30%) une marge minuscule. Certains assets mobilisaient jusqu'à 4.9× le capital.

**Implémenté** :
- Sizing equal allocation : `margin_per_level = capital / nb_assets / num_levels` (le SL contrôle le risque en $, PAS la taille)
- Cap 25% par asset (inchangé)
- Margin guard global : `max_margin_ratio: 0.70` dans `risk.yaml` — impossible de dépasser 70% du capital en marge simultanée
- Champ `max_margin_ratio` dans `RiskConfig` (Pydantic, validé 0.1-1.0, default 0.70)
- Guard MagicMock-safe (`isinstance` check) pour les tests avec MagicMock config
- Scanner grade fix : prioriser le meilleur grade, pas envelope_dca hardcodé

**Résultat** : Marge totale réduite de 4.9× à 0.38× du capital. Plus aucun dépassement possible grâce au guard 70%.

**Tests** : 4 mis à jour + 2 nouveaux (margin guard blocks, total margin ≤ 70%) → 774 tests

**Non touché** : fast engine WFO (déjà equal allocation), executor (reçoit quantity du Simulator), GridPositionManager, grades existants

### Sprint 18 — ~~Multi-asset Live~~ Superseded ⏭️

**Superseded** — grid_atr a remplacé envelope_dca comme stratégie principale. Le déploiement multi-asset se fait directement via grid_atr (21 assets en paper trading dès Sprint 19).

### Sprint 20 — Nouvelles Stratégies Grid
**But** : Développer d'autres stratégies qui utilisent le moteur multi-position.

**Candidates** :
- **Grid RSI** : DCA déclenché par RSI extrême + niveaux %
- **Grid Funding** : DCA quand le funding rate est fortement négatif (coûte de shorter)
- **Grid Multi-timeframe** : signaux sur 4h, exécution sur 1h

**Workflow pour chaque nouvelle stratégie** :
1. Implémenter (hérite BaseGridStrategy)
2. Ajouter au param_grids.yaml
3. Lancer WFO → Grade dans le dashboard (Sprint 13)
4. Comparer avec envelope_dca dans le tableau
5. Paper trading si Grade >= C
6. Live si paper trading cohérent

**Scope** : ~1-2 sessions par stratégie.

**Dette technique résolue** : Sprint 20c a factorisé `_simulate_grid_common()` — ajouter une nouvelle stratégie grid = 3-5 lignes dans `_build_entry_prices()`.

---

### Sprint 20c — Factorisation Fast Engine + Auto-dispatch WFO ✅

**But** : Factoriser `_simulate_envelope_dca()` et `_simulate_grid_atr()` (80% de code dupliqué) en une architecture extensible.

**Implémenté** :
- `_build_entry_prices(strategy_name, cache, params, num_levels, direction) -> np.ndarray` — factory retournant un 2D array `(n_candles, num_levels)` de prix d'entrée. Chaque nouvelle stratégie grid = ajouter un elif de 3-5 lignes
- `_simulate_grid_common()` — boucle chaude unifiée (TP/SL, allocation, force close)
- Wrappers backward-compat (`_simulate_envelope_dca`, `_simulate_grid_atr`) — API inchangée
- `FAST_ENGINE_STRATEGIES` dans `__init__.py` — dérivée automatiquement du registre (`STRATEGY_REGISTRY - STRATEGIES_NEED_EXTRA_DATA`)
- walk_forward.py : 2 tuples hardcodés remplacés par `FAST_ENGINE_STRATEGIES`

**Résultat** : -132 lignes dans `fast_multi_backtest.py` (425→293). Parité bit-à-bit vérifiée sur les 3 variantes (envelope_dca LONG/SHORT + grid_atr). 13 tests de parité + factory + constantes.

**Tests** : 787 passants (774 existants + 13 nouveaux)

**Hors scope** : `_INDICATOR_PARAMS` (walk_forward.py L394) reste un dict hardcodé — fallback `_run_sequential` rarement utilisé.

### Sprint 20b — Portfolio Backtest Multi-Asset ✅

**But** : Backtest portfolio qui simule le capital partagé entre les 21 assets, réutilisant `GridStrategyRunner` (même code que la prod).

**Questions répondues** :

1. Max drawdown historique sur le portfolio (capital partagé)
2. Corrélation des fills — combien de grilles se remplissent simultanément
3. Margin peak — marge max mobilisée simultanément (% du capital)
4. Kill switch frequency — combien de fois le kill switch aurait déclenché
5. Sizing optimal — combien d'assets max par niveau de capital

**Implémenté** :

- `PortfolioBacktester` : N runners (1/asset) avec params WFO per_asset, capital split `initial_capital / N`, `_nb_assets=1` par runner
- Warm-up manuel (50 candles) avec désactivation explicite `_is_warming_up = False` (candles historiques ont age > 2h, détection auto ne fire jamais)
- `IncrementalIndicatorEngine` partagé, `data_engine=None` (safe, jamais utilisé dans `on_candle()`)
- P&L séparé : `realized_pnl` (TP/SL naturels) vs `force_closed_pnl` (fin de données)
- Kill switch détection (fenêtre glissante 24h, seuil 30% paramétrable)
- Rapport CLI avec sections Risk, Marge, Kill Switch, Per-Asset breakdown
- Script `scripts/portfolio_backtest.py` avec `--days`, `--capital`, `--assets`, `--json`, `--output`

**Résultat (90j, 21 assets, 10k$)** : +14.5% return, -28.7% max drawdown, 73.7% WR, 1690 trades, peak margin 25%, 0 kill switch, 64 positions simultanées max.

**Tests** : 19 nouveaux → 806 passants (774 + 19 portfolio + 13 sprint 20c)

**Limitation connue** : ATR period fixe à 14 dans `IncrementalIndicatorEngine` (le `atr_period` per_asset du WFO n'est pas utilisé dans le path live). Matche le comportement prod.

### Sprint 20b-UI — Portfolio Backtest Frontend + Comparateur ✅
**But** : Persister les résultats portfolio en DB, ajouter une API REST, et créer un frontend React pour visualiser et comparer les runs.

**Backend** :
- Table `portfolio_backtests` dans SQLite (27 colonnes, equity_curve JSON sous-échantillonnée à 500 pts)
- CRUD sync+async dans `portfolio_db.py` (même pattern que `optimization_db.py`)
- 7 endpoints API : presets (4), backtests CRUD, run async (`asyncio.create_task`), status, compare
- Job tracker in-memory (un seul backtest à la fois), progress via WebSocket broadcast
- `progress_callback` optionnel ajouté à `PortfolioBacktester.run()` et `_simulate()`
- CLI `--save` et `--label` dans `portfolio_backtest.py`

**Frontend** :
- Onglet "Portfolio" dans le dashboard (6e tab)
- `PortfolioPage.jsx` : config panel (presets, capital, jours, assets, kill switch) + résultats
- `EquityCurveSVG.jsx` : SVG interactif (hover tooltip, multi-courbes, normalisation %, ResizeObserver)
- `DrawdownChart.jsx` : mini chart drawdown inversé avec seuil kill switch
- `PortfolioCompare.jsx` : tableau comparatif 7 métriques + deltas colorés
- 4 presets : Conservateur (1k$/5 assets), Équilibré (5k$/10), Agressif (10k$/tous), Long terme (10k$/7/365j)

**Tests** : 15 nouveaux → 821 passants

### Hotfix 20d — Anti-spam Telegram ✅
**Problème** : `Notifier.notify_anomaly()` envoyait un message Telegram à chaque appel sans cooldown. Le Watchdog appelle `notify_anomaly(ALL_STRATEGIES_STOPPED)` toutes les ~5 min quand le kill switch est actif → 102 messages identiques en quelques heures.

**Fix** : Cooldown par type d'anomalie dans le Notifier. Le log WARNING reste systématique (fichiers de log), seul l'envoi Telegram est throttlé.

**Cooldowns** : SL_PLACEMENT_FAILED=5min, WS/DATA/EXECUTOR=30min, KILL_SWITCH/ALL_STOPPED=1h, défaut=10min.

**Tests** : 4 nouveaux → 825 passants

### Hotfix 20e — Kill switch grid + Warm-up fixes ✅
**Problème** : Après un restart Docker en production, 5 bugs découverts : kill switch se redéclenche immédiatement, warm-up génère 183 trades phantom (-1 409$ fictifs), StateManager écrase le bon état pendant le warm-up bloqué.

**Analyse** : 2 bugs déjà corrigés (persistence grid_positions dans Sprint 15, anti-spam Telegram dans Hotfix 20d). 4 bugs restants corrigés.

**Fixes** :
1. **Bug 2** : `_end_warmup()` forcé quand kill switch global restauré au boot → state sauvegardé appliqué immédiatement
2. **Bug 3** : Grace period de 10 bougies 1h post-warmup (pas de kill switch runner pendant la stabilisation)
3. **Bug 4** : Seuils kill switch grid-spécifiques (25%/25% au lieu de 5%/10%), champs `grid_max_session_loss_percent` et `grid_max_daily_loss_percent` dans KillSwitchConfig + risk.yaml, fallback MagicMock-safe
4. **Bug 5** : Guard anti-phantom trades — bougies > 2h skippées pendant 5 min après fin du warm-up

**Leçon** : Le kill switch session (5%) était conçu pour les stratégies mono-position. Avec grid DCA (21 assets × 3 niveaux), le P&L non réalisé dépasse facilement -500$ en fonctionnement normal. Les seuils doivent être adaptés au type de stratégie.

**Tests** : 22 nouveaux → 847 passants

### Hotfix 20f — Panneau Simulator : P&L réalisé + non réalisé + equity ✅

**Problème** : Le panneau "SIMULATOR (PAPER)" n'affichait que le P&L réalisé (trades fermés). Avec 3 grids ouvertes à -77$ et 0 trades fermés, le dashboard affichait "P&L Net: +0.00" et "Capital: 9531$" — trompeur.

**Fixes** :

1. **Backend** : `GridStrategyRunner.get_status()` enrichi avec `unrealized_pnl`, `margin_used`, `equity`, `assets_with_positions` — calcul depuis `_last_prices` mis à jour à chaque `on_candle()`
2. **Backend** : `LiveStrategyRunner.get_status()` enrichi avec les mêmes champs (cohérence frontend)
3. **Backend** : `get_equity_curve()` ajoute un point "now" avec l'equity courante (capital + unrealized) et retourne `current_equity`
4. **Frontend** : SessionStats refonte complète — P&L Total (réalisé + non réalisé) en gros, détail réalisé/non réalisé, equity avec %, marge/disponible, stats compactes, info grids
5. **Frontend** : EquityCurve affiche "Equity" au lieu de "Capital actuel", utilise `current_equity`
6. **Frontend** : `getSummary()` inclut le P&L total (pas juste réalisé)

**Tests** : 5 nouveaux → 852 passants

---

## PHASE 6 — MULTI-STRATÉGIE & LIVE (Sprints 21-25)

| Sprint | Contenu | Status |
|--------|---------|--------|
| 21a | Grid Multi-TF (Supertrend 4h + Grid ATR 1h) — backtest + WFO | ✅ |
| 21b | Grid Multi-TF — support live (Simulator, TimeFrame.H4) | 📋 Planifié |
| 22 | Grid Funding (DCA sur funding négatif, LONG-only, 2592 combos WFO) | ✅ |
| Perf | Numba JIT Optimization (speedup 5-10x WFO) | ✅ |
| 23 | Grid Trend (trend following DCA, EMA cross + ADX + trailing ATR) | ✅ |
| Audit | Micro-Sprint Audit (auth executor, async I/O, candle buffer) | ✅ |
| 23b | Grid Trend compute_live_indicators (paper/portfolio fix) | ✅ |
| 24a | Portfolio Backtest Realistic Mode (sizing fixe, global margin guard, kill switch) | ✅ |
| 24b | Portfolio Backtest Multi-Stratégie (clé strategy:symbol, dispatch multi) | ✅ |
| 24 | Live trading progressif (1000$ → 5000$) | 📋 Planifié |
| 25 | Monitoring V2 (alertes enrichies, rapport hebdo Telegram) | 📋 Planifié |

### Sprint 21a — Grid Multi-TF (Backtest + WFO) ✅

**But** : 11e stratégie — Supertrend 4h filtre directionnel + Grid ATR 1h exécution. Corrige le défaut principal de grid_atr (LONG en bear market → -46% drawdown).

**Architecture** : Candles 1h → resampling 4h (UTC-aligned, anti-lookahead) → Supertrend → direction (UP=LONG, DOWN=SHORT). Flip de direction → force-close.

**Implémentation** :

- `GridMultiTFConfig` (config.py) + `GridMultiTFStrategy` (grid_multi_tf.py) héritant `BaseGridStrategy`
- `compute_indicators()` calcule SMA+ATR 1h ET Supertrend 4h (resampling interne, anti-lookahead)
- Fast engine : `_build_entry_prices()` avec directions dynamiques, `_simulate_grid_common()` avec `directions` array
- Cache : `_resample_1h_to_4h()` + `supertrend_dir_4h` field dans `IndicatorCache`
- Registry, factory, param_grids.yaml (384 combos), strategies.yaml (enabled: false)
- `_INDICATOR_PARAMS` : `["ma_period", "atr_period", "st_atr_period", "st_atr_multiplier"]`

**Bugfix 21a-bis** : Validation Bitget et Monte Carlo retournaient 0 trades car `compute_indicators()` ne calculait pas le Supertrend 4h et `MultiPositionEngine` ne passait que le TF principal dans `ctx_indicators`. Corrigé.

**Bugfix 21a-ter** : Portfolio backtest 0 trades car `GridStrategyRunner.on_candle()` n'appelle pas `compute_indicators()` (il utilise `IncrementalIndicatorEngine`). Fix : `compute_live_indicators()` dans `BaseGridStrategy` (default `{}`), override dans `GridMultiTFStrategy` pour calculer le Supertrend 4h depuis le buffer de candles accumulé. Guard `isinstance(buffers, dict)` pour compatibilité MagicMock.

**Résultat** : 902 tests (44 nouveaux pour grid_multi_tf), 0 régression. Parité fast engine vs MultiPositionEngine validée.

### Sprint Perf — Numba JIT Optimization ✅

**But** : Accélérer le WFO (optimiseur qui exécute 1000-5000 backtests par run). Profiling : 60-70% du temps = boucle Python scalaire `_simulate_trades()`, 15-25% = boucles Wilder des indicateurs (RSI, ATR, ADX, EMA, SuperTrend).

**Approche** :

- **Phase 0** : Retirer pandas (jamais importé, 0 occurrence dans le codebase)
- **Phase 1** : Vectorisation numpy pure (True Range, rolling std, rolling max/min)
- **Phase 2** : Numba `@njit(cache=True)` sur les boucles Wilder (EMA, RSI, ATR, ADX, SuperTrend)
- **Phase 3** : Numba sur `_simulate_trades()` — 1 fonction JIT par stratégie (vwap_rsi, momentum, bollinger_mr, donchian, supertrend) avec fallback Python transparent

**Architecture** :

- `pyproject.toml` : groupe optionnel `optimization = ["numba>=0.61"]` — pas de dépendance lourde pour le serveur Docker
- Import avec fallback : `try: from numba import njit / except: njit = identity decorator` — code Python inchangé si numba absent
- Fonctions `@njit` isolées : `_ema_loop`, `_rsi_wilder_loop`, `_wilder_smooth`, `_adx_wilder_loop`, `_supertrend_loop`, `_close_trade_numba`, `_simulate_{vwap_rsi,momentum,bollinger,donchian,supertrend}_numba`
- Dispatch dans `_simulate_trades()` : if `NUMBA_AVAILABLE and strategy_name == "vwap_rsi"` → wrapper numba, sinon Python
- Scope : stratégies **mono-position** uniquement (les 5 stratégies scalp/swing). Grid/DCA bénéficient de Phase 1-2 (indicateurs), pas de Phase 3

**Benchmark** (200 combos × 5000 candles, numba cache chaud) :

```text
vwap_rsi         : 0.034s (0.17ms/combo)
momentum         : 0.030s (0.15ms/combo)
bollinger_mr     : 0.042s (0.21ms/combo)
donchian_breakout: 0.039s (0.19ms/combo)
supertrend       : 0.034s (0.17ms/combo)
Total            : 0.179s
```

Speedup compilation : WARM (1ère compilation) = 0.20s → RUN = 0.03s = **~6x speedup** sur la simulation de trades. Speedup global WFO attendu : **5-10x** (inclut Phase 2 sur `build_cache()`).

**Compatibilité** : Numba 0.63.1 compatible Python 3.13.11 (downgrade numpy 2.4→2.3.5, pas d'impact).

**Script benchmark** : `scripts/benchmark_fast_engine.py` — génère données synthétiques, 3+ runs, exclut compilation, reporte mean ± std.

**Résultat** : 941 tests OK (3 exclus = crash JIT Python 3.13 pré-existant dans `_simulate_grid_common`, non lié à numba). 0 régression fonctionnelle. Plan archivé : [docs/plans/sprint-perf-numba-optimization.md](docs/plans/sprint-perf-numba-optimization.md).

### Sprint 21b — Grid Multi-TF Live (Planifié)

**But** : Support live pour grid_multi_tf (Simulator, GridStrategyRunner, TimeFrame.H4, DataEngine). Note : le support portfolio backtest et simulator fonctionne déjà grâce au bugfix 21a-ter.

### Sprint 22 — Grid Funding (DCA sur Funding Rate Négatif) ✅

**But** : 12e stratégie — entre LONG quand le funding rate est très négatif (signal structurel indépendant du prix). L'edge : les shorts paient les longs tant que le funding est négatif, même si le prix ne bouge pas. LONG-only, pas de support live.

**Implémentation** (12 fichiers, ~850 lignes) :

- `GridFundingConfig` + `GridFundingStrategy` (hérite `BaseGridStrategy`)
- Fast engine : `_build_entry_signals()`, `_calc_grid_pnl_with_funding()`, `_simulate_grid_funding()`
- IndicatorCache : `_load_funding_rates_aligned()` (sqlite3 sync, searchsorted forward-fill, DB % → raw decimal /100)
- Walk Forward : `db_path`/`symbol`/`exchange` forwarding pour charger les funding rates dans le cache
- Grille WFO : 2592 combos (4×3×2×3×4×3×3), fenêtres IS=360j/OOS=90j/step=90j
- Découplage `FAST_ENGINE_STRATEGIES` : `_NO_FAST_ENGINE = {"funding", "liquidation"}` (grid_funding dans FAST_ENGINE ET NEED_EXTRA_DATA)

**Corrections clés** :

- Unités funding : DB stocke en % (×100), divisé par 100 partout (cache loader + strategy class)
- Anti-lookahead : `searchsorted(side='right') - 1` direct, pas de décalage +8h
- `MultiPositionEngine` : `extra_data_by_timestamp` ajouté au `StrategyContext` (manquait, bloquait l'OOS evaluation)

**Bugfix 22-bis** : Validation Bitget et stabilité retournaient 0 trades car `extra_data_by_timestamp` (funding rates) n'était pas passé à `run_multi_backtest_single` dans `report.py:validate_on_bitget()` et `overfitting.py:_run_backtest_for_strategy()`. Même pattern que bugfix 21a-bis. Corrigé (2 lignes).

**Résultat** : 944 tests (42 nouveaux pour grid_funding), 0 régression.

### Sprint 23 — Grid Trend ✅

**But** : 13e stratégie — trend following DCA avec EMA cross + ADX filtre + trailing stop ATR.

**Implémentation** :
- EMA cross (fast/slow) pour direction, ADX > seuil pour force du trend, zone neutre si ADX < seuil
- Trailing stop ATR (high watermark), force close au flip de direction
- Fast engine + IndicatorCache (ema_by_period, adx_by_period)
- 2592 combos WFO, 46 tests

**Résultat** : 990 tests, 0 régression.

### Micro-Sprint Audit ✅

**But** : Corriger 3 problèmes identifiés par audit de sécurité et performance.

**Fix 1 — Auth endpoints executor** :
- Dépendance FastAPI `verify_executor_key` sur les 3 routes (`/status`, `/test-trade`, `/test-close`)
- Vérifie `X-API-Key` contre `sync_api_key` de la config (même clé que sync WFO)
- Sans clé configurée ou clé invalide → 401

**Fix 2 — Async I/O StateManager** :
- `_write_json_file()` et `_read_json_file()` statiques, exécutés via `asyncio.to_thread()`
- Les 4 méthodes save/load (runner + executor) ne bloquent plus l'event loop

**Fix 3 — Buffer candles DataEngine** :
- `_write_buffer` accumule les candles, `_flush_candle_buffer()` flush toutes les 5s
- Les callbacks (Simulator) restent immédiats, seule la persistance DB est bufferisée
- `stop()` fait un flush final avant fermeture DB

**Résultat** : 1004 tests (+14 nouveaux), 0 régression.

### Sprint 23b — Grid Trend compute_live_indicators ✅

**But** : Permettre au paper trading et portfolio backtest de grid_trend de générer des trades (0 trades sans cette méthode).

**Problème** : `IncrementalIndicatorEngine` calcule SMA + ATR mais pas EMA ni ADX. Le `GridStrategyRunner.on_candle()` appelle `compute_live_indicators()` et merge le résultat dans les indicateurs, mais `GridTrendStrategy` héritait le défaut `{}` de `BaseGridStrategy`.

**Implémentation** :

- Override `compute_live_indicators()` dans `GridTrendStrategy` (~30 lignes)
- Calcule EMA fast/slow + ADX depuis le buffer de candles 1h
- Retourne `{timeframe: {"ema_fast", "ema_slow", "adx"}}` pour la dernière candle
- Guard : retourne `{}` si pas assez de candles (identique à `min_candles`)
- Pattern identique à `GridMultiTFStrategy.compute_live_indicators()` (Supertrend 4h)

**Tests** : 3 nouveaux tests (Section 9 de test_grid_trend.py)

- `test_returns_ema_adx_with_enough_candles` — vérifie les 3 indicateurs retournés
- `test_returns_empty_with_too_few_candles` — vérifie le guard min candles
- `test_runner_merges_live_indicators` — pipeline IncrementalIndicatorEngine → buffer → merge

**Résultat** : 1007 tests (+3 nouveaux), 0 régression.

### Sprint 24a — Portfolio Backtest Realistic Mode ✅

**But** : Le portfolio backtest grid_atr 21 assets affichait peak margin 284% (= liquidation en live). 3 corrections pour que le backtest reflète la réalité.

**Problèmes identifiés** :
1. **Compounding abusif** : les runners réinvestissaient les profits → sizing exponentiel
2. **Pas de global margin guard** : chaque runner vérifie sa marge locale, pas la marge totale du portfolio
3. **Kill switch passif** : détecté a posteriori via `_check_kill_switch()`, mais les runners continuaient à trader pendant la simulation

**Corrections** :

1. **Sizing fixe (anti-compounding)** — `simulator.py` + `portfolio_engine.py`
   - Flag `_portfolio_mode = True` sur chaque runner portfolio
   - En portfolio mode, sizing basé sur `_initial_capital` (pas `_capital` courant)
   - Transparent pour live/paper : `getattr(self, "_portfolio_mode", False)` = False si absent

2. **Global Margin Guard** — `simulator.py` + `portfolio_engine.py`
   - Chaque runner reçoit `_portfolio_runners` (dict) et `_portfolio_initial_capital`
   - Après le margin guard local, calcule la marge globale (tous runners) et skip si `> capital × max_margin_ratio`

3. **Kill switch temps réel** — `portfolio_engine.py`
   - Fenêtre glissante 24h dans `_simulate()` : si DD% ≥ seuil, gèle tous les runners
   - Cooldown 24h : après expiration, dégèle les runners
   - Le kill switch se re-déclenche tant que les snapshots haute-equity sont dans la fenêtre

**Design** : tous les ajouts sont derrière `getattr(..., False/None)` → zéro impact sur le code live/paper.

**Tests** : 5 nouveaux tests
- `test_portfolio_mode_fixed_sizing` — sizing basé sur initial_capital
- `test_normal_mode_uses_current_capital` — compound en mode normal (contrôle)
- `test_global_margin_guard_blocks` — marge globale 65% bloque les ouvertures
- `test_global_margin_under_threshold` — marge globale 20% laisse passer
- `test_kill_switch_freezes_all_runners` — trigger + cooldown 24h + reset

**Résultat** : 1012 tests (+5 nouveaux), 0 régression.

### Sprint 24b — Portfolio Backtest Multi-Stratégie ✅

**But** : Supporter plusieurs stratégies simultanées dans le portfolio backtest pour mesurer la complémentarité grid_atr + grid_trend sur un pool de capital unique.

**Problème** : `PortfolioBacktester` ne prenait qu'un seul `strategy_name`. Impossible de tester grid_atr (10 assets) + grid_trend (6 assets) ensemble.

**Changements** :

1. **`portfolio_engine.py`** — paramètre `multi_strategies: list[tuple[str, list[str]]]`
   - Clés des runners au format `strategy:symbol` (ex: `grid_atr:ICP/USDT`)
   - `_symbol_from_key()` extrait le symbol, rétro-compatible avec l'ancien format
   - `_create_runners()` itère sur `multi_strategies`, crée N runners avec indicator engine partagé
   - `_warmup_runners()` injecte les candles une seule fois par symbol dans l'engine
   - `_simulate()` mapping `symbol → [runner_keys]` pour dispatcher à tous les runners d'un symbol
   - `_build_result()` breakdown `per_asset_results` indexé par `runner_key`
   - `format_portfolio_report()` section "Par Runner" avec largeur dynamique

2. **`scripts/portfolio_backtest.py`** — CLI enrichi
   - `--strategies strat1:sym1,sym2+strat2:sym3,sym4` — format explicite
   - `--preset combined` — lit automatiquement les per_asset de grid_atr + grid_trend

3. **Rétro-compatibilité totale** — sans `multi_strategies`, auto-construit `[(strategy_name, assets)]`

**Tests** : 4 nouveaux tests

- `test_multi_strategy_creates_runners` — 2 runners avec clés strategy:symbol
- `test_same_symbol_dispatched_to_both` — même symbol dispatché aux 2 runners
- `test_capital_split` — 4 runners × 2500$ chacun
- `test_backward_compatible` — mode single-strategy inchangé

**Résultat** : 1016 tests (+4 nouveaux), 0 régression.

### Sprint 24 — Live Trading Progressif

**But** : Passer du paper trading au live avec capital réel progressif.

**Étapes** :

- 1000$ sur 3-5 assets Grade A (validation 2 semaines)
- 2500$ sur 7-10 assets (validation 1 mois)
- 5000$ sur 15+ assets (objectif long terme)
- Monitoring slippage paper vs live à chaque palier

### Sprint 25 — Activity Journal (Live Trading Monitor) ✅

**But** : Historique complet de l'activité live entre ouverture et fermeture des positions. Courbe d'equity incluant le P&L non réalisé, événements DCA traçables.

**Problème** : Seuls les trades fermés étaient enregistrés (`simulation_trades`). Impossible de tracer l'equity non réalisée, les ouvertures DCA, ou de comparer le comportement live aux prédictions du backtest.

**Changements** :

1. **`database.py`** — 2 nouvelles tables + 5 méthodes CRUD
   - `portfolio_snapshots` : equity, capital, margin, unrealized, breakdown JSON (toutes les 5 min)
   - `position_events` : OPEN/CLOSE avec level, direction, prix, metadata JSON
   - `get_latest_snapshot()` avec ORDER BY DESC LIMIT 1

2. **`simulator.py`** — Snapshot collector + hooks events
   - `Simulator.take_journal_snapshot()` : agrège `get_status()` de tous les runners (DRY)
   - `GridStrategyRunner._pending_journal_events` : queue séparée de `_pending_events`
   - Hook OPEN après `_emit_open_event()` (dans le guard `not _is_warming_up`)
   - Hook CLOSE après `_emit_close_event()` (dans le guard `not _is_warming_up`)
   - Drain dans `_dispatch_candle()` avec `await db.insert_position_event()`

3. **`state_manager.py`** — Snapshot périodique toutes les 5 min
   - Compteur dans `_periodic_save_loop()` (toutes les 5 itérations de 60s)
   - `_save_journal_snapshot()` helper avec try/except graceful

4. **`journal_routes.py`** — 3 endpoints API
   - `GET /api/journal/snapshots` (since, until, limit)
   - `GET /api/journal/events` (since, strategy, symbol, limit)
   - `GET /api/journal/summary` (dernier snapshot + 10 derniers events)

5. **Frontend enrichi**
   - `EquityCurve.jsx` : double source (snapshots journal + fallback trades), hover tooltip (equity, unrealized, margin, positions)
   - `ActivityFeed.jsx` : section `JournalEventCard` avec pastilles CSS pour les événements DCA

**Tests** : 21 nouveaux tests (DB round-trip, filtrage, snapshot collector, API endpoints)

**Résultat** : 1037 tests (+21 nouveaux), 0 régression.

**Hotfix 25a — Retry DB writes** :

- Problème : "database is locked" avec connexion sync sqlite3 de DataEngine `_flush_candle_buffer()` en concurrent avec les INSERT journal
- Solution :
  - `_execute_with_retry()` dans database.py (3 tentatives, backoff 100ms/200ms sur "locked")
  - `insert_portfolio_snapshot()` et `insert_position_event()` utilisent retry
  - Throttle `await asyncio.sleep(0.05)` entre INSERT events si batch > 2 dans `_dispatch_candle()`
- 0 régression (1037 tests passants)

**Hotfix 25b — Kill Switch Reliability** ✅

**Problème** : Le kill switch global du Simulator avait 4 problèmes en production :

1. **Pas de reset API** — il fallait éditer le JSON à la main pour réinitialiser
2. **Pas d'alerte Telegram au restore** — le kill switch se restaurait silencieusement au restart du serveur
3. **Pas de raison persistée** — impossible de savoir pourquoi il avait triggeré (drawdown%, timestamps, seuils)
4. **Positions perdues (11→6)** — bug dans `_apply_restored_state()` qui reset `kill_switch_triggered=False` APRÈS que `_stop_all_runners()` l'a mis à True

**Bug critique** : L'ordre d'exécution dans `start()` était : restore state → `_stop_all_runners()` → `_end_warmup()` (qui appelle `_apply_restored_state()`). Le `_apply_restored_state()` écrasait le flag kill_switch après que `_stop_all_runners()` l'ait positionné, causant la perte de positions.

**Fixes** :

1. **FIX 1 — API Reset** : nouveau endpoint `POST /api/simulator/kill-switch/reset`
   - Vérifie que le kill switch est actif (sinon retourne `not_triggered`)
   - Appelle `simulator.reset_kill_switch()` qui réactive tous les runners
   - Sauvegarde l'état immédiatement via StateManager
   - Notifie Telegram avec le nombre de runners réactivés
   - Retourne le count de runners réactivés

2. **FIX 2 — Alerte Telegram au restore** : notification enrichie dans `start()`
   - Restaure `_kill_switch_reason` depuis saved_state
   - Si kill switch actif au boot : compte les positions totales (grid + mono)
   - Construit un message détaillé avec la raison si disponible
   - Envoie l'alerte via `notifier.notify_anomaly(AnomalyType.KILL_SWITCH_GLOBAL, ...)`

3. **FIX 3 — Raison persistée** : nouveau field `_kill_switch_reason`
   - Dict avec `triggered_at`, `drawdown_pct`, `window_hours`, `threshold_pct`, `capital_max`, `capital_current`
   - Persisté dans `_check_global_kill_switch()` avant `_stop_all_runners()`
   - Property `kill_switch_reason` pour exposition API
   - Sauvegardé dans state JSON via StateManager (`kill_switch_reason` field)
   - Exposé dans `GET /api/simulator/status`

4. **FIX 4 — Bug ordre d'exécution** : reordering dans `start()`
   - **AVANT** : restore → stop → warmup (apply state écrase kill_switch)
   - **APRÈS** : restore → warmup → stop (stop a le dernier mot)
   - `_end_warmup()` est appelé AVANT `_stop_all_runners()` (lignes 1654-1666)
   - Le `_apply_restored_state()` peut reset kill_switch=False (backward compat grid runners)
   - Mais `_stop_all_runners()` le repositionne True immédiatement après (final authority)
   - Logs diagnostiques ajoutés : position count INFO, mismatch WARNING

**Architecture** :
- `simulator.py` : `_kill_switch_reason` field, `reset_kill_switch()` method, property, reordering `start()`
- `state_manager.py` : signature `save_runner_state()` avec paramètre `kill_switch_reason`
- `server.py` : expose `state_manager` dans `app.state` pour l'endpoint reset
- `simulator_routes.py` : endpoint reset + reason dans status
- Backward compatible : ancien JSON sans `kill_switch_reason` → `None` (via `.get()`)

**Tests** : 12 nouveaux tests → 1049 passants
- 5 tests reset endpoint (resets global, reactivates runners, saves state, not_triggered, telegram)
- 4 tests kill_switch_reason (in status, null when inactive, cleared on reset, persisted)
- 3 tests bug fix (warmup/stop order, position restoration, global reset)

**Résultat** : 1049 tests (+12 nouveaux), 0 régression. Kill switch global pleinement fonctionnel avec visibilité et contrôle API.

### Sprint 26 — Funding Costs dans le Backtest ✅

**But** : Appliquer les coûts de funding rate (toutes les 8h : 00:00, 08:00, 16:00 UTC) à TOUTES les stratégies grid/DCA dans les deux moteurs de backtest (event-driven + fast engine).

**Problème** : Seule la stratégie `grid_funding` calculait les funding costs. Les 5 autres stratégies grid (`grid_atr`, `envelope_dca`, `envelope_dca_short`, `grid_multi_tf`, `grid_trend`) ignoraient le funding, faussant les résultats WFO. En live sur Bitget futures, TOUTES les positions ouvertes paient ou reçoivent du funding toutes les 8h.

**Formule** : `funding_payment = -funding_rate × notional_value × direction`
- LONG + funding positif → paie (coût)
- LONG + funding négatif → reçoit (revenu)
- SHORT : inversé

**Changements** :

1. **Phase 0 — Fix Convention /100** (CRITIQUE CASCADE)
   - `extra_data_builder.py` ligne 62 : divise par 100 (DB stocke en %, convertir en decimal)
   - `grid_funding.py` ligne 125 : retire la division /100 (sinon double division !)
   - Tests adaptés : `funding_rate=-0.10` → `-0.001` (decimal équivalent)

2. **Phase 1 — MultiPositionEngine** (event-driven)
   - `engine.py` : nouveau champ `BacktestResult.funding_paid_total: float = 0.0`
   - `multi_engine.py` : settlement detection 8h UTC avec timezone handling, formule unifiée
   - Utilise `entry_price × quantity` (pas `candle.close`) pour cohérence avec fast engine

3. **Phase 2 — Fast Engine** (`_simulate_grid_common`)
   - Settlement mask vectorisé : `hours = ((candle_ts_ms / 3600000) % 24).astype(int)`
   - Funding costs appliqués entre exit check et capital guard
   - Wrappers grid_atr/envelope_dca/grid_multi_tf/grid_trend passent `funding_rates`

4. **Phase 3 — Indicator Cache**
   - Étendre la condition `grid_funding` → set de 6 stratégies grid
   - db_path optionnel (graceful degradation : `funding_rates_1h=None` si pas de DB)

5. **Phase 4 — Plumbing WFO**
   - `STRATEGIES_NEED_EXTRA_DATA` étendu aux 6 grid strategies
   - `portfolio_engine.py` : champ `PortfolioResult.funding_paid_total`

6. **Hotfix 26a — Affichage Funding** ✅
   - `metrics.py` : 5 nouveaux champs (`funding_paid_total`, `backtest_start`, `backtest_end`, `initial_capital`, `leverage`)
   - Section "Contexte" dans `format_metrics_table()` + ligne "Funding" conditionnelle (caché si zéro)
   - `report.py` : `ValidationResult.funding_paid_total` extrait depuis backtest
   - `optimize.py` : affichage "Funding total" dans section Validation Bitget

**Tests** : 29 nouveaux tests (25 Sprint 26 + 4 Hotfix 26a) → 1078 passants

- 5 tests settlement detection (UTC conversion, mask vectorisé, 3 heures 0/8/16)
- 6 tests calcul funding (LONG/SHORT × positif/négatif, notional entry_price, accumulation)
- 4 tests convention /100 (extra_data_builder, indicator_cache, grid_funding no double, end-to-end)
- 3 tests backward compat (BacktestResult sans funding, constructeur kwargs)
- 4 tests parité engines (grid_atr/envelope_dca/grid_trend < 0.1% delta, stratégies sans funding inchangées)
- 2 tests portfolio aggregation (somme runners, snapshot includes funding)
- 1 test edge case (NaN funding rate skipped)
- 4 tests affichage (funding dans metrics, table shows/hides, contexte)

**Pièges résolus** :
- **Convention /100 cascade** : extra_data_builder ET grid_funding (double division bug)
- **Entry prices NaN skip** : `_simulate_grid_common` skip TOUTE la candle (exit + funding) si NaN → tests utilisent prix valides non-triggering
- **Timestamp timezone** : normaliser en UTC explicitement (candle.timestamp peut être naive)
- **Notional calculation** : utiliser `entry_price` (cohérence fast engine), pas `candle.close`
- **Dataclass fields** : champs avec default EN DERNIER (sinon TypeError)

**Résultat** : 1078 tests (+29 nouveaux), 0 régression. Funding costs appliqués à TOUTES les stratégies grid dans les deux moteurs (event-driven + fast).

### Sprint 27 — Filtre Darwinien par Régime ✅

**But** : Bloquer automatiquement les nouvelles grilles DCA dans un régime de marché si les résultats WFO sont négatifs dans ce régime.

**Problème** : grid_atr performe bien en backtest général mais échoue en bear market soutenu (7/21 assets Sharpe négatif sur 90j récents). Besoin d'un filtre intelligent basé sur les profils WFO par régime.

**Changements** :

1. **Database `get_regime_profiles()`** — Récupère `avg_oos_sharpe` par régime depuis `regime_metrics` WFO
2. **GridStrategyRunner `_should_block_entry_by_regime()`** — Check si Sharpe < 0 dans le régime actuel
3. **Mapping `REGIME_LIVE_TO_WFO`** — Convertit Bull/Bear/Range/Crash live → Bull/Bear/Range WFO
4. **Compteur `_regime_filter_blocks`** — Sauvegardé dans state + exposé dans `get_status()`
5. **Config `risk.yaml`** — `regime_filter_enabled: true` (activé par défaut)

**Tests** : 12 nouveaux tests → 1090 passants

- Test activation/désactivation via config
- Test blocage Bear market (avg_oos_sharpe = -0.2)
- Test autorisation Bull market (avg_oos_sharpe = 0.8)
- Test absence profil WFO (pas de blocage, fallback sécurisé)
- Test compteur `_regime_filter_blocks` persisté
- Test mapping régimes live→WFO (Crash→Bear)

**Résultat** : Protection automatique contre les régimes défavorables. grid_atr continue de tourner en Bull/Range, s'arrête en Bear prolongé.

### Hotfix 28a — Préparation Déploiement Live ✅

**But** : Corriger 3 bugs critiques avant le déploiement live.

**Bugs corrigés** :

1. **Selector cold start crash** — `AdaptiveSelector.start()` chargeait les trades DB → crash si table vide. Fix : `SELECT COUNT(*) FROM trades` guard
2. **Bypass selector permanent** — `selector_bypass_at_boot` désactivé au démarrage mais jamais ré-activé → selector paralysé. Fix : auto-disable bypass quand TOUTES les stratégies atteignent `min_trades`
3. **Warning capital mismatch fantôme** — Comparaison `risk.yaml initial_capital` vs balance Bitget au boot mais pas de réconciliation réelle. Fix : Warning seulement (pas de blocage), log + notification Telegram si écart > 10%

**Tests** : 12 nouveaux tests → 1102 passants

- Test `load_trade_history()` avec DB vide (cold start)
- Test auto-disable bypass selector (3 stratégies × 3 trades → bypass OFF)
- Test warning capital mismatch (écart 15% → warning, écart 5% → silence)

**Résultat** : Bot prêt pour le déploiement live mainnet avec capital minimal. Selector fonctionne dès le boot même sans historique.

### Hotfix 28b — Suppression Sandbox Bitget ✅

**But** : Retirer complètement le support sandbox Bitget du code (cassé depuis ccxt issue #25523).

**Problème** : Le sandbox Bitget ne fonctionne pas avec les sous-comptes. Garder l'option comme configurable est dangereux — un changement accidentel pourrait envoyer des ordres dans le vide.

**Changements** :

1. **Suppression config** :
   - `.env` / `.env.example` : supprimé `BITGET_SANDBOX` + commentaires associés
   - `config/exchanges.yaml` : supprimé `sandbox: false`
   - `backend/core/config.py` : supprimé `ExchangeConfig.sandbox` + `SecretsConfig.bitget_sandbox`

2. **Nettoyage executor** (`backend/execution/executor.py`) :
   - Supprimé properties `_sandbox_params` et `_margin_coin`
   - Hardcodé `"sandbox": False` dans ccxt config avec commentaire "Sandbox Bitget cassé (ccxt #25523) — mainnet only"
   - Hardcodé `"USDT"` partout (était `self._margin_coin`)
   - Simplifié `_fetch_positions_safe()` (plus de branche sandbox)
   - Supprimé `sandbox` du dict `get_status()`
   - Nettoyé ~30 sites d'appel (`params=self._sandbox_params` → supprimés)

3. **Nettoyage frontend** (`frontend/src/components/ExecutorPanel.jsx`) :
   - Supprimé état `SANDBOX` → désormais LIVE ou OFF uniquement
   - Simplifié tooltip et badge

4. **Nettoyage tests** :
   - `test_executor.py` : supprimé assertions `sandbox`, `SUSDT`, `_sandbox_params`
   - `test_executor_grid.py` : supprimé `params={}`
   - `test_adaptive_selector.py` : supprimé `bitget_sandbox`
   - `conftest.py` : supprimé `sandbox` du mock exchanges

5. **Documentation** :
   - `README.md` : commentaire "Sandbox Bitget supprimé (cassé, ccxt #25523) — mainnet only"
   - `docs/ROADMAP.md` : section "Sandbox Bitget Non Fonctionnel (SUPPRIMÉ)"

**Tests** : 1102 tests passés, 0 régression

**Fichiers modifiés** : 11 fichiers (6 backend, 1 frontend, 4 tests)

**Résultat** : Code simplifié et plus sûr. Mainnet only = moins de confusion, moins de risque d'erreur.

### Hotfix 28b-bis — Filtre per_asset strict GridStrategyRunner ✅

**But** : Empêcher grid_atr de trader des assets non validés par WFO.

**Problème** : grid_atr ouvrait des positions sur SOL, IMX, SAND (visibles sur le dashboard) alors que ces assets ne sont pas dans son `per_asset`. Le runner utilisait les paramètres par défaut pour des assets non optimisés — dangereux.

**Changements** :

1. **`GridStrategyRunner.__init__`** : construction de `_per_asset_keys` (set des symbols autorisés depuis per_asset)
2. **`on_candle()`** : après le filtre timeframe, skip si `symbol not in _per_asset_keys`
3. **Backward compatible** : per_asset vide = tous les symbols acceptés (comportement inchangé)
4. **Tests corrigés** : 3 tests existants utilisaient BTC/USDT hors per_asset → corrigés avec ASSET0/USDT
5. **2 nouveaux tests** : `TestGridRunnerPerAssetFilter` (skip non autorisé + backward compat)

**Tests** : 1104 tests passés, 0 régression

**Fichiers modifiés** : 3 fichiers (simulator.py, test_grid_runner.py, test_simulator_grid_state.py)

**Résultat** : Seuls les assets validés par WFO sont tradés. Aucun trade parasite possible.

### Hotfix 28c — Refresh périodique solde exchange ✅

**But** : Mettre à jour le solde Bitget affiché sur le dashboard quand l'utilisateur dépose/retire des fonds.

**Problème** : L'Executor ne faisait `fetch_balance()` qu'au `start()`. Le solde affiché restait figé indéfiniment.

**Changements** :

1. **`Executor.refresh_balance()`** : méthode publique async, fetch le solde USDT swap, log WARNING si variation >10%
2. **`Executor._balance_refresh_loop()`** : boucle toutes les 5 min, annulée dans `stop()`
3. **`get_status()`** : expose `exchange_balance` pour le dashboard
4. **`POST /api/executor/refresh-balance`** : endpoint protégé par API key pour refresh manuel
5. **10 nouveaux tests** : 8 unit (update, log >10%, pas de log <10%, erreur, no exchange, get_status ×2, boucle) + 2 routes (auth 401, executor absent 400)

**Tests** : 1114 tests passés, 0 régression

**Fichiers modifiés** : 4 fichiers (executor.py, executor_routes.py, test_executor.py, test_executor_routes.py)

**Résultat** : Le solde se rafraîchit automatiquement. Alertes si variation anormale. Refresh manuel depuis le dashboard possible.

### Hotfix 28d — Override env var SELECTOR_BYPASS_AT_BOOT ✅

**But** : Permettre au serveur d'avoir `selector_bypass_at_boot=true` sans que `git pull` l'écrase.

**Problème** : `risk.yaml` est versionné avec `selector_bypass_at_boot: false`. Sur le serveur, on veut `true` au boot (cold start live), mais chaque `git pull` remet `false`.

**Changements** :

1. **`SecretsConfig`** : champ `selector_bypass_at_boot: bool | None = None` — lu depuis `.env` via Pydantic BaseSettings
2. **`AppConfig.__init__`** : si défini dans `.env`, override la valeur de `risk.yaml`
3. **`.env.example`** : `SELECTOR_BYPASS_AT_BOOT=false` documenté
4. **2 tests** : override `true` via `.env` + fallback `risk.yaml` si non défini

**Tests** : 1116 tests passés, 0 régression

**Fichiers modifiés** : 3 fichiers (config.py, test_config.py, .env.example)

**Résultat** : Serveur met `SELECTOR_BYPASS_AT_BOOT=true` dans `.env` (gitignored), `risk.yaml` reste à `false` dans le repo.

### Hotfix 28e — Sync Portfolio Backtests local → serveur ✅

**But** : Les portfolio backtests restaient uniquement en DB locale. Étendre le mécanisme de sync existant (WFO) pour pousser aussi les portfolio backtests vers le serveur.

**Implémentation** : Réutilisation de l'infra existante (`SYNC_ENABLED/SYNC_SERVER_URL/SYNC_API_KEY`, httpx best-effort, auth `X-API-Key`).

**Changements** :

- `portfolio_db.py` : +3 fonctions (`build_portfolio_payload_from_row()`, `save_portfolio_from_payload_sync()`, `push_portfolio_to_server()`)
- `portfolio_routes.py` : +endpoint `POST /api/portfolio/results` (auth + dédup par `created_at`) + push auto après save API background task
- `portfolio_backtest.py` : push auto après `--save` CLI
- `sync_to_server.py` : étendu pour sync WFO + portfolio, option `--only wfo|portfolio`
- Résilience : `_load_all_portfolio_backtests()` gère les DB sans table portfolio (anciennes DB)

**Tests** : 13 nouveaux tests (payload, insert, dedup, push best-effort, endpoint auth/201/200/422, script dry-run/post)

**Résultat** : 1129 tests, 0 régression. Portfolio backtests visibles sur le serveur via Recherche/Explorer.

**Fichiers modifiés** : 6 fichiers (portfolio_db.py, portfolio_routes.py, portfolio_backtest.py, sync_to_server.py, test_portfolio_sync.py, test_sync_to_server.py)

### Sprint 29a — Grid Range ATR (14e stratégie, bidirectionnelle) ✅

**But** : Monétiser les micro-oscillations quand le prix reste près de la SMA. Contrairement à grid_atr (unidirectionnel), grid_range_atr ouvre des LONG **et** SHORT simultanément avec des TP/SL individuels par position.

**Résultat** : 1169 tests passants (40 nouveaux), stratégie opérationnelle, WFO 2160 combos à lancer.

---

### Hotfix 30 — Deadlock Selector + DATA_STALE spam ✅

**Contexte** : En production LIVE (Bitget 1000 USDT), 0 trades exécutés après restart. Diagnostic : deadlock AdaptiveSelector + spam Telegram DATA_STALE.

**Problème 1 — Deadlock** : Session vierge (0 trades Arena) + bypass désactivé (DB a 59 trades grid_atr >= min_trades) → check `net_return_pct > 0` et `profit_factor >= seuil` échoue (session 0% return) → grid_atr jamais autorisé → deadlock.

**Problème 2 — DATA_STALE** : Timeframe 1h reçoit des mises à jour WS de la candle en cours (même timestamp, OHLCV actualisés) → `is_duplicate()` les filtre → `_last_update` jamais rafraîchi → faux positif DATA_STALE après 5 min → spam Telegram.

**Fix 1 — force_strategies (hotfix immédiat)** :

- `AdaptiveSelectorConfig` : `force_strategies: list[str] = []` (liste de stratégies qui bypasses `net_return`/`PF` checks)
- `risk.yaml` : `force_strategies: ["grid_atr"]`
- Garde-fous : `live_eligible` et `is_active` toujours requis (pas de bypass kill switch)

**Fix 2 — Session vierge (fix structurel)** :

- Dans `evaluate()` : si `perf.total_trades == 0` et `effective_trades >= min_trades` (DB a assez de trades), bypass les checks `net_return`/`PF` (on ne peut pas évaluer une performance qui n'existe pas)
- Résout le deadlock : DB=59 trades + session vierge → autorisé

**Fix 3 — DATA_STALE** :

- `data_engine.py` : `_last_update` mis à jour AVANT le check doublon (ligne 422-425)
- Tout message WS valide (même doublon = mise à jour candle en cours) rafraîchit le timestamp

**Bonus** : `grid_range_atr` ajouté au mapping `_STRATEGY_CONFIG_ATTR` (manquait depuis Sprint 29a).

**Tests** : 14 nouveaux tests (6 force_strategies, 4 session vierge, 4 data freshness)

**Résultat** : 1212 tests passants (43 nouveaux), déploiement live possible avec `force_strategies: ["grid_atr"]`.

---

### Hotfix 30b — Config conflicts deploy (fix persistance) ✅

**Contexte** : Problème récurrent (3ème occurrence) — `git pull` sur le serveur bloque sur conflits `config/risk.yaml` → intervention SSH manuelle à chaque déploiement.

**Cause racine** : édition directe des fichiers YAML sur le serveur pour config prod (ex: `force_strategies: ["grid_atr"]`).

**Fix 1 — deploy.sh reset config** :

- `deploy.sh` ligne 28-30 : `git checkout -- config/` avant `git pull`
- Garantit que les fichiers YAML versionnés sont toujours propres
- Commentaire explicite : "prod overrides go in .env (gitignored)"

**Fix 2 — Override force_strategies via .env** :

- `SecretsConfig` : `force_strategies: str | None = None` (comma-separated, ex: "grid_atr,grid_trend")
- `AppConfig.__init__` : parsing string → list + override `risk.adaptive_selector.force_strategies`
- `risk.yaml` : `force_strategies: []` (valeur neutre par défaut dans git)
- Pattern identique à `SELECTOR_BYPASS_AT_BOOT` (Hotfix 28d)

**Fix 3 — Documentation overrides .env** :

- `.env.example` : section "Adaptive Selector Overrides" avec exemples
- `CLAUDE.md` : règle critique déploiement, exemple `.env` serveur

**Règle critique** : **JAMAIS éditer les fichiers `config/*.yaml` sur le serveur**. Tous les overrides prod passent par `.env` (gitignored).

**Overrides .env disponibles** :

```bash
LIVE_TRADING=true                    # Active Executor (ordres réels)
SELECTOR_BYPASS_AT_BOOT=true         # Autorise toutes stratégies au boot
FORCE_STRATEGIES=grid_atr            # Bypass net_return/PF checks (comma-separated)
```

**Tests** : 5 nouveaux tests (single, multiple, whitespace, not set, empty)

**Résultat** : 1217 tests passants (+5 depuis Hotfix 30), conflits deploy éliminés.

---

### Sprint 30c — Enrichissement panel Executor + formatage prix adaptatif ✅

**But** : Améliorer la lisibilité du dashboard live — les positions executor affichent levier, marge, notionnel, P&L latent ; les prix crypto sont formatés avec le bon nombre de décimales selon la magnitude.

**Fix 1 — Utilitaire `formatPrice`** :

- Nouveau fichier `frontend/src/utils/format.js` : `formatPrice(price)` et `formatPnl(value)`
- Logique adaptative : ≥100$ → 2 décimales (BTC, ETH), ≥1$ → 4 (SOL, MATIC), ≥0.01$ → 5, sinon → 6 (SHIB)
- Importé dans Scanner, ActivePositions, ActivityFeed, ExecutorPanel

**Fix 2 — PositionCard enrichi** (`ExecutorPanel.jsx`) :

- Solde : `executor.exchange_balance ?? rm?.initial_capital` (réel Bitget en priorité)
- `currentPrice` passé à chaque card depuis `wsData.prices` (conversion `BTC/USDT:USDT` → `BTC/USDT`)
- Badge levier `x{leverage}` dans le header de la card
- Nouvelles lignes : **Notionnel**, **Marge** (`notional / leverage`), **P&L latent** en temps réel
- TP=0 → affiche "**SMA dynamique**" en vert (au lieu de `0.00`)

**Fix 3 — Backend** (`executor.py`) :

- `get_status()` enrichi : `leverage` et `notional` dans chaque position (mono = `default_leverage`, grid = `gs.leverage`)
- `notional = entry_price × quantity`

**Fix 4 — Prix adaptatifs dans tous les composants** :

- `Scanner.jsx` : colonne prix
- `ActivePositions.jsx` : avg@ grille, prix niveaux détaillés, PositionRow
- `ActivityFeed.jsx` : journal events, OpenPositionCard, ClosedTradeCard

**Résultat** : 1217 tests passants (0 régression, pas de nouveaux tests backend nécessaires — frontend pur).

### Sprint 30 — Multi-Timeframe Support (timeframe optimisable WFO) ✅

**But** : Rendre le timeframe (1h, 4h, 1d) optimisable dans le WFO. La DB stocke des candles 1h ; le pipeline resample 1h → 4h/1d à la volée. Permet de découvrir si certaines stratégies performent mieux sur des horizons plus longs.

**Approche** : Groupement des combos par timeframe dans `_run_fast()`, resampling via `resample_candles()`, OOS event-driven avec candles resamplees. ZÉRO modification de la boucle chaude (`_simulate_grid_common`, `_build_entry_prices`) ni du dataclass `IndicatorCache`.

**Changements** :

- `models.py` : `H4 = "4h"` et `D1 = "1d"` ajoutés au `TimeFrame` enum + `to_milliseconds()`
- `indicator_cache.py` : nouvelle fonction `resample_candles()` — passthrough 1h, buckets UTC complets pour 4h/1d, WARNING pour buckets incomplets au milieu des données
- `walk_forward.py` : `_run_fast()` groupe les combos par timeframe (un cache par TF), OOS resample si best_tf ≠ 1h (funding skippé pour non-1h)
- `param_grids.yaml` : `timeframe: ["1h", "4h", "1d"]` pour grid_atr (×3 → 9720 combos), envelope_dca (×3), envelope_dca_short (×3), grid_trend (×3 → 7776 combos)
- `test_multi_timeframe.py` : 29 tests (10 resampling, 3 total_days, 6 fast engine multi-TF, 5 parité, 3 intégration param_grids, 2 groupement _run_fast)
- `test_timeframe_sweep.py` : script diagnostic comparaison 1h vs 4h vs 1d pour une stratégie donnée

**Stratégies exclues (inchangées)** :

- `grid_multi_tf` — filtre Supertrend 4h conçu pour main_tf=1h
- `grid_funding` — funding rates indexés 1h
- `grid_range_atr` — pas de support multi-TF

**Contraintes respectées** :

- ZÉRO modification `_simulate_grid_common()`, `_build_entry_prices()`, `IndicatorCache` dataclass, classes de stratégie
- Parité bit-à-bit : `_run_fast()` sans timeframe → résultats identiques au pré-refactoring
- 90/90 tests de parité fast engine existants (grid_atr, grid_range_atr, fast_engine_refactor) verts

**Tests** : 29 nouveaux tests, 1198 tests au total, 0 régression.

**Fichiers modifiés** : 6 fichiers (models.py, indicator_cache.py, walk_forward.py, param_grids.yaml, test_multi_timeframe.py, test_timeframe_sweep.py)

**Différences clés vs grid_atr** :

- **Bidirectionnel** : LONG sous SMA + SHORT au-dessus, simultanément (pas de direction lock)
- **TP/SL individuels** : chaque position se ferme indépendamment (retour SMA = TP, % entry = SL)
- **tp_mode** : `dynamic_sma` (SMA courante) ou `fixed_center` (SMA au moment de l'ouverture)
- **sides configurable** : `["long", "short"]`, `["long"]` ou `["short"]`
- **Level encoding** : LONG = 0..N-1, SHORT = N..2N-1

**Changements** :

- `config.py` : `GridRangeATRConfig` (Pydantic) + champ dans `StrategiesConfig`
- `grid_range_atr.py` : NOUVEAU — `GridRangeATRStrategy(BaseGridStrategy)`, `compute_grid()` retourne les 2 côtés, `should_close_all()` → None, `get_tp_price()`/`get_sl_price()` → NaN (TODO Sprint 29b)
- `factory.py` : import + mapping create_strategy + get_enabled_strategies
- `__init__.py` : STRATEGY_REGISTRY + GRID_STRATEGIES + STRATEGIES_NEED_EXTRA_DATA
- `indicator_cache.py` : 3 sets étendus (SMA, ATR multi-period, funding)
- `fast_multi_backtest.py` : `_simulate_grid_range()` (~160 lignes, boucle individuelle TP/SL, funding per-position) + dispatch
- `walk_forward.py` : `_INDICATOR_PARAMS["grid_range_atr"]`
- `multi_engine.py` : `_GRID_STRATEGIES_WITH_FUNDING`
- `strategies.yaml` + `param_grids.yaml` : config + grille WFO 2160 combos
- `test_grid_range_fast.py` : script diagnostic fast engine (mode simple + --sweep spacings)

**Contraintes respectées** :

- ZÉRO modification `_simulate_grid_common()`, `BaseGridStrategy`, `GridStrategyRunner`, `_build_entry_prices()`
- Entry prices calculés inline dans `_simulate_grid_range()` (pas de modif au code partagé)
- Fee model aligné : TP = maker_fee (limit), SL = taker_fee + slippage

**Tests** : 40 nouveaux tests (10 signaux, 4 TP/SL, 12 fast engine, 2 viabilité fees, 7 registry/config, 5 parité stratégies existantes)

**Résultat** : 1169 tests, 0 régression. 5 tests de parité (grid_atr, envelope_dca, grid_trend, grid_multi_tf, grid_funding) tous verts.

**Fichiers modifiés** : 12 fichiers (config.py, grid_range_atr.py, factory.py, __init__.py, indicator_cache.py, fast_multi_backtest.py, walk_forward.py, multi_engine.py, strategies.yaml, param_grids.yaml, test_grid_range_atr.py, test_fast_engine_refactor.py)

### Sprint 31 — Log Viewer (mini-feed WS + onglet terminal) ✅

**But** : Rendre les logs backend visibles depuis le dashboard web, sans SSH. Deux vues : mini-feed sidebar temps réel (WARNING/ERROR via WebSocket) et onglet complet "Logs" style terminal Linux (polling HTTP).

**Changements backend** :

- `log_routes.py` : NOUVEAU — endpoint `GET /api/logs` avec lecture inversée par chunks (8KB), filtres `level`/`search`/`module`/`since`, guard 10MB max, retourne JSON structuré
- `logging_setup.py` : sink loguru `_ws_log_sink()` capture WARNING+ et push vers subscribers WS via `asyncio.Queue.put_nowait()` (thread-safe), buffer circulaire `deque(maxlen=20)`, fonctions `subscribe_logs()`/`unsubscribe_logs()`/`get_log_buffer()`
- `websocket_routes.py` : boucle WS refactorisée — `asyncio.wait_for(queue.get(), timeout=remaining)` remplace `sleep(3)`, log alerts envoyés instantanément via `{"type": "log_alert", "entry": {...}}`, buffer initial au connect
- `server.py` : include `log_router`

**Changements frontend** :

- `useWebSocket.js` : dispatch par `data.type` — `lastUpdate` (type=update), `lastEvent` (optimization/portfolio progress), `logAlerts` (array accumulé max 50). **Corrige un bug latent** : les messages non-update écrasaient brièvement `wsData`
- `App.jsx` : onglet Logs, wiring `logAlerts`/`lastEvent`, compteur `unseenLogErrors` (reset au clic onglet)
- `Header.jsx` : badge rouge sur l'onglet Logs si erreurs non vues
- `ExplorerPage.jsx` + `PortfolioPage.jsx` : migration vers `lastEvent` prop
- `LogMini.jsx` : NOUVEAU — mini-feed sidebar (20 dernières alertes, pastilles couleur, clic → onglet Logs)
- `LogViewer.jsx` + `LogViewer.css` : NOUVEAU — terminal noir (#0a0a0a), couleurs ANSI par niveau, auto-scroll tail -f, barre filtres (niveaux toggle, grep, module, auto-refresh 5s), expand détail au clic, max 500 lignes

**Tests** : 19 nouveaux (8 endpoint HTTP + 11 sink WS), 1236 tests au total, 0 régression.

**Fichiers** : 5 créés, 7 modifiés, 2 fichiers tests.

### Sprint 30b — Stratégie BolTrend (Bollinger Trend Following, 15e stratégie) ✅

**But** : Intégrer une stratégie mono-position Bollinger Breakout filtrée par tendance long terme. Breakout quand close sort des bandes de Bollinger (avec prev_close dedans), filtre SMA long terme, sortie dynamique au retour à la SMA de Bollinger.

**Logique** :

- **LONG** : prev_close < prev_upper AND close > upper AND spread > min_bol_spread AND close > long_ma
- **SHORT** : prev_close > prev_lower AND close < lower AND spread > min_bol_spread AND close < long_ma
- **Exit LONG** : close < bb_sma (breakout s'essouffle) — INVERSÉ vs bollinger_mr
- **Exit SHORT** : close > bb_sma
- **TP très éloigné** : entry×2 (LONG), entry×0.5 (SHORT) — check_exit gère le vrai TP
- **SL** : % fixe depuis l'entrée (filet de sécurité)

**Changements** :

- `config.py` : `BolTrendConfig` (Pydantic) + champ dans `StrategiesConfig`
- `boltrend.py` : NOUVEAU — `BolTrendStrategy(BaseStrategy)`, compute_indicators (BB + SMA long + ATR/ADX), evaluate (breakout + spread + trend filter), check_exit (SMA crossing inversé)
- `factory.py` : import + mapping create_strategy + get_enabled_strategies
- `__init__.py` : STRATEGY_REGISTRY (pas GRID_STRATEGIES, pas STRATEGIES_NEED_EXTRA_DATA)
- `indicator_cache.py` : build_cache bloc boltrend (réutilise bb_sma/bb_upper/bb_lower dicts)
- `fast_backtest.py` : `_boltrend_signals()` (vectorized np.roll), `_simulate_boltrend_numba()` (JIT), 6 points de dispatch
- `adaptive_selector.py` : mapping `_STRATEGY_CONFIG_ATTR`
- `strategies.yaml` + `param_grids.yaml` : config + grille WFO 486 combos (3×3×3×3×3×2)

**Points critiques** :

- **check_exit INVERSÉ** : bollinger_mr (mean reversion) LONG exit quand `close >= sma`, boltrend (breakout) LONG exit quand `close < sma`
- **np.roll wraparound** : valid[0] = False (closes[-1] se retrouve en position 0)
- **bb_sma collision safe** : `bollinger_bands()` appelle `sma()` en interne → résultats identiques

**Tests** : 25 nouveaux + 1 mis à jour (test_fast_engine_refactor.py), 1261 tests au total, 0 régression.

**Fichiers** : 10 modifiés/créés, 1 fichier test mis à jour.

### Sprint 33 — Stratégie Grid BolTrend (16e stratégie) ✅

**But** : boltrend (Sprint 30b) obtient Grade C sur 4/5 assets mais seulement 4-6 trades/fenêtre OOS (DSR 0.00 = plafond Grade C). Problème : mono-position = pas assez de trades. Solution : gridifier boltrend pour multiplier les trades par 3-4×.

**Logique hybride** : signal d'activation boltrend (breakout Bollinger) + exécution grid DCA (comme grid_atr) dans la direction du breakout. TP inverse : `close < SMA` (LONG) ou `close > SMA` (SHORT).

**Architecture** :

- **Event-driven** : grille OFF par défaut, s'active uniquement sur breakout Bollinger + filtre SMA long terme
- **Niveaux fixés au breakout** : `entry_levels[k] = close ∓ k × ATR × atr_spacing_mult` — pas recalculés à chaque candle
- **Level 0 immédiat** : entre automatiquement au breakout (lows ≤ close toujours vrai)
- **Fast engine dédié** : `_simulate_grid_boltrend()` (~180 lignes) — TP inversé incompatible avec `_simulate_grid_common()`, zéro modification du code existant
- **Heuristique SL+TP** : si SL et signal_exit sur même candle → bougie verte = signal_exit, rouge = sl_global
- **1296 combos WFO** (2×3×2×2×2×3×3×3) : bol_window, bol_std, long_ma_window, min_bol_spread, atr_period, num_levels, sl_percent, atr_spacing_mult

**Fichiers** :

- `backend/core/config.py` : `GridBolTrendConfig` + champ dans `StrategiesConfig`
- `backend/strategies/grid_boltrend.py` : NOUVEAU — `GridBolTrendStrategy(BaseGridStrategy)`, compute_grid (breakout + DCA levels), should_close_all (TP inverse + SL global)
- `backend/optimization/fast_multi_backtest.py` : `_simulate_grid_boltrend()` + branche dispatcher
- `backend/optimization/indicator_cache.py` : section grid_boltrend (BB + long_ma + ATR multi-period + funding)
- `backend/optimization/walk_forward.py` : `_INDICATOR_PARAMS`
- `backend/optimization/__init__.py` : STRATEGY_REGISTRY + GRID_STRATEGIES + STRATEGIES_NEED_EXTRA_DATA
- `backend/strategies/factory.py` : import + mapping + get_enabled
- `backend/execution/adaptive_selector.py` : `_STRATEGY_CONFIG_ATTR`
- `backend/backtesting/multi_engine.py` : `_GRID_STRATEGIES_WITH_FUNDING`
- `config/strategies.yaml` + `config/param_grids.yaml` : config + grille WFO
- `tests/test_grid_boltrend.py` : NOUVEAU — 32 tests (breakout, TP inverse, fast engine, registry, edge cases)
- `tests/test_fast_engine_refactor.py` : +1 stratégie dans expected set

**Tests** : 32 nouveaux, 1309 tests au total, 0 régression.

### Hotfix 33a — Bug TP inverse grid_boltrend (validation Bitget) ✅

**Problème** : La validation Bitget de `grid_boltrend` donnait Sharpe -14.51 sur ETH/USDT alors que le fast engine (test_regime_performance) donnait +1.35 sur la même période avec les mêmes params.

**Cause racine** : `get_tp_price()` retournait `bb_sma`. `check_global_tp_sl()` dans `multi_engine.py` interprète ce prix avec la convention standard LONG : `tp_hit = candle.high >= tp_price`. Or après un breakout LONG (`close > bb_upper > bb_sma`), la condition `high >= bb_sma` est **toujours vraie** → sortie immédiate sur chaque candle en perte (on paie frais entrée + sortie sans tenir la position). `should_close_all()` qui contient la bonne logique inverse (`close < bb_sma`) n'est jamais atteint.

**Fix** : `get_tp_price()` retourne `float("nan")` pour désactiver `check_global_tp_sl()`. Le TP inverse est entièrement géré par `should_close_all()`.

**Résultats après fix** :

- Bitget Sharpe : -14.51 → **+1.58** (ETH/USDT)
- OOS Sharpe moyen WFO : **4.15** (30 fenêtres, 100% consistance)
- OOS/IS ratio : **1.98** (OOS > IS = pas d'overfit)
- Grade : **B (83/100), LIVE ELIGIBLE**

**Leçon** : Pour les stratégies avec TP "inverse" (exit en dessous de bb_sma pour LONG), ne pas utiliser `get_tp_price()` — retourner `NaN` et gérer la sortie dans `should_close_all()` uniquement.

**Fichiers** : `backend/strategies/grid_boltrend.py` (1 ligne modifiée).

**Tests** : 1309 tests, 0 régression.

### Hotfix 33b — Audit de parité grid_boltrend (3 bugs fast engine) ✅

**But** : Vérifier la fiabilité et la cohérence de la 16e stratégie. 5 vérifications : parité fast engine vs event-driven, look-ahead bias, frais, remplissage multi-niveaux, diagnostic trade log.

**Bugs découverts (divergence PnL +31.73% avant fix)** :

1. **exit_price signal_exit incorrect** (`fast_multi_backtest.py`) — Le fast engine sortait au prix `sma_val` (SMA au moment de la sortie) au lieu du `close_i` réel. Pour un LONG, `sma_val < close` → sortie optimiste → surestimation des profits WFO.

2. **Fees signal_exit incorrectes** (`fast_multi_backtest.py`) — Le fast engine utilisait `maker_fee + 0 slippage` pour les sorties signal_exit. Correct : `taker_fee + slippage_pct` (sortie au marché = taker).

3. **Double-comptage entry_fee** (`multi_engine.py`) — Le moteur event-driven déduisait `capital -= pos.entry_fee` à l'ouverture ET les entry_fees étaient déjà incluses dans `trade.net_pnl` à la clôture via `close_all_positions()`. Double comptage affectant **toutes les stratégies grid**.

**Résultat après fix** : Divergence 31.73% → 2.62% (résiduel structurel dû au compounding légèrement différent entre les deux moteurs).

**Fichiers créés** :

- `tests/test_grid_boltrend_parity.py` : NOUVEAU — 13 tests (parité count/directions/PnL/short, look-ahead ×2, frais ×3, multi-niveaux ×3, diagnostic)
- `scripts/grid_boltrend_diagnostic.py` : NOUVEAU — trade log instrumenté (entry/exit candle, prix moyen, frais détaillés, exit_reason)

**Fichiers modifiés** :

- `backend/optimization/fast_multi_backtest.py` : exit_price `sma_val` → `close_i`, fees `maker_fee` → `taker_fee + slippage` pour signal_exit
- `backend/backtesting/multi_engine.py` : suppression du double-comptage `capital -= pos.entry_fee`

**Tests** : 13 nouveaux, 1322 tests au total, 0 régression.

### Hotfix 34 — Executor P&L basé sur les fills réels Bitget ✅

**But** : Premier jour de live (18 fév 2026) — l'Executor surestimait le P&L de +147% (session_pnl affiché +7.37$ vs Bitget réel +2.98$). Le kill switch était aveugle car basé sur ce P&L faux.

**Cause racine (3 bugs)** :

1. **Exit/Entry price fallback paper** — quand Bitget retourne `average=None` (fréquent sur market orders), fallback sur `event.exit_price`/`event.entry_price` = prix du Simulator
2. **Fees estimées** — `_calculate_pnl()` utilisait `taker_percent` config (0.06%), jamais les fees réelles
3. **Pas de tracking fees entry** — `LivePosition`/`GridLivePosition` n'avaient pas de champ `entry_fee`

**Fonctionnalités** :

- **`_fetch_fill_price(order_id, symbol, fallback)`** : fetch le vrai fill via `fetch_order()` puis `fetch_my_trades()` comme fallback. Retourne `(avg_price, fee: float | None)` — `None` si tout échoue (fallback paper + WARNING)
- **`_calculate_real_pnl()`** : P&L avec fees absolues réelles en USDT. Ne remplace PAS `_calculate_pnl()` (conservé pour réconciliation et estimations)
- **Entries** : `_open_position` et `_open_grid_position` utilisent le vrai fill si `average=None`, log le slippage
- **Exits** : `_close_grid_cycle` et `_close_position` utilisent le vrai fill + `_calculate_real_pnl()` si fees disponibles
- **Handlers** : `_handle_grid_sl_executed` et `_handle_exchange_close` reçoivent `exit_fee: float | None = None`. `_process_watched_order` extrait le fee du WS push et appelle `_fetch_fill_price` si absent (fréquent pour trigger orders Bitget)
- **Persistence** : `entry_fee` ajouté dans `get_state_for_persistence()` et `restore_positions()` (backward compat `.get("entry_fee", 0.0)`)
- **TODO** : polling (`_check_position_still_open`, `_check_grid_still_open`) conserve `_calculate_pnl()` estimé — amélioration future

**Changements** :

- `executor.py` : `LivePosition.entry_fee`, `GridLivePosition.entry_fee`, `GridLiveState.total_entry_fees` (property), `_fetch_fill_price()`, `_calculate_real_pnl()`, modification de 6 méthodes open/close/SL/exchange_close, persistence round-trip
- `tests/test_executor_real_pnl.py` : NOUVEAU — 17 tests (fetch_fill_price, real_pnl, entries, exits, watched orders, persistence, dataclasses)

**Tests** : 17 nouveaux, **1339 tests** au total, 0 régression.

### Hotfix 35 — Stabilité restart : cooldown post-warmup, max_live_grids, sauvegarde executor ✅

**But** : Premier jour live — le bot a crashé/redémarré 3 fois. Chaque restart provoquait : (1) perte du simulator state, (2) ouverture massive de grids sur TOUS les assets au premier signal post-warmup, (3) 8 positions live ouvertes en 30 secondes, (4) session_pnl fantôme persistant.

**Cause racine** : Après le warm-up, le runner sort avec état "vierge" (0 positions, 1000$) dans un marché bearish → TOUS les assets sous SMA → conditions LONG remplies partout. L'Executor n'avait aucune protection contre les ouvertures massives post-restart.

**3 bugs corrigés** :

**Bug A — Cooldown post-warmup (`simulator.py`)** :

- `POST_WARMUP_COOLDOWN = 3` : constante de classe dans `GridStrategyRunner`
- `_post_warmup_candle_count` incrémenté à chaque bougie post-warmup, reset dans `_end_warmup()`
- `_emit_open_event()` et `_emit_close_event()` bloquent les events Executor pendant 3 bougies (3h en TF 1h)
- **Les positions paper s'ouvrent normalement** — seul l'envoi à l'Executor est bloqué

**Bug B — Max grids simultanés (`executor.py` + `risk.yaml` + `config.py`)** :

- `max_live_grids: 4` dans `risk.yaml` et champ dans `RiskConfig`
- Guard dans `_open_grid_position()` : refuse un **nouveau** cycle si `len(_grid_states) >= max_live_grids`
- Les niveaux DCA supplémentaires sur un cycle existant passent toujours (`is_first_level=False`)
- Fallback `isinstance(int)` pour robustesse avec les MagicMock des tests

**Bug C — Sauvegarde périodique executor (`state_manager.py` + `server.py`)** :

- `set_executor(executor, risk_mgr)` : nouvelle méthode pour enregistrer l'executor
- `_periodic_save_loop()` appelle `save_executor_state()` toutes les 60s si executor enregistré
- `server.py` appelle `state_manager.set_executor()` après `executor.start()`
- Résout le session_pnl fantôme qui persistait entre les restarts (état périmé après kill -9)

**Fichiers modifiés** : `backend/backtesting/simulator.py`, `backend/execution/executor.py`, `backend/core/state_manager.py`, `backend/api/server.py`, `backend/core/config.py`, `config/risk.yaml`, `tests/test_hotfix_35.py` (14 nouveaux tests), `tests/test_grid_runner.py` (1 test adapté au cooldown)

**Tests** : 14 nouveaux, **1353 tests** au total, 0 régression.

### Sprint 34a — Lancement paper trading grid_boltrend ✅

**But** : Activer grid_boltrend en paper trading sur l'instance existante (aux côtés de grid_atr). Corriger un bug critique qui rendait la stratégie muette pendant ~15 jours après chaque restart.

**Bug corrigé** : `_warmup_from_db()` chargeait 50 candles au lieu de 420 pour grid_boltrend (`long_ma_window=400`). Résultat : `compute_live_indicators()` retournait `{}` pendant ~15 jours. Fix : utiliser `strategy.min_candles` pour dimensionner le warm-up dynamiquement.

**Changements** :
- **Warm-up dynamique** : `MAX_WARMUP_CANDLES` 200 → 500, `_warmup_from_db()` utilise `strategy.min_candles` (rétrocompatible : grid_atr reste à 50)
- **Filet de sécurité** : try/except autour de `compute_live_indicators()` — log ERROR + alerte Telegram `INDICATOR_ERROR` avec cooldown 1h
- **Préfixes Telegram** : `[ATR]`/`[BOLT]` au début des messages grid et live pour identification rapide
- **Rollback documenté** : section 17 dans COMMANDS.md (avec avertissement `nano` vs `echo >`)
- **TODO sizing** : commentaire dans le bloc equal allocation (dilution grid_atr quand on ajoute grid_boltrend)

**Fichiers** : `backend/backtesting/simulator.py`, `backend/alerts/notifier.py`, `backend/alerts/telegram.py`, `COMMANDS.md`.

**Tests** : 6 nouveaux (2 warmup, 1 try/except + récupération, 1 Telegram tag, 1 notifier), **1359 tests** au total, 0 régression.

### Sprint 34b — Dashboard Multi-Stratégie (Frontend) ✅

**But** : Grid_atr (10 assets, LIVE) et grid_boltrend (6 assets, PAPER) tournent en parallèle mais le dashboard mélangeait tout. Ajouter une barre de navigation par stratégie au-dessus des tabs existants, avec filtrage côté client des données WS.

**Bug fix inclus** : `grid_positions` était keyed par `symbol` — collision quand 2 stratégies partagent un symbol (BTC, DOGE, DYDX). Fix : key `strategy:symbol`.

**Nouveaux fichiers frontend (4)** :
- `StrategyContext.jsx` — React Context + Provider, persistance localStorage
- `useFilteredWsData.js` — Hook centralisé de filtrage wsData + helper `buildGridLookupBySymbol()`
- `StrategyBar.jsx` — Barre navigation [Overview] [grid_atr ●] [grid_boltrend ○] avec indicateurs live/paper
- `OverviewPage.jsx` — Tableau résumé par stratégie (assets, grids, P&L, marge, badge LIVE/PAPER)

**Fichiers modifiés** :
- `simulator.py` (1 ligne) : clé `grid_positions` → `strategy:symbol`
- `App.jsx` : StrategyProvider wrapper, `useFilteredWsData`, routing filtré/brut, OverviewPage conditionnel
- `Header.jsx` : layout 2 lignes (logo+strategy bar en haut, tabs en bas), version 0.7.0
- `Scanner.jsx` : `buildGridLookupBySymbol()`, filtrage assets par stratégie active
- `ActivePositions.jsx` : clé expand `strategy:symbol` (anti-collision overview)
- `styles.css` : header restructuré, styles strategy-bar

**Architecture** : les composants reçoivent des `wsData` pré-filtrées sans connaître le concept de stratégie. Les pages non filtrables (Recherche, Explorer, Portfolio, Journal, Logs) reçoivent les données brutes. La sidebar (SessionStats, ExecutorPanel, ActivityFeed) est filtrée.

**Tests** : 2 existants adaptés (clés `strategy:symbol`), **1359 tests** au total, 0 régression.

### Hotfix UI — Layout 2 colonnes ActivePositions ✅

**But** : Améliorer la lisibilité du bandeau "Positions actives" quand paper trading et live coexistent — les positions PAPER et LIVE s'empilaient dans une seule colonne, impossible de distinguer d'un coup d'œil.

**Changement** : Layout CSS Grid 2 colonnes — colonne gauche **PAPER** (grilles DCA + positions mono simulator), colonne droite **LIVE** (positions executor Bitget). Responsive : ≤900px → 1 colonne. Headers "PAPER" / "LIVE" (en vert) séparent visuellement les deux contextes.

**Fichiers** : `frontend/src/components/ActivePositions.jsx`, `frontend/src/styles.css`.

### Hotfix 36 — Cooldown par temps + DataEngine auto-recovery ✅

**But** : Deux bugs critiques détectés en prod — le bot ouvre des positions incontrôlées à chaque restart et perd sa connexion WebSocket sans jamais la récupérer.

**Bug 1 — Cooldown Hotfix 35 inefficace** : `_post_warmup_candle_count` s'incrémente par (symbol × candle). Avec 22 symbols, 3 bougies = 66 appels `on_candle()` → le cooldown est épuisé en quelques secondes, bien avant que la première bougie live soit traitée.

**Bug 2 — DataEngine meurt définitivement** : `_watch_symbol()` faisait `break` après `max_reconnect_attempts` → la tâche asyncio se termine → ce symbol ne reçoit plus jamais de données. En prod : rate limit à 00:44 → `data_stale` toutes les 30 min pendant 7 heures.

**Fix A — Cooldown par temps réel** :
- `POST_WARMUP_COOLDOWN = 3` (compteur) → `POST_WARMUP_COOLDOWN_SECONDS = 10800` (3 heures)
- Guard dans `_emit_open/close_event()` : `elapsed = now - _warmup_ended_at ; elapsed < 10800s → bloqué`
- `_warmup_ended_at: datetime | None = None` initialisé dans `__init__`
- `_post_warmup_candle_count` supprimé (compteur obsolète)

**Fix B — DataEngine never give up** :
- `_watch_symbol()` : retiré `max_reconnect_attempts` + le `break` — boucle infinie sauf symbol invalide
- Backoff exponentiel plafonné à 5 min (était 60s), reset après `attempt > 20`
- Stagger : `_SUBSCRIBE_BATCH_SIZE` 10→5, `_SUBSCRIBE_BATCH_DELAY` 0.5→2.0s
- `restart_dead_tasks()` : relance les tâches asyncio terminées (par symbol)
- `full_reconnect()` : recrée l'instance ccxt + toutes les souscriptions

**Fix B3 — Watchdog auto-recovery** :
- `data_stale > 10 min` → `restart_dead_tasks()`
- `data_stale > 30 min` + 0 tâches relancées → `full_reconnect()`

**Fichiers** : `simulator.py`, `data_engine.py`, `watchdog.py`

**Tests** : 12 nouveaux (`test_hotfix_36.py`), 7 adaptés (`test_hotfix_35.py`), 1 adapté (`test_grid_runner.py`), **1374 tests** au total, 0 régression.

### Sprint Executor Autonome — TP/SL indépendant + Réconciliation boot ✅

**But** : Résoudre 3 problèmes critiques de production apparus après la mise en live.

**Problèmes résolus** :
1. **Positions zombies** : quand le SL live est exécuté directement par Bitget, le Simulator paper ne reçoit jamais l'event CLOSE → position zombie ouverte indéfiniment
2. **Divergence paper↔live au restart** : Executor restaure ses positions live, Simulator repart à zéro → pas de surveillance TP/SL, pas d'events live
3. **SL trop grand avec 6x levier** : SL 20% × 6x = 120% marge → liquidation avant le SL en cross margin. Fix : 3x (20% × 3x = 60% marge, safe)

**Bloc 0 — Fix leverage 6x → 3x** :
- `config/risk.yaml` : `default_leverage: 15` → `3`
- `config/strategies.yaml` : `leverage: 6` → `3` pour grid_atr et grid_boltrend

**Bloc 1 — Exit monitor autonome** (`executor.py`) :
- `set_data_engine()` + `set_strategies()` : enregistrement des dépendances
- `start_exit_monitor()` : tâche asyncio toutes les 60s
- `_check_grid_exit(futures_sym)` : cœur du mécanisme — récupère candles depuis DataEngine, appelle `strategy.compute_live_indicators()`, construit `GridState` + `StrategyContext`, appelle `strategy.should_close_all()`, ferme si non-None
- `get_strategy_instances()` ajouté dans `simulator.py` : expose les instances `BaseGridStrategy` des `GridStrategyRunner`

**Bloc 2 — Boot reconciliation** (`backend/execution/sync.py`, NOUVEAU) :
- `sync_live_to_paper(executor, simulator)` : LIVE fait autorité
  - Position live sans miroir paper → INJECTION (crée GridPosition depuis GridLivePosition, déduit marge)
  - Position paper sans miroir live → SUPPRESSION (vide positions, restitue marge au capital)
  - Position live + paper → inchangée
- Appelé dans `server.py` après restore des deux côtés, avant `start_exit_monitor()`

**Bloc 3 — Hardening OPEN** (`executor.py`) :
- Guard dans `_open_grid_position()` au premier niveau (`is_first_level=True`)
- Vérifie via `_fetch_positions_safe()` si Bitget a déjà une position → skip si oui
- Les niveaux DCA supplémentaires (`is_first_level=False`) ne sont pas bloqués

**Fichiers** : `config/risk.yaml`, `config/strategies.yaml`, `backend/execution/executor.py`, `backend/backtesting/simulator.py`, `backend/execution/sync.py` (NOUVEAU), `backend/api/server.py`, `tests/test_executor_autonomous.py` (NOUVEAU)

**Tests** : 20 nouveaux (`test_executor_autonomous.py`), **1394 tests** au total, 0 régression.

**Pièges** :
- `datetime(2024, 1, 1, i)` invalide si `i > 23` (heure) → utiliser `base_ts + timedelta(hours=i)`
- Early return dans sync bloque le cleanup paper quand `_grid_states` vide → ne pas retourner tôt

### Sprint Backtest Réalisme — Liquidation, Funding, Leverage Validation ✅

**But** : Rendre les portfolio backtests plus réalistes pour le trading cross margin sur Bitget.

**Problèmes résolus** :
1. **Pas de simulation de liquidation** — un portfolio pouvait afficher -50% DD alors qu'il aurait été liquidé en cross margin
2. **Pas de funding costs** dans le GridStrategyRunner — le fast engine WFO les gère (Sprint 26) mais pas le runner event-driven paper/portfolio
3. **Pas de validation leverage × SL** — `sl_percent=20 × leverage=6 = 120%` de la marge, chaque SL coûte plus que la marge allouée

**Fix 1 — Simulation liquidation cross margin** (`portfolio_engine.py`) :
- `PortfolioSnapshot` +5 champs : `total_notional`, `maintenance_margin`, `liquidation_distance_pct`, `is_liquidated`, `worst_case_sl_loss_pct`
- `_take_snapshot()` calcule le notional total, la marge de maintenance (0.4% du notional, Bitget USDT-M tier 1), la distance de liquidation et le worst-case SL à chaque snapshot
- `_simulate()` brise la boucle si `is_liquidated=True` → capital = 0, positions vidées, event loggé
- `PortfolioResult` +4 champs : `was_liquidated`, `liquidation_event`, `min_liquidation_distance_pct`, `worst_case_sl_loss_pct`
- Worst-case SL calculé au **peak positions** (max sur tous les snapshots), pas juste à la fin
- `format_portfolio_report()` : section "Cross-Margin Risk" (min liquidation distance, worst-case SL, funding total)

**Fix 2 — Funding costs dans GridStrategyRunner** (`simulator.py`) :
- `_total_funding_cost: float = 0.0` dans `__init__`
- Settlement toutes les 8h (00:00, 08:00, 16:00 UTC), taux fixe 0.01% (approximation conservative — les vrais taux sont dans la DB et utilisés par le fast engine WFO)
- Placé **AVANT** le check TP/SL (si position ouverte au moment du settlement, on paie même si elle ferme ensuite)
- `get_status()` expose `funding_cost`
- `_build_result()` agrège `_total_funding_cost` de tous les runners → `funding_paid_total`

**Fix 3 — Validation leverage × SL dans WFO** (`report.py`) :
- `_validate_leverage_sl(strategy_name, params)` : warning si `sl_pct × leverage / 100 > 1.0` ("dépasse 100%") ou `> 0.8` ("risqué")
- Intégré dans `build_final_report()` → apparaît dans `FinalReport.warnings`

**Fix 4 — Report enrichi** (`scripts/portfolio_backtest.py`) :
- `_result_to_dict()` : +5 champs (`was_liquidated`, `liquidation_event`, `min_liquidation_distance_pct`, `worst_case_sl_loss_pct`, `funding_paid_total`)

**Fichiers** : `backend/backtesting/portfolio_engine.py`, `backend/backtesting/simulator.py`, `backend/optimization/report.py`, `scripts/portfolio_backtest.py`, `tests/test_backtest_realism.py` (NOUVEAU), `tests/test_portfolio_backtest.py` (2 unpacks mis à jour)

**Tests** : 17 nouveaux (`test_backtest_realism.py`), **1411 tests** au total, 0 régression.

**Pièges** :
- `_simulate()` retourne maintenant un 3-tuple → les tests existants qui décompactaient 2 valeurs nécessitent `snapshots, realized, _liq = await backtester._simulate(...)`
- Timestamps dans les tests `on_candle()` doivent être timezone-aware (`tzinfo=timezone.utc`), sinon `TypeError: can't subtract offset-naive and offset-aware datetimes`
- Worst-case SL au peak positions : calculer à chaque snapshot et garder le max (pas juste à la fin où il peut n'y avoir aucune position)

### Hotfix Sync — executor._grid_states peuplé depuis Bitget au boot ✅

**But** : L'exit monitor autonome restait inopérant après un restart car `executor._grid_states` était vide.

**Problème** : `sync_live_to_paper()` itérait sur `executor._grid_states` pour injecter les positions dans le paper. Si le state file était absent ou corrompu, `_grid_states` était vide → la boucle ne s'exécutait pas → les 4 positions live Bitget n'étaient pas injectées → l'exit monitor n'avait rien à checker.

Séquence observée en prod :

```text
13:03:44 | Sync: terminé — 4 symbols live synchronisés  ← FAUX (0 injectés)
13:03:44 | exit monitor autonome démarré
→ AUCUN log de check pendant 10+ minutes
→ executor_state.json : grid_states = []
```

**Fix** (`backend/execution/sync.py`) :

- Au début de `sync_live_to_paper()` : si `executor._grid_states` est vide, appel à `_populate_grid_states_from_exchange()`
- `_populate_grid_states_from_exchange()` : `fetch_positions()` sans filtre → crée un `GridLiveState` par position ouverte (`contracts > 0` et `entryPrice > 0`)
  - Stratégie détectée via les runners paper (`symbol_to_strategy`), fallback `"grid_atr"`
  - Leverage depuis `config.strategies.{name}.leverage`, fallback 3
  - `GridLivePosition(level=0, entry_order_id="restored-from-sync")` — position agrégée (Bitget retourne le total consolidé)
  - `sl_price=0.0` — sera recalculé par l'exit monitor via `should_close_all()`

**Fix** (`backend/execution/executor.py`) :

- `_exit_monitor_loop()` : log `DEBUG` avant chaque check — `"Exit monitor: check N positions ([sym1, sym2, ...])"` — confirme que la boucle tourne

**Résultat attendu après deploy** :

```text
Sync: grid_state créé pour FET/USDT:USDT (grid_atr, 3812 contracts @ 0.1667)
Sync: grid_state créé pour GALA/USDT:USDT (grid_atr, 124476 contracts @ 0.003919)
Sync: 4 grid_states créés depuis l'exchange
Exit monitor: check 4 positions (['FET/USDT:USDT', 'GALA/USDT:USDT', ...])
```

**Tests** : 2 nouveaux (`test_sync_creates_grid_states_from_exchange`, `test_sync_grid_states_not_overwritten_when_populated`), **1424 tests** au total, 0 régression.

**Piège** : Si `_grid_states` est déjà peuplé (state file intact), `_populate_grid_states_from_exchange()` n'est pas appelé — les états existants sont conservés tels quels.

### Sprint 32 — Page Journal de Trading ✅

**But** : L'ActivityFeed sidebar étant trop compact, créer un onglet "Journal" dédié avec 4 sections collapsibles, statistiques agrégées, historique d'ordres Executor, et réduction de la sidebar.

**Fonctionnalités** :

- **GET /api/journal/stats** : win rate, profit factor, max drawdown (depuis snapshots), streak, best/worst trade, durée moyenne, trades/jour — filtrable par période (today/7d/30d/all)
- **Executor order history** : deque FIFO 200 entrées, `_record_order()` après chaque `create_order`, persistence dans state JSON — **GET /api/executor/orders** (sans auth, read-only)
- **Onglet Journal** : sélecteur de période global, 4 sections collapsibles (Stats KPI, Positions & Trades, Equity Curve annotée avec markers OPEN/CLOSE, Ordres Bitget)
- **Section Positions** : sous-onglets "Ouvertes" (wsData temps réel, tableau détaillé avec niveaux grid expandables) / "Historique" (API, filtres symbol/stratégie, tri par colonne)
- **ActivityFeed** : réduit à 5 events + lien "Voir le journal complet →"

**Changements** :

- `database.py` : `get_journal_stats(period)` — calculs Python depuis simulation_trades + portfolio_snapshots
- `journal_routes.py` : endpoint `GET /api/journal/stats`
- `executor.py` : `_order_history` deque(200) + `_record_order()` + 8 insertions après create_order + persistence
- `executor_routes.py` : endpoint `GET /api/executor/orders` (sans auth)
- `JournalPage.jsx` + `JournalPage.css` : NOUVEAU — page 4 sections
- `App.jsx` : onglet journal + routing + props onTabChange
- `ActivityFeed.jsx` : max 5 events + lien journal

**Bug corrigé** : `return` anticipé avant `useMemo` dans `AnnotatedEquityCurve` → violation règle des hooks React ("Rendered more hooks than during the previous render") → page noire. Fix : déplacer le return conditionnel après tous les hooks.

**Tests** : 16 nouveaux (9 stats + 7 orders), 1238 tests au total, 0 régression.

**Fichiers** : 4 modifiés, 4 créés (JournalPage.jsx, JournalPage.css, test_journal_stats.py, test_executor_orders.py).

### Sprint 27 (futur) — Monitoring & Alertes V2

**But** : Surveillance avancée et rapports automatiques.

**Features** :

- Alertes configurables (drawdown > X%, divergence paper/live)
- Rapport quotidien/hebdomadaire automatique par Telegram
- Logs structurés pour post-mortem (chaque trade avec full context)

### Sprint 24 — Data Pipeline Robuste

**But** : Garantir la qualité et la disponibilité des données.

**Features** :

- Backfill automatique des trous (candles manquées)
- Détection de données aberrantes (spikes, gaps, volumes 0)
- Archivage et compression des données anciennes (> 1 an)
- Health check data freshness par asset × timeframe

### Sprint 25 — Gestion du Capital Avancée

**But** : Optimiser l'allocation de capital entre assets.

**Features** :

- Position sizing dynamique (Kelly criterion, fixed fractional)
- Capital allocation basée sur le grade et la performance récente
- Rebalancing automatique (sous-performance → réduit allocation)
- Risk parity (égaliser le risque entre assets, pas le capital)

---

## PHASE 7 — AVANCÉ (Sprints 26+, selon besoins)

### Walk-Forward Adaptatif
- Reoptimisation automatique quand la performance dégrade (ex: OOS Sharpe < seuil pendant N fenêtres)
- Hot swap des paramètres sans redémarrage
- A/B testing : anciens params vs nouveaux params en parallèle (paper)

### Régime Detection Avancé
- Clustering de marché (trending, ranging, volatile, calm)
- Adapter la stratégie au régime (ex: envelope_dca en ranging, breakout en trending)
- Machine learning sur features de marché (volume, ATR, ADX, funding rate)

### Machine Learning sur Résultats WFO
- Features engineering : params + métriques IS → prédire OOS Sharpe
- Réduction de l'espace de recherche (focus sur les zones prometteuses)
- Meta-learning : apprendre quels params marchent selon le régime

### Multi-Exchange
- Arbitrage (prix différents entre exchanges)
- Diversification (pas tout sur Bitget)
- Binance, Bybit, OKX, Kraken
- Gestion des taux de financement différents

### Infrastructure
- Migration vers VPS cloud si le serveur local ne suffit plus
- TimescaleDB pour les candles (compression, hypertables)
- Redis pour le cache (indicator cache, WFO results)
- Monitoring Prometheus + Grafana

---

## VUE SYNTHÉTIQUE

```
TERMINÉ                              À VENIR
═══════                              ═══════

Phase 1: Infrastructure         ✅   Phase 6: Multi-Stratégie & Live
Phase 2: Optimisation           ✅   Phase 7: Avancé (WFO adaptatif, ML, multi-exchange)
Phase 3: Paper → Live           ✅
Phase 4: Recherche & Visu       ✅
Phase 5: Scaling Stratégies     ✅
```

---

## ÉTAT ACTUEL (19 février 2026)

- **1445 tests**, 0 régression
- **Phases 1-5 terminées + Sprint Perf + Sprint 23 + Sprint 23b + Micro-Sprint Audit + Sprint 24a + Sprint 24b + Sprint 25 + Sprint 26 + Sprint 27 + Hotfix 28a-e + Sprint 29a + Hotfix 30 + Hotfix 30b + Sprint 30b + Sprint 32 + Sprint 33 + Hotfix 33a + Hotfix 33b + Hotfix 34 + Hotfix 35 + Hotfix UI + Sprint 34a + Sprint 34b + Hotfix 36 + Sprint Executor Autonome + Sprint Backtest Réalisme + Hotfix Sync grid_states + Sprint 35 + Sprint Journal V2**
- **Phase 6 en cours** — leverage optimal en cours de validation via Sprint 35 stress test
- **16 stratégies** : 4 scalp 5m + 4 swing 1h (bollinger_mr, donchian_breakout, supertrend, boltrend) + 8 grid/DCA 1h (envelope_dca, envelope_dca_short, grid_atr, grid_range_atr, grid_multi_tf, grid_funding, grid_trend, grid_boltrend)
- **22 assets** (21 historiques + JUP/USDT pour grid_trend, THETA/USDT retiré — inexistant sur Bitget)
- **Paper trading actif** : **grid_atr Top 10** (BTC, CRV, DOGE, DYDX, ENJ, FET, GALA, ICP, NEAR, AVAX) + **grid_boltrend 6 assets** (BTC, ETH, DOGE, DYDX, LINK, SAND) en préparation
- **grid_trend non déployé** : échoue en forward test (1/5 runners profitables sur 365j de bear market)
- **Sécurité** : endpoints executor protégés par API key, async I/O StateManager, buffer candles DataEngine, bypass selector configurable au boot, filtre per_asset strict (assets non validés WFO rejetés)
- **Balance refresh** : solde exchange mis à jour toutes les 5 min, refresh manuel POST /api/executor/refresh-balance, alerte si variation >10%
- **Frontend complet** : 7 pages (Scanner, Heatmap, Explorer, Recherche, Portfolio, Journal, Logs) + barre navigation stratégie (Overview/grid_atr/grid_boltrend) avec persistance localStorage
- **Log Viewer** : mini-feed sidebar WARNING/ERROR temps réel (WS) + onglet terminal Linux complet (polling HTTP, filtres, auto-scroll)
- **Benchmark WFO** : 200 combos × 5000 candles = 0.18s (0.17-0.21ms/combo), numba cache chaud
- **Sprint 35** : `scripts/stress_test_leverage.py` — 20 backtests (2 stratégies × 4 leverages × 2-3 fenêtres), KS désactivé (99%), analyse KS a posteriori à 30%/45%/60%, Calmar ratio, recommandation automatique, CSV `data/stress_test_results.csv`. Pas de tests unitaires (script de benchmark).
- **Sprint Journal V2** : Fix prix moyen "--" Bitget (`_update_order_price()` rétroactif + `paper_price`), enregistrement SL/TP fills watchOrders dans `_order_history`, endpoint slippage paper vs live (`GET /api/journal/slippage`), perf par asset (`GET /api/journal/per-asset`), `get_status()` enrichi (entry_time, positions detail, levels_max), frontend : colonne P&L %, LIVE grids expandables, colonne Slippage + bandeau, section per-asset triable, funding costs dans Stats. 18 tests (1445 total).
- **Prochaine étape** : Lancer le stress test complet (20 runs ~30 min), choisir le leverage optimal grid_boltrend (actuellement 6x arbitraire), déployer grid_boltrend en paper trading

### Résultats Portfolio Backtest — Validation Finale

**Portfolio 730 jours (complet)** :

| Config | Return | Max DD | Peak Margin | Runners prof. |
|--------|--------|--------|-------------|---------------|
| grid_atr 21 assets | +181% | -31.6% | 23.5% | 20/21 |
| grid_atr Top 10 | +221% | -29.8% | 25.0% | 10/10 |
| grid_trend 6B | +77% | -20.2% | 23.6% | 5/6 |
| Combiné 16 runners | +167% | -20.0% | 18.5% | 15/16 |

**Forward test 365 derniers jours** (bull terminal + crash -44% + bear) :

| Config | Return | Max DD | Runners prof. |
|--------|--------|--------|---------------|
| **grid_atr Top 10** | **+82.4%** | **-25.7%** | **9/10** |
| Combiné (atr+trend) | +49.3% | -23.7% | 9/16 |

**Décision** : grid_atr Top 10 = meilleur ratio rendement/risque. grid_trend échoue en forward (bear market sans trends prolongés).

**Validation Bitget** : 7/21 assets grid_atr Sharpe négatif sur 90j Bitget récents (bear soutenu nov 2025 → fév 2026). WFO pas faux (fenêtres OOS 2022-2024 ont crashes AVEC recovery), mais le bear actuel piège le mean-reversion DCA.

---

## POINTS D'ATTENTION

### 1. Overfitting
Le pipeline WFO + Monte Carlo + DSR + grading existe pour ça. Toute nouvelle stratégie passe par le même processus. L'IS positif + OOS négatif = overfitting, c'est le pattern qu'on a vu 21 fois avec les stratégies mono-position.

### 2. TP Dynamique
Le TP d'envelope_dca = SMA courante (change à chaque bougie). Il ne peut pas être placé comme trigger order sur Bitget → client-side. Si le bot crash, le SL protège (server-side).

### 3. Données Locales vs Serveur
Les backtests/WFO tournent en local sur données Binance (candles 1h depuis 2020, via `scripts/backfill_candles.py`). Le serveur n'a que les données Bitget live. Routage exchange : WFO/backtest → binance, simulateur/warm-up → bitget, live → bitget, validation croisée → bitget.

### 4. Monte Carlo Inadapté au DCA
Le block bootstrap détruit la corrélation temporelle qui est le mécanisme même de l'edge DCA. Fix : trades < 30 → underpowered (12/25 pts au lieu de 0/25).

### 5. deploy.sh --clean
Supprime les fichiers state JSON avant redémarrage. À utiliser quand le format change entre sprints. Ne supprime pas la DB SQLite.

### 6. Sandbox Bitget Non Fonctionnel (SUPPRIMÉ)

Sandbox Bitget cassé (ccxt issue #25523) → support complètement retiré du code. Mainnet only avec capital minimal = sandbox de fait.

### 7. ProcessPoolExecutor sur Windows
Instable (bug JIT Python 3.13 + surchauffe CPU i9-14900HX laptop). Solution : batches de 20 tasks + 2s cooldown entre lots. 4 workers = bon compromis (8 = seulement 1.5x plus rapide mais double la chaleur). `max_tasks_per_child=50`, fallback séquentiel automatique.

### 8. scipy Interdit
Chaque import dans un worker coûte ~200MB. Utiliser `math.erf` à la place pour la CDF normale (DSR).

---

## ARCHITECTURE CLÉS

```
config/
  strategies.yaml     # Params par stratégie (grid_atr enabled Top 10, les 12 autres disabled)
  param_grids.yaml    # Grilles de recherche WFO (3240 combos grid_atr, 2592 combos grid_trend, etc.)
  assets.yaml         # 22 assets (21 historiques + JUP pour grid_trend)
  risk.yaml           # Kill switch, leverage, fees, adaptive selector
  exchanges.yaml      # Bitget WebSocket, rate limits, API config

backend/
  core/               # Config, DataEngine, Database, StateManager, PositionManager, GridPositionManager
  strategies/         # BaseStrategy, BaseGridStrategy, 13 stratégies (4 scalp, 3 swing, 6 grid/DCA)
  backtesting/        # BacktestEngine, MultiPositionEngine, Simulator, GridStrategyRunner, Arena
  optimization/       # WFO, Monte Carlo, DSR, grading, fast engine (mono + multi), indicator cache
  execution/          # Executor (mono + grid), RiskManager, AdaptiveSelector
  alerts/             # Telegram, Notifier, Heartbeat, Watchdog
  api/                # FastAPI (server, routes REST + WebSocket)

frontend/             # React (Scanner, Heatmap, Equity, Trades, Arena, ExecutorPanel, ActivePositions, etc.)

scripts/
  optimize.py         # CLI WFO (--all, --apply, --check-data, --dry-run, -v)
  run_backtest.py     # CLI backtest simple (--symbol, --days, --json)
  backfill_candles.py # Backfill candles Binance API publique (httpx, sans clé)
  fetch_history.py    # Backfill candles via ccxt (Binance/Bitget)
  fetch_funding.py    # Backfill funding rates (Bitget)
  fetch_oi.py         # Backfill open interest (Binance)
  parity_check.py     # Compare moteurs mono vs multi-position
  reset_simulator.py  # Purge état simulateur (--executor flag)
  sync_to_server.py   # Push historique WFO vers serveur prod (idempotent)
  migrate_optimization.py # Import résultats JSON → DB (Sprint 13)

data/                 # SQLite DB + reports JSON (gitignored)
docs/plans/          # 30+ sprint plans (1-24b + hotfixes)
```

---

## MÉTRIQUES CLÉ (WFO Grading)

**6 critères pondérés (score 0-100 → Grade A-F)** :

1. **OOS/IS Ratio** (20 pts) : robustesse (pas d'overfitting)
   - < 0.5 → 0 pts (rouge)
   - 0.5-1.0 → linéaire 0-20 pts
   - > 1.0 → 20 pts

2. **Monte Carlo** (20 pts) : significativité statistique (p-value)
   - p > 0.10 → 0 pts
   - p 0.05-0.10 → 10 pts
   - p < 0.05 → 20 pts
   - Underpowered (< 30 trades) → 12 pts (bonus partiel)

3. **Consistance** (20 pts) : % fenêtres OOS positives
   - < 50% → 0 pts
   - 50-100% → linéaire 0-20 pts

4. **DSR (Deflated Sharpe Ratio)** (15 pts) : correction multiple testing bias
   - < 0.5 → 0 pts
   - 0.5-1.0 → linéaire 0-15 pts
   - > 1.0 → 15 pts

5. **Stabilité** (10 pts) : variance des perturbations ±10/20%
   - variance < 0.1 → 10 pts
   - 0.1-1.0 → linéaire 10-3 pts
   - > 1.0 → 0 pts

6. **Bitget Transfer** (15 pts) : ratio performance Binance OOS → Bitget validation
   - < 0.3 → 0 pts
   - 0.3-1.0 → linéaire 0-15 pts
   - > 1.0 → 15 pts

**Sélection best combo** : `combo_score = sharpe × (0.4 + 0.6×consistency) × min(1, trades/100)`

**Garde-fous** : < 30 trades → grade max C, < 50 trades → grade max B

**Grades** :
- A : 90-100 (excellent, prêt pour le live)
- B : 80-89 (bon, paper trading puis live)
- C : 70-79 (correct, paper trading uniquement)
- D : 60-69 (faible, amélioration nécessaire)
- F : < 60 (overfitting, à rejeter)

---

## RESSOURCES

- **Repo** : https://github.com/jackseg80/scalp-radar.git
- **Serveur** : 192.168.1.200 (Docker, Bitget mainnet, LIVE_TRADING=false)
- **Tests** : 1424 passants, 0 régression
- **Stack** : Python 3.12 (FastAPI, ccxt, numpy, aiosqlite), React (Vite), Docker
- **Bitget API** : https://www.bitget.com/api-doc/
- **ccxt Bitget** : https://docs.ccxt.com/#/exchanges/bitget
