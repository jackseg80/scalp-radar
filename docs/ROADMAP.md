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
| 15     | Stratégie Envelope DCA SHORT (miroir LONG)              | ✅           |

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

---

## PHASE 5 — SCALING STRATÉGIES (Sprints 16-19) ← ON EST ICI

### Sprint 16 — WFO envelope_dca_short + Passage Live
**But** : Valider SHORT via WFO, puis passer envelope_dca LONG en live si paper trading OK.

**Phase A — WFO SHORT (prioritaire)** :
- [ ] Lancer WFO envelope_dca_short sur BTC/ETH/SOL (via Explorateur dashboard)
- [ ] Analyser les résultats (grade, OOS Sharpe, consistency, heatmap)
- [ ] Comparer LONG vs SHORT dans ResearchPage
- [ ] Décision : enabled: true si Grade >= C

**Phase B — Passage Live LONG (si paper trading validé)** :
**Prérequis** : Paper trading envelope_dca cohérent (1-2 semaines observation).

**Checklist** :
- [ ] Paper trading cohérent (trades sur candles 1h fraîches, pas juste replay)
- [ ] Pas de bugs critiques (SL placés, TP détectés, P&L cohérent)
- [ ] Capital suffisant sur Bitget (minimum ~100-200 USDT pour des trades significatifs)
- [ ] Monitoring étroit les premiers jours (alertes Telegram actives)

**Actions** :
- Ajouter du capital sur Bitget (minimum ~100-200 USDT pour des trades significatés)
- `LIVE_TRADING=true` dans .env
- `live_eligible: true` dans strategies.yaml (déjà fait pour LONG)
- Redéployer avec deploy.sh (sans --clean pour garder le state paper)
- Observer les premiers trades en live
- Vérifier : ordres passés, SL placés, TP détectés, P&L cohérent

**Scope** : ~1-2 sessions (WFO SHORT + surveillance live LONG si validé).

### Sprint 17 — Monitoring DCA Live Amélioré
**But** : Voir l'état du DCA en temps réel, pas juste les trades clôturés.

**Frontend — Onglet "DCA" ou amélioration Scanner** :
- Niveaux d'enveloppe actuels vs prix (déjà visible partiellement dans le Scanner)
- Positions grid ouvertes avec P&L non réalisé (par niveau + global)
- Historique des cycles complets (multi-niveaux → fermeture)
- Graphique prix + enveloppes + points d'entrée/sortie (SVG ou canvas)
- Temps moyen d'un cycle, fréquence par asset
- Alertes visuelles : prix proche d'un niveau, SL proche

**Backend** :
- Endpoint GET /api/simulator/grid-state?symbol= (niveaux, enveloppes, positions ouvertes)
- WebSocket push : prix + enveloppes + positions en temps réel

**Scope** : ~1 session.

**Dépendances** :
- GridStrategyRunner expose grid_levels calculés
- API endpoint grid-state
- Frontend GridMonitor.jsx ou amélioration Scanner.jsx

### Sprint 18 — Multi-asset Live
**But** : Déployer envelope_dca sur ETH, SOL (et potentiellement DOGE, LINK si grades OK après reoptimisation).

**Prérequis** : Paper trading validé sur BTC, capital suffisant pour 2-3 assets.

**Actions** :
- Reoptimiser par asset si nécessaire (params différents par crypto)
- Config per_asset dans strategies.yaml (déjà prévu)
- Capital allocation : répartition fixe ou proportionnelle au grade
- Gestion des corrélations (pas tout LONG en même temps sur BTC+ETH)
- Activer les assets un par un (observer 1-2 jours entre chaque)

**Scope** : ~1 session (config) + surveillance échelonnée.

**Questions** :
- Capital allocation : fixe (ex: 100$ par asset) ou proportionnelle au grade (ex: Grade B = 200$, Grade C = 100$) ?
- Corrélation groups : max_concurrent_same_direction = 2 sur crypto_major (BTC/ETH/SOL) ?

### Sprint 19 — Nouvelles Stratégies Grid
**But** : Développer d'autres stratégies qui utilisent le moteur multi-position.

**Candidates** :
- **Grid ATR** : enveloppes basées sur la volatilité (ATR) au lieu de % fixe
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

---

## PHASE 6 — ROBUSTESSE & PRODUCTION (Sprints 20-22)

### Sprint 20 — Gestion du Capital Avancée
**But** : Optimiser l'allocation de capital entre stratégies et assets.

**Features** :
- Position sizing dynamique (Kelly criterion, fixed fractional)
- Capital allocation par stratégie basée sur le grade et la performance récente
- Max drawdown par stratégie et global
- Rebalancing automatique (ex: stratégie sous-performe → réduit allocation)
- Risk parity (égaliser le risque entre assets, pas le capital)

**Scope** : ~2 sessions.

### Sprint 21 — Data Pipeline Robuste
**But** : Garantir la qualité et la disponibilité des données.

**Features** :
- Backfill automatique des trous (candles manquées)
- Détection de données aberrantes (spikes, gaps, volumes 0)
- Multi-exchange (Binance spot pour backtest, Bitget futures pour live)
- Archivage et compression des données anciennes (> 1 an)
- Health check data freshness par asset × timeframe

**Scope** : ~1-2 sessions.

### Sprint 22 — Monitoring & Alertes V2
**But** : Surveillance avancée et rapports automatiques.

**Features** :
- Dashboard de performance live (P&L cumulé, drawdown, Sharpe rolling)
- Alertes configurables (drawdown > X%, stratégie sous-performe, divergence paper/live)
- Rapport quotidien/hebdomadaire automatique par Telegram
- Comparaison paper vs live (slippage réel, fills partiels, latence)
- Logs structurés pour post-mortem (chaque trade avec full context)

**Scope** : ~2 sessions.

---

## PHASE 7 — AVANCÉ (Sprints 23+, selon besoins)

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
TERMINÉ                          EN COURS              À VENIR
═══════                          ════════              ═══════

Phase 1: Infrastructure     ✅   Phase 4: Recherche    Phase 5: Scaling
Phase 2: Validation         ✅   Sprint 13: DB+Dash    Sprint 16: Live
Phase 3: Paper/Live ready   ✅   Sprint 14: Explorer   Sprint 17: SHORT
                                  Sprint 15: DCA UI     Sprint 18: Multi-asset
                                                        Sprint 19: Nouvelles strats

                                                        Phase 6: Production
                                                        Phase 7: Avancé
```

---

## ÉTAT ACTUEL (14 février 2026)

- **613 tests**, 0 régression
- **15 sprints** complétés (Phase 1-4 terminées)
- **9 stratégies** : 4 scalp 5m (vwap_rsi, momentum, funding, liquidation) + 3 swing 1h (bollinger_mr, donchian_breakout, supertrend) + 2 grid/DCA 1h (envelope_dca LONG, envelope_dca_short SHORT)
- **1 stratégie validée LONG** : envelope_dca Grade B (BTC), enabled en paper trading
- **1 stratégie SHORT prête pour WFO** : envelope_dca_short (enabled: false, validation WFO en attente)
- **Paper trading actif** : envelope_dca sur 5 assets
- **Executor Grid prêt** : LIVE_TRADING=false, à activer après validation paper
- **Explorateur WFO** : lance des optimisations depuis le dashboard, heatmap 2D 100% dense (324 combos), charts analytiques
- **Prochaine étape** : Sprint 16 (WFO envelope_dca_short + passage Live si validé)

---

## POINTS D'ATTENTION

### 1. Overfitting
Le pipeline WFO + Monte Carlo + DSR + grading existe pour ça. Toute nouvelle stratégie passe par le même processus. L'IS positif + OOS négatif = overfitting, c'est le pattern qu'on a vu 21 fois avec les stratégies mono-position.

### 2. TP Dynamique
Le TP d'envelope_dca = SMA courante (change à chaque bougie). Il ne peut pas être placé comme trigger order sur Bitget → client-side. Si le bot crash, le SL protège (server-side).

### 3. Données Locales vs Serveur
Les backtests tournent en local (candles Binance 1h). Le serveur n'a que les données Bitget live. Sprint 13 adresse le stockage des résultats en DB.

### 4. Monte Carlo Inadapté au DCA
Le block bootstrap détruit la corrélation temporelle qui est le mécanisme même de l'edge DCA. Fix : trades < 30 → underpowered (12/25 pts au lieu de 0/25).

### 5. deploy.sh --clean
Supprime les fichiers state JSON avant redémarrage. À utiliser quand le format change entre sprints. Ne supprime pas la DB SQLite.

### 6. Sandbox Bitget Non Fonctionnel
Sandbox Bitget ne marche pas avec le sous-compte → mainnet avec capital minimal = sandbox de fait. `BITGET_SANDBOX=false` en prod, toujours mainnet.

### 7. ProcessPoolExecutor sur Windows
Instable (bug JIT Python 3.13 + surchauffe CPU i9-14900HX laptop). Solution : batches de 20 tasks + 2s cooldown entre lots. 4 workers = bon compromis (8 = seulement 1.5x plus rapide mais double la chaleur). `max_tasks_per_child=50`, fallback séquentiel automatique.

### 8. scipy Interdit
Chaque import dans un worker coûte ~200MB. Utiliser `math.erf` à la place pour la CDF normale (DSR).

---

## ARCHITECTURE CLÉS

```
config/
  strategies.yaml     # Params par stratégie (envelope_dca enabled, les 7 autres disabled)
  param_grids.yaml    # Grilles de recherche WFO (324 combos envelope_dca)
  assets.yaml         # 5 assets (BTC, ETH, SOL, DOGE, LINK)
  risk.yaml           # Kill switch, leverage, fees, adaptive selector
  exchanges.yaml      # Bitget WebSocket, rate limits, API config

backend/
  core/               # Config, DataEngine, Database, StateManager, PositionManager, GridPositionManager
  strategies/         # BaseStrategy, BaseGridStrategy, envelope_dca, 7 autres stratégies Grade F
  backtesting/        # BacktestEngine, MultiPositionEngine, Simulator, GridStrategyRunner, Arena
  optimization/       # WFO, Monte Carlo, DSR, grading, fast engine (mono + multi), indicator cache
  execution/          # Executor (mono + grid), RiskManager, AdaptiveSelector
  alerts/             # Telegram, Notifier, Heartbeat, Watchdog
  api/                # FastAPI (server, routes REST + WebSocket)

frontend/             # React (Scanner, Heatmap, Equity, Trades, Arena, ExecutorPanel, ActivePositions, etc.)

scripts/
  optimize.py         # CLI WFO (--all, --apply, --check-data, --dry-run, -v)
  run_backtest.py     # CLI backtest simple (--symbol, --days, --json)
  fetch_history.py    # Backfill candles (Binance/Bitget)
  fetch_funding.py    # Backfill funding rates (Bitget)
  fetch_oi.py         # Backfill open interest (Binance)
  parity_check.py     # Compare moteurs mono vs multi-position

data/                 # SQLite DB + reports JSON (gitignored)
docs/plans/          # 16 sprint plans (1-12 + hotfix)
```

---

## MÉTRIQUES CLÉ (WFO Grading)

**5 critères pondérés (score 0-100 → Grade A-F)** :

1. **OOS Sharpe Ratio** (25 pts) : performance ajustée au risque OOS
   - < 0 → 0 pts
   - 0-2 → linéaire 0-15 pts
   - 2-5 → linéaire 15-25 pts
   - > 5 → 25 pts

2. **Consistance** (25 pts) : % fenêtres OOS positives
   - < 50% → 0 pts
   - 50-100% → linéaire 0-25 pts

3. **OOS/IS Ratio** (20 pts) : robustesse (pas d'overfitting)
   - < 0.3 → 0 pts
   - 0.3-1.0 → linéaire 0-20 pts
   - > 1.0 → 20 pts

4. **DSR (Deflated Sharpe Ratio)** (15 pts) : correction multiple testing bias
   - < 0.5 → 0 pts
   - 0.5-1.0 → linéaire 0-15 pts
   - > 1.0 → 15 pts

5. **Stabilité** (15 pts) : variance des perturbations ±10/20%
   - variance < 0.1 → 15 pts
   - 0.1-1.0 → linéaire 15-5 pts
   - > 1.0 → 0 pts

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
- **Tests** : 513 passants (28 fichiers), 0 régression
- **Stack** : Python 3.12 (FastAPI, ccxt, numpy, aiosqlite), React (Vite), Docker
- **Bitget API** : https://www.bitget.com/api-doc/
- **ccxt Bitget** : https://docs.ccxt.com/#/exchanges/bitget
