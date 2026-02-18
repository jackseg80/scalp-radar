# CLAUDE.md — Scalp Radar Project Brief

## Project Overview

Scalp Radar is a multi-strategy automated scalping tool for crypto futures.
It detects trading opportunities, scores them, runs strategies in parallel (simulation then live),
and presents results via a real-time dashboard.

## Repository

- GitHub: <https://github.com/jackseg80/scalp-radar.git>
- Local dev: D:\Python\scalp-radar (Windows + VSCode)
- Production: Linux server at 192.168.1.200 (~/scalp-radar), deployed via Docker Compose

## Documentation

- **[ROADMAP.md](docs/ROADMAP.md)** — Roadmap complète Phases 1-7, sprints détaillés, état actuel
  - **IMPORTANT pour Claude Code** : Mettre à jour ce fichier après chaque plan de sprint complété
  - Ajouter les résultats, leçons apprises, bugs corrigés dans la section du sprint concerné
  - Mettre à jour "ÉTAT ACTUEL" avec le nouveau nombre de tests et la prochaine étape
  - **[STRATEGIES.md](docs/STRATEGIES.md)** — Guide complet des 16 stratégies de trading (logique, paramètres, exemples)
  - **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Architecture runtime, flux de données, boot/shutdown, persistence
  - **[COMMANDS.md](COMMANDS.md)** — Toutes les commandes CLI, requêtes DB, déploiement, méthodologie
  - **IMPORTANT pour Claude Code** : Consulter ce fichier avant de proposer des commandes

## Developer Profile

- Experienced crypto trader (swing trading, futures x10-x30 on Bitget)
- Building this tool to learn and automate scalping specifically
- Language: French — comments and UI can be in French
- **Platform: Windows** — tout le code doit fonctionner sur Windows (paths, line endings CRLF pour .bat)
- **Git: ne jamais ajouter de Co-Authored-By dans les commits**

## Architecture Decisions

| Decision           | Choice            | Reason                                                  |
|--------------------|-------------------|---------------------------------------------------------|
| Backend language   | Python 3.12+      | Fast enough for 1-5min scalping, best trading ecosystem |
| Package manager    | uv                | Fast pip replacement, uses standard .venv               |
| API framework      | FastAPI           | Async native, WebSocket support, Pydantic               |
| Database           | SQLite → TimescaleDB | SQLite for dev/early prod, TimescaleDB when volume grows |
| Exchange           | Bitget (primary)  | Already has bots there, good API, good futures fees     |
| Exchange lib       | ccxt              | Unified API, easy to add Binance/Kraken later           |
| Frontend           | React + Vite      | Real-time dashboard via WebSocket                       |
| Dev environment    | Windows/VSCode    | No Docker in dev — just uvicorn + vite dev              |
| Production         | Docker Compose    | On Linux server 192.168.1.200, bot runs 24/7            |
| Config format      | YAML              | Editable without code changes or redeployment           |
| Testing            | pytest            | Critical components must have unit tests (1359 passants) |

## Key Architecture Principles

- **pyproject.toml à la racine** (pas dans backend/) → imports propres `from backend.core.models`
- **Process unique** : DataEngine intégré dans le lifespan FastAPI (pas de process séparé)
- **100% async** : database, data engine, scripts CLI utilisent `asyncio.run()`
- **Buffer rolling borné** : max 500 bougies par (symbol, timeframe) en mémoire
- **Rate limiter par catégorie** : market_data, trade, account, position (token bucket)
- **SL réaliste** : coût SL inclut distance + taker_fee + slippage (configurable)
- **Groupes de corrélation** : limite l'exposition sur assets corrélés (max concurrent same direction)

## Project Structure

```text
scalp-radar/
├── config/                       # YAML configs (assets, strategies, risk, exchanges, param_grids)
├── backend/
│   ├── core/                     # models, config, database, indicators, position_manager, grid_position_manager, state_manager, data_engine
│   ├── strategies/               # base, base_grid, factory + 16 stratégies (vwap_rsi, momentum, funding, liquidation, bollinger_mr, donchian_breakout, supertrend, boltrend, envelope_dca, envelope_dca_short, grid_atr, grid_range_atr, grid_multi_tf, grid_funding, grid_trend, grid_boltrend)
│   ├── optimization/             # walk_forward, overfitting, report, indicator_cache, fast_backtest, fast_multi_backtest
│   ├── backtesting/              # engine, multi_engine, metrics, simulator, arena, extra_data_builder, portfolio_engine, portfolio_db
│   ├── execution/                # executor, risk_manager, adaptive_selector
│   ├── api/                      # server, health, simulator_routes, conditions_routes, arena_routes, executor_routes, websocket_routes, portfolio_routes
│   ├── alerts/                   # telegram, notifier, heartbeat
│   └── monitoring/               # watchdog
├── frontend/                     # React + Vite (32 components: Scanner, Heatmap, Explorer, Research, Portfolio, Diagnostic, etc.)
├── tests/                        # 1359 tests (pytest)
├── scripts/                      # backfill_candles, fetch_history, fetch_funding, fetch_oi, run_backtest, optimize, parity_check, reset_simulator, sync_to_server, portfolio_backtest
└── docs/plans/                   # Sprint plans 1-33 archivés
```

## Trading Strategies (16 implémentées)

### 5m Scalp (4)

1. **VWAP+RSI** — Mean reversion RANGING, filtre 15m anti-trend
2. **Momentum** — Breakout volume, filtre tendance
3. **Funding** — Arbitrage taux extrêmes (paper only)
4. **Liquidation** — Cascade OI zones (paper only)

### 1h Swing (4)

1. **Bollinger MR** — TP dynamique SMA crossing
2. **Donchian Breakout** — TP/SL ATR multiples
3. **SuperTrend** — Trend-following ATR-based
4. **BolTrend** — Bollinger Breakout + filtre SMA, mono-position, TP inverse (close < SMA = exit LONG) (`enabled: true`)

### 1h Grid/DCA (8)

1. **Envelope DCA** — Multi-niveaux asymétriques LONG, TP=SMA, SL=% prix moyen (`enabled: false`, remplacé par grid_atr)
2. **Envelope DCA SHORT** — Miroir SHORT d'envelope_dca, enveloppes hautes (`enabled: false`, validation WFO en attente)
3. **Grid ATR** — Enveloppes adaptatives basées sur ATR (volatilité), `entry = SMA ± ATR × multiplier` (`enabled: true`, paper trading actif sur Top 10 assets)
4. **Grid Range ATR** — Range trading bidirectionnel, LONG+SHORT simultanés, TP/SL individuels par position, `entry = SMA ± ATR × spacing`, TP = retour SMA (`enabled: false`, WFO à lancer)
5. **Grid Multi-TF** — Supertrend 4h filtre directionnel + Grid ATR 1h exécution, LONG quand ST=UP / SHORT quand ST=DOWN, force-close au flip (`enabled: false`, WFO en cours)
6. **Grid Funding** — DCA sur funding rate négatif (LONG-only), multi-niveaux par seuil, TP = funding positif / SMA cross, funding payments accumulés 8h (`enabled: false`, WFO à lancer)
7. **Grid Trend** — Trend following DCA, EMA cross + ADX + trailing stop ATR, force-close au flip, zone neutre ADX < seuil (`enabled: false`, échoue en forward test bear market)
8. **Grid BolTrend** — DCA event-driven sur breakout Bollinger + filtre SMA long, TP inversé (close < SMA = exit LONG), niveaux fixés au breakout (`enabled: true`, paper trading en préparation 6 assets : BTC, ETH, DOGE, DYDX, LINK, SAND)

## Multi-Strategy Arena

All strategies run in parallel on the same data feed with identical virtual capital.
Simulator tracks: win rate, profit factor, max drawdown, Sharpe ratio, net P&L after fees.
Adaptive selector allocates more capital to top performers, pauses underperformers.

## Critical Design Requirements

### Database & State Management
- SQLite async (WAL mode) for dev, TimescaleDB for heavy prod
- StateManager : crash recovery, sauvegarde périodique (60s), écriture atomique
- Au boot : restore session P&L, open positions, runners state

### Risk Management (non-negotiable)
- Kill switch : auto-stop if X% capital lost (persisted across restarts)
- Position sizing based on SL distance + max risk per trade
- Liquidation distance always calculated
- Max concurrent positions + correlation groups limits
- **Règle #1** : JAMAIS de position sans SL (retry 2x → emergency close)

### Fee-Aware P&L
- All P&L calculations net of fees (Bitget: taker 0.06%, maker 0.02%)
- At x20 leverage targeting 0.3% moves, fees are ~40% of gross profit

### Multi-Timeframe
- Data engine aggregates 1m, 5m, 15m, 1h simultaneously
- Strategy base class receives dict of timeframes: `{"1m": [...], "5m": [...], "15m": [...], "1h": [...]}`

### Monitoring & Alerts
- `/health` endpoint: WS connected, data freshness, strategies running, open positions
- Telegram heartbeat every hour + anomaly alerts (WS disconnect, kill switch, etc.)
- Watchdog : checks toutes les 30s, anti-spam cooldown 5 min

### Security
- `.env` never committed, IP whitelist to 192.168.1.200
- API key permissions: futures read + trade ONLY, no withdrawal

### Sync local → serveur (WFO + Portfolio)

- Après chaque run d'optimisation local, push automatique du résultat vers le serveur prod via POST API
- Best-effort : serveur down ne casse jamais un run local
- Endpoint `POST /api/optimization/results` avec auth `X-API-Key`
- Transaction sûre : INSERT OR IGNORE + UPDATE is_latest seulement si inséré (pas de perte du flag)
- **Portfolio backtests** : même mécanisme, endpoint `POST /api/portfolio/results`, push auto après save CLI/API, dédup par `created_at`
- Script `sync_to_server.py` pour pousser l'historique existant (idempotent, `--only wfo|portfolio`)
- Colonne `source` ('local' ou 'server') pour tracer l'origine des résultats
- Config : `SYNC_ENABLED`, `SYNC_SERVER_URL`, `SYNC_API_KEY` dans `.env`

## Config Files (5 YAML)

- `assets.yaml` — 22 assets (21 historiques + JUP/USDT pour grid_trend), timeframes [1m/5m/15m/1h ou 1h], groupes corrélation
- `strategies.yaml` — 16 stratégies + custom + per_asset overrides
- `risk.yaml` — kill switch, position sizing, fees, slippage, margin cross, max_margin_ratio
- `exchanges.yaml` — Bitget WebSocket, rate limits par catégorie
- `param_grids.yaml` — Espaces de recherche WFO + per-strategy config (is_days, oos_days, step_days)

## État Actuel du Projet

**Sprints complétés (1-15d + hotfixes + Sprint 16+17 + Sprint 19 + Sprint 20a-b-UI + Hotfix 20d-f + Sprint 21a + Sprint 22 + Perf + Sprint 23 + Audit + Sprint 23b + Sprint 24a + Sprint 24b + Sprint 25 + Sprint 26 + Sprint 27 + Hotfix 28a-e + Sprint 29a + Hotfix 30 + Hotfix 30b + Sprint 30b + Sprint 33 + Hotfix 33a + Hotfix 33b + Hotfix 34 + Hotfix 35 + Sprint 34a) : 1359 tests passants**

### Sprint 1-4 : Foundations & Production
- Sprint 1 : Infrastructure de base (configs, models, database, DataEngine, API, 40 tests)
- Sprint 2 : Backtesting engine + VWAP+RSI (95 tests)
- Sprint 3 : Simulator (paper trading) + 4 stratégies + Arena + Frontend MVP (166 tests)
- Sprint 4 : StateManager, Telegram, Watchdog, Docker Compose, deploy.sh (200 tests)

### Sprint 5 : Executor (Live Trading)
- Sprint 5a : Executor minimal (1 stratégie, 1 paire) + LiveRiskManager (248 tests)
- Sprint 5b : Multi-position (3 paires × 4 stratégies) + AdaptiveSelector (284 tests)

### Sprint 6 : Dashboard V2
- Sprint 6 Phase 1 : Refonte complète frontend (Scanner, Heatmap, RiskCalc, 20 components)
- Sprint 6 Phase 2 : Polish (sparkline, gradient heatmap, tri par score)
- Sprint 6b : UX overhaul (ActivePositions, sidebar redimensionnable, collapsible cards)

### Sprint 7 : Optimization
- Sprint 7 : WFO grid search (2 passes, ProcessPool), overfitting detection (Monte Carlo, DSR, stabilité), grading A-F (330 tests)
- Sprint 7b : Funding/OI historiques (fetch scripts, extra_data_builder) (352 tests)

### Sprint 9-15 : Advanced Strategies & Research
- Sprint 9 : 3 stratégies 1h (Bollinger, Donchian, SuperTrend) + fast engine numpy (419 tests)
- Sprint 10 : Infrastructure grid/DCA (BaseGridStrategy, GridPositionManager, MultiPositionEngine) (451 tests)
- Sprint 11 : GridStrategyRunner paper trading (envelope_dca live simulation) (484 tests)
- Sprint 12 : Executor grid/DCA support (multi-niveaux Bitget, 8 bugs corrigés) (513 tests)
- Sprint 13 : DB optimization_results + page Recherche (visualisation WFO, migration 49 JSON) (533 tests)
- Sprint 14 : Explorateur de paramètres (JobManager, WFO background, heatmap 2D, 6 endpoints API) (597 tests)
- Sprint 14b : Heatmap dense (wfo_combo_results), charts analytiques, tooltips (603 tests)
- Sprint 15 : Stratégie Envelope DCA SHORT (miroir LONG, fast engine direction=-1) (613 tests)
- Hotfix : Monte Carlo underpowered detection fix (envelope_dca Grade D→B)
- Hotfix : P&L overflow GridStrategyRunner — margin accounting + realized_pnl tracking (628 tests)
- Hotfix : Orphan cleanup + collision warning — positions orphelines au boot, détection collision paper (632 tests)
- Sprint 15b : Analyse par régime de marché (Bull/Bear/Range/Crash) + fix exchange WFO depuis config principale (650 tests)
- Hotfix : Warm-up compound overflow — capital fixe 10k pendant warm-up, candles plafonnées à 200, reset auto (662 tests)
- Nettoyage tests : registry centralisé (test_strategy_registry.py), fixture make_indicator_cache, fix schéma DB, suppression doublons (679 tests)
- Hotfix : Kill switch grid/DCA désactivé (pertes temporaires normales) + Kill switch global Simulator (drawdown 30%/24h, grace period warm-up, alerte Telegram, persisté dans state) (689 tests)
- Fix grading : best combo par score composite (consistance + volume), seuils OOS/IS rehaussés, extraction diagnosticUtils.js, ExportButton, TOP 5 CLI (695 tests)
- Fix grading : métriques WFO reflètent le best combo (pas les médianes fenêtre), MC + DSR reçoivent OOS Sharpe (pas IS), debug breakdown compute_grade, seuil trades 50→100 (698 tests)
- Sprint 15c : MC observed_sharpe IS→OOS fix, combo_score seuil 100 trades, garde-fou <30 trades → max C, grille étendue 0.05-0.15, DB purgée (698 tests)
- Sprint 15d : Consistance dans le grade (20 pts/100), Top 5 trié par combo_score, fetch 18 nouvelles paires Binance, WFO 23 assets (21 Grade A/B), `--apply` auto per_asset, auto-add assets.yaml via ccxt, bouton "Appliquer A/B" frontend (714 tests)
- Hotfix sizing : capital configurable (`risk.yaml initial_capital`), position sizing proportionnel (`capital / nb_assets / levels`), equal risk per trade (`margin = risk_budget / sl_pct`, cap 25%) (714 tests)
- Sprint 16+17 : Dashboard Scanner (colonnes Grade + Grid), ActivePositions GridSummary, endpoint `GET /api/simulator/grid-state`, WS push grid_state 3s, DataEngine batching anti-rate-limit (30006), fix warm-up compound post-restore (727 tests)
- Sprint 19 : Stratégie Grid ATR (10e stratégie, enveloppes ATR adaptatives, fast engine, 3240 combos WFO, 37 tests) (772 tests)
- Sprint 19b : wfo_combo_results grid_atr (collecte dense 3240 combos, heatmap + scatter + distribution)
- Sprint 19c : Régimes de marché grid_atr + fix warning trades insuffisants
- Sprint 19d : Grading Bitget transfer 3 paliers + guard bitget_trades < 15 + fix or-on-float (772 tests)
- Sprint 20a : Sizing equal allocation (`capital/nb_assets/levels`) + margin guard 70% (`max_margin_ratio`), Scanner grade fix (774 tests)
- Hotfix 19e : Scanner Grid Fix — colonnes dynamiques (Score/Signaux masquées si aucune stratégie mono), colonne Dist.SMA, direction grid, tri par grade, GridDetail (niveaux vert/rouge, TP/SL, P&L) — frontend pur, zéro backend (774 tests)
- Sprint 20b : Portfolio Backtest Multi-Asset — N runners capital partagé, 21 assets × 90j, snapshots equity/margin/positions, drawdown peak-to-trough, kill switch fenêtre glissante, force-close séparé, rapport CLI + JSON (806 tests)
- Hotfix 20d : Anti-spam Telegram — cooldown par type d'anomalie dans Notifier (SL=5min, WS/DATA=30min, KS/ALL_STOPPED=1h), log WARNING toujours, envoi Telegram throttlé (825 tests)
- Sprint 20b-UI : Frontend Portfolio Backtest — table portfolio_backtests DB, portfolio_db.py CRUD sync+async, 7 endpoints API REST (presets, CRUD, run async, status, compare), PortfolioPage React (config panel, equity curve SVG, drawdown chart, comparateur multi-runs), progress_callback engine, CLI --save/--label (825 tests)
- Hotfix 20e : Kill switch grid-compatible + warm-up fixes — grace period 10 bougies, seuils grid 25%/25%, guard anti-phantom trades post-warmup (847 tests)
- Hotfix 20f : Panneau Simulator P&L réalisé + non réalisé + equity — `get_status()` enrichi (unrealized_pnl, margin_used, equity), equity curve avec point "now", SessionStats refonte complète, EquityCurve affiche equity (852 tests)
- Sprint 21a : Stratégie Grid Multi-TF (11e stratégie, Supertrend 4h + Grid ATR 1h, resampling anti-lookahead, fast engine directions dynamiques, 384 combos WFO, 40 tests) (898 tests)
- Bugfix 21a-bis : Validation Bitget + Monte Carlo 0 trades — compute_indicators() retourne 4h Supertrend, MultiPositionEngine passe tous les TFs (898 tests)
- Bugfix 21a-ter : Portfolio backtest 0 trades — compute_live_indicators() dans BaseGridStrategy pour mode live/portfolio, GridStrategyRunner merge les indicateurs 4h depuis le buffer IncrementalIndicatorEngine (902 tests)
- Hotfix 21a-quater : Serveur bloqué au warm-up + résilience assets manquants — skip compute_live_indicators() pendant warm-up, détection symbol invalide immédiate, sleep 1s sur erreur non-rate-limit, THETA/USDT retiré (21 assets restants) (902 tests)
- Sprint 22 : Stratégie Grid Funding (12e stratégie, DCA sur funding rate négatif, LONG-only, fast engine avec funding payments 8h, 2592 combos WFO, 42 tests) (944 tests)
- Bugfix 22-bis : Validation Bitget + stabilité 0 trades — extra_data_by_timestamp manquant dans run_multi_backtest_single pour report.py et overfitting.py (944 tests)
- Sprint Perf : Numba JIT Optimization — speedup 5-10x WFO (indicateurs Wilder + boucle trades), fallback transparent si absent (941 tests excl. JIT)
- Sprint 23 : Grid Trend (13e stratégie, trend following DCA, EMA cross + ADX + trailing stop ATR, 2592 combos WFO, 46 tests) (990 tests)
- Micro-Sprint Audit : Auth endpoints executor (API key), async I/O StateManager (asyncio.to_thread), buffer candles DataEngine (flush 5s) (1004 tests)
- Sprint 23b : grid_trend compute_live_indicators (paper/portfolio fix, calcul EMA+ADX depuis buffer candles) (1007 tests)
- Sprint 24a : Portfolio Backtest Realistic Mode — sizing fixe (_portfolio_mode), global margin guard (tous runners), kill switch temps réel (freeze/dégel 24h) (1012 tests)
- Sprint 24b : Portfolio Backtest Multi-Stratégie — clé runner strategy:symbol, dispatch multi-runners par symbol, CLI --strategies/--preset combined (1016 tests)
- Sprint 25 : Activity Journal — 2 tables DB (portfolio_snapshots, position_events), snapshots equity/margin/unrealized 5min, hooks OPEN/CLOSE DCA, 3 endpoints API, frontend EquityCurve double source + ActivityFeed événements journal (1037 tests)
- Hotfix 25a : Retry DB writes journal — `_execute_with_retry()` 3 tentatives avec backoff 100ms/200ms sur "database is locked", throttle 50ms entre INSERT events si batch > 2 (1037 tests)
- Sprint 26 : Funding Costs Backtest — funding rate 8h settlement costs dans toutes les stratégies grid (event-driven + fast engine), fix convention /100, 25 tests (1074 tests)
- Sprint 27 : Filtre Darwinien par Régime — bloque nouvelles grilles si WFO avg_oos_sharpe < 0 dans le régime actuel, mapping `REGIME_LIVE_TO_WFO`, compteur `_regime_filter_blocks` dans `get_status()`, DB `get_regime_profiles()`, configurable via `regime_filter_enabled` (1090 tests)
- Hotfix 28a : Préparation déploiement live — Selector charge trades DB au boot (survit à --clean), bypass configurable au cold start (auto-désactivé quand toutes les stratégies atteignent min_trades), warning capital mismatch Bitget vs risk.yaml (1102 tests)
- Hotfix 28b : Suppression sandbox Bitget + filtre per_asset strict GridStrategyRunner — assets non listés dans per_asset sont rejetés par on_candle() (1104 tests)
- Hotfix 28c : Refresh périodique solde exchange — fetch_balance toutes les 5 min, log WARNING si >10% change, POST /api/executor/refresh-balance, exchange_balance dans get_status() (1114 tests)
- Hotfix 28d : Override env var SELECTOR_BYPASS_AT_BOOT — `.env` (gitignored) prend priorité sur risk.yaml (versionné), même pattern que LIVE_TRADING (1116 tests)
- Hotfix 28e : Sync portfolio backtests local → serveur — réutilisation infra sync WFO (config, httpx, auth X-API-Key), endpoint POST /api/portfolio/results, push auto après save (CLI + API), sync_to_server.py étendu --only portfolio (1129 tests)
- Sprint 29a : Grid Range ATR (14e stratégie) — range trading bidirectionnel, LONG+SHORT simultanés, TP/SL individuels par position, fast engine dédié `_simulate_grid_range()`, 2160 combos WFO, script diagnostic `test_grid_range_fast.py` (1169 tests)
- Hotfix 30 : Deadlock Selector + DATA_STALE — force_strategies bypass (grid_atr forcé), session vierge skip net_return/PF si DB a des trades, `_last_update` avant check doublon, grid_range_atr ajouté au mapping (1183 tests)
- Hotfix 30b : Config conflicts deploy — deploy.sh reset config/ avant git pull, force_strategies override .env (comma-separated), règle JAMAIS éditer config/*.yaml sur serveur (1188 tests)
- Sprint 30b : Stratégie BolTrend (15e stratégie) — Bollinger Breakout + filtre SMA trend, mono-position, check_exit INVERSÉ vs bollinger_mr (close < SMA = exit LONG), fast engine Numba, 486 combos WFO, 25 tests (1213 tests)
- Sprint 33 : Stratégie Grid BolTrend (16e stratégie) — DCA event-driven sur breakout Bollinger + filtre SMA long, fast engine dédié `_simulate_grid_boltrend()`, TP inversé (`get_tp_price()` retourne NaN), niveaux fixés au breakout, 1296 combos WFO, 32 tests (1309 tests)
- Hotfix 33a : TP inverse grid_boltrend — `get_tp_price()` retourne `float("nan")` pour désactiver `check_global_tp_sl()`, TP géré par `should_close_all()` uniquement. Sharpe -14.51 → +1.58, Grade F → B (83/100) (1309 tests)
- Hotfix 33b : Audit parité grid_boltrend — 3 bugs fast engine (exit_price sma_val→close_i, fees maker→taker+slip, double entry_fee dans multi_engine.py), divergence 31.73% → 2.62%, 13 tests parité (1322 tests)
- Hotfix 34 : Executor P&L réel Bitget — `_fetch_fill_price()` (fetch_order → fetch_my_trades), `entry_fee` dans dataclasses LivePosition/GridLivePosition, `_calculate_real_pnl()` séparée, 17 tests (1339 tests)
- Hotfix 35 : Stabilité restart live — cooldown post-warmup 3 bougies (`POST_WARMUP_COOLDOWN`) dans GridStrategyRunner, guard `max_live_grids` isinstance-safe, sauvegarde périodique executor via `StateManager._periodic_save_loop()` (60s), 14 tests (1353 tests)
- Sprint 34a : Lancement paper trading grid_boltrend — warm-up dynamique via `strategy.min_candles` (MAX_WARMUP_CANDLES 200→500), try/except `compute_live_indicators()` + alerte Telegram INDICATOR_ERROR (cooldown 1h), préfixes stratégie `[ATR]`/`[BOLT]` dans Telegram, rollback d'urgence COMMANDS.md section 17, 6 tests (1359 tests)

Sprint 8 (Backtest Dashboard) planifié mais non implémenté.

### Décisions Clés Transverses

**Optimisation (Sprint 7) :**
- Deux chemins params : optimisation (grid explicite) vs production (`_resolve_param` per_asset)
- scipy retiré (remplacé par `math.erf`) — évite MemoryError ProcessPoolExecutor
- Fallback séquentiel automatique si multiprocessing crashe sur Windows
- Workers limités à 4 (`min(cpu_count, 4)`), `max_tasks_per_child=50`

**Fast Engine (Sprint 9) :**
- IndicatorCache pré-calculé (toutes variantes du grid)
- Bollinger TP dynamique (SMA crossing dans `check_exit()`, pas prix fixe)
- SuperTrend pré-calculé (boucle itérative, ~5ms/48k pts)

**Grid/DCA (Sprint 10-12, 20a) :**
- BaseGridStrategy hérite BaseStrategy (compatibilité Arena/Simulator/Dashboard)
- Equal allocation sizing : `margin = capital / nb_assets / levels`, cap 25% du capital par asset, margin guard 70% (`max_margin_ratio` dans risk.yaml)
- TP/SL global (pas par position), SMA dynamic TP
- Un seul côté actif (positions LONG → pas niveaux SHORT simultanés)
- Enveloppes asymétriques : `upper = 1/(1-lower) - 1` (aller-retour cohérent)
- Leverage 6 depuis config (pas le défaut 15)
- Exclusion mutuelle mono/grid par symbol (Bitget agrège par symbol+direction)

**Grid Margin Accounting (Hotfix) :**
- GridStrategyRunner déduit `margin = notional / leverage` à l'ouverture, restitue à la fermeture
- `_realized_pnl` séparé de `_capital` — kill switch utilise uniquement le P&L réalisé
- Entry fees inclus dans `net_pnl` à la fermeture (pas déduites à l'ouverture, sinon double-counting)
- StateManager sauvegarde `realized_pnl` avec guard `isinstance(..., (int, float))` (MagicMock-safe)
- `scripts/reset_simulator.py` pour purger un état corrompu (idempotent, `--executor` flag)

**Warm-up Compound Overflow (Hotfix) :**

- Au démarrage, `watch_ohlcv()` retourne un gros batch de candles historiques → GridStrategyRunner les traitait avec compound sizing → capital de 10k à 83M$
- Fix 1 : `_warmup_from_db()` plafonné à `MAX_WARMUP_CANDLES = 500` (Sprint 34a : 200→500 pour couvrir grid_boltrend `long_ma_window=400`)
- Fix 2 : Flag `_is_warming_up = True` dans GridStrategyRunner, capital fixé à `_initial_capital` (10k) pendant le warm-up
- Détection auto fin warm-up : candle avec age < `WARMUP_AGE_THRESHOLD` (2h) → `_end_warmup()`
- `_end_warmup()` : reset capital/realized_pnl/stats, ferme positions, garde trades en historique
- Pendant warm-up : pas de modification de `_capital`, pas d'événements Executor, trades enregistrés en historique seulement
- `restore_state()` désactive automatiquement le warm-up (restart = pas besoin de re-warmer)
- Sprint 34a : `_warmup_from_db()` utilise `strategy.min_candles.get(tf, 50)` (plus `_ma_period + 20`) — dynamique par stratégie, rétrocompatible (grid_atr=50 inchangé, grid_boltrend=420 corrigé)

**Orphan Cleanup & Collision (Hotfix) :**
- Au boot, `_cleanup_orphan_runners()` détecte les runners dans saved_state non présents dans enabled_names
- Paper : log WARNING + `OrphanClosure` dataclass avec fee estimé ; Live : log CRITICAL (pas d'action Bitget)
- Collision paper : snapshot positions avant/après chaque `on_candle()`, log WARNING si même symbol
- `_trades` (liste mémoire) vidée à chaque boot ; `_stats.total_trades` restauré (compteur cumulatif)

**Executor Grid (Sprint 12) :**
- Dispatch grid vs mono via `_is_grid_strategy()`
- Pre-trade check au 1er niveau seulement (un cycle = 1 slot max_concurrent)
- SL global server-side recalculé à chaque niveau (cancel ancien + place nouveau)
- TP client-side (SMA dynamique, détecté par GridStrategyRunner)
- State persistence round-trip (`grid_states` dans get_state/restore_positions)
- Réconciliation grid au boot (`_reconcile_grid_symbol`)

**Explorateur WFO (Sprint 14) :**
- JobManager : file FIFO asyncio.Queue (max 5 pending), worker loop avec `asyncio.to_thread()`
- Thread-safety : sqlite3 sync + `run_coroutine_threadsafe()` pour broadcast WebSocket
- Progress callback : appelé à chaque fenêtre WFO, met à jour DB + broadcast WS temps réel
- Annulation : `threading.Event` vérifié dans callback, interrompt le WFO proprement
- Normalisation `params_override` : frontend envoie scalaires `{ma_period: 7}`, backend convertit en listes `{ma_period: [7]}`
- Recovery boot : jobs "running" orphelins → status failed (serveur redémarré)
- Heatmap responsive : ResizeObserver + cellSize 60-300px + flexbox centering
- 6 endpoints API : run, jobs (liste), jobs/{id}, cancel, param-grid, heatmap

**Grading 6 critères (Sprint 15c/15d) :**
- 6 critères : OOS/IS ratio (20 pts), Monte Carlo (20 pts), Consistance (20 pts), DSR (15 pts), Stabilité (10 pts), Bitget transfer (15 pts)
- combo_score : `sharpe × (0.4 + 0.6×consistency) × min(1, trades/100)` — sélection best combo WFO
- Garde-fous : < 30 trades → grade max C, < 50 trades → grade max B
- Bitget transfer 3 paliers : `>0.50 + significant` → 15 pts, `>0.50` seul → 10 pts, `>0.30` → 5 pts ; guard `bitget_trades < 15` → cap 8 pts
- Bouton "Appliquer A/B" frontend : `POST /api/optimization/apply` → per_asset strategies.yaml + auto-add assets.yaml via ccxt
- `fetch_history.py --symbols` : bypass assets.yaml pour télécharger des paires spécifiques

**Portfolio Backtest Réaliste (Sprint 24a-b) :**

- `_portfolio_mode = True` → sizing fixe sur `_initial_capital` (anti-compounding)
- Global margin guard : marge totale tous runners < `capital × max_margin_ratio` (70%)
- Kill switch temps réel : fenêtre glissante 24h, freeze/dégel tous runners
- Multi-stratégie : clé `strategy:symbol`, dispatch candles à tous runners du même symbol
- Résultats : grid_atr Top 10 = +221% (730j), +82.4% (forward 365j), DD max -29.8%
- grid_trend non déployé (1/5 profitables en forward, bear market sans trends)
- Paper trading actif : **grid_atr Top 10** (BTC, CRV, DOGE, DYDX, ENJ, FET, GALA, ICP, NEAR, AVAX)

**Sync WFO :**

- Push automatique après chaque run local : `save_report()` → `push_to_server()` (best-effort)
- Transaction sûre : INSERT OR IGNORE d'abord, UPDATE is_latest seulement si inséré
- Endpoint `POST /api/optimization/results` avec auth X-API-Key (refuse si `sync_api_key` vide côté serveur)
- 4 nouvelles fonctions dans `optimization_db.py` : `save_result_from_payload_sync`, `build_push_payload`, `build_payload_from_db_row`, `push_to_server`
- Script `sync_to_server.py` pour pousser l'historique (idempotent, --dry-run)
- Colonne `source TEXT DEFAULT 'local'` + migration ALTER TABLE idempotente

## Lifespan Complet (server.py)

**Startup :**
DB → Telegram/Notifier → DataEngine → StateManager load → Simulator.start(saved_state) → periodic save → Arena → AdaptiveSelector → Executor(selector) → selector.start() → Watchdog → Heartbeat → notify_startup

**Shutdown :**
notify_shutdown → heartbeat.stop() → watchdog.stop() → selector.stop() → executor state save + stop → state_manager.save() → simulator.stop() → data_engine.stop() → db.close()

## Dev Workflow

```text
Windows (VSCode)                    Linux Server (192.168.1.200)
     │                                       │
  Code + test locally               Bot H24 + dashboard
  uvicorn + vite dev                 Docker Compose
     │                                       │
  git push ──────────────────► SSH + deploy.sh
                                      │
                                git pull
                                docker compose build
                                docker compose up -d
                                curl /health → check OK
```

**Flags deploy.sh :**

- `--clean` / `-c` : kill brutal + supprime state files (fresh start sans perdre la DB)

**Règle critique déploiement (Hotfix 30b) :**

- **JAMAIS éditer les fichiers `config/*.yaml` sur le serveur**
- `deploy.sh` reset automatiquement `config/` avant `git pull` (ligne 28-30)
- Tous les overrides prod passent par `.env` (gitignored) :
  - `LIVE_TRADING=true` : active l'Executor (ordres réels)
  - `SELECTOR_BYPASS_AT_BOOT=true` : autorise toutes les stratégies au boot (cold start)
  - `FORCE_STRATEGIES=grid_atr` : bypass net_return/PF checks (comma-separated)
- Exemple `.env` serveur :

```bash
LIVE_TRADING=true
SELECTOR_BYPASS_AT_BOOT=true
FORCE_STRATEGIES=grid_atr
```

## References

- Bitget API docs: <https://www.bitget.com/api-doc/>
- ccxt Bitget: <https://docs.ccxt.com/#/exchanges/bitget>
- Frontend prototype: `docs/prototypes/Scalp radar v2.jsx` (référence design)
- Plans détaillés : `docs/plans/sprint-{n}-*.md` (1-33 archivés)
