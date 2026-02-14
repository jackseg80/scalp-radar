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
| Testing            | pytest            | Critical components must have unit tests (632 passants) |

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
│   ├── strategies/               # base, base_grid, factory + 9 stratégies (vwap_rsi, momentum, funding, liquidation, bollinger_mr, donchian_breakout, supertrend, envelope_dca, envelope_dca_short)
│   ├── optimization/             # walk_forward, overfitting, report, indicator_cache, fast_backtest, fast_multi_backtest
│   ├── backtesting/              # engine, multi_engine, metrics, simulator, arena, extra_data_builder
│   ├── execution/                # executor, risk_manager, adaptive_selector
│   ├── api/                      # server, health, simulator_routes, conditions_routes, arena_routes, executor_routes, websocket_routes
│   ├── alerts/                   # telegram, notifier, heartbeat
│   └── monitoring/               # watchdog
├── frontend/                     # React + Vite (20 components: Scanner, Heatmap, RiskCalc, ExecutorPanel, etc.)
├── tests/                        # 650 tests (pytest)
├── scripts/                      # fetch_history, fetch_funding, fetch_oi, run_backtest, optimize, parity_check, reset_simulator
└── docs/plans/                   # Sprint plans 1-15 archivés
```

## Trading Strategies (9 implémentées)

### 5m Scalp (4)

1. **VWAP+RSI** — Mean reversion RANGING, filtre 15m anti-trend
2. **Momentum** — Breakout volume, filtre tendance
3. **Funding** — Arbitrage taux extrêmes (paper only)
4. **Liquidation** — Cascade OI zones (paper only)

### 1h Swing (3)

1. **Bollinger MR** — TP dynamique SMA crossing
2. **Donchian Breakout** — TP/SL ATR multiples
3. **SuperTrend** — Trend-following ATR-based

### 1h Grid/DCA (2)

1. **Envelope DCA** — Multi-niveaux asymétriques LONG, TP=SMA, SL=% prix moyen (`enabled: true`, paper trading actif)
2. **Envelope DCA SHORT** — Miroir SHORT d'envelope_dca, enveloppes hautes (`enabled: false`, validation WFO en attente)

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

### Sync WFO local → serveur (NEW)

- Après chaque run d'optimisation local, push automatique du résultat vers le serveur prod via POST API
- Best-effort : serveur down ne casse jamais un run local
- Endpoint `POST /api/optimization/results` avec auth `X-API-Key`
- Transaction sûre : INSERT OR IGNORE + UPDATE is_latest seulement si inséré (pas de perte du flag)
- Script `sync_to_server.py` pour pousser l'historique existant (idempotent)
- Colonne `source` ('local' ou 'server') pour tracer l'origine des résultats
- Config : `SYNC_ENABLED`, `SYNC_SERVER_URL`, `SYNC_API_KEY` dans `.env`

## Config Files (5 YAML)

- `assets.yaml` — 5 assets (BTC, ETH, SOL, DOGE, LINK), timeframes [1m, 5m, 15m, 1h], groupes corrélation
- `strategies.yaml` — 9 stratégies + custom + per_asset overrides
- `risk.yaml` — kill switch, position sizing, fees, slippage, margin cross
- `exchanges.yaml` — Bitget WebSocket, rate limits par catégorie
- `param_grids.yaml` — Espaces de recherche WFO + per-strategy config (is_days, oos_days, step_days)

## État Actuel du Projet

**Sprints complétés (1-15b + hotfixes) : 650 tests passants**

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

**Sprint 8** (Backtest Dashboard) planifié mais non implémenté.

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

**Grid/DCA (Sprint 10-12) :**
- BaseGridStrategy hérite BaseStrategy (compatibilité Arena/Simulator/Dashboard)
- Allocation fixe par niveau : `notional = capital/levels × leverage`
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

**Sync WFO (NEW) :**

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

## References

- Bitget API docs: <https://www.bitget.com/api-doc/>
- ccxt Bitget: <https://docs.ccxt.com/#/exchanges/bitget>
- Frontend prototype: `docs/prototypes/Scalp radar v2.jsx` (référence design)
- Plans détaillés : `docs/plans/sprint-{n}-*.md` (1-15 archivés)
