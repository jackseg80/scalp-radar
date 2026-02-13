# CLAUDE.md — Scalp Radar Project Brief

## Project Overview

Scalp Radar is a multi-strategy automated scalping tool for crypto futures.
It detects trading opportunities, scores them, runs strategies in parallel (simulation then live),
and presents results via a real-time dashboard.

## Repository

- GitHub: https://github.com/jackseg80/scalp-radar.git
- Local dev: D:\Python\scalp-radar (Windows + VSCode)
- Production: Linux server at 192.168.1.200 (~/scalp-radar), deployed via Docker Compose

## Developer Profile

- Experienced crypto trader (swing trading, futures x10-x30 on Bitget)
- Already has trading bots running on Bitget
- Beginner in scalping specifically — building this tool to learn and automate
- Uses Bitget (futures), Kraken (spot), and Saxo (indices/forex — phase 2)
- Comfortable with Python, knows risk management (stop loss sizing, position sizing)
- Language: French — comments and UI can be in French
- **Platform: Windows** — tout le code doit fonctionner sur Windows (paths, line endings CRLF pour .bat, pas de commandes Unix-only). Le `.gitattributes` gère la normalisation des fins de ligne.
- **Git: ne jamais ajouter de Co-Authored-By dans les commits**

## Architecture Decisions (validated)

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
| Deployment         | git push → SSH → deploy.sh → docker compose             |
| Config format      | YAML              | Editable without code changes or redeployment           |
| Testing            | pytest            | Critical components must have unit tests                |
| Forex/Saxo         | Phase 2           | Different API, different strategies, added later        |

## Architecture Sprint 1 (implémenté)

### Décisions clés

- **pyproject.toml à la racine** (pas dans backend/) → imports propres `from backend.core.models import Candle`
- **Process unique** : DataEngine intégré dans le lifespan FastAPI (pas de process séparé)
- **100% async** : database, data engine, scripts CLI utilisent `asyncio.run()` — pas de wrappers sync
- **Flag ENABLE_WEBSOCKET** : variable d'env (défaut `true`), permet de désactiver le DataEngine en dev pour éviter les reconnexions WebSocket lors du `--reload` d'uvicorn
- **Buffer rolling borné** : max 500 bougies par (symbol, timeframe) en mémoire
- **Rate limiter par catégorie** : market_data, trade, account, position (token bucket)
- **Mark price vs last price** : distinction dans les modèles pour calculs de liquidation
- **SL réaliste** : coût SL inclut distance + taker_fee + slippage (configurable)
- **Groupes de corrélation** : limite l'exposition sur assets corrélés (max concurrent same direction)
- **Marge cross** : suivi du min_free_margin_percent pour éviter la liquidation en cascade

## Project Structure

```text
scalp-radar/
├── CLAUDE.md                     # Project brief (ce fichier)
├── README.md                     # Setup & usage guide
├── pyproject.toml                # Python deps (à la RACINE, géré par uv)
├── .env.example                  # API keys template (never commit real .env)
├── .gitignore
├── .dockerignore                 # Exclut .venv, node_modules, data, logs, .git
├── dev.bat                       # Windows: lance backend + frontend ensemble
├── Dockerfile.backend            # python:3.12-slim + uv
├── Dockerfile.frontend           # node:18 build → nginx:alpine
├── docker-compose.yml            # 2 services : backend (8000) + frontend (80)
├── nginx.conf                    # Proxy /api, /health, /ws → backend:8000
├── deploy.sh                     # Déploiement : graceful shutdown + rollback (--clean pour fresh start)
│
├── config/                       # ALL tunable parameters — no hardcoding in code
│   ├── assets.yaml               # Traded pairs + correlation groups (5 assets, 2 groupes)
│   ├── strategies.yaml           # Strategy parameters (8 stratégies + custom + per_asset)
│   ├── risk.yaml                 # Kill switch, position sizing, fees, slippage, margin, SL/TP
│   ├── exchanges.yaml            # Bitget: WebSocket, rate limits par catégorie, API config
│   └── param_grids.yaml          # Espaces de recherche pour l'optimisation WFO (5m + 1h)
│
├── backend/
│   ├── __init__.py
│   ├── main.py                   # Point d'entrée standalone (DataEngine sans API)
│   ├── __main__.py               # python -m backend support
│   │
│   ├── core/
│   │   ├── models.py             # Enums + 15 modèles Pydantic (Candle, Signal, Trade, Position...)
│   │   ├── config.py             # YAML loader + validation croisée (Pydantic)
│   │   ├── database.py           # SQLite async (aiosqlite, WAL mode)
│   │   ├── indicators.py          # RSI, VWAP, ADX+DI, ATR, SMA, EMA, Bollinger, SuperTrend, régime (pur numpy)
│   │   ├── incremental_indicators.py # Buffers rolling pour indicateurs live (Simulator)
│   │   ├── position_manager.py   # Sizing, fees, slippage, TP/SL (mono-position, réutilisé par engine + simulator)
│   │   ├── grid_position_manager.py # Sizing multi-position DCA (N niveaux, prix moyen, TP/SL global)
│   │   ├── state_manager.py      # Crash recovery : save/load état runners (JSON atomique)
│   │   ├── data_engine.py        # ccxt Pro WebSocket + polling OI/funding + buffer rolling
│   │   ├── rate_limiter.py       # Token bucket par catégorie d'endpoint
│   │   └── logging_setup.py      # loguru: console + fichier JSON + fichier erreurs
│   │
│   ├── strategies/
│   │   ├── base.py               # BaseStrategy ABC, StrategyContext, StrategySignal, EXTRA_* constants
│   │   ├── base_grid.py          # BaseGridStrategy ABC pour stratégies grid/DCA multi-position
│   │   ├── factory.py            # create_strategy() + get_enabled_strategies()
│   │   ├── vwap_rsi.py           # VWAP+RSI mean reversion (5m, RANGING)
│   │   ├── momentum.py           # Momentum Breakout (5m, TRENDING)
│   │   ├── funding.py            # Funding Rate Arbitrage (15m, paper trading only)
│   │   ├── liquidation.py        # Liquidation Zone Hunting (5m, paper trading only)
│   │   ├── bollinger_mr.py       # Bollinger Band Mean Reversion (1h)
│   │   ├── donchian_breakout.py  # Donchian Channel Breakout (1h)
│   │   ├── supertrend.py         # SuperTrend trend-following (1h)
│   │   └── envelope_dca.py       # Envelope DCA multi-niveaux (1h, grid/DCA)
│   ├── optimization/
│   │   ├── __init__.py           # STRATEGY_REGISTRY + create_strategy_with_params() + is_grid_strategy()
│   │   ├── walk_forward.py       # WFO grid search 2 passes + ProcessPool/fallback séquentiel
│   │   ├── overfitting.py        # Monte Carlo, DSR, stabilité, convergence cross-asset
│   │   ├── report.py             # Grading A-F, validation Bitget, apply_to_yaml
│   │   ├── indicator_cache.py    # Cache numpy pré-calculé pour fast engine (toutes variantes du grid)
│   │   ├── fast_backtest.py      # Fast engine mono-position (5m + 1h simple, numpy-only)
│   │   └── fast_multi_backtest.py # Fast engine multi-position (grid/DCA, numpy-only)
│   ├── backtesting/
│   │   ├── engine.py             # Moteur event-driven mono-position (délègue au PositionManager)
│   │   ├── multi_engine.py       # Moteur event-driven multi-position (grid/DCA, GridPositionManager)
│   │   ├── metrics.py            # BacktestMetrics + format table console
│   │   ├── simulator.py          # LiveStrategyRunner + GridStrategyRunner + Simulator (paper trading)
│   │   ├── arena.py              # StrategyArena (classement parallèle)
│   │   └── extra_data_builder.py # Alignement funding/OI par timestamp (forward-fill)
│   ├── execution/
│   │   ├── executor.py          # Executor multi-position (3 paires × 4 stratégies, SL/TP, watchOrders)
│   │   ├── risk_manager.py      # LiveRiskManager (pre-trade checks, kill switch, corrélation groups)
│   │   └── adaptive_selector.py # Gate stratégies live (évalue Arena + live_eligible config)
│   │
│   ├── api/
│   │   ├── server.py             # FastAPI + lifespan (DataEngine, Simulator, Executor, Arena, StateManager, Watchdog, Heartbeat)
│   │   ├── health.py             # GET /health → status, data_engine, database, uptime, watchdog, executor
│   │   ├── simulator_routes.py   # GET /api/simulator/* (status, positions, trades, performance)
│   │   ├── conditions_routes.py  # GET /api/simulator/conditions, /signals/matrix, /simulator/equity
│   │   ├── arena_routes.py       # GET /api/arena/* (ranking, strategy detail)
│   │   ├── signals_routes.py     # GET /api/signals/recent
│   │   └── websocket_routes.py   # WS /ws/live (push temps réel + prix + executor)
│   │
│   ├── alerts/
│   │   ├── telegram.py           # Client Telegram via httpx (API Bot directe)
│   │   ├── notifier.py           # Notifier centralisé + AnomalyType enum
│   │   └── heartbeat.py          # Heartbeat Telegram périodique (intervalle configurable)
│   └── monitoring/
│       └── watchdog.py           # Surveillance WS, data freshness, stratégies (dépendances explicites)
│
├── frontend/                     # React + Vite (dashboard dark theme)
│   ├── package.json
│   ├── vite.config.js            # Proxy /api, /health, /ws → backend:8000
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx               # Layout grid + WebSocket
│       ├── styles.css            # Dark theme (variables CSS)
│       ├── hooks/
│       │   ├── useApi.js         # Polling hook (configurable interval)
│       │   └── useWebSocket.js   # WS avec reconnexion auto (backoff exponentiel)
│       └── components/
│           ├── Header.jsx         # Logo, tabs Scanner/Heatmap/Risque, status dots
│           ├── Scanner.jsx        # Table principale assets (conditions, indicateurs, détail extensible)
│           ├── SignalDots.jsx     # Grille de dots colorés par stratégie
│           ├── SignalDetail.jsx   # Panneau extensible (ScoreRing + breakdown + indicateurs)
│           ├── SignalBreakdown.jsx # Barres de progression par stratégie
│           ├── ScoreRing.jsx     # Anneau SVG score global
│           ├── Spark.jsx         # Sparkline SVG minimaliste
│           ├── Heatmap.jsx       # Matrice assets × stratégies (conditions colorées)
│           ├── RiskCalc.jsx      # Calculatrice de risque client-side
│           ├── ExecutorPanel.jsx # Statut executor + multi-positions + selector
│           ├── ActivePositions.jsx # Multi-position display
│           ├── ActivityFeed.jsx  # Trade activity timeline
│           ├── SessionStats.jsx  # Sidebar P&L, trades, win rate (via WS)
│           ├── EquityCurve.jsx   # Courbe d'equity SVG (poll 30s)
│           ├── AlertFeed.jsx     # Timeline signaux chronologique
│           ├── TradeHistory.jsx  # Trades récents collapsible (poll 10s)
│           ├── ArenaRankingMini.jsx # Classement compact arena (via WS)
│           ├── Tooltip.jsx       # Tooltips explicatifs
│           └── CollapsibleCard.jsx # Conteneur réutilisable collapsible
│
├── tests/
│   ├── conftest.py               # Fixtures partagées (config_dir temporaire)
│   ├── test_models.py            # 17 tests : enums, Candle, OrderBook, Trade, Signal, SessionState
│   ├── test_config.py            # 11 tests : chargement, validation, erreurs
│   ├── test_database.py          # 12 tests : CRUD candles, session state, signals, trades (async)
│   ├── test_indicators.py        # 27 tests : RSI, VWAP, ADX, ATR, SMA, EMA, Bollinger, SuperTrend, régime
│   ├── test_strategy_vwap_rsi.py # 13 tests : signaux, filtres, check_exit, compute_indicators
│   ├── test_backtesting.py       # 17 tests : engine, métriques, OHLC, sizing, Sortino
│   ├── test_position_manager.py  # 10 tests : sizing, fees, slippage, TP/SL, heuristique
│   ├── test_incremental_indicators.py # 5 tests : buffer rolling, trim, indicateurs
│   ├── test_strategy_momentum.py # 10 tests : breakout, volume, trend filter, ADX exit
│   ├── test_strategy_funding.py  # 7 tests : extreme rates, delay, neutral exit
│   ├── test_strategy_liquidation.py # 7 tests : zones, OI threshold, proximity score
│   ├── test_simulator.py         # 14 tests : runner, on_candle, kill switch, regime change
│   ├── test_arena.py             # 9 tests : ranking, profit factor, drawdown, detail
│   ├── test_api_simulator.py     # 7 tests : endpoints status, trades, ranking, signals
│   ├── test_state_manager.py    # 16 tests : save, load, restore, round-trip, periodic save
│   ├── test_telegram.py         # 10 tests : send_message, trade alert, kill switch, notifier, grid messages
│   ├── test_heartbeat.py        # 3 tests : format message, no trades, stop
│   ├── test_watchdog.py         # 8 tests : all ok, WS down, data stale, cooldown, lifecycle
│   ├── test_executor.py         # 53 tests : multi-position, selector, persistence, reconciliation
│   ├── test_risk_manager.py     # 22 tests : pre-trade checks, kill switch, corrélation groups, leverage_override
│   ├── test_adaptive_selector.py # 12 tests : critères perf, live_eligible, symboles actifs
│   ├── test_optimization.py      # 52 tests : WFO, Monte Carlo, DSR, stabilité, convergence, grading
│   ├── test_funding_oi_data.py  # 23 tests : fetch funding/OI, extra_data_builder, alignement
│   ├── test_new_strategies.py   # 48 tests : bollinger_mr, donchian_breakout, supertrend (1h)
│   ├── test_fast_backtest.py    # 18 tests : fast engine, indicator_cache, parité
│   ├── test_multi_engine.py     # 32 tests : MultiPositionEngine, grid/DCA backtest
│   ├── test_grid_runner.py      # 28 tests : GridStrategyRunner, paper trading grid/DCA
│   └── test_executor_grid.py   # 25 tests : Executor grid DCA, ouverture/fermeture/surveillance/state
│
├── scripts/
│   ├── fetch_history.py          # Backfill async ccxt REST + tqdm (6 mois, reprise auto, --exchange)
│   ├── fetch_funding.py          # Fetch historique funding rates (Bitget API)
│   ├── fetch_oi.py               # Fetch historique open interest (Binance API)
│   ├── run_backtest.py           # CLI backtest runner (--symbol, --days, --json)
│   ├── optimize.py               # CLI optimisation WFO (--all, --apply, --check-data, --dry-run)
│   ├── parity_check.py           # Compare moteurs mono vs multi-position
│   └── __main__.py               # python -m scripts support
│
├── data/                         # SQLite DB + données (gitignored)
│
└── docs/
    ├── plans/
    │   ├── sprint-1-foundations.md
    │   ├── sprint-2-backtesting.md
    │   ├── sprint-3-simulator-arena.md
    │   ├── sprint-4-production.md
    │   ├── sprint-5a-executor.md
    │   ├── sprint-5b-scaling.md
    │   ├── sprint-6-dashboard-v2.md
    │   ├── sprint-6b-dashboard-ux-overhaul.md
    │   ├── sprint-7-optimization.md
    │   ├── sprint-7b-funding-oi-optimization.md
    │   ├── sprint-8-backtest-dashboard.md
    │   ├── sprint-9-new-1h-strategies.md
    │   ├── sprint-10-multi-position-engine.md
    │   ├── sprint-11-paper-trading-grid.md
    │   ├── hotfix-monte-carlo-underpowered.md
    │   └── sprint-12-executor-grid-dca.md
    └── prototypes/
        └── Scalp radar v2.jsx    # Prototype React (référence design Sprint 3)
```

## Trading Strategies

### Strategy 1 — VWAP + RSI Mean Reversion (implement first)
- Timeframe: 1-5 min, with 15min trend confirmation
- Entry: Price touches VWAP with RSI extreme (<25 long, >75 short) + volume spike
- Multi-timeframe: do NOT take a 1min long if 15min trend is bearish
- Assets: BTC/USDT, ETH/USDT, SOL/USDT
- Parameters in config/strategies.yaml

### Strategy 2 — Liquidation Zone Hunting
- Maps estimated liquidation zones via open interest + average leverage
- Trades the cascade when price approaches these zones
- Data source: Bitget open interest API + estimated liquidation levels

### Strategy 3 — Order Flow Imbalance
- Bid/ask ratio on L2 order book, large order detection, volume delta, absorptions
- BEST USED AS CONFIRMATION FILTER for other strategies, not standalone
- Adds confidence score to signals from strategies 1, 2, 4

### Strategy 4 — Momentum Breakout
- Range breakout on volume with multi-timeframe confirmation
- Best during high volatility regimes

### Strategy 5 — Funding Rate Arbitrage
- Extreme negative funding → go long, extreme positive → go short
- Slower frequency, highly automatable, complementary to faster strategies

### Strategy 6 — Bollinger Band Mean Reversion (1h)

- Timeframe: 1h
- Entry: prix touche bande inférieure (long) ou supérieure (short) de Bollinger
- Exit: retour à la SMA (TP dynamique via check_exit, pas un prix fixe)
- TP distant fictif (entry×2 / entry×0.5) — le vrai TP est géré par check_exit (SMA crossing)
- Parité fast engine / BacktestEngine

### Strategy 7 — Donchian Channel Breakout (1h)

- Timeframe: 1h
- Entry: cassure du canal Donchian (N bougies high/low)
- TP/SL basés sur ATR multiples
- Trend-following, complémentaire au mean reversion

### Strategy 8 — SuperTrend (1h)

- Timeframe: 1h
- Entry: changement de direction SuperTrend (ATR-based)
- TP/SL en pourcentage configurable
- Trend-following

### Strategy 9 — Envelope DCA Multi-Niveaux (1h, grid)

- Timeframe: 1h, stratégie grid/DCA (multi-position)
- SMA + N enveloppes asymétriques (bandes pas symétriques, asymétrie log-return)
- Entrée à chaque niveau touché (DCA progressif)
- TP = retour à la SMA. SL = % depuis prix moyen pondéré
- Hérite de BaseGridStrategy (pas BaseStrategy directement)
- Gérée par GridStrategyRunner (paper) et MultiPositionEngine (backtest)
- Seule stratégie `enabled: true` actuellement (paper trading Sprint 11)

## Multi-Strategy Arena

All strategies run in parallel on the same data feed with identical virtual capital.
Simulator tracks: win rate, profit factor, max drawdown, Sharpe ratio, net P&L after fees.
Adaptive selector allocates more capital to top performers, pauses underperformers.

## Critical Design Requirements

### Database & Persistence
- All klines, trades, signals stored in database (SQLite for dev, TimescaleDB for heavy prod)
- Database abstraction layer so the switch is transparent
- Structured JSON logging for every trade with full context

### State Management (crash recovery)
- On startup: check Bitget API for open positions and pending orders
- Restore session P&L counter (kill switch must survive restarts)
- Log last known state to database, reload on boot

### Risk Management (non-negotiable)
- Kill switch: auto-stop if X% capital lost in a session (persisted across restarts)
- Position sizing based on stop loss distance and max risk per trade
- Liquidation distance always calculated and displayed
- Max concurrent positions limit
- Prefer limit orders (maker: 0.02%) over market orders (taker: 0.06%) when strategy allows

### Fee-Aware P&L
- All P&L calculations MUST be net of fees
- Backtester and simulator use real Bitget fee structure
- Dashboard always shows gross AND net P&L
- At x20 leverage targeting 0.3% moves, fees are ~40% of gross profit — this must be visible

### Backtesting Rigor
- No look-ahead bias: strategy sees only data available at time T
- Slippage model based on order book depth, not flat percentage
- Limit order fill probability < 100% (model partial fills)
- Results must show: with fees vs without fees comparison

### Multi-Timeframe

- Data engine aggregates klines in 1min, 5min, 15min, 1h simultaneously
- Strategy base class receives dict of timeframes: {"1m": [...], "5m": [...], "15m": [...], "1h": [...]}
- Strategies can use higher timeframes as trend filter
- 1h inclus pour le swing baseline (Arena) et les filtres de tendance long terme

### Rate Limiting
- Centralized rate limiter in core/ shared by ALL strategies and execution
- Respects Bitget limits (e.g., 20 req/s on certain endpoints)
- Queues requests, does not drop them

### Monitoring & Alerts (production)
- /health endpoint: WS connected, data freshness, strategies running, open positions
- Telegram heartbeat every hour: "alive, session P&L: +X%, 3 trades, 2 wins"
- Anomaly alerts: WS disconnect >30s, API latency >2s, strategy silent >1h

### Security
- .env never committed, in .gitignore
- Bitget API key: IP whitelist to server IP only (192.168.1.200)
- Docker secrets for production (or .env injected, not baked in image)
- API key permissions: futures read + trade ONLY, no withdrawal

### Testing (pytest)
- Risk manager: position sizing, liquidation distance, kill switch trigger
- Strategies: known input → expected signal output
- Simulator: fee calc, slippage model, fill probability
- State manager: simulate crash, verify recovery
- Config loader: missing keys, invalid values, defaults

## Config Files

Les 4 fichiers YAML dans `config/` sont la source de vérité pour tous les paramètres.
Voir les fichiers directement — ils sont exhaustifs et commentés.

- `assets.yaml` — 5 assets (BTC, ETH, SOL, DOGE, LINK), timeframes [1m, 5m, 15m, 1h], groupes de corrélation
- `strategies.yaml` — 8 stratégies (4 scalp 5m + 3 swing 1h + 1 grid/DCA 1h) + custom_strategies (swing baseline)
- `risk.yaml` — kill switch, position sizing, fees, slippage, margin cross, SL/TP server-side
- `exchanges.yaml` — Bitget: WebSocket, rate limits par catégorie, API config (USDT-M, mark_price)
- `param_grids.yaml` — Espaces de recherche WFO par stratégie (5m + 1h) + per-strategy WFO config (is_days, oos_days, step_days)

## Sprint Plan

### Sprint 1 — Foundations ✅

Complet. Infrastructure de base : scaffold, configs YAML, modèles Pydantic, database async,
DataEngine WebSocket (ccxt Pro), API FastAPI avec health check, rate limiter, logging loguru,
script fetch_history, frontend scaffold, 40 tests passants.

### Sprint 2 — Backtesting & First Strategy ✅

Complet. Moteur de backtesting event-driven + stratégie VWAP+RSI mean reversion.
Plan détaillé : `docs/plans/sprint-2-backtesting.md`

**Fichiers créés :**
- `backend/core/indicators.py` — RSI (Wilder), VWAP rolling 24h, ADX+DI+/DI-, ATR, SMA, EMA, détection régime
- `backend/strategies/base.py` — BaseStrategy ABC, StrategyContext, StrategySignal, OpenPosition
- `backend/strategies/vwap_rsi.py` — Stratégie VWAP+RSI (entry/exit/compute_indicators)
- `backend/backtesting/engine.py` — Moteur event-driven (OHLC heuristique, sizing SL réel, multi-TF alignment)
- `backend/backtesting/metrics.py` — Métriques (Sharpe, Sortino, profit factor net+gross, fee drag)
- `scripts/run_backtest.py` — CLI runner (`uv run python -m scripts.run_backtest`)
- 95 tests passants (44 Sprint 2 + 51 Sprint 1)

**Décisions clés :**
- Indicateurs pré-calculés une seule fois via `compute_indicators()` (numpy)
- Alignement multi-TF géré par le moteur (`last_available_before()`)
- Heuristique OHLC pour résolution TP/SL intra-bougie
- Position sizing intègre le coût SL réel (distance + taker_fee + slippage)
- Dual profit factor : net (scalping) + gross (benchmarks)
- Equity curve par bougie (pas par trade) pour drawdown duration fiable

**Résultats backtest baseline (180j, 3 paires) :**

- 85 trades total (BTC:20, ETH:25, SOL:40), PF gross 1.44-2.11
- Edge solide en RANGING (+$2,457, 53% WR sur 45 trades)
- Pertes en TRENDING_DOWN (-$2,517 sur 28 trades) annulent les gains
- Net global ~breakeven (+$11) — baseline fonctionnelle

**TODO quick fix (Sprint 3 ou hotfix) :**

- `check_exit()` devrait couper la position quand le régime bascule de RANGING → TRENDING
- Ajouterait une sortie anticipée sur changement de régime pendant le trade
- Aurait éliminé une bonne partie des 28 trades trending perdants

### Sprint 3 — Simulator, Stratégies & Frontend ✅

Complet. Paper trading live, 4 stratégies, Arena, API REST+WS, frontend dashboard.

**Infrastructure extraite :**

- `PositionManager` — sizing, fees, slippage, TP/SL réutilisé par BacktestEngine ET Simulator
- `IncrementalIndicatorEngine` — buffers numpy rolling (500 candles), recalcul < 1ms par update
- `StrategyFactory` — create_strategy() et get_enabled_strategies() depuis la config YAML

**4 stratégies implémentées :**

- **VWAP+RSI** (Sprint 2) — mean reversion en RANGING, filtre 15m anti-trend
- **Momentum Breakout** — trade AVEC la tendance, complémentaire à VWAP+RSI
- **Funding Rate Arbitrage** — scalp lent sur taux extrêmes (paper trading only)
- **Liquidation Zone Hunting** — cascade OI + zones de liquidation estimées (paper trading only)

**Simulator (paper trading live) :**

- `LiveStrategyRunner` — un runner par stratégie, capital virtuel isolé (10k$)
- `Simulator` — orchestrateur, câblé sur DataEngine via on_candle callback
- Kill switch : coupe si perte session >= 5% (configurable dans risk.yaml)
- Quick fix régime : coupe la position quand RANGING → TRENDING

**Arena :**

- Classement parallèle par net_return_pct, profit factor, max drawdown
- Isolation totale : chaque stratégie = capital séparé

**API (8 endpoints + WebSocket) :**

- `/api/simulator/*` — status, positions, trades, performance
- `/api/arena/*` — ranking, strategy detail
- `/api/signals/recent` — derniers signaux
- `/ws/live` — push temps réel (3s interval)

**Frontend MVP (dark theme) :**

- 5 composants : Header, ArenaRanking, SignalFeed, SessionStats, TradeHistory
- Hooks : useApi (polling configurable), useWebSocket (reconnexion backoff exponentiel)
- 166 tests passants (69 nouveaux + 97 existants)

**Décisions clés Sprint 3 :**

- Constantes `EXTRA_*` dans base.py (pas de magic strings pour extra_data)
- Champ `source` en DB (backtest/simulation/live) pour filtrer les trades
- Funding et Liquidation non backtestables (pas de données historiques OI/funding)
- `zone_buffer_percent` élargi de 0.5% à 1.5% (estimation levier trop grossière)

### Sprint 4 — Production ✅

Complet. Crash recovery, alertes Telegram, monitoring Watchdog, Docker Compose.
Plan détaillé : `docs/plans/sprint-4-production.md`

**Crash Recovery (StateManager) :**

- `StateManager` — sauvegarde périodique (60s) + restauration au boot
- Écriture atomique (tmp + `os.replace`), lecture robuste (fichier absent/corrompu → fresh start)
- `saved_state` passé à `simulator.start()` — runners créés avec le bon capital AVANT le callback `on_candle`
- `LiveStrategyRunner.restore_state()` — restaure capital, stats, kill_switch, position ouverte

**Alertes Telegram :**

- `TelegramClient` — httpx (pas de dépendance supplémentaire), retry 1x
- `Notifier` — point d'entrée unique, graceful si telegram=None (log only)
- `AnomalyType` enum — types structurés (WS_DISCONNECTED, DATA_STALE, ALL_STRATEGIES_STOPPED, KILL_SWITCH_GLOBAL)
- `Heartbeat` — intervalle configurable (défaut 3600s, env `HEARTBEAT_INTERVAL`)
- Notifications : startup, shutdown, trade, kill switch, anomalies

**Monitoring (Watchdog) :**

- Dépendances explicites (data_engine, simulator, notifier) — pas app.state — testable unitairement
- Checks toutes les 30s : WS connecté, data freshness < 5min, stratégies actives
- Anti-spam : cooldown 5 min par type d'anomalie
- Status intégré dans `/health`

**Docker & Déploiement :**

- `Dockerfile.backend` — python:3.12-slim + uv
- `Dockerfile.frontend` — node:18 build → nginx:alpine
- `nginx.conf` — proxy vers `http://backend:8000` (service Docker Compose, pas localhost)
- `docker-compose.yml` — 2 services + healthcheck + volumes persistants
- `deploy.sh` — graceful shutdown + rollback sur health check échoué. Flag `--clean` / `-c` : kill brutal + supprime state files (fresh start sans perdre la DB)

**Lifespan complet :**

Startup : DB → Telegram/Notifier → DataEngine → StateManager load → Simulator.start(saved_state) → periodic save → Arena → Watchdog → Heartbeat → notify_startup
Shutdown : notify_shutdown → heartbeat → watchdog → state save → simulator → engine → db

- 200 tests passants (166 existants + 34 nouveaux)

### Sprint 5a — Executor Minimal + Safety ✅

Complet. Executor live trading minimal : 1 stratégie (VWAP+RSI), 1 paire (BTC/USDT:USDT).
Plan détaillé : `docs/plans/sprint-5a-executor.md`

**Executor (`execution/executor.py`) :**

- Pattern observer : Simulator émet TradeEvent → Executor réplique en ordres réels Bitget
- Market order d'entrée (taker 0.06%, cohérent avec backtester/simulator)
- SL = market trigger (garanti), TP = limit trigger (maker fee)
- **Règle #1 : JAMAIS de position sans SL** — retry 2x, sinon close market immédiat + alerte Telegram
- `watchOrders()` via ccxt Pro (détection fills quasi temps réel) + polling 5s en fallback
- Réconciliation 4 cas au boot (position exchange ↔ état local, dont fetch P&L downtime)
- Mapping symboles `BTC/USDT` → `BTC/USDT:USDT` dans executor uniquement (assets.yaml inchangé)
- `load_markets()` au start pour min_order_size/tick_size réels Bitget
- Rate limiting : `asyncio.sleep(0.1)` entre chaque ordre séquentiel
- Leverage vérifié avant changement (pas de set si position ouverte)
- Instance ccxt Pro authentifiée séparée du DataEngine

**LiveRiskManager (`execution/risk_manager.py`) :**

- Pre-trade checks : kill switch, position dupliquée, max concurrent, marge suffisante
- Double kill switch : Simulator (virtuel) + RiskManager (argent réel, >= 5% perte)
- State persistence : get_state/restore_state pour crash recovery

**Intégration :**

- `simulator.py` : `_pending_events` queue + swap atomique pour callback Executor
- `server.py` : section 4b lifespan (après Simulator, avant Watchdog)
- `state_manager.py` : save/load executor state (écriture atomique)
- `health.py` : section executor dans /health
- `watchdog.py` : checks executor déconnecté + kill switch live
- `notifier.py` + `telegram.py` : alertes ordres live (ouverture, fermeture, SL échoué)
- `config.py` : `LIVE_TRADING=false` (défaut, simulation only)

**Lifespan mis à jour :**

Startup : DB → Telegram/Notifier → DataEngine → StateManager → Simulator → **Executor** → Arena → Watchdog → Heartbeat
Shutdown : notify_shutdown → heartbeat → watchdog → **executor state save + stop** → state save → simulator → engine → db

- 248 tests passants (200 existants + 48 nouveaux)

### Sprint 5b — Scaling (3 paires × 4 stratégies) ✅

Complet. Multi-position, multi-stratégie, sélection adaptative.
Plan détaillé : `docs/plans/sprint-5b-scaling.md`

**AdaptiveSelector (`execution/adaptive_selector.py`) :**

- Gate quelles stratégies peuvent trader en live (OPEN seulement, CLOSE passe toujours)
- Évalue périodiquement (5 min) depuis Arena.get_ranking()
- Critères : `live_eligible` config, `is_active`, `min_trades >= 3`, `net_return_pct > 0`, `profit_factor >= 1.0`
- `set_active_symbols()` appelé par Executor après leverage setup (try/except par symbole)
- `_STRATEGY_CONFIG_ATTR` mapping pour accès live_eligible depuis config stratégies

**Executor multi-position (`execution/executor.py`) :**

- `self._positions: dict[str, LivePosition]` (était `_position: LivePosition | None`)
- Properties `position` (backward compat) et `positions` (liste)
- `selector: AdaptiveSelector | None` remplace `_ALLOWED_STRATEGIES` / `_ALLOWED_SYMBOLS` hardcodés
- `start()` : leverage setup try/except par symbole (si SOL échoue, BTC+ETH continuent)
- `_reconcile_on_boot()` : itère tous les symboles configurés via `_reconcile_symbol()`
- `_watch_orders_loop()` : watch ALL ordres (sans filtre symbole)
- `_process_watched_order()` : scan toutes les positions + log debug ordres non matchés
- `_cancel_orphan_orders()` : tracked_ids depuis TOUTES les positions
- `restore_positions()` : backward compat ancien format `"position"` → nouveau `"positions"`

**RiskManager — corrélation groups :**

- `pre_trade_check()` : limite direction dans groupe de corrélation (`max_concurrent_same_direction: 2`)
- Helpers : `_get_correlation_group(symbol)` (strip `:USDT`), `_get_max_same_direction(group)`

**Config :**

- `live_eligible: bool` par stratégie (True pour vwap_rsi/momentum, False pour funding/liquidation)
- `AdaptiveSelectorConfig` dans risk.yaml (min_trades, min_profit_factor, eval_interval_seconds)
- 4 stratégies enabled (momentum, funding, liquidation activées en simulation)

**Frontend (ExecutorPanel.jsx) :**

- Multi-positions via `.map()` avec PositionCard (symbole + stratégie + direction)
- Affichage statut AdaptiveSelector (stratégies live autorisées)
- Backward compat : fallback `executor.position` si `positions` absent

**API routes (`executor_routes.py`) :**

- `POST /api/executor/test-trade?symbol=BTC/USDT` — query param symbole (défaut BTC)
- `POST /api/executor/test-close?symbol=BTC/USDT` — idem
- Quantités minimales par asset

**Lifespan mis à jour :**

Startup : ... → Simulator → Arena → **AdaptiveSelector** → **Executor(selector=selector)** → selector.start() → Watchdog
Shutdown : selector.stop() → executor state save + stop → ...

- 284 tests passants (252 existants + 32 nouveaux)

### Sprint 6 Phase 1 — Dashboard V2 ✅

Refonte complète du frontend. Plan détaillé : `docs/plans/sprint-6-dashboard-v2.md`

**Backend (5 fichiers modifiés, 1 créé) :**

- `base.py` : méthode abstraite `get_current_conditions()` — read-only pour le dashboard
- 4 stratégies : implémentation `get_current_conditions()` (VWAP+RSI: 4 conditions, Momentum: 3, Funding: 2, Liquidation: 2)
- `simulator.py` : `get_conditions()`, `get_signal_matrix()`, `get_equity_curve()` avec cache invalidé par candle
- `conditions_routes.py` : 3 endpoints (`/api/simulator/conditions`, `/api/signals/matrix`, `/api/simulator/equity`)
- `websocket_routes.py` : enrichi avec prix live et statut executor
- `server.py` : ajout conditions_router

**Frontend (12 créés, 5 modifiés, 2 supprimés) :**

- Layout 2 colonnes : zone principale (tabs Scanner/Heatmap/Risque) + sidebar
- Scanner : table assets avec prix, var%, direction, sparkline, score, SignalDots + détail extensible (ScoreRing, SignalBreakdown, indicateurs)
- Heatmap : matrice assets × stratégies, cellules à gradient d'intensité proportionnel + colonne score Σ
- RiskCalc : calculateur interactif (capital, levier slider, SL%, résultats: taille position, perte max, distance liquidation)
- ExecutorPanel : badge PAPER/LIVE, position ouverte, P&L non réalisé
- EquityCurve : courbe SVG avec aire sous la courbe, baseline capital initial
- AlertFeed : timeline signaux chronologique inverse
- ArenaRankingMini : classement compact sidebar (remplace ArenaRanking)
- ScoreRing : anneau SVG score global (couleur par ratio)
- Spark : sparkline SVG avec gradient fill, aire sous la courbe, dot animé (alimentée par 60 derniers close)
- SignalDots : pastilles 22px avec initiales stratégie (V/M/F/L), bordure et fond colorés par ratio
- SignalBreakdown : barres de progression par stratégie
- SessionStats : réécrit pour utiliser wsData (pas de polling)
- TradeHistory : collapsible (5 visible, expandable)
- styles.css : refonte complète (471 lignes, CSS variables, dark theme)
- Supprimés : SignalFeed.jsx, ArenaRanking.jsx

**Décisions clés Sprint 6 :**

- `get_current_conditions()` séparé de `check_entry()` — read-only, pas d'impact sur le trading
- Backend renvoie données brutes structurées, frontend formate (noms français côté frontend)
- SVG inline pour tous les graphiques (pas de dépendance chart.js/recharts)
- Cache conditions invalidé par candle dans `_dispatch_candle()`
- CSS classes (pas d'inline styles), CSS variables dark theme existantes

- 252 tests passants (0 régression)

### Sprint 6 Phase 2 — Dashboard Polish ✅

Alignement visuel du dashboard sur le prototype (`docs/prototypes/Scalp radar v2.jsx`).

**Backend :**

- `simulator.py` : `get_conditions()` renvoie `sparkline` (60 derniers close prices depuis le buffer 1m)

**Frontend (5 fichiers modifiés) :**

- `Spark.jsx` : gradient fill sous la courbe + dot animé (pulsation) sur le dernier point
- `Scanner.jsx` : colonnes Var. (%), Dir. (LONG/SHORT basé sur RSI/VWAP), Score (meilleure stratégie), tri par score décroissant
- `Heatmap.jsx` : gradient d'intensité proportionnel au ratio (pas statique), colonne Σ score par asset
- `SignalDots.jsx` : pastilles 22×22px avec initiales stratégie (V/M/F/L), bordure colorée
- `styles.css` : `.signal-dot` 22px avec bordure, `.score-number`, `.asset-dot`

- 284 tests passants (0 régression)

### Sprint 7 — Parameter Optimization & Overfitting Detection ✅

Complet. Optimisation automatique des paramètres par stratégie × asset avec détection d'overfitting.
Plan détaillé : `docs/plans/sprint-7-optimization.md`

**Phase 0 — Données & config :**

- `config/assets.yaml` : +DOGE/USDT, +LINK/USDT, +groupe corrélation altcoins
- `config/strategies.yaml` : +section `per_asset: {}` sur vwap_rsi et momentum
- `backend/core/models.py` : +champ `exchange: str = "bitget"` sur Candle
- `backend/core/config.py` : +`per_asset` + `get_params_for_symbol()` sur VwapRsiConfig/MomentumConfig
- `backend/core/database.py` : migration idempotente (backup auto, PK 4 colonnes avec exchange, index)
- `backend/strategies/base.py` : +`_resolve_param()` (overrides per_asset au runtime)
- `backend/strategies/vwap_rsi.py` / `momentum.py` : TP/SL via `_resolve_param()`
- `scripts/fetch_history.py` : +`--exchange binance|bitget`, factory exchange

**Phase 1 — Walk-Forward Optimizer :**

- `backend/optimization/__init__.py` : STRATEGY_REGISTRY + `create_strategy_with_params()`
- `config/param_grids.yaml` : espaces de recherche par stratégie (default + per-asset overrides)
- `backend/optimization/walk_forward.py` : WFO complet (~600 lignes)
  - Grid search 2 passes (coarse LHS 500 → fine ±1 step autour du top 20)
  - ProcessPoolExecutor avec initializer (candles chargées 1× par worker)
  - Fallback séquentiel automatique si pool crashe (BrokenExecutor)
  - `max_tasks_per_child=50` pour recycler la mémoire workers
  - Workers limités à 4 (`min(cpu_count, 4)`)
- `backend/backtesting/engine.py` : +`run_backtest_single()` (module-level pour les workers)

**Phase 2 — Détection d'overfitting :**

- `backend/optimization/overfitting.py` : `OverfitDetector` (~390 lignes)
  - Monte Carlo block bootstrap (blocs de 7 trades, 1000 sims, seed configurable)
  - Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014, `math.erf` au lieu de scipy)
  - Parameter stability (perturbation ±10/20%, score plateau vs cliff)
  - Cross-asset convergence (coefficient de variation)

**Phase 3 — Validation & rapport :**

- `backend/optimization/report.py` : grading A-F, validation Bitget 90j + bootstrap CI
  - `compute_grade()` : score 0-100 → A/B/C/D/F
  - `validate_on_bitget()` : backtest params optimaux sur Bitget, bootstrap CI Sharpe
  - `apply_to_yaml()` : écrit params grade A/B dans strategies.yaml per_asset (backup horodaté)
  - `save_report()` : JSON dans `data/optimization/`

**Phase 4 — CLI :**

- `scripts/optimize.py` : orchestrateur complet
  - `--check-data` : vérifie données disponibles par exchange × symbol
  - `--strategy X --symbol Y` ou `--all` ou `--all-symbols`
  - `--dry-run` : affiche le grid sans exécuter
  - `--apply` : applique les params grade A/B dans strategies.yaml
  - `-v` : résultats détaillés par fenêtre

**Décisions clés Sprint 7 :**

- Deux chemins params : optimisation (params explicites du grid) vs production (`_resolve_param` per_asset)
- scipy retiré (remplacé par `math.erf` pour la CDF normale) — évite MemoryError avec ProcessPoolExecutor
- Fallback séquentiel automatique si multiprocessing crashe sur Windows
- Retour worker léger (pas de trades, juste métriques scalaires) pour minimiser le pickling
- Funding/Liquidation exclus (pas de données historiques OI/funding sur Binance)
- Migration DB avec backup auto horodaté avant altération du schéma

- 330 tests passants (46 nouveaux + 284 existants, 0 régression)

### Sprint 6b — Dashboard UX Overhaul ✅

Complet. Refonte UX du dashboard : positions actives, sidebar redimensionnable, sections collapsibles.

**Fichiers créés :**

- `frontend/src/components/ActivePositions.jsx` — Bannière positions live + paper avec P&L
- `frontend/src/components/CollapsibleCard.jsx` — Wrapper collapsible réutilisable (localStorage)
- `frontend/src/components/ActivityFeed.jsx` — Timeline trades (renommé depuis AlertFeed)
- `frontend/src/components/Tooltip.jsx` — Tooltips explicatifs

**Fichiers modifiés :**

- `simulator.py` : tuples `(symbol, trade)`, `get_all_trades()` enrichi (symbol, tp/sl, exit_reason), `get_open_positions()`
- `websocket_routes.py` : push `simulator_positions` dans WS `/ws/live`
- `App.jsx` : layout refactoré (resize handler, collapsible, ActivePositions bannière)
- `Scanner.jsx` : panneau détail avec barres de conditions
- `TradeHistory.jsx` : colonnes asset, entry→exit, exit_reason, durée
- `SessionStats.jsx` : kill switch red-tinted card
- `styles.css` : ~80 nouvelles classes (active-positions, resize-handle, condition bars)

### Sprint 7b — Funding/OI Historiques + Optimisation ✅

Complet. Données funding rates et open interest historiques, injection dans le moteur de backtest, WFO activé pour funding et liquidation.
Plan détaillé : `docs/plans/sprint-7b-funding-oi-optimization.md`

**Fichiers créés :**

- `scripts/fetch_funding.py` — Fetch historique funding rates (Bitget API)
- `scripts/fetch_oi.py` — Fetch historique open interest (Binance API)
- `backend/backtesting/extra_data_builder.py` — Alignement funding/OI par timestamp (forward-fill)
- `tests/test_funding_oi_data.py` — 23 tests

**Fichiers modifiés :**

- `database.py` : 2 nouvelles tables (funding_rates, open_interest) + 6 méthodes CRUD
- `config.py` : `per_asset` + `get_params_for_symbol()` sur FundingConfig, LiquidationConfig
- `engine.py` : paramètre `extra_data_by_timestamp` dans `run()`
- `walk_forward.py` : chargement extra_data, propagation aux workers/OOS
- `param_grids.yaml` : grids funding (192 combos), liquidation (576 combos)

**Décisions clés :**

- Forward-fill funding rates entre les updates 8h
- OI change % calculé vs snapshot précédent (parité DataEngine)
- Binance OI pour validation (pas d'API historique Bitget)

- 352 tests passants (22 nouveaux)

### Sprint 8 — Backtest Dashboard (planifié, non implémenté)

Plan détaillé : `docs/plans/sprint-8-backtest-dashboard.md`
Sprint planifié mais non exécuté — les composants frontend prévus n'ont pas été créés.

### Sprint 9 — 3 Nouvelles Stratégies 1h ✅

Complet. 3 stratégies haute timeframe (1h) + fast engine + support WFO.
Plan détaillé : `docs/plans/sprint-9-new-1h-strategies.md`

**Fichiers créés :**

- `backend/strategies/bollinger_mr.py` — Bollinger Bands mean reversion (TP dynamique SMA crossing)
- `backend/strategies/donchian_breakout.py` — Donchian channel breakout (TP/SL ATR multiples)
- `backend/strategies/supertrend.py` — SuperTrend direction flips
- `backend/optimization/indicator_cache.py` — Cache numpy pré-calculé (toutes variantes du grid)
- `backend/optimization/fast_backtest.py` — Fast engine mono-position (numpy-only, ~10× plus rapide)
- `tests/test_new_strategies.py` — 48 tests
- `tests/test_fast_backtest.py` — 18 tests

**Fichiers modifiés :**

- `indicators.py` : `bollinger_bands()`, `supertrend()` fonctions pure numpy
- `config.py` : BollingerMRConfig, DonchianBreakoutConfig, SuperTrendConfig + per_asset
- `factory.py` : create/enable pour les 3 nouvelles stratégies
- `walk_forward.py` : `_INDICATOR_PARAMS` pour les 3 stratégies, sélection fast engine automatique
- `param_grids.yaml` : grids + per-strategy WFO configs (IS: 180j, OOS: 60j, step: 60j)

**Décisions clés Sprint 9 :**

- 1h = plus de données → meilleure validation WFO qu'en 5m
- Bollinger TP dynamique (SMA crossing dans `check_exit()`, pas un % fixe)
- SuperTrend pré-calculé (boucle itérative, ~5ms/48k pts)
- ATR multi-period support (Donchian et SuperTrend utilisent `atr_period` variable)
- `live_eligible: false` pour les 3 (paper trading only jusqu'à grade A/B)

- 419 tests passants (89 nouveaux)

### Sprint 10 — Moteur Multi-Position Modulaire ✅

Complet. Infrastructure grid/DCA : BaseGridStrategy, GridPositionManager, MultiPositionEngine, fast engine multi-position.
Plan détaillé : `docs/plans/sprint-10-multi-position-engine.md`

**Fichiers créés :**

- `backend/strategies/base_grid.py` — BaseGridStrategy ABC héritant de BaseStrategy
- `backend/core/grid_position_manager.py` — GridPositionManager pour N positions simultanées
- `backend/backtesting/multi_engine.py` — MultiPositionEngine parallèle au BacktestEngine
- `backend/strategies/envelope_dca.py` — Envelope DCA mean reversion (enveloppes asymétriques, TP retour SMA)
- `backend/optimization/fast_multi_backtest.py` — Fast engine multi-position (numpy-only)
- `scripts/parity_check.py` — Compare moteurs mono vs multi-position
- `tests/test_multi_engine.py` — 32 tests

**Fichiers modifiés :**

- `config.py` : EnvelopeDCAConfig + StrategiesConfig
- `optimization/__init__.py` : GRID_STRATEGIES set, `is_grid_strategy()` helper
- `walk_forward.py` : leverage depuis config stratégie, switch engine auto, per-strategy WFO
- `indicator_cache.py` : SMA par période pour envelope_dca
- `strategies.yaml` : section envelope_dca
- `param_grids.yaml` : grid envelope_dca + WFO config

**Décisions clés Sprint 10 :**

- Deux moteurs indépendants (mono inchangé pour la vitesse)
- BaseGridStrategy hérite BaseStrategy pour compatibilité Arena/Simulator/Dashboard
- Allocation fixe par niveau : `notional = capital/levels × leverage` (pas risk-based)
- TP/SL global (pas par position), SMA dynamic TP
- Un seul côté actif (positions LONG → pas de niveaux SHORT)
- Enveloppes asymétriques : `upper = 1/(1-lower) - 1` (aller-retour cohérent)
- Leverage 6 depuis config (pas le défaut 15)

- 451 tests passants (32 nouveaux)

### Sprint 11 — Paper Trading Grid/DCA (Envelope DCA) ✅

Complet. Envelope DCA en paper trading live (Simulator). Executor ajouté en Sprint 12.
Plan détaillé : `docs/plans/sprint-11-paper-trading-grid.md`

**Fichiers créés :**

- `tests/test_grid_runner.py` — 28 tests

**Fichiers modifiés :**

- `simulator.py` : classe `GridStrategyRunner` (~250 lignes), détection dans `start()`, support `get_open_positions()`
- `state_manager.py` : sérialisation/restauration grid_positions dans l'état runner
- `database.py` : `get_recent_candles()` pour le warm-up
- `server.py` : passage `db` au Simulator
- `strategies.yaml` : `envelope_dca` seul enabled, stratégies Grade F désactivées

**Décisions clés Sprint 11 :**

- GridStrategyRunner parallèle à LiveStrategyRunner (même duck-type interface)
- Buffer SMA interne + merge dans indicators dict (pas de modification d'IncrementalIndicatorEngine)
- Warm-up depuis DB (N candles injectées dans le runner AVANT on_candle)
- TradeEvent format pour opens/closes grid (prêt pour futur Executor)
- Détection régime via ADX/ATR si disponible, sinon RANGING (DCA non filtré)

- 484 tests passants (28 nouveaux + hotfix Monte Carlo)

### Hotfix — Monte Carlo underpowered detection ✅

Fix détection underpowered dans Monte Carlo (pénalisait incorrectement les stratégies à faible nombre de trades comme envelope_dca).

- Seuil underpowered < 30 trades → retourne p_value=0.50 (score neutre 12/25 pts)
- ASCII fallback pour console Windows (cp1252 compat)
- Impact : BTC envelope_dca Grade D → B

### Sprint 12 — Executor Grid DCA + Alertes Telegram ✅

Complet. L'Executor (exécution live Bitget) supporte maintenant les cycles grid/DCA multi-niveaux (envelope_dca).
Plan détaillé : `docs/plans/sprint-12-executor-grid-dca.md`

**Fichiers créés :**

- `tests/test_executor_grid.py` — 25 tests

**Fichiers modifiés :**

- `executor.py` : +2 dataclasses (`GridLivePosition`, `GridLiveState`), ~12 méthodes grid (~350 lignes)
- `risk_manager.py` : +param `leverage_override` dans `pre_trade_check`
- `adaptive_selector.py` : +4 mappings stratégies 1h dans `_STRATEGY_CONFIG_ATTR`
- `notifier.py` : +2 méthodes grid (`notify_grid_level_opened`, `notify_grid_cycle_closed`)
- `telegram.py` : +2 méthodes format messages grid
- `strategies.yaml` : `envelope_dca.live_eligible: true`
- `test_risk_manager.py` : +3 tests `leverage_override`
- `test_telegram.py` : +3 tests format messages grid

**8 bugs corrigés vs plan original :**

1. AdaptiveSelector bloquait envelope_dca (mapping manquant + `live_eligible: false`)
2. RiskManager rejetait le 2ème niveau grid (`position_already_open`)
3. `record_pnl()` n'existait pas (utiliser `record_trade_result(LiveTradeResult(...))`)
4. `_watch_orders_loop` dormait sans positions mono (condition inclut `_grid_states`)
5. `_poll_positions_loop` ignorait les grids (itérer aussi `_grid_states`)
6. `_cancel_orphan_orders` supprimait le SL grid (inclure grid IDs dans `tracked_ids`)
7. Leverage 15 au lieu de 6 dans le margin check (`leverage_override` param)
8. Conflit mono/grid sur le même symbol (exclusion mutuelle bidirectionnelle)

**Décisions clés Sprint 12 :**

- Dispatch grid vs mono dans `handle_event()` via `_is_grid_strategy()`
- Pre-trade check au 1er niveau seulement (un cycle = 1 slot pour max_concurrent)
- SL global server-side recalculé à chaque niveau (cancel ancien + place nouveau)
- TP client-side (SMA dynamique, détecté par GridStrategyRunner)
- Règle #1 : JAMAIS de position sans SL → `_emergency_close_grid()` si SL impossible
- Exclusion mutuelle mono/grid par symbol (Bitget agrège positions par symbol+direction)
- State persistence round-trip (`grid_states` dans `get_state_for_persistence/restore_positions`)
- Réconciliation grid au boot (`_reconcile_grid_symbol`)

- 513 tests passants (29 nouveaux)

## Dev Workflow

```
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

## References
- Bitget API docs: https://www.bitget.com/api-doc/
- ccxt Bitget: https://docs.ccxt.com/#/exchanges/bitget
- Frontend prototype: exists as React JSX with signal scoring, heatmap, risk calc, alert feed (built during design phase)
- Frontend prototype: docs/prototypes/dashboard-prototype.jsx (React component with signal scoring, 
  heatmap, risk calc, alert feed — use as visual/UX reference for Sprint 3)