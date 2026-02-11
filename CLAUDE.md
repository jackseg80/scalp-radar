# CLAUDE.md — Scalp Radar Project Brief

## Project Overview

Scalp Radar is a multi-strategy automated scalping tool for crypto futures.
It detects trading opportunities, scores them, runs strategies in parallel (simulation then live),
and presents results via a real-time dashboard.

## Repository

- GitHub: https://github.com/jackseg80/scalp-radar.git
- Local dev: D:\Python\scalp-radar (Windows + VSCode)
- Production: Linux server at 192.168.1.200, deployed via Docker Compose

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
├── dev.bat                       # Windows: lance backend + frontend ensemble
│
├── config/                       # ALL tunable parameters — no hardcoding in code
│   ├── assets.yaml               # Traded pairs + correlation groups
│   ├── strategies.yaml           # Strategy parameters (5 stratégies + custom)
│   ├── risk.yaml                 # Kill switch, position sizing, fees, slippage, margin, SL/TP
│   └── exchanges.yaml            # Bitget: WebSocket, rate limits par catégorie, API config
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
│   │   ├── indicators.py          # RSI, VWAP, ADX+DI, ATR, SMA, EMA, régime (pur numpy)
│   │   ├── incremental_indicators.py # Buffers rolling pour indicateurs live (Simulator)
│   │   ├── position_manager.py   # Sizing, fees, slippage, TP/SL (réutilisé par engine + simulator)
│   │   ├── data_engine.py        # ccxt Pro WebSocket + polling OI/funding + buffer rolling
│   │   ├── rate_limiter.py       # Token bucket par catégorie d'endpoint
│   │   └── logging_setup.py      # loguru: console + fichier JSON + fichier erreurs
│   │
│   ├── strategies/
│   │   ├── base.py               # BaseStrategy ABC, StrategyContext, StrategySignal, EXTRA_* constants
│   │   ├── factory.py            # create_strategy() + get_enabled_strategies()
│   │   ├── vwap_rsi.py           # VWAP+RSI mean reversion (RANGING)
│   │   ├── momentum.py           # Momentum Breakout (TRENDING)
│   │   ├── funding.py            # Funding Rate Arbitrage (paper trading only)
│   │   └── liquidation.py        # Liquidation Zone Hunting (paper trading only)
│   ├── backtesting/
│   │   ├── engine.py             # Moteur event-driven (délègue au PositionManager)
│   │   ├── metrics.py            # BacktestMetrics + format table console
│   │   ├── simulator.py          # LiveStrategyRunner + Simulator (paper trading live)
│   │   └── arena.py              # StrategyArena (classement parallèle)
│   ├── execution/                # (Sprint 5)
│   │
│   ├── api/
│   │   ├── server.py             # FastAPI + lifespan (DataEngine + Simulator + Arena)
│   │   ├── health.py             # GET /health → status, data_engine, database, uptime
│   │   ├── simulator_routes.py   # GET /api/simulator/* (status, positions, trades, performance)
│   │   ├── arena_routes.py       # GET /api/arena/* (ranking, strategy detail)
│   │   ├── signals_routes.py     # GET /api/signals/recent
│   │   └── websocket_routes.py   # WS /ws/live (push temps réel)
│   │
│   ├── alerts/                   # (Sprint 4)
│   └── monitoring/               # (Sprint 4)
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
│           ├── Header.jsx        # Logo, status indicators (Engine/DB/WS)
│           ├── ArenaRanking.jsx  # Table classement stratégies (poll 30s)
│           ├── SignalFeed.jsx    # Stratégies live via WebSocket
│           ├── SessionStats.jsx  # Sidebar PnL, trades, win rate (poll 3s)
│           └── TradeHistory.jsx  # Trades récents (poll 10s)
│
├── tests/
│   ├── conftest.py               # Fixtures partagées (config_dir temporaire)
│   ├── test_models.py            # 17 tests : enums, Candle, OrderBook, Trade, Signal, SessionState
│   ├── test_config.py            # 11 tests : chargement, validation, erreurs
│   ├── test_database.py          # 12 tests : CRUD candles, session state, signals, trades (async)
│   ├── test_indicators.py        # 22 tests : RSI, VWAP, ADX, ATR, SMA, EMA, régime
│   ├── test_strategy_vwap_rsi.py # 11 tests : signaux, filtres, check_exit, compute_indicators
│   ├── test_backtesting.py       # 17 tests : engine, métriques, OHLC, sizing, Sortino
│   ├── test_position_manager.py  # 10 tests : sizing, fees, slippage, TP/SL, heuristique
│   ├── test_incremental_indicators.py # 5 tests : buffer rolling, trim, indicateurs
│   ├── test_strategy_momentum.py # 10 tests : breakout, volume, trend filter, ADX exit
│   ├── test_strategy_funding.py  # 7 tests : extreme rates, delay, neutral exit
│   ├── test_strategy_liquidation.py # 7 tests : zones, OI threshold, proximity score
│   ├── test_simulator.py         # 14 tests : runner, on_candle, kill switch, regime change
│   ├── test_arena.py             # 9 tests : ranking, profit factor, drawdown, detail
│   └── test_api_simulator.py     # 7 tests : endpoints status, trades, ranking, signals
│
├── scripts/
│   ├── fetch_history.py          # Backfill async ccxt REST + tqdm (6 mois, reprise auto)
│   ├── run_backtest.py           # CLI backtest runner (--symbol, --days, --json)
│   └── __main__.py               # python -m scripts support
│
├── data/                         # SQLite DB + données (gitignored)
│
└── docs/
    ├── plans/
    │   └── sprint-2-backtesting.md  # Plan détaillé Sprint 2 (décisions arch, corrections)
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

- `assets.yaml` — 3 assets (BTC, ETH, SOL), timeframes [1m, 5m, 15m, 1h], groupes de corrélation
- `strategies.yaml` — 5 stratégies scalping + section custom_strategies (swing baseline)
- `risk.yaml` — kill switch, position sizing, fees, slippage, margin cross, SL/TP server-side
- `exchanges.yaml` — Bitget: WebSocket, rate limits par catégorie, API config (USDT-M, mark_price)

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

### Sprint 4 — Production

- docker-compose.yml, Dockerfiles, deploy.sh
- State manager (crash recovery)
- monitoring/watchdog.py
- alerts/telegram.py + alerts/heartbeat.py
- First deployment on 192.168.1.200

### Sprint 5 — Live Trading

- execution/executor.py — real order execution
- execution/risk_manager.py — kill switch, position sizing
- Adaptive strategy selector
- Progressive rollout: small capital first, monitor closely

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