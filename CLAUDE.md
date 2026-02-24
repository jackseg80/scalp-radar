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

- **[ROADMAP.md](docs/ROADMAP.md)** — Roadmap Phases 1-7, sprints détaillés, **décompte de tests (fait autorité)**
  - Mettre à jour après chaque sprint : résultats, bugs, nombre de tests, prochaine étape
- **[docs/plans/](docs/plans/)** — Plans de sprint archivés (1 fichier par sprint)
  - Copier le plan dans `docs/plans/sprint-{n}-{nom}.md` à chaque fin de sprint
- **[docs/audit/](docs/audit/)** — Rapports d'audit (`audit-{sujet}-{YYYYMMDD}.md`)
- **[STRATEGIES.md](docs/STRATEGIES.md)** — Guide complet des 16 stratégies
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Architecture runtime, flux de données, boot/shutdown
- **[COMMANDS.md](COMMANDS.md)** — Toutes les commandes CLI — **consulter avant de proposer des commandes**
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** — Dépannage production

## Developer Profile

- Experienced crypto trader (swing trading, futures x10-x30 on Bitget)
- Language: French — comments and UI can be in French
- **Platform: Windows** — code doit fonctionner sur Windows (paths, CRLF pour .bat)
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
| Testing            | pytest            | Critical components must have unit tests                |

## Key Architecture Principles

- **pyproject.toml à la racine** (pas dans backend/) → imports propres `from backend.core.models`
- **Process unique** : DataEngine intégré dans le lifespan FastAPI (pas de process séparé)
- **100% async** : database, data engine, scripts CLI utilisent `asyncio.run()`
- **Buffer rolling borné** : max 500 bougies par (symbol, timeframe) en mémoire
- **Rate limiter par catégorie** : market_data, trade, account, position (token bucket)
- **SL réaliste** : coût SL inclut distance + taker_fee + slippage (configurable)
- **Groupes de corrélation** : limite l'exposition sur assets corrélés

## Project Structure

```text
scalp-radar/
├── config/            # YAML configs (assets, strategies, risk, exchanges, param_grids)
├── backend/
│   ├── core/          # models, config, database, indicators, state_manager, data_engine
│   ├── strategies/    # base, base_grid, factory + 16 stratégies
│   ├── optimization/  # walk_forward, overfitting, report, indicator_cache, fast_backtest
│   ├── backtesting/   # engine, multi_engine, simulator, arena, portfolio_engine
│   ├── execution/     # executor, executor_manager, risk_manager, adaptive_selector
│   ├── api/           # server, routes (simulator, conditions, arena, executor, websocket, portfolio)
│   ├── alerts/        # telegram, notifier, heartbeat
│   └── monitoring/    # watchdog
├── frontend/          # React + Vite (32 components)
├── tests/             # pytest (~1807 tests — voir ROADMAP.md)
├── scripts/           # backfill, fetch_history, optimize, portfolio_backtest, stress_test_leverage
├── docs/plans/        # Sprint plans archivés
└── docs/audit/        # Rapports d'audit
```

## Trading Strategies (17 implémentées)

**5m Scalp (4)** : vwap_rsi, momentum, funding (paper), liquidation (paper)

**1h Swing (4)** : bollinger_mr, donchian_breakout, supertrend, boltrend (`enabled: false`)

**1h Grid/DCA (9)** :
- `grid_atr` — Enveloppes ATR adaptatives (LIVE 7x, 14 assets)
- `grid_multi_tf` — Supertrend 4h + Grid ATR 1h (LIVE 3x, 14 assets)
- `grid_boltrend` — DCA Bollinger breakout + SMA, TP inversé (paper, 2 assets, mise en pause Sprint 38b)
- `grid_momentum` — Donchian breakout + DCA pullback (`enabled: false`, WFO à lancer)
- `grid_range_atr` — Bidirectionnel LONG+SHORT (`enabled: false`)
- `grid_funding` — DCA sur funding négatif (`enabled: false`)
- `grid_trend` — EMA cross + ADX + trailing stop ATR (`enabled: false`, échoue bear market)
- `envelope_dca` / `envelope_dca_short` — (`enabled: false`, remplacés par grid_atr)

Détails complets : voir **[STRATEGIES.md](docs/STRATEGIES.md)** | Workflow WFO : voir **[WORKFLOW_WFO.md](docs/WORKFLOW_WFO.md)**

## Critical Design Requirements

- **Kill switch** : auto-stop à 45% drawdown (global + per-runner), persisté across restarts
- **Margin accounting** : `margin = notional / leverage` déduit à l'ouverture, restitué à fermeture
- **Entry fees** : comptabilisées à la clôture via `net_pnl` uniquement (jamais à l'ouverture)
- **Position sizing** : `capital / nb_assets / levels`, margin guard 70% (`max_margin_ratio`)
- **Fees** : Bitget taker 0.06%, maker 0.02% — toujours net de fees
- **Règle #1** : JAMAIS de position sans SL (retry 2x → emergency close)
- **Multi-Executor** : un Executor par stratégie live, `ExecutorManager` agrège via duck typing
- **Sync WFO** : push auto local → serveur après chaque run, best-effort

## Config Files (5 YAML)

- `assets.yaml` — 21 assets, timeframes, groupes corrélation
- `strategies.yaml` — 16 stratégies + per_asset overrides
- `risk.yaml` — kill switch, sizing, fees, slippage, max_margin_ratio
- `exchanges.yaml` — Bitget WebSocket, rate limits
- `param_grids.yaml` — espaces de recherche WFO + config per-strategy

## Lifespan Complet (server.py)

**Startup :** DB → Telegram → DataEngine → StateManager → Simulator.start() → Arena → AdaptiveSelector → ExecutorManager → Watchdog → Heartbeat

**Shutdown :** notify → heartbeat → watchdog → selector → executor save+stop → state_manager → simulator → data_engine → db

## Dev Workflow

```text
Windows (VSCode) → git push → SSH + deploy.sh → docker compose up -d → curl /health
```

- `deploy.sh --clean` : fresh start sans perdre la DB
- **JAMAIS éditer `config/*.yaml` sur le serveur** — overrides via `.env` uniquement
- Voir **[COMMANDS.md](COMMANDS.md)** et **`.claude/rules/deployment.md`** pour détails

## References

- Bitget API docs: <https://www.bitget.com/api-doc/>
- ccxt Bitget: <https://docs.ccxt.com/#/exchanges/bitget>
- Frontend prototype: `docs/prototypes/Scalp radar v2.jsx`
- Plans détaillés : `docs/plans/sprint-{n}-*.md`
- Pièges résolus : `.claude/rules/` (git, testing, grid-strategy, deployment, wfo-optimization)
