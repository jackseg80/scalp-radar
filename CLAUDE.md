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

- **[ROADMAP.md](docs/ROADMAP.md)** — Roadmap Phases 1-9, detailed sprints, **2233 tests (authoritative)**
  - Update after each sprint: results, bugs, test count, next step
- **[docs/plans/](docs/plans/)** — Archived sprint plans (1 file per sprint)
  - Copy plan to `docs/plans/sprint-{n}-{name}.md` at the end of each sprint
- **[docs/audit/](docs/audit/)** — Audit reports (`audit-{subject}-{YYYYMMDD}.md`)
- **[STRATEGIES.md](docs/STRATEGIES.md)** — Full guide for the 18 strategies
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Runtime architecture, data flow, boot/shutdown
- **[COMMANDS.md](COMMANDS.md)** — All CLI commands — **consult before proposing commands**
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** — Production troubleshooting

## Developer Profile

- Experienced crypto trader (swing trading, futures x10-x30 on Bitget)
- **Communication: French**, **Documentation/Code: English** (Comments/UI can be French per user preference)
- **Platform: Windows** — code must work on Windows (paths, CRLF for .bat)
- **Git: NEVER add Co-Authored-By in commits**

## Architecture Decisions

| Decision           | Choice            | Reason                                                  |
|--------------------|-------------------|---------------------------------------------------------|
| Backend language   | Python 3.12+      | Fast enough for 1-5min scalping, best trading ecosystem |
| JIT Compiler       | Numba             | JIT-accelerated simulation loops (5-10x speedup)        |
| Package manager    | uv                | Fast pip replacement, uses standard .venv               |
| API framework      | FastAPI           | Async native, WebSocket support, Pydantic               |
| Database           | SQLite → TimescaleDB | SQLite for dev/early prod, TimescaleDB when volume grows |
| Exchange           | Bitget (primary)  | Already has bots there, good API, good futures fees     |
| Exchange lib       | ccxt              | Unified API, easy to add Binance/Kraken later           |
| Frontend           | React + Vite      | Real-time dashboard via WebSocket                       |
| Dev environment    | Windows/VSCode    | No Docker in dev — just uvicorn + vite dev              |
| Production         | Docker Compose    | On Linux server 192.168.1.200, bot runs 24/7            |
| Config format      | YAML              | Editable without code changes or redeployment           |
| Testing            | pytest            | Critical components must have unit tests (2231 tests)   |

## Key Architecture Principles

- **pyproject.toml at root** (not in backend/) → clean imports `from backend.core.models`
- **Single process**: DataEngine integrated into FastAPI lifespan (no separate process)
- **100% async**: database, data engine, CLI scripts use `asyncio.run()`
- **Multi-Timeframe**: Native support for 1h, 4h, 1d resampling in WFO and Simulator
- **Bounded rolling buffer**: max 500 candles per (symbol, timeframe) in memory
- **Rate limiter by category**: market_data, trade, account, position (token bucket)
- **Realistic SL**: SL cost includes distance + taker_fee + slippage (configurable)
- **Correlation groups**: limits exposure on correlated assets

## Project Structure

```text
scalp-radar/
├── config/            # YAML configs (28 assets, 18 strategies, risk, exchanges, param_grids)
├── backend/
│   ├── core/          # models, config, database, indicators, state_manager, data_engine
│   ├── strategies/    # base, base_grid, factory + 18 strategies
│   ├── optimization/  # walk_forward, overfitting, report, indicator_cache, fast_backtest
│   ├── backtesting/   # engine, multi_engine, simulator, arena, portfolio_engine
│   ├── execution/     # executor, executor_manager, risk_manager, adaptive_selector
│   ├── api/           # server, routes (simulator, conditions, arena, executor, websocket, portfolio)
│   ├── alerts/        # telegram, notifier, heartbeat
│   └── monitoring/    # watchdog
├── frontend/          # React + Vite (48 components)
├── tests/             # pytest (2231 tests — see ROADMAP.md)
├── scripts/           # backfill, fetch_history, optimize, portfolio_backtest, stress_test_leverage
├── docs/plans/        # Archived sprint plans
└── docs/audit/        # Audit reports
```

## Trading Strategies (18 implemented)

**Scalp/Swing (9)**: bollinger_mr, donchian_breakout, supertrend, boltrend, vwap_rsi, momentum, funding (paper), liquidation (paper), **trend_follow_daily (1d, new)**.

**1h Grid/DCA (9)**:
- `grid_atr` — Adaptive ATR envelopes (LIVE 7x, 14 assets)
- `grid_multi_tf` — Supertrend 4h + Grid ATR 1h (LIVE 3x, 14 assets)
- `grid_boltrend` — DCA Bollinger breakout + SMA, inverse TP (paper, 2 assets, paused Sprint 38b)
- `grid_momentum` — Donchian breakout + DCA pullback (`enabled: false`, ABANDONED)
- `grid_range_atr` — Bidirectional LONG+SHORT (`enabled: false`)
- `grid_funding` — DCA on negative funding (`enabled: false`, ABANDONED)
- `grid_trend` — EMA cross + ADX + ATR trailing stop (`enabled: false`, ABANDONED)
- `envelope_dca` / `envelope_dca_short` — (`enabled: false`, replaced by grid_atr)

Full details: see **[STRATEGIES.md](docs/STRATEGIES.md)** | WFO Workflow: see **[WORKFLOW_WFO.md](docs/WORKFLOW_WFO.md)**

## Critical Design Requirements

- **Kill switch**: auto-stop at 45% drawdown (global + per-runner), persisted across restarts
- **Margin accounting**: `margin = notional / leverage` deducted at open, returned at close
- **Entry fees**: accounted for at close via `net_pnl` only (never at open)
- **Position sizing**: `capital / nb_assets / levels`, margin guard 70% (`max_margin_ratio`)
- **Fees**: Bitget taker 0.06%, maker 0.02% — always net of fees
- **Rule #1**: NEVER a position without SL (retry 2x → emergency close)
- **Multi-Executor**: one Executor per live strategy, `ExecutorManager` aggregates via duck typing
- **Audit Hardening**: async locks on critical sections, anti double-instance guards, crash recovery logic.

## Config Files (5 YAML)

- `assets.yaml` — 28 assets, timeframes, correlation groups
- `strategies.yaml` — 18 strategies + per_asset overrides
- `risk.yaml` — kill switch, sizing, fees, slippage, max_margin_ratio
- `exchanges.yaml` — Bitget WebSocket, rate limits
- `param_grids.yaml` — WFO search spaces + per-strategy config

## Complete Lifespan (server.py)

**Startup:** DB → Telegram → DataEngine → StateManager → Simulator.start() → Arena → AdaptiveSelector → ExecutorManager → Watchdog → Heartbeat

**Shutdown:** notify → heartbeat → watchdog → selector → executor save+stop → state_manager → simulator → data_engine → db

## Dev Workflow

```text
Windows (VSCode) → git push → SSH + deploy.sh → docker compose up -d → curl /health
```

- `deploy.sh --clean`: fresh start without losing the DB
- **NEVER edit `config/*.yaml` on the server** — overrides via `.env` only
- See **[COMMANDS.md](COMMANDS.md)** and **`.claude/rules/deployment.md`** for details

## References

- Bitget API docs: <https://www.bitget.com/api-doc/>
- ccxt Bitget: <https://docs.ccxt.com/#/exchanges/bitget>
- Frontend prototype: `docs/prototypes/Scalp radar v2.jsx`
- Detailed plans: `docs/plans/sprint-{n}-*.md`
- Resolved pitfalls: `.claude/rules/` (git, testing, grid-strategy, deployment, wfo-optimization)
