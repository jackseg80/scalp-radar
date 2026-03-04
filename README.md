# Scalp Radar

Multi-strategy automated trading platform for crypto futures (optimized for Bitget).
Features 18 strategies (Scalping 5m, Swing 1h, Grid/DCA 1h, and Trend Daily), automatic Walk-Forward Optimization (WFO), real-time Paper Trading, Live Mainnet Executor, and a comprehensive React dashboard.

## Key Features

- **18 Strategies Implemented**: 4 Scalp (5m), 5 Swing/Trend (1h/1d), and 9 Grid/DCA (1h).
- **Advanced Validation**: Walk-Forward Optimization (WFO) with anti-overfitting metrics (Monte Carlo, DSR, Stability).
- **Real-time Performance**: High-performance 100% async Python backend (FastAPI) + React/Vite frontend.
- **JIT Acceleration**: Numba-powered simulation loops for 5-10x faster optimization.
- **Production Ready**: Full Docker Compose deployment, Telegram alerts, Watchdog monitoring, and automatic crash recovery.
- **Risk Management**: Global Kill Switch (45% drawdown), Margin Guard (70%), and Regime-based entry filters.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Fast Python package manager)
- Node.js 18+ (for the frontend)

## Installation

```bash
# Clone the repository
git clone https://github.com/jackseg80/scalp-radar.git
cd scalp-radar

# Sync Python dependencies
uv sync

# Configure environment variables
cp .env.example .env
# Edit .env with your Bitget API keys and preferences

# Install frontend dependencies
cd frontend && npm install && cd ..
```

## Development (Windows)

```bash
# All-in-one: backend (port 8000) + frontend (port 5173)
dev.bat

# Or separately:
uv run uvicorn backend.api.server:app --reload --port 8000
cd frontend && npm run dev
```

To disable WebSocket in dev (prevents reconnect spam during --reload):
```bash
# In .env
ENABLE_WEBSOCKET=false
```

## Data Management

```bash
# Backfill candles from Binance (Public API, no key needed)
uv run python -m scripts.backfill_candles --symbol BTC/USDT --since 2023-01-01 --timeframe 1h

# Fetch recent history via CCXT (Bitget or Binance)
uv run python -m scripts.fetch_history --symbol BTC/USDT --timeframe 5m --days 7

# Historical Funding Rates & Open Interest
uv run python -m scripts.fetch_funding
uv run python -m scripts.fetch_oi
```

## Testing

```bash
uv run pytest tests/ -v
```
**2231 tests** covering models, indicators, strategies, multi-position engines, simulator, arena, live executor, and WFO optimization.

## Tech Stack

| Component | Technology |
| --- | --- |
| **Backend** | Python 3.12+, FastAPI, CCXT Pro, Numba (JIT) |
| **Database** | SQLite (dev) / TimescaleDB (prod) |
| **Frontend** | React 19, Vite 6, Tailwind CSS |
| **Optimization** | Walk-Forward, Monte Carlo, Deflated Sharpe Ratio (DSR) |
| **Deployment** | Docker Compose, Nginx, Linux/Windows compatible |

## Project Structure

```text
config/              # YAML parameters (28 assets, 18 strategies, risk, exchanges)
backend/core/        # DataEngine, Indicators, StateManager, RiskManager
backend/strategies/  # 18 strategies (GridATR, GridMultiTF, BolTrend, TrendDaily, etc.)
backend/optimization/# WFO, Overfitting detection, Fast Engines (JIT)
backend/backtesting/ # Engines (Mono, Multi, Portfolio), Simulator, Arena
backend/execution/   # Live Executor (Bitget), Adaptive Selector, Sync
backend/api/         # FastAPI endpoints & WebSockets
scripts/             # CLI tools for optimization, history, and deployment
frontend/src/        # React Dashboard (Scanner, Heatmap, Portfolio, Research)
```

## Strategy Overview (18 Total)

- **Grid/DCA 1h (9)**: `grid_atr` (Main LIVE), `grid_multi_tf` (LIVE), `grid_boltrend`, `grid_funding`, `grid_trend`, etc.
- **Swing/Trend (5)**: `bollinger_mr`, `donchian_breakout`, `supertrend`, `boltrend`, `trend_follow_daily` (1d).
- **Scalp 5m (4)**: `vwap_rsi`, `momentum`, `funding`, `liquidation`.

## Roadmap Progress

- [x] **Phase 1-4**: Infrastructure, Backtesting, Production (Docker, Telegram), Live Trading.
- [x] **Phase 5**: Scaling (Adaptive Selector, Multi-Asset Live, Grid ATR).
- [x] **Phase 6**: Optimization & Visualization (WFO Dashboard, Heatmaps, Diagnostic Panel).
- [x] **Phase 7**: Portfolio Backtesting (Multi-Asset correlation, Liquidation simulation).
- [x] **Phase 8 — Audit & Hardening**: Race conditions, Crash recovery, Multi-executor guard, Async I/O (Sprints 50-60).
- [x] **Phase 9 — Operational Excellence**: Regime Monitor, Dashboard Overview, Enhanced Equity Charts, Telegram History (Sprints 61-63b).

**Last Updated:** 2026-03-04 (Sprint 63b)
