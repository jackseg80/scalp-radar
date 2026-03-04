# Scalp Radar

Multi-strategy automated trading platform for crypto futures (optimized for Bitget).
Features 17 strategies (Scalp 5m, Swing 1h, and Grid/DCA 1h), automatic Walk-Forward Optimization (WFO), real-time Paper Trading, Live Mainnet Executor, and a comprehensive React dashboard.

## Key Features

- **17 Strategies Implemented**: 4 Scalp (5m), 4 Swing (1h), and 9 Grid/DCA (1h).
- **Advanced Validation**: Walk-Forward Optimization (WFO) with anti-overfitting metrics (Monte Carlo, DSR, Stability).
- **Real-time Performance**: High-performance 100% async Python backend (FastAPI) + React/Vite frontend.
- **Production Ready**: Full Docker Compose deployment, Telegram alerts, Watchdog monitoring, and automatic crash recovery.
- **Risk Management**: Global Kill Switch (45% drawdown), Margin Guard (70%), and realistic SL/Fee accounting.

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
**~1840 tests** covering models, indicators, strategies, multi-position engines, simulator, arena, live executor, and WFO optimization.

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
config/              # YAML parameters (assets, strategies, risk, exchanges)
backend/core/        # DataEngine, Indicators, StateManager, RiskManager
backend/strategies/  # 17 strategies (GridATR, GridMultiTF, BolTrend, etc.)
backend/optimization/# WFO, Overfitting detection, Fast Engines (JIT)
backend/backtesting/ # Engines (Mono, Multi, Portfolio), Simulator, Arena
backend/execution/   # Live Executor (Bitget), Adaptive Selector, Sync
backend/api/         # FastAPI endpoints & WebSockets
scripts/             # CLI tools for optimization, history, and deployment
frontend/src/        # React Dashboard (Scanner, Heatmap, Portfolio, Research)
```

## Strategy Overview (17 Total)

- **Grid/DCA 1h (9)**: `grid_atr` (Main), `grid_multi_tf`, `grid_boltrend`, `grid_funding`, `grid_trend`, `envelope_dca`, etc.
- **Swing 1h (4)**: `bollinger_mr`, `donchian_breakout`, `supertrend`, `boltrend`.
- **Scalp 5m (4)**: `vwap_rsi`, `momentum`, `funding`, `liquidation`.

## Roadmap Progress

- [x] **Phase 1-4**: Infrastructure, Backtesting, Production (Docker, Telegram), Live Trading.
- [x] **Phase 5**: Scaling (Adaptive Selector, Multi-Asset Live, Grid ATR).
- [x] **Phase 6**: Optimization & Visualization (WFO Dashboard, Heatmaps, Diagnostic Panel).
- [x] **Phase 7**: Portfolio Backtesting (Multi-Asset correlation, Liquidation simulation).
- [x] **Sprint 30-38**: Multi-Timeframe WFO, JIT Numba Optimization (5-10x speedup), BolTrend Strategy, Log Viewer, Realistic P&L Tracking.

**Last Updated:** 2026-03-04 (Sprint 38b)
