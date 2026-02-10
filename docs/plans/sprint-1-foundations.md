# Plan Sprint 1 — Fondations

> Plan reconstruit a posteriori depuis le code implemente.
> Sprint 1 livre dans le commit `194204a`.

## Contexte

Premier sprint du projet Scalp Radar. Objectif : poser l'infrastructure complete
(modeles, database, config, data engine, API, tests) pour supporter les sprints
suivants (backtesting, strategies, live trading).

## Decisions architecturales cles

### 1. pyproject.toml a la racine (pas dans backend/)

- Imports propres : `from backend.core.models import Candle`
- Un seul package Python, pas de confusion de paths

### 2. Process unique (DataEngine dans le lifespan FastAPI)

- Pas de process separe pour le data engine
- Le lifespan FastAPI gere le cycle de vie : start engine au demarrage, stop au shutdown
- Simplifie le deploiement et la communication inter-composants

### 3. 100% async

- Database (aiosqlite), data engine (ccxt Pro WebSocket), scripts CLI utilisent `asyncio.run()`
- Pas de wrappers sync — tout est nativement async

### 4. Flag ENABLE_WEBSOCKET

- Variable d'env (defaut `true`)
- Permet de desactiver le DataEngine en dev pour eviter les reconnexions WebSocket lors du `--reload` d'uvicorn

### 5. Buffer rolling borne

- Max 500 bougies par (symbol, timeframe) en memoire
- Evite la croissance memoire infinie sur un bot 24/7

### 6. Rate limiter par categorie

- 4 categories : market_data, trade, account, position
- Token bucket : pas de drop de requetes, queuing seulement
- Partage par tous les composants (strategies, execution, data engine)

### 7. Mark price vs last price

- Distinction dans les modeles (Candle.mark_price, TickerData.mark_price)
- Necessaire pour les calculs de liquidation en futures

### 8. SL realiste dans la config

- `risk.yaml` definit le cout SL = distance + taker_fee + slippage
- Configurable via `sl_real_cost_includes: [distance, taker_fee, slippage]`

### 9. Groupes de correlation

- Limite l'exposition sur assets correles (BTC, ETH, SOL = crypto_major)
- `max_concurrent_same_direction: 2`, `max_exposure_percent: 60`

### 10. YAML comme source de verite

- 4 fichiers de config (`assets.yaml`, `strategies.yaml`, `risk.yaml`, `exchanges.yaml`)
- Tous les parametres tunables dans YAML, zero hardcoding dans le code
- Validation Pydantic au chargement

---

## Fichiers implementes

### backend/core/models.py (~350 lignes)

17 modeles Pydantic + enums pour le domaine trading.

**Enums :**
- `Direction` (LONG, SHORT)
- `OrderType` (MARKET, LIMIT)
- `OrderSide` (BUY, SELL)
- `OrderStatus` (PENDING, OPEN, FILLED, PARTIALLY_FILLED, CANCELLED)
- `SignalStrength` (STRONG, MODERATE, WEAK)
- `MarketRegime` (TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY, LOW_VOLATILITY)
- `TimeFrame` (M1, M5, M15, H1) avec `from_string()` et `to_milliseconds()`

**Modeles de donnees :**
- `Candle` — OHLCV + vwap + mark_price, validation stricte (low <= min(open, close), etc.)
- `OrderBookLevel`, `OrderBookSnapshot` — L2 bids/asks avec mid_price & spread
- `TickerData` — last_price, mark_price, index_price, funding_rate, open_interest

**Modeles de trading :**
- `Signal` — direction, strength, score (0-1), entry/tp/sl, regime, signals_detail
- `Order` — id, symbol, side, type, price, quantity, status, sl/tp IDs, fees
- `Position` — entry_price, quantity, leverage, margin (initial + maintenance), unrealized_pnl
- `Trade` — position fermee avec gross_pnl, fee_cost, slippage_cost, net_pnl (valide)

**Etat :**
- `SessionState` — P&L session, trades, wins/losses, drawdown, kill_switch_triggered
- `MultiTimeframeData` — dict[timeframe -> Candle[]], orderbook, ticker

### backend/core/config.py (~375 lignes)

Chargeur YAML avec validation Pydantic croisee.

- `AssetConfig` — symbol, exchange, type, timeframes, max_leverage, tick_size, correlation_group
- `VwapRsiConfig` — RSI period/thresholds, VWAP deviation, TP/SL %, volume spike multiplier
- 4 autres configs de strategies (Liquidation, OrderFlow, Momentum, Funding)
- `RiskConfig` — kill_switch, position sizing, fees (maker/taker), slippage, margin, SL/TP
- `ExchangeConfig` — Bitget WebSocket, rate limits par categorie, API config
- `SecretsConfig` (pydantic-settings) — .env (API keys, database_url, flags)
- `AppConfig` — agregation + validation croisee (leverage coherent, correlation groups valides)
- `get_config()` — singleton avec option `force_reload`

### backend/core/database.py (~310 lignes)

Abstraction SQLite async (aiosqlite, WAL mode).

**4 tables :**
- `candles` (PK: symbol, timeframe, timestamp) — OHLCV + vwap + mark_price
- `signals` — 11 colonnes (JSON metadata)
- `trades` — 15 colonnes
- `session_state` — 1 row (id=1 constraint) pour crash recovery

**10 methodes async :**
- `init()`, `close()`
- `insert_candles_batch()` — INSERT OR IGNORE (deduplication)
- `get_candles(symbol, tf, start, end, limit)`
- `get_latest_candle_timestamp()` — pour la reprise du fetch
- `delete_candles()` — dev/testing
- `insert_signal()`, `insert_trade()`
- `save_session_state()`, `load_session_state()`

### backend/core/data_engine.py (~250 lignes)

WebSocket streaming via ccxt Pro + validation + buffer rolling.

- `DataValidator` — validate_candle(), check_gap(), is_duplicate() (statique)
- `DataEngine` — buffers par (symbol, tf), callbacks, reconnexion avec backoff exponentiel
- Methodes : `start()`, `stop()`, `get_data(symbol)`, `on_candle(callback)`
- Max 500 bougies par buffer, erreurs loguees mais non fatales

### backend/core/rate_limiter.py (~80 lignes)

Token bucket par categorie d'endpoint.

- `TokenBucket` — refill lineaire, `acquire()` bloque jusqu'a disponibilite
- `RateLimiter` — 4 buckets (market_data, trade, account, position)
- `from_exchange_config()` — factory depuis la config

### backend/core/logging_setup.py (~70 lignes)

Loguru : console coloree + fichiers JSON.

- Console : `HH:mm:ss | LEVEL | module:function:line | message`
- Fichiers : `logs/scalp_radar.log` (50MB rotation, 30j retention, gzip) + `logs/errors.log` (ERROR only)

### backend/api/server.py (~60 lignes)

FastAPI avec lifespan (DataEngine integre).

- Startup : config -> logging -> DB init -> DataEngine start (si ENABLE_WEBSOCKET)
- Shutdown : engine stop -> DB close
- CORS pour Vite dev (localhost:5173)

### backend/api/health.py (~30 lignes)

GET /health — liveness check.

- `engine_connected`, `engine_last_update`, `uptime_seconds`, database status

### scripts/fetch_history.py (~180 lignes)

Backfill REST depuis Bitget via ccxt.

- CLI : `--symbol`, `--timeframe`, `--days`, `--force`
- Reprise automatique depuis la derniere bougie en DB
- Pagination (1000 bougies/requete), rate limit respecte
- tqdm pour la progress bar

### config/ (4 fichiers YAML)

- `assets.yaml` — 3 assets (BTC, ETH, SOL), 4 TF (1m, 5m, 15m, 1h), 1 correlation group
- `strategies.yaml` — 5 strategies avec params tunables + custom_strategies
- `risk.yaml` — kill switch (5%/10%), position (2% risk, x15 levier), fees (0.02/0.06%), slippage (0.05%)
- `exchanges.yaml` — Bitget WebSocket, rate limits, API config (USDT-M, mark_price)

### frontend/ (scaffold)

- React + Vite, proxy `/api` et `/health` vers backend:8000
- Placeholder `App.jsx` — implementation Sprint 3

---

## Tests (40 passants)

| Fichier | Tests | Couverture |
|---------|-------|-----------|
| `test_models.py` | 17 | Enums, Candle validation, OrderBook, Trade.net_pnl, Signal, SessionState |
| `test_config.py` | 11 | Chargement YAML, validation croisee, erreurs, defauts |
| `test_database.py` | 12 | CRUD candles (async), session state, signals, trades |

---

## Dependances Python

| Categorie | Libraries |
|-----------|-----------|
| Data & Async | ccxt>=4.0, pandas>=2.2, numpy>=2.0, aiosqlite>=0.20, websockets>=13.0 |
| API & Web | FastAPI>=0.115, uvicorn[standard]>=0.30, httpx>=0.27 |
| Validation | pydantic>=2.8, pydantic-settings>=2.4, pyyaml>=6.0 |
| Observabilite | loguru>=0.7, python-dotenv>=1.0, tqdm>=4.66 |
| Testing | pytest>=8.0, pytest-asyncio>=0.24 |

---

## Resume

| # | Fichier | Lignes est. | Description |
|---|---------|-------------|-------------|
| 1 | `backend/core/models.py` | ~350 | 17 modeles Pydantic + 7 enums |
| 2 | `backend/core/config.py` | ~375 | YAML loader + validation croisee |
| 3 | `backend/core/database.py` | ~310 | SQLite async (4 tables, 10 methodes) |
| 4 | `backend/core/data_engine.py` | ~250 | WebSocket ccxt Pro + validation + buffer |
| 5 | `backend/core/rate_limiter.py` | ~80 | Token bucket par categorie |
| 6 | `backend/core/logging_setup.py` | ~70 | Loguru console + JSON files |
| 7 | `backend/api/server.py` | ~60 | FastAPI + lifespan |
| 8 | `backend/api/health.py` | ~30 | GET /health |
| 9 | `scripts/fetch_history.py` | ~180 | Backfill REST Bitget |
| 10 | `tests/` (3 fichiers) | ~400 | 40 tests |
| 11 | `config/` (4 YAML) | ~200 | Parametres tunables |
| 12 | `frontend/` (scaffold) | ~50 | React + Vite placeholder |
| **Total** | | **~2350** | |
