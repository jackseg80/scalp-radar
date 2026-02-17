# Scalp Radar

Outil de trading multi-stratégies pour crypto futures (Bitget).
13 stratégies (4 scalp 5m + 3 swing 1h + 6 grid/DCA 1h), optimisation Walk-Forward automatique,
paper trading live, executor mainnet, et dashboard temps réel.

## Prérequis

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Node.js 18+ (pour le frontend)

## Installation

```bash
# Cloner le repo
git clone https://github.com/jackseg80/scalp-radar.git
cd scalp-radar

# Installer les dépendances Python
uv sync

# Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos clés API Bitget

# Installer les dépendances frontend
cd frontend && npm install && cd ..
```

## Lancement en dev (Windows)

```bash
# Tout-en-un : backend (port 8000) + frontend (port 5173)
dev.bat

# Ou séparément :
uv run uvicorn backend.api.server:app --reload --port 8000
cd frontend && npm run dev
```

Pour désactiver le WebSocket en dev (évite les reconnexions lors du --reload) :

```bash
# Dans .env
ENABLE_WEBSOCKET=false
```

## Télécharger l'historique

```bash
# Backfill candles Binance (API publique, sans clé) — données WFO depuis 2020
uv run python -m scripts.backfill_candles
uv run python -m scripts.backfill_candles --symbol BTC/USDT --since 2023-01-01 --timeframe 4h

# Candles via ccxt (Bitget ou Binance) — données live récentes
uv run python -m scripts.fetch_history
uv run python -m scripts.fetch_history --symbol BTC/USDT --timeframe 5m --days 7
uv run python -m scripts.fetch_history --exchange binance

# Funding rates historiques
uv run python -m scripts.fetch_funding

# Open interest historique
uv run python -m scripts.fetch_oi
```

## Tests

```bash
uv run pytest tests/ -v
```

1037+ tests couvrant : modèles, config, database, indicateurs, 13 stratégies (4 scalp 5m + 3 swing 1h + 6 grid/DCA), backtesting (mono + multi-position + portfolio), simulator, arena, API, state manager, telegram, watchdog, executor (mono + grid DCA), risk manager, optimisation WFO, fast engines, funding/OI data, regime analysis, combo results, activity journal.

## Endpoints

### Core & Simulator

| Endpoint | Description |
| --- | --- |
| `GET /health` | Status du système (data engine, database, uptime) |
| `GET /api/simulator/status` | Statut du simulateur et stratégies actives (inclut kill_switch_reason) |
| `GET /api/simulator/positions` | Positions ouvertes par stratégie |
| `GET /api/simulator/trades` | Trades récents (paginé, ?limit=50) |
| `GET /api/simulator/performance` | Métriques de performance par stratégie |
| `GET /api/simulator/conditions` | Indicateurs courants + conditions par stratégie/asset |
| `GET /api/simulator/equity` | Courbe d'equity (depuis trades + journal, ?since= filter) |
| `GET /api/simulator/grid-state` | État grid/DCA des runners actifs |
| `POST /api/simulator/kill-switch/reset` | Reset kill switch global et réactive tous les runners |

### Arena & Signals

| Endpoint | Description |
| --- | --- |
| `GET /api/arena/ranking` | Classement des stratégies par return % |
| `GET /api/arena/strategy/{name}` | Détail d'une stratégie (status + trades + perf) |
| `GET /api/signals/recent` | Derniers signaux (paginé, ?limit=20) |
| `GET /api/signals/matrix` | Matrice simplifiée heatmap (stratégie × asset) |

### Executor (Live Trading)

| Endpoint | Description |
| --- | --- |
| `GET /api/executor/status` | Statut executor (position, SL/TP, kill switch) |
| `POST /api/executor/test-trade?symbol=BTC/USDT` | Ouvre un trade test (capital minimal, symbole configurable) |
| `POST /api/executor/test-close?symbol=BTC/USDT` | Ferme la position ouverte par market close |

### Portfolio Backtest

| Endpoint | Description |
| --- | --- |
| `GET /api/portfolio/presets` | Presets de configuration portfolio (top_10, all_assets, etc.) |
| `GET /api/portfolio/backtests` | Liste des runs portfolio sauvegardés |
| `GET /api/portfolio/backtests/{id}` | Détail d'un run portfolio |
| `POST /api/portfolio/backtests` | Créer et lancer un backtest portfolio |
| `POST /api/portfolio/run` | Lancer un backtest portfolio async (returns job_id) |
| `GET /api/portfolio/status/{job_id}` | Status d'un job portfolio en cours |
| `POST /api/portfolio/compare` | Comparer plusieurs runs portfolio |

### Activity Journal

| Endpoint | Description |
| --- | --- |
| `GET /api/journal/snapshots` | Snapshots equity/margin/unrealized (?limit=288 par défaut, 24h) |
| `GET /api/journal/events` | Événements positions (OPEN/CLOSE/ADJUST, ?limit=100) |
| `GET /api/journal/stats` | Statistiques journal (premier/dernier snapshot, count events) |

### WebSocket

| Endpoint | Description |
| --- | --- |
| `WS /ws/live` | Push temps réel (status, ranking, prix, executor, positions, grid_state) |

## Stack technique

| Composant       | Technologie                      |
| --------------- | -------------------------------- |
| Backend         | Python 3.12+, FastAPI, ccxt Pro  |
| Database        | SQLite (aiosqlite, WAL mode)     |
| Frontend        | React 19, Vite 6                 |
| Config          | YAML (Pydantic validation)       |
| Logging         | loguru (console + fichiers JSON) |
| Package manager | uv                               |

## Structure du projet

Voir [CLAUDE.md](CLAUDE.md) pour l'architecture complète et les décisions techniques.

```text
config/              # Paramètres YAML (assets, strategies, risk, exchanges, param_grids)
backend/core/        # Modèles, config, database, data engine, indicateurs, position managers (mono + grid)
backend/strategies/  # 13 stratégies (vwap_rsi, momentum, funding, liquidation, bollinger_mr, donchian_breakout, supertrend, envelope_dca, envelope_dca_short, grid_atr, grid_multi_tf, grid_funding, grid_trend) + base_grid + factory
backend/optimization/# WFO, overfitting detection, fast engines (mono + multi), indicator cache, grading
backend/backtesting/ # Engines (mono + multi-position + portfolio), metrics, simulator (paper trading), arena, portfolio_engine, portfolio_db
backend/execution/   # Executor live trading (Bitget), risk manager, adaptive selector
backend/api/         # FastAPI + endpoints simulator/arena/signals/executor/portfolio + WebSocket
backend/alerts/      # Telegram client, Notifier, Heartbeat
backend/monitoring/  # Watchdog (data freshness, WS, stratégies)
scripts/             # backfill_candles, fetch_history, fetch_funding, fetch_oi, run_backtest, optimize, parity_check, reset_simulator, sync_to_server, portfolio_backtest
frontend/src/        # React dashboard (32+ composants, Scanner/Heatmap/Explorer/Research/Portfolio/Diagnostic)
tests/               # pytest (1037+ tests, 55+ fichiers)
```

## Déploiement production

```bash
# Sur le serveur Linux (192.168.1.200)
ssh jack@192.168.1.200
cd ~/scalp-radar
bash deploy.sh          # normal : state préservé
bash deploy.sh --clean  # fresh start : supprime state files (pas la DB)
```

Le bot tourne H24 en Docker Compose : backend (port 8000) + frontend nginx (port 80).
Alertes Telegram : startup/shutdown, heartbeat horaire, trades live, anomalies watchdog.

### Logs production

```bash
# Logs temps réel du backend
docker compose logs -f backend

# 100 dernières lignes
docker compose logs --tail 100 backend

# Fichiers de logs persistants (volume Docker)
ls ~/scalp-radar/logs/

# Diagnostic rapide
curl http://localhost:8000/health | python3 -m json.tool
```

### Variables d'environnement (production)

```bash
LIVE_TRADING=true       # Active l'executor (défaut: false = simulation only)
# Note: Sandbox Bitget supprimé (cassé, ccxt #25523) — mainnet only
```

## Optimisation WFO

```bash
# Vérifier les données disponibles
uv run python -m scripts.optimize --check-data

# Optimiser une stratégie/symbole
uv run python -m scripts.optimize --strategy bollinger_mr --symbol BTC/USDT -v

# Optimiser toutes les stratégies, tous les symboles
uv run python -m scripts.optimize --all

# Appliquer les résultats grade A/B dans strategies.yaml
uv run python -m scripts.optimize --all --apply
```

## Avancement

- [x] Sprint 1 — Fondations (config, modèles, database, data engine, API, tests)
- [x] Sprint 2 — Backtesting & stratégie VWAP+RSI
- [x] Sprint 3 — Simulator, 4 stratégies, Arena, API, frontend MVP
- [x] Sprint 4 — Production (Docker, crash recovery, monitoring, Telegram)
- [x] Sprint 5a — Trading live (executor, risk manager, pipeline validé mainnet)
- [x] Sprint 5b — Scaling (adaptive selector, 3 paires, 4 stratégies)
- [x] Sprint 6 — Dashboard V2 (Scanner/Heatmap/Risque, conditions live, executor panel, equity curve)
- [x] Sprint 6b — Dashboard UX Overhaul (sidebar collapsible, positions actives, activité, resize)
- [x] Sprint 7 — Optimisation WFO + détection overfitting (Monte Carlo, DSR, grading A-F)
- [x] Sprint 7b — Funding/OI historiques + optimisation funding/liquidation
- [ ] Sprint 8 — Backtest Dashboard (planifié, non implémenté)
- [x] Sprint 9 — 3 nouvelles stratégies 1h (Bollinger MR, Donchian Breakout, SuperTrend)
- [x] Sprint 10 — Moteur multi-position modulaire (grid/DCA, EnvelopeDCA)
- [x] Sprint 11 — Paper trading Grid/DCA (GridStrategyRunner, warm-up, state persistence)
- [x] Sprint 12 — Executor Grid DCA + Alertes Telegram (8 bugs corrigés, multi-niveaux live)
- [x] Sprint 13 — Résultats WFO en DB + page Recherche (migration 49 JSON, visualisation)
- [x] Sprint 14 — Explorateur de paramètres (WFO background, heatmap 2D, 6 endpoints API)
- [x] Sprint 14b — Heatmap dense (wfo_combo_results), charts analytiques, tooltips
- [x] Sprint 14c — DiagnosticPanel (analyse intelligente WFO, 6 règles)
- [x] Sprint 15 — Stratégie Envelope DCA SHORT (miroir LONG, fast engine direction=-1)
- [x] Sprint 15b — Analyse par régime de marché (Bull/Bear/Range/Crash)
- [x] Sprint 15c-15d — Grading refinement (combo_score, consistance, top 5, bouton apply A/B, auto-add assets)
- [x] Sprint 16+17 — Dashboard Scanner Grid (GridSummary, endpoint grid-state, WS push, DataEngine batching)
- [x] Sprint 19 — Stratégie Grid ATR (10e stratégie, enveloppes ATR adaptatives, 3240 combos WFO)
- [x] Sprint 19b-19d — Grid ATR wfo_combo_results + régimes marché + grading Bitget transfer
- [x] Sprint 20a — Sizing equal allocation + margin guard 70%
- [x] Sprint 20b — Portfolio Backtest Multi-Asset (21 assets × 90j, kill switch temps réel, snapshots equity/margin)
- [x] Sprint 20b-UI — Frontend Portfolio Backtest (CRUD, equity curve SVG, drawdown chart, comparateur multi-runs)
- [x] Sprint 21a — Stratégie Grid Multi-TF (11e stratégie, Supertrend 4h + Grid ATR 1h, 384 combos WFO)
- [x] Sprint 22 — Stratégie Grid Funding (12e stratégie, DCA funding négatif, funding payments 8h, 2592 combos WFO)
- [x] Sprint Perf — Numba JIT Optimization (speedup 5-10x WFO, fallback transparent)
- [x] Sprint 23 — Grid Trend (13e stratégie, trend following DCA, EMA cross + ADX + trailing stop ATR, 2592 combos WFO)
- [x] Micro-Sprint Audit — Auth executor (API key), async I/O StateManager, buffer candles DataEngine
- [x] Sprint 23b — Grid Trend compute_live_indicators (paper/portfolio fix)
- [x] Sprint 24a — Portfolio Backtest Realistic Mode (sizing fixe, global margin guard, kill switch temps réel)
- [x] Sprint 24b — Portfolio Backtest Multi-Stratégie (clé runner strategy:symbol, dispatch multi-runners)
- [x] Sprint 25 — Activity Journal (snapshots equity/margin 5min, hooks OPEN/CLOSE DCA, ActivityFeed frontend)
- [x] Hotfix 25a — Retry DB writes journal (3 tentatives, backoff, throttle batch events)
- [ ] Hotfix 25b — Kill Switch Reliability (en cours : reset API, alerte restore, raison persistée, fix positions perdues)

---

**Dernière mise à jour :** 2026-02-17
