# Scalp Radar

Outil de scalping multi-stratégies pour crypto futures (Bitget).
Détecte les opportunités, score les signaux, exécute les stratégies en parallèle
et présente les résultats via un dashboard temps réel.

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
# 6 mois complets (BTC, ETH, SOL × 4 timeframes)
uv run python -m scripts.fetch_history

# Test rapide : 7 jours, un seul symbole
uv run python -m scripts.fetch_history --symbol BTC/USDT --timeframe 5m --days 7
```

## Tests

```bash
uv run pytest tests/ -v
```

252 tests couvrant : modèles, config, database, indicateurs, 4 stratégies, backtesting, simulator, arena, API, state manager, telegram, watchdog, executor, risk manager.

## Endpoints

| Endpoint | Description |
| --- | --- |
| `GET /health` | Status du système (data engine, database, uptime) |
| `GET /api/simulator/status` | Statut du simulateur et stratégies actives |
| `GET /api/simulator/positions` | Positions ouvertes par stratégie |
| `GET /api/simulator/trades` | Trades récents (paginé, ?limit=50) |
| `GET /api/simulator/performance` | Métriques de performance par stratégie |
| `GET /api/arena/ranking` | Classement des stratégies par return % |
| `GET /api/arena/strategy/{name}` | Détail d'une stratégie (status + trades + perf) |
| `GET /api/signals/recent` | Derniers signaux (paginé, ?limit=20) |
| `GET /api/simulator/conditions` | Indicateurs courants + conditions par stratégie/asset |
| `GET /api/signals/matrix` | Matrice simplifiée heatmap (stratégie × asset) |
| `GET /api/simulator/equity` | Courbe d'equity (depuis trades, ?since= filter) |
| `GET /api/executor/status` | Statut executor (position, SL/TP, kill switch) |
| `POST /api/executor/test-trade` | Ouvre un trade test LONG BTC (capital minimal) |
| `POST /api/executor/test-close` | Ferme la position ouverte par market close |
| `WS /ws/live` | WebSocket push temps réel (status, ranking, prix, executor) |

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
config/              # Paramètres YAML (assets, strategies, risk, exchanges)
backend/core/        # Modèles, config, database, data engine, indicateurs, position manager
backend/strategies/  # 4 stratégies (vwap_rsi, momentum, funding, liquidation) + factory
backend/backtesting/ # Engine, metrics, simulator (paper trading), arena (classement)
backend/execution/   # Executor live trading (Bitget), risk manager
backend/api/         # FastAPI + endpoints simulator/arena/signals/executor + WebSocket
backend/alerts/      # Telegram client, Notifier, Heartbeat
backend/monitoring/  # Watchdog (data freshness, WS, stratégies)
scripts/             # fetch_history, run_backtest
frontend/src/        # React dashboard V2 (15 composants, Scanner/Heatmap/Risque, hooks polling + WS)
tests/               # pytest (252 tests)
```

## Déploiement production

```bash
# Sur le serveur Linux (192.168.1.200)
ssh jack@192.168.1.200
cd ~/scalp-radar
bash deploy.sh
# → graceful shutdown, git pull, docker build, health check, rollback si erreur
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
BITGET_SANDBOX=false    # Mainnet (true = demo trading, non fonctionnel actuellement)
```

## Avancement

- [x] Sprint 1 — Fondations (config, modèles, database, data engine, API, tests)
- [x] Sprint 2 — Backtesting & stratégie VWAP+RSI
- [x] Sprint 3 — Simulator, 4 stratégies, Arena, API, frontend MVP
- [x] Sprint 4 — Production (Docker, crash recovery, monitoring, Telegram)
- [x] Sprint 5a — Trading live (executor, risk manager, pipeline validé mainnet)
- [ ] Sprint 5b — Scaling (adaptive selector, 3 paires, 4 stratégies)
- [x] Sprint 6 Phase 1 — Dashboard V2 (Scanner/Heatmap/Risque, conditions live, executor panel, equity curve, 15 composants)
