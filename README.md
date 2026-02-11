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

200 tests couvrant : modèles, config, database, indicateurs, 4 stratégies, backtesting, simulator, arena, API, state manager, telegram, watchdog.

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
| `WS /ws/live` | WebSocket push temps réel (status, ranking) |

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
backend/api/         # FastAPI + endpoints simulator/arena/signals + WebSocket
backend/alerts/      # Telegram client, Notifier, Heartbeat
backend/monitoring/  # Watchdog (data freshness, WS, stratégies)
scripts/             # fetch_history, run_backtest
frontend/src/        # React dashboard (5 composants, hooks polling + WS)
tests/               # pytest (200 tests)
```

## Avancement

- [x] Sprint 1 — Fondations (config, modèles, database, data engine, API, tests)
- [x] Sprint 2 — Backtesting & stratégie VWAP+RSI
- [x] Sprint 3 — Simulator, 4 stratégies, Arena, API, frontend MVP
- [x] Sprint 4 — Production (Docker, crash recovery, monitoring, Telegram)
- [ ] Sprint 5 — Trading live
