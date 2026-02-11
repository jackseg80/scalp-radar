# Plan Sprint 4 — Production (Docker, Monitoring, Telegram, Crash Recovery)

## Contexte

Sprints 1-3 terminés : infrastructure complète (config, DB, DataEngine), backtesting event-driven, 4 stratégies (VWAP+RSI, Momentum, Funding, Liquidation), Simulator paper trading, Arena classement, API REST + WebSocket, frontend MVP React. 166 tests passants.

**Objectif Sprint 4** : rendre le système déployable en production sur le serveur Linux 192.168.1.200 via Docker Compose, avec crash recovery, monitoring automatique et alertes Telegram.

**Livrables** :
- StateManager : sauvegarde/restauration de l'état du Simulator (crash recovery)
- Telegram : alertes trades, kill switch, heartbeat horaire
- Watchdog : surveillance data freshness, WS connecté, stratégies actives
- Docker : conteneurisation backend + frontend, docker-compose.yml
- deploy.sh : déploiement automatisé sur le serveur

---

## Décisions architecturales

### 1. StateManager = wrapper autour de DB existante

Le `SessionState` model et `Database.save_session_state()`/`load_session_state()` existent déjà. Le StateManager ajoute la couche métier : sauvegarde périodique (toutes les 60s), restauration au boot, et snapshot des positions ouvertes des runners dans un fichier JSON (`data/simulator_state.json`). Pas de nouvelle table DB — on réutilise `session_state` + un fichier JSON pour les positions sérialisées.

**Écriture atomique** : écriture dans un fichier `.tmp` puis `os.replace()` vers le fichier final. Si crash pendant l'écriture, le `.tmp` est incomplet mais le fichier final reste intact.

**Lecture robuste** : `load_runner_state()` wrappe `json.load()` dans try/except. Si le fichier est absent, vide, ou corrompu → log warning + retourne None → démarrage fresh.

**Pas de race condition** : le `saved_state` est passé directement à `simulator.start(saved_state=...)` pour que les runners soient créés avec le bon capital dès le départ. Le callback `data_engine.on_candle()` n'est enregistré qu'APRÈS la restauration complète.

### 2. Telegram via httpx (pas de dépendance supplémentaire)

`httpx` est déjà en dépendance. On utilise l'API Bot Telegram directement (`https://api.telegram.org/bot<token>/sendMessage`). Pas de lib `python-telegram-bot` — trop lourd pour juste envoyer des messages.

### 3. Watchdog = tâche asyncio dans le lifespan, dépendances explicites

Le Watchdog tourne en boucle (toutes les 30s) et vérifie : WS connecté, data freshness < 5min, stratégies pas silencieuses > 1h. Si anomalie → alerte Telegram. Reçoit ses dépendances explicitement (`data_engine`, `simulator`, `notifier`) — pas `app.state` — pour rester testable unitairement.

### 4. Docker multi-stage

Backend : `python:3.12-slim` + uv pour installer les deps. Frontend : build Vite → servir via nginx. Un `docker-compose.yml` avec 2 services + volume pour SQLite.

### 5. deploy.sh = script avec graceful shutdown + rollback

`docker compose down` (graceful shutdown → SIGTERM → lifespan sauvegarde l'état) → `git pull` → `docker compose build` → `docker compose up -d` → `curl /health`. Si le health check échoue, rollback vers l'image précédente. Crée `data/` et `logs/` si absents (évite les problèmes de permissions Docker).

### 6. Pas de check Bitget positions au boot (Sprint 5)

CLAUDE.md mentionne "check Bitget API for open positions on startup" — mais c'est pour le trading live (Sprint 5). En Sprint 4, le Simulator gère du capital virtuel, donc on restaure les positions depuis le fichier JSON local. Le check Bitget sera ajouté dans l'executor Sprint 5.

---

## Phases d'implémentation

### Phase 1 — StateManager (~200 lignes)

**But** : crash recovery pour le Simulator.

#### `backend/core/state_manager.py` (NEW, ~150 lignes)

```python
class StateManager:
    """Sauvegarde et restauration de l'état du Simulator."""

    def __init__(self, db: Database, state_file: str = "data/simulator_state.json")
    async def save_runner_state(self, runners: list[LiveStrategyRunner]) -> None
        # Sérialise : capital, trades, positions ouvertes, kill_switch, stats
        # Écrit dans data/simulator_state.json (atomique via tmp + rename)
        # + save_session_state() en DB pour les stats globales
    async def load_runner_state(self) -> dict | None
        # Lit data/simulator_state.json
        # try/except json.JSONDecodeError + FileNotFoundError + KeyError
        # Si fichier absent, vide ou corrompu → log warning + retourne None (fresh start)
    async def start_periodic_save(self, simulator: Simulator, interval: int = 60) -> None
        # Boucle asyncio : save toutes les 60s
    async def stop(self) -> None
```

**Format JSON** (`data/simulator_state.json`) :
```json
{
  "saved_at": "2025-01-15T10:30:00Z",
  "runners": {
    "vwap_rsi": {
      "capital": 10234.50,
      "net_pnl": 234.50,
      "total_trades": 15,
      "wins": 9,
      "losses": 6,
      "kill_switch": false,
      "position": null
    }
  }
}
```

#### `backend/backtesting/simulator.py` (MODIFY, +40 lignes)

- Ajouter `restore_state(state: dict) -> None` au `LiveStrategyRunner` : restaure capital, stats, kill_switch
- Ajouter `saved_state: dict | None = None` en paramètre de `Simulator.start()` : si fourni, distribue les états aux runners AVANT d'enregistrer le callback `data_engine.on_candle()`. Cela évite la race condition où une candle arriverait avec le capital initial au lieu du capital restauré.
- Séparer l'enregistrement du callback : `start()` crée les runners → restaure l'état → PUIS appelle `data_engine.on_candle(self._dispatch_candle)`

#### Tests Phase 1 (~16 tests)

- `tests/test_state_manager.py` : save, load, restore, fichier absent/corrompu/vide, sauvegarde atomique, round-trip complet, restore_state avec position, kill switch

---

### Phase 2 — Telegram Alerts (~200 lignes)

**But** : notifications Telegram pour trades, kill switch, heartbeat.

#### `backend/alerts/telegram.py` (NEW, ~80 lignes)

```python
class TelegramClient:
    """Client Telegram via API Bot (httpx)."""

    def __init__(self, bot_token: str, chat_id: str)
    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool
    async def send_trade_alert(self, trade: dict, strategy: str) -> bool
    async def send_kill_switch_alert(self, strategy: str, loss_pct: float) -> bool
    async def send_startup_message(self, strategies: list[str]) -> bool
    async def send_shutdown_message(self) -> bool
```

#### `backend/alerts/heartbeat.py` (NEW, ~60 lignes)

Heartbeat Telegram à intervalle configurable (défaut 3600s, overridable via .env `HEARTBEAT_INTERVAL`).

#### `backend/alerts/notifier.py` (NEW, ~60 lignes)

```python
class AnomalyType(str, Enum):
    WS_DISCONNECTED = "ws_disconnected"
    DATA_STALE = "data_stale"
    ALL_STRATEGIES_STOPPED = "all_strategies_stopped"
    KILL_SWITCH_GLOBAL = "kill_switch_global"

class Notifier:
    """Centralise les notifications : dispatche vers Telegram (+ futurs canaux)."""
    # Si telegram est None → notifications juste loguées
```

#### Tests Phase 2 (~10 tests)

- `tests/test_telegram.py` : 7 tests (send_message, trade alert, kill switch, startup, notifier sans telegram, notifier dispatch, anomaly)
- `tests/test_heartbeat.py` : 3 tests (format message, no trades, stop)

---

### Phase 3 — Watchdog (~120 lignes)

**But** : surveillance automatique avec alertes sur anomalies.

#### `backend/monitoring/watchdog.py` (NEW, ~100 lignes)

Dépendances explicites (data_engine, simulator, notifier) — pas app_state.
Anti-spam : cooldown 5 min par type d'anomalie.

#### `backend/api/health.py` (MODIFY, +10 lignes)

Ajout watchdog status dans la réponse health.

#### Tests Phase 3 (~8 tests)

- `tests/test_watchdog.py` : all ok, WS down, data stale, all strategies stopped, cooldown, get_status (initial + after check), lifecycle

---

### Phase 4 — Docker & Déploiement

- `Dockerfile.backend` — python:3.12-slim + uv
- `Dockerfile.frontend` — node:18 build → nginx:alpine
- `nginx.conf` — proxy vers `http://backend:8000` (service Docker Compose)
- `docker-compose.yml` — 2 services + healthcheck
- `deploy.sh` — graceful shutdown + rollback
- `.dockerignore`

---

### Phase 5 — Câblage lifespan

Intégration complète dans `backend/api/server.py` :
1. Database → 2. Telegram/Notifier → 3. DataEngine → 4. StateManager load + Simulator.start(saved_state) + periodic save + Arena → 5. Watchdog + Heartbeat → notify_startup → yield → shutdown inverse

---

## Résultat final

- **200 tests passants** (166 existants + 34 nouveaux)
- **11 fichiers créés** + **4 fichiers modifiés**
- ~950 lignes (Python + Docker config)
