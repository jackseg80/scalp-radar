# Plan Sprint 3 — Simulator, Arena & Stratégies 2-5

## Contexte

Sprint 2 terminé : backtesting event-driven + VWAP+RSI. Baseline validée : 85 trades/180j/3 paires, edge solide en RANGING (+$2,457) mais pertes en TRENDING (-$2,517). Net ~breakeven.

**Objectif Sprint 3** : paper trading live (simulateur), stratégies 2-5 pour augmenter le volume de trades, Arena pour comparaison parallèle, API + frontend MVP.

**Motivation utilisateur** : "Le vrai test c'est le live (paper trading)" et "les stratégies 2-5 vont générer des signaux sur des conditions différentes".

---

## Décisions architecturales clés

### 1. Extraction d'un PositionManager réutilisable

Le `BacktestEngine` contient la logique de position sizing, fees, slippage, TP/SL qui est battle-tested. Plutôt que de la dupliquer, on extrait un `PositionManager` utilisé par BacktestEngine ET Simulator.

### 2. Indicateurs incrémentaux = recalcul sur fenêtre rolling

En backtest, `compute_indicators()` est appelé une fois sur tout le dataset. En live, on maintient des arrays numpy rolling (300-500 éléments) et on recalcule à chaque nouvelle candle. Sur 300 floats, c'est < 1ms. Pas besoin de variantes incrémentales de RSI/ADX — les mêmes fonctions `indicators.py` sont réutilisées.

### 3. Arena = isolation totale des stratégies

Chaque stratégie tourne avec son capital virtuel isolé. Pas de partage de positions. L'Arena est un orchestrateur qui maintient un classement par performance.

### 4. Données OI/Funding via polling ccxt

Pas de WebSocket pour l'OI et le funding rate — polling via `fetchOpenInterest()` (60s) et `fetchFundingRate()` (5min). Stockés dans DataEngine, exposés via `get_funding_rate(symbol)` et `get_open_interest(symbol)`.

### 5. OrderFlow = filtre de confirmation, pas une stratégie

Strategy 3 n'hérite pas de `BaseStrategy`. C'est un filtre qui boost/réduit les scores des autres stratégies via le carnet d'ordres L2. Priorité basse.

### 6. Frontend MVP

5 composants essentiels (Arena ranking, signal feed, session stats, trade history, header). Pas de heatmap ni risk calculator pour le moment. Dark theme du prototype. WebSocket avec reconnexion automatique (backoff exponentiel).

### 7. Constantes pour extra_data (pas de magic strings)

Les clés de `StrategyContext.extra_data` sont définies comme constantes dans `backend/strategies/base.py` :

```python
EXTRA_FUNDING_RATE = "funding_rate"
EXTRA_OPEN_INTEREST = "open_interest"
EXTRA_OI_CHANGE_PCT = "oi_change_pct"
EXTRA_ORDERBOOK = "orderbook"
```

Les stratégies et le Simulator utilisent ces constantes — jamais de strings hardcodées.

### 8. Distinction backtest / simulation / live en DB

Ajout d'un champ `source: str` ("backtest", "simulation", "live") dans les tables `trades` et `signals`. Permet au dashboard de filtrer par source. Le BacktestEngine écrit `source="backtest"`, le Simulator écrit `source="simulation"`.

### 9. Stratégie Funding = paper trading only (pas backtestable)

Le `fetch_history.py` ne télécharge pas l'historique des funding rates. La stratégie Funding ne peut être validée qu'en paper trading via le Simulator. Un script `fetch_funding_history.py` pourra être ajouté plus tard si nécessaire — pas bloquant pour Sprint 3.

### 10. Kill switch dans le Simulator (non négociable)

Le `LiveStrategyRunner` vérifie après chaque trade si le kill switch doit s'activer (depuis `risk.yaml` : `max_session_loss_percent: 5%`, `max_daily_loss_percent: 10%`). Si activé, la stratégie (ou toutes les stratégies) est arrêtée. Le `Simulator` expose `is_kill_switch_triggered()` pour l'API.

---

## Phases d'implémentation

### Phase 1 — PositionManager + IncrementalIndicators (~270 lignes)

**But** : extraire la logique réutilisable, préparer l'infrastructure pour le simulateur.

#### `backend/core/position_manager.py` (NEW, ~150 lignes)

```python
@dataclass
class PositionManagerConfig:
    leverage: int
    maker_fee: float
    taker_fee: float
    slippage_pct: float
    high_vol_slippage_mult: float
    max_risk_per_trade: float

class PositionManager:
    """Gestion des positions : sizing, fees, slippage, TP/SL.
    Utilisé par BacktestEngine ET Simulator."""

    def open_position(self, signal, timestamp, capital) -> OpenPosition | None
    def check_position_exit(self, candle, position, strategy, ctx, regime) -> TradeResult | None
    def close_position(self, position, exit_price, exit_time, exit_reason, regime) -> TradeResult
    def force_close(self, position, candle, regime) -> TradeResult
    def unrealized_pnl(self, position, current_price) -> float
```

Extraction directe depuis `BacktestEngine._open_position()` (L304-348), `_close_position()` (L350-404), `_check_position_exit()` (L241-282), `_ohlc_heuristic()` (L406-415), `_unrealized_pnl()` (L417-421).

#### `backend/backtesting/engine.py` (MODIFY, ~-80 lignes refactored)

Remplacer les méthodes privées par délégation à `self._pm: PositionManager`. Tous les tests existants doivent passer sans modification.

#### `backend/core/incremental_indicators.py` (NEW, ~120 lignes)

```python
class IncrementalIndicatorEngine:
    """Buffers numpy rolling par (symbol, timeframe).
    Recalcule les indicateurs sur la fenêtre complète à chaque update."""

    def __init__(self, strategies: list[BaseStrategy])  # Déduit max_candles par TF
    def update(self, symbol: str, timeframe: str, candle: Candle) -> None
    def get_indicators(self, symbol: str) -> dict[str, dict[str, Any]]
    # Retourne {"5m": {"rsi": 23.5, "vwap": 98500, ...}, "15m": {"rsi": 45, "adx": 22, ...}}
```

Utilise les fonctions existantes de `backend/core/indicators.py` (rsi, vwap_rolling, adx, atr, sma).

#### Tests Phase 1 (~8 tests)

- `tests/test_position_manager.py` : open, close, tp/sl, fee calc, sizing, OHLC heuristic (6)
- `tests/test_incremental_indicators.py` : update + get, fenêtre rolling (2)

---

### Phase 2 — Stratégies 4 (Momentum) + 5 (Funding) (~310 lignes)

**But** : deux stratégies supplémentaires utilisant l'infrastructure existante.

#### `backend/strategies/momentum.py` (NEW, ~180 lignes)

Momentum Breakout — trade AVEC la tendance (complémentaire à VWAP+RSI qui trade CONTRE).

**Logique d'entrée :**
- Prix casse le max des N dernières bougies (breakout_lookback=20) → LONG
- Volume > volume_sma × multiplier (2.0) — confirmation
- ADX 15m > 25 et DI+ > DI- — tendance confirmée (inverse du filtre VWAP+RSI)
- TP = entry + ATR × atr_multiplier_tp (2.0), SL = entry - ATR × atr_multiplier_sl (1.0)
- Les % config (tp_percent/sl_percent) servent de cap

**check_exit** : ADX chute sous 20 → momentum essoufflé → "signal_exit"

**compute_indicators** : ATR, ADX+DI, volume SMA, rolling high/low sur breakout_lookback

#### `backend/strategies/funding.py` (NEW, ~130 lignes)

Funding Rate Arbitrage — scalp lent sur taux de financement extrêmes.

**Logique d'entrée :**
- `funding_rate < -0.03%` → LONG (les shorts paient → pression short excessive)
- `funding_rate > 0.03%` → SHORT (les longs paient → pression long excessive)
- Entry delay de 5 min après détection (attente de confirmation)

**check_exit** : funding rate revient à neutre (< |0.01%|) → "signal_exit"

**Données** : `ctx.extra_data.get(EXTRA_FUNDING_RATE)` — alimenté par DataEngine (Phase 3).

**Note** : la stratégie Funding ne peut PAS être backtestée sur données historiques (pas d'historique funding rates en DB). Elle sera validée uniquement en paper trading via le Simulator.

#### `backend/strategies/base.py` (MODIFY, +10 lignes)

- Ajouter `extra_data: dict[str, Any] = field(default_factory=dict)` à `StrategyContext`
- Ajouter les constantes `EXTRA_FUNDING_RATE`, `EXTRA_OPEN_INTEREST`, `EXTRA_OI_CHANGE_PCT`, `EXTRA_ORDERBOOK`

#### Tests Phase 2 (~10 tests)

- `tests/test_strategy_momentum.py` : long breakout, short breakout, no breakout, volume filter, trend filter, ATR TP/SL, check_exit ADX drop (7)
- `tests/test_strategy_funding.py` : extreme neg→long, extreme pos→short, neutral→none (3)

---

### Phase 3 — Extensions DataEngine (OI + Funding polling) (~150 lignes)

**But** : alimenter les stratégies 2 et 5 en données live.

#### `backend/core/data_engine.py` (MODIFY, +100 lignes)

```python
# Nouveaux attributs
self._funding_rates: dict[str, float] = {}
self._open_interest: dict[str, list[OISnapshot]] = {}

# Nouvelles tâches dans start()
asyncio.create_task(self._poll_funding_rates())   # toutes les 5 min
asyncio.create_task(self._poll_open_interest())    # toutes les 60s

# Nouvelles méthodes publiques
def get_funding_rate(self, symbol: str) -> float | None
def get_open_interest(self, symbol: str) -> list[OISnapshot]
```

Appels ccxt wrappés en try/except avec logging warning si non supporté.

#### `backend/core/models.py` (MODIFY, +15 lignes)

```python
class OISnapshot(BaseModel):
    timestamp: datetime
    symbol: str
    value: float         # Open interest en USDT
    change_pct: float = 0.0  # Variation vs snapshot précédent
```

#### `backend/core/database.py` (MODIFY, +10 lignes)

- Ajouter colonne `source TEXT DEFAULT 'backtest'` aux tables `trades` et `signals`
- Mettre à jour `insert_signal()` et `insert_trade()` pour accepter le paramètre `source`
- Ajouter filtre `source` aux méthodes de requête

#### Tests Phase 3 (~4 tests)

- `tests/test_data_engine_extensions.py` : mock ccxt, funding polling, OI polling, get methods (4)

---

### Phase 4 — Stratégie 2 (Liquidation Zone Hunting) (~200 lignes)

**But** : stratégie basée sur les zones de liquidation estimées.

#### `backend/strategies/liquidation.py` (NEW, ~200 lignes)

**Logique d'entrée :**
1. Estimer les zones de liquidation :
   - `liq_long_zone = price × (1 - 1/leverage_estimate)` — où les longs se font liquider
   - `liq_short_zone = price × (1 + 1/leverage_estimate)` — où les shorts se font liquider
2. Si OI a augmenté > `oi_change_threshold` (5%) → leviers chargés
3. Si le prix approche une zone (< `zone_buffer_percent` = 1.5%) :
   - LONG si prix approche la zone de liq des shorts (short squeeze anticipé)
   - SHORT si prix approche la zone de liq des longs (cascade de liquidation)
   - Buffer large (1.5% au lieu de 0.5%) car l'estimation du levier moyen est grossière

**check_exit** : OI chute brutalement (cascade terminée) → "signal_exit"

**Données** : `ctx.extra_data.get(EXTRA_OPEN_INTEREST)` et `ctx.extra_data.get(EXTRA_OI_CHANGE_PCT)`

**Note** : comme Funding, la stratégie Liquidation ne peut être backtestée que si on a l'historique OI. En Sprint 3, validation en paper trading uniquement.

**Config YAML** : modifier `zone_buffer_percent: 0.5` → `zone_buffer_percent: 1.5` dans `config/strategies.yaml` et default dans `config.py`.

#### Tests Phase 4 (~5 tests)

- `tests/test_strategy_liquidation.py` : OI haut + zone proche → signal, OI bas → None, exit OI drop, calcul zones, score (5)

---

### Phase 5 — Simulator + Arena (~420 lignes)

**But** : le livrable principal — paper trading live.

#### `backend/strategies/factory.py` (NEW, ~50 lignes)

```python
def create_strategy(name: str, config: AppConfig) -> BaseStrategy
def get_enabled_strategies(config: AppConfig) -> list[BaseStrategy]
```

Mapping : vwap_rsi, momentum, funding, liquidation → classes correspondantes.

#### `backend/backtesting/simulator.py` (NEW, ~280 lignes)

```python
class LiveStrategyRunner:
    """Exécute une stratégie sur données live (paper trading).
    Un runner par stratégie. Capital virtuel isolé."""

    def __init__(self, strategy, config, indicator_engine, position_manager, database)
    async def on_candle(self, symbol, timeframe, candle) -> None
        # 1. Update indicators via IncrementalIndicatorEngine
        # 2. Build StrategyContext (avec extra_data: funding, OI via constantes)
        # 3. Si position ouverte:
        #    a. Détection changement régime → si RANGING→TRENDING : force close ("regime_change")
        #    b. Check exit via PositionManager (TP/SL/signal_exit)
        # 4. Si pas de position et kill switch non activé: evaluate via strategy
        # 5. Persist signal/trade en DB (source="simulation")
        # 6. Update session stats
        # 7. Vérifier kill switch (max_session_loss_percent, max_daily_loss_percent)
    def get_status(self) -> dict
    def get_trades(self) -> list[TradeResult]
    def get_performance(self) -> BacktestMetrics

class Simulator:
    """Orchestrateur du paper trading."""

    def __init__(self, data_engine, config, database)
    async def start(self) -> None
        # 1. Créer IncrementalIndicatorEngine
        # 2. Créer un LiveStrategyRunner par stratégie enabled (via factory)
        # 3. Enregistrer self._on_candle auprès du DataEngine :
        #    data_engine.on_candle(self._dispatch_candle)
    async def _dispatch_candle(self, symbol, timeframe, candle) -> None
        # Fan-out vers tous les runners
    async def stop(self) -> None
    def get_all_status(self) -> dict
    def get_all_trades(self) -> list
    def is_kill_switch_triggered(self) -> bool
```

**Câblage Simulator ↔ DataEngine** : dans `Simulator.start()`, on appelle `self._data_engine.on_candle(self._dispatch_candle)` pour brancher le flux de données. Le Simulator doit être créé APRÈS le DataEngine dans le lifespan FastAPI.

Le `LiveStrategyRunner` réutilise `PositionManager` pour les positions et `IncrementalIndicatorEngine` pour les indicateurs. La logique est la même que `BacktestEngine.run()` mais event-driven (callback) au lieu d'itératif.

**Quick fix intégré** : dans `on_candle()`, si position ouverte et que le régime passe de RANGING → TRENDING, couper la position (exit_reason="regime_change"). Résout le problème identifié en backtest (-$2,517 en trending).

**Kill switch** : après chaque trade clôturé, vérifier `session_pnl / initial_capital` contre `max_session_loss_percent` (5%) et `max_daily_loss_percent` (10%) de `risk.yaml`. Si déclenché → `self._kill_switch_triggered = True`, plus aucune nouvelle position.

**Source field** : tous les trades/signaux persistés avec `source="simulation"` pour les distinguer des backtests (`source="backtest"`) et du futur live (`source="live"`).

#### `backend/backtesting/arena.py` (NEW, ~120 lignes)

```python
@dataclass
class StrategyPerformance:
    name: str
    capital: float
    net_pnl: float
    net_return_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    is_active: bool

class StrategyArena:
    """Comparaison parallèle des stratégies."""

    def __init__(self, simulator: Simulator)
    def get_ranking(self) -> list[StrategyPerformance]
    def get_strategy_detail(self, name: str) -> dict
```

#### Tests Phase 5 (~12 tests)

- `tests/test_simulator.py` : on_candle processing, position open/close, fee calc, multi-strategy isolation, session state, kill switch, regime change exit (8)
- `tests/test_arena.py` : ranking, multi-strategy, factory (4)

---

### Phase 6 — API Endpoints (~230 lignes)

**But** : exposer les données du simulateur au frontend.

#### `backend/api/simulator.py` (NEW, ~100 lignes)

```
GET /api/simulator/status      → running, strategies actives, uptime
GET /api/simulator/positions   → positions ouvertes par stratégie
GET /api/simulator/trades      → trades récents (paginated, limit=50)
GET /api/simulator/performance → métriques par stratégie
```

#### `backend/api/arena.py` (NEW, ~40 lignes)

```
GET /api/arena/ranking → classement des stratégies
```

#### `backend/api/signals.py` (NEW, ~30 lignes)

```
GET /api/signals/recent → derniers signaux (limit=20)
```

#### `backend/api/websocket.py` (NEW, ~60 lignes)

```python
class ConnectionManager:
    """Gère les connexions WebSocket frontend."""
    async def connect(ws) / disconnect(ws) / broadcast(data)

@router.websocket("/ws/live")
async def live_feed(websocket):
    """Push temps réel : signaux, trades, status."""
```

#### `backend/api/server.py` (MODIFY, +25 lignes)

- Include des nouveaux routers (simulator, arena, signals, websocket)
- Lifespan, dans cet ordre :
  1. Config, logging, DB (existant)
  2. DataEngine start (existant)
  3. **Simulator = Simulator(data_engine, config, db)** → `app.state.simulator`
  4. **await simulator.start()** (crée runners, s'enregistre sur DataEngine)
  5. **Arena = StrategyArena(simulator)** → `app.state.arena`
  6. Shutdown : simulator.stop() avant engine.stop()

#### `frontend/vite.config.js` (MODIFY, +5 lignes)

- Ajouter proxy `/ws` pour WebSocket

#### Tests Phase 6 (~6 tests)

- `tests/test_api_simulator.py` : status, positions, trades, performance (4)
- `tests/test_api_arena.py` : ranking (1)
- `tests/test_api_signals.py` : recent signals (1)

---

### Phase 7 — Frontend MVP (~500 lignes JSX/CSS)

**But** : dashboard minimal connecté à l'API réelle. Dark theme du prototype.

Pas de nouvelles dépendances npm — React 19 natif (fetch + WebSocket API).

#### Hooks

- `frontend/src/hooks/useApi.js` (NEW, ~40 lignes) — polling avec interval configurable par composant
- `frontend/src/hooks/useWebSocket.js` (NEW, ~60 lignes) — connexion WS avec **reconnexion automatique** (backoff exponentiel : 1s → 2s → 4s → max 30s). Indispensable car uvicorn --reload coupe le WS fréquemment en dev.

#### Composants

- `frontend/src/components/Header.jsx` (NEW, ~40 lignes) — logo, status indicator, live toggle
- `frontend/src/components/ArenaRanking.jsx` (NEW, ~80 lignes) — table comparaison stratégies (poll: **30s**)
- `frontend/src/components/SignalFeed.jsx` (NEW, ~70 lignes) — flux signaux live via **WebSocket** (temps réel)
- `frontend/src/components/SessionStats.jsx` (NEW, ~50 lignes) — sidebar PnL, trades, win rate (poll: **3s**)
- `frontend/src/components/TradeHistory.jsx` (NEW, ~80 lignes) — table trades récents (poll: **10s**)

#### Layout

- `frontend/src/App.jsx` (REWRITE, ~80 lignes) — grid layout, dark theme
- `frontend/src/styles.css` (NEW, ~60 lignes) — variables CSS du prototype (#06080d, #00e68a)

Pas de tests frontend en Sprint 3.

---

### Phase 8 (optionnelle) — OrderFlow Confirmation (~150 lignes)

**Priorité basse** — peut être différé au Sprint 4.

#### `backend/strategies/orderflow.py` (NEW, ~150 lignes)

```python
class OrderFlowConfirmation:
    """PAS un BaseStrategy — filtre de confirmation L2."""

    def compute_confirmation_score(self, orderbook: OrderBookSnapshot) -> float
    # Score basé sur: bid/ask imbalance, large orders, absorption
```

Intégration dans `LiveStrategyRunner` : si orderflow enabled, multiplie le score du signal par le score de confirmation avant d'entrer.

#### `backend/core/data_engine.py` (MODIFY, +30 lignes)

Ajout `_watch_orderbook()` + `get_orderbook(symbol)`.

#### Tests Phase 8 (~3 tests)

- `tests/test_orderflow.py` : imbalance, large orders, confirmation score (3)

---

## Récapitulatif des fichiers

### Nouveaux fichiers (17)

| # | Fichier | Lignes | Phase |
|---|---------|--------|-------|
| 1 | `backend/core/position_manager.py` | ~150 | 1 |
| 2 | `backend/core/incremental_indicators.py` | ~120 | 1 |
| 3 | `backend/strategies/momentum.py` | ~180 | 2 |
| 4 | `backend/strategies/funding.py` | ~130 | 2 |
| 5 | `backend/strategies/liquidation.py` | ~200 | 4 |
| 6 | `backend/strategies/factory.py` | ~50 | 5 |
| 7 | `backend/backtesting/simulator.py` | ~250 | 5 |
| 8 | `backend/backtesting/arena.py` | ~120 | 5 |
| 9 | `backend/api/simulator.py` | ~100 | 6 |
| 10 | `backend/api/arena.py` | ~40 | 6 |
| 11 | `backend/api/signals.py` | ~30 | 6 |
| 12 | `backend/api/websocket.py` | ~60 | 6 |
| 13 | `frontend/src/hooks/useApi.js` | ~40 | 7 |
| 14 | `frontend/src/hooks/useWebSocket.js` | ~50 | 7 |
| 15 | `frontend/src/components/*.jsx` (5 fichiers) | ~320 | 7 |
| 16 | `frontend/src/styles.css` | ~60 | 7 |
| 17 | `backend/strategies/orderflow.py` | ~150 | 8 (opt) |

### Fichiers modifiés (8)

| Fichier | Changements | Phase |
|---------|-------------|-------|
| `backend/backtesting/engine.py` | Délégation au PositionManager | 1 |
| `backend/strategies/base.py` | Ajout `extra_data` + constantes EXTRA_* à StrategyContext | 2 |
| `backend/core/data_engine.py` | Polling OI + funding + orderbook | 3, 8 |
| `backend/core/models.py` | Ajout OISnapshot | 3 |
| `backend/core/database.py` | Colonne `source` dans trades/signals | 3 |
| `config/strategies.yaml` | zone_buffer_percent 0.5 → 1.5 | 4 |
| `backend/api/server.py` | Wire simulator, arena, routers dans lifespan | 6 |
| `frontend/vite.config.js` | Proxy WS | 6 |

### Tests (~48 nouveaux, 97 existants → ~145 total)

| Fichier | Tests | Phase |
|---------|-------|-------|
| `tests/test_position_manager.py` | 6 | 1 |
| `tests/test_incremental_indicators.py` | 2 | 1 |
| `tests/test_strategy_momentum.py` | 7 | 2 |
| `tests/test_strategy_funding.py` | 3 | 2 |
| `tests/test_data_engine_extensions.py` | 4 | 3 |
| `tests/test_strategy_liquidation.py` | 5 | 4 |
| `tests/test_simulator.py` | 8 (dont kill switch + regime change exit) | 5 |
| `tests/test_arena.py` | 4 | 5 |
| `tests/test_api_simulator.py` | 6 | 6 |
| `tests/test_orderflow.py` | 3 | 8 (opt) |
| **Total** | **~48** | |

---

## Graphe de dépendances

```
Phase 1 (PositionManager + IncrementalIndicators)
    │
    ├──→ Phase 2 (Momentum + Funding)
    │        │
    │        └──→ Phase 3 (DataEngine extensions)
    │                 │
    │                 └──→ Phase 4 (Liquidation)
    │
    └──→ Phase 5 (Simulator + Arena) — dépend de Phase 1 + 2
             │
             └──→ Phase 6 (API) → Phase 7 (Frontend)
                                       │
                                  Phase 8 (OrderFlow, optionnel)
```

---

## Vérification

1. **Après Phase 1** : `pytest tests/` — tous les 97 tests existants + 8 nouveaux passent
2. **Après Phase 2** : run backtests avec Momentum sur données existantes pour valider
3. **Après Phase 5** : lancer le simulateur en local avec `ENABLE_WEBSOCKET=true`, vérifier que les signaux sont émis
4. **Après Phase 6** : `curl localhost:8000/api/simulator/status` + test WS
5. **Après Phase 7** : `dev.bat` → dashboard affiche les données live
6. **En continu** : `pytest tests/` doit toujours passer (97 existants + ~48 nouveaux = ~145)

---

## Estimation totale

~2,300 lignes de code nouveau + ~500 lignes frontend = **~2,800 lignes**
