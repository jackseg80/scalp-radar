# Architecture — Scalp Radar

Guide complet du flux de données au runtime. Tout est extrait du code source.

---

## Vue d'ensemble

```
                          Bitget Exchange
                               |
                     WebSocket (ccxt.pro)
                               |
                         +-----v------+
                         | DataEngine |   (data_engine.py)
                         +-----+------+
                               |
              +----------------+----------------+
              |                |                |
         watch_ohlcv     poll_funding      poll_OI
         (par symbol)     (5 min)          (60s)
              |                |                |
              v                v                v
     +--------+--------+  _funding_rates   _open_interest
     | _buffers        |  (dict)           (dict[list])
     | {sym: {tf: []}} |
     | max 500/sym/tf  |
     +--------+--------+
              |
              | _write_buffer → flush DB (5s)
              |
              | _callbacks → on_candle(symbol, tf, candle)
              |
     +--------v---------+
     |    Simulator      |   (simulator.py)
     +--------+----------+
              |
              | _dispatch_candle()
              |
   +----------+-----------+
   |                      |
   | indicator_engine     |
   | .update()            |
   |                      |
   v                      v
+--+--+   +---------+   +-----------+
| LSR |   | GSR #1  |   | GSR #N    |  (LiveStrategyRunner / GridStrategyRunner)
+--+--+   +----+----+   +-----+-----+
   |           |               |
   |    on_candle() → compute_grid() → check entries → check exits
   |           |               |
   |    +------v------+  +----v-----+
   |    | GridPosition|  | GridPos  |
   |    | Manager     |  | Manager  |
   |    +------+------+  +----+-----+
   |           |               |
   v           v               v
+--+-----------+---------------+----+
|         TradeResult (net_pnl)     |
+--+--------------------------------+
   |              |
   |    +---------v----------+
   |    |   StateManager     |   (state_manager.py)
   |    |   save toutes      |
   |    |   les 60s           |
   |    +----+----+----------+
   |         |    |
   |  JSON file   DB (trades, snapshots)
   |
   +-----------> TradeEvent → Executor → Bitget (ordres réels)
   |
   +-----------> WebSocket /ws/live → Frontend React (3s push)
```

---

## 1. Boot Sequence (Lifespan)

Le démarrage est orchestré par `lifespan()` dans [server.py](../backend/api/server.py). Ordre strict :

### Startup

```
1. Database           → db.init() (SQLite WAL mode)
2. JobManager         → job_manager.start() (worker loop WFO)
3. Telegram + Notifier → TelegramClient + Notifier (si token configuré)
4. DataEngine         → engine.start() (WebSocket ccxt, si ENABLE_WEBSOCKET=true)
5. Simulator          → Simulator(data_engine, config, db)
   5a. StateManager   → state_manager.load_runner_state() (crash recovery)
   5b. simulator.set_notifier(notifier)
   5c. simulator.start(saved_state) → crée runners, restore, warm-up, câble callback
   5d. state_manager.start_periodic_save(simulator) → boucle 60s
6. Arena              → StrategyArena(simulator) (classement)
7. Executor           → (si LIVE_TRADING=true)
   7a. LiveRiskManager(config)
   7b. AdaptiveSelector(arena, config)
   7c. Executor(config, risk_mgr, notifier, selector)
   7d. state_manager.load_executor_state() → restore
   7e. executor.start(), selector.start()
   7f. simulator.set_trade_event_callback(executor.handle_event)
8. Watchdog           → watchdog.start() (checks 30s)
9. Heartbeat          → heartbeat.start() (Telegram toutes les heures)
10. notifier.notify_startup(strategies)
```

### Shutdown (ordre inverse)

```
1. notifier.notify_shutdown()
2. heartbeat.stop()
3. watchdog.stop()
4. selector.stop()
5. state_manager.save_executor_state() → executor.stop()
6. state_manager.save_runner_state() → state_manager.stop()
7. simulator.stop()
8. engine.stop() (flush final candles → DB)
9. job_manager.stop()
10. db.close()
```

### Variables partagées (app.state)

| Variable | Type | Description |
|----------|------|-------------|
| `app.state.db` | Database | Connexion SQLite |
| `app.state.engine` | DataEngine | null | Flux de données |
| `app.state.config` | AppConfig | Configuration YAML |
| `app.state.simulator` | Simulator | null | Paper trading |
| `app.state.arena` | StrategyArena | null | Classement stratégies |
| `app.state.executor` | Executor | null | Trading live |
| `app.state.selector` | AdaptiveSelector | null | Sélection stratégies |
| `app.state.notifier` | Notifier | Alertes Telegram |
| `app.state.watchdog` | Watchdog | null | Surveillance système |
| `app.state.job_manager` | JobManager | File d'attente WFO |

---

## 2. Flux de données (DataEngine)

Fichier : [data_engine.py](../backend/core/data_engine.py)

### Connexion WebSocket

```python
# ccxt.pro en mode swap (futures perpétuels)
self._exchange = ccxtpro.bitget({
    "enableRateLimit": True,
    "options": {"defaultType": "swap"},
})
```

Au `start()`, une tâche asyncio est créée par asset :
```
asyncio.create_task(_watch_symbol(symbol, timeframes))
```

Les souscriptions sont staggerées par batch de 10 avec 0.5s de délai entre les batchs (anti rate-limit Bitget code 30006).

### Réception des candles

`_watch_symbol()` → `_subscribe_klines()` → `watch_ohlcv()` (ccxt) → `_on_candle_received()`

Pour chaque candle reçue :

1. **Parsing** : timestamp ms → Candle dataclass
2. **Validation** (DataValidator) :
   - `low <= high`, `volume >= 0`, `open > 0`, `close > 0`
   - Pas de doublon (check 5 dernières candles)
   - Détection gap (> 1.5× la durée attendue)
3. **Buffer** : ajout au buffer rolling `_buffers[symbol][timeframe]`, borné à `MAX_BUFFER_SIZE = 500`
4. **Write buffer** : ajout à `_write_buffer` (flush DB toutes les 5s par `_flush_candle_buffer()`)
5. **Callbacks** : appel de tous les callbacks enregistrés via `on_candle(callback)`

### Structure du buffer

```python
_buffers: dict[str, dict[str, list[Candle]]]
# Exemple: _buffers["BTC/USDT"]["1h"] = [Candle, Candle, ...] (max 500)
```

### Données additionnelles

| Donnée | Source | Fréquence | Stockage |
|--------|--------|-----------|----------|
| Funding rate | `fetch_funding_rate()` | 5 min | `_funding_rates[symbol]` (float, en %) |
| Open Interest | `fetch_open_interest()` | 60s | `_open_interest[symbol]` (list[OISnapshot], max 60) |

### Gestion des erreurs

- **Rate limit** (codes 30006, 429) : retry 3× avec backoff `2s × attempt`, puis backoff global exponentiel
- **Symbol invalide** ("does not have market symbol") : abandon immédiat, log WARNING
- **Toute autre erreur** : `await asyncio.sleep(1.0)` pour yield à l'event loop (évite affamement)
- **Reconnexion** : backoff exponentiel `delay × 2^attempt` (max 60s), jusqu'à `max_reconnect_attempts`

---

## 3. Flux de trading — Paper (Simulator)

Fichier : [simulator.py](../backend/backtesting/simulator.py)

### Initialisation (start)

`Simulator.start(saved_state)` :

1. `get_enabled_strategies(config)` → liste des stratégies `enabled: true` dans `strategies.yaml`
2. Crée `IncrementalIndicatorEngine(strategies)` — moteur d'indicateurs partagé
3. Crée `PositionManager` (fees, slippage, leverage)
4. Pour chaque stratégie :
   - Si grid (`is_grid_strategy()`) → `GridStrategyRunner` + `GridPositionManager`
   - Sinon → `LiveStrategyRunner` + `PositionManager` partagé
5. Cleanup orphelins (stratégies désactivées avec positions sauvegardées)
6. `runner.restore_state(state)` pour chaque runner (crash recovery)
7. Warm-up grid runners : `runner._warmup_from_db(db, symbol)` — charge max 200 candles depuis la DB
8. Restaure le kill switch global si sauvegardé
9. **Câblage** : `self._data_engine.on_candle(self._dispatch_candle)` — c'est ce qui connecte le flux

### Dispatch candle (_dispatch_candle)

Appelé par le DataEngine à chaque candle reçue :

```
_dispatch_candle(symbol, timeframe, candle)
    |
    +-- Si kill switch global → return
    |
    +-- indicator_engine.update(symbol, timeframe, candle)  # UNE seule fois
    |
    +-- Pour chaque runner :
    |       runner.on_candle(symbol, timeframe, candle)
    |       |
    |       +-- Drain pending_events → Executor callback
    |       +-- Drain journal_events → DB
    |       +-- Détection collision (même symbol, runners différents)
    |
    +-- Snapshot capital + check kill switch global
```

### GridStrategyRunner.on_candle() — le coeur du système

Fichier : [simulator.py:695](../backend/backtesting/simulator.py) (classe GridStrategyRunner)

```
on_candle(symbol, timeframe, candle)
    |
    +-- Filtre : seul le timeframe de la stratégie (ex: "1h") est traité
    |
    +-- Détection fin warm-up : candle_age <= 2h → _end_warmup()
    |
    +-- Mise à jour buffer closes + SMA interne
    |
    +-- Merge indicateurs :
    |     indicators[tf]["sma"] = SMA calculée
    |     indicators[tf]["close"] = candle.close
    |     + compute_live_indicators() → indicateurs extra (EMA, ADX, Supertrend 4h)
    |
    +-- Construire StrategyContext (symbol, timestamp, indicators, capital)
    |
    +-- Construire GridState via gpm.compute_grid_state(positions, close)
    |
    +-- 1. SI POSITIONS OUVERTES :
    |     +-- get_tp_price() / get_sl_price() → TP/SL dynamiques
    |     +-- gpm.check_global_tp_sl(positions, candle, tp, sl)  # OHLC heuristic
    |     +-- strategy.should_close_all(ctx, grid_state)         # signal (direction_flip, etc.)
    |     +-- Si exit :
    |           +-- Restituer marge (capital += margin)
    |           +-- gpm.close_all_positions() → TradeResult
    |           +-- _record_trade() → capital += net_pnl, DB, kill switch check
    |
    +-- 2. SI GRILLE PAS PLEINE :
          +-- strategy.compute_grid(ctx, grid_state) → list[GridLevel]
          +-- Pour chaque level non rempli :
                +-- Touché ? LONG: candle.low <= entry_price / SHORT: candle.high >= entry_price
                +-- Sizing : margin_per_level = capital / nb_assets / num_levels (cap 25%)
                +-- Margin guard : total_margin < capital × max_margin_ratio (70%)
                +-- Global margin guard (portfolio mode) : marge totale tous runners
                +-- gpm.open_grid_position(level, timestamp, capital, max_positions)
                +-- capital -= margin (notional / leverage)
```

### LiveStrategyRunner.on_candle() (stratégies mono-position)

Flux plus simple :

```
on_candle(symbol, timeframe, candle)
    |
    +-- Récupérer indicateurs depuis l'engine
    +-- Build StrategyContext + extra_data (funding, OI)
    +-- Détecter régime de marché
    |
    +-- SI POSITION OUVERTE :
    |     +-- Régime change RANGING→TRENDING → close (regime_change)
    |     +-- pm.check_position_exit(candle, position, strategy, ctx) → TP/SL/signal_exit
    |
    +-- SI PAS DE POSITION :
          +-- strategy.evaluate(ctx) → StrategySignal | None
          +-- pm.open_position(signal, timestamp, capital)
```

### Kill switch

Deux niveaux :
- **Par runner** : perte session > `max_session_loss_percent` → runner stoppé
- **Global** : drawdown fenêtre glissante > `global_max_loss_pct` sur `global_window_hours` → tous les runners stoppés, alerte Telegram

Grace period : pas de check global pendant 1h après la fin du warm-up.

---

## 4. Flux de trading — Live (Executor)

Fichier : [executor.py](../backend/execution/executor.py)

### Flux signal → ordre Bitget

```
Simulator → TradeEvent → executor.handle_event()
    |
    +-- TradeEvent.OPEN :
    |     +-- LiveRiskManager : pre-trade check
    |     +-- _is_grid_strategy() ? dispatch grid : dispatch mono
    |     +-- Grid : SL server-side (place/cancel sur Bitget), TP client-side (SMA dynamique)
    |     +-- Mono : TP + SL server-side
    |     +-- ccxt : create_order() sur Bitget mainnet
    |
    +-- TradeEvent.CLOSE :
          +-- ccxt : close_position() sur Bitget
          +-- Annuler ordres SL/TP pendants
```

### AdaptiveSelector

Fichier : [adaptive_selector.py](../backend/execution/adaptive_selector.py)

- Classe l'Arena par performance (Sharpe, win rate, net P&L)
- Alloue plus de capital aux top performers
- Pause les stratégies sous-performantes
- Réévalue périodiquement

### Réconciliation au boot

Au démarrage, l'Executor :
1. Charge son état sauvegardé (`executor_state.json`)
2. Récupère les positions ouvertes sur Bitget via ccxt
3. Réconcilie : positions locales vs Bitget
4. Grid : `_reconcile_grid_symbol` pour chaque symbol

---

## 5. Flux de backtest

### Single-asset (WFO)

Fichier : [fast_multi_backtest.py](../backend/optimization/fast_multi_backtest.py)

Optimisé pour la vitesse (100-1000× plus rapide que le backtester standard) :

1. **Pré-calcul** : IndicatorCache calcule tous les indicateurs sur tout le dataset (vectorisé numpy)
2. **Boucle** : itère les candles, check les niveaux de grille, accumule les trades
3. **Pas d'async** : pur Python synchrone, exécuté dans ProcessPoolExecutor (4 workers)
4. **Numba JIT** (optionnel) : accélère 5-10× les boucles indicateurs + trades

```
IndicatorCache.build(candles, params)
    → sma_arr, atr_arr, bb_*, supertrend, ema, adx, atr_by_period
    |
_simulate_grid(closes, highs, lows, cache, params)
    → itère chaque candle
    → check entry levels (SMA ± ATR × multiplier)
    → check TP (retour SMA) / SL (% prix moyen)
    → accumule TradeResult[]
    |
BacktestResult (net_return_pct, sharpe, win_rate, trades, ...)
```

### Portfolio (multi-asset)

Fichier : [portfolio_engine.py](../backend/backtesting/portfolio_engine.py)

```
PortfolioBacktester.run(start, end)
    |
    1. Charger candles 1h depuis DB (tous assets)
    2. Créer N GridStrategyRunners (capital = initial / N)
       - Clé runner : "strategy:symbol" (ex: "grid_atr:BTC/USDT")
       - _portfolio_mode = True → sizing fixe sur _initial_capital
    3. Warm-up : 50 premières candles par symbol → indicator_engine.update()
    4. Merge candles (tous symbols) trié par timestamp
    5. _simulate() :
       - Pour chaque candle :
         - Identifier le symbol → trouver les runners concernés
         - indicator_engine.update(symbol, "1h", candle)
         - runner.on_candle(symbol, "1h", candle) pour chaque runner du symbol
         - Snapshot equity/margin/positions
         - Kill switch temps réel : DD fenêtre 24h → freeze, dégel après 24h
    6. Force-close positions restantes
    7. Build PortfolioResult
```

**Particularités portfolio** :
- `_portfolio_mode = True` : sizing fixe (pas de compounding)
- `_portfolio_runners` dict partagé : check marge globale < `max_margin_ratio` (70%)
- Kill switch actif : freeze/dégel tous runners avec fenêtre glissante 24h
- Multi-stratégie : plusieurs runners par symbol possible (clé `strategy:symbol`)

---

## 6. Persistence

### StateManager

Fichier : [state_manager.py](../backend/core/state_manager.py)

| Quoi | Fichier | Fréquence | Méthode |
|------|---------|-----------|---------|
| État runners (capital, stats, positions) | `data/simulator_state.json` | 60s | `save_runner_state()` |
| État executor (positions Bitget) | `data/executor_state.json` | Au shutdown | `save_executor_state()` |
| Snapshots journal | DB `portfolio_snapshots` | 5 min | `_save_journal_snapshot()` |

**Écriture atomique** : écrit dans `.tmp` puis `os.replace()` (pas de corruption si crash).

**I/O async** : toutes les opérations fichier via `asyncio.to_thread()` (ne bloque pas l'event loop).

### Structure simulator_state.json

```json
{
  "saved_at": "2026-02-17T12:00:00+00:00",
  "global_kill_switch": false,
  "runners": {
    "grid_atr": {
      "capital": 9850.50,
      "net_pnl": -149.50,
      "realized_pnl": -149.50,
      "total_trades": 12,
      "wins": 7,
      "losses": 5,
      "kill_switch": false,
      "is_active": true,
      "position": null,
      "position_symbol": null,
      "grid_positions": [
        {
          "symbol": "BTC/USDT",
          "level": 0,
          "direction": "long",
          "entry_price": 95000.0,
          "quantity": 0.001,
          "entry_time": "2026-02-17T10:00:00+00:00",
          "entry_fee": 0.057
        }
      ]
    }
  }
}
```

### Tables DB principales

| Table | Contenu | Écrit par |
|-------|---------|-----------|
| `candles` | OHLCV historiques | DataEngine (flush 5s) |
| `simulation_trades` | Trades paper fermés | GridStrategyRunner / LiveStrategyRunner |
| `optimization_results` | Résultats WFO | walk_forward.py |
| `wfo_combo_results` | Tous les combos WFO (heatmap) | walk_forward.py |
| `optimization_jobs` | Jobs explorateur | JobManager |
| `portfolio_backtests` | Résultats portfolio | portfolio_db.py |
| `portfolio_snapshots` | Snapshots journal (5 min) | StateManager |
| `position_events` | Événements positions (Sprint 25) | Simulator |

---

## 7. Frontend (WebSocket)

Fichier : [websocket_routes.py](../backend/api/websocket_routes.py)

### Endpoint

`/ws/live` — push toutes les **3 secondes**

### Données envoyées

```json
{
  "type": "update",
  "strategies": [...],           // get_all_status() — capital, P&L, trades par runner
  "kill_switch": false,          // kill switch global
  "simulator_positions": [...],  // positions ouvertes
  "grid_state": {...},           // niveaux de grille actifs par symbol
  "ranking": [...],              // Arena : classement par net_pnl
  "executor": {...},             // statut executor live
  "prices": {                    // prix live par symbol
    "BTC/USDT": {"last": 95000.0, "change_pct": -0.5},
    ...
  }
}
```

### ConnectionManager

- `connect()` : accepte la connexion, ajoute à la liste
- `disconnect()` : retire de la liste
- `broadcast()` : envoie à tous les clients (retire les déconnectés)
- Utilisé aussi par le JobManager pour les mises à jour de progression WFO

### Composants React principaux

| Composant | Données consommées |
|-----------|-------------------|
| Scanner | prices, strategies, grid_state |
| GridDetail | grid_state (niveaux, TP/SL, P&L par symbol) |
| EquityCurve | equity snapshots (endpoint REST) |
| ActivePositions | simulator_positions |
| ArenaRanking | ranking |
| HeatmapChart | wfo_combo_results (endpoint REST) |
| PortfolioPage | portfolio_backtests (endpoint REST) |

---

## 8. Monitoring

### Watchdog

Fichier : [watchdog.py](../backend/monitoring/watchdog.py)

Checks toutes les **30 secondes** :

| Check | Anomalie | Cooldown |
|-------|----------|----------|
| WebSocket connecté | `WS_DISCONNECTED` | 30 min |
| Data freshness (< 5 min) | `DATA_STALE` | 30 min |
| Stratégies actives | `ALL_STOPPED` | 1h |
| Kill switch déclenché | `KILL_SWITCH` | 1h |
| SL manquant (position sans SL) | `MISSING_SL` | 5 min |

Anti-spam : cooldown par type d'anomalie. Le log WARNING est toujours émis, seul l'envoi Telegram est throttlé.

### Heartbeat

Fichier : [heartbeat.py](../backend/alerts/heartbeat.py)

- Envoi Telegram toutes les **heures** (configurable `heartbeat_interval`)
- Contenu : nombre de stratégies actives, positions ouvertes, P&L session, uptime

### Notifier

Fichier : [notifier.py](../backend/alerts/notifier.py)

Cooldown par type d'anomalie (Sprint 20d) :
- `MISSING_SL` : 5 min
- `WS_DISCONNECTED`, `DATA_STALE` : 30 min
- `KILL_SWITCH`, `KILL_SWITCH_GLOBAL`, `ALL_STOPPED` : 1h

---

## 9. Diagramme récapitulatif — Chemin d'une candle

```
Bitget WS
    |
    v
DataEngine._subscribe_klines()
    | watch_ohlcv(symbol, tf)
    v
DataEngine._on_candle_received()
    | validate → buffer → write_buffer → callbacks
    v
Simulator._dispatch_candle(symbol, tf, candle)
    | indicator_engine.update()
    v
GridStrategyRunner.on_candle()
    |
    +-- compute_live_indicators() (EMA, ADX, Supertrend 4h)
    +-- strategy.compute_grid(ctx, grid_state) → niveaux
    +-- candle.low <= entry_price ? → open_grid_position()
    +-- candle touches TP/SL ? → close_all_positions() → TradeResult
    |
    +-- _record_trade()
    |     +-- capital += net_pnl
    |     +-- DB: INSERT simulation_trades (via thread)
    |     +-- check kill switch runner
    |
    +-- _emit_event() → Executor → ccxt → Bitget (ordres réels)
    |
    v
StateManager (60s) → data/simulator_state.json
WebSocket /ws/live (3s) → Frontend React
```
