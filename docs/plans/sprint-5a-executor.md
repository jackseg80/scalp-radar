# Sprint 5a — Executor Live Trading Minimal

## Contexte

Le Sprint 5a ajoute l'exécution d'ordres réels Bitget au-dessus du Simulator existant. Scope volontairement réduit : **1 stratégie (VWAP+RSI), 1 paire (BTC/USDT:USDT), capital minimal**. Un bug dans l'executor = perte d'argent réel, donc on battle-teste le pipeline avant d'ajouter de la complexité.

**Design** : Pattern observer — le `LiveStrategyRunner` émet un callback quand il ouvre/ferme une position virtuelle, l'`Executor` écoute et réplique en ordres réels sur Bitget.

---

## Fichiers CRÉÉS

### 1. `backend/execution/executor.py`

Classe `Executor` — orchestrateur principal :

- **`TradeEvent`** (dataclass) + **`TradeEventType`** (enum OPEN/CLOSE) — événements émis par le runner
  - Champs : event_type, strategy_name, symbol, direction, entry_price, quantity, tp_price, sl_price, score, timestamp, **market_regime**, exit_reason (CLOSE), exit_price (CLOSE)
- **`LivePosition`** (dataclass) — position live ouverte (order_id, sl_order_id, tp_order_id)
- **`to_futures_symbol(spot) → str`** — mapping `BTC/USDT` → `BTC/USDT:USDT` (ne PAS changer assets.yaml)
- **`Executor.__init__(config, risk_manager, notifier)`** — crée l'instance, pas encore connectée
- **`start()`** :
  1. Créer ccxt.pro.bitget authentifié (`defaultType: "swap"`)
  2. `exchange.load_markets()` — cache les min_order_size, tick_size réels de Bitget
  3. Vérifier positions ouvertes via `fetch_positions()`
  4. Set leverage UNIQUEMENT s'il n'y a pas de position ouverte (sinon log warning et garder le leverage existant)
  5. Set margin mode (cross)
  6. Réconcilier positions avec état sauvegardé
  7. Lancer `_watch_orders_loop()` (principal) + `_poll_positions_loop()` (fallback)
- **`stop()`** — stoppe watchers/polling, sauvegarde état. NE ferme PAS les positions (TP/SL restent sur exchange)
- **`handle_event(event: TradeEvent)`** — callback du Simulator, filtre VWAP+RSI + BTC uniquement
- **`_open_position(event)`** :
  1. `risk_manager.pre_trade_check()` — marge, kill switch, max positions
  2. Convertir symbol via `to_futures_symbol()`
  3. Arrondir quantité via `_round_quantity()` (utilise données `load_markets()`)
  4. **Market order** d'entrée (taker 0.06%, cohérent avec le backtester/simulator)
  5. Attendre confirmation fill (timeout 30s)
  6. Placer SL (market trigger order, `triggerType: "mark_price"`, `reduceOnly: true`)
  7. **SI le SL échoue** : retry 2x avec `await asyncio.sleep(0.2)` entre chaque → si toujours échec, **CLOSE MARKET IMMÉDIAT** + alerte Telegram urgente. JAMAIS de position sans SL.
  8. Placer TP (limit trigger order, `reduceOnly: true`)
  9. Si le TP échoue : log warning (moins critique, le SL protège)
  10. `await asyncio.sleep(0.1)` entre chaque ordre (rate limiting)
  11. Enregistrer `LivePosition`, notifier Telegram
- **`_close_position(event)`** — annule SL/TP pending → market close → notify → record trade
- **`_watch_orders_loop()`** — **mécanisme principal** : `exchange.watch_orders()` via ccxt Pro pour détecter les fills TP/SL en quasi temps réel
- **`_poll_positions_loop()`** — **fallback de sécurité** toutes les 5s : `fetch_positions()` pour vérifier que le state est synchronisé (rattrape tout ce que watchOrders aurait manqué)
- **`_reconcile_on_boot()`** — fetch_positions Bitget, compare avec état sauvegardé, **4 cas** :
  1. Position exchange + état local → OK, reprendre le suivi
  2. Position exchange, PAS d'état local → position orpheline, alerter, ne pas toucher
  3. PAS de position exchange, état local présent → fermée pendant downtime (TP/SL hit ou liquidée). **Fetch P&L réel** via `fetchClosedOrders()` / `fetchMyTrades()`, mettre à jour risk_manager, notifier
  4. Aucune position des deux côtés → clean, rien à faire
- **`_round_quantity(qty, symbol)`** — arrondi à `min_order_size` de `load_markets()` (valeurs réelles Bitget, pas config)
- **`get_status() → dict`** — pour /health et dashboard

Instance ccxt **Pro** séparée de celle du DataEngine (authentifiée, pour ordres + watchOrders).

### 2. `backend/execution/risk_manager.py`

Classe `LiveRiskManager` — gardien pré/post-trade :

- **`pre_trade_check(symbol, direction, quantity, entry_price, free_margin, total_balance) → (bool, str)`** — vérifie : kill switch, position déjà ouverte, max concurrent, marge suffisante
- **`register_position(position)`** / **`unregister_position(symbol)`** — tracking positions ouvertes
- **`record_trade_result(result)`** — met à jour session P&L, vérifie kill switch live (>= 5% perte)
- **`get_state() → dict`** / **`restore_state(state)`** — persistance pour crash recovery
- **`is_kill_switch_triggered`** (property)

Double kill switch : le Simulator a le sien (virtuel), le RiskManager a le sien (argent réel).

### 3. `tests/test_executor.py` (32 tests)

- Symbol mapping (spot → futures, symbole inconnu → ValueError)
- Event filtering (seul vwap_rsi + BTC accepté, autres ignorés)
- Ouverture position (create_order market, SL/TP trigger orders placés)
- **Rollback SL échoué** : si SL échoue après retries → close market immédiat
- TP échoué : log warning, position gardée (SL protège)
- Échec entry order : pas de SL/TP placés
- Quantité arrondie via données load_markets()
- Fermeture position (cancel SL/TP + market close)
- Réconciliation au boot (4 scénarios dont P&L fetch)
- Leverage : pas de set_leverage si position ouverte
- Lifecycle (start/stop)
- Mock ccxt via AsyncMock

### 4. `tests/test_risk_manager.py` (16 tests)

- Pre-trade check OK / rejeté (kill switch, position dupliquée, max positions, marge)
- Kill switch : déclenché à >= 5%, accumulation pertes
- Position tracking : register/unregister
- State persistence : get_state/restore_state round-trip

---

## Fichiers MODIFIÉS

### 5. `backend/backtesting/simulator.py`

Callback `TradeEvent` dans `LiveStrategyRunner` :

- `_pending_events: list` — queue d'événements
- Après ouverture position dans `on_candle` : `_emit_open_event()` → `_pending_events.append(TradeEvent(OPEN, ...))`
- Dans `_record_trade` : `_emit_close_event()` → `_pending_events.append(TradeEvent(CLOSE, ...))`
- Param `symbol` ajouté à `_record_trade`
- Drain via swap atomique `events, self._pending_events = self._pending_events, []`

`Simulator` :

- `_trade_event_callback: Callable | None`
- `set_trade_event_callback(callback)` — appelé par le lifespan
- Dans `_dispatch_candle` : drain `runner._pending_events` via swap atomique → callback

### 6. `backend/api/server.py`

Section 4b dans le lifespan (après Simulator, avant Watchdog) :

```python
if config.secrets.live_trading and engine and simulator:
    risk_mgr = LiveRiskManager(config)
    executor = Executor(config, risk_mgr, notifier)
    executor_state = await state_manager.load_executor_state()
    if executor_state:
        risk_mgr.restore_state(executor_state.get("risk_manager", {}))
        executor.restore_position(executor_state)
    await executor.start()
    simulator.set_trade_event_callback(executor.handle_event)
```

Shutdown : `state_manager.save_executor_state()` + `executor.stop()` avant `simulator.stop()`

### 7. `backend/core/config.py`

Ajout dans `SecretsConfig` :
```python
live_trading: bool = False  # LIVE_TRADING env var, défaut false = simulation only
```

### 8. `backend/alerts/notifier.py`

Nouvelles méthodes :
- `notify_live_order_opened(symbol, direction, qty, entry, sl, tp, strategy, order_id)`
- `notify_live_order_closed(symbol, direction, entry, exit, net_pnl, reason, strategy)`
- `notify_live_sl_failed(symbol, strategy)` — alerte urgente
- `notify_reconciliation(result: str)`

Nouveaux `AnomalyType` :
- `EXECUTOR_DISCONNECTED`, `KILL_SWITCH_LIVE`, `SL_PLACEMENT_FAILED`

### 9. `backend/alerts/telegram.py`

Nouvelles méthodes :
- `send_live_order_opened(...)` — format HTML structuré
- `send_live_order_closed(...)` — format HTML avec P&L
- `send_live_sl_failed(...)` — alerte urgente formatée

### 10. `backend/core/state_manager.py`

- `_executor_state_file` param (défaut `data/executor_state.json`)
- `save_executor_state(executor, risk_manager)` — écriture atomique
- `load_executor_state() → dict | None` — lecture robuste

### 11. `backend/monitoring/watchdog.py`

- Param optionnel `executor=None` dans `__init__`
- Checks : executor connecté ?, kill switch live ?

### 12. `.env.example`

```
LIVE_TRADING=false
```

### 13. `backend/api/health.py`

Section `executor` dans la réponse `/health` (status, positions, kill switch live).

---

## Décisions clés

1. **Symboles inchangés dans assets.yaml** — mapping `BTC/USDT` → `BTC/USDT:USDT` fait dans executor.py uniquement, évite toute régression
2. **Deux instances ccxt Pro** — DataEngine (non authentifié, lecture) vs Executor (authentifié, ordres + watchOrders)
3. **Graceful shutdown = positions ouvertes** — TP/SL server-side protègent même si bot down
4. **Entry = market order** (taker 0.06%) — cohérent avec le backtester et le simulator qui modélisent l'entry comme taker. Évite la divergence systématique d'un limit order non fill en scalping rapide
5. **SL = market trigger** (garanti), **TP = limit trigger** (maker fee)
6. **JAMAIS de position sans SL** — si le placement SL échoue après 2 retries, close market immédiat + alerte urgente Telegram. C'est la règle de sécurité #1
7. **watchOrders() comme mécanisme principal** de détection des fills TP/SL (quasi temps réel via ccxt Pro). Polling 5s en fallback de sécurité uniquement
8. **Callback via queue + swap atomique** — `_pending_events` dans le runner (sync), drainés par le Simulator (async) via `events, self._pending = self._pending, []`
9. **load_markets() au start** — cache les min_order_size/tick_size réels de Bitget (pas les valeurs config qui peuvent être obsolètes)
10. **Leverage : vérifier avant de changer** — ne set le leverage que s'il n'y a PAS de position ouverte (changer le leverage avec une position ouverte peut déclencher une liquidation)
11. **Rate limiting entre ordres** — `await asyncio.sleep(0.1)` entre entry, SL et TP pour éviter de taper les rate limits Bitget
12. **Réconciliation 4 cas** — dont fetch P&L réel via `fetchClosedOrders()` quand une position a été fermée pendant le downtime
13. **Testnet Bitget obligatoire** — valider le pipeline complet sur le demo trading avant mainnet

---

## Résultats

- **248 tests passent** (200 existants + 32 executor + 16 risk_manager)
- Aucune régression sur les tests existants
- `LIVE_TRADING=false` (défaut) → aucun changement de comportement
- `LIVE_TRADING=true` sans DataEngine → warning loggé, executor non créé

---

## Prochaines étapes (Sprint 5b)

1. Validation sur testnet Bitget (`sandbox: true` dans exchanges.yaml)
2. Adaptive strategy selector (allocation capital basée sur performance)
3. Extension à 3 paires (BTC, ETH, SOL)
4. Extension à 4 stratégies en parallèle
5. Rollout progressif du capital
