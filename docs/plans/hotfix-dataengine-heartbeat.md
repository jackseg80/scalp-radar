# Hotfix DataEngine Heartbeat — Détection silence WS + auto-reconnect

## Contexte

Le DataEngine `watch_ohlcv` peut se déconnecter silencieusement ("connection closed" sans exception captée).
La boucle de reconnexion `_watch_symbol` ne se déclenche pas. Le bot continue à checker des prix figés.
Position live ouverte, bot aveugle, seul le SL server-side Bitget protège.

## Fichiers modifiés

| Fichier | Rôle |
|---------|------|
| `backend/core/data_engine.py` | Heartbeat loop + notifier + timer |
| `backend/api/server.py` | Passer notifier au DataEngine |
| `tests/test_dataengine_heartbeat.py` | 9 tests (créé) |

## Implémentation

### DataEngine.__init__

```python
def __init__(self, config, database, notifier=None):
    ...
    self._notifier = notifier
    self._last_candle_received: float = time.time()
    self._heartbeat_interval: int = 300  # 5 minutes
    self._heartbeat_task: asyncio.Task | None = None
```

### _on_candle_received (ligne ~515)

Après `self._last_update = datetime.now(...)` :
```python
self._last_candle_received = time.time()
```

### _heartbeat_loop (nouvelle méthode)

- Check toutes les 60s
- Si `elapsed > 300s` → WARNING + `notify_anomaly(DATA_STALE)` + `full_reconnect()` + reset timer
- Sinon → DEBUG "heartbeat OK"
- `CancelledError` → break proprement
- Erreur `full_reconnect` → log ERROR mais boucle continue

### start()

```python
self._last_candle_received = time.time()
self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="heartbeat")
```

### stop()

Cancel `_heartbeat_task` avant flush/autres tâches.

### server.py

```python
engine = DataEngine(config, db, notifier=notifier)
```

## Tests (9)

1. `test_heartbeat_triggers_reconnect_on_silence` — silence 310s → reconnect
2. `test_heartbeat_sends_telegram_alert_on_silence` — silence → DATA_STALE
3. `test_heartbeat_no_reconnect_when_candles_fresh` — candle 10s ago → pas de reconnect
4. `test_heartbeat_reconnect_failure_doesnt_crash_loop` — reconnect plante → boucle continue
5. `test_heartbeat_resets_timer_after_reconnect` — timer reset après reconnect
6. `test_heartbeat_no_telegram_when_notifier_none` — notifier=None → pas d'AttributeError
7. `test_notifier_stored_in_init` — notifier bien stocké
8. `test_no_notifier_by_default` — défaut = None
9. `test_last_candle_received_initialized` — timer init au boot

## Résultat

1535 tests, 0 régression.
