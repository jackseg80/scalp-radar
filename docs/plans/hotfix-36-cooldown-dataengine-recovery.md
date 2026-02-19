# Hotfix 36 — Cooldown par temps + DataEngine auto-recovery

## Contexte critique

Le bot live ouvre des positions incontrôlées à chaque restart et perd sa connexion WebSocket sans la récupérer. C'est la priorité #1.

### Bug 1 — Le cooldown du Hotfix 35 ne fonctionne pas

Le compteur `_post_warmup_candle_count` s'incrémente **par appel `on_candle()`**, soit une fois par (symbol × candle). Avec 22 symbols et 50 bougies de warm-up par symbol, le batch initial de `watch_ohlcv()` génère des dizaines d'appels en quelques secondes. Le cooldown de 3 est épuisé instantanément.

**Preuve** : À 07:26:05, le runner ouvre des grids AVAX → dès la 4ème bougie historique le cooldown est expiré → events passent vers l'Executor → 5 ordres live à 08:27.

**Fix requis** : Cooldown basé sur le **temps réel écoulé depuis `_warmup_ended_at`**, pas sur un compteur.

### Bug 2 — DataEngine meurt sans reconnexion

`_watch_symbol()` a un `max_reconnect_attempts` (défaut: configurable). Après les retries, la boucle fait `break` → la tâche asyncio se termine → ce symbol ne reçoit plus JAMAIS de données. Si tous les symbols meurent (rate limit global), le DataEngine est mort.

Le Watchdog détecte `data_stale` et envoie une alerte Telegram mais **ne fait rien pour corriger**.

**Preuve** : À 00:44:25, rate limit sur 20+ symbols en même temps. Ensuite, `data_stale` toutes les 30 min pendant 7 heures. Zéro candle reçue = bot aveugle.

## Fix A — Cooldown par temps (simulator.py)

- Supprimé `POST_WARMUP_COOLDOWN = 3` (compteur) → `POST_WARMUP_COOLDOWN_SECONDS = 10800` (3h)
- Supprimé `_post_warmup_candle_count` du `__init__` → `_warmup_ended_at: datetime | None = None`
- `_emit_open_event()` et `_emit_close_event()` : guard basé sur `elapsed < POST_WARMUP_COOLDOWN_SECONDS`
- Supprimé l'incrément `_post_warmup_candle_count += 1` dans `on_candle()`
- Supprimé le reset dans `_end_warmup()` (plus de compteur)
- `_candles_since_warmup` conservé (utilisé par le kill switch grace period)

## Fix B — DataEngine auto-recovery (data_engine.py + watchdog.py)

### B1 — `_watch_symbol` : never give up
- Retiré `max_attempts` et le `if attempt >= max_attempts: break`
- Backoff exponentiel plafonné à 5 min (était 60s)
- Reset attempt counter après `attempt > 20` pour éviter overflow
- Reset attempt counter à 0 sur connexion réussie

### B2 — Stagger des souscriptions
- `_SUBSCRIBE_BATCH_SIZE` : 10 → 5
- `_SUBSCRIBE_BATCH_DELAY` : 0.5s → 2.0s
- Boot : 5 batchs × 2s = 10s (était 3 × 0.5s = 1.5s)

### B3 — `restart_dead_tasks()`
- Nouvelle méthode dans DataEngine
- Parcourt `_tasks`, relance les tâches `done()` et non `cancelled()`
- Extrait le symbol du nom de la tâche (`watch_BTC/USDT`)

### B4 — `full_reconnect()`
- Nouvelle méthode dans DataEngine
- Ferme l'exchange, recrée l'instance ccxt, relance toutes les tâches

### Watchdog auto-recovery
- `data_stale > 600s` (10 min) → `restart_dead_tasks()`
- `data_stale > 1800s` (30 min) + 0 tasks restarted → `full_reconnect()`

## Fichiers modifiés

| Fichier | Modification |
|---------|-------------|
| `backend/backtesting/simulator.py` | Cooldown par temps (4 zones modifiées) |
| `backend/core/data_engine.py` | Never give up + stagger + restart_dead_tasks + full_reconnect |
| `backend/monitoring/watchdog.py` | Auto-recovery sur data_stale |
| `tests/test_hotfix_35.py` | 7 tests adaptés au cooldown par temps |
| `tests/test_hotfix_36.py` | 12 nouveaux tests |
| `tests/test_grid_runner.py` | 1 ligne adaptée (cooldown passé) |

## Tests

**1374 tests passent** (1359 existants + 12 nouveaux + 3 modifiés dans hotfix_35)
