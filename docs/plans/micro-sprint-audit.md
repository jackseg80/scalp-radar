# Micro-Sprint Audit — 3 fixes (auth + async I/O + candle buffer)

## Contexte

Un audit a identifié 3 problèmes confirmés par vérification du code source.

## Fix 1 — Auth sur endpoints executor (CRITIQUE)

**Fichier** : `backend/api/executor_routes.py`

**Problème** : POST /api/executor/test-trade et POST /api/executor/test-close
n'ont aucune authentification. En LIVE_TRADING=true, n'importe qui sur le réseau
peut passer des ordres réels.

**Solution** :
- Dépendance FastAPI `verify_executor_key` qui vérifie le header `X-API-Key`
  contre `config.secrets.sync_api_key` (même clé que optimization_routes)
- Appliquée sur les 3 routes du router executor (test-trade, test-close, status)
- Si la clé n'est pas configurée côté serveur -> 401 "API key non configurée"
- Si la clé est absente ou mauvaise -> 401 "API key invalide"

**Tests** : 6 tests dans `tests/test_executor_routes.py`

## Fix 2 — Async I/O dans StateManager

**Fichier** : `backend/core/state_manager.py`

**Problème** : `save_runner_state()` et `save_executor_state()` font json.dump()
+ os.replace() synchrones dans des méthodes async. Bloque l'event loop toutes
les 60 secondes.

**Solution** :
- `_write_json_file(file_path, data)` : mkdir + open + json.dump + os.replace
- `_read_json_file(file_path)` : exists + open + json.load -> None si absent/corrompu
- Les 4 méthodes async appellent ces helpers via `asyncio.to_thread()`
- Pas de dépendance aiofiles ajoutée

**Tests** : 3 nouveaux tests dans `tests/test_state_manager.py` (19 total)

## Fix 3 — Buffer d'écriture candles dans DataEngine

**Fichier** : `backend/core/data_engine.py`

**Problème** : `_on_candle_received()` appelle `insert_candles_batch([candle])`
avec une liste d'un seul élément -> 1 COMMIT par candle.

**Solution** :
- `_write_buffer: list[Candle]` dans DataEngine.__init__
- `_on_candle_received` : append au buffer au lieu d'insérer immédiatement
- `_flush_candle_buffer()` : flush toutes les 5 secondes
- `stop()` : flush final avant fermeture DB
- Les callbacks (Simulator) restent immédiats

**Tests** : 5 tests dans `tests/test_data_engine_buffer.py`

## Résultat

- 990 -> 1004 tests (+14 nouveaux, 0 régression)
- 3 fichiers modifiés, 2 fichiers de test créés
