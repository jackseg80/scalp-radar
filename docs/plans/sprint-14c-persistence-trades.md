# Plan — Persistence des trades simulateur en SQLite

## Contexte

Les trades du simulateur (paper trading) sont stockés uniquement en mémoire (`_trades: list[tuple[str, TradeResult]]`). Après un restart, la liste est vidée mais les compteurs cumulatifs (`total_trades`, `wins`, `losses`, `net_pnl`) sont restaurés via le StateManager. Résultat : incohérence entre les stats ("40 trades") et l'historique visible (seulement les trades du boot courant). L'audit complet est impossible.

**Objectif** : persister chaque trade en DB SQLite au moment de sa clôture, et lire depuis la DB pour l'API frontend.

## Structure actuelle (résumé)

- **TradeResult** (`backend/core/position_manager.py:30`) — dataclass avec : `direction`, `entry_price`, `exit_price`, `quantity`, `entry_time`, `exit_time`, `gross_pnl`, `fee_cost`, `slippage_cost`, `net_pnl`, `exit_reason`, `market_regime`. Pas de `symbol`, `strategy_name`, `leverage`, `notional`.
- **LiveStrategyRunner._record_trade()** (`:233`) — reçoit `(trade, symbol)`, append à `self._trades`, met à jour stats/kill switch
- **GridStrategyRunner._record_trade()** (`:648`) — idem, avec `_realized_pnl` en plus
- **Simulator** (`:844`) — a `self._db: Database | None`, crée les runners dans `start()`
- **Simulator.get_all_trades()** (`:1156`) — itère `runner.get_trades()` en mémoire, retourne des dicts
- **GET /api/simulator/trades** (`simulator_routes.py:48`) — appelle `simulator.get_all_trades()[:limit]`
- **Database._create_tables()** (`database.py:101`) — chaîne d'appels pour créer les tables

## Schéma de la table

Adapté aux champs **réellement disponibles** dans TradeResult + contexte runner :

```sql
CREATE TABLE IF NOT EXISTS simulation_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    quantity REAL NOT NULL,
    gross_pnl REAL NOT NULL,
    fee_cost REAL NOT NULL,
    slippage_cost REAL NOT NULL,
    net_pnl REAL NOT NULL,
    exit_reason TEXT NOT NULL,
    market_regime TEXT,
    entry_time TEXT NOT NULL,
    exit_time TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sim_trades_strategy ON simulation_trades(strategy_name);
CREATE INDEX IF NOT EXISTS idx_sim_trades_exit_time ON simulation_trades(exit_time);
```

**Différences vs proposition** : pas de `notional`/`leverage`/`entry_fee`/`exit_fee`/`duration_seconds` (indisponibles dans TradeResult). On stocke `gross_pnl`, `fee_cost`, `slippage_cost` qui sont les champs réels. `duration_seconds` est calculable côté requête (`exit_time - entry_time`).

## Fichiers à modifier (6) + tests

### 1. `backend/core/database.py`

- Ajouter `_create_simulator_trades_table()` (async, pattern identique à `_create_sprint14b_tables`)
- Appeler depuis `_create_tables()` après `_create_sprint14_tables()`
- Ajouter méthode async `get_simulation_trades(strategy_name=None, limit=50)` pour la lecture API
- Ajouter méthode async `clear_simulation_trades()` pour le reset

### 2. `backend/backtesting/simulator.py`

**Fonction helper synchrone** (au top du module) :
```python
def _save_trade_to_db_sync(db_path: str, strategy_name: str, symbol: str, trade: TradeResult) -> None:
```
- Ouvre une connexion `sqlite3` synchrone, INSERT, ferme
- Try/except : log warning si échec (ne crashe jamais le runner)
- Backward compatible : si table absente → `OperationalError` catché → warning

**LiveStrategyRunner** :
- Ajouter `db_path: str | None = None` au `__init__()` (`:91`)
- Dans `_record_trade()` (`:233`), après l'append, appeler `_save_trade_to_db_sync` si `self._db_path`

**GridStrategyRunner** :
- Idem : `db_path: str | None = None` au `__init__()` (`:420`)
- Dans `_record_trade()` (`:648`), idem

**Simulator.start()** (`:989`) :
- Extraire `db_path = self._db.db_path if self._db else None`
- Passer `db_path` aux constructeurs des runners (`LiveStrategyRunner(..., db_path=db_path)` et `GridStrategyRunner(..., db_path=db_path)`)

**Simulator.get_all_trades()** (`:1156`) :
- Garder tel quel (reste la source mémoire pour l'equity curve interne, conditions, etc.)
- L'API trades utilisera la DB directement

### 3. `backend/api/simulator_routes.py`

**GET /api/simulator/trades** (`:48`) :
- Lire depuis la DB via `request.app.state.db.get_simulation_trades(limit=limit)`
- Fallback sur `simulator.get_all_trades()` si `db` absent (backward compat)
- Le format retourné reste **identique** (même clés de dict) — pas de changement frontend

### 4. `scripts/reset_simulator.py`

- Ajouter le vidage de la table `simulation_trades`
- Utiliser `sqlite3` synchrone (comme le reste du script)
- Avant le DELETE, afficher le nombre de trades en DB
- Le `db_path` est `data/scalp_radar.db` (même constante que Database)

### 5. `backend/backtesting/simulator.py` — Migration one-shot au start

Dans `Simulator.start()`, après la restauration des runners et avant le câblage DataEngine :
- Si `self._db` est disponible, vérifier si la table `simulation_trades` est vide
- Si vide ET des trades existent en mémoire (restaurés par un ancien état) → les insérer en DB
- Ceci ne concerne que les trades du boot courant (les pré-restart sont perdus, accepté)

En pratique : au tout premier démarrage avec cette feature, la table sera vide et `_trades` sera vide aussi (pas de migration à faire). Les trades pré-restart sont perdus — c'est le problème qu'on résout pour l'avenir.

### 6. `tests/test_simulator.py` — 4 tests

```
test_trade_persisted_to_db:
  - Créer un runner avec db_path temp
  - Enregistrer 1 trade via _record_trade()
  - Lire la DB directement → vérifier les champs

test_trades_survive_restart:
  - Enregistrer 5 trades
  - Vider _trades (simule restart)
  - Lire la DB → 5 trades

test_trades_ordered_by_exit_time:
  - 3 trades avec exit_time désordonnées
  - get_simulation_trades() → triés DESC par exit_time

test_reset_clears_trades:
  - Insérer des trades
  - Exécuter le DELETE (simuler reset_simulator)
  - Vérifier table vide
```

## Flux de données résultant

```
on_candle() → _record_trade()
                ├── self._trades.append()     ← mémoire (cache session, equity curve)
                └── _save_trade_to_db_sync()  ← DB (historique permanent)

GET /api/simulator/trades
  └── db.get_simulation_trades()              ← lit depuis la DB

StateManager (inchangé)
  └── sauvegarde capital/stats/positions (pas _trades)
```

## Contraintes respectées

- INSERT synchrone sqlite3 (pas d'async dans _record_trade)
- WAL mode → pas de conflit avec la connexion aiosqlite du Database
- db_path passé aux runners via constructeur (source : `self._db.db_path`)
- Backward compatible : si table absente → warning, pas de crash
- Frontend inchangé : même format de réponse API
- Zéro impact perf : 1 INSERT SQLite < 1ms, trades arrivent toutes les quelques minutes

## Estimation

- database.py : ~30 lignes (table + 2 méthodes)
- simulator.py : ~40 lignes (helper sync + 2 constructeurs + 2 _record_trade + migration)
- simulator_routes.py : ~10 lignes
- reset_simulator.py : ~15 lignes
- tests : ~80 lignes
- **Total : ~175 lignes**
