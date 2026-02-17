# Sprint 25 — Activity Journal (Plan d'implémentation corrigé)

## Contexte

Le paper trading grid_atr tourne en prod sur 10 assets Grade A. Le système enregistre les trades fermés (`simulation_trades`) et diffuse l'état temps réel via WS, mais il n'existe **aucun historique** entre l'ouverture et la fermeture des positions. Ce sprint ajoute 2 tables DB + hooks dans le code existant + API + enrichissement frontend pour tracer l'equity non réalisée et les événements DCA.

## Corrections vs plan initial

| # | Problème dans le plan initial | Correction |
|---|-------------------------------|------------|
| 1 | `ActivityFeed.jsx` marqué "(NOUVEAU)" | **Existe déjà** (117 lignes). Le MODIFIER, pas le créer |
| 2 | `import json` supposé "peut-être présent" dans database.py | **Absent** en haut. Importé inline (L571). À ajouter |
| 3 | `take_journal_snapshot()` recalcule unrealized/margin inline | Utiliser **`get_status()`** (L1097-1148) — DRY |
| 4 | Tables créées directement dans `_create_tables()` | Sous-méthode **`_create_journal_tables()`** (pattern existant) |
| 5 | `_dispatch_candle` "peut être sync ou async" | **Est async** (L1523). Utiliser `await` directement |
| 6 | `_save_journal_snapshot` dans StateManager accède à `self._db` | OK — StateManager a bien `self._db` (L39) |

---

## Phase 1 — Tables DB

### Fichier : `backend/core/database.py`

**1a.** Ajouter `import json` en haut (après `from pathlib import Path`, L12)

**1b.** Ajouter `await self._create_journal_tables()` dans `_create_tables()` (L174, avant `await self._conn.commit()`)

**1c.** Nouvelle méthode `_create_journal_tables()` — 2 tables :

```sql
-- portfolio_snapshots : snapshot equity/margin toutes les 5 min
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    equity REAL NOT NULL,
    capital REAL NOT NULL,
    margin_used REAL NOT NULL,
    margin_ratio REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    n_positions INTEGER NOT NULL,
    n_assets INTEGER NOT NULL,
    breakdown_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON portfolio_snapshots(timestamp);

-- position_events : ouverture/fermeture de positions
CREATE TABLE IF NOT EXISTS position_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    event_type TEXT NOT NULL,
    level INTEGER,
    direction TEXT NOT NULL,
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    unrealized_pnl REAL,
    margin_used REAL,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON position_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_symbol ON position_events(strategy_name, symbol);
```

**1d.** 5 méthodes async CRUD :
- `insert_portfolio_snapshot(snapshot: dict)`
- `get_portfolio_snapshots(since, until, limit=2000)` → ORDER BY timestamp ASC
- `get_latest_snapshot()` → ORDER BY timestamp **DESC** LIMIT 1 (pour le summary)
- `insert_position_event(event: dict)`
- `get_position_events(since, limit=100, strategy, symbol)` → ORDER BY timestamp DESC

---

## Phase 2 — Snapshot Collector

### Fichier : `backend/backtesting/simulator.py`

Ajouter `take_journal_snapshot()` dans la classe `Simulator` (après `periodic_check()`, L1286).

**Correction vs plan** : utiliser `runner.get_status()` au lieu de recalculer inline :

```python
async def take_journal_snapshot(self) -> dict | None:
    if not self._runners:
        return None

    now = datetime.now(tz=timezone.utc)
    total_capital = 0.0
    total_realized = 0.0
    total_unrealized = 0.0
    total_margin = 0.0
    n_positions = 0
    breakdown = {}

    assets_set = set()
    for runner in self._runners:
        status = runner.get_status()  # ← RÉUTILISE get_status() (DRY)
        total_capital += status.get("capital", 0.0)
        total_realized += status.get("net_pnl", 0.0)
        total_unrealized += status.get("unrealized_pnl", 0.0)
        total_margin += status.get("margin_used", 0.0)
        n_pos = status.get("open_positions", 0)
        n_positions += n_pos

        # Breakdown par symbol (grid runners seulement)
        if isinstance(runner, GridStrategyRunner) and n_pos > 0:
            for symbol, positions in runner._positions.items():
                if not positions:
                    continue
                assets_set.add(symbol)
                last_price = runner._last_prices.get(symbol, 0.0)
                upnl = runner._gpm.unrealized_pnl(positions, last_price)
                margin = sum(p.entry_price * p.quantity / runner._leverage for p in positions)
                breakdown[symbol] = {
                    "strategy": runner.name,
                    "positions": len(positions),
                    "unrealized": round(upnl, 2),
                    "margin": round(margin, 2),
                    "last_price": round(last_price, 2),
                }

    equity = total_capital + total_unrealized
    initial = sum(r._initial_capital for r in self._runners)
    margin_ratio = total_margin / initial if initial > 0 else 0.0

    return {
        "timestamp": now.isoformat(),
        "equity": round(equity, 2),
        "capital": round(total_capital, 2),
        "margin_used": round(total_margin, 2),
        "margin_ratio": round(margin_ratio, 4),
        "realized_pnl": round(total_realized, 2),
        "unrealized_pnl": round(total_unrealized, 2),
        "n_positions": n_positions,
        "n_assets": len(assets_set),
        "breakdown": breakdown if breakdown else None,
    }
```

### Fichier : `backend/core/state_manager.py`

Modifier `_periodic_save_loop()` (L191-210) — ajouter compteur snapshot 5min :

```python
async def _periodic_save_loop(self, simulator, interval):
    snapshot_counter = 0
    while self._running:
        try:
            await asyncio.sleep(interval)
            if self._running and simulator.runners:
                await simulator.periodic_check()
                await self.save_runner_state(
                    simulator.runners,
                    global_kill_switch=simulator._global_kill_switch,
                )
                # Journal snapshot toutes les 5 itérations (300s = 5 min)
                snapshot_counter += 1
                if snapshot_counter >= 5:
                    snapshot_counter = 0
                    await self._save_journal_snapshot(simulator)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("StateManager: erreur sauvegarde périodique: {}", e)
```

Nouvelle méthode helper `_save_journal_snapshot()` (comme dans le plan initial).

---

## Phase 3 — Position Events

### Fichier : `backend/backtesting/simulator.py`

**3a.** Ajouter `self._pending_journal_events: list[dict] = []` dans `GridStrategyRunner.__init__()` (L547, à côté de `_pending_events`)

**3b.** Hook OPEN — DANS le bloc `if not self._is_warming_up:` (L930-931), APRÈS `_emit_open_event()` :
```python
self._pending_journal_events.append({...event OPEN...})
```
→ Automatiquement skippé pendant le warm-up (même guard)

**3c.** Hook CLOSE — DANS le bloc `if not self._is_warming_up:` (L832-833), APRÈS `_emit_close_event()` :
```python
self._pending_journal_events.append({...event CLOSE...})
```

**3d.** Drain dans `_dispatch_candle()` (L1567) — APRÈS le drain `_pending_events` :
```python
# Drain journal events (async insert)
journal_events, runner._pending_journal_events = runner._pending_journal_events, []
if journal_events and self._db:
    for event in journal_events:
        try:
            await self._db.insert_position_event(event)
        except Exception as e:
            logger.warning("Journal: erreur insert event: {}", e)
```
→ `_dispatch_candle` EST async (L1523) — `await` direct OK

---

## Phase 4 — API Routes

### Fichier : `backend/api/journal_routes.py` (NOUVEAU)

3 endpoints comme dans le plan initial :
- `GET /api/journal/snapshots` (since, until, limit)
- `GET /api/journal/events` (since, strategy, symbol, limit)
- `GET /api/journal/summary` (dernier snapshot + 10 derniers events)

Parser `breakdown_json` / `metadata_json` dans les réponses.

**Bug corrigé** : le summary utilise `get_latest_snapshot()` (ORDER BY DESC LIMIT 1) au lieu de `get_portfolio_snapshots(limit=1)[-1]` qui retournerait le plus ancien.

### Fichier : `backend/api/server.py`

Ajouter import + `app.include_router(journal_router)` (L224, après portfolio_router).

---

## Phase 5 — Frontend

### `frontend/src/components/EquityCurve.jsx` (MODIFIER)

Double source de données :
1. Source existante : `GET /api/simulator/equity` (trades fermés, polling 30s) — inchangée
2. Source journal : `GET /api/journal/snapshots?since={24h_ago}` (polling 60s)
3. Si snapshots journal existent → utiliser pour la courbe (equity = capital + unrealized)
4. Sinon → fallback sur la source existante

Ajouter tooltip au hover montrant : equity, unrealized P&L, margin ratio, n_positions.

### `frontend/src/components/ActivityFeed.jsx` (MODIFIER — PAS NOUVEAU)

**Correction majeure** : ce fichier EXISTE déjà (117 lignes, importé dans App.jsx L13).

Enrichir avec :
1. Ajouter `useApi('/api/journal/events?limit=20', 30000)` pour les événements journal
2. Nouveau composant `JournalEventCard` pour afficher OPEN/CLOSE DCA avec pastilles CSS
3. Insérer les événements journal ENTRE les positions ouvertes et les trades fermés :
```
{/* 1. Positions ouvertes (existant, inchangé) */}
{openPositions.map(pos => <OpenPositionCard ... />)}

{/* 2. Événements journal récents (NOUVEAU) */}
{journalEvents.map(event => <JournalEventCard ... />)}

{/* 3. Trades fermés (existant, inchangé) */}
{trades.map(trade => <ClosedTradeCard ... />)}
```

---

## Phase 6 — Tests

### Fichier : `tests/test_journal.py` (NOUVEAU)

~12 tests :
1. `test_insert_and_get_snapshot` — round-trip DB
2. `test_filter_since_until` — filtrage timestamp
3. `test_breakdown_json_roundtrip` — sérialisation/désérialisation JSON
4. `test_insert_and_get_event` — round-trip événement
5. `test_close_event_with_pnl` — événement CLOSE avec P&L
6. `test_filter_by_strategy_and_symbol` — filtrage
7. `test_events_ordered_desc` — tri DESC
8. `test_take_journal_snapshot_no_runners` — retourne None
9. `test_take_journal_snapshot_with_runners` — structure correcte via get_status()
10. `test_snapshots_endpoint_empty` — API sans DB
11. `test_events_endpoint_empty` — API sans DB
12. `test_summary_endpoint` — structure de réponse

---

## Résumé des fichiers modifiés

| Fichier | Type | ~Lignes | Description |
|---------|------|---------|-------------|
| `backend/core/database.py` | MODIFIÉ | +80 | 2 tables + 4 méthodes CRUD |
| `backend/backtesting/simulator.py` | MODIFIÉ | +70 | `take_journal_snapshot()` + hooks events + drain |
| `backend/core/state_manager.py` | MODIFIÉ | +20 | Compteur snapshot 5min + helper |
| `backend/api/journal_routes.py` | NOUVEAU | +90 | 3 endpoints API |
| `backend/api/server.py` | MODIFIÉ | +3 | Import + include router |
| `frontend/src/components/ActivityFeed.jsx` | **MODIFIÉ** | +50 | Section événements journal (JournalEventCard) |
| `frontend/src/components/EquityCurve.jsx` | MODIFIÉ | +40 | Double source snapshots + fallback |
| `tests/test_journal.py` | NOUVEAU | +200 | 12 tests |
| **Total** | | **~550** | |

## Ordre d'implémentation

1. Phase 1 (DB tables + CRUD) → tests DB
2. Phase 2 (snapshot collector) → test snapshot
3. Phase 3 (position events hooks + drain)
4. Phase 4 (API routes) → tests API
5. Phase 5 (frontend enrichissement)
6. Régression complète

## Validation

```bash
# 1. Nouveaux tests
uv run python -m pytest tests/test_journal.py -x -v

# 2. Régression complète
uv run python -m pytest --tb=short -q
# Attendu : 1016 + ~12 = ~1028 tests, 0 échec

# 3. Tables DB après démarrage
sqlite3 data/scalp_radar.db ".tables"
# → portfolio_snapshots, position_events

# 4. Endpoints API
curl http://localhost:8000/api/journal/summary
curl http://localhost:8000/api/journal/snapshots?limit=5
curl http://localhost:8000/api/journal/events?limit=10

# 5. Après 5 min de paper trading
sqlite3 data/scalp_radar.db "SELECT COUNT(*) FROM portfolio_snapshots"
```

## Points d'attention implémentation

1. **Warm-up** : hooks journal dans les mêmes `if not self._is_warming_up` que les emit events — skippés automatiquement
2. **`get_status()`** pour les snapshots — pas de duplication du calcul unrealized/margin
3. **`_dispatch_candle` est async** — utiliser `await` direct pour les INSERT journal
4. **Pas de purge** en phase 1 — les snapshots s'accumulent (~300/jour, négligeable)
5. **Pas de WS push** pour les events journal — polling useApi suffit
6. **`_pending_journal_events`** est une queue séparée de `_pending_events` (Executor)
