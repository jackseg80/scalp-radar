# Sprint 20b-UI — Portfolio Backtest Frontend + Comparateur

## Contexte

Le Sprint 20b a créé `PortfolioBacktester` (CLI) qui simule N assets avec capital partagé via `GridStrategyRunner`. On ajoute maintenant la persistence DB, l'API REST, et le frontend React pour visualiser/comparer les résultats — **sans toucher au moteur de backtest**.

## Fichiers

### À créer (9)
| Fichier | Description |
|---------|-------------|
| `backend/backtesting/portfolio_db.py` | CRUD DB sync+async (même pattern que `optimization_db.py`) |
| `backend/api/portfolio_routes.py` | 6 endpoints REST + job tracker in-memory |
| `frontend/src/components/PortfolioPage.jsx` | Page principale (config panel + résultats) |
| `frontend/src/components/PortfolioPage.css` | Styles dark theme |
| `frontend/src/components/EquityCurveSVG.jsx` | Equity curve SVG interactive (hover tooltip, multi-courbes) |
| `frontend/src/components/DrawdownChart.jsx` | Mini chart drawdown inversé |
| `frontend/src/components/PortfolioCompare.jsx` | Tableau comparatif métriques + deltas |
| `tests/test_portfolio_db.py` | Tests CRUD DB (~5 tests) |
| `tests/test_portfolio_routes.py` | Tests API routes (~8 tests) |

### À modifier (5)
| Fichier | Changement |
|---------|-----------|
| `backend/core/database.py` | Ajouter `_create_portfolio_tables()` dans `_create_tables()` |
| `backend/backtesting/portfolio_engine.py` | Ajouter `progress_callback` optionnel à `run()` et `_simulate()` |
| `backend/api/server.py` | Import + `app.include_router(portfolio_router)` |
| `scripts/portfolio_backtest.py` | Flags `--save` (default) et `--label` |
| `frontend/src/App.jsx` | Import `PortfolioPage` + tab "Portfolio" dans `TABS` |

---

## Étape 1 — DB : table `portfolio_backtests`

### 1A. `backend/core/database.py` — nouvelle sous-méthode

Ajouter `await self._create_portfolio_tables()` dans `_create_tables()` (après `_create_simulator_trades_table`, ligne 173).

```sql
CREATE TABLE IF NOT EXISTS portfolio_backtests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    initial_capital REAL NOT NULL,
    n_assets INTEGER NOT NULL,
    period_days INTEGER NOT NULL,
    assets TEXT NOT NULL,                    -- JSON array
    exchange TEXT NOT NULL DEFAULT 'binance',
    kill_switch_pct REAL NOT NULL DEFAULT 30.0,
    kill_switch_window_hours INTEGER NOT NULL DEFAULT 24,
    final_equity REAL NOT NULL,
    total_return_pct REAL NOT NULL,
    total_trades INTEGER NOT NULL,
    win_rate REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    force_closed_pnl REAL NOT NULL,
    max_drawdown_pct REAL NOT NULL,
    max_drawdown_date TEXT,
    max_drawdown_duration_hours REAL NOT NULL,
    peak_margin_ratio REAL NOT NULL,
    peak_open_positions INTEGER NOT NULL,
    peak_concurrent_assets INTEGER NOT NULL,
    kill_switch_triggers INTEGER NOT NULL DEFAULT 0,
    kill_switch_events TEXT,                 -- JSON
    equity_curve TEXT NOT NULL,              -- JSON (max 500 points sous-échantillonnés)
    per_asset_results TEXT NOT NULL,         -- JSON dict
    created_at TEXT NOT NULL,
    duration_seconds REAL,
    label TEXT
);
CREATE INDEX IF NOT EXISTS idx_portfolio_created ON portfolio_backtests(created_at);
```

### 1B. `backend/backtesting/portfolio_db.py` — CRUD

Pattern identique à `optimization_db.py` : fonctions sync (`sqlite3`) pour le CLI, async (`aiosqlite`) pour l'API.

Fonctions :
- `_result_to_row(result, strategy_name, exchange, duration, label, created_at) -> dict` — sérialise, sous-échantillonne equity à 500 pts
- `save_result_sync(db_path, result, ...) -> int` — INSERT, retourne id
- `save_result_async(db_path, result, ...) -> int` — idem async
- `get_backtests_async(db_path, limit=20) -> list[dict]` — liste sans `equity_curve` (perf)
- `get_backtest_by_id_async(db_path, id) -> dict | None` — détail complet, JSON parsé
- `delete_backtest_async(db_path, id) -> bool`

Snapshot sous-échantillonné (8 champs) :
```json
{"timestamp": "ISO", "equity": 10234.56, "capital": 9800.0, "realized_pnl": 234.56,
 "unrealized_pnl": 200.0, "margin_ratio": 0.42, "positions": 12, "assets_active": 8}
```

---

## Étape 2 — Progress callback dans `portfolio_engine.py`

Ajouter paramètre optionnel `progress_callback: Callable[[float, str], None] | None = None` à `run()` (ligne 146) et le passer à `_simulate()`.

Dans `_simulate()`, appeler le callback au même endroit que le `logger.info` existant (toutes les ~5% de progression). Le callback est synchrone — pas besoin de thread car `_simulate` est async avec des `await` réguliers qui cèdent le contrôle à l'event loop.

---

## Étape 3 — Routes API `portfolio_routes.py`

### Job tracker in-memory

Un seul backtest à la fois. Pas de table `portfolio_jobs` — un simple dict `_current_job` en mémoire :
```python
_current_job: dict | None = None  # {id, status, progress_pct, phase, task, result_id}
```

### 6 endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/portfolio/backtests` | Liste des runs (sans equity_curve) |
| `GET /api/portfolio/backtests/{id}` | Détail complet d'un run |
| `DELETE /api/portfolio/backtests/{id}` | Supprimer un run |
| `POST /api/portfolio/run` | Lancer un backtest → `asyncio.create_task()` |
| `GET /api/portfolio/status` | Status du job en cours (progress, phase) |
| `GET /api/portfolio/compare?ids=1,3` | N runs pour comparaison |

### POST `/run` — body
```json
{
  "strategy_name": "grid_atr",
  "initial_capital": 10000,
  "days": 90,
  "assets": null,
  "exchange": "binance",
  "kill_switch_pct": 30.0,
  "kill_switch_window": 24,
  "label": "test balanced"
}
```

La fonction `_run_backtest()` :
1. Instancie `PortfolioBacktester` avec les params
2. Définit un `progress_callback` qui met à jour `_current_job` + broadcast WS `portfolio_progress`
3. Appelle `await backtester.run(...)`
4. Sauvegarde en DB via `save_result_async()`
5. Broadcast WS `portfolio_completed` + met à jour `_current_job`
6. Try/except pour capter les crashs → broadcast `portfolio_failed`

Broadcast WebSocket via `ws_manager.broadcast()` importé de `websocket_routes.py`.

### `server.py`
Ajouter import + `app.include_router(portfolio_router)` (2 lignes).

---

## Étape 4 — CLI `--save`

Dans `scripts/portfolio_backtest.py` :
- Ajouter `--save` (flag, store_true) et `--label` (str, optional)
- Mesurer `duration_seconds` avec `time.monotonic()`
- Appeler `save_result_sync()` si `--save`

---

## Étape 5 — Frontend

### 5A. `App.jsx` — tab "Portfolio"
- Import `PortfolioPage`
- Ajouter `{ id: 'portfolio', label: 'Portfolio' }` dans `TABS`
- Rendu : `{activeTab === 'portfolio' && <PortfolioPage wsData={wsData} />}`

### 5B. `PortfolioPage.jsx` — page principale

Layout 2 colonnes (même pattern que ExplorerPage) : config panel 320px | résultats flex.

**Panneau gauche** :
- Champs : strategy (select), capital (number), jours (range + span), label (text)
- Assets : auto (tous per_asset) ou sélection manuelle (checkboxes)
- Kill switch % (number)
- Bouton "Lancer" + barre de progression (écoute WS `portfolio_progress`)
- Liste des runs précédents (cliquables + checkbox pour comparaison)

**Panneau droit** :
- `EquityCurveSVG` (equity curve du run sélectionné)
- `DrawdownChart` (drawdown sous l'equity)
- Grille de métriques (return, trades, DD, margin peak, KS)
- Table par asset (symbol, trades, WR, P&L, triée par P&L)

**Mode comparaison** (quand ≥2 runs cochés) :
- Section pleine largeur en bas
- `EquityCurveSVG` multi-courbes normalisées en %
- `PortfolioCompare` tableau métriques + deltas

**WebSocket** : écouter `wsData.type === 'portfolio_progress'` / `'portfolio_completed'` / `'portfolio_failed'`.

### 5C. `EquityCurveSVG.jsx` — SVG interactif

Props : `{ curves: [{label, color, points, initialCapital}], height }`.

- SVG pur (même philosophie que `HeatmapChart.jsx`)
- Axes minimaux : dates en X, valeurs en Y
- Polyline par courbe + fill gradient sous la 1ère courbe
- Baseline capital initial en pointillés
- Hover tooltip : `onMouseMove` → index le plus proche → affiche date, equity, DD%, margin%, positions
- Mode multi-courbes : normalisation en % return quand `curves.length > 1`
- `ResizeObserver` pour le responsive (comme `HeatmapChart`)

### 5D. `DrawdownChart.jsx`

SVG compact (height ~100px). Calcule le drawdown depuis les points equity.
- Axe Y inversé (0% en haut, -max% en bas)
- Fill rouge transparent
- Ligne pointillée au seuil kill switch
- Mode multi-courbes si comparaison

### 5E. `PortfolioCompare.jsx`

Tableau HTML comparatif :
- Colonnes : Métrique | Run A | Run B | Δ
- 7 métriques : Return%, MaxDD%, WR, Trades, Margin peak, KS triggers, P&L réalisé
- Couleur delta : vert si meilleur, rouge si pire (avec `inverted` pour DD/margin)

### 5F. `PortfolioPage.css`

Dark theme cohérent (mêmes variables CSS que `ExplorerPage.css`).

---

## Étape 6 — Tests

### `tests/test_portfolio_db.py` (~5 tests)
1. `test_save_and_get_backtest` — round-trip complet
2. `test_list_backtests_no_equity_curve` — champ exclu de la liste
3. `test_get_backtest_parses_json` — JSON blobs bien parsés
4. `test_delete_backtest` — suppression OK
5. `test_subsample_equity_curve` — max 500 points même avec 2000 snapshots

### `tests/test_portfolio_routes.py` (~8 tests)
1. `test_list_backtests_empty` — 200, liste vide
2. `test_get_backtest_detail` — 200 avec equity_curve
3. `test_get_backtest_not_found` — 404
4. `test_run_returns_job_id` — 200 + job_id
5. `test_run_conflict` — 409 si déjà en cours
6. `test_status_idle` — `{"running": false}`
7. `test_delete_backtest` — 200/204
8. `test_compare_backtests` — structure avec N runs

---

## Ordre d'implémentation

```
Parallèle 1:  database.py (table)  |  portfolio_engine.py (callback)
              ↓                     |
         portfolio_db.py            |
              ↓                     |
         test_portfolio_db.py       |
              ↓                     ↓
         portfolio_routes.py + server.py
              ↓
         test_portfolio_routes.py
              ↓
         portfolio_backtest.py (--save)

Parallèle 2 (frontend, indépendant du backend):
         EquityCurveSVG.jsx → DrawdownChart.jsx → PortfolioCompare.jsx
              ↓
         PortfolioPage.jsx + CSS
              ↓
         App.jsx (tab)
```

## Vérification

```bash
# Tests unitaires
uv run python -m pytest tests/test_portfolio_db.py tests/test_portfolio_routes.py -v

# Tous les tests (806+ existants + ~13 nouveaux)
uv run python -m pytest --tb=short -q

# CLI avec --save
uv run python -m scripts.portfolio_backtest --days 30 --capital 5000 --save --label "test"

# Frontend build
cd frontend && npm run build
```
