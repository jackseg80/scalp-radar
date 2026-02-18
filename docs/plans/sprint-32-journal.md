# Sprint 32 — Page Journal de Trading

## Contexte

L'ActivityFeed en sidebar est trop compact pour suivre efficacement les operations de trading. On cree un onglet "Journal" dedie avec 4 sections collapsibles, un endpoint stats backend, et un historique d'ordres dans l'Executor. Un selecteur de periode global filtre Stats + Equity Curve. Le sous-onglet "Ouvertes" aura un rendu specifique plus detaille que l'ActivePositions existant.

---

## Fichiers a modifier/creer

| Fichier | Action |
|---------|--------|
| `backend/core/database.py` | Ajouter `get_journal_stats()` |
| `backend/api/journal_routes.py` | Ajouter endpoint `GET /api/journal/stats` |
| `backend/execution/executor.py` | Ajouter `_order_history` deque + `_record_order()` + persistence |
| `backend/api/executor_routes.py` | Ajouter endpoint `GET /api/executor/orders` (sans auth) |
| `frontend/src/components/JournalPage.jsx` | **NOUVEAU** — page 4 sections |
| `frontend/src/components/JournalPage.css` | **NOUVEAU** — styles |
| `frontend/src/App.jsx` | Tab journal + routing + props |
| `frontend/src/components/ActivityFeed.jsx` | Reduire a 5 events + lien journal |
| `tests/test_journal_stats.py` | **NOUVEAU** — tests stats |
| `tests/test_executor_orders.py` | **NOUVEAU** — tests order history |

---

## Phase 1 — Backend : `GET /api/journal/stats`

### 1a. `backend/core/database.py` — Methode `get_journal_stats()`

Ajouter apres `clear_simulation_trades()` (~ligne 890) :

```python
async def get_journal_stats(self, period: str = "all") -> dict:
```

**Logique :**
1. Calculer `since` ISO timestamp selon period : `"today"` = debut du jour UTC, `"7d"` = -7j, `"30d"` = -30j, `"all"` = None
2. Query SQL : `SELECT net_pnl, exit_time, entry_time, symbol, exit_reason FROM simulation_trades WHERE exit_time >= ? ORDER BY exit_time ASC`
3. Calculs Python depuis les rows :
   - `total_trades`, `wins` (net_pnl > 0), `losses`, `win_rate` (%)
   - `total_pnl` = sum(net_pnl), `gross_profit` = sum(positifs), `gross_loss` = sum(negatifs)
   - `profit_factor` = gross_profit / abs(gross_loss) (0 si pas de losses)
   - `best_trade` = {symbol, pnl} du max, `worst_trade` = {symbol, pnl} du min
   - `avg_duration_hours` = moyenne(exit_time - entry_time) en heures
   - `trades_per_day` = total_trades / max(1, jours_dans_periode)
   - `current_streak` = iterer depuis le plus recent, compter wins/losses consecutifs → `{"type": "win"|"loss", "count": N}`
4. Drawdown depuis `portfolio_snapshots` : peak-to-trough sur equity dans la meme periode. **Cas 0 snapshots** : retourner `max_drawdown_pct = 0.0` (serveur fraichement redemarre ou mode dev sans snapshots)
5. Retourner dict complet

### 1b. `backend/api/journal_routes.py` — Endpoint

Ajouter apres `get_summary()` (~ligne 92) :

```python
@router.get("/stats")
async def get_stats(
    request: Request,
    period: str = Query("all", description="today, 7d, 30d, all"),
) -> dict:
    db = getattr(request.app.state, "db", None)
    if db is None:
        return {"stats": None}
    if period not in {"today", "7d", "30d", "all"}:
        period = "all"
    stats = await db.get_journal_stats(period=period)
    return {"stats": stats}
```

Pas de nouveau router a monter (`journal_router` deja dans server.py ligne 24).

---

## Phase 2 — Backend : Executor Order History

### 2a. `backend/execution/executor.py` — `_order_history` + `_record_order()`

**Dans `__init__` (apres ligne 174)** :
```python
from collections import deque
self._order_history: deque[dict] = deque(maxlen=200)
```

**Nouvelle methode `_record_order()`** :
```python
def _record_order(self, order_type, symbol, side, quantity, order_result, strategy_name="", context=""):
    self._order_history.appendleft({
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "order_type": order_type,  # "entry", "close", "sl", "tp", "emergency_close"
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "filled": float(order_result.get("filled") or 0),
        "average_price": float(order_result.get("average") or 0),
        "order_id": order_result.get("id", ""),
        "status": order_result.get("status", ""),
        "strategy_name": strategy_name,
        "context": context,
    })
```

**8 points d'insertion** apres chaque `create_order` reussi :

| Ligne | Methode | order_type | context |
|-------|---------|------------|---------|
| 453 | `_open_position` | `"entry"` | `"mono"` |
| 492 | `_open_position` (emergency) | `"emergency_close"` | `"sl_failed"` |
| 549 | `_place_sl_with_retry` | `"sl"` | `f"retry_{attempt}"` |
| 582 | `_place_tp` | `"tp"` | `""` |
| 665 | `_open_grid_position` | `"entry"` | `"grid"` |
| 777 | `_emergency_close_grid` | `"emergency_close"` | `"grid_sl_failed"` |
| 815 | `_close_grid_cycle` | `"close"` | `"grid_cycle"` |
| 913 | `_close_position` | `"close"` | `f"mono_{exit_reason}"` |

**Persistence** dans `get_state_for_persistence()` :
```python
"order_history": list(self._order_history),
```

**Restore** dans `restore_positions()` :
```python
order_history = state.get("order_history", [])
self._order_history = deque(order_history, maxlen=200)
```

### 2b. `backend/api/executor_routes.py` — Endpoint sans auth

```python
@router.get("/orders")
async def executor_orders(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    """Historique des ordres Bitget (read-only, sans auth)."""
    executor = getattr(request.app.state, "executor", None)
    if executor is None:
        return {"orders": [], "count": 0}
    orders = list(executor._order_history)[:limit]
    return {"orders": orders, "count": len(orders)}
```

---

## Phase 3 — Frontend : JournalPage

### 3a. `frontend/src/components/JournalPage.jsx` — NOUVEAU

**Architecture** : selecteur de periode global en haut, 4 sections CollapsibleCard en dessous.

```
JournalPage ({ wsData, onTabChange })
├── Header + Selecteur periode global (today/7d/30d/all)
├── Section 1: StatsOverview (CollapsibleCard)
│   └── 6 KPI cards en grid + ligne secondaire
├── Section 2: PositionsAndTrades (CollapsibleCard)
│   ├── Sub-tab "Ouvertes" → tableau detaille (wsData temps reel)
│   └── Sub-tab "Historique" → tableau trades fermes + filtres
├── Section 3: AnnotatedEquityCurve (CollapsibleCard)
│   └── SVG large (800x300) + markers OPEN/CLOSE
└── Section 4: BitgetOrders (CollapsibleCard)
    └── Tableau ordres executor
```

### 3b. `frontend/src/components/JournalPage.css` — NOUVEAU

---

## Phase 4 — Frontend : Header + App wiring

### `frontend/src/App.jsx`

1. Import JournalPage
2. TABS — ajouter `{ id: 'journal', label: 'Journal' }` entre portfolio et logs
3. loadActiveTab — ajouter `'journal'` dans validTabs
4. Rendu conditionnel
5. ActivityFeed — passer `onTabChange={handleTabChange}` en props

Header.jsx ne change pas (data-driven par le prop `tabs`).

---

## Phase 5 — Frontend : Reduire ActivityFeed

1. Accepter prop `onTabChange`
2. Reduire journalEvents de 15 a 5
3. Ajouter lien "Voir le journal complet →"

---

## Phase 6 — Tests

### `tests/test_journal_stats.py` — 9 tests
1. `test_stats_empty_db` — retourne zeros
2. `test_stats_with_trades` — 3 wins + 1 loss → WR=75%, PF correct
3. `test_stats_period_filter_7d` — trades vieux exclus de "7d"
4. `test_stats_period_filter_today` — seuls trades du jour
5. `test_stats_streak_win` — 3 wins consecutifs recents
6. `test_stats_streak_loss` — 2 losses consecutives recentes
7. `test_stats_profit_factor_no_losses` — PF = 0 guard
8. `test_stats_drawdown_from_snapshots` — peak-to-trough sur equity
9. `test_stats_drawdown_no_snapshots` — 0 snapshots → 0.0

### `tests/test_executor_orders.py` — 7 tests
1. `test_order_history_initialized` — deque maxlen=200
2. `test_record_order_format` — append correct avec tous les champs
3. `test_order_history_maxlen` — 250 inserts → 200 gardes
4. `test_order_history_persistence_roundtrip` — list/deque round-trip
5. `test_orders_endpoint_no_executor` — retourne vide
6. `test_orders_endpoint_with_data` — retourne les ordres
7. `test_orders_endpoint_with_limit` — limit respecte

---

## Resultats

- **16 nouveaux tests** (9 stats + 7 orders), tous passent
- **1238 tests** au total (zero regression)
- Backend : 1 nouvel endpoint stats + 1 nouvel endpoint orders + order history executor
- Frontend : nouvel onglet Journal avec 4 sections collapsibles, selecteur de periode, sous-onglets positions/historique, courbe d'equity annotee, tableau ordres Bitget
- ActivityFeed sidebar reduit a 5 events avec lien vers Journal
