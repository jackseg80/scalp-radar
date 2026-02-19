# Sprint Journal V2 — Slippage, Prix Reels & Enrichissements

## Contexte

Le journal de trading (Sprint 32) a un probleme critique : la colonne "PRIX MOYEN" du tableau Ordres Bitget affiche "--" sur quasi tous les ordres. Cause : `_record_order()` est appele juste apres `create_order()`, AVANT `_fetch_fill_price()`. A ce stade Bitget retourne souvent `average=None` pour les market orders.

De plus, il manque des metriques essentielles pour valider le live vs paper : slippage, P&L % positions ouvertes, funding costs, perf par asset.

---

## Fichiers a modifier

| Fichier | Blocs |
|---------|-------|
| [executor.py](backend/execution/executor.py) | 1, 3b |
| [journal_routes.py](backend/api/journal_routes.py) | 2, 3c |
| [database.py](backend/core/database.py) | 3c |
| [JournalPage.jsx](frontend/src/components/JournalPage.jsx) | 2, 3a-d |
| [test_executor_orders.py](tests/test_executor_orders.py) | Tests bloc 1 |
| Nouveau: `tests/test_journal_slippage.py` | Tests blocs 2-3 |

---

## Bloc 1 — Fix prix moyen dans order_history (CRITIQUE)

### 1.1 Enrichir `_record_order()` avec `paper_price`

**Fichier**: `backend/execution/executor.py` ligne 229

- Ajouter parametre `paper_price: float = 0.0`
- Ajouter `"paper_price": paper_price` dans le dict (ligne 240)

### 1.2 Nouvelle methode `_update_order_price()`

Ajouter apres `_record_order()` (apres ligne 252) :

```python
def _update_order_price(self, order_id: str, real_price: float, fee: float | None = None) -> None:
    """Patche le prix reel et la fee dans l'historique pour un order_id donne."""
    if not order_id or real_price <= 0:
        return
    for record in self._order_history:
        if record.get("order_id") == order_id:
            record["average_price"] = real_price
            if fee is not None:
                record["fee"] = fee
            return
```

Safe : deque max 200, single-threaded asyncio, modification dict en place.

### 1.3 Brancher aux sites entry/close

Chercher par pattern (pas par numero de ligne) :

| Pattern a chercher | Contexte | paper_price | _update_order_price apres le bloc if/else avg_price |
|--------------------|----------|-------------|------------------------------------------------------|
| `_record_order("entry",...,"mono")` | Mono entry | `event.entry_price` | `_update_order_price(entry_order_id, avg_price, entry_fee or None)` |
| `_record_order("entry",...,"grid")` | Grid entry | `event.entry_price` | `_update_order_price(order_id, avg_price, entry_fee or None)` |
| `_record_order("close",...,"grid_cycle")` | Grid close | `event.exit_price or 0` | `_update_order_price(close_order.get("id",""), exit_price, exit_fee)` |
| `_record_order("close",...,f"mono_{...}")` | Mono close | `event.exit_price or 0` | `_update_order_price(close_order.get("id",""), exit_price, exit_fee)` |

Les 4 autres call sites (`_record_order` pour SL/TP/emergency) : `paper_price=0.0` (pas d'equivalent paper, slippage non pertinent).

### 1.5 Enregistrer les SL/TP fills dans `_order_history` (NOUVEAU)

**Fichier**: `backend/execution/executor.py`, methode `_process_watched_order()`

Ce path (SL/TP fill detecte par watchOrders) n'appelle PAS `_record_order` — les SL/TP triggers n'apparaissent pas dans le tableau Ordres Bitget. C'est un trou majeur : la majorite des exits sont des SL triggers server-side.

**Fix** : Ajouter un `_record_order()` dans `_process_watched_order` pour les 2 paths :

**Path mono SL/TP** (chercher `if order_id == pos.sl_order_id` / `pos.tp_order_id`) :
```python
# Apres avoir determine exit_reason et exit_price, AVANT _handle_exchange_close :
close_side = "sell" if pos.direction == "LONG" else "buy"
self._record_order(
    exit_reason, symbol, close_side, pos.quantity,
    order, pos.strategy_name, f"watched_{exit_reason}",
    paper_price=0.0,  # pas de prix paper pour SL/TP server-side
)
```

**Path grid SL** (chercher `if order_id == grid_state.sl_order_id`) :
```python
close_side = "sell" if grid_state.direction == "LONG" else "buy"
self._record_order(
    "sl", futures_sym, close_side, grid_state.total_quantity,
    order, grid_state.strategy_name, "watched_grid_sl",
    paper_price=0.0,
)
```

Note : `paper_price=0.0` car ces ordres sont executes par Bitget (trigger orders), pas par le bot — il n'y a pas d'equivalent paper direct. Le slippage ne sera pas calcule pour ces ordres (filtre `paper_price > 0`).

**Verifications avant de coder** :
- Confirmer que `pos.direction`, `pos.quantity`, `pos.strategy_name` existent sur `LivePosition` (L61) et `grid_state.direction`, `grid_state.total_quantity` (property), `grid_state.strategy_name` existent sur `GridLiveState` (L91). En particulier `total_quantity` est une `@property` calculee.
- L'objet `order` dans `_process_watched_order` vient du WS push (ccxt `watchOrders`). Sa structure peut differer d'un `create_order` result. Verifier que `order.get("filled")`, `order.get("average")`, `order.get("id")`, `order.get("status")` sont bien les cles utilisees par ccxt pour les WS orders. Si la cle est differente (ex: `avgPrice`), adapter. `_record_order` fait `float(order_result.get("average") or 0)` → si absent, sera 0, mais `_update_order_price` ne sera PAS appele dans ce path (pas de `_fetch_fill_price` apres). Donc la valeur initiale doit etre correcte.

### 1.4 Persistance

Aucun changement requis. `get_state_for_persistence()` serialise `list(self._order_history)` — les nouveaux champs (`paper_price`, `fee`) sont inclus automatiquement. `restore_positions()` accepte tous les dicts. Backward compat : anciens ordres sans `paper_price` → `o.get("paper_price", 0)` = 0 → exclus du calcul slippage.

---

## Bloc 2 — Mesure slippage paper vs live

### 2.1 Endpoint `GET /api/journal/slippage`

**Fichier**: `backend/api/journal_routes.py`

- Source : `executor._order_history` (filtre `average_price > 0` ET `paper_price > 0`)
- Calcul : `slippage_pct = (average_price - paper_price) / paper_price * 100`
- Retour :
  - `orders_analyzed`, `avg_slippage_pct`, `total_slippage_cost`
  - `by_asset: {symbol: {avg_pct, count}}`
  - `by_strategy: {name: {avg_pct, count}}`
  - `by_type: {order_type: {avg_pct, count}}`
  - `note: "Based on last N orders in memory (max 200)"` — avertissement deque non exhaustive

### 2.2 Frontend: colonne SLIPPAGE + resume

**Fichier**: `frontend/src/components/JournalPage.jsx`, component `BitgetOrders` (ligne 595)

- Ajouter `useApi('/api/journal/slippage', 30000)` pour le resume en haut
- Bandeau resume : "Slippage moyen: X.XX% | Cout total: X.XX$"
- Colonne SLIPPAGE dans le tableau (calcul inline par row) :
  - `slip = (o.average_price - o.paper_price) / o.paper_price * 100`
  - Couleur : vert si favorable (< -0.01%), rouge si defavorable (> 0.01%), gris sinon
- La colonne "PRIX MOYEN" affichera desormais les vraies valeurs (grace au Bloc 1)

---

## Bloc 3 — Enrichissements frontend

### 3a. Colonne P&L % sur positions ouvertes

**Fichier**: `frontend/src/components/JournalPage.jsx`, component `OpenPositions`

- Ajouter colonne "P&L %" apres "P&L latent" dans le thead/tbody
- Grid paper : `pos.unrealized_pnl_pct` (deja expose par `get_grid_state()`, ligne 2089 simulator.py)
- Grid live : calculer `pnl / (entryPrice * qty / leverage) * 100` avec leverage du `executor.positions`
- Mono : calculer `pnl / (entryPrice * qty / leverage) * 100` — leverage dispo dans `executor.positions` pour LIVE, utiliser `config.risk.default_leverage` (=3) pour paper mono (NE PAS hardcoder 6)

### 3b. Niveaux + duree sur positions LIVE

**Fichier backend**: `backend/execution/executor.py`, methode `get_status()` (ligne 1904)

Enrichir la boucle grid dans `get_status()` (chercher `for gs in self._grid_states.values()`) :
```python
"entry_time": gs.opened_at.isoformat(),       # NOUVEAU — opened_at existe deja sur GridLiveState (L106)
"levels_max": self._get_grid_num_levels(gs.strategy_name),  # NOUVEAU
"positions": [{                                # NOUVEAU - detail niveaux
    "level": p.level, "entry_price": p.entry_price,
    "quantity": p.quantity, "entry_time": p.entry_time.isoformat(),
} for p in gs.positions],
```

Enrichir la boucle mono (chercher `for pos in self._positions.values()`) :
```python
"entry_time": pos.entry_time.isoformat(),  # NOUVEAU
```

Nouvelle methode helper (meme pattern que `_get_grid_sl_percent` et `_get_grid_leverage`) :
```python
def _get_grid_num_levels(self, strategy_name: str) -> int:
    strat_config = getattr(self._config.strategies, strategy_name, None)
    if strat_config and hasattr(strat_config, "num_levels"):
        return strat_config.num_levels
    return 4  # default
```

**Fichier frontend**: Adapter `OpenPositions` pour :
- Detecter `pos.type === 'grid'` pour les positions executor
- Row expandable si `pos.positions` existe (comme paper grids)
- Afficher `pos.levels` / `pos.levels_max` pour les grids LIVE
- Afficher `timeAgo(pos.entry_time)` pour la duree

### 3c. Resume perf par asset

**Fichier backend**: `backend/core/database.py`

Ajouter methode `get_journal_per_asset_stats(period)` :
- SQL `GROUP BY symbol` sur `simulation_trades`
- Retourne: symbol, total_trades, wins, losses, win_rate, total_pnl, avg_pnl

**Note** : `simulation_trades` contient uniquement les trades paper (simulator). Les trades live fermes par l'Executor ne sont PAS dans cette table — ils sont dans `_order_history` en memoire. L'endpoint per-asset reflete donc la performance paper. C'est acceptable car paper et live traitent les memes signaux (memes strategies, memes candles). Un enrichissement futur pourrait persister les trades live en DB.

**Fichier backend**: `backend/api/journal_routes.py`

Ajouter endpoint `GET /api/journal/per-asset?period=all` → appelle `db.get_journal_per_asset_stats()`

**Fichier frontend**: Nouveau component `PerAssetSummary` dans JournalPage, section collapsible apres Stats. Tableau triable: Symbol, Trades, Win Rate, P&L Net.

### 3d. Funding costs visibles

Pas de changement backend. `funding_cost` deja expose par `runner.get_status()` → `simulator.get_all_status()` → WS `data["strategies"]`.

**Fichier frontend**: Dans `StatsOverview` (ou stats secondaires) :
```jsx
const totalFunding = Object.values(wsData?.strategies || {})
  .reduce((sum, s) => sum + (s.funding_cost || 0), 0)
```
Afficher "Funding: {totalFunding}$" dans la barre stats secondaires.

---

## Ordre d'implementation

1. **Bloc 1.1-1.3** : `_update_order_price()` + enrichir `_record_order()` + brancher 4 sites entry/close
2. **Bloc 1.5** : Enregistrer SL/TP fills dans `_process_watched_order`
3. **Bloc 3b backend** : enrichir `get_status()` executor (entry_time, positions, levels_max) + helper `_get_grid_num_levels`
4. **Bloc 2 backend** : endpoint `/api/journal/slippage`
5. **Bloc 3c backend** : methode DB + endpoint `/api/journal/per-asset`
6. **Tests** : 17 tests minimum (voir section)
7. **Frontend** : tous les blocs en une passe (slippage, P&L%, LIVE niveaux, per-asset, funding)

---

## Tests (17 minimum)

**Fichier**: `tests/test_executor_orders.py` (enrichir)

1. `test_update_order_price_patches_record` — patche average_price dans la deque
2. `test_update_order_price_no_match` — order_id inexistant, pas de crash
3. `test_update_order_price_adds_fee` — champ fee ajoute au dict
4. `test_update_order_price_ignores_zero_price` — real_price=0, pas de maj
5. `test_record_order_has_paper_price` — champ paper_price present
6. `test_persistence_roundtrip_new_fields` — serialise/restaure avec paper_price+fee
7. `test_persistence_restore_legacy_orders` — restaure des ordres au format ancien (sans paper_price ni fee), verifie que tout fonctionne

**Nouveau fichier**: `tests/test_journal_slippage.py`

8. `test_slippage_endpoint_no_executor` — retourne `{slippage: None}`
9. `test_slippage_endpoint_with_data` — calculs corrects + champ `note` present
10. `test_slippage_by_asset_grouping` — groupement par symbol
11. `test_slippage_filters_invalid_orders` — exclusion ordres sans paper_price

**Fichier**: `tests/test_journal_slippage.py` (suite)

12. `test_per_asset_stats_empty` — retourne liste vide
13. `test_per_asset_stats_aggregation` — 2 assets, verifie wins/losses/pnl

**Fichier**: `tests/test_executor_orders.py` ou `tests/test_executor.py`

14. `test_executor_get_status_grid_entry_time` — verifie entry_time + positions dans get_status() grid
15. `test_executor_get_status_mono_entry_time` — verifie entry_time dans get_status() mono
16. `test_get_grid_num_levels_helper` — helper retourne num_levels depuis config, fallback 4
17. `test_watched_order_records_sl_fill` — verifie qu'un SL fill via watchOrders est enregistre dans _order_history

---

## Pieges identifies

| Piege | Mitigation |
|-------|------------|
| Ordres anciens sans `paper_price` | `o.get("paper_price", 0)` → exclus du slippage |
| Positions LIVE grid dans `execPositions` | Detecter `pos.type === "grid"`, expandable rows |
| `levels_max` non dispo dans executor | Helper `_get_grid_num_levels()` lit `num_levels` depuis config strategies |
| P&L % mono paper sans leverage | Fallback sur `default_leverage` (=3) de risk.yaml, PAS 6 |
| Race condition deque | Pas de risque : asyncio single-threaded |
| SL/TP fills Bitget hors historique | Ajouter `_record_order` dans `_process_watched_order` (Bloc 1.5) |
| `simulation_trades` = paper only | L'endpoint per-asset reflete le paper (trades live non persistes en DB) |

---

## Validation

```bash
# Tests complets
uv run python -m pytest tests/ -x --tb=short -q

# Ordres avec prix reels
curl http://localhost:8000/api/executor/orders?limit=5 | python -m json.tool

# Rapport slippage
curl http://localhost:8000/api/journal/slippage | python -m json.tool

# Perf par asset
curl http://localhost:8000/api/journal/per-asset | python -m json.tool
```
