# Sprint 16+17 — Dashboard Scanner amélioré + Monitoring DCA Live

## Contexte

Le Scanner multi-stratégie fonctionne correctement mais manque de visibilité sur les positions grid DCA. Avec 21 assets en paper trading (envelope_dca), le dashboard affiche "Aucune position" dans ActivePositions (il ne montre que les mono-positions). Ce sprint **ajoute** des colonnes Grade et Grid sans toucher aux colonnes Score/Signaux existantes, et enrichit le widget Positions Actives pour afficher les grilles DCA.

## Fichiers à modifier

| Fichier | Action |
|---------|--------|
| `backend/backtesting/simulator.py` | Ajouter `Simulator.get_grid_state()` |
| `backend/api/simulator_routes.py` | Endpoint `GET /api/simulator/grid-state` |
| `backend/api/websocket_routes.py` | Pusher `grid_state` via WS (3s) |
| `frontend/src/components/Scanner.jsx` | Colonnes Grade + Grid, tri positions-first |
| `frontend/src/components/ActivePositions.jsx` | Résumé grid agrégé + détail dépliable |
| `frontend/src/styles.css` | Classes `.grade-badge` globales |
| `tests/test_api_simulator.py` | 3 tests endpoint grid-state |
| `tests/test_simulator_grid_state.py` (nouveau) | 3 tests logique métier get_grid_state |

---

## Étape 1 : Backend — `Simulator.get_grid_state()`

**Fichier :** `backend/backtesting/simulator.py` — ajouter après `get_open_positions()` (ligne ~1505)

Nouvelle méthode sur la classe `Simulator` :

```python
def get_grid_state(self) -> dict:
    """État détaillé des grilles DCA actives avec P&L non réalisé."""
    grids: list[dict] = []

    for runner in self._runners:
        if not hasattr(runner, "_gpm"):
            continue  # Pas un GridStrategyRunner

        for symbol, positions in runner._positions.items():
            if not positions:
                continue

            # Prix courant depuis DataEngine
            data = self._data_engine.get_data(symbol)
            candles_1m = data.candles.get("1m", [])
            if not candles_1m:
                continue
            current_price = candles_1m[-1].close
            if current_price <= 0:
                continue

            # GridState agrégé via GPM
            grid_state = runner._gpm.compute_grid_state(positions, current_price)

            # Indicateurs pour TP/SL dynamique
            ctx = runner.build_context(symbol)
            main_tf = getattr(runner._strategy._config, "timeframe", "1h")
            main_ind = ctx.indicators.get(main_tf, {}) if ctx and ctx.indicators else {}

            tp_price = runner._strategy.get_tp_price(grid_state, main_ind)
            sl_price = runner._strategy.get_sl_price(grid_state, main_ind)

            leverage = runner._leverage
            margin_used = grid_state.total_notional / leverage if leverage > 0 else 0
            unrealized_pnl_pct = (
                grid_state.unrealized_pnl / margin_used * 100 if margin_used > 0 else 0.0
            )

            import math
            grids.append({
                "symbol": symbol,
                "strategy": runner.name,
                "direction": positions[0].direction.value,
                "levels_open": len(positions),
                "levels_max": runner._strategy.max_positions,
                "avg_entry": round(grid_state.avg_entry_price, 6),
                "current_price": current_price,
                "unrealized_pnl": round(grid_state.unrealized_pnl, 2),
                "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
                "tp_price": round(tp_price, 6) if not math.isnan(tp_price) else None,
                "sl_price": round(sl_price, 6) if not math.isnan(sl_price) else None,
                "tp_distance_pct": round((tp_price - current_price) / current_price * 100, 2) if not math.isnan(tp_price) else None,
                "sl_distance_pct": round((sl_price - current_price) / current_price * 100, 2) if not math.isnan(sl_price) else None,
                "margin_used": round(margin_used, 2),
                "leverage": leverage,
                "positions": [
                    {"level": p.level, "entry_price": p.entry_price,
                     "quantity": p.quantity, "entry_time": p.entry_time.isoformat(),
                     "direction": p.direction.value}
                    for p in positions
                ],
            })

    total_margin = sum(g["margin_used"] for g in grids)
    total_upnl = sum(g["unrealized_pnl"] for g in grids)

    return {
        "grid_positions": {g["symbol"]: g for g in grids},
        "summary": {
            "total_positions": sum(g["levels_open"] for g in grids),
            "total_assets": len(grids),
            "total_margin_used": round(total_margin, 2),
            "total_unrealized_pnl": round(total_upnl, 2),
            "capital_available": round(
                sum(r._capital for r in self._runners if hasattr(r, "_gpm")),
                2,
            ) if any(hasattr(r, "_gpm") for r in self._runners) else 0,
        },
    }
```

**Fonctions existantes réutilisées :**
- `runner._gpm.compute_grid_state(positions, current_price)` → [grid_position_manager.py:209](backend/core/grid_position_manager.py#L209)
- `runner.build_context(symbol)` → [simulator.py:938](backend/backtesting/simulator.py#L938)
- `runner._strategy.get_tp_price(grid_state, indicators)` → [envelope_dca.py:152](backend/strategies/envelope_dca.py#L152)
- `runner._strategy.get_sl_price(grid_state, indicators)` → [envelope_dca.py:158](backend/strategies/envelope_dca.py#L158)

**Note :** `import math` à placer en haut du fichier (vérifier s'il y est déjà).

---

## Étape 2 : Backend — Endpoint API

**Fichier :** `backend/api/simulator_routes.py` — ajouter après le dernier endpoint

```python
@router.get("/grid-state")
async def simulator_grid_state(request: Request) -> dict:
    """État détaillé des grilles DCA actives."""
    simulator = getattr(request.app.state, "simulator", None)
    if simulator is None:
        return {"grid_positions": {}, "summary": {
            "total_positions": 0, "total_assets": 0,
            "total_margin_used": 0, "total_unrealized_pnl": 0,
            "capital_available": 0,
        }}
    return simulator.get_grid_state()
```

---

## Étape 3 : Backend — WebSocket push grid_state

**Fichier :** `backend/api/websocket_routes.py` — ligne 89, après `simulator_positions`

Ajouter une seule ligne :
```python
data["grid_state"] = simulator.get_grid_state()
```

Cela pousse le grid_state toutes les 3s dans le WS existant, évitant un polling API séparé côté frontend.

---

## Étape 4 : Frontend — Styles globaux grade-badge

**Fichier :** `frontend/src/styles.css` — ajouter à la fin (ou dans la section badges)

```css
/* Grade badges (Scanner + Research) */
.grade-badge { padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; display: inline-block; }
.grade-A { background: #22c55e; color: white; }
.grade-B { background: #3b82f6; color: white; }
.grade-C { background: #eab308; color: white; }
.grade-D { background: #ef4444; color: white; }
.grade-F { background: #6b7280; color: white; }

/* Grid cell */
.grid-cell { font-weight: 600; font-size: 13px; font-family: var(--mono); }
.grid-cell--profit { color: var(--accent); }
.grid-cell--loss { color: var(--red); }
.grid-cell--empty { color: var(--text-dim); font-weight: 400; }
```

---

## Étape 5 : Frontend — Scanner.jsx

**Modifications :**

### 5a. Nouveaux imports et API calls
- Ajouter `useApi('/api/optimization/results?latest_only=true&limit=500', 60000)` pour les grades (polling 60s, données quasi-statiques)
- Le grid_state vient de `wsData.grid_state` (via WS, étape 3)

### 5b. Construire les lookups
- `gradesLookup[symbol] → { grade, strategy }` — pour chaque asset, prendre le grade envelope_dca prioritairement, sinon le meilleur grade
- `gridLookup[symbol] → grid data` — directement depuis `wsData.grid_state.grid_positions`

### 5c. Modifier le tri
```
positions grid ouvertes → en premier (triées par unrealized_pnl desc)
puis par score décroissant (existant)
```

### 5d. Ajouter 2 colonnes dans le thead (9 colonnes au total)
```
Actif | Prix | Var. | Dir. | Trend | Score | Grade | Signaux | Grid
```
Mettre à jour tous les `colSpan={7}` → `colSpan={9}`

### 5e. Cellule Grade
Badge coloré `<span className="grade-badge grade-{grade}">{grade}</span>` ou "--" si pas de WFO.

### 5f. Cellule Grid
`"2/4"` coloré vert/rouge selon P&L, ou "--" si pas de position grid.

### 5g. Ajuster les largeurs
| Colonne | Width |
|---------|-------|
| Actif | 13% |
| Prix | 9% |
| Var. | 7% |
| Dir. | 7% |
| Trend | 12% |
| Score | 8% |
| Grade | 7% |
| Signaux | 22% |
| Grid | 7% |

---

## Étape 6 : Frontend — ActivePositions.jsx

### 6a. Séparer positions grid vs mono
- `monoSimPositions = simPositions.filter(p => p.type !== 'grid')`
- Grid data depuis `wsData.grid_state`

### 6b. Résumé grid en haut
Quand des grilles existent, afficher un bandeau :
```
9 grids sur 6 assets · Marge: 4 523$ · P&L: +234.56$
```

### 6c. Ligne par asset (grid agrégé)
Pour chaque grille dans `grid_state.grid_positions` :
- Direction badge + Symbol + Strategy + niveaux "2/4" + avg entry + P&L + TP/SL distances
- Clic → déplier les positions individuelles (niveau, prix, qty, heure)

### 6d. Conserver PositionRow pour mono-positions
`PositionRow` existant reste inchangé pour les futures stratégies mono-position.

---

## Étape 7 : Tests

### 7a. `tests/test_api_simulator.py` — 3 tests endpoint

```python
test_grid_state_no_simulator     # simulator=None → JSON vide
test_grid_state_empty            # simulator mockée, pas de grille → grids vide
test_grid_state_with_data        # simulator mockée avec données → JSON complet vérifié
```

Pattern : mock `simulator.get_grid_state = MagicMock(return_value={...})`, même fixture `mock_app`.

### 7b. `tests/test_simulator_grid_state.py` — 3 tests logique métier

```python
test_empty_no_runners            # Simulator sans runners → grids=[]
test_grid_state_with_positions   # GridStrategyRunner avec 2 positions injectées → vérifie fields
test_grid_state_unrealized_pnl   # entry=100, current=110, qty=1, LONG → upnl=+10
```

Pattern : vrais `GridStrategyRunner` avec mock strategy/config/data_engine, `GridPositionManager` réel pour les calculs.

---

## Ordre d'implémentation

1. `simulator.py` — `get_grid_state()` (backend pur, testable unitairement)
2. `simulator_routes.py` — endpoint API
3. `websocket_routes.py` — push WS
4. `styles.css` — classes CSS (indépendant)
5. `Scanner.jsx` — colonnes Grade + Grid
6. `ActivePositions.jsx` — résumé grid + détail
7. Tests (endpoint + logique métier)

## Vérification

```bash
uv run python -m pytest --tb=short -q     # 714+ tests passent
```
- Visuellement : colonnes Score et Signaux toujours présentes et fonctionnelles
- Nouvelle colonne Grade avec badges A/B/C/D/F colorés
- Nouvelle colonne Grid avec "2/4" coloré
- Widget Positions Actives montre les grids avec P&L et détail dépliable
- WebSocket transmet grid_state en temps réel (3s)
