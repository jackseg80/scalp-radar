# Sprint 39 — Métriques Live Enrichies

## Contexte

Le dashboard affiche des métriques riches (TP, SL, P&L, indicateurs, distances) pour les positions **paper** (Simulator.get_grid_state) mais seulement des données basiques pour les positions **live** (Executor.get_status). L'objectif est d'aligner les deux pour que le frontend affiche les mêmes métriques live et paper, sans modifier la logique de trading.

**Observation clé** : L'Executor a déjà accès à tout ce qu'il faut :
- `self._data_engine` (set via `set_data_engine()`, ligne 496)
- `self._simulator` (set via `set_strategies()`, ligne 507)
- `self._strategies` (dict de BaseGridStrategy instances)
- `_live_state_to_grid_state()` (ligne 710) construit déjà un GridState avec P&L
- `_check_grid_exit()` (exit monitor, lignes 757-850) lit déjà les indicateurs du runner paper et appelle `get_tp_price()`/`get_sl_price()`

Le format cible = celui du paper `get_grid_state()` (simulator.py:2092-2191).

---

## Fichiers à modifier

| Fichier | Changement |
|---------|-----------|
| `backend/execution/executor.py` | Ajouter `_enrich_grid_position()`, modifier `get_status()` |
| `backend/api/websocket_routes.py` | Merger les grids live dans `grid_state` du push WS |
| `frontend/src/components/ActivePositions.jsx` | Badge LIVE/PAPER dynamique, filtrer grid du LIVE column |
| `frontend/src/components/ExecutorPanel.jsx` | Utiliser P&L backend, TP/SL distances, durée, P&L par niveau |
| `tests/test_executor_enriched_status.py` | ~10 tests |

---

## Étape 1 — `_enrich_grid_position()` dans Executor

Ajouter une méthode privée qui enrichit un `GridLiveState` en réutilisant le même pattern que `_check_grid_exit()` :

1. Convertir `futures_sym` → `spot_sym` (strip `:USDT`)
2. Lire prix depuis DataEngine buffers (fallback 1m→5m→15m→1h)
3. Calculer P&L via `_live_state_to_grid_state(futures_sym, current_price)`
4. Lire indicateurs paper via `self._simulator.get_runner_context(strategy_name, spot_sym)`
5. Appeler `strategy.get_tp_price(grid_state, tf_indicators)` et `get_sl_price()`
6. Calculer distances %, marge, durée, P&L par niveau

Retourne un dict au format paper-compatible :
```python
{
    "symbol": gs.symbol,           # futures format (backward compat)
    "direction": gs.direction,
    "entry_price": avg_entry,      # backward compat ExecutorPanel
    "quantity": total_quantity,
    "strategy_name": gs.strategy_name,
    "type": "grid",
    "levels": len(gs.positions),
    "levels_max": ...,
    "leverage": gs.leverage,
    "notional": ...,
    "entry_time": ...,
    # — Champs enrichis —
    "current_price": float | None,
    "unrealized_pnl": float,
    "unrealized_pnl_pct": float,
    "tp_price": float | None,      # SMA estimée (None si NaN/indisponible)
    "sl_price": float,             # SL réel ou 0.0
    "tp_distance_pct": float | None,
    "sl_distance_pct": float | None,
    "margin_used": float,
    "duration_hours": float,
    "positions": [{level, entry_price, quantity, entry_time, direction, pnl_usd, pnl_pct}],
}
```

**Graceful degradation** : si `_data_engine` ou `_simulator` sont None, les champs enrichis valent None/0.

---

## Étape 2 — Modifier `get_status()` dans Executor

Remplacer la boucle grid (lignes 2287-2315) par `_enrich_grid_position()`.

Ajouter un champ `executor_grid_state` au format paper-compatible :
```python
result["executor_grid_state"] = {
    "grid_positions": {
        "grid_atr:BTC/USDT": { /* format paper avec source="live" */ },
        ...
    },
    "summary": { total_positions, total_assets, total_margin_used, total_unrealized_pnl },
}
```

Les clés sont `strategy_name:spot_symbol` (spot, pas futures) pour matcher le format paper.

---

## Étape 3 — Merger live grids dans le WS push

Dans `_build_update_data()` (websocket_routes.py), après avoir construit `grid_state` depuis le Simulator et `executor` status :

1. Extraire `executor_grid_state` depuis le statut executor
2. Déterminer les stratégies live : `allowed_strategies` depuis `executor.selector`
3. **Supprimer les entrées paper** des stratégies live du `grid_state` (éviter la confusion paper+live)
4. Pour chaque entrée live, convertir au format paper (renommages: `entry_price`→`avg_entry`, `levels`→`levels_open`, `strategy_name`→`strategy`)
5. Ajouter `source: "live"` aux entrées live, `source: "paper"` aux entrées paper restantes
6. Recalculer le `summary` avec les données mergées

Résultat : le `grid_state` dans le WS contient les positions live pour les stratégies live, et les positions paper pour les stratégies paper-only. Pas de doublons. Le frontend existant (Scanner, GridDetail, ActivePositions.GridList) affiche tout sans modification structurelle.

---

## Étape 4 — Frontend ActivePositions : colonne unique

**ActivePositions.jsx** — fusionner PAPER et LIVE en **une seule colonne** :
- Supprimer les deux colonnes PAPER/LIVE séparées
- Une seule liste de positions : grids (depuis `grid_state` mergé) + mono (paper + live)
- Badge dynamique par position basé sur `g.source` :
  ```jsx
  <span className={`badge ${g.source === 'live' ? 'badge-active' : 'badge-simulation'}`}>
    {g.source === 'live' ? 'LIVE' : 'PAPER'}
  </span>
  ```
- Filtrer les grid des `execPositions` pour éviter le double-affichage :
  ```jsx
  const execPositions = (wsData?.executor?.positions || []).filter(p => p.type !== 'grid')
  ```
- Les mono live (execPositions sans grid) restent affichées avec badge LIVE

---

## Étape 5 — Frontend ExecutorPanel

**ExecutorPanel.jsx** `PositionCard` :
- Utiliser `position.unrealized_pnl` / `unrealized_pnl_pct` du backend (fallback client-side)
- Afficher `current_price`
- Afficher TP avec `tp_price` + `tp_distance_pct` (plus "SMA dynamique" quand tp=0 ET pas d'enrichissement)
- Afficher SL avec `sl_distance_pct`
- Afficher `duration_hours` (formaté heures ou jours)
- Afficher P&L par niveau pour les grids (section dépliable si >1 niveau)
- Utiliser `margin_used` du backend au lieu de calculer `notional/leverage`

---

## Étape 6 — Tests

Nouveau fichier `tests/test_executor_enriched_status.py` (~10 tests) :

1. `test_enrich_grid_position_basic` — avec mock DataEngine + Simulator, vérifie tous les champs
2. `test_enrich_grid_position_no_data_engine` — graceful degradation (current_price=None)
3. `test_enrich_grid_position_no_simulator` — tp_price fallback 0.0
4. `test_enrich_grid_position_per_level_pnl` — 3 niveaux LONG avec prix différents
5. `test_enrich_grid_position_short` — P&L correct pour SHORT
6. `test_enrich_grid_position_boltrend_nan_tp` — get_tp_price() retourne NaN → tp_price=None
7. `test_get_status_includes_enriched_fields` — get_status() contient les nouveaux champs
8. `test_get_status_executor_grid_state` — format paper-compatible avec summary
9. `test_ws_merge_live_grids` — `_build_update_data()` merge live dans grid_state
10. `test_ws_live_overrides_paper` — même clé strategy:symbol → live gagne

Pattern : réutiliser les helpers de `test_executor.py` (`_make_config()`, MagicMock, GridLiveState)

---

## Vérification

1. `uv run pytest tests/test_executor_enriched_status.py -v` — tous les tests passent
2. `uv run pytest tests/ -x` — aucune régression
3. Démarrer le serveur local (`dev.bat`), ouvrir le dashboard :
   - ExecutorPanel : vérifier TP estimé, SL distance, P&L %, durée, niveaux
   - ActivePositions : badge LIVE sur les grids live, PAPER sur les paper
   - Scanner : GridDetail affiche les mêmes données pour live et paper
4. Vérifier que sans DataEngine (startup), les champs enrichis sont None sans erreur

## Commit

```
feat(executor): enriched live position metrics — TP estimate, P&L, distances, per-level details
```
