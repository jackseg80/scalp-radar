# Plan : Isolation des données sidebar par stratégie

## Contexte

Problème : quand on navigue vers `grid_boltrend` (paper-only) dans la StrategyBar,
la sidebar droite affiche encore les données globales de l'executor :
- ExecutorPanel montre "LIVE" au lieu de "SIMULATION ONLY"
- EquityCurve affiche la courbe agrégée de toutes les stratégies

Cause racine :
1. `useFilteredWsData.js` filtre `executor.positions` par stratégie mais laisse `executor.enabled` (global → toujours LIVE si l'executor est actif), `risk_manager.session_pnl` et `kill_switch` non filtrés
2. `EquityCurve` ne reçoit aucun prop, appelle `/api/simulator/equity` et `/api/journal/snapshots` sans filtre stratégie

---

## Changements

### 1. `frontend/src/hooks/useFilteredWsData.js`

Quand `strategyFilter` est actif :

**a) Executor** — déterminer si la stratégie est réellement live :
```javascript
const isStrategyLive = wsData.executor?.enabled &&
  (wsData.executor?.selector?.allowed_strategies || []).includes(strategyFilter)

const filteredExecutor = isStrategyLive
  ? { ...wsData.executor, positions: filteredPositions }
  : null  // → ExecutorPanel affiche "SIMULATION ONLY"
```

**b) kill_switch** — utiliser le kill switch du runner individuel (pas le global) :
```javascript
kill_switch: filteredStrategies[strategyFilter]?.kill_switch ?? wsData.kill_switch,
```

### 2. `backend/backtesting/simulator.py` — `Simulator.get_equity_curve()`

Ajouter paramètre `strategy: str | None = None` :
```python
def get_equity_curve(self, since: str | None = None, strategy: str | None = None) -> dict:
    runners = [r for r in self._runners if r.name == strategy] if strategy else self._runners
    # ... utiliser `runners` au lieu de `self._runners` pour le calcul
```
Cache invalidé séparément (ne pas partager le cache global si strategy est spécifié — utiliser un calcul direct sans cache pour les requêtes filtrées).

### 3. `backend/core/database.py` — `get_equity_curve_from_trades()`

Ajouter `strategy: str | None = None` :
```python
async def get_equity_curve_from_trades(self, since: str | None = None, strategy: str | None = None) -> list[dict]:
    query = "SELECT net_pnl, exit_time FROM simulation_trades"
    conditions, params = [], []
    if since:
        conditions.append("exit_time > ?"); params.append(since)
    if strategy:
        conditions.append("strategy_name = ?"); params.append(strategy)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY exit_time ASC"
    # Capital de base si since + strategy combinés : ajouter strategy_name = ? au calcul de base
```

### 4. `backend/api/conditions_routes.py` — `get_equity_curve()`

Ajouter `strategy: str | None = Query(None)` et passer aux deux appels :
```python
result = simulator.get_equity_curve(since=since, strategy=strategy)
equity = await db.get_equity_curve_from_trades(since=since, strategy=strategy)
```

### 5. `frontend/src/components/EquityCurve.jsx`

Accepter `strategyFilter` prop :
```javascript
export default function EquityCurve({ strategyFilter = null }) {
  const stratParam = strategyFilter ? `&strategy=${strategyFilter}` : ''
  const { data: equityData } = useApi(`/api/simulator/equity${stratParam ? '?' + stratParam.slice(1) : ''}`, 30000)

  // Journal snapshots = toujours global (pas de filtre stratégie possible)
  // Quand strategyFilter actif : forcer fallback trades (ignorer journal)
  const snapshots = journalData?.snapshots || []
  const useJournal = !strategyFilter && snapshots.length >= 2
```

### 6. `frontend/src/App.jsx`

Passer `strategyFilter` à `EquityCurve` :
```jsx
<EquityCurve strategyFilter={strategyFilter} />
```

---

## Fichiers modifiés

| Fichier | Changement |
|---------|-----------|
| `frontend/src/hooks/useFilteredWsData.js` | executor null si stratégie non-live, kill_switch par runner |
| `frontend/src/components/EquityCurve.jsx` | prop strategyFilter, skip journal si filtré |
| `frontend/src/App.jsx` | passer strategyFilter à EquityCurve |
| `backend/backtesting/simulator.py` | get_equity_curve(strategy=) |
| `backend/core/database.py` | get_equity_curve_from_trades(strategy=) |
| `backend/api/conditions_routes.py` | param strategy sur endpoint |

---

## Notes

- **Journal snapshots ignorés si stratégie filtrée** : les snapshots portfolio sont globaux (une ligne = toutes les stratégies). Impossible de les filtrer sans modifier le schéma DB. Fallback sur trades = suffisant pour une courbe par stratégie.
- **Cache `get_equity_curve`** : quand `strategy` est spécifié, recalculer sans utiliser le cache global (simpler — le cache est utile principalement pour l'affichage global temps réel).
- **Scalabilité** : toute nouvelle stratégie grid fonctionne automatiquement (même logique `allowed_strategies`).

---

## Vérification

1. `uv run pytest tests/ -x -q` — vérifier 0 régression (les signatures ajoutent des kwargs optionnels)
2. Dashboard avec grid_boltrend sélectionné :
   - ExecutorPanel → "SIMULATION ONLY" (plus "LIVE")
   - EquityCurve → courbe isolée de grid_boltrend uniquement
3. Dashboard avec grid_atr sélectionné (stratégie live) :
   - ExecutorPanel → "LIVE" avec positions filtrées
   - EquityCurve → courbe isolée de grid_atr
4. Overview (aucune stratégie) :
   - Comportement inchangé — toutes données globales
