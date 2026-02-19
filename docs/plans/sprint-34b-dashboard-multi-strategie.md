# Sprint 34b — Dashboard Multi-Stratégie (Frontend)

## Contexte

Grid_atr (10 assets, LIVE) et grid_boltrend (6 assets, PAPER) tournent en parallèle.
Le dashboard actuel ne distingue pas les stratégies : positions, stats, scanner mélangent tout.
Ce sprint ajoute une barre de navigation par stratégie au-dessus des tabs existants,
avec filtrage côté client des données WS.

**Bug fix inclus** : `grid_positions` est keyed par `symbol` — collision quand 2 stratégies
partagent un symbol (BTC, DOGE, DYDX). Fix : key `strategy:symbol`.

---

## Fichiers modifiés / créés

### Backend (1 fichier, 1 ligne)

- [simulator.py](backend/backtesting/simulator.py) L2110 :
  - Changer `{g["symbol"]: g for g in grids}` → `{f'{g["strategy"]}:{g["symbol"]}': g for g in grids}`

### Frontend créés (4 fichiers)

- `frontend/src/contexts/StrategyContext.jsx` (~40 lignes)
- `frontend/src/hooks/useFilteredWsData.js` (~80 lignes)
- `frontend/src/components/StrategyBar.jsx` (~60 lignes)
- `frontend/src/components/OverviewPage.jsx` (~90 lignes)

### Frontend modifiés (6 fichiers)

- `frontend/src/App.jsx` — wrapper StrategyProvider, filtrage wsData, overview
- `frontend/src/components/Header.jsx` — intégrer StrategyBar
- `frontend/src/components/Scanner.jsx` — adapter gridLookup aux nouvelles clés
- `frontend/src/components/ActivePositions.jsx` — adapter aux nouvelles clés
- `frontend/src/components/SessionStats.jsx` — aucun changement de code nécessaire (reçoit wsData déjà filtrées)
- `frontend/src/styles.css` — styles strategy-bar

---

## Plan détaillé

### Étape 0 — Fix backend : clé grid_positions (1 ligne)

**Fichier** : [simulator.py:2110](backend/backtesting/simulator.py#L2110)

```python
# Avant
"grid_positions": {g["symbol"]: g for g in grids},
# Après
"grid_positions": {f'{g["strategy"]}:{g["symbol"]}': g for g in grids},
```

Impact frontend : tous les accès à `grid_positions[symbol]` doivent chercher
par suffix `:symbol` ou par itération sur les values. Le hook `useFilteredWsData`
gère cette adaptation.

### Étape 1 — StrategyContext

**Fichier** : `frontend/src/contexts/StrategyContext.jsx`

```jsx
const StrategyContext = createContext({
  activeStrategy: "overview",   // "overview" | "grid_atr" | "grid_boltrend" | ...
  setActiveStrategy: () => {},
  strategyFilter: null,         // null (overview) | "grid_atr" | "grid_boltrend"
})
```

- `activeStrategy` persisté dans `localStorage('scalp-radar-strategy')`
- `strategyFilter` dérivé : `overview → null`, sinon = activeStrategy
- Provider wrappé autour de tout dans App.jsx

### Étape 2 — StrategyBar

**Fichier** : `frontend/src/components/StrategyBar.jsx`

- Reçoit `wsData` en props
- Extrait la liste des stratégies depuis :
  - `Object.keys(wsData.strategies)` (noms des runners)
  - Plus "overview" toujours en premier
- Indicateur live/paper : `wsData.executor?.enabled` + `wsData.executor?.selector?.allowed_strategies`
  contient le nom → ● (live), sinon ○ (paper)
- Boutons pill, style distinct des tabs (fond plus foncé, bordure accent)
- Quand activeStrategy non reconnu dans les données → fallback "overview"

### Étape 3 — useFilteredWsData hook

**Fichier** : `frontend/src/hooks/useFilteredWsData.js`

Le hook centralise TOUT le filtrage. Les composants enfants reçoivent des wsData
pré-filtrées sans connaître le concept de stratégie.

```js
function useFilteredWsData(wsData, strategyFilter) {
  return useMemo(() => {
    if (!strategyFilter || !wsData) return wsData

    // 1. Filtrer grid_state.grid_positions (clé strategy:symbol)
    //    Garder uniquement les entrées dont strategy === strategyFilter
    const filteredGridPositions = {}
    for (const [key, g] of Object.entries(wsData.grid_state?.grid_positions || {})) {
      if (g.strategy === strategyFilter) {
        filteredGridPositions[key] = g
      }
    }
    // Recalculer summary
    const filteredGridState = {
      grid_positions: filteredGridPositions,
      summary: recalcSummary(filteredGridPositions),
    }

    // 2. Filtrer strategies (dict keyed par runner name)
    const filteredStrategies = {}
    for (const [name, s] of Object.entries(wsData.strategies || {})) {
      if (name === strategyFilter) {
        filteredStrategies[name] = s
      }
    }

    // 3. Filtrer executor.positions par strategy_name
    const filteredExecutor = wsData.executor ? {
      ...wsData.executor,
      positions: (wsData.executor.positions || [])
        .filter(p => p.strategy_name === strategyFilter),
    } : wsData.executor

    // 4. Filtrer simulator_positions par strategy
    const filteredSimPositions = (wsData.simulator_positions || [])
      .filter(p => (p.strategy_name || p.strategy) === strategyFilter)

    return {
      ...wsData,
      grid_state: filteredGridState,
      strategies: filteredStrategies,
      executor: filteredExecutor,
      simulator_positions: filteredSimPositions,
    }
  }, [wsData, strategyFilter])
}
```

### Étape 4 — OverviewPage

**Fichier** : `frontend/src/components/OverviewPage.jsx`

Tableau résumé quand `activeStrategy === "overview"` et `activeTab === "scanner"`.

Pour chaque stratégie (issue des clés de `wsData.strategies`) :
- Nombre d'assets = nombre d'entrées dans grid_positions pour cette stratégie
- Nombre de grids = somme levels_open
- P&L = somme unrealized_pnl des grids + realized (net_pnl) de strategies[name]
- Marge = somme margin_used des grids
- Badge ● LIVE / ○ PAPER

Clic sur une ligne → `setActiveStrategy(strategyName)`.

Sous le tableau, afficher `ActivePositions` non filtré (toutes stratégies).

### Étape 5 — Adaptation Scanner.jsx

Le `gridLookup` actuel fait `wsData.grid_state.grid_positions` keyed par symbol.
Avec la nouvelle clé `strategy:symbol`, il faut construire un lookup par symbol
pour le Scanner (qui affiche une ligne par symbol, pas par strategy:symbol).

```js
// Construire gridLookup par symbol (pour le Scanner)
// Si plusieurs stratégies sur le même symbol : prendre celui filtré
// Grâce au filtrage useFilteredWsData, grid_positions ne contient
// que la stratégie sélectionnée (ou toutes en overview)
const gridLookup = useMemo(() => {
  const positions = wsData?.grid_state?.grid_positions || {}
  const lookup = {}
  for (const g of Object.values(positions)) {
    // Si même symbol pour 2 stratégies (overview), prendre le 1er trouvé
    if (!lookup[g.symbol]) lookup[g.symbol] = g
  }
  return lookup
}, [wsData?.grid_state?.grid_positions])
```

Scanner filtre aussi les assets affichés quand une stratégie est sélectionnée :
- Extraire la liste des symbols actifs pour cette stratégie depuis grid_positions
- Ne montrer que ces symbols dans le tableau (pas tous les 21 assets)

### Étape 6 — Adaptation ActivePositions.jsx

Même adaptation que Scanner : construire le dict par symbol depuis les values
de `grid_positions` (qui sont déjà filtrées par useFilteredWsData).

```js
const gridLookup = {}
const grids = Object.values(gridState?.grid_positions || {})
for (const g of grids) {
  gridLookup[g.symbol] = g  // OK car déjà filtré par stratégie
}
```

Le reste du composant utilise déjà `grid_positions` via les values, pas les clés.
Adaptation minimale.

### Étape 7 — Intégration App.jsx

```jsx
import { StrategyProvider, useStrategyContext } from './contexts/StrategyContext'
import useFilteredWsData from './hooks/useFilteredWsData'
import OverviewPage from './components/OverviewPage'

function AppContent() {
  // ... tout le code actuel de App() ...
  const { activeStrategy, strategyFilter } = useStrategyContext()
  const filteredWsData = useFilteredWsData(lastUpdate, strategyFilter)

  // Remplacer wsData par filteredWsData partout
  // Sauf pour Header/StrategyBar qui reçoivent lastUpdate brut

  // Conditionnel Scanner vs OverviewPage
  {activeTab === 'scanner' && activeStrategy === 'overview'
    ? <OverviewPage wsData={lastUpdate} />
    : activeTab === 'scanner'
    ? <Scanner wsData={filteredWsData} />
    : null}
}

export default function App() {
  return (
    <StrategyProvider>
      <AppContent />
    </StrategyProvider>
  )
}
```

Pages non filtrées (Recherche, Explorer, Portfolio, Journal, Logs) : reçoivent
`lastUpdate` brut (pas filtré). La strategy bar est visible mais n'affecte pas
ces pages.

### Étape 8 — Header.jsx + StrategyBar

Header reçoit la StrategyBar en tant que composant enfant au-dessus des tabs :

```jsx
<header className="header">
  <div className="header-top">
    <span className="header-logo">SCALP RADAR</span>
    <span className="header-version">v0.7.0</span>
    <StrategyBar wsData={wsData} />
    <div className="header-right">...</div>
  </div>
  <div className="tabs">...</div>
</header>
```

Réorganisation : le header passe sur 2 lignes (logo+strategy bar en haut, tabs en bas).

### Étape 9 — CSS styles.css

Ajouter les styles pour `.strategy-bar` :

```css
/* Strategy bar */
.strategy-bar {
  display: flex;
  gap: 4px;
  padding: 2px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: var(--radius-sm);
}
.strategy-btn {
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
  color: var(--text-muted);
  cursor: pointer;
  border: 1px solid transparent;
  background: transparent;
  transition: all 0.15s;
  display: flex;
  align-items: center;
  gap: 6px;
}
.strategy-btn:hover {
  color: var(--text-secondary);
  border-color: var(--border);
}
.strategy-btn.active {
  background: rgba(255, 255, 255, 0.06);
  color: var(--text-primary);
  font-weight: 600;
  border-color: var(--accent);
}
.strategy-dot--live { color: var(--accent); }
.strategy-dot--paper { color: var(--yellow); }
```

---

## Résumé des changements par composant

| Composant | Type de changement |
|---|---|
| simulator.py | 1 ligne : clé `strategy:symbol` |
| StrategyContext | NOUVEAU : context + provider |
| StrategyBar | NOUVEAU : barre navigation stratégie |
| useFilteredWsData | NOUVEAU : hook filtrage centralisé |
| OverviewPage | NOUVEAU : résumé stratégies |
| App.jsx | Wrapper provider, filtrage, overview conditionnel |
| Header.jsx | Layout 2 lignes, intègre StrategyBar |
| Scanner.jsx | gridLookup par values (pas par clé), filtre assets |
| ActivePositions.jsx | gridLookup par values |
| styles.css | Styles strategy-bar |
| SessionStats.jsx | Aucun changement (reçoit wsData pré-filtrées) |
| ExecutorPanel.jsx | Aucun changement (reçoit wsData pré-filtrées) |
| ActivityFeed.jsx | Aucun changement (reçoit wsData pré-filtrées) |

---

## Vérification

1. `uv run pytest tests/ -x -q` — les tests backend passent (1 ligne changée)
2. `dev.bat` — vérifier visuellement :
   - Strategy bar visible avec [Overview] [grid_atr ●] [grid_boltrend ○]
   - Clic grid_atr → Scanner montre 10 assets, positions ATR seulement
   - Clic grid_boltrend → Scanner montre 6 assets, positions BOLT seulement
   - Clic Overview → OverviewPage résumé + tout non filtré dans sidebar
   - Pages Recherche/Explorer/Portfolio/Journal/Logs non affectées
   - Refresh → sélection persistée dans localStorage
   - Sidebar (SessionStats, ExecutorPanel) filtrée correctement
