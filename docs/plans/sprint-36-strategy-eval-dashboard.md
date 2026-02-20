# Sprint 36 — Strategy Evaluation Dashboard

## Context

Le workflow d'evaluation d'une strategie se fait manuellement entre CLI, conversations et navigation entre onglets sans lien. Ce sprint ajoute un guidage visuel du processus de decision **Explorer -> Recherche -> Portfolio** sans imposer un parcours lineaire. Regles deterministes, pas d'IA.

---

## Ajustements par rapport au spec (valides par exploration code)

| # | Spec | Code reel | Ajustement |
|---|------|-----------|------------|
| 1 | `funding_costs_total` dans PortfolioCompare | Pas en DB `portfolio_backtests` | Remplace par `max_drawdown_duration_hours` (deja en DB) |
| 2 | Route `GET /strategy-summary` | `GET /{result_id}` catch-all ligne 647 | Placer AVANT ligne 588 (apres `/apply`) |
| 3 | `per_asset_results` champ `pnl` | Champ reel = `net_pnl` | Utiliser `net_pnl` dans CompareVerdict |
| 4 | PortfolioPage hardcode `grid_atr` | Ligne 216 `launchBacktest` | Ajouter dropdown strategie + lecture localStorage |
| 5 | `portfolio_runs` query `label LIKE` | `strategy_name` existe en DB | Filtrer par `strategy_name = ?` (plus fiable) + fallback `label LIKE` |
| 6 | `best_params` simple JSON | Double-encoding possible (JSON in JSON) | Guard double-decode dans `get_strategy_summary_async` |
| 7 | Sync localStorage StrategyEvalBar <-> ResearchPage | `storage` event ne marche qu'entre onglets | App.jsx detient `evalStrategy` state, passe en props aux deux composants |
| 8 | Endpoint strategies a creer | `GET /api/optimization/strategies` existe deja (ligne 266) | Reutiliser directement dans StrategyEvalBar |

---

## Phase 1 — Backend : endpoint resume strategie

### 1A. `backend/optimization/optimization_db.py` — nouvelle fonction

Ajouter `get_strategy_summary_async(db_path, strategy_name) -> dict` :
- Query : `SELECT ... FROM optimization_results WHERE strategy_name = ? AND is_latest = 1`
- Calcul Python : grade distribution (Counter), red flags (seuils), convergence params (mode/pct), avg metrics
- **Double-JSON guard** pour `best_params` (convergence) :
  ```python
  params = json.loads(r["best_params"]) if isinstance(r["best_params"], str) else r["best_params"]
  if isinstance(params, str):
      params = json.loads(params)  # double-encoded
  ```
- Query portfolio : `SELECT ... FROM portfolio_backtests WHERE strategy_name = ? ORDER BY created_at DESC LIMIT 10`
- Retourne le dict complet tel que defini dans le spec (grades, red_flags, param_convergence, portfolio_runs)

### 1B. `backend/api/optimization_routes.py` — nouvelle route

```python
@router.get("/strategy-summary")
async def get_strategy_summary(strategy: str = Query(...)) -> dict:
```

**CRITIQUE** : placer AVANT le bloc `GET /combo-results/{result_id}` (ligne 588) et AVANT le catch-all `GET /{result_id}` (ligne 647). Sinon FastAPI matche "strategy-summary" comme un `result_id`.

Import a ajouter ligne 9-15 : `get_strategy_summary_async`

### 1C. Tests — `tests/test_strategy_summary.py` (4 tests)

1. `test_summary_empty_strategy` — 0 resultats, retourne `total_assets: 0`
2. `test_summary_grade_distribution` — grades correctement comptes
3. `test_summary_red_flags` — oos_is_ratio suspect, underpowered, etc.
4. `test_summary_param_convergence` — mode + pourcentage calcules

Utiliser une DB temp en memoire avec les tables `optimization_results` + `portfolio_backtests`.

---

## Phase 2 — Frontend : StrategyEvalBar (composant global)

### 2A. `frontend/src/components/StrategyEvalBar.jsx` (nouveau)

Bandeau horizontal sous les tabs, visible uniquement sur Explorer/Recherche/Portfolio.

**Contenu** :
- Dropdown strategie (fetch `GET /api/optimization/strategies` — endpoint existant ligne 266) — persiste via `evalStrategy`/`setEvalStrategy` recus en props depuis App.jsx
- 3 pills cliquables (WFO / Audit / Portfolio) avec couleurs conditionnelles
- Badge recommandations (nombre + dropdown panel au clic)
- Utilise le hook `useRecommendations` (Phase 3)

**Props** : `activeTab`, `onNavigate`, `evalStrategy`, `setEvalStrategy`, `summary` (optionnel, fetch interne sinon)

**Logique affichage** : `if (!['explorer', 'research', 'portfolio'].includes(activeTab)) return null`

### 2B. `frontend/src/components/StrategyEvalBar.css` (nouveau)

Styles dark theme : `.eval-bar`, `.eval-pill--green/yellow/orange/gray`, `.eval-reco-panel`, `.eval-reco-badge`

### 2C. `frontend/src/App.jsx` — integration

**State central** : App.jsx detient `evalStrategy` via `usePersistedState('eval-strategy', '')` et le passe en props. Cela evite la synchronisation localStorage manuelle entre composants du meme onglet.

```jsx
import StrategyEvalBar from './components/StrategyEvalBar'
const [evalStrategy, setEvalStrategy] = usePersistedState('eval-strategy', '')

// Apres <Header .../>, avant <div className="main-grid" ...>
<StrategyEvalBar
  activeTab={activeTab}
  onNavigate={handleTabChange}
  evalStrategy={evalStrategy}
  setEvalStrategy={setEvalStrategy}
/>
```

Passer `evalStrategy` + `onTabChange` a ResearchPage et PortfolioPage :

```jsx
{activeTab === 'research' && (
  <ResearchPage onTabChange={handleTabChange} evalStrategy={evalStrategy} setEvalStrategy={setEvalStrategy} />
)}
{activeTab === 'portfolio' && (
  <PortfolioPage wsData={lastUpdate} lastEvent={lastEvent} evalStrategy={evalStrategy} />
)}
```

---

## Phase 3 — Frontend : hook useRecommendations

### 3A. `frontend/src/hooks/useRecommendations.js` (nouveau)

Hook pur `useMemo` qui prend un `summary` et retourne une liste de recommandations triees par priorite.

Regles :
- **error** : 0 A/B, >50% underpowered, forward test negatif
- **recommended** : >=5 A/B sans portfolio run, backtest OK sans forward test
- **warning** : >50% OOS/IS suspects, forward faible (<20%)
- **info** : WFO > 30 jours
- **success** : forward >= 20%

Tri : error > recommended > warning > info > success

---

## Phase 4 — Frontend : StrategySummaryPanel (Recherche)

### 4A. `frontend/src/components/StrategySummaryPanel.jsx` (nouveau)

Affiche en haut de ResearchPage quand `filters.strategy` est non-vide.

**Sections** :
1. **Grade distribution** : 5 badges colores (reutiliser `.grade-badge .grade-A` etc de ResearchPage.css)
2. **Red flags** : liste avec icones, seuils definis (oos_is_ratio > 1.5, sharpe > 20, underpowered, consistency < 0.5, stability < 0.3)
3. **Convergence params** : tableau compact (check/warning/cross selon mode_pct)
4. **Verdict** : texte genere par regles deterministes (`generateVerdict(summary)`)
5. **Bouton action** : "Tester en portfolio" -> navigue via `onNavigatePortfolio`

**Props** : `strategyName`, `onNavigatePortfolio`
**Data** : fetch `GET /api/optimization/strategy-summary?strategy=X`

### 4B. `frontend/src/components/ResearchPage.jsx` — integration

- Accepter nouvelles props `onTabChange`, `evalStrategy`, `setEvalStrategy`
- **Sync bidirectionnelle par props** (pas localStorage) :
  - Quand `evalStrategy` change (venant de la EvalBar) : `useEffect` met a jour `filters.strategy`
  - Quand `filters.strategy` change (dropdown Recherche) : appeler `setEvalStrategy(filters.strategy)`
- Apres les filtres, avant le tableau :

  ```jsx
  {filters.strategy && (
    <StrategySummaryPanel
      strategyName={filters.strategy}
      onNavigatePortfolio={() => onTabChange?.('portfolio')}
    />
  )}
  ```

---

## Phase 5 — Frontend : Portfolio — Compare enrichi

### 5A. `frontend/src/components/PortfolioCompare.jsx` — enrichissement

Ajouter a METRICS :
```javascript
{ label: 'Return/DD ratio', key: '_return_dd_ratio', fmt: v => v.toFixed(1), computed: true },
{ label: 'DD duree max', key: 'max_drawdown_duration_hours', fmt: v => {
  const d = Math.floor(v / 24); const h = Math.round(v % 24);
  return d > 0 ? `${d}j ${h}h` : `${h}h`
}, inverted: true },
{ label: '% runners perte', key: '_losers_pct', fmt: v => `${v.toFixed(0)}%`, inverted: true, computed: true },
```

Champs computes : calculer `_return_dd_ratio` et `_losers_pct` depuis les donnees existantes avant le render.

### 5B. `frontend/src/components/CompareVerdict.jsx` (nouveau)

Sous-composant affiche sous le tableau delta. Regles :
- Return/DD ratio : si best > worst * 1.5 -> flag
- DD duree > 720h (30j) -> flag
- % losers > 20% -> flag
- Verdict final si un run domine tous les criteres

### 5C. `frontend/src/components/PortfolioPage.jsx` — modifications

1. **Strategie selectionnable** : ajouter `usePersistedState('portfolio-strategy', 'grid_atr')` + dropdown dans config panel. Lire `localStorage.getItem('scalp-radar-eval-strategy')` au montage comme valeur initiale si presente.
2. **`launchBacktest`** : remplacer `strategy_name: 'grid_atr'` par la strategie selectionnee
3. **Filtrer `backtests`** : `backtests.filter(b => !selectedStrategy || b.strategy_name === selectedStrategy)`
4. **CompareVerdict** : ajouter `<CompareVerdict runs={compareDetails} />` apres `<PortfolioCompare />`
5. **Lire preset** : si `localStorage.getItem('scalp-radar-portfolio-preset-strategy')` present, pre-selectionner

---

## Recapitulatif fichiers

### A creer (6)

| Fichier | Description |
|---------|-------------|
| `frontend/src/components/StrategyEvalBar.jsx` | Barre globale dropdown strategie + pills + recommandations |
| `frontend/src/components/StrategyEvalBar.css` | Styles dark theme |
| `frontend/src/components/StrategySummaryPanel.jsx` | Resume WFO dans Recherche (grades, red flags, convergence, verdict) |
| `frontend/src/components/CompareVerdict.jsx` | Verdict automatique dans Portfolio compare |
| `frontend/src/hooks/useRecommendations.js` | Moteur de regles recommandations (useMemo) |
| `tests/test_strategy_summary.py` | 4 tests backend endpoint |

### A modifier (6)

| Fichier | Changement |
|---------|-----------|
| `backend/optimization/optimization_db.py` | + `get_strategy_summary_async()` (~60 lignes) |
| `backend/api/optimization_routes.py` | + import + endpoint `GET /strategy-summary` (~15 lignes, AVANT catch-all) |
| `frontend/src/App.jsx` | + import StrategyEvalBar, insertion entre Header et main-grid, passer `onTabChange` a ResearchPage |
| `frontend/src/components/ResearchPage.jsx` | + prop `onTabChange`, + StrategySummaryPanel conditionnel, sync localStorage |
| `frontend/src/components/PortfolioPage.jsx` | + strategy dropdown, - hardcode grid_atr, + CompareVerdict, + lire preset localStorage |
| `frontend/src/components/PortfolioCompare.jsx` | + 3 metriques (return/DD, duree DD, % losers) |

---

## Ordre d'implementation

| # | Phase | Dependances | Fichiers |
|---|-------|-------------|----------|
| 1 | Backend DB function | Aucune | optimization_db.py |
| 2 | Backend route | Phase 1 | optimization_routes.py |
| 3 | Tests backend | Phases 1-2 | test_strategy_summary.py |
| 4 | useRecommendations hook | Aucune | useRecommendations.js |
| 5 | StrategyEvalBar | Phase 4 | StrategyEvalBar.jsx/css, App.jsx |
| 6 | StrategySummaryPanel | Phases 1-2 | StrategySummaryPanel.jsx, ResearchPage.jsx |
| 7 | CompareVerdict + enrichissement | Aucune | CompareVerdict.jsx, PortfolioCompare.jsx |
| 8 | PortfolioPage modifications | Phase 7 | PortfolioPage.jsx |

Phases 4-5 et 7-8 peuvent etre paralleles (Recherche et Portfolio independants).

---

## Verification

### Tests automatises
```bash
uv run pytest tests/test_strategy_summary.py -v
uv run pytest tests/ -k "optimization" --tb=short
```

### Tests manuels (checklist)
- [ ] StrategyEvalBar visible sur Explorer/Recherche/Portfolio, cache sur Scanner/Journal/Logs
- [ ] Dropdown liste les strategies avec resultats WFO
- [ ] Pills changent de couleur selon les donnees
- [ ] Clic sur pill navigue vers la bonne page
- [ ] StrategySummaryPanel appara quand filtre strategie actif dans Recherche
- [ ] Red flags, convergence, verdict affiches correctement
- [ ] "Tester en portfolio" navigue vers Portfolio
- [ ] CompareVerdict s'affiche quand 2 runs coches dans Portfolio
- [ ] Metriques enrichies dans le tableau compare
- [ ] PortfolioPage utilise la strategie selectionnee (pas grid_atr hardcode)
