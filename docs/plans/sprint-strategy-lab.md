# Sprint Strategy Lab — Documentation Interactive des Stratégies

## Contexte

Scalp Radar a 16 stratégies (4 Scalp, 4 Swing, 8 Grid/DCA), 2 actives (grid_atr LIVE, grid_boltrend PAPER). Le dashboard a 6 tabs. On ajoute un 7ème tab **"Stratégies"** entre Scanner et Recherche — documentation interactive pour comprendre chaque stratégie.

## Résultats de l'exploration

### Patterns existants confirmés
- **Tabs data-driven** : array `TABS` dans `App.jsx`, passé à `Header.jsx` qui mappe. Ajouter = 1 entry + validTabs + UNFILTERED_TABS + rendu conditionnel
- **CSS** : fichier `.css` co-localisé par composant. Variables globales dans `styles.css` (`--bg-card`, `--border`, `--accent`, `--red`, `--yellow`, `--blue`, `--text-primary`, `--text-secondary`, `--text-muted`, `--radius`, `--radius-sm`)
- **Navigation** : state-based (pas de router). Sub-views = `useState` local dans la page
- **Recharts NON installé** — à ajouter (`npm install recharts`)
- **`frontend/src/data/` n'existe PAS** — à créer
- **StrategyEvalBar** : visible seulement sur explorer/research/portfolio → pas de conflit
- **Grade badges** : classes `.grade-badge .grade-A` etc. déjà dans `styles.css`

### Fichiers à créer (7)
```
frontend/src/data/strategyMeta.js          ← Métadonnées 16 stratégies
frontend/src/components/StrategiesPage.jsx ← Catalogue (tab principal)
frontend/src/components/StrategiesPage.css
frontend/src/components/StrategyDetail.jsx ← Vue détail
frontend/src/components/StrategyDetail.css
frontend/src/components/guides/GridAtrGuide.jsx   ← Tutoriel interactif (Recharts)
frontend/src/components/guides/GenericGuide.jsx   ← Fallback SVG par type
```

### Fichier à modifier (1)
```
frontend/src/App.jsx  ← Ajouter tab + import + rendu conditionnel
```

---

## Plan d'implémentation

### Étape 1 — Installer Recharts
```bash
cd frontend && npm install recharts
```

### Étape 2 — `frontend/src/data/strategyMeta.js`
- Créer le dossier `frontend/src/data/`
- Source unique des métadonnées des 16 stratégies
- Exporter `STRATEGY_FAMILIES` (3 familles avec label, couleur CSS var, ordre d'affichage)
- Exporter `STRATEGIES` (array de 16 objets avec id, name, family, timeframe, type, direction, status, hasGuide, shortDesc, edge, strengths, weaknesses, keyParams, entryLogic, exitLogic, wfoGrade)
- **COPIER VERBATIM** le contenu du spec utilisateur — ne pas réinterpréter les descriptions, IDs, ou statuts

### Étape 3 — `StrategiesPage.jsx` + `.css`
**Composant** :
- Props : `{ onNavigate, setEvalStrategy }` (callback tab + pré-filtre stratégie pour Recherche)
- State : `selectedStrategy` (null = catalogue, string = vue détail), `familyFilter` ('all'|'grid'|'swing'|'scalp')
- Si `selectedStrategy !== null` → render `<StrategyDetail>`
- Sinon → catalogue de cartes groupées par famille
- Filtres famille = 4 boutons toggle en haut
- Cartes en CSS grid `repeat(auto-fill, minmax(240px, 1fr))`
- Badge statut (live=vert, paper=jaune, disabled=gris, replaced=gris barré)
- Badge grade (réutiliser `.grade-badge .grade-X` existant)
- Border-left coloré pour live/paper

**CSS** :
- Root `.strategies-page` avec padding 20px, max-width 1600px
- Utiliser les CSS variables existantes (`--bg-card`, `--border`, `--accent`, `--yellow`, `--text-primary`, etc.)
- Cards : background `var(--bg-card)`, border `var(--border)`, hover `var(--bg-card-hover)`
- Filtres : boutons type pill, actif = background accent-dim

### Étape 4 — `guides/GenericGuide.jsx`
- SVG inline (0 dépendance externe)
- Props : `{ strategy }` (objet depuis strategyMeta)
- `viewBox="0 0 600 250"`, `width="100%"`
- Couleurs via CSS variables (currentColor + hardcoded dark theme compat)

**Mapping explicite type → schéma SVG** :

| Schéma | Types mappés | Stratégies |
|--------|-------------|------------|
| **Mean Reversion** (prix oscillant SMA, flèches achat/vente) | `Mean Reversion` | grid_atr, envelope_dca, envelope_dca_short, bollinger_mr, vwap_rsi |
| **Trend Following** (prix en tendance, entrées directionnelles) | `Trend Following`, `Trend DCA`, `Trend + DCA`, `Trend Following DCA` | supertrend, grid_boltrend, grid_multi_tf, grid_trend |
| **Breakout** (channel horizontal, sortie violente) | `Breakout`, `Event-Driven` | donchian_breakout, boltrend, momentum, liquidation |
| **Range Trading** (canal, achats bas / ventes haut) | `Range Trading` | grid_range_atr |
| **Funding Arbitrage** (histogramme funding rate) | `Funding Arbitrage` | grid_funding, funding |

### Étape 5 — `StrategyDetail.jsx` + `.css`
**Composant** :
- Props : `{ strategyId, onBack, onNavigate, setEvalStrategy }`
- Lookup dans `STRATEGIES` par `strategyId`
- Sections :
  1. Header avec bouton retour + nom + badge statut
  2. Fiche résumé (type, direction, timeframe, grade, edge, strengths/weaknesses)
  3. Paramètres clés (tableau si `keyParams.length > 0`)
  4. Logique entrée/sortie (entryLogic, exitLogic)
  5. Guide interactif : si `hasGuide` → `React.lazy(() => import('./guides/GridAtrGuide'))` avec `Suspense`, sinon → `<GenericGuide strategy={strategy} />`
  6. "Aller plus loin" : liens cliquables vers research/explorer/portfolio. **Clic = pré-filtre** la stratégie via `setEvalStrategy(strategyId)` + `dispatchEvent('eval-strategy-change')` puis `onNavigate('research')`

**CSS** :
- Sections en cards `.sd-section` avec bg `#1a1a1a`, border `1px solid #333`, border-radius 8px
- Paramètres en grid `repeat(auto-fill, minmax(200px, 1fr))`
- Bouton retour : réutiliser style `.btn-back` existant (ResearchPage.css est global via import)

### Étape 6 — `guides/GridAtrGuide.jsx`
**Le composant le plus complexe** — tutoriel interactif en 7 étapes avec Recharts.

- Imports : `LineChart, Line, XAxis, YAxis, CartesianGrid, Area, ReferenceLine, ReferenceDot, ResponsiveContainer` depuis recharts
- `React.lazy` dans StrategyDetail pour lazy-loading

**Données — ZÉRO prix d'achat hardcodé** :
- ~18 points BTC-like (prix ~42000) avec `price`, `sma`, `atr` à chaque point
- ATR **variable** : ~450 au début → monte à ~640 pendant le crash → redescend à ~520 en recovery
- SMA avec lag crédible sur le prix (pas juste un décalage linéaire)
- Niveaux d'achat **calculés dynamiquement à chaque point** via :
  `level_i_price = dataPoint.sma - dataPoint.atr * (2.0 + i * 1.0)` pour i=0,1,2
- 2 scénarios partageant les ~12 premiers points (crash commun) :
  - `recovery` : 6 points supplémentaires où le prix remonte et croise la SMA
  - `disaster` : 6 points où le prix continue de chuter et touche le SL (avg_price × (1 - sl_percent/100))

**7 étapes** :
1. Prix + SMA seuls (explication moyenne)
2. ATR zone (volatilité variable)
3. Grille adaptative (3 niveaux qui bougent)
4. Crash → Achat Niveau 1
5. DCA → Niveaux 2 et 3 + prix moyen pondéré
6. Recovery → TP au croisement SMA
7. Scénario catastrophe → SL touché

**Navigation** : boutons Précédent/Suivant + dots cliquables + indicateur progression
**Panneau info** : description + encadré "Chiffres" + concept clé en gras
**Responsive** : `<ResponsiveContainer width="100%" height={350}>`

### Étape 7 — Intégration `App.jsx`

Modifications minimales :

1. Import : `import StrategiesPage from './components/StrategiesPage'`
2. TABS : ajouter `{ id: 'strategies', label: 'Stratégies' }` en position 2 (entre scanner et research)
3. UNFILTERED_TABS : ajouter `'strategies'`
4. `loadActiveTab` validTabs : ajouter `'strategies'`
5. Rendu conditionnel :

```jsx
{activeTab === 'strategies' && (
  <StrategiesPage onNavigate={handleTabChange} setEvalStrategy={setEvalStrategy} />
)}
```

Passer `setEvalStrategy` permet à StrategyDetail de pré-filtrer la page Recherche quand l'utilisateur clique "Voir les résultats WFO".

---

## Ordre d'exécution

1. `npm install recharts`
2. `strategyMeta.js`
3. `StrategiesPage.jsx` + `.css`
4. `GenericGuide.jsx`
5. `StrategyDetail.jsx` + `.css`
6. `GridAtrGuide.jsx`
7. `App.jsx` modifications
8. Vérification visuelle (navigateur)

---

## Vérification

1. Lancer `cd frontend && npm run dev`
2. Ouvrir le navigateur sur le dashboard
3. Vérifier que le tab "Stratégies" apparaît entre Scanner et Recherche
4. Vérifier les 16 cartes groupées par famille (Grid/DCA en premier)
5. Tester les filtres famille
6. Cliquer sur grid_atr → vérifier la vue détail avec le tutoriel interactif 7 étapes
7. Naviguer les 7 étapes, vérifier les dots cliquables
8. Vérifier que l'ATR varie et les niveaux sont calculés dynamiquement
9. Cliquer sur une stratégie sans guide → vérifier le GenericGuide SVG
10. Tester "Aller plus loin" → navigation vers Recherche/Explorer/Portfolio
11. Bouton retour → retour au catalogue
12. Vérifier que les 6 autres tabs fonctionnent toujours (pas de régression)
13. Vérifier la persistence localStorage du tab actif
14. `npm run build` pour vérifier le build de production (lazy loading recharts)
