# Sprint 63a — Overview Dashboard + Equity Live Enrichie + Drawdown Chart

## Contexte
Le dashboard scalp-radar a toutes les données mais la page d'accueil (Overview) est basique et l'equity curve live manque d'interactivité. Ce sprint enrichit les 3 composants les plus visibles : OverviewPage, equity curve live, drawdown chart.

## Divergence trouvee
**`DrawdownChart.jsx` existe deja** (115 lignes, SVG) et est importe par `PortfolioPage.jsx` (ligne 3, utilise ligne 679) avec l'interface `{ curves, height, killSwitchPct }`. Le spec dit "Fichiers a creer" mais il existe.

**Approche retenue** : Reecrire DrawdownChart en Recharts en gardant la compatibilite arriere (accepte `curves` OU `equityPoints`). PortfolioPage continue de fonctionner sans modification.

---

## Fichiers crees (4)

| Fichier | Contenu |
|---------|---------|
| `frontend/src/components/EnhancedEquityCurve.jsx` | Recharts AreaChart, gradient, tooltip, regime overlay, period selector |
| `frontend/src/components/EnhancedEquityCurve.css` | Tooltip custom, period buttons, container |
| `frontend/src/components/OverviewPage.css` | KPI grid 4 cols, responsive 2x2, assets grid, bottom grid |
| `frontend/src/components/DrawdownChart.css` | Tooltip custom DD |

## Fichiers modifies (2)

| Fichier | Changements |
|---------|-------------|
| `frontend/src/components/OverviewPage.jsx` | Refonte complete : KPI cards + EnhancedEquityCurve + DrawdownChart + Top/Bottom assets + Regime + Activity |
| `frontend/src/components/JournalPage.jsx` | Remplacer LiveEquityCurve par EnhancedEquityCurve + ajouter DrawdownChart, supprimer fonction LiveEquityCurve inline |

## Fichier reecrit (1)

| Fichier | Changements |
|---------|-------------|
| `frontend/src/components/DrawdownChart.jsx` | SVG -> Recharts AreaChart, accepter `equityPoints` ET `curves` (backward compat PortfolioPage) |

---

## Decisions techniques

1. **DrawdownChart autonome** : fetch ses propres donnees via `useApi` — zero couplage avec EnhancedEquityCurve. PortfolioPage passe `curves={[...]}` (mode legacy), JournalPage/OverviewPage passent `strategy`+`days` (mode autonome).

2. **Regime bands** : grouper les entries history consecutives par regime → blocs `{regime, startTs, endTs}` → trouver timestamp equity le plus proche via scan lineaire → `ReferenceArea x1/x2`. Pas d'interpolation necessaire car Recharts matche sur dataKey exact.

3. **CSS uniquement via CSS variables** : `var(--accent)`, `var(--red)`, `var(--yellow)`, `var(--orange)` — zero hex hardcode dans les CSS.

4. **`/api/journal/events` confirme existant** (journal_routes.py:47) — utilise pour la section activite dans OverviewPage.

## Layout OverviewPage

```
+------+------+------+------+
| Cap  | P&L  | WR%  | DD%  |  <- KPI cards (4 cols, 2x2 mobile)
+------+------+------+------+
|  EnhancedEquityCurve       |  <- pleine largeur, 250px
|  DrawdownChart             |  <- 120px sous equity
+----------------------------+
| Top 3 gagnants | Bot 3 per |  <- 2 cols
+----------------------------+
| Regime BTC    | Activite   |  <- 2 cols
+----------------------------+
| Strategies (collapsed)     |
| Positions (collapsed)      |
+----------------------------+
```

## Resultats

- **0 nouveau test** (frontend pur)
- **2172 tests, 2166 passants** (6 pre-existants), 0 regression
- Build `npm run build` : OK, 742 modules transformes

## Verification

```bash
cd frontend && npm run build
```

Checklist :
- [x] Build passe sans erreur
- [x] PortfolioPage fonctionne (DrawdownChart backward compat avec `curves`)
- [x] OverviewPage s'affiche quand `activeStrategy === 'overview'`
- [x] EnhancedEquityCurve : period selector 7j/30j/90j/Tout fonctionne
- [x] Regime bands visibles sur equity curve
- [x] DrawdownChart avec kill switch -45% en pointille rouge
- [x] KPI cards responsive : 4 cols desktop, 2x2 mobile
