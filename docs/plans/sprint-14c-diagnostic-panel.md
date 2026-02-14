# Plan — Sprint 14c : DiagnosticPanel (Analyse intelligente WFO)

## Objectif

Ajouter un encart "Diagnostic" dans l'Explorer qui analyse automatiquement les résultats WFO et produit des verdicts textuels en langage clair. L'utilisateur comprend en 10 secondes si sa stratégie est viable et pourquoi.

## Emplacement

Sous le header "Analyse des combos (N testées)", **AVANT** le Top 10 et les charts. Première chose visible après la heatmap.

## Fichiers modifiés

| Fichier | Action | Lignes estimées |
|---------|--------|-----------------|
| `frontend/src/components/DiagnosticPanel.jsx` | **CRÉER** | ~180 lignes |
| `frontend/src/components/DiagnosticPanel.css` | **CRÉER** | ~80 lignes |
| `frontend/src/components/ExplorerPage.jsx` | MODIFIER | ~8 lignes (import + insertion JSX) |

**0 changement backend.** Tout calculé côté frontend à partir de données déjà fetchées.

---

## 1. Nouveau composant `DiagnosticPanel.jsx`

### Props

```jsx
DiagnosticPanel({ combos, grade, totalScore, nWindows })
```

- `combos` : array des combo results (depuis `GET /api/optimization/combo-results/{id}`, déjà chargé dans `comboResults.combos`)
- `grade` : string `"A"|"B"|"C"|"D"|"F"` du run sélectionné (depuis `availableRuns.find(r => r.id === selectedRunId).grade`)
- `totalScore` : number 0-100 (depuis `availableRuns.find(r => r.id === selectedRunId).total_score`)
- `nWindows` : number — calculé dynamiquement via `Math.max(...combos.map(c => c.n_windows_evaluated))` (option 2 du brief, pas de hardcode)

### `nWindows` — source

Calculé dans ExplorerPage avant de passer la prop :
```jsx
const nWindows = Math.max(...comboData.combos.map(c => c.n_windows_evaluated || 0))
```
Fallback : si le résultat est 0 ou -Infinity (combos vide), ne pas afficher le diagnostic.

### Fonction helper `median(arr)`

```javascript
function median(arr) {
    if (!arr.length) return 0
    const sorted = [...arr].sort((a, b) => a - b)
    const mid = Math.floor(sorted.length / 2)
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}
```

### Fonction `analyzeResults(combos, grade, totalScore, nWindows)` → `verdicts[]`

Retourne un array de `{ level: "green"|"orange"|"red", title: string, text: string }`.

6 règles dans cet ordre :

**Règle 1 — Grade global**
- A/B → vert "Stratégie viable" + message grade/score + "Prête pour le paper trading."
- C → orange "Stratégie moyenne" + "Déployable avec surveillance renforcée."
- D/F → rouge "Stratégie non viable" + "Non recommandée pour le déploiement."

**Règle 2 — Consistance du best combo**
- Trouver `best = combos.find(c => c.is_best) || combos[0]`
- Calculer `consistencyPct = Math.round(best.consistency * 100)` et `consistencyWindows = Math.round(best.consistency * nWindows)`
- < 0.2 → rouge "Consistance catastrophique" (bruit statistique)
- < 0.5 → rouge "Consistance faible" (moins de la moitié)
- < 0.8 → orange "Consistance acceptable" (correct mais pas robuste)
- ≥ 0.8 → vert "Consistance excellente" (robuste)

**Règle 3 — Transfert IS → OOS**
- Si `best.is_sharpe > 5 && best.oos_sharpe < 1` → rouge "Overfitting détecté"
- Sinon si `best.oos_is_ratio < 0.3` → rouge "Dégradation sévère IS→OOS"
- Sinon si `best.oos_is_ratio < 0.7` → orange "Dégradation normale IS→OOS"
- Sinon → vert "Bon transfert IS→OOS"

**Règle 4 — Edge structurel (distribution)**
- Calculer `pctAbove1`, `pctAbove05`, `pctPositive` depuis `allOosSharpe` (tous les combos)
- `pctAbove1 > 0.5` → vert "Edge structurel fort"
- `pctAbove05 > 0.5` → orange "Edge modéré"
- `pctPositive > 0.5` → orange "Edge faible"
- Sinon → rouge "Pas d'edge structurel"

**Règle 5 — Volume de trades**
- Si `best.oos_trades < 30` → orange "Données insuffisantes" (min 30 pour signification)

**Règle 6 — Fenêtres partielles**
- Compter combos où `n_windows_evaluated < nWindows`
- Si > 30% du total → orange "Combos partielles" (stats moins fiables)

### Rendu

Structure JSX :
```
<div class="diagnostic-panel" style="border-left-color: {couleur la plus sévère}">
  <h4 class="diagnostic-title">
    <svg .../>  DIAGNOSTIC
  </h4>
  <div class="diagnostic-verdicts">
    {verdicts.map(v => (
      <div class="verdict-item">
        <svg class="verdict-icon" .../>  {/* cercle coloré SVG inline */}
        <div>
          <div class="verdict-title">{v.title}</div>
          <div class="verdict-text">{v.text}</div>
        </div>
      </div>
    ))}
  </div>
</div>
```

### Icônes SVG inline

Cercles colorés simples (pas d'emoji — rendu inconsistant cross-platform) :
```jsx
function StatusCircle({ level }) {
  const colors = { green: '#10b981', orange: '#f59e0b', red: '#ef4444' }
  return (
    <svg width="10" height="10" viewBox="0 0 10 10">
      <circle cx="5" cy="5" r="5" fill={colors[level]} />
    </svg>
  )
}
```

Icône titre "Diagnostic" : petit graphique SVG (barres) en blanc/gris.

### Guard

Si `combos` est vide ou nul, ou si `nWindows <= 0` → retourner `null` (ne pas afficher).

---

## 2. Nouveau fichier `DiagnosticPanel.css`

Style cohérent avec le dark theme de l'Explorer :

```css
.diagnostic-panel {
  background: #111827;          /* légèrement différent du fond #1a1a1a */
  border: 1px solid #333;
  border-left: 4px solid #666;  /* couleur dynamique via style inline */
  border-radius: 8px;
  padding: 16px;
  max-height: 300px;
  overflow-y: auto;
}

.diagnostic-title {
  margin: 0 0 12px 0;
  color: #fff;
  font-size: 14px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.diagnostic-verdicts {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.verdict-item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
}

.verdict-icon {
  flex-shrink: 0;
  margin-top: 3px;
}

.verdict-title {
  color: #fff;
  font-size: 13px;
  font-weight: 600;
}

.verdict-text {
  color: #9ca3af;          /* gris clair */
  font-size: 12px;
  line-height: 1.4;
  margin-top: 2px;
}
```

La `border-left-color` est appliquée dynamiquement en `style` inline selon le verdict le plus sévère :
- rouge si ≥1 verdict rouge → `#ef4444`
- orange si ≥1 verdict orange (et 0 rouge) → `#f59e0b`
- vert sinon → `#10b981`

---

## 3. Modification `ExplorerPage.jsx`

### 3a. Import (ligne 11, après InfoTooltip)

```jsx
import DiagnosticPanel from './DiagnosticPanel'
```

### 3b. Calcul du selectedRun (nouveau `useMemo`, après ligne 365)

```jsx
const selectedRun = useMemo(() => {
  return availableRuns.find(r => r.id === selectedRunId) || null
}, [availableRuns, selectedRunId])
```

### 3c. Insertion JSX (entre le `<h3>` et `<div className="analysis-top10">`, lignes 596-599)

Remplacer la section analysis existante pour insérer le diagnostic AVANT le Top10 :

```jsx
{comboResults && comboResults.combos && comboResults.combos.length > 0 && (
  <div className="analysis-section">
    <h3>Analyse des combos ({comboResults.combos.length} testées)</h3>

    {/* Diagnostic — première chose visible */}
    <DiagnosticPanel
      combos={comboResults.combos}
      grade={selectedRun?.grade || "?"}
      totalScore={selectedRun?.total_score || 0}
      nWindows={Math.max(...comboResults.combos.map(c => c.n_windows_evaluated || 0))}
    />

    {/* Top 10 pleine largeur */}
    <div className="analysis-top10">
      <Top10Table ... />
    </div>

    {/* Scatter + Distribution */}
    <div className="analysis-charts">
      <ScatterChart ... />
      <DistributionChart ... />
    </div>
  </div>
)}
```

Le reste du JSX (Top10, Scatter, Distribution) ne change pas. Seul le `DiagnosticPanel` est ajouté en première position dans `analysis-section`.

---

## Résumé effort

| Tâche | Effort |
|-------|--------|
| `DiagnosticPanel.jsx` (composant + logique 6 règles + SVG) | ~180 lignes |
| `DiagnosticPanel.css` (style dark theme) | ~80 lignes |
| `ExplorerPage.jsx` (import + useMemo selectedRun + insertion JSX) | ~8 lignes modifiées |
| **Total** | ~270 lignes, 3 fichiers |

**Complexité** : Faible. Tout est 100% frontend, pas de nouvel endpoint, pas de migration DB. Les données sont déjà disponibles dans `comboResults` et `availableRuns`.

**Tests** : Pas de test unitaire nécessaire — c'est un composant purement visuel avec de la logique conditionnelle simple. Testable manuellement en sélectionnant différents runs WFO dans l'Explorer.
