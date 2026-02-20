# Plan : Multi-Asset WFO Launch dans l'Explorateur

## Contexte

La page Explorer permet actuellement de sélectionner une stratégie + un seul asset, puis de lancer un WFO. L'utilisateur doit répéter l'opération manuellement pour chaque asset, ce qui est fastidieux sur 21 assets.

La demande : sélectionner plusieurs assets d'un coup, voir ceux déjà testés (avec leur grade) et ceux non testés (grisés), puis lancer le WFO en un clic pour tous les assets cochés.

Choix UX confirmés :
- **Heatmap indépendante** : un clic "simple" sur un asset = visualisation heatmap (état `viewAsset`), une checkbox = sélection WFO (état `selectedAssets`). Les deux sont découplés.
- **Source assets** : tous les assets de `assets.yaml` (pas seulement ceux déjà testés en DB)

---

## Fichiers à modifier

1. `backend/api/optimization_routes.py` — nouvel endpoint
2. `frontend/src/components/ExplorerPage.jsx` — refonte sélection assets
3. `frontend/src/components/ExplorerPage.css` — styles nouveaux éléments

Aucun changement dans `job_manager.py` ni en DB (submit_job accepte déjà N appels séquentiels).

---

## 1. Backend — `optimization_routes.py`

### Nouvel endpoint `GET /api/optimization/assets-status`

```python
@router.get("/assets-status")
async def get_assets_status(
    strategy: str = Query(..., description="Nom de la stratégie"),
) -> dict:
    """Retourne tous les assets configurés avec leur statut WFO pour la stratégie.

    Lit assets.yaml pour la liste complète, joint avec optimization_results (is_latest=1)
    pour avoir le grade/score si déjà testé.

    Returns:
        {
            "assets": [
                {"symbol": "BTC/USDT", "tested": true,  "grade": "A", "total_score": 92, "result_id": 42},
                {"symbol": "ETH/USDT", "tested": true,  "grade": "B", "total_score": 75, "result_id": 43},
                {"symbol": "SOL/USDT", "tested": false, "grade": null, "total_score": null, "result_id": null},
                ...
            ]
        }
    """
```

Logique :
1. Charger `config/assets.yaml`, extraire la liste de symbols (trié alphabétiquement)
2. Query DB : `SELECT asset, grade, total_score, id FROM optimization_results WHERE strategy_name=? AND is_latest=1`
3. Construire un dict `{symbol: {grade, total_score, id}}` depuis DB
4. Retourner la liste complète avec `tested=True/False` selon présence dans le dict DB

---

## 2. Frontend — `ExplorerPage.jsx`

### Nouveaux états

```javascript
// Remplacer l'ancien : const [asset, setAsset] = usePersistedState('explorer-asset', '')
const [viewAsset, setViewAsset] = usePersistedState('explorer-view-asset', '')   // pour heatmap
const [selectedAssets, setSelectedAssets] = usePersistedState('explorer-wfo-assets', [])  // pour WFO

// Nouvelle source d'assets (inclut le statut WFO)
const [assetsStatus, setAssetsStatus] = useState([])  // [{symbol, tested, grade, total_score}]
```

### Chargement des assets

Remplacer le `useEffect` qui chargeait depuis `/api/optimization/results` par :

```javascript
useEffect(() => {
  if (!strategy) { setAssetsStatus([]); return }
  fetch(`/api/optimization/assets-status?strategy=${encodeURIComponent(strategy)}`)
    .then(r => r.json())
    .then(data => setAssetsStatus(data.assets || []))
    .catch(() => setAssetsStatus([]))
}, [strategy])
```

### Compatibilité `fetchAvailableRuns`

Remplacer toutes les références à `asset` par `viewAsset` dans :
- `fetchAvailableRuns()` (ligne 151)
- `useEffect([strategy, asset])` (ligne 196)
- `useEffect(heatmap)` (ligne 202)
- Messages UI "Sélectionnez une stratégie et un asset" (ligne 637)
- Titre heatmap "Heatmap {strategy} × {asset}" (ligne 649)

### Nouvelle UI de sélection d'assets (remplace le `<select>`)

Structure HTML :
```jsx
<div className="form-group">
  <div className="assets-list-header">
    <label>Assets</label>
    <div className="assets-shortcuts">
      <button onClick={selectAll}>Tout</button>
      <button onClick={selectUntested}>Non testés</button>
      <button onClick={clearAll}>Effacer</button>
    </div>
  </div>
  <div className="assets-list">
    {assetsStatus.map(a => (
      <div
        key={a.symbol}
        className={`asset-row ${viewAsset === a.symbol ? 'view-active' : ''} ${!a.tested ? 'untested' : ''}`}
        onClick={() => setViewAsset(a.symbol)}   // clic simple = heatmap
      >
        <input
          type="checkbox"
          checked={selectedAssets.includes(a.symbol)}
          onChange={e => toggleAssetSelection(a.symbol, e.target.checked)}
          onClick={e => e.stopPropagation()}      // empêche le double-clic
        />
        <span className="asset-symbol">{a.symbol.replace('/USDT', '')}</span>
        {a.tested
          ? <span className={`grade-badge grade-${a.grade}`}>{a.grade}</span>
          : <span className="grade-badge grade-none">—</span>
        }
      </div>
    ))}
  </div>
</div>
```

### `handleSubmitJob` — loop multi-assets

```javascript
const handleSubmitJob = async () => {
  if (!strategy || selectedAssets.length === 0) {
    alert('Sélectionner une stratégie et au moins un asset')
    return
  }

  // Filtrer les assets déjà en pending/running
  const alreadyRunning = selectedAssets.filter(asset =>
    jobs.some(j => j.strategy_name === strategy && j.asset === asset &&
      (j.status === 'pending' || j.status === 'running'))
  )
  if (alreadyRunning.length > 0) {
    alert(`${alreadyRunning.length} asset(s) déjà en cours : ${alreadyRunning.join(', ')}`)
    // Continuer quand même pour les autres
  }

  const toRun = selectedAssets.filter(a => !alreadyRunning.includes(a))
  if (toRun.length === 0) return

  setLoading(true)
  setError(null)
  const finalOverride = Object.keys(override).length > 0 ? override : null

  let submitted = 0
  for (const asset of toRun) {
    try {
      const resp = await fetch('/api/optimization/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strategy_name: strategy, asset, params_override: finalOverride }),
      })
      if (resp.ok) submitted++
    } catch { /* continue */ }
  }

  await fetchJobs()
  setLoading(false)
}
```

### Bouton Lancer WFO

```jsx
<button
  onClick={handleSubmitJob}
  disabled={!strategy || selectedAssets.length === 0 || loading || activeJobsCount >= 5}
  className="btn btn-primary"
>
  {loading ? 'Lancement...' : `Lancer WFO (${selectedAssets.length})`}
</button>
```

### `activeJobsCount` guard

Remplacer la condition dans le bouton par `activeJobsCount >= 5` comme avant (le backend gère le doublon avec 409).

---

## 3. CSS — `ExplorerPage.css`

Ajouter les styles suivants :

```css
/* Liste assets multi-sélection */
.assets-list-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
}

.assets-shortcuts {
  display: flex;
  gap: 4px;
}

.assets-shortcuts button {
  font-size: 10px;
  padding: 2px 6px;
  border: 1px solid #444;
  border-radius: 4px;
  background: #2a2a2a;
  color: #aaa;
  cursor: pointer;
}

.assets-shortcuts button:hover { background: #333; color: #fff; }

.assets-list {
  max-height: 280px;
  overflow-y: auto;
  border: 1px solid #333;
  border-radius: 6px;
  background: #111;
}

.asset-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 8px;
  cursor: pointer;
  transition: background 0.1s;
  border-bottom: 1px solid #1e1e1e;
}

.asset-row:hover { background: #1f1f1f; }
.asset-row.view-active { background: #1a2a1a; border-left: 2px solid #10b981; }
.asset-row.untested { opacity: 0.55; }

.asset-symbol { flex: 1; font-size: 12px; color: #ccc; }

/* Badges grades */
.grade-badge {
  font-size: 10px;
  font-weight: 700;
  padding: 1px 5px;
  border-radius: 3px;
  min-width: 18px;
  text-align: center;
}
.grade-A  { background: #064e3b; color: #34d399; }
.grade-B  { background: #1e3a5f; color: #60a5fa; }
.grade-C  { background: #451a03; color: #fb923c; }
.grade-D  { background: #3b1515; color: #f87171; }
.grade-F  { background: #1f1f1f; color: #6b7280; }
.grade-none { background: #1f1f1f; color: #444; }
```

---

## Ordre d'implémentation

1. **Backend** : ajouter `GET /assets-status` dans `optimization_routes.py`
2. **Frontend** : modifier `ExplorerPage.jsx` (états, chargement, UI, submit)
3. **CSS** : ajouter les styles dans `ExplorerPage.css`

---

## Vérification

1. Ouvrir l'Explorer, sélectionner une stratégie (ex: `grid_atr`)
2. La liste affiche les 21 assets → les déjà testés ont un grade (A/B/...), les autres sont grisés avec "—"
3. Cocher plusieurs assets non testés (checkboxes)
4. Clic simple sur un asset testé → heatmap s'affiche à droite
5. "Lancer WFO (3)" → 3 jobs apparaissent dans la liste des jobs
6. "Non testés" → sélectionne automatiquement tous les assets sans grade
7. Vérifier que les 409 (doublons) sont gérés sans bloquer les autres assets

Aucun test backend nouveau requis (le endpoint est trivial — lecture YAML + requête DB simple).
