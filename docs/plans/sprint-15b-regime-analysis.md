# Plan — Sprint 15b : Analyse par régime de marché

## Objectif

Classifier automatiquement chaque fenêtre OOS du WFO selon le régime de marché (Bull, Bear, Range, Crash) et afficher la performance de la stratégie par régime dans le DiagnosticPanel. L'utilisateur comprend en 10 secondes dans quelles conditions sa stratégie fonctionne.

## Flux de données

```
WFO loop (walk_forward.py)
  → OOS candles déjà slicées → _classify_regime() → window_regimes[]
  → combo_accumulator + window_idx → regime_analysis (best combo par régime)
  → WFOResult.regime_analysis + WFOResult.window_regimes

scripts/optimize.py
  → windows_serialized[i].regime + regime fields
  → save_report(regime_analysis=...)

report.py → save_result_sync(regime_analysis=...)
optimization_db.py → INSERT regime_analysis JSON dans optimization_results

API: GET /combo-results/{id}
  → fetch optimization_results.regime_analysis
  → response: { combos: [...], regime_analysis: {...} }

Frontend: ExplorerPage → DiagnosticPanel(regimeAnalysis={...})
  → Nouvelle section "PERFORMANCE PAR RÉGIME DE MARCHÉ"
```

---

## 1. Backend — Classification du régime (`walk_forward.py`)

### 1a. Nouvelle fonction `_classify_regime`

**Emplacement :** après `_slice_candles` (~ligne 201), avant la classe `WalkForwardOptimizer`.

```python
def _classify_regime(candles: list[Candle]) -> dict[str, Any]:
    """Classifie le régime de marché d'une période OOS.

    Ordre d'évaluation : Crash (prioritaire) → Bull → Bear → Range.

    Critères :
    - Crash : max drawdown > 30% survenant en < 14 jours
    - Bull : rendement fenêtre > +20%
    - Bear : rendement fenêtre < -20%
    - Range : ni bull ni bear (rendement entre -20% et +20%)
    """
    if len(candles) < 2:
        return {"regime": "range", "return_pct": 0.0, "max_dd_pct": 0.0}

    closes = [c.close for c in candles]
    timestamps = [c.timestamp for c in candles]
    return_pct = (closes[-1] - closes[0]) / closes[0] * 100

    # Max drawdown global
    peak = closes[0]
    max_dd = 0.0
    for close in closes:
        if close > peak:
            peak = close
        dd = (close - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    # Fast crash detection : peak-to-trough > 30% en ≤ 14 jours
    # Algorithme O(n) : sliding window maximum via deque
    from collections import deque
    is_crash = False
    max_seconds = 14 * 86400
    peak_deque = deque()  # Indices des peaks potentiels (décroissant)

    for i in range(len(closes)):
        ts_i = timestamps[i].timestamp()

        # Retirer les éléments hors fenêtre de 14 jours
        while peak_deque and (ts_i - timestamps[peak_deque[0]].timestamp()) > max_seconds:
            peak_deque.popleft()

        # Retirer les éléments plus petits que le courant (ne seront jamais le max)
        while peak_deque and closes[peak_deque[-1]] <= closes[i]:
            peak_deque.pop()

        peak_deque.append(i)

        # Le max dans la fenêtre glissante est closes[peak_deque[0]]
        local_peak = closes[peak_deque[0]]
        if local_peak > 0:
            dd_14d = (closes[i] - local_peak) / local_peak * 100
            if dd_14d < -30:
                is_crash = True
                break

    # Classification
    if is_crash:
        regime = "crash"
    elif return_pct > 20:
        regime = "bull"
    elif return_pct < -20:
        regime = "bear"
    else:
        regime = "range"

    return {
        "regime": regime,
        "return_pct": round(return_pct, 2),
        "max_dd_pct": round(max_dd, 2),
    }
```

**Complexité :** O(n) grâce au deque glissant. Pour 17 280 candles 5m (60 jours OOS), c'est instantané.

### 1b. Ajouter `window_idx` au `combo_accumulator`

**Fichier :** `walk_forward.py`, dans la boucle window (~ligne 619-626).

Ajouter `"window_idx": w_idx` à chaque entrée pour pouvoir lier les résultats par combo aux régimes par fenêtre :

```python
combo_accumulator[params_key].append({
    "is_sharpe": is_r[1],
    "is_return_pct": is_r[2],
    "is_trades": is_r[4],
    "oos_sharpe": oos_r[1] if oos_r else None,
    "oos_return_pct": oos_r[2] if oos_r else None,
    "oos_trades": oos_r[4] if oos_r else None,
    "window_idx": w_idx,  # NOUVEAU
})
```

### 1c. Calculer le régime dans la boucle WFO

**Fichier :** `walk_forward.py`, dans la boucle (~ligne 577, après le slice des candles OOS).

```python
# Avant la boucle (initialisation ~ligne 502) :
window_regimes: list[dict[str, Any]] = []

# Dans la boucle, après le slice des candles OOS (~ligne 577) :
oos_regime = _classify_regime(oos_candles_by_tf.get(main_tf, []))
window_regimes.append(oos_regime)
```

**Note :** `oos_candles_by_tf` est déjà slicé à ce point (lignes 574-576). Pas de fetch supplémentaire.

### 1d. Agréger `regime_analysis` pour le best combo

**Fichier :** `walk_forward.py`, après l'agrégation des combo results (~ligne 741).

```python
# Agrégation regime_analysis (best combo par régime)
regime_analysis: dict[str, dict[str, Any]] | None = None
if collect_combo_results and combo_accumulator and window_regimes:
    recommended_key = json.dumps(recommended, sort_keys=True)
    best_window_data = combo_accumulator.get(recommended_key, [])

    if best_window_data:
        # Grouper les fenêtres du best combo par régime
        regime_groups: dict[str, list[dict]] = {}
        for wd in best_window_data:
            w_idx = wd.get("window_idx", -1)
            if 0 <= w_idx < len(window_regimes):
                regime = window_regimes[w_idx]["regime"]
                regime_groups.setdefault(regime, []).append(wd)

        regime_analysis = {}
        for regime, entries in regime_groups.items():
            oos_sharpes = [e["oos_sharpe"] for e in entries if e["oos_sharpe"] is not None]
            oos_returns = [e["oos_return_pct"] for e in entries if e["oos_return_pct"] is not None]
            n_positive = sum(1 for s in oos_sharpes if s > 0)

            regime_analysis[regime] = {
                "n_windows": len(entries),
                "avg_oos_sharpe": round(float(np.nanmean(oos_sharpes)), 4) if oos_sharpes else 0.0,
                "consistency": round(n_positive / len(oos_sharpes), 4) if oos_sharpes else 0.0,
                "avg_return_pct": round(float(np.mean(oos_returns)), 4) if oos_returns else 0.0,
            }
```

### 1e. Ajouter les champs au `WFOResult`

**Fichier :** `walk_forward.py`, dataclass `WFOResult` (~ligne 63).

```python
@dataclass
class WFOResult:
    # ... champs existants ...
    combo_results: list[dict[str, Any]] = field(default_factory=list)
    window_regimes: list[dict[str, Any]] = field(default_factory=list)  # NOUVEAU
    regime_analysis: dict[str, dict[str, Any]] | None = None           # NOUVEAU
```

Et dans le constructeur final (~ligne 743) :
```python
return WFOResult(
    # ... existant ...
    combo_results=combo_results,
    window_regimes=window_regimes,        # NOUVEAU
    regime_analysis=regime_analysis,      # NOUVEAU
)
```

---

## 2. Backend — Sérialisation (`scripts/optimize.py`)

### 2a. Ajouter le régime aux windows sérialisées

**Fichier :** `scripts/optimize.py`, sérialisation des windows (~lignes 342-360).

Ajouter 3 champs à chaque window :

```python
windows_serialized = [
    {
        # ... champs existants ...
        "oos_trades": w.oos_trades,
        # NOUVEAU : régime de la fenêtre OOS
        "regime": wfo.window_regimes[i]["regime"] if i < len(wfo.window_regimes) else None,
        "regime_return_pct": wfo.window_regimes[i]["return_pct"] if i < len(wfo.window_regimes) else None,
        "regime_max_dd_pct": wfo.window_regimes[i]["max_dd_pct"] if i < len(wfo.window_regimes) else None,
    }
    for i, w in enumerate(wfo.windows)
]
```

### 2b. Passer `regime_analysis` à `save_report`

**Fichier :** `scripts/optimize.py`, appel `save_report()` (~ligne 363).

```python
filepath, result_id = save_report(
    report,
    wfo_windows=windows_serialized,
    duration=None,
    timeframe=main_tf,
    combo_results=wfo.combo_results,
    regime_analysis=wfo.regime_analysis,  # NOUVEAU
)
```

### 2c. Faire la même chose dans `_save_wfo_intermediate`

**Fichier :** `scripts/optimize.py`, `_save_wfo_intermediate` (~lignes 140-159).

Ajouter les mêmes 3 champs par window (pour ne pas perdre les régimes en cas de crash intermédiaire).

---

## 3. Backend — Persistance DB

### 3a. Nouvelle colonne `regime_analysis` dans `optimization_results`

**Fichier :** `backend/core/database.py`

Ajouter une migration idempotente (même pattern que `_migrate_optimization_source`) :

```python
async def _migrate_regime_analysis(self) -> None:
    """Migration idempotente : ajoute regime_analysis à optimization_results."""
    assert self._conn is not None
    cursor = await self._conn.execute("PRAGMA table_info(optimization_results)")
    columns = await cursor.fetchall()
    if not columns:
        return
    col_names = [col["name"] for col in columns]
    if "regime_analysis" in col_names:
        return
    await self._conn.execute(
        "ALTER TABLE optimization_results ADD COLUMN regime_analysis TEXT"
    )
    await self._conn.commit()
    logger.info("Migration optimization_results : colonne regime_analysis ajoutée")
```

Appeler depuis `_create_sprint13_tables` (après `_migrate_optimization_source`).

### 3b. Modifier `save_result_sync` pour stocker `regime_analysis`

**Fichier :** `backend/optimization/optimization_db.py`, `save_result_sync()`.

Ajouter le paramètre `regime_analysis: dict | None = None` et l'insérer dans la query INSERT :

```python
regime_analysis_json = json.dumps(_sanitize_dict(regime_analysis)) if regime_analysis else None
```

Ajouter `regime_analysis` à la liste des colonnes et des valeurs de l'INSERT.

### 3c. Modifier `save_report` pour propager `regime_analysis`

**Fichier :** `backend/optimization/report.py`, `save_report()`.

Ajouter le paramètre `regime_analysis: dict | None = None` et le passer à `save_result_sync()`.

### 3d. Modifier `build_push_payload` et `save_result_from_payload_sync`

**Fichier :** `backend/optimization/optimization_db.py`

- `build_push_payload` : ajouter `regime_analysis` au payload
- `save_result_from_payload_sync` : insérer la colonne `regime_analysis`

Ceci assure que la synchronisation serveur transporte aussi les données de régime.

---

## 4. Backend — API

### 4a. Retourner `regime_analysis` dans `/combo-results/{result_id}`

**Fichier :** `backend/api/optimization_routes.py`, endpoint `get_combo_results` (~ligne 542).

Après avoir récupéré les combos, aussi récupérer le `regime_analysis` du résultat parent :

```python
@router.get("/combo-results/{result_id}")
async def get_combo_results(result_id: int) -> dict:
    db_path = _get_db_path()
    combos = await get_combo_results_async(db_path, result_id)

    # Récupérer regime_analysis depuis le résultat parent
    regime_analysis = None
    result = await get_result_by_id_async(db_path, result_id)
    if result:
        regime_analysis = result.get("regime_analysis")

    if not combos:
        if result is None:
            raise HTTPException(404, f"Résultat {result_id} non trouvé")
        return {
            "result_id": result_id,
            "combos": [],
            "regime_analysis": regime_analysis,
            "message": "...",
        }

    return {
        "result_id": result_id,
        "combos": combos,
        "regime_analysis": regime_analysis,
    }
```

### 4b. Parser `regime_analysis` dans `get_result_by_id_async`

**Fichier :** `backend/optimization/optimization_db.py`, `get_result_by_id_async()`.

Ajouter le parsing JSON du champ `regime_analysis` (après les autres JSON blobs) :

```python
if result.get("regime_analysis"):
    result["regime_analysis"] = json.loads(result["regime_analysis"])
```

---

## 5. Frontend — `DiagnosticPanel.jsx`

### 5a. Nouvelle prop `regimeAnalysis`

```jsx
export default function DiagnosticPanel({ combos, grade, totalScore, nWindows, regimeAnalysis })
```

### 5b. Nouvelle section "Performance par régime"

Après les verdicts existants, ajouter un bloc conditionnel :

```jsx
{regimeAnalysis && Object.keys(regimeAnalysis).length > 0 && (
  <div className="regime-section">
    <h4 className="regime-title">
      <RegimeIcon />
      PERFORMANCE PAR RÉGIME
    </h4>
    <div className="regime-grid">
      {regimeOrder
        .filter(r => regimeAnalysis[r.key])
        .map(r => {
          const data = regimeAnalysis[r.key]
          const level = data.avg_oos_sharpe > 1 ? 'green'
            : data.avg_oos_sharpe > 0 ? 'orange' : 'red'
          return (
            <div key={r.key} className="regime-item">
              <div className="regime-header">
                <RegimeLabel regime={r.key} />
                <span className="regime-windows">({data.n_windows} fen.)</span>
              </div>
              <div className="regime-metrics">
                <span>Sharpe : {data.avg_oos_sharpe.toFixed(2)}</span>
                <span>Consist. : {Math.round(data.consistency * 100)}%</span>
                <StatusCircle level={level} />
              </div>
            </div>
          )
        })}
    </div>
    <div className="regime-conclusion">
      → {generateConclusion(regimeAnalysis)}
    </div>
  </div>
)}
```

### 5c. Icônes régime (SVG inline, pas d'emoji)

```jsx
const regimeOrder = [
  { key: 'bull', label: 'Bull Market', color: '#10b981' },
  { key: 'bear', label: 'Bear Market', color: '#ef4444' },
  { key: 'range', label: 'Range', color: '#6b7280' },
  { key: 'crash', label: 'Crash', color: '#f59e0b' },
]

function RegimeLabel({ regime }) {
  const info = regimeOrder.find(r => r.key === regime) || regimeOrder[2]
  return (
    <span style={{ color: info.color, fontWeight: 600, fontSize: '13px' }}>
      <svg width="12" height="12" viewBox="0 0 12 12" style={{ marginRight: 4, verticalAlign: 'middle' }}>
        {regime === 'bull' && <path d="M2 10 L6 2 L10 10" fill="none" stroke={info.color} strokeWidth="2"/>}
        {regime === 'bear' && <path d="M2 2 L6 10 L10 2" fill="none" stroke={info.color} strokeWidth="2"/>}
        {regime === 'range' && <line x1="1" y1="6" x2="11" y2="6" stroke={info.color} strokeWidth="2"/>}
        {regime === 'crash' && <path d="M2 2 L5 5 L3 5 L10 10" fill="none" stroke={info.color} strokeWidth="2"/>}
      </svg>
      {info.label}
    </span>
  )
}
```

### 5d. Conclusion automatique

```jsx
function generateConclusion(regimeAnalysis) {
  const bull = regimeAnalysis.bull?.avg_oos_sharpe ?? 0
  const bear = regimeAnalysis.bear?.avg_oos_sharpe ?? 0
  const range_ = regimeAnalysis.range?.avg_oos_sharpe ?? 0
  const crash = regimeAnalysis.crash?.avg_oos_sharpe ?? 0

  const bearCrashAvg = ((bear || 0) + (crash || 0)) / 2
  const allSimilar = Math.abs(bull - bear) < 1 && Math.abs(bear - range_) < 1

  if (allSimilar && bull > 0 && bear > 0 && range_ > 0)
    return "Stratégie all-weather, performance stable tous régimes."
  if (bearCrashAvg > bull * 2 && bearCrashAvg > 1)
    return "Stratégie de mean-reversion, performante en bear/crash."
  if (bull > bearCrashAvg * 2 && bull > 1)
    return "Stratégie momentum/trend-following, performante en bull."
  if (Math.max(bull, bear, range_, crash) <= 0)
    return "Aucun régime favorable identifié."

  // Identifier le meilleur régime
  const best = Object.entries(regimeAnalysis)
    .sort((a, b) => (b[1]?.avg_oos_sharpe ?? 0) - (a[1]?.avg_oos_sharpe ?? 0))
  if (best.length > 0 && (best[0][1]?.avg_oos_sharpe ?? 0) > 0) {
    const label = regimeOrder.find(r => r.key === best[0][0])?.label || best[0][0]
    return `Meilleure performance en ${label}.`
  }
  return "Profil de régime indéterminé."
}
```

### 5e. Styles CSS (dans `DiagnosticPanel.css`)

```css
.regime-section {
  margin-top: 16px;
  padding-top: 14px;
  border-top: 1px solid #333;
}

.regime-title {
  margin: 0 0 10px 0;
  color: #fff;
  font-size: 13px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.regime-grid {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.regime-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 8px;
  background: rgba(255,255,255,0.03);
  border-radius: 4px;
}

.regime-header {
  display: flex;
  align-items: center;
  gap: 6px;
}

.regime-windows {
  color: #6b7280;
  font-size: 11px;
}

.regime-metrics {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 12px;
  color: #9ca3af;
}

.regime-conclusion {
  margin-top: 10px;
  font-size: 12px;
  color: #d1d5db;
  font-style: italic;
}
```

---

## 6. Frontend — `ExplorerPage.jsx`

### 6a. Passer `regimeAnalysis` au `DiagnosticPanel`

**Modification minimale (~3 lignes).**

```jsx
<DiagnosticPanel
  combos={comboResults.combos}
  grade={selectedRun?.grade || '?'}
  totalScore={selectedRun?.total_score || 0}
  nWindows={Math.max(...comboResults.combos.map((c) => c.n_windows_evaluated || 0))}
  regimeAnalysis={comboResults.regime_analysis || null}
/>
```

Le `regime_analysis` est déjà inclus dans la réponse `/combo-results/{id}` (étape 4a).

---

## 7. Tests

### 7a. Test `_classify_regime` (dans tests/)

```python
def test_classify_regime_bull():
    """Rendement > 20% → bull."""
    candles = make_candles(start=100, end=125, n=100)  # +25%
    result = _classify_regime(candles)
    assert result["regime"] == "bull"

def test_classify_regime_bear():
    """Rendement < -20% → bear."""
    candles = make_candles(start=100, end=75, n=100)  # -25%
    result = _classify_regime(candles)
    assert result["regime"] == "bear"

def test_classify_regime_range():
    """Rendement entre -20% et +20% → range."""
    candles = make_candles(start=100, end=105, n=100)  # +5%
    result = _classify_regime(candles)
    assert result["regime"] == "range"

def test_classify_regime_crash():
    """Drawdown > 30% en < 14 jours → crash (prioritaire)."""
    # Prix: 100 → 65 en 10 jours → recovery à 110
    candles = make_crash_candles(peak=100, trough=65, days_to_trough=10, recovery=110)
    result = _classify_regime(candles)
    assert result["regime"] == "crash"

def test_classify_regime_slow_decline_not_crash():
    """Drawdown > 30% mais > 14 jours → bear, pas crash."""
    candles = make_candles(start=100, end=65, n=500)
    result = _classify_regime(candles)
    assert result["regime"] == "bear"

def test_classify_regime_empty():
    """< 2 candles → range par défaut."""
    result = _classify_regime([])
    assert result["regime"] == "range"
```

### 7b. Test migration DB

```python
async def test_regime_analysis_column_exists():
    """La colonne regime_analysis est créée par la migration."""
    db = Database(":memory:")
    await db.init()
    async with aiosqlite.connect(db.db_path) as conn:
        cursor = await conn.execute("PRAGMA table_info(optimization_results)")
        cols = [row[1] for row in await cursor.fetchall()]
        assert "regime_analysis" in cols
    await db.close()
```

### 7c. Test API combo-results inclut regime_analysis

```python
async def test_combo_results_includes_regime_analysis():
    """L'endpoint combo-results retourne regime_analysis."""
    # Insert un résultat avec regime_analysis en DB
    # GET /api/optimization/combo-results/{id}
    # Vérifier que la réponse contient regime_analysis
```

---

## 8. Fichiers modifiés — Résumé

| Fichier | Action | Lignes estimées |
|---------|--------|-----------------|
| `backend/optimization/walk_forward.py` | MODIFIER — `_classify_regime()`, `window_regimes`, `regime_analysis`, champs `WFOResult` | ~65 |
| `scripts/optimize.py` | MODIFIER — sérialisation regime, passage à `save_report` | ~12 |
| `backend/optimization/report.py` | MODIFIER — paramètre `regime_analysis` dans `save_report` | ~5 |
| `backend/optimization/optimization_db.py` | MODIFIER — `save_result_sync`, `build_push_payload`, `save_result_from_payload_sync`, `get_result_by_id_async` | ~25 |
| `backend/core/database.py` | MODIFIER — migration `_migrate_regime_analysis` | ~15 |
| `backend/api/optimization_routes.py` | MODIFIER — `/combo-results/{id}` retourne regime_analysis | ~10 |
| `frontend/src/components/DiagnosticPanel.jsx` | MODIFIER — section régime + conclusion + helpers | ~90 |
| `frontend/src/components/DiagnosticPanel.css` | MODIFIER — styles section régime | ~40 |
| `frontend/src/components/ExplorerPage.jsx` | MODIFIER — passer prop regimeAnalysis | ~3 |
| Tests | CRÉER/MODIFIER | ~80 |
| **Total** | | **~345 lignes, 10 fichiers** |

---

## 9. Rétrocompatibilité

- **Anciens runs** : colonne DB `NULL`, API retourne `null`, frontend n'affiche pas la section.
- **Sync serveur** : payload inclut `regime_analysis` si présent. Ancien serveur ignore le champ.
- **Migration DB** : idempotente via `ALTER TABLE ADD COLUMN` (pattern existant).

## 10. Points d'attention

1. **Performance :** `_classify_regime` est O(n) et appelé 1× par fenêtre. Négligeable.
2. **Seuils** (20% bull, -20% bear, 30% crash) : calibrés crypto. Hardcodés pour le MVP.
3. **Best combo = recommended_key** : le combo `is_best` (médiane) est utilisé pour l'analyse par régime.
4. **Fenêtres sans match** : si `window_idx` ne correspond à aucun régime, ignoré silencieusement.
