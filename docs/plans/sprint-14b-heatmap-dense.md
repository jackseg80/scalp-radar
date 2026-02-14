# Sprint 14b — Heatmap dense + Charts analytiques + Tooltips d'aide

## Contexte

Le Sprint 14 a ajouté l'Explorateur de Paramètres avec heatmap WFO. Problème : la heatmap est **sparse** (une cellule par run historique, basée sur `best_params`), donc quasi vide (2-3 cellules sur 324 possibles). Ce sprint rend la heatmap **dense** en sauvegardant les résultats de CHAQUE combo testée par le WFO, ajoute 3 charts analytiques, et des tooltips d'aide sur les termes techniques.

---

## Analyse du code existant

### Flux WFO actuel (`walk_forward.py:496-635`)

Pour chaque fenêtre :
1. **IS** : `_parallel_backtest(coarse_grid)` → `coarse_results: list[_ISResult]` (200 combos)
2. **IS fine** : `_parallel_backtest(fine_grid)` → résultats fins (top 20 ± 1 step)
3. `all_is_results = coarse + fine` → trié par sharpe, seul `best_is[0]` est gardé
4. **OOS** : `run_backtest_single(best_params)` → UNE SEULE combo évaluée en OOS

**Problème** : les résultats IS de toutes les combos sont calculés puis jetés. L'OOS n'est fait que pour le best.

### Solution : OOS batch pour toutes les combos

Le fast engine (`_run_fast`) tourne à ~0.1ms/backtest. Pour 324 combos × 12 fenêtres :
- IS batch : déjà fait (coarse 200 + fine ~80 = ~280 backtests/fenêtre)
- **OOS batch (nouveau)** : ~324 backtests/fenêtre × 0.1ms = ~30ms/fenêtre
- Coût total OOS additionnel : ~360ms pour 12 fenêtres → **négligeable**

Pour les stratégies sans fast engine (funding, liquidation) : OOS batch via `_run_sequential` ~1s/bt → trop cher (324 × 12 = ~65 min). **Skip les combo results pour ces stratégies** (colonne `combo_results` vide, message frontend).

### Type de retour `_ISResult`

```python
_ISResult = tuple[dict[str, Any], float, float, float, int]
#                  params          sharpe  return%  PF     trades
```

Positions : `[0]=params, [1]=sharpe, [2]=net_return_pct, [3]=profit_factor, [4]=n_trades`

Disponible pour IS ET OOS (via `_parallel_backtest`). **Pas de win_rate** → colonne `oos_win_rate` nullable dans la table.

### Gestion du 2-pass (coarse + fine)

- `coarse_grid` : même set pour toutes les fenêtres (200 combos LHS ou grid complet si ≤ 200)
- `fine_grid` : varie par fenêtre (dépend du top 20 IS de cette fenêtre)
- **Certaines combos n'existent que dans certaines fenêtres** (fine grid variable)
- Agrégation : moyenne/somme sur les fenêtres où la combo a été évaluée
- Champ `n_windows_evaluated` pour distinguer combos partielles

---

## Plan d'implémentation

### Bloc A — Table `wfo_combo_results` (DB)

**Fichier** : `backend/core/database.py`

Ajouter dans `_create_tables()` un appel à `_create_sprint14b_tables()` :

```sql
CREATE TABLE IF NOT EXISTS wfo_combo_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    optimization_result_id INTEGER NOT NULL,
    params TEXT NOT NULL,
    oos_sharpe REAL,
    oos_return_pct REAL,
    oos_trades INTEGER,
    oos_win_rate REAL,        -- nullable, non peuplé pour l'instant
    is_sharpe REAL,
    is_return_pct REAL,
    is_trades INTEGER,
    consistency REAL,
    oos_is_ratio REAL,
    is_best INTEGER DEFAULT 0,
    n_windows_evaluated INTEGER,
    FOREIGN KEY (optimization_result_id) REFERENCES optimization_results(id)
);

CREATE INDEX IF NOT EXISTS idx_combo_opt_id ON wfo_combo_results(optimization_result_id);
CREATE INDEX IF NOT EXISTS idx_combo_best ON wfo_combo_results(is_best) WHERE is_best = 1;
```

**Volume** : ~324 rows par run × 5 assets = ~1600 rows par run complet. Très petit.

---

### Bloc B — Collecte des combo results dans le WFO

**Fichier** : `backend/optimization/walk_forward.py`

#### B1. Ajouter `combo_results` à `WFOResult`

```python
@dataclass
class WFOResult:
    # ... champs existants ...
    combo_results: list[dict[str, Any]] = field(default_factory=list)  # NEW
```

#### B2. Modifier `optimize()` — collecte per-combo per-window

Dans la boucle des fenêtres (ligne 496), après les passes IS :

1. Construire `all_tested_params` : set unique de params depuis `all_is_results`
2. **Appeler `_parallel_backtest()` sur les candles OOS** avec `all_tested_params` → `all_oos_results`
3. Stocker `(is_results, oos_results)` dans un accumulateur per-combo

**Détail du code** (pseudo-code des changements dans la boucle fenêtre) :

```python
# Accumulateur cross-fenêtre : {params_key: [per_window_data]}
combo_accumulator: dict[str, list[dict]] = {}

for w_idx, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
    # ... IS passes existantes (coarse + fine) → all_is_results ...
    # ... OOS pour le best (existant) ...

    # NOUVEAU : OOS batch pour toutes les combos
    all_tested_params_list = [r[0] for r in all_is_results]
    # Dédupliquer
    seen = set()
    unique_params = []
    for p in all_tested_params_list:
        key = json.dumps(p, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique_params.append(p)

    # OOS batch (utilise fast engine si disponible)
    oos_batch_results = self._parallel_backtest(
        unique_params, oos_candles_by_tf, strategy_name, symbol,
        bt_config_dict, main_tf, n_workers, metric,
        extra_data_map=oos_extra_data_map,
    )

    # Index les résultats IS et OOS par params_key
    is_by_key = {json.dumps(r[0], sort_keys=True): r for r in all_is_results}
    oos_by_key = {json.dumps(r[0], sort_keys=True): r for r in oos_batch_results}

    for params_key in is_by_key:
        is_r = is_by_key[params_key]
        oos_r = oos_by_key.get(params_key)

        if params_key not in combo_accumulator:
            combo_accumulator[params_key] = []

        combo_accumulator[params_key].append({
            "is_sharpe": is_r[1],
            "is_return_pct": is_r[2],
            "is_trades": is_r[4],
            "oos_sharpe": oos_r[1] if oos_r else None,
            "oos_return_pct": oos_r[2] if oos_r else None,
            "oos_trades": oos_r[4] if oos_r else None,
        })
```

#### B3. Agrégation cross-fenêtre (après la boucle)

```python
# Après la boucle des fenêtres
combo_results = []
recommended_key = json.dumps(recommended, sort_keys=True)

for params_key, window_data in combo_accumulator.items():
    params = json.loads(params_key)

    is_sharpes = [d["is_sharpe"] for d in window_data]
    oos_sharpes = [d["oos_sharpe"] for d in window_data if d["oos_sharpe"] is not None]
    oos_returns = [d["oos_return_pct"] for d in window_data if d["oos_return_pct"] is not None]
    oos_trades_list = [d["oos_trades"] for d in window_data if d["oos_trades"] is not None]

    avg_is_sharpe = float(np.nanmean(is_sharpes)) if is_sharpes else 0.0
    avg_oos_sharpe = float(np.nanmean(oos_sharpes)) if oos_sharpes else 0.0
    total_oos_return = sum(oos_returns) if oos_returns else 0.0
    total_oos_trades = sum(oos_trades_list) if oos_trades_list else 0
    n_oos_positive = sum(1 for s in oos_sharpes if s > 0)
    consistency = n_oos_positive / len(oos_sharpes) if oos_sharpes else 0.0
    oos_is_ratio = avg_oos_sharpe / avg_is_sharpe if avg_is_sharpe > 0 else 0.0

    combo_results.append({
        "params": params,
        "is_sharpe": round(avg_is_sharpe, 4),
        "is_return_pct": round(sum(d["is_return_pct"] for d in window_data), 4),
        "is_trades": sum(d["is_trades"] for d in window_data),
        "oos_sharpe": round(avg_oos_sharpe, 4),
        "oos_return_pct": round(total_oos_return, 4),
        "oos_trades": total_oos_trades,
        "oos_win_rate": None,  # Non disponible via fast engine
        "consistency": round(consistency, 4),
        "oos_is_ratio": round(oos_is_ratio, 4),
        "is_best": params_key == recommended_key,
        "n_windows_evaluated": len(window_data),
    })

# Ajouter au WFOResult
return WFOResult(
    ...,
    combo_results=combo_results,
)
```

#### B4. Guard pour stratégies sans fast engine

Si `strategy_name not in ("vwap_rsi", "momentum", "bollinger_mr", "donchian_breakout", "supertrend", "envelope_dca")`, skip l'OOS batch et retourner `combo_results=[]`.

**Risques Bloc B** :
- Le 2-pass (coarse+fine) crée des combos partielles → mitigé par agrégation sur fenêtres évaluées
- Mémoire : `combo_accumulator` stocke ~324 combos × 12 fenêtres × 7 floats = ~30KB → négligeable
- L'OOS batch réutilise `_parallel_backtest` qui gère déjà le fallback séquentiel

---

### Bloc C — Persistence des combo results

**Fichier** : `backend/optimization/optimization_db.py`

#### C1. `save_combo_results_sync(db_path, result_id, combo_results)`

```python
def save_combo_results_sync(db_path: str, result_id: int, combo_results: list[dict]) -> int:
    """Insère les combo results en DB (sync). Retourne le nombre inséré."""
    conn = sqlite3.connect(db_path)
    try:
        data = [
            (
                result_id,
                json.dumps(cr["params"], sort_keys=True),
                cr.get("oos_sharpe"),
                cr.get("oos_return_pct"),
                cr.get("oos_trades"),
                cr.get("oos_win_rate"),
                cr.get("is_sharpe"),
                cr.get("is_return_pct"),
                cr.get("is_trades"),
                cr.get("consistency"),
                cr.get("oos_is_ratio"),
                1 if cr.get("is_best") else 0,
            )
            for cr in combo_results
        ]
        conn.executemany(
            """INSERT INTO wfo_combo_results
               (optimization_result_id, params, oos_sharpe, oos_return_pct, oos_trades,
                oos_win_rate, is_sharpe, is_return_pct, is_trades, consistency, oos_is_ratio, is_best)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            data,
        )
        conn.commit()
        return len(data)
    finally:
        conn.close()
```

#### C2. `get_combo_results_async(db_path, result_id)`

Fonction async pour l'API :

```python
async def get_combo_results_async(db_path: str, result_id: int) -> list[dict]:
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute(
            """SELECT params, oos_sharpe, oos_return_pct, oos_trades, oos_win_rate,
                      is_sharpe, is_return_pct, is_trades, consistency, oos_is_ratio, is_best
               FROM wfo_combo_results
               WHERE optimization_result_id = ?
               ORDER BY oos_sharpe DESC""",
            (result_id,),
        )
        rows = await cursor.fetchall()
        return [
            {**dict(row), "params": json.loads(row["params"])}
            for row in rows
        ]
```

#### C3. Modifier `save_report()` dans `report.py`

Ajouter paramètre `combo_results` :

```python
def save_report(
    report, wfo_windows=None, duration=None, timeframe=None,
    output_dir="data/optimization", db_path=None,
    combo_results=None,  # NEW
) -> tuple[Path, int | None]:
    # ... existant ...
    result_id = save_result_sync(db_path, report, wfo_windows, duration, timeframe)

    # NEW : sauver les combo results
    if result_id and combo_results:
        from backend.optimization.optimization_db import save_combo_results_sync
        n_saved = save_combo_results_sync(db_path, result_id, combo_results)
        logger.info("Combo results sauvés : {} combos pour result_id={}", n_saved, result_id)

    # Push serveur (inclut combo_results dans le payload)
    if timeframe is not None:
        push_to_server(report, wfo_windows, duration, timeframe, combo_results=combo_results)

    return filepath, result_id
```

#### C4. Modifier `build_push_payload()` + `push_to_server()`

Ajouter `combo_results` au payload :

```python
def build_push_payload(report, wfo_windows, duration, timeframe, source="local", combo_results=None):
    payload = { ... }  # existant
    if combo_results:
        payload["combo_results"] = combo_results
    return payload
```

#### C5. Modifier `save_result_from_payload_sync()` (réception serveur)

Après insertion du résultat principal, insérer les combo_results si présents :

```python
# Dans save_result_from_payload_sync, après le commit principal :
combo_results = payload.get("combo_results")
if combo_results and new_id:
    save_combo_results_sync(db_path, new_id, combo_results)
```

---

### Bloc D — API endpoints

**Fichier** : `backend/api/optimization_routes.py`

#### D1. Nouveau endpoint `GET /api/optimization/combo-results/{result_id}`

```python
@router.get("/combo-results/{result_id}")
async def get_combo_results(result_id: int) -> dict:
    db_path = _get_db_path()
    combos = await get_combo_results_async(db_path, result_id)
    if not combos:
        # Vérifier si le result_id existe
        result = await get_result_by_id_async(db_path, result_id)
        if result is None:
            raise HTTPException(404, f"Résultat {result_id} non trouvé")
        # Existe mais pas de combos → ancien run
        return {"result_id": result_id, "combos": [], "message": "Données détaillées non disponibles pour ce run (lancez un nouveau WFO)"}
    return {"result_id": result_id, "combos": combos}
```

#### D2. Modifier `GET /api/optimization/heatmap`

Ajouter paramètre optionnel `result_id`. Si fourni, lire `wfo_combo_results`. Sinon, chercher le latest result_id pour (strategy, asset) et lire ses combos. Fallback sur l'ancien comportement (sparse) si pas de combo results.

```python
@router.get("/heatmap")
async def get_optimization_heatmap(
    strategy: str = Query(...),
    asset: str = Query(...),
    param_x: str = Query(...),
    param_y: str = Query(...),
    metric: str = Query(default="oos_sharpe"),  # Change default de total_score à oos_sharpe
    result_id: int | None = Query(default=None),
) -> dict:
```

Logique :
1. Si `result_id` absent → trouver le latest pour (strategy, asset, is_latest=1)
2. Lire `wfo_combo_results` pour ce result_id
3. Construire la matrice (param_x × param_y) → valeur = metric de chaque combo
4. Si pas de combo results → fallback ancien comportement (sparse, optimization_results)

#### D3. Endpoint listing : ajouter `id` dans les résultats pour le run selector

Déjà présent dans `get_optimization_results()` (retourne `id`). Vérifier que le frontend le reçoit.

---

### Bloc E — Frontend : Run Selector + Heatmap dense

**Fichier** : `frontend/src/components/ExplorerPage.jsx`

#### E1. Run Selector

Nouveau state : `selectedRunId`. Nouveau fetch : quand strategy + asset changent, charger les runs disponibles via `GET /api/optimization/results?strategy=...&asset=...&latest_only=false`.

Dropdown avec : "Run du 13/02 — Grade B (score 72) [latest]" etc.

#### E2. Modifier le fetch heatmap

Passer `result_id` dans la query. Ajouter les nouvelles métriques au sélecteur :

```javascript
const metrics = [
    { value: 'oos_sharpe', label: 'OOS Sharpe' },
    { value: 'oos_return_pct', label: 'OOS Return %' },
    { value: 'consistency', label: 'Consistance' },
    { value: 'oos_is_ratio', label: 'Ratio OOS/IS' },
    { value: 'is_sharpe', label: 'IS Sharpe' },
]
```

#### E3. Adapter HeatmapChart.jsx

La structure de données ne change pas (même format `cells[y][x]`). Mais les cellules n'ont plus de `grade` individuel — on utilise l'échelle numérique couleur (rouge→jaune→vert). Supprimer le rendu du grade par cellule, garder uniquement la valeur numérique + tooltip avec les params.

---

### Bloc F — Frontend : Charts analytiques

#### F1. `Top10Table.jsx` (NOUVEAU)

**Fichier** : `frontend/src/components/Top10Table.jsx`

Tableau HTML simple :
- Données : `combo_results` trié par métrique sélectionnée, top 10
- Colonnes : Rang, [params...], OOS Sharpe, IS Sharpe, OOS/IS, Consistance, Trades
- La combo `is_best=true` est surlignée (background différent)
- Clic sur une ligne → optionnel (future : highlight heatmap)

Props : `{ combos, paramNames, metric }`

#### F2. `ScatterChart.jsx` (NOUVEAU)

**Fichier** : `frontend/src/components/ScatterChart.jsx`

SVG pur (pattern WfoChart.jsx) :
- Axe X : IS Sharpe, Axe Y : OOS Sharpe
- Un point par combo (cx, cy calculés avec scaleX/scaleY)
- Diagonale pointillée IS = OOS
- Point best : couleur + taille différente (star ou cercle plus gros)
- Couleur des points par consistance : rouge < 0.5, orange 0.5-0.8, vert > 0.8
- Hover (title SVG) : params de la combo

SVG responsive via `viewBox="0 0 600 400"` + `width="100%"` + `preserveAspectRatio="xMidYMid meet"`.

#### F3. `DistributionChart.jsx` (NOUVEAU)

**Fichier** : `frontend/src/components/DistributionChart.jsx`

SVG pur, histogramme :
- Input : `oos_sharpe` de toutes les combos
- Bins automatiques : `Math.ceil(Math.sqrt(n_combos))` bins, bornes arrondi
- Barres : rouge si bin < 0, vert si bin ≥ 0
- Ligne verticale à OOS Sharpe = 0
- Marqueur pour le OOS Sharpe du best combo (flèche ou triangle en haut)
- Axes : X = OOS Sharpe, Y = nombre de combos

SVG responsive via `viewBox="0 0 600 300"` + `width="100%"` + `preserveAspectRatio="xMidYMid meet"`.

#### F4. Intégration dans ExplorerPage.jsx

Sous la heatmap, ajouter une section "Analyse" avec les 3 charts :
- Fetch `GET /api/optimization/combo-results/{result_id}` quand `selectedRunId` change
- Si pas de combos → message "Données détaillées non disponibles"
- Layout : Top10 (pleine largeur) + Scatter (gauche) + Distribution (droite) en grid 2 colonnes

---

### Bloc G — Tooltips d'aide

#### G1. `InfoTooltip.jsx` (NOUVEAU)

**Fichier** : `frontend/src/components/InfoTooltip.jsx`

Composant réutilisable :
```jsx
<InfoTooltip term="oos_sharpe" />
```

Comportement :
- Icone `(i)` en SVG inline (petit cercle + lettre i), couleur #666
- Clic → toggle popover
- Popover : position auto (au-dessus par défaut, en dessous si pas d'espace)
- Fermeture : clic ailleurs (useEffect + document.addEventListener)
- Style : background #1a1a1a, border #444, border-radius 8px, padding 12px, max-width 320px

Contenu du popover :
```jsx
<div className="tooltip-title">{GLOSSARY[term].title}</div>
<div className="tooltip-desc">{GLOSSARY[term].description}</div>
<div className="tooltip-interp">{GLOSSARY[term].interpretation}</div>
```

#### G2. Glossaire `GLOSSARY`

Défini dans InfoTooltip.jsx (objet JS statique, ~14 termes) :
- oos_sharpe, is_sharpe, oos_is_ratio, consistency, dsr
- monte_carlo_pvalue, param_stability, grade, total_score
- ci_sharpe, transfer_ratio, wfo, is_vs_oos_chart

Contenu exactement comme spécifié par l'utilisateur dans le prompt.

#### G3. Intégration ExplorerPage

- Header heatmap : `Heatmap ... <InfoTooltip term={metric} />`
- Top 10 : headers colonnes avec tooltip
- Scatter : labels axes avec tooltip
- Distribution : titre avec tooltip

#### G4. Intégration ResearchPage

- Tableau : headers "OOS Sharpe", "Consistance", "OOS/IS", "DSR", "Stabilité" avec tooltip
- Vue détail : titres sections "Critères de notation", "Monte Carlo", "Validation Bitget"
- WfoChart : titre "Equity Curve IS vs OOS" avec tooltip `is_vs_oos_chart`

---

### Bloc H — Propagation dans optimize.py et JobManager

**Fichier** : `scripts/optimize.py`

#### H1. Passer combo_results à save_report()

Dans `run_optimization()`, après le WFO :

```python
wfo = await optimizer.optimize(...)
# ...
filepath, result_id = save_report(
    report,
    wfo_windows=windows_serialized,
    duration=None,
    timeframe=main_tf,
    combo_results=wfo.combo_results,  # NEW
)
```

**Fichier** : `backend/optimization/job_manager.py`

Le JobManager appelle déjà `run_optimization()` qui appelle `save_report()`. Pas de modification nécessaire — le flux passe automatiquement par les changements de `optimize.py`.

---

## Ordre d'implémentation

| # | Bloc | Dépendances | Estimation |
|---|------|-------------|------------|
| 1 | A — Table DB | Aucune | 20 min |
| 2 | B — WFO combo collect | A | 45 min |
| 3 | C — Persistence | A, B | 30 min |
| 4 | H — optimize.py propagation | B, C | 10 min |
| 5 | D — API endpoints | C | 25 min |
| 6 | E — Run Selector + Heatmap | D | 30 min |
| 7 | F — Charts analytiques | D, E | 45 min |
| 8 | G — Tooltips | Aucune (indépendant) | 30 min |
| 9 | Tests | Tout | 40 min |

**Total estimé : ~4h30**

G peut être fait en parallèle de B-F car indépendant.

---

## Fichiers à modifier

| Fichier | Action | Bloc |
|---------|--------|------|
| `backend/core/database.py` | Ajouter `_create_sprint14b_tables()` | A |
| `backend/optimization/walk_forward.py` | Ajouter champ `combo_results` à WFOResult, modifier `optimize()` | B |
| `backend/optimization/optimization_db.py` | Ajouter `save_combo_results_sync()`, `get_combo_results_async()` | C |
| `backend/optimization/report.py` | Modifier `save_report()` signature + appel combo save | C |
| `scripts/optimize.py` | Passer `combo_results` à `save_report()` | H |
| `backend/api/optimization_routes.py` | Nouveau endpoint combo-results, modifier heatmap | D |
| `frontend/src/components/ExplorerPage.jsx` | Run selector, section analyse, fetch combo data | E, F |
| `frontend/src/components/HeatmapChart.jsx` | Adapter au mode dense (supprimer grade par cellule) | E |

## Fichiers à créer

| Fichier | Bloc |
|---------|------|
| `frontend/src/components/InfoTooltip.jsx` | G |
| `frontend/src/components/Top10Table.jsx` | F |
| `frontend/src/components/ScatterChart.jsx` | F |
| `frontend/src/components/DistributionChart.jsx` | F |

---

## Tests à ajouter

**Fichier** : `tests/test_combo_results.py` (NOUVEAU)

1. **test_wfo_combo_results_populated** : WFO avec mini-grid (2×2=4 combos) retourne `combo_results` non vide
2. **test_combo_results_aggregation** : vérifier que la moyenne/somme cross-fenêtre est correcte
3. **test_combo_best_flag** : vérifier qu'exactement une combo a `is_best=True`
4. **test_save_combo_results_sync** : insertion en DB + relecture
5. **test_combo_results_empty_for_old_runs** : endpoint retourne `combos: []` pour un run sans combo data
6. **test_heatmap_dense** : endpoint heatmap retourne matrice dense avec N×M cellules pour un result_id
7. **test_push_payload_includes_combos** : `build_push_payload()` inclut le champ combo_results
8. **test_save_from_payload_with_combos** : `save_result_from_payload_sync()` insère aussi les combos

---

## Risques identifiés

| Risque | Impact | Mitigation |
|--------|--------|------------|
| OOS batch trop lent pour stratégies sans fast engine | Temps WFO × N_combos | Skip combo_results si pas de fast engine |
| Fine grid crée des combos avec données partielles | Combos avec peu de fenêtres évaluées → stats peu fiables | Afficher `n_windows_evaluated` dans le tooltip |
| Taille du payload sync avec 324 combos | ~100KB de JSON par push | Acceptable, compression HTTP |
| collision de params_key (JSON sort order) | Combos dupliquées dans l'accumulateur | Utiliser `json.dumps(p, sort_keys=True)` comme clé |
| Migration DB : table déjà existante | `CREATE TABLE IF NOT EXISTS` → idempotent | Déjà le pattern utilisé partout |

---

## Vérification

1. **Backend** : `uv run pytest tests/test_combo_results.py -v`
2. **Régression** : `uv run pytest tests/ -x` (tous les 597 tests doivent passer)
3. **E2E manuelle** :
   - Lancer un WFO via CLI : `uv run python -m scripts.optimize --strategy envelope_dca --symbol BTC/USDT`
   - Vérifier en DB : `SELECT COUNT(*) FROM wfo_combo_results WHERE optimization_result_id = <id>`
   - Lancer via Explorer (frontend) : soumettre un job, attendre completion
   - Vérifier que la heatmap est dense (toutes les cellules remplies)
   - Vérifier les 3 charts (Top 10, Scatter, Distribution)
   - Vérifier les tooltips sur Explorer ET Recherche
4. **Dev** : `dev.bat` (lance backend + frontend en parallèle)
