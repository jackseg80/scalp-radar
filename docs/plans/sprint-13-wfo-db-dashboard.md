# Sprint 13 — Résultats WFO en DB + Dashboard Recherche

## Contexte

Les résultats d'optimisation WFO sont stockés en JSON locaux (`data/optimization/`). 49 final reports + 23 intermediates existent. Le FinalReport JSON ne contient **pas** les détails par fenêtre WFO (IS/OOS Sharpe par window) — ces données sont uniquement dans les fichiers `wfo_*_intermediate.json`, qui sont écrasés à chaque run.

**Problème** : pas de visualisation centralisée, pas d'accès depuis le dashboard serveur, pas d'historique des runs.

**Objectif** : Persister en DB + créer une page "Recherche" pour visualiser grades, scores, equity curve IS vs OOS.

---

## 1. Schéma DB

```sql
CREATE TABLE IF NOT EXISTS optimization_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    asset TEXT NOT NULL,          -- ex: "BTC/USDT"
    timeframe TEXT NOT NULL,      -- ex: "1h", "5m"

    -- Métadonnées
    created_at TEXT NOT NULL,     -- ISO 8601 (ex: "2026-02-13T12:35:32")
    duration_seconds REAL,       -- durée du run (nullable, pas dispo dans les JSON legacy)

    -- Grading
    grade TEXT NOT NULL,          -- A/B/C/D/F
    total_score REAL NOT NULL,   -- score numérique 0-100 (avant conversion en lettre)
    oos_sharpe REAL,
    consistency REAL,             -- % fenêtres OOS > 0
    oos_is_ratio REAL,
    dsr REAL,
    param_stability REAL,
    monte_carlo_pvalue REAL,
    mc_underpowered INTEGER DEFAULT 0,  -- boolean
    n_windows INTEGER NOT NULL,
    n_distinct_combos INTEGER,

    -- Données détaillées (JSON blobs)
    best_params TEXT NOT NULL,     -- JSON dict des params retenus
    wfo_windows TEXT,              -- JSON array [{window_index, is_start, is_end, oos_start, oos_end, best_params, is_sharpe, is_net_return_pct, oos_sharpe, oos_net_return_pct, is_trades, oos_trades}, ...]
    monte_carlo_summary TEXT,     -- JSON {p_value, significant, underpowered}
    validation_summary TEXT,      -- JSON {bitget_sharpe, transfer_ratio, ci_low, ci_high, ...}
    warnings TEXT,                -- JSON array de strings

    -- Flag
    is_latest INTEGER DEFAULT 1,  -- 1=dernier run pour ce (strategy, asset, timeframe)

    UNIQUE(strategy_name, asset, timeframe, created_at)
);

CREATE INDEX IF NOT EXISTS idx_opt_strategy_asset ON optimization_results(strategy_name, asset);
CREATE INDEX IF NOT EXISTS idx_opt_grade ON optimization_results(grade);
CREATE INDEX IF NOT EXISTS idx_opt_latest ON optimization_results(is_latest) WHERE is_latest = 1;
CREATE INDEX IF NOT EXISTS idx_opt_created ON optimization_results(created_at);
```

**Justification** :
- Colonnes SQL pour métriques clés → filtrage/tri rapide côté API
- JSON blobs pour fenêtres WFO, validation, params → flexibilité, pas requêtable directement
- `total_score REAL` : permet tri fin (B à 72 vs B à 68)
- `n_windows INTEGER` : évite de parser le JSON pour une info fréquemment affichée
- `created_at TEXT` ISO 8601 : lisible, triable, sans ambiguïté (vs REAL epoch)
- `is_latest` : index partiel pour récupérer vite les résultats courants

---

## 2. Fichiers à créer

### 2.1. `backend/optimization/optimization_db.py` (NOUVEAU) — Taille M

Module **synchrone** (sqlite3 standard) pour l'écriture depuis optimize.py.
Module **async** (aiosqlite) pour la lecture depuis l'API FastAPI.

```
Fonctions sync (pour optimize.py CLI) :
  save_result_sync(db_path, report: FinalReport, wfo_windows: list[dict], duration: float | None)
    → INSERT avec transaction : UPDATE is_latest=0 pour ancien, INSERT nouveau
  _compute_total_score(report) → float
    → Recalcule le score numérique depuis les métriques (réutilise la logique de compute_grade)

Fonctions async (pour l'API) :
  get_results_async(db_path, filters) → list[dict]
    → SELECT avec filtres (strategy, asset, grade min, is_latest)
  get_result_by_id_async(db_path, id) → dict | None
    → SELECT complet (inclut JSON blobs)
  get_comparison_async(db_path) → list[dict]
    → Tableau croisé strategies × assets (is_latest=1 seulement)
```

**Note sur total_score** : Refactorer `compute_grade()` pour retourner `(grade: str, score: int)` au lieu de seulement `str`. Tous les appelants existants (`build_final_report`) sont mis à jour. Pas de fonction parallèle `compute_total_score()` — une seule source de vérité. Le module optimization_db.py appelle `compute_grade()` et récupère le score directement.

### 2.2. `backend/api/optimization_routes.py` (NOUVEAU) — Taille M

```
Router FastAPI, préfixe /api/optimization

GET /api/optimization/results
  Query params : strategy (optionnel), asset (optionnel), min_grade (optionnel), latest_only (défaut true)
  Pagination : offset (défaut 0), limit (défaut 50)
  Response : { results: [...], total: int }
  Chaque résultat : { id, strategy_name, asset, grade, total_score, oos_sharpe, consistency,
                       oos_is_ratio, dsr, param_stability, n_windows, created_at, is_latest }

GET /api/optimization/{id}
  Response : résultat complet incluant best_params, wfo_windows, monte_carlo_summary,
             validation_summary, warnings (JSON parsés)
  404 si id inexistant

GET /api/optimization/comparison
  Response : tableau croisé {strategies: [...], assets: [...], matrix: {strategy: {asset: {grade, total_score, oos_sharpe, ...}}}}
  Uniquement is_latest=1
  Cases vides (stratégie non testée sur un asset) : clé absente du dict (pas null).
  Le frontend affiche "—" grisé pour les combos manquants.
```

**Pattern** : Même pattern que les autres routes (app.state.db pour la connexion). Mais ici on lit directement la DB sans passer par un composant app.state, car les données WFO ne dépendent pas du Simulator/DataEngine.

### 2.3. `scripts/migrate_optimization.py` (NOUVEAU) — Taille S

Script CLI idempotent pour importer les JSON existants dans la DB.

```
1. Créer la table optimization_results (si absente)
2. Pour chaque fichier *_YYYYMMDD_HHMMSS.json dans data/optimization/ :
   a. Parser le JSON avec .get() défensif + valeurs par défaut
   b. Chercher le fichier intermediate correspondant (wfo_{strategy}_{symbol}_intermediate.json)
   c. Si trouvé : extraire les wfo_windows. Sinon : wfo_windows = NULL + log warning
   d. Déduire timeframe via STRATEGY_REGISTRY (acceptable en migration one-shot)
   e. Calculer total_score via compute_grade() refactoré
   f. INSERT OR IGNORE avec is_latest=0 (basé sur UNIQUE constraint)
3. Pass finale is_latest : pour chaque (strategy, asset, timeframe),
   UPDATE is_latest=1 sur la row avec created_at MAX
4. Log détaillé :
   - X fichiers importés
   - Y skippés (doublons)
   - Z erreurs (avec chemin du fichier)
   - W reports sans données fenêtre WFO (intermediate manquant)

Mode dry-run : --dry-run liste les fichiers qui seraient importés sans écrire en DB.
Idempotent : relanceable sans doubler les données (UNIQUE constraint + INSERT OR IGNORE).
```

**Gestion valeurs spéciales JSON** :
- `oos_sharpe: null` → stocker comme NULL
- `profit_factor: Infinity` → remplacer par None (pas stocké en colonnes, seulement dans le JSON blob)
- `convergence: null` → acceptable (pas tous les runs ont la convergence cross-asset)

### 2.4. `frontend/src/components/ResearchPage.jsx` (NOUVEAU) — Taille L

Page principale "Recherche" avec 2 vues : tableau + détail.

```
État local :
  - filters: {strategy, asset, minGrade}
  - selectedId: null | number
  - view: 'table' | 'detail'

Vue tableau (défaut) :
  - Barre de filtres (selects : stratégie, asset, grade min)
  - Tableau comparatif : colonnes = Stratégie, Asset, Grade (badge couleur), Score,
    OOS Sharpe, Consistance, OOS/IS Ratio, DSR, Stabilité, Date
  - Tri cliquable sur chaque colonne (défaut : total_score DESC)
  - Clic sur une ligne → vue détail

Vue détail :
  - Bouton retour vers le tableau
  - Section "Paramètres retenus" (liste key=value)
  - Section "Scores" (5 critères avec barre de progression visuelle)
  - Section "Equity Curve IS vs OOS" (chart : 1 point par fenêtre WFO)
  - Section "Validation Bitget" (Sharpe, CI, transfer ratio)
  - Section "Monte Carlo" (p-value, significant/underpowered)
  - Section "Warnings" (liste si non vide)
```

### 2.5. `frontend/src/components/WfoChart.jsx` (NOUVEAU) — Taille S

Composant chart pour l'equity curve IS vs OOS par fenêtre WFO.

```
Props : windows (array de {window_index, is_sharpe, oos_sharpe, is_net_return_pct, oos_net_return_pct})

Rendu SVG :
  - Axe X : numéro de fenêtre (0..N-1)
  - Axe Y : Sharpe ratio (ou net return %, toggle)
  - 2 lignes : IS (bleu) + OOS (orange)
  - Le décrochage IS→OOS = overfitting visible
  - Tooltip au hover sur chaque point
```

Note : SVG natif, pas de lib externe (Recharts/Chart.js). Le projet n'a pas de dépendance chart, et les données sont petites (20-30 points max).

---

## 3. Fichiers à modifier

### 3.1. `backend/core/database.py` — Taille S

Ajouter la création de la table `optimization_results` dans `_create_tables()`.
Suivre le pattern existant : appel idempotent `CREATE TABLE IF NOT EXISTS`.
Pas de méthodes CRUD dans Database — le module optimization_db.py s'en charge.

### 3.2. `backend/optimization/report.py` — Taille S

**Refactorer `compute_grade()`** : retourne `tuple[str, int]` (grade, score) au lieu de `str`.
Mettre à jour `build_final_report()` pour récupérer le tuple. Ajouter `total_score: int` au FinalReport dataclass.
**Important** : Mettre à jour `_report_to_dict()` pour inclure `total_score` dans le JSON (backward compat : anciens JSON sans ce champ).

Modifier `save_report()` pour aussi sauvegarder en DB :

```python
def save_report(report, wfo_windows=None, duration=None, timeframe=None, output_dir="data/optimization", db_path="data/scalp_radar.db"):
    # 1. Sauvegarde JSON (existant, inchangé)
    filepath = _save_json(report, output_dir)
    # 2. Sauvegarde DB (nouveau)
    save_result_sync(db_path, report, wfo_windows, duration, timeframe)
    return filepath
```

**Note DB path** : Le paramètre `db_path` a le même défaut que dans `Database()`. Idéalement, importer depuis config plutôt que hardcoder, mais pour cohérence immédiate, le défaut est identique à database.py.

Passer `wfo_windows` + `timeframe` depuis `optimize.py` (déjà disponible dans `wfo.windows` et config stratégie).

### 3.3. `scripts/optimize.py` — Taille S

Modifier `run_optimization()` pour passer les window details + timeframe à `save_report()` :

```python
# Après build_final_report()
report = build_final_report(wfo, overfit, validation)
save_report(report, wfo_windows=[...serialize wfo.windows...], duration=elapsed, timeframe=main_tf)
```

Sérialiser les WindowResult en dict avant passage (comme dans `_save_wfo_intermediate()`).
`main_tf` est déjà disponible dans `run_optimization()` via `default_cfg.timeframe`.

### 3.4. `backend/api/server.py` — Taille XS

Ajouter le router :
```python
from backend.api.optimization_routes import router as optimization_router
app.include_router(optimization_router)
```

### 3.5. `frontend/src/App.jsx` — Taille XS

Ajouter le tab "Recherche" :
```jsx
const TABS = [
  { id: 'scanner', label: 'Scanner' },
  { id: 'heatmap', label: 'Heatmap' },
  { id: 'risk', label: 'Risque' },
  { id: 'research', label: 'Recherche' },  // NEW
]

// Dans le render :
{activeTab === 'research' && <ResearchPage />}
```

La page "Recherche" n'a pas besoin de wsData (données statiques, polling API).

---

## 4. Ordre d'implémentation

### Bloc 1 — DB Schema + Module DB (S)
1. Modifier `database.py` : ajouter `CREATE TABLE optimization_results` dans `_create_tables()`
2. Refactorer `compute_grade()` dans `report.py` → retourne `(grade, score)` + ajouter `total_score` à FinalReport
3. Créer `optimization_db.py` : fonctions sync + async
4. **Test** : test unitaire insertion + lecture + compute_grade retourne bien le tuple

### Bloc 2 — Migration (S)
5. Créer `scripts/migrate_optimization.py`
6. Lancer en dry-run → vérifier les 49 fichiers parsés correctement
7. Lancer pour de vrai → vérifier en DB
8. **Test** : test migration idempotent (2 runs → même résultat)

### Bloc 3 — optimize.py modifié (S)
9. Modifier `report.py` : `save_report()` écrit aussi en DB
10. Modifier `optimize.py` : passe wfo_windows + duration à save_report
11. **Test** : run optimize.py sur une stratégie simple (vwap_rsi × BTC) → vérifier DB

### Bloc 4 — API endpoints (M)
12. Créer `optimization_routes.py` avec les 3 endpoints
13. Enregistrer dans `server.py`
14. **Test** : tests API (GET /results, GET /{id}, GET /comparison)

### Bloc 5 — Frontend (M)
15. Créer `ResearchPage.jsx` (tableau + filtres)
16. Créer `WfoChart.jsx` (SVG chart IS vs OOS)
17. Modifier `App.jsx` (nouveau tab)
18. **Test** : vérifier visuellement dans le navigateur

---

## 5. Points techniques critiques

### 5.1. Transaction is_latest
Quand optimize.py insère un nouveau run :
```python
conn.execute("BEGIN")
conn.execute(
    "UPDATE optimization_results SET is_latest=0 WHERE strategy_name=? AND asset=? AND timeframe=? AND is_latest=1",
    (strategy, asset, timeframe)
)
conn.execute("INSERT INTO optimization_results ...", ...)
conn.execute("COMMIT")
```

### 5.2. Sync/Async cohabitation
- `optimize.py` (CLI) : `sqlite3` standard (sync). Ouvre/ferme sa propre connexion.
- API FastAPI : `aiosqlite` via `app.state.db`. Lit la même DB file.
- WAL mode (déjà activé dans database.py) garantit que les lectures async ne bloquent pas pendant une écriture sync.
- **Important** : les fonctions async dans optimization_db.py utilisent `aiosqlite.connect()` avec leur propre connexion, pas `app.state.db._conn`. Raison : l'API pourrait tourner SANS DataEngine (en mode dev).

### 5.3. Valeurs JSON spéciales
- `null` → None en Python → NULL en SQLite (OK)
- `Infinity` dans profit_factor → `float('inf')` → remplacer par None avant json.dumps (json.dumps crashe sur Infinity)
- `NaN` dans oos_sharpe → remplacer par None

### 5.4. Timeframe — passé explicitement

Le FinalReport n'a pas de champ `timeframe`. Plutôt que de le deviner via `config_cls().timeframe` (fragile si le constructeur change), **optimize.py le passe explicitement** à `save_report()`. optimize.py connaît déjà le timeframe via la config de la stratégie. La migration le déduit via STRATEGY_REGISTRY (acceptable en one-shot).

### 5.5. CORS
Déjà configuré dans server.py pour localhost:5173. Pas de changement nécessaire (les nouvelles routes sont sur le même serveur).

### 5.6. SPA routing
Pas de React Router. Le frontend utilise des tabs (state local). Ajouter un tab "Recherche" = 1 entrée dans TABS + 1 condition de rendu. Zero risque de conflit avec FastAPI.

---

## 6. Tests à ajouter (~25-30 tests)

### `tests/test_optimization_db.py` (NOUVEAU)
- `test_save_result_sync` : insertion OK, vérifie toutes les colonnes
- `test_save_result_updates_is_latest` : 2 insertions → seul le dernier a is_latest=1
- `test_get_results_filter_strategy` : filtre par stratégie
- `test_get_results_filter_grade` : filtre par grade minimum
- `test_get_results_pagination` : offset + limit
- `test_get_result_by_id` : récupère le détail complet avec JSON parsés
- `test_get_result_by_id_not_found` : retourne None
- `test_get_comparison` : tableau croisé correct
- `test_special_values` : null, NaN, Infinity gérés correctement
- `test_compute_grade_returns_tuple` : compute_grade retourne (grade, score) avec score cohérent

### `tests/test_optimization_routes.py` (NOUVEAU)
- `test_get_results_ok` : 200 + pagination
- `test_get_results_filtered` : filtres strategy + asset
- `test_get_result_detail` : 200 avec JSON complet
- `test_get_result_not_found` : 404
- `test_get_comparison` : matrice strategies × assets

### `tests/test_migrate_optimization.py` (NOUVEAU)
- `test_migrate_final_report` : import 1 JSON → vérifie DB
- `test_migrate_with_intermediate` : merge final + intermediate → windows en DB
- `test_migrate_idempotent` : 2 runs → même count en DB
- `test_migrate_handles_missing_fields` : JSON incomplet → .get() avec défauts
- `test_migrate_dry_run` : --dry-run ne modifie pas la DB

---

## 7. Risques identifiés

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| Format JSON legacy inconsistant | Moyenne | Faible | .get() défensif + log des anomalies |
| Conflits WAL (optimize écrit pendant API lit) | Faible | Faible | WAL gère ça nativement |
| Performance tableau comparison (49 rows) | Nulle | Nul | 49 rows, pas de problème |
| SVG chart complexe | Faible | Moyen | Max 30 points, SVG basique suffit |
| compute_grade refactoring casse des appelants | Faible | Faible | Un seul appelant (build_final_report), facile à mettre à jour |

---

## 8. Estimation par bloc

| Bloc | Contenu | Taille |
|------|---------|--------|
| 1 | DB Schema + optimization_db.py + tests | S |
| 2 | Migration script + tests | S |
| 3 | optimize.py + report.py modifiés | S |
| 4 | API routes + tests | M |
| 5 | Frontend (ResearchPage + WfoChart + App.jsx) | M |
| **Total** | | **~1 session** |

---

## 9. Ce qui est explicitement hors scope

- Lanceur WFO depuis le dashboard (Sprint 14)
- Table `optimization_jobs` (Sprint 14)
- React Router / SPA routing avancé
- Suppression des JSON existants (gardés comme backup)
- Chart library externe (Recharts, Chart.js)
