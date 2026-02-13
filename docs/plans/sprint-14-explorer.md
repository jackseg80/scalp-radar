# Sprint 14 — Explorateur de Paramètres

## Contexte

Le projet scalp-radar a 8 stratégies optimisables via WFO (Walk-Forward Optimization). Actuellement, les optimisations ne se lancent que via CLI (`scripts/optimize.py`). Le Sprint 14 ajoute un **explorateur visuel** dans le dashboard pour lancer des WFO depuis le navigateur, suivre la progression en temps réel, et visualiser les résultats dans une heatmap interactive 2D.

---

## 1. Schéma DB — Table `optimization_jobs`

```sql
CREATE TABLE IF NOT EXISTS optimization_jobs (
    id TEXT PRIMARY KEY,                          -- UUID v4
    strategy_name TEXT NOT NULL,
    asset TEXT NOT NULL,
    timeframe TEXT NOT NULL,

    -- Statut
    status TEXT NOT NULL DEFAULT 'pending',        -- pending | running | completed | failed | cancelled
    progress_pct REAL DEFAULT 0,                   -- 0-100
    current_phase TEXT DEFAULT '',                  -- "WFO fenetre 3/12", "Monte Carlo", "Grading"

    -- Paramètres
    params_override TEXT,                           -- JSON : sous-grille custom (null = grille complète)

    -- Timing
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    duration_seconds REAL,

    -- Résultat
    result_id INTEGER,                             -- FK → optimization_results.id (quand terminé)
    error_message TEXT                              -- Si failed
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON optimization_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON optimization_jobs(created_at);
```

---

## 2. Liste des fichiers à créer/modifier

### Fichiers NOUVEAUX (4)

| Fichier | Rôle |
|---------|------|
| `backend/optimization/job_manager.py` | JobManager : FIFO queue, exécution background, callbacks |
| `frontend/src/components/ExplorerPage.jsx` | Page principale de l'explorateur |
| `frontend/src/components/HeatmapChart.jsx` | Heatmap SVG interactive |
| `tests/test_job_manager.py` | Tests unitaires du JobManager |

### Fichiers MODIFIES (7)

| Fichier | Modification |
|---------|-------------|
| `backend/core/database.py` | Ajouter `CREATE TABLE optimization_jobs` dans `_create_optimization_tables()` |
| `backend/optimization/walk_forward.py` | Ajouter `progress_callback` param à `optimize()` + `_cancel_event` |
| `backend/api/optimization_routes.py` | 4 nouveaux endpoints (POST run, GET jobs, GET jobs/{id}, GET param-grid, GET heatmap, DELETE cancel) |
| `backend/api/server.py` | Instancier `JobManager` dans lifespan + cleanup au shutdown |
| `backend/api/websocket_routes.py` | Ajouter broadcast des events `optimization_progress` |
| `frontend/src/App.jsx` | Ajouter tab "Explorer" + import ExplorerPage |
| `scripts/optimize.py` | Factoriser `run_optimization()` pour être réutilisable (extraire la logique core) |

---

## 3. Détail par fichier

### 3.1 `backend/optimization/job_manager.py` (NOUVEAU)

**Rôle** : Gestionnaire de jobs WFO avec file d'attente FIFO et exécution séquentielle.

**Classes/fonctions** :

```python
@dataclass
class OptimizationJob:
    id: str                    # uuid4
    strategy_name: str
    asset: str
    timeframe: str
    status: str                # pending|running|completed|failed|cancelled
    progress_pct: float
    current_phase: str
    params_override: dict | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    duration_seconds: float | None
    result_id: int | None
    error_message: str | None

class JobManager:
    def __init__(self, db_path: str, ws_broadcast: Callable)

    async def start(self) -> None
        # Lance la boucle _worker_loop en background task

    async def stop(self) -> None
        # Annule le job en cours si besoin, arrête la boucle

    async def submit_job(self, strategy_name, asset, params_override=None) -> str
        # Crée le job en DB (status=pending), l'ajoute à la queue
        # Retourne job_id (UUID)

    async def cancel_job(self, job_id: str) -> bool
        # Si pending → passer en cancelled
        # Si running → setter le cancel_event → le WFO check à chaque fenêtre

    async def get_job(self, job_id: str) -> OptimizationJob | None
        # Lire depuis DB

    async def list_jobs(self, status: str | None = None) -> list[OptimizationJob]
        # Lire depuis DB avec filtre optionnel

    async def _worker_loop(self) -> None
        # Boucle infinie : dépile la queue, exécute _run_job(), repeat
        # asyncio.Queue interne

    async def _run_job(self, job: OptimizationJob) -> None
        # 1. Update status=running en DB
        # 2. Broadcast WS
        # 3. Lance run_optimization() dans asyncio.to_thread()
        # 4. Update status=completed/failed en DB
        # 5. Broadcast WS final

    def _make_progress_callback(self, job_id: str) -> Callable
        # Retourne un callback qui :
        #   - Update le job en DB (progress_pct, current_phase)
        #   - Broadcast WS {"type": "optimization_progress", ...}
        #   - Check cancel_event → raise CancelledError si set
```

**Dépendances** : `aiosqlite`, `asyncio`, `uuid`, `websocket_routes.manager`

**Points critiques** :
- `run_optimization()` est async mais CPU-bound (WFO utilise ProcessPool). On utilise `asyncio.to_thread()` pour la partie CPU pour ne pas bloquer l'event loop.
- Probleme : `run_optimization()` appelle `await optimizer.optimize()` qui est async. On ne peut pas simplement le mettre dans `to_thread`. **Solution** : garder l'appel async dans le main event loop mais le WFO interne utilise déjà `ProcessPoolExecutor` qui ne bloque pas l'event loop (les CPU tasks sont dans des process séparés).
  - En fait, `optimize()` est `async def` mais les parties CPU (`_parallel_backtest`, `_run_fast`, `_run_pool`) sont synchrones (ProcessPoolExecutor.map bloque le thread courant). Donc **l'event loop FastAPI est bloqué** pendant le WFO.
  - **Vraie solution** : wrapper `run_optimization()` dans `asyncio.to_thread()`. Pour ça, il faut une version sync de `run_optimization()` OU lancer un event loop secondaire dans le thread. Le plus simple : `asyncio.to_thread(lambda: asyncio.run(run_optimization(...)))` — crée un event loop dédié dans le thread.
  - **Alternative plus propre** : refactorer `run_optimization()` en extrayant les parties sync (WFO, overfit, validation sont en fait sync sauf le chargement DB initial). Mais c'est un gros refactoring.
  - **Décision** : `asyncio.to_thread()` avec `asyncio.run()` interne. Simple, isolé, le thread a son propre event loop pour les appels DB async de WFO.

**Annulation** : `threading.Event` vérifié dans le progress callback. Si set, le callback lève `asyncio.CancelledError` qui interrompt le WFO.

### 3.2 `backend/optimization/walk_forward.py` (MODIFIE)

**Modifications** :

1. **Signature `optimize()`** : ajouter `progress_callback: Callable[[float, str, str], None] | None = None`
   - Appelé avec `(pct, phase, detail)` ex: `(25.0, "WFO", "Fenêtre 3/12")`
   - Si None → comportement inchangé (CLI)

2. **Appels callback** dans la boucle fenêtres (ligne ~481) :
   ```python
   # Après chaque fenêtre WFO complétée
   if progress_callback:
       pct = (w_idx + 1) / len(windows) * 80  # WFO = 80% du total
       progress_callback(pct, "WFO", f"Fenêtre {w_idx + 1}/{len(windows)}")
   ```

3. **Ajouter `cancel_event: threading.Event | None = None`** :
   - Check au début de chaque fenêtre : `if cancel_event and cancel_event.is_set(): raise asyncio.CancelledError("Job annulé")`

**Impact** : minimal — 2 params optionnels, ~10 lignes ajoutées. Zéro changement pour le CLI.

### 3.3 `scripts/optimize.py` (MODIFIE)

**Factorisation** : extraire `run_optimization()` pour accepter un `progress_callback` et un `cancel_event` qu'elle passe à `optimizer.optimize()` et aux phases suivantes.

```python
async def run_optimization(
    strategy_name: str,
    symbol: str,
    config_dir: str = "config",
    verbose: bool = False,
    all_symbols_results: dict[str, dict] | None = None,
    db: Database | None = None,
    progress_callback: Callable | None = None,  # NOUVEAU
    cancel_event: threading.Event | None = None,  # NOUVEAU
    params_override: dict | None = None,  # NOUVEAU : sous-grille custom
) -> FinalReport:
```

Les callbacks sont passés à `optimizer.optimize()`.

Le `params_override` permet de passer une sous-grille spécifique (pour l'explorateur).

**Phase de progression** :
- WFO : 0-80%
- Overfitting : 80-90%
- Validation Bitget : 90-95%
- Grading + Save : 95-100%

### 3.4 `backend/api/optimization_routes.py` (MODIFIE)

**6 nouveaux endpoints** :

```
POST /api/optimization/run
  Body: { strategy_name, asset, params_override? }
  → Crée un job (pending) + l'ajoute à la queue
  → Retourne: { job_id, status: "pending" }
  → 409 si un job running/pending existe déjà pour cette (strategy, asset)

GET /api/optimization/jobs
  Query: status? (filter)
  → Liste des jobs avec statut, progression, durée
  → Retourne: { jobs: [...] }

GET /api/optimization/jobs/{job_id}
  → Détail d'un job
  → 404 si inexistant

DELETE /api/optimization/jobs/{job_id}
  → Annule un job pending/running
  → 404 si inexistant, 409 si déjà terminé

GET /api/optimization/param-grid/{strategy_name}
  → Retourne les paramètres disponibles avec min/max/step/default
  → Source : param_grids.yaml + strategies.yaml
  → Retourne: { strategy, params: { param_name: {values, min, max, step, default} } }

GET /api/optimization/heatmap
  Query: strategy, asset, param_x, param_y, metric (default: "total_score")
  → Matrice 2D depuis optimization_results existants
  → Retourne: { x_param, y_param, metric, x_values, y_values, cells: [[{value, grade, result_id}]] }
```

**Accès au JobManager** : via `request.app.state.job_manager`

### 3.5 `backend/api/server.py` (MODIFIE)

**Dans `lifespan()`** :

```python
# Après database init, avant yield
from backend.optimization.job_manager import JobManager
from backend.api.websocket_routes import manager as ws_manager

job_manager = JobManager(db_path=db_path_str, ws_broadcast=ws_manager.broadcast)
await job_manager.start()
app.state.job_manager = job_manager

# Dans shutdown (avant db.close())
if hasattr(app.state, 'job_manager') and app.state.job_manager:
    await app.state.job_manager.stop()
```

### 3.6 `backend/api/websocket_routes.py` (MODIFIE)

Pas de modification structurelle nécessaire. Le `JobManager` utilise directement `manager.broadcast()` importé depuis ce module. Le broadcast est déjà async et thread-safe (appelé depuis l'event loop).

**Format du message WS** :
```json
{
  "type": "optimization_progress",
  "job_id": "uuid-...",
  "status": "running",
  "progress_pct": 42.5,
  "current_phase": "WFO Fenêtre 5/12",
  "strategy_name": "envelope_dca",
  "asset": "BTC/USDT"
}
```

### 3.7 `backend/core/database.py` (MODIFIE)

Ajouter dans `_create_optimization_tables()` :
```sql
CREATE TABLE IF NOT EXISTS optimization_jobs (...)
```
(schéma vu en section 1)

### 3.8 `frontend/src/App.jsx` (MODIFIE)

```jsx
import ExplorerPage from './components/ExplorerPage'

const TABS = [
  { id: 'scanner', label: 'Scanner' },
  { id: 'heatmap', label: 'Heatmap' },
  { id: 'risk', label: 'Risque' },
  { id: 'research', label: 'Recherche' },
  { id: 'explorer', label: 'Explorer' },  // NOUVEAU
]

// Dans le JSX :
{activeTab === 'explorer' && <ExplorerPage wsData={wsData} />}
```

### 3.9 `frontend/src/components/ExplorerPage.jsx` (NOUVEAU)

**Layout** : 3 zones
- **Gauche (300px)** : Panneau de contrôle
- **Centre (flex)** : Heatmap + WFO chart du résultat sélectionné
- **Bas (200px)** : Liste des jobs

**Panneau de contrôle (gauche)** :
- `<select>` stratégie (8 stratégies de STRATEGY_REGISTRY)
- `<select>` asset (5 assets de assets.yaml)
- Paramètres dynamiques : chargés depuis `GET /api/optimization/param-grid/{strategy}`
  - Chaque param : slider `<input type="range">` + valeur numérique `<input type="number">`
  - Min/max/step depuis le backend
  - Default = valeur actuelle strategies.yaml
- Sélection axes heatmap :
  - `<select>` Axe X (un des paramètres)
  - `<select>` Axe Y (un des paramètres)
  - `<select>` Métrique couleur : total_score, oos_sharpe, consistency, dsr
- Bouton "Lancer WFO" → `POST /api/optimization/run`
  - Disabled si un job running/pending existe

**State management** :
```jsx
const [strategy, setStrategy] = useState('')
const [asset, setAsset] = useState('')
const [paramGrid, setParamGrid] = useState(null)    // depuis GET param-grid
const [paramValues, setParamValues] = useState({})   // valeurs slider courantes
const [axisX, setAxisX] = useState('')
const [axisY, setAxisY] = useState('')
const [metric, setMetric] = useState('total_score')
const [heatmapData, setHeatmapData] = useState(null) // depuis GET heatmap
const [jobs, setJobs] = useState([])                  // depuis GET jobs
```

**Jobs en temps réel** :
- Écouter `wsData` (prop) pour les messages `type === "optimization_progress"`
- Mettre à jour la progress bar du job correspondant
- À la complétion d'un job : re-fetch heatmap + jobs list

**Styles** : styled-jsx, dark theme (#1a1a1a, #333 borders), pattern ResearchPage.jsx

### 3.10 `frontend/src/components/HeatmapChart.jsx` (NOUVEAU)

**SVG pur** (comme WfoChart.jsx, pas de lib externe).

**Structure** :
```
┌──────────────────────────────────────┐
│  Y-axis label (param Y)             │
│  ┌──────────────────────────────┐    │
│  │  [cellule] [cellule] [...]   │    │
│  │  [cellule] [cellule] [...]   │    │
│  │  [...]                       │    │
│  └──────────────────────────────┘    │
│        X-axis label (param X)        │
│  [légende : échelle de couleur]      │
└──────────────────────────────────────┘
```

**Props** :
```jsx
HeatmapChart({ data, xParam, yParam, metric, onCellClick })
// data = { x_values, y_values, cells: [[{value, grade, result_id}]] }
```

**Rendu** :
- Grille de `<rect>` SVG, une par cellule
- Couleur : échelle rouge→jaune→vert (interpolation linéaire sur la plage min-max des valeurs)
- Cellules sans données : gris foncé (#2a2a2a)
- Hover : `<title>` avec grade, score, params complets (pattern WfoChart.jsx)
- Clic : `onCellClick(result_id)` → ouvre dans ResearchPage ou détail inline

**Fonction de couleur** :
```javascript
function metricToColor(value, min, max) {
  const t = (value - min) / (max - min)  // 0=pire, 1=meilleur
  // Rouge → Jaune → Vert
  if (t < 0.5) {
    return interpolate('#ef4444', '#f59e0b', t * 2)
  } else {
    return interpolate('#f59e0b', '#10b981', (t - 0.5) * 2)
  }
}
```

**Dimensions** : responsive, width=600, height calculée selon nombre de valeurs Y, cellSize ~40-60px.

### 3.11 `tests/test_job_manager.py` (NOUVEAU)

**Tests** (~15-20 tests) :

1. **DB** : create table, insert job, read job, list jobs, update status
2. **JobManager.submit_job()** : crée un job pending en DB
3. **JobManager.cancel_job()** : pending → cancelled, running → cancelled via event
4. **JobManager.list_jobs()** : filtre par status
5. **Worker loop** : un job submitted → passe en running → completed
6. **FIFO** : 2 jobs soumis, le 2e attend que le 1er finisse
7. **Erreur** : job qui fail → status=failed + error_message
8. **Progress callback** : callback appelé, met à jour le job en DB
9. **Endpoint POST /run** : retourne job_id, crée en DB
10. **Endpoint GET /jobs** : retourne la liste
11. **Endpoint GET /param-grid** : retourne les params depuis param_grids.yaml
12. **Endpoint GET /heatmap** : retourne la matrice depuis optimization_results
13. **Endpoint DELETE cancel** : annule un job pending

---

## 4. Ordre d'implémentation

### Bloc A — DB + JobManager core (M)
1. Modifier `database.py` : ajouter `CREATE TABLE optimization_jobs`
2. Créer `job_manager.py` : dataclass `OptimizationJob`, classe `JobManager` (submit, cancel, get, list, CRUD DB)
3. Tests : DB CRUD jobs, submit/cancel logic
4. **Pas de worker loop encore** — juste la structure

### Bloc B — Progress callback WFO (S)
1. Modifier `walk_forward.py` : ajouter `progress_callback` + `cancel_event` à `optimize()`
2. Modifier `scripts/optimize.py` : factoriser `run_optimization()` avec les nouveaux params
3. Tests : vérifier que le CLI fonctionne toujours (pas de régression), callback appelé correctement

### Bloc C — Worker loop + thread (M)
1. Implémenter `_worker_loop()` et `_run_job()` dans JobManager
2. Intégrer `asyncio.to_thread()` avec event loop dédié
3. Brancher le progress callback → update DB + broadcast WS
4. Intégrer dans `server.py` lifespan (start/stop)
5. Tests : job complet de bout en bout (mock WFO rapide)

### Bloc D — Endpoints API (S)
1. `POST /api/optimization/run`
2. `GET /api/optimization/jobs` et `GET /api/optimization/jobs/{id}`
3. `DELETE /api/optimization/jobs/{id}`
4. `GET /api/optimization/param-grid/{strategy}`
5. `GET /api/optimization/heatmap`
6. Tests : endpoints avec TestClient

### Bloc E — Frontend ExplorerPage (L)
1. Ajouter tab "Explorer" dans App.jsx
2. Créer `ExplorerPage.jsx` : layout, sélection stratégie/asset, panneau contrôle
3. Charger param-grid dynamiquement, sliders
4. Bouton "Lancer WFO" → POST
5. Liste des jobs avec progression WS temps réel
6. Créer `HeatmapChart.jsx` : SVG, échelle couleur, hover/click
7. Brancher heatmap sur `GET /api/optimization/heatmap`

---

## 5. Risques identifiés

| Risque | Impact | Mitigation |
|--------|--------|-----------|
| **WFO bloque l'event loop FastAPI** | Dashboard freeze pendant le WFO | `asyncio.to_thread()` avec event loop dédié — le WFO tourne dans son propre thread |
| **Crash ProcessPool pendant un job** | Job stuck en "running" | Timeout watchdog (max 60 min par job), fallback séquentiel déjà en place dans WFO |
| **Serveur restart pendant un job** | Job orphelin en "running" | Au boot, scanner les jobs `running` et les passer en `failed` avec message "Serveur redémarré" |
| **Mémoire** | WFO consomme ~2-4GB sur un gros grid | Les workers et le batch size limitent déjà la conso. Le job s'exécute seul (FIFO). |
| **Candles insuffisantes sur le serveur** | WFO échoue immédiatement | Le WFO lève `ValueError("Pas de candles")` → capturé, job=failed avec message clair |
| **Heatmap vide** | Aucun résultat existant pour la combo | Afficher un message "Aucun résultat. Lancez un WFO pour remplir la heatmap." + cellules grises |
| **Annulation mid-WFO** | Resources leakées | `cancel_event` vérifié à chaque fenêtre, ProcessPool terminé proprement via context manager |

---

## 6. Tests à ajouter

| Fichier test | Tests | Type |
|-------------|-------|------|
| `tests/test_job_manager.py` | ~15 tests : DB CRUD, submit, cancel, FIFO, progress, erreurs | Unit |
| `tests/test_optimization_routes.py` (existant ou nouveau) | ~10 tests : endpoints POST/GET/DELETE, param-grid, heatmap | Integration |
| `tests/test_walk_forward.py` (existant) | ~2 tests : progress callback appelé, cancel_event respecté | Unit |

**Total estimé** : ~25-30 tests supplémentaires → ~580+ tests au total

---

## 7. Estimation par bloc

| Bloc | Description | Taille | Estimé |
|------|-------------|--------|--------|
| **A** | DB schema + JobManager structure | M | Session 1 |
| **B** | Progress callback WFO + refactoring optimize.py | S | Session 1 |
| **C** | Worker loop + thread + WS broadcast | M | Session 2 |
| **D** | Endpoints API (6 routes) | S | Session 2 |
| **E** | Frontend ExplorerPage + HeatmapChart | L | Session 3 |

**Total : ~3 sessions** (conforme au ROADMAP)

---

## 8. Détails techniques importants

### Thread isolation du WFO

Le WFO (`optimize()`) est `async def` mais contient des appels bloquants (ProcessPoolExecutor.map, calculs numpy). On ne peut pas le lancer directement dans l'event loop FastAPI car il bloquerait les requêtes HTTP.

**Solution retenue** :
```python
async def _run_job(self, job):
    # Préparer le callback qui utilise l'event loop principal pour le broadcast WS
    loop = asyncio.get_event_loop()

    def progress_cb(pct, phase, detail):
        # Update DB (sync, le thread a sa propre connexion sqlite3)
        self._update_job_progress_sync(job.id, pct, phase)
        # Broadcast WS via l'event loop principal
        asyncio.run_coroutine_threadsafe(
            self._ws_broadcast({"type": "optimization_progress", ...}),
            loop,
        )
        # Check annulation
        if self._cancel_events.get(job.id, threading.Event()).is_set():
            raise CancelledError("Job annulé")

    # Lancer dans un thread dédié avec son propre event loop
    await asyncio.to_thread(self._run_optimization_sync, job, progress_cb)

def _run_optimization_sync(self, job, progress_cb):
    """Exécuté dans un thread séparé."""
    asyncio.run(run_optimization(
        strategy_name=job.strategy_name,
        symbol=job.asset,
        progress_callback=progress_cb,
        cancel_event=self._cancel_events.get(job.id),
    ))
```

### Endpoint heatmap — extraction des données

```python
@router.get("/heatmap")
async def get_heatmap(
    strategy: str, asset: str,
    param_x: str, param_y: str,
    metric: str = "total_score",
):
    db_path = _get_db_path()
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        rows = await conn.execute_fetchall(
            """SELECT id, total_score, oos_sharpe, consistency, dsr,
                      best_params, grade
               FROM optimization_results
               WHERE strategy_name = ? AND asset = ?""",
            (strategy, asset),
        )

    # Extraire les valeurs de param_x et param_y depuis best_params (JSON)
    # Construire la matrice x_values × y_values
    # Chaque cellule = {value: metric_value, grade, result_id}
```

### Recovery au boot

Dans `JobManager.start()` :
```python
# Passer tous les jobs "running" en "failed" (serveur a redémarré)
await self._recover_orphaned_jobs()
# Relancer les jobs "pending" dans la queue
await self._enqueue_pending_jobs()
```

### params_override depuis l'explorateur

Le `params_override` est un dict `{param_name: [values]}` envoyé par le frontend. Dans `optimize()`, il sera fusionné dans les grids :
```python
# walk_forward.py ligne ~437
strategy_grids = self._grids.get(strategy_name, {})
if params_override:
    # Fusionner : override remplace les valeurs dans default
    merged_default = {**strategy_grids.get("default", {}), **params_override}
    strategy_grids = {**strategy_grids, "default": merged_default}
full_grid = _build_grid(strategy_grids, symbol)
```

### Endpoint param-grid — extraction min/max/step

Les valeurs dans `param_grids.yaml` sont des listes discrètes (ex: `[5, 7, 10]`). L'endpoint retourne :
```json
{
  "strategy": "envelope_dca",
  "params": {
    "ma_period": {"values": [5, 7, 10], "default": 7},
    "num_levels": {"values": [2, 3, 4], "default": 2}
  }
}
```
Le frontend affiche un slider discret (snap aux valeurs de la liste) plutôt qu'un slider continu avec step calculé. Plus simple, plus fiable.

---

## 9. Vérification end-to-end

### Tests automatisés
```bash
# Lancer tous les tests (doit rester vert)
uv run pytest tests/ -x -q

# Tests spécifiques Sprint 14
uv run pytest tests/test_job_manager.py -v
uv run pytest tests/test_optimization_routes.py -v -k "job or param_grid or heatmap"
```

### Test manuel backend
```bash
# 1. Démarrer le serveur
uv run uvicorn backend.api.server:app --reload

# 2. Vérifier les nouveaux endpoints
curl http://localhost:8000/api/optimization/param-grid/envelope_dca
curl http://localhost:8000/api/optimization/jobs

# 3. Lancer un WFO (si candles dispo)
curl -X POST http://localhost:8000/api/optimization/run \
  -H "Content-Type: application/json" \
  -d '{"strategy_name": "envelope_dca", "asset": "BTC/USDT"}'

# 4. Suivre la progression
curl http://localhost:8000/api/optimization/jobs/{job_id}

# 5. Vérifier le heatmap (après WFO terminé)
curl "http://localhost:8000/api/optimization/heatmap?strategy=envelope_dca&asset=BTC/USDT&param_x=envelope_start&param_y=envelope_step&metric=total_score"
```

### Test manuel frontend
1. Ouvrir le dashboard (`npm run dev`)
2. Aller sur l'onglet "Explorer"
3. Sélectionner envelope_dca + BTC/USDT
4. Vérifier que les sliders apparaissent avec les bonnes valeurs
5. Ajuster les sliders, sélectionner les axes heatmap
6. Cliquer "Lancer WFO" → vérifier la progress bar en temps réel
7. À la fin, vérifier la heatmap + clic sur une cellule → détail
8. Tester l'annulation d'un job en cours

### Non-régression
- Vérifier que `uv run python -m scripts.optimize --strategy envelope_dca --symbol BTC/USDT` fonctionne toujours (progress_callback=None → pas de régression)
- Vérifier que la page Recherche (Sprint 13) fonctionne toujours
- Vérifier que le WebSocket `/ws/live` continue à pousser les updates toutes les 3s
