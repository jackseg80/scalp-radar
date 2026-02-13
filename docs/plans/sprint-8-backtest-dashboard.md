# Sprint 8 — Backtest & Optimization Dashboard

## Objectif

Exposer les résultats de backtest et d'optimisation (Sprint 7) dans le dashboard existant, avec persistance en DB, endpoints API REST en lecture seule, suivi de progression temps réel via WebSocket, et visualisation complète (equity curves Recharts, métriques, grades, fenêtres WFO, détection d'overfitting).

Les scripts CLI (`run_backtest`, `optimize`) continuent à fonctionner en standalone ET écrivent désormais en DB. Les backtests et optimisations sont lancés uniquement via CLI — le lancement depuis le dashboard est hors scope (Sprint 9, nécessite un vrai job runner).

**Règle importante** : Les backtests intermédiaires générés par le WFO (centaines par fenêtre) ne sont PAS persistés dans `backtest_runs`. Seuls les backtests lancés explicitement par l'utilisateur (`scripts/run_backtest.py`) y sont stockés.

---

## 1. Nouvelles tables DB

### 1.1 Table `backtest_runs`

Stocke chaque exécution de backtest lancé explicitement par l'utilisateur (CLI).

```sql
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id TEXT PRIMARY KEY,                    -- UUID v4
    strategy TEXT NOT NULL,                     -- "vwap_rsi", "momentum"
    symbol TEXT NOT NULL,                       -- "BTC/USDT"
    exchange TEXT NOT NULL DEFAULT 'bitget',    -- "binance", "bitget"
    start_date TEXT NOT NULL,                   -- ISO 8601
    end_date TEXT NOT NULL,                     -- ISO 8601
    initial_capital REAL NOT NULL DEFAULT 10000,
    final_capital REAL NOT NULL,
    leverage INTEGER NOT NULL DEFAULT 15,
    -- Métriques performance
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    win_rate REAL NOT NULL DEFAULT 0,
    net_pnl REAL NOT NULL DEFAULT 0,
    net_return_pct REAL NOT NULL DEFAULT 0,
    gross_pnl REAL NOT NULL DEFAULT 0,
    total_fees REAL NOT NULL DEFAULT 0,
    total_slippage REAL NOT NULL DEFAULT 0,
    profit_factor REAL NOT NULL DEFAULT 0,
    gross_profit_factor REAL NOT NULL DEFAULT 0,
    -- Risque
    sharpe_ratio REAL NOT NULL DEFAULT 0,
    sortino_ratio REAL NOT NULL DEFAULT 0,
    max_drawdown_pct REAL NOT NULL DEFAULT 0,
    max_drawdown_duration_hours REAL NOT NULL DEFAULT 0,
    -- Détails
    fee_drag_pct REAL NOT NULL DEFAULT 0,
    avg_win REAL NOT NULL DEFAULT 0,
    avg_loss REAL NOT NULL DEFAULT 0,
    risk_reward_ratio REAL NOT NULL DEFAULT 0,
    expectancy REAL NOT NULL DEFAULT 0,
    -- JSON blobs
    strategy_params_json TEXT,                  -- {"rsi_period": 14, ...}
    regime_stats_json TEXT,                     -- {"RANGING": {"trades": 45, ...}, ...}
    equity_curve_json TEXT,                     -- [10000, 10012.5, ...] (sous-échantillonné à 500 pts max)
    equity_timestamps_json TEXT,                -- ["2024-01-01T00:00:00", ...] (même taille)
    trades_summary_json TEXT,                   -- [{entry_time, exit_time, direction, net_pnl, regime}, ...]
    -- Métadonnées
    source TEXT NOT NULL DEFAULT 'cli',         -- "cli" uniquement dans ce sprint
    created_at TEXT NOT NULL                    -- ISO 8601
);

CREATE INDEX IF NOT EXISTS idx_backtest_runs_strategy_symbol
    ON backtest_runs (strategy, symbol);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_created
    ON backtest_runs (created_at DESC);
```

**Sous-échantillonnage equity curve** : Si > 500 points, on prend 1 point sur N pour garder exactement 500 points (plus le premier et le dernier). Cela représente ~2 Ko par run au lieu de ~200 Ko pour 720j en 5m. Recharts gère bien 500 points.

**`trades_summary_json`** : Résumé compact de chaque trade, pas le `TradeResult` complet. Permet d'afficher les trades sur l'equity curve (markers) et de détailler le breakdown par régime. Structure :

```json
[
  {"entry_time": "2024-01-15T10:05:00", "exit_time": "2024-01-15T10:35:00",
   "direction": "LONG", "net_pnl": 42.50, "regime": "RANGING"},
  ...
]
```

Taille estimée : ~100 bytes/trade × 200 trades = ~20 Ko (acceptable).

### 1.2 Table `optimization_runs`

Stocke chaque exécution d'optimisation (CLI). Les colonnes principales (grade, status, WFO summary, progression) sont plates et requêtables. Les détails overfitting et validation (qui pourraient évoluer si on ajoute Kraken en phase 2) sont dans un `details_json` unique.

```sql
CREATE TABLE IF NOT EXISTS optimization_runs (
    run_id TEXT PRIMARY KEY,                    -- UUID v4
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',     -- "running" | "completed" | "failed"
    -- Grade & décision (colonnes plates — requêtables)
    grade TEXT,                                 -- "A" | "B" | "C" | "D" | "F" (null tant que running)
    live_eligible INTEGER NOT NULL DEFAULT 0,
    recommended_params_json TEXT,               -- {"rsi_period": 14, ...}
    -- WFO summary (colonnes plates — utilisées pour trier/filtrer)
    wfo_n_windows INTEGER,
    wfo_avg_is_sharpe REAL,
    wfo_avg_oos_sharpe REAL,
    oos_is_ratio REAL,
    wfo_consistency_rate REAL,
    n_distinct_combos INTEGER,
    -- Détails overfitting + validation (JSON flexible)
    details_json TEXT,                          -- Voir structure ci-dessous
    -- Progression (mis à jour pendant l'exécution)
    progress_pct REAL NOT NULL DEFAULT 0,       -- 0.0 à 100.0
    progress_phase TEXT NOT NULL DEFAULT 'init', -- "init" | "wfo" | "overfitting" | "validation" | "done"
    progress_detail TEXT,                        -- "Fenêtre 3/20 — fine pass"
    -- Timestamps
    started_at TEXT NOT NULL,
    completed_at TEXT,
    -- Résultats additionnels
    warnings_json TEXT,                         -- ["Paramètres instables: sl_percent", ...]
    error_message TEXT,                         -- Message si status=failed
    -- Métadonnées
    source TEXT NOT NULL DEFAULT 'cli',         -- "cli" uniquement dans ce sprint
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_optimization_runs_strategy_symbol
    ON optimization_runs (strategy, symbol);
CREATE INDEX IF NOT EXISTS idx_optimization_runs_status
    ON optimization_runs (status);
CREATE INDEX IF NOT EXISTS idx_optimization_runs_created
    ON optimization_runs (created_at DESC);
```

**Structure de `details_json`** :

```json
{
  "overfitting": {
    "mc_p_value": 0.012,
    "mc_significant": true,
    "dsr": 0.96,
    "dsr_max_expected_sharpe": 1.45,
    "stability": 0.84,
    "convergence": 0.78,
    "cliff_params": ["sl_percent"],
    "divergent_params": ["tp_percent"]
  },
  "validation": {
    "bitget_sharpe": 0.71,
    "bitget_net_return_pct": 4.2,
    "bitget_trades": 28,
    "bitget_sharpe_ci_low": 0.32,
    "bitget_sharpe_ci_high": 1.14,
    "transfer_ratio": 0.76,
    "volume_warning": false,
    "volume_warning_detail": ""
  }
}
```

Ce design évite un ALTER TABLE si on ajoute un nouveau type de validation (ex: Kraken phase 2) ou un nouveau test d'overfitting. En pratique, on filtre par `grade` et `status`, pas par `mc_p_value`.

### 1.3 Table `optimization_windows`

Résultats détaillés par fenêtre WFO. Jointure sur `optimization_run_id`.

```sql
CREATE TABLE IF NOT EXISTS optimization_windows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    optimization_run_id TEXT NOT NULL,           -- FK → optimization_runs.run_id
    window_index INTEGER NOT NULL,
    is_start TEXT NOT NULL,
    is_end TEXT NOT NULL,
    oos_start TEXT NOT NULL,
    oos_end TEXT NOT NULL,
    best_params_json TEXT,
    is_sharpe REAL NOT NULL DEFAULT 0,
    is_net_return_pct REAL NOT NULL DEFAULT 0,
    is_profit_factor REAL NOT NULL DEFAULT 0,
    is_trades INTEGER NOT NULL DEFAULT 0,
    oos_sharpe REAL NOT NULL DEFAULT 0,
    oos_net_return_pct REAL NOT NULL DEFAULT 0,
    oos_profit_factor REAL NOT NULL DEFAULT 0,
    oos_trades INTEGER NOT NULL DEFAULT 0,
    UNIQUE(optimization_run_id, window_index)
);

CREATE INDEX IF NOT EXISTS idx_optim_windows_run
    ON optimization_windows (optimization_run_id);
```

### 1.4 Migration

Dans `database.py`, ajouter une méthode `_create_sprint8_tables()` appelée dans `_create_tables()`. Tables créées avec `IF NOT EXISTS` — migration idempotente, pas de backup nécessaire (tables nouvelles, aucune altération).

---

## 2. Modifications des scripts existants

### 2.1 `backend/core/database.py` — Nouvelles méthodes CRUD

**Méthodes ajoutées** (~120 lignes) :

```python
# ─── BACKTEST RUNS ────────────────────────────────────────────

async def insert_backtest_run(self, run: dict) -> None:
    """Insère un résultat de backtest."""

async def get_backtest_runs(
    self,
    strategy: str | None = None,
    symbol: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Liste les backtests avec filtres optionnels, triés par created_at DESC."""

async def get_backtest_run(self, run_id: str) -> dict | None:
    """Récupère un backtest par run_id."""

# ─── OPTIMIZATION RUNS ───────────────────────────────────────

async def insert_optimization_run(self, run: dict) -> None:
    """Insère une optimisation (status=running au départ)."""

async def update_optimization_progress(
    self, run_id: str, progress_pct: float, phase: str, detail: str,
) -> None:
    """Met à jour la progression (appelé pendant l'exécution)."""

async def update_optimization_result(self, run_id: str, result: dict) -> None:
    """Met à jour le résultat final (grade, details_json, etc.)."""

async def get_optimization_runs(
    self,
    strategy: str | None = None,
    symbol: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Liste les optimisations avec filtres, triés par created_at DESC."""

async def get_optimization_run(self, run_id: str) -> dict | None:
    """Récupère une optimisation par run_id."""

async def get_optimization_windows(self, run_id: str) -> list[dict]:
    """Récupère les fenêtres WFO d'une optimisation."""

async def insert_optimization_windows(
    self, run_id: str, windows: list[dict],
) -> None:
    """Insère les fenêtres WFO en batch."""

async def get_active_optimization(self) -> dict | None:
    """Retourne l'optimisation en cours (status=running), ou None."""
```

### 2.2 `scripts/run_backtest.py` — Persistance en DB

**Modifications** (~60 lignes ajoutées) :

**Attention — piège `asyncio.run()`** : Le script actuel fait `asyncio.run(_load_data(...))` pour charger les candles, puis le backtest est synchrone. On ne peut PAS faire un second `asyncio.run(_persist_backtest(...))` après — Python interdit deux `asyncio.run()` dans le même processus si la première event loop n'a pas été correctement fermée. La solution est de refactorer `run_backtest()` pour englober le tout dans un seul `asyncio.run()` :

```python
def run_backtest(args: argparse.Namespace) -> None:
    """Point d'entrée principal du backtest."""
    setup_logging()
    # ... parsing args, config ...
    asyncio.run(_run_backtest_async(args, config, bt_config, strategy, ...))

async def _run_backtest_async(args, config, bt_config, strategy, ...):
    """Charge données, lance le backtest (sync), persiste en DB."""
    db = Database()
    await db.init()
    try:
        # 1. Charger les candles (async)
        candles_by_tf = await load_candles(db, ...)

        # 2. Backtest (synchrone — pas de await)
        engine = BacktestEngine(bt_config, strategy)
        result = engine.run(candles_by_tf, main_tf=main_tf)
        metrics = calculate_metrics(result)

        # 3. Persister en DB (async)
        run_id = str(uuid.uuid4())
        equity_curve, equity_ts = _subsample_equity(...)
        trades_summary = _trades_to_summary(result.trades)
        await db.insert_backtest_run({...})

        # 4. Afficher résultats (identique à avant)
        ...
    finally:
        await db.close()
```

Le dict de persistance :

```python
backtest_data = {
    "run_id": run_id,
    "strategy": result.strategy_name,
    "symbol": bt_config.symbol,
    "exchange": "bitget",
    "start_date": bt_config.start_date.isoformat(),
    "end_date": bt_config.end_date.isoformat(),
    "initial_capital": bt_config.initial_capital,
    "final_capital": result.final_capital,
    "leverage": bt_config.leverage,
    # ... toutes les métriques ...
    "strategy_params_json": json.dumps(result.strategy_params),
    "regime_stats_json": json.dumps(metrics.regime_stats),
    "equity_curve_json": json.dumps(equity_curve),
    "equity_timestamps_json": json.dumps(equity_ts),
    "trades_summary_json": json.dumps(trades_summary),
    "source": "cli",
    "created_at": datetime.now(timezone.utc).isoformat(),
}
await db.insert_backtest_run(backtest_data)
```

**Fonctions utilitaires** (~25 lignes) :

```python
def _subsample_equity(
    curve: list[float], timestamps: list[datetime], max_points: int = 500,
) -> tuple[list[float], list[str]]:
    """Sous-échantillonne l'equity curve pour le stockage DB."""
    n = len(curve)
    if n <= max_points:
        return [round(v, 2) for v in curve], [t.isoformat() for t in timestamps]
    step = max(1, n // max_points)
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    return (
        [round(curve[i], 2) for i in indices],
        [timestamps[i].isoformat() for i in indices],
    )


def _trades_to_summary(trades: list) -> list[dict]:
    """Extrait un résumé compact de chaque trade pour le stockage DB."""
    return [
        {
            "entry_time": t.entry_time.isoformat(),
            "exit_time": t.exit_time.isoformat(),
            "direction": t.direction.value,
            "net_pnl": round(t.net_pnl, 2),
            "regime": t.market_regime.value if t.market_regime else None,
        }
        for t in trades
    ]
```

**Impact** : Le CLI fonctionne exactement comme avant (même output console/JSON), mais les résultats sont aussi écrits en DB.

### 2.3 `scripts/optimize.py` — Persistance en DB + progression

**Modifications** (~80 lignes ajoutées) :

Le mécanisme central est un **`ProgressCallback`** passé à `WalkForwardOptimizer.optimize()` :

```python
class ProgressCallback:
    """Callback pour mettre à jour la progression en DB et notifier le WS."""

    def __init__(self, db: Database, run_id: str, broadcaster=None):
        self._db = db
        self._run_id = run_id
        self._broadcaster = broadcaster  # None en CLI, ConnectionManager en API (Sprint 9)

    async def update(self, pct: float, phase: str, detail: str) -> None:
        await self._db.update_optimization_progress(
            self._run_id, pct, phase, detail,
        )
        # Sprint 9 : si broadcaster disponible (mode API), pusher la progression
        if self._broadcaster:
            await self._broadcaster.broadcast({
                "type": "optimization_progress",
                "run_id": self._run_id,
                "progress_pct": pct,
                "phase": phase,
                "detail": detail,
                "status": "running",
            })
```

**Flux modifié dans `run_optimization()`** :

```python
async def run_optimization(strategy_name, symbol, ..., db=None, progress_cb=None):
    run_id = str(uuid.uuid4())

    # 1. Créer l'enregistrement DB (status=running)
    await db.insert_optimization_run({
        "run_id": run_id,
        "strategy": strategy_name,
        "symbol": symbol,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source": "cli",
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    try:
        # 2. WFO (progress_cb appelé par l'optimizer)
        wfo = await optimizer.optimize(strategy_name, symbol, progress_cb=progress_cb)

        # 3. Overfitting
        if progress_cb:
            await progress_cb.update(70, "overfitting", "Monte Carlo + DSR + stabilité")
        overfit = detector.full_analysis(...)

        # 4. Validation
        if progress_cb:
            await progress_cb.update(85, "validation", "Backtest Bitget 90j + bootstrap CI")
        validation = await validate_on_bitget(...)

        # 5. Build report + persist
        report = build_final_report(wfo, overfit, validation)
        save_report(report)  # JSON existant (backward compat)

        await db.update_optimization_result(run_id, _report_to_db_dict(report))
        await db.insert_optimization_windows(run_id, _windows_to_db(wfo.windows))

        if progress_cb:
            await progress_cb.update(100, "done", f"Grade {report.grade}")

    except Exception as e:
        await db.update_optimization_result(run_id, {
            "status": "failed",
            "error_message": str(e),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        raise

    return report
```

**`_report_to_db_dict(report: FinalReport)`** construit le dict pour `update_optimization_result()`. Mapping exact depuis les champs de `FinalReport` (défini dans `backend/optimization/report.py`) :

```python
def _report_to_db_dict(report: FinalReport) -> dict:
    """Convertit un FinalReport en dict pour update_optimization_result()."""
    return {
        "status": "completed",
        "grade": report.grade,
        "live_eligible": 1 if report.live_eligible else 0,
        "recommended_params_json": json.dumps(report.recommended_params),
        # WFO summary (colonnes plates)
        "wfo_n_windows": report.wfo_n_windows,
        "wfo_avg_is_sharpe": round(report.wfo_avg_is_sharpe, 4),
        "wfo_avg_oos_sharpe": round(report.wfo_avg_oos_sharpe, 4),
        "oos_is_ratio": round(report.oos_is_ratio, 4),
        "wfo_consistency_rate": round(report.wfo_consistency_rate, 4),
        "n_distinct_combos": report.n_distinct_combos,
        # Overfitting + validation → details_json
        "details_json": json.dumps({
            "overfitting": {
                "mc_p_value": round(report.mc_p_value, 4),
                "mc_significant": report.mc_significant,
                "dsr": round(report.dsr, 4),
                "dsr_max_expected_sharpe": round(report.dsr_max_expected_sharpe, 4),
                "stability": round(report.stability, 4),
                "convergence": round(report.convergence, 4) if report.convergence is not None else None,
                "cliff_params": report.cliff_params,
                "divergent_params": report.divergent_params,
            },
            "validation": {
                "bitget_sharpe": round(report.validation.bitget_sharpe, 4),
                "bitget_net_return_pct": round(report.validation.bitget_net_return_pct, 2),
                "bitget_trades": report.validation.bitget_trades,
                "bitget_sharpe_ci_low": round(report.validation.bitget_sharpe_ci_low, 4),
                "bitget_sharpe_ci_high": round(report.validation.bitget_sharpe_ci_high, 4),
                "transfer_ratio": round(report.validation.transfer_ratio, 4),
                "volume_warning": report.validation.volume_warning,
                "volume_warning_detail": report.validation.volume_warning_detail,
            },
        }),
        # Métadonnées
        "warnings_json": json.dumps(report.warnings) if report.warnings else None,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
```

**Champs `FinalReport` non mappés** (intentionnel) : `strategy_name`, `symbol`, `timestamp` sont déjà dans la row initiale (`insert_optimization_run`). `bitget_transfer` est identique à `validation.transfer_ratio`. `oos_is_ratio` est une colonne plate ET dans le calcul du grade — pas de duplication dans `details_json`.

### 2.4 `backend/optimization/walk_forward.py` — Support progression

**Modifications** (~20 lignes) :

Ajouter un paramètre `progress_cb` à `optimize()` et appeler le callback après chaque fenêtre :

```python
async def optimize(
    self,
    strategy_name: str,
    symbol: str,
    ...,
    progress_cb=None,  # NOUVEAU
) -> WFOResult:
    ...
    for w_idx, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
        ...
        # Après chaque fenêtre, notifier la progression
        if progress_cb:
            pct = (w_idx + 1) / len(windows) * 65  # WFO = 0-65% du total
            await progress_cb.update(
                pct, "wfo",
                f"Fenêtre {w_idx + 1}/{len(windows)}"
            )
```

Le poids des phases dans la progression totale :
- WFO : 0% → 65%
- Overfitting : 65% → 80%
- Validation Bitget : 80% → 95%
- Finalisation : 95% → 100%

### 2.5 `frontend/src/hooks/useApi.js` — Support interval=null

**Bug actuel** : `setInterval(fetchData, 0)` cause un polling infini (~4ms). Le `BacktestDetail` a besoin d'un fetch unique (pas de polling).

**Modification** (~3 lignes) :

```javascript
useEffect(() => {
    fetchData()
    if (!interval) return  // NOUVEAU : interval=null ou 0 → pas de polling
    const id = setInterval(fetchData, interval)
    return () => clearInterval(id)
}, [fetchData, interval])
```

Le hook retourne aussi une fonction `refetch` pour les composants qui veulent rafraîchir manuellement :

```javascript
return { data, error, loading, refetch: fetchData }
```

---

## 3. Nouveaux endpoints API (lecture seule)

**Pas de POST ni DELETE dans ce sprint.** Le lancement de backtests/optimisations se fait exclusivement via CLI. Le dashboard est en consultation uniquement. Le lancement depuis l'UI est prévu pour le Sprint 9 avec un vrai job runner (gestion d'erreur, limite de concurrence, cleanup).

### 3.1 Fichier `backend/api/backtest_routes.py` (nouveau)

```python
router = APIRouter(prefix="/api/backtest", tags=["backtest"])
```

| Méthode | Route | Params | Réponse |
|---------|-------|--------|---------|
| `GET` | `/api/backtest/runs` | `?strategy=&symbol=&limit=50&offset=0` | `[{run_id, strategy, symbol, net_return_pct, sharpe_ratio, total_trades, win_rate, created_at, ...}]` |
| `GET` | `/api/backtest/runs/{run_id}` | — | `{...toutes les colonnes, equity_curve, regime_stats, trades_summary}` |

L'endpoint list ne retourne PAS `equity_curve_json`, `trades_summary_json`, `strategy_params_json` ni `regime_stats_json` (trop volumineux) — uniquement les métriques scalaires. L'endpoint detail retourne tout.

### 3.2 Fichier `backend/api/optimization_routes.py` (nouveau)

```python
router = APIRouter(prefix="/api/optimization", tags=["optimization"])
```

| Méthode | Route | Params | Réponse |
|---------|-------|--------|---------|
| `GET` | `/api/optimization/runs` | `?strategy=&symbol=&status=&limit=50&offset=0` | `[{run_id, strategy, symbol, status, grade, oos_is_ratio, wfo_avg_oos_sharpe, progress_pct, started_at, ...}]` |
| `GET` | `/api/optimization/runs/{run_id}` | — | `{...toutes les colonnes, details_json parsé}` |
| `GET` | `/api/optimization/runs/{run_id}/windows` | — | `[{window_index, is_sharpe, oos_sharpe, ...}]` |

L'endpoint list ne retourne PAS `details_json` ni `warnings_json` (potentiellement volumineux) — uniquement les colonnes scalaires (`grade`, `status`, `oos_is_ratio`, `wfo_avg_oos_sharpe`, etc.). Seul l'endpoint detail retourne tout.

L'endpoint detail parse `details_json` et le retourne comme objet imbriqué (pas de JSON dans du JSON côté frontend).

### 3.3 Modifications `backend/api/server.py`

```python
from backend.api.backtest_routes import router as backtest_router
from backend.api.optimization_routes import router as optimization_router

app.include_router(backtest_router)
app.include_router(optimization_router)
```

2 lignes d'import + 2 lignes d'include.

---

## 4. Message WebSocket pour progression

### 4.1 Modification `backend/api/websocket_routes.py`

Ajouter la progression de l'optimisation active dans la boucle `/ws/live` :

```python
# Dans live_feed(), après le bloc executor :
db = getattr(websocket.app.state, "db", None)
if db is not None:
    active_optim = await db.get_active_optimization()
    if active_optim:
        data["optimization_progress"] = {
            "run_id": active_optim["run_id"],
            "strategy": active_optim["strategy"],
            "symbol": active_optim["symbol"],
            "progress_pct": active_optim["progress_pct"],
            "phase": active_optim["progress_phase"],
            "detail": active_optim["progress_detail"],
            "status": active_optim["status"],
        }
```

**Payload WebSocket** (ajouté au message `type: "update"` existant) :

```json
{
  "type": "update",
  "strategies": [...],
  "ranking": [...],
  "prices": {...},
  "optimization_progress": {
    "run_id": "abc-123",
    "strategy": "vwap_rsi",
    "symbol": "BTC/USDT",
    "progress_pct": 42.5,
    "phase": "wfo",
    "detail": "Fenêtre 9/20",
    "status": "running"
  }
}
```

Le champ `optimization_progress` est absent (pas `null`) quand il n'y a aucune optimisation en cours. Le frontend vérifie `wsData?.optimization_progress` avant de rendre le widget.

**Mécanisme** : Le WS loop lit la DB toutes les 3 secondes (même fréquence que le reste). La progression est écrite en DB par le `ProgressCallback` dans le process CLI. Pas besoin de broadcast direct dans ce sprint puisque les optimisations sont lancées via CLI (pas depuis le serveur API).

---

## 5. Composants frontend

### 5.1 Dépendance Recharts

```bash
cd frontend && npm install recharts
```

Recharts est nécessaire pour les equity curves longues (720j+ = des milliers de points). Les sparklines SVG inline existantes restent pour le Scanner (60 points), mais les courbes de backtest utilisent Recharts (`AreaChart`, `ResponsiveContainer`, `Tooltip`, `XAxis`, `YAxis`).

### 5.2 Architecture des composants

**Nouveau tab "Backtest"** dans `App.jsx` avec sous-tabs internes :

```
Tab "Backtest" (dans Header, à côté de Scanner/Heatmap/Risque)
├── Sous-tab "Résultats" (BacktestPanel)
│   ├── BacktestTable           — liste des runs passés
│   └── BacktestDetail          — détail extensible (equity curve + métriques)
└── Sous-tab "Optimisation" (OptimizationPanel)
    ├── OptimizationTable       — liste des runs avec grade badge
    └── OptimizationDetail      — détail extensible (WFO + overfitting + validation)

Sidebar (toujours visible)
└── OptimizationProgress        — widget compact quand optimisation active
```

Pas de Launcher dans ce sprint — les backtests/optimisations sont lancés via CLI.

### 5.3 `App.jsx` — Modifications

```jsx
import BacktestPanel from './components/BacktestPanel'
import OptimizationPanel from './components/OptimizationPanel'
import OptimizationProgress from './components/OptimizationProgress'

const TABS = [
  { id: 'scanner', label: 'Scanner' },
  { id: 'heatmap', label: 'Heatmap' },
  { id: 'risk', label: 'Risque' },
  { id: 'backtest', label: 'Backtest' },  // NOUVEAU
]

// Sous-tabs pour le tab Backtest
const [backtestSubTab, setBacktestSubTab] = useState('results')

// Dans le JSX content :
{activeTab === 'backtest' && backtestSubTab === 'results' && <BacktestPanel />}
{activeTab === 'backtest' && backtestSubTab === 'optimization' && <OptimizationPanel />}

// Dans la sidebar, avant ExecutorPanel :
<OptimizationProgress wsData={lastMessage} />
```

### 5.4 `BacktestPanel.jsx` (nouveau, ~80 lignes)

**Container** avec sous-tabs + `BacktestTable` + `BacktestDetail`.

```jsx
// Props : aucune (self-contained, utilise useApi)
// State : selectedRunId

export default function BacktestPanel() {
  const { data: runs, refetch } = useApi('/api/backtest/runs?limit=20', 60000) // polling 60s
  const [selectedRunId, setSelectedRunId] = useState(null)

  return (
    <div className="backtest-panel">
      <div className="panel-header">
        <h3>Résultats de backtest</h3>
        <button className="btn-refresh" onClick={refetch}>Rafraîchir</button>
      </div>
      <BacktestTable runs={runs} onSelect={setSelectedRunId} selectedId={selectedRunId} />
      {selectedRunId && <BacktestDetail runId={selectedRunId} />}
    </div>
  )
}
```

**Polling à 60 secondes** + bouton "Rafraîchir" manuel. La liste des backtests ne change que quand on en lance un nouveau (via CLI) — 60s est suffisant.

### 5.5 `BacktestTable.jsx` (nouveau, ~100 lignes)

Table des résultats de backtests passés, triée par date décroissante.

```jsx
// Props : runs: array, onSelect: (runId) => void, selectedId: string | null

// Colonnes :
// | Date | Stratégie | Symbole | Trades | Win% | Net P&L | Sharpe | Max DD |
// Ligne cliquable → sélectionne le run pour afficher le détail
// Ligne sélectionnée = highlight (bg-card-hover)
// Badge couleur sur Net P&L : vert si positif, rouge si négatif
```

### 5.6 `BacktestDetail.jsx` (nouveau, ~180 lignes)

Vue détaillée d'un backtest sélectionné. Charge les données via `GET /api/backtest/runs/{runId}`.

```jsx
// Props : runId: string
// Données : useApi(`/api/backtest/runs/${runId}`, null)  // fetch unique, pas de polling

// Layout :
// ┌────────────────────────────────────────────────┐
// │  Equity Curve (Recharts AreaChart)              │
// │  - Ligne verte si net_pnl > 0, rouge sinon     │
// │  - Aire sous la courbe avec opacité             │
// │  - Tooltip : date + valeur                      │
// │  - Ligne horizontale = capital initial          │
// ├────────────┬───────────────────────────────────┤
// │ Performance│  Risque          │ Frais            │
// │ Trades: 85 │  Sharpe: 1.42    │ Gross: +$3,200   │
// │ Win: 58%   │  Sortino: 1.87   │ Fees: -$890      │
// │ Net: +$2,4 │  MaxDD: -4.2%    │ Slip: -$120      │
// │ PF: 1.62   │  DD dur: 12j     │ Net: +$2,190     │
// ├────────────┴───────────────────────────────────┤
// │  Breakdown par régime (barres horizontales)     │
// │  RANGING:  45 trades, 62% win, +$2,457          │
// │  TRENDING: 28 trades, 42% win, -$267            │
// │  VOLATILE: 12 trades, 50% win, +$0              │
// └────────────────────────────────────────────────┘
```

**Equity curve avec Recharts** :

```jsx
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

<ResponsiveContainer width="100%" height={250}>
  <AreaChart data={equityData}>
    <XAxis dataKey="date" tick={{ fontSize: 10 }} />
    <YAxis domain={['dataMin - 100', 'dataMax + 100']} tick={{ fontSize: 10 }} />
    <ReferenceLine y={initialCapital} stroke="var(--text-muted)" strokeDasharray="3 3" />
    <Tooltip />
    <Area
      type="monotone"
      dataKey="equity"
      stroke={netPnl >= 0 ? 'var(--accent)' : 'var(--red)'}
      fill={netPnl >= 0 ? 'var(--accent-dim)' : 'var(--red-dim)'}
    />
  </AreaChart>
</ResponsiveContainer>
```

### 5.7 `OptimizationPanel.jsx` (nouveau, ~80 lignes)

Container avec `OptimizationTable` + `OptimizationDetail`.

```jsx
// Props : aucune
// State : selectedRunId

export default function OptimizationPanel() {
  const { data: runs, refetch } = useApi('/api/optimization/runs?limit=20', 60000) // polling 60s
  const [selectedRunId, setSelectedRunId] = useState(null)

  return (
    <div className="optimization-panel">
      <div className="panel-header">
        <h3>Résultats d'optimisation</h3>
        <button className="btn-refresh" onClick={refetch}>Rafraîchir</button>
      </div>
      <OptimizationTable runs={runs} onSelect={setSelectedRunId} selectedId={selectedRunId} />
      {selectedRunId && <OptimizationDetail runId={selectedRunId} />}
    </div>
  )
}
```

### 5.8 `OptimizationTable.jsx` (nouveau, ~120 lignes)

```jsx
// Colonnes :
// | Date | Stratégie | Symbole | Grade | Status | OOS/IS | Sharpe OOS | Progression |

// Grade badge : cercle coloré (A=vert, B=bleu, C=jaune, D=orange, F=rouge)
// Status badge : "running" animé (pulse), "completed", "failed" (rouge)
// Progression : mini barre de progression si status=running
```

### 5.9 `OptimizationDetail.jsx` (nouveau, ~220 lignes)

Vue la plus riche. Charge les données via `GET /api/optimization/runs/{runId}` et `/runs/{runId}/windows`.

```jsx
// Props : runId: string
// Données :
//   useApi(`/api/optimization/runs/${runId}`, null)  // fetch unique
//   useApi(`/api/optimization/runs/${runId}/windows`, null)  // fetch unique

// Layout :
// ┌──────────────────────────────────────────────────┐
// │  GRADE A  │ VWAP_RSI × BTC/USDT  │ LIVE ✓       │
// ├──────────────────────────────────────────────────┤
// │  Walk-Forward (Recharts BarChart)                 │
// │  Barres IS (gris) + barres OOS (couleur)          │
// │  par fenêtre, X = window index, Y = Sharpe        │
// ├──────────┬────────────────────────────────────────┤
// │ WFO      │ Overfitting         │ Validation        │
// │ IS: 1.82 │ MC p: 0.012 ✓       │ Bitget: 0.71     │
// │ OOS:0.94 │ DSR: 0.96 ✓         │ CI: [0.32-1.14]  │
// │ Ratio:52%│ Stab: 0.84 ✓        │ Transfer: 0.76   │
// │ Cons:75% │ Conv: 0.78 ✓        │ Vol warn: Non    │
// ├──────────┴────────────────────────────────────────┤
// │  Paramètres recommandés (table key: value)        │
// ├───────────────────────────────────────────────────┤
// │  Warnings (si présents)                           │
// └───────────────────────────────────────────────────┘
```

**WFO BarChart** :

```jsx
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

// Data : [{window: 1, is_sharpe: 1.82, oos_sharpe: 0.94}, ...]
<ResponsiveContainer width="100%" height={200}>
  <BarChart data={windowsData}>
    <XAxis dataKey="window" tick={{ fontSize: 10 }} />
    <YAxis tick={{ fontSize: 10 }} />
    <ReferenceLine y={0} stroke="var(--text-muted)" />
    <Tooltip />
    <Bar dataKey="is_sharpe" fill="var(--text-dim)" name="IS Sharpe" />
    <Bar dataKey="oos_sharpe" fill="var(--accent)" name="OOS Sharpe" />
  </BarChart>
</ResponsiveContainer>
```

**Overfitting indicators** : barres horizontales avec seuils visuels (vert si MC p < 0.05, DSR > 0.95, stabilité > 0.80, etc.). Les données viennent du `details_json` déjà parsé par l'API.

### 5.10 `GradeBadge.jsx` (nouveau, ~20 lignes)

Petit composant réutilisable pour le badge de grade.

```jsx
const GRADE_COLORS = {
  A: 'var(--accent)', B: 'var(--blue)', C: 'var(--yellow)',
  D: 'var(--orange)', F: 'var(--red)',
}

export default function GradeBadge({ grade }) {
  return (
    <span className="grade-badge" style={{ '--grade-color': GRADE_COLORS[grade] || 'var(--text-muted)' }}>
      {grade}
    </span>
  )
}
```

### 5.11 `OptimizationProgress.jsx` (nouveau, ~60 lignes)

Widget compact dans la sidebar. Visible uniquement quand une optimisation est en cours.

```jsx
// Props : wsData
// Données : wsData?.optimization_progress

export default function OptimizationProgress({ wsData }) {
  const progress = wsData?.optimization_progress
  if (!progress || progress.status !== 'running') return null

  return (
    <div className="card optimization-progress">
      <div className="card-title">Optimisation en cours</div>
      <div className="optim-info">
        <span>{progress.strategy.toUpperCase()}</span>
        <span>{progress.symbol}</span>
      </div>
      <div className="progress-bar">
        <div
          className="progress-bar__fill"
          style={{ width: `${progress.progress_pct}%` }}
        />
      </div>
      <div className="optim-detail">
        {progress.detail} — {Math.round(progress.progress_pct)}%
      </div>
    </div>
  )
}
```

### 5.12 CSS — Ajouts dans `styles.css`

Classes ajoutées (~70 lignes) :

```css
/* Backtest & Optimization Panels */
.backtest-panel, .optimization-panel { display: flex; flex-direction: column; gap: 12px; }
.panel-header { display: flex; justify-content: space-between; align-items: center; }
.panel-header h3 { font-size: 14px; font-weight: 600; }
.btn-refresh {
  background: none; border: 1px solid var(--border); color: var(--text-secondary);
  border-radius: var(--radius-sm); padding: 4px 10px; cursor: pointer; font-size: 11px;
}
.btn-refresh:hover { color: var(--text-primary); border-color: var(--text-secondary); }

/* Tables */
.bt-table { width: 100%; border-collapse: collapse; }
.bt-table th { text-align: left; color: var(--text-secondary); font-weight: 500; padding: 8px; border-bottom: 1px solid var(--border); font-size: 11px; }
.bt-table td { padding: 8px; border-bottom: 1px solid var(--border); font-size: 12px; }
.bt-table tr { cursor: pointer; }
.bt-table tr:hover { background: var(--bg-card-hover); }
.bt-table tr.selected { background: var(--bg-card-hover); border-left: 2px solid var(--accent); }

/* Sub-tabs */
.sub-tabs { display: flex; gap: 4px; margin-bottom: 12px; }
.sub-tab { background: none; border: 1px solid var(--border); color: var(--text-secondary);
  border-radius: var(--radius-sm); padding: 4px 12px; cursor: pointer; font-size: 12px; }
.sub-tab.active { background: var(--accent-dim); color: var(--accent); border-color: var(--accent); }

/* Grade badge */
.grade-badge {
  display: inline-flex; align-items: center; justify-content: center;
  width: 28px; height: 28px; border-radius: 50%;
  background: var(--grade-color); color: var(--bg-primary);
  font-weight: 700; font-size: 13px;
}

/* Progress bar */
.progress-bar { width: 100%; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
.progress-bar__fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width 0.5s ease; }

/* Optimization progress sidebar widget */
.optimization-progress .optim-info { display: flex; justify-content: space-between; margin-bottom: 6px; }
.optimization-progress .optim-detail { font-size: 11px; color: var(--text-secondary); margin-top: 4px; }

/* Backtest detail metrics grid */
.bt-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 12px; }
.bt-metric-group h4 { color: var(--text-secondary); font-size: 11px; text-transform: uppercase; margin-bottom: 6px; }
.bt-metric-row { display: flex; justify-content: space-between; padding: 2px 0; }
.bt-metric-row .label { color: var(--text-secondary); }
.bt-metric-row .value { font-family: var(--font-mono); }

/* Regime breakdown bars */
.regime-bar { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
.regime-bar__label { width: 100px; font-size: 11px; color: var(--text-secondary); }
.regime-bar__fill { height: 8px; border-radius: 4px; }
.regime-bar__stats { font-size: 11px; font-family: var(--font-mono); }
```

---

## 6. Tests

### 6.1 Fichier `tests/test_backtest_dashboard.py` (nouveau, ~28 tests, ~580 lignes)

**Tests DB — backtest_runs** (~5 tests) :

1. `test_insert_backtest_run` — Insertion et lecture correcte (tous les champs)
2. `test_get_backtest_runs_filtered_by_strategy` — Filtre par stratégie
3. `test_get_backtest_runs_filtered_by_symbol` — Filtre par symbole
4. `test_get_backtest_runs_pagination` — limit/offset fonctionnent
5. `test_get_backtest_run_not_found` — run_id inexistant → None

**Tests DB — optimization_runs** (~7 tests) :

6. `test_insert_optimization_run` — Insertion status=running
7. `test_update_optimization_progress` — Progression mise à jour
8. `test_update_optimization_result_with_details_json` — Résultat final (grade, details_json parsé)
9. `test_get_active_optimization` — Retourne le run en cours
10. `test_get_active_optimization_none_when_completed` — Aucun run running → None
11. `test_get_optimization_runs_filtered_by_status` — Filtre completed/running/failed
12. `test_insert_and_get_optimization_windows` — Fenêtres WFO persistées

**Tests API — backtest** (~4 tests) :

13. `test_api_backtest_runs_empty` — GET /api/backtest/runs quand DB vide → `[]`
14. `test_api_backtest_runs_list` — Insertion + GET → résultat correct (sans equity_curve)
15. `test_api_backtest_run_detail` — GET /api/backtest/runs/{id} → détail complet (avec equity_curve)
16. `test_api_backtest_run_not_found` — GET /api/backtest/runs/xxx → 404

**Tests API — optimization** (~4 tests) :

17. `test_api_optimization_runs_empty` — GET /api/optimization/runs → `[]`
18. `test_api_optimization_runs_list` — Insertion + GET → résultat
19. `test_api_optimization_run_detail_parses_details_json` — GET → details_json retourné comme objet
20. `test_api_optimization_windows` — GET /api/optimization/runs/{id}/windows → fenêtres

**Tests persistance CLI** (~5 tests) :

21. `test_run_backtest_persists_to_db` — run_backtest écrit en DB
22. `test_subsample_equity_500_points` — Sous-échantillonnage correct
23. `test_subsample_equity_short_unchanged` — Courbe < 500 pts non modifiée
24. `test_trades_to_summary` — Résumé compact correct
25. `test_optimization_persists_to_db` — Optimisation écrit en DB avec details_json

**Tests progression** (~3 tests) :

26. `test_progress_callback_updates_db` — ProgressCallback écrit en DB
27. `test_progress_callback_no_broadcast_when_no_broadcaster` — Pas d'erreur si broadcaster=None
28. `test_ws_includes_optimization_progress` — /ws/live inclut la progression active

---

## 7. Ordre d'implémentation par phases

### Phase 1 — DB & persistance backend (fondations)

1. `backend/core/database.py` — 3 nouvelles tables + méthodes CRUD
2. `frontend/src/hooks/useApi.js` — Support `interval=null` + `refetch`
3. `scripts/run_backtest.py` — Persistance après calcul des métriques + `_subsample_equity` + `_trades_to_summary`
4. `scripts/optimize.py` — ProgressCallback + persistance en DB + `_report_to_db_dict`
5. `backend/optimization/walk_forward.py` — Paramètre `progress_cb`
6. Tests DB + tests persistance CLI

**Dépendances** : aucune — ne modifie pas le comportement existant.

### Phase 2 — API REST (lecture seule)

7. `backend/api/backtest_routes.py` — 2 endpoints GET
8. `backend/api/optimization_routes.py` — 3 endpoints GET
9. `backend/api/server.py` — Enregistrer les 2 nouveaux routers
10. Tests API

**Dépendances** : Phase 1 (DB et méthodes CRUD).

### Phase 3 — WebSocket progression

11. `backend/api/websocket_routes.py` — Champ `optimization_progress`
12. Tests WS

**Dépendances** : Phase 1 (DB `get_active_optimization`).

### Phase 4 — Frontend Backtest

13. `npm install recharts` dans `frontend/`
14. `App.jsx` — Nouveau tab + sous-tabs
15. `BacktestPanel.jsx` — Container avec polling 60s + bouton refresh
16. `BacktestTable.jsx` — Liste
17. `BacktestDetail.jsx` — Détail + equity curve Recharts
18. `styles.css` — Classes backtest

**Dépendances** : Phase 2 (API endpoints).

### Phase 5 — Frontend Optimisation

19. `OptimizationPanel.jsx` — Container avec polling 60s + bouton refresh
20. `OptimizationTable.jsx` — Liste avec badges
21. `OptimizationDetail.jsx` — Détail (WFO chart + overfitting + validation)
22. `GradeBadge.jsx` — Composant réutilisable
23. `OptimizationProgress.jsx` — Widget sidebar
24. `styles.css` — Classes optimisation + progression

**Dépendances** : Phase 3 (WS progression) + Phase 4 (Recharts déjà installé).

---

## 8. Estimation en lignes par fichier

### Fichiers modifiés

| Fichier | Lignes ajoutées | Modification |
|---------|---------------:|-------------|
| `backend/core/database.py` | ~170 | 3 tables + 11 méthodes CRUD |
| `scripts/run_backtest.py` | ~60 | Persistance DB + subsample + trades_summary |
| `scripts/optimize.py` | ~80 | ProgressCallback + insert/update DB + _report_to_db_dict |
| `backend/optimization/walk_forward.py` | ~15 | Param `progress_cb` + appels callback |
| `backend/api/server.py` | ~4 | 2 imports + 2 include_router |
| `backend/api/websocket_routes.py` | ~12 | Champ optimization_progress dans boucle WS |
| `frontend/src/hooks/useApi.js` | ~5 | Support interval=null + refetch |
| `frontend/src/App.jsx` | ~20 | Tab Backtest + sous-tabs + imports |
| `frontend/src/styles.css` | ~70 | Classes backtest, optimization, progress |

### Fichiers créés

| Fichier | Lignes | Contenu |
|---------|-------:|---------|
| `backend/api/backtest_routes.py` | ~60 | 2 endpoints GET (list, detail) |
| `backend/api/optimization_routes.py` | ~80 | 3 endpoints GET (list, detail, windows) |
| `frontend/src/components/BacktestPanel.jsx` | ~80 | Container + refresh button |
| `frontend/src/components/BacktestTable.jsx` | ~100 | Table résultats |
| `frontend/src/components/BacktestDetail.jsx` | ~180 | Détail + equity Recharts + métriques |
| `frontend/src/components/OptimizationPanel.jsx` | ~80 | Container + refresh button |
| `frontend/src/components/OptimizationTable.jsx` | ~120 | Table avec grade badges |
| `frontend/src/components/OptimizationDetail.jsx` | ~220 | WFO chart + overfitting + validation |
| `frontend/src/components/GradeBadge.jsx` | ~20 | Badge grade A-F |
| `frontend/src/components/OptimizationProgress.jsx` | ~60 | Widget sidebar progression |
| `tests/test_backtest_dashboard.py` | ~580 | 28 tests |

### Récapitulatif

| Catégorie | Lignes |
|-----------|-------:|
| Backend modifié | ~345 |
| Backend créé | ~140 |
| Frontend créé | ~860 |
| Frontend modifié | ~95 |
| Tests | ~580 |
| **Total** | **~2 020** |

---

## Contraintes & risques

1. **Taille equity curve en DB** : Sous-échantillonné à 500 points max (~2 Ko JSON). Un backtest de 720j en 5m = 207k points → 500 points suffisent pour le rendu Recharts.

2. **`trades_summary_json`** : Résumé compact (~100 bytes/trade). Pour 200 trades ≈ 20 Ko. Acceptable. Permet d'afficher des markers sur l'equity curve plus tard sans backfill.

3. **Optimisation longue** : Un WFO complet (5 assets × 20 fenêtres × 700 combos) prend ~20 min. Le `ProgressCallback` met à jour la DB après chaque fenêtre (~30s). Le WS loop lit la DB toutes les 3s et pousse la progression au frontend.

4. **Pas de lancement depuis le dashboard** : Les POST endpoints sont hors scope (Sprint 9). `asyncio.create_task()` pour un job CPU-bound est dangereux : pas de gestion d'erreur propre, pas de limite de concurrence, pas de cleanup si le serveur redémarre. Un vrai job runner est nécessaire.

5. **`details_json` flexible** : Les colonnes overfitting/validation qui changeront (ex: Kraken phase 2) sont dans un JSON unique. On filtre par `grade` et `status` (colonnes plates), pas par `mc_p_value`.

6. **Backward compat CLI** : Les scripts fonctionnent exactement comme avant (même output console/JSON). La persistance DB est ajoutée en plus. Les rapports JSON dans `data/optimization/` sont toujours générés.

7. **Recharts bundle size** : Recharts ajoute ~40 Ko gzippé au bundle. Les composants SVG inline existants (Spark, ScoreRing, SignalDots) ne changent pas.

8. **Migration DB** : Les 3 tables sont créées avec `IF NOT EXISTS` — pas de migration destructive.

9. **Refactoring `run_backtest.py`** : Le script actuel utilise `asyncio.run(_load_data(...))` puis fait le backtest de manière synchrone. On ne peut pas ajouter un second `asyncio.run()` pour la persistance DB. Le `run_backtest()` doit être refactoré pour englober load + backtest + persist dans un seul `asyncio.run(_run_backtest_async(...))`. Le backtest lui-même reste synchrone (`engine.run()` sans `await`).

10. **`useApi` avec interval=null** : Le hook actuel ne gère pas `interval=0` (polling infini). Correction nécessaire en Phase 1 : `if (!interval) return` avant le `setInterval`. Le hook retourne aussi `refetch` pour les composants qui veulent rafraîchir manuellement.

11. **Backtests WFO internes** : NON persistés dans `backtest_runs`. Seuls les backtests explicites (CLI `run_backtest`) y sont stockés. Les centaines de backtests intermédiaires du WFO sont des résultats jetables — les persister remplirait la table avec des milliers de rows inutiles.

---

## Hors scope (Sprint 9)

- `POST /api/backtest/run` — Lancement backtest depuis le dashboard
- `POST /api/optimization/run` — Lancement optimisation depuis le dashboard
- `DELETE /api/backtest/runs/{run_id}` — Suppression de runs
- `BacktestLauncher.jsx` — Formulaire de lancement
- `OptimizationLauncher.jsx` — Formulaire de lancement
- Job runner (TaskManager avec queue, status tracking, cleanup)
- Comparaison côte-à-côte de deux backtests

---

## Vérification finale

1. `uv run pytest` — tous les tests passent (330 existants + ~28 nouveaux = ~358)
2. `uv run python -m scripts.run_backtest --symbol BTC/USDT --days 90` — résultat console + persisté en DB
3. `uv run python -m scripts.optimize --strategy vwap_rsi --symbol BTC/USDT` — progression en DB, résultat persisté avec details_json
4. Dashboard → tab Backtest → liste des backtests passés → clic → equity curve Recharts
5. Dashboard → tab Backtest → sous-tab Optimisation → liste avec grades → clic → détail WFO + overfitting
6. Bouton "Rafraîchir" → actualise la liste
7. Lancer une optimisation en CLI → progression visible en temps réel dans la sidebar du dashboard
8. WebSocket `/ws/live` → champ `optimization_progress` présent pendant l'optimisation
9. `useApi('/api/backtest/runs/xxx', null)` → fetch unique, pas de polling
10. Optimisation terminée → `OptimizationProgress` disparaît de la sidebar
