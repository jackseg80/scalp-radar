# Sprint 61 — Regime Monitor

## Contexte

Grid_atr performe bien en volatilité mais souffre en range calme. L'objectif est un outil d'aide à la décision pour ajuster manuellement le leverage via une alerte Telegram quotidienne + un widget frontend. **Aucune automatisation du leverage.**

Il existe déjà un système de régime dans `backend/regime/` (EMAATRDetector, Sprint 50b) et `backend/backtesting/metrics.py` (`_classify_regime`). Le nouveau module **réutilise** `_classify_regime` (comme demandé) et `atr()` existant — pas de duplication.

---

## Fichiers à créer (5)

### 1. `backend/regime/regime_monitor.py` — Module principal

- **`RegimeSnapshot`** dataclass : regime, regime_days, btc_atr_14d_pct, btc_change_30d_pct, volatility_level, suggested_leverage, timestamp
- **`_classify_volatility(atr_pct) -> (str, int)`** : seuils ATR → volatility_level + leverage suggéré
- **`compute_regime_snapshot(db, exchange="bitget") -> RegimeSnapshot`** :
  1. Charge BTC/USDT 1h (45 derniers jours) via `db.get_candles()`
  2. Resample 1h → daily (group par date, OHLCV agrégés)
  3. Appelle `_classify_regime(daily_candles[-30:])` de `metrics.py` pour le label
  4. Calcule regime_days : re-classify fenêtres glissantes 30j pour les 15 derniers jours, compte les jours consécutifs avec le même label depuis la fin
  5. Calcule ATR(14) daily via `backend.core.indicators.atr()`, convertit en % du prix
  6. Calcule return 30j
  7. Retourne `RegimeSnapshot`
- **`RegimeMonitor`** classe (pattern identique à `WeeklyReporter` de [weekly_reporter.py](backend/alerts/weekly_reporter.py)) :
  - `__init__(telegram, db)` — stocke `_latest: RegimeSnapshot | None` et `_history: list[dict]` (max 30)
  - `start()` — calcule snapshot initial immédiatement + lance `_loop()`
  - `_loop()` — attend 00:05 UTC, compute snapshot, envoie Telegram, sleep anti-doublons
  - `_format_telegram(snapshot)` — message formaté HTML avec emojis par régime
  - `stop()` — `_running = False` + cancel task

### 2. `backend/api/regime_routes.py` — Endpoints API

- `GET /api/regime/snapshot` → `{"snapshot": {...}}` depuis `monitor.latest`
- `GET /api/regime/history?days=30` → `{"history": [...], "count": N}` depuis `monitor.history`
- Pattern identique aux autres routes (router avec prefix `/api/regime`)

### 3. `frontend/src/components/RegimeWidget.jsx` — Widget sidebar

- Utilise `useApi('/api/regime/snapshot', 60000)` (polling 60s)
- Header : dot coloré + label régime + jours consécutifs
- Métriques : BTC 30j, ATR 14j, Leverage suggéré (grille 3 colonnes)
- Barre ATR : track + fill gradient (vert→jaune→rouge), labels LOW/MED/HIGH
- Pas de sparkline dans le widget sidebar (trop compact) — les données history sont disponibles via l'API pour une extension future

### 4. `frontend/src/components/RegimeWidget.css` — Styles

- CSS variables du projet (--accent, --red, --yellow, --text-dim, --font-mono)
- Barre ATR avec gradient, labels typographiques

### 5. `tests/test_regime_monitor.py` — 8 tests

- `test_classify_volatility_low` — ATR < 2% → LOW, 3x
- `test_classify_volatility_medium` — ATR 2-3% → MEDIUM, 4x
- `test_classify_volatility_high` — ATR > 4% → HIGH, 6x
- `test_compute_snapshot_returns_valid` — snapshot valide avec mock DB
- `test_compute_snapshot_no_candles` — ValueError si pas de candles
- `test_monitor_start_stop` — start/stop sans erreur
- `test_monitor_telegram_format` — message contient infos clés
- `test_seconds_until_0005_positive` — toujours dans le futur, ≤ 24h

---

## Fichiers à modifier (2)

### 6. `backend/api/server.py` — 4 insertions

- **Import** (ligne ~31) : `from backend.api.regime_routes import router as regime_router`
- **Lifespan startup** (après WeeklyReporter, ~ligne 257) : créer `RegimeMonitor(telegram, db)`, `await start()`, stocker dans `app.state.regime_monitor`
- **Lifespan shutdown** (avant weekly_reporter.stop, ~ligne 270) : `await regime_monitor.stop()`
- **Router** (ligne ~339) : `app.include_router(regime_router)`

### 7. `frontend/src/App.jsx` — 2 insertions

- **Import** (~ligne 18) : `import RegimeWidget from './components/RegimeWidget'`
- **Sidebar** (après le `CollapsibleCard` "Executor", ~ligne 209) :
```jsx
<CollapsibleCard title="Regime BTC" defaultOpen={true} storageKey="regime">
  <RegimeWidget />
</CollapsibleCard>
```

---

## Ce qu'on NE touche PAS

- `config/risk.yaml`, `config/strategies.yaml` — pas de config YAML (horaire hardcodé comme `WeeklyReporter`)
- `backend/core/config.py` — pas de nouveau model Pydantic
- Le leverage réel, l'executor, le grid_position_manager
- Les régimes dans le portfolio backtest (séparés)

## Réutilisation de code existant

| Fonction | Fichier | Usage |
|----------|---------|-------|
| `_classify_regime(candles)` | [metrics.py:322-388](backend/backtesting/metrics.py#L322-L388) | Classification 4-class (crash/bull/bear/range) |
| `atr(highs, lows, closes, 14)` | [indicators.py:167-202](backend/core/indicators.py#L167-L202) | ATR(14) Wilder sur daily |
| `db.get_candles()` | [database.py](backend/core/database.py) | Chargement BTC/USDT 1h |
| `WeeklyReporter` pattern | [weekly_reporter.py:370-430](backend/alerts/weekly_reporter.py#L370-L430) | start/stop/loop scheduler |
| `Spark.jsx` | [Spark.jsx](frontend/src/components/Spark.jsx) | Disponible si sparkline ajoutée plus tard |
| `useApi` hook | [useApi.js](frontend/src/hooks/useApi.js) | Polling API |
| `CollapsibleCard` | [CollapsibleCard.jsx](frontend/src/components/CollapsibleCard.jsx) | Wrapper sidebar |

---

## Ordre d'implémentation

1. `backend/regime/regime_monitor.py` — logique pure, testable isolément
2. `tests/test_regime_monitor.py` — valider immédiatement
3. `backend/api/regime_routes.py` — endpoints API
4. `backend/api/server.py` — intégration lifespan + router
5. `frontend/src/components/RegimeWidget.css` — styles
6. `frontend/src/components/RegimeWidget.jsx` — composant
7. `frontend/src/App.jsx` — câblage sidebar

## Vérification

```bash
uv run pytest tests/test_regime_monitor.py -v
uv run pytest tests/ -x -q
```
