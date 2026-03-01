# Sprint 63b — Overlay Paper vs Live + Historique Alertes Telegram

## Contexte

Deux fonctionnalités complémentaires pour le suivi de production :
1. **Comparer visuellement** la performance paper vs live sur une même courbe normalisée
2. **Historiser** toutes les alertes Telegram envoyées (actuellement volatiles) avec filtrage par type/stratégie

---

## Erreurs et améliorations identifiées dans le plan initial

### Feature 1 : PaperLiveOverlay

| # | Problème | Correction |
|---|----------|------------|
| 1 | Plan dit `equity: [{timestamp, equity}]` pour le paper | L'endpoint réel retourne `equity: [{timestamp, capital, trade_pnl}]` — champ `capital`, pas `equity` |
| 2 | Plan utilise `days` pour le paper | L'endpoint paper utilise `since` (ISO8601), pas `days`. Frontend doit convertir : `new Date(Date.now() - days * 86400000).toISOString()` |
| 3 | Plan propose Recharts alors que toutes les equity curves prod utilisent du SVG custom | Recharts est disponible (`recharts@^3.7.0`) mais utilisé uniquement dans les `guides/`. Recharts reste le bon choix ici (2 courbes + tooltip + normalisation), c'est juste une inconsistance de style à noter |
| 4 | "inner join sur la date" | Les timestamps live (snapshots horaires) et paper (par trade, irréguliers) ne matchent pas. Utiliser un **union** avec `connectNulls` dans Recharts, pas un inner join |
| 5 | Live `equity` inclut unrealized_pnl, Paper `capital` = réalisé uniquement | Comparaison pommes vs oranges. Ajouter une note dans l'UI : "Live inclut les P&L non réalisés" |

### Feature 2 : Historique Alertes Telegram

| # | Problème | Correction |
|---|----------|------------|
| 1 | **Architecture critique** : plan modifie seulement `Notifier.notify_*` | Heartbeat, WeeklyReporter, RegimeMonitor appellent `telegram.send_message()` **directement**, sans passer par Notifier → ~40% des alertes non capturées |
| 2 | Plan dit modifier `_send_telegram_message()` | Cette méthode n'existe pas. C'est `TelegramClient.send_message()` dans `telegram.py` |
| 3 | `self.db.execute()` dans le plan | Notifier n'a pas d'accès DB. TelegramClient non plus. Il faut ajouter un `set_db()` |
| 4 | Types d'alerte manquants | `notify_live_order_opened/closed` → 'trade', `notify_grid_level/cycle` → 'trade', `notify_anomaly` (11 sous-types) → 'anomaly', `notify_reconciliation/leverage_divergence` → 'reconciliation', startup/shutdown → 'system' |
| 5 | Pas de fichier route `alerts_routes.py` | Le plan liste seulement "ajout endpoint dans routes existantes" — il faut un nouveau fichier `backend/api/alerts_routes.py` |
| 6 | Table SQL a `symbol` et `metadata JSON` | Inutile et sur-ingéniéré. `strategy` suffit (le symbol est dans le message). `metadata` JSON est un anti-pattern pour SQLite → simplifier |

---

## Approche recommandée

### Feature 2 — Choix architectural : hook au niveau `TelegramClient.send_message()`

**Pourquoi** : `send_message()` est le **point unique** par lequel TOUS les messages passent (chaque `send_*` method appelle `self.send_message(text)` à la fin). Ajouter `alert_type: str = "unknown"` en kwarg capture 100% des messages.

**Avantages** :
- Modifications minimales sur les callers (juste ajouter `alert_type=`)
- Pas de nouvelle classe wrapper
- Les callers non modifiés fonctionnent toujours (default `"unknown"`)

---

## Fichiers à créer (3)

1. `frontend/src/components/PaperLiveOverlay.jsx` — Composant overlay Recharts
2. `backend/api/alerts_routes.py` — Routes API alertes Telegram
3. `tests/test_telegram_persistence.py` — Tests DB + persistence

## Fichiers à modifier (6)

1. **`backend/core/database.py`** — Table `telegram_alerts` + `insert_telegram_alert()` + `get_telegram_alerts()`
2. **`backend/alerts/telegram.py`** — `set_db()`, param `alert_type` sur `send_message()`, mise à jour de chaque `send_*`
3. **`backend/alerts/notifier.py`** — `alert_type=` sur 3 appels directs à `send_message()` (anomaly, reconciliation, leverage)
4. **`backend/alerts/heartbeat.py`** — `alert_type="heartbeat"` sur le `send_message()`
5. **`backend/alerts/weekly_reporter.py`** — `alert_type="report"` sur le `send_message()`
6. **`backend/regime/regime_monitor.py`** — `alert_type="regime"` sur le `send_message()`
7. **`backend/api/server.py`** — `telegram.set_db(db)` au startup + `include_router(alerts_router)`
8. **`frontend/src/components/JournalPage.jsx`** — Import + 2 nouvelles CollapsibleCards (PaperLiveOverlay + TelegramAlerts inline)

---

## Plan d'implémentation détaillé

### 1. Database — `backend/core/database.py`

Table simplifiée (pas de `symbol`/`metadata`) :
```sql
CREATE TABLE IF NOT EXISTS telegram_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    message TEXT NOT NULL,
    strategy TEXT,
    success INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_telegram_alerts_ts ON telegram_alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_telegram_alerts_type ON telegram_alerts(alert_type);
```

Méthodes : `_create_telegram_alerts_table()`, `insert_telegram_alert()`, `get_telegram_alerts(alert_type, strategy, since, limit)`

### 2. TelegramClient — `backend/alerts/telegram.py`

- Ajouter `self._db = None` dans `__init__`
- Ajouter `set_db(db)`
- Modifier signature `send_message()` : `*, alert_type: str = "unknown", strategy: str | None = None`
- Après le résultat d'envoi, persister en best-effort (try/except, log DEBUG si erreur)
- Tronquer message à 2000 chars pour la DB
- Mettre à jour chaque `send_*` pour passer `alert_type=` et `strategy=`

### 3. Callers directs — 4 fichiers

- `notifier.py` : `notify_anomaly` → `alert_type="anomaly"`, `notify_reconciliation` → `"reconciliation"`, `notify_leverage_divergence` → `"reconciliation"`
- `heartbeat.py` : `alert_type="heartbeat"`
- `weekly_reporter.py` : `alert_type="report"`
- `regime_monitor.py` : `alert_type="regime"`

### 4. Wiring — `backend/api/server.py`

Après la création du TelegramClient (ligne ~87) : `telegram.set_db(db)`

### 5. API route — `backend/api/alerts_routes.py` (NOUVEAU)

```
GET /api/alerts/telegram?alert_type=trade&strategy=grid_atr&since=ISO&limit=100
→ {"alerts": [...], "count": N}
```

### 6. Mapping complet alert_type

| alert_type | Sources |
|------------|---------|
| `trade` | send_trade_alert, send_live_order_opened/closed, send_grid_level_opened, send_grid_cycle_closed |
| `kill_switch` | send_kill_switch_alert |
| `anomaly` | notify_anomaly (11 sous-types), send_live_sl_failed |
| `system` | send_startup_message, send_shutdown_message |
| `heartbeat` | Heartbeat loop |
| `report` | WeeklyReporter loop |
| `regime` | RegimeMonitor loop |
| `reconciliation` | notify_reconciliation, notify_leverage_divergence |

### 7. PaperLiveOverlay — `frontend/src/components/PaperLiveOverlay.jsx` (NOUVEAU)

- Fetch live via `useApi('/api/journal/live-equity?days=${days}&strategy=...')`
- Fetch paper via `useApi('/api/simulator/equity?since=${sinceISO}&strategy=...')`
- Normaliser : live `(p.equity - first.equity) / first.equity * 100`, paper `(p.capital - initial_capital) / initial_capital * 100`
- Merger en union (Map par timestamp), trier chronologiquement
- Recharts `LineChart` : live = trait plein `var(--accent)`, paper = pointillé `#f0ad4e`
- `connectNulls` sur les deux `Line` (timestamps différents)
- Tooltip : date FR, return % live, return % paper
- Boutons période : 7j / 30j
- Note : "Live inclut les P&L non réalisés"

### 8. JournalPage — `frontend/src/components/JournalPage.jsx`

Dans `LiveJournal`, après "Equity Curve Live" :
```jsx
<CollapsibleCard title="Paper vs Live" defaultOpen={false} storageKey="journal-paper-live">
  <PaperLiveOverlay strategy={stratParam} />
</CollapsibleCard>
```

Après "Performance par Asset" (ou dans LogViewer selon préférence) :
```jsx
<CollapsibleCard title="Alertes Telegram" defaultOpen={false} storageKey="journal-telegram-alerts">
  <TelegramAlerts strategy={stratParam} />
</CollapsibleCard>
```

`TelegramAlerts` = composant inline dans JournalPage (dropdown type, tableau filtrable, badge coloré par type).

### 9. Tests — `tests/test_telegram_persistence.py`

- `test_insert_and_get_alert` — round-trip DB
- `test_filter_by_type` — filtre par alert_type
- `test_filter_by_strategy` — filtre par stratégie
- `test_filter_by_since` — filtre par date
- `test_limit` — respect du limit
- `test_send_message_persists_to_db` — TelegramClient + DB mocké
- `test_send_message_without_db_no_crash` — graceful sans DB
- `test_send_trade_alert_passes_alert_type` — vérif kwarg
- `test_failed_send_persists_with_success_false` — échec = success=0

---

## Vérification

```bash
# Tests
uv run pytest tests/test_telegram_persistence.py -v
uv run pytest tests/test_telegram.py tests/test_notifier.py -v  # pas de régression
uv run pytest tests/ -x -q  # suite complète

# Frontend
cd frontend && npm run build

# Manuel
# 1. Démarrer le serveur, vérifier GET /api/alerts/telegram retourne 200
# 2. Déclencher une alerte (startup), vérifier qu'elle apparaît dans /api/alerts/telegram
# 3. Vérifier le composant PaperLiveOverlay dans JournalPage > Live
```

## Ordre d'implémentation

1. `database.py` (table + méthodes)
2. `telegram.py` (set_db + alert_type)
3. `notifier.py` + `heartbeat.py` + `weekly_reporter.py` + `regime_monitor.py` (callers)
4. `server.py` (wiring + route)
5. `alerts_routes.py` (nouveau)
6. `test_telegram_persistence.py` (tests)
7. `PaperLiveOverlay.jsx` (nouveau)
8. `JournalPage.jsx` (intégration des 2 cards)
