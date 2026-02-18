# Sprint 31 — Log Viewer (mini-feed WS + onglet terminal)

## Contexte

Les logs backend (loguru `serialize=True`) ne sont accessibles qu'en SSH. On veut :
1. **LogMini** : mini-feed sidebar temps réel (WARNING/ERROR via WS, toujours visible)
2. **LogViewer** : onglet complet "Logs" style terminal Linux (polling HTTP + auto-refresh)

---

## Phase 1 — Backend : endpoint HTTP `GET /api/logs`

**Nouveau fichier** : [backend/api/log_routes.py](backend/api/log_routes.py)

- Lit le fichier log JSON à l'envers par chunks (seek depuis la fin, chunks 8KB)
- Guard : fichier > 100MB → ne lire que les derniers 10MB
- Paramètres query : `limit` (max 500), `level`, `search`, `module`, `since` (ISO datetime)
- Chaque ligne = JSON loguru sérialisé → parser `record.level.name`, `record.time`, `record.name`, `record.function`, `record.line`, `record.message`
- Retourne `{"logs": [{"timestamp", "level", "module", "function", "line", "message"}]}`
- Chemin log depuis `logs/scalp_radar.log` (même que [logging_setup.py:43](backend/core/logging_setup.py#L43))

**Modifier** : [backend/api/server.py](backend/api/server.py)
- Ajouter `from backend.api.log_routes import router as log_router` + `app.include_router(log_router)`

---

## Phase 2 — Backend : log alerts WS temps réel

### 2a. Sink loguru dans [logging_setup.py](backend/core/logging_setup.py)

Ajouter buffer circulaire + subscriber pattern (comme décrit dans le spec) :

```python
_log_buffer: deque[dict] = deque(maxlen=20)
_log_subscribers: set[asyncio.Queue] = set()

def _ws_log_sink(message):
    """Capture WARNING+ et push aux subscribers WS. Synchrone, thread-safe."""
    record = message.record
    if record["level"].no < 30:  # 30 = WARNING
        return
    entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
    }
    _log_buffer.append(entry)
    for queue in list(_log_subscribers):  # snapshot copy = thread-safe
        try:
            queue.put_nowait(entry)
        except asyncio.QueueFull:
            pass  # Client lent, on drop

def subscribe_logs() -> asyncio.Queue: ...
def unsubscribe_logs(q: asyncio.Queue): ...
def get_log_buffer() -> list[dict]: ...
```

Dans `setup_logging()`, ajouter en dernier : `logger.add(_ws_log_sink, level="WARNING", format="{message}")`

### 2b. Intégration WS dans [websocket_routes.py](backend/api/websocket_routes.py)

**Point critique** : la boucle WS actuelle fait `sleep(3)` → on ne peut pas envoyer les alertes immédiatement. Solution : remplacer `asyncio.sleep(3)` par `asyncio.wait_for(queue.get(), timeout=remaining)`.

Modification du handler `live_feed()` :
1. Au connect : `log_queue = subscribe_logs()`
2. Envoyer `get_log_buffer()` comme messages initiaux `{"type": "log_alert", "entry": ...}`
3. Boucle principale : `wait_for(log_queue.get(), timeout=temps_restant_avant_update)`
   - Timeout → envoyer update normal `{"type": "update", ...}`, reset timer 3s
   - Queue item → envoyer `{"type": "log_alert", "entry": {...}}` immédiatement, NE PAS reset le timer
4. Au disconnect : `unsubscribe_logs(log_queue)`

Tous les sends passent par un seul coroutine par connexion → pas de race condition.

---

## Phase 3 — Frontend : refactoring useWebSocket + LogMini

### 3a. Refactoring [useWebSocket.js](frontend/src/hooks/useWebSocket.js)

**Problème actuel** : `lastMessage` est écrasé par TOUT type de message WS. Un `log_alert` écraserait le dernier `update` → `wsData.strategies` serait `undefined` brièvement.

**Fix** : dispatcher par `data.type` dans `onmessage` :
- `type === "update"` → `setLastUpdate(data)` (remplace `lastMessage`)
- `type === "log_alert"` → `setLogAlerts(prev => [data.entry, ...prev].slice(0, 50))`
- Autre → `setLastEvent(data)` (pour `optimization_progress`, `portfolio_progress`)

Retourne `{ lastUpdate, lastEvent, logAlerts, connected }` au lieu de `{ lastMessage, connected }`

### 3b. Mise à jour [App.jsx](frontend/src/App.jsx)

- `const { lastUpdate, lastEvent, logAlerts, connected } = useWebSocket(wsUrl)`
- `const wsData = lastUpdate` (remplacement direct, zéro impact sur les composants existants)
- Ajouter onglet `{ id: 'logs', label: 'Logs' }` dans `TABS`
- State `unseenLogErrors` (compteur, reset quand on clique l'onglet Logs)
- `useEffect` sur `logAlerts` : incrémenter `unseenLogErrors` si `activeTab !== 'logs'`
- Passer `logAlerts` à `LogMini` dans la sidebar
- Passer `logAlerts` à `LogViewer` dans le content area
- Passer `lastEvent` à `ExplorerPage` et `PortfolioPage`
- Mettre à jour `loadActiveTab()` pour inclure `'logs'`

### 3c. Mise à jour composants qui écoutent des messages non-update

**[ExplorerPage.jsx:257](frontend/src/components/ExplorerPage.jsx#L257)** :
- Ajouter prop `lastEvent`, utiliser `lastEvent?.type === 'optimization_progress'` au lieu de `wsData?.type`

**[PortfolioPage.jsx:178](frontend/src/components/PortfolioPage.jsx#L178)** :
- Ajouter prop `lastEvent`, utiliser `lastEvent?.type` pour `portfolio_progress` ET `portfolio_completed`

**Sécurité du refactoring** : les 7 autres composants consommant `wsData` (Scanner, ActivityFeed, ActivePositions, ExecutorPanel, SessionStats, ArenaRankingMini, AlertFeed) ne lisent que des champs `type: "update"` (`.strategies`, `.executor`, `.prices`, `.ranking`, `.simulator_positions`, `.grid_state`). Le remplacement `wsData = lastUpdate` est transparent pour eux.

### 3d. Nouveau composant : `LogMini.jsx`

Sidebar, wrappé dans `CollapsibleCard` (sous Activité) :
- Reçoit `logAlerts` en prop (array, max 50)
- Affiche les 20 derniers WARNING/ERROR
- Pastille couleur : orange WARNING, rouge ERROR/CRITICAL
- Chaque entrée : `HH:mm:ss` + message tronqué 1 ligne (ellipsis CSS)
- Clic sur entrée → `onTabChange('logs')` (prop callback)
- Si vide : "Aucune alerte" en vert
- `LogMini.getSummary()` : nombre d'alertes non vues ou "OK"

---

## Phase 4 — Frontend : onglet LogViewer.jsx

**Nouveau composant** : `LogViewer.jsx` + `LogViewer.css`

### Style terminal
- Fond noir pur `#0a0a0a`, font monospace (`var(--font-mono)`), taille 12-13px
- Couleurs ANSI : vert INFO, jaune WARNING, rouge ERROR, gris DEBUG, rouge vif CRITICAL
- Format ligne : `HH:mm:ss.SSS | LEVEL    | module:function:line | message`
- Auto-scroll (tail -f) + bouton pause scroll
- Sélection texte possible

### Barre de filtres (toolbar sombre au-dessus)
- Boutons toggle par niveau (multi-select, défaut INFO+WARNING+ERROR)
- Input recherche `grep...` (debounce 300ms)
- Dropdown module : "Tous", "executor", "simulator", etc.
- Toggle auto-refresh (poll `GET /api/logs?since=...` toutes les 5s)
- Indicateur vert pulsant quand live

### Comportement
- Polling HTTP `GET /api/logs` (pas WS) — tourne uniquement quand onglet actif
- `since` = timestamp du dernier log reçu (refresh incrémental)
- Max 500 lignes en mémoire, purge les anciennes par le haut
- Nouvelles lignes en bas avec flash highlight 1s
- Bouton "Charger plus" en haut
- Clic ligne → expand détails (function, line, full message)

---

## Phase 5 — Header + wiring

### Mise à jour [Header.jsx](frontend/src/components/Header.jsx)

- Recevoir `unseenLogErrors` en prop
- Onglet "Logs" avec badge rouge si `unseenLogErrors > 0`
- Le badge = petit cercle rouge avec nombre, style existant des badges

---

## Fichiers modifiés/créés

| Fichier | Action |
|---------|--------|
| `backend/api/log_routes.py` | **Nouveau** — endpoint GET /api/logs |
| `backend/core/logging_setup.py` | Modifier — ajouter sink WS + subscribe/unsubscribe |
| `backend/api/websocket_routes.py` | Modifier — log queue per-connection + wait_for pattern |
| `backend/api/server.py` | Modifier — inclure log_router |
| `frontend/src/hooks/useWebSocket.js` | Modifier — split par type (lastUpdate/lastEvent/logAlerts) |
| `frontend/src/App.jsx` | Modifier — wiring logAlerts, onglet Logs, unseenLogErrors |
| `frontend/src/components/Header.jsx` | Modifier — badge erreurs non vues |
| `frontend/src/components/ExplorerPage.jsx` | Modifier — lastEvent prop (ligne 257) |
| `frontend/src/components/PortfolioPage.jsx` | Modifier — lastEvent prop (lignes 178, 183) |
| `frontend/src/components/LogMini.jsx` | **Nouveau** — mini-feed sidebar |
| `frontend/src/components/LogViewer.jsx` | **Nouveau** — onglet terminal complet |
| `frontend/src/components/LogViewer.css` | **Nouveau** — styles terminal |
| `tests/test_log_routes.py` | **Nouveau** — 5 tests endpoint HTTP |
| `tests/test_log_ws.py` | **Nouveau** — 3 tests sink + subscribe |

---

## Vérification

1. **Tests backend** : `pytest tests/test_log_routes.py tests/test_log_ws.py -v`
   - HTTP : JSON valide, filtre level/search/module, limit/since
   - WS : sink capture WARNING+ uniquement, buffer maxlen, subscribe/unsubscribe cleanup
2. **Régression** : `pytest tests/ -x` → 1217+ tests passants (0 régression)
3. **Test manuel** :
   - Lancer `dev.bat`, ouvrir le dashboard
   - Vérifier que LogMini affiche les WARNING/ERROR en temps réel dans la sidebar
   - Cliquer l'onglet Logs → vérifier le style terminal, auto-scroll, filtres
   - Provoquer un WARNING (ex: data stale) → vérifier apparition immédiate dans LogMini ET LogViewer
   - Badge rouge Header quand erreurs non vues, reset au clic
