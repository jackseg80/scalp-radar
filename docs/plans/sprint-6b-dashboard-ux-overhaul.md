# Sprint 6b — Dashboard UX Overhaul

## Contexte

Le dashboard est fonctionnel mais pas exploitable en conditions réelles. Les infos critiques (positions actives, résultat des trades, conditions d'entrée) sont noyées dans un layout rigide. Ce sprint refond l'UX pour qu'un trader puisse comprendre l'état du système en un coup d'oeil.

## Changements Backend (2 fichiers)

### B1. Ajouter `symbol` aux trades — `backend/backtesting/simulator.py`

`get_all_trades()` (ligne 507) ne renvoie pas `symbol`. Le runner reçoit `symbol` dans `on_candle()` mais ne le stocke pas avec le trade.

**Solution :** Stocker `(symbol, trade)` dans `_trades` au lieu de `trade` seul.
- `_trades: list[tuple[str, TradeResult]]` (était `list[TradeResult]`)
- `_record_trade(self, trade, symbol)` → `self._trades.append((symbol, trade))`
- `get_trades()` → adapter pour retourner les tuples
- `get_all_trades()` → ajouter `"symbol": symbol` dans le dict, + `"tp_price"`, `"sl_price"` (depuis OpenPosition au moment de la fermeture, ou exit_reason pour déduire)
- `get_all_status()` → adapter les calculs (itèrent `_trades`)
- `restore_state()` → backward compat : si ancien format détecté (trade sans symbol), migrer avec `symbol="UNKNOWN"`

**Note :** `exit_reason` existe déjà (`"tp"`, `"sl"`, `"signal_exit"`, `"regime_change"`, `"end_of_data"`) — le frontend peut l'utiliser directement.

### B2. Enrichir les positions dans le WS — `backend/api/websocket_routes.py`

Ajouter un champ `positions` au push WS `/ws/live` (ligne 84) :
```python
data["simulator_positions"] = simulator.get_open_positions()
```

Ajouter `get_open_positions()` dans `Simulator` :
```python
def get_open_positions(self) -> list[dict]:
    """Positions ouvertes de tous les runners avec symbol."""
    # Itère runners × symbols, retourne celles avec _position != None
```

`get_conditions()` (ligne 630) contient déjà cette info par asset, mais elle n'est pas dans le WS push (seulement en polling 10s). L'ajouter au WS permet un affichage temps réel du bandeau positions.

## Changements Frontend (10 fichiers modifiés, 2 créés)

### F1. Nouveau composant `ActivePositions.jsx` (créer)

Bandeau au-dessus du Scanner dans `.content` (pas dans la sidebar).

**Données :** `wsData.simulator_positions` (WS) + `wsData.executor?.positions` (executor live)
**Affichage :**
- Positions simulator (paper) + positions executor (live) séparées
- Chaque position : emoji direction, asset, stratégie, entry price, P&L non réalisé (calculé avec `wsData.prices`)
- Si aucune position : message contextuel ("Aucune position" / "Kill switch actif" / "En attente de signal")
- Badge PAPER/LIVE pour distinguer les sources

### F2. Refonte `AlertFeed.jsx` → `ActivityFeed.jsx` (renommer + réécrire)

Renommer "Signaux" en "Activité". Chaque entrée = carte lisible au lieu d'une ligne compacte.

**Données :** `/api/simulator/trades?limit=20` (polling 10s) au lieu de `/api/signals/recent`
— Les trades sont plus riches que les signaux (entry/exit price, P&L, exit_reason, symbol).

**Format carte :**
```
🔴 SHORT SOL/USDT                    il y a 2h
vwap_rsi · Entry 81.34 → SL 81.58
Résultat: -243.31$ (fermé par SL)
```

- Positions ouvertes (depuis `wsData.simulator_positions`) affichées en premier avec fond distinct
- Trades fermés ensuite, triés par `exit_time` desc
- Temps relatif (il y a Xmin/Xh) via helper `timeAgo()`
- Exit reason traduit : `"sl"` → "fermé par SL", `"tp"` → "TP atteint", `"signal_exit"` → "sortie signal", `"regime_change"` → "changement régime"

### F3. Panneau redimensionnable — `App.jsx` + `styles.css`

Remplacer la grid fixe `1fr 340px` par un layout resizable.

**Implémentation :** CSS `resize` n'est pas assez flexible. Utiliser un simple drag handler JS :
- State `sidebarWidth` (défaut 35%, min 25%, max 50%)
- Div `.resize-handle` (6px, cursor col-resize) entre content et sidebar
- `onMouseDown` → track `mousemove` → update `sidebarWidth`
- Sauver en `localStorage('scalp-radar-sidebar-width')`
- Grid : `grid-template-columns: 1fr ${sidebarWidth}px`

### F4. Scanner enrichi — `Scanner.jsx` + `SignalDetail.jsx`

Garder le détail en expand inline (comme actuellement) mais l'enrichir considérablement.

**Changement :** Le panneau expand sous chaque asset affiche maintenant :
- Barres de progression visuelles pour chaque condition (value vs threshold)
- Texte explicite : "Manque : volume" (conditions non remplies)
- Score agrégé + dernier signal avec temps relatif
- Indicateurs avec barres (RSI 73.0 sur une barre 0-100, VWAP dist sur une barre etc.)

**Garder l'expand inline** plutôt qu'un split 50/50 — ça scale mieux avec 5+ assets.

Modifier `SignalDetail.jsx` pour afficher les conditions avec barres de progression et texte manquant.

### F5. Sidebar collapsible — `App.jsx` + nouveau hook/composant

Chaque section de sidebar devient collapsible. Créer un wrapper `CollapsibleCard`.

**Composant `CollapsibleCard.jsx` (créer) :**
```jsx
function CollapsibleCard({ title, summary, defaultOpen, children })
```
- `title` : titre de la section
- `summary` : texte affiché quand fermé (ex: "Simulator -520$", "14 trades", "#1 funding +167$")
- `defaultOpen` : état initial
- Click titre → toggle
- Sauver l'état ouvert/fermé en localStorage par section

**Sections et défauts :**
| Section | defaultOpen | Summary (fermé) |
|---------|------------|-----------------|
| Executor | true | mode badge |
| Simulator | false | P&L net |
| Equity Curve | true | -- |
| Activité | true | -- |
| Trades Récents | false | "{n} trades" |
| Arena | false | "#1 {name} {pnl}" |

### F6. TradeHistory enrichi — `TradeHistory.jsx`

Ajouter les colonnes manquantes :

| Colonne actuelle | Ajout |
|-----------------|-------|
| Stratégie | **Asset** (symbol) |
| Dir | -- |
| P&L | **Entry → Exit** |
| Heure | **Exit reason** (badge SL/TP/Signal) |
| -- | **Durée** (entry→exit) |

**Données :** Le backend `get_all_trades()` renvoie déjà `entry_price`, `exit_price`, `exit_reason`, `entry_time`, `exit_time` — mais pas encore `symbol` (ajouté en B1).

Table scrollable horizontalement si trop large. Garder le mode collapsible (5 visibles par défaut).

### F7. Kill switch visuel — `SessionStats.jsx` + `styles.css`

Si `wsData.kill_switch === true` : appliquer `.card--kill-switch` sur le wrapper `.card` du SessionStats.

```css
.card--kill-switch {
  background: rgba(255, 68, 102, 0.06);
  border-color: rgba(255, 68, 102, 0.2);
}
```

Le fond rouge subtil est visible immédiatement, pas juste le badge texte.

### F8. Mise à jour styles — `styles.css`

Nouvelles classes CSS :
- `.card--kill-switch` : fond rouge subtil
- `.resize-handle` : séparateur draggable
- `.activity-card` : carte trade dans ActivityFeed
- `.activity-card--open` : fond distinct pour positions ouvertes
- `.condition-bar` : barre de progression condition (valeur vs seuil)
- `.condition-bar__fill` : remplissage
- `.active-positions-banner` : bandeau positions actives
- `.scanner-detail-panel` : panneau détail fixe en bas du scanner

### F9. Mise à jour `App.jsx`

- Import `ActivePositions` + `CollapsibleCard` + `ActivityFeed`
- Ajouter `ActivePositions` au-dessus du Scanner dans `.content`
- Remplacer les composants sidebar par `CollapsibleCard` wrappers
- Remplacer `AlertFeed` par `ActivityFeed`
- Ajouter le resize handler (state + events)

## Ordre d'implémentation

1. **B1** — Backend : ajouter `symbol` aux trades (+ `get_open_positions`)
2. **B2** — Backend : enrichir WS push avec positions
3. **F5** — `CollapsibleCard.jsx` (fondation pour sidebar)
4. **F8** — CSS nouvelles classes
5. **F1** — `ActivePositions.jsx` (bandeau)
6. **F2** — `ActivityFeed.jsx` (refonte AlertFeed)
7. **F4** — Scanner enrichi + SignalDetail refondu
8. **F6** — TradeHistory enrichi
9. **F7** — Kill switch visuel
10. **F3** — Panneau redimensionnable
11. **F9** — App.jsx (assemblage final)

## Fichiers impactés

### Backend (2 fichiers)
- `backend/backtesting/simulator.py` — `_trades` format, `get_all_trades()`, `get_open_positions()`
- `backend/api/websocket_routes.py` — `simulator_positions` dans le push WS

### Frontend (10 fichiers modifiés, 2 créés)
- **Créer** : `frontend/src/components/ActivePositions.jsx`
- **Créer** : `frontend/src/components/CollapsibleCard.jsx`
- **Renommer+réécrire** : `AlertFeed.jsx` → `ActivityFeed.jsx`
- **Modifier** : `App.jsx` (layout, imports, resize handler)
- **Modifier** : `Scanner.jsx` (détail en panneau fixe, pas expand)
- **Modifier** : `SignalDetail.jsx` (barres de progression conditions)
- **Modifier** : `TradeHistory.jsx` (colonnes enrichies)
- **Modifier** : `SessionStats.jsx` (kill switch fond rouge)
- **Modifier** : `ExecutorPanel.jsx` (minor — summary pour collapsible)
- **Modifier** : `ArenaRankingMini.jsx` (minor — summary pour collapsible)
- **Modifier** : `styles.css` (nouvelles classes + resize)

### Pas de nouveaux tests backend
Les changements backend sont mineurs (ajout de champs aux dicts). Les tests existants du simulator couvrent déjà `get_all_trades()` — ils devront être adaptés pour le nouveau format `_trades` (tuples).

## Vérification

1. `uv run pytest` — tous les tests passent (adapter ceux qui testent `get_all_trades`)
2. `dev.bat` — lancer le dashboard, vérifier :
   - Bandeau positions actives visible (ou message contextuel)
   - Clic sur asset → panneau détail en bas avec barres de progression
   - Sidebar collapsible (clic titre → toggle, état sauvé en localStorage)
   - Resize handle fonctionnel (drag → largeur change, sauvé en localStorage)
   - Activité : cartes avec asset, entry/exit, P&L, temps relatif
   - Trades récents : colonnes asset, entry→exit, exit reason, durée
   - Kill switch → fond rouge sur Simulator
3. Vérifier responsivité : le resize respecte min 25% / max 50%
