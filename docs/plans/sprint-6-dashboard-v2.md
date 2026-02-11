# Sprint 6 — Dashboard V2

## Contexte

Le frontend actuel (Sprint 3) est un MVP fonctionnel avec 5 composants basiques : Header, ArenaRanking, SignalFeed, SessionStats, TradeHistory. Layout simple 2 colonnes, tables brutes, pas de visualisations.

Le Sprint 6 refond le dashboard en s'inspirant du prototype `docs/prototypes/Scalp radar v2.jsx` : tabs Scanner/Heatmap/Risque, composants visuels (ScoreRing, Sparkline, Heatmap), panel Executor live, equity curve, et alertes Telegram-style.

**Contraintes** : React 19 + Vite 6, pas de dépendance lourde (pas de recharts/d3/chart.js), SVG inline pour les graphiques, classes CSS (pas de inline styles), CSS variables (dark theme existant), desktop only (pas de responsive mobile).

---

## Architecture Frontend V2

### Layout — 2 colonnes + sidebar

Avec 3 assets × 4 stratégies (dont 2 paper-only), un layout 3 colonnes serait creux 95% du temps. On garde **2 colonnes** (contenu principal + sidebar) comme le MVP, mais avec du contenu beaucoup plus riche. Passage à 3 colonnes prévu quand le Sprint 5b ajoutera du volume (plus d'assets, plus de stratégies).

```
┌──────────────────────────────────────────────────────────┐
│ HEADER : logo + tabs [Scanner|Heatmap|Risque] + status  │
├──────────────────────────────────┬───────────────────────┤
│                                  │                       │
│   ZONE PRINCIPALE (tab content)  │   SIDEBAR DROITE      │
│                                  │                       │
│   Scanner: table assets          │   Executor Panel      │
│            + détail expandable   │   Session Stats       │
│   Heatmap: matrix + conditions   │   Equity Curve        │
│   Risque:  calculateur           │   Alert Feed          │
│                                  │   Trade History       │
│                                  │   Arena Ranking       │
│                                  │                       │
├──────────────────────────────────┴───────────────────────┤
│ FOOTER                                                    │
└──────────────────────────────────────────────────────────┘
```

En mode **Scanner** (tab par défaut) :
- **Principale** : table des assets avec conditions live (indicateurs courants, pas juste les signaux), clic sur une ligne → détail expandable inline (ScoreRing + SignalBreakdown + sparkline + entry/TP/SL)
- **Sidebar** : ExecutorPanel, SessionStats, EquityCurve, AlertFeed, TradeHistory (collapsible), ArenaRanking

En mode **Heatmap** :
- **Principale** : heatmap pleine largeur (assets × stratégies) + conditions courantes par asset
- **Sidebar** : inchangé

En mode **Risque** :
- **Principale** : RiskCalc pleine largeur
- **Sidebar** : inchangé

---

## Nouveaux endpoints backend

### `GET /api/simulator/conditions`

**Le point clé du sprint.** Expose les indicateurs courants par asset en permanence, pas seulement quand un signal se déclenche. Ça rend le dashboard utile 100% du temps.

Le backend retourne des **données brutes structurées** — pas de texte formaté pour l'UI. Le frontend formate les messages explicatifs ("RSI trop neutre") côté client. Ça évite de coupler le backend au texte français et de nécessiter un redéploiement Docker pour chaque changement de wording.

```json
{
  "assets": {
    "BTC/USDT": {
      "price": 68250.5,
      "change_pct": 0.35,
      "regime": "RANGING",
      "indicators": {
        "rsi_14": 42.3,
        "vwap_distance_pct": -0.15,
        "adx": 18.7,
        "atr_pct": 0.28
      },
      "strategies": {
        "vwap_rsi": {
          "last_signal": {"score": 0.72, "direction": "LONG", "timestamp": "...", "age_minutes": 45},
          "conditions": [
            {"name": "rsi_extreme", "met": false, "value": 42.3, "threshold": 30},
            {"name": "vwap_proximity", "met": true, "value": -0.15, "threshold": 0.3},
            {"name": "volume_spike", "met": false, "value": 1.2, "threshold": 2.0},
            {"name": "regime_ok", "met": true, "value": "RANGING", "threshold": "RANGING"}
          ]
        },
        "momentum": {
          "last_signal": null,
          "conditions": [
            {"name": "adx_strong", "met": false, "value": 18.7, "threshold": 25},
            {"name": "breakout", "met": false, "value": 0.02, "threshold": 0.5},
            {"name": "volume_confirm", "met": true, "value": 2.3, "threshold": 1.5}
          ]
        },
        "funding": { ... },
        "liquidation": { ... }
      },
      "position": null
    },
    "ETH/USDT": { ... },
    "SOL/USDT": { ... }
  },
  "timestamp": "2026-02-11T14:30:00Z"
}
```

**Source** : chaque stratégie expose ses conditions via une nouvelle méthode `get_current_conditions()` dans `BaseStrategy` (voir fichiers modifiés #23-27). L'`IncrementalIndicatorEngine` fournit les indicateurs bruts. Le Simulator agrège le tout.

**Cache** : le Simulator cache le résultat de `get_conditions()` et l'invalide à chaque nouvelle bougie 1m (quand les indicateurs changent réellement). L'endpoint retourne le cache, pas de recalcul à chaque appel.

**Polling** : 10s (les indicateurs changent à chaque bougie 1m, le cache absorbe les appels intermédiaires).

### `GET /api/signals/matrix`

Matrice simplifiée pour la Heatmap : dernier score par (stratégie, asset).

```json
{
  "matrix": {
    "BTC/USDT": {
      "vwap_rsi": 0.72,
      "momentum": 0.45,
      "funding": null,
      "liquidation": null
    },
    "ETH/USDT": { ... },
    "SOL/USDT": { ... }
  }
}
```

**Polling** : 30s (les signaux changent rarement, pas besoin de poller vite).

### `GET /api/simulator/equity`

Courbe d'equity calculée **depuis les trades en DB**, pas depuis des snapshots mémoire (robuste aux redémarrages).

Paramètre optionnel `?since=<ISO8601>` — ne retourne que les points après ce timestamp. Le frontend garde les points précédents en mémoire et ne demande que les nouveaux. Évite le scan complet à chaque poll.

```json
{
  "equity": [
    {"timestamp": "2026-02-11T10:05:00Z", "capital": 10025.5, "trade_pnl": 25.5},
    {"timestamp": "2026-02-11T10:32:00Z", "capital": 10012.3, "trade_pnl": -13.2},
    ...
  ],
  "current_capital": 10012.3,
  "initial_capital": 10000.0
}
```

**Calcul** : itère les trades triés par `exit_time`, accumule `net_pnl` sur le capital initial. Ajoute un point "now" avec le capital courant du Simulator (inclut les positions non fermées). Cache côté serveur, invalidé quand un nouveau trade est enregistré.

**Polling** : 30s.

### WebSocket enrichi `/ws/live`

Ajouter au push existant (toutes les 3s) :
- `executor` : statut executor (position live, SL/TP, kill switch live)
- `prices` : `{symbol: {last, change_pct}}` — prix live des 3 assets
- `latest_signal` : dernier signal émis (pour animation "new signal" temps réel)

Les prix passent **uniquement par le WebSocket** (pas de polling séparé).

---

## Fichiers à CRÉER

### 1. `frontend/src/components/ScoreRing.jsx`

Anneau SVG circulaire affichant un score 0-100 avec code couleur.
- Props : `score` (0-1), `size` (défaut 72px)
- Couleur : vert >= 75, jaune >= 55, orange >= 35, rouge < 35
- Animation CSS transition sur le strokeDashoffset
- **Classes CSS** (pas d'inline styles) : `.score-ring`, `.score-ring__value`
- Réf prototype : lignes 94-111

### 2. `frontend/src/components/Spark.jsx`

Sparkline SVG légère (pas de librairie).
- Props : `data` (array numbers), `width`, `height`, `stroke`
- Gradient fill sous la courbe
- Point animé (pulsing) à la dernière valeur
- Couleur auto : vert si montant, rouge si descendant
- Réf prototype : lignes 69-91

### 3. `frontend/src/components/Scanner.jsx`

Table des assets avec **conditions live** (toujours visible, jamais vide).
- Colonnes : Asset (+ pastille couleur), Prix, Var.%, Régime (badge), RSI, VWAP dist., Score, Conditions
- Chaque ligne montre les indicateurs courants (RSI actuel, distance VWAP, régime marché) — **toujours renseignés**
- Clic sur une ligne → expand inline : ScoreRing + SignalBreakdown + sparkline + entry/TP/SL
- **État "pas de signal"** : le frontend formate les conditions brutes en texte lisible (ex: "RSI 42 / seuil 30") — jamais de ligne vide
- Données : `/api/simulator/conditions` (poll 10s) + prix via WebSocket
- Composants enfants : `Spark`, `SignalDots`, `SignalDetail`

### 4. `frontend/src/components/SignalDots.jsx`

Mini-grille de pastilles colorées représentant le nombre de conditions remplies par stratégie.
- Props : `strategies` (objet {strategy_name: {conditions_met, conditions_total}})
- Icônes : ◎ VWAP, ◆ Momentum, ⚡ Liquidation, ∿ Funding
- Couleur par ratio : vert > 75%, jaune > 50%, orange > 25%, gris sinon
- Tooltip sur hover : "VWAP+RSI : 3/4 conditions remplies"
- Réf prototype : lignes 114-131

### 5. `frontend/src/components/SignalDetail.jsx`

Panel de détail expandable pour l'asset sélectionné dans le Scanner.
- Titre + direction badge + score label (EXCELLENT/BON/MOYEN/FAIBLE)
- `ScoreRing` avec le score composite
- `SignalBreakdown` (barres de progression par stratégie)
- Sparkline élargie (60 ticks)
- Entry / TP / SL : affichés **uniquement si un signal récent existe** (les paramètres stratégie ne sont pas exposés par l'API, donc pas de calcul client-side sans signal)
- **État vide** : affiche les indicateurs bruts formatés côté client ("RSI: 42, VWAP: -0.15%, Régime: RANGING") → info utile même sans signal
- Animation slideIn à l'apparition
- Réf prototype : lignes 491-540

### 6. `frontend/src/components/SignalBreakdown.jsx`

Barres de progression détaillées pour chaque stratégie sur un asset.
- Props : `strategies` (objet {strategy_name: {conditions: [...]}})
- Barre horizontale : remplie au ratio conditions remplies / total
- Label texte : **formaté côté client** depuis les données brutes (name, met, value, threshold) — ex: "ADX 18.7 / seuil 25"
- Réf prototype : lignes 134-153

### 7. `frontend/src/components/Heatmap.jsx`

Matrice couleur assets × stratégies avec conditions courantes.
- Grid CSS : colonne "Asset" + 1 colonne par stratégie + colonne "Régime"
- Cellules : ratio conditions remplies (vert = ready, gris = loin du signal)
- Ligne supplémentaire par asset : indicateurs clés (RSI, VWAP dist., ADX)
- **État vide** : la heatmap montre TOUJOURS les conditions (jamais vide)
- Données : `/api/simulator/conditions` (poll 10s)
- Réf prototype : lignes 218-261

### 8. `frontend/src/components/RiskCalc.jsx`

Calculateur de risque interactif (100% client-side, pas d'API).
- Inputs : capital ($), levier (slider 2x-50x), stop loss (%)
- Outputs : taille position, perte max, distance liquidation, ratio R:R
- Warning si levier > 20x
- **Toujours fonctionnel** (pas de dépendance aux données live)
- Réf prototype : lignes 156-215

### 9. `frontend/src/components/AlertFeed.jsx`

Feed d'alertes style Telegram.
- Affiche les derniers signaux exécutés en timeline verticale
- Chaque alerte : pastille couleur, symbol + direction badge, score, trigger, prix, timestamp
- Animation slideIn sur les nouvelles entrées
- **État vide** : "Scan en cours... Dernier signal il y a Xh" (pas un écran vide)
- Données : `/api/signals/recent` (poll 30s) + WebSocket `latest_signal`
- Max 25 alertes affichées, scroll interne
- Réf prototype : lignes 264-303

### 10. `frontend/src/components/ExecutorPanel.jsx`

Panel Executor live (nouveau, pas dans le prototype).
- Si `LIVE_TRADING=false` : badge "SIMULATION ONLY", pas de boutons
- Si actif + position ouverte : symbol, direction, entry price, SL/TP, order IDs, P&L non réalisé
- Si actif + pas de position : "En attente de signal..."
- Kill switch live : warning rouge si déclenché
- **Distinction sandbox/mainnet** : lire `sandbox` depuis `/api/executor/status` pour adapter le message de confirmation ("ordre RÉEL sur Bitget MAINNET" vs "ordre sur le TESTNET Bitget")
- **Boutons test-trade / test-close** :
  - Masqués si `LIVE_TRADING=false`
  - Désactivés si kill switch actif
  - Double confirmation avec texte adapté sandbox/mainnet
  - Log chaque clic dans la console
- Données : WebSocket `executor` (temps réel, pas de polling séparé)
- **Loading state** : skeleton pendant la première réponse WS

### 11. `frontend/src/components/EquityCurve.jsx`

Graphique SVG de la courbe d'equity session.
- SVG inline, pas de librairie externe
- Ligne de base à 10 000$ (capital initial)
- Zones vertes au-dessus, rouges en dessous
- Axes : temps (heures) et capital ($)
- **État vide** : ligne plate à 10 000$ avec texte "Pas encore de trades"
- Données : `/api/simulator/equity` (poll 30s)

### 12. `frontend/src/components/ArenaRankingMini.jsx`

Version compacte du classement Arena pour la sidebar.
- 4 lignes (une par stratégie) : nom + P&L + badge actif/stop
- **Clic** → expand le ArenaRanking complet en modal/panel (profit factor, drawdown, win rate)
- Données : WebSocket `ranking`

### 13. `backend/api/conditions_routes.py`

Nouveau router FastAPI :
- `GET /api/simulator/conditions` — indicateurs courants + état stratégies par asset
- `GET /api/signals/matrix` — matrice simplifiée pour heatmap
- `GET /api/simulator/equity` — courbe equity calculée depuis les trades

---

## Fichiers à MODIFIER

### 14. `frontend/src/App.jsx`

Refonte du layout :
- State : `tab` (scanner/heatmap/risk), `selectedAsset`
- Layout CSS Grid **2 colonnes** (proportions ~65% / 35%)
- Tabs Scanner/Heatmap/Risque dans le header
- Zone principale : contenu du tab actif
- Sidebar : ExecutorPanel, SessionStats, EquityCurve, AlertFeed, TradeHistory, ArenaRankingMini
- **Loading states** : chaque composant affiche un skeleton/placeholder avant la première réponse API

### 15. `frontend/src/components/Header.jsx`

Enrichir le header :
- Ajouter les tabs [Scanner | Heatmap | Risque]
- Toggle LIVE/PAUSED (contrôle le polling et WS)
- Indicateur "Best score" du moment
- Garder les status dots existants (Engine, DB, WS)

### 16. `frontend/src/components/SessionStats.jsx`

Enrichir avec :
- P&L total, capital total, trades, win rate (existant)
- Ratio LONG/SHORT
- Max drawdown
- Affichage plus compact pour la sidebar

### 17. `frontend/src/components/TradeHistory.jsx`

**Garder** (pas supprimer) mais adapter :
- Version compacte pour la sidebar (5 derniers trades)
- Collapsible : clic pour expand la liste complète
- Colonnes : stratégie, direction, P&L net, temps (format compact)
- C'est la vue analytique table que l'AlertFeed ne remplace pas

### 18. `frontend/src/styles.css`

Refonte du CSS :
- Nouvelles variables CSS (couleurs du prototype : orange, blue, dim)
- Layout 2 colonnes
- **Classes CSS explicites** pour chaque composant (`.scanner-row`, `.score-ring`, `.heatmap-cell`, etc.)
- Pas d'inline styles — tout en classes CSS
- Scrollbar custom (prototype style)
- Keyframes : slideIn, blink, pulse
- Classes utilitaires : `.mono`, `.muted`, `.dim`
- Loading skeleton : `.skeleton` avec animation pulse

### 19. `frontend/src/hooks/useWebSocket.js`

Pas de changement structurel. Le hook existant gère déjà le reconnect.
Le format du message WebSocket change (champs ajoutés), mais le hook est agnostique au contenu.

### 20. `backend/api/server.py`

- Inclure le nouveau router `conditions_routes`
- Passer `data_engine` et `executor` aux routes qui en ont besoin

### 21. `backend/api/websocket_routes.py`

Enrichir le message push avec :
```python
data["executor"] = executor.get_status() if executor else None
data["prices"] = _get_current_prices(engine)  # {symbol: {last, change_pct}}
```

### 22. `backend/backtesting/simulator.py`

Ajouter :

- `_conditions_cache: dict | None` — cache invalidé à chaque nouvelle bougie 1m
- `_equity_cache: list[dict] | None` — cache invalidé quand un nouveau trade est enregistré
- `get_conditions() → dict` — indicateurs courants par asset + conditions par stratégie. Retourne le cache si valide, sinon appelle `runner.strategy.get_current_conditions()` pour chaque runner et construit la réponse
- `get_signal_matrix() → dict` — dernier score par (strategy, symbol)
- `get_equity_curve(since: str | None) → dict` — calculé depuis les trades, avec filtre `since` optionnel. Cache invalidé quand `_record_trade` est appelé

### 23. `backend/strategies/base.py`

Ajouter méthode abstraite à `BaseStrategy` :

```python
@abstractmethod
def get_current_conditions(self, context: StrategyContext) -> list[dict]:
    """Retourne les conditions d'entrée avec leur état actuel.

    Chaque condition : {"name": str, "met": bool, "value": float|str, "threshold": float|str}
    Ne modifie PAS la logique de trading (check_entry reste le point d'entrée).
    Méthode read-only pour le dashboard.
    """
```

C'est la pièce manquante entre le Simulator et le dashboard. Chaque stratégie expose ses conditions d'entrée de manière structurée, sans toucher à `check_entry()`.

### 24. `backend/strategies/vwap_rsi.py`

Implémenter `get_current_conditions()` :

- `rsi_extreme` : RSI < 30 (long) ou > 70 (short), valeur courante vs seuil
- `vwap_proximity` : distance prix/VWAP < X%, valeur courante vs seuil
- `volume_spike` : volume > X× moyenne, valeur courante vs seuil
- `regime_ok` : régime RANGING, valeur courante vs attendu

### 25. `backend/strategies/momentum.py`

Implémenter `get_current_conditions()` :

- `adx_strong` : ADX > 25, valeur courante vs seuil
- `breakout` : cassure de range, valeur courante vs seuil
- `volume_confirm` : volume de confirmation, valeur courante vs seuil

### 26. `backend/strategies/funding.py`

Implémenter `get_current_conditions()` :

- `funding_extreme` : taux funding extrême, valeur courante vs seuil
- `delay_ok` : délai depuis dernier trade, valeur courante vs minimum

### 27. `backend/strategies/liquidation.py`

Implémenter `get_current_conditions()` :

- `zone_proximity` : distance à la zone de liquidation, valeur courante vs seuil
- `oi_threshold` : changement d'OI suffisant, valeur courante vs seuil
- `volume_confirm` : volume de confirmation, valeur courante vs seuil

---

## Fichiers à SUPPRIMER

### 23. `frontend/src/components/SignalFeed.jsx`

Remplacé par `AlertFeed.jsx` (même rôle, meilleur design).

### 24. `frontend/src/components/ArenaRanking.jsx`

Remplacé par `ArenaRankingMini.jsx` (sidebar, expandable vers le détail complet).

---

## Ordre d'implémentation — 5 phases

### Phase 1 : Infrastructure (CSS + Layout + Backend)

| Fichier | Action |
|---------|--------|
| `styles.css` | Refonte CSS : variables, layout 2 colonnes, classes, animations, skeletons |
| `App.jsx` | Structure 2 colonnes + tabs (composants placeholder) |
| `base.py` | Méthode abstraite `get_current_conditions()` |
| `vwap_rsi.py` | Implémenter `get_current_conditions()` |
| `momentum.py` | Implémenter `get_current_conditions()` |
| `funding.py` | Implémenter `get_current_conditions()` |
| `liquidation.py` | Implémenter `get_current_conditions()` |
| `simulator.py` | `get_conditions()` (avec cache), `get_signal_matrix()`, `get_equity_curve(?since=)` |
| `conditions_routes.py` | 3 nouveaux endpoints |
| `server.py` | Include nouveau router |
| `websocket_routes.py` | Ajouter executor + prices au push |

**Résultat visible** : layout vide avec tabs fonctionnels, endpoints backend testables via curl, `npm run build` OK.

**Smoke test** : `npm run build` + `uv run pytest` à la fin de chaque phase.

### Phase 2 : Composants visuels de base

| Fichier | Action |
|---------|--------|
| `ScoreRing.jsx` | Composant SVG anneau score |
| `Spark.jsx` | Composant SVG sparkline |
| `SignalDots.jsx` | Mini-grille conditions |
| `SignalBreakdown.jsx` | Barres de progression |

**Résultat visible** : composants isolés, testables unitairement.

### Phase 3 : Vues principales (tabs)

| Fichier | Action |
|---------|--------|
| `Scanner.jsx` + `SignalDetail.jsx` | Table assets + détail expandable |
| `Heatmap.jsx` | Matrice assets × stratégies |
| `RiskCalc.jsx` | Calculateur de risque |
| `Header.jsx` | Tabs, toggle LIVE, best score |

**Résultat visible** : les 3 tabs fonctionnent avec données réelles.

### Phase 4 : Sidebar

| Fichier | Action |
|---------|--------|
| `ExecutorPanel.jsx` | Panel executor live + garde-fous boutons |
| `SessionStats.jsx` | Version enrichie |
| `EquityCurve.jsx` | Graphique SVG equity |
| `AlertFeed.jsx` | Feed Telegram-style |
| `TradeHistory.jsx` | Version compacte collapsible |
| `ArenaRankingMini.jsx` | Ranking compact expandable |

**Résultat visible** : sidebar complète, dashboard fonctionnel.

### Phase 5 : Intégration + Polish

| Fichier | Action |
|---------|--------|
| `App.jsx` | Câblage final tous composants |
| `SignalFeed.jsx` | Supprimer (remplacé par AlertFeed) |
| `ArenaRanking.jsx` | Supprimer (remplacé par ArenaRankingMini) |
| Tous | Empty states, loading skeletons, animations, polish |

**Résultat visible** : dashboard V2 complet.

---

## Décisions clés

1. **Pas de librairie de charts** — SVG inline pour Sparkline, ScoreRing, EquityCurve. Élimine recharts/d3/chart.js (200-500kB). Le prototype prouve que c'est suffisant
2. **CSS variables + classes (pas Tailwind, pas inline)** — on étend le système existant. Le prototype utilise des inline styles partout — on les migre vers des classes CSS pour la maintenabilité
3. **2 colonnes (pas 3)** — avec 3 assets et des signaux rares, 3 colonnes serait creux. Extensible vers 3 colonnes quand Sprint 5b ajoutera du volume
4. **Endpoint `/api/simulator/conditions`** — expose les indicateurs courants (RSI, VWAP, régime) en permanence, pas seulement les signaux. Le dashboard est utile 100% du temps, pas juste quand un signal arrive
5. **Equity depuis les trades en DB** — pas de snapshots mémoire (perdus au restart). L'endpoint calcule la courbe à partir des trades persistés
6. **Prix via WebSocket only** — les prix changent chaque seconde, pas de polling séparé. Les signaux (rares) sont pollés à 30s
7. **TradeHistory conservé** — l'AlertFeed est un flux chronologique mixte, pas un remplacement de la vue analytique table. TradeHistory reste en sidebar (version compacte collapsible)
8. **ArenaRanking expandable** — le mini ranking en sidebar est cliquable vers le détail complet (profit factor, drawdown, win rate)
9. **Empty states informatifs** — chaque composant a un état vide utile. Le frontend formate les conditions brutes en texte lisible ("RSI 42 / seuil 30"). Jamais un écran gris vide
10. **Loading skeletons** — chaque composant affiche un placeholder animé avant la première réponse API (pas de flash "undefined")
11. **Boutons executor sécurisés** — masqués si LIVE_TRADING=false, désactivés si kill switch, double confirmation adaptée sandbox/mainnet
12. **Données brutes, formatage client** — le backend ne retourne jamais de texte UX. Les conditions sont structurées (name, met, value, threshold), le frontend formate en français. Ça découple le backend du wording et évite les redéploiements Docker pour un changement de texte
13. **Cache conditions** — le Simulator cache `get_conditions()`, invalidé à chaque bougie 1m. L'endpoint retourne le cache, pas de recalcul à chaque appel API
14. **Cache equity** — `get_equity_curve()` est caché, invalidé quand un trade est enregistré. Paramètre `?since=` pour ne renvoyer que les nouveaux points
15. **`get_current_conditions()` dans BaseStrategy** — méthode read-only séparée de `check_entry()`. Expose les conditions d'entrée sans toucher à la logique de trading. Implémentée dans les 4 stratégies
16. **Fonts système** — `ui-monospace, SFMono-Regular, monospace` au lieu de Google Fonts. Évite le chargement externe et suffit pour le dashboard
17. **5 phases d'implémentation** — chaque phase finit par `npm run build` + `uv run pytest`. Résultat visible et testable

---

## Mapping Prototype → Composants

| Prototype (inline) | Nouveau composant | Adaptation |
|---------------------|-------------------|------------|
| `ScoreRing` (l.94) | `ScoreRing.jsx` | Inline styles → classes CSS |
| `Spark` (l.69) | `Spark.jsx` | Inline styles → classes CSS |
| `SignalDots` (l.114) | `SignalDots.jsx` | 5 indicateurs → 4 stratégies, scores → conditions ratio |
| `SignalBreakdown` (l.134) | `SignalBreakdown.jsx` | Scores → conditions brutes, formatage client |
| `Heatmap` (l.218) | `Heatmap.jsx` | 8×5 → 3×4, + conditions courantes |
| `RiskCalc` (l.156) | `RiskCalc.jsx` | Quasi identique, inline → classes CSS |
| `AlertFeed` (l.264) | `AlertFeed.jsx` | Données simulées → `/api/signals/recent` |
| `ScalpRadar` main (l.306) | `App.jsx` + `Scanner.jsx` | 3 colonnes → 2 colonnes, inline → classes |
| — | `ExecutorPanel.jsx` | Nouveau (Sprint 5a) |
| — | `EquityCurve.jsx` | Nouveau, SVG inline |
| — | `ArenaRankingMini.jsx` | Version compacte expandable |

---

## Palette couleurs (du prototype, étend l'existant)

```css
--bg-primary: #06080d;     /* existant */
--bg-card: rgba(255,255,255,0.018);  /* plus subtil que #141922 */
--border: rgba(255,255,255,0.055);
--text-primary: #e8eaed;
--text-secondary: rgba(255,255,255,0.55); /* existant */
--text-muted: rgba(255,255,255,0.35);
--text-dim: rgba(255,255,255,0.18);
--accent: #00e68a;         /* existant (vert) */
--red: #ff4466;            /* légèrement ajusté */
--yellow: #ffc53d;         /* nouveau */
--orange: #ff8c42;         /* nouveau */
--blue: #4da6ff;           /* nouveau */
```

---

## Estimation

- **Frontend** : 12 composants (10 nouveaux, 5 modifiés, 2 supprimés), ~1500 lignes JSX + CSS
- **Backend** : 1 nouveau fichier routes, 5 stratégies + Simulator + WS modifiés, ~300 lignes Python
- **Tests** : `npm run build` à chaque phase + `uv run pytest` (252 existants + tests nouveaux endpoints)

---

## Vérification

1. `npm run build` — aucune erreur (frontend)
2. `uv run pytest` — aucune régression backend (252 tests existants + nouveaux)
3. **Scanner toujours informatif** : les 3 assets affichent RSI, VWAP dist, régime même sans signal actif
4. **Tab Heatmap** : matrice 3×4 avec conditions, jamais vide
5. **Tab Risque** : calculateur interactif (toujours fonctionnel)
6. **Sidebar** : ExecutorPanel, SessionStats, EquityCurve, AlertFeed, TradeHistory, ArenaRanking
7. **ExecutorPanel** : "SIMULATION ONLY" si `LIVE_TRADING=false`, boutons masqués
8. **ExecutorPanel** : double confirmation sur test-trade si `LIVE_TRADING=true`
9. **WebSocket** : mise à jour temps réel des prix, executor
10. **Empty states** : chaque composant est informatif même sans données (pas d'écran vide)
11. **Loading** : skeleton/placeholder avant la première réponse API (pas de flash)
12. **TradeHistory** : accessible en sidebar, expandable vers la liste complète
13. Pas de dépendance ajoutée dans `package.json`
