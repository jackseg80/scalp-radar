# TROUBLESHOOTING.md — Guide de dépannage Scalp Radar

> Toutes les commandes s'exécutent sur le serveur (`ssh jack@192.168.1.200`, dans `~/scalp-radar`).

---

## 1. Checklist de diagnostic rapide

Une séquence de 8-10 commandes à exécuter dans l'ordre dès qu'un problème survient, AVANT de plonger dans les sections détaillées :

```bash
# 1. Vérifier si les containers tournent et sont sains
docker compose ps

# 2. Vérifier la santé du backend API
curl -sf http://localhost:8000/health || echo "Backend DOWN"

# 3. Checker les dernières erreurs critiques
docker compose logs backend --since 15m | grep -i "error\|exception" | tail -10

# 4. Statut global de l'executor et kill switch
curl -s http://localhost:8000/api/executor/status -H "X-API-Key: $(grep SYNC_API_KEY .env | cut -d= -f2)" | grep -E "enabled|kill_switch|session_pnl"

# 5. Etat des flux de données et WS Bitget (heartbeat)
curl -s http://localhost:8000/api/data/status | grep -E "active|stale"

# 6. Vérifier la RAM / CPU (fuite mémoire potentielle)
docker stats --no-stream

# 7. Regarder s'il y a un décalage de positions paper/live
docker compose logs backend --since 5m | grep -i "sync.*terminé" | tail -5

# 8. Checker les rejets d'ordres par Bitget
docker compose logs backend --since 1h | grep -i "rejected\|insufficient" | tail -5
```

---

## Le bot n'ouvre aucune position

### Symptôme
Dashboard affiche "En attente de signal", aucune activité depuis des heures.

### Diagnostic

```bash
# 1. Kill switch actif ?
docker compose logs backend --since 5m | grep -i "kill_switch"

# 2. Executor connecté ?
curl -s http://localhost:8000/api/executor/status \
  -H "X-API-Key: $(grep SYNC_API_KEY .env | cut -d= -f2)" | python3 -m json.tool

# 3. DataEngine reçoit des données ?
docker compose logs backend --since 5m | grep -E "(heartbeat|DataEngine.*actifs)"

# 4. Stratégie autorisée ?
docker compose logs backend --since 5m | grep -i "live_eligible\|force_strategies\|autorisé"

# 5. AdaptiveSelector bloqué ?
docker compose logs backend --since 5m | grep -i "selector\|bypass\|evaluate"
```

### Fixes

| Cause | Commande |
|-------|---------|
| Kill switch actif | `curl -X POST http://localhost:8000/api/executor/kill-switch/reset -H "X-API-Key: $(grep SYNC_API_KEY .env | cut -d= -f2)"` |
| DataEngine mort | `docker compose restart backend` |
| Selector bloqué | Vérifier `FORCE_STRATEGIES=grid_atr` dans `.env` |

---

## Les positions ne se ferment pas (TP jamais détecté)

### Symptôme
Prix au-dessus de la SMA sur Bitget/TradingView mais le bot ne ferme pas.

### Diagnostic

```bash
# 1. Que voit l'exit monitor ?
docker compose logs backend --since 10m | grep -E "Exit monitor|no exit" | tail -10

# 2. Le prix bouge-t-il ?
watch -n 30 "docker compose logs backend --since 1m | grep 'price.*sma' | tail -1"

# 3. Si le prix est figé — DataEngine par symbol
curl -s http://localhost:8000/api/data/status | python3 -m json.tool

# 4. Le runner paper a-t-il des indicateurs ?
docker compose logs backend --since 5m | grep -E "(warm-up|build_context|indicateurs incomplets)" | tail -5
```

### Causes fréquentes

| Cause | Symptôme dans les logs | Fix |
|-------|----------------------|-----|
| Prix figé (buffer 1m vide) | `price=X.XX` ne change pas entre les checks | `docker compose restart backend` |
| SMA du paper pas encore calculée (warm-up) | `indicateurs incomplets` ou `pas de contexte runner` | Attendre 2-3 min après restart |
| Exit monitor ne tourne pas | Aucun log "Exit monitor: check" | Vérifier `start_exit_monitor` dans les logs de boot |
| DataEngine déconnecté pour ce symbol | `curl /api/data/status` montre "stale" | Le heartbeat devrait auto-reconnecter, sinon restart |

### Rappel SMA
La SMA du bot est calculée sur les **bougies 1h fermées** uniquement. TradingView inclut la bougie en cours → la SMA affichée est légèrement différente. Le TP trigger quand le prix temps réel (buffer 1m) dépasse la SMA des bougies fermées.

---

## 3. Kill switch (Live & Manuel)

### Symptôme
Dashboard affiche "KILL SWITCH LIVE ACTIF", aucune nouvelle position.

### Diagnostic

```bash
# Pourquoi s'est-il déclenché ?
docker compose logs backend --since 2h | grep -i "kill switch" | head -10

# Quel est le session P&L ?
curl -s http://localhost:8000/api/executor/status \
  -H "X-API-Key: $(grep SYNC_API_KEY .env | cut -d= -f2)" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Session PnL: {d.get(\"session_pnl\", \"?\")}\$')
print(f'Kill switch: {d.get(\"kill_switch\", \"?\")}')
"
```

### Fix

```bash
# Reset via API (préféré)
curl -X POST http://localhost:8000/api/executor/kill-switch/reset \
  -H "X-API-Key: $(grep SYNC_API_KEY .env | cut -d= -f2)"

# Si l'API ne répond pas — reset manuel
docker compose stop backend
sudo python3 -c "
import json
with open('data/executor_state.json') as f:
    state = json.load(f)
state['executor']['risk_manager']['kill_switch'] = False
state['executor']['risk_manager']['session_pnl'] = 0.0
with open('data/executor_state.json', 'w') as f:
    json.dump(state, f, indent=2)
print('OK')
"
docker compose start backend
```

> IMPORTANT : `docker compose stop` puis `start`, PAS `restart`. Le restart sauvegarde l'état en mémoire avant d'arrêter → écrase votre edit.
> IMPORTANT : `sudo` requis car le fichier appartient à root (créé par Docker).

### Discordance de calcul entre backtest engine et trading live
**Symptôme** : Le live déclenche le kill switch très vite, alors que le backtest n'indiquait aucun risque extrême.  
**Cause probable** : Le backtest inclut le funding négatif (bonus perçu sur Bitget) sans slippage, alors que le live subit des fees et un slippage réel d'exécution, poussant le PnL dans le rouge plus vite.  
**Diagnostic** : Comparez le `session_pnl` de l'API avec les trades dans le journal de backtest.  
**Résolution** : Relâcher très légèrement la contrainte de risque (ex: max_session_loss_percent) ou réduire le levier.  
**Prévention** : L'outil Hotfix 34 enregistre la fee exacte en USDT (fetch fill Bitget) pour aligner les calculs futurs au plus proche de la réalité.

### Procédure de déclenchement manuel
**Symptôme** : Action préventive requise lors d'un flash crash, l'opérateur veut forcer l'arrêt du bot avant que le seuil de 25% (grid) ne soit atteint.  
**Cause probable** : Volatilité extrême non gérée par le bot.  
**Diagnostic** : Observation visuelle du marché divergente.  
**Résolution** : Exécuter la commande d'urgence (kill-switch trigger) :
```bash
curl -X POST http://localhost:8000/api/executor/kill-switch/trigger \
  -H "X-API-Key: $(grep SYNC_API_KEY .env | cut -d= -f2)"
```
**Prévention** : Disposer de ce snippet accessible en 1 clic.

### Flowchart de décision : kill switch manuel vs automatique
- **Automatique** : Déclenché par les limites fixées (`grid_max_session_loss_percent` à 25% ou `max_session_loss_percent` à 5%). Nécessite un appel manuel à `/reset` après que l'opérateur ait analysé les dégâts.
- **Manuel** : Appelé manuellement via l'API `/trigger`. Stoppe immédiatement toute nouvelle prise de position. Les positions existantes se gèrent de façon autonome (TP/SL gérés par Bitget).

### Seuils
- Stratégies grid : 25% session loss (`grid_max_session_loss_percent`)
- Stratégies scalp : 5% session loss (`max_session_loss_percent`)
- Global : 45% drawdown sur fenêtre 24h
- Reset automatique du session_pnl à minuit UTC

---

## Divergence paper / live (positions différentes)

### Symptôme
Le paper affiche plus de positions que le live, ou les P&L divergent.

### Diagnostic

```bash
# 1. Comparer les positions
docker compose logs backend --since 5m | grep -E "(Sync|positions)" | tail -10

# 2. Positions fantômes créées par le warm-up ?
docker compose logs backend --since 30m | grep -E "GRID LONG level 0" | tail -10

# 3. Max grids atteint ?
docker compose logs backend --since 2h | grep "max grids" | tail -5
```

### Fix — Resynchroniser

```bash
# Option douce : restart (le sync au boot nettoie les fantômes)
docker compose stop backend
sudo rm -f data/simulator_state.json
docker compose start backend

# Vérifier après 2 min
docker compose logs backend --since 2m | grep "Sync.*terminé"
```

Le sync au boot injecte les positions live dans le paper et supprime les positions paper-only.

---

## DataEngine / WebSocket mort

### Symptôme
Prix figé dans les logs, heartbeat montre "X symbols sans données".

### Diagnostic

```bash
# 1. Heartbeat status
docker compose logs backend --since 5m | grep -E "(heartbeat|DataEngine.*actifs|stale)" | tail -10

# 2. Status per-symbol
curl -s http://localhost:8000/api/data/status | python3 -m json.tool

# 3. Tasks mortes ?
docker compose logs backend --since 5m | grep -E "(task.*morte|relancé)" | tail -5

# 4. Connexions fermées ?
docker compose logs backend --since 30m | grep -i "connection closed" | tail -5
```

### Fix
Le heartbeat devrait auto-reconnecter (full_reconnect après 5 min sans données). Si ça ne fonctionne pas :

```bash
docker compose restart backend
```

---

## Permissions Docker (sudo)

### Symptôme
`PermissionError: [Errno 13] Permission denied` lors de l'édition de fichiers dans `data/`.

### Règle
Les fichiers dans `data/` appartiennent à root (créé par le container Docker). **Toujours utiliser `sudo`** pour les éditer :

```bash
sudo python3 -c "..."
sudo nano data/executor_state.json
```

Et **toujours stopper** le backend avant d'éditer (sinon le save_state écrase vos changements) :

```bash
docker compose stop backend   # PAS restart
sudo python3 -c "..."         # éditer le fichier
docker compose start backend  # redémarrer
```

---

## Monitoring quotidien

```bash
# Résumé rapide
docker compose logs backend --since 1h | grep -E "(GRID CLOSE|GRID ENTRY|EXIT AUTONOME|kill_switch|ERROR)" | tail -20

# P&L session
curl -s http://localhost:8000/api/executor/status \
  -H "X-API-Key: $(grep SYNC_API_KEY .env | cut -d= -f2)" | python3 -m json.tool

# Santé DataEngine
curl -s http://localhost:8000/api/data/status | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Symbols: {d[\"active\"]}/{d[\"total_symbols\"]} actifs')
if d['stale'] > 0:
    stale = [s for s,v in d['symbols'].items() if v['status']=='stale']
    print(f'Stale: {stale}')
"

# Positions ouvertes
docker compose logs backend --since 1m | grep "Exit monitor.*check" | tail -1
```

---

## Checklist après restart / deploy

À vérifier après chaque `docker compose restart backend` ou `./deploy.sh` :

```bash
# 1. Boot OK ?
docker compose logs backend --since 2m | grep -E "(warm-up terminé|Executor.*grid restaurée|Sync.*terminé)" | tail -10

# 2. DataEngine connecté ?
docker compose logs backend --since 2m | grep "DataEngine.*connecté"

# 3. Exit monitor actif ?
docker compose logs backend --since 2m | grep "Exit monitor.*check" | tail -3

# 4. Pas de kill switch ?
docker compose logs backend --since 2m | grep -i "kill_switch"

# 5. Positions synchro ?
docker compose logs backend --since 2m | grep "Sync.*terminé"
```

---

## 2. Executor & gestion des ordres

### Échec du rollback sur notional pending
**Symptôme** : Rejets d'ordres en boucle pour marge insuffisante (`Insufficient balance`) alors que l'equity ou le solde d'exchange semblent corrects.  
**Cause probable** : Une race condition ou un timeout de l'API Bitget pendant l'engagement de la marge (dans `_pending_notional`) n'a pas déclenché le `raise` requis, bloquant cette marge indéfiniment (état zombie).  
**Diagnostic** :  
```bash
docker compose logs backend --since 2h | grep "_pending_notional"
```  
**Résolution** :  
1. Stopper le backend (`docker compose stop backend` — PAS restart).
2. Reset l'état mémoire : `sudo rm -f data/executor_state.json`.
3. Redémarrer le backend (`docker compose start backend`).  
**Prévention** : L'Audit 9 garantit que le code d'ouverture des ordres propage l'erreur afin de rollback `_pending_notional`.

### Stop-loss manquant sur positions grid
**Symptôme** : Une position grid est ouverte sur Bitget sans SL trigger order associé.  
**Cause probable** : Sur Bitget, les stop-loss nécessitent un appel séparé à l'API (endpoint `placePlanOrder`). Si cet appel échoue ou subit un timeout réseau juste après le fill du market order, la position reste orpheline de protection.  
**Diagnostic** :  
```bash
docker compose logs backend --since 1h | grep -i "trigger order\|SL"
```  
**Résolution** : L'ordre sera détecté comme une position "sandbox" ou resynchronisé, mais mieux vaut placer un SL manuellement sur l'interface Bitget en urgence, ou lancer le script de resynchro complet.  
**Prévention** : La logique de `GridLiveState` effectue un retry automatique lors de la pose d'ordres SL.

### Race condition entre order watcher et polling
**Symptôme** : Un fill est logué en double dans le terminal ou l'on observe une ouverture de position "double" avant correction.  
**Cause probable** : Le WebSocket push `watchOrders` et la tâche asynchrone de polling REST détectent le même fill exact au même millième de seconde, déclenchant l'enrichissement deux fois.  
**Diagnostic** :  
```bash
docker compose logs backend --since 30m | grep -i "matché" | sort
```  
**Résolution** : L'état sera réconcilié correctement sans doubler la position réelle, car l'API de gestion ignore les identifiants d'ordres déjà mis en cache local. Aucune action manuelle requise.  
**Prévention** : Maintien d'un cache LRU d'IDs de trades (hotfix).

### Ordres rejetés ou bloqués en statut inconnu côté Bitget
**Symptôme** : Logs affichant le message `FALLBACK prix paper pour order`.  
**Cause probable** : Bitget ne transmet pas toujours la "fee" (frais d'exécution) dans les flux Websocket pour les trigger orders. L'Executor ne peut pas enrichir le prix net.  
**Diagnostic** :  
```bash
docker compose logs backend --since 30m | grep -i "fill Bitget indisponible"
```  
**Résolution** : Le process est géré automatiquement. L'Executor initie la méthode de secours `fetch_fill_price` via l'API REST ; en cas de blocage (rate limit), le prix papier simulé est utilisé comme fallback.  
**Prévention** : Hotfix 34 pour chercher activement les frais d'exécution sur le serveur.

---

## 4. Connectivité Bitget

### Erreurs API (rate limit, auth, timeout)
**Symptôme** : Spams dans les logs du type `Rate limit: attente` ou erreur bloquante `ccxt.RateLimitExceeded`.  
**Cause probable** : L'API est submergée par de trop nombreuses requêtes concurrentes (polling REST ou reconnexions massives).  
**Diagnostic** :  
```bash
docker compose logs backend --since 10m | grep -i "rate limit"
```  
**Résolution** : S'assurer que le DataEngine est en mode WebSockets (`ENABLE_WEBSOCKET=true` dans `.env`) afin d'économiser les calls REST.

### Échec d'authentification sur sous-comptes
**Symptôme** : L'application crashe au boot avec l'erreur `ccxt.AuthenticationError` pour une stratégie précise.  
**Cause probable** : Les clés API définies pour un Multi-Executor (`BITGET_API_KEY_{STRATEGY}`) sont invalides, expiraient, ou n'ont pas l'habilitation "Futures & Trade".  
**Diagnostic** :  
```bash
grep -E "BITGET_API_KEY" .env
```  
**Résolution** : Connectez-vous sur Bitget, rubrique API Sub-accounts, et vérifiez/régénérez la clé concernée.  
**Prévention** : Si les clés dédiées sont absentes, le système utilise par défaut les clés globales.

### Comportements inattendus selon l'environnement (paper vs live)
**Symptôme** : Ordres de paper trading exécutés dans un environnement Sandbox qui plante.  
**Cause probable** : Le "Sandbox" de Bitget pour l'API Futures V2 comporte un bug natif (référencé `ccxt #25523`).  
**Diagnostic** : Identifier les configurations `sandbox: true` dans le repo (supprimées depuis le sprint 5a).  
**Résolution** : Tester le mode de simulation locale avec `LIVE_TRADING=false`. Pour exécuter du vrai trade, paramétrer `LIVE_TRADING=true`. Le sandbox n'est et ne sera pas réactivé.  
**Prévention** : Forcage en dur dans le code de l'argument `sandbox: False`.

---

## 5. Base de données PostgreSQL / SQLite

> *Note technique* : Le backend de Scalp Radar utilise SQLite (`aiosqlite`) via l'ORM asynchrone, mais l'architecture et les typages s'assimilent aux modes de défaillance habituels de PostgreSQL.

### Perte de connexion / pool exhausted / Database locked
**Symptôme** : Blocage de requêtes ou logs indiquant `sqlite3.OperationalError: database is locked`.  
**Cause probable** : L'accès concurrent en écriture est limité (contention), souvent lié à un buffer flush massif du DataEngine ou à la sauvegarde d'un WFO lourd.  
**Diagnostic** :  
```bash
docker compose logs backend | grep -i "database is locked"
```  
**Résolution** : SQLite possède un mécanisme de retry natif (`timeout=10`s). Si un deadlock persiste, relancer le container (`docker compose restart backend`) pour libérer les locks orphelins.  
**Prévention** : Paramètre `_FLUSH_INTERVAL = 30` implémenté pour soulager les I/O disque.

### Échec du backup quotidien automatisé
**Symptôme** : Aucun nouveau fichier de type `.bak` ou `_backup_ts` n'est créé lors des déploiements.  
**Cause probable** : Espace disque de l'hôte plein ou problème de permission Linux sur le dossier `data/` monté en volume.  
**Diagnostic** :  
```bash
ls -lh data/*_backup_*.db
df -h
```  
**Résolution** : Purger l'historique des backups obsolètes avec `sudo rm data/*_backup_2025*.db`.  
**Prévention** : Créer un cron-job d'auto-suppression des vieux dumps de DB.

### Erreurs de migration
**Symptôme** : Crash backend au démarrage avec l'erreur `no such table: xxx` (par ex. pour les tables Telegram).  
**Cause probable** : Exécution incomplète d'une méthode de `_create_xxx_table` suite à un kill brusque pendant l'initialisation.  
**Diagnostic** : Vérifier la taille et l'état du fichier `data/scalp_radar.db`.  
**Résolution** : Supprimer la base et redémarrer (attention, supprime les logs WFO locaux et les stats !). `docker compose stop backend && sudo rm data/scalp_radar.db && docker compose start backend`.

### Requêtes SQL de diagnostic utiles
À exécuter directement sur le serveur :
```bash
# Positions et résultats récents d'optimisations
sqlite3 data/scalp_radar.db "SELECT strategy_name, oos_sharpe FROM optimization_results ORDER BY timestamp DESC LIMIT 5;"

# Liste des dernières alertes Telegram
sqlite3 data/scalp_radar.db "SELECT timestamp, alert_type, message FROM telegram_alerts ORDER BY timestamp DESC LIMIT 3;"
```

---

## 6. WFO & backtesting

### Échec de compilation Numba JIT (fallback Python)
**Symptôme** : Le Worker log un message `Fast engine échoué (xxx), fallback pool/séquentiel...`.  
**Cause probable** : Les stratégies Fast Multi-Engine dépendent de Numba pour la compilation JIT. Sur certains serveurs / versions Python 3.13, le JIT peut échouer silencieusement.  
**Diagnostic** :  
```bash
docker compose logs backend --since 1h | grep -i "fallback pool"
```  
**Résolution** : Aucune. Le backtest se déroulera en mode séquentiel via un `ProcessPoolExecutor` de secours. La tâche est valide, mais prendra beaucoup plus de temps (heures au lieu de minutes).  
**Prévention** : L'intégration d'un mode de secours natif évite le crash du job manager.

### Corruption de fichier de paramètres
**Symptôme** : Erreur bloquante au démarrage : `ValueError: Stratégie inconnue` ou `yaml.scanner.ScannerError` lors du parsing config.  
**Cause probable** : Un processus automatisé (ou un développeur maladroit) a écrasé ou tronqué `config/strategies.yaml`.  
**Diagnostic** :  
```bash
cat config/strategies.yaml | tail -5
```  
**Résolution** : Lors de sa modification automatique (post-WFO), le système génère un `.bak`. Copiez ce backup :
```bash
cp config/strategies.yaml.bak.<timestamp> config/strategies.yaml
docker compose restart backend
```

### Résultats suspects (Calmar > 20, consistency score à 100%)
**Symptôme** : Le backtest affiche des scores OOS hors-normes et une "Grade A" évidente, mais le bot live perd de l'argent.  
**Cause probable** : Le problème de la "Shallow validation". L'algorithme sur-optimise (Overfit) sur une grille de fenêtres trop courte (< 24 fenêtres).  
**Diagnostic** :  
```bash
docker compose logs backend --since 24h | grep -i "Shallow validation"
```  
**Résolution** : Augmenter la fenêtre d'embargo ou imposer un `n_windows` supérieur à 24.  
**Prévention** : Une pénalité de couverture (`-X pts`) est désormais calculée sur la grade si le nombre de fenêtres est insuffisant.

---

## 7. Déploiement & infrastructure

### Échecs de deploy.sh avec points de contrôle
**Symptôme** : Lors du lancement de `./deploy.sh`, le message final est `[ERREUR] Health check échoué — rollback`.  
**Cause probable** : Une erreur de syntaxe dans le dernier commit git empêche l'API FastAPI de démarrer ou la health-route (`/health`) renvoie un statut 500.  
**Diagnostic** : Analysez les logs du processus de démarrage juste avant l'interruption :  
```bash
docker compose logs backend --tail 50
```  
**Résolution** : Le script de déploiement effectue automatiquement un fallback (`docker compose up -d --no-build`). Corrigez le code sur Github et relancez le déploiement.  

### Boucles de redémarrage Docker
**Symptôme** : La commande `docker compose ps` affiche l'état `Restarting` en boucle.  
**Cause probable** : Un crash instantané (syntax error, port 8000 déjà alloué, variable d'environnement critique manquante).  
**Diagnostic** :  
```bash
docker compose logs backend -f
```  
**Résolution** : Restaurer le paramètre `.env` de base avec `.env.example` s'il a été altéré, et vérifier que le port est libre (`lsof -i:8000`).

### Override .env non appliqué
**Symptôme** : Après avoir modifié `FORCE_STRATEGIES` ou `SELECTOR_BYPASS_AT_BOOT` dans le `.env`, aucun effet n'est constaté.  
**Cause probable** : Éditer le `.env` requiert de ré-initialiser le container pour prendre effet. Un simple redémarrage in-place peut ne pas injecter le paramètre.  
**Résolution** :  
```bash
docker compose up -d
```  
*(Rappel formel : ne modifiez jamais les fichiers `.yaml` directement en environnement de production !).*

---

## 8. Alertes Telegram

### Token invalide ou expiré
**Symptôme** : Le log d'erreur indique `Erreur envoi alerte Telegram` ou `Unauthorized`.  
**Cause probable** : Le token de bot API `TELEGRAM_BOT_TOKEN` a été révoqué par BotFather ou mal copié dans le `.env`.  
**Diagnostic** :  
```bash
docker compose logs backend | grep -i "erreur envoi alerte"
```  
**Résolution** : Re-générer un token Telegram et mettre à jour l'identifiant du canal `TELEGRAM_CHAT_ID`.

### Alertes manquantes (silence suspect)
**Symptôme** : Le bot cesse d'envoyer le snapshot de régime quotidien à 00:05 UTC.  
**Cause probable** : Les connexions vers `api.telegram.org` sont instables ou bloquées par un pare-feu serveur silencieusement.  
**Diagnostic** : Le code (Sprint 34a) intercepte les erreurs réseau pour que le process complet ne crashe pas. Regardez la DB :
```bash
curl -s http://localhost:8000/api/alerts/telegram | jq .
```  
**Résolution** : Si le log affiche des Timeout répétitifs, redémarrez le container (`docker compose restart backend`).

### Storm d'alertes en boucle
**Symptôme** : Réception de centaines d'alertes Telegram d'affilées pour le Heartbeat DataEngine.  
**Cause probable** : Une instabilité réseau WebSocket provoque des cycles infinis de "connect-disconnect" sans délai exponentiel (`backoff`).  
**Résolution** : Couper le flux DataEngine via Telegram en modifiant l'intervalle dans `.env` (`HEARTBEAT_INTERVAL=36000`), relancer et investiguer la qualité du réseau.

---

## 9. Frontend & WebSocket

### Courbes d'equity oscillantes (filtre paper vs live)
**Symptôme** : Sur le dashboard (SessionStats / SVG), le capital total clignote frénétiquement ou affiche deux valeurs l'une après l'autre de façon illogique.  
**Cause probable** : Le composant React mélange les arrays `positions` envoyés par le mode PAPER et les vrais trades de l'Executor LIVE reçus sur le même port de WebSocket.  
**Diagnostic** : Apparition cyclique rapide des badges "PAPER".  
**Résolution** : Sélectionnez une stratégie unifiée (Live ou Paper) en haut à droite. Le `useFilteredWsData` isolera proprement le tableau sans confusion. Un `F5` ou "Ctrl+R" resynchronise tout de suite.

### Déconnexion WebSocket / données figées
**Symptôme** : Le widget "Scanner" des prix se fige, plus aucune variation.  
**Cause probable** : Le `useWebSocket.js` du frontend a été déconnecté (sleep du PC client, coupure micro internet) et la routine `onclose()` backoff n'est pas parvenue à relinker le port ws://.  
**Diagnostic** : Appuyez sur `F12` (Console Web) et cherchez la mention `[WS] Déconnecté, reconnexion...`.  
**Résolution** : Recharger intégralement l'onglet du navigateur. 

### Overlay de régime absent ou incorrect
**Symptôme** : L'affichage en jauge/bouton du régime de marché marque "N/A".  
**Cause probable** : Le process `RegimeMonitor` asynchrone n'a pas pu télécharger ou calculer ses bougies BTC (daily et horaires).  
**Diagnostic** :  
```bash
docker compose logs backend --since 24h | grep -i "RegimeMonitor"
```  
**Résolution** : Attendre le passage à l'heure pile (`H:00`) pour le re-trigger de la routine de calcul interne du DataEngine.

---

## 10. Paper trading (grid_boltrend)

### Confusion compte paper vs live
**Symptôme** : Dans l'onglet Stratégies, vous constatez que le sous-badge (jaune - PAPER) affiche un P&L qui ne reflète absolument pas votre solde en temps réel, ou qu'il inclut des assets "inattendus".  
**Cause probable** : Vous consultez l'instance Simulator du compte, qui s'alimente sur tous les marchés pour des tests en aveugle, et non l'Executor.  
**Diagnostic** : S'assurer que `LIVE_TRADING=true` si on désire la réplication réelle Bitget.

### Désynchronisation des positions simulées et comportement anormal
**Symptôme** : Les positions virtuelles ne se ferment plus (TP) même quand le cours a traversé le target ou d'énormes lignes "GRID LONG level 0" prolifèrent.  
**Cause probable** : Lors des redémarrages fréquents de l'API (Warm-up phase), le bot crée des grilles de test dans l'environnement virtuel pour initier le calcul des SMA qui n'ont jamais été clôturées (Positions fantômes).  
**Diagnostic** :  
```bash
docker compose logs backend --since 1h | grep -i "warm-up"
```  
**Résolution** : Procédure de reset propre du papier de simulation complet sans perte ou risque financier sur le Live Bitget :
```bash
docker compose stop backend
sudo rm -f data/simulator_state.json
docker compose start backend
```
*(Ceci va détruire l'historique du journal paper-trading et forcer l'Executor à re-copier l'état propre depuis Bitget).*
