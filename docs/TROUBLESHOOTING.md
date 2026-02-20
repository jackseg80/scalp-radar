# TROUBLESHOOTING.md — Guide de dépannage Scalp Radar

> Toutes les commandes s'exécutent sur le serveur (`ssh jack@192.168.1.200`, dans `~/scalp-radar`).

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

## Kill switch live déclenché

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
Les fichiers dans `data/` appartiennent à root (créés par le container Docker). **Toujours utiliser `sudo`** pour les éditer :

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
