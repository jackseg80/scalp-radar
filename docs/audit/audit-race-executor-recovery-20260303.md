# Audit Triple P0 — Race Conditions, Multi-Executor, Crash Recovery

**Date :** 2026-03-03
**Scope :** 3 audits P0 consolidés — concurrence async, isolation multi-executor, récupération crash
**Fichiers audités :** executor.py, executor_manager.py, risk_manager.py, state_manager.py, data_engine.py, simulator.py, boot_reconciler.py, sync.py, order_monitor.py, watchdog.py, server.py, rate_limiter.py, routes API

### Fixes appliqués (Phase 1)

| Fix | Issue | Fichier modifié | Résolution |
|-----|-------|----------------|------------|
| P0-CR-2 | Close error supprimait grid_states | `executor.py:1834` | Return sans supprimer l'état — exit monitor retente |
| P0-CR-1 | Crash entry→SL sans state sauvé | `executor.py:1632`, `server.py:238` | `_save_state_now()` callback après chaque fill |
| P0-RC-2 | StateManager snapshot non-atomique | `state_manager.py:79` | **Faux positif** — boucle déjà synchrone, commentaire ajouté |
| P0-RC-1 | Mutation partagée strategy._config | `executor.py:set_strategies()` | Deep copy des configs stratégie dans l'Executor |
| P0-ME-1/2 | Symboles partagés + clés partagées | `server.py:219` | Hard block au boot si chevauchement détecté |

**Tests :** 2226 passed (5 pré-existants) après chaque fix.

---

## Résumé exécutif

| Catégorie | P0 | P1 | P2 | Total |
|-----------|----|----|----|----|
| Race Conditions Async | 2 | 5 | 5 | 12 |
| Multi-Executor Isolation | 2 | 4 | 4 | 10 |
| Crash Recovery | 4 | 6 | 7 | 17 |
| **TOTAL (dédupliqués)** | **6** | **12** | **13** | **31** |

> Note : certains bugs Multi-Executor sont **dormants** tant qu'un seul executor live tourne (état actuel : grid_atr uniquement). Ils s'activent dès qu'une 2e stratégie est promue live avec clés partagées.

---

## P0 — CRITIQUES (6 issues)

### P0-RC-1 : Mutation partagée de `strategy._config` entre Simulator et Executor

**Fichiers :** `server.py:230-236`, `executor.py:882,895,1172,1179`, `simulator.py:1035-1044`

**Description :** `simulator.get_strategy_instances()` retourne les **mêmes objets** `runner._strategy` partagés entre Paper et Live. L'Executor et le Simulator mutent `strategy._config.min_profit_pct` et `min_grid_spacing_pct` avec un pattern save/restore en `try/finally`. Si le Simulator fait un `await` (DB write) pendant la mutation, l'Executor peut démarrer avec la mauvaise valeur.

**Impact :** TP/SL calculé avec les paramètres d'un autre symbol. Sorties prématurées ou manquées = pertes directes.

**Fix :** Ne jamais muter `strategy._config`. Passer `min_profit_pct` et `min_grid_spacing_pct` comme arguments aux méthodes `should_close_all()` / `compute_grid()`, ou cloner la config.

---

### P0-RC-2 : StateManager lit runner state sans synchronisation pendant dispatch candle

**Fichiers :** `state_manager.py:79-135`, `simulator.py:1989-2093`

**Description :** `save_runner_state()` (toutes les 60s) lit `runner._capital`, `runner._positions`, `runner._kill_switch_triggered`. Puis fait `await asyncio.to_thread(self._write_json_file, ...)` qui rend la main à l'event loop. Pendant ce temps, `_dispatch_candle()` peut fermer des positions, modifier le capital, etc.

**Impact :** State persisté incohérent (capital d'avant + positions d'après). Après crash : positions fantômes et capital incorrect.

**Fix :** Faire un snapshot atomique (deep copy) de tous les runners AVANT le premier `await`.

---

### P0-ME-1 : Aucun guard global de marge inter-executors

**Fichiers :** `executor.py:633,852,930-949`, `risk_manager.py:113-126`

**Description :** Chaque Executor fetch le solde **total** du compte et applique `max_margin_ratio: 0.70` indépendamment. Avec 2 executors à clés partagées : chacun croit pouvoir utiliser 70% de 1646 USDT = total possible 2304 USDT (140% du compte).

**Impact :** Sur-utilisation de marge → risque liquidation Bitget.

**Condition :** Dormant tant qu'un seul executor live tourne.

**Fix :** Diviser `available_balance` par le nombre d'executors partageant les mêmes clés, ou ajouter un coordinateur global de marge dans ExecutorManager.

---

### P0-ME-2 : Conflit de positions Bitget en mode one-way sur symboles partagés

**Fichiers :** `executor.py:697-701,1464-1473`, `boot_reconciler.py:36-43`

**Description :** Bitget en mode one-way = UNE position par symbole par sous-compte. Si `grid_atr` et `grid_multi_tf` tradent BTC/USDT sur le même compte : positions mergées, SL écrasés, close ferme TOUT.

**Impact :** Corruption SL, fermeture involontaire, désync état. 7 symboles en commun entre grid_atr et grid_multi_tf.

**Condition :** Dormant tant qu'un seul executor live.

**Fix immédiat :** Transformer le warning `server.py:219-225` en **hard block au boot** si chevauchement de symboles détecté entre executors à clés partagées.

---

### P0-CR-1 : Crash entre entry fill et SL placement — position sans stop-loss

**Fichiers :** `executor.py:1547-1621`

**Description :** `_open_grid_position()` place l'ordre d'entrée (ligne 1550), crée le `GridLiveState` (1589-1596), puis appelle `_update_grid_sl()` (1621). Si crash entre le fill et le SL : position ouverte sur Bitget sans SL. Le state n'a pas été sauvé (prochain save dans ~60s). La fenêtre d'exposition peut durer des minutes à des heures.

Au reboot, `boot_reconciler` détecte la position et replace un SL — mais seulement si le state a été sauvé. Si crash avant le save : `sync.py:_populate_grid_states_from_exchange()` crée un state avec `sl_price=0.0` et le reconciler le détecte et place un SL.

**Impact :** Position live sans SL pendant crash→reboot. Flash crash = perte non-contrôlée.

**Fix :** `await self._save_state_immediately()` juste après le fill (ligne 1609), AVANT le SL placement.

---

### P0-CR-2 : `_close_grid_cycle` erreur → supprime l'état alors que la position est toujours ouverte

**Fichiers :** `executor.py:1834-1839`

**Description :** Si le market close order échoue (exception), le code supprime quand même la position de `_grid_states`, `_risk_manager`, et `_record_grid_close`. La position reste ouverte sur Bitget mais invisible au bot.

**Impact :** Position orpheline sur l'exchange, sans monitoring, sans SL management, sans exit strategy.

**Fix :** Sur erreur de close, **NE PAS** supprimer de `_grid_states`. Logger l'erreur et laisser l'exit monitor retenter au cycle suivant.

---

## P1 — IMPORTANTS (12 issues)

### P1-RC-3 : `_grid_states` et `_positions` modifiés par 3+ tâches concurrentes sans lock

**Fichiers :** `executor.py:186-187`, `order_monitor.py:57,87,120-131`

Les dicts `_positions` et `_grid_states` sont mutés par : `_on_candle()`, `_watch_orders_loop()`, `_poll_positions_loop()`, `_exit_monitor_loop()`. Les `await` (appels Bitget) au milieu des mutations créent des fenêtres de race condition.

**Fix :** `asyncio.Lock()` par Executor pour toute modification de `_grid_states`/`_positions`.

---

### P1-RC-4 : Executor accède directement aux buffers internes du DataEngine

**Fichiers :** `executor.py:1097-1147`, `data_engine.py:794-836`

L'Executor lit `self._data_engine._buffers` directement (pas via `get_data()` qui retourne une copie). Le callback WS peut modifier le buffer in-place ou le tronquer pendant la lecture.

**Fix :** Utiliser `get_data()` ou passer le prix dans le callback candle.

---

### P1-RC-5 : Fire-and-forget tasks pour les alertes kill switch

**Fichiers :** `risk_manager.py:194,253`

`asyncio.create_task(self._notifier.notify_anomaly(...))` sans tracking. Exceptions perdues silencieusement. Alertes kill switch (les plus critiques) peuvent ne jamais arriver.

**Fix :** Stocker les tasks dans `_pending_alert_tasks` et les nettoyer.

---

### P1-RC-7 : API kill switch reset mute l'état sans synchronisation

**Fichiers :** `simulator_routes.py:117-156`, `executor_routes.py:73-155`

Les endpoints POST mutent directement `_kill_switch_triggered` et `_stats` pendant que le dispatch candle tourne.

**Fix :** Pattern command queue : le dispatch loop applique le reset au prochain cycle.

---

### P1-RC-8 : `full_reconnect()` ne cancel pas `_flush_task` et `_heartbeat_task`

**Fichiers :** `data_engine.py:227-229,388-438`

Après reconnect WS, le flush et heartbeat continuent avec l'ancien contexte. Le heartbeat peut trigger un deuxième reconnect → boucle.

**Fix :** Cancel/recréer `_flush_task` et `_heartbeat_task` dans `full_reconnect()`.

---

### P1-ME-3 : Kill switch per-executor sans propagation inter-executor

**Fichiers :** `risk_manager.py:55,148-203`, `executor_manager.py:127-149`

Chaque executor a son propre kill switch. Si executor A déclenche, executor B continue. En flash crash : pertes aggravées.

**Fix :** Flag kill switch partagé dans ExecutorManager, propagation immédiate.

---

### P1-ME-4 : Capital initial = solde total du compte par executor

**Fichiers :** `executor.py:399`, `risk_manager.py:172`

Chaque executor utilise le solde complet comme `_initial_capital`. Le seuil kill switch session (25%) est calculé contre le total → seuil effectif 50% avec 2 executors.

**Fix :** Diviser `initial_capital` par le nombre d'executors sur le même compte.

---

### P1-ME-5 : RateLimiter implémenté mais jamais utilisé

**Fichiers :** `rate_limiter.py` (103 lignes), `exchanges.yaml:8-15`

Le `RateLimiter` est entièrement codé mais **jamais importé/instancié** dans le codebase. Chaque executor utilise le rate limiter interne ccxt (per-instance, pas per-compte). Avec clés partagées → dépassement des limites Bitget.

**Fix :** Intégrer le RateLimiter existant ou le supprimer comme code mort.

---

### P1-ME-6 : Double comptage du solde dans ExecutorManager

**Fichiers :** `executor_manager.py:62-70`

`exchange_balance` additionne le solde de chaque executor. Avec clés partagées : solde affiché 2× le réel. Dashboard et décisions faussés.

**Fix :** Grouper par sous-compte, prendre le solde une seule fois par groupe.

---

### P1-CR-3 : Kill switch live non persisté immédiatement

**Fichiers :** `risk_manager.py:183-202`

Le flag `_kill_switch_triggered = True` est en mémoire uniquement. Sauvé au prochain periodic save (jusqu'à 60s). Crash dans la fenêtre → kill switch perdu → trading reprend au reboot.

**Fix :** Callback vers StateManager pour forcer un flush immédiat.

---

### P1-CR-4 : Boot reconciler ignore les positions orphelines (mono)

**Fichiers :** `executor.py:1254-1339`, `boot_reconciler.py:93-104`

Si crash entre entry fill et création de `_positions[sym]` : au reboot, reconciler détecte "position orpheline sur exchange" et log "Non touchée." — AUCUN SL placé.

**Fix :** Sur détection orphan : placer un SL protectif, ou fermer la position, ou au minimum créer un tracking local + SL.

---

### P1-CR-5 : Pas de guard contre instances concurrentes du bot

**Fichiers :** `server.py:164-249`

Aucun lock file ni mécanisme empêchant deux instances de tourner simultanément sur le même compte Bitget.

**Fix :** Lock file `data/.executor.lock` avec PID au startup.

---

## P2 — MINEURS (13 issues)

| ID | Fichier | Description |
|----|---------|-------------|
| P2-RC-10 | `data_engine.py:843-849` | Callbacks candle exécutés séquentiellement — un callback lent bloque tout |
| P2-RC-11 | `watchdog.py:168,172` | Watchdog accède directement aux attributs internes des executors |
| P2-RC-12 | `data_engine.py:843-849` | Exception dans callback executor avalée silencieusement (pas d'alerte) |
| P2-RC-13 | `data_engine.py:227-229` | `_flush_task` non inclus dans `_tasks`, pas cancel dans full_reconnect |
| P2-RC-14 | `simulator.py:380-387` | `run_in_executor` pour DB writes sans tracking du Future |
| P2-ME-7 | `risk_manager.py:101-111` | Correlation group check per-executor seulement |
| P2-ME-8 | `boot_reconciler.py:36-43` | Réconciliation boot non filtrée aux assets de la stratégie |
| P2-ME-9 | `executor.py:665-705` | Race condition setup leverage sur symboles partagés |
| P2-ME-10 | `rate_limiter.py` | Code mort (103 lignes jamais utilisées) |
| P2-CR-6 | `executor.py:2170-2182` | `_fetch_exit_price()` retourne 0.0 en cas d'échec |
| P2-CR-7 | `executor_manager.py:169-175` | `stop_all()` sans timeout per-executor |
| P2-CR-8 | `watchdog.py` | Pas de check positions sans SL |
| P2-CR-9 | `executor.py:1649-1698` | Cancel-then-place SL crée fenêtre sans protection |
| P2-CR-10 | `state_manager.py:266-267` | Pas d'escalade sur échecs de sauvegarde consécutifs |
| P2-CR-11 | `database.py:1556-1557` | Index unique peut causer échec silencieux de persistence trade |
| P2-CR-12 | `boot_reconciler.py:154-177` | Pas de gestion SL partiellement rempli au reboot |

---

## Points positifs confirmés

| Aspect | Détail |
|--------|--------|
| Écriture state atomique | `state_manager.py:180-193` : tmp + `os.fsync()` + `os.replace()` — gold standard |
| SL retry + emergency close | `executor.py:1376-1410` : 3 retries, puis market close — Rule #1 respectée |
| Boot reconciliation | `boot_reconciler.py` : détecte orphans, closes pendant downtime, SL manquants |
| Shutdown timeout | `server.py:67-74` : `_safe_stop` avec 30s timeout par composant |
| WAL + write lock | `database.py` : asyncio.Lock pour sérialiser les écritures SQLite |
| Vérification résiduelle | `executor.py:2325-2375` : check Bitget après chaque grid close |
| Instance ccxt isolée | Chaque Executor crée sa propre instance ccxt |
| State files isolés | `data/executor_{strategy}_state.json` par executor |

---

## Locks existants dans le codebase (seulement 3)

| Lock | Fichier | Protège |
|------|---------|---------|
| `Database._write_lock` | `database.py` | Écritures SQLite |
| `RateLimiter._lock` | `rate_limiter.py` | Token bucket (jamais utilisé) |
| `JobManager._semaphore` | `job_manager.py` | Jobs WFO concurrents |

**Cruellement manquants :**
- Lock sur `Executor._grid_states` / `_positions`
- Lock/snapshot sur runner state dans `StateManager.save_runner_state()`
- Lock sur `strategy._config` pendant les évaluations per-asset

---

## Plan d'action recommandé

### Phase 1 — Quick wins (1-2 sprints)

| # | Action | Issues résolues | Effort |
|---|--------|----------------|--------|
| 1 | Hard block boot si symboles partagés + clés partagées | P0-ME-1, P0-ME-2 | Léger |
| 2 | `_save_state_immediately()` après entry fill | P0-CR-1 | Léger |
| 3 | Ne pas supprimer `_grid_states` sur erreur close | P0-CR-2 | Léger |
| 4 | Snapshot atomique dans `save_runner_state()` | P0-RC-2 | Moyen |
| 5 | Passer per-asset params en argument (pas muter config) | P0-RC-1 | Moyen |

### Phase 2 — Hardening (2-3 sprints)

| # | Action | Issues résolues | Effort |
|---|--------|----------------|--------|
| 6 | asyncio.Lock dans Executor pour grid_states | P1-RC-3 | Moyen |
| 7 | Flush immédiat kill switch | P1-CR-3 | Léger |
| 8 | Boot reconciler : SL sur positions orphelines | P1-CR-4 | Moyen |
| 9 | Lock file anti double-instance | P1-CR-5 | Léger |
| 10 | Kill switch propagation inter-executor | P1-ME-3, P1-ME-4 | Moyen |

### Phase 3 — Polish (optionnel)

- Intégrer ou supprimer RateLimiter
- Watchdog check SL manquants
- Place-before-cancel pour SL updates
- Callbacks DataEngine en parallèle
- Escalade alertes sur échecs save consécutifs
