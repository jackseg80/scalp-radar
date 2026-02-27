# Audit Sprint 57 — Moteur Live/Paper : Diagnostic Complet

**Date :** 2026-02-27
**Objectif :** Identifier les bugs potentiels AVANT déploiement Sprint 56 sur 26 assets (15 grid_atr + 11 grid_multi_tf)
**Scope :** executor.py, risk_manager.py, executor_manager.py, state_manager.py, data_engine.py, database.py, watchdog.py, notifier.py, telegram.py, grid_multi_tf.py, grid_atr.py, sync_bitget_trades.py

---

## 1. Flux signal → ordre (executor.py)

### Sizing

| Aspect | Verdict | Détail |
|--------|---------|--------|
| Formule de base | ✅ OK | `quantity = size_fraction × allocated_balance × leverage / entry_price` (L860-862) |
| Capital source | ✅ OK | `_ensure_balance()` fetch le solde Bitget réel (L556-571), pas le capital config |
| Division par nb_assets | ✅ OK | `allocated_balance = available_balance / nb_assets` (L799), empêche 1 asset de consommer toute la marge |
| per_asset overrides | ✅ OK | `_get_per_asset_float()` résout `min_grid_spacing_pct`, `min_profit_pct` par symbol (L588-599) |
| Margin guard 70% | ✅ OK | Sprint 56 : vérifie `(total_margin + level_margin) / available > max_margin_ratio` avant chaque entrée (L867-879) |

### Types d'ordres

| Type | Verdict | Détail |
|------|---------|--------|
| Entrées | ✅ OK | Market orders — `create_order(futures_sym, "market", side, quantity)` (L1473) |
| SL | ✅ OK | Market trigger server-side — `triggerPrice + triggerType=mark_price + reduceOnly` (L1310-1316) |
| TP (mono) | ⚠️ Risque | Limit trigger — mais avec `triggerPrice` ET `limit price` (L1344-1350). Si l'exchange exécute le trigger mais la limit ne fill pas (gap), la position reste sans protection |
| TP (grid) | ✅ OK | Pas de TP server-side. TP = SMA dynamique, détecté par `_check_grid_exit()` (L964) toutes les 60s |
| Close cycles | ✅ OK | Market `reduceOnly` (L1717-1719) |

### Gestion des erreurs

| Scénario | Verdict | Détail |
|----------|---------|--------|
| Bitget rejette l'entrée | ✅ OK | Exception capturée, log error, return sans position (L1477-1479) |
| SL impossible | ✅ OK | 3 retries (L1308-1333), si échec total → emergency close market + alerte Telegram (L1244-1262) |
| TP impossible | ✅ OK | Log warning, position reste ouverte avec SL uniquement (L1356-1358) |
| Insufficient balance | ✅ OK | `pre_trade_check()` vérifie marge libre avant chaque 1er niveau (L1449-1456) |
| Min notional | ✅ OK | `_round_quantity()` applique `min_amount` du market (L2766-2772) |
| Timeout / erreur réseau | ⚠️ Risque | Pas de retry sur l'ordre d'entrée — un timeout peut laisser un ordre "pending" côté Bitget sans suivi local |

### Concurrence (2 assets simultanés)

| Aspect | Verdict | Détail |
|--------|---------|--------|
| Async single-loop | ✅ OK | Python asyncio = pas de parallélisme réel, les ordres sont séquentiels dans la boucle `_on_candle` |
| Anti double-trigger | ✅ OK | `_pending_levels` set + `_pending_notional` tracker (L846-883) |
| Marge partagée | ⚠️ Risque | `_pending_notional` est mis à jour manuellement et réinitialisé au prochain `refresh_balance()`. En cas de cascade rapide (10+ assets trigger en 1 min), le compteur peut être imprécis |

---

## 2. Gestion des positions — GridLiveState + persistence

### État des positions

| Aspect | Verdict | Détail |
|--------|---------|--------|
| Mémoire | ✅ OK | `_grid_states: dict[str, GridLiveState]` — toutes les positions en RAM (L182) |
| Persistence fichier | ✅ OK | `get_state_for_persistence()` sauvegarde tous les champs (L2894-2947), `restore_positions()` les restaure (L2949-3019) |
| Persistence DB | ✅ OK | Chaque entry/close persisté via `_persist_live_trade()` (best-effort, L313-352) |
| Atomic write | ✅ OK | `state_manager._write_json_file()` écrit en `.tmp` puis `os.replace()` |
| Fréquence save | ⚠️ Risque | Toutes les 60s — jusqu'à 60s de state perdus en cas de crash |

### Récupération après restart Docker

| Aspect | Verdict | Détail |
|--------|---------|--------|
| Restauration state | ✅ OK | `restore_positions()` reconstruit `_positions` et `_grid_states` depuis le JSON (L2949-3019) |
| Réconciliation Bitget | ✅ OK | `_reconcile_on_boot()` compare état local vs positions Bitget réelles (L2282-2329) |
| Position orpheline exchange | ✅ OK | Détectée et loguée, non touchée (conservatrice) — notification Telegram (L2350-2361) |
| Position fermée pendant downtime | ✅ OK | Détectée, P&L estimé via `fetch_my_trades`, comptabilisée dans le kill switch (L2363-2393) |
| SL exécuté pendant downtime | ✅ OK | `_reconcile_grid_symbol()` vérifie le status du SL order (L2411-2428) |
| Ordres orphelins | ✅ OK | `_cancel_orphan_orders()` nettoie les triggers sans position associée (L2474-2537) |

### WebSocket down pendant 5 minutes

| Aspect | Verdict | Détail |
|--------|---------|--------|
| SL server-side | ✅ OK | Les SL sont des trigger orders Bitget (`triggerPrice + mark_price`), ils survivent au crash/WS down |
| TP grid (SMA) | 🔴 Bug potentiel | Le TP grid est **client-side** (exit monitor toutes les 60s). Si WS down + bot alive, le prix est stale → `_check_grid_exit()` utilise un fallback `fetch_ticker` REST après 2 min de stale (L1037-1063). Mais si le bot crash, le TP n'est plus surveillé → seul le SL server-side protège |
| Détection stale | ✅ OK | DataEngine heartbeat détecte silence >5 min → `restart_dead_tasks()` puis `full_reconnect()` (data_engine.py) |

### Réconciliation local vs Bitget

| Aspect | Verdict | Détail |
|--------|---------|--------|
| Boot reconciliation | ✅ OK | 4 cas gérés : both open, orpheline exchange, fermée pendant downtime, clean (L2331-2397) |
| Runtime reconciliation | ✅ OK | `_poll_positions_loop()` vérifie toutes les 5s que les positions locales existent encore sur Bitget (L2131-2146) |
| Position fermée par liquidation | ⚠️ Risque | Détectée par polling comme "position fermée côté exchange" (L2159-2183), exit_reason = "unknown". Le P&L est estimé via `fetch_my_trades` last 5 trades — peut être imprécis si la liquidation a généré >5 fills |

---

## 3. Multi-stratégie isolation

### Capital et sous-comptes

| Aspect | Verdict | Détail |
|--------|---------|--------|
| API keys par stratégie | ✅ OK | `config.get_executor_keys(strategy_name)` retourne clés spécifiques `BITGET_API_KEY_{STRATEGY}` (L361-363) |
| Sous-comptes Bitget | ✅ OK | Chaque Executor a sa propre instance ccxt avec ses clés (L369-376). Isolation complète au niveau exchange |
| Balance par Executor | ✅ OK | Chaque Executor fetch son propre `fetch_balance()` (L388-398) |
| Symboles par stratégie | ✅ OK | `_per_asset_filter` limite chaque Executor aux assets de sa stratégie (L403-409) |

### Partage du Simulator

| Aspect | Verdict | Détail |
|--------|---------|--------|
| Paper runners partagés | ⚠️ Risque | Grid_atr et grid_multi_tf partagent le même Simulator, mais chaque runner a son propre capital paper isolé. Si les deux tradent DYDX : 2 runners paper indépendants |
| Live capital isolation | ✅ OK | Chaque Executor a son propre sous-compte Bitget. Pas de partage de capital live entre stratégies |
| Indicateurs | ✅ OK | Chaque runner calcule ses propres indicateurs via `compute_live_indicators()`. grid_multi_tf a ses indicateurs 4h spécifiques |

### Margin guard

| Aspect | Verdict | Détail |
|--------|---------|--------|
| Scope | ✅ OK | Le margin guard 70% est **par Executor** (il vérifie `available_balance` de son sous-compte, L867-879) |
| Max live grids | ✅ OK | `max_live_grids=4` par Executor (L1401-1412), pas global |
| Max concurrent positions | ✅ OK | `risk_manager.pre_trade_check()` vérifie `max_concurrent_positions` par Executor (risk_manager.py L97-99) |

---

## 4. Kill switch et safety

### Architecture kill switch

| Niveau | Seuil | Scope | Verdict | Détail |
|--------|-------|-------|---------|--------|
| Session (per-runner) | 25% (grid) / 5% (scalp) | Par stratégie | ✅ OK | `_session_pnl` accumulé à chaque trade, reset quotidien minuit UTC (risk_manager.py L149-202) |
| Global (sliding window) | 45% | Toutes stratégies | ✅ OK | `record_balance_snapshot()` toutes les 5 min, calcul drawdown peak→current sur 24h (risk_manager.py L210-262) |

### Comportement au trigger

| Aspect | Verdict | Détail |
|--------|---------|--------|
| Blocage nouveaux trades | ✅ OK | `is_kill_switch_triggered` vérifié dans `pre_trade_check()` (L88) ET aux niveaux 2+ grid (executor.py L1459-1464) |
| Fermeture positions existantes | ⚠️ Risque | Le kill switch **ne ferme PAS** les positions existantes. Il bloque uniquement les nouvelles entrées. Les positions ouvertes continuent avec leurs SL/TP normaux |
| Alerte Telegram | ✅ OK | `asyncio.create_task()` pour fire-and-forget (risk_manager.py L194). Mais... |
| 🔴 Bug `create_task` | 🔴 Bug | `asyncio.get_event_loop().create_task()` (risk_manager.py L194, L253) — si appelé hors event loop actif (ex: pendant shutdown), lève `RuntimeError`. Les alertes kill switch critiques pourraient ne pas être envoyées |
| Reset | ✅ OK | Endpoint `POST /api/executor/kill-switch/reset` disponible (L2320) |
| Persistence | ✅ OK | `_kill_switch_triggered` sauvegardé dans le state et restauré au boot (risk_manager.py L266-309) |

### Dead man's switch

| Aspect | Verdict | Détail |
|--------|---------|--------|
| Watchdog | ✅ OK | Boucle 30s vérifie : WS connecté, data freshness, strategies actives, disk space, executor connected (watchdog.py) |
| Auto-recovery WS | ✅ OK | Stale >5 min → restart tasks, >10 min → full_reconnect (data_engine.py heartbeat) |
| Bot crash | ⚠️ Risque | Si le bot crash et ne redémarre pas : les SL server-side protègent, mais les TP grid (client-side SMA) ne sont plus surveillés. Les positions restent ouvertes jusqu'au SL |
| Docker healthcheck | ⚠️ Risque | Non audité — si le container crash, Docker Compose doit `restart: always` pour relancer automatiquement |

### SL server-side vs client-side

| Type | Nature | Survit au crash | Verdict |
|------|--------|-----------------|---------|
| SL (tous) | Server-side Bitget trigger (`mark_price`) | ✅ Oui | ✅ OK |
| TP (mono) | Server-side Bitget limit trigger | ✅ Oui | ✅ OK |
| TP (grid) | Client-side, `_check_grid_exit()` toutes les 60s | ❌ Non | ⚠️ Risque — seul le SL protège en cas de crash |

---

## 5. Supertrend direction_flip (grid_multi_tf spécifique)

### Mécanisme du flip

| Aspect | Verdict | Détail |
|--------|---------|--------|
| Détection | ✅ OK | `should_close_all()` vérifie `st_direction` vs direction positions (grid_multi_tf.py L217-224) |
| Anti-lookahead | ✅ OK | Resampling 4h utilise la direction du bucket **précédent** complété via `np.searchsorted(..., side="left") - 1` (grid_multi_tf.py L347-353) |
| Close en live | ✅ OK | Market order immédiat via `_close_grid_cycle()` (executor.py L1693-1748) |
| Close en backtest | ✅ OK | Exit à `candle.close` (multi_engine.py L189) |

### Timing du flip

| Aspect | Verdict | Détail |
|--------|---------|--------|
| Détection live | ⚠️ Risque | Exit monitor vérifie toutes les 60s (`_EXIT_CHECK_INTERVAL = 60`). Si le ST flip à 12:01, le close peut n'arriver qu'à 12:02 — 1 min de latence max |
| Weekend / basse liquidité | ⚠️ Risque | **Pas de slippage guard spécifique** au flip. Le market order est envoyé sans vérifier le spread. Le slippage réel dépend du carnet d'ordres Bitget à ce moment |
| Cooldown post-flip | 🔴 Bug | **Aucun cooldown** après un flip. `grid_multi_tf` n'a pas de paramètre `cooldown_candles`. Après un close LONG, une entrée SHORT peut se faire **sur la même candle** (grid_multi_tf.py compute_grid ne vérifie pas de cooldown). Risque de churning aller-retour pendant les whipsaws du Supertrend |

### Paramètres manquants vs grid_atr

| Paramètre | grid_atr | grid_multi_tf | Risque |
|-----------|----------|---------------|--------|
| `min_grid_spacing_pct` | ✅ Oui | ❌ Absent | En basse volatilité (ATR comprimé), les niveaux grid peuvent se rapprocher dangereusement → tous les niveaux fill d'un coup |
| `cooldown_candles` | ✅ Oui | ❌ Absent | Churning après flip ST |
| `max_hold_candles` | ✅ Oui | ❌ Absent | Positions zombies possibles (prix entre SMA et SL indéfiniment) |
| `min_profit_pct` | ✅ Oui | ❌ Absent | TP sur tout touch de SMA, même pour profit quasi-nul |

---

## 6. Edge cases critiques

### Asset delisted ou suspendu

| Aspect | Verdict | Détail |
|--------|---------|--------|
| Entrée | ✅ OK | `create_order` échouerait → exception capturée → log error, pas de position créée |
| Position ouverte | ⚠️ Risque | Si l'asset est suspendu avec une position ouverte, le SL trigger Bitget ne s'exécuterait pas non plus. Le polling détecterait la position comme "toujours ouverte" (contracts > 0). L'exit monitor ne pourrait pas fermer (market orders rejetés). **Aucune alerte spécifique pour ce cas** |
| DataEngine | ✅ OK | `_watch_symbol()` détecte "does not have market symbol" et abandonne le symbol (data_engine.py L653) |

### Funding rate extrême (>0.1%)

| Aspect | Verdict | Détail |
|--------|---------|--------|
| Alerte | ❌ Absent | Pas d'alerte ni d'action pour les funding rates extrêmes. Le DataEngine poll les funding rates (pour `grid_funding`) mais aucun guard global ne vérifie un funding excessif sur les positions ouvertes |
| Impact | ⚠️ Risque | En cross-margin, un funding rate extrême peut grignoter la marge progressivement. Les positions grid qui durent plusieurs jours accumulent les funding charges sans visibilité |

### Margin call en cross-margin

| Aspect | Verdict | Détail |
|--------|---------|--------|
| Détection | ⚠️ Risque | Pas de détection spécifique du margin call Bitget. Le `_balance_refresh_loop()` détecte un changement >10% du solde (L502-507) mais ne distingue pas un margin call d'un trade normal |
| Protection | ✅ Partiel | Le kill switch global 45% devrait se déclencher avant un margin call si les SL fonctionnent. Mais si 26 positions SL touchent en cascade, le drawdown peut dépasser le kill switch threshold |

### Double exécution (TP limit fill + close bot simultané)

| Aspect | Verdict | Détail |
|--------|---------|--------|
| Positions mono | ✅ OK | Le close bot utilise `reduceOnly: True` (L1719). Si le TP a déjà fermé la position, le market close échoue gracieusement avec "no position to reduce" |
| Positions grid | ✅ OK | Pas de TP server-side pour les grids, donc pas de race condition possible |
| Résiduel check | ✅ OK | `_verify_no_residual_position()` vérifie 1.5s après le close qu'il ne reste rien (L2691-2741) |

### Partial fill

| Aspect | Verdict | Détail |
|--------|---------|--------|
| Entrée | ⚠️ Risque | `filled_qty = float(entry_order.get("filled") or quantity)` (L1481) — si le market order est partiellement rempli, on utilise `filled_qty` pour la suite. Mais le SL est placé avec `filled_qty`, pas la quantité demandée → OK |
| Close | ✅ OK | `_handle_partial_close_fill()` détecte les fills partiels, envoie un 2ème market order sur le résidu, alerte PARTIAL_FILL (L2642-2689) |

---

## 7. Logging et monitoring

### Trades logués en DB

| Aspect | Verdict | Détail |
|--------|---------|--------|
| Entries | ✅ OK | `_persist_live_trade("entry", ...)` à chaque entry grid/mono (executor.py L1560-1570) |
| Closes (TP/SL/signal) | ✅ OK | `_persist_live_trade("tp_close"/"sl_close", ...)` avec P&L (L1796-1807) |
| SL exchange | ✅ OK | `_handle_grid_sl_executed()` persiste avec context `"grid_sl_global"` (L1876-1890) |
| Réconciliation | ⚠️ Risque | Les positions fermées pendant downtime sont comptabilisées dans le risk manager mais **pas persistées en DB** comme live_trade (L2363-2393). Le `sync_bitget_trades.py` doit rattraper ces trades |
| Best-effort | ⚠️ Risque | Toutes les insertions DB sont `try/except` avec log warning (L351-352). Un échec DB silencieux ne bloque pas le trading mais peut créer des trous dans l'historique |

### sync_bitget_trades.py

| Aspect | Verdict | Détail |
|--------|---------|--------|
| Dédup | ✅ OK | Check `order_id` in memory set avant insertion (L432) |
| Fill aggregation | ✅ OK | VWAP sur fills multiples du même order_id (L91-134) |
| Close burst merge | ✅ OK | Fusion closes <5 min pour SL multi-niveaux (L157-205) |
| 🔴 Bug order_id NOT UNIQUE | 🔴 Bug | La table `live_trades` a un INDEX sur `order_id` mais **pas de contrainte UNIQUE** (database.py L1533-1534). Si le script tourne 2× sans `--purge`, les trades sont dédupliqués en mémoire mais la DB pourrait avoir des doublons de runs précédentes |
| Cycles ouverts | ⚠️ Risque | Les cycles non fermés à la fin de la fenêtre sync sont logués mais pas insérés. Re-run nécessaire après fermeture |

### Alertes Telegram

| Scénario | Couvert | Cooldown | Détail |
|----------|---------|----------|--------|
| Kill switch per-strategy | ✅ | 1h | risk_manager.py L191-202 |
| Kill switch global | ✅ | 1h | risk_manager.py L249-262 |
| SL placement failed | ✅ | 0 (toujours) | notifier.py L175-182 — CRITICAL |
| Partial fill | ✅ | 1 min | notifier.py, AnomalyType.PARTIAL_FILL |
| WS disconnected | ✅ | 30 min | watchdog.py → notifier |
| Data stale | ✅ | 30 min | watchdog.py + data_engine heartbeat |
| All strategies stopped | ✅ | 5 min | watchdog.py |
| Executor disconnected | ✅ | 5 min | watchdog.py |
| Position orpheline | ✅ | 0 | reconciliation boot uniquement |
| Leverage divergence | ✅ | 0 | boot uniquement |
| Funding rate extrême | ❌ | — | **Non couvert** |
| Margin call | ❌ | — | **Non couvert** |
| Asset suspendu | ❌ | — | **Non couvert** |
| Position zombie (>Xh sans mouvement) | ❌ | — | **Non couvert** |

---

## Synthèse des findings

### 🔴 Bugs (3)

| # | Sévérité | Fichier | Ligne | Description |
|---|----------|---------|-------|-------------|
| B1 | HAUTE | risk_manager.py | L194, L253 | `asyncio.get_event_loop().create_task()` pour les alertes kill switch — peut lever `RuntimeError` si pas d'event loop actif (shutdown). Les alertes les plus critiques du système pourraient ne pas être envoyées |
| B2 | MOYENNE | database.py | L1533 | `order_id` indexé mais pas UNIQUE dans `live_trades`. Doublons possibles si sync_bitget_trades.py tourne 2× ou si l'Executor persiste 2× le même trade |
| B3 | MOYENNE | grid_multi_tf.py | — | Aucun `cooldown_candles` après direction_flip. Churning aller-retour LONG→SHORT→LONG possible sur whipsaw Supertrend, accumulant 2× taker fees à chaque flip |

### ⚠️ Risques (12)

| # | Sévérité | Composant | Description |
|---|----------|-----------|-------------|
| R1 | HAUTE | executor.py | TP grid 100% client-side (SMA check toutes les 60s). Si le bot crash, seul le SL server-side protège. Pas de TP server-side backup |
| R2 | HAUTE | grid_multi_tf | Pas de `min_grid_spacing_pct`. En ATR comprimé, tous les niveaux peuvent se remplir simultanément |
| R3 | MOYENNE | executor.py | Pas de retry sur l'ordre d'entrée market. Timeout → position possible côté Bitget sans suivi local |
| R4 | MOYENNE | state_manager.py | `_write_json_file()` n'appelle pas `fsync()` avant `os.replace()`. Power loss → state file potentiellement vide sur Linux |
| R5 | MOYENNE | notifier.py | Pas d'alerte pour : funding rate extrême, margin call, asset suspendu, position zombie |
| R6 | MOYENNE | risk_manager.py | Kill switch NE ferme PAS les positions existantes — bloque uniquement les nouvelles entrées |
| R7 | MOYENNE | data_engine.py | Symbol stale abandonné après 3 retries pour toute la session. Si Bitget maintenance temporaire, le symbol est perdu |
| R8 | MOYENNE | data_engine.py | `_write_buffer.copy()` + `clear()` pas atomique. Si `insert_candles_batch()` échoue, candles perdues sans re-queue |
| R9 | FAIBLE | executor.py L1977 | P&L % mono hardcodé à `leverage=3` : `margin = pos.entry_price * pos.quantity / 3`. Incorrect si leverage différent |
| R10 | FAIBLE | executor_manager.py | `exchange_balance` somme partielle si un executor fail → dashboard affiche solde trompeur |
| R11 | FAIBLE | data_engine.py | `_connected = True` (L242) avant que le WS soit réellement connecté — faux positif possible pour les consommateurs |
| R12 | FAIBLE | risk_manager.py | `_trade_history` list grow unbounded. Après semaines de prod → mémoire croissante |

### ✅ Points forts

1. **SL 100% server-side** : Bitget trigger orders avec `mark_price`, survivent au crash
2. **Reconciliation au boot robuste** : 4 cas gérés (both open, orpheline, fermée downtime, clean)
3. **Nettoyage ordres orphelins** : `_cancel_orphan_orders()` au boot
4. **Partial fill protection** : détection + retry + alerte
5. **Residual position check** : `_verify_no_residual_position()` 1.5s après chaque close
6. **Anti-lookahead Supertrend 4h** : resampling correct avec `searchsorted`
7. **Atomic state write** : `.tmp` + `os.replace()`
8. **Multi-tier WS recovery** : per-symbol retry → dead task restart → full reconnect
9. **Dual monitoring** : watchOrders (temps réel) + poll_positions_loop (fallback 5s)
10. **Emergency close** : règle #1 respectée — JAMAIS de position sans SL

---

## Priorisation des fixes (pour sprint suivant)

### P0 — Avant déploiement 26 assets

| Fix | Effort | Ticket |
|-----|--------|--------|
| B1 : Remplacer `get_event_loop().create_task()` par pattern safe | Petit | risk_manager.py L194, L253 |
| B3 : Ajouter `cooldown_candles` à grid_multi_tf (config + compute_grid + fast engine) | Moyen | grid_multi_tf.py, config.py, param_grids.yaml |
| R2 : Ajouter `min_grid_spacing_pct` à grid_multi_tf | Moyen | grid_multi_tf.py, config.py |

### P1 — Semaine suivante

| Fix | Effort |
|-----|--------|
| B2 : Ajouter UNIQUE constraint sur `order_id` dans `live_trades` (migration) | Petit |
| R4 : Ajouter `f.flush(); os.fsync(f.fileno())` dans `_write_json_file()` | Petit |
| R5 : Alertes funding extrême + position zombie | Moyen |
| R9 : Utiliser le vrai leverage dans le calcul pnl_pct mono | Petit |

### P2 — Sprint suivant

| Fix | Effort |
|-----|--------|
| R1 : Server-side TP backup pour les grids (trigger order à X% au-dessus de l'entrée) | Gros |
| R3 : Retry + idempotency key sur entry market orders | Moyen |
| R6 : Option kill switch → market close all | Moyen |
| R7 : Mécanisme de retry périodique pour les symbols abandonnés | Petit |
| R12 : `maxlen` sur `_trade_history` dans risk_manager | Petit |
