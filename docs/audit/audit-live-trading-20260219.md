# Audit Live Trading — 2026-02-19

## Résumé exécutif

- **3 bugs P0** trouvés (bloquants — sécurité et opérabilité du bot)
- **7 bugs P1** trouvés (importants — protection incomplète)
- **3 bugs P2** trouvés (nice to have)
- **Le bot n'est PAS safe pour le live** tant que les P0 ne sont pas corrigés

---

## Findings par composant

### A. Risk Manager (`backend/execution/risk_manager.py`)

#### Ce que fait le code

`record_trade_result()` accumule `_session_pnl += result.net_pnl` et compare :

```python
max_loss = self._config.risk.kill_switch.max_session_loss_percent  # TOUJOURS 5.0
if loss_pct >= max_loss:
    self._kill_switch_triggered = True
```

`pre_trade_check()` vérifie (dans l'ordre) : kill_switch, position_already_open, max_concurrent_positions, correlation_group_limit, insufficient_margin.

`session_pnl` ne se reset jamais automatiquement. Il persiste dans `executor_state.json`.

Il n'y a pas de méthode `can_trade()` — seulement `pre_trade_check()`.

`daily_pnl` n'existe pas dans le code (seulement dans risk.yaml).

#### Bugs trouvés

- **[P0] `grid_max_session_loss_percent: 25.0` jamais utilisé** : La config a ce champ mais `record_trade_result()` lit **toujours** `max_session_loss_percent: 5.0` pour toutes les stratégies. Avec grid_atr (capital $1000, 10 assets, 3x leverage), **un seul SL hit peut représenter 6%+ de session** -> kill switch live déclenché sur une perte normale.

- **[P0] Kill switch live irréinitialisable** : Aucun endpoint `POST /api/executor/kill-switch/reset` n'existe (contrairement au paper qui a `POST /api/simulator/kill-switch/reset`). Seule manipulation : modifier `data/executor_state.json` puis redémarrer.

- **[P1] session_pnl jamais reset** : S'accumule depuis le premier démarrage du bot. Empoisonnement inter-sessions possible.

- **[P1] `max_daily_loss_percent` et `grid_max_daily_loss_percent` jamais vérifiés** : Dead code côté live.

- **[P1] Pas d'alerte Telegram quand kill switch live déclenché** : Seulement `logger.warning()`. Bot bloqué silencieusement.

---

### B. Executor (`backend/execution/executor.py`)

#### Boot sequence complète

```
restore_positions(executor_state)     <- positions/grid_states depuis state file
executor.start()                      <- markets, balance, setup_leverage
  |_ _reconcile_on_boot()             <- sync avec positions Bitget réelles
selector.start()
simulator.set_trade_event_callback(executor.handle_event)
executor.set_data_engine(engine)
executor.set_strategies(..., simulator=simulator)
sync_live_to_paper(executor, simulator)
executor.start_exit_monitor()
state_manager.set_executor(executor, risk_mgr)
```

#### Bugs trouvés

- **[P0] Niveaux DCA 2+ ignorent le kill switch live** : `pre_trade_check()` est appelé UNIQUEMENT au niveau 1. Les niveaux suivants continuent à s'ouvrir même si le kill switch est actif.

- **[P1] Réconciliation downtime sans fees** : `_reconcile_grid_symbol()` cas "fermé pendant downtime" utilise `_calculate_pnl()` (fees approximatives) et non `_calculate_real_pnl()`.

- **[P2] `GET /api/executor/orders` sans authentification** : Read-only mais incohérent avec les autres endpoints.

#### `_close_grid_cycle()` — OK

`record_trade_result()` appelé -> RiskManager mis à jour. `unregister_position()` appelé. Correct.

#### Kill switch live au restart

`executor_state.json` avec `kill_switch: true` -> `pre_trade_check()` retourne `(False, "kill_switch_live")` -> bot bloqué définitivement.

---

### C. State Manager (`backend/core/state_manager.py`)

#### Ce que fait le code

`save_executor_state()` : positions, grid_states, risk_manager state, order_history.

`_periodic_save_loop()` : toutes les 60s. Shutdown SIGTERM : sauvegardé avant stop.

#### Bugs trouvés

- **[P1] `_open_positions` non restaurée dans `restore_state()`** : Entre restauration et réconciliation, `max_concurrent_positions` peut être faussement satisfait (0 positions trackées).

- **[P2] Race condition mineure** : Signal de fermeture pendant clôture de position. Risque faible.

---

### D. Sync (`backend/execution/sync.py`)

#### Ce que fait le code

`sync_live_to_paper()` : sync grid_states avec paper. Cas 0 positions Bitget = positions paper supprimées. **Correct.**

#### Bugs trouvés

- **[P1] Fallback `strategy_name = "grid_atr"`** dans `_populate_grid_states_from_exchange()` : Si le symbol appartient à grid_boltrend, mauvais seuil SL utilisé par l'exit monitor.

---

### E. Exit Monitor

`_check_grid_exit()` lit depuis `simulator.get_runner_context()` — source unique. **OK.**

Warm-up : SMA absente -> skip. **OK.**
Exception handling : boucle survit. **OK.**

- **[P2] Skip logué en DEBUG seulement** : Difficile à diagnostiquer.

---

### F. Config risk.yaml — Tableau des seuils

| Paramètre | Utilisé par | Valeur | Commentaire |
|---|---|---|---|
| `max_session_loss_percent: 5.0` | RiskManager live (TOUTES) | 5% | TROP RESTRICTIF pour grid |
| `max_daily_loss_percent: 10.0` | PERSONNE | 10% | Dead code |
| `grid_max_session_loss_percent: 25.0` | PERSONNE | 25% | Dead code |
| `grid_max_daily_loss_percent: 25.0` | PERSONNE | 25% | Dead code |
| `global_max_loss_pct: 45` | Simulator paper uniquement | 45% | Pas d'équivalent live |
| `global_window_hours: 24` | Simulator paper uniquement | 24h | Idem |
| `max_margin_ratio: 0.70` | GridStrategyRunner paper | 70% | Pas d'équivalent live |
| `max_live_grids: 4` | Executor live | 4 | OK |

- **[P1] `global_max_loss_pct: 45` absent côté live** : Si le portfolio perd 45% en 24h, rien ne se passe côté executor.

---

### G. Server boot (`backend/api/server.py`)

Séquence complète : DB -> JobManager -> Telegram/Notifier -> DataEngine -> Simulator -> StateManager -> Executor -> Selector -> exit monitor -> Watchdog -> Heartbeat.

Kill switch paper et live sont indépendants. **OK.**

---

### H. API Endpoints

**executor_routes.py** :
- `GET /api/executor/status` (X-API-Key) OK
- `POST /api/executor/refresh-balance` (X-API-Key) OK
- `POST /api/executor/test-trade` (X-API-Key) OK
- `POST /api/executor/test-close` (X-API-Key) OK
- `GET /api/executor/orders` (sans auth — P2)
- **MANQUANT** : `POST /api/executor/kill-switch/reset` **P0**

**simulator_routes.py** :
- `POST /api/simulator/kill-switch/reset` OK
- Pas d'équivalent live

---

## Plan de fix priorisé

1. **[P0]** Utiliser `grid_max_session_loss_percent` pour les stratégies grid (risk_manager.py)
2. **[P0]** Endpoint `POST /api/executor/kill-switch/reset` (executor_routes.py)
3. **[P0]** Guard kill switch niveaux DCA 2+ (executor.py)
4. **[P1]** Alerte Telegram quand kill switch live déclenché (risk_manager.py)
5. **[P1]** Reset quotidien session_pnl à minuit UTC (risk_manager.py)
6. **[P1]** Kill switch global live 45% drawdown (risk_manager.py + executor.py)

---

## Checklist avant relance live

- [ ] Fix grid_max_session_loss_percent
- [ ] Endpoint reset kill switch live
- [ ] Guard kill switch niveaux DCA 2+
- [ ] Alerte Telegram kill switch live
- [ ] Reset quotidien session_pnl
- [ ] Kill switch global live (45% drawdown)
- [ ] Vérifier executor_state.json : kill_switch=false, session_pnl cohérent
- [ ] Tests unitaires pour chaque fix
- [ ] Déployer et monitorer 24h
