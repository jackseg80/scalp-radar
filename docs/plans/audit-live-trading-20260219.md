# Plan : Audit Live Trading + Plan de Fix

## Contexte

L'objectif est double :
1. Produire un rapport d'audit exhaustif dans `docs/audit/audit-live-trading-20260219.md`
2. Planifier et implémenter les fixes P0

---

## Contenu du rapport d'audit (à créer)

```markdown
# Audit Live Trading — 2026-02-19

## Résumé exécutif

- **3 bugs P0** trouvés (bloquants — sécurité et opérabilité du bot)
- **7 bugs P1** trouvés (importants — protection incomplète, dont 2 ajoutés post-revue)
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
    logger.warning("KILL SWITCH LIVE: perte session {:.1f}% >= {:.1f}%", ...)
```

`pre_trade_check()` vérifie (dans l'ordre) : kill_switch, position_already_open, max_concurrent_positions, correlation_group_limit, insufficient_margin.

`session_pnl` ne se reset jamais automatiquement. Il persiste dans le fichier `executor_state.json`.

Il n'y a pas de méthode `can_trade()` — seulement `pre_trade_check()`.

`daily_pnl` n'existe pas dans le code (seulement dans risk.yaml).

#### Bugs trouvés

- **[P0] `grid_max_session_loss_percent: 25.0` jamais utilisé** : La config a ce champ mais `record_trade_result()` lit **toujours** `max_session_loss_percent: 5.0` pour toutes les stratégies. Avec grid_atr (capital $1000, 10 assets, 3x leverage), **un seul SL hit peut représenter 6%+ de session** → kill switch live déclenché sur une perte normale. Ce bug a probablement causé les -83$ : l'exit monitor a fermé des positions, le risk manager a accumulé les pertes et a déclenché le kill switch.

- **[P0] Kill switch live irréinitialisable** : Aucun endpoint `POST /api/executor/kill-switch/reset` n'existe (contrairement au paper qui a `POST /api/simulator/kill-switch/reset`). Seule manipulation possible : modifier manuellement `data/executor_state.json` puis redémarrer.

- **[P1] session_pnl jamais reset** : S'accumule depuis le premier démarrage du bot. Une session perdante + sessions gagnantes + une session perdante peut déclencher le kill switch par cumul inter-sessions.

- **[P1] `max_daily_loss_percent: 10.0` et `grid_max_daily_loss_percent: 25.0` jamais vérifiés** : Ces champs risk.yaml sont dead code côté live.

- **[P1] Pas d'alerte Telegram quand kill switch live déclenché** : Seulement `logger.warning()`. Le bot se bloque silencieusement sans notification push.

#### Recommandations

```python
# Dans record_trade_result(), détecter le type de stratégie :
is_grid = is_grid_strategy(result.exit_reason, ...)  # ou passer strategy_name
max_loss = (
    self._config.risk.kill_switch.grid_max_session_loss_percent
    if is_grid else
    self._config.risk.kill_switch.max_session_loss_percent
)
```

Alternativement (plus simple) : passer `strategy_name: str` à `record_trade_result()` et appeler `is_grid_strategy(strategy_name)`.

---

### B. Executor (`backend/execution/executor.py`)

#### Boot sequence complète

```
restore_positions(executor_state)     ← positions/grid_states depuis state file
executor.start()                      ← markets, balance, setup_leverage
  └─ _reconcile_on_boot()             ← sync avec positions Bitget réelles
selector.start()
simulator.set_trade_event_callback(executor.handle_event)
executor.set_data_engine(engine)
executor.set_strategies(..., simulator=simulator)  ← OK après dernier hotfix
sync_live_to_paper(executor, simulator)
executor.start_exit_monitor()
state_manager.set_executor(executor, risk_mgr)
```

#### Bugs trouvés

- **[P0] Niveaux DCA 2+ ignorent le kill switch live** : `pre_trade_check()` est appelé UNIQUEMENT au niveau 1 (`if is_first_level`). Si le kill switch se déclenche entre le niveau 1 et le niveau 2 (ex: level 1 clôture avec grosse perte puis level 2 ouvre), les niveaux suivants continuent à s'ouvrir sans vérification.

- **[P1] Réconciliation downtime sans fees** : `_reconcile_grid_symbol()` cas "fermé pendant downtime" utilise `_calculate_pnl()` (fees approximatives) et non `_calculate_real_pnl()`. Le P&L enregistré dans session_pnl peut être légèrement erroné.

- **[P2] `GET /api/executor/orders` sans authentification** : L'historique des ordres Bitget est accessible sans X-API-Key. Pas de risque de sécurité critique (read-only) mais incohérent avec les autres endpoints.

#### `_close_grid_cycle()` — comportement correct

Après fermeture : `record_trade_result()` appelé → RiskManager mis à jour. `unregister_position()` appelé. `del self._grid_states[futures_sym]`. OK.

#### Kill switch live au restart

Si `executor_state.json` contient `kill_switch: true` → `risk_mgr.restore_state()` le restaure → `pre_trade_check()` retourne `(False, "kill_switch_live")` → aucun nouveau trade. Bot bloqué définitivement.

---

### C. State Manager (`backend/core/state_manager.py`)

#### Ce que fait le code

`save_executor_state()` : appelle `executor.get_state_for_persistence()` → positions, grid_states, risk_manager (session_pnl + kill_switch + ...), order_history.

`_periodic_save_loop()` : toutes les 60s sauvegarde simulator state + executor state. Résout le problème de state périmé après kill -9.

Shutdown SIGTERM : `save_executor_state()` appelé dans le lifespan avant `executor.stop()`. ✓

#### Bugs trouvés

- **[P1] `_open_positions` non restaurée dans `restore_state()`** : `risk_manager.restore_state()` ne restaure pas `_open_positions` (commentaire dit "source de vérité = exchange"). Mais entre la restauration et la réconciliation, `max_concurrent_positions` peut être faussement satisfait (0 positions trackées), permettant l'ouverture de trop de positions.

- **[P2] Race condition mineure** : Si le bot reçoit un signal de fermeture PENDANT qu'une position se clôture, le state sauvegardé peut ne pas refléter le trade. Risque faible (save périodique 60s).

---

### D. Sync (`backend/execution/sync.py`)

#### Ce que fait le code

`sync_live_to_paper()` : si `executor._grid_states` vide → `_populate_grid_states_from_exchange()` → inject dans paper. Sinon : compare grid_states avec runner paper, nettoie les positions orphelines.

#### Comportement si Bitget a 0 positions et paper en a encore

1. `_reconcile_on_boot()` détecte "position fermée pendant downtime" → `del self._grid_states[sym]`
2. `sync_live_to_paper()` reçoit `_grid_states` vide
3. `_populate_grid_states_from_exchange()` appelée → fetch Bitget → 0 positions → rien créé
4. `live_symbols_by_runner` vide → toutes les positions paper supprimées, marge restituée ✓

Ce cas est géré correctement.

#### Bugs trouvés

- **[P1] `strategy_name` fallback "grid_atr" dans `_populate_grid_states_from_exchange()`** : Si un symbol n'est pas dans `symbol_to_strategy` (runner paper n'a pas de positions pour ce symbol), fallback `strategy_name = "grid_atr"`. Si en réalité c'est du `grid_boltrend`, le mauvais seuil SL sera utilisé par l'exit monitor.

---

### E. Exit Monitor (`executor.py` — `_check_grid_exit`, `_exit_monitor_loop`)

#### Ce que fait le code

`_check_grid_exit()` lit depuis `self._simulator.get_runner_context(strategy_name, spot_sym)` — SOURCE UNIQUE, plus de recalcul indépendant. ✓

Si `ctx is None` → debug log + return. ✓
Si `sma` ou `close` absent des indicateurs → debug log + return. ✓
Exception loop : `except Exception as e: logger.error(...)` — survit aux exceptions. ✓

#### Pendant le warm-up paper (~1-2 min)

`build_context(symbol)` retourne un contexte. Si le `IncrementalIndicatorEngine` n'a pas encore calculé la SMA (buffer insuffisant), `sma` sera absent → skip avec debug log. Pas de faux TP. ✓

#### Bugs trouvés

- **[P2] Skip "pas de contexte" logué en DEBUG seulement** : Si le simulator n'est pas encore prêt ou que le runner n'existe pas, c'est du DEBUG silencieux. Difficile à diagnostiquer si le monitoring manque.

---

### F. Config risk.yaml

| Paramètre | Utilisé par | Valeur | Commentaire |
|-----------|-------------|--------|-------------|
| `max_session_loss_percent: 5.0` | RiskManager live (TOUTES stratégies) | 5% | **TROP RESTRICTIF pour grid** |
| `max_daily_loss_percent: 10.0` | **PERSONNE** | 10% | Dead code |
| `grid_max_session_loss_percent: 25.0` | **PERSONNE** | 25% | Dead code — devrait remplacer le 5% pour grid |
| `grid_max_daily_loss_percent: 25.0` | **PERSONNE** | 25% | Dead code |
| `global_max_loss_pct: 45` | Simulator **paper** uniquement | 45% | Pas de protection équivalente en live |
| `global_window_hours: 24` | Simulator paper uniquement | 24h | Idem |
| `max_margin_ratio: 0.70` | GridStrategyRunner paper | 70% | L'executor live n'a pas ce guard |
| `max_live_grids: 4` | Executor live | 4 | ✓ |

**Conclusion** : Le seul vrai kill switch live est à 5% session_pnl, sans reset automatique, pour toutes les stratégies confondues. C'est trop restrictif et non adapté aux grid strategies.

- **[P1] `global_max_loss_pct: 45` absent côté live** : Le Simulator vérifie le drawdown peak-to-trough sur fenêtre glissante 24h via `_check_global_kill_switch()` (snapshots capital toutes les 60s). L'Executor live n'a aucun équivalent — si le portfolio perd 45% en 24h, rien ne se passe.
- **[P1] `max_margin_ratio: 0.70` absent côté live** : Le GridStrategyRunner paper vérifie que la marge totale ne dépasse pas 70% du capital. L'Executor live n'a pas ce guard (TODO futur, moins critique car Bitget a ses propres guards de marge).

---

### G. Server boot (`backend/api/server.py`)

**Séquence complète** : DB → JobManager → Telegram/Notifier → DataEngine → Simulator (+ crash recovery) → StateManager periodic save → Executor (+ restore state) → Selector → trade callback → exit monitor → Watchdog → Heartbeat

`set_strategies()` reçoit bien `simulator=simulator` (après dernier hotfix). ✓

Kill switch paper et live sont **indépendants** :
- Paper kill switch → Simulator arrête d'émettre des TradeEvents → no new OPEN via executor
- Live kill switch → executor refuse pre_trade_check → positions Bitget restent jusqu'au TP/SL/exit monitor

**Observation** : `simulator.set_trade_event_callback(executor.handle_event)` est enregistré après `simulator.start()`. Pendant le warm-up (1-2 min), les runners ne tradent pas (bloqués par `_is_warming_up`). Pas de race condition pratique. ✓

---

### H. API Endpoints

**executor_routes.py** :
- `GET /api/executor/status` — statut (X-API-Key requis) ✓
- `POST /api/executor/refresh-balance` — refresh solde (X-API-Key) ✓
- `POST /api/executor/test-trade` — injection test (X-API-Key) ✓
- `POST /api/executor/test-close` — fermeture test (X-API-Key) ✓
- `GET /api/executor/orders` — historique ordres (**sans auth** — P2)
- **MANQUANT** : `POST /api/executor/kill-switch/reset` ← **P0**

**simulator_routes.py** :
- `POST /api/simulator/kill-switch/reset` — reset kill switch paper ✓
- **Pas d'équivalent live**

---

## Plan de fix priorisé

### 1. [P0] Utiliser `grid_max_session_loss_percent` pour les stratégies grid

**Fichier** : `backend/execution/risk_manager.py`

```python
# Ajouter strategy_name au LiveTradeResult
@dataclass
class LiveTradeResult:
    net_pnl: float
    timestamp: datetime
    symbol: str
    direction: str
    exit_reason: str
    strategy_name: str = ""  # nouveau champ

# Dans record_trade_result() :
def record_trade_result(self, result: LiveTradeResult) -> None:
    self._session_pnl += result.net_pnl
    self._trade_history.append(result)

    if self._initial_capital <= 0:
        return

    from backend.optimization import is_grid_strategy
    loss_pct = abs(min(0, self._session_pnl)) / self._initial_capital * 100

    ks_config = self._config.risk.kill_switch
    if is_grid_strategy(result.strategy_name):
        max_loss = ks_config.grid_max_session_loss_percent
    else:
        max_loss = ks_config.max_session_loss_percent

    if loss_pct >= max_loss:
        self._kill_switch_triggered = True
        logger.warning("KILL SWITCH LIVE: perte session {:.1f}% >= {:.1f}% ({})", ...)
```

**Cascades** : Ajouter `strategy_name=""` à tous les appels `LiveTradeResult(...)` dans executor.py (~5 endroits).

**AppConfig confirmé** : `backend/core/config.py:455` — `grid_max_session_loss_percent: Optional[float] = None`. Le champ existe mais est `Optional`. Dans le code, utiliser `ks_config.grid_max_session_loss_percent or 25.0` comme fallback sécurisé.

`is_grid_strategy()` est disponible dans `backend/optimization/__init__.py:85`.

---

### 2. [P0] Endpoint API pour reset kill switch live

**Fichier** : `backend/api/executor_routes.py`

```python
@router.post("/kill-switch/reset", dependencies=[Depends(verify_executor_key)])
async def reset_live_kill_switch(request: Request) -> dict:
    """Reset le kill switch live et réactive le trading."""
    executor = getattr(request.app.state, "executor", None)
    if executor is None:
        raise HTTPException(status_code=400, detail="Executor non actif")

    if not executor.risk_manager.is_kill_switch_triggered:
        return {"status": "not_triggered", "message": "Kill switch live non actif"}

    # Reset
    executor.risk_manager._kill_switch_triggered = False
    executor.risk_manager._session_pnl = 0.0  # Reset session aussi

    # Sauvegarder l'état
    state_manager = getattr(request.app.state, "state_manager", None)
    if state_manager:
        risk_mgr = getattr(request.app.state, "risk_mgr", None)
        if risk_mgr:
            await state_manager.save_executor_state(executor, risk_mgr)

    # Notification Telegram
    notifier = getattr(request.app.state, "notifier", None)
    if notifier:
        await notifier.notify_anomaly(
            AnomalyType.KILL_SWITCH_GLOBAL,
            "Kill switch LIVE reset manuellement — trading réactivé",
        )

    return {"status": "reset", "message": "Kill switch live réinitialisé"}
```

Exposer `risk_mgr` dans `app.state` dans server.py (actuellement non exposé).

---

### 3. [P0] Vérifier kill switch aux niveaux DCA supérieurs

**Fichier** : `backend/execution/executor.py`, méthode `_open_grid_position()`

```python
# Après le bloc `if is_first_level:` ... `else:`
else:
    # Niveaux suivants : vérifier QUAND MÊME le kill switch live
    if self._risk_manager.is_kill_switch_triggered:
        logger.warning(
            "Executor: kill switch live actif, niveau grid {} ignoré pour {}",
            len(state.positions) + 1, futures_sym,
        )
        return
    quantity = self._round_quantity(event.quantity, futures_sym)
    ...
```

---

### 4. [P1] Alerte Telegram quand kill switch live déclenché

**Fichier** : `backend/execution/risk_manager.py`

Problème : `LiveRiskManager` n'a pas de référence au `Notifier` (dépendance circulaire potentielle).

Solution : passer le notifier dans `__init__` ou faire un callback.

```python
def __init__(self, config: AppConfig, notifier=None) -> None:
    ...
    self._notifier = notifier

# Dans record_trade_result() après avoir déclenché :
if self._notifier:
    import asyncio
    asyncio.create_task(
        self._notifier.notify_anomaly(
            AnomalyType.KILL_SWITCH_GLOBAL,
            f"KILL SWITCH LIVE déclenché ! perte={loss_pct:.1f}% / seuil={max_loss:.1f}%"
        )
    )
```

---

### 5. [P1] Reset automatique session_pnl quotidien

**Fichier** : `backend/execution/risk_manager.py`

Ajouter un attribut `_session_start_date` et vérifier dans `record_trade_result()` si le jour UTC a changé :

```python
def __init__(self, config, notifier=None) -> None:
    ...
    self._session_start_date: date = datetime.now(tz=timezone.utc).date()

def record_trade_result(self, result: LiveTradeResult) -> None:
    # Auto-reset quotidien
    today = datetime.now(tz=timezone.utc).date()
    if today != self._session_start_date:
        logger.info(
            "RiskManager: reset session_pnl quotidien ({:+.2f} → 0.0)",
            self._session_pnl,
        )
        self._session_pnl = 0.0
        self._session_start_date = today

    self._session_pnl += result.net_pnl
    ...
```

Persister `_session_start_date` dans `get_state()` / `restore_state()`.

---

### 6. [P1] Kill switch global live (équivalent `global_max_loss_pct: 45`)

**Fichier** : `backend/execution/risk_manager.py`

Le Simulator a un kill switch global basé sur une fenêtre glissante de snapshots de capital (`_capital_snapshots`, check peak-to-trough drawdown). L'Executor live n'a rien d'équivalent.

**Implémentation** : tracker les snapshots de `_exchange_balance` (rafraîchi toutes les 5 min par `_balance_refresh_loop()`) et vérifier le drawdown :

```python
def __init__(self, config, notifier=None) -> None:
    ...
    self._balance_snapshots: deque[tuple[datetime, float]] = deque(maxlen=288)  # 24h @ 5min

def record_balance_snapshot(self, balance: float) -> None:
    """Appelé par Executor._balance_refresh_loop() après chaque fetch."""
    self._balance_snapshots.append((datetime.now(tz=timezone.utc), balance))
    self._check_global_kill_switch(balance)

def _check_global_kill_switch(self, current_balance: float) -> None:
    if self._kill_switch_triggered:
        return
    if len(self._balance_snapshots) < 2:
        return

    ks = self._config.risk.kill_switch
    threshold = getattr(ks, "global_max_loss_pct", 45)
    window_hours = getattr(ks, "global_window_hours", 24)
    if not isinstance(threshold, (int, float)):
        return

    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=window_hours)
    peak = max(b for ts, b in self._balance_snapshots if ts >= cutoff)

    if peak <= 0:
        return

    drawdown_pct = (peak - current_balance) / peak * 100
    if drawdown_pct >= threshold:
        self._kill_switch_triggered = True
        logger.critical(
            "KILL SWITCH LIVE GLOBAL: drawdown {:.1f}% >= {:.1f}% (peak={:.2f}, now={:.2f})",
            drawdown_pct, threshold, peak, current_balance,
        )
```

**Cascade** : dans `Executor._balance_refresh_loop()`, appeler `self._risk_manager.record_balance_snapshot(new_total)` après chaque refresh.

---

## Checklist avant relance live

- [ ] **[P0] Fix grid_max_session_loss_percent** — risque de kill switch immédiat sur SL normal
- [ ] **[P0] Endpoint reset kill switch live** — sans ça, toute intervention nécessite accès SSH
- [ ] **[P0] Guard kill switch niveaux DCA 2+**
- [ ] **[P1] Alerte Telegram kill switch live** — actuellement silencieux
- [ ] **[P1] Reset quotidien session_pnl** — empoisonnement inter-sessions
- [ ] **[P1] Kill switch global live (45% drawdown)** — parité avec le paper
- [ ] Vérifier `executor_state.json` actuel : kill_switch=false ? session_pnl cohérent ?
- [ ] Vérifier risk.yaml : `grid_max_session_loss_percent` est bien parsé par AppConfig
- [ ] Test : déclencher manuellement le kill switch via un SL test + vérifier reset via API
- [ ] Déployer + monitorer les premières 24h

---

## Fichiers à créer/modifier

| Fichier | Action |
|---------|--------|
| `docs/audit/audit-live-trading-20260219.md` | Créer le rapport complet |
| `backend/execution/risk_manager.py` | Fix P0 seuil grid + P1 notifier + P1 daily reset + P1 global kill switch |
| `backend/execution/executor.py` | Fix P0 guard kill switch DCA niveaux 2+ + cascade balance snapshot |
| `backend/api/executor_routes.py` | Fix P0 endpoint reset kill switch live |
| `backend/api/server.py` | Exposer `risk_mgr` dans app.state + passer notifier au RiskManager |
| `tests/test_risk_manager.py` (ou équivalent) | Tests pour les 3 P0 + 3 P1 |
```

---

## Vérification

1. `uv run pytest tests/ -x -q -k risk_manager` → vérifier seuil grid
2. `uv run pytest tests/ -x -q -k executor` → vérifier guard kill switch DCA
3. `curl -X POST http://localhost:8000/api/executor/kill-switch/reset -H "X-API-Key: ..."` → 200 OK
4. `uv run pytest tests/ -x -q` → 0 régression (>= 1422 tests)
5. Vérifier dans les logs : "KILL SWITCH LIVE: ... (grid_atr)" avec seuil 25% et non 5%
