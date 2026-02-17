# Hotfix 25b — Kill Switch Reliability

## Contexte

Le kill switch global du Simulator a 4 problèmes en production :
1. **Pas de reset API** — il faut éditer le JSON à la main
2. **Pas d'alerte Telegram au restore** — le kill switch se restaure silencieusement au restart
3. **Pas de raison persistée** — on ne sait pas pourquoi il a triggeré
4. **Positions perdues (11→6)** — bug dans `_apply_restored_state()` qui reset `kill_switch_triggered=False` après que `_stop_all_runners()` l'a mis à True

**Point clé architecture** : dans le lifespan (`server.py:98`), `set_notifier()` est appelé AVANT `start()` — le notifier est disponible au moment de la restauration.

---

## Fichiers à modifier

| Fichier | Changement |
|---------|-----------|
| `backend/backtesting/simulator.py` | FIX 1-4 : reason, reset, alerte restore, bug _apply_restored_state |
| `backend/core/state_manager.py` | Persister `kill_switch_reason` dans le JSON |
| `backend/api/simulator_routes.py` | POST /kill-switch/reset + reason dans GET /status |
| `backend/api/server.py` | Exposer `state_manager` dans `app.state` |
| `tests/test_kill_switch_reliability.py` | 10 tests nouveaux |

---

## Étape 1 — `backend/backtesting/simulator.py`

### 1A. Ajouter `_kill_switch_reason` dans `Simulator.__init__` (après ligne 1227)
```python
self._kill_switch_reason: dict | None = None
```

### 1B. Persister la raison dans `_check_global_kill_switch()` (ligne ~1299)
Dans le bloc `if drawdown_pct >= threshold_pct:`, avant `_stop_all_runners()` :
```python
self._kill_switch_reason = {
    "triggered_at": datetime.now(tz=timezone.utc).isoformat(),
    "drawdown_pct": round(drawdown_pct, 2),
    "window_hours": window_hours,
    "threshold_pct": threshold_pct,
    "capital_max": round(capital_max, 2),
    "capital_current": round(current_capital, 2),
}
```

### 1C. Méthode `reset_kill_switch()` (après `_stop_all_runners`)
```python
def reset_kill_switch(self) -> int:
    """Reset le kill switch global et réactive tous les runners."""
    self._global_kill_switch = False
    self._kill_switch_reason = None
    reactivated = 0
    for runner in self._runners:
        if runner._kill_switch_triggered:
            runner._kill_switch_triggered = False
            runner._stats.is_active = True
            reactivated += 1
    logger.critical("KILL SWITCH GLOBAL RESET — {} runners réactivés", reactivated)
    return reactivated
```

### 1D. Property `kill_switch_reason` (après `is_kill_switch_triggered`)
```python
@property
def kill_switch_reason(self) -> dict | None:
    return self._kill_switch_reason
```

### 1E. Alerte Telegram au restore dans `start()` (lignes 1596-1613)
Restaurer `_kill_switch_reason` depuis saved_state (juste après `_global_kill_switch`) :
```python
self._kill_switch_reason = saved_state.get("kill_switch_reason", None)
```
Puis si `_global_kill_switch` est True :
- Compter les positions totales (grid + mono)
- Construire un message avec la raison si disponible
- Appeler `self._notifier.notify_anomaly(AnomalyType.KILL_SWITCH_GLOBAL, ...)` dans un try/except
- Garder le force `_end_warmup()` existant

### 1F. Corriger `_apply_restored_state()` (ligne 668) — **BUG CRITIQUE**
Remplacer :
```python
self._kill_switch_triggered = False  # BUG: écrase _stop_all_runners()
```
Par :
```python
if not self._kill_switch_triggered:
    self._kill_switch_triggered = state.get("kill_switch", False)
```
Et pour `is_active` :
```python
if not self._kill_switch_triggered:
    self._stats.is_active = state.get("is_active", True)
```
Ajouter des logs INFO pour les positions restaurées et WARNING si mismatch count.

### 1G. Log WARNING dans `_dispatch_candle` (après le return kill switch, ligne 1637)
Pas dans le return path (on ne peut pas logger après un return). À la place, ajouter un log one-shot au démarrage dans `start()` — déjà couvert par le log CRITICAL enrichi en 1E avec le comptage des positions.

---

## Étape 2 — `backend/core/state_manager.py`

### 2A. Signature `save_runner_state` : ajouter paramètre
```python
async def save_runner_state(
    self, runners, global_kill_switch=False, kill_switch_reason=None,
) -> None:
```

### 2B. Ajouter au dict `state`
```python
"kill_switch_reason": kill_switch_reason,
```

### 2C. Mettre à jour `_periodic_save_loop` (ligne ~205)
Passer `kill_switch_reason=simulator._kill_switch_reason`.

---

## Étape 3 — `backend/api/server.py`

### 3A. Exposer `state_manager` dans `app.state`
Après `await state_manager.start_periodic_save(simulator)` (ligne 104) :
```python
app.state.state_manager = state_manager
```
Après `app.state.arena = None` (branche else, ligne 111) :
```python
app.state.state_manager = None
```

### 3B. Mettre à jour le shutdown (lignes 183-186)
Passer `kill_switch_reason=simulator._kill_switch_reason` dans l'appel `save_runner_state`.

---

## Étape 4 — `backend/api/simulator_routes.py`

### 4A. GET /status : ajouter `kill_switch_reason`
```python
"kill_switch_reason": simulator.kill_switch_reason,
```

### 4B. POST /api/simulator/kill-switch/reset
- Vérifier `simulator._global_kill_switch` (sinon retourner `not_triggered`)
- Appeler `simulator.reset_kill_switch()`
- Sauvegarder via `state_manager.save_runner_state(...)`
- Notifier Telegram via `notifier.notify_anomaly(...)`
- Retourner `{"status": "reset", "runners_reactivated": N}`
- Commentaire `# TODO: ajouter auth quand exposé hors réseau local`

---

## Étape 5 — `tests/test_kill_switch_reliability.py` (~10 tests)

1. `test_reset_endpoint_resets_global` — POST reset → appelle `reset_kill_switch()`
2. `test_reset_endpoint_reactivates_runners` — vérifie le count retourné
3. `test_reset_endpoint_saves_state` — vérifie `save_runner_state` appelé
4. `test_reset_endpoint_not_triggered` — kill switch inactif → `not_triggered`
5. `test_reset_endpoint_notifies_telegram` — vérifie `notify_anomaly` appelé
6. `test_kill_switch_reason_in_status` — GET status inclut la raison
7. `test_kill_switch_reason_null_when_inactive` — raison null si pas déclenché
8. `test_kill_switch_reason_cleared_on_reset` — reset → raison None
9. `test_apply_restored_state_respects_kill_switch` — `_kill_switch_triggered` pas écrasé
10. `test_apply_restored_state_restores_positions` — positions grid restaurées avec log
11. `test_reset_reactivates_globally_stopped_runner` — runner stoppé par `_stop_all_runners()` (global) est bien réactivé par reset (pas seulement ceux stoppés par leur propre seuil)

---

## Vérification

1. `uv run python -m pytest tests/test_kill_switch_reliability.py -v` — 10 nouveaux tests passent
2. `uv run python -m pytest --tb=short -q` — 1037 + ~10 tests, 0 régression
3. Vérifier que le state JSON contient `kill_switch_reason` quand le kill switch trigger
4. Vérifier rétrocompatibilité : ancien JSON sans `kill_switch_reason` → `None` (via `.get()`)
