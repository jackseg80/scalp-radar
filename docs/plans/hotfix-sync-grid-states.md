# Hotfix — Sync injecte dans executor._grid_states

**Date** : 19 février 2026

## Problème

Au boot après un restart, `sync_live_to_paper()` itérait sur `executor._grid_states`
qui était vide (state file absent ou corrompu). La boucle ne s'exécutait pas →
les positions live Bitget n'étaient pas injectées dans l'executor → l'exit monitor
autonome n'avait rien à checker → les TP dynamiques (SMA) n'étaient jamais évalués.

Séquence observée :
```
13:03:44 | Sync: terminé — 4 symbols live synchronisés  ← FAUX (0 injectés)
13:03:44 | exit monitor autonome démarré
→ AUCUN log de check pendant 10+ minutes
→ executor_state.json : grid_states = []
```

## Solution

**backend/execution/sync.py** :
- Début de `sync_live_to_paper()` : si `executor._grid_states` est vide → appel `_populate_grid_states_from_exchange()`
- `_populate_grid_states_from_exchange()` : `fetch_positions()` → crée un `GridLiveState` par position
  - `contracts > 0` et `entryPrice > 0` obligatoires
  - Stratégie via runners paper, fallback `grid_atr`
  - Leverage via `config.strategies.{name}.leverage`, fallback 3
  - `GridLivePosition(level=0, entry_order_id="restored-from-sync")`
  - `sl_price=0.0` (recalculé par `should_close_all()`)

**backend/execution/executor.py** :
- `_exit_monitor_loop()` : log DEBUG avant chaque check (confirmation que la boucle tourne)

## Tests

- `test_sync_creates_grid_states_from_exchange` : 3 positions mockées (2 valides + 1 contracts=0) → 2 GridLiveState créés
- `test_sync_grid_states_not_overwritten_when_populated` : si déjà peuplé → populate non appelé

**Total : 1424 tests, 0 régression**

## Validation prod

```bash
docker compose logs -f backend | grep -E "(Sync:|grid_state|Exit monitor)"
```

Attendu :
```
Sync: grid_state créé pour FET/USDT:USDT (grid_atr, 3812 contracts @ 0.1667)
Sync: 4 grid_states créés depuis l'exchange
Exit monitor: check 4 positions ([...])
```
