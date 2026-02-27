# Sprint 57b — Audit moteur live + 6 fixes critiques avant déploiement 26 assets

## Contexte

Avant le déploiement Sprint 56 sur 26 assets (15 grid_atr + 11 grid_multi_tf), audit complet du moteur live/paper sur 7 axes, suivi de 6 corrections prioritaires.

---

## Phase 1 — Audit (Sprint 57)

Audit exhaustif sur 7 axes :
1. Signal → Order flow
2. Position management (GridLiveState, reconciliation)
3. Multi-strategy isolation (ExecutorManager)
4. Kill switch / safety (per-session, global, margin guard)
5. Supertrend direction_flip (grid_multi_tf)
6. Edge cases (partial fill, exchange down, rate limits)
7. Logging / monitoring (Telegram, watchdog)

**Résultat** : 3 bugs (B1-B3), 12 risques (R1-R12), 10 points forts.
Rapport complet : `docs/audit/audit-live-paper-sprint57-20260227.md`

---

## Phase 2 — Fixes (Sprint 57b)

### Fix 1 — B1 : Alertes kill switch silencieuses (risk_manager.py)

`asyncio.get_event_loop()` → `asyncio.get_running_loop()` aux 2 endroits d'alerte kill switch.
Le `except Exception` existant capture déjà le `RuntimeError`.

### Fix 2 — B3 + R2 : Parité grid_multi_tf avec grid_atr

- **Config** (`config.py`) : ajout `cooldown_candles=3`, `min_grid_spacing_pct=0.0`, `max_hold_candles=0` à `GridMultiTFConfig`
- **Stratégie** (`grid_multi_tf.py`) : plancher ATR dans `compute_grid()`, 3 params dans `get_params()`
- **Fast engine** (`fast_multi_backtest.py`) : plancher ATR vectorisé dans `_build_entry_prices`, propagation `max_hold_candles`/`cooldown_candles`/`min_profit_pct` dans `_simulate_grid_multi_tf`
- **param_grids.yaml** : `min_grid_spacing_pct: [0.0, 0.8]` (384 → 768 combos). cooldown_candles fixé à 3 (pas dans param_grids).

### Fix 3 — B2 : UNIQUE constraint sur order_id (database.py)

Index UNIQUE partiel `WHERE order_id IS NOT NULL AND order_id != ''` sur `live_trades`.

### Fix 4 — R4 : fsync dans state_manager.py

`f.flush()` + `os.fsync(f.fileno())` avant `os.replace()` dans `_write_json_file()`.

### Fix 5 — R9 : Leverage réel dans pnl_pct mono (executor.py)

`margin = pos.entry_price * pos.quantity / 3` → `/self._config.risk.position.default_leverage` (2 endroits).

### Fix collatéral — test_sprint46_journal.py

`order_id` auto-incrémenté via `itertools.count()` pour compatibilité avec le nouvel index UNIQUE.

---

## Résultat

- **2081 tests, 2081 passants** (0 échec, 0 régression)
- Moteur live prêt pour déploiement 26 assets
