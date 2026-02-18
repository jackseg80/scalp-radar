# Hotfix 35 — Stabilité restart : warm-up, state, et garde-fous Executor

## Contexte

Premier jour live trading. Le bot a crashé/redémarré 3 fois, provoquant à chaque fois :
1. Perte du simulator state → paper repart à zéro avec 1000$ et 0 positions
2. La première bougie "réelle" post warm-up → le runner ouvre des grids sur TOUS les assets où les conditions sont remplies
3. L'Executor reçoit ces événements et ouvre des positions LIVE sur Bitget (8 positions ouvertes en 30 secondes)
4. Le `session_pnl` fantôme (+7.37$) persiste car le RiskManager restaure depuis le state file

### Cause racine

Le warm-up (`_warmup_from_db`) ne charge que les indicateurs (SMA, ATR) dans les buffers. Quand le DataEngine envoie la première bougie live, le runner sort du warm-up avec un état "vierge" (0 positions, 1000$ capital). Comme le marché est bearish, TOUS les assets se trouvent sous leur SMA → les conditions de la grille LONG sont remplies partout → ouverture massive.

Le problème n'est PAS pendant le warm-up (le guard `_is_warming_up` empêche les events). C'est **juste après** le warm-up quand il n'y a pas de state à restaurer.

## Bugs corrigés

### Bug A — Cooldown post-warmup pour les OPEN events (CRITIQUE)

**Fichier** : `backend/backtesting/simulator.py`

**Implémentation** :

- `POST_WARMUP_COOLDOWN = 3` : constante de classe dans `GridStrategyRunner`
- `_post_warmup_candle_count: int = 0` ajouté dans `__init__`
- `_end_warmup()` reset le compteur à 0
- `on_candle()` incrémente `_post_warmup_candle_count` après le guard warm-up (même endroit que `_candles_since_warmup`)
- `_emit_open_event()` et `_emit_close_event()` : guard au début — log INFO + return si `count < POST_WARMUP_COOLDOWN`
- **Les positions paper s'ouvrent normalement** — seul l'envoi à l'Executor est bloqué

### Bug B — Executor : limiter les positions simultanées (IMPORTANT)

**Fichiers** : `backend/execution/executor.py`, `config/risk.yaml`, `backend/core/config.py`

**Implémentation** :

- `max_live_grids: 4` ajouté dans `risk.yaml`
- `max_live_grids: int = Field(default=4, ge=1)` dans `RiskConfig`
- Guard dans `_open_grid_position()` AVANT le pre-trade check :
  - `is_first_level` détecté avant le guard (ne bloque pas les niveaux DCA existants)
  - Lecture sécurisée : `isinstance(max_live_grids_raw, int)` avec fallback 4 (robustesse MagicMock)

### Bug C — Sauvegarde executor_state dans periodic_save (IMPORTANT)

**Fichiers** : `backend/core/state_manager.py`, `backend/api/server.py`

**Implémentation** :

- `_executor: Any = None` et `_risk_manager: Any = None` dans `StateManager.__init__`
- `set_executor(executor, risk_manager)` : nouvelle méthode publique
- `_periodic_save_loop()` appelle `save_executor_state()` toutes les 60s si `_executor is not None`
- `server.py` appelle `state_manager.set_executor(executor, risk_mgr)` après `executor.start()`

## Tests — `tests/test_hotfix_35.py` (14 nouveaux tests)

### TestCooldownPostWarmup (7 tests)

| Test | Vérification |
|------|-------------|
| `test_emit_open_blocked_during_cooldown` | `_emit_open_event` supprime l'event avec count=0 |
| `test_emit_open_allowed_after_cooldown` | `_emit_open_event` émet normalement avec count=3 |
| `test_emit_close_blocked_during_cooldown` | `_emit_close_event` supprime l'event avec count=1 |
| `test_cooldown_counter_increments_on_candle` | count s'incrémente à chaque `on_candle` post-warmup |
| `test_warmup_end_resets_counter` | `_end_warmup()` reset count à 0 |
| `test_paper_positions_open_during_cooldown` | Positions paper s'ouvrent malgré cooldown |
| `test_events_emitted_after_cooldown` | Events émis normalement après count=3 |

### TestMaxLiveGrids (3 tests)

| Test | Vérification |
|------|-------------|
| `test_max_grids_blocks_new_cycle` | 5ème cycle refusé avec max=2 et 2 actifs |
| `test_max_grids_allows_additional_levels` | Level 1 sur cycle existant passe toujours |
| `test_max_grids_default_from_config` | Lecture config.risk.max_live_grids |

### TestPeriodicSaveExecutor (4 tests)

| Test | Vérification |
|------|-------------|
| `test_set_executor_registers_references` | `set_executor()` enregistre les références |
| `test_initial_executor_is_none` | `_executor` None par défaut |
| `test_periodic_save_includes_executor` | `save_executor_state` appelé si executor enregistré |
| `test_periodic_save_without_executor_no_error` | Pas d'erreur si executor None |

## Tests adaptés

- `tests/test_grid_runner.py::test_pending_events_trade_event_format` : ajout `runner._post_warmup_candle_count = 3` pour simuler le cooldown passé

## Fichiers modifiés

| Fichier | Modification |
|---------|-------------|
| `backend/backtesting/simulator.py` | Constante + attribut + _end_warmup + on_candle + _emit_open/close_event |
| `backend/execution/executor.py` | Guard max_live_grids dans `_open_grid_position` |
| `backend/core/state_manager.py` | `_executor`, `_risk_manager`, `set_executor()`, boucle périodique |
| `backend/api/server.py` | `state_manager.set_executor(executor, risk_mgr)` |
| `backend/core/config.py` | `max_live_grids: int = Field(default=4, ge=1)` dans `RiskConfig` |
| `config/risk.yaml` | `max_live_grids: 4` |
| `tests/test_hotfix_35.py` | NOUVEAU — 14 tests |
| `tests/test_grid_runner.py` | 1 test adapté (cooldown count forcé à 3) |

## Résultats

- **1353 tests** au total (14 nouveaux), 0 régression
- Crashs ProcessPool/Numba (`test_job_manager_wfo_integration`, `test_walk_forward_callback`) préexistants, non liés

## Monitoring post-deploy

```bash
docker compose restart backend
docker compose logs -f backend | grep -E "(COOLDOWN|max grids|executor sauvegard)"

# Attendu :
# [grid_atr] COOLDOWN post-warmup (1/3) — event OPEN BTC/USDT supprimé pour Executor
# [grid_atr] COOLDOWN post-warmup (2/3) — event OPEN DOGE/USDT supprimé pour Executor
# StateManager: état executor sauvegardé
```
