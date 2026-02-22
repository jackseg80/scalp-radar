# Phase 2 — Anti-churning : Cooldown par Symbol après Close

## Objectif

Empêcher la réouverture immédiate d'une grille sur le même symbol après fermeture.
Le churning (rouverture excessive) amplifie les fees et peut transformer un signal
légèrement positif en perte nette.

## Mécanique

Après chaque close grid, `_last_close_time[symbol]` est horodaté.
Avant `compute_grid()`, on vérifie que `elapsed >= cooldown_candles × tf_seconds`.
Si en cooldown → skip (retour immédiat, sans évaluer les niveaux DCA).

## Paramètres

- `cooldown_candles: int = Field(default=3, ge=0)` dans `GridATRConfig` et `GridBolTrendConfig`
- Default 3 bougies 1h = 3h de refroidissement
- `ge=0` → 0 désactive complètement (backward compat)
- Configurable par stratégie dans `config/strategies.yaml`

## Bugs identifiés et corrigés avant implémentation

### BUG 1 — Guard warm-up manquant (CRITIQUE)

Sans guard, les closes pendant le warm-up (bougies historiques) enregistrent
`_last_close_time`, bloquant les vraies entrées post-warmup pendant 3h.

**Fix** : `_record_close()` appelé uniquement sous `if not self._is_warming_up`.

### BUG 2 — `force_close_grid()` n'a pas de candle

Le plan utilisait `candle.timestamp` dans `force_close_grid()`, mais cette méthode
n'a pas de candle en paramètre (c'est un close synchrone).

**Fix** : Utiliser `trade.exit_time` (l'heure réelle du close dans le trade).

### BUG 3 — Persistence simulator manquante (CRITIQUE)

Sans sauvegarde, les cooldowns sont perdus à chaque restart → pas d'effet après reboot.

**Fix** : StateManager sauvegarde `last_close_times` dans `simulator_state.json`,
`_apply_restored_state()` le restaure.

### BUG 4 — `_TF_SECONDS` dupliqué (3 copies)

`grid_atr.py` a `_TF_SECONDS`, `grid_boltrend.py` avait son propre dict.
Le plan proposait d'en ajouter un 3e dans `simulator.py`.

**Fix** : `TF_SECONDS` extrait dans `base_grid.py` (source unique), importé partout.

### BUG 5 — MagicMock trap (CRITIQUE en tests)

`getattr(mock._config, "cooldown_candles", 0)` retourne un MagicMock (pas 0)
car MagicMock crée les attributs à la volée. Ensuite `MagicMock > 0` → TypeError.

**Fix** : `isinstance(raw_cd, (int, float))` avant la comparaison.

### BUG 6 — 5 chemins `del _grid_states`, pas 4

L'audit a identifié un 5e chemin (emergency close) manqué dans le plan.

**Fix** : Tous les 5 `del self._grid_states[futures_sym]` dans executor.py
appellent `_record_grid_close()` avant suppression.

## Architecture finale

### Paper (GridStrategyRunner)

```python
# __init__
self._last_close_time: dict[str, datetime] = {}

# Nouveau
def _record_close(self, symbol: str, close_timestamp: datetime) -> None:
    self._last_close_time[symbol] = close_timestamp

# _on_candle_inner — après darwinian filter, avant compute_grid
if not positions:
    raw_cd = getattr(self._strategy._config, "cooldown_candles", 0)
    cooldown = raw_cd if isinstance(raw_cd, (int, float)) else 0
    if cooldown > 0 and symbol in self._last_close_time:
        tf_seconds = TF_SECONDS.get(self._strategy_tf, 3600)
        elapsed = (candle.timestamp - self._last_close_time[symbol]).total_seconds()
        if elapsed < cooldown * tf_seconds:
            return

# Chemin TP/SL close
if not self._is_warming_up:
    self._record_trade(trade, symbol)
    self._record_close(symbol, candle.timestamp)  # ← warm-up guard implicite

# force_close_grid (sync)
runner._record_close(symbol, trade.exit_time)
```

### StateManager

```python
# save_runner_state
close_times = getattr(runner, "_last_close_time", {})
if isinstance(close_times, dict) and close_times:
    runner_state["last_close_times"] = {
        sym: ts.isoformat() for sym, ts in close_times.items()
    }

# _apply_restored_state
for sym, ts_str in state.get("last_close_times", {}).items():
    try:
        self._last_close_time[sym] = datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        pass
```

### Live (Executor)

```python
# __init__
self._last_close_time: dict[str, datetime] = {}

# Nouveau
def _record_grid_close(self, futures_sym: str) -> None:
    self._last_close_time[futures_sym] = datetime.now(tz=timezone.utc)

# _on_candle — après stratégie/timeframe filter, avant get_runner_context
raw_cd = getattr(strategy._config, "cooldown_candles", 0)
cooldown = raw_cd if isinstance(raw_cd, (int, float)) else 0
if cooldown > 0 and futures_sym not in self._grid_states:
    if futures_sym in self._last_close_time:
        tf_seconds = TF_SECONDS.get(strat_tf, 3600)
        elapsed = (candle.timestamp - self._last_close_time[futures_sym]).total_seconds()
        if elapsed < cooldown * tf_seconds:
            continue

# 5 chemins del _grid_states → _record_grid_close avant chaque del
```

### Persistence Executor

```python
# get_state_for_persistence
"last_close_times": {sym: ts.isoformat() for sym, ts in self._last_close_time.items()}

# restore_positions
for sym, ts_str in state.get("last_close_times", {}).items():
    self._last_close_time[sym] = datetime.fromisoformat(ts_str)
```

### Fast engine

Warning uniquement : `cooldown_candles > 0` → `warnings.warn(...)`. Non implémenté en
Numba (backtests légèrement optimistes si cooldown configuré).

## Tests — 20 dans `tests/test_cooldown_churning.py`

### `TestCooldownRunner` (7 tests)
- `test_cooldown_blocks_reentry_after_close` : T+1h/T+2h bloqués, T+3h OK
- `test_cooldown_zero_disables` : cooldown=0 → compute_grid immédiat
- `test_cooldown_per_symbol` : BTC en cooldown n'affecte pas ETH
- `test_cooldown_does_not_block_exits` : positions ouvertes → exits toujours évalués
- `test_record_close_called_on_tp` : close TP → `_last_close_time` enregistré
- `test_cooldown_not_recorded_during_warmup` : warm-up → PAS enregistré
- `test_record_close_called_on_should_close_all` : close signal → enregistré

### `TestCooldownExecutor` (4 tests)
- `test_executor_cooldown_blocks_entry` : skip avant get_runner_context
- `test_executor_cooldown_expired_allows_entry` : 4h → get_runner_context appelé
- `test_executor_record_close_in_close_grid_cycle` : `_record_grid_close()` fonctionne
- `test_executor_cooldown_survives_restart` : get_state/restore round-trip

### `TestCooldownConfig` (4 tests)
- Default grid_atr = 3
- Default grid_boltrend = 3
- Zéro valide
- Inclus dans `get_params_for_symbol()`

### `TestTFSeconds` (3 tests)
- Importable depuis base_grid
- grid_atr n'a plus `_TF_SECONDS` local
- grid_boltrend n'a plus `_TF_SECONDS` local

### `TestCooldownPersistence` (2 tests)
- StateManager sérialise `_last_close_time`
- `_apply_restored_state` restaure correctement

## Résultats

- **20 tests nouveaux** — tous passants
- **1716 tests totaux**, 0 régression
- Guard MagicMock : pattern `isinstance(raw, (int, float))` ajouté au MEMORY.md comme piège résolu
