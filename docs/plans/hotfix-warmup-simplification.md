# Hotfix — Supprimer le warmup tracking, garder uniquement is_warming_up

## Contexte

Le warmup tracking (_warmup_position_symbols) et l'ancien cooldown 3h résolvaient
un problème qui n'existe pas : les protections existantes (MarginGuard 70%,
kill switch 45%, max positions, correlation groups) couvrent déjà tous les risques
de mass entry au restart.

Le warmup tracking crée un vrai problème : paper et live divergent, rendant la
comparaison du journal impossible.

## Objectif

Simplifier : la seule protection nécessaire est `if self._is_warming_up: return`
dans les _emit_*_event(). Rien d'autre.

## Modifications dans `backend/backtesting/simulator.py`

### Supprimer

1. L'attribut `_warmup_position_symbols: set[str]` dans `__init__()`
2. Tout le code de snapshot warmup symbols dans `_end_warmup()` — garder uniquement
   le clear des positions, l'apply restored state, et les flags
   `_is_warming_up = False` / `_warmup_ended_at = datetime.now()`
3. Dans `_emit_open_event()` : remplacer `if symbol in _warmup_position_symbols`
   par `if self._is_warming_up: return`
4. Dans `_emit_close_event()` : idem (sans le discard())

### Garder

1. `self._is_warming_up` flag — utilisé partout
2. `self._warmup_ended_at` — utilisé par phantom trade guard et kill switch grace
3. Le guard `if self._is_warming_up: return` au DÉBUT de `_emit_open_event()`
   et `_emit_close_event()`

### `_end_warmup()` simplifiée

```python
def _end_warmup(self) -> None:
    warmup_trade_count = len(self._trades)
    # Clear positions warm-up
    self._positions.clear()

    # Restaurer l'état sauvegardé si disponible
    pending = getattr(self, "_pending_restore", None)
    if pending is not None:
        self._apply_restored_state(pending)
        self._pending_restore = None
        logger.info(...)
    else:
        # Reset propre
        self._trades.clear()
        self._capital = self._initial_capital
        self._stats = RunnerStats(...)

    self._is_warming_up = False
    self._warmup_ended_at = datetime.now(tz=timezone.utc)
    logger.info("[{}] Warm-up terminé", self.name)
```

## Tests

### `tests/test_hotfix_36.py` — `TestWarmupPositionTracking` → `TestIsWarmingUpGuard`

7 tests basés sur `is_warming_up` :
- test_emit_open_blocked_during_warmup
- test_emit_open_passes_after_warmup
- test_emit_close_blocked_during_warmup
- test_emit_close_passes_after_warmup
- test_end_warmup_clears_positions_and_sets_flags
- test_end_warmup_restores_saved_state
- test_paper_positions_open_during_warmup (bougie ancienne age > 2h pour rester en warmup)

### `tests/test_hotfix_35.py` — `TestWarmupPositionTracking35` → `TestIsWarmingUpGuard35`

8 tests basés sur `is_warming_up` (même pattern)

## Piège résolu

**Bougie récente déclenche _end_warmup() automatiquement** : `on_candle()` appelle
`_end_warmup()` si `candle_age <= WARMUP_AGE_THRESHOLD` (2h). Pour tester que
les events sont bloqués pendant le warmup via `on_candle()`, utiliser une bougie
ancienne (`ts = datetime.now() - timedelta(hours=3)`).

## Validation

```bash
uv run python -m pytest tests/test_hotfix_35.py tests/test_hotfix_36.py -x -v
# → 29/29
uv run python -m pytest -q
# → 1667 passants
```

## Résultat

- Paper et live parfaitement synchronisés (plus de blocage ciblé par symbol)
- Zéro divergence au restart
- Code simplifié : `_end_warmup()` -10 lignes, `_emit_*_event()` -8 lignes
- Les protections existantes (MarginGuard 70%, kill switch 45%, max positions,
  correlation groups) gèrent les risques de mass entry
