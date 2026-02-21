# Hotfix — Remplacer cooldown 3h par tracking positions warm-up

## Contexte

`POST_WARMUP_COOLDOWN_SECONDS = 10800` (3h) bloque TOUS les TradeEvents (OPEN et CLOSE) vers l'Executor pendant 3h après chaque restart. Chaque `deploy.sh` = 3h sans live trading. Le but original (ne pas répliquer les positions warm-up en live) est bon, mais la méthode est trop brutale.

## Approche — Warm-up position tagging

Remplacer le cooldown temporel par un tracking ciblé par symbol : seuls les symbols qui avaient des positions pendant le warm-up sont bloqués, et uniquement pour leur premier cycle open→close.

### Logique

| Event | Symbol dans warmup set ? | Action |
|-------|-------------------------|--------|
| OPEN | NON | PASS → Executor |
| OPEN | OUI | BLOCK (paper seulement) |
| CLOSE | NON | PASS → Executor |
| CLOSE | OUI | BLOCK + retirer symbol du set |

Après retrait, le symbol est "propre" et les cycles suivants passent normalement.

## Fichiers à modifier

### 1. `backend/backtesting/simulator.py` — GridStrategyRunner

#### a) `__init__()` (~L659) — Ajouter le set de tracking

Remplacer le commentaire `POST_WARMUP_COOLDOWN_SECONDS` par `_warmup_position_symbols`:

```python
# Warm-up position tracking : symbols avec positions ouvertes pendant le warm-up
self._warmup_position_symbols: set[str] = set()
```

Garder `_warmup_ended_at` (utilisé par phantom trade guard L922 et kill switch grace L1543).

#### b) `_end_warmup()` (~L801) — Snapshot avant clear

Avant `self._positions.clear()`, capturer les symbols avec positions warm-up. Après `_apply_restored_state()`, exclure les symbols restaurés (ils ont des counterparts live) :

```python
def _end_warmup(self) -> None:
    warmup_trade_count = len(self._trades)

    # Snapshot symbols avec positions pendant le warm-up
    warmup_symbols = {s for s, p in self._positions.items() if p}

    self._positions.clear()

    pending = getattr(self, "_pending_restore", None)
    if pending is not None:
        self._apply_restored_state(pending)
        self._pending_restore = None
        # Exclure les symbols restaurés (counterparts live existants)
        restored_symbols = {s for s, p in self._positions.items() if p}
        warmup_symbols -= restored_symbols
        logger.info(...)
    else:
        # Reset propre (inchangé)
        ...

    self._warmup_position_symbols = warmup_symbols
    self._is_warming_up = False
    self._warmup_ended_at = datetime.now(tz=timezone.utc)

    if warmup_symbols:
        logger.info(
            "[{}] Warmup tracking: {} symbols en cooldown ciblé: {}",
            self.name, len(warmup_symbols), warmup_symbols,
        )
```

#### c) `_emit_open_event()` (~L1277) — Remplacer cooldown par tracking

Supprimer le check `elapsed < POST_WARMUP_COOLDOWN_SECONDS`. Remplacer par :

```python
if symbol in self._warmup_position_symbols:
    logger.info(
        "[{}] WARMUP TRACKING — event OPEN {} supprimé (position warm-up en cours)",
        self.name, symbol,
    )
    return
```

#### d) `_emit_close_event()` (~L1310) — Remplacer cooldown par tracking + retrait

Supprimer le check `elapsed < POST_WARMUP_COOLDOWN_SECONDS`. Remplacer par :

```python
if symbol in self._warmup_position_symbols:
    self._warmup_position_symbols.discard(symbol)
    logger.info(
        "[{}] WARMUP TRACKING — event CLOSE {} supprimé + symbol libéré ({} restants)",
        self.name, symbol, len(self._warmup_position_symbols),
    )
    return
```

#### e) Supprimer `POST_WARMUP_COOLDOWN_SECONDS` (~L577)

La constante n'est plus utilisée, la supprimer (ou la commenter si on veut garder l'historique — je préfère supprimer).

### 2. `tests/test_hotfix_36.py` — Adapter les tests existants + ajouter nouveaux

#### Tests à modifier

- `test_cooldown_blocks_within_time_window` → refactorer en `test_warmup_tracking_blocks_open_for_warmup_symbols`
- `test_cooldown_allows_after_time_window` → refactorer en `test_warmup_tracking_allows_open_for_non_warmup_symbols`
- `test_cooldown_close_also_blocked` → refactorer en `test_warmup_tracking_blocks_close_and_removes_symbol`
- `test_paper_positions_open_during_time_cooldown` → adapter : positions paper s'ouvrent, events bloqués pour warmup symbols
- `test_cooldown_none_warmup_ended_at` → adapter : si warmup set vide, events passent

#### Nouveaux tests

- `test_warmup_tracking_allows_close_after_removal` : après retrait du set, CLOSE passe
- `test_warmup_tracking_complete_cycle` : open bloqué → close bloqué + retrait → prochain open passe
- `test_warmup_tracking_excludes_restored_symbols` : symbols restaurés par `_pending_restore` ne sont pas tagués
- `test_end_warmup_populates_warmup_set` : vérifie que `_end_warmup()` peuple correctement le set

## Vérification

1. `uv run pytest tests/test_hotfix_36.py -v` — tous les tests passent
2. `uv run pytest tests/ -x -q` — pas de régression (1648+ tests)
3. Vérification manuelle : les logs doivent montrer "WARMUP TRACKING" au lieu de "COOLDOWN post-warmup"
