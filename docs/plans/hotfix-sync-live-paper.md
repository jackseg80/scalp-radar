# Plan : Synchronisation fermetures Live → Paper

## Contexte

L'exit monitor Executor ferme les positions live en intra-candle (prix temps réel toutes les 60s). Le Simulator paper ne ferme qu'au close de la bougie 1h via `should_close_all()`. Résultat : divergence croissante de P&L et de positions entre live et paper, rendant la comparaison sans valeur.

**Solution** : après chaque fermeture live dans `_close_grid_cycle()`, forcer immédiatement la fermeture paper au même prix via une nouvelle méthode `Simulator.force_close_grid()`.

## Fichiers à modifier

| Fichier | Rôle |
|---------|------|
| `backend/backtesting/simulator.py` | Ajouter `force_close_grid()` sur `Simulator` |
| `backend/execution/executor.py` | Appeler `force_close_grid()` dans `_close_grid_cycle()` ET `_handle_grid_sl_executed()` |
| `tests/test_executor_sync_paper.py` | Nouveaux tests (créer) |

## Implémentation

### 1. `backend/backtesting/simulator.py` — méthode `force_close_grid()`

Ajouter une méthode publique sur `Simulator` (après `get_runner_context()`, ligne ~2445) :

```python
def force_close_grid(
    self,
    strategy_name: str,
    symbol: str,
    exit_price: float,
    exit_reason: str,
) -> None:
    """Ferme une position grid paper au même prix que le live (sync live→paper)."""
    from datetime import datetime, timezone
    for runner in self._runners:
        if runner.name != strategy_name:
            continue
        if not hasattr(runner, "_positions"):
            return
        positions = runner._positions.get(symbol, [])
        if not positions:
            return
        # Calcul marge à restituer (même logique que GridStrategyRunner.on_candle)
        margin_to_return = sum(
            (p.notional / runner._leverage) for p in positions
            if hasattr(p, "notional")
        )
        trade = runner._gpm.close_all_positions(
            positions,
            exit_price,
            datetime.now(tz=timezone.utc),
            exit_reason,
            "unknown",
        )
        runner._capital += trade.net_pnl + margin_to_return
        runner._realized_pnl += trade.net_pnl
        runner._positions[symbol] = []
        runner._stats.total_trades += 1
        runner._trades.append((symbol, trade))  # list[tuple[str, TradeResult]]
        logger.info(
            "[{}] SYNC CLOSE {} — {} niveaux, exit={:.6f}, net={:+.2f} ({})",
            strategy_name,
            symbol,
            len(positions),
            exit_price,
            trade.net_pnl,
            exit_reason,
        )
        return
```

**Précautions** :
- Guard `if not positions: return` — si paper déjà fermé (ex: SL naturel), pas d'erreur
- `margin_to_return` recalculé localement (pas de méthode publique existante sur le runner)
- `runner._leverage` : attribut existant sur `GridStrategyRunner`
- `runner._stats.total_trades` incrémenté pour cohérence compteur
- `runner._trades.append(trade)` pour l'historique en mémoire

### 2. `backend/execution/executor.py` — deux emplacements

#### `_close_grid_cycle()` (ligne ~1242)

Après `del self._grid_states[futures_sym]`, ajouter le bloc de sync. `spot_sym` est déjà calculé en début de fonction, `state.strategy_name` est disponible, `exit_price` est connu :

```python
# Synchroniser la fermeture vers le runner paper
if self._simulator is not None:
    try:
        self._simulator.force_close_grid(
            strategy_name=state.strategy_name,
            symbol=spot_sym,
            exit_price=exit_price,
            exit_reason=event.exit_reason or "tp_global",
        )
    except Exception as e:
        logger.warning(
            "Executor: sync close vers paper échoué pour {}: {}", futures_sym, e
        )
```

#### `_handle_grid_sl_executed()` (ligne ~1283)

Même pattern après `del self._grid_states[futures_sym]`. `spot_sym` est disponible dans cette méthode (passé en paramètre), `state.strategy_name` accessible :

```python
if self._simulator is not None:
    try:
        self._simulator.force_close_grid(
            strategy_name=state.strategy_name,
            symbol=spot_sym,
            exit_price=exit_price,
            exit_reason="sl_global",
        )
    except Exception as e:
        logger.warning(
            "Executor: sync SL vers paper échoué pour {}: {}", futures_sym, e
        )
```

**Précautions communes** :

- `try/except` large : la fermeture live ne doit JAMAIS être bloquée par une erreur paper
- Pas de sync dans le cas d'erreur (ligne ~1196, `return` anticipé) : exit_price inconnu

### 3. Tests `tests/test_executor_sync_paper.py`

```python
async def test_close_grid_syncs_to_paper():
    """Fermeture live → paper fermé au même prix."""
    # Setup : mock Executor avec _simulator mock
    # _close_grid_cycle() appelée → vérifier simulator.force_close_grid() appelé
    # avec strategy_name, symbol correct, exit_price = prix réel

async def test_close_grid_paper_sync_failure_doesnt_break_live():
    """Si sync paper échoue, la fermeture live n'est pas affectée."""
    # simulator.force_close_grid() lève une exception
    # Vérifier que la fermeture live s'est bien passée (del _grid_states ok)

async def test_force_close_grid_no_positions():
    """force_close_grid sur symbol sans positions = no-op silencieux."""

async def test_force_close_grid_wrong_strategy():
    """force_close_grid sur mauvais strategy_name = no-op silencieux."""

async def test_force_close_grid_updates_capital():
    """Après force_close_grid, capital du runner reflète le P&L."""
```

## Vérification

1. **Tests** : `uv run pytest tests/test_executor_sync_paper.py -v`
2. **Intégration** : Vérifier dans les logs que le message `SYNC CLOSE` apparaît après chaque `GRID CLOSE` de l'exit monitor
3. **Cohérence** : `GET /api/simulator/status` → vérifier que les positions paper correspondent aux positions live après une fermeture exit monitor

## Points d'attention

- `runner._leverage` : vérifier que l'attribut est bien `_leverage` (pas `_config.leverage`) dans `GridStrategyRunner`
- Si le paper a déjà fermé naturellement (SMA crossing avant l'exit monitor), `positions = []` → guard `if not positions: return` évite tout problème
- Les trades papier générés par `force_close_grid` ne déclenchent PAS d'event vers l'Executor (pas de callback), ce qui est correct (on est côté paper)
