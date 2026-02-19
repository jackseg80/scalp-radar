"""Synchronisation positions live ↔ paper au démarrage.

Principe : le LIVE fait autorité (c'est l'argent réel).
- Position live sans miroir paper → injectée dans le paper
- Position paper sans miroir live → supprimée du paper
- Position live + paper → pas touchée
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from backend.backtesting.simulator import GridStrategyRunner, Simulator
    from backend.execution.executor import Executor, GridLiveState


async def sync_live_to_paper(executor: Executor, simulator: Simulator) -> None:
    """Synchronise les positions live avec le paper après un restart."""
    from backend.backtesting.simulator import GridStrategyRunner

    # Si executor._grid_states est vide, tenter de le peupler depuis Bitget
    if not executor._grid_states:
        await _populate_grid_states_from_exchange(executor, simulator)

    # Mapper runners par nom de stratégie
    runner_map: dict[str, GridStrategyRunner] = {
        r.name: r for r in simulator.runners if isinstance(r, GridStrategyRunner)
    }

    live_symbols_by_runner: dict[str, set[str]] = {}

    for futures_sym, state in executor._grid_states.items():
        spot_sym = futures_sym.split(":")[0] if ":" in futures_sym else futures_sym
        runner = runner_map.get(state.strategy_name)

        if runner is None:
            logger.warning(
                "Sync: runner {} introuvable pour {}",
                state.strategy_name, futures_sym,
            )
            continue

        # Tracker les symbols live par runner
        live_symbols_by_runner.setdefault(runner.name, set()).add(spot_sym)

        # Le paper a-t-il des positions sur ce symbol ?
        paper_positions = runner._positions.get(spot_sym, [])

        if not paper_positions:
            # INJECTION : créer des positions paper miroir
            _inject_live_to_paper(runner, spot_sym, state)
            logger.info(
                "Sync: {} injecté dans paper {} ({} niveaux, avg={:.6f})",
                spot_sym, runner.name, len(state.positions), state.avg_entry_price,
            )

    # Nettoyer les positions paper sans miroir live
    for runner_name, runner in runner_map.items():
        live_syms = live_symbols_by_runner.get(runner_name, set())
        for spot_sym in list(runner._positions.keys()):
            if spot_sym not in live_syms and runner._positions[spot_sym]:
                removed_count = len(runner._positions[spot_sym])
                # Rendre la marge au capital
                leverage = runner._leverage
                for pos in runner._positions[spot_sym]:
                    margin = pos.entry_price * pos.quantity / leverage
                    runner._capital += margin
                runner._positions[spot_sym] = []
                logger.info(
                    "Sync: {} supprimé du paper {} ({} positions, pas de miroir live)",
                    spot_sym, runner_name, removed_count,
                )

    total_live = sum(len(s) for s in live_symbols_by_runner.values())
    logger.info("Sync: terminé — {} symbols live synchronisés", total_live)


async def _populate_grid_states_from_exchange(
    executor: Executor,
    simulator: Simulator,
) -> None:
    """Fetch les positions Bitget et crée les GridLiveState manquants dans l'executor.

    Appelé quand executor._grid_states est vide au boot (state file absent ou corrompu).
    Sans cette étape, l'exit monitor autonome n'a rien à checker.
    """
    from backend.backtesting.simulator import GridStrategyRunner
    from backend.execution.executor import GridLivePosition, GridLiveState

    try:
        all_positions = await executor._exchange.fetch_positions()
    except Exception as e:
        logger.error("Sync: erreur fetch_positions pour peupler grid_states: {}", e)
        return

    # Mapper symbol spot → stratégie via les runners paper
    symbol_to_strategy: dict[str, str] = {}
    for r in simulator.runners:
        if isinstance(r, GridStrategyRunner):
            for spot_sym in r._positions:
                symbol_to_strategy[spot_sym] = r.name

    created = 0
    for pos in all_positions:
        contracts = float(pos.get("contracts", 0) or 0)
        if contracts <= 0:
            continue

        futures_sym: str = pos["symbol"]  # ex: "FET/USDT:USDT"
        spot_sym = futures_sym.split(":")[0] if ":" in futures_sym else futures_sym

        entry_price = float(pos.get("entryPrice", 0) or 0)
        if entry_price <= 0:
            continue

        direction = (pos.get("side", "long") or "long").upper()  # "LONG" | "SHORT"

        # Stratégie depuis le paper, fallback grid_atr
        strategy_name = symbol_to_strategy.get(spot_sym, "grid_atr")

        # Leverage depuis la config de stratégie
        leverage = 3  # fallback
        try:
            strat_cfg = getattr(executor._config.strategies, strategy_name, None)
            raw_lev = getattr(strat_cfg, "leverage", None)
            if isinstance(raw_lev, (int, float)) and raw_lev > 0:
                leverage = int(raw_lev)
        except Exception:
            pass

        grid_state = GridLiveState(
            symbol=futures_sym,
            direction=direction,
            strategy_name=strategy_name,
            leverage=leverage,
            positions=[
                GridLivePosition(
                    level=0,
                    entry_price=entry_price,
                    quantity=contracts,
                    entry_order_id="restored-from-sync",
                )
            ],
            sl_price=0.0,
        )

        executor._grid_states[futures_sym] = grid_state
        created += 1
        logger.info(
            "Sync: grid_state créé pour {} ({}, {:.6g} contracts @ {:.6g})",
            futures_sym, strategy_name, contracts, entry_price,
        )

    if created:
        logger.info("Sync: {} grid_states créés depuis l'exchange", created)


def _inject_live_to_paper(
    runner: GridStrategyRunner,
    spot_sym: str,
    live_state: GridLiveState,
) -> None:
    """Injecte les positions live dans le runner paper."""
    from backend.core.models import Direction
    from backend.strategies.base_grid import GridPosition

    positions = []
    leverage = runner._leverage

    for lp in live_state.positions:
        gp = GridPosition(
            level=lp.level,
            direction=Direction(live_state.direction),
            entry_price=lp.entry_price,
            quantity=lp.quantity,
            entry_time=lp.entry_time,
            entry_fee=getattr(lp, "entry_fee", lp.entry_price * lp.quantity * 0.0006),
        )
        positions.append(gp)

        # Déduire la marge du capital paper
        margin = lp.entry_price * lp.quantity / leverage
        runner._capital -= margin

    runner._positions[spot_sym] = positions
