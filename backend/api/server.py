"""Serveur FastAPI avec DataEngine intégré via lifespan.

Un seul process gère l'API REST + le DataEngine WebSocket.
Le flag ENABLE_WEBSOCKET permet de désactiver le DataEngine en dev.
Sprint 4 : StateManager, Notifier, Watchdog, Heartbeat.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.alerts.heartbeat import Heartbeat
from backend.alerts.notifier import Notifier
from backend.alerts.telegram import TelegramClient
from backend.api.arena_routes import router as arena_router
from backend.api.conditions_routes import router as conditions_router
from backend.api.data_routes import router as data_router
from backend.api.executor_routes import router as executor_router
from backend.api.health import router as health_router
from backend.api.journal_routes import router as journal_router
from backend.api.log_routes import router as log_router
from backend.api.optimization_routes import router as optimization_router
from backend.api.portfolio_routes import router as portfolio_router
from backend.api.signals_routes import router as signals_router
from backend.api.simulator_routes import router as simulator_router
from backend.api.websocket_routes import router as ws_router
from backend.backtesting.arena import StrategyArena
from backend.backtesting.simulator import Simulator
from backend.core.config import get_config
from backend.core.data_engine import DataEngine
from backend.core.database import Database
from backend.core.logging_setup import setup_logging
from backend.core.state_manager import StateManager
from backend.execution.adaptive_selector import AdaptiveSelector
from backend.execution.executor import Executor
from backend.execution.risk_manager import LiveRiskManager
from backend.monitoring.watchdog import Watchdog


def _get_live_eligible_strategies(config) -> list[str]:
    """Retourne les noms de stratégies enabled + live_eligible."""
    result: list[str] = []
    for name in config.strategies.model_fields:
        if name == "custom_strategies":
            continue
        cfg = getattr(config.strategies, name, None)
        if cfg is None:
            continue
        if getattr(cfg, "enabled", False) and getattr(cfg, "live_eligible", False):
            result.append(name)
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Démarre tous les composants au lancement, les arrête proprement."""
    config = get_config()
    setup_logging(level=config.secrets.log_level)

    # 1. Database
    db = Database()
    await db.init()
    db.start_maintenance_loop()
    app.state.db = db
    app.state.start_time = datetime.now(tz=timezone.utc)

    # 1b. JobManager (Sprint 14)
    from backend.optimization.job_manager import JobManager
    from backend.api.websocket_routes import manager as ws_manager

    db_path_str = str(db.db_path) if db.db_path else "data/scalp_radar.db"
    job_manager = JobManager(db_path=db_path_str, ws_broadcast=ws_manager.broadcast)
    await job_manager.start()
    app.state.job_manager = job_manager
    logger.info("JobManager démarré (worker loop actif)")

    # 2. Telegram + Notifier (si token configuré)
    telegram: TelegramClient | None = None
    if config.secrets.telegram_bot_token:
        telegram = TelegramClient(
            config.secrets.telegram_bot_token,
            config.secrets.telegram_chat_id,
        )
    notifier = Notifier(telegram)
    app.state.notifier = notifier

    # 3. DataEngine (optionnel via ENABLE_WEBSOCKET)
    engine: DataEngine | None = None
    if config.secrets.enable_websocket:
        engine = DataEngine(config, db, notifier=notifier)
        await engine.start()
        logger.info("DataEngine démarré via lifespan")
    else:
        logger.info("DataEngine désactivé (ENABLE_WEBSOCKET=false)")

    app.state.engine = engine
    app.state.config = config

    # 3b. CandleUpdater (daily backfill)
    from backend.core.candle_updater import CandleUpdater

    candle_updater = CandleUpdater(config, db, ws_broadcast=ws_manager.broadcast)
    await candle_updater.start()
    app.state.candle_updater = candle_updater

    # 4. Simulator + Arena (avec crash recovery)
    simulator: Simulator | None = None
    state_manager: StateManager | None = None
    if engine is not None:
        simulator = Simulator(data_engine=engine, config=config, db=db)

        # Crash recovery : charger l'état AVANT start
        state_manager = StateManager(db)
        saved_state = await state_manager.load_runner_state()

        # Notifier pour le kill switch global
        simulator.set_notifier(notifier)

        # start() crée les runners avec le bon état ET enregistre le callback on_candle
        await simulator.start(saved_state=saved_state)

        # Sauvegardes périodiques
        await state_manager.start_periodic_save(simulator)

        app.state.simulator = simulator
        app.state.state_manager = state_manager
        app.state.arena = StrategyArena(simulator)
        logger.info("Simulator + Arena démarrés")
    else:
        app.state.simulator = None
        app.state.state_manager = None
        app.state.arena = None

    # 4b. Multi-Executor live (Sprint 36b) — un Executor par stratégie live
    from backend.execution.executor_manager import ExecutorManager

    executor_mgr = ExecutorManager()
    selector: AdaptiveSelector | None = None

    if config.secrets.live_trading and engine and simulator:
        arena = app.state.arena
        selector = AdaptiveSelector(arena, config, db=db)
        live_strategies = _get_live_eligible_strategies(config)

        for strat_name in live_strategies:
            risk_mgr = LiveRiskManager(config, notifier=notifier)
            executor = Executor(
                config, risk_mgr, notifier,
                selector=selector, strategy_name=strat_name,
            )

            # Restaurer l'état per-strategy avant start
            strat_state = await state_manager.load_executor_state(
                strategy_name=strat_name,
            )
            if strat_state:
                risk_mgr.restore_state(strat_state.get("risk_manager", {}))
                executor.restore_positions(strat_state)

            await executor.start()

            # Warning capital mismatch par sous-compte
            if executor.exchange_balance is not None:
                config_capital = config.risk.initial_capital
                real_balance = executor.exchange_balance
                if config_capital > 0:
                    diff_pct = abs(real_balance - config_capital) / config_capital * 100
                    if diff_pct > 20:
                        msg = (
                            f"Capital mismatch [{strat_name}]: "
                            f"risk.yaml={config_capital:.0f}$ "
                            f"vs Bitget={real_balance:.0f}$ (écart {diff_pct:.0f}%)"
                        )
                        logger.warning(msg)
                        await notifier.notify_reconciliation(msg)

            dedicated = config.has_dedicated_keys(strat_name)
            logger.info(
                "Executor[{}] démarré (sous-compte: {})",
                strat_name, "dédié" if dedicated else "global/partagé",
            )
            executor_mgr.add(strat_name, executor, risk_mgr)

        # Warning si clés globales partagées entre >1 executor
        if len(live_strategies) > 1 and not all(
            config.has_dedicated_keys(s) for s in live_strategies
        ):
            logger.warning(
                "Multi-Executor: certains executors partagent les mêmes clés API "
                "— risque de rate limit. Recommandé : sous-comptes dédiés."
            )

        await selector.start()

        # Câblage exit monitor + entrées autonomes PAR EXECUTOR
        strategy_instances = simulator.get_strategy_instances()
        from backend.execution.sync import sync_live_to_paper

        for strat_name, executor in executor_mgr.executors.items():
            executor.set_db(db)  # Sprint 45 : persist live trades
            executor.set_data_engine(engine)
            executor.set_strategies(strategy_instances, simulator=simulator)
            await sync_live_to_paper(executor, simulator)
            await executor.start_exit_monitor()

            # Phase 1 : entrées autonomes — chaque Executor reçoit les candles
            # INVARIANT : simulator.start() a déjà enregistré son callback (index 0).
            engine.on_candle(executor._on_candle)

        # Sauvegarde périodique multi-executor
        if state_manager is not None:
            state_manager.set_executors(executor_mgr)

        logger.info(
            "Multi-Executor: {} executors démarrés ({})",
            len(executor_mgr.executors),
            list(executor_mgr.executors.keys()),
        )
    elif config.secrets.live_trading:
        logger.warning("LIVE_TRADING=true mais DataEngine/Simulator absents — executor non créé")

    # app.state — backward compat via duck typing (ExecutorManager.get_status())
    app.state.executor = executor_mgr if executor_mgr.executors else None
    app.state.executor_mgr = executor_mgr
    app.state.risk_mgr = None  # Plus de singleton, utiliser executor_mgr.risk_managers
    app.state.selector = selector

    # 5. Watchdog + Heartbeat + WeeklyReporter (dépendances explicites)
    watchdog: Watchdog | None = None
    heartbeat: Heartbeat | None = None
    weekly_reporter = None
    if engine is not None and simulator is not None:
        watchdog = Watchdog(
            data_engine=engine, simulator=simulator, notifier=notifier,
            executor_mgr=executor_mgr if executor_mgr.executors else None,
        )
        await watchdog.start()

        if telegram:
            heartbeat = Heartbeat(
                telegram,
                simulator,
                interval_seconds=config.secrets.heartbeat_interval,
            )
            await heartbeat.start()

            # Sprint 49 : rapport hebdomadaire (lundi 08:00 UTC)
            from backend.alerts.weekly_reporter import WeeklyReporter

            weekly_reporter = WeeklyReporter(telegram, db, config)
            await weekly_reporter.start()

    app.state.watchdog = watchdog

    # Notification startup
    if simulator:
        strategies = [r.name for r in simulator.runners]
        await notifier.notify_startup(strategies)

    yield

    # Shutdown (ordre inverse)
    if simulator:
        await notifier.notify_shutdown()
    if weekly_reporter:
        await weekly_reporter.stop()
    if heartbeat:
        await heartbeat.stop()
    if watchdog:
        await watchdog.stop()

    # Executor + Selector : sauvegarder état + stop AVANT simulator
    if selector:
        await selector.stop()
    if executor_mgr and executor_mgr.executors and state_manager:
        for _name, _ex in executor_mgr.executors.items():
            _rm = executor_mgr.risk_managers.get(_name)
            if _rm:
                await state_manager.save_executor_state(_ex, _rm, strategy_name=_name)
        await executor_mgr.stop_all()
        logger.info("Multi-Executor: tous les executors arrêtés et états sauvegardés")

    if state_manager and simulator:
        await state_manager.save_runner_state(
            simulator.runners,
            global_kill_switch=simulator._global_kill_switch,
            kill_switch_reason=simulator._kill_switch_reason,
        )
        await state_manager.stop()
    if simulator:
        await simulator.stop()
    if engine:
        await engine.stop()

    # CandleUpdater : arrêter la boucle quotidienne
    if hasattr(app.state, "candle_updater") and app.state.candle_updater:
        await app.state.candle_updater.stop()

    # JobManager : arrêter le worker loop
    if hasattr(app.state, "job_manager") and app.state.job_manager:
        await app.state.job_manager.stop()
        logger.info("JobManager arrêté")

    await db.close()
    logger.info("Shutdown complet")


app = FastAPI(
    title="Scalp Radar API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS pour le frontend en dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(simulator_router)
app.include_router(arena_router)
app.include_router(signals_router)
app.include_router(conditions_router)
app.include_router(ws_router)
app.include_router(executor_router)
app.include_router(optimization_router)
app.include_router(portfolio_router)
app.include_router(journal_router)
app.include_router(log_router)
app.include_router(data_router)
