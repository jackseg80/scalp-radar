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
from backend.api.executor_routes import router as executor_router
from backend.api.health import router as health_router
from backend.api.optimization_routes import router as optimization_router
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Démarre tous les composants au lancement, les arrête proprement."""
    config = get_config()
    setup_logging(level=config.secrets.log_level)

    # 1. Database
    db = Database()
    await db.init()
    app.state.db = db
    app.state.start_time = datetime.now(tz=timezone.utc)

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
        engine = DataEngine(config, db)
        await engine.start()
        logger.info("DataEngine démarré via lifespan")
    else:
        logger.info("DataEngine désactivé (ENABLE_WEBSOCKET=false)")

    app.state.engine = engine
    app.state.config = config

    # 4. Simulator + Arena (avec crash recovery)
    simulator: Simulator | None = None
    state_manager: StateManager | None = None
    if engine is not None:
        simulator = Simulator(data_engine=engine, config=config, db=db)

        # Crash recovery : charger l'état AVANT start
        state_manager = StateManager(db)
        saved_state = await state_manager.load_runner_state()

        # start() crée les runners avec le bon état ET enregistre le callback on_candle
        await simulator.start(saved_state=saved_state)

        # Sauvegardes périodiques
        await state_manager.start_periodic_save(simulator)

        app.state.simulator = simulator
        app.state.arena = StrategyArena(simulator)
        logger.info("Simulator + Arena démarrés")
    else:
        app.state.simulator = None
        app.state.arena = None

    # 4b. Executor live (Sprint 5b) — après Simulator/Arena, avant Watchdog
    executor: Executor | None = None
    risk_mgr: LiveRiskManager | None = None
    selector: AdaptiveSelector | None = None
    if config.secrets.live_trading and engine and simulator:
        risk_mgr = LiveRiskManager(config)
        arena = app.state.arena
        selector = AdaptiveSelector(arena, config)
        executor = Executor(config, risk_mgr, notifier, selector=selector)

        # Restaurer l'état avant start
        executor_state = await state_manager.load_executor_state()
        if executor_state:
            risk_mgr.restore_state(executor_state.get("risk_manager", {}))
            executor.restore_positions(executor_state)

        await executor.start()
        await selector.start()
        simulator.set_trade_event_callback(executor.handle_event)
        logger.info("Executor live démarré (sandbox={})", config.secrets.bitget_sandbox)
    elif config.secrets.live_trading:
        logger.warning("LIVE_TRADING=true mais DataEngine/Simulator absents — executor non créé")

    app.state.executor = executor
    app.state.selector = selector

    # 5. Watchdog + Heartbeat (dépendances explicites)
    watchdog: Watchdog | None = None
    heartbeat: Heartbeat | None = None
    if engine is not None and simulator is not None:
        watchdog = Watchdog(
            data_engine=engine, simulator=simulator, notifier=notifier,
            executor=executor,
        )
        await watchdog.start()

        if telegram:
            heartbeat = Heartbeat(
                telegram,
                simulator,
                interval_seconds=config.secrets.heartbeat_interval,
            )
            await heartbeat.start()

    app.state.watchdog = watchdog

    # Notification startup
    if simulator:
        strategies = [r.name for r in simulator.runners]
        await notifier.notify_startup(strategies)

    yield

    # Shutdown (ordre inverse)
    if simulator:
        await notifier.notify_shutdown()
    if heartbeat:
        await heartbeat.stop()
    if watchdog:
        await watchdog.stop()

    # Executor + Selector : sauvegarder état + stop AVANT simulator
    if selector:
        await selector.stop()
    if executor and state_manager and risk_mgr:
        await state_manager.save_executor_state(executor, risk_mgr)
        await executor.stop()
        logger.info("Executor live arrêté et état sauvegardé")

    if state_manager and simulator:
        await state_manager.save_runner_state(simulator.runners)
        await state_manager.stop()
    if simulator:
        await simulator.stop()
    if engine:
        await engine.stop()
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
