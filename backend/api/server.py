"""Serveur FastAPI avec DataEngine intégré via lifespan.

Un seul process gère l'API REST + le DataEngine WebSocket.
Le flag ENABLE_WEBSOCKET permet de désactiver le DataEngine en dev.
Sprint 4 : StateManager, Notifier, Watchdog, Heartbeat.
Sprint Audit-A : hardening lifespan (try-except, timeout shutdown, health status).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

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
from backend.api.alerts_routes import router as alerts_router
from backend.api.regime_routes import router as regime_router
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

# Timeout pour chaque étape de shutdown (secondes)
_SHUTDOWN_TIMEOUT = 30


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


async def _safe_stop(name: str, coro, timeout: float = _SHUTDOWN_TIMEOUT) -> None:
    """Arrête un composant avec timeout et gestion d'erreur."""
    try:
        await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("Shutdown timeout ({}s) pour {}", timeout, name)
    except Exception as exc:
        logger.error("Erreur shutdown {} : {}", name, exc)


# ─── Helpers d'initialisation ─────────────────────────────────────────────


async def _init_database(app: FastAPI) -> Database:
    """Initialise la base de données. CRITIQUE — raise si échec."""
    db = Database()
    await db.init()
    db.start_maintenance_loop()
    app.state.db = db
    app.state.start_time = datetime.now(tz=timezone.utc)
    return db


async def _init_job_manager(app: FastAPI, db: Database) -> Any:
    """Initialise le JobManager pour les jobs WFO."""
    from backend.optimization.job_manager import JobManager
    from backend.api.websocket_routes import manager as ws_manager

    db_path_str = str(db.db_path) if db.db_path else "data/scalp_radar.db"
    job_manager = JobManager(db_path=db_path_str, ws_broadcast=ws_manager.broadcast)
    await job_manager.start()
    app.state.job_manager = job_manager
    logger.info("JobManager démarré (worker loop actif)")
    return job_manager


def _init_notifier(
    app: FastAPI, config, db: Database,
) -> tuple[TelegramClient | None, Notifier]:
    """Initialise Telegram + Notifier."""
    telegram: TelegramClient | None = None
    if config.secrets.telegram_bot_token:
        telegram = TelegramClient(
            config.secrets.telegram_bot_token,
            config.secrets.telegram_chat_id,
        )
        telegram.set_db(db)
    notifier = Notifier(telegram)
    app.state.notifier = notifier
    return telegram, notifier


async def _init_data_engine(
    app: FastAPI, config, db: Database, notifier: Notifier,
) -> DataEngine | None:
    """Initialise le DataEngine (optionnel via ENABLE_WEBSOCKET)."""
    engine: DataEngine | None = None
    if config.secrets.enable_websocket:
        engine = DataEngine(config, db, notifier=notifier)
        await engine.start()
        logger.info("DataEngine démarré via lifespan")
    else:
        logger.info("DataEngine désactivé (ENABLE_WEBSOCKET=false)")
    app.state.engine = engine
    app.state.config = config
    return engine


async def _init_candle_updater(app: FastAPI, config, db: Database) -> Any:
    """Initialise le CandleUpdater (daily backfill)."""
    from backend.core.candle_updater import CandleUpdater
    from backend.api.websocket_routes import manager as ws_manager

    candle_updater = CandleUpdater(config, db, ws_broadcast=ws_manager.broadcast)
    await candle_updater.start()
    app.state.candle_updater = candle_updater
    return candle_updater


async def _init_simulator(
    app: FastAPI, config, db: Database, engine: DataEngine, notifier: Notifier,
) -> tuple[Simulator, StateManager]:
    """Initialise Simulator + StateManager + Arena."""
    simulator = Simulator(data_engine=engine, config=config, db=db)
    state_manager = StateManager(db)
    saved_state = await state_manager.load_runner_state()
    simulator.set_notifier(notifier)
    await simulator.start(saved_state=saved_state)
    await state_manager.start_periodic_save(simulator)

    app.state.simulator = simulator
    app.state.state_manager = state_manager
    app.state.arena = StrategyArena(simulator)
    logger.info("Simulator + Arena démarrés")
    return simulator, state_manager


async def _init_executors(
    app: FastAPI, config, db: Database, engine: DataEngine,
    simulator: Simulator, state_manager: StateManager,
    notifier: Notifier,
) -> tuple[Any, AdaptiveSelector | None]:
    """Initialise le Multi-Executor et le Selector."""
    from backend.execution.executor_manager import ExecutorManager

    executor_mgr = ExecutorManager()
    selector: AdaptiveSelector | None = None

    if not config.secrets.live_trading:
        return executor_mgr, selector

    arena = app.state.arena
    selector = AdaptiveSelector(arena, config, db=db)
    live_strategies = _get_live_eligible_strategies(config)

    for strat_name in live_strategies:
        risk_mgr = LiveRiskManager(config, notifier=notifier)
        executor = Executor(
            config, risk_mgr, notifier,
            selector=selector, strategy_name=strat_name,
        )

        strat_state = await state_manager.load_executor_state(
            strategy_name=strat_name,
        )
        if strat_state:
            risk_mgr.restore_state(strat_state.get("risk_manager", {}))
            executor.restore_positions(strat_state)

        await executor.start()

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
        executor.set_db(db)
        executor.set_data_engine(engine)
        executor.set_strategies(strategy_instances, simulator=simulator)
        await sync_live_to_paper(executor, simulator)
        await executor.start_exit_monitor()
        engine.on_candle(executor._on_candle)

    if state_manager is not None:
        state_manager.set_executors(executor_mgr)

    logger.info(
        "Multi-Executor: {} executors démarrés ({})",
        len(executor_mgr.executors),
        list(executor_mgr.executors.keys()),
    )
    return executor_mgr, selector


async def _init_monitoring(
    engine: DataEngine, simulator: Simulator, notifier: Notifier,
    executor_mgr, telegram: TelegramClient | None, db: Database, config,
) -> tuple[Watchdog | None, Heartbeat | None, Any]:
    """Initialise Watchdog + Heartbeat + WeeklyReporter."""
    watchdog = Watchdog(
        data_engine=engine, simulator=simulator, notifier=notifier,
        executor_mgr=executor_mgr if executor_mgr.executors else None,
    )
    await watchdog.start()

    heartbeat: Heartbeat | None = None
    weekly_reporter = None
    if telegram:
        heartbeat = Heartbeat(
            telegram, simulator,
            interval_seconds=config.secrets.heartbeat_interval,
        )
        await heartbeat.start()

        from backend.alerts.weekly_reporter import WeeklyReporter

        weekly_reporter = WeeklyReporter(telegram, db, config)
        await weekly_reporter.start()

    return watchdog, heartbeat, weekly_reporter


async def _init_regime_monitor(
    telegram: TelegramClient | None, db: Database,
) -> Any:
    """Initialise le RegimeMonitor."""
    from backend.regime.regime_monitor import RegimeMonitor

    regime_monitor = RegimeMonitor(telegram, db)
    await regime_monitor.start()
    return regime_monitor


# ─── Lifespan principal ──────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Démarre tous les composants au lancement, les arrête proprement.

    Chaque étape est protégée par try-except. Si un composant non-critique
    échoue, les suivants tentent quand même de démarrer. Le statut de chaque
    composant est exposé via /health (app.state.startup_components).
    """
    config = get_config()
    setup_logging(level=config.secrets.log_level)

    # Suivi d'état pour /health
    components: dict[str, str] = {}
    app.state.startup_components = components

    # Variables locales (toujours initialisées pour le shutdown)
    db: Database | None = None
    telegram: TelegramClient | None = None
    notifier = Notifier(None)
    engine: DataEngine | None = None
    simulator: Simulator | None = None
    state_manager: StateManager | None = None
    executor_mgr = None
    selector: AdaptiveSelector | None = None
    watchdog: Watchdog | None = None
    heartbeat: Heartbeat | None = None
    weekly_reporter = None
    regime_monitor = None

    # ── 1. Database (CRITIQUE) ────────────────────────────────────────
    try:
        db = await _init_database(app)
        components["database"] = "ok"
    except Exception as exc:
        logger.critical("DATABASE INIT FAILED : {} — mode dégradé", exc)
        components["database"] = f"error: {exc}"
        # Créer des stubs pour que le reste ne crash pas
        app.state.db = None
        app.state.start_time = datetime.now(tz=timezone.utc)

    # ── 1b. JobManager ────────────────────────────────────────────────
    if db:
        try:
            await _init_job_manager(app, db)
            components["job_manager"] = "ok"
        except Exception as exc:
            logger.error("JobManager init failed : {}", exc)
            components["job_manager"] = f"error: {exc}"
            app.state.job_manager = None
    else:
        app.state.job_manager = None

    # ── 2. Telegram + Notifier ────────────────────────────────────────
    try:
        telegram, notifier = _init_notifier(app, config, db)
        components["notifier"] = "ok"
    except Exception as exc:
        logger.error("Notifier init failed : {}", exc)
        components["notifier"] = f"error: {exc}"
        notifier = Notifier(None)
        app.state.notifier = notifier

    # ── 3. DataEngine ─────────────────────────────────────────────────
    if db:
        try:
            engine = await _init_data_engine(app, config, db, notifier)
            components["data_engine"] = "ok" if engine else "disabled"
        except Exception as exc:
            logger.error("DataEngine init failed : {}", exc)
            components["data_engine"] = f"error: {exc}"
            app.state.engine = None
            app.state.config = config
    else:
        app.state.engine = None
        app.state.config = config
        components["data_engine"] = "skipped"

    # ── 3b. CandleUpdater ─────────────────────────────────────────────
    if db:
        try:
            await _init_candle_updater(app, config, db)
            components["candle_updater"] = "ok"
        except Exception as exc:
            logger.error("CandleUpdater init failed : {}", exc)
            components["candle_updater"] = f"error: {exc}"
            app.state.candle_updater = None
    else:
        app.state.candle_updater = None

    # ── 4. Simulator + Arena ──────────────────────────────────────────
    if engine is not None:
        try:
            simulator, state_manager = await _init_simulator(
                app, config, db, engine, notifier,
            )
            components["simulator"] = "ok"
        except Exception as exc:
            logger.error("Simulator init failed : {}", exc)
            components["simulator"] = f"error: {exc}"
            app.state.simulator = None
            app.state.state_manager = None
            app.state.arena = None
    else:
        app.state.simulator = None
        app.state.state_manager = None
        app.state.arena = None
        components["simulator"] = "skipped"

    # ── 4b. Multi-Executor ────────────────────────────────────────────
    from backend.execution.executor_manager import ExecutorManager

    executor_mgr = ExecutorManager()

    if engine and simulator and state_manager:
        try:
            executor_mgr, selector = await _init_executors(
                app, config, db, engine, simulator, state_manager, notifier,
            )
            components["executor"] = "ok" if executor_mgr.executors else "no_strategies"
        except Exception as exc:
            logger.error("Executor init failed : {}", exc)
            components["executor"] = f"error: {exc}"
            executor_mgr = ExecutorManager()
    elif config.secrets.live_trading:
        logger.warning("LIVE_TRADING=true mais DataEngine/Simulator absents — executor non créé")
        components["executor"] = "skipped"
    else:
        components["executor"] = "disabled"

    app.state.executor = executor_mgr if executor_mgr.executors else None
    app.state.executor_mgr = executor_mgr
    app.state.risk_mgr = None
    app.state.selector = selector

    # ── 5. Watchdog + Heartbeat + WeeklyReporter ──────────────────────
    if engine is not None and simulator is not None:
        try:
            watchdog, heartbeat, weekly_reporter = await _init_monitoring(
                engine, simulator, notifier, executor_mgr, telegram, db, config,
            )
            components["watchdog"] = "ok"
            components["heartbeat"] = "ok" if heartbeat else "disabled"
        except Exception as exc:
            logger.error("Monitoring init failed : {}", exc)
            components["watchdog"] = f"error: {exc}"
            components["heartbeat"] = f"error: {exc}"
    else:
        components["watchdog"] = "skipped"
        components["heartbeat"] = "skipped"

    app.state.watchdog = watchdog

    # ── 6. Regime Monitor ─────────────────────────────────────────────
    if db:
        try:
            regime_monitor = await _init_regime_monitor(telegram, db)
            components["regime_monitor"] = "ok"
        except Exception as exc:
            logger.error("RegimeMonitor init failed : {}", exc)
            components["regime_monitor"] = f"error: {exc}"
    else:
        components["regime_monitor"] = "skipped"
    app.state.regime_monitor = regime_monitor

    # ── Notification startup ──────────────────────────────────────────
    if simulator:
        try:
            strategies = [r.name for r in simulator.runners]
            await notifier.notify_startup(strategies)
        except Exception as exc:
            logger.error("Notification startup failed : {}", exc)

    # Log résumé
    failed = [k for k, v in components.items() if v.startswith("error")]
    if failed:
        logger.warning("Startup partiel — composants en erreur : {}", failed)
    else:
        logger.info("Startup complet — tous les composants OK")

    yield

    # ── Shutdown (ordre inverse, chaque étape protégée) ───────────────
    if simulator:
        try:
            await notifier.notify_shutdown()
        except Exception as exc:
            logger.error("Notification shutdown failed : {}", exc)

    if regime_monitor:
        await _safe_stop("RegimeMonitor", regime_monitor.stop())
    if weekly_reporter:
        await _safe_stop("WeeklyReporter", weekly_reporter.stop())
    if heartbeat:
        await _safe_stop("Heartbeat", heartbeat.stop())
    if watchdog:
        await _safe_stop("Watchdog", watchdog.stop())

    # Executor + Selector : sauvegarder état + stop AVANT simulator
    if selector:
        await _safe_stop("AdaptiveSelector", selector.stop())
    if executor_mgr and executor_mgr.executors and state_manager:
        for _name, _ex in executor_mgr.executors.items():
            _rm = executor_mgr.risk_managers.get(_name)
            if _rm:
                await _safe_stop(
                    f"SaveState[{_name}]",
                    state_manager.save_executor_state(_ex, _rm, strategy_name=_name),
                )
        await _safe_stop("ExecutorManager", executor_mgr.stop_all())
        logger.info("Multi-Executor: tous les executors arrêtés et états sauvegardés")

    if state_manager and simulator:
        await _safe_stop(
            "SaveRunnerState",
            state_manager.save_runner_state(
                simulator.runners,
                global_kill_switch=simulator._global_kill_switch,
                kill_switch_reason=simulator._kill_switch_reason,
            ),
        )
        await _safe_stop("StateManager", state_manager.stop())
    if simulator:
        await _safe_stop("Simulator", simulator.stop())
    if engine:
        await _safe_stop("DataEngine", engine.stop())

    if getattr(app.state, "candle_updater", None):
        await _safe_stop("CandleUpdater", app.state.candle_updater.stop())
    if getattr(app.state, "job_manager", None):
        await _safe_stop("JobManager", app.state.job_manager.stop())
        logger.info("JobManager arrêté")

    if db:
        await _safe_stop("Database", db.close())
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
app.include_router(regime_router)
app.include_router(alerts_router)
