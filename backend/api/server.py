"""Serveur FastAPI avec DataEngine intégré via lifespan.

Un seul process gère l'API REST + le DataEngine WebSocket.
Le flag ENABLE_WEBSOCKET permet de désactiver le DataEngine en dev.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.api.arena_routes import router as arena_router
from backend.api.health import router as health_router
from backend.api.signals_routes import router as signals_router
from backend.api.simulator_routes import router as simulator_router
from backend.api.websocket_routes import router as ws_router
from backend.backtesting.arena import StrategyArena
from backend.backtesting.simulator import Simulator
from backend.core.config import get_config
from backend.core.data_engine import DataEngine
from backend.core.database import Database
from backend.core.logging_setup import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Démarre le DataEngine et la DB au lancement, les arrête proprement."""
    config = get_config()
    setup_logging(level=config.secrets.log_level)

    # Database
    db = Database()
    await db.init()
    app.state.db = db
    app.state.start_time = datetime.now(tz=timezone.utc)

    # DataEngine (optionnel via ENABLE_WEBSOCKET)
    engine: DataEngine | None = None
    if config.secrets.enable_websocket:
        engine = DataEngine(config, db)
        await engine.start()
        logger.info("DataEngine démarré via lifespan")
    else:
        logger.info("DataEngine désactivé (ENABLE_WEBSOCKET=false)")

    app.state.engine = engine
    app.state.config = config

    # Simulator + Arena (après DataEngine)
    simulator: Simulator | None = None
    if engine is not None:
        simulator = Simulator(data_engine=engine, config=config)
        await simulator.start()
        app.state.simulator = simulator
        app.state.arena = StrategyArena(simulator)
        logger.info("Simulator + Arena démarrés")
    else:
        app.state.simulator = None
        app.state.arena = None

    yield

    # Shutdown
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
app.include_router(ws_router)
