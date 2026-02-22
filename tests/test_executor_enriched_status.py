"""Tests Sprint 39 : métriques live enrichies dans Executor.get_status().

Vérifie :
- _enrich_grid_position() retourne les champs enrichis (prix, P&L, TP/SL, durée)
- Graceful degradation sans DataEngine ou sans Simulator
- P&L par niveau correct pour LONG et SHORT
- TP NaN (grid_boltrend) → tp_price=None
- get_status() contient executor_grid_state au format paper
- WS merge _merge_live_grids_into_state() fonctionne
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock
import math

import pytest

from backend.execution.executor import (
    Executor,
    GridLivePosition,
    GridLiveState,
)
from backend.execution.risk_manager import LiveRiskManager


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_config():
    config = MagicMock()
    config.secrets.live_trading = True
    config.secrets.bitget_api_key = "test_key"
    config.secrets.bitget_secret = "test_secret"
    config.secrets.bitget_passphrase = "test_pass"
    config.risk.position.max_concurrent_positions = 3
    config.risk.position.default_leverage = 6
    config.risk.margin.min_free_margin_percent = 20
    config.risk.margin.mode = "cross"
    config.risk.fees.taker_percent = 0.06
    config.risk.fees.maker_percent = 0.02
    config.risk.kill_switch.max_session_loss_percent = 5.0
    config.risk.kill_switch.grid_max_session_loss_percent = 25.0
    config.assets = []
    config.correlation_groups = {}
    # Strategy configs
    config.strategies.grid_atr.num_levels = 3
    config.strategies.grid_atr.leverage = 6
    config.strategies.grid_atr.live_eligible = True
    config.strategies.grid_atr.timeframe = "1h"
    config.strategies.grid_boltrend.num_levels = 3
    config.strategies.grid_boltrend.leverage = 8
    config.strategies.grid_boltrend.live_eligible = True
    config.strategies.grid_boltrend.timeframe = "1h"
    return config


def _make_executor(config=None) -> Executor:
    if config is None:
        config = _make_config()
    risk_manager = LiveRiskManager(config)
    risk_manager.set_initial_capital(10_000.0)
    notifier = AsyncMock()
    executor = Executor(config, risk_manager, notifier)
    executor._running = True
    executor._connected = True
    # Pas d'exchange (pas nécessaire pour get_status)
    return executor


def _make_grid_state(
    symbol="BTC/USDT:USDT",
    direction="LONG",
    strategy_name="grid_atr",
    entry_prices=None,
    quantity=0.01,
    leverage=6,
    sl_price=0.0,
    opened_at=None,
):
    if entry_prices is None:
        entry_prices = [50000.0]
    if opened_at is None:
        opened_at = datetime.now(tz=timezone.utc) - timedelta(hours=12)
    positions = [
        GridLivePosition(
            level=i,
            entry_price=ep,
            quantity=quantity,
            entry_order_id=f"entry_{i}",
            entry_time=opened_at + timedelta(hours=i),
        )
        for i, ep in enumerate(entry_prices)
    ]
    return GridLiveState(
        symbol=symbol,
        direction=direction,
        strategy_name=strategy_name,
        leverage=leverage,
        positions=positions,
        sl_order_id="sl_123",
        sl_price=sl_price,
        opened_at=opened_at,
    )


def _mock_data_engine(symbol="BTC/USDT", close_price=52000.0):
    """Crée un mock DataEngine avec un buffer 1m contenant un prix."""
    engine = MagicMock()
    candle = MagicMock()
    candle.close = close_price
    engine._buffers = {symbol: {"1m": [candle]}}
    return engine


def _mock_simulator(strategy_name="grid_atr", symbol="BTC/USDT", sma=51000.0):
    """Crée un mock Simulator qui retourne un runner context avec SMA."""
    sim = MagicMock()
    ctx = MagicMock()
    ctx.indicators = {"1h": {"sma": sma, "close": 52000.0}}
    sim.get_runner_context.return_value = ctx
    return sim


def _mock_strategy(tp_price=51000.0, sl_price=45000.0):
    """Crée un mock BaseGridStrategy avec get_tp_price/get_sl_price."""
    strategy = MagicMock()
    strategy.get_tp_price.return_value = tp_price
    strategy.get_sl_price.return_value = sl_price
    strategy._config.timeframe = "1h"
    return strategy


# ── Tests _enrich_grid_position ──────────────────────────────────────────


def test_enrich_grid_position_basic():
    """Avec DataEngine + Simulator, tous les champs enrichis sont remplis."""
    executor = _make_executor()
    gs = _make_grid_state(
        entry_prices=[50000.0, 49000.0],
        sl_price=45000.0,
    )
    executor._grid_states["BTC/USDT:USDT"] = gs
    executor._data_engine = _mock_data_engine(close_price=52000.0)
    executor._simulator = _mock_simulator(sma=51000.0)
    executor._strategies = {"grid_atr": _mock_strategy(tp_price=51000.0, sl_price=45000.0)}

    info = executor._enrich_grid_position("BTC/USDT:USDT", gs)

    assert info["current_price"] == 52000.0
    assert info["type"] == "grid"
    assert info["levels"] == 2
    assert info["levels_max"] == 3
    assert info["leverage"] == 6
    # P&L : LONG, avg_entry ~49500, current 52000 → positive
    assert info["unrealized_pnl"] > 0
    assert info["unrealized_pnl_pct"] > 0
    # TP/SL
    assert info["tp_price"] == 51000.0
    assert info["sl_price"] == 45000.0
    # Distances
    assert info["tp_distance_pct"] is not None
    assert info["sl_distance_pct"] is not None
    # Marge
    assert info["margin_used"] > 0
    # Durée
    assert info["duration_hours"] > 0
    # Per-level P&L
    assert len(info["positions"]) == 2
    assert info["positions"][0]["pnl_usd"] is not None
    assert info["positions"][0]["pnl_pct"] is not None
    assert info["positions"][0]["direction"] == "LONG"


def test_enrich_grid_position_no_data_engine():
    """Sans DataEngine, current_price=None, P&L=0."""
    executor = _make_executor()
    gs = _make_grid_state(entry_prices=[50000.0])
    executor._grid_states["BTC/USDT:USDT"] = gs
    executor._data_engine = None

    info = executor._enrich_grid_position("BTC/USDT:USDT", gs)

    assert info["current_price"] is None
    assert info["unrealized_pnl"] == 0.0
    assert info["positions"][0]["pnl_usd"] is None


def test_enrich_grid_position_no_simulator():
    """Sans Simulator, tp_price fallback 0.0, sl_price fallback exchange."""
    executor = _make_executor()
    gs = _make_grid_state(entry_prices=[50000.0], sl_price=45000.0)
    executor._grid_states["BTC/USDT:USDT"] = gs
    executor._data_engine = _mock_data_engine(close_price=52000.0)
    executor._simulator = None
    executor._strategies = {}

    info = executor._enrich_grid_position("BTC/USDT:USDT", gs)

    assert info["tp_price"] == 0.0  # Pas de stratégie → fallback
    assert info["sl_price"] == 45000.0  # Fallback exchange
    assert info["current_price"] == 52000.0  # DataEngine fonctionne
    assert info["unrealized_pnl"] > 0  # P&L calculé quand même


def test_enrich_grid_position_per_level_pnl():
    """3 niveaux LONG à prix différents, P&L correct par niveau."""
    executor = _make_executor()
    gs = _make_grid_state(
        entry_prices=[50000.0, 48000.0, 46000.0],
        quantity=0.01,
    )
    executor._grid_states["BTC/USDT:USDT"] = gs
    executor._data_engine = _mock_data_engine(close_price=52000.0)

    info = executor._enrich_grid_position("BTC/USDT:USDT", gs)

    # Niveau 0 : entry=50000, current=52000 → pnl = 2000 * 0.01 = 20$
    assert info["positions"][0]["pnl_usd"] == 20.0
    # Niveau 1 : entry=48000, current=52000 → pnl = 4000 * 0.01 = 40$
    assert info["positions"][1]["pnl_usd"] == 40.0
    # Niveau 2 : entry=46000, current=52000 → pnl = 6000 * 0.01 = 60$
    assert info["positions"][2]["pnl_usd"] == 60.0


def test_enrich_grid_position_short():
    """P&L correct pour positions SHORT."""
    executor = _make_executor()
    gs = _make_grid_state(
        entry_prices=[50000.0],
        direction="SHORT",
        quantity=0.01,
    )
    executor._grid_states["BTC/USDT:USDT"] = gs
    executor._data_engine = _mock_data_engine(close_price=48000.0)

    info = executor._enrich_grid_position("BTC/USDT:USDT", gs)

    # SHORT : entry=50000, current=48000 → pnl = (50000-48000)*0.01 = 20$
    assert info["unrealized_pnl"] == 20.0
    assert info["positions"][0]["pnl_usd"] == 20.0
    assert info["positions"][0]["pnl_pct"] > 0


def test_enrich_grid_position_boltrend_nan_tp():
    """grid_boltrend get_tp_price() retourne NaN → tp_price=None."""
    executor = _make_executor()
    gs = _make_grid_state(
        entry_prices=[50000.0],
        strategy_name="grid_boltrend",
        leverage=8,
    )
    executor._grid_states["BTC/USDT:USDT"] = gs
    executor._data_engine = _mock_data_engine(close_price=52000.0)
    executor._simulator = _mock_simulator(sma=51000.0)
    # BolTrend retourne NaN pour TP
    executor._strategies = {
        "grid_boltrend": _mock_strategy(tp_price=float("nan"), sl_price=45000.0),
    }

    info = executor._enrich_grid_position("BTC/USDT:USDT", gs)

    assert info["tp_price"] is None or info["tp_price"] == 0.0
    assert info["tp_distance_pct"] is None
    assert info["sl_price"] == 45000.0  # SL normal
    assert info["sl_distance_pct"] is not None


def test_enrich_duration_hours():
    """duration_hours calculée correctement depuis opened_at."""
    executor = _make_executor()
    opened_48h_ago = datetime.now(tz=timezone.utc) - timedelta(hours=48)
    gs = _make_grid_state(
        entry_prices=[50000.0],
        opened_at=opened_48h_ago,
    )
    executor._grid_states["BTC/USDT:USDT"] = gs

    info = executor._enrich_grid_position("BTC/USDT:USDT", gs)

    # Environ 48h (tolérance 1h pour le temps d'exécution du test)
    assert 47.0 <= info["duration_hours"] <= 49.0


# ── Tests get_status() ───────────────────────────────────────────────────


def test_get_status_includes_enriched_fields():
    """get_status() retourne les champs enrichis pour les grids."""
    executor = _make_executor()
    gs = _make_grid_state(entry_prices=[50000.0, 49000.0])
    executor._grid_states["BTC/USDT:USDT"] = gs
    executor._data_engine = _mock_data_engine(close_price=52000.0)
    executor._simulator = _mock_simulator(sma=51000.0)
    executor._strategies = {"grid_atr": _mock_strategy(tp_price=51000.0, sl_price=45000.0)}

    status = executor.get_status()

    assert status["enabled"] is True
    assert len(status["positions"]) == 1
    pos = status["positions"][0]
    assert pos["current_price"] == 52000.0
    assert pos["unrealized_pnl"] > 0
    assert pos["tp_price"] == 51000.0
    assert pos["margin_used"] > 0
    assert pos["duration_hours"] > 0
    assert len(pos["positions"]) == 2


def test_get_status_executor_grid_state():
    """get_status() contient executor_grid_state au format paper."""
    executor = _make_executor()
    gs = _make_grid_state(entry_prices=[50000.0])
    executor._grid_states["BTC/USDT:USDT"] = gs
    executor._data_engine = _mock_data_engine(close_price=52000.0)

    status = executor.get_status()

    assert "executor_grid_state" in status
    egs = status["executor_grid_state"]
    assert "grid_positions" in egs
    assert "summary" in egs
    # Clé format paper : strategy:spot_symbol
    assert "grid_atr:BTC/USDT" in egs["grid_positions"]
    assert egs["summary"]["total_assets"] == 1
    assert egs["summary"]["total_positions"] == 1
    assert egs["summary"]["total_margin_used"] > 0


# ── Tests WS merge ──────────────────────────────────────────────────────


def test_ws_merge_live_grids():
    """_merge_live_grids_into_state() ajoute les grids live avec source='live'."""
    from backend.api.websocket_routes import _merge_live_grids_into_state

    grid_state = {
        "grid_positions": {
            "grid_atr:CRV/USDT": {
                "symbol": "CRV/USDT",
                "strategy": "grid_atr",
                "direction": "LONG",
                "levels_open": 1,
                "margin_used": 100,
                "unrealized_pnl": 5.0,
            },
        },
        "summary": {
            "total_positions": 1,
            "total_assets": 1,
            "total_margin_used": 100,
            "total_unrealized_pnl": 5.0,
            "capital_available": 9000,
        },
    }

    exec_status = {
        "executor_grid_state": {
            "grid_positions": {
                "grid_atr:BTC/USDT": {
                    "symbol": "BTC/USDT:USDT",
                    "strategy_name": "grid_atr",
                    "direction": "LONG",
                    "levels": 2,
                    "levels_max": 3,
                    "entry_price": 50000.0,
                    "current_price": 52000.0,
                    "unrealized_pnl": 200.0,
                    "unrealized_pnl_pct": 5.0,
                    "tp_price": 51000.0,
                    "sl_price": 45000.0,
                    "margin_used": 1000.0,
                    "leverage": 6,
                    "positions": [],
                },
            },
        },
    }

    _merge_live_grids_into_state(grid_state, exec_status)

    gp = grid_state["grid_positions"]
    # CRV paper supprimé (stratégie grid_atr = live)
    assert "grid_atr:CRV/USDT" not in gp
    # BTC live ajouté
    assert "grid_atr:BTC/USDT" in gp
    btc = gp["grid_atr:BTC/USDT"]
    assert btc["source"] == "live"
    assert btc["symbol"] == "BTC/USDT"  # Spot format (sans :USDT)
    assert btc["strategy"] == "grid_atr"
    assert btc["levels_open"] == 2
    assert btc["unrealized_pnl"] == 200.0
    # Summary recalculé
    assert grid_state["summary"]["total_assets"] == 1
    assert grid_state["summary"]["total_unrealized_pnl"] == 200.0


def test_ws_live_overrides_paper():
    """Quand paper et live ont le même strategy:symbol, live gagne."""
    from backend.api.websocket_routes import _merge_live_grids_into_state

    grid_state = {
        "grid_positions": {
            "grid_atr:BTC/USDT": {
                "symbol": "BTC/USDT",
                "strategy": "grid_atr",
                "direction": "LONG",
                "levels_open": 1,
                "margin_used": 500,
                "unrealized_pnl": -10.0,
            },
        },
        "summary": {
            "total_positions": 1,
            "total_assets": 1,
            "total_margin_used": 500,
            "total_unrealized_pnl": -10.0,
            "capital_available": 9000,
        },
    }

    exec_status = {
        "executor_grid_state": {
            "grid_positions": {
                "grid_atr:BTC/USDT": {
                    "symbol": "BTC/USDT:USDT",
                    "strategy_name": "grid_atr",
                    "direction": "LONG",
                    "levels": 2,
                    "levels_max": 3,
                    "entry_price": 50000.0,
                    "current_price": 52000.0,
                    "unrealized_pnl": 200.0,
                    "margin_used": 1000.0,
                    "leverage": 6,
                    "positions": [],
                },
            },
        },
    }

    _merge_live_grids_into_state(grid_state, exec_status)

    gp = grid_state["grid_positions"]
    assert "grid_atr:BTC/USDT" in gp
    btc = gp["grid_atr:BTC/USDT"]
    # Live gagne : unrealized_pnl=200 (pas -10 du paper)
    assert btc["unrealized_pnl"] == 200.0
    assert btc["source"] == "live"
    assert btc["levels_open"] == 2


def test_ws_merge_paper_only_strategies_kept():
    """Les positions paper de stratégies non-live restent avec source='paper'."""
    from backend.api.websocket_routes import _merge_live_grids_into_state

    grid_state = {
        "grid_positions": {
            "grid_boltrend:ETH/USDT": {
                "symbol": "ETH/USDT",
                "strategy": "grid_boltrend",
                "direction": "LONG",
                "levels_open": 1,
                "margin_used": 200,
                "unrealized_pnl": 10.0,
            },
        },
        "summary": {
            "total_positions": 1,
            "total_assets": 1,
            "total_margin_used": 200,
            "total_unrealized_pnl": 10.0,
            "capital_available": 9000,
        },
    }

    # Live ne concerne que grid_atr, pas grid_boltrend
    exec_status = {
        "executor_grid_state": {
            "grid_positions": {
                "grid_atr:BTC/USDT": {
                    "symbol": "BTC/USDT:USDT",
                    "strategy_name": "grid_atr",
                    "direction": "LONG",
                    "levels": 1,
                    "levels_max": 3,
                    "unrealized_pnl": 100.0,
                    "margin_used": 500.0,
                    "leverage": 6,
                    "positions": [],
                },
            },
        },
    }

    _merge_live_grids_into_state(grid_state, exec_status)

    gp = grid_state["grid_positions"]
    # grid_boltrend paper conservé
    assert "grid_boltrend:ETH/USDT" in gp
    eth = gp["grid_boltrend:ETH/USDT"]
    assert eth["source"] == "paper"
    assert eth["unrealized_pnl"] == 10.0
    # grid_atr live ajouté
    assert "grid_atr:BTC/USDT" in gp
    assert gp["grid_atr:BTC/USDT"]["source"] == "live"
    # Summary combiné
    assert grid_state["summary"]["total_assets"] == 2
