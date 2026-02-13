"""Simulator (paper trading live) pour Scalp Radar.

Exécute les stratégies sur données live en capital virtuel isolé.
Réutilise PositionManager (fees, slippage, TP/SL) et IncrementalIndicatorEngine.

Sprint 11 : GridStrategyRunner pour les stratégies grid/DCA (envelope_dca).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from loguru import logger

from backend.core.config import AppConfig
from backend.core.data_engine import DataEngine
from backend.core.grid_position_manager import GridPositionManager
from backend.core.incremental_indicators import IncrementalIndicatorEngine
from backend.core.indicators import detect_market_regime
from backend.core.models import Candle, Direction, MarketRegime
from backend.core.position_manager import (
    PositionManager,
    PositionManagerConfig,
    TradeResult,
)
from backend.optimization import is_grid_strategy
from backend.strategies.base import (
    EXTRA_FUNDING_RATE,
    EXTRA_OI_CHANGE_PCT,
    EXTRA_OPEN_INTEREST,
    BaseStrategy,
    OpenPosition,
    StrategyContext,
)
from backend.strategies.base_grid import BaseGridStrategy, GridLevel, GridPosition
from backend.strategies.factory import get_enabled_strategies

if TYPE_CHECKING:
    from backend.core.database import Database


def _safe_round(val: Any, decimals: int = 1) -> float | None:
    """Arrondi safe — retourne None si NaN ou None."""
    if val is None:
        return None
    try:
        import math
        if math.isnan(val):
            return None
        return round(val, decimals)
    except (TypeError, ValueError):
        return None


@dataclass
class RunnerStats:
    """Stats d'un LiveStrategyRunner."""

    capital: float
    initial_capital: float
    net_pnl: float = 0.0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    is_active: bool = True


class LiveStrategyRunner:
    """Exécute une stratégie sur données live (paper trading).

    Un runner par stratégie. Capital virtuel isolé.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        config: AppConfig,
        indicator_engine: IncrementalIndicatorEngine,
        position_manager: PositionManager,
        data_engine: DataEngine,
    ) -> None:
        self._strategy = strategy
        self._config = config
        self._indicator_engine = indicator_engine
        self._pm = position_manager
        self._data_engine = data_engine

        self._initial_capital = 10_000.0
        self._capital = self._initial_capital
        self._position: OpenPosition | None = None
        self._position_symbol: str | None = None
        self._trades: list[tuple[str, TradeResult]] = []
        self._current_regime = MarketRegime.RANGING
        self._previous_regime = MarketRegime.RANGING
        self._kill_switch_triggered = False
        self._stats = RunnerStats(
            capital=self._capital,
            initial_capital=self._initial_capital,
        )

        # Sprint 5a : queue d'événements pour l'Executor (drainée par le Simulator)
        self._pending_events: list[Any] = []

    @property
    def name(self) -> str:
        return self._strategy.name

    @property
    def is_kill_switch_triggered(self) -> bool:
        return self._kill_switch_triggered

    async def on_candle(self, symbol: str, timeframe: str, candle: Candle) -> None:
        """Traitement d'une nouvelle candle."""
        if self._kill_switch_triggered:
            return

        # 1. Les indicateurs sont déjà à jour (mis à jour par le Simulator avant l'appel)

        # 2. Récupérer les indicateurs
        indicators = self._indicator_engine.get_indicators(symbol)
        if not indicators:
            return

        # Vérifier que le timeframe principal de la stratégie a des données
        main_tf = None
        for tf in self._strategy.min_candles:
            if tf in indicators:
                main_tf = tf
                break
        if main_tf is None:
            return

        main_ind = indicators.get(main_tf, {})
        if not main_ind:
            return

        # 3. Build extra_data (funding, OI)
        extra_data: dict[str, Any] = {}
        funding = self._data_engine.get_funding_rate(symbol)
        if funding is not None:
            extra_data[EXTRA_FUNDING_RATE] = funding

        oi_snapshots = self._data_engine.get_open_interest(symbol)
        if oi_snapshots:
            extra_data[EXTRA_OPEN_INTEREST] = oi_snapshots
            extra_data[EXTRA_OI_CHANGE_PCT] = oi_snapshots[-1].change_pct

        # Build StrategyContext
        ctx = StrategyContext(
            symbol=symbol,
            timestamp=candle.timestamp,
            candles={},
            indicators=indicators,
            current_position=self._position,
            capital=self._capital,
            config=self._config,
            extra_data=extra_data,
        )

        # Détecter le régime
        self._previous_regime = self._current_regime
        regime_data = main_ind
        self._current_regime = detect_market_regime(
            regime_data.get("adx", float("nan")),
            regime_data.get("di_plus", float("nan")),
            regime_data.get("di_minus", float("nan")),
            regime_data.get("atr", float("nan")),
            regime_data.get("atr_sma", float("nan")),
        )

        # 4. Si position ouverte
        if self._position is not None:
            # Quick fix : couper si régime change RANGING → TRENDING
            if self._previous_regime in (MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY):
                if self._current_regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
                    trade = self._pm.close_position(
                        self._position, candle.close, candle.timestamp,
                        "regime_change", self._current_regime
                    )
                    self._record_trade(trade, symbol)
                    self._position = None
                    self._position_symbol = None
                    return

            # Check TP/SL/signal_exit
            exit_result = self._pm.check_position_exit(
                candle, self._position, self._strategy, ctx, self._current_regime
            )
            if exit_result is not None:
                self._record_trade(exit_result, symbol)
                self._position = None
                self._position_symbol = None
                return

        # 5. Si pas de position : évaluer l'entrée
        if self._position is None and not self._kill_switch_triggered:
            signal = self._strategy.evaluate(ctx)
            if signal is not None:
                self._position = self._pm.open_position(
                    signal, candle.timestamp, self._capital
                )
                if self._position is not None:
                    self._position_symbol = symbol
                    self._capital -= self._position.entry_fee
                    logger.info(
                        "[{}] {} {} @ {:.2f} (score={:.2f})",
                        self.name,
                        signal.direction.value,
                        symbol,
                        signal.entry_price,
                        signal.score,
                    )
                    # Sprint 5a : notifier l'Executor
                    self._emit_open_event(symbol, signal)

    def _record_trade(self, trade: TradeResult, symbol: str = "") -> None:
        """Enregistre un trade et vérifie le kill switch."""
        self._capital += trade.net_pnl
        self._trades.append((symbol, trade))
        self._stats.total_trades += 1
        self._stats.net_pnl = self._capital - self._initial_capital

        if trade.net_pnl > 0:
            self._stats.wins += 1
        else:
            self._stats.losses += 1

        self._stats.capital = self._capital

        logger.info(
            "[{}] Trade clos : {} {:.2f} → {:.2f}, net={:+.2f} ({})",
            self.name,
            trade.direction.value,
            trade.entry_price,
            trade.exit_price,
            trade.net_pnl,
            trade.exit_reason,
        )

        # Sprint 5a : notifier l'Executor
        if symbol:
            self._emit_close_event(symbol, trade)

        # Kill switch
        session_loss_pct = abs(min(0, self._stats.net_pnl)) / self._initial_capital * 100
        max_session = self._config.risk.kill_switch.max_session_loss_percent
        max_daily = self._config.risk.kill_switch.max_daily_loss_percent

        if session_loss_pct >= max_session:
            self._kill_switch_triggered = True
            self._stats.is_active = False
            logger.warning(
                "[{}] KILL SWITCH : perte session {:.1f}% >= {:.1f}%",
                self.name, session_loss_pct, max_session,
            )

    # ─── Sprint 5a : émission d'événements pour l'Executor ───────────

    def _emit_open_event(self, symbol: str, signal: Any) -> None:
        """Crée un TradeEvent OPEN et l'ajoute à la queue."""
        from backend.execution.executor import TradeEvent, TradeEventType

        self._pending_events.append(TradeEvent(
            event_type=TradeEventType.OPEN,
            strategy_name=self.name,
            symbol=symbol,
            direction=signal.direction.value,
            entry_price=signal.entry_price,
            quantity=self._position.quantity if self._position else 0,
            tp_price=signal.tp_price,
            sl_price=signal.sl_price,
            score=signal.score,
            timestamp=self._position.entry_time if self._position else datetime.now(tz=timezone.utc),
            market_regime=self._current_regime.value,
        ))

    def _emit_close_event(self, symbol: str, trade: TradeResult) -> None:
        """Crée un TradeEvent CLOSE et l'ajoute à la queue."""
        from backend.execution.executor import TradeEvent, TradeEventType

        self._pending_events.append(TradeEvent(
            event_type=TradeEventType.CLOSE,
            strategy_name=self.name,
            symbol=symbol,
            direction=trade.direction.value,
            entry_price=trade.entry_price,
            quantity=trade.quantity,
            tp_price=0,
            sl_price=0,
            score=0,
            timestamp=trade.exit_time,
            market_regime=trade.market_regime.value,
            exit_reason=trade.exit_reason,
            exit_price=trade.exit_price,
        ))

    def restore_state(self, state: dict) -> None:
        """Restaure l'état du runner depuis un snapshot sauvegardé."""
        self._capital = state.get("capital", self._initial_capital)
        self._kill_switch_triggered = state.get("kill_switch", False)

        self._stats.capital = self._capital
        self._stats.net_pnl = state.get("net_pnl", 0.0)
        self._stats.total_trades = state.get("total_trades", 0)
        self._stats.wins = state.get("wins", 0)
        self._stats.losses = state.get("losses", 0)
        self._stats.is_active = state.get("is_active", True)

        # Restaurer la position ouverte si présente
        pos_data = state.get("position")
        if pos_data is not None:
            from datetime import datetime

            self._position = OpenPosition(
                direction=Direction(pos_data["direction"]),
                entry_price=pos_data["entry_price"],
                quantity=pos_data["quantity"],
                entry_time=datetime.fromisoformat(pos_data["entry_time"]),
                tp_price=pos_data["tp_price"],
                sl_price=pos_data["sl_price"],
                entry_fee=pos_data["entry_fee"],
            )
            # Backward compat : ancien format sans position_symbol
            self._position_symbol = state.get("position_symbol", "UNKNOWN")

        logger.info(
            "[{}] État restauré : capital={:.2f}, trades={}, kill_switch={}",
            self.name,
            self._capital,
            self._stats.total_trades,
            self._kill_switch_triggered,
        )

    @property
    def strategy(self) -> BaseStrategy:
        return self._strategy

    @property
    def current_regime(self) -> MarketRegime:
        return self._current_regime

    def build_context(self, symbol: str) -> StrategyContext | None:
        """Construit un StrategyContext pour un symbol (pour le dashboard)."""
        if not self._indicator_engine:
            return None

        indicators = self._indicator_engine.get_indicators(symbol)
        if not indicators:
            return None

        extra_data: dict[str, Any] = {}
        funding = self._data_engine.get_funding_rate(symbol)
        if funding is not None:
            extra_data[EXTRA_FUNDING_RATE] = funding

        oi_snapshots = self._data_engine.get_open_interest(symbol)
        if oi_snapshots:
            extra_data[EXTRA_OPEN_INTEREST] = oi_snapshots
            extra_data[EXTRA_OI_CHANGE_PCT] = oi_snapshots[-1].change_pct

        return StrategyContext(
            symbol=symbol,
            timestamp=datetime.now(tz=timezone.utc),
            candles={},
            indicators=indicators,
            current_position=self._position,
            capital=self._capital,
            config=self._config,
            extra_data=extra_data,
        )

    def get_status(self) -> dict:
        return {
            "name": self.name,
            "capital": self._capital,
            "net_pnl": self._stats.net_pnl,
            "total_trades": self._stats.total_trades,
            "wins": self._stats.wins,
            "losses": self._stats.losses,
            "win_rate": self._stats.wins / self._stats.total_trades * 100 if self._stats.total_trades > 0 else 0.0,
            "is_active": self._stats.is_active,
            "kill_switch": self._kill_switch_triggered,
            "has_position": self._position is not None,
        }

    def get_trades(self) -> list[tuple[str, TradeResult]]:
        return list(self._trades)

    def get_stats(self) -> RunnerStats:
        return self._stats


class GridStrategyRunner:
    """Exécute une stratégie grid/DCA sur données live (paper trading).

    Différences avec LiveStrategyRunner :
    - Gère N positions simultanées via GridPositionManager
    - Utilise compute_grid() au lieu de evaluate()
    - TP/SL global (pas par position)
    - Un "trade" = cycle complet (toutes positions ouvertes → toutes fermées)
    """

    def __init__(
        self,
        strategy: BaseGridStrategy,
        config: AppConfig,
        indicator_engine: IncrementalIndicatorEngine,
        grid_position_manager: GridPositionManager,
        data_engine: DataEngine,
    ) -> None:
        self._strategy = strategy
        self._config = config
        self._indicator_engine = indicator_engine
        self._gpm = grid_position_manager
        self._data_engine = data_engine

        self._initial_capital = 10_000.0
        self._capital = self._initial_capital
        self._positions: list[GridPosition] = []
        self._grid_symbol: str | None = None  # Symbole actif de la grille
        self._trades: list[tuple[str, TradeResult]] = []
        self._current_regime = MarketRegime.RANGING
        self._kill_switch_triggered = False
        self._stats = RunnerStats(
            capital=self._capital,
            initial_capital=self._initial_capital,
        )

        # Sprint 5a : queue d'événements pour l'Executor (drainée par le Simulator)
        self._pending_events: list[Any] = []

        # COMPAT: duck typing pour Simulator/StateManager qui accèdent à ces attributs.
        # Voir self._positions pour le vrai état grid.
        self._position = None
        self._position_symbol = None

        # Buffer de closes pour calcul SMA interne
        self._strategy_tf = getattr(strategy._config, "timeframe", "1h")
        self._ma_period = getattr(strategy._config, "ma_period", 7)
        self._close_buffer: dict[str, deque] = {}

    @property
    def name(self) -> str:
        return self._strategy.name

    @property
    def is_kill_switch_triggered(self) -> bool:
        return self._kill_switch_triggered

    @property
    def strategy(self) -> BaseStrategy:
        return self._strategy

    @property
    def current_regime(self) -> MarketRegime:
        return self._current_regime

    async def _warmup_from_db(self, db: Database, symbol: str) -> None:
        """Pré-charge les bougies historiques depuis la DB pour le warm-up SMA."""
        needed = max(self._ma_period + 20, 50)
        candles = await db.get_recent_candles(symbol, self._strategy_tf, needed)
        if not candles:
            logger.info(
                "[{}] Warm-up: 0 bougies {} en DB pour {}",
                self.name, self._strategy_tf, symbol,
            )
            return

        if symbol not in self._close_buffer:
            self._close_buffer[symbol] = deque(
                maxlen=max(self._ma_period + 20, 50)
            )

        for candle in candles:
            self._close_buffer[symbol].append(candle.close)
            self._indicator_engine.update(symbol, self._strategy_tf, candle)

        logger.info(
            "[{}] Warm-up: {} bougies {} chargées pour {}",
            self.name, len(candles), self._strategy_tf, symbol,
        )

    async def on_candle(
        self, symbol: str, timeframe: str, candle: Candle
    ) -> None:
        """Traitement d'une nouvelle candle — logique grid."""
        if self._kill_switch_triggered:
            return

        # Filtre strict : seul le timeframe de la stratégie est traité
        if timeframe != self._strategy_tf:
            return

        # Maintenir le buffer de closes
        if symbol not in self._close_buffer:
            self._close_buffer[symbol] = deque(
                maxlen=max(self._ma_period + 20, 50)
            )
        self._close_buffer[symbol].append(candle.close)

        # Calculer SMA
        closes = list(self._close_buffer[symbol])
        if len(closes) < self._ma_period:
            return

        sma_val = float(np.mean(closes[-self._ma_period:]))

        # Récupérer les indicateurs de l'engine et merger SMA
        indicators = self._indicator_engine.get_indicators(symbol)
        if not indicators:
            indicators = {}
        indicators.setdefault(self._strategy_tf, {}).update({
            "sma": sma_val,
            "close": candle.close,
        })

        # Détecter le régime (si ADX/ATR disponibles)
        main_ind = indicators.get(self._strategy_tf, {})
        self._current_regime = detect_market_regime(
            main_ind.get("adx", float("nan")),
            main_ind.get("di_plus", float("nan")),
            main_ind.get("di_minus", float("nan")),
            main_ind.get("atr", float("nan")),
            main_ind.get("atr_sma", float("nan")),
        )

        # Construire le contexte
        ctx = StrategyContext(
            symbol=symbol,
            timestamp=candle.timestamp,
            candles={},
            indicators=indicators,
            current_position=None,
            capital=self._capital,
            config=self._config,
        )

        # Construire le GridState
        grid_state = self._gpm.compute_grid_state(
            self._positions, candle.close
        )

        # 1. Si positions ouvertes → check TP/SL global
        if self._positions:
            tp_price = self._strategy.get_tp_price(grid_state, main_ind)
            sl_price = self._strategy.get_sl_price(grid_state, main_ind)

            # Check via OHLC heuristic
            exit_reason, exit_price = self._gpm.check_global_tp_sl(
                self._positions, candle, tp_price, sl_price
            )

            # Si pas de TP/SL OHLC, check should_close_all (signal)
            if exit_reason is None:
                close_reason = self._strategy.should_close_all(ctx, grid_state)
                if close_reason:
                    exit_reason = close_reason
                    exit_price = candle.close

            if exit_reason:
                trade = self._gpm.close_all_positions(
                    self._positions, exit_price,
                    candle.timestamp, exit_reason,
                    self._current_regime,
                )
                self._record_trade(trade, symbol)
                self._positions = []
                self._grid_symbol = None
                self._emit_close_event(symbol, trade)
                return

        # 2. Ouvrir de nouveaux niveaux si grille pas pleine
        if len(self._positions) < self._strategy.max_positions:
            levels = self._strategy.compute_grid(ctx, grid_state)

            for level in levels:
                if level.index in {p.level for p in self._positions}:
                    continue

                touched = False
                if level.direction == Direction.LONG:
                    touched = candle.low <= level.entry_price
                else:
                    touched = candle.high >= level.entry_price

                if touched:
                    position = self._gpm.open_grid_position(
                        level, candle.timestamp,
                        self._capital,
                        self._strategy.max_positions,
                    )
                    if position:
                        self._positions.append(position)
                        self._grid_symbol = symbol
                        logger.info(
                            "[{}] GRID {} level {} @ {:.2f} ({}) — {}/{} positions",
                            self.name,
                            level.direction.value,
                            level.index,
                            level.entry_price,
                            symbol,
                            len(self._positions),
                            self._strategy.max_positions,
                        )
                        self._emit_open_event(symbol, level, position)

    def _record_trade(self, trade: TradeResult, symbol: str = "") -> None:
        """Enregistre un trade grid (fermeture de toutes les positions)."""
        self._capital += trade.net_pnl
        self._trades.append((symbol, trade))
        self._stats.total_trades += 1
        self._stats.net_pnl = self._capital - self._initial_capital
        self._stats.capital = self._capital

        if trade.net_pnl > 0:
            self._stats.wins += 1
        else:
            self._stats.losses += 1

        logger.info(
            "[{}] Grid trade clos : {} avg={:.2f} → {:.2f}, net={:+.2f} ({})",
            self.name,
            trade.direction.value,
            trade.entry_price,
            trade.exit_price,
            trade.net_pnl,
            trade.exit_reason,
        )

        # Kill switch
        session_loss_pct = (
            abs(min(0, self._stats.net_pnl)) / self._initial_capital * 100
        )
        max_session = self._config.risk.kill_switch.max_session_loss_percent
        if session_loss_pct >= max_session:
            self._kill_switch_triggered = True
            self._stats.is_active = False
            logger.warning(
                "[{}] KILL SWITCH : perte session {:.1f}% >= {:.1f}%",
                self.name, session_loss_pct, max_session,
            )

    def _emit_open_event(
        self, symbol: str, level: GridLevel, position: GridPosition
    ) -> None:
        """Crée un TradeEvent OPEN pour un niveau de grille."""
        from backend.execution.executor import TradeEvent, TradeEventType

        self._pending_events.append(TradeEvent(
            event_type=TradeEventType.OPEN,
            strategy_name=self.name,
            symbol=symbol,
            direction=position.direction.value,
            entry_price=position.entry_price,
            quantity=position.quantity,
            tp_price=0.0,
            sl_price=0.0,
            score=0.0,
            timestamp=position.entry_time,
            market_regime=self._current_regime.value,
        ))

    def _emit_close_event(self, symbol: str, trade: TradeResult) -> None:
        """Crée un TradeEvent CLOSE pour la fermeture grid globale."""
        from backend.execution.executor import TradeEvent, TradeEventType

        self._pending_events.append(TradeEvent(
            event_type=TradeEventType.CLOSE,
            strategy_name=self.name,
            symbol=symbol,
            direction=trade.direction.value,
            entry_price=trade.entry_price,
            quantity=trade.quantity,
            tp_price=0.0,
            sl_price=0.0,
            score=0.0,
            timestamp=trade.exit_time,
            market_regime=self._current_regime.value,
            exit_reason=trade.exit_reason,
            exit_price=trade.exit_price,
        ))

    def restore_state(self, state: dict) -> None:
        """Restaure l'état du runner depuis un snapshot sauvegardé."""
        self._capital = state.get("capital", self._initial_capital)
        self._kill_switch_triggered = state.get("kill_switch", False)

        self._stats.capital = self._capital
        self._stats.net_pnl = state.get("net_pnl", 0.0)
        self._stats.total_trades = state.get("total_trades", 0)
        self._stats.wins = state.get("wins", 0)
        self._stats.losses = state.get("losses", 0)
        self._stats.is_active = state.get("is_active", True)

        # Restaurer les positions grid ouvertes
        grid_positions = state.get("grid_positions", [])
        self._positions = []
        for gp in grid_positions:
            self._positions.append(GridPosition(
                level=gp["level"],
                direction=Direction(gp["direction"]),
                entry_price=gp["entry_price"],
                quantity=gp["quantity"],
                entry_time=datetime.fromisoformat(gp["entry_time"]),
                entry_fee=gp["entry_fee"],
            ))
        if self._positions:
            self._grid_symbol = state.get("grid_symbol", "UNKNOWN")

        logger.info(
            "[{}] État restauré : capital={:.2f}, trades={}, positions_grid={}, kill_switch={}",
            self.name,
            self._capital,
            self._stats.total_trades,
            len(self._positions),
            self._kill_switch_triggered,
        )

    def build_context(self, symbol: str) -> StrategyContext | None:
        """Construit un StrategyContext pour le dashboard (get_conditions)."""
        if not self._indicator_engine:
            return None

        indicators = self._indicator_engine.get_indicators(symbol)
        if not indicators:
            indicators = {}

        # Merger SMA depuis le buffer interne
        closes = list(self._close_buffer.get(symbol, []))
        if len(closes) >= self._ma_period:
            sma_val = float(np.mean(closes[-self._ma_period:]))
            indicators.setdefault(self._strategy_tf, {}).update({
                "sma": sma_val,
                "close": closes[-1] if closes else 0.0,
            })

        return StrategyContext(
            symbol=symbol,
            timestamp=datetime.now(tz=timezone.utc),
            candles={},
            indicators=indicators,
            current_position=None,
            capital=self._capital,
            config=self._config,
        )

    def get_grid_positions(self) -> list[dict]:
        """Retourne les positions grid ouvertes pour le dashboard."""
        result = []
        for p in self._positions:
            result.append({
                "symbol": self._grid_symbol or "UNKNOWN",
                "strategy": self.name,
                "direction": p.direction.value,
                "entry_price": p.entry_price,
                "quantity": p.quantity,
                "entry_time": p.entry_time.isoformat(),
                "level": p.level,
                "type": "grid",
            })
        return result

    def get_status(self) -> dict:
        """Même interface que LiveStrategyRunner.get_status() + champs grid."""
        return {
            "name": self.name,
            "capital": self._capital,
            "net_pnl": self._stats.net_pnl,
            "total_trades": self._stats.total_trades,
            "wins": self._stats.wins,
            "losses": self._stats.losses,
            "win_rate": (
                self._stats.wins / self._stats.total_trades * 100
                if self._stats.total_trades > 0 else 0.0
            ),
            "is_active": self._stats.is_active,
            "kill_switch": self._kill_switch_triggered,
            "has_position": len(self._positions) > 0,
            "open_positions": len(self._positions),
            "max_positions": self._strategy.max_positions,
            "avg_entry_price": (
                sum(p.entry_price * p.quantity for p in self._positions)
                / sum(p.quantity for p in self._positions)
                if self._positions else 0.0
            ),
        }

    def get_trades(self) -> list[tuple[str, TradeResult]]:
        return list(self._trades)

    def get_stats(self) -> RunnerStats:
        return self._stats


class Simulator:
    """Orchestrateur du paper trading.

    Crée un LiveStrategyRunner ou GridStrategyRunner par stratégie enabled.
    Se câble sur le DataEngine via on_candle.
    """

    def __init__(
        self,
        data_engine: DataEngine,
        config: AppConfig,
        db: Database | None = None,
    ) -> None:
        self._data_engine = data_engine
        self._config = config
        self._db = db
        self._runners: list[LiveStrategyRunner | GridStrategyRunner] = []
        self._indicator_engine: IncrementalIndicatorEngine | None = None
        self._running = False
        self._trade_event_callback: Callable | None = None

        # Sprint 6 : caches pour le dashboard
        self._conditions_cache: dict | None = None
        self._conditions_cache_time: float = 0.0
        self._equity_cache: list[dict] | None = None
        self._trade_count_at_cache: int = 0

    def set_trade_event_callback(self, callback: Callable) -> None:
        """Enregistre le callback de l'Executor pour recevoir les TradeEvent."""
        self._trade_event_callback = callback

    async def start(self, saved_state: dict | None = None) -> None:
        """Démarre le simulateur : crée les runners et s'enregistre sur le DataEngine.

        Si saved_state est fourni, restaure l'état des runners AVANT
        d'enregistrer le callback on_candle (évite la race condition).
        """
        strategies = get_enabled_strategies(self._config)
        if not strategies:
            logger.warning("Simulator: aucune stratégie activée")
            return

        # Créer l'engine d'indicateurs incrémentaux
        self._indicator_engine = IncrementalIndicatorEngine(strategies)

        # PositionManager partagé (mêmes params fees/slippage)
        pm_config = PositionManagerConfig(
            leverage=self._config.risk.position.default_leverage,
            maker_fee=self._config.risk.fees.maker_percent / 100,
            taker_fee=self._config.risk.fees.taker_percent / 100,
            slippage_pct=self._config.risk.slippage.default_estimate_percent / 100,
            high_vol_slippage_mult=self._config.risk.slippage.high_volatility_multiplier,
            max_risk_per_trade=self._config.risk.position.max_risk_per_trade_percent / 100,
        )
        pm = PositionManager(pm_config)

        # Créer un runner par stratégie (grid ou mono-position)
        for strategy in strategies:
            if is_grid_strategy(strategy.name):
                gpm_config = PositionManagerConfig(
                    leverage=getattr(strategy._config, "leverage", 15),
                    maker_fee=self._config.risk.fees.maker_percent / 100,
                    taker_fee=self._config.risk.fees.taker_percent / 100,
                    slippage_pct=self._config.risk.slippage.default_estimate_percent / 100,
                    high_vol_slippage_mult=self._config.risk.slippage.high_volatility_multiplier,
                    max_risk_per_trade=self._config.risk.position.max_risk_per_trade_percent / 100,
                )
                gpm = GridPositionManager(gpm_config)
                runner: LiveStrategyRunner | GridStrategyRunner = GridStrategyRunner(
                    strategy=strategy,
                    config=self._config,
                    indicator_engine=self._indicator_engine,
                    grid_position_manager=gpm,
                    data_engine=self._data_engine,
                )
            else:
                runner = LiveStrategyRunner(
                    strategy=strategy,
                    config=self._config,
                    indicator_engine=self._indicator_engine,
                    position_manager=pm,
                    data_engine=self._data_engine,
                )
            self._runners.append(runner)
            logger.info(
                "Simulator: stratégie '{}' ajoutée ({})",
                strategy.name,
                "grid" if is_grid_strategy(strategy.name) else "mono",
            )

        # Restaurer l'état AVANT d'enregistrer le callback on_candle
        if saved_state is not None:
            runners_state = saved_state.get("runners", {})
            for runner in self._runners:
                if runner.name in runners_state:
                    runner.restore_state(runners_state[runner.name])

        # Warm-up grid runners depuis la DB
        if self._db is not None:
            symbols = self._data_engine.get_all_symbols()
            for runner in self._runners:
                if isinstance(runner, GridStrategyRunner):
                    for symbol in symbols:
                        await runner._warmup_from_db(self._db, symbol)

        # Câblage DataEngine → Simulator (APRÈS restauration + warm-up)
        self._data_engine.on_candle(self._dispatch_candle)
        self._running = True

        logger.info(
            "Simulator: démarré avec {} stratégies{}",
            len(self._runners),
            " (état restauré)" if saved_state else "",
        )

    async def _dispatch_candle(
        self, symbol: str, timeframe: str, candle: Candle
    ) -> None:
        """Fan-out vers tous les runners."""
        if not self._running or not self._indicator_engine:
            return

        # Mettre à jour les indicateurs (une seule fois pour tous les runners)
        self._indicator_engine.update(symbol, timeframe, candle)

        # Invalider le cache conditions à chaque bougie (indicateurs changent)
        self._conditions_cache = None

        # Dispatcher à chaque runner
        for runner in self._runners:
            try:
                await runner.on_candle(symbol, timeframe, candle)
            except Exception as e:
                logger.error(
                    "Simulator: erreur runner '{}': {}",
                    runner.name, e,
                )

            # Sprint 5a : drain pending_events vers l'Executor (swap atomique)
            if self._trade_event_callback and runner._pending_events:
                events, runner._pending_events = runner._pending_events, []
                for event in events:
                    try:
                        await self._trade_event_callback(event)
                    except Exception as e:
                        logger.error(
                            "Simulator: erreur callback trade event: {}", e,
                        )

    async def stop(self) -> None:
        """Arrête le simulateur."""
        self._running = False
        logger.info("Simulator: arrêté")

    def get_all_status(self) -> dict[str, dict]:
        """Retourne le status de tous les runners."""
        return {runner.name: runner.get_status() for runner in self._runners}

    def get_all_trades(self) -> list[dict]:
        """Retourne tous les trades de tous les runners."""
        all_trades = []
        for runner in self._runners:
            for symbol, trade in runner.get_trades():
                all_trades.append({
                    "strategy": runner.name,
                    "symbol": symbol,
                    "direction": trade.direction.value,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "quantity": trade.quantity,
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": trade.exit_time.isoformat(),
                    "gross_pnl": trade.gross_pnl,
                    "fee_cost": trade.fee_cost,
                    "slippage_cost": trade.slippage_cost,
                    "net_pnl": trade.net_pnl,
                    "exit_reason": trade.exit_reason,
                    "market_regime": trade.market_regime.value,
                })
        # Trier par exit_time
        all_trades.sort(key=lambda t: t["exit_time"], reverse=True)
        return all_trades

    def get_open_positions(self) -> list[dict]:
        """Retourne les positions ouvertes de tous les runners."""
        positions = []
        for runner in self._runners:
            # Positions grid (GridStrategyRunner)
            if hasattr(runner, "get_grid_positions"):
                positions.extend(runner.get_grid_positions())
            # Position mono-position (LiveStrategyRunner)
            elif runner._position is not None and runner._position_symbol:
                pos = runner._position
                positions.append({
                    "symbol": runner._position_symbol,
                    "strategy": runner.name,
                    "direction": pos.direction.value,
                    "entry_price": pos.entry_price,
                    "quantity": pos.quantity,
                    "tp_price": pos.tp_price,
                    "sl_price": pos.sl_price,
                    "entry_time": pos.entry_time.isoformat(),
                })
        return positions

    def is_kill_switch_triggered(self) -> bool:
        """Vérifie si au moins un runner a déclenché le kill switch."""
        return any(r.is_kill_switch_triggered for r in self._runners)

    def get_conditions(self) -> dict:
        """Indicateurs courants par asset + conditions par stratégie.

        Cache invalidé à chaque nouvelle bougie (dans _dispatch_candle).
        """
        if self._conditions_cache is not None:
            return self._conditions_cache

        symbols = self._data_engine.get_all_symbols()
        assets: dict[str, dict] = {}

        for symbol in symbols:
            asset_data: dict[str, Any] = {
                "price": None,
                "change_pct": None,
                "regime": None,
                "indicators": {},
                "strategies": {},
                "position": None,
                "sparkline": [],
            }

            # Prix courant depuis le buffer 1m du DataEngine
            data = self._data_engine.get_data(symbol)
            if data.candles.get("1m"):
                last_candle = data.candles["1m"][-1]
                asset_data["price"] = last_candle.close
                # Sparkline : 60 derniers close prices
                asset_data["sparkline"] = [
                    c.close for c in data.candles["1m"][-60:]
                ]
                if len(data.candles["1m"]) >= 2:
                    prev_close = data.candles["1m"][-2].close
                    if prev_close > 0:
                        asset_data["change_pct"] = round(
                            (last_candle.close - prev_close) / prev_close * 100, 2
                        )

            # Indicateurs et conditions par runner/stratégie
            for runner in self._runners:
                ctx = runner.build_context(symbol)
                if ctx is None:
                    continue

                # Régime (du runner, mis à jour à chaque candle)
                if asset_data["regime"] is None:
                    asset_data["regime"] = runner.current_regime.value

                # Indicateurs bruts depuis le context
                main_tf = list(runner.strategy.min_candles.keys())[0]
                main_ind = ctx.indicators.get(main_tf, {})
                if main_ind and not asset_data["indicators"]:
                    asset_data["indicators"] = {
                        "rsi_14": _safe_round(main_ind.get("rsi"), 1),
                        "vwap_distance_pct": None,
                        "adx": _safe_round(main_ind.get("adx"), 1),
                        "atr_pct": None,
                    }
                    # VWAP distance
                    close = main_ind.get("close")
                    vwap = main_ind.get("vwap")
                    if close and vwap and vwap > 0:
                        asset_data["indicators"]["vwap_distance_pct"] = round(
                            (close - vwap) / vwap * 100, 2
                        )
                    # ATR %
                    atr_val = main_ind.get("atr")
                    if close and atr_val and close > 0:
                        asset_data["indicators"]["atr_pct"] = round(
                            atr_val / close * 100, 2
                        )

                # Conditions de la stratégie
                try:
                    conditions = runner.strategy.get_current_conditions(ctx)
                except Exception:
                    conditions = []

                # Dernier signal (trade le plus récent de ce runner pour ce symbol)
                last_signal = None
                for trade_dict in self.get_all_trades():
                    if trade_dict["strategy"] == runner.name:
                        last_signal = {
                            "score": trade_dict.get("score"),
                            "direction": trade_dict["direction"],
                            "timestamp": trade_dict["entry_time"],
                        }
                        break

                asset_data["strategies"][runner.name] = {
                    "last_signal": last_signal,
                    "conditions": conditions,
                }

                # Position ouverte sur cet asset (vérifier le symbol)
                if runner._position is not None and runner._position_symbol == symbol:
                    asset_data["position"] = {
                        "direction": runner._position.direction.value,
                        "entry_price": runner._position.entry_price,
                        "tp_price": runner._position.tp_price,
                        "sl_price": runner._position.sl_price,
                        "strategy": runner.name,
                        "entry_time": runner._position.entry_time.isoformat(),
                    }

            assets[symbol] = asset_data

        result = {
            "assets": assets,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._conditions_cache = result
        return result

    def get_signal_matrix(self) -> dict:
        """Matrice simplifiée pour la Heatmap : dernier score par (strategy, asset)."""
        symbols = self._data_engine.get_all_symbols()
        matrix: dict[str, dict[str, float | None]] = {}

        for symbol in symbols:
            matrix[symbol] = {}
            for runner in self._runners:
                # Chercher le dernier trade de ce runner pour ce symbol
                last_score = None
                for _sym, trade in runner.get_trades():
                    if hasattr(trade, "entry_price"):
                        # TradeResult n'a pas de score — on utilise les conditions
                        break
                matrix[symbol][runner.name] = last_score

        return {"matrix": matrix}

    def get_equity_curve(self, since: str | None = None) -> dict:
        """Courbe d'equity depuis les trades. Cache invalidé quand un trade est enregistré."""
        total_trades = sum(len(r.get_trades()) for r in self._runners)

        # Cache valide ?
        if self._equity_cache is not None and self._trade_count_at_cache == total_trades:
            equity = self._equity_cache
        else:
            # Recalculer
            all_trades: list[TradeResult] = []
            for runner in self._runners:
                for _sym, trade in runner.get_trades():
                    all_trades.append(trade)
            all_trades.sort(key=lambda t: t.exit_time)

            capital = 10_000.0  # Capital initial par convention
            equity = []
            for trade in all_trades:
                capital += trade.net_pnl
                equity.append({
                    "timestamp": trade.exit_time.isoformat(),
                    "capital": round(capital, 2),
                    "trade_pnl": round(trade.net_pnl, 2),
                })

            self._equity_cache = equity
            self._trade_count_at_cache = total_trades

        # Filtre since
        if since:
            equity = [p for p in equity if p["timestamp"] > since]

        # Capital courant (somme de tous les runners)
        current_capital = sum(r._capital for r in self._runners) if self._runners else 10_000.0
        # Si un seul runner, capital initial = 10k. Si multiple, somme.
        initial_capital = sum(r._initial_capital for r in self._runners) if self._runners else 10_000.0

        return {
            "equity": equity,
            "current_capital": round(current_capital, 2),
            "initial_capital": round(initial_capital, 2),
        }

    @property
    def runners(self) -> list[LiveStrategyRunner | GridStrategyRunner]:
        return self._runners
