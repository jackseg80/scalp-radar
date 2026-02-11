"""Simulator (paper trading live) pour Scalp Radar.

Exécute les stratégies sur données live en capital virtuel isolé.
Réutilise PositionManager (fees, slippage, TP/SL) et IncrementalIndicatorEngine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from loguru import logger

from backend.core.config import AppConfig
from backend.core.data_engine import DataEngine
from backend.core.incremental_indicators import IncrementalIndicatorEngine
from backend.core.indicators import detect_market_regime
from backend.core.models import Candle, Direction, MarketRegime
from backend.core.position_manager import (
    PositionManager,
    PositionManagerConfig,
    TradeResult,
)
from backend.strategies.base import (
    EXTRA_FUNDING_RATE,
    EXTRA_OI_CHANGE_PCT,
    EXTRA_OPEN_INTEREST,
    BaseStrategy,
    OpenPosition,
    StrategyContext,
)
from backend.strategies.factory import get_enabled_strategies


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
        self._trades: list[TradeResult] = []
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
                    return

            # Check TP/SL/signal_exit
            exit_result = self._pm.check_position_exit(
                candle, self._position, self._strategy, ctx, self._current_regime
            )
            if exit_result is not None:
                self._record_trade(exit_result, symbol)
                self._position = None
                return

        # 5. Si pas de position : évaluer l'entrée
        if self._position is None and not self._kill_switch_triggered:
            signal = self._strategy.evaluate(ctx)
            if signal is not None:
                self._position = self._pm.open_position(
                    signal, candle.timestamp, self._capital
                )
                if self._position is not None:
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
        self._trades.append(trade)
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

    def get_trades(self) -> list[TradeResult]:
        return list(self._trades)

    def get_stats(self) -> RunnerStats:
        return self._stats


class Simulator:
    """Orchestrateur du paper trading.

    Crée un LiveStrategyRunner par stratégie enabled.
    Se câble sur le DataEngine via on_candle.
    """

    def __init__(
        self,
        data_engine: DataEngine,
        config: AppConfig,
    ) -> None:
        self._data_engine = data_engine
        self._config = config
        self._runners: list[LiveStrategyRunner] = []
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

        # Créer un runner par stratégie
        for strategy in strategies:
            runner = LiveStrategyRunner(
                strategy=strategy,
                config=self._config,
                indicator_engine=self._indicator_engine,
                position_manager=pm,
                data_engine=self._data_engine,
            )
            self._runners.append(runner)
            logger.info("Simulator: stratégie '{}' ajoutée", strategy.name)

        # Restaurer l'état AVANT d'enregistrer le callback on_candle
        if saved_state is not None:
            runners_state = saved_state.get("runners", {})
            for runner in self._runners:
                if runner.name in runners_state:
                    runner.restore_state(runners_state[runner.name])

        # Câblage DataEngine → Simulator (APRÈS restauration)
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
            for trade in runner.get_trades():
                all_trades.append({
                    "strategy": runner.name,
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
            }

            # Prix courant depuis le buffer 1m du DataEngine
            data = self._data_engine.get_data(symbol)
            if data.candles.get("1m"):
                last_candle = data.candles["1m"][-1]
                asset_data["price"] = last_candle.close
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

                # Position ouverte sur cet asset
                if runner._position is not None:
                    asset_data["position"] = {
                        "direction": runner._position.direction.value,
                        "entry_price": runner._position.entry_price,
                        "strategy": runner.name,
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
                for trade in runner.get_trades():
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
            all_trades = []
            for runner in self._runners:
                for trade in runner.get_trades():
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
    def runners(self) -> list[LiveStrategyRunner]:
        return self._runners
