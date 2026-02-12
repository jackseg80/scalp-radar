"""Gestion des positions : sizing, fees, slippage, TP/SL.

Réutilisé par BacktestEngine ET Simulator.
Extrait depuis BacktestEngine (Sprint 2) pour éviter la duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from backend.core.models import Candle, Direction, MarketRegime
from backend.strategies.base import BaseStrategy, OpenPosition, StrategyContext


@dataclass
class PositionManagerConfig:
    """Configuration du PositionManager."""

    leverage: int = 15
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0006  # 0.06%
    slippage_pct: float = 0.0005  # 0.05%
    high_vol_slippage_mult: float = 2.0
    max_risk_per_trade: float = 0.02  # 2%


@dataclass
class TradeResult:
    """Résultat d'un trade clôturé."""

    direction: Direction
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    gross_pnl: float
    fee_cost: float
    slippage_cost: float
    net_pnl: float
    exit_reason: str  # "tp", "sl", "signal_exit", "end_of_data", "regime_change"
    market_regime: MarketRegime


class PositionManager:
    """Gestion des positions : sizing, fees, slippage, TP/SL.

    Utilisé par BacktestEngine ET Simulator.
    """

    def __init__(self, config: PositionManagerConfig) -> None:
        self._config = config

    def open_position(
        self,
        signal: Any,
        timestamp: datetime,
        capital: float,
    ) -> OpenPosition | None:
        """Ouvre une position avec position sizing basé sur le coût SL réel."""
        entry_price = signal.entry_price
        sl_price = signal.sl_price

        if entry_price <= 0 or sl_price <= 0:
            return None

        # Distance SL en %
        sl_distance_pct = abs(entry_price - sl_price) / entry_price

        # Coût SL réel = distance + taker_fee + slippage
        sl_real_cost = sl_distance_pct + self._config.taker_fee + self._config.slippage_pct

        if sl_real_cost <= 0:
            return None

        # Position sizing
        risk_amount = capital * self._config.max_risk_per_trade
        notional = risk_amount / sl_real_cost
        quantity = notional / entry_price

        if quantity <= 0:
            return None

        # Fee d'entrée (taker = market order)
        entry_fee = quantity * entry_price * self._config.taker_fee

        if entry_fee >= capital:
            return None

        return OpenPosition(
            direction=signal.direction,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=timestamp,
            tp_price=signal.tp_price,
            sl_price=sl_price,
            entry_fee=entry_fee,
        )

    def check_position_exit(
        self,
        candle: Candle,
        position: OpenPosition,
        strategy: BaseStrategy,
        ctx: StrategyContext,
        regime: MarketRegime,
    ) -> TradeResult | None:
        """Vérifie TP/SL (heuristique OHLC) puis check_exit."""
        tp_hit = False
        sl_hit = False

        if position.direction == Direction.LONG:
            tp_hit = candle.high >= position.tp_price
            sl_hit = candle.low <= position.sl_price
        else:
            tp_hit = candle.low <= position.tp_price
            sl_hit = candle.high >= position.sl_price

        if tp_hit and sl_hit:
            exit_reason = self._ohlc_heuristic(candle, position)
        elif tp_hit:
            exit_reason = "tp"
        elif sl_hit:
            exit_reason = "sl"
        else:
            # Ni TP ni SL → check_exit via la stratégie
            exit_signal = strategy.check_exit(ctx, position)
            if exit_signal:
                return self.close_position(
                    position, candle.close, candle.timestamp, "signal_exit", regime
                )
            return None

        if exit_reason == "tp":
            exit_price = position.tp_price
        else:
            exit_price = position.sl_price

        return self.close_position(
            position, exit_price, candle.timestamp, exit_reason, regime
        )

    def close_position(
        self,
        position: OpenPosition,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        regime: MarketRegime,
    ) -> TradeResult:
        """Clôture une position et calcule le P&L."""
        slippage_cost = 0.0
        actual_exit_price = exit_price

        if exit_reason in ("sl", "signal_exit", "end_of_data", "regime_change"):
            slippage_rate = self._config.slippage_pct
            if regime == MarketRegime.HIGH_VOLATILITY:
                slippage_rate *= self._config.high_vol_slippage_mult

            slippage_cost = position.quantity * exit_price * slippage_rate

            if position.direction == Direction.LONG:
                actual_exit_price = exit_price * (1 - slippage_rate)
            else:
                actual_exit_price = exit_price * (1 + slippage_rate)

        # Gross P&L
        if position.direction == Direction.LONG:
            gross_pnl = (actual_exit_price - position.entry_price) * position.quantity
        else:
            gross_pnl = (position.entry_price - actual_exit_price) * position.quantity

        # Fee de sortie
        if exit_reason == "tp":
            exit_fee = position.quantity * exit_price * self._config.maker_fee
        else:
            exit_fee = position.quantity * exit_price * self._config.taker_fee

        fee_cost = position.entry_fee + exit_fee
        net_pnl = gross_pnl - fee_cost - slippage_cost

        return TradeResult(
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=actual_exit_price,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=exit_time,
            gross_pnl=gross_pnl,
            fee_cost=fee_cost,
            slippage_cost=slippage_cost,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
            market_regime=regime,
        )

    def force_close(
        self,
        position: OpenPosition,
        candle: Candle,
        regime: MarketRegime,
    ) -> TradeResult:
        """Clôture forcée (market order au close)."""
        return self.close_position(
            position, candle.close, candle.timestamp, "end_of_data", regime
        )

    def unrealized_pnl(self, position: OpenPosition, current_price: float) -> float:
        """P&L non réalisé (pour l'equity curve)."""
        if position is None:
            return 0.0
        if position.direction == Direction.LONG:
            return (current_price - position.entry_price) * position.quantity
        return (position.entry_price - current_price) * position.quantity

    def _ohlc_heuristic(self, candle: Candle, position: OpenPosition) -> str:
        """Heuristique OHLC pour déterminer TP ou SL quand les deux sont touchés.

        Bougie verte (close > open) → mouvement inféré : open → high → low → close
        Bougie rouge (close < open) → mouvement inféré : open → low → high → close
        Doji (close == open) → SL priorisé (conservateur)
        """
        if candle.close > candle.open:
            if position.direction == Direction.LONG:
                return "tp"
            return "sl"
        elif candle.close < candle.open:
            if position.direction == Direction.LONG:
                return "sl"
            return "tp"
        else:
            return "sl"
