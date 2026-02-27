"""Gestion multi-position pour stratégies grid/DCA.

Gère N positions ouvertes simultanément avec prix moyen pondéré.
Sizing : allocation fixe par niveau (notional = capital/levels × leverage).
Le SL global protège le capital total, pas le sizing par position.
"""

from __future__ import annotations

from datetime import datetime

from backend.core.models import Candle, Direction, MarketRegime
from backend.core.position_manager import PositionManagerConfig, TradeResult
from backend.strategies.base_grid import GridLevel, GridPosition, GridState


class GridPositionManager:
    """Gestion multi-position pour stratégies grid/DCA.

    Différences avec PositionManager :
    - Gère une LISTE de positions (pas une seule)
    - Calcule le prix moyen pondéré automatiquement
    - Ferme TOUTES les positions en un seul TradeResult agrégé
    - Allocation fixe par niveau (pas risk-based)
    """

    def __init__(self, config: PositionManagerConfig) -> None:
        self._config = config

    def open_grid_position(
        self,
        level: GridLevel,
        timestamp: datetime,
        capital: float,
        total_levels: int,
    ) -> GridPosition | None:
        """Ouvre une position pour un niveau de grille.

        Allocation fixe par niveau (comme le live) :
        notional = capital × (1/total_levels) × leverage
        quantity = notional / entry_price
        """
        if level.entry_price <= 0 or capital <= 0:
            return None

        # Sprint 56: slippage à l'entrée (prix d'exécution défavorable)
        slippage_pct = self._config.slippage_pct
        if level.direction == Direction.LONG:
            actual_entry = level.entry_price * (1 + slippage_pct)
        else:
            actual_entry = level.entry_price * (1 - slippage_pct)

        notional = capital * (1.0 / total_levels) * self._config.leverage
        quantity = notional / actual_entry

        if quantity <= 0:
            return None

        entry_fee = quantity * actual_entry * self._config.taker_fee

        if entry_fee >= capital:
            return None

        return GridPosition(
            level=level.index,
            direction=level.direction,
            entry_price=actual_entry,
            quantity=quantity,
            entry_time=timestamp,
            entry_fee=entry_fee,
        )

    def close_all_positions(
        self,
        positions: list[GridPosition],
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        regime: MarketRegime,
    ) -> TradeResult:
        """Ferme toutes les positions, retourne UN TradeResult agrégé."""
        if not positions:
            return TradeResult(
                direction=Direction.LONG,
                entry_price=0.0,
                exit_price=exit_price,
                quantity=0.0,
                entry_time=exit_time,
                exit_time=exit_time,
                gross_pnl=0.0,
                fee_cost=0.0,
                slippage_cost=0.0,
                net_pnl=0.0,
                exit_reason=exit_reason,
                market_regime=regime,
            )

        direction = positions[0].direction
        total_qty = sum(p.quantity for p in positions)
        if total_qty <= 0:
            return TradeResult(
                direction=direction,
                entry_price=0.0,
                exit_price=exit_price,
                quantity=0.0,
                entry_time=exit_time,
                exit_time=exit_time,
                gross_pnl=0.0,
                fee_cost=0.0,
                slippage_cost=0.0,
                net_pnl=0.0,
                exit_reason=exit_reason,
                market_regime=regime,
            )
        avg_entry = (
            sum(p.entry_price * p.quantity for p in positions) / total_qty
        )
        total_entry_fees = sum(p.entry_fee for p in positions)
        earliest_entry = min(p.entry_time for p in positions)

        # Gross PnL sur prix brut (pas d'actual_exit ajusté)
        if direction == Direction.LONG:
            gross_pnl = (exit_price - avg_entry) * total_qty
        else:
            gross_pnl = (avg_entry - exit_price) * total_qty

        # Slippage : flat cost 1 seule fois (exit seulement, sauf TP)
        slippage_cost = 0.0
        if exit_reason in ("sl_global", "signal_exit", "end_of_data"):
            slippage_rate = self._config.slippage_pct
            if regime == MarketRegime.HIGH_VOLATILITY:
                slippage_rate *= self._config.high_vol_slippage_mult
            slippage_cost = total_qty * exit_price * slippage_rate

        # Fee de sortie
        if exit_reason == "tp_global":
            exit_fee = total_qty * exit_price * self._config.maker_fee
        else:
            exit_fee = total_qty * exit_price * self._config.taker_fee

        fee_cost = total_entry_fees + exit_fee
        net_pnl = gross_pnl - fee_cost - slippage_cost

        return TradeResult(
            direction=direction,
            entry_price=avg_entry,
            exit_price=exit_price,
            quantity=total_qty,
            entry_time=earliest_entry,
            exit_time=exit_time,
            gross_pnl=gross_pnl,
            fee_cost=fee_cost,
            slippage_cost=slippage_cost,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
            market_regime=regime,
        )

    def check_global_tp_sl(
        self,
        positions: list[GridPosition],
        candle: Candle,
        tp_price: float,
        sl_price: float,
    ) -> tuple[str | None, float]:
        """Vérifie TP/SL global avec heuristique OHLC.

        Returns:
            (exit_reason, exit_price) ou (None, 0.0)
        """
        if not positions:
            return None, 0.0

        import math
        if math.isnan(tp_price) and math.isnan(sl_price):
            return None, 0.0

        direction = positions[0].direction

        tp_hit = False
        sl_hit = False

        if not math.isnan(tp_price):
            if direction == Direction.LONG:
                tp_hit = candle.high >= tp_price
            else:
                tp_hit = candle.low <= tp_price

        if not math.isnan(sl_price):
            if direction == Direction.LONG:
                sl_hit = candle.low <= sl_price
            else:
                sl_hit = candle.high >= sl_price

        if tp_hit and sl_hit:
            exit_reason = self._ohlc_heuristic(candle, direction)
        elif tp_hit:
            exit_reason = "tp_global"
        elif sl_hit:
            exit_reason = "sl_global"
        else:
            return None, 0.0

        if exit_reason == "tp_global":
            return exit_reason, tp_price

        # Sprint 56: SL gap slippage — fill mi-chemin entre SL et extrême
        actual_sl = sl_price
        if direction == Direction.LONG:
            gap = max(0.0, sl_price - candle.low)
            actual_sl = sl_price - 0.5 * gap
        else:
            gap = max(0.0, candle.high - sl_price)
            actual_sl = sl_price + 0.5 * gap
        return exit_reason, actual_sl

    def compute_grid_state(
        self,
        positions: list[GridPosition],
        current_price: float,
    ) -> GridState:
        """Calcule l'état agrégé de la grille."""
        if not positions:
            return GridState(
                positions=positions,
                avg_entry_price=0.0,
                total_quantity=0.0,
                total_notional=0.0,
                unrealized_pnl=0.0,
            )

        total_qty = sum(p.quantity for p in positions)
        avg_entry = sum(p.entry_price * p.quantity for p in positions) / total_qty
        total_notional = sum(p.entry_price * p.quantity for p in positions)
        upnl = self.unrealized_pnl(positions, current_price)

        return GridState(
            positions=positions,
            avg_entry_price=avg_entry,
            total_quantity=total_qty,
            total_notional=total_notional,
            unrealized_pnl=upnl,
        )

    def unrealized_pnl(
        self,
        positions: list[GridPosition],
        current_price: float,
    ) -> float:
        """P&L non réalisé total."""
        if not positions:
            return 0.0

        pnl = 0.0
        for p in positions:
            if p.direction == Direction.LONG:
                pnl += (current_price - p.entry_price) * p.quantity
            else:
                pnl += (p.entry_price - current_price) * p.quantity
        return pnl

    @staticmethod
    def _ohlc_heuristic(candle: Candle, direction: Direction) -> str:
        """Heuristique OHLC quand TP et SL touchés sur la même bougie."""
        if candle.close > candle.open:  # Bougie verte
            return "tp_global" if direction == Direction.LONG else "sl_global"
        elif candle.close < candle.open:  # Bougie rouge
            return "sl_global" if direction == Direction.LONG else "tp_global"
        else:  # Doji → SL (conservateur)
            return "sl_global"
