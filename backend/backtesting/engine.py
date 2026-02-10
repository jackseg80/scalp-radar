"""Moteur de backtesting event-driven pour Scalp Radar.

Traite chaque bougie séquentiellement, gère le TP/SL avec heuristique OHLC,
le position sizing avec coût SL réel, et l'alignement multi-timeframe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger

from backend.core.models import Candle, Direction, MarketRegime
from backend.strategies.base import BaseStrategy, OpenPosition, StrategyContext


@dataclass
class BacktestConfig:
    """Configuration d'un backtest."""

    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10_000.0
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
    exit_reason: str  # "tp", "sl", "signal_exit", "end_of_data"
    market_regime: MarketRegime


@dataclass
class BacktestResult:
    """Résultat complet d'un backtest."""

    config: BacktestConfig
    strategy_name: str
    strategy_params: dict[str, Any]
    trades: list[TradeResult]
    equity_curve: list[float]
    equity_timestamps: list[datetime]
    final_capital: float


class BacktestEngine:
    """Moteur de backtesting event-driven.

    Boucle principale :
    1. Pré-calcul des indicateurs (une seule fois)
    2. Pour chaque bougie du TF principal :
       a. Mise à jour des buffers multi-TF
       b. TP/SL check (heuristique OHLC si les deux touchés)
       c. check_exit si ni TP ni SL touchés
       d. evaluate si pas de position
       e. Mise à jour de l'equity curve (point par bougie)
    3. Clôture forcée en fin de données
    """

    def __init__(self, config: BacktestConfig, strategy: BaseStrategy) -> None:
        self._config = config
        self._strategy = strategy

    def run(
        self,
        candles_by_tf: dict[str, list[Candle]],
        main_tf: str = "5m",
    ) -> BacktestResult:
        """Lance le backtest. candles_by_tf = {"5m": [...], "15m": [...]}."""
        # Vérifier les données
        if main_tf not in candles_by_tf or not candles_by_tf[main_tf]:
            raise ValueError(f"Pas de données pour le timeframe principal {main_tf}")

        min_candles = self._strategy.min_candles
        for tf, needed in min_candles.items():
            available = len(candles_by_tf.get(tf, []))
            if available < needed:
                logger.warning(
                    "TF {} : {} bougies disponibles, {} requises",
                    tf, available, needed,
                )

        # 0. Pré-calcul des indicateurs
        logger.info("Pré-calcul des indicateurs...")
        indicators_by_tf = self._strategy.compute_indicators(candles_by_tf)

        # Construire l'index des timestamps 15m pour lookup rapide
        higher_tf_timestamps: dict[str, list[str]] = {}
        for tf, ind_dict in indicators_by_tf.items():
            if tf != main_tf:
                higher_tf_timestamps[tf] = sorted(ind_dict.keys())

        # État du backtest
        capital = self._config.initial_capital
        position: OpenPosition | None = None
        trades: list[TradeResult] = []
        equity_curve: list[float] = []
        equity_timestamps: list[datetime] = []
        current_regime = MarketRegime.RANGING

        main_candles = candles_by_tf[main_tf]
        main_indicators = indicators_by_tf.get(main_tf, {})

        logger.info(
            "Backtest {} : {} bougies {}, capital={}, levier={}",
            self._config.symbol,
            len(main_candles),
            main_tf,
            self._config.initial_capital,
            self._config.leverage,
        )

        # 1. Boucle principale
        for candle in main_candles:
            ts_iso = candle.timestamp.isoformat()

            # Indicateurs pour cette bougie
            ctx_indicators: dict[str, dict[str, Any]] = {}

            # Indicateurs du TF principal
            if ts_iso in main_indicators:
                ctx_indicators[main_tf] = main_indicators[ts_iso]

            # Indicateurs des TF supérieurs (last_available_before)
            for tf, ts_list in higher_tf_timestamps.items():
                last_ts = self._last_available_before(ts_iso, ts_list)
                if last_ts and last_ts in indicators_by_tf[tf]:
                    ctx_indicators[tf] = indicators_by_tf[tf][last_ts]

            # Construire le context
            ctx = StrategyContext(
                symbol=self._config.symbol,
                timestamp=candle.timestamp,
                candles=candles_by_tf,
                indicators=ctx_indicators,
                current_position=position,
                capital=capital,
                config=None,  # type: ignore[arg-type] — pas besoin en backtest
            )

            # Détecter le régime si les indicateurs sont dispo
            main_ind = ctx_indicators.get(main_tf, {})
            if main_ind:
                from backend.core.indicators import detect_market_regime
                current_regime = detect_market_regime(
                    main_ind.get("adx", float("nan")),
                    main_ind.get("di_plus", float("nan")),
                    main_ind.get("di_minus", float("nan")),
                    main_ind.get("atr", float("nan")),
                    main_ind.get("atr_sma", float("nan")),
                )

            # 2. Si position ouverte : vérifier TP/SL puis check_exit
            if position is not None:
                exit_result = self._check_position_exit(
                    candle, position, ctx, current_regime
                )
                if exit_result is not None:
                    capital += exit_result.net_pnl
                    trades.append(exit_result)
                    position = None

            # 3. Si pas de position : évaluer l'entrée
            if position is None and main_ind:
                signal = self._strategy.evaluate(ctx)
                if signal is not None:
                    position = self._open_position(signal, candle.timestamp, capital)
                    if position is not None:
                        capital -= position.entry_fee

            # 4. Equity curve (point par bougie)
            current_equity = capital
            if position is not None:
                current_equity += self._unrealized_pnl(position, candle.close)
            equity_curve.append(current_equity)
            equity_timestamps.append(candle.timestamp)

        # 5. Clôture forcée des positions ouvertes
        if position is not None and main_candles:
            last_candle = main_candles[-1]
            trade = self._force_close(position, last_candle, current_regime)
            capital += trade.net_pnl
            trades.append(trade)
            position = None
            equity_curve[-1] = capital

        logger.info(
            "Backtest terminé : {} trades, capital final={:.2f} ({:+.2f}%)",
            len(trades),
            capital,
            (capital / self._config.initial_capital - 1) * 100,
        )

        strategy_params = {}
        if hasattr(self._strategy, "get_params"):
            strategy_params = self._strategy.get_params()

        return BacktestResult(
            config=self._config,
            strategy_name=self._strategy.name,
            strategy_params=strategy_params,
            trades=trades,
            equity_curve=equity_curve,
            equity_timestamps=equity_timestamps,
            final_capital=capital,
        )

    def _last_available_before(
        self, target_ts: str, sorted_timestamps: list[str]
    ) -> str | None:
        """Retourne le dernier timestamp <= target dans la liste triée."""
        result = None
        for ts in sorted_timestamps:
            if ts <= target_ts:
                result = ts
            else:
                break
        return result

    def _check_position_exit(
        self,
        candle: Candle,
        position: OpenPosition,
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
            # Heuristique OHLC : déterminer lequel est touché en premier
            exit_reason = self._ohlc_heuristic(candle, position)
        elif tp_hit:
            exit_reason = "tp"
        elif sl_hit:
            exit_reason = "sl"
        else:
            # Ni TP ni SL → check_exit
            exit_signal = self._strategy.check_exit(ctx, position)
            if exit_signal:
                return self._close_position(
                    position, candle.close, candle.timestamp, "signal_exit", regime
                )
            return None

        if exit_reason == "tp":
            exit_price = position.tp_price
        else:
            exit_price = position.sl_price

        return self._close_position(
            position, exit_price, candle.timestamp, exit_reason, regime
        )

    def _ohlc_heuristic(self, candle: Candle, position: OpenPosition) -> str:
        """Heuristique OHLC pour déterminer TP ou SL quand les deux sont touchés.

        Bougie verte (close > open) → mouvement inféré : open → high → low → close
        Bougie rouge (close < open) → mouvement inféré : open → low → high → close
        Doji (close == open) → SL priorisé (conservateur)
        """
        if candle.close > candle.open:
            # Bougie verte : high d'abord
            if position.direction == Direction.LONG:
                return "tp"  # TP en haut touché d'abord
            return "sl"  # SHORT : SL en haut touché d'abord
        elif candle.close < candle.open:
            # Bougie rouge : low d'abord
            if position.direction == Direction.LONG:
                return "sl"  # SL en bas touché d'abord
            return "tp"  # SHORT : TP en bas touché d'abord
        else:
            return "sl"  # Conservateur

    def _open_position(
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

    def _close_position(
        self,
        position: OpenPosition,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        regime: MarketRegime,
    ) -> TradeResult:
        """Clôture une position et calcule le P&L."""
        # Slippage sur les market orders (SL et signal_exit)
        slippage_cost = 0.0
        actual_exit_price = exit_price

        if exit_reason in ("sl", "signal_exit"):
            slippage_rate = self._config.slippage_pct
            # Haute volatilité → slippage doublé
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

    def _force_close(
        self,
        position: OpenPosition,
        candle: Candle,
        regime: MarketRegime,
    ) -> TradeResult:
        """Clôture forcée en fin de données (market order au close)."""
        return self._close_position(
            position, candle.close, candle.timestamp, "end_of_data", regime
        )

    def _unrealized_pnl(self, position: OpenPosition, current_price: float) -> float:
        """P&L non réalisé (pour l'equity curve)."""
        if position.direction == Direction.LONG:
            return (current_price - position.entry_price) * position.quantity
        return (position.entry_price - current_price) * position.quantity
