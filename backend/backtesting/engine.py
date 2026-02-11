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
from backend.core.position_manager import (
    PositionManager,
    PositionManagerConfig,
    TradeResult,
)
from backend.strategies.base import BaseStrategy, OpenPosition, StrategyContext

# Re-export pour compatibilité (tests et scripts importent TradeResult depuis engine)
__all__ = ["BacktestConfig", "BacktestEngine", "BacktestResult", "TradeResult"]


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
        self._pm = PositionManager(PositionManagerConfig(
            leverage=config.leverage,
            maker_fee=config.maker_fee,
            taker_fee=config.taker_fee,
            slippage_pct=config.slippage_pct,
            high_vol_slippage_mult=config.high_vol_slippage_mult,
            max_risk_per_trade=config.max_risk_per_trade,
        ))

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
                exit_result = self._pm.check_position_exit(
                    candle, position, self._strategy, ctx, current_regime
                )
                if exit_result is not None:
                    capital += exit_result.net_pnl
                    trades.append(exit_result)
                    position = None

            # 3. Si pas de position : évaluer l'entrée
            if position is None and main_ind:
                signal = self._strategy.evaluate(ctx)
                if signal is not None:
                    position = self._pm.open_position(signal, candle.timestamp, capital)
                    if position is not None:
                        capital -= position.entry_fee

            # 4. Equity curve (point par bougie)
            current_equity = capital
            if position is not None:
                current_equity += self._pm.unrealized_pnl(position, candle.close)
            equity_curve.append(current_equity)
            equity_timestamps.append(candle.timestamp)

        # 5. Clôture forcée des positions ouvertes
        if position is not None and main_candles:
            last_candle = main_candles[-1]
            trade = self._pm.force_close(position, last_candle, current_regime)
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
