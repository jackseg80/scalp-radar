"""Moteur de backtesting multi-position pour stratégies grid/DCA.

Boucle similaire à BacktestEngine mais gère N positions simultanées
via GridPositionManager. Produit le même BacktestResult.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from backend.backtesting.engine import BacktestConfig, BacktestResult
from backend.core.grid_position_manager import GridPositionManager
from backend.core.models import Candle, Direction, MarketRegime
from backend.core.position_manager import PositionManagerConfig, TradeResult
from backend.strategies.base import StrategyContext
from backend.strategies.base_grid import BaseGridStrategy, GridPosition


class MultiPositionEngine:
    """Moteur de backtesting multi-position.

    Boucle :
    1. Pré-calcul indicateurs (une seule fois)
    2. Pour chaque bougie :
       a. Si positions ouvertes : check TP/SL global + should_close_all
       b. Si grille pas pleine : compute_grid + ouvrir niveaux touchés
       c. Mise à jour equity curve
    3. Clôture forcée en fin de données
    """

    def __init__(self, config: BacktestConfig, strategy: BaseGridStrategy) -> None:
        self._config = config
        self._strategy = strategy
        self._gpm = GridPositionManager(PositionManagerConfig(
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
        main_tf: str = "1h",
        precomputed_indicators: dict[str, dict[str, dict[str, Any]]] | None = None,
    ) -> BacktestResult:
        """Lance le backtest multi-position."""
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
        if precomputed_indicators is not None:
            indicators_by_tf = precomputed_indicators
        else:
            indicators_by_tf = self._strategy.compute_indicators(candles_by_tf)

        # État
        capital = self._config.initial_capital
        positions: list[GridPosition] = []
        trades: list[TradeResult] = []
        equity_curve: list[float] = []
        equity_timestamps: list[datetime] = []
        current_regime = MarketRegime.RANGING

        main_candles = candles_by_tf[main_tf]
        main_indicators = indicators_by_tf.get(main_tf, {})

        from datetime import datetime

        logger.info(
            "Multi-backtest {} : {} bougies {}, capital={}, levier={}",
            self._config.symbol,
            len(main_candles),
            main_tf,
            self._config.initial_capital,
            self._config.leverage,
        )

        # Boucle principale
        for candle in main_candles:
            ts_iso = candle.timestamp.isoformat()

            # Indicateurs
            ctx_indicators: dict[str, dict[str, Any]] = {}
            if ts_iso in main_indicators:
                ctx_indicators[main_tf] = main_indicators[ts_iso]

            main_ind = ctx_indicators.get(main_tf, {})

            # Régime
            if main_ind:
                from backend.core.indicators import detect_market_regime
                current_regime = detect_market_regime(
                    main_ind.get("adx", float("nan")),
                    main_ind.get("di_plus", float("nan")),
                    main_ind.get("di_minus", float("nan")),
                    main_ind.get("atr", float("nan")),
                    main_ind.get("atr_sma", float("nan")),
                )

            # Grid state
            grid_state = self._gpm.compute_grid_state(positions, candle.close)

            # Context
            ctx = StrategyContext(
                symbol=self._config.symbol,
                timestamp=candle.timestamp,
                candles=candles_by_tf,
                indicators=ctx_indicators,
                current_position=None,
                capital=capital,
                config=None,  # type: ignore[arg-type]
            )

            # a. Si positions ouvertes : check TP/SL global puis should_close_all
            if positions:
                tp_price = self._strategy.get_tp_price(grid_state, main_ind)
                sl_price = self._strategy.get_sl_price(grid_state, main_ind)

                exit_reason, exit_price = self._gpm.check_global_tp_sl(
                    positions, candle, tp_price, sl_price,
                )

                if exit_reason is None:
                    # Vérifier should_close_all (sortie sur signal)
                    signal_exit = self._strategy.should_close_all(ctx, grid_state)
                    if signal_exit:
                        exit_reason = signal_exit
                        exit_price = candle.close

                if exit_reason is not None:
                    trade = self._gpm.close_all_positions(
                        positions, exit_price, candle.timestamp,
                        exit_reason, current_regime,
                    )
                    capital += trade.net_pnl
                    trades.append(trade)
                    positions = []

            # b. Si grille pas pleine : ouvrir de nouvelles positions
            if len(positions) < self._strategy.max_positions and main_ind:
                grid_state = self._gpm.compute_grid_state(positions, candle.close)
                levels = self._strategy.compute_grid(ctx, grid_state)

                # Double sécurité : filtrer les niveaux du mauvais côté
                if positions:
                    active_dir = positions[0].direction
                    levels = [lv for lv in levels if lv.direction == active_dir]

                # Trier par distance au prix courant (plus proche d'abord)
                levels.sort(key=lambda lv: abs(lv.entry_price - candle.close))

                for level in levels:
                    if len(positions) >= self._strategy.max_positions:
                        break

                    # Vérifier si le prix a touché ce niveau
                    touched = False
                    if level.direction == Direction.LONG:
                        touched = candle.low <= level.entry_price
                    else:
                        touched = candle.high >= level.entry_price

                    if touched:
                        pos = self._gpm.open_grid_position(
                            level, candle.timestamp, capital,
                            self._strategy.max_positions,
                        )
                        if pos is not None:
                            positions.append(pos)
                            capital -= pos.entry_fee

            # c. Equity curve
            upnl = self._gpm.unrealized_pnl(positions, candle.close)
            equity_curve.append(capital + upnl)
            equity_timestamps.append(candle.timestamp)

        # Clôture forcée en fin de données
        if positions and main_candles:
            last_candle = main_candles[-1]
            trade = self._gpm.close_all_positions(
                positions, last_candle.close, last_candle.timestamp,
                "end_of_data", current_regime,
            )
            capital += trade.net_pnl
            trades.append(trade)
            positions = []
            if equity_curve:
                equity_curve[-1] = capital

        logger.info(
            "Multi-backtest terminé : {} trades, capital final={:.2f} ({:+.2f}%)",
            len(trades),
            capital,
            (capital / self._config.initial_capital - 1) * 100,
        )

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


def run_multi_backtest_single(
    strategy_name: str,
    params: dict[str, Any],
    candles_by_tf: dict[str, list[Candle]],
    bt_config: BacktestConfig,
    main_tf: str = "1h",
    precomputed_indicators: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> BacktestResult:
    """Lance un backtest multi-position unique avec paramètres custom.

    Équivalent de run_backtest_single() pour les stratégies grid/DCA.
    Utilisé par le WFO pour l'évaluation OOS et le fallback séquentiel.
    """
    from backend.optimization import create_strategy_with_params

    strategy = create_strategy_with_params(strategy_name, params)
    engine = MultiPositionEngine(bt_config, strategy)  # type: ignore[arg-type]
    return engine.run(
        candles_by_tf, main_tf=main_tf,
        precomputed_indicators=precomputed_indicators,
    )
