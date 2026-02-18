"""Factory pour créer les stratégies depuis la config."""

from __future__ import annotations

from backend.core.config import AppConfig
from backend.strategies.base import BaseStrategy
from backend.strategies.bollinger_mr import BollingerMRStrategy
from backend.strategies.boltrend import BolTrendStrategy
from backend.strategies.donchian_breakout import DonchianBreakoutStrategy
from backend.strategies.envelope_dca import EnvelopeDCAStrategy
from backend.strategies.envelope_dca_short import EnvelopeDCAShortStrategy
from backend.strategies.funding import FundingStrategy
from backend.strategies.grid_atr import GridATRStrategy
from backend.strategies.grid_boltrend import GridBolTrendStrategy
from backend.strategies.grid_funding import GridFundingStrategy
from backend.strategies.grid_multi_tf import GridMultiTFStrategy
from backend.strategies.grid_range_atr import GridRangeATRStrategy
from backend.strategies.grid_trend import GridTrendStrategy
from backend.strategies.liquidation import LiquidationStrategy
from backend.strategies.momentum import MomentumStrategy
from backend.strategies.supertrend import SuperTrendStrategy
from backend.strategies.vwap_rsi import VwapRsiStrategy


def create_strategy(name: str, config: AppConfig) -> BaseStrategy:
    """Crée une stratégie par nom depuis la config."""
    strategies_config = config.strategies
    mapping: dict[str, tuple] = {
        "vwap_rsi": (VwapRsiStrategy, strategies_config.vwap_rsi),
        "momentum": (MomentumStrategy, strategies_config.momentum),
        "funding": (FundingStrategy, strategies_config.funding),
        "liquidation": (LiquidationStrategy, strategies_config.liquidation),
        "bollinger_mr": (BollingerMRStrategy, strategies_config.bollinger_mr),
        "donchian_breakout": (DonchianBreakoutStrategy, strategies_config.donchian_breakout),
        "supertrend": (SuperTrendStrategy, strategies_config.supertrend),
        "boltrend": (BolTrendStrategy, strategies_config.boltrend),
        "envelope_dca": (EnvelopeDCAStrategy, strategies_config.envelope_dca),
        "envelope_dca_short": (EnvelopeDCAShortStrategy, strategies_config.envelope_dca_short),
        "grid_atr": (GridATRStrategy, strategies_config.grid_atr),
        "grid_range_atr": (GridRangeATRStrategy, strategies_config.grid_range_atr),
        "grid_multi_tf": (GridMultiTFStrategy, strategies_config.grid_multi_tf),
        "grid_funding": (GridFundingStrategy, strategies_config.grid_funding),
        "grid_trend": (GridTrendStrategy, strategies_config.grid_trend),
        "grid_boltrend": (GridBolTrendStrategy, strategies_config.grid_boltrend),
    }
    if name not in mapping:
        raise ValueError(f"Stratégie inconnue : {name}")

    cls, strat_config = mapping[name]
    return cls(strat_config)


def get_enabled_strategies(config: AppConfig) -> list[BaseStrategy]:
    """Retourne la liste des stratégies activées dans la config."""
    strategies: list[BaseStrategy] = []
    strats = config.strategies

    if strats.vwap_rsi.enabled:
        strategies.append(VwapRsiStrategy(strats.vwap_rsi))
    if strats.momentum.enabled:
        strategies.append(MomentumStrategy(strats.momentum))
    if strats.funding.enabled:
        strategies.append(FundingStrategy(strats.funding))
    if strats.liquidation.enabled:
        strategies.append(LiquidationStrategy(strats.liquidation))
    if strats.bollinger_mr.enabled:
        strategies.append(BollingerMRStrategy(strats.bollinger_mr))
    if strats.donchian_breakout.enabled:
        strategies.append(DonchianBreakoutStrategy(strats.donchian_breakout))
    if strats.supertrend.enabled:
        strategies.append(SuperTrendStrategy(strats.supertrend))
    if strats.boltrend.enabled:
        strategies.append(BolTrendStrategy(strats.boltrend))
    if strats.envelope_dca.enabled:
        strategies.append(EnvelopeDCAStrategy(strats.envelope_dca))
    if strats.envelope_dca_short.enabled:
        strategies.append(EnvelopeDCAShortStrategy(strats.envelope_dca_short))
    if strats.grid_atr.enabled:
        strategies.append(GridATRStrategy(strats.grid_atr))
    if strats.grid_range_atr.enabled:
        strategies.append(GridRangeATRStrategy(strats.grid_range_atr))
    if strats.grid_multi_tf.enabled:
        strategies.append(GridMultiTFStrategy(strats.grid_multi_tf))
    if strats.grid_funding.enabled:
        strategies.append(GridFundingStrategy(strats.grid_funding))
    if strats.grid_trend.enabled:
        strategies.append(GridTrendStrategy(strats.grid_trend))
    if strats.grid_boltrend.enabled:
        strategies.append(GridBolTrendStrategy(strats.grid_boltrend))

    return strategies
