"""Configuration pour la stratégie trend_follow_daily.

Trend Following sur Daily — EMA cross ou Donchian breakout.
Position unique, trailing stop ATR / channel exit, SL fixe.
Fast engine only (pas de live runner tant que le WFO n'a pas prouvé la viabilité).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrendFollowDailyConfig:
    """Configuration pour trend_follow_daily."""

    name: str = "trend_follow_daily"
    enabled: bool = False
    live_eligible: bool = False
    timeframe: str = "1d"

    # Entry mode
    entry_mode: str = "donchian"  # "ema_cross" ou "donchian"

    # Signal — EMA cross (ignorés si entry_mode == "donchian")
    ema_fast: int = 9
    ema_slow: int = 50

    # Signal — Donchian (ignorés si entry_mode == "ema_cross")
    donchian_entry_period: int = 50  # Breakout N-day high/low
    donchian_exit_period: int = 20   # Canal de sortie (plus court)

    # Filtres partagés
    adx_period: int = 14
    adx_threshold: float = 20.0  # 0 = ADX désactivé

    # Exit
    atr_period: int = 14
    trailing_atr_mult: float = 4.0
    exit_mode: str = "trailing"  # "trailing", "signal" ou "channel"
    sl_percent: float = 10.0  # SL catastrophe

    # Sizing
    cooldown_candles: int = 3
    sides: list[str] = field(default_factory=lambda: ["long"])
    leverage: int = 6
    position_fraction: float = 0.3  # fraction du capital engagée par position
