"""Modèles de données partagés pour Scalp Radar.

Tous les types Pydantic utilisés à travers le projet :
enums, candles, signaux, ordres, positions, trades, état de session.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ─── ENUMS ──────────────────────────────────────────────────────────────────


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"


class SignalStrength(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


class MarketRegime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


class TimeFrame(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"

    @classmethod
    def from_string(cls, value: str) -> TimeFrame:
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"TimeFrame inconnu : {value}")

    def to_milliseconds(self) -> int:
        mapping = {
            TimeFrame.M1: 60_000,
            TimeFrame.M5: 300_000,
            TimeFrame.M15: 900_000,
            TimeFrame.H1: 3_600_000,
        }
        return mapping[self]


# ─── DATA MODELS ────────────────────────────────────────────────────────────


class Candle(BaseModel):
    """Bougie OHLCV avec métadonnées."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = Field(ge=0)
    symbol: str
    timeframe: TimeFrame
    vwap: Optional[float] = None
    mark_price: Optional[float] = None

    @model_validator(mode="after")
    def validate_ohlc(self) -> Candle:
        if self.low > min(self.open, self.close):
            raise ValueError(
                f"low ({self.low}) doit être ≤ min(open, close) "
                f"({min(self.open, self.close)})"
            )
        if self.high < max(self.open, self.close):
            raise ValueError(
                f"high ({self.high}) doit être ≥ max(open, close) "
                f"({max(self.open, self.close)})"
            )
        if self.low > self.high:
            raise ValueError(
                f"low ({self.low}) ne peut pas être > high ({self.high})"
            )
        return self


class OrderBookLevel(BaseModel):
    """Niveau du carnet d'ordres."""

    price: float = Field(gt=0)
    quantity: float = Field(gt=0)


class OrderBookSnapshot(BaseModel):
    """Snapshot du carnet d'ordres L2."""

    timestamp: datetime
    symbol: str
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)
    mark_price: Optional[float] = None

    @property
    def spread(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return self.asks[0].price - self.bids[0].price

    @property
    def mid_price(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return (self.asks[0].price + self.bids[0].price) / 2


class TickerData(BaseModel):
    """Données ticker temps réel."""

    symbol: str
    last_price: float
    mark_price: Optional[float] = None
    index_price: Optional[float] = None
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    timestamp: datetime


# ─── TRADING MODELS ─────────────────────────────────────────────────────────


class Signal(BaseModel):
    """Signal de trading émis par une stratégie."""

    timestamp: datetime
    strategy_name: str
    symbol: str
    direction: Direction
    strength: SignalStrength
    score: float = Field(ge=0, le=1)
    entry_price: float = Field(gt=0)
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    market_regime: Optional[MarketRegime] = None
    signals_detail: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)


class Order(BaseModel):
    """Ordre sur l'exchange."""

    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    price: Optional[float] = None
    quantity: float = Field(gt=0)
    status: OrderStatus = OrderStatus.PENDING
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    is_sl_order: bool = False
    is_tp_order: bool = False
    fees: float = 0.0
    timestamp: datetime


class Position(BaseModel):
    """Position ouverte sur l'exchange."""

    symbol: str
    direction: Direction
    entry_price: float = Field(gt=0)
    quantity: float = Field(gt=0)
    leverage: int = Field(ge=1)
    margin_used: float = Field(ge=0)
    initial_margin: float = Field(ge=0)
    maintenance_margin: float = Field(ge=0)
    unrealized_pnl: float = 0.0
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    open_time: datetime


class Trade(BaseModel):
    """Trade clôturé avec P&L complet."""

    id: str
    symbol: str
    direction: Direction
    entry_price: float = Field(gt=0)
    exit_price: float = Field(gt=0)
    quantity: float = Field(gt=0)
    leverage: int = Field(ge=1)
    gross_pnl: float
    fee_cost: float = Field(ge=0)
    slippage_cost: float = Field(ge=0)
    net_pnl: float
    entry_time: datetime
    exit_time: datetime
    strategy_name: str
    market_regime: Optional[MarketRegime] = None

    @model_validator(mode="after")
    def validate_net_pnl(self) -> Trade:
        expected = self.gross_pnl - self.fee_cost - self.slippage_cost
        if abs(self.net_pnl - expected) > 0.01:
            raise ValueError(
                f"net_pnl ({self.net_pnl}) != gross_pnl ({self.gross_pnl}) "
                f"- fee_cost ({self.fee_cost}) - slippage_cost ({self.slippage_cost}) "
                f"= {expected}"
            )
        return self


# ─── STATE MODELS ───────────────────────────────────────────────────────────


class SessionState(BaseModel):
    """État de la session de trading (persisté pour crash recovery)."""

    start_time: datetime
    total_pnl: float = 0.0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    max_drawdown: float = 0.0
    available_margin: float = 0.0
    kill_switch_triggered: bool = False

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades


class MultiTimeframeData(BaseModel):
    """Données agrégées multi-timeframe pour un symbol."""

    model_config = {"arbitrary_types_allowed": True}

    symbol: str
    candles: dict[str, list[Candle]] = Field(default_factory=dict)
    orderbook: Optional[OrderBookSnapshot] = None
    ticker: Optional[TickerData] = None
    last_update: Optional[datetime] = None
