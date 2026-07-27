"""Modèles de données partagés pour Scalp Radar.

Tous les types Pydantic utilisés à travers le projet :
enums, candles, signaux, ordres, positions, trades, état de session.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Any, Optional

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
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"


class SignalStrength(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


class CertificationStatus(str, Enum):
    RESEARCH_ONLY = "RESEARCH_ONLY"
    HISTORICAL_FAIL = "HISTORICAL_FAIL"
    PAPER_READY = "PAPER_READY"
    LIVE_CANARY_READY = "LIVE_CANARY_READY"
    LIVE_APPROVED = "LIVE_APPROVED"
    REJECTED = "REJECTED"


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
    H4 = "4h"
    D1 = "1d"

    @classmethod
    def from_string(cls, value: str) -> TimeFrame:
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"TimeFrame inconnu : {value}")

    def to_minutes(self) -> int:
        return self.to_milliseconds() // 60_000

    def to_milliseconds(self) -> int:
        mapping = {
            TimeFrame.M1: 60_000,
            TimeFrame.M5: 300_000,
            TimeFrame.M15: 900_000,
            TimeFrame.H1: 3_600_000,
            TimeFrame.H4: 14_400_000,
            TimeFrame.D1: 86_400_000,
        }
        return mapping[self]

    def floor_timestamp(self, dt: datetime) -> datetime:
        """Aligne un datetime sur le début de la période du timeframe."""
        minutes = self.to_minutes()
        if minutes < 60:
            return dt.replace(
                minute=(dt.minute // minutes) * minutes,
                second=0,
                microsecond=0,
            )
        elif minutes < 1440:
            hours = minutes // 60
            return dt.replace(
                hour=(dt.hour // hours) * hours,
                minute=0,
                second=0,
                microsecond=0,
            )
        else:
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)


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
    exchange: str = "bitget"
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


class OrderIntent(BaseModel):
    """Canonical strategy-to-broker order request.

    The same payload can be consumed by the Bitget executor, paper broker and
    historical broker.  It deliberately contains no exchange response fields.
    """

    id: str
    account_scope: str
    strategy_name: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float = Field(gt=0)
    created_at: datetime
    price: Optional[float] = Field(default=None, gt=0)
    leverage: int = Field(default=1, ge=1)
    reduce_only: bool = False
    sl_price: Optional[float] = Field(default=None, gt=0)
    tp_price: Optional[float] = Field(default=None, gt=0)
    expires_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_limit_price(self) -> OrderIntent:
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("A LIMIT OrderIntent requires a price")
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        return self


class FillEvent(BaseModel):
    """Canonical broker event for fills and terminal order states."""

    event_id: str
    order_intent_id: str
    symbol: str
    side: OrderSide
    status: OrderStatus
    timestamp: datetime
    fill_price: Optional[float] = Field(default=None, gt=0)
    fill_quantity: float = Field(default=0.0, ge=0)
    cumulative_quantity: float = Field(default=0.0, ge=0)
    fee: float = Field(default=0.0, ge=0)
    exchange_order_id: Optional[str] = None
    reason: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionSpec(BaseModel):
    """Versioned assumptions shared by certification execution paths."""

    model_version: str = "closed_bar_v3"
    scenario: str = "nominal"
    exchange: str = "bitget"
    execution_timeframe: TimeFrame = TimeFrame.M1
    grid_entry_type: OrderType = OrderType.LIMIT
    grid_replace_drift_pct: float = Field(default=0.2, ge=0)
    grid_order_expiry_minutes: int = Field(default=120, ge=1)
    tp_order_type: OrderType = OrderType.MARKET
    sl_server_side: bool = True
    maker_fee_pct: float = Field(default=0.02, ge=0)
    taker_fee_pct: float = Field(default=0.06, ge=0)
    slippage_pct: float = Field(default=0.03, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    missed_fill_probability: float = Field(default=0.0, ge=0, le=1)
    partial_fill_probability: float = Field(default=0.0, ge=0, le=1)
    fee_multiplier: float = Field(default=1.0, gt=0)
    slippage_multiplier: float = Field(default=1.0, ge=0)
    funding_multiplier: float = Field(default=1.0, ge=0)
    sl_gap_fill_fraction: float = Field(default=0.5, ge=0, le=1)
    calibration_id: Optional[str] = None
    calibration_sample_size: int = Field(default=0, ge=0)
    calibration_unfilled_sample_size: int = Field(default=0, ge=0)
    calibration_partial_sample_size: int = Field(default=0, ge=0)
    calibration_observation_hash: Optional[str] = None
    calibration_window_start: Optional[datetime] = None
    calibration_window_end: Optional[datetime] = None
    calibration_latency_p95_ms: int = Field(default=0, ge=0)
    calibration_slippage_p95_pct: float = Field(default=0.0, ge=0)
    random_seed: int = 0

    def with_scenario(self, scenario: str) -> ExecutionSpec:
        """Derive deterministic execution assumptions from one calibration."""
        normalized = scenario.lower()
        if normalized == "nominal":
            return self.model_copy(update={"scenario": "nominal"})
        if normalized == "favorable":
            return self.model_copy(update={
                "scenario": "favorable",
                "latency_ms": 0,
                "missed_fill_probability": 0.0,
                "partial_fill_probability": 0.0,
                "fee_multiplier": 1.0,
                "slippage_multiplier": 0.5,
                "funding_multiplier": 1.0,
                "sl_gap_fill_fraction": 0.0,
            })
        if normalized == "adverse":
            adverse_slippage_multiplier = (
                self.calibration_slippage_p95_pct / self.slippage_pct
                if self.slippage_pct > 0 else 2.0
            )
            return self.model_copy(update={
                "scenario": "adverse",
                "latency_ms": max(self.calibration_latency_p95_ms, self.latency_ms, 2_000),
                "missed_fill_probability": max(self.missed_fill_probability, 0.10),
                "partial_fill_probability": max(self.partial_fill_probability, 0.20),
                "fee_multiplier": max(self.fee_multiplier, 1.25),
                "slippage_multiplier": max(
                    self.slippage_multiplier, adverse_slippage_multiplier, 2.0,
                ),
                "funding_multiplier": max(self.funding_multiplier, 1.5),
                "sl_gap_fill_fraction": max(self.sl_gap_fill_fraction, 1.0),
            })
        raise ValueError(f"Unknown execution scenario: {scenario}")


class AccountRiskSpec(BaseModel):
    """Risk limits applied to one real exchange-account scope."""

    account_scope: str = "default"
    max_live_grids: int = Field(default=4, ge=1)
    max_margin_ratio: float = Field(default=0.70, gt=0, le=1)
    max_simultaneous_sl_loss_ratio: float = Field(default=0.30, gt=0, le=1)
    kill_switch_ratio: float = Field(default=0.45, gt=0, le=1)
    min_liquidation_distance_ratio: float = Field(default=0.50, ge=0, le=1)
    nominal_oos_drawdown_ratio: float = Field(default=0.30, gt=0, le=1)
    adverse_drawdown_ratio: float = Field(default=0.40, gt=0, le=1)
    correlation_groups: dict[str, list[str]] = Field(default_factory=dict)


class UniverseSelectionSpec(BaseModel):
    """Frozen, IS-only universe decision used by a certification snapshot."""

    strategy_name: str = "grid_atr"
    evaluation_scope: str = "universe_discovery"
    universe_symbols: list[str] = Field(min_length=1)
    calendar_start: datetime
    signal_timeframe: str = "1h"
    is_window_days: int = Field(default=180, ge=1)
    embargo_days: int = Field(default=7, ge=0)
    oos_window_days: int = Field(default=60, ge=1)
    step_days: int = Field(default=60, ge=1)
    top_n: int = Field(default=8, ge=1)
    min_is_sharpe: float = 0.0
    min_is_net_return_pct: float = 0.0
    min_is_trades: int = Field(default=10, ge=1)
    search_mode: str = "exhaustive"
    primary_leverage: int = Field(default=4, ge=1)
    leverage_scenarios: list[int] = Field(default_factory=lambda: [2, 4, 6])
    portfolio_initial_capital: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_universe(self) -> "UniverseSelectionSpec":
        symbols = [symbol.strip() for symbol in self.universe_symbols]
        if len(symbols) != len(set(symbols)):
            raise ValueError("universe_symbols must be unique")
        if self.search_mode != "exhaustive":
            raise ValueError("universe discovery requires exhaustive search")
        if self.signal_timeframe != "1h":
            raise ValueError("universe discovery requires 1h signals")
        if self.primary_leverage not in self.leverage_scenarios:
            raise ValueError("primary_leverage must be included in leverage_scenarios")
        if any(leverage < 1 for leverage in self.leverage_scenarios):
            raise ValueError("leverage_scenarios must be positive")
        self.universe_symbols = sorted(symbols)
        self.leverage_scenarios = sorted(set(self.leverage_scenarios))
        return self

    def resolve_portfolio_initial_capital(
        self,
        requested: float | None,
        *,
        legacy_default: float = 1000.0,
    ) -> float:
        """Resolve CLI capital without allowing a frozen snapshot override."""
        frozen = self.portfolio_initial_capital
        if frozen is None:
            return float(requested if requested is not None else legacy_default)
        if requested is not None and not math.isclose(
            float(requested),
            float(frozen),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Portfolio capital differs from immutable snapshot selection: "
                f"requested={requested}, snapshot={frozen}"
            )
        return float(frozen)


class ExperimentManifest(BaseModel):
    """Immutable provenance attached to certifiable research results."""

    snapshot_id: str
    cutoff: datetime
    git_commit: str
    config_hashes: dict[str, str]
    data_hashes: dict[str, str]
    seed: int
    execution_model_version: str
    created_at: datetime
    strategy_name: Optional[str] = None
    params_hash: Optional[str] = None
    dirty_worktree: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


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


class OISnapshot(BaseModel):
    """Snapshot d'open interest pour un symbol."""

    timestamp: datetime
    symbol: str
    value: float  # Open interest en USDT
    change_pct: float = 0.0  # Variation vs snapshot précédent


class MultiTimeframeData(BaseModel):
    """Données agrégées multi-timeframe pour un symbol."""

    model_config = {"arbitrary_types_allowed": True}

    symbol: str
    candles: dict[str, list[Candle]] = Field(default_factory=dict)
    orderbook: Optional[OrderBookSnapshot] = None
    ticker: Optional[TickerData] = None
    last_update: Optional[datetime] = None
