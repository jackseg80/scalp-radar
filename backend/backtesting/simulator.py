"""Simulator (paper trading live) pour Scalp Radar.

Exécute les stratégies sur données live en capital virtuel isolé.
Réutilise PositionManager (fees, slippage, TP/SL) et IncrementalIndicatorEngine.

Sprint 11 : GridStrategyRunner pour les stratégies grid/DCA (envelope_dca).
"""

from __future__ import annotations

import asyncio
import math
import sqlite3
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from loguru import logger

from backend.core.config import AppConfig
from backend.core.data_engine import DataEngine
from backend.core.grid_position_manager import GridPositionManager
from backend.core.incremental_indicators import IncrementalIndicatorEngine
from backend.core.indicators import detect_market_regime
from backend.core.models import Candle, Direction, MarketRegime
from backend.core.position_manager import (
    PositionManager,
    PositionManagerConfig,
    TradeResult,
)
from backend.optimization import is_grid_strategy
from backend.strategies.base import (
    EXTRA_FUNDING_RATE,
    EXTRA_OI_CHANGE_PCT,
    EXTRA_OPEN_INTEREST,
    BaseStrategy,
    OpenPosition,
    StrategyContext,
)
from backend.strategies.base_grid import BaseGridStrategy, GridLevel, GridPosition
from backend.strategies.factory import get_enabled_strategies

if TYPE_CHECKING:
    from backend.core.database import Database

# Sprint 27 : Mapping MarketRegime live → clé WFO regime_analysis
REGIME_LIVE_TO_WFO: dict[MarketRegime, str] = {
    MarketRegime.TRENDING_UP: "bull",
    MarketRegime.TRENDING_DOWN: "bear",
    MarketRegime.RANGING: "range",
    MarketRegime.HIGH_VOLATILITY: "crash",
    # LOW_VOLATILITY intentionnellement absent → toujours autorisé
}


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


def _save_trade_to_db_sync(
    db_path: str,
    strategy_name: str,
    symbol: str,
    trade: TradeResult,
) -> None:
    """Sauvegarde synchrone d'un trade dans la DB avec retry sur lock.

    Backward compatible : log warning si table absente, ne crashe jamais.
    timeout=10s laisse SQLite attendre le lock (busy wait interne).
    Retry 3x avec backoff 100ms/200ms en cas d'échec malgré le timeout.
    """
    import time

    for attempt in range(3):
        try:
            conn = sqlite3.connect(db_path, timeout=10)
            try:
                conn.execute(
                    """INSERT INTO simulation_trades
                       (strategy_name, symbol, direction, entry_price, exit_price, quantity,
                        gross_pnl, fee_cost, slippage_cost, net_pnl, exit_reason,
                        market_regime, entry_time, exit_time)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        strategy_name,
                        symbol,
                        trade.direction.value,
                        trade.entry_price,
                        trade.exit_price,
                        trade.quantity,
                        trade.gross_pnl,
                        trade.fee_cost,
                        trade.slippage_cost,
                        trade.net_pnl,
                        trade.exit_reason,
                        trade.market_regime.value if trade.market_regime else None,
                        trade.entry_time.isoformat(),
                        trade.exit_time.isoformat(),
                    ),
                )
                conn.commit()
                return  # Success
            finally:
                conn.close()
        except sqlite3.OperationalError as e:
            if "no such table" in str(e).lower():
                logger.warning(
                    "Table simulation_trades absente (DB ancienne), trade non persisté"
                )
                return
            if "locked" in str(e).lower() and attempt < 2:
                time.sleep(0.1 * (attempt + 1))  # 100ms, 200ms
                continue
            logger.warning(
                "Trade non sauvegardé (DB locked): {} {} {:.2f}",
                strategy_name, symbol, trade.net_pnl,
            )
            return
        except Exception as e:
            logger.warning("Erreur inattendue sauvegarde trade: {}", e)
            return


@dataclass
class OrphanClosure:
    """Position orpheline fermée au boot (stratégie désactivée)."""

    strategy_name: str
    symbol: str
    direction: str
    entry_price: float
    quantity: float
    estimated_fee_cost: float
    reason: str  # "strategy_disabled"


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
        db_path: str | None = None,
    ) -> None:
        self._strategy = strategy
        self._config = config
        self._indicator_engine = indicator_engine
        self._pm = position_manager
        self._data_engine = data_engine
        self._db_path = db_path

        self._initial_capital = config.risk.initial_capital
        self._capital = self._initial_capital
        self._position: OpenPosition | None = None
        self._position_symbol: str | None = None
        self._trades: list[tuple[str, TradeResult]] = []
        self._current_regime = MarketRegime.RANGING
        self._previous_regime = MarketRegime.RANGING
        self._kill_switch_triggered = False
        self._stats = RunnerStats(
            capital=self._capital,
            initial_capital=self._initial_capital,
        )

        # Circuit breaker — désactive le runner après trop de crashes
        self._crash_times: list[float] = []
        self._circuit_breaker_open: bool = False
        self._CIRCUIT_BREAKER_MAX_CRASHES: int = 3
        self._CIRCUIT_BREAKER_WINDOW_SECONDS: float = 600.0  # 10 minutes

        # Sprint 34a : signal erreur pour Telegram via Simulator
        self._last_indicator_error: tuple[str, str] | None = None

    def _record_crash(self, symbol: str, error: Exception) -> None:
        """Enregistre un crash et déclenche le circuit breaker si seuil atteint."""
        import time as _time

        now = _time.monotonic()
        self._crash_times.append(now)
        self._crash_times = [
            t for t in self._crash_times
            if now - t < self._CIRCUIT_BREAKER_WINDOW_SECONDS
        ]

        crash_count = len(self._crash_times)
        logger.error(
            "[{}] Runner crash #{} sur {}: {}",
            self.name, crash_count, symbol, error,
        )

        if crash_count >= self._CIRCUIT_BREAKER_MAX_CRASHES:
            self._circuit_breaker_open = True
            logger.critical(
                "[{}] CIRCUIT BREAKER OUVERT — runner désactivé après {} crashes en {}s. "
                "Redémarrage du container nécessaire pour réactiver.",
                self.name, self._CIRCUIT_BREAKER_MAX_CRASHES,
                int(self._CIRCUIT_BREAKER_WINDOW_SECONDS),
            )

    @property
    def name(self) -> str:
        return self._strategy.name

    @property
    def is_kill_switch_triggered(self) -> bool:
        return self._kill_switch_triggered

    async def on_candle(self, symbol: str, timeframe: str, candle: Candle) -> None:
        """Traitement d'une nouvelle candle."""
        if self._circuit_breaker_open:
            return

        if self._kill_switch_triggered:
            return

        try:
            await self._on_candle_inner(symbol, timeframe, candle)
        except Exception as e:
            self._record_crash(symbol, e)
            self._last_indicator_error = (symbol, f"CRASH on_candle: {e}")

    async def _on_candle_inner(self, symbol: str, timeframe: str, candle: Candle) -> None:
        """Corps interne de on_candle — séparé pour circuit breaker."""
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
                    self._position_symbol = None
                    return

            # Check TP/SL/signal_exit
            exit_result = self._pm.check_position_exit(
                candle, self._position, self._strategy, ctx, self._current_regime
            )
            if exit_result is not None:
                self._record_trade(exit_result, symbol)
                self._position = None
                self._position_symbol = None
                return

        # 5. Si pas de position : évaluer l'entrée
        if self._position is None and not self._kill_switch_triggered:
            signal = self._strategy.evaluate(ctx)
            if signal is not None:
                self._position = self._pm.open_position(
                    signal, candle.timestamp, self._capital
                )
                if self._position is not None:
                    self._position_symbol = symbol
                    self._capital -= self._position.entry_fee
                    logger.info(
                        "[{}] {} {} @ {:.2f} (score={:.2f})",
                        self.name,
                        signal.direction.value,
                        symbol,
                        signal.entry_price,
                        signal.score,
                    )

    def _record_trade(self, trade: TradeResult, symbol: str = "") -> None:
        """Enregistre un trade et vérifie le kill switch."""
        self._capital += trade.net_pnl
        self._trades.append((symbol, trade))
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

        # Persister en DB (non-bloquant via thread)
        if self._db_path and symbol:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.run_in_executor(
                    None, _save_trade_to_db_sync,
                    self._db_path, self.name, symbol, trade,
                )
            except RuntimeError:
                _save_trade_to_db_sync(self._db_path, self.name, symbol, trade)

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
            # Backward compat : ancien format sans position_symbol
            self._position_symbol = state.get("position_symbol", "UNKNOWN")

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

    def update_indicators_only(
        self, symbol: str, timeframe: str, candle: Candle,
    ) -> None:
        """No-op pour LiveStrategyRunner (pas de close_buffer interne).

        Existe pour duck-typing avec GridStrategyRunner.update_indicators_only().
        """

    def get_status(self) -> dict:
        # Champs unrealized pour cohérence avec GridStrategyRunner
        unrealized_pnl = 0.0
        margin_used = 0.0
        assets_with_positions = 0
        if self._position is not None and self._position_symbol:
            assets_with_positions = 1
        equity = self._capital + margin_used + unrealized_pnl

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
            "unrealized_pnl": round(unrealized_pnl, 2),
            "margin_used": round(margin_used, 2),
            "equity": round(equity, 2),
            "initial_capital": self._initial_capital,
            "assets_with_positions": assets_with_positions,
            "circuit_breaker": self._circuit_breaker_open,
            "crash_count": len(self._crash_times),
        }

    def get_trades(self) -> list[tuple[str, TradeResult]]:
        return list(self._trades)

    def get_stats(self) -> RunnerStats:
        return self._stats


class GridStrategyRunner:
    """Exécute une stratégie grid/DCA sur données live (paper trading).

    Différences avec LiveStrategyRunner :
    - Gère N positions simultanées via GridPositionManager
    - Utilise compute_grid() au lieu de evaluate()
    - TP/SL global (pas par position)
    - Un "trade" = cycle complet (toutes positions ouvertes → toutes fermées)
    """

    # Nombre max de candles chargées depuis la DB pour le warm-up indicateurs
    MAX_WARMUP_CANDLES = 500
    # Seuil d'âge : une candle plus récente que ça = fin du warm-up
    WARMUP_AGE_THRESHOLD = timedelta(hours=2)

    def __init__(
        self,
        strategy: BaseGridStrategy,
        config: AppConfig,
        indicator_engine: IncrementalIndicatorEngine,
        grid_position_manager: GridPositionManager,
        data_engine: DataEngine,
        db_path: str | None = None,
        regime_profile: dict[str, dict] | None = None,
    ) -> None:
        self._strategy = strategy
        self._config = config
        self._indicator_engine = indicator_engine
        self._gpm = grid_position_manager
        self._data_engine = data_engine
        self._db_path = db_path

        # Sprint 27 : profil WFO par régime, indexé par symbol
        # Format: {"BTC/USDT": {"bull": {"avg_oos_sharpe": 6.88, ...}, ...}}
        self._regime_profile = regime_profile
        self._regime_filter_blocks: int = 0

        self._initial_capital = config.risk.initial_capital
        self._capital = self._initial_capital
        self._realized_pnl = 0.0  # P&L réalisé (trades clôturés uniquement)
        self._positions: dict[str, list[GridPosition]] = {}  # Positions par symbol
        self._trades: list[tuple[str, TradeResult]] = []
        self._current_regime = MarketRegime.RANGING
        self._kill_switch_triggered = False
        self._stats = RunnerStats(
            capital=self._capital,
            initial_capital=self._initial_capital,
        )

        # Sprint 25 : queue d'événements pour le journal d'activité
        self._pending_journal_events: list[dict] = []

        # Sprint 34a : signal erreur compute_live_indicators → Telegram via Simulator
        self._last_indicator_error: tuple[str, str] | None = None

        # COMPAT: duck typing pour Simulator/StateManager qui accèdent à ces attributs.
        # Voir self._positions pour le vrai état grid.
        self._position = None
        self._position_symbol = None

        # Leverage pour calcul marge (réservation capital)
        self._leverage = getattr(strategy._config, "leverage", 15)

        # Prix courants par symbol (mis à jour à chaque on_candle)
        self._last_prices: dict[str, float] = {}

        # Buffer de closes pour calcul SMA interne
        self._strategy_tf = getattr(strategy._config, "timeframe", "1h")
        self._ma_period = getattr(strategy._config, "ma_period", 7)
        self._close_buffer: dict[str, deque] = {}

        # Nombre d'assets pour diviser le capital proportionnellement
        per_asset = getattr(strategy._config, "per_asset", {})
        nb_from_per_asset = len(per_asset) if isinstance(per_asset, dict) else 0
        self._nb_assets = nb_from_per_asset if nb_from_per_asset > 0 else len(config.assets)
        if self._nb_assets < 1:
            self._nb_assets = 1

        # Whitelist des symbols autorisés (si per_asset défini, sinon aucune restriction)
        self._per_asset_keys: set[str] = (
            set(per_asset.keys()) if isinstance(per_asset, dict) and per_asset else set()
        )

        # Warm-up : capital fixe pendant le replay des candles historiques
        self._is_warming_up = True
        self._pending_restore: dict | None = None  # État à appliquer après warm-up

        # Grace period : pas de kill switch runner pendant les N premières bougies
        self._candles_since_warmup: int = 0
        self._grace_period_candles: int = 10  # 10 bougies 1h = ~10h de grâce

        self._warmup_ended_at: datetime | None = None  # set dans _end_warmup()

        # Phase 2 : cooldown anti-churning — timestamp du dernier close par symbol
        self._last_close_time: dict[str, datetime] = {}

        # Funding costs tracking (approximation 0.01% par settlement)
        self._total_funding_cost: float = 0.0

        # Circuit breaker — désactive le runner après trop de crashes
        self._crash_times: list[float] = []
        self._circuit_breaker_open: bool = False
        self._CIRCUIT_BREAKER_MAX_CRASHES: int = 3
        self._CIRCUIT_BREAKER_WINDOW_SECONDS: float = 600.0  # 10 minutes

    def _record_crash(self, symbol: str, error: Exception) -> None:
        """Enregistre un crash et déclenche le circuit breaker si seuil atteint."""
        import time as _time

        now = _time.monotonic()
        self._crash_times.append(now)
        # Garder seulement les crashes dans la fenêtre
        self._crash_times = [
            t for t in self._crash_times
            if now - t < self._CIRCUIT_BREAKER_WINDOW_SECONDS
        ]

        crash_count = len(self._crash_times)
        logger.error(
            "[{}] Runner crash #{} sur {}: {}",
            self.name, crash_count, symbol, error,
        )

        if crash_count >= self._CIRCUIT_BREAKER_MAX_CRASHES:
            self._circuit_breaker_open = True
            logger.critical(
                "[{}] CIRCUIT BREAKER OUVERT — runner désactivé après {} crashes en {}s. "
                "Redémarrage du container nécessaire pour réactiver.",
                self.name, self._CIRCUIT_BREAKER_MAX_CRASHES,
                int(self._CIRCUIT_BREAKER_WINDOW_SECONDS),
            )

    def _record_close(self, symbol: str, close_timestamp: datetime) -> None:
        """Enregistre le timestamp du dernier close pour le cooldown anti-churning."""
        self._last_close_time[symbol] = close_timestamp

    def _get_sl_percent(self, symbol: str) -> float:
        """Résout le sl_percent pour un symbol (avec override per_asset)."""
        config = self._strategy._config
        default_sl = getattr(config, "sl_percent", 25.0)
        per_asset = getattr(config, "per_asset", {})
        if isinstance(per_asset, dict):
            overrides = per_asset.get(symbol, {})
            if isinstance(overrides, dict) and "sl_percent" in overrides:
                return float(overrides["sl_percent"])
        return float(default_sl) if isinstance(default_sl, (int, float)) else 25.0

    def _get_num_levels(self, symbol: str) -> int:
        """Résout num_levels pour un symbol (avec override per_asset)."""
        config = self._strategy._config
        default = self._strategy.max_positions
        per_asset = getattr(config, "per_asset", {})
        if isinstance(per_asset, dict):
            overrides = per_asset.get(symbol, {})
            if isinstance(overrides, dict) and "num_levels" in overrides:
                return int(overrides["num_levels"])
        return default

    def _should_allow_new_grid(self, symbol: str) -> bool:
        """Sprint 27 : Filtre Darwinien — bloque si régime WFO défavorable.

        Logique :
        - Filtre désactivé dans config → True
        - Pas de profil → True (backward compat)
        - Régime live non mappé (LOW_VOLATILITY) → True
        - Régime WFO non couvert → True (bénéfice du doute)
        - avg_oos_sharpe < 0 → False (bloqué)
        - Sinon → True
        """
        if not getattr(self._config.risk, "regime_filter_enabled", True):
            return True
        if not self._regime_profile:
            return True
        symbol_profile = self._regime_profile.get(symbol)
        if not symbol_profile:
            return True
        wfo_key = REGIME_LIVE_TO_WFO.get(self._current_regime)
        if wfo_key is None:
            return True
        regime_data = symbol_profile.get(wfo_key)
        if regime_data is None:
            return True
        avg_sharpe = regime_data.get("avg_oos_sharpe", 0.0)
        if not isinstance(avg_sharpe, (int, float)):
            return True
        if avg_sharpe < 0:
            self._regime_filter_blocks += 1
            logger.info(
                "[{}] REGIME FILTER : {} bloqué (régime={}, sharpe={:.2f})",
                self.name, symbol, wfo_key, avg_sharpe,
            )
            return False
        return True

    @property
    def name(self) -> str:
        return self._strategy.name

    @property
    def is_kill_switch_triggered(self) -> bool:
        return self._kill_switch_triggered

    @property
    def strategy(self) -> BaseStrategy:
        return self._strategy

    @property
    def current_regime(self) -> MarketRegime:
        return self._current_regime

    async def _warmup_from_db(self, db: Database, symbol: str) -> None:
        """Pré-charge les bougies historiques depuis la DB pour le warm-up SMA."""
        # Skip les symbols hors whitelist per_asset (ex: grid_boltrend = 6 assets sur 22)
        if self._per_asset_keys and symbol not in self._per_asset_keys:
            return
        # Sprint 34a : utiliser strategy.min_candles pour le warm-up dynamique
        strat_min = self._strategy.min_candles.get(self._strategy_tf, 50)
        needed = min(max(strat_min, 50), self.MAX_WARMUP_CANDLES)
        candles = await db.get_recent_candles(symbol, self._strategy_tf, needed)
        if not candles:
            logger.info(
                "[{}] Warm-up: 0 bougies {} en DB pour {}",
                self.name, self._strategy_tf, symbol,
            )
            return

        if symbol not in self._close_buffer:
            self._close_buffer[symbol] = deque(
                maxlen=max(self._ma_period + 20, 50)
            )

        for candle in candles:
            self._close_buffer[symbol].append(candle.close)
            self._indicator_engine.update(symbol, self._strategy_tf, candle)

        logger.info(
            "[{}] Warm-up: {} bougies {} chargées pour {}",
            self.name, len(candles), self._strategy_tf, symbol,
        )

    def _end_warmup(self) -> None:
        """Termine le warm-up : restaure l'état sauvegardé ou reset à initial_capital."""
        warmup_trade_count = len(self._trades)

        # Fermer toutes les positions warm-up (pas de record trade)
        self._positions.clear()

        # Si un état était en attente de restauration (restart avec state)
        pending = getattr(self, "_pending_restore", None)
        if pending is not None:
            self._apply_restored_state(pending)
            self._pending_restore = None
            logger.info(
                "[{}] Warm-up terminé : {} trades historiques, état restauré (capital={:.2f}$)",
                self.name, warmup_trade_count, self._capital,
            )
        else:
            # Pas d'état à restaurer → reset propre à initial_capital
            self._capital = self._initial_capital
            self._realized_pnl = 0.0
            self._stats = RunnerStats(
                capital=self._capital,
                initial_capital=self._initial_capital,
            )
            logger.info(
                "[{}] Warm-up terminé : {} trades historiques, capital reset à {:.0f}$",
                self.name, warmup_trade_count, self._capital,
            )

        self._is_warming_up = False
        self._warmup_ended_at = datetime.now(tz=timezone.utc)

    def _apply_restored_state(self, state: dict) -> None:
        """Applique un état sauvegardé (appelé après le warm-up).

        Note : kill_switch est toujours remis à False ici car les grid runners
        n'ont pas de kill switch runner-level. Si le kill switch global est actif,
        _stop_all_runners() le remettra à True APRÈS _end_warmup() (voir start()).
        """
        self._capital = state.get("capital", self._initial_capital)
        self._kill_switch_triggered = False
        self._realized_pnl = state.get("realized_pnl", state.get("net_pnl", 0.0))

        self._stats.capital = self._capital
        self._stats.net_pnl = self._realized_pnl
        self._stats.total_trades = state.get("total_trades", 0)
        self._stats.wins = state.get("wins", 0)
        self._stats.losses = state.get("losses", 0)
        self._stats.is_active = state.get("is_active", True)

        # Restaurer les positions grid ouvertes
        grid_positions = state.get("grid_positions", [])
        for gp in grid_positions:
            symbol = gp.get("symbol", "UNKNOWN")
            pos = GridPosition(
                level=gp["level"],
                direction=Direction(gp["direction"]),
                entry_price=gp["entry_price"],
                quantity=gp["quantity"],
                entry_time=datetime.fromisoformat(gp["entry_time"]),
                entry_fee=gp["entry_fee"],
            )
            self._positions.setdefault(symbol, []).append(pos)

        # Phase 2 : restaurer cooldown anti-churning
        for sym, ts_str in state.get("last_close_times", {}).items():
            try:
                self._last_close_time[sym] = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                pass

        # Log diagnostic positions restaurées
        total_restored = sum(len(p) for p in self._positions.values())
        expected = len(grid_positions)
        if total_restored != expected:
            logger.warning(
                "[{}] Mismatch positions restaurées: {} attendues, {} effectives",
                self.name, expected, total_restored,
            )
        elif total_restored > 0:
            logger.info(
                "[{}] {} positions grid restaurées (kill_switch={}, is_active={})",
                self.name, total_restored,
                self._kill_switch_triggered, self._stats.is_active,
            )

    async def on_candle(
        self, symbol: str, timeframe: str, candle: Candle
    ) -> None:
        """Traitement d'une nouvelle candle — logique grid."""
        if self._circuit_breaker_open:
            return

        if self._kill_switch_triggered:
            return

        try:
            await self._on_candle_inner(symbol, timeframe, candle)
        except Exception as e:
            self._record_crash(symbol, e)
            self._last_indicator_error = (symbol, f"CRASH on_candle: {e}")

    async def _on_candle_inner(
        self, symbol: str, timeframe: str, candle: Candle
    ) -> None:
        """Corps interne de on_candle — séparé pour circuit breaker."""
        # Filtre per_asset : skip si symbol n'est pas dans la whitelist
        if self._per_asset_keys and symbol not in self._per_asset_keys:
            return

        # Rafraîchir le prix courant pour TOUTES les timeframes
        # (P&L temps réel dans get_status(), mis à jour chaque 1m au lieu de 1h)
        self._last_prices[symbol] = candle.close

        # Filtre strict : seul le timeframe de la stratégie est traité
        if timeframe != self._strategy_tf:
            return

        # Détection fin warm-up : candle récente = données live
        if self._is_warming_up:
            now = datetime.now(tz=timezone.utc)
            candle_age = now - candle.timestamp
            if candle_age <= self.WARMUP_AGE_THRESHOLD:
                self._end_warmup()

        # Guard anti-phantom trades : dans les premières minutes post-warmup,
        # skip les bougies historiques (> 2h). Protège contre le batch de watch_ohlcv().
        if not self._is_warming_up:
            warmup_ended = getattr(self, '_warmup_ended_at', None)
            if warmup_ended is not None:
                real_elapsed = (
                    datetime.now(tz=timezone.utc) - warmup_ended
                ).total_seconds()
                if real_elapsed < 300:  # 5 min post-warmup
                    candle_age_s = (
                        datetime.now(tz=timezone.utc) - candle.timestamp
                    ).total_seconds()
                    if candle_age_s > 7200:
                        return  # Bougie historique, ne pas trader

            # Compteur grace period (kill switch runner)
            self._candles_since_warmup += 1

        # Maintenir le buffer de closes
        if symbol not in self._close_buffer:
            self._close_buffer[symbol] = deque(
                maxlen=max(self._ma_period + 20, 50)
            )
        self._close_buffer[symbol].append(candle.close)

        # Calculer SMA
        closes = list(self._close_buffer[symbol])
        if len(closes) < self._ma_period:
            return

        sma_val = float(np.mean(closes[-self._ma_period:]))

        # Récupérer les indicateurs de l'engine et merger SMA
        indicators = self._indicator_engine.get_indicators(symbol)
        if not indicators:
            indicators = {}
        indicators.setdefault(self._strategy_tf, {}).update({
            "sma": sma_val,
            "close": candle.close,
        })

        # Indicateurs supplémentaires (ex: Supertrend 4h pour grid_multi_tf)
        # Skippé pendant le warm-up : indicateurs pas encore valides
        if not self._is_warming_up:
            buffers = getattr(self._indicator_engine, "_buffers", None)
            if isinstance(buffers, dict):
                candle_buf = buffers.get((symbol, self._strategy_tf), [])
                if candle_buf:
                    try:
                        extra = self._strategy.compute_live_indicators(
                            list(candle_buf),
                        )
                        for tf_key, tf_data in extra.items():
                            indicators.setdefault(tf_key, {}).update(tf_data)
                    except Exception as e:
                        logger.error(
                            "[{}] compute_live_indicators ERREUR pour {}: {}",
                            self.name, symbol, e,
                        )
                        self._last_indicator_error = (symbol, str(e))

        # Détecter le régime (si ADX/ATR disponibles)
        main_ind = indicators.get(self._strategy_tf, {})
        self._current_regime = detect_market_regime(
            main_ind.get("adx", float("nan")),
            main_ind.get("di_plus", float("nan")),
            main_ind.get("di_minus", float("nan")),
            main_ind.get("atr", float("nan")),
            main_ind.get("atr_sma", float("nan")),
        )

        # Capital utilisé pour le sizing : fixe pendant warm-up, réel en live
        raw_capital = self._initial_capital if self._is_warming_up else self._capital
        sizing_capital = raw_capital / self._nb_assets

        # Construire le contexte
        ctx = StrategyContext(
            symbol=symbol,
            timestamp=candle.timestamp,
            candles={},
            indicators=indicators,
            current_position=None,
            capital=sizing_capital,
            config=self._config,
        )

        # Récupérer les positions de ce symbol uniquement
        positions = self._positions.get(symbol, [])

        # Funding cost (settlement toutes les 8h : 00:00, 08:00, 16:00 UTC)
        # Appliqué AVANT TP/SL : si position ouverte au settlement, on paie le funding
        if not self._is_warming_up and positions and candle.timestamp.hour in (0, 8, 16):
            # Lire le funding rate réel depuis DataEngine (en %, ex: 0.01 = 0.01%)
            # Fallback sur 0.01% si non disponible (approximation conservative)
            raw_fr = self._data_engine.get_funding_rate(symbol) if self._data_engine else None
            funding_rate = (raw_fr / 100) if isinstance(raw_fr, (int, float)) else 0.0001
            for pos in positions:
                notional = pos.entry_price * pos.quantity
                if pos.direction == Direction.LONG:
                    cost = notional * funding_rate   # LONG paie
                else:
                    cost = -notional * funding_rate  # SHORT reçoit
                self._capital -= cost
                self._total_funding_cost += cost

        # Construire le GridState
        grid_state = self._gpm.compute_grid_state(positions, candle.close)

        # 1. Si positions ouvertes → check TP/SL global
        if positions:
            tp_price = self._strategy.get_tp_price(grid_state, main_ind)
            sl_price = self._strategy.get_sl_price(grid_state, main_ind)

            # Check via OHLC heuristic
            exit_reason, exit_price = self._gpm.check_global_tp_sl(
                positions, candle, tp_price, sl_price
            )

            # Si pas de TP/SL OHLC, check should_close_all (signal)
            if exit_reason is None:
                close_reason = self._strategy.should_close_all(ctx, grid_state)
                if close_reason:
                    exit_reason = close_reason
                    exit_price = candle.close

            if exit_reason:
                # Rendre la marge réservée pour toutes les positions
                total_notional = sum(
                    p.entry_price * p.quantity for p in positions
                )
                margin_to_return = total_notional / self._leverage
                if not self._is_warming_up:
                    self._capital += margin_to_return

                trade = self._gpm.close_all_positions(
                    positions, exit_price,
                    candle.timestamp, exit_reason,
                    self._current_regime,
                )
                if self._is_warming_up:
                    # Warm-up : enregistrer dans l'historique sans modifier capital/stats
                    self._trades.append((symbol, trade))
                else:
                    self._record_trade(trade, symbol)
                    self._record_close(symbol, candle.timestamp)
                self._positions[symbol] = []
                if not self._is_warming_up:
                    self._pending_journal_events.append({
                        "timestamp": trade.exit_time.isoformat(),
                        "strategy_name": self.name,
                        "symbol": symbol,
                        "event_type": "CLOSE",
                        "level": None,
                        "direction": trade.direction.value,
                        "price": trade.exit_price,
                        "quantity": trade.quantity,
                        "unrealized_pnl": round(trade.net_pnl, 2),
                        "metadata": {
                            "exit_reason": trade.exit_reason,
                            "entry_price": trade.entry_price,
                            "gross_pnl": round(trade.gross_pnl, 2),
                            "net_pnl": round(trade.net_pnl, 2),
                            "fee_cost": round(trade.fee_cost, 2),
                        },
                    })
                return

        # Sprint 27 : Filtre Darwinien — bloquer si régime défavorable
        # Uniquement quand aucune position ouverte (ne touche pas les grilles existantes)
        if not positions and not self._should_allow_new_grid(symbol):
            return

        # Phase 2 : cooldown anti-churning — bloquer si close récent
        if not positions:
            raw_cd = getattr(self._strategy._config, "cooldown_candles", 0)
            cooldown = raw_cd if isinstance(raw_cd, (int, float)) else 0
            if cooldown > 0 and symbol in self._last_close_time:
                from backend.strategies.base_grid import TF_SECONDS
                tf_seconds = TF_SECONDS.get(self._strategy_tf, 3600)
                elapsed = (candle.timestamp - self._last_close_time[symbol]).total_seconds()
                if elapsed < cooldown * tf_seconds:
                    return

        # 2. Ouvrir de nouveaux niveaux si grille pas pleine
        effective_max = self._get_num_levels(symbol)
        if len(positions) < effective_max:
            # Patcher temporairement num_levels pour que compute_grid()
            # génère le bon nombre de niveaux (per_asset override)
            original_num_levels = self._strategy._config.num_levels
            self._strategy._config.num_levels = effective_max
            try:
                levels = self._strategy.compute_grid(ctx, grid_state)
            finally:
                self._strategy._config.num_levels = original_num_levels

            for level in levels:
                if level.index in {p.level for p in positions}:
                    continue

                touched = False
                if level.direction == Direction.LONG:
                    touched = candle.low <= level.entry_price
                else:
                    touched = candle.high >= level.entry_price

                if touched:
                    # Sizing : fixe en portfolio mode, fixe pendant warm-up, compound en live
                    if getattr(self, "_portfolio_mode", False):
                        pos_raw = self._initial_capital
                    elif self._is_warming_up:
                        pos_raw = self._initial_capital
                    else:
                        pos_raw = self._capital
                    pos_per_asset = pos_raw / self._nb_assets

                    # Equal allocation sizing (Sprint 20a)
                    # Note : l'ajout de runners multi-stratégie dilue l'allocation
                    # (10 assets → 16 runners = -37% par runner). Accepté pour le paper.
                    # Options si besoin : capital séparé par stratégie (config.weight)
                    # ou compter les assets uniques au lieu des runners.
                    # Marge fixe par niveau = capital / nb_assets / num_levels
                    # Le SL contrôle le risque en $, PAS la taille de position
                    num_levels = effective_max
                    margin_per_level = pos_per_asset / num_levels

                    # Cap de sécurité : jamais plus de 25% du capital sur un seul asset
                    max_margin_per_asset = pos_raw * 0.25
                    margin_per_level = min(margin_per_level, max_margin_per_asset / num_levels)

                    # Margin guard (Sprint 20a) — skip si marge totale dépasse le seuil
                    max_margin_ratio = getattr(self._config.risk, "max_margin_ratio", 0.70)
                    if not isinstance(max_margin_ratio, (int, float)):
                        max_margin_ratio = 0.70
                    total_margin_used = sum(
                        p.entry_price * p.quantity / self._leverage
                        for positions_list in self._positions.values()
                        for p in positions_list
                    )
                    if total_margin_used + margin_per_level > pos_raw * max_margin_ratio:
                        _h = getattr(self, "_on_skip_local", None)
                        if callable(_h): _h(symbol)
                        continue  # Skip ce niveau, pas assez de marge

                    # Global margin guard (Sprint 24a) — portfolio backtest seulement
                    portfolio_runners = getattr(self, "_portfolio_runners", None)
                    portfolio_cap = getattr(self, "_portfolio_initial_capital", None)
                    if portfolio_runners is not None and portfolio_cap is not None:
                        global_margin = sum(
                            p.entry_price * p.quantity / r._leverage
                            for r in portfolio_runners.values()
                            for positions_list in r._positions.values()
                            for p in positions_list
                        )
                        if global_margin + margin_per_level > portfolio_cap * max_margin_ratio:
                            _h = getattr(self, "_on_skip_global", None)
                            if callable(_h): _h(symbol)
                            continue  # Skip — marge globale dépasserait le seuil

                    pos_capital = margin_per_level * num_levels

                    position = self._gpm.open_grid_position(
                        level, candle.timestamp,
                        pos_capital,
                        effective_max,
                    )
                    if position:
                        # Réserver la marge (les fees sont dans net_pnl à la fermeture)
                        notional = position.entry_price * position.quantity
                        margin_used = notional / self._leverage
                        if not self._is_warming_up:
                            if self._capital < margin_used:
                                logger.warning(
                                    "[{}] Capital insuffisant pour level {} "
                                    "({:.2f} requis, {:.2f} disponible)",
                                    self.name, level.index,
                                    margin_used, self._capital,
                                )
                                continue
                            self._capital -= margin_used
                        self._positions.setdefault(symbol, []).append(position)
                        _h = getattr(self, "_on_position_opened", None)
                        if callable(_h): _h(symbol)
                        logger.info(
                            "[{}] GRID {} level {} @ {:.2f} ({}) — {}/{} positions "
                            "(marge={:.2f}, capital={:.2f})",
                            self.name,
                            level.direction.value,
                            level.index,
                            level.entry_price,
                            symbol,
                            len(self._positions[symbol]),
                            effective_max,
                            margin_used,
                            self._capital,
                        )
                        if not self._is_warming_up:
                            self._pending_journal_events.append({
                                "timestamp": position.entry_time.isoformat(),
                                "strategy_name": self.name,
                                "symbol": symbol,
                                "event_type": "OPEN",
                                "level": level.index,
                                "direction": position.direction.value,
                                "price": position.entry_price,
                                "quantity": position.quantity,
                                "margin_used": round(
                                    position.entry_price * position.quantity / self._leverage, 2,
                                ),
                                "metadata": {
                                    "levels_open": len(self._positions.get(symbol, [])),
                                    "levels_max": effective_max,
                                },
                            })

    def _record_trade(self, trade: TradeResult, symbol: str = "") -> None:
        """Enregistre un trade grid (fermeture de toutes les positions)."""
        self._capital += trade.net_pnl
        self._realized_pnl += trade.net_pnl
        self._trades.append((symbol, trade))
        self._stats.total_trades += 1
        self._stats.net_pnl = self._realized_pnl
        self._stats.capital = self._capital

        if trade.net_pnl > 0:
            self._stats.wins += 1
        else:
            self._stats.losses += 1

        logger.info(
            "[{}] Grid trade clos : {} avg={:.2f} → {:.2f}, net={:+.2f} ({})",
            self.name,
            trade.direction.value,
            trade.entry_price,
            trade.exit_price,
            trade.net_pnl,
            trade.exit_reason,
        )

        # Persister en DB (non-bloquant via thread)
        if self._db_path and symbol:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.run_in_executor(
                    None, _save_trade_to_db_sync,
                    self._db_path, self.name, symbol, trade,
                )
            except RuntimeError:
                _save_trade_to_db_sync(self._db_path, self.name, symbol, trade)

        # Kill switch grid : seuils plus larges (drawdown structurel normal)
        # Grace period : pas de kill switch pendant les N premières bougies post-warmup
        if self._candles_since_warmup < self._grace_period_candles:
            return

        ks = self._config.risk.kill_switch
        grid_session = getattr(ks, 'grid_max_session_loss_percent', None)
        std_session = getattr(ks, 'max_session_loss_percent', None)
        max_session = (
            grid_session if isinstance(grid_session, (int, float)) else
            std_session if isinstance(std_session, (int, float)) else None
        )
        if max_session is None:
            return  # Pas de config kill switch → skip

        session_loss_pct = abs(min(0, self._realized_pnl)) / self._initial_capital * 100
        if session_loss_pct >= max_session:
            self._kill_switch_triggered = True
            self._stats.is_active = False
            logger.warning(
                "[{}] KILL SWITCH GRID : perte session {:.1f}% >= {:.1f}%",
                self.name, session_loss_pct, max_session,
            )

    def restore_state(self, state: dict) -> None:
        """Restaure l'état du runner depuis un snapshot sauvegardé.

        Le warm-up reste actif : watch_ohlcv() renvoie un batch historique
        au démarrage qui doit être traité avec initial_capital (pas le capital
        restauré). _end_warmup() appliquera l'état sauvegardé.
        """
        # Garder le warm-up actif pour que les candles historiques initiales
        # utilisent initial_capital. L'état sera appliqué à _end_warmup().
        self._pending_restore = state
        self._is_warming_up = True

        logger.info(
            "[{}] État sauvegardé pour restauration post-warmup : capital={:.2f}, trades={}",
            self.name,
            state.get("capital", self._initial_capital),
            state.get("total_trades", 0),
        )

    def build_context(self, symbol: str) -> StrategyContext | None:
        """Construit un StrategyContext pour le dashboard (get_conditions)."""
        if not self._indicator_engine:
            return None

        indicators = self._indicator_engine.get_indicators(symbol)
        if not indicators:
            indicators = {}

        # Merger SMA depuis le buffer interne
        closes = list(self._close_buffer.get(symbol, []))
        if len(closes) >= self._ma_period:
            sma_val = float(np.mean(closes[-self._ma_period:]))
            indicators.setdefault(self._strategy_tf, {}).update({
                "sma": sma_val,
                "close": closes[-1] if closes else 0.0,
            })

        return StrategyContext(
            symbol=symbol,
            timestamp=datetime.now(tz=timezone.utc),
            candles={},
            indicators=indicators,
            current_position=None,
            capital=self._capital,
            config=self._config,
        )

    def update_indicators_only(
        self, symbol: str, timeframe: str, candle: Candle,
    ) -> None:
        """Met à jour le close_buffer/SMA sans évaluer de trades.

        Appelé par _dispatch_candle quand le kill switch global est actif.
        L'Executor live lit ces données via get_runner_context() / build_context().
        """
        if timeframe != self._strategy_tf:
            return
        if symbol not in self._close_buffer:
            self._close_buffer[symbol] = deque(
                maxlen=max(self._ma_period + 20, 50),
            )
        self._close_buffer[symbol].append(candle.close)

    def get_grid_positions(self) -> list[dict]:
        """Retourne les positions grid ouvertes pour le dashboard."""
        result = []
        for symbol, positions in self._positions.items():
            for p in positions:
                result.append({
                    "symbol": symbol,
                    "strategy": self.name,
                    "direction": p.direction.value,
                    "entry_price": p.entry_price,
                    "quantity": p.quantity,
                    "entry_time": p.entry_time.isoformat(),
                    "level": p.level,
                    "type": "grid",
                })
        return result

    def get_status(self) -> dict:
        """Même interface que LiveStrategyRunner.get_status() + champs grid."""
        total_positions = sum(len(positions) for positions in self._positions.values())
        all_positions = [p for positions in self._positions.values() for p in positions]

        # P&L non réalisé + marge utilisée
        unrealized_pnl = 0.0
        margin_used = 0.0
        assets_with_positions = 0
        for symbol, positions in self._positions.items():
            if not positions:
                continue
            assets_with_positions += 1
            current_price = self._last_prices.get(symbol, 0.0)
            if current_price <= 0:
                continue
            for pos in positions:
                if pos.direction.value == "LONG":
                    unrealized_pnl += (current_price - pos.entry_price) * pos.quantity
                else:
                    unrealized_pnl += (pos.entry_price - current_price) * pos.quantity
                margin_used += (pos.entry_price * pos.quantity) / self._leverage

        equity = self._capital + margin_used + unrealized_pnl

        return {
            "name": self.name,
            "capital": self._capital,
            "net_pnl": self._stats.net_pnl,
            "total_trades": self._stats.total_trades,
            "wins": self._stats.wins,
            "losses": self._stats.losses,
            "win_rate": (
                self._stats.wins / self._stats.total_trades * 100
                if self._stats.total_trades > 0 else 0.0
            ),
            "is_active": self._stats.is_active,
            "kill_switch": self._kill_switch_triggered,
            "has_position": total_positions > 0,
            "open_positions": total_positions,
            "max_positions": self._strategy.max_positions,
            "avg_entry_price": (
                sum(p.entry_price * p.quantity for p in all_positions)
                / (sum(p.quantity for p in all_positions) or 1.0)
                if all_positions else 0.0
            ),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "margin_used": round(margin_used, 2),
            "equity": round(equity, 2),
            "initial_capital": self._initial_capital,
            "assets_with_positions": assets_with_positions,
            "regime_filter_blocks": self._regime_filter_blocks,
            "funding_cost": round(self._total_funding_cost, 2),
            "watched_symbols": sorted(self._per_asset_keys) if self._per_asset_keys else [],
            "leverage": self._leverage,
            "is_warming_up": self._is_warming_up,
            "circuit_breaker": self._circuit_breaker_open,
            "crash_count": len(self._crash_times),
        }

    def get_trades(self) -> list[tuple[str, TradeResult]]:
        return list(self._trades)

    def get_stats(self) -> RunnerStats:
        return self._stats


class Simulator:
    """Orchestrateur du paper trading.

    Crée un LiveStrategyRunner ou GridStrategyRunner par stratégie enabled.
    Se câble sur le DataEngine via on_candle.
    """

    def __init__(
        self,
        data_engine: DataEngine,
        config: AppConfig,
        db: Database | None = None,
    ) -> None:
        self._data_engine = data_engine
        self._config = config
        self._db = db
        self._runners: list[LiveStrategyRunner | GridStrategyRunner] = []
        self._indicator_engine: IncrementalIndicatorEngine | None = None
        self._running = False
        self._orphan_closures: list[OrphanClosure] = []
        self._collision_warnings: list[dict] = []

        # Sprint 6 : caches pour le dashboard
        self._conditions_cache: dict | None = None
        self._conditions_cache_time: float = 0.0
        self._equity_cache: list[dict] | None = None
        self._trade_count_at_cache: int = 0

        # Kill switch global (filet de sécurité toutes stratégies)
        self._capital_snapshots: deque[tuple[datetime, float]] = deque(maxlen=1440)
        self._global_kill_switch = False
        self._kill_switch_reason: dict | None = None
        self._notifier: Any = None
        self._warmup_ended_at: datetime | None = None

    def set_notifier(self, notifier: Any) -> None:
        """Injecte le Notifier pour les alertes kill switch global."""
        self._notifier = notifier

    # ─── Kill switch global ──────────────────────────────────────────────

    def _snapshot_capital(self) -> None:
        """Prend un snapshot du capital total (toutes stratégies)."""
        if not self._runners:
            return
        total = sum(r._capital for r in self._runners)
        self._capital_snapshots.append(
            (datetime.now(tz=timezone.utc), total)
        )

    async def _check_global_kill_switch(self) -> None:
        """Vérifie le drawdown global sur la fenêtre glissante.

        Déclenche le kill switch si le drawdown depuis le pic dans la
        fenêtre dépasse le seuil configuré.
        """
        if self._global_kill_switch:
            return
        if not self._capital_snapshots:
            return

        # Grace period : pas de check pendant 1h après la fin du warm-up
        # Les candles historiques post-warmup arrivent en rafale et faussent
        # le drawdown calculé.
        if self._warmup_ended_at is None:
            # Récupérer le dernier warm-up terminé parmi les grid runners
            for r in self._runners:
                ended = getattr(r, "_warmup_ended_at", None)
                if isinstance(ended, datetime):
                    if self._warmup_ended_at is None or ended > self._warmup_ended_at:
                        self._warmup_ended_at = ended
        if self._warmup_ended_at is not None:
            elapsed = (datetime.now(tz=timezone.utc) - self._warmup_ended_at).total_seconds()
            if elapsed < 3600:
                return  # Grace period 1h après warm-up

        # Guard MagicMock-safe (tests avec config=MagicMock())
        window_hours = self._config.risk.kill_switch.global_window_hours
        threshold_pct = self._config.risk.kill_switch.global_max_loss_pct
        if not isinstance(window_hours, (int, float)) or not isinstance(
            threshold_pct, (int, float)
        ):
            return

        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=window_hours)

        # Capital max dans la fenêtre
        in_window = [cap for ts, cap in self._capital_snapshots if ts >= cutoff]
        if not in_window:
            return
        capital_max = max(in_window)
        if capital_max <= 0:
            return

        # Capital actuel
        current_capital = sum(r._capital for r in self._runners)
        drawdown_pct = (capital_max - current_capital) / capital_max * 100

        if drawdown_pct >= threshold_pct:
            self._global_kill_switch = True
            self._kill_switch_reason = {
                "triggered_at": datetime.now(tz=timezone.utc).isoformat(),
                "drawdown_pct": round(drawdown_pct, 2),
                "window_hours": window_hours,
                "threshold_pct": threshold_pct,
                "capital_max": round(capital_max, 2),
                "capital_current": round(current_capital, 2),
            }
            self._stop_all_runners()
            logger.critical(
                "KILL SWITCH GLOBAL : drawdown {:.1f}% sur {}h >= {:.0f}%",
                drawdown_pct, window_hours, threshold_pct,
            )
            if self._notifier:
                try:
                    from backend.alerts.notifier import AnomalyType
                    await self._notifier.notify_anomaly(
                        AnomalyType.KILL_SWITCH_GLOBAL,
                        f"Drawdown {drawdown_pct:.1f}% sur {window_hours}h "
                        f"(seuil {threshold_pct:.0f}%)",
                    )
                except Exception as e:
                    logger.error("Erreur notification kill switch global: {}", e)

    def _stop_all_runners(self) -> None:
        """Coupe tous les runners (kill switch global)."""
        for runner in self._runners:
            runner._kill_switch_triggered = True
            runner._stats.is_active = False

    def reset_kill_switch(self) -> int:
        """Reset le kill switch global et réactive tous les runners.

        Retourne le nombre de runners réactivés.
        """
        self._global_kill_switch = False
        self._kill_switch_reason = None
        reactivated = 0
        for runner in self._runners:
            if runner._kill_switch_triggered:
                runner._kill_switch_triggered = False
                runner._stats.is_active = True
                reactivated += 1
        logger.critical(
            "KILL SWITCH GLOBAL RESET — {} runners réactivés", reactivated,
        )
        return reactivated

    async def periodic_check(self) -> None:
        """Snapshot capital + check global kill switch (appelé toutes les 60s)."""
        self._snapshot_capital()
        await self._check_global_kill_switch()

    # ─── Journal (Sprint 25) ─────────────────────────────────────────────

    async def take_journal_snapshot(self) -> dict | None:
        """Prend un snapshot du portfolio pour le journal d'activité.

        Retourne le snapshot dict ou None si aucun runner actif.
        Appelé toutes les 5 minutes par le StateManager.
        """
        if not self._runners:
            return None

        now = datetime.now(tz=timezone.utc)
        total_capital = 0.0
        total_realized = 0.0
        total_unrealized = 0.0
        total_margin = 0.0
        n_positions = 0
        breakdown: dict[str, dict] = {}

        assets_set: set[str] = set()
        for runner in self._runners:
            status = runner.get_status()
            total_capital += status.get("capital", 0.0)
            total_realized += status.get("net_pnl", 0.0)
            n_pos = status.get("open_positions", 0)
            n_positions += n_pos

            # Breakdown par symbol (grid runners seulement)
            if isinstance(runner, GridStrategyRunner) and n_pos > 0:
                for symbol, positions in runner._positions.items():
                    if not positions:
                        continue
                    assets_set.add(symbol)
                    # Utiliser DataEngine pour prix courant (correct même après kill switch)
                    last_price = self._get_current_price(symbol) or 0.0
                    upnl = runner._gpm.unrealized_pnl(positions, last_price)
                    margin = sum(
                        p.entry_price * p.quantity / runner._leverage
                        for p in positions
                    )
                    total_unrealized += upnl
                    total_margin += margin
                    breakdown[symbol] = {
                        "strategy": runner.name,
                        "positions": len(positions),
                        "unrealized": round(upnl, 2),
                        "margin": round(margin, 2),
                        "last_price": round(last_price, 2),
                    }
            else:
                # Mono runner : unrealized/margin depuis get_status()
                total_unrealized += status.get("unrealized_pnl", 0.0)
                total_margin += status.get("margin_used", 0.0)

        equity = total_capital + total_unrealized
        initial = sum(r._initial_capital for r in self._runners)
        margin_ratio = total_margin / initial if initial > 0 else 0.0

        return {
            "timestamp": now.isoformat(),
            "equity": round(equity, 2),
            "capital": round(total_capital, 2),
            "margin_used": round(total_margin, 2),
            "margin_ratio": round(margin_ratio, 4),
            "realized_pnl": round(total_realized, 2),
            "unrealized_pnl": round(total_unrealized, 2),
            "n_positions": n_positions,
            "n_assets": len(assets_set),
            "breakdown": breakdown if breakdown else None,
        }

    # ─── Orphan cleanup ─────────────────────────────────────────────────

    def _cleanup_orphan_runners(
        self,
        saved_state: dict,
        enabled_names: set[str],
    ) -> None:
        """Nettoie les runners orphelins (stratégie désactivée avec positions).

        Pour chaque runner dans le saved_state absent de enabled_names :
        - Paper : log WARNING + enregistre OrphanClosure (fee-only loss)
        - Live  : log CRITICAL (position peut-être encore ouverte sur Bitget)
        """
        runners_state = saved_state.get("runners", {})
        is_live = getattr(
            getattr(self._config, "secrets", None), "live_trading", False,
        )
        taker_rate = self._config.risk.fees.taker_percent / 100

        for runner_name, state in runners_state.items():
            if runner_name in enabled_names:
                continue

            mono_pos = state.get("position")
            grid_positions = state.get("grid_positions", [])

            if mono_pos is None and not grid_positions:
                logger.info(
                    "Orphan runner '{}' désactivé sans positions — ignoré",
                    runner_name,
                )
                continue

            # ── Live : warning critique ──────────────────────────────
            if is_live:
                if mono_pos is not None:
                    logger.critical(
                        "ORPHAN LIVE : runner '{}' désactivé avec position {} "
                        "@ {:.2f} sur {} — VÉRIFIER BITGET MANUELLEMENT",
                        runner_name,
                        mono_pos["direction"],
                        mono_pos["entry_price"],
                        state.get("position_symbol", "UNKNOWN"),
                    )
                for gp in grid_positions:
                    logger.critical(
                        "ORPHAN LIVE : runner '{}' désactivé avec grid level {} "
                        "{} @ {:.2f} sur {} — VÉRIFIER BITGET MANUELLEMENT",
                        runner_name,
                        gp["level"],
                        gp["direction"],
                        gp["entry_price"],
                        gp.get("symbol", "UNKNOWN"),
                    )

            # ── Paper : enregistrer les closures ─────────────────────
            if mono_pos is not None:
                qty = mono_pos["quantity"]
                entry = mono_pos["entry_price"]
                symbol = state.get("position_symbol", "UNKNOWN")
                fee = mono_pos.get("entry_fee", 0.0) + qty * entry * taker_rate

                self._orphan_closures.append(OrphanClosure(
                    strategy_name=runner_name,
                    symbol=symbol,
                    direction=mono_pos["direction"],
                    entry_price=entry,
                    quantity=qty,
                    estimated_fee_cost=fee,
                    reason="strategy_disabled",
                ))
                logger.warning(
                    "ORPHAN PAPER : runner '{}' désactivé — position {} {} "
                    "@ {:.2f} sur {} fermée (fees≈{:.2f}$)",
                    runner_name, mono_pos["direction"], symbol,
                    entry, symbol, fee,
                )

            for gp in grid_positions:
                qty = gp["quantity"]
                entry = gp["entry_price"]
                symbol = gp.get("symbol", "UNKNOWN")
                fee = gp.get("entry_fee", 0.0) + qty * entry * taker_rate

                self._orphan_closures.append(OrphanClosure(
                    strategy_name=runner_name,
                    symbol=symbol,
                    direction=gp["direction"],
                    entry_price=entry,
                    quantity=qty,
                    estimated_fee_cost=fee,
                    reason="strategy_disabled",
                ))
                logger.warning(
                    "ORPHAN PAPER : runner '{}' désactivé — grid level {} {} "
                    "@ {:.2f} sur {} fermée (fees≈{:.2f}$)",
                    runner_name, gp["level"], gp["direction"],
                    entry, symbol, fee,
                )

    # ─── Position symbols helper ────────────────────────────────────────

    def _get_position_symbols(
        self, runner: LiveStrategyRunner | GridStrategyRunner,
    ) -> set[str]:
        """Retourne les symbols sur lesquels ce runner a des positions."""
        if isinstance(runner, GridStrategyRunner):
            return {s for s, p in runner._positions.items() if p}
        if runner._position is not None and runner._position_symbol:
            return {runner._position_symbol}
        return set()

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

        # Extraire db_path pour les runners
        db_path = self._db.db_path if self._db else None

        # Créer un runner par stratégie (grid ou mono-position)
        for strategy in strategies:
            if is_grid_strategy(strategy.name):
                gpm_config = PositionManagerConfig(
                    leverage=getattr(strategy._config, "leverage", 15),
                    maker_fee=self._config.risk.fees.maker_percent / 100,
                    taker_fee=self._config.risk.fees.taker_percent / 100,
                    slippage_pct=self._config.risk.slippage.default_estimate_percent / 100,
                    high_vol_slippage_mult=self._config.risk.slippage.high_volatility_multiplier,
                    max_risk_per_trade=self._config.risk.position.max_risk_per_trade_percent / 100,
                )
                gpm = GridPositionManager(gpm_config)
                runner: LiveStrategyRunner | GridStrategyRunner = GridStrategyRunner(
                    strategy=strategy,
                    config=self._config,
                    indicator_engine=self._indicator_engine,
                    grid_position_manager=gpm,
                    data_engine=self._data_engine,
                    db_path=db_path,
                )
            else:
                runner = LiveStrategyRunner(
                    strategy=strategy,
                    config=self._config,
                    indicator_engine=self._indicator_engine,
                    position_manager=pm,
                    data_engine=self._data_engine,
                    db_path=db_path,
                )
            self._runners.append(runner)
            logger.info(
                "Simulator: stratégie '{}' ajoutée ({})",
                strategy.name,
                "grid" if is_grid_strategy(strategy.name) else "mono",
            )

        # Cleanup orphans (stratégies désactivées avec positions)
        if saved_state is not None:
            enabled_names = {r.name for r in self._runners}
            self._cleanup_orphan_runners(saved_state, enabled_names)

        # Restaurer l'état AVANT d'enregistrer le callback on_candle
        if saved_state is not None:
            runners_state = saved_state.get("runners", {})
            for runner in self._runners:
                if runner.name in runners_state:
                    runner.restore_state(runners_state[runner.name])

        # Warm-up grid runners depuis la DB
        if self._db is not None:
            symbols = self._data_engine.get_all_symbols()
            logger.debug(
                "Simulator: warm-up {} grid runners × {} symbols",
                sum(1 for r in self._runners if isinstance(r, GridStrategyRunner)),
                len(symbols),
            )
            for runner in self._runners:
                if isinstance(runner, GridStrategyRunner):
                    for symbol in symbols:
                        await runner._warmup_from_db(self._db, symbol)
            logger.info("Simulator: warm-up terminé")

        # Sprint 27 : Charger les profils régime WFO pour les grid runners
        if self._db is not None:
            for runner in self._runners:
                if isinstance(runner, GridStrategyRunner):
                    try:
                        profiles = await self._db.get_regime_profiles(
                            runner.name,
                        )
                        if profiles:
                            runner._regime_profile = profiles
                            logger.info(
                                "Simulator: regime profiles chargés pour "
                                "'{}' ({} assets)",
                                runner.name, len(profiles),
                            )
                    except Exception as e:
                        logger.warning(
                            "Simulator: erreur chargement regime profiles "
                            "'{}': {}",
                            runner.name, e,
                        )

        # Restaurer le kill switch global
        if saved_state is not None:
            self._global_kill_switch = saved_state.get(
                "global_kill_switch", False,
            )
            self._kill_switch_reason = saved_state.get(
                "kill_switch_reason", None,
            )
            if self._global_kill_switch:
                # Forcer la fin du warm-up D'ABORD pour que le state sauvegardé
                # soit appliqué (sinon StateManager écrase avec capital=10000).
                # _stop_all_runners() est appelé APRÈS pour avoir le dernier mot
                # sur kill_switch_triggered (car _apply_restored_state le reset).
                for runner in self._runners:
                    if hasattr(runner, '_is_warming_up') and runner._is_warming_up:
                        runner._end_warmup()
                        logger.info(
                            "[{}] Warm-up forcé terminé (kill switch global restauré)",
                            runner.name,
                        )
                self._stop_all_runners()

                # Compter les positions pour le log
                total_positions = 0
                for runner in self._runners:
                    if hasattr(runner, '_positions') and isinstance(runner._positions, dict):
                        total_positions += sum(len(p) for p in runner._positions.values())
                    elif hasattr(runner, '_position') and runner._position is not None:
                        total_positions += 1

                reason_str = ""
                if self._kill_switch_reason:
                    reason_str = (
                        f" (drawdown {self._kill_switch_reason.get('drawdown_pct', '?')}%"
                        f" sur {self._kill_switch_reason.get('window_hours', '?')}h,"
                        f" déclenché à {self._kill_switch_reason.get('triggered_at', '?')})"
                    )
                logger.critical(
                    "KILL SWITCH GLOBAL restauré{} — {} runners stoppés, {} positions",
                    reason_str, len(self._runners), total_positions,
                )

                # Notification Telegram
                if self._notifier:
                    try:
                        from backend.alerts.notifier import AnomalyType
                        await self._notifier.notify_anomaly(
                            AnomalyType.KILL_SWITCH_GLOBAL,
                            f"Kill switch restauré depuis le state{reason_str}. "
                            f"{len(self._runners)} runners stoppés, {total_positions} positions. "
                            f"Reset via POST /api/simulator/kill-switch/reset",
                        )
                    except Exception as e:
                        logger.error("Erreur notification kill switch restore: {}", e)

        # Snapshot initial du capital (référence pour la fenêtre glissante)
        self._snapshot_capital()

        # Câblage DataEngine → Simulator (APRÈS restauration + warm-up)
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

        # TOUJOURS mettre à jour les indicateurs (même si kill switch actif).
        # L'Executor live lit ces indicateurs via get_runner_context().
        self._indicator_engine.update(symbol, timeframe, candle)

        # Invalider le cache conditions à chaque bougie (indicateurs changent)
        self._conditions_cache = None

        # Kill switch global : indicateurs à jour, mais pas de trades paper.
        # On met à jour les close_buffers des runners pour que build_context()
        # retourne des SMA fraîches à l'Executor.
        if self._global_kill_switch:
            for runner in self._runners:
                if hasattr(runner, "update_indicators_only"):
                    runner.update_indicators_only(symbol, timeframe, candle)
            return

        # Snapshot positions AVANT dispatch (détection collision)
        positions_before: dict[str, set[str]] = {
            r.name: self._get_position_symbols(r) for r in self._runners
        }

        # Compteur trades pour détecter les fermetures
        trades_before = sum(r._stats.total_trades for r in self._runners)

        # Dispatcher à chaque runner
        for runner in self._runners:
            try:
                await runner.on_candle(symbol, timeframe, candle)
            except Exception as e:
                logger.error(
                    "Simulator: erreur runner '{}': {}",
                    runner.name, e,
                )

            # Sprint 36a : alerte Telegram si circuit breaker vient de s'ouvrir
            if getattr(runner, '_circuit_breaker_open', False) and self._notifier:
                try:
                    from backend.alerts.notifier import AnomalyType
                    await self._notifier.notify_anomaly(
                        AnomalyType.CIRCUIT_BREAKER,
                        f"[{runner.name}] CIRCUIT BREAKER — runner désactivé après "
                        f"{runner._CIRCUIT_BREAKER_MAX_CRASHES} crashes",
                    )
                except Exception:
                    pass

            # Sprint 34a : forward erreurs compute_live_indicators → Telegram
            alert = getattr(runner, '_last_indicator_error', None)
            if alert and self._notifier:
                err_symbol, err_msg = alert
                runner._last_indicator_error = None
                try:
                    from backend.alerts.notifier import AnomalyType
                    await self._notifier.notify_anomaly(
                        AnomalyType.INDICATOR_ERROR,
                        f"[{runner.name}] {err_symbol}: {err_msg}",
                    )
                except Exception:
                    pass  # Telegram down ne doit pas bloquer le dispatch

            # Sprint 25 : drain journal events vers la DB
            if self._db and hasattr(runner, "_pending_journal_events") and runner._pending_journal_events:
                journal_events, runner._pending_journal_events = runner._pending_journal_events, []
                for i, event in enumerate(journal_events):
                    try:
                        await self._db.insert_position_event(event)
                        if len(journal_events) > 2 and i < len(journal_events) - 1:
                            await asyncio.sleep(0.05)
                    except Exception as e:
                        logger.warning("Journal: erreur insert event: {}", e)

            # Détection collision : runner vient d'ouvrir sur un symbol déjà pris
            new_symbols = (
                self._get_position_symbols(runner)
                - positions_before.get(runner.name, set())
            )
            for new_sym in new_symbols:
                for other in self._runners:
                    if other.name == runner.name:
                        continue
                    if new_sym in self._get_position_symbols(other):
                        self._collision_warnings.append({
                            "timestamp": candle.timestamp.isoformat(),
                            "symbol": new_sym,
                            "runner_opening": runner.name,
                            "runner_existing": other.name,
                        })
                        logger.warning(
                            "COLLISION : '{}' ouvre sur {} alors que "
                            "'{}' a déjà une position",
                            runner.name, new_sym, other.name,
                        )

        # Kill switch global : snapshot + check si un trade a été enregistré
        trades_after = sum(r._stats.total_trades for r in self._runners)
        if trades_after > trades_before:
            self._snapshot_capital()
            await self._check_global_kill_switch()

    async def stop(self) -> None:
        """Arrête le simulateur."""
        self._running = False
        logger.info("Simulator: arrêté")

    def get_all_status(self) -> dict[str, dict]:
        """Retourne le status de tous les runners."""
        statuses = {runner.name: runner.get_status() for runner in self._runners}
        # Ajouter les collision warnings par runner
        for warning in self._collision_warnings:
            name = warning["runner_opening"]
            if name in statuses:
                statuses[name].setdefault("collision_warnings", []).append(
                    warning,
                )
        return statuses

    def get_all_trades(self) -> list[dict]:
        """Retourne tous les trades de tous les runners."""
        all_trades = []
        for runner in self._runners:
            for symbol, trade in runner.get_trades():
                all_trades.append({
                    "strategy": runner.name,
                    "symbol": symbol,
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

    def get_open_positions(self) -> list[dict]:
        """Retourne les positions ouvertes de tous les runners."""
        positions = []
        for runner in self._runners:
            # Positions grid (GridStrategyRunner)
            if hasattr(runner, "get_grid_positions"):
                positions.extend(runner.get_grid_positions())
            # Position mono-position (LiveStrategyRunner)
            elif runner._position is not None and runner._position_symbol:
                pos = runner._position
                positions.append({
                    "symbol": runner._position_symbol,
                    "strategy": runner.name,
                    "direction": pos.direction.value,
                    "entry_price": pos.entry_price,
                    "quantity": pos.quantity,
                    "tp_price": pos.tp_price,
                    "sl_price": pos.sl_price,
                    "entry_time": pos.entry_time.isoformat(),
                })
        return positions

    def get_grid_state(self) -> dict:
        """État détaillé des grilles DCA actives avec P&L non réalisé."""
        grids: list[dict] = []

        for runner in self._runners:
            if not hasattr(runner, "_gpm"):
                continue  # Pas un GridStrategyRunner

            for symbol, positions in runner._positions.items():
                if not positions:
                    continue

                # Prix courant depuis DataEngine — fallback 1m → 5m → 1h
                current_price = self._get_current_price(symbol)
                if current_price is None or current_price <= 0:
                    continue

                # GridState agrégé via GPM
                grid_state = runner._gpm.compute_grid_state(positions, current_price)

                # Indicateurs pour TP/SL dynamique
                ctx = runner.build_context(symbol)
                main_tf = getattr(runner._strategy._config, "timeframe", "1h")
                main_ind = (
                    ctx.indicators.get(main_tf, {})
                    if ctx and ctx.indicators
                    else {}
                )

                tp_price = runner._strategy.get_tp_price(grid_state, main_ind)
                sl_price = runner._strategy.get_sl_price(grid_state, main_ind)

                leverage = runner._leverage
                margin_used = (
                    grid_state.total_notional / leverage if leverage > 0 else 0.0
                )
                unrealized_pnl_pct = (
                    grid_state.unrealized_pnl / margin_used * 100
                    if margin_used > 0
                    else 0.0
                )

                tp_valid = not math.isnan(tp_price) if tp_price is not None else False
                sl_valid = not math.isnan(sl_price) if sl_price is not None else False

                grids.append({
                    "symbol": symbol,
                    "strategy": runner.name,
                    "direction": positions[0].direction.value,
                    "levels_open": len(positions),
                    "levels_max": runner._get_num_levels(symbol),
                    "avg_entry": round(grid_state.avg_entry_price, 6),
                    "current_price": current_price,
                    "unrealized_pnl": round(grid_state.unrealized_pnl, 2),
                    "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
                    "tp_price": round(tp_price, 6) if tp_valid else None,
                    "sl_price": round(sl_price, 6) if sl_valid else None,
                    "tp_distance_pct": (
                        round((tp_price - current_price) / current_price * 100, 2)
                        if tp_valid
                        else None
                    ),
                    "sl_distance_pct": (
                        round((sl_price - current_price) / current_price * 100, 2)
                        if sl_valid
                        else None
                    ),
                    "margin_used": round(margin_used, 2),
                    "leverage": leverage,
                    "positions": [
                        {
                            "level": p.level,
                            "entry_price": p.entry_price,
                            "quantity": p.quantity,
                            "entry_time": p.entry_time.isoformat(),
                            "direction": p.direction.value,
                        }
                        for p in positions
                    ],
                })

        total_margin = sum(g["margin_used"] for g in grids)
        total_upnl = sum(g["unrealized_pnl"] for g in grids)

        # Capital disponible : somme du capital des runners grid
        capital_available = 0.0
        for runner in self._runners:
            if hasattr(runner, "_gpm"):
                capital_available += runner._capital

        return {
            "grid_positions": {f'{g["strategy"]}:{g["symbol"]}': g for g in grids},
            "summary": {
                "total_positions": sum(g["levels_open"] for g in grids),
                "total_assets": len(grids),
                "total_margin_used": round(total_margin, 2),
                "total_unrealized_pnl": round(total_upnl, 2),
                "capital_available": round(capital_available, 2),
            },
        }

    def _get_current_price(self, symbol: str) -> float | None:
        """Prix courant depuis DataEngine avec fallback multi-timeframe."""
        data = self._data_engine.get_data(symbol)
        for tf in ("1m", "5m", "1h"):
            candles = data.candles.get(tf, [])
            if candles and candles[-1].close > 0:
                return candles[-1].close
        return None

    def is_kill_switch_triggered(self) -> bool:
        """Vérifie si le kill switch global ou un runner a déclenché le kill switch."""
        if self._global_kill_switch:
            return True
        return any(r.is_kill_switch_triggered for r in self._runners)

    @property
    def kill_switch_reason(self) -> dict | None:
        """Raison du kill switch global (None si pas déclenché)."""
        return self._kill_switch_reason

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
                "sparkline": [],
            }

            # Prix courant avec fallback multi-timeframe (1m → 5m → 1h → 4h)
            data = self._data_engine.get_data(symbol)
            price_candles = None
            for _tf in ("1m", "5m", "1h", "4h"):
                c = data.candles.get(_tf, [])
                if c:
                    price_candles = c
                    break

            if price_candles:
                last_candle = price_candles[-1]
                asset_data["price"] = last_candle.close
                # Sparkline : 60 derniers close prices (timeframe disponible)
                asset_data["sparkline"] = [
                    c.close for c in price_candles[-60:]
                ]
                if len(price_candles) >= 2:
                    prev_close = price_candles[-2].close
                    if prev_close > 0:
                        asset_data["change_pct"] = round(
                            (last_candle.close - prev_close) / prev_close * 100, 2
                        )

            # Indicateurs de base avec fallback multi-timeframe (5m → 1h)
            if self._indicator_engine:
                ind_tf = None
                indicators_data = self._indicator_engine.get_indicators(symbol)
                for _tf in ("5m", "1h"):
                    if data.candles.get(_tf) and indicators_data and indicators_data.get(_tf):
                        ind_tf = _tf
                        break

                if ind_tf:
                    ind = indicators_data[ind_tf]
                    close = ind.get("close")
                    vwap = ind.get("vwap")
                    atr = ind.get("atr")

                    asset_data["indicators"] = {
                        "rsi_14": _safe_round(ind.get("rsi"), 1),
                        "vwap_distance_pct": None,
                        "adx": _safe_round(ind.get("adx"), 1),
                        "atr_pct": None,
                    }

                    # VWAP distance
                    if close and vwap and vwap > 0:
                        asset_data["indicators"]["vwap_distance_pct"] = round(
                            (close - vwap) / vwap * 100, 2
                        )

                    # ATR %
                    if close and atr and close > 0:
                        asset_data["indicators"]["atr_pct"] = round(
                            atr / close * 100, 2
                        )

                    # Régime de marché par défaut (depuis indicateurs scalaires)
                    adx_val = ind.get("adx")
                    di_plus_val = ind.get("di_plus")
                    di_minus_val = ind.get("di_minus")
                    atr_val = atr
                    atr_sma_val = ind.get("atr_sma")
                    if all(v is not None for v in (adx_val, di_plus_val, di_minus_val, atr_val, atr_sma_val)):
                        asset_data["regime"] = detect_market_regime(
                            adx_val, di_plus_val, di_minus_val, atr_val, atr_sma_val
                        ).value

            # Conditions par runner/stratégie (enrichissement si runner actif)
            for runner in self._runners:
                ctx = runner.build_context(symbol)
                if ctx is None:
                    continue

                # Régime (du runner, mis à jour à chaque candle) — priorité au runner actif
                asset_data["regime"] = runner.current_regime.value

                # Indicateurs : si déjà calculés globalement, les enrichir avec les données du runner si différent TF
                # Sinon, calculer depuis le context du runner
                if not asset_data["indicators"]:
                    main_tf = list(runner.strategy.min_candles.keys())[0]
                    main_ind = ctx.indicators.get(main_tf, {})
                    if main_ind:
                        close = main_ind.get("close")
                        vwap = main_ind.get("vwap")
                        atr_val = main_ind.get("atr")

                        asset_data["indicators"] = {
                            "rsi_14": _safe_round(main_ind.get("rsi"), 1),
                            "vwap_distance_pct": None,
                            "adx": _safe_round(main_ind.get("adx"), 1),
                            "atr_pct": None,
                        }

                        # VWAP distance
                        if close and vwap and vwap > 0:
                            asset_data["indicators"]["vwap_distance_pct"] = round(
                                (close - vwap) / vwap * 100, 2
                            )

                        # ATR %
                        if close and atr_val and close > 0:
                            asset_data["indicators"]["atr_pct"] = round(
                                atr_val / close * 100, 2
                            )

                # Conditions de la stratégie (uniquement pour les stratégies mono, pas grid)
                # Les stratégies grid ont leur propre section dans grid_state
                if not isinstance(runner, GridStrategyRunner):
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

                # Position ouverte sur cet asset (vérifier le symbol)
                if runner._position is not None and runner._position_symbol == symbol:
                    asset_data["position"] = {
                        "direction": runner._position.direction.value,
                        "entry_price": runner._position.entry_price,
                        "tp_price": runner._position.tp_price,
                        "sl_price": runner._position.sl_price,
                        "strategy": runner.name,
                        "entry_time": runner._position.entry_time.isoformat(),
                    }

            assets[symbol] = asset_data

        result = {
            "assets": assets,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._conditions_cache = result
        return result

    def get_signal_matrix(self) -> dict:
        """Matrice conditions met/total pour la Heatmap : ratio par (strategy, asset)."""
        conditions_data = self.get_conditions()
        assets = conditions_data.get("assets", {})
        matrix: dict[str, dict[str, float | None]] = {}

        for symbol, asset_data in assets.items():
            matrix[symbol] = {}
            strategies = asset_data.get("strategies", {})
            for strat_name, strat_data in strategies.items():
                conditions = strat_data.get("conditions", [])
                if conditions:
                    met = sum(1 for c in conditions if c.get("met"))
                    total = len(conditions)
                    matrix[symbol][strat_name] = met / total if total > 0 else 0.0
                else:
                    matrix[symbol][strat_name] = None

        return {"matrix": matrix}

    def get_equity_curve(self, since: str | None = None, strategy: str | None = None) -> dict:
        """Courbe d'equity depuis les trades.

        strategy : si spécifié, filtre sur un seul runner. Cache non utilisé dans ce cas.
        """
        default_capital = self._config.risk.initial_capital

        # Sélectionner les runners concernés
        runners = [r for r in self._runners if r.name == strategy] if strategy else self._runners

        if strategy:
            # Calcul direct sans cache (requête filtrée — rare, pas besoin d'optimiser)
            all_trades: list[TradeResult] = []
            for runner in runners:
                for _sym, trade in runner.get_trades():
                    all_trades.append(trade)
            all_trades.sort(key=lambda t: t.exit_time)

            capital = sum(r._initial_capital for r in runners) if runners else default_capital
            equity: list[dict] = []
            for trade in all_trades:
                capital += trade.net_pnl
                equity.append({
                    "timestamp": trade.exit_time.isoformat(),
                    "capital": round(capital, 2),
                    "trade_pnl": round(trade.net_pnl, 2),
                })
        else:
            # Calcul global avec cache
            total_trades = sum(len(r.get_trades()) for r in self._runners)
            if self._equity_cache is not None and self._trade_count_at_cache == total_trades:
                equity = self._equity_cache
            else:
                all_trades = []
                for runner in self._runners:
                    for _sym, trade in runner.get_trades():
                        all_trades.append(trade)
                all_trades.sort(key=lambda t: t.exit_time)

                capital = self._config.risk.initial_capital
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

        # Métriques courantes sur les runners sélectionnés
        current_capital = sum(r._capital for r in runners) if runners else default_capital
        initial_capital = sum(r._initial_capital for r in runners) if runners else default_capital
        current_equity = sum(
            r.get_status().get("equity", r._capital) for r in runners
        )

        # Point "now" avec l'equity courante (inclut non réalisé)
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        equity_with_now = list(equity)
        equity_with_now.append({
            "timestamp": now_iso,
            "capital": round(current_equity, 2),
            "trade_pnl": 0,
        })

        return {
            "equity": equity_with_now,
            "current_capital": round(current_capital, 2),
            "current_equity": round(current_equity, 2),
            "initial_capital": round(initial_capital, 2),
        }

    def get_strategy_instances(self) -> dict[str, BaseGridStrategy]:
        """Retourne les instances de stratégie grid par nom (pour l'Executor autonome)."""
        return {
            runner.name: runner._strategy
            for runner in self._runners
            if isinstance(runner, GridStrategyRunner)
        }

    def get_runner_context(
        self, strategy_name: str, symbol: str,
    ) -> StrategyContext | None:
        """Retourne le StrategyContext du runner paper (source unique de vérité pour l'exit monitor)."""
        for runner in self._runners:
            if runner.name == strategy_name:
                return runner.build_context(symbol)
        return None

    def force_close_grid(
        self,
        strategy_name: str,
        symbol: str,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        """Ferme une position grid paper au même prix que le live (sync live→paper)."""
        for runner in self._runners:
            if runner.name != strategy_name:
                continue
            if not hasattr(runner, "_positions"):
                return
            positions = runner._positions.get(symbol, [])
            if not positions:
                return
            total_notional = sum(
                p.entry_price * p.quantity for p in positions
            )
            margin_to_return = total_notional / runner._leverage
            trade = runner._gpm.close_all_positions(
                positions,
                exit_price,
                datetime.now(tz=timezone.utc),
                exit_reason,
                "unknown",
            )
            runner._capital += trade.net_pnl + margin_to_return
            runner._realized_pnl += trade.net_pnl
            runner._positions[symbol] = []
            runner._record_close(symbol, trade.exit_time)
            runner._stats.total_trades += 1
            runner._trades.append((symbol, trade))
            if trade.net_pnl >= 0:
                runner._stats.wins += 1
            else:
                runner._stats.losses += 1
            runner._stats.net_pnl = runner._capital - runner._initial_capital
            runner._stats.capital = runner._capital
            logger.info(
                "[{}] SYNC CLOSE {} — {} niveaux, exit={:.6f}, net={:+.2f} ({})",
                strategy_name,
                symbol,
                len(positions),
                exit_price,
                trade.net_pnl,
                exit_reason,
            )
            return

    @property
    def runners(self) -> list[LiveStrategyRunner | GridStrategyRunner]:
        return self._runners

    @property
    def orphan_closures(self) -> list[OrphanClosure]:
        """Positions orphelines fermées au boot (stratégies désactivées)."""
        return list(self._orphan_closures)

    @property
    def collision_warnings(self) -> list[dict]:
        """Avertissements de collision inter-runners (même symbol)."""
        return list(self._collision_warnings)
