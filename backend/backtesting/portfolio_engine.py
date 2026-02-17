"""Portfolio Backtest Engine — simulation multi-asset à capital partagé.

Réutilise GridStrategyRunner (le même code qu'en production) pour simuler
N assets simultanément avec un pool de capital unique.

Réponses aux 5 questions clés :
1. Max drawdown historique sur le portfolio (capital partagé)
2. Corrélation des fills — combien de grilles se remplissent simultanément
3. Margin peak — marge max mobilisée simultanément (% du capital)
4. Kill switch frequency — combien de fois le kill switch aurait déclenché
5. Sizing optimal — pour 1k/5k/10k$, combien d'assets max
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import groupby
from typing import Any, Callable

import numpy as np
from loguru import logger

from backend.backtesting.simulator import GridStrategyRunner, RunnerStats
from backend.core.config import AppConfig
from backend.core.database import Database
from backend.core.grid_position_manager import GridPositionManager
from backend.core.incremental_indicators import IncrementalIndicatorEngine
from backend.core.models import Candle, MarketRegime
from backend.core.position_manager import PositionManagerConfig, TradeResult
from backend.optimization import create_strategy_with_params


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PortfolioSnapshot:
    """Snapshot de l'état du portfolio à un instant t."""

    timestamp: datetime
    total_equity: float  # capital + unrealized
    total_capital: float  # cash disponible (après marge)
    total_realized_pnl: float
    total_unrealized_pnl: float
    total_margin_used: float
    margin_ratio: float  # margin / initial_capital
    n_open_positions: int
    n_assets_with_positions: int


@dataclass
class PortfolioResult:
    """Résultat complet du backtest portfolio."""

    # Config
    initial_capital: float
    n_assets: int
    period_days: int
    assets: list[str]

    # Aggregate
    final_equity: float
    total_return_pct: float
    total_trades: int
    win_rate: float

    # P&L séparé (realized = TP/SL naturels, force_closed = fin de données)
    realized_pnl: float
    force_closed_pnl: float

    # Risk
    max_drawdown_pct: float
    max_drawdown_date: datetime | None
    max_drawdown_duration_hours: float
    peak_margin_ratio: float
    peak_open_positions: int
    peak_concurrent_assets: int

    # Kill switch
    kill_switch_triggers: int
    kill_switch_events: list[dict]

    # Equity curve
    snapshots: list[PortfolioSnapshot]

    # Per-asset breakdown
    per_asset_results: dict[str, dict]

    # Trades
    all_trades: list[tuple[str, TradeResult]] = field(default_factory=list)

    # Funding costs (agrégé de tous les runners, 0.0 si non calculé)
    funding_paid_total: float = 0.0


# ---------------------------------------------------------------------------
# Portfolio Backtester
# ---------------------------------------------------------------------------


class PortfolioBacktester:
    """Orchestre N GridStrategyRunners avec capital partagé.

    Chaque runner gère 1 seul asset avec les params WFO per_asset.
    Le capital est divisé également entre les runners.

    NOTE — limitation connue (ATR period) :
    L'IncrementalIndicatorEngine calcule l'ATR avec period=14 fixe.
    Le atr_period per_asset du WFO n'est PAS utilisé dans le path live.
    Ceci est le comportement actuel de la prod — pas une régression.
    Le ma_period est correctement per_asset (le runner calcule sa propre SMA).
    TODO: Sprint futur — injecter atr_period per_asset dans l'indicator engine.
    """

    DEFAULT_WARMUP = 50

    def __init__(
        self,
        config: AppConfig,
        initial_capital: float = 10_000.0,
        strategy_name: str = "grid_atr",
        assets: list[str] | None = None,
        exchange: str = "binance",
        kill_switch_pct: float = 30.0,
        kill_switch_window_hours: int = 24,
        multi_strategies: list[tuple[str, list[str]]] | None = None,
    ) -> None:
        self._config = config
        self._initial_capital = initial_capital
        self._strategy_name = strategy_name
        self._exchange = exchange
        self._kill_switch_pct = kill_switch_pct
        self._kill_switch_window_hours = kill_switch_window_hours

        # Multi-stratégie : liste de (strategy_name, [symbols])
        if multi_strategies:
            self._multi_strategies = multi_strategies
        else:
            # Résoudre les assets pour une seule stratégie (rétro-compatible)
            if assets:
                resolved_assets = assets
            else:
                strat_config = getattr(config.strategies, strategy_name, None)
                per_asset = getattr(strat_config, "per_asset", {}) if strat_config else {}
                resolved_assets = sorted(per_asset.keys()) if per_asset else []
            self._multi_strategies = [(strategy_name, resolved_assets)]

        # Calculer les assets uniques (pour le chargement des candles)
        all_symbols: set[str] = set()
        for _, symbols in self._multi_strategies:
            all_symbols.update(symbols)
        self._assets = sorted(all_symbols)

        self._kill_freeze_until: datetime | None = None  # Sprint 24a

        if not self._assets:
            raise ValueError("Aucun asset sélectionné pour le portfolio backtest")

    @staticmethod
    def _symbol_from_key(runner_key: str) -> str:
        """Extrait le symbol depuis 'strategy:symbol' ou retourne la clé telle quelle."""
        return runner_key.split(":", 1)[1] if ":" in runner_key else runner_key

    async def run(
        self,
        start: datetime,
        end: datetime,
        db_path: str = "data/scalp_radar.db",
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> PortfolioResult:
        """Lance le backtest portfolio complet."""
        # Compter le nombre total de runners (pas de symbols uniques)
        n_total_runners = sum(len(syms) for _, syms in self._multi_strategies)
        per_runner_capital = self._initial_capital / n_total_runners
        strat_names = [s for s, _ in self._multi_strategies]
        logger.info(
            "Portfolio backtest : {} runners ({}), {:.0f}$ total ({:.0f}$/runner), {} → {}",
            n_total_runners,
            "+".join(strat_names),
            self._initial_capital,
            per_runner_capital,
            start.date(),
            end.date(),
        )

        # 1. Charger candles (symbols uniques)
        db = Database(db_path)
        await db.init()
        candles_by_symbol = await self._load_candles(db, start, end)

        # Sprint 27 : Charger les profils régime WFO avant fermeture DB
        regime_profiles_by_strategy: dict[str, dict[str, dict]] = {}
        for strat_name, _ in self._multi_strategies:
            try:
                profiles = await db.get_regime_profiles(strat_name)
                if profiles:
                    regime_profiles_by_strategy[strat_name] = profiles
            except Exception:
                pass

        await db.close()

        if not candles_by_symbol:
            raise ValueError("Aucune candle chargée depuis la DB")

        # Filtrer assets sans données suffisantes
        valid_symbols = {
            s: c
            for s, c in candles_by_symbol.items()
            if len(c) >= self.DEFAULT_WARMUP + 10
        }
        if not valid_symbols:
            raise ValueError(
                f"Aucun asset avec assez de données (min {self.DEFAULT_WARMUP + 10} candles)"
            )
        skipped = set(candles_by_symbol) - set(valid_symbols)
        if skipped:
            logger.warning("Assets ignorés (données insuffisantes) : {}", skipped)

        # Filtrer les multi_strategies pour ne garder que les symbols valides
        filtered_multi = [
            (sname, [s for s in syms if s in valid_symbols])
            for sname, syms in self._multi_strategies
        ]
        filtered_multi = [(s, syms) for s, syms in filtered_multi if syms]
        n_total_runners = sum(len(syms) for _, syms in filtered_multi)
        per_runner_capital = self._initial_capital / n_total_runners

        # 2. Créer les runners
        runners, indicator_engine = self._create_runners(
            filtered_multi, per_runner_capital, regime_profiles_by_strategy
        )

        # 3. Warm-up
        warmup_ends = self._warmup_runners(
            runners, valid_symbols, indicator_engine, self.DEFAULT_WARMUP
        )

        # 4. Merge et simulate
        merged = self._merge_candles(valid_symbols)
        snapshots, realized_trades = await self._simulate(
            runners, indicator_engine, merged, warmup_ends, progress_callback
        )

        # 5. Force-close positions restantes
        force_closed_trades = self._force_close_all(runners)

        # 6. Build result
        period_days = (end - start).days
        runner_keys = list(runners.keys())
        return self._build_result(
            runners,
            snapshots,
            realized_trades,
            force_closed_trades,
            runner_keys,
            period_days,
        )

    # ------------------------------------------------------------------
    # Chargement données
    # ------------------------------------------------------------------

    async def _load_candles(
        self,
        db: Database,
        start: datetime,
        end: datetime,
    ) -> dict[str, list[Candle]]:
        """Charge les candles 1h pour tous les assets."""
        result: dict[str, list[Candle]] = {}
        for symbol in self._assets:
            candles = await db.get_candles(
                symbol, "1h", start, end, limit=1_000_000, exchange=self._exchange
            )
            if candles:
                result[symbol] = candles
                logger.info(
                    "  {} : {} candles ({} → {})",
                    symbol,
                    len(candles),
                    candles[0].timestamp.date(),
                    candles[-1].timestamp.date(),
                )
            else:
                logger.warning("  {} : aucune candle", symbol)
        return result

    # ------------------------------------------------------------------
    # Création des runners
    # ------------------------------------------------------------------

    def _create_runners(
        self,
        multi_strategies: list[tuple[str, list[str]]],
        per_runner_capital: float,
        regime_profiles: dict[str, dict[str, dict]] | None = None,
    ) -> tuple[dict[str, GridStrategyRunner], IncrementalIndicatorEngine]:
        """Crée 1 runner par (stratégie, asset) avec params WFO per_asset.

        Les clés des runners sont au format 'strategy_name:symbol' pour
        supporter plusieurs stratégies sur le même symbol.
        """
        strategies_list: list = []
        runner_entries: list[tuple[str, str, Any]] = []  # (runner_key, symbol, strategy)

        for strat_name, symbols in multi_strategies:
            strat_config = getattr(self._config.strategies, strat_name, None)
            per_asset_overrides = (
                getattr(strat_config, "per_asset", {}) if strat_config else {}
            )

            for symbol in symbols:
                params = per_asset_overrides.get(symbol, {})
                strategy = create_strategy_with_params(strat_name, params)
                strategies_list.append(strategy)
                runner_key = f"{strat_name}:{symbol}"
                runner_entries.append((runner_key, symbol, strategy))

        # Indicator engine partagé
        indicator_engine = IncrementalIndicatorEngine(strategies_list)

        runners: dict[str, GridStrategyRunner] = {}
        for runner_key, symbol, strategy in runner_entries:
            leverage = getattr(strategy._config, "leverage", 15)
            gpm_config = PositionManagerConfig(
                leverage=leverage,
                maker_fee=self._config.risk.fees.maker_percent / 100,
                taker_fee=self._config.risk.fees.taker_percent / 100,
                slippage_pct=self._config.risk.slippage.default_estimate_percent / 100,
                high_vol_slippage_mult=self._config.risk.slippage.high_volatility_multiplier,
                max_risk_per_trade=self._config.risk.position.max_risk_per_trade_percent / 100,
            )
            gpm = GridPositionManager(gpm_config)

            # Sprint 27 : profil régime pour cette stratégie
            strat_name = runner_key.split(":", 1)[0] if ":" in runner_key else runner_key
            strat_profiles = (
                regime_profiles.get(strat_name) if regime_profiles else None
            )

            runner = GridStrategyRunner(
                strategy=strategy,
                config=self._config,
                indicator_engine=indicator_engine,
                grid_position_manager=gpm,
                data_engine=None,  # type: ignore[arg-type]
                db_path=None,
                regime_profile=strat_profiles,
            )

            runner._nb_assets = 1
            runner._capital = per_runner_capital
            runner._initial_capital = per_runner_capital
            runner._portfolio_mode = True
            runner._stats = RunnerStats(
                capital=per_runner_capital,
                initial_capital=per_runner_capital,
            )

            runners[runner_key] = runner

        # Références croisées pour global margin guard
        for runner in runners.values():
            runner._portfolio_runners = runners
            runner._portfolio_initial_capital = self._initial_capital

        logger.info(
            "Créé {} runners (capital={:.0f}$/runner)",
            len(runners),
            per_runner_capital,
        )
        return runners, indicator_engine

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------

    def _warmup_runners(
        self,
        runners: dict[str, GridStrategyRunner],
        candles_by_symbol: dict[str, list[Candle]],
        indicator_engine: IncrementalIndicatorEngine,
        warmup_count: int = 50,
    ) -> dict[str, int]:
        """Alimente les buffers indicateurs, puis désactive le warm-up.

        Retourne l'index de la première candle de simulation par symbol.
        Les candles d'un symbol ne sont injectées dans l'indicator engine
        qu'une seule fois (même si plusieurs runners utilisent ce symbol).
        """
        warmup_ends: dict[str, int] = {}
        symbol_warmed: set[str] = set()  # éviter double injection dans l'engine

        for runner_key, runner in runners.items():
            symbol = self._symbol_from_key(runner_key)
            candles = candles_by_symbol.get(symbol)
            if not candles:
                continue

            n = min(warmup_count, len(candles) - 1)
            ma_period = runner._ma_period

            runner._close_buffer[symbol] = deque(
                maxlen=max(ma_period + 20, 50)
            )

            # Injecter dans l'indicator engine seulement une fois par symbol
            if symbol not in symbol_warmed:
                for i in range(n):
                    indicator_engine.update(symbol, "1h", candles[i])
                symbol_warmed.add(symbol)

            # Alimenter le close_buffer de chaque runner
            for i in range(n):
                runner._close_buffer[symbol].append(candles[i].close)

            # Désactiver warm-up et reset propre
            runner._is_warming_up = False
            per_cap = runner._initial_capital
            runner._capital = per_cap
            runner._realized_pnl = 0.0
            runner._trades = []
            runner._positions = {}
            runner._stats = RunnerStats(capital=per_cap, initial_capital=per_cap)

            # warmup_ends indexé par symbol (partagé entre runners du même symbol)
            warmup_ends[symbol] = n

        logger.info(
            "Warm-up terminé ({} candles/asset), {} runners prêts",
            warmup_count,
            len(runners),
        )
        return warmup_ends

    # ------------------------------------------------------------------
    # Merge & Simulation
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_candles(
        candles_by_symbol: dict[str, list[Candle]],
    ) -> list[Candle]:
        """Merge toutes les candles, tri chronologique stable."""
        all_candles: list[Candle] = []
        for candles in candles_by_symbol.values():
            all_candles.extend(candles)
        all_candles.sort(key=lambda c: (c.timestamp, c.symbol))
        return all_candles

    async def _simulate(
        self,
        runners: dict[str, GridStrategyRunner],
        indicator_engine: IncrementalIndicatorEngine,
        merged_candles: list[Candle],
        warmup_ends: dict[str, int],
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> tuple[list[PortfolioSnapshot], list[tuple[str, TradeResult]]]:
        """Boucle de simulation principale."""
        snapshots: list[PortfolioSnapshot] = []
        all_trades: list[tuple[str, TradeResult]] = []

        # Mapping symbol → [runner_keys] pour dispatcher une candle à tous les runners du symbol
        symbol_to_runners: dict[str, list[str]] = {}
        for key in runners:
            sym = self._symbol_from_key(key)
            symbol_to_runners.setdefault(sym, []).append(key)

        # Compteurs pour savoir quand on dépasse le warmup (indexé par symbol)
        all_symbols = set(symbol_to_runners.keys())
        candle_count_per_symbol: dict[str, int] = {s: 0 for s in all_symbols}
        last_closes: dict[str, float] = {}

        total = len(merged_candles)
        log_interval = max(total // 20, 1)

        for i, candle in enumerate(merged_candles):
            symbol = candle.symbol
            runner_keys = symbol_to_runners.get(symbol)
            if not runner_keys:
                continue

            candle_count_per_symbol[symbol] += 1

            # Skip candles de warmup
            warmup_end = warmup_ends.get(symbol, 0)
            if candle_count_per_symbol[symbol] <= warmup_end:
                continue

            # Mettre à jour l'indicator engine AVANT les runners (une seule fois par symbol)
            indicator_engine.update(symbol, "1h", candle)

            # Dispatcher à TOUS les runners de ce symbol
            for runner_key in runner_keys:
                runner = runners[runner_key]
                trades_before = len(runner._trades)

                await runner.on_candle(symbol, "1h", candle)

                # Collecter les nouveaux trades, re-keyed avec runner_key
                if len(runner._trades) > trades_before:
                    for t in runner._trades[trades_before:]:
                        # t est (symbol, TradeResult) — remplacer symbol par runner_key
                        all_trades.append((runner_key, t[1]))

            last_closes[symbol] = candle.close

            # Snapshot à chaque changement de timestamp
            next_ts = merged_candles[i + 1].timestamp if i + 1 < total else None
            if next_ts != candle.timestamp and last_closes:
                snap = self._take_snapshot(runners, candle.timestamp, last_closes)
                snapshots.append(snap)

                # Kill switch temps réel (Sprint 24a)
                if len(snapshots) >= 2:
                    window_hours = self._kill_switch_window_hours
                    current_ts = snap.timestamp

                    window_start_equity = snap.total_equity
                    for prev_snap in reversed(snapshots[:-1]):
                        if (current_ts - prev_snap.timestamp).total_seconds() > window_hours * 3600:
                            break
                        window_start_equity = prev_snap.total_equity

                    if window_start_equity > 0:
                        dd_pct = (1 - snap.total_equity / window_start_equity) * 100
                        if dd_pct >= self._kill_switch_pct:
                            if not any(r._kill_switch_triggered for r in runners.values()):
                                logger.warning(
                                    "KILL SWITCH PORTFOLIO: DD={:.1f}% equity={:.0f}$",
                                    dd_pct, snap.total_equity,
                                )
                            for r in runners.values():
                                r._kill_switch_triggered = True
                            self._kill_freeze_until = current_ts + timedelta(hours=24)

                    freeze_until = self._kill_freeze_until
                    if freeze_until and current_ts >= freeze_until:
                        for r in runners.values():
                            r._kill_switch_triggered = False
                        self._kill_freeze_until = None

            # Log de progression
            if (i + 1) % log_interval == 0:
                pct = (i + 1) / total * 100
                logger.info("Simulation : {:.0f}% ({}/{})", pct, i + 1, total)
                if progress_callback:
                    progress_callback(round(pct, 1), f"Simulation {pct:.0f}%")

        return snapshots, all_trades

    def _take_snapshot(
        self,
        runners: dict[str, GridStrategyRunner],
        timestamp: datetime,
        last_closes: dict[str, float],
    ) -> PortfolioSnapshot:
        """Agrège l'état de tous les runners en un snapshot."""
        total_capital = 0.0
        total_realized = 0.0
        total_unrealized = 0.0
        total_margin = 0.0
        n_positions = 0
        n_assets_active = 0

        for runner_key, runner in runners.items():
            symbol = self._symbol_from_key(runner_key)
            total_capital += runner._capital
            total_realized += runner._realized_pnl

            close_price = last_closes.get(symbol, 0.0)
            leverage = runner._leverage

            for pos_symbol, positions in runner._positions.items():
                if not positions:
                    continue
                upnl = runner._gpm.unrealized_pnl(positions, close_price)
                total_unrealized += upnl
                margin = sum(
                    p.entry_price * p.quantity / leverage for p in positions
                )
                total_margin += margin
                n_positions += len(positions)
                n_assets_active += 1

        total_equity = total_capital + total_unrealized
        margin_ratio = total_margin / self._initial_capital if self._initial_capital > 0 else 0.0

        return PortfolioSnapshot(
            timestamp=timestamp,
            total_equity=total_equity,
            total_capital=total_capital,
            total_realized_pnl=total_realized,
            total_unrealized_pnl=total_unrealized,
            total_margin_used=total_margin,
            margin_ratio=margin_ratio,
            n_open_positions=n_positions,
            n_assets_with_positions=n_assets_active,
        )

    # ------------------------------------------------------------------
    # Force-close
    # ------------------------------------------------------------------

    def _force_close_all(
        self,
        runners: dict[str, GridStrategyRunner],
    ) -> list[tuple[str, TradeResult]]:
        """Force-close toutes les positions restantes.

        Retourne les trades force-closed avec la clé runner (strategy:symbol).
        """
        force_closed: list[tuple[str, TradeResult]] = []

        for runner_key, runner in runners.items():
            for pos_symbol, positions in list(runner._positions.items()):
                if not positions:
                    continue

                closes = runner._close_buffer.get(pos_symbol)
                if closes:
                    exit_price = closes[-1]
                else:
                    exit_price = positions[-1].entry_price

                total_notional = sum(
                    p.entry_price * p.quantity for p in positions
                )
                margin_to_return = total_notional / runner._leverage
                runner._capital += margin_to_return

                trade = runner._gpm.close_all_positions(
                    positions,
                    exit_price,
                    positions[-1].entry_time,
                    "end_of_data",
                    MarketRegime.RANGING,
                )
                runner._capital += trade.net_pnl
                runner._realized_pnl += trade.net_pnl

                force_closed.append((runner_key, trade))
                runner._positions[pos_symbol] = []

        if force_closed:
            total_pnl = sum(t.net_pnl for _, t in force_closed)
            logger.info(
                "Force-close : {} trades, P&L={:+.2f}$",
                len(force_closed),
                total_pnl,
            )

        return force_closed

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_drawdown(
        snapshots: list[PortfolioSnapshot],
    ) -> tuple[float, datetime | None, float]:
        """Max drawdown %, date, et durée en heures."""
        if not snapshots:
            return 0.0, None, 0.0

        peak = snapshots[0].total_equity
        max_dd = 0.0
        max_dd_date: datetime | None = None
        peak_time = snapshots[0].timestamp
        max_dd_duration = 0.0

        for snap in snapshots:
            if snap.total_equity > peak:
                peak = snap.total_equity
                peak_time = snap.timestamp
            dd = (snap.total_equity / peak - 1) * 100 if peak > 0 else 0.0
            if dd < max_dd:
                max_dd = dd
                max_dd_date = snap.timestamp
                duration = (snap.timestamp - peak_time).total_seconds() / 3600
                max_dd_duration = duration

        return max_dd, max_dd_date, max_dd_duration

    def _check_kill_switch(
        self,
        snapshots: list[PortfolioSnapshot],
    ) -> list[dict]:
        """Détecte les déclenchements du kill switch (fenêtre glissante)."""
        events: list[dict] = []
        if not snapshots:
            return events

        window = timedelta(hours=self._kill_switch_window_hours)
        threshold = self._kill_switch_pct

        # Index fenêtre glissante
        window_start_idx = 0
        in_trigger = False

        for i, snap in enumerate(snapshots):
            # Avancer le début de la fenêtre
            while (
                window_start_idx < i
                and snap.timestamp - snapshots[window_start_idx].timestamp > window
            ):
                window_start_idx += 1

            # Equity au début de la fenêtre
            start_equity = snapshots[window_start_idx].total_equity
            if start_equity <= 0:
                continue

            dd_pct = (1 - snap.total_equity / start_equity) * 100

            if dd_pct >= threshold and not in_trigger:
                in_trigger = True
                events.append({
                    "timestamp": snap.timestamp.isoformat(),
                    "drawdown_pct": round(dd_pct, 2),
                    "equity": round(snap.total_equity, 2),
                    "window_start_equity": round(start_equity, 2),
                })
            elif dd_pct < threshold * 0.5:
                # Sortie de zone critique (reset pour détecter le prochain)
                in_trigger = False

        return events

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------

    def _build_result(
        self,
        runners: dict[str, GridStrategyRunner],
        snapshots: list[PortfolioSnapshot],
        realized_trades: list[tuple[str, TradeResult]],
        force_closed_trades: list[tuple[str, TradeResult]],
        runner_keys: list[str],
        period_days: int,
    ) -> PortfolioResult:
        """Construit le PortfolioResult final."""
        # P&L
        realized_pnl = sum(t.net_pnl for _, t in realized_trades)
        force_closed_pnl = sum(t.net_pnl for _, t in force_closed_trades)

        # Equity finale
        final_equity = sum(r._capital for r in runners.values())
        total_return_pct = (
            (final_equity / self._initial_capital - 1) * 100
            if self._initial_capital > 0
            else 0.0
        )

        # Trades
        all_trades = realized_trades + force_closed_trades
        total_trades = len(all_trades)
        wins = sum(1 for _, t in all_trades if t.net_pnl > 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

        # Drawdown
        max_dd, max_dd_date, max_dd_duration = self._compute_drawdown(snapshots)

        # Peaks
        peak_margin = max(
            (s.margin_ratio for s in snapshots), default=0.0
        )
        peak_positions = max(
            (s.n_open_positions for s in snapshots), default=0
        )
        peak_assets = max(
            (s.n_assets_with_positions for s in snapshots), default=0
        )

        # Kill switch
        ks_events = self._check_kill_switch(snapshots)

        # Per-runner breakdown (clé = runner_key, e.g. "grid_atr:ICP/USDT")
        per_asset: dict[str, dict] = {}
        for rk in runner_keys:
            rk_trades = [t for s, t in all_trades if s == rk]
            rk_realized = [t for s, t in realized_trades if s == rk]
            rk_force = [t for s, t in force_closed_trades if s == rk]
            n = len(rk_trades)
            w = sum(1 for t in rk_trades if t.net_pnl > 0)
            pnl = sum(t.net_pnl for t in rk_trades)
            per_asset[rk] = {
                "trades": n,
                "wins": w,
                "win_rate": (w / n * 100) if n > 0 else 0.0,
                "net_pnl": round(pnl, 2),
                "realized_trades": len(rk_realized),
                "force_closed_trades": len(rk_force),
                "realized_pnl": round(sum(t.net_pnl for t in rk_realized), 2),
                "force_closed_pnl": round(sum(t.net_pnl for t in rk_force), 2),
            }

        # Assets uniques pour le result
        unique_assets = sorted(set(self._symbol_from_key(k) for k in runner_keys))

        return PortfolioResult(
            initial_capital=self._initial_capital,
            n_assets=len(runner_keys),
            period_days=period_days,
            assets=unique_assets,
            final_equity=round(final_equity, 2),
            total_return_pct=round(total_return_pct, 2),
            total_trades=total_trades,
            win_rate=round(win_rate, 1),
            realized_pnl=round(realized_pnl, 2),
            force_closed_pnl=round(force_closed_pnl, 2),
            max_drawdown_pct=round(max_dd, 2),
            max_drawdown_date=max_dd_date,
            max_drawdown_duration_hours=round(max_dd_duration, 1),
            peak_margin_ratio=round(peak_margin, 4),
            peak_open_positions=peak_positions,
            peak_concurrent_assets=peak_assets,
            kill_switch_triggers=len(ks_events),
            kill_switch_events=ks_events,
            snapshots=snapshots,
            per_asset_results=per_asset,
            all_trades=all_trades,
        )


# ---------------------------------------------------------------------------
# Rapport CLI
# ---------------------------------------------------------------------------


def format_portfolio_report(result: PortfolioResult) -> str:
    """Formate le rapport portfolio pour la console."""
    lines: list[str] = []
    sep = "=" * 65

    lines.append("")
    lines.append(sep)
    lines.append("  PORTFOLIO BACKTEST REPORT")
    lines.append(sep)
    lines.append("")

    # Résumé
    lines.append(f"  Capital initial     : {result.initial_capital:>10,.0f} $")
    lines.append(f"  Equity finale       : {result.final_equity:>10,.0f} $")
    lines.append(f"  Return total        : {result.total_return_pct:>+9.1f} %")
    lines.append(f"  P&L réalisé (TP/SL) : {result.realized_pnl:>+10,.2f} $")
    lines.append(f"  P&L force-closed    : {result.force_closed_pnl:>+10,.2f} $")
    lines.append(f"  Période             : {result.period_days} jours, {result.n_assets} assets")
    lines.append("")

    # Trades
    lines.append("  --- Trades ---")
    lines.append(f"  Total trades        : {result.total_trades}")
    lines.append(f"  Win rate            : {result.win_rate:.1f} %")
    lines.append("")

    # Risque
    lines.append("  --- Risque ---")
    lines.append(f"  Max drawdown        : {result.max_drawdown_pct:.1f} %")
    if result.max_drawdown_date:
        lines.append(f"  Date max DD         : {result.max_drawdown_date}")
    lines.append(f"  Durée max DD        : {result.max_drawdown_duration_hours:.0f} h")
    lines.append("")

    # Marge
    lines.append("  --- Marge ---")
    lines.append(f"  Peak margin ratio   : {result.peak_margin_ratio:.1%}")
    lines.append(f"  Peak positions      : {result.peak_open_positions}")
    lines.append(f"  Peak assets actifs  : {result.peak_concurrent_assets}")
    lines.append("")

    # Kill switch
    lines.append("  --- Kill Switch ---")
    lines.append(f"  Déclenchements      : {result.kill_switch_triggers}")
    for evt in result.kill_switch_events[:5]:
        lines.append(f"    {evt['timestamp'][:19]}  DD={evt['drawdown_pct']:.1f}%  equity={evt['equity']:.0f}$")
    lines.append("")

    # Per-asset (trié par P&L)
    lines.append("  --- Par Runner ---")
    sorted_assets = sorted(
        result.per_asset_results.items(),
        key=lambda x: x[1].get("net_pnl", 0),
        reverse=True,
    )
    # Largeur dynamique pour les clés (strategy:symbol peut être long)
    max_key_len = max((len(k) for k in result.per_asset_results), default=14)
    col_w = max(max_key_len, 14)
    for key, stats in sorted_assets:
        trades = stats.get("trades", 0)
        wr = stats.get("win_rate", 0)
        pnl = stats.get("net_pnl", 0)
        fc = stats.get("force_closed_trades", 0)
        fc_label = f" ({fc}fc)" if fc > 0 else ""
        lines.append(
            f"    {key:{col_w}s}  {trades:4d} trades{fc_label:6s}  "
            f"WR {wr:5.1f}%  P&L {pnl:+9.2f} $"
        )

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)
