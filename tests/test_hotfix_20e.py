"""Tests Hotfix 20e — Persistence grid, kill switch grid, warm-up fixes.

Bugs corrigés :
- Bug 1 : Positions grid sérialisées dans le state (déjà implémenté, vérifié ici)
- Bug 2 : _end_warmup() forcé quand kill switch global restauré
- Bug 3 : Grace period post-warmup (pas de kill switch runner pendant N bougies)
- Bug 4 : Seuils kill switch grid-spécifiques (25% au lieu de 5%)
- Bug 5 : Guard bougies historiques post-warmup (pas de phantom trades)
- Bug 6 : Anti-spam Telegram (déjà implémenté, vérifié ici)
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.backtesting.simulator import (
    GridStrategyRunner,
    LiveStrategyRunner,
    RunnerStats,
    Simulator,
)
from backend.core.grid_position_manager import GridPositionManager
from backend.core.incremental_indicators import IncrementalIndicatorEngine
from backend.core.models import Candle, Direction, MarketRegime, TimeFrame
from backend.core.position_manager import PositionManagerConfig, TradeResult
from backend.core.state_manager import StateManager
from backend.strategies.base_grid import (
    BaseGridStrategy,
    GridLevel,
    GridPosition,
    GridState,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_gpm_config(leverage: int = 6) -> PositionManagerConfig:
    return PositionManagerConfig(
        leverage=leverage,
        maker_fee=0.0002,
        taker_fee=0.0006,
        slippage_pct=0.0005,
        high_vol_slippage_mult=2.0,
        max_risk_per_trade=0.02,
    )


def _make_mock_strategy(
    name: str = "grid_atr",
    timeframe: str = "1h",
    ma_period: int = 7,
    max_positions: int = 3,
    grid_levels: list[GridLevel] | None = None,
) -> MagicMock:
    strategy = MagicMock(spec=BaseGridStrategy)
    strategy.name = name
    config = MagicMock()
    config.timeframe = timeframe
    config.ma_period = ma_period
    config.leverage = 6
    config.per_asset = {}
    config.sl_percent = 20.0
    strategy._config = config
    strategy.min_candles = {"1h": 50}
    strategy.max_positions = max_positions
    strategy.compute_grid.return_value = grid_levels or []
    strategy.should_close_all.return_value = None
    strategy.get_tp_price.return_value = float("nan")
    strategy.get_sl_price.return_value = float("nan")
    strategy.get_current_conditions.return_value = []
    return strategy


def _make_mock_config(
    grid_max_session: float | None = 25.0,
    grid_max_daily: float | None = 25.0,
    n_assets: int = 1,
) -> MagicMock:
    config = MagicMock()
    config.risk.initial_capital = 10_000.0
    config.risk.max_margin_ratio = 0.70
    config.risk.fees.maker_percent = 0.02
    config.risk.fees.taker_percent = 0.06
    config.risk.slippage.default_estimate_percent = 0.05
    config.risk.slippage.high_volatility_multiplier = 2.0
    config.risk.position.max_risk_per_trade_percent = 2.0
    config.risk.position.default_leverage = 15
    config.risk.kill_switch.max_session_loss_percent = 5.0
    config.risk.kill_switch.max_daily_loss_percent = 10.0
    config.risk.kill_switch.grid_max_session_loss_percent = grid_max_session
    config.risk.kill_switch.grid_max_daily_loss_percent = grid_max_daily
    config.risk.kill_switch.global_max_loss_pct = 30.0
    config.risk.kill_switch.global_window_hours = 24
    config.assets = [MagicMock(symbol=f"ASSET{i}/USDT") for i in range(n_assets)]
    return config


def _make_grid_runner(
    strategy=None,
    config=None,
    warmup: bool = False,
) -> GridStrategyRunner:
    if strategy is None:
        strategy = _make_mock_strategy()
    if config is None:
        config = _make_mock_config()

    indicator_engine = MagicMock(spec=IncrementalIndicatorEngine)
    indicator_engine.get_indicators.return_value = {}
    indicator_engine.update = MagicMock()

    gpm = GridPositionManager(_make_gpm_config())
    data_engine = MagicMock()
    data_engine.get_funding_rate.return_value = None
    data_engine.get_open_interest.return_value = []

    runner = GridStrategyRunner(
        strategy=strategy,
        config=config,
        indicator_engine=indicator_engine,
        grid_position_manager=gpm,
        data_engine=data_engine,
    )
    if not warmup:
        runner._is_warming_up = False
    return runner


def _fill_buffer(
    runner: GridStrategyRunner,
    symbol: str = "BTC/USDT",
    n: int = 10,
    base_close: float = 100_000.0,
) -> None:
    runner._close_buffer[symbol] = deque(maxlen=50)
    for i in range(n):
        runner._close_buffer[symbol].append(base_close + i * 10)


def _make_candle(
    close: float = 100_000.0,
    high: float | None = None,
    low: float | None = None,
    volume: float = 100.0,
    ts: datetime | None = None,
) -> Candle:
    h = high or close * 1.001
    lo = low or close * 0.999
    return Candle(
        timestamp=ts or datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        open=close,
        high=h,
        low=lo,
        close=close,
        volume=volume,
        symbol="BTC/USDT",
        timeframe=TimeFrame.H1,
    )


def _make_trade(net_pnl: float = -1_000.0) -> TradeResult:
    return TradeResult(
        direction=Direction.LONG,
        entry_price=100_000.0,
        exit_price=90_000.0,
        quantity=0.01,
        entry_time=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        exit_time=datetime(2024, 6, 15, 13, 0, tzinfo=timezone.utc),
        gross_pnl=net_pnl + 5.0,
        fee_cost=3.0,
        slippage_cost=2.0,
        net_pnl=net_pnl,
        exit_reason="sl",
        market_regime=MarketRegime.RANGING,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Bug 1 — Sérialisation / restauration positions grid (déjà implémenté)
# ═══════════════════════════════════════════════════════════════════════════════


class TestGridPositionsSerialization:
    """Vérifie que les positions grid sont sérialisées dans le state."""

    @pytest.mark.asyncio
    async def test_serialize_grid_positions(self, tmp_path):
        """Runner avec positions grid → JSON contient grid_positions."""
        state_file = str(tmp_path / "state.json")
        sm = StateManager(db=MagicMock(), state_file=state_file)

        runner = _make_grid_runner()
        runner._positions["BTC/USDT"] = [
            GridPosition(
                level=0, direction=Direction.LONG,
                entry_price=95_000.0, quantity=0.01,
                entry_time=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
                entry_fee=0.57,
            ),
        ]
        runner._positions["ETH/USDT"] = [
            GridPosition(
                level=1, direction=Direction.LONG,
                entry_price=3_000.0, quantity=0.5,
                entry_time=datetime(2024, 6, 15, 13, 0, tzinfo=timezone.utc),
                entry_fee=0.09,
            ),
        ]

        await sm.save_runner_state([runner])

        data = json.loads((tmp_path / "state.json").read_text())
        gp = data["runners"]["grid_atr"]["grid_positions"]
        assert len(gp) == 2
        symbols = {p["symbol"] for p in gp}
        assert symbols == {"BTC/USDT", "ETH/USDT"}

    @pytest.mark.asyncio
    async def test_restore_grid_positions(self):
        """JSON avec grid_positions → positions restaurées via _apply_restored_state."""
        runner = _make_grid_runner(warmup=True)
        state = {
            "capital": 9_500.0,
            "kill_switch": False,
            "realized_pnl": -500.0,
            "total_trades": 5,
            "wins": 2,
            "losses": 3,
            "is_active": True,
            "grid_positions": [
                {
                    "symbol": "BTC/USDT",
                    "level": 0,
                    "direction": "LONG",
                    "entry_price": 95_000.0,
                    "quantity": 0.01,
                    "entry_time": "2024-06-15T12:00:00+00:00",
                    "entry_fee": 0.57,
                },
            ],
        }
        runner.restore_state(state)
        runner._end_warmup()

        assert len(runner._positions.get("BTC/USDT", [])) == 1
        pos = runner._positions["BTC/USDT"][0]
        assert pos.entry_price == 95_000.0
        assert pos.level == 0
        assert pos.direction == Direction.LONG

    @pytest.mark.asyncio
    async def test_restore_backward_compat_no_grid_positions(self):
        """JSON sans grid_positions (ancien format) → pas de crash."""
        runner = _make_grid_runner(warmup=True)
        state = {
            "capital": 10_000.0,
            "kill_switch": False,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "is_active": True,
        }
        runner.restore_state(state)
        runner._end_warmup()

        assert runner._capital == 10_000.0
        assert not runner._positions  # Vide, pas d'erreur

    @pytest.mark.asyncio
    async def test_roundtrip_grid_positions(self, tmp_path):
        """save → load → restore → positions identiques."""
        state_file = str(tmp_path / "state.json")
        sm = StateManager(db=MagicMock(), state_file=state_file)

        original = _make_grid_runner()
        original._positions["SOL/USDT"] = [
            GridPosition(
                level=2, direction=Direction.LONG,
                entry_price=150.0, quantity=10.0,
                entry_time=datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc),
                entry_fee=0.09,
            ),
        ]

        await sm.save_runner_state([original])
        data = await sm.load_runner_state()

        restored = _make_grid_runner(warmup=True)
        restored.restore_state(data["runners"]["grid_atr"])
        restored._end_warmup()

        assert len(restored._positions.get("SOL/USDT", [])) == 1
        pos = restored._positions["SOL/USDT"][0]
        assert pos.level == 2
        assert pos.entry_price == 150.0
        assert pos.quantity == 10.0

    @pytest.mark.asyncio
    async def test_serialize_skips_empty_symbols(self, tmp_path):
        """Symbols sans positions → pas dans le JSON."""
        state_file = str(tmp_path / "state.json")
        sm = StateManager(db=MagicMock(), state_file=state_file)

        runner = _make_grid_runner()
        runner._positions["BTC/USDT"] = []  # Vide
        runner._positions["ETH/USDT"] = [
            GridPosition(
                level=0, direction=Direction.LONG,
                entry_price=3_000.0, quantity=0.5,
                entry_time=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
                entry_fee=0.09,
            ),
        ]

        await sm.save_runner_state([runner])

        data = json.loads((tmp_path / "state.json").read_text())
        gp = data["runners"]["grid_atr"]["grid_positions"]
        symbols = {p["symbol"] for p in gp}
        assert "BTC/USDT" not in symbols
        assert "ETH/USDT" in symbols


# ═══════════════════════════════════════════════════════════════════════════════
# Bug 2 — Warm-up forcé quand kill switch global restauré
# ═══════════════════════════════════════════════════════════════════════════════


class TestKillSwitchForcesEndWarmup:
    """Vérifie que _end_warmup() est forcé quand kill switch global restauré."""

    @pytest.mark.asyncio
    async def test_kill_switch_forces_end_warmup(self):
        """Kill switch restored + warm-up actif → warm-up terminé, state restauré."""
        config = _make_mock_config()
        data_engine = MagicMock()
        data_engine.get_all_symbols.return_value = []
        data_engine.on_candle = MagicMock()
        db = MagicMock()

        sim = Simulator(data_engine=data_engine, config=config, db=db)

        # Simuler start() manuellement avec un runner grid en warm-up
        strategy = _make_mock_strategy()
        runner = _make_grid_runner(strategy=strategy, config=config, warmup=True)

        # Préparer un pending_restore
        runner._pending_restore = {
            "capital": 5_599.0,
            "kill_switch": False,
            "realized_pnl": -4_401.0,
            "total_trades": 71,
            "wins": 30,
            "losses": 41,
            "is_active": True,
        }

        sim._runners = [runner]

        # Simuler la restauration du kill switch global
        saved_state = {
            "global_kill_switch": True,
            "runners": {},
        }
        sim._global_kill_switch = saved_state["global_kill_switch"]
        sim._stop_all_runners()

        # Forcer _end_warmup comme dans start()
        for r in sim._runners:
            if hasattr(r, '_is_warming_up') and r._is_warming_up:
                r._end_warmup()

        assert not runner._is_warming_up
        assert runner._capital == 5_599.0
        assert runner._stats.total_trades == 71

    @pytest.mark.asyncio
    async def test_state_not_overwritten_by_blocked_warmup(self, tmp_path):
        """State restauré (capital=5k) → kill switch → StateManager sauvegarde → capital préservé."""
        state_file = str(tmp_path / "state.json")
        sm = StateManager(db=MagicMock(), state_file=state_file)

        config = _make_mock_config()
        runner = _make_grid_runner(config=config, warmup=True)

        # Simuler pending_restore avec le bon état
        runner._pending_restore = {
            "capital": 5_000.0,
            "realized_pnl": -5_000.0,
            "total_trades": 50,
            "wins": 20,
            "losses": 30,
            "is_active": True,
        }

        # Forcer fin warmup (comme le kill switch global le ferait)
        runner._end_warmup()

        # StateManager sauvegarde l'état
        await sm.save_runner_state([runner])

        data = json.loads((tmp_path / "state.json").read_text())
        saved_capital = data["runners"]["grid_atr"]["capital"]
        assert saved_capital == 5_000.0  # Pas 10_000


# ═══════════════════════════════════════════════════════════════════════════════
# Bug 3 — Grace period post-warmup
# ═══════════════════════════════════════════════════════════════════════════════


class TestGracePeriod:
    """Vérifie la grace period du kill switch runner post-warmup."""

    def test_grace_period_prevents_runner_kill_switch(self):
        """Perte > seuil pendant grace period → kill switch NOT triggered."""
        runner = _make_grid_runner()
        runner._candles_since_warmup = 3  # < 10 = grace period
        runner._realized_pnl = 0.0

        trade = _make_trade(net_pnl=-3_000.0)  # 30% loss > 25% threshold
        runner._record_trade(trade, "BTC/USDT")

        assert not runner._kill_switch_triggered

    def test_grace_period_expired_allows_kill_switch(self):
        """Après 10 bougies → kill switch triggered normalement."""
        runner = _make_grid_runner()
        runner._candles_since_warmup = 15  # > 10 = grace period expirée
        runner._realized_pnl = 0.0

        trade = _make_trade(net_pnl=-3_000.0)  # 30% loss > 25% threshold
        runner._record_trade(trade, "BTC/USDT")

        assert runner._kill_switch_triggered

    @pytest.mark.asyncio
    async def test_global_kill_switch_ignores_grace_period(self):
        """Le kill switch global (30%) n'est PAS affecté par la grace period."""
        config = _make_mock_config()
        data_engine = MagicMock()
        data_engine.get_all_symbols.return_value = []
        data_engine.on_candle = MagicMock()

        sim = Simulator(data_engine=data_engine, config=config)

        runner = _make_grid_runner(config=config)
        runner._candles_since_warmup = 0  # Grace period active
        runner._capital = 5_000.0  # Grosse perte

        sim._runners = [runner]

        # Le global kill switch ne regarde pas la grace period du runner
        # Il regarde les capital_snapshots dans la fenêtre glissante
        sim._capital_snapshots.append(
            (datetime.now(tz=timezone.utc), 10_000.0)
        )
        sim._warmup_ended_at = datetime.now(tz=timezone.utc) - timedelta(hours=2)

        # Le check global devrait fonctionner même en grace period runner
        # (drawdown 50% > 30% threshold)
        await sim._check_global_kill_switch()

        assert sim._global_kill_switch


# ═══════════════════════════════════════════════════════════════════════════════
# Bug 4 — Seuils kill switch grid vs mono
# ═══════════════════════════════════════════════════════════════════════════════


class TestGridKillSwitchThresholds:
    """Vérifie les seuils kill switch spécifiques au grid."""

    def test_grid_runner_uses_grid_thresholds(self):
        """GridStrategyRunner avec perte 10% → NOT triggered (seuil grid 25%)."""
        runner = _make_grid_runner()
        runner._candles_since_warmup = 20  # Grace period expirée
        runner._realized_pnl = 0.0

        trade = _make_trade(net_pnl=-1_000.0)  # 10% loss < 25% grid threshold
        runner._record_trade(trade, "BTC/USDT")

        assert not runner._kill_switch_triggered

    def test_grid_runner_kill_switch_at_grid_threshold(self):
        """GridStrategyRunner avec perte 25% → triggered."""
        runner = _make_grid_runner()
        runner._candles_since_warmup = 20
        runner._realized_pnl = 0.0

        trade = _make_trade(net_pnl=-2_500.0)  # 25% loss = grid threshold
        runner._record_trade(trade, "BTC/USDT")

        assert runner._kill_switch_triggered

    def test_live_runner_uses_standard_thresholds(self):
        """LiveStrategyRunner avec perte 6% → triggered (seuil standard 5%)."""
        config = _make_mock_config()
        strategy = MagicMock()
        strategy.name = "vwap_rsi"
        strategy.min_candles = {"5m": 50}

        indicator_engine = MagicMock(spec=IncrementalIndicatorEngine)
        indicator_engine.get_indicators.return_value = {}

        from backend.core.position_manager import PositionManager

        pm_config = PositionManagerConfig(
            leverage=15, maker_fee=0.0002, taker_fee=0.0006,
            slippage_pct=0.0005, high_vol_slippage_mult=2.0,
            max_risk_per_trade=0.02,
        )
        pm = PositionManager(pm_config)

        data_engine = MagicMock()
        data_engine.get_funding_rate.return_value = None
        data_engine.get_open_interest.return_value = []

        runner = LiveStrategyRunner(
            strategy=strategy, config=config,
            indicator_engine=indicator_engine,
            position_manager=pm, data_engine=data_engine,
        )

        trade = _make_trade(net_pnl=-600.0)  # 6% loss > 5% threshold
        runner._record_trade(trade, "BTC/USDT")

        assert runner._kill_switch_triggered

    def test_grid_thresholds_fallback_to_standard(self):
        """Config sans grid thresholds → fallback sur standard."""
        config = _make_mock_config(
            grid_max_session=None,
            grid_max_daily=None,
        )
        runner = _make_grid_runner(config=config)
        runner._candles_since_warmup = 20
        runner._realized_pnl = 0.0

        # Perte 6% > 5% standard threshold (fallback car grid_max = None)
        trade = _make_trade(net_pnl=-600.0)
        runner._record_trade(trade, "BTC/USDT")

        assert runner._kill_switch_triggered


# ═══════════════════════════════════════════════════════════════════════════════
# Bug 5 — Trades phantom (bougies historiques post-warmup)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPhantomTradesGuard:
    """Vérifie que les bougies historiques post-warmup ne génèrent pas de trades."""

    @pytest.mark.asyncio
    async def test_old_candles_skipped_after_warmup(self):
        """Bougie > 2h post-warmup (dans la fenêtre de 5 min) → pas de trade."""
        strategy = _make_mock_strategy(
            grid_levels=[
                GridLevel(
                    index=0, entry_price=95_000.0,
                    direction=Direction.LONG, size_fraction=0.33,
                ),
            ],
        )
        runner = _make_grid_runner(strategy=strategy)
        _fill_buffer(runner, n=10)

        # Simuler un warm-up qui vient de se terminer (guard actif 5 min)
        runner._warmup_ended_at = datetime.now(tz=timezone.utc)

        # Bougie historique (3h old)
        old_ts = datetime.now(tz=timezone.utc) - timedelta(hours=3)
        candle = _make_candle(
            close=96_000.0, low=94_500.0, high=97_000.0, ts=old_ts,
        )

        await runner.on_candle("BTC/USDT", "1h", candle)

        positions = runner._positions.get("BTC/USDT", [])
        assert len(positions) == 0

    @pytest.mark.asyncio
    async def test_live_candles_trade_normally(self):
        """Bougie fraîche (< 2h) → trade normalement même post-warmup."""
        strategy = _make_mock_strategy(
            grid_levels=[
                GridLevel(
                    index=0, entry_price=95_000.0,
                    direction=Direction.LONG, size_fraction=0.33,
                ),
            ],
        )
        runner = _make_grid_runner(strategy=strategy)
        _fill_buffer(runner, n=10)

        # Simuler un warm-up qui vient de se terminer
        runner._warmup_ended_at = datetime.now(tz=timezone.utc)

        # Bougie récente (30 min)
        fresh_ts = datetime.now(tz=timezone.utc) - timedelta(minutes=30)
        candle = _make_candle(
            close=96_000.0, low=94_500.0, high=97_000.0, ts=fresh_ts,
        )

        await runner.on_candle("BTC/USDT", "1h", candle)

        positions = runner._positions.get("BTC/USDT", [])
        assert len(positions) == 1

    @pytest.mark.asyncio
    async def test_candle_counter_increments_on_live_only(self):
        """Le compteur grace period ne s'incrémente que pour les bougies live."""
        runner = _make_grid_runner()
        _fill_buffer(runner, n=10)

        # Simuler warm-up qui vient de terminer
        runner._warmup_ended_at = datetime.now(tz=timezone.utc)

        assert runner._candles_since_warmup == 0

        # Bougie historique → skippée (pas d'incrément)
        old_ts = datetime.now(tz=timezone.utc) - timedelta(hours=5)
        old_candle = _make_candle(ts=old_ts)
        await runner.on_candle("BTC/USDT", "1h", old_candle)
        assert runner._candles_since_warmup == 0

        # Bougie live → incrément
        fresh_ts = datetime.now(tz=timezone.utc) - timedelta(minutes=10)
        live_candle = _make_candle(ts=fresh_ts)
        await runner.on_candle("BTC/USDT", "1h", live_candle)
        assert runner._candles_since_warmup == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Bug 6 — Anti-spam Telegram (déjà implémenté, vérifié ici)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTelegramAntiSpam:
    """Vérifie le cooldown par type d'anomalie dans le Notifier."""

    @pytest.mark.asyncio
    async def test_anomaly_cooldown_same_type(self):
        """Même anomalie 3x rapidement → 1 seul message Telegram."""
        from backend.alerts.notifier import AnomalyType, Notifier

        telegram = AsyncMock()
        notifier = Notifier(telegram=telegram)

        for _ in range(3):
            await notifier.notify_anomaly(
                AnomalyType.ALL_STRATEGIES_STOPPED, "test"
            )

        assert telegram.send_message.call_count == 1

    @pytest.mark.asyncio
    async def test_different_anomalies_not_throttled(self):
        """2 types différents → 2 messages envoyés."""
        from backend.alerts.notifier import AnomalyType, Notifier

        telegram = AsyncMock()
        notifier = Notifier(telegram=telegram)

        await notifier.notify_anomaly(AnomalyType.WS_DISCONNECTED, "ws")
        await notifier.notify_anomaly(AnomalyType.DATA_STALE, "data")

        assert telegram.send_message.call_count == 2

    @pytest.mark.asyncio
    async def test_cooldown_expired_resends(self):
        """Après expiration du cooldown → renvoi autorisé."""
        import time
        from backend.alerts.notifier import AnomalyType, Notifier

        telegram = AsyncMock()
        notifier = Notifier(telegram=telegram)

        await notifier.notify_anomaly(
            AnomalyType.SL_PLACEMENT_FAILED, "test1"
        )
        assert telegram.send_message.call_count == 1

        # Simuler l'expiration du cooldown (SL = 300s)
        notifier._last_anomaly_sent[AnomalyType.SL_PLACEMENT_FAILED] = (
            time.monotonic() - 400
        )

        await notifier.notify_anomaly(
            AnomalyType.SL_PLACEMENT_FAILED, "test2"
        )
        assert telegram.send_message.call_count == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Config — Vérification KillSwitchConfig grid thresholds
# ═══════════════════════════════════════════════════════════════════════════════


class TestKillSwitchConfig:
    """Vérifie que le modèle Pydantic accepte les nouveaux champs grid."""

    def test_config_with_grid_thresholds(self):
        """KillSwitchConfig accepte grid_max_session/daily_loss_percent."""
        from backend.core.config import KillSwitchConfig

        ks = KillSwitchConfig(
            max_session_loss_percent=5.0,
            max_daily_loss_percent=10.0,
            grid_max_session_loss_percent=25.0,
            grid_max_daily_loss_percent=25.0,
        )
        assert ks.grid_max_session_loss_percent == 25.0
        assert ks.grid_max_daily_loss_percent == 25.0

    def test_config_without_grid_thresholds(self):
        """KillSwitchConfig sans grid thresholds → None (backward compat)."""
        from backend.core.config import KillSwitchConfig

        ks = KillSwitchConfig()
        assert ks.grid_max_session_loss_percent is None
        assert ks.grid_max_daily_loss_percent is None
        assert ks.max_session_loss_percent == 5.0
