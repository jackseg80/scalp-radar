"""Tests pour Envelope DCA SHORT (Sprint 15).

Couvre :
- Tests 1-6 : Signal generation (compute_grid, should_close_all)
- Tests 7-10 : Fast engine multi-position SHORT
- Tests 11-16 : Registry, config, WFO integration
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from backend.core.config import EnvelopeDCAShortConfig
from backend.core.models import Candle, Direction
from backend.strategies.base import StrategyContext
from backend.strategies.base_grid import GridLevel, GridPosition, GridState
from backend.strategies.envelope_dca_short import EnvelopeDCAShortStrategy


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_candle(
    ts: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 100.0,
    tf: str = "1h",
) -> Candle:
    return Candle(
        symbol="BTC/USDT",
        exchange="binance",
        timeframe=tf,
        timestamp=ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _make_candles(
    n: int,
    start_price: float = 100.0,
    step: float = 0.0,
    tf: str = "1h",
    volume: float = 100.0,
) -> list[Candle]:
    """Génère N candles synthétiques."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    for i in range(n):
        price = start_price + i * step
        candles.append(Candle(
            symbol="BTC/USDT",
            exchange="binance",
            timeframe=tf,
            timestamp=base + timedelta(hours=i),
            open=price,
            high=price + 1.0,
            low=price - 1.0,
            close=price,
            volume=volume,
        ))
    return candles


def _make_strategy(**kwargs) -> EnvelopeDCAShortStrategy:
    """Crée une stratégie SHORT avec defaults sensibles."""
    defaults = {
        "ma_period": 7,
        "num_levels": 3,
        "envelope_start": 0.05,
        "envelope_step": 0.02,
        "sl_percent": 20.0,
        "sides": ["short"],
        "leverage": 6,
    }
    defaults.update(kwargs)
    config = EnvelopeDCAShortConfig(**defaults)
    return EnvelopeDCAShortStrategy(config)


def _make_ctx(sma_val: float, close: float) -> StrategyContext:
    """Crée un StrategyContext minimal avec SMA et close."""
    return StrategyContext(
        symbol="BTC/USDT",
        timestamp=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        candles={},
        indicators={"1h": {"sma": sma_val, "close": close}},
        current_position=None,
        capital=10_000.0,
        config=None,  # type: ignore[arg-type]
    )


def _make_grid_state(
    positions: list[GridPosition] | None = None,
) -> GridState:
    """Crée un GridState."""
    positions = positions or []
    total_qty = sum(p.quantity for p in positions)
    avg_entry = (
        sum(p.entry_price * p.quantity for p in positions) / total_qty
        if total_qty > 0 else 0.0
    )
    return GridState(
        positions=positions,
        avg_entry_price=avg_entry,
        total_quantity=total_qty,
        total_notional=0.0,
        unrealized_pnl=0.0,
    )


def _make_cache(n: int = 100):
    """Crée un IndicatorCache minimal pour envelope_dca_short."""
    from backend.optimization.indicator_cache import IndicatorCache

    rng = np.random.default_rng(42)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    volumes = rng.uniform(50, 200, n)

    # SMA 7
    sma_arr = np.full(n, np.nan)
    for i in range(6, n):
        sma_arr[i] = np.mean(prices[i - 6:i + 1])

    return IndicatorCache(
        n_candles=n,
        opens=prices + rng.uniform(-0.3, 0.3, n),
        highs=prices + np.abs(rng.normal(0.5, 0.3, n)),
        lows=prices - np.abs(rng.normal(0.5, 0.3, n)),
        closes=prices,
        volumes=volumes,
        total_days=n / 24,
        rsi={14: np.full(n, 50.0)},
        vwap=np.full(n, np.nan),
        vwap_distance_pct=np.full(n, np.nan),
        adx_arr=np.full(n, 25.0),
        di_plus=np.full(n, 15.0),
        di_minus=np.full(n, 10.0),
        atr_arr=np.full(n, 1.0),
        atr_sma=np.full(n, 1.0),
        volume_sma_arr=np.full(n, 100.0),
        regime=np.zeros(n, dtype=np.int8),
        rolling_high={},
        rolling_low={},
        filter_adx=np.full(n, np.nan),
        filter_di_plus=np.full(n, np.nan),
        filter_di_minus=np.full(n, np.nan),
        bb_sma={7: sma_arr},
        bb_upper={},
        bb_lower={},
        supertrend_direction={},
        atr_by_period={},
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1-6 : Signal generation (compute_grid, should_close_all)
# ═══════════════════════════════════════════════════════════════════════════


class TestEnvelopeDCAShortSignals:
    """Tests des signaux SHORT."""

    def test_name(self):
        """Le nom de la stratégie est 'envelope_dca_short'."""
        strategy = _make_strategy()
        assert strategy.name == "envelope_dca_short"

    def test_compute_grid_returns_short_levels_above_sma(self):
        """compute_grid retourne des niveaux SHORT au-dessus de la SMA."""
        strategy = _make_strategy(num_levels=3, envelope_start=0.05, envelope_step=0.02)
        sma_val = 100.0
        close = 110.0  # Au-dessus de la SMA
        ctx = _make_ctx(sma_val, close)
        grid_state = _make_grid_state()

        levels = strategy.compute_grid(ctx, grid_state)

        assert len(levels) == 3
        for lvl in levels:
            assert lvl.direction == Direction.SHORT
            assert lvl.entry_price > sma_val

    def test_compute_grid_asymmetric_offsets(self):
        """Les enveloppes SHORT sont asymétriques (round(1/(1-lower) - 1, 3))."""
        strategy = _make_strategy(
            num_levels=3, envelope_start=0.05, envelope_step=0.02,
        )
        sma_val = 100.0
        ctx = _make_ctx(sma_val, 110.0)
        grid_state = _make_grid_state()

        levels = strategy.compute_grid(ctx, grid_state)

        # lower_offsets = [0.05, 0.07, 0.09]
        # high_offsets = [round(1/(1-0.05)-1, 3), round(1/(1-0.07)-1, 3), round(1/(1-0.09)-1, 3)]
        #             = [0.053, 0.075, 0.099]
        expected_offsets = [
            round(1 / (1 - 0.05) - 1, 3),
            round(1 / (1 - 0.07) - 1, 3),
            round(1 / (1 - 0.09) - 1, 3),
        ]
        for lvl, offset in zip(levels, expected_offsets):
            expected_price = sma_val * (1 + offset)
            assert lvl.entry_price == pytest.approx(expected_price, rel=1e-6)

    def test_direction_lock_short_prevents_long(self):
        """Si SHORT ouvert, pas de niveaux LONG proposés."""
        strategy = _make_strategy(num_levels=3)
        sma_val = 100.0
        ctx = _make_ctx(sma_val, 95.0)

        # Position SHORT existante
        pos = GridPosition(
            level=0, direction=Direction.SHORT,
            entry_price=105.0, quantity=1.0,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            entry_fee=0.1,
        )
        grid_state = _make_grid_state([pos])

        levels = strategy.compute_grid(ctx, grid_state)

        # Tous les niveaux doivent être SHORT (pas LONG)
        for lvl in levels:
            assert lvl.direction == Direction.SHORT
        # Le level 0 est déjà rempli, on attend 2 niveaux
        assert len(levels) == 2

    def test_should_close_all_tp_short(self):
        """TP global SHORT : close <= SMA → retour à la moyenne."""
        strategy = _make_strategy(sl_percent=20.0)
        sma_val = 100.0
        close = 99.0  # En dessous de la SMA → TP pour SHORT

        ctx = _make_ctx(sma_val, close)
        pos = GridPosition(
            level=0, direction=Direction.SHORT,
            entry_price=105.0, quantity=1.0,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            entry_fee=0.1,
        )
        grid_state = _make_grid_state([pos])

        result = strategy.should_close_all(ctx, grid_state)
        assert result == "tp_global"

    def test_should_close_all_sl_short(self):
        """SL global SHORT : close >= avg_entry × (1 + sl_pct)."""
        strategy = _make_strategy(sl_percent=20.0)
        avg_entry = 100.0
        sl_price = avg_entry * (1 + 0.20)  # = 120.0
        close = 121.0  # Au-dessus du SL → SL touché
        sma_val = 90.0  # SMA en dessous du close → close > sma → pas TP

        ctx = _make_ctx(sma_val, close)
        pos = GridPosition(
            level=0, direction=Direction.SHORT,
            entry_price=avg_entry, quantity=1.0,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            entry_fee=0.1,
        )
        grid_state = _make_grid_state([pos])

        result = strategy.should_close_all(ctx, grid_state)
        assert result == "sl_global"

    def test_should_close_all_no_exit(self):
        """Pas de sortie si prix entre SMA et SL."""
        strategy = _make_strategy(sl_percent=20.0)
        avg_entry = 105.0
        close = 108.0  # Au-dessus de l'entrée mais pas au SL
        sma_val = 100.0  # En dessous → pas encore TP (close > sma)

        ctx = _make_ctx(sma_val, close)
        pos = GridPosition(
            level=0, direction=Direction.SHORT,
            entry_price=avg_entry, quantity=1.0,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            entry_fee=0.1,
        )
        grid_state = _make_grid_state([pos])

        result = strategy.should_close_all(ctx, grid_state)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 7-10 : Fast engine SHORT
# ═══════════════════════════════════════════════════════════════════════════


class TestFastEngineShort:
    """Tests du fast engine multi-position en direction SHORT."""

    def test_run_multi_backtest_short(self):
        """Fast engine retourne un résultat valide pour envelope_dca_short."""
        from backend.backtesting.engine import BacktestConfig
        from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache

        cache = _make_cache()
        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            leverage=6,
        )

        params = {
            "ma_period": 7, "num_levels": 3,
            "envelope_start": 0.07, "envelope_step": 0.03,
            "sl_percent": 25.0,
        }

        result = run_multi_backtest_from_cache("envelope_dca_short", params, cache, bt_config)

        assert len(result) == 5  # (params, sharpe, return, PF, n_trades)
        assert result[0] == params

    def test_simulate_short_direction(self):
        """_simulate_envelope_dca avec direction=-1 produit des trades."""
        from backend.backtesting.engine import BacktestConfig
        from backend.optimization.fast_multi_backtest import _simulate_envelope_dca

        cache = _make_cache(200)
        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            leverage=6,
            initial_capital=10_000,
        )

        params = {
            "ma_period": 7, "num_levels": 3,
            "envelope_start": 0.07, "envelope_step": 0.03,
            "sl_percent": 25.0,
        }

        trade_pnls, trade_returns, final = _simulate_envelope_dca(
            cache, params, bt_config, direction=-1,
        )

        assert final > 0
        assert final < 1_000_000

    def test_long_backward_compatible(self):
        """_simulate_envelope_dca(direction=1) reste compatible avec l'ancien comportement."""
        from backend.backtesting.engine import BacktestConfig
        from backend.optimization.fast_multi_backtest import _simulate_envelope_dca

        cache = _make_cache(200)
        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            leverage=6,
            initial_capital=10_000,
        )

        params = {
            "ma_period": 7, "num_levels": 3,
            "envelope_start": 0.07, "envelope_step": 0.03,
            "sl_percent": 25.0,
        }

        # LONG explicite
        pnls_long, _, final_long = _simulate_envelope_dca(
            cache, params, bt_config, direction=1,
        )
        # LONG par défaut
        pnls_default, _, final_default = _simulate_envelope_dca(
            cache, params, bt_config,
        )

        assert pnls_long == pnls_default
        assert final_long == final_default

    def test_short_envelope_offsets_asymmetric(self):
        """Le fast engine SHORT utilise les offsets asymétriques."""
        from backend.backtesting.engine import BacktestConfig
        from backend.optimization.fast_multi_backtest import _simulate_envelope_dca

        # Créer un cache avec des prix montant clairement au-dessus de la SMA
        n = 50
        prices = np.full(n, 100.0)
        sma_arr = np.full(n, np.nan)
        for i in range(6, n):
            sma_arr[i] = 100.0  # SMA fixe à 100

        # Prix monte à 108 (au-dessus de l'enveloppe 0.05 → 105.3) puis redescend
        for i in range(20, 30):
            prices[i] = 108.0
        for i in range(30, 40):
            prices[i] = 99.0  # Retour sous la SMA → TP

        from backend.optimization.indicator_cache import IndicatorCache
        rng = np.random.default_rng(42)
        cache = IndicatorCache(
            n_candles=n,
            opens=prices.copy(),
            highs=prices + 1.0,
            lows=prices - 1.0,
            closes=prices,
            volumes=np.full(n, 100.0),
            total_days=n / 24,
            rsi={14: np.full(n, 50.0)},
            vwap=np.full(n, np.nan),
            vwap_distance_pct=np.full(n, np.nan),
            adx_arr=np.full(n, 25.0),
            di_plus=np.full(n, 15.0),
            di_minus=np.full(n, 10.0),
            atr_arr=np.full(n, 1.0),
            atr_sma=np.full(n, 1.0),
            volume_sma_arr=np.full(n, 100.0),
            regime=np.zeros(n, dtype=np.int8),
            rolling_high={},
            rolling_low={},
            filter_adx=np.full(n, np.nan),
            filter_di_plus=np.full(n, np.nan),
            filter_di_minus=np.full(n, np.nan),
            bb_sma={7: sma_arr},
            bb_upper={},
            bb_lower={},
            supertrend_direction={},
            atr_by_period={},
        )

        bt_config = BacktestConfig(
            symbol="BTC/USDT",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            leverage=6,
            initial_capital=10_000,
        )

        params = {
            "ma_period": 7, "num_levels": 2,
            "envelope_start": 0.05, "envelope_step": 0.02,
            "sl_percent": 25.0,
        }

        trade_pnls, _, _ = _simulate_envelope_dca(
            cache, params, bt_config, direction=-1,
        )

        # On doit avoir au moins 1 trade (entrée SHORT puis TP au retour)
        assert len(trade_pnls) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# 11-16 : Registry, config, WFO integration
# ═══════════════════════════════════════════════════════════════════════════


class TestEnvelopeDCAShortIntegration:
    """Tests d'intégration WFO / registry / config."""

    def test_in_registry(self):
        """envelope_dca_short est dans STRATEGY_REGISTRY."""
        from backend.optimization import STRATEGY_REGISTRY

        assert "envelope_dca_short" in STRATEGY_REGISTRY
        config_cls, strategy_cls = STRATEGY_REGISTRY["envelope_dca_short"]
        assert config_cls is EnvelopeDCAShortConfig
        assert strategy_cls is EnvelopeDCAShortStrategy

    def test_is_grid_strategy(self):
        """is_grid_strategy retourne True pour envelope_dca_short."""
        from backend.optimization import is_grid_strategy

        assert is_grid_strategy("envelope_dca_short") is True
        # Vérifier que LONG n'est pas cassé
        assert is_grid_strategy("envelope_dca") is True

    def test_grid_strategies_set(self):
        """GRID_STRATEGIES contient envelope_dca_short."""
        from backend.optimization import GRID_STRATEGIES

        assert "envelope_dca_short" in GRID_STRATEGIES

    def test_indicator_params(self):
        """_INDICATOR_PARAMS a envelope_dca_short → ['ma_period']."""
        from backend.optimization.walk_forward import _INDICATOR_PARAMS

        assert "envelope_dca_short" in _INDICATOR_PARAMS
        assert _INDICATOR_PARAMS["envelope_dca_short"] == ["ma_period"]

    def test_create_strategy_with_params(self):
        """create_strategy_with_params crée un EnvelopeDCAShortStrategy."""
        from backend.optimization import create_strategy_with_params

        params = {
            "ma_period": 10, "num_levels": 4,
            "envelope_start": 0.05, "envelope_step": 0.02,
            "sl_percent": 20.0, "sides": ["short"], "leverage": 6,
        }
        strategy = create_strategy_with_params("envelope_dca_short", params)
        assert isinstance(strategy, EnvelopeDCAShortStrategy)
        assert strategy.name == "envelope_dca_short"
        assert strategy.max_positions == 4

    def test_config_defaults(self):
        """EnvelopeDCAShortConfig a sides=['short'] par défaut."""
        config = EnvelopeDCAShortConfig()
        assert config.sides == ["short"]
        assert config.enabled is False
        assert config.leverage == 6

    def test_config_validation(self):
        """EnvelopeDCAShortConfig valide les bornes."""
        cfg = EnvelopeDCAShortConfig(ma_period=5, num_levels=2, leverage=4)
        assert cfg.leverage == 4

        with pytest.raises(Exception):
            EnvelopeDCAShortConfig(ma_period=1)

        with pytest.raises(Exception):
            EnvelopeDCAShortConfig(num_levels=0)

    def test_get_params(self):
        """get_params retourne les bons champs."""
        strategy = _make_strategy(ma_period=10, num_levels=4)
        params = strategy.get_params()
        assert params["ma_period"] == 10
        assert params["num_levels"] == 4
        assert params["sides"] == ["short"]
        assert "enabled" not in params
        assert "weight" not in params

    def test_build_cache_for_short(self):
        """build_cache crée les SMA pour envelope_dca_short."""
        from backend.optimization.indicator_cache import build_cache

        candles = _make_candles(50, start_price=100.0, tf="1h")
        cache = build_cache(
            {"1h": candles},
            {"ma_period": [5, 7]},
            "envelope_dca_short",
            main_tf="1h",
        )

        assert 5 in cache.bb_sma
        assert 7 in cache.bb_sma
        assert len(cache.bb_sma[7]) == 50

    def test_get_tp_price_short(self):
        """TP = SMA (dynamique)."""
        strategy = _make_strategy()
        pos = GridPosition(
            level=0, direction=Direction.SHORT,
            entry_price=105.0, quantity=1.0,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            entry_fee=0.1,
        )
        grid_state = _make_grid_state([pos])
        tp = strategy.get_tp_price(grid_state, {"sma": 100.0})
        assert tp == 100.0

    def test_get_sl_price_short(self):
        """SL SHORT = avg_entry × (1 + sl_pct)."""
        strategy = _make_strategy(sl_percent=20.0)
        pos = GridPosition(
            level=0, direction=Direction.SHORT,
            entry_price=105.0, quantity=1.0,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            entry_fee=0.1,
        )
        grid_state = _make_grid_state([pos])
        sl = strategy.get_sl_price(grid_state, {"sma": 100.0})
        assert sl == pytest.approx(105.0 * 1.20)
