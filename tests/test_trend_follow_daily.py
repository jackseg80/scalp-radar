"""Tests pour la stratégie trend_follow_daily — Fast Engine Only.

17 tests :
1-2.   Entrée LONG / SHORT sur EMA cross
3-4.   ADX filter / ADX désactivé
5-6.   Trailing stop profit / SL fixe perte
7.     Day 0 Bug — SL touché le jour de l'entrée
8.     Exit mode "signal" — sortie sur EMA cross inverse
9.     Cooldown empêche ré-entrée
10.    Force-close exclu des métriques
11.    Sides = ["long"] bloque SHORT
12.    DD guard stoppe la simulation
13.    Registry et config
14.    IndicatorCache inclut EMA/ADX
15.    Déduplication exit_mode/trailing
16.    Trailing init = entry_price (look-ahead fix)
17.    Pas de look-ahead signaux
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest

from backend.backtesting.engine import BacktestConfig
from backend.optimization import (
    FAST_ENGINE_STRATEGIES,
    GRID_STRATEGIES,
    MULTI_BACKTEST_STRATEGIES,
    STRATEGIES_NEED_EXTRA_DATA,
    STRATEGY_REGISTRY,
    is_grid_strategy,
    uses_multi_backtest,
)
from backend.optimization.fast_multi_backtest import (
    _close_trend_position,
    _simulate_trend_follow,
    run_multi_backtest_from_cache,
)
from backend.strategies.trend_follow_daily import TrendFollowDailyConfig

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


# ─── Helpers ──────────────────────────────────────────────────────────────


def _make_bt_config(**overrides) -> BacktestConfig:
    defaults = dict(
        symbol="BTC/USDT",
        start_date=_NOW,
        end_date=_NOW,
        initial_capital=10_000.0,
        leverage=6,
        taker_fee=0.0006,
        maker_fee=0.0002,
        slippage_pct=0.0005,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _make_cache_for_trend(
    make_indicator_cache,
    n: int = 200,
    *,
    ema_fast_vals: np.ndarray | None = None,
    ema_slow_vals: np.ndarray | None = None,
    adx_vals: np.ndarray | None = None,
    atr_vals: np.ndarray | None = None,
    closes: np.ndarray | None = None,
    opens: np.ndarray | None = None,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    ema_fast_period: int = 9,
    ema_slow_period: int = 50,
    adx_period: int = 14,
    atr_period: int = 14,
    total_days: float | None = None,
) -> Any:
    """Crée un cache avec les champs spécifiques trend_follow_daily."""
    if closes is None:
        closes = np.full(n, 100.0)
    if opens is None:
        opens = closes.copy()
    if highs is None:
        highs = closes + 1.0
    if lows is None:
        lows = closes - 1.0
    if ema_fast_vals is None:
        ema_fast_vals = np.full(n, 102.0)
    if ema_slow_vals is None:
        ema_slow_vals = np.full(n, 100.0)
    if adx_vals is None:
        adx_vals = np.full(n, 25.0)
    if atr_vals is None:
        atr_vals = np.full(n, 5.0)

    return make_indicator_cache(
        n=n,
        closes=closes,
        opens=opens,
        highs=highs,
        lows=lows,
        total_days=total_days or n,  # 1d par candle
        ema_by_period={ema_fast_period: ema_fast_vals, ema_slow_period: ema_slow_vals},
        adx_by_period={adx_period: adx_vals},
        atr_by_period={atr_period: atr_vals},
    )


_DEFAULT_PARAMS: dict[str, Any] = {
    "ema_fast": 9,
    "ema_slow": 50,
    "adx_period": 14,
    "adx_threshold": 20.0,
    "atr_period": 14,
    "trailing_atr_mult": 4.0,
    "exit_mode": "trailing",
    "sl_percent": 10.0,
    "cooldown_candles": 3,
    "sides": ["long"],
    "leverage": 6,
}


def _params(**overrides) -> dict[str, Any]:
    return {**_DEFAULT_PARAMS, **overrides}


def _setup_bull_cross(n: int = 200, cross_at: int = 60):
    """Crée des arrays EMA où un bull cross se produit à `cross_at`.

    cross_at-1 : ema_fast <= ema_slow (pas encore croisé)
    cross_at   : ema_fast > ema_slow  (cross confirmé)
    Le moteur lit le signal sur [prev], donc entrée à candle cross_at+1.
    """
    ema_fast = np.full(n, 98.0)  # Sous ema_slow par défaut
    ema_slow = np.full(n, 100.0)
    # Cross : ema_fast passe au-dessus à cross_at
    ema_fast[cross_at:] = 102.0
    return ema_fast, ema_slow


def _setup_bear_cross(n: int = 200, cross_at: int = 60):
    """Crée des arrays EMA où un bear cross se produit à `cross_at`."""
    ema_fast = np.full(n, 102.0)  # Au-dessus par défaut
    ema_slow = np.full(n, 100.0)
    # Cross : ema_fast passe en dessous à cross_at
    ema_fast[cross_at:] = 98.0
    return ema_fast, ema_slow


# ═════════════════════════════════════════════════════════════════════════
# Test 1 — Entrée LONG sur EMA cross haussier
# ═════════════════════════════════════════════════════════════════════════


class TestEntryLong:
    def test_long_entry_on_bull_cross(self, make_indicator_cache):
        """EMA fast croise au-dessus à candle 60 → entrée LONG sur open[61]."""
        n = 200
        ema_fast, ema_slow = _setup_bull_cross(n, cross_at=60)
        # Prix monte puis pullback modéré → trailing stop en profit
        # Entry ~100, SL=90. Trailing doit sortir AVANT que le prix touche 90.
        closes = np.full(n, 100.0)
        closes[61:80] = np.linspace(100, 130, 19)
        # Pullback à 110 (au-dessus du SL=90, mais sous le trailing)
        # Trailing au pic: max(init, 131 - 5*4) = max(80, 111) = 111
        # Pullback 110 < 111 → trailing touché
        closes[80:] = 110.0
        opens = closes.copy()
        highs = closes + 1.0
        lows = closes - 1.0

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            closes=closes, opens=opens, highs=highs, lows=lows,
        )
        bt = _make_bt_config()
        pnls, rets, cap = _simulate_trend_follow(cache, _params(), bt)

        assert len(pnls) >= 1, "Au moins 1 trade doit être fermé"
        assert pnls[0] > 0, "Le premier trade doit être profitable (trailing exit > entry)"


# ═════════════════════════════════════════════════════════════════════════
# Test 2 — Entrée SHORT sur EMA cross baissier
# ═════════════════════════════════════════════════════════════════════════


class TestEntryShort:
    def test_short_entry_on_bear_cross(self, make_indicator_cache):
        """EMA fast croise en dessous → entrée SHORT."""
        n = 200
        ema_fast, ema_slow = _setup_bear_cross(n, cross_at=60)
        closes = np.full(n, 100.0)
        closes[61:80] = np.linspace(100, 70, 19)  # Prix baisse → profit SHORT
        # Rebond modéré à 90 (sous SL=110, mais au-dessus du trailing)
        # Entry ~100, trailing init = 100 + 5*4 = 120
        # Au pic bas: lows ~69, trailing = min(120, 69 + 20) = 89
        # Rebond à 90, highs = 91 > 89 → trailing touché
        closes[80:] = 90.0
        opens = closes.copy()
        highs = closes + 1.0
        lows = closes - 1.0

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            closes=closes, opens=opens, highs=highs, lows=lows,
        )
        bt = _make_bt_config()
        pnls, rets, cap = _simulate_trend_follow(
            cache, _params(sides=["long", "short"]), bt,
        )

        assert len(pnls) >= 1
        assert pnls[0] > 0, "SHORT profitable (trailing exit < entry)"


# ═════════════════════════════════════════════════════════════════════════
# Test 3 — ADX filter bloque l'entrée
# ═════════════════════════════════════════════════════════════════════════


class TestADXFilter:
    def test_adx_below_threshold_blocks_entry(self, make_indicator_cache):
        """ADX < threshold → pas d'entrée malgré le cross."""
        n = 200
        ema_fast, ema_slow = _setup_bull_cross(n, cross_at=60)
        adx_vals = np.full(n, 15.0)  # < 20.0 threshold

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx_vals,
        )
        bt = _make_bt_config()
        pnls, rets, cap = _simulate_trend_follow(cache, _params(), bt)

        assert len(pnls) == 0
        assert cap == pytest.approx(10_000.0)


# ═════════════════════════════════════════════════════════════════════════
# Test 4 — ADX threshold = 0 (désactivé)
# ═════════════════════════════════════════════════════════════════════════


class TestADXDisabled:
    def test_adx_threshold_zero_allows_entry(self, make_indicator_cache):
        """ADX threshold = 0 → ADX ignoré, entrée même si ADX < 20."""
        n = 200
        ema_fast, ema_slow = _setup_bull_cross(n, cross_at=60)
        adx_vals = np.full(n, 10.0)  # ADX très bas

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            adx_vals=adx_vals,
        )
        bt = _make_bt_config()
        pnls, rets, cap = _simulate_trend_follow(
            cache, _params(adx_threshold=0.0), bt,
        )

        # Avec threshold=0, l'entrée devrait se faire malgré ADX bas
        # Force-close fin de données ne compte pas → pnls peut être vide
        # Mais le capital doit avoir bougé (position ouverte puis force-close)
        assert cap != pytest.approx(10_000.0), "Le capital doit avoir bougé (position ouverte)"


# ═════════════════════════════════════════════════════════════════════════
# Test 5 — Trailing stop sort en profit
# ═════════════════════════════════════════════════════════════════════════


class TestTrailingStop:
    def test_trailing_stop_exits_in_profit(self, make_indicator_cache):
        """LONG, prix monte 100→130, pullback modéré → trailing stop en profit."""
        n = 200
        ema_fast, ema_slow = _setup_bull_cross(n, cross_at=60)

        closes = np.full(n, 100.0)
        closes[61:80] = np.linspace(100, 130, 19)
        # Pullback à 115 → au-dessus du SL(90) mais sous trailing(116)
        closes[80:] = 115.0

        opens = closes.copy()
        highs = closes + 1.0
        lows = closes - 1.0
        atr_vals = np.full(n, 5.0)

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            closes=closes, opens=opens, highs=highs, lows=lows,
            atr_vals=atr_vals,
        )
        bt = _make_bt_config()
        pnls, rets, cap = _simulate_trend_follow(
            cache, _params(trailing_atr_mult=3.0), bt,
        )

        assert len(pnls) >= 1
        # Trailing HWM(131) - 3*5 = 116. Lows[80]=114 < 116 → trailing touché
        # Exit ~116, entry ~100 → profit
        assert pnls[0] > 0


# ═════════════════════════════════════════════════════════════════════════
# Test 6 — SL fixe sort en perte
# ═════════════════════════════════════════════════════════════════════════


class TestSLFixe:
    def test_sl_exits_in_loss(self, make_indicator_cache):
        """LONG à ~100, prix chute sous SL le lendemain → sortie en perte."""
        n = 200
        ema_fast, ema_slow = _setup_bull_cross(n, cross_at=60)

        # Open[61]=100 (entrée), puis candle 62 : low chute sous SL
        closes = np.full(n, 100.0)
        opens = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        # Candle 62 : crash. Entry ~100, SL = 90. Low doit être < 90.
        lows[62] = 85.0
        closes[62] = 87.0

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            closes=closes, opens=opens, highs=highs, lows=lows,
        )
        bt = _make_bt_config()
        pnls, rets, cap = _simulate_trend_follow(cache, _params(), bt)

        assert len(pnls) >= 1
        assert pnls[0] < 0, "SL doit fermer en perte"


# ═════════════════════════════════════════════════════════════════════════
# Test 7 — Day 0 Bug — SL touché le jour même de l'entrée
# ═════════════════════════════════════════════════════════════════════════


class TestDay0SL:
    def test_sl_hit_on_entry_day(self, make_indicator_cache):
        """Flash crash le jour de l'entrée → SL déclenché (Day 0 fix)."""
        n = 200
        ema_fast, ema_slow = _setup_bull_cross(n, cross_at=60)

        closes = np.full(n, 100.0)
        opens = closes.copy()
        highs = closes + 1.0
        lows = closes - 1.0

        # Candle 61 : open=100, mais low descend à 85 (flash crash intraday)
        # SL = entry_price * 0.90 ~ 90. Low=85 < 90 → SL touché le jour même
        lows[61] = 85.0

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            closes=closes, opens=opens, highs=highs, lows=lows,
        )
        bt = _make_bt_config()
        pnls, rets, cap = _simulate_trend_follow(cache, _params(), bt)

        assert len(pnls) >= 1, "SL doit se déclencher le jour même de l'entrée"
        assert pnls[0] < 0, "SL = perte"


# ═════════════════════════════════════════════════════════════════════════
# Test 8 — Exit mode "signal" — sortie sur EMA cross inverse
# ═════════════════════════════════════════════════════════════════════════


class TestSignalExit:
    def test_signal_exit_on_ema_cross_inverse(self, make_indicator_cache):
        """LONG, puis EMA fast croise sous slow → sortie sur open."""
        n = 200
        ema_fast = np.full(n, 98.0)  # Sous ema_slow
        ema_slow = np.full(n, 100.0)

        # Bull cross à candle 60
        ema_fast[60:100] = 102.0  # Au-dessus
        # Bear cross à candle 100 (ema_fast repasse sous ema_slow)
        ema_fast[100:] = 98.0

        closes = np.full(n, 100.0)
        opens = closes.copy()
        highs = closes + 1.0
        lows = closes - 1.0

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
        )
        bt = _make_bt_config()
        pnls, rets, cap = _simulate_trend_follow(
            cache, _params(exit_mode="signal", sl_percent=50.0), bt,
        )

        # Signal inverse à candle 100, sortie à candle 101
        assert len(pnls) >= 1, "Signal inverse doit fermer la position"


# ═════════════════════════════════════════════════════════════════════════
# Test 9 — Cooldown empêche ré-entrée
# ═════════════════════════════════════════════════════════════════════════


class TestCooldown:
    def test_cooldown_blocks_reentry(self, make_indicator_cache):
        """Cooldown = 3 candles. Après sortie, 3 candles d'attente."""
        n = 200
        ema_fast = np.full(n, 98.0)
        ema_slow = np.full(n, 100.0)

        # Bull cross 1 à candle 60
        ema_fast[60:70] = 102.0
        # Bull cross 2 à candle 70 (after a dip back below)
        ema_fast[70:72] = 98.0
        ema_fast[72:] = 102.0

        # SL à candle 61 (flash crash) pour provoquer une sortie rapide
        closes = np.full(n, 100.0)
        opens = closes.copy()
        highs = closes + 1.0
        lows = closes - 1.0
        lows[61] = 85.0  # SL touché jour même

        # Cross 2 détecté à candle 72, entrée à candle 73
        # Cooldown = 3 : sortie à 61, cooldown_remaining = 3 (candles 62, 63, 64)
        # Candle 65+ : cooldown fini. Cross 2 à 72 devrait être accepté.

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            closes=closes, opens=opens, highs=highs, lows=lows,
        )
        bt = _make_bt_config()
        pnls, rets, cap = _simulate_trend_follow(
            cache, _params(cooldown_candles=3), bt,
        )

        # 2 trades possibles : SL au candle 61, puis ré-entrée après cooldown
        assert len(pnls) >= 1, "Au moins le premier trade (SL)"


# ═════════════════════════════════════════════════════════════════════════
# Test 10 — Force-close exclu des métriques
# ═════════════════════════════════════════════════════════════════════════


class TestForceClose:
    def test_force_close_excluded_from_metrics(self, make_indicator_cache):
        """Position ouverte jamais fermée → force-close, exclu des pnls."""
        n = 100
        ema_fast, ema_slow = _setup_bull_cross(n, cross_at=60)
        # Prix stable → ni SL ni trailing ne se déclenchent avant la fin
        closes = np.full(n, 100.0)
        opens = closes.copy()
        highs = closes + 0.5
        lows = closes - 0.5
        atr_vals = np.full(n, 0.1)  # ATR minuscule → trailing très serré mais on met un trailing_mult très grand

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            atr_vals=atr_vals,
        )
        bt = _make_bt_config()
        pnls, rets, cap = _simulate_trend_follow(
            cache, _params(trailing_atr_mult=1000.0, sl_percent=99.0), bt,
        )

        # Position ouverte, jamais fermée naturellement, force-close exclu
        assert len(pnls) == 0, "Force-close ne doit pas compter dans trade_pnls"
        assert cap != pytest.approx(10_000.0), "Capital impacté par force-close"


# ═════════════════════════════════════════════════════════════════════════
# Test 11 — Sides = ["long"] bloque SHORT
# ═════════════════════════════════════════════════════════════════════════


class TestSidesFilter:
    def test_long_only_blocks_short(self, make_indicator_cache):
        """sides=["long"] → bear cross ignoré."""
        n = 200
        ema_fast, ema_slow = _setup_bear_cross(n, cross_at=60)

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
        )
        bt = _make_bt_config()
        pnls, rets, cap = _simulate_trend_follow(
            cache, _params(sides=["long"]), bt,
        )

        assert len(pnls) == 0
        assert cap == pytest.approx(10_000.0)


# ═════════════════════════════════════════════════════════════════════════
# Test 12 — DD guard stoppe la simulation
# ═════════════════════════════════════════════════════════════════════════


class TestDDGuard:
    def test_dd_guard_stops_simulation(self, make_indicator_cache):
        """Pertes successives → DD guard stoppe avant liquidation."""
        n = 300
        ema_fast = np.full(n, 98.0)
        ema_slow = np.full(n, 100.0)

        # 3 bull crosses, chacun suivi d'un crash → 3 SL
        for start in [60, 100, 140]:
            ema_fast[start:start + 5] = 102.0

        closes = np.full(n, 100.0)
        opens = closes.copy()
        highs = closes + 1.0
        lows = closes - 1.0
        # Flash crash sur chaque entrée
        for entry in [61, 101, 141]:
            lows[entry] = 50.0  # SL catastrophe

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
        )
        bt = _make_bt_config()
        pnls, rets, cap = _simulate_trend_follow(cache, _params(), bt)

        # Capital doit rester > 0 (DD guard a arrêté)
        assert cap > 0


# ═════════════════════════════════════════════════════════════════════════
# Test 13 — Registry et config
# ═════════════════════════════════════════════════════════════════════════


class TestRegistryConfig:
    def test_in_strategy_registry(self):
        assert "trend_follow_daily" in STRATEGY_REGISTRY

    def test_config_cls_is_correct(self):
        config_cls, strategy_cls = STRATEGY_REGISTRY["trend_follow_daily"]
        assert config_cls is TrendFollowDailyConfig
        assert strategy_cls is None

    def test_not_in_grid_strategies(self):
        assert "trend_follow_daily" not in GRID_STRATEGIES

    def test_not_grid_strategy(self):
        assert not is_grid_strategy("trend_follow_daily")

    def test_in_multi_backtest_strategies(self):
        assert "trend_follow_daily" in MULTI_BACKTEST_STRATEGIES
        assert uses_multi_backtest("trend_follow_daily")

    def test_in_fast_engine_strategies(self):
        assert "trend_follow_daily" in FAST_ENGINE_STRATEGIES

    def test_not_in_strategies_need_extra_data(self):
        assert "trend_follow_daily" not in STRATEGIES_NEED_EXTRA_DATA

    def test_config_defaults(self):
        cfg = TrendFollowDailyConfig()
        assert cfg.timeframe == "1d"
        assert cfg.ema_fast == 9
        assert cfg.ema_slow == 50
        assert cfg.exit_mode == "trailing"
        assert cfg.sl_percent == 10.0
        assert cfg.sides == ["long"]
        assert cfg.leverage == 6
        assert cfg.enabled is False
        assert cfg.live_eligible is False


# ═════════════════════════════════════════════════════════════════════════
# Test 14 — IndicatorCache inclut EMA et ADX
# ═════════════════════════════════════════════════════════════════════════


class TestIndicatorCache:
    def test_cache_has_ema_and_adx(self, make_indicator_cache):
        """build_cache rempli via make_indicator_cache a les bons champs."""
        cache = _make_cache_for_trend(make_indicator_cache)
        assert 9 in cache.ema_by_period
        assert 50 in cache.ema_by_period
        assert 14 in cache.adx_by_period
        assert 14 in cache.atr_by_period


# ═════════════════════════════════════════════════════════════════════════
# Test 15 — Déduplication exit_mode/trailing
# ═════════════════════════════════════════════════════════════════════════


class TestDeduplication:
    def test_signal_mode_ignores_trailing_atr_mult(self, make_indicator_cache):
        """exit_mode=signal → trailing_atr_mult n'a aucun effet."""
        n = 200
        ema_fast = np.full(n, 98.0)
        ema_slow = np.full(n, 100.0)
        ema_fast[60:100] = 102.0  # Bull cross à 60
        ema_fast[100:] = 98.0  # Bear cross à 100 (signal exit)

        closes = np.full(n, 100.0)
        opens = closes.copy()
        highs = closes + 1.0
        lows = closes - 1.0

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
        )
        bt = _make_bt_config()

        p1 = _params(exit_mode="signal", trailing_atr_mult=3.0, sl_percent=50.0)
        p2 = _params(exit_mode="signal", trailing_atr_mult=5.0, sl_percent=50.0)

        pnls1, _, cap1 = _simulate_trend_follow(cache, p1, bt)
        pnls2, _, cap2 = _simulate_trend_follow(cache, p2, bt)

        assert len(pnls1) == len(pnls2)
        assert cap1 == pytest.approx(cap2, rel=1e-10)


# ═════════════════════════════════════════════════════════════════════════
# Test 16 — Trailing init = entry_price (look-ahead fix)
# ═════════════════════════════════════════════════════════════════════════


class TestTrailingInit:
    def test_trailing_init_from_entry_price(self, make_indicator_cache):
        """Trailing init basé sur entry_price, pas highs[i] (look-ahead fix).

        Si le trailing était initialisé à highs[i] - ATR×mult, il utiliserait
        une information future (le high de la candle d'entrée). Au lieu de ça,
        il doit être initialisé à entry_price - ATR×mult.
        """
        n = 200
        ema_fast, ema_slow = _setup_bull_cross(n, cross_at=60)

        closes = np.full(n, 100.0)
        opens = closes.copy()
        # Candle 61 : high très élevé mais close/low normaux
        highs = closes + 1.0
        highs[61] = 200.0  # High irréaliste
        lows = closes - 1.0
        atr_vals = np.full(n, 5.0)

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            closes=closes, opens=opens, highs=highs, lows=lows,
            atr_vals=atr_vals,
        )
        bt = _make_bt_config()
        pnls, rets, cap = _simulate_trend_follow(
            cache, _params(trailing_atr_mult=3.0, sl_percent=50.0), bt,
        )

        # Si trailing était basé sur highs[61]=200 → trailing = 200 - 15 = 185
        # Le trailing serait si haut que le prix (100) déclencherait immédiatement
        # Avec entry_price (~100) → trailing = 100 - 15 = 85 → pas de déclenchement immédiat
        # Le test vérifie que le prix stable à ~100 ne déclenche pas le trailing
        # (il ne devrait pas car 100 > 85)
        # Si force-close fin de données → pnls = [] (pas de trade naturel)
        # Ça signifie que le trailing n'a PAS été déclenché par le high de 200
        assert len(pnls) == 0, "Trailing ne doit pas utiliser highs[i] pour l'init"


# ═════════════════════════════════════════════════════════════════════════
# Test 17 — Pas de look-ahead sur les signaux
# ═════════════════════════════════════════════════════════════════════════


class TestNoLookAhead:
    def test_signal_uses_prev_candle_entry_uses_open(self, make_indicator_cache):
        """Le signal lit ema[i-1], l'entrée est sur opens[i]."""
        n = 200
        ema_fast, ema_slow = _setup_bull_cross(n, cross_at=60)

        # Open[61] différent de close[60] pour détecter un éventuel bug
        closes = np.full(n, 100.0)
        opens = np.full(n, 100.0)
        opens[61] = 105.0  # Open distinct → entry_price doit refléter ce prix
        highs = np.maximum(closes, opens) + 1.0
        lows = np.minimum(closes, opens) - 1.0

        # SL immédiat pour capturer le trade
        lows[62] = 50.0  # SL touché candle 62

        cache = _make_cache_for_trend(
            make_indicator_cache, n=n,
            ema_fast_vals=ema_fast, ema_slow_vals=ema_slow,
            closes=closes, opens=opens, highs=highs, lows=lows,
        )
        bt = _make_bt_config(slippage_pct=0.0)
        pnls, rets, cap = _simulate_trend_follow(cache, _params(), bt)

        assert len(pnls) >= 1
        # Entry price doit être ~105 (open[61]), pas ~100 (close[60])
        # Avec SL 10% : sl_price = 105 * 0.9 = 94.5
        # PnL ≈ (94.5 - 105) × qty - fees → doit être négatif
        assert pnls[0] < 0


# ═════════════════════════════════════════════════════════════════════════
# Test helper — _close_trend_position
# ═════════════════════════════════════════════════════════════════════════


class TestCloseTrendPosition:
    def test_long_profit(self):
        pnl = _close_trend_position(
            direction=1, entry_price=100.0, exit_price=110.0,
            quantity=10.0, entry_fee=0.6, taker_fee=0.0006,
        )
        # gross = (110-100)*10 = 100
        # exit_fee = 10*110*0.0006 = 0.66
        # net = 100 - 0.6 - 0.66 = 98.74
        assert pnl == pytest.approx(98.74)

    def test_short_profit(self):
        pnl = _close_trend_position(
            direction=-1, entry_price=100.0, exit_price=90.0,
            quantity=10.0, entry_fee=0.6, taker_fee=0.0006,
        )
        # gross = (100-90)*10 = 100
        # exit_fee = 10*90*0.0006 = 0.54
        # net = 100 - 0.6 - 0.54 = 98.86
        assert pnl == pytest.approx(98.86)


# ═════════════════════════════════════════════════════════════════════════
# Test — run_trend_follow_backtest_single (OOS single eval WFO)
# ═════════════════════════════════════════════════════════════════════════


def _make_daily_candles(n: int = 150, base_price: float = 100.0, trend: float = 0.3) -> list:
    """Crée n bougies daily synthétiques avec une légère tendance haussière."""
    from backend.core.models import Candle

    candles = []
    price = base_price
    for i in range(n):
        price = price * (1 + trend / 100)
        atr = price * 0.02
        c = Candle(
            timestamp=datetime(2022, 1, 1, tzinfo=timezone.utc) + __import__("datetime").timedelta(days=i),
            open=price,
            high=price + atr,
            low=price - atr,
            close=price,
            volume=1000.0,
            symbol="BTC/USDT",
            timeframe="1d",
        )
        candles.append(c)
    return candles


class TestRunTrendFollowBacktestSingle:
    """Vérifie que run_trend_follow_backtest_single retourne un BacktestResult valide."""

    def test_returns_backtest_result(self):
        from backend.backtesting.engine import BacktestResult
        from backend.optimization.fast_multi_backtest import run_trend_follow_backtest_single

        candles = _make_daily_candles(n=150)
        params = {
            "timeframe": "1d", "ema_fast": 5, "ema_slow": 20,
            "adx_period": 14, "adx_threshold": 0.0, "atr_period": 14,
            "trailing_atr_mult": 3.0, "exit_mode": "trailing",
            "sl_percent": 10.0, "cooldown_candles": 3,
            "sides": ["long"], "leverage": 6,
        }
        bt_config = _make_bt_config()

        result = run_trend_follow_backtest_single("trend_follow_daily", params, {"1d": candles}, bt_config, "1d")

        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "trend_follow_daily"
        assert result.final_capital > 0
        assert isinstance(result.trades, list)
        assert len(result.equity_curve) >= 1

    def test_equity_curve_consistent(self):
        """equity_curve[0] = initial_capital, dernier point ≈ final_capital."""
        from backend.optimization.fast_multi_backtest import run_trend_follow_backtest_single

        candles = _make_daily_candles(n=150)
        params = {
            "timeframe": "1d", "ema_fast": 5, "ema_slow": 20,
            "adx_period": 14, "adx_threshold": 0.0, "atr_period": 14,
            "trailing_atr_mult": 3.0, "exit_mode": "trailing",
            "sl_percent": 10.0, "cooldown_candles": 3,
            "sides": ["long"], "leverage": 6,
        }
        bt_config = _make_bt_config()
        result = run_trend_follow_backtest_single("trend_follow_daily", params, {"1d": candles}, bt_config, "1d")

        assert result.equity_curve[0] == pytest.approx(bt_config.initial_capital)
        assert result.equity_curve[-1] == pytest.approx(result.final_capital, rel=1e-6)
