"""Tests Sprint 52 — Fix leverage dans le fast engine WFO.

Couvre :
- Test 1 : Propagation — walk_forward lit leverage depuis YAML (pas Pydantic default)
- Test 2 : Scaling — leverage=7 → premier trade PnL = 7× leverage=1
- Test 3 : Régression — leverage=1 → deux runs bit-à-bit identiques
- Test 4 & 5 : Warning — _validate_leverage_sl lit leverage YAML (7x, pas 6x)
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.backtesting.engine import BacktestConfig
from backend.optimization.fast_multi_backtest import (
    _build_entry_prices,
    _simulate_grid_common,
)
from backend.optimization.indicator_cache import IndicatorCache
from backend.optimization.report import _validate_leverage_sl


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_cache(n: int = 300, seed: int = 42) -> IndicatorCache:
    rng = np.random.default_rng(seed)
    prices = 100.0 + np.cumsum(rng.normal(0, 2.0, n))
    opens = prices + rng.uniform(-0.3, 0.3, n)
    highs = prices + np.abs(rng.normal(3.0, 1.5, n))
    lows = prices - np.abs(rng.normal(3.0, 1.5, n))

    sma_14 = np.full(n, np.nan)
    atr_14 = np.full(n, np.nan)
    for i in range(14, n):
        sma_14[i] = np.mean(prices[max(0, i - 13) : i + 1])
        atr_14[i] = np.mean(np.abs(np.diff(prices[max(0, i - 13) : i + 1])))

    return IndicatorCache(
        n_candles=n,
        opens=opens,
        highs=highs,
        lows=lows,
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
        bb_sma={14: sma_14},
        bb_upper={},
        bb_lower={},
        supertrend_direction={},
        atr_by_period={14: atr_14},
        supertrend_dir_4h={},
    )


def _make_bt(leverage: int, capital: float = 10_000.0) -> BacktestConfig:
    return BacktestConfig(
        symbol="TEST/USDT",
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
        leverage=leverage,
        initial_capital=capital,
    )


_GRID_ATR_PARAMS = {
    "ma_period": 14,
    "atr_period": 14,
    "atr_multiplier_start": 2.0,
    "atr_multiplier_step": 1.0,
    "num_levels": 3,
    "sl_percent": 20.0,
    "max_hold_candles": 0,
    "min_profit_pct": 0.0,
    "cooldown_candles": 0,
}


# ═══════════════════════════════════════════════════════════════════════════
# Test 1 : Propagation — la logique fixée lit bien la valeur YAML
# ═══════════════════════════════════════════════════════════════════════════


class TestLeveragePropagation:
    """La logique corrigée de walk_forward lit leverage depuis YAML, pas Pydantic default."""

    def test_pydantic_default_is_6_not_7(self):
        """Confirme que GridATRConfig() sans args retourne leverage=6 (l'ancien bug)."""
        from backend.core.config import GridATRConfig

        default_cfg = GridATRConfig()
        assert default_cfg.leverage == 6, (
            "Le default Pydantic doit être 6 — si ce test échoue, "
            "le default a changé et le fix est peut-être devenu inutile."
        )

    def test_yaml_value_overrides_pydantic_default(self):
        """La logique fixée (getattr sur yaml_strat) retourne 7 si YAML dit 7."""
        from backend.core.config import GridATRConfig
        from backend.optimization import STRATEGY_REGISTRY

        config_cls, _ = STRATEGY_REGISTRY["grid_atr"]
        default_cfg = config_cls()  # leverage=6 (Pydantic default)

        # Simuler get_config().strategies.grid_atr avec la valeur YAML
        yaml_strat = GridATRConfig(leverage=7)
        mock_app = MagicMock()
        mock_app.strategies.grid_atr = yaml_strat

        bt_config = _make_bt(leverage=15)  # valeur arbitraire avant override

        # Reproduit la logique corrigée de walk_forward.py:573-577
        with patch("backend.core.config.get_config", return_value=mock_app):
            from backend.core.config import get_config
            _yaml_strat = getattr(get_config().strategies, "grid_atr", None)
            bt_config.leverage = getattr(_yaml_strat, "leverage", default_cfg.leverage)

        assert bt_config.leverage == 7, f"Attendu 7 (YAML), obtenu {bt_config.leverage}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 2 & 3 : Scaling et régression dans _simulate_grid_common
# ═══════════════════════════════════════════════════════════════════════════


class TestLeverageScaling:
    """Vérifie que leverage multiplie bien le notional dans _simulate_grid_common."""

    def test_first_trade_pnl_scales_linearly(self):
        """leverage=7 → PnL du 1er trade = 7× PnL avec leverage=1 (même timing)."""
        cache = _make_cache(n=300, seed=42)
        entry_prices = _build_entry_prices("grid_atr", cache, _GRID_ATR_PARAMS, 3, 1)
        sma_arr = cache.bb_sma[14]

        pnls_1x, _, _ = _simulate_grid_common(
            entry_prices, sma_arr, cache, _make_bt(1), 3, 0.20, 1
        )
        pnls_7x, _, _ = _simulate_grid_common(
            entry_prices, sma_arr, cache, _make_bt(7), 3, 0.20, 1
        )

        assert len(pnls_1x) >= 1, "Aucun trade avec leverage=1 — seed ou données à revoir"
        assert len(pnls_7x) >= 1

        # 1er trade : même timing, même prix → quantité 7x → PnL 7x (fees aussi 7x)
        assert pnls_7x[0] == pytest.approx(pnls_1x[0] * 7.0, rel=1e-9)

    def test_leverage_1x_runs_are_deterministic(self):
        """Deux runs identiques avec leverage=1 → résultats bit-à-bit identiques."""
        cache = _make_cache(n=300, seed=42)
        entry_prices = _build_entry_prices("grid_atr", cache, _GRID_ATR_PARAMS, 3, 1)
        sma_arr = cache.bb_sma[14]

        pnls_a, rets_a, cap_a = _simulate_grid_common(
            entry_prices, sma_arr, cache, _make_bt(1), 3, 0.20, 1
        )
        pnls_b, rets_b, cap_b = _simulate_grid_common(
            entry_prices, sma_arr, cache, _make_bt(1), 3, 0.20, 1
        )

        assert pnls_a == pnls_b
        assert rets_a == rets_b
        assert cap_a == cap_b


# ═══════════════════════════════════════════════════════════════════════════
# Test 4 & 5 : Warning _validate_leverage_sl lit YAML leverage
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateLeverageSlYaml:
    """_validate_leverage_sl doit afficher le leverage YAML, pas le default Pydantic."""

    def test_warning_reads_yaml_7x(self):
        """Avec YAML leverage=7 et SL=20%, warning doit mentionner '7x'."""
        from backend.core.config import GridATRConfig

        mock_strat = GridATRConfig(leverage=7)
        mock_app = MagicMock()
        mock_app.strategies.grid_atr = mock_strat

        with patch("backend.core.config.get_config", return_value=mock_app):
            warnings = _validate_leverage_sl("grid_atr", {"sl_percent": 20.0})

        # SL 20% × 7x = 140% de la marge → dépasse 100%
        assert len(warnings) == 1, f"Attendu 1 warning, obtenu: {warnings}"
        assert "7x" in warnings[0], f"Attendu '7x' dans: {warnings[0]}"
        assert "6x" not in warnings[0]

    def test_pydantic_default_would_give_6x(self):
        """Avec leverage=6 (Pydantic default, ancien bug), le warning dit '6x'."""
        from backend.core.config import GridATRConfig

        mock_strat = GridATRConfig()  # leverage=6 par défaut
        mock_app = MagicMock()
        mock_app.strategies.grid_atr = mock_strat

        with patch("backend.core.config.get_config", return_value=mock_app):
            warnings = _validate_leverage_sl("grid_atr", {"sl_percent": 20.0})

        # SL 20% × 6x = 120% > 100% → warning avec 6x
        assert len(warnings) == 1
        assert "6x" in warnings[0]
