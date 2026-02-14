"""Tests centralisés : STRATEGY_REGISTRY, GRID_STRATEGIES, factory, _INDICATOR_PARAMS.

Remplace les tests registry dupliqués dans :
- test_optimization.py (TestStrategyRegistry)
- test_new_strategies.py (TestRegistryAndConfig — registry subset)
- test_multi_engine.py (TestWFOIntegration)
- test_envelope_dca_short.py (TestEnvelopeDCAShortIntegration — registry subset)
- test_funding_oi_data.py (TestStrategyRegistry)
"""

import pytest

from backend.optimization import (
    GRID_STRATEGIES,
    STRATEGY_REGISTRY,
    create_strategy_with_params,
    is_grid_strategy,
)
from backend.optimization.walk_forward import _INDICATOR_PARAMS


# ─── Toutes les stratégies ──────────────────────────────────────────────

ALL_STRATEGIES = [
    "vwap_rsi",
    "momentum",
    "funding",
    "liquidation",
    "bollinger_mr",
    "donchian_breakout",
    "supertrend",
    "envelope_dca",
    "envelope_dca_short",
]


@pytest.mark.parametrize("name", ALL_STRATEGIES)
def test_in_registry(name):
    """Chaque stratégie est dans STRATEGY_REGISTRY."""
    assert name in STRATEGY_REGISTRY
    config_cls, strategy_cls = STRATEGY_REGISTRY[name]
    assert config_cls is not None
    assert strategy_cls is not None


def test_registry_excludes_non_optimizable():
    """Les stratégies non optimisables ne sont pas dans le registre."""
    assert "orderflow" not in STRATEGY_REGISTRY


@pytest.mark.parametrize("name", ALL_STRATEGIES)
def test_create_strategy_with_params(name):
    """create_strategy_with_params crée chaque stratégie avec les defaults."""
    strategy = create_strategy_with_params(name, {})
    assert strategy is not None
    assert strategy.name == name


def test_create_strategy_unknown_raises():
    """Stratégie inconnue → ValueError."""
    with pytest.raises(ValueError, match="non optimisable"):
        create_strategy_with_params("unknown", {})


# ─── Stratégies grid ────────────────────────────────────────────────────

GRID_STRATEGY_NAMES = ["envelope_dca", "envelope_dca_short"]
NON_GRID_STRATEGY_NAMES = [s for s in ALL_STRATEGIES if s not in GRID_STRATEGY_NAMES]


@pytest.mark.parametrize("name", GRID_STRATEGY_NAMES)
def test_grid_in_grid_strategies(name):
    """Les stratégies grid sont dans GRID_STRATEGIES."""
    assert name in GRID_STRATEGIES


@pytest.mark.parametrize("name", GRID_STRATEGY_NAMES)
def test_is_grid_strategy_true(name):
    """is_grid_strategy retourne True pour les stratégies grid."""
    assert is_grid_strategy(name) is True


@pytest.mark.parametrize("name", NON_GRID_STRATEGY_NAMES)
def test_is_grid_strategy_false(name):
    """is_grid_strategy retourne False pour les stratégies non-grid."""
    assert is_grid_strategy(name) is False


# ─── _INDICATOR_PARAMS ──────────────────────────────────────────────────

INDICATOR_PARAMS_EXPECTED = {
    "envelope_dca": ["ma_period"],
    "envelope_dca_short": ["ma_period"],
}


@pytest.mark.parametrize(
    "name,expected_params",
    list(INDICATOR_PARAMS_EXPECTED.items()),
)
def test_indicator_params(name, expected_params):
    """_INDICATOR_PARAMS contient les bons paramètres indicateurs."""
    assert name in _INDICATOR_PARAMS
    assert _INDICATOR_PARAMS[name] == expected_params


# ─── Tests spécifiques avec params ──────────────────────────────────────


def test_create_vwap_rsi_with_params():
    """create_strategy_with_params applique les overrides vwap_rsi."""
    strategy = create_strategy_with_params("vwap_rsi", {"rsi_period": 20})
    assert strategy._config.rsi_period == 20


def test_create_vwap_rsi_default_params():
    """create_strategy_with_params avec {} utilise les defaults."""
    strategy = create_strategy_with_params("vwap_rsi", {})
    assert strategy._config.rsi_period == 14


def test_create_momentum_with_params():
    """create_strategy_with_params applique les overrides momentum."""
    strategy = create_strategy_with_params("momentum", {"breakout_lookback": 30})
    assert strategy._config.breakout_lookback == 30


def test_create_funding_with_params():
    """create_strategy_with_params applique les overrides funding."""
    strategy = create_strategy_with_params("funding", {
        "extreme_positive_threshold": 0.05,
        "tp_percent": 0.3,
        "sl_percent": 0.15,
    })
    assert strategy.name == "funding"


def test_funding_mirroring():
    """extreme_negative_threshold = -extreme_positive_threshold automatiquement."""
    strategy = create_strategy_with_params("funding", {
        "extreme_positive_threshold": 0.05,
    })
    assert strategy._config.extreme_negative_threshold == -0.05


def test_create_liquidation_with_params():
    """create_strategy_with_params applique les overrides liquidation."""
    strategy = create_strategy_with_params("liquidation", {
        "oi_change_threshold": 7.0,
        "leverage_estimate": 20,
    })
    assert strategy.name == "liquidation"


def test_create_envelope_dca_with_params():
    """create_strategy_with_params crée un EnvelopeDCAStrategy."""
    from backend.strategies.envelope_dca import EnvelopeDCAStrategy

    params = {
        "ma_period": 10, "num_levels": 4,
        "envelope_start": 0.05, "envelope_step": 0.02,
        "sl_percent": 20.0, "sides": ["long"], "leverage": 6,
    }
    strategy = create_strategy_with_params("envelope_dca", params)
    assert isinstance(strategy, EnvelopeDCAStrategy)
    assert strategy.max_positions == 4


def test_create_envelope_dca_short_with_params():
    """create_strategy_with_params crée un EnvelopeDCAShortStrategy."""
    from backend.strategies.envelope_dca_short import EnvelopeDCAShortStrategy

    params = {
        "ma_period": 10, "num_levels": 4,
        "envelope_start": 0.05, "envelope_step": 0.02,
        "sl_percent": 20.0, "sides": ["short"], "leverage": 6,
    }
    strategy = create_strategy_with_params("envelope_dca_short", params)
    assert isinstance(strategy, EnvelopeDCAShortStrategy)
    assert strategy.name == "envelope_dca_short"
    assert strategy.max_positions == 4
