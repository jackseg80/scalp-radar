"""Every registered strategy is classified before certification replay."""

from backend.backtesting.certification_capabilities import (
    canonical_certification_capability,
    certification_capability_matrix,
)
from backend.optimization import GRID_STRATEGIES, STRATEGY_REGISTRY


def test_all_18_strategies_have_a_fail_closed_capability_decision():
    matrix = certification_capability_matrix()
    assert len(STRATEGY_REGISTRY) == 18
    assert set(matrix) == set(STRATEGY_REGISTRY)
    assert all(isinstance(item["supported"], bool) and item["reason"] for item in matrix.values())


def test_existing_canonical_portfolio_is_used_only_for_grid_strategies():
    for name in STRATEGY_REGISTRY:
        supported, _ = canonical_certification_capability(name)
        assert supported is (name in GRID_STRATEGIES)


def test_trend_follow_daily_cannot_be_certified_without_live_runner():
    supported, reason = canonical_certification_capability("trend_follow_daily")
    assert supported is False
    assert "no live/canonical runner" in reason
