"""Fail-closed capability registry for the canonical certification replay.

This is deliberately derived from the existing optimization registry and
strategy base classes.  It does not duplicate strategy definitions; it only
states whether the current canonical portfolio path can represent them.
"""

from __future__ import annotations

from backend.optimization import STRATEGY_REGISTRY
from backend.strategies.base_grid import BaseGridStrategy


def canonical_certification_capability(strategy_name: str) -> tuple[bool, str]:
    registered = STRATEGY_REGISTRY.get(strategy_name)
    if registered is None:
        return False, "strategy is not registered for optimization"
    _config_class, strategy_class = registered
    if strategy_class is None:
        return False, "strategy has no live/canonical runner"
    if not issubclass(strategy_class, BaseGridStrategy):
        return False, "canonical shared-account replay for mono-position strategies is pending"
    return True, "canonical grid replay available; intrabar gate still applies"


def certification_capability_matrix() -> dict[str, dict[str, str | bool]]:
    return {
        name: {"supported": result[0], "reason": result[1]}
        for name in sorted(STRATEGY_REGISTRY)
        for result in [canonical_certification_capability(name)]
    }
