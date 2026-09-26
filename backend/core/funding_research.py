"""Pre-registered synthetic funding assumptions, never historical observations."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict


FUNDING_RESEARCH_SCENARIOS = (
    "central", "positive_stress", "negative_stress",
    "positive_shocks", "negative_shocks",
)


class ResearchFundingSpec(BaseModel):
    """Fixed v1 family; changing assumptions requires a new version/study."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    profile: Literal["boltrend_funding_v1"] = "boltrend_funding_v1"
    scenario: Literal[
        "central", "positive_stress", "negative_stress",
        "positive_shocks", "negative_shocks",
    ] = "central"
    central_rate_pct: Literal[0.01] = 0.01
    stress_magnitude_pct: Literal[0.03] = 0.03
    shock_magnitude_pct: Literal[0.10] = 0.10
    shock_first_days_utc: Literal[7] = 7
    settlement_hours_utc: tuple[Literal[0], Literal[8], Literal[16]] = (0, 8, 16)
    evidence_kind: Literal["synthetic_research_only"] = "synthetic_research_only"

    def rate_pct(self, timestamp: datetime) -> float:
        if timestamp.tzinfo is None:
            raise ValueError("Synthetic funding requires a timezone-aware timestamp")
        utc = timestamp.astimezone(timezone.utc)
        if self.scenario.endswith("stress"):
            sign = -1 if self.scenario.startswith("negative") else 1
            return sign * self.stress_magnitude_pct
        if self.scenario.endswith("shocks") and utc.day <= self.shock_first_days_utc:
            sign = -1 if self.scenario.startswith("negative") else 1
            return sign * self.shock_magnitude_pct
        return self.central_rate_pct

    def for_scenario(self, scenario: str) -> "ResearchFundingSpec":
        return type(self).model_validate({**self.model_dump(), "scenario": scenario})


def require_central_research_strategy(strategy: str, spec: ResearchFundingSpec) -> None:
    if strategy != "grid_boltrend" or spec.scenario != "central":
        raise ValueError("Funding research snapshot/WFO requires grid_boltrend central")


def require_observed_funding(manifest: dict) -> None:
    if manifest.get("metadata", {}).get("execution_spec", {}).get("research_funding"):
        raise ValueError("Synthetic funding is RESEARCH_ONLY and cannot be certified")
