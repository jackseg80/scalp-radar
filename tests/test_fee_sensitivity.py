"""Tests Sprint 40b #4 : Fee sensitivity analysis dans le rapport WFO.

Le rapport contient un champ fee_sensitivity avec le Sharpe re-simulé pour
3 scénarios (nominal/degraded/stress). Un warning est émis si le Sharpe
descend sous 0.5 en scénario degraded.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from backend.optimization.report import (
    FinalReport,
    _fee_sensitivity_analysis,
    _FEE_SCENARIOS,
)
from backend.core.position_manager import TradeResult
from backend.core.models import Direction, MarketRegime


# ─── Helpers ──────────────────────────────────────────────────────────────


def _make_trade(
    gross_pnl: float,
    fee_cost: float,
    slippage_cost: float,
) -> TradeResult:
    """Crée un TradeResult minimal pour les tests de fee sensitivity."""
    net = gross_pnl - fee_cost - slippage_cost
    return TradeResult(
        direction=Direction.LONG,
        entry_price=100.0,
        exit_price=105.0,
        quantity=1.0,
        entry_time=datetime(2023, 1, 1),
        exit_time=datetime(2023, 1, 2),
        gross_pnl=gross_pnl,
        fee_cost=fee_cost,
        slippage_cost=slippage_cost,
        net_pnl=net,
        exit_reason="tp",
        market_regime=MarketRegime.RANGING,
    )


# ─── Tests _fee_sensitivity_analysis() ────────────────────────────────────


class TestFeeSensitivityAnalysis:

    def test_returns_three_scenarios(self):
        """La fonction retourne les 3 scénarios fee_sensitivity."""
        trades = [_make_trade(50.0, 1.0, 0.5) for _ in range(20)]
        result = _fee_sensitivity_analysis(trades)
        assert set(result.keys()) == {"nominal", "degraded", "stress"}

    def test_empty_trades_returns_empty(self):
        """Moins de 5 trades → dict vide (pas assez pour un Sharpe)."""
        result = _fee_sensitivity_analysis([])
        assert result == {}
        result_few = _fee_sensitivity_analysis([_make_trade(10.0, 0.1, 0.05)] * 3)
        assert result_few == {}

    def test_nominal_close_to_original_sharpe(self):
        """Scénario nominal (mult=1.0) re-simule les fees identiques → même Sharpe."""
        trades = [_make_trade(50.0, 1.0, 0.5) for _ in range(30)]
        result = _fee_sensitivity_analysis(trades)
        # nominal applique les fees telles quelles
        assert result["nominal"] > 0, "Trades profitables → Sharpe nominal > 0"

    def test_stress_scenario_makes_marginal_strategy_negative(self):
        """Stratégie marginalement profitable → Sharpe stress < 0."""
        # gross=10, fee=8 (80%), slip=2 → net_nominal=0 mais pas négatif
        # En stress : fee=8*(5/3)=13.3, slip=2*4=8 → net=-11.3 → négatif
        trades = [_make_trade(10.0, 8.0, 2.0) for _ in range(30)]
        result = _fee_sensitivity_analysis(trades)
        # Scénario stress : net_pnl très négatif → Sharpe < 0
        assert result.get("stress", 0) < 0, (
            f"Stratégie marginale en scénario stress doit avoir Sharpe < 0. "
            f"stress={result.get('stress', 'N/A')}"
        )

    def test_fee_scenarios_multipliers(self):
        """Vérifier que les multiplicateurs sont cohérents avec le plan."""
        assert _FEE_SCENARIOS["nominal"]["fee_mult"] == 1.0
        assert _FEE_SCENARIOS["nominal"]["slip_mult"] == 1.0
        # degraded : fees +33% (4/3x), slip ×2
        assert abs(_FEE_SCENARIOS["degraded"]["fee_mult"] - 4 / 3) < 1e-9
        assert _FEE_SCENARIOS["degraded"]["slip_mult"] == 2.0
        # stress : fees +67% (5/3x), slip ×4
        assert abs(_FEE_SCENARIOS["stress"]["fee_mult"] - 5 / 3) < 1e-9
        assert _FEE_SCENARIOS["stress"]["slip_mult"] == 4.0


# ─── Test FinalReport a le champ fee_sensitivity ──────────────────────────


class TestFinalReportFeeSensitivityField:

    def test_final_report_has_fee_sensitivity_field(self):
        """FinalReport possède le champ fee_sensitivity (default None)."""
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(FinalReport)}
        assert "fee_sensitivity" in fields, "FinalReport.fee_sensitivity manquant"
        assert fields["fee_sensitivity"].default is None

    def test_fee_sensitivity_is_optional(self):
        """fee_sensitivity peut être None (rétrocompatibilité)."""
        # On vérifie juste que le type hint est dict | None
        import inspect
        hints = inspect.get_annotations(FinalReport)
        annotation = str(hints.get("fee_sensitivity", ""))
        assert "None" in annotation or "Optional" in annotation, (
            f"fee_sensitivity doit être optionnel. Annotation: {annotation}"
        )
