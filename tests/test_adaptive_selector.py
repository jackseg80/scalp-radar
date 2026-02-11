"""Tests pour l'AdaptiveSelector (Sprint 5b)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from backend.execution.adaptive_selector import AdaptiveSelector


# ─── Helpers ───────────────────────────────────────────────────────────────


@dataclass
class FakePerformance:
    """Simule StrategyPerformance sans importer Arena."""

    name: str
    total_trades: int = 10
    net_return_pct: float = 5.0
    profit_factor: float = 1.5
    is_active: bool = True
    capital: float = 10_000.0
    net_pnl: float = 500.0
    win_rate: float = 60.0
    max_drawdown_pct: float = 2.0


def _make_config(
    min_trades: int = 3,
    min_profit_factor: float = 1.0,
    eval_interval: int = 300,
    vwap_eligible: bool = True,
    momentum_eligible: bool = True,
    funding_eligible: bool = False,
    liquidation_eligible: bool = False,
) -> MagicMock:
    """Config mock pour le selector."""
    config = MagicMock()

    # Adaptive selector config
    config.risk.adaptive_selector.min_trades = min_trades
    config.risk.adaptive_selector.min_profit_factor = min_profit_factor
    config.risk.adaptive_selector.eval_interval_seconds = eval_interval

    # Strategy configs avec live_eligible
    config.strategies.vwap_rsi.live_eligible = vwap_eligible
    config.strategies.momentum.live_eligible = momentum_eligible
    config.strategies.funding.live_eligible = funding_eligible
    config.strategies.liquidation.live_eligible = liquidation_eligible

    # Assets
    config.assets = [
        MagicMock(symbol="BTC/USDT"),
        MagicMock(symbol="ETH/USDT"),
        MagicMock(symbol="SOL/USDT"),
    ]

    return config


def _make_arena(ranking: list[FakePerformance] | None = None) -> MagicMock:
    arena = MagicMock()
    arena.get_ranking.return_value = ranking or []
    return arena


def _make_selector(
    ranking: list[FakePerformance] | None = None,
    active_symbols: set[str] | None = None,
    **config_kwargs,
) -> AdaptiveSelector:
    config = _make_config(**config_kwargs)
    arena = _make_arena(ranking)
    selector = AdaptiveSelector(arena, config)
    if active_symbols is not None:
        selector.set_active_symbols(active_symbols)
    return selector


# ─── Tests ────────────────────────────────────────────────────────────────


class TestAdaptiveSelector:
    def test_arena_vide_aucune_strategie_autorisee(self):
        """Arena vide → aucune stratégie autorisée."""
        selector = _make_selector(ranking=[], active_symbols={"BTC/USDT"})
        selector.evaluate()
        assert selector.is_allowed("vwap_rsi", "BTC/USDT") is False

    def test_strategie_performante_autorisee(self):
        """Stratégie live_eligible avec bonnes perfs → autorisée."""
        ranking = [FakePerformance(name="vwap_rsi", total_trades=10, net_return_pct=5.0, profit_factor=1.5)]
        selector = _make_selector(ranking=ranking, active_symbols={"BTC/USDT"})
        selector.evaluate()
        assert selector.is_allowed("vwap_rsi", "BTC/USDT") is True

    def test_live_eligible_false_rejete(self):
        """Stratégie live_eligible=false rejetée même si performante."""
        ranking = [FakePerformance(name="funding", total_trades=20, net_return_pct=10.0, profit_factor=2.0)]
        selector = _make_selector(ranking=ranking, active_symbols={"BTC/USDT"})
        selector.evaluate()
        assert selector.is_allowed("funding", "BTC/USDT") is False

    def test_sous_seuil_min_trades_rejete(self):
        """Pas assez de trades → rejeté."""
        ranking = [FakePerformance(name="vwap_rsi", total_trades=2, net_return_pct=5.0, profit_factor=1.5)]
        selector = _make_selector(ranking=ranking, active_symbols={"BTC/USDT"}, min_trades=3)
        selector.evaluate()
        assert selector.is_allowed("vwap_rsi", "BTC/USDT") is False

    def test_net_return_negatif_rejete(self):
        """Net return négatif → rejeté."""
        ranking = [FakePerformance(name="vwap_rsi", total_trades=10, net_return_pct=-2.0, profit_factor=0.8)]
        selector = _make_selector(ranking=ranking, active_symbols={"BTC/USDT"})
        selector.evaluate()
        assert selector.is_allowed("vwap_rsi", "BTC/USDT") is False

    def test_profit_factor_trop_bas_rejete(self):
        """PF < seuil → rejeté."""
        ranking = [FakePerformance(name="vwap_rsi", total_trades=10, net_return_pct=1.0, profit_factor=0.5)]
        selector = _make_selector(
            ranking=ranking, active_symbols={"BTC/USDT"}, min_profit_factor=1.0,
        )
        selector.evaluate()
        assert selector.is_allowed("vwap_rsi", "BTC/USDT") is False

    def test_kill_switch_simulation_rejete(self):
        """Stratégie inactive (kill switch simulation) → rejetée."""
        ranking = [FakePerformance(name="vwap_rsi", total_trades=10, net_return_pct=5.0, is_active=False)]
        selector = _make_selector(ranking=ranking, active_symbols={"BTC/USDT"})
        selector.evaluate()
        assert selector.is_allowed("vwap_rsi", "BTC/USDT") is False

    def test_evaluation_dynamique_ajout_retrait(self):
        """Évaluation dynamique : ajout puis retrait."""
        arena = _make_arena([])
        config = _make_config()
        selector = AdaptiveSelector(arena, config)
        selector.set_active_symbols({"BTC/USDT"})

        # Phase 1 : vwap_rsi performante → autorisée
        arena.get_ranking.return_value = [
            FakePerformance(name="vwap_rsi", total_trades=5, net_return_pct=3.0, profit_factor=1.2),
        ]
        selector.evaluate()
        assert selector.is_allowed("vwap_rsi", "BTC/USDT") is True

        # Phase 2 : performances dégradées → retirée
        arena.get_ranking.return_value = [
            FakePerformance(name="vwap_rsi", total_trades=8, net_return_pct=-1.0, profit_factor=0.7),
        ]
        selector.evaluate()
        assert selector.is_allowed("vwap_rsi", "BTC/USDT") is False

    def test_symbole_hors_active_rejete(self):
        """Symbole hors _active_symbols → rejeté même si stratégie autorisée."""
        ranking = [FakePerformance(name="vwap_rsi")]
        selector = _make_selector(ranking=ranking, active_symbols={"BTC/USDT"})
        selector.evaluate()
        assert selector.is_allowed("vwap_rsi", "BTC/USDT") is True
        assert selector.is_allowed("vwap_rsi", "ETH/USDT") is False

    def test_set_active_symbols_met_a_jour(self):
        """set_active_symbols met à jour le filtre."""
        ranking = [FakePerformance(name="vwap_rsi")]
        selector = _make_selector(ranking=ranking, active_symbols={"BTC/USDT"})
        selector.evaluate()

        assert selector.is_allowed("vwap_rsi", "ETH/USDT") is False
        selector.set_active_symbols({"BTC/USDT", "ETH/USDT"})
        assert selector.is_allowed("vwap_rsi", "ETH/USDT") is True

    def test_get_status_format(self):
        """get_status() retourne le bon format."""
        ranking = [FakePerformance(name="vwap_rsi")]
        selector = _make_selector(
            ranking=ranking,
            active_symbols={"BTC/USDT", "ETH/USDT"},
            min_trades=3,
            min_profit_factor=1.0,
            eval_interval=300,
        )
        selector.evaluate()

        status = selector.get_status()
        assert "allowed_strategies" in status
        assert "active_symbols" in status
        assert status["min_trades"] == 3
        assert status["min_profit_factor"] == 1.0
        assert status["eval_interval_seconds"] == 300
        assert "vwap_rsi" in status["allowed_strategies"]
        assert "BTC/USDT" in status["active_symbols"]

    def test_strategie_inconnue_rejetee(self):
        """Stratégie non mappée dans _STRATEGY_CONFIG_ATTR → rejetée."""
        ranking = [FakePerformance(name="unknown_strategy", total_trades=50, net_return_pct=10.0)]
        selector = _make_selector(ranking=ranking, active_symbols={"BTC/USDT"})
        selector.evaluate()
        assert selector.is_allowed("unknown_strategy", "BTC/USDT") is False
