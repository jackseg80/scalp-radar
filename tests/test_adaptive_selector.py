"""Tests pour l'AdaptiveSelector (Sprint 5b + Hotfix 28a)."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
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

    # Hotfix 28a : défauts explicites (MagicMock retourne truthy sinon)
    config.risk.selector_bypass_at_boot = False
    config.secrets.live_trading = False

    # Strategy configs avec live_eligible
    config.strategies.vwap_rsi.live_eligible = vwap_eligible
    config.strategies.momentum.live_eligible = momentum_eligible
    config.strategies.funding.live_eligible = funding_eligible
    config.strategies.liquidation.live_eligible = liquidation_eligible

    # Grid strategies — live_eligible par défaut
    config.strategies.grid_atr.live_eligible = True
    config.strategies.grid_multi_tf.live_eligible = False
    config.strategies.grid_funding.live_eligible = False
    config.strategies.grid_trend.live_eligible = False
    config.strategies.envelope_dca.live_eligible = False
    config.strategies.envelope_dca_short.live_eligible = False
    config.strategies.bollinger_mr.live_eligible = False
    config.strategies.donchian_breakout.live_eligible = False
    config.strategies.supertrend.live_eligible = False

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


# ─── Tests existants (Sprint 5b) ─────────────────────────────────────────


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
        # Hotfix 28a : nouveaux champs
        assert "bypass_active" in status
        assert "db_trade_counts" in status

    def test_strategie_inconnue_rejetee(self):
        """Stratégie non mappée dans _STRATEGY_CONFIG_ATTR → rejetée."""
        ranking = [FakePerformance(name="unknown_strategy", total_trades=50, net_return_pct=10.0)]
        selector = _make_selector(ranking=ranking, active_symbols={"BTC/USDT"})
        selector.evaluate()
        assert selector.is_allowed("unknown_strategy", "BTC/USDT") is False


# ─── Tests Hotfix 28a — FIX 1 : DB trade counts ─────────────────────────


async def _create_test_db(
    trades: list[tuple[str, int]] | None = None,
) -> "Database":
    """Crée une DB temporaire avec simulation_trades peuplée.

    trades: liste de (strategy_name, count) — insère count trades factices.
    """
    from backend.core.database import Database

    tmpdir = tempfile.mkdtemp()
    db_path = str(Path(tmpdir) / "test.db")
    db = Database(db_path)
    await db.init()

    if trades:
        assert db._conn is not None
        for strategy_name, count in trades:
            for _ in range(count):
                await db._conn.execute(
                    """INSERT INTO simulation_trades
                       (strategy_name, symbol, direction, entry_price, exit_price,
                        quantity, gross_pnl, fee_cost, slippage_cost, net_pnl,
                        exit_reason, entry_time, exit_time)
                       VALUES (?, 'BTC/USDT', 'LONG', 100.0, 101.0,
                               1.0, 1.0, 0.06, 0.05, 0.89,
                               'tp', '2024-01-01T00:00:00', '2024-01-01T01:00:00')""",
                    (strategy_name,),
                )
        await db._conn.commit()

    return db


class TestSelectorDBTrades:
    """FIX 1 : Selector charge les trades historiques depuis la DB."""

    @pytest.mark.asyncio
    async def test_selector_loads_trades_from_db(self):
        """DB a 5 trades grid_atr, runner a 0 → autorisé (5 >= min_trades=3)."""
        db = await _create_test_db([("grid_atr", 5)])
        try:
            ranking = [FakePerformance(
                name="grid_atr", total_trades=0,
                net_return_pct=5.0, profit_factor=1.5,
            )]
            config = _make_config(min_trades=3)
            arena = _make_arena(ranking)

            selector = AdaptiveSelector(arena, config, db=db)
            await selector._load_trade_counts_from_db()
            selector.set_active_symbols({"BTC/USDT"})
            selector.evaluate()

            assert selector._db_trade_counts["grid_atr"] == 5
            assert selector.is_allowed("grid_atr", "BTC/USDT") is True
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_selector_empty_db(self):
        """DB vide → compteurs vides, runner a 0 trades → rejeté."""
        db = await _create_test_db()
        try:
            ranking = [FakePerformance(
                name="grid_atr", total_trades=0,
                net_return_pct=5.0, profit_factor=1.5,
            )]
            config = _make_config(min_trades=3)
            arena = _make_arena(ranking)

            selector = AdaptiveSelector(arena, config, db=db)
            await selector._load_trade_counts_from_db()
            selector.set_active_symbols({"BTC/USDT"})
            selector.evaluate()

            assert selector._db_trade_counts == {}
            assert selector.is_allowed("grid_atr", "BTC/USDT") is False
        finally:
            await db.close()


# ─── Tests Hotfix 28a — FIX 2 : Bypass au boot ──────────────────────────


class TestSelectorBypass:
    """FIX 2 : Bypass selector au boot pour cold start."""

    def test_bypass_allows_all_eligible(self):
        """Bypass actif + LIVE_TRADING → autorise même sans trades/return/PF."""
        ranking = [FakePerformance(
            name="grid_atr", total_trades=0,
            net_return_pct=-2.0, profit_factor=0.5,
        )]
        config = _make_config(min_trades=3)
        config.risk.selector_bypass_at_boot = True
        config.secrets.live_trading = True
        arena = _make_arena(ranking)

        selector = AdaptiveSelector(arena, config)
        selector.set_active_symbols({"BTC/USDT"})
        selector.evaluate()

        assert selector.is_allowed("grid_atr", "BTC/USDT") is True
        assert selector._bypass_active is True

    def test_bypass_ignored_paper_mode(self):
        """Bypass ignoré si live_trading=False."""
        config = _make_config()
        config.risk.selector_bypass_at_boot = True
        config.secrets.live_trading = False
        arena = _make_arena([])

        selector = AdaptiveSelector(arena, config)
        assert selector._bypass_active is False

    def test_bypass_auto_deactivates_all_ready(self):
        """Bypass se désactive quand TOUTES les eligible atteignent min_trades."""
        ranking = [
            FakePerformance(name="grid_atr", total_trades=5, net_return_pct=3.0, profit_factor=1.5),
            FakePerformance(name="vwap_rsi", total_trades=4, net_return_pct=2.0, profit_factor=1.2),
        ]
        config = _make_config(min_trades=3)
        config.risk.selector_bypass_at_boot = True
        config.secrets.live_trading = True
        arena = _make_arena(ranking)

        selector = AdaptiveSelector(arena, config)
        selector.set_active_symbols({"BTC/USDT"})
        selector.evaluate()

        # Toutes ont >= 3 trades → bypass désactivé
        assert selector._bypass_active is False
        # Stratégies toujours autorisées (ajoutées avant désactivation)
        assert selector.is_allowed("grid_atr", "BTC/USDT") is True
        assert selector.is_allowed("vwap_rsi", "BTC/USDT") is True

    def test_bypass_stays_if_not_all_ready(self):
        """Bypass reste actif si une stratégie n'a pas assez de trades."""
        ranking = [
            FakePerformance(name="grid_atr", total_trades=5, net_return_pct=3.0, profit_factor=1.5),
            FakePerformance(name="vwap_rsi", total_trades=1, net_return_pct=-1.0, profit_factor=0.5),
        ]
        config = _make_config(min_trades=3)
        config.risk.selector_bypass_at_boot = True
        config.secrets.live_trading = True
        arena = _make_arena(ranking)

        selector = AdaptiveSelector(arena, config)
        selector.set_active_symbols({"BTC/USDT"})
        selector.evaluate()

        # vwap_rsi n'a que 1 trade < 3 → bypass reste actif
        assert selector._bypass_active is True
        # Les deux restent autorisées grâce au bypass
        assert selector.is_allowed("grid_atr", "BTC/USDT") is True
        assert selector.is_allowed("vwap_rsi", "BTC/USDT") is True


# ─── Tests Hotfix 28a — FIX 3 : exchange_balance property ───────────────


class TestExecutorExchangeBalance:
    """FIX 3 : Executor expose exchange_balance après start."""

    def test_exchange_balance_none_before_start(self):
        """Avant start, exchange_balance est None."""
        from backend.execution.executor import Executor

        config = MagicMock()
        config.secrets.live_trading = True

        risk_mgr = MagicMock()
        notifier = MagicMock()

        executor = Executor(config, risk_mgr, notifier)
        assert executor.exchange_balance is None

    def test_exchange_balance_set_after_fetch(self):
        """Après affectation, exchange_balance retourne la valeur."""
        from backend.execution.executor import Executor

        config = MagicMock()
        config.secrets.live_trading = True

        risk_mgr = MagicMock()
        notifier = MagicMock()

        executor = Executor(config, risk_mgr, notifier)
        executor._exchange_balance = 8500.0
        assert executor.exchange_balance == 8500.0
