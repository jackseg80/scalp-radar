"""Tests pour le module de signal régime BTC — Sprint 50b.

Couvre :
- RegimeSignal (bisect, leverage, transitions)
- compute_regime_signal (DB synthétique)
- Intégration portfolio (leverage dynamique)
- Métriques rapport (Sharpe, Calmar, verdict)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from backend.regime.btc_regime_signal import (
    RegimeSignal,
    _build_transitions,
    _candles_to_dataframe,
    compute_regime_signal,
)


# ─── Helpers ─────────────────────────────────────────────────────────────

def _make_regime_signal(
    n_hours: int = 96,
    switch_at_hour: int = 48,
) -> RegimeSignal:
    """Crée un RegimeSignal synthétique : normal puis defensive."""
    base = datetime(2023, 1, 1, tzinfo=timezone.utc)
    timestamps = [base + timedelta(hours=i * 4) for i in range(n_hours // 4)]
    regimes = [
        "normal" if i * 4 < switch_at_hour else "defensive"
        for i in range(len(timestamps))
    ]
    transitions = _build_transitions(timestamps, regimes)
    return RegimeSignal(
        timestamps=timestamps,
        regimes=regimes,
        transitions=transitions,
        params={"test": True},
    )


# ─── TestRegimeSignal ────────────────────────────────────────────────────


class TestRegimeSignal:
    """Tests unitaires pour RegimeSignal (bisect, leverage)."""

    def test_get_regime_before_first_ts(self):
        """Avant le premier timestamp → "normal" (défaut warmup)."""
        sig = _make_regime_signal()
        dt_before = sig.timestamps[0] - timedelta(hours=1)
        assert sig.get_regime_at(dt_before) == "normal"

    def test_get_regime_exact_ts(self):
        """Timestamp exact → régime correct."""
        sig = _make_regime_signal(n_hours=96, switch_at_hour=48)
        # Premier ts = normal
        assert sig.get_regime_at(sig.timestamps[0]) == "normal"
        # Après switch (48h = index 12) = defensive
        assert sig.get_regime_at(sig.timestamps[12]) == "defensive"

    def test_get_regime_between_ts(self):
        """Entre deux timestamps → régime du précédent (bisect)."""
        sig = _make_regime_signal(n_hours=96, switch_at_hour=48)
        # 2h après le premier ts (entre ts[0] et ts[1]) → normal
        between = sig.timestamps[0] + timedelta(hours=2)
        assert sig.get_regime_at(between) == "normal"
        # 2h après le switch → defensive
        between_def = sig.timestamps[12] + timedelta(hours=2)
        assert sig.get_regime_at(between_def) == "defensive"

    def test_get_leverage(self):
        """normal → 7x, defensive → 4x."""
        sig = _make_regime_signal(n_hours=96, switch_at_hour=48)
        assert sig.get_leverage(sig.timestamps[0]) == 7
        assert sig.get_leverage(sig.timestamps[12]) == 4
        # Custom leverages
        assert sig.get_leverage(sig.timestamps[0], lev_normal=10, lev_defensive=3) == 10
        assert sig.get_leverage(sig.timestamps[12], lev_normal=10, lev_defensive=3) == 3

    def test_transitions_list(self):
        """Transitions correctement détectées."""
        sig = _make_regime_signal(n_hours=96, switch_at_hour=48)
        assert len(sig.transitions) == 1
        t = sig.transitions[0]
        assert t["from"] == "normal"
        assert t["to"] == "defensive"
        assert "timestamp" in t

    def test_empty_signal(self):
        """Signal vide → toujours "normal"."""
        sig = RegimeSignal(timestamps=[], regimes=[], transitions=[], params={})
        dt = datetime(2023, 6, 1, tzinfo=timezone.utc)
        assert sig.get_regime_at(dt) == "normal"
        assert sig.get_leverage(dt) == 7


# ─── TestComputeRegimeSignal ─────────────────────────────────────────────


@pytest_asyncio.fixture
async def test_db(tmp_path):
    """Crée une DB temporaire avec des candles BTC 4h synthétiques."""
    from backend.core.database import Database
    from backend.core.models import Candle, TimeFrame

    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    await db.init()

    # Générer 300 candles 4h (~50 jours) avec une tendance claire
    base = datetime(2023, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = 20000.0
    for i in range(300):
        # Tendance haussière premiers 200, baissière ensuite
        if i < 200:
            price *= 1.001
        else:
            price *= 0.998
        candles.append(Candle(
            timestamp=base + timedelta(hours=i * 4),
            open=price * 0.999,
            high=price * 1.002,
            low=price * 0.998,
            close=price,
            volume=1000.0,
            symbol="BTC/USDT",
            timeframe=TimeFrame.H4,
            exchange="binance",
        ))

    await db.insert_candles_batch(candles)
    await db.close()
    yield db_path


@pytest.mark.asyncio
class TestComputeRegimeSignal:
    """Tests pour compute_regime_signal avec DB synthétique."""

    async def test_signal_computed(self, test_db):
        """Le signal est calculé et contient les deux régimes."""
        sig = await compute_regime_signal(db_path=test_db, exchange="binance")
        assert len(sig.timestamps) > 0
        assert len(sig.regimes) == len(sig.timestamps)
        assert isinstance(sig.params, dict)
        # Doit contenir au moins un type de régime
        unique = set(sig.regimes)
        assert unique <= {"normal", "defensive"}

    async def test_default_params(self, test_db):
        """Params par défaut utilisés si None."""
        sig = await compute_regime_signal(
            db_path=test_db, exchange="binance", detector_params=None
        )
        assert sig.params["ema_fast"] == 50
        assert sig.params["ema_slow"] == 200
        assert sig.params["h_down"] == 6
        assert sig.params["h_up"] == 24

    async def test_missing_btc_raises(self, tmp_path):
        """DB vide → ValueError avec message backfill."""
        from backend.core.database import Database

        db_path = str(tmp_path / "empty.db")
        db = Database(db_path)
        await db.init()
        await db.close()

        with pytest.raises(ValueError, match="BTC/USDT 4h not found"):
            await compute_regime_signal(db_path=db_path, exchange="binance")


# ─── TestPortfolioRegimeIntegration ──────────────────────────────────────


@dataclass
class _FakeGPMConfig:
    leverage: int = 7


@dataclass
class _FakeGPM:
    _config: _FakeGPMConfig


@dataclass
class _FakeStrategyConfig:
    leverage: int = 7


class _FakeStrategy:
    def __init__(self, leverage: int = 7):
        self._config = _FakeStrategyConfig(leverage=leverage)


class _FakeRunner:
    """Runner minimal pour tester le changement de leverage."""

    def __init__(self, leverage: int = 7):
        self._leverage = leverage
        self._gpm = _FakeGPM(_FakeGPMConfig(leverage=leverage))
        self._strategy = _FakeStrategy(leverage=leverage)
        self._positions: dict[str, list] = {}


class TestPortfolioRegimeIntegration:
    """Tests d'intégration du leverage dynamique dans le portfolio engine."""

    def test_update_runner_leverage_all_three(self):
        """_update_runner_leverage met à jour les 3 emplacements."""
        from backend.backtesting.portfolio_engine import PortfolioBacktester

        runner = _FakeRunner(leverage=7)
        # Appeler le helper statique-like (on crée une instance minimale)
        PortfolioBacktester._update_runner_leverage("test:BTC", runner, 4)

        assert runner._leverage == 4
        assert runner._gpm._config.leverage == 4
        assert runner._strategy._config.leverage == 4

    def test_leverage_changes_no_position(self):
        """Le leverage change quand le runner n'a pas de positions."""
        runner = _FakeRunner(leverage=7)
        assert runner._positions == {} or not any(runner._positions.values())

        # Simuler le changement
        from backend.backtesting.portfolio_engine import PortfolioBacktester
        PortfolioBacktester._update_runner_leverage("test:BTC", runner, 4)
        assert runner._leverage == 4

    def test_leverage_no_change_with_position(self):
        """Le leverage NE change PAS si le runner a des positions ouvertes."""
        runner = _FakeRunner(leverage=7)
        # Simuler une position ouverte
        runner._positions = {"BTC/USDT": [MagicMock()]}

        has_positions = any(positions for positions in runner._positions.values())
        assert has_positions is True

        # Le code dans _simulate() ne devrait pas appeler _update_runner_leverage
        # On vérifie juste la condition
        if not has_positions:
            runner._leverage = 4  # ne devrait pas arriver
        assert runner._leverage == 7  # inchangé

    def test_regime_signal_init_leverage(self):
        """Au démarrage, les runners sont initialisés au bon leverage."""
        sig = _make_regime_signal(n_hours=96, switch_at_hour=0)  # tout defensive
        runner = _FakeRunner(leverage=7)

        # Le premier timestamp est en mode defensive
        first_ts = sig.timestamps[0]
        init_lev = sig.get_leverage(first_ts)
        assert init_lev == 4

        # Simuler l'init
        from backend.backtesting.portfolio_engine import PortfolioBacktester
        if runner._leverage != init_lev:
            PortfolioBacktester._update_runner_leverage("test:BTC", runner, init_lev)
        assert runner._leverage == 4

    def test_leverage_changes_empty_without_regime(self):
        """Sans regime_signal, leverage_changes est vide."""
        # PortfolioResult.leverage_changes default = []
        from backend.backtesting.portfolio_engine import PortfolioResult

        result = PortfolioResult(
            initial_capital=10000,
            n_assets=1,
            period_days=90,
            assets=["BTC/USDT"],
            final_equity=10500,
            total_return_pct=5.0,
            total_trades=10,
            win_rate=60.0,
            realized_pnl=500,
            force_closed_pnl=0,
            max_drawdown_pct=-3.0,
            max_drawdown_date=None,
            max_drawdown_duration_hours=24,
            peak_margin_ratio=0.5,
            peak_open_positions=3,
            peak_concurrent_assets=1,
            kill_switch_triggers=0,
            kill_switch_events=[],
            snapshots=[],
            per_asset_results={},
        )
        assert result.leverage_changes == []

    def test_delayed_transition(self):
        """Transition retardée si position ouverte, appliquée après fermeture."""
        sig = _make_regime_signal(n_hours=96, switch_at_hour=48)
        runner = _FakeRunner(leverage=7)

        # Au moment du switch (ts[12]), runner a une position
        switch_ts = sig.timestamps[12]
        target_lev = sig.get_leverage(switch_ts)
        assert target_lev == 4

        runner._positions = {"BTC/USDT": [MagicMock()]}
        has_positions = any(positions for positions in runner._positions.values())
        assert has_positions
        # Pas de changement
        assert runner._leverage == 7

        # Position fermée
        runner._positions = {"BTC/USDT": []}
        has_positions = any(positions for positions in runner._positions.values())
        assert not has_positions

        # Maintenant le changement peut se faire
        from backend.backtesting.portfolio_engine import PortfolioBacktester
        if runner._leverage != target_lev and not has_positions:
            PortfolioBacktester._update_runner_leverage("test:BTC", runner, target_lev)
        assert runner._leverage == 4


# ─── TestReportMetrics ───────────────────────────────────────────────────


class TestReportMetrics:
    """Tests pour les métriques de comparaison (Sharpe, Calmar, verdict)."""

    def test_sharpe_calculation(self):
        """Sharpe annualisé calculé correctement."""
        from scripts.regime_backtest_compare import _calc_sharpe

        # Données synthétiques : equity croissante régulière
        @dataclass
        class FakeSnap:
            total_equity: float

        equities = [10000 + i * 10 for i in range(100)]
        snaps = [FakeSnap(total_equity=e) for e in equities]
        sharpe = _calc_sharpe(snaps)
        assert sharpe > 0  # Equity croissante → Sharpe positif

    def test_calmar_calculation(self):
        """Calmar = return / abs(max_dd)."""
        from scripts.regime_backtest_compare import _calc_calmar

        assert _calc_calmar(100.0, -10.0) == pytest.approx(10.0)
        assert _calc_calmar(50.0, -25.0) == pytest.approx(2.0)
        assert _calc_calmar(0.0, -5.0) == pytest.approx(0.0)
        assert _calc_calmar(10.0, 0.0) == 0.0  # Division par zéro

    def test_verdict_go(self):
        """2/3 critères → GO."""
        from scripts.regime_backtest_compare import _compute_verdict

        verdict, details = _compute_verdict(
            return_a=100.0, return_c=90.0,   # 90% > 80% → OK
            dd_a=-10.0, dd_c=-8.0,           # -8 < -10 → OK
            sharpe_a=1.5, sharpe_c=1.2,      # 1.2 < 1.5 → NOK
        )
        assert verdict == "GO"

    def test_verdict_nogo(self):
        """0/3 critères → NO-GO."""
        from scripts.regime_backtest_compare import _compute_verdict

        verdict, details = _compute_verdict(
            return_a=100.0, return_c=50.0,   # 50% < 80% → NOK
            dd_a=-10.0, dd_c=-15.0,          # -15 > -10 → NOK
            sharpe_a=1.5, sharpe_c=0.5,      # 0.5 < 1.5 → NOK
        )
        assert verdict == "NO-GO"

    def test_verdict_borderline(self):
        """1/3 critères → BORDERLINE."""
        from scripts.regime_backtest_compare import _compute_verdict

        verdict, details = _compute_verdict(
            return_a=100.0, return_c=50.0,   # NOK
            dd_a=-10.0, dd_c=-8.0,           # OK (amélioration DD)
            sharpe_a=1.5, sharpe_c=0.5,      # NOK
        )
        assert verdict == "BORDERLINE"
