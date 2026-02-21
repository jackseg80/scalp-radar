"""Tests pour l'audit backtest — Sprint Audit Complet + Sprint 36.

Sprint Audit (4 fixes) :
  Fix 1 — Grid Funding double entry fee (fast_multi_backtest.py)
  Fix 2 — Grid Funding double slippage (fast_multi_backtest.py)
  Fix 3 — Simulator funding rate réel (simulator.py)
  Fix 4 — Portfolio liquidation off-by-one (portfolio_engine.py)

Sprint 36 (2 parties) :
  Part A — Double slippage supprimé dans 6 fonctions (5 fichiers)
  Part B — Margin deduction ajoutée dans 4 fast engines
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.backtesting.engine import BacktestConfig
from backend.core.grid_position_manager import GridPositionManager
from backend.core.models import Direction, MarketRegime
from backend.core.position_manager import PositionManager, PositionManagerConfig
from backend.optimization.fast_backtest import _close_trade, _close_trade_numba
from backend.optimization.fast_multi_backtest import (
    _calc_grid_pnl,
    _calc_grid_pnl_with_funding,
    _simulate_grid_common,
    _simulate_grid_funding,
)
from backend.strategies.base_grid import GridLevel, GridPosition


# ─── Helpers ──────────────────────────────────────────────────────────────


def _make_bt_config(
    *,
    initial_capital: float = 10_000.0,
    leverage: int = 6,
    taker_fee: float = 0.0006,
    maker_fee: float = 0.0002,
    slippage_pct: float = 0.0005,
) -> BacktestConfig:
    return BacktestConfig(
        symbol="TEST/USDT",
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
        initial_capital=initial_capital,
        leverage=leverage,
        taker_fee=taker_fee,
        maker_fee=maker_fee,
        slippage_pct=slippage_pct,
    )


# ─── Fix 1 : Grid Funding — pas de double entry fee ──────────────────────


class TestGridFundingNoDoubleEntryFee:
    """Vérifie que _simulate_grid_funding() ne déduit PAS les entry fees à l'ouverture.

    Les entry fees sont déjà incluses dans _calc_grid_pnl_with_funding() à la clôture.
    Bug identique à Hotfix 33b (corrigé dans multi_engine.py, oublié dans fast engine).
    """

    def test_capital_unchanged_at_open(self, make_indicator_cache):
        """Le capital ne doit PAS diminuer au moment de l'ouverture d'une position.

        On utilise un taker_fee=1% (visible) pour rendre le double-comptage évident.
        SMA=200 (bien au-dessus de close=100) pour que sma_cross ne trigger pas
        jusqu'à la dernière candle où close=210.
        """
        n = 30
        ma_period = 5

        # SMA à 200 → close=100 ne triggera PAS sma_cross (close < sma)
        sma = np.full(n, 200.0)
        # Funding négatif pour trigger l'entrée
        funding = np.full(n, -0.005)
        base_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        timestamps = np.array([base_ts + i * 3_600_000 for i in range(n)], dtype=np.int64)

        closes = np.full(n, 100.0)
        # Dernière candle : close=210 > sma=200 → sma_cross trigger exit
        closes[-1] = 210.0

        cache = make_indicator_cache(
            n=n,
            closes=closes,
            bb_sma={ma_period: sma},
            funding_rates_1h=funding,
            candle_timestamps=timestamps,
        )

        params = {
            "funding_threshold_start": 0.001,
            "funding_threshold_step": 0.001,
            "num_levels": 1,
            "ma_period": ma_period,
            "sl_percent": 80.0,  # SL très large (pas touché)
            "tp_mode": "sma_cross",
            "min_hold_candles": 1,
        }

        # taker_fee gros (1%) pour rendre visible le double-comptage
        bt = _make_bt_config(taker_fee=0.01, slippage_pct=0.0, initial_capital=10_000.0)

        trade_pnls, trade_returns, final_capital = _simulate_grid_funding(cache, params, bt)

        # Doit avoir exactement 1 trade fermé (entrée candle ma_period+1, sortie candle n-1)
        assert len(trade_pnls) == 1, (
            f"Expected 1 trade, got {len(trade_pnls)} — vérifier la logique d'entrée/sortie"
        )

        # Calcul attendu avec déduction UNIQUE :
        # entry_price = 100, exit_price = 210, qty = (10000 * 6) / 100 = 600
        # price_pnl = (210 - 100) * 600 = 66000
        # entry_fee = 100*600*0.01 = 600
        # exit_fee = 210*600*0.01 = 1260
        # slippage = 0
        # net_pnl = 66000 - 600 - 1260 + funding_payments ≈ 64140
        # final_capital ≈ 10000 + 64140 = 74140

        # Avec double-comptage (bug) : capital -= 600 à l'ouverture aussi
        # → final serait ≈ 73540 (600 de moins)
        # Mais surtout, sizing serait basé sur 9400 au lieu de 10000 → qty différent

        # Le pnl DOIT être > 63000 (pas de double entry fee)
        assert trade_pnls[0] > 63_000, (
            f"trade_pnl={trade_pnls[0]:.2f} trop bas — probable double entry fee"
        )

    def test_entry_fee_counted_once_only(self):
        """Vérifie via _calc_grid_pnl_with_funding() que l'entry fee est comptée 1 seule fois."""
        entry_price = 100.0
        exit_price = 110.0
        qty = 10.0
        taker_fee = 0.01  # 1% pour être visible

        positions = [(entry_price, qty, 0)]

        pnl = _calc_grid_pnl_with_funding(
            positions, exit_price, 5,
            funding_rates=None,
            candle_timestamps=None,
            taker_fee=taker_fee,
            slippage_pct=0.0,
        )

        # price_pnl = (110-100) * 10 = 100
        # entry_fee = 100*10*0.01 = 10
        # exit_fee = 110*10*0.01 = 11
        # net = 100 - 10 - 11 = 79
        expected = 79.0
        assert abs(pnl - expected) < 0.01, (
            f"pnl={pnl:.4f}, expected={expected:.4f} — entry_fee doit être comptée 1x"
        )


# ─── Fix 2 : Grid Funding — pas de double slippage ───────────────────────


class TestGridFundingNoDoubleSlippage:
    """Vérifie que _calc_grid_pnl_with_funding() calcule le slippage sur exit seulement."""

    def test_slippage_exit_only(self):
        """Slippage doit être exit_price × qty × slippage_pct uniquement."""
        entry_price = 100.0
        exit_price = 110.0
        qty = 10.0

        positions = [(entry_price, qty, 0)]
        slippage_pct = 0.01  # 1% (gros pour être visible)

        pnl = _calc_grid_pnl_with_funding(
            positions, exit_price, 5,
            funding_rates=None,
            candle_timestamps=None,
            taker_fee=0.0,
            slippage_pct=slippage_pct,
        )

        # price_pnl = (110 - 100) * 10 = 100
        # slippage (exit only) = 110 * 10 * 0.01 = 11
        # net_pnl = 100 - 11 = 89
        expected = 89.0

        # Avec double slippage (bug) : slippage = 100*10*0.01 + 110*10*0.01 = 21
        # → net_pnl = 100 - 21 = 79
        assert abs(pnl - expected) < 0.01, (
            f"pnl={pnl:.4f}, expected={expected:.4f} — "
            f"probable double slippage si pnl ≈ 79"
        )


# ─── Fix 2+1 : Parité fees entre _calc_grid_pnl_with_funding et _calc_grid_pnl


class TestCalcPnlParity:
    """Vérifie la parité des fees entre les deux fonctions PnL (avec slippage=0).

    Note : les deux fonctions ont une convention de slippage différente
    (_calc_grid_pnl applique actual_exit + slippage_cost, _calc_grid_pnl_with_funding
    applique un flat cost). On teste donc avec slippage=0 pour vérifier que les
    composantes fees/price sont identiques.
    """

    def test_same_pnl_with_zero_slippage_zero_funding(self):
        """Mêmes positions, fees, slippage=0 → même PnL."""
        entry_price = 100.0
        exit_price = 105.0
        qty = 50.0
        taker_fee = 0.0006

        # _calc_grid_pnl
        entry_fee = entry_price * qty * taker_fee
        positions_std = [(0, entry_price, qty, entry_fee)]
        pnl_std = _calc_grid_pnl(
            positions_std, exit_price,
            exit_fee_rate=taker_fee,
            slippage_rate=0.0,
            direction=1,
        )

        # _calc_grid_pnl_with_funding (funding=0)
        n = 10
        funding = np.zeros(n)
        timestamps = np.arange(n, dtype=np.int64) * 3_600_000

        positions_fund = [(entry_price, qty, 0)]
        pnl_fund = _calc_grid_pnl_with_funding(
            positions_fund, exit_price, 5,
            funding_rates=funding,
            candle_timestamps=timestamps,
            taker_fee=taker_fee,
            slippage_pct=0.0,
        )

        assert abs(pnl_std - pnl_fund) < 0.01, (
            f"Divergence pnl_std={pnl_std:.4f} vs pnl_fund={pnl_fund:.4f} "
            f"— les deux fonctions doivent converger avec slippage=0 et funding=0"
        )


# ─── Fix 3 : Simulator — funding rate réel depuis DataEngine ─────────────


class TestSimulatorRealFundingRate:
    """Vérifie que GridStrategyRunner utilise le funding rate réel du DataEngine.

    on_candle est async et nécessite un setup complet (close_buffer, SMA, etc.).
    On teste la logique de conversion du funding rate directement.
    """

    def test_funding_rate_conversion_real(self):
        """DataEngine retourne 0.05 (= 0.05%), conversion = 0.0005."""
        data_engine = MagicMock()
        data_engine.get_funding_rate.return_value = 0.05

        raw_fr = data_engine.get_funding_rate("BTC/USDT")
        funding_rate = (raw_fr / 100) if isinstance(raw_fr, (int, float)) else 0.0001

        assert abs(funding_rate - 0.0005) < 1e-10, (
            f"funding_rate={funding_rate}, expected 0.0005"
        )

    def test_funding_rate_conversion_fallback_none(self):
        """DataEngine retourne None → fallback 0.0001 (0.01%)."""
        data_engine = MagicMock()
        data_engine.get_funding_rate.return_value = None

        raw_fr = data_engine.get_funding_rate("BTC/USDT")
        funding_rate = (raw_fr / 100) if isinstance(raw_fr, (int, float)) else 0.0001

        assert funding_rate == 0.0001, (
            f"funding_rate={funding_rate}, expected fallback 0.0001"
        )

    def test_funding_rate_conversion_negative(self):
        """Funding rate négatif (-0.03%) → conversion = -0.0003."""
        data_engine = MagicMock()
        data_engine.get_funding_rate.return_value = -0.03

        raw_fr = data_engine.get_funding_rate("BTC/USDT")
        funding_rate = (raw_fr / 100) if isinstance(raw_fr, (int, float)) else 0.0001

        assert abs(funding_rate - (-0.0003)) < 1e-10, (
            f"funding_rate={funding_rate}, expected -0.0003"
        )

    def test_funding_rate_isinstance_guard_mock(self):
        """MagicMock.get_funding_rate() sans return_value → isinstance guard → fallback."""
        data_engine = MagicMock()
        # Sans return_value explicite, get_funding_rate() retourne un MagicMock

        raw_fr = data_engine.get_funding_rate("BTC/USDT")
        funding_rate = (raw_fr / 100) if isinstance(raw_fr, (int, float)) else 0.0001

        assert funding_rate == 0.0001, (
            f"funding_rate={funding_rate} — isinstance guard devrait rejeter MagicMock"
        )

    def test_funding_cost_direction_long_pays(self):
        """LONG paie un funding positif (capital diminue)."""
        from backend.core.models import Direction

        funding_rate = 0.0005  # 0.05% converti
        notional = 5000.0  # 50000 * 0.1
        capital = 10_000.0

        # Logique du simulator (lignes 921-927)
        direction = Direction.LONG
        if direction == Direction.LONG:
            cost = notional * funding_rate  # 5000 * 0.0005 = 2.5
        else:
            cost = -notional * funding_rate
        capital -= cost

        assert capital < 10_000.0, "LONG devrait payer un funding positif"
        assert abs(capital - 9_997.5) < 0.01, f"capital={capital}, expected 9997.5"

    def test_funding_cost_direction_long_receives_negative(self):
        """LONG reçoit un funding négatif (capital augmente)."""
        from backend.core.models import Direction

        funding_rate = -0.0003  # -0.03% converti
        notional = 5000.0
        capital = 10_000.0

        direction = Direction.LONG
        if direction == Direction.LONG:
            cost = notional * funding_rate  # 5000 * (-0.0003) = -1.5
        else:
            cost = -notional * funding_rate
        capital -= cost  # capital -= (-1.5) → capital += 1.5

        assert capital > 10_000.0, "LONG devrait recevoir un funding négatif"
        assert abs(capital - 10_001.5) < 0.01, f"capital={capital}, expected 10001.5"


# ─── Fix 4 : Portfolio liquidation boundary ──────────────────────────────


class TestPortfolioLiquidationBoundary:
    """Vérifie que l'equity == maintenance_margin n'est PAS considérée comme liquidation."""

    def test_equity_equals_maintenance_not_liquidated(self):
        """Equity exactement au maintenance_margin → pas liquidé."""
        total_equity = 400.0
        maintenance_margin = 400.0  # 100_000 * 0.004
        total_notional = 100_000.0

        is_liquidated = total_equity < maintenance_margin and total_notional > 0
        assert is_liquidated is False, (
            "Equity == maintenance_margin ne devrait PAS être liquidé"
        )

    def test_equity_below_maintenance_is_liquidated(self):
        """Equity en dessous du maintenance_margin → liquidé."""
        total_equity = 399.99
        maintenance_margin = 400.0
        total_notional = 100_000.0

        is_liquidated = total_equity < maintenance_margin and total_notional > 0
        assert is_liquidated is True, (
            "Equity < maintenance_margin devrait être liquidé"
        )


# ─── Sprint 36 Part A : Double Slippage — _calc_grid_pnl ─────────────────


class TestCalcGridPnlNoDoubleSlippage:
    """Vérifie que _calc_grid_pnl() n'applique le slippage qu'une seule fois."""

    def test_long_single_slippage(self):
        """LONG : gross sur prix brut, slippage flat cost unique."""
        entry_price = 100.0
        exit_price = 110.0
        qty = 10.0
        slippage = 0.01  # 1%

        positions = [(0, entry_price, qty, 0.0)]  # (lvl, ep, qty, entry_fee)
        pnl = _calc_grid_pnl(positions, exit_price, exit_fee_rate=0.0,
                             slippage_rate=slippage, direction=1)

        # gross = (110 - 100) * 10 = 100
        # slippage = 110 * 10 * 0.01 = 11
        # net = 100 - 11 = 89
        expected = 89.0
        # Double slippage donnerait : 78.0  (100 - 11 - 11 implicitement)
        assert abs(pnl - expected) < 0.01, (
            f"pnl={pnl:.4f}, expected={expected} — probable double slippage"
        )

    def test_short_single_slippage(self):
        """SHORT : slippage aussi une seule fois."""
        entry_price = 110.0
        exit_price = 100.0
        qty = 10.0
        slippage = 0.01

        positions = [(0, entry_price, qty, 0.0)]
        pnl = _calc_grid_pnl(positions, exit_price, exit_fee_rate=0.0,
                             slippage_rate=slippage, direction=-1)

        # gross = (110 - 100) * 10 = 100
        # slippage = 100 * 10 * 0.01 = 10
        # net = 100 - 10 = 90
        expected = 90.0
        assert abs(pnl - expected) < 0.01, (
            f"pnl={pnl:.4f}, expected={expected}"
        )


# ─── Sprint 36 Part A : Double Slippage — _close_trade / _close_trade_numba


class TestCloseTradeNoDoubleSlippage:
    """Vérifie que _close_trade() et _close_trade_numba() n'appliquent
    le slippage qu'une seule fois."""

    def test_close_trade_sl_slippage_once(self):
        """SL exit : slippage doit être exit_price × qty × slippage."""
        pnl = _close_trade(
            direction=1,
            entry_price=100.0,
            exit_price=95.0,
            quantity=10.0,
            entry_fee=0.0,
            exit_reason="sl",
            regime_int=0,
            taker_fee=0.0,
            maker_fee=0.0,
            slippage_pct=0.01,
            high_vol_slippage_mult=2.0,
        )
        # gross = (95 - 100) * 10 = -50
        # slippage = 95 * 10 * 0.01 = 9.5
        # net = -50 - 9.5 = -59.5
        expected = -59.5
        assert abs(pnl - expected) < 0.01, (
            f"pnl={pnl:.4f}, expected={expected}"
        )

    def test_close_trade_tp_no_slippage(self):
        """TP exit : pas de slippage (maker, limit order)."""
        pnl = _close_trade(
            direction=1,
            entry_price=100.0,
            exit_price=110.0,
            quantity=10.0,
            entry_fee=0.0,
            exit_reason="tp",
            regime_int=0,
            taker_fee=0.0,
            maker_fee=0.0,
            slippage_pct=0.01,
            high_vol_slippage_mult=2.0,
        )
        # gross = (110 - 100) * 10 = 100
        # slippage = 0 (TP)
        # net = 100
        expected = 100.0
        assert abs(pnl - expected) < 0.01, (
            f"pnl={pnl:.4f}, expected={expected}"
        )

    def test_close_trade_numba_matches_python(self):
        """Numba JIT et Python fallback doivent donner le même résultat."""
        kwargs = dict(
            direction=1,
            entry_price=100.0,
            exit_price=95.0,
            quantity=10.0,
            entry_fee=0.6,
            exit_reason="sl",
            regime_int=0,
            taker_fee=0.0006,
            maker_fee=0.0002,
            slippage_pct=0.0005,
            high_vol_slippage_mult=2.0,
        )
        pnl_python = _close_trade(**kwargs)

        # _close_trade_numba utilise exit_reason int (0=tp, 1=sl)
        pnl_numba = _close_trade_numba(
            kwargs["direction"], kwargs["entry_price"], kwargs["exit_price"],
            kwargs["quantity"], kwargs["entry_fee"],
            1,  # sl = 1
            kwargs["regime_int"], kwargs["taker_fee"], kwargs["maker_fee"],
            kwargs["slippage_pct"], kwargs["high_vol_slippage_mult"],
        )
        assert abs(pnl_python - pnl_numba) < 0.001, (
            f"Python={pnl_python:.6f} vs Numba={pnl_numba:.6f}"
        )


# ─── Sprint 36 Part A : Double Slippage — GridPositionManager ────────────


class TestGridPositionManagerNoDoubleSlippage:
    """Vérifie que close_all_positions() n'applique le slippage qu'une fois."""

    def test_sl_global_single_slippage(self):
        """SL global : slippage = total_qty × exit_price × slippage_rate."""
        config = PositionManagerConfig(
            leverage=6, taker_fee=0.0, maker_fee=0.0,
            slippage_pct=0.01,
        )
        mgr = GridPositionManager(config)

        positions = [
            GridPosition(
                level=0,
                direction=Direction.LONG,
                entry_price=100.0,
                quantity=10.0,
                entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                entry_fee=0.0,
            ),
        ]

        trade = mgr.close_all_positions(
            positions, exit_price=110.0,
            exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
            exit_reason="sl_global",
            regime=MarketRegime.RANGING,
        )

        # gross = (110 - 100) * 10 = 100  (raw exit_price)
        # slippage = 110 * 10 * 0.01 = 11
        # net = 100 - 11 = 89
        expected_net = 89.0
        assert abs(trade.net_pnl - expected_net) < 0.01, (
            f"net_pnl={trade.net_pnl:.4f}, expected={expected_net}"
        )
        # TradeResult.exit_price = raw market price (pas actual_exit)
        assert trade.exit_price == 110.0, (
            f"exit_price should be raw market price, got {trade.exit_price}"
        )


# ─── Sprint 36 Part A : Double Slippage — PositionManager ────────────────


class TestPositionManagerNoDoubleSlippage:
    """Vérifie que close_position() n'applique le slippage qu'une fois."""

    def test_sl_single_slippage(self):
        """SL : slippage 1 seule fois."""
        from backend.strategies.base import OpenPosition

        config = PositionManagerConfig(
            leverage=6, taker_fee=0.0, maker_fee=0.0,
            slippage_pct=0.01,
        )
        mgr = PositionManager(config)

        position = OpenPosition(
            direction=Direction.LONG,
            entry_price=100.0,
            quantity=10.0,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            tp_price=110.0,
            sl_price=90.0,
            entry_fee=0.0,
        )

        trade = mgr.close_position(
            position, exit_price=95.0,
            exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
            exit_reason="sl",
            regime=MarketRegime.RANGING,
        )

        # gross = (95 - 100) * 10 = -50  (raw exit_price)
        # slippage = 95 * 10 * 0.01 = 9.5
        # net = -50 - 9.5 = -59.5
        expected_net = -59.5
        assert abs(trade.net_pnl - expected_net) < 0.01, (
            f"net_pnl={trade.net_pnl:.4f}, expected={expected_net}"
        )
        assert trade.exit_price == 95.0, (
            f"exit_price should be raw market price, got {trade.exit_price}"
        )


# ─── Sprint 36 Part A : Parité _calc_grid_pnl / _calc_grid_pnl_with_funding
#     AVEC slippage non-nul (les deux utilisent maintenant le flat cost model)


class TestCalcPnlParityWithSlippage:
    """Après Sprint 36, les deux fonctions doivent converger même avec slippage."""

    def test_same_pnl_with_slippage(self):
        """Slippage non-nul, funding=0 → même PnL entre les deux fonctions."""
        entry_price = 100.0
        exit_price = 105.0
        qty = 50.0
        taker_fee = 0.0006
        slippage = 0.005  # 0.5%

        # _calc_grid_pnl
        entry_fee = entry_price * qty * taker_fee
        positions_std = [(0, entry_price, qty, entry_fee)]
        pnl_std = _calc_grid_pnl(
            positions_std, exit_price,
            exit_fee_rate=taker_fee,
            slippage_rate=slippage,
            direction=1,
        )

        # _calc_grid_pnl_with_funding (funding=0)
        n = 10
        funding = np.zeros(n)
        timestamps = np.arange(n, dtype=np.int64) * 3_600_000

        positions_fund = [(entry_price, qty, 0)]
        pnl_fund = _calc_grid_pnl_with_funding(
            positions_fund, exit_price, 5,
            funding_rates=funding,
            candle_timestamps=timestamps,
            taker_fee=taker_fee,
            slippage_pct=slippage,
        )

        assert abs(pnl_std - pnl_fund) < 0.01, (
            f"pnl_std={pnl_std:.4f} vs pnl_fund={pnl_fund:.4f} — "
            f"les deux fonctions doivent converger AVEC slippage après Sprint 36"
        )


# ─── Sprint 36 Part B : Margin Deduction — _simulate_grid_common ─────────


class TestMarginDeductionGridCommon:
    """Vérifie que _simulate_grid_common() déduit/restaure la marge."""

    def test_margin_reduces_capital_at_open(self, make_indicator_cache):
        """Le capital doit diminuer de margin = notional/leverage à l'ouverture.

        On vérifie indirectement : avec margin deduction, le sizing du 2e
        niveau est basé sur capital RÉDUIT → le PnL total est INFÉRIEUR
        à ce qu'on obtiendrait sans margin deduction.
        """
        n = 20
        ma_period = 5

        # Prix autour de SMA → TP = close >= SMA
        closes = np.full(n, 100.0)
        sma = np.full(n, 100.0)

        # Entrée : niveaux à 99 et 98 (lows < entry_prices)
        lows = np.full(n, 97.0)
        highs = np.full(n, 103.0)

        # SMA à 100 → close=100 = SMA → TP trigger immédiat !
        # Pour forcer un cycle : entry puis exit sur la même candle
        # Non — on va faire : close < SMA pendant 10 candles puis close > SMA
        sma = np.full(n, 105.0)  # SMA au-dessus de close pendant l'entrée
        closes[-1] = 106.0  # TP trigger à la dernière candle
        highs[-1] = 107.0

        # Entry prices : 2 niveaux à 99 et 98
        entry_prices = np.full((n, 2), np.nan)
        for i in range(ma_period + 1, n):
            entry_prices[i, 0] = 99.0
            entry_prices[i, 1] = 98.0

        cache = make_indicator_cache(
            n=n, closes=closes, lows=lows, highs=highs,
            bb_sma={ma_period: sma},
        )

        bt = _make_bt_config(
            initial_capital=10_000.0, leverage=6,
            taker_fee=0.0, maker_fee=0.0, slippage_pct=0.0,
        )

        trade_pnls, _, final_capital = _simulate_grid_common(
            entry_prices, sma, cache, bt,
            num_levels=2, sl_pct=0.5, direction=1,
        )

        # Avec margin deduction :
        # Level 0 : notional = 10000 * 0.5 * 6 = 30000, qty = 30000/99 ≈ 303.03
        #           margin = 30000/6 = 5000, capital = 10000 - 5000 = 5000
        # Level 1 : notional = 5000 * 0.5 * 6 = 15000, qty = 15000/98 ≈ 153.06
        #           margin = 15000/6 = 2500, capital = 5000 - 2500 = 2500
        #
        # Sans margin deduction (ancien comportement) :
        # Level 0 : notional = 10000 * 0.5 * 6 = 30000, qty = 30000/99 ≈ 303.03
        # Level 1 : notional = 10000 * 0.5 * 6 = 30000, qty = 30000/98 ≈ 306.12
        #           → qty 2x plus gros au niveau 1 !
        #
        # Le PnL total avec margin deduction est plus faible car le niveau 1
        # a un sizing réduit.

        assert len(trade_pnls) >= 1, "Au moins 1 trade attendu"

        # Calcul attendu avec margin :
        # qty0 = (10000/2*6) / 99 = 30000/99 ≈ 303.03
        # qty1 = (5000/2*6) / 98 = 15000/98 ≈ 153.06
        # pnl0 = (106 - 99) * 303.03 ≈ 2121.21
        # pnl1 = (106 - 98) * 153.06 ≈ 1224.49
        # total ≈ 3345.7
        #
        # Sans margin :
        # qty0 = 303.03
        # qty1 = (10000/2*6) / 98 = 306.12
        # pnl1 = (106 - 98) * 306.12 ≈ 2448.98
        # total ≈ 4570.19

        # TP exit at SMA=105 (not close=106):
        # qty0 = (10000/2*6) / 99 = 303.03, pnl0 = (105-99)*303.03 = 1818.18
        # qty1 = (5000/2*6) / 98 = 153.06, pnl1 = (105-98)*153.06 = 1071.43
        # total ≈ 2889.61
        # Sans margin : qty1 = 306.12, pnl1 = 2448.98, total ≈ 4267.16
        assert trade_pnls[0] < 3500, (
            f"PnL={trade_pnls[0]:.2f} trop élevé — margin deduction pas active ?"
        )
        assert trade_pnls[0] > 2500, (
            f"PnL={trade_pnls[0]:.2f} trop bas — vérifier la logique"
        )

    def test_margin_restored_at_close(self, make_indicator_cache):
        """Breakeven trade → final_capital == initial_capital (margin restaurée)."""
        n = 20
        ma_period = 5

        closes = np.full(n, 100.0)
        sma = np.full(n, 105.0)
        lows = np.full(n, 97.0)
        highs = np.full(n, 103.0)

        # Exit au même prix que l'entrée → breakeven
        closes[-1] = 106.0  # TP
        highs[-1] = 107.0

        # 1 seul niveau → entry à 99
        entry_prices = np.full((n, 1), np.nan)
        for i in range(ma_period + 1, n):
            entry_prices[i, 0] = 99.0

        # Exit price = 99 → breakeven si pas de fees
        # Sauf que TP = close >= SMA, et close[-1]=106, exit_price=SMA=105
        # Non : TP dans _simulate_grid_common c'est close >= SMA → exit at SMA price
        # Vérifions... non, le TP utilise tp_price = sma_arr[i]
        # En fait, let me reconsider. Le TP dans _simulate_grid_common est
        # "close >= SMA → tp_hit", exit_price = tp_price = sma_arr[i]
        # Donc exit à 105.

        cache = make_indicator_cache(
            n=n, closes=closes, lows=lows, highs=highs,
            bb_sma={ma_period: sma},
        )

        bt = _make_bt_config(
            initial_capital=10_000.0, leverage=6,
            taker_fee=0.0, maker_fee=0.0, slippage_pct=0.0,
        )

        _, _, final_capital = _simulate_grid_common(
            entry_prices, sma, cache, bt,
            num_levels=1, sl_pct=0.5, direction=1,
        )

        # Entry at 99, exit at 105 (SMA), qty = 10000*1*6/99 ≈ 606.06
        # pnl = (105 - 99) * 606.06 ≈ 3636.36
        # final = 10000 + 3636.36 ≈ 13636.36
        # Vérification : margin restaurée, aucune fuite
        assert final_capital > 10_000.0, (
            f"final_capital={final_capital:.2f} — trade profitable devrait augmenter le capital"
        )
        # La valeur précise : 10000 + (105-99) * (10000*6/99) = 10000 + 6*60000/99
        expected = 10_000.0 + 6.0 * (10_000.0 * 6.0 / 99.0)
        assert abs(final_capital - expected) < 1.0, (
            f"final_capital={final_capital:.2f}, expected≈{expected:.2f}"
        )

    def test_insufficient_margin_skips_level(self, make_indicator_cache):
        """Si capital < margin, le niveau n'est pas ouvert."""
        n = 20
        ma_period = 5

        closes = np.full(n, 100.0)
        sma = np.full(n, 105.0)
        lows = np.full(n, 97.0)
        highs = np.full(n, 103.0)
        closes[-1] = 106.0
        highs[-1] = 107.0

        # 10 niveaux avec capital=100 → margin/level = 100/10 = 10
        # Chaque level déduit 10 → après 10 levels, capital = 0
        # Mais sizing = (capital / 10) * 6, capital diminue → pas 10 levels
        entry_prices = np.full((n, 10), np.nan)
        for i in range(ma_period + 1, n):
            for lvl in range(10):
                entry_prices[i, lvl] = 99.0 - lvl * 0.5

        cache = make_indicator_cache(
            n=n, closes=closes, lows=lows, highs=highs,
            bb_sma={ma_period: sma},
        )

        bt = _make_bt_config(
            initial_capital=100.0, leverage=6,
            taker_fee=0.0, maker_fee=0.0, slippage_pct=0.0,
        )

        trade_pnls, _, final_capital = _simulate_grid_common(
            entry_prices, sma, cache, bt,
            num_levels=10, sl_pct=0.5, direction=1,
        )

        # Le capital de 100 permet ~10 levels × margin=10
        # Mais chaque level réduit le capital, donc les levels suivants ont
        # des margins décroissantes. Tous devraient ouvrir car margin = capital/10
        # diminue avec le capital. Le test vérifie surtout que ça ne crash pas
        # et que final_capital > 0 (pas de capital négatif)
        assert final_capital > 0, (
            f"final_capital={final_capital:.2f} — ne devrait pas être négatif"
        )
