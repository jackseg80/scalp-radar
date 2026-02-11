"""Tests pour le LiveRiskManager (Sprint 5a)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from backend.execution.risk_manager import LiveRiskManager, LiveTradeResult


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_config() -> MagicMock:
    """Config mock avec les valeurs par défaut de risk.yaml."""
    config = MagicMock()
    config.risk.kill_switch.max_session_loss_percent = 5.0
    config.risk.kill_switch.max_daily_loss_percent = 10.0
    config.risk.position.max_concurrent_positions = 3
    config.risk.position.default_leverage = 15
    config.risk.margin.min_free_margin_percent = 20
    return config


def _make_rm(config=None) -> LiveRiskManager:
    if config is None:
        config = _make_config()
    rm = LiveRiskManager(config)
    rm.set_initial_capital(10_000.0)
    return rm


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# ─── Pre-trade checks ─────────────────────────────────────────────────────


class TestPreTradeCheck:
    def test_ok_when_no_issues(self):
        rm = _make_rm()
        ok, reason = rm.pre_trade_check(
            symbol="BTC/USDT:USDT",
            direction="LONG",
            quantity=0.001,
            entry_price=100_000.0,
            free_margin=5_000.0,
            total_balance=10_000.0,
        )
        assert ok is True
        assert reason == "ok"

    def test_rejected_kill_switch(self):
        rm = _make_rm()
        rm._kill_switch_triggered = True
        ok, reason = rm.pre_trade_check(
            "BTC/USDT:USDT", "LONG", 0.001, 100_000.0, 5_000.0, 10_000.0,
        )
        assert ok is False
        assert reason == "kill_switch_live"

    def test_rejected_position_already_open(self):
        rm = _make_rm()
        rm.register_position({"symbol": "BTC/USDT:USDT", "direction": "LONG"})
        ok, reason = rm.pre_trade_check(
            "BTC/USDT:USDT", "LONG", 0.001, 100_000.0, 5_000.0, 10_000.0,
        )
        assert ok is False
        assert reason == "position_already_open"

    def test_rejected_max_concurrent_positions(self):
        rm = _make_rm()
        for sym in ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]:
            rm.register_position({"symbol": sym, "direction": "LONG"})

        ok, reason = rm.pre_trade_check(
            "DOGE/USDT:USDT", "LONG", 1.0, 0.1, 5_000.0, 10_000.0,
        )
        assert ok is False
        assert reason == "max_concurrent_positions"

    def test_rejected_insufficient_margin(self):
        rm = _make_rm()
        # Quantité énorme → marge requise dépasse la marge libre
        ok, reason = rm.pre_trade_check(
            "BTC/USDT:USDT", "LONG",
            quantity=1.0,
            entry_price=100_000.0,
            free_margin=3_000.0,
            total_balance=10_000.0,
        )
        assert ok is False
        assert reason == "insufficient_margin"

    def test_margin_check_respects_min_free_percent(self):
        """La marge résiduelle doit rester >= 20% du total balance."""
        rm = _make_rm()
        # total=10000, min_free=20% → 2000 USDT doit rester libre
        # free=2500, required = 0.01 * 100000 / 15 = 66.67 → résidu=2433 > 2000 → OK
        ok, reason = rm.pre_trade_check(
            "BTC/USDT:USDT", "LONG",
            quantity=0.01,
            entry_price=100_000.0,
            free_margin=2_500.0,
            total_balance=10_000.0,
        )
        assert ok is True
        assert reason == "ok"


# ─── Kill switch live ──────────────────────────────────────────────────────


class TestKillSwitchLive:
    def test_triggers_on_loss_threshold(self):
        rm = _make_rm()
        # Perte de 500$ = 5% du capital initial (10k) → kill switch
        rm.record_trade_result(LiveTradeResult(
            net_pnl=-500.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="LONG", exit_reason="sl",
        ))
        assert rm.is_kill_switch_triggered is True

    def test_not_triggered_under_threshold(self):
        rm = _make_rm()
        rm.record_trade_result(LiveTradeResult(
            net_pnl=-400.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="LONG", exit_reason="sl",
        ))
        assert rm.is_kill_switch_triggered is False

    def test_accumulates_losses(self):
        rm = _make_rm()
        # Deux pertes de 250$ = 500$ total = 5%
        rm.record_trade_result(LiveTradeResult(
            net_pnl=-250.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="LONG", exit_reason="sl",
        ))
        assert rm.is_kill_switch_triggered is False

        rm.record_trade_result(LiveTradeResult(
            net_pnl=-250.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="SHORT", exit_reason="sl",
        ))
        assert rm.is_kill_switch_triggered is True

    def test_wins_offset_losses(self):
        rm = _make_rm()
        rm.record_trade_result(LiveTradeResult(
            net_pnl=-300.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="LONG", exit_reason="sl",
        ))
        rm.record_trade_result(LiveTradeResult(
            net_pnl=200.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="SHORT", exit_reason="tp",
        ))
        # Session PnL = -100 → 1% → pas de kill switch
        assert rm.is_kill_switch_triggered is False
        assert rm._session_pnl == pytest.approx(-100.0)


# ─── Position tracking ────────────────────────────────────────────────────


class TestPositionTracking:
    def test_register_and_count(self):
        rm = _make_rm()
        assert rm.open_positions_count == 0
        rm.register_position({"symbol": "BTC/USDT:USDT"})
        assert rm.open_positions_count == 1

    def test_unregister_returns_position(self):
        rm = _make_rm()
        rm.register_position({"symbol": "BTC/USDT:USDT", "direction": "LONG"})
        pos = rm.unregister_position("BTC/USDT:USDT")
        assert pos is not None
        assert pos["direction"] == "LONG"
        assert rm.open_positions_count == 0

    def test_unregister_unknown_returns_none(self):
        rm = _make_rm()
        assert rm.unregister_position("BTC/USDT:USDT") is None


# ─── State persistence ────────────────────────────────────────────────────


class TestStatePersistence:
    def test_get_state_serialization(self):
        rm = _make_rm()
        rm.record_trade_result(LiveTradeResult(
            net_pnl=-100.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="LONG", exit_reason="sl",
        ))
        state = rm.get_state()
        assert state["session_pnl"] == pytest.approx(-100.0)
        assert state["kill_switch"] is False
        assert state["total_orders"] == 0
        assert state["initial_capital"] == pytest.approx(10_000.0)
        assert len(state["trade_history"]) == 1

    def test_restore_state_roundtrip(self):
        rm = _make_rm()
        rm.record_trade_result(LiveTradeResult(
            net_pnl=-200.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="LONG", exit_reason="sl",
        ))
        rm._total_orders = 5

        state = rm.get_state()

        rm2 = _make_rm()
        rm2.restore_state(state)
        assert rm2._session_pnl == pytest.approx(-200.0)
        assert rm2._total_orders == 5
        assert rm2._initial_capital == pytest.approx(10_000.0)

    def test_restore_state_with_kill_switch(self):
        rm = _make_rm()
        rm._kill_switch_triggered = True
        state = rm.get_state()

        rm2 = _make_rm()
        rm2.restore_state(state)
        assert rm2.is_kill_switch_triggered is True


# ─── Correlation groups ──────────────────────────────────────────────────


def _make_config_with_correlation() -> MagicMock:
    """Config mock avec groupes de corrélation comme dans assets.yaml."""
    config = _make_config()

    # Assets avec correlation_group
    btc = MagicMock()
    btc.symbol = "BTC/USDT"
    btc.correlation_group = "crypto_major"
    eth = MagicMock()
    eth.symbol = "ETH/USDT"
    eth.correlation_group = "crypto_major"
    sol = MagicMock()
    sol.symbol = "SOL/USDT"
    sol.correlation_group = "crypto_major"

    config.assets = [btc, eth, sol]

    # Correlation group config
    group_cfg = MagicMock()
    group_cfg.max_concurrent_same_direction = 2
    config.correlation_groups = {"crypto_major": group_cfg}

    return config


class TestCorrelationGroups:
    def test_3eme_long_meme_groupe_rejete(self):
        """2 LONG déjà ouverts dans crypto_major → 3ème LONG rejeté."""
        config = _make_config_with_correlation()
        rm = LiveRiskManager(config)
        rm.set_initial_capital(10_000.0)

        # Ouvrir 2 positions LONG dans le même groupe
        rm.register_position({"symbol": "BTC/USDT:USDT", "direction": "LONG"})
        rm.register_position({"symbol": "ETH/USDT:USDT", "direction": "LONG"})

        # 3ème LONG SOL → rejeté
        ok, reason = rm.pre_trade_check(
            "SOL/USDT:USDT", "LONG", 1.0, 150.0, 5_000.0, 10_000.0,
        )
        assert ok is False
        assert "correlation_group_limit" in reason

    def test_direction_differente_ok(self):
        """2 LONG + 1 SHORT dans le même groupe → OK (directions différentes)."""
        config = _make_config_with_correlation()
        rm = LiveRiskManager(config)
        rm.set_initial_capital(10_000.0)

        rm.register_position({"symbol": "BTC/USDT:USDT", "direction": "LONG"})
        rm.register_position({"symbol": "ETH/USDT:USDT", "direction": "LONG"})

        # SHORT SOL → OK car c'est une direction différente
        ok, reason = rm.pre_trade_check(
            "SOL/USDT:USDT", "SHORT", 1.0, 150.0, 5_000.0, 10_000.0,
        )
        assert ok is True
        assert reason == "ok"

    def test_sans_groupe_pas_de_limite(self):
        """Asset sans correlation_group → pas de limite de corrélation."""
        config = _make_config()
        # Assets sans correlation_group
        asset_no_group = MagicMock()
        asset_no_group.symbol = "DOGE/USDT"
        asset_no_group.correlation_group = None
        config.assets = [asset_no_group]
        config.correlation_groups = {}

        rm = LiveRiskManager(config)
        rm.set_initial_capital(10_000.0)

        ok, reason = rm.pre_trade_check(
            "DOGE/USDT:USDT", "LONG", 100.0, 0.1, 5_000.0, 10_000.0,
        )
        assert ok is True
        assert reason == "ok"
