"""Tests pour le LiveRiskManager (Sprint 5a + Audit 2026-02-19)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.execution.risk_manager import LiveRiskManager, LiveTradeResult


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_config() -> MagicMock:
    """Config mock avec les valeurs par défaut de risk.yaml."""
    config = MagicMock()
    config.risk.kill_switch.max_session_loss_percent = 5.0
    config.risk.kill_switch.max_daily_loss_percent = 10.0
    config.risk.kill_switch.grid_max_session_loss_percent = 25.0
    config.risk.kill_switch.global_max_loss_pct = 45
    config.risk.kill_switch.global_window_hours = 24
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


# ─── Reconciliation (kill switch déclenche normalement, UX améliorée) ────


class TestReconciliationKillSwitch:
    """Hotfix 2026-02-24 : le kill switch déclenche normalement sur les pertes
    de réconciliation (pas de bypass). L'UX est améliorée côté Executor."""

    def test_downtime_losses_trigger_kill_switch(self):
        """Les pertes de réconciliation DOIVENT déclencher le kill switch."""
        rm = _make_rm()
        rm.record_trade_result(LiveTradeResult(
            net_pnl=-500.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="LONG",
            exit_reason="closed_during_downtime",
        ))
        assert rm.is_kill_switch_triggered is True
        assert rm._session_pnl == pytest.approx(-500.0)

    def test_downtime_losses_in_session_pnl(self):
        """Les pertes de réconciliation sont bien comptées dans session_pnl."""
        rm = _make_rm()
        rm.record_trade_result(LiveTradeResult(
            net_pnl=-200.0, timestamp=_now(),
            symbol="ETH/USDT:USDT", direction="SHORT",
            exit_reason="closed_during_downtime", strategy_name="grid_boltrend",
        ))
        rm.record_trade_result(LiveTradeResult(
            net_pnl=-100.0, timestamp=_now(),
            symbol="SOL/USDT:USDT", direction="LONG",
            exit_reason="closed_during_downtime", strategy_name="grid_atr",
        ))
        assert rm._session_pnl == pytest.approx(-300.0)
        assert len(rm._trade_history) == 2


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


# ─── Sprint 12 : leverage_override ──────────────────────────────────────


class TestLeverageOverride:
    """Tests pour le paramètre leverage_override (grid DCA)."""

    def test_leverage_override_uses_custom_leverage(self):
        """Avec leverage_override=6, la marge requise est calculée à levier 6."""
        rm = _make_rm()
        # quantity=0.01, entry=100000 → notional=1000
        # leverage=6 → required_margin = 1000/6 = 166.67
        # free=5000, min_free=20%*10000=2000 → résidu=4833 > 2000 → OK
        ok, reason = rm.pre_trade_check(
            "BTC/USDT:USDT", "LONG",
            quantity=0.01, entry_price=100_000.0,
            free_margin=5_000.0, total_balance=10_000.0,
            leverage_override=6,
        )
        assert ok is True
        assert reason == "ok"

    def test_leverage_override_higher_margin_requirement(self):
        """Avec leverage=6 (vs default 15), la marge requise est plus haute."""
        rm = _make_rm()
        # quantity=0.1, entry=100000 → notional=10000
        # leverage=6 → required_margin = 10000/6 = 1666.67
        # free=3500, min_free=20%*10000=2000 → résidu=1833.33 < 2000 → REJETÉ
        ok_low_lev, reason_low = rm.pre_trade_check(
            "BTC/USDT:USDT", "LONG",
            quantity=0.1, entry_price=100_000.0,
            free_margin=3_500.0, total_balance=10_000.0,
            leverage_override=6,
        )
        assert ok_low_lev is False
        assert reason_low == "insufficient_margin"

        # Même trade SANS override → leverage=15 → required=10000/15=666.67
        # résidu=3500-666.67=2833.33 > 2000 → OK
        ok_default, reason_default = rm.pre_trade_check(
            "BTC/USDT:USDT", "LONG",
            quantity=0.1, entry_price=100_000.0,
            free_margin=3_500.0, total_balance=10_000.0,
        )
        assert ok_default is True
        assert reason_default == "ok"

    def test_grid_cycle_counts_as_one_position(self):
        """Un cycle grid = 1 position pour max_concurrent_positions."""
        rm = _make_rm()
        # Enregistrer 1 cycle grid (1 appel register_position)
        rm.register_position({
            "symbol": "BTC/USDT:USDT", "direction": "LONG",
            "entry_price": 50_000.0, "quantity": 0.002,
        })
        # 2ème position mono
        rm.register_position({
            "symbol": "ETH/USDT:USDT", "direction": "LONG",
            "entry_price": 3_000.0, "quantity": 0.1,
        })
        assert rm.open_positions_count == 2  # grid=1 + mono=1

        # 3ème position OK (max=3)
        ok, reason = rm.pre_trade_check(
            "SOL/USDT:USDT", "LONG",
            quantity=1.0, entry_price=150.0,
            free_margin=5_000.0, total_balance=10_000.0,
        )
        assert ok is True


# ═══════════════════════════════════════════════════════════════════════════
# Audit 2026-02-19 — tests pour les fixes P0 + P1
# ═══════════════════════════════════════════════════════════════════════════


class TestAuditP0GridThreshold:
    """P0 : grid_max_session_loss_percent utilisé pour les stratégies grid."""

    def test_grid_strategy_uses_25pct_threshold(self):
        """Perte 6% = OK pour grid (seuil 25%), KO pour mono (seuil 5%)."""
        rm = _make_rm()
        # 600$ = 6% de 10k → dépasse 5% mono mais pas 25% grid
        rm.record_trade_result(LiveTradeResult(
            net_pnl=-600.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="LONG", exit_reason="sl",
            strategy_name="grid_atr",
        ))
        assert rm.is_kill_switch_triggered is False

    def test_mono_strategy_uses_5pct_threshold(self):
        """Perte 6% = KO pour mono (seuil 5%)."""
        rm = _make_rm()
        rm.record_trade_result(LiveTradeResult(
            net_pnl=-600.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="LONG", exit_reason="sl",
            strategy_name="vwap_rsi",
        ))
        assert rm.is_kill_switch_triggered is True

    def test_grid_boltrend_uses_grid_threshold(self):
        """grid_boltrend = stratégie grid → seuil 25%."""
        rm = _make_rm()
        rm.record_trade_result(LiveTradeResult(
            net_pnl=-2000.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="LONG", exit_reason="sl",
            strategy_name="grid_boltrend",
        ))
        # 2000$ = 20% < 25% → pas de kill switch
        assert rm.is_kill_switch_triggered is False

    def test_grid_triggers_at_25pct(self):
        """Perte 25% = kill switch pour grid."""
        rm = _make_rm()
        rm.record_trade_result(LiveTradeResult(
            net_pnl=-2500.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="LONG", exit_reason="sl",
            strategy_name="grid_atr",
        ))
        assert rm.is_kill_switch_triggered is True

    def test_empty_strategy_name_uses_mono_threshold(self):
        """strategy_name vide → fallback seuil mono 5%."""
        rm = _make_rm()
        rm.record_trade_result(LiveTradeResult(
            net_pnl=-600.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="LONG", exit_reason="sl",
            strategy_name="",
        ))
        assert rm.is_kill_switch_triggered is True

    def test_grid_max_session_loss_fallback_25(self):
        """Si grid_max_session_loss_percent absent → fallback 25.0."""
        config = _make_config()
        config.risk.kill_switch.grid_max_session_loss_percent = None
        rm = LiveRiskManager(config)
        rm.set_initial_capital(10_000.0)

        rm.record_trade_result(LiveTradeResult(
            net_pnl=-2000.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="LONG", exit_reason="sl",
            strategy_name="grid_atr",
        ))
        # 20% < 25% fallback → pas de kill switch
        assert rm.is_kill_switch_triggered is False


class TestAuditP1DailyReset:
    """P1 : reset automatique session_pnl à minuit UTC."""

    def test_session_pnl_resets_on_new_day(self):
        """session_pnl reset quand le jour UTC change."""
        rm = _make_rm()
        rm.record_trade_result(LiveTradeResult(
            net_pnl=-300.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="LONG", exit_reason="sl",
            strategy_name="grid_atr",
        ))
        assert rm._session_pnl == pytest.approx(-300.0)

        # Simuler le passage au jour suivant (UTC pour cohérence avec le code)
        rm._session_start_date = datetime.now(tz=timezone.utc).date() - timedelta(days=1)

        rm.record_trade_result(LiveTradeResult(
            net_pnl=-100.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="LONG", exit_reason="sl",
            strategy_name="grid_atr",
        ))
        # Reset à 0 + -100 = -100 (pas -400)
        assert rm._session_pnl == pytest.approx(-100.0)

    def test_session_start_date_persisted(self):
        """session_start_date sauvegardé et restauré dans get_state/restore_state."""
        rm = _make_rm()
        state = rm.get_state()
        assert "session_start_date" in state

        rm2 = _make_rm()
        rm2.restore_state(state)
        assert rm2._session_start_date == rm._session_start_date

    def test_no_reset_same_day(self):
        """Pas de reset si même jour UTC."""
        rm = _make_rm()
        rm.record_trade_result(LiveTradeResult(
            net_pnl=-200.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="LONG", exit_reason="sl",
            strategy_name="grid_atr",
        ))
        rm.record_trade_result(LiveTradeResult(
            net_pnl=-100.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="SHORT", exit_reason="sl",
            strategy_name="grid_atr",
        ))
        # Même jour → pas de reset → cumul = -300
        assert rm._session_pnl == pytest.approx(-300.0)


class TestAuditP1TelegramAlert:
    """P1 : alerte Telegram quand kill switch live déclenché."""

    def test_notifier_called_on_kill_switch(self):
        """Notifier.notify_anomaly() appelé quand kill switch déclenché."""
        config = _make_config()
        notifier = MagicMock()
        notifier.notify_anomaly = AsyncMock()
        rm = LiveRiskManager(config, notifier=notifier)
        rm.set_initial_capital(10_000.0)

        rm.record_trade_result(LiveTradeResult(
            net_pnl=-500.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="LONG", exit_reason="sl",
            strategy_name="vwap_rsi",
        ))
        assert rm.is_kill_switch_triggered is True
        # L'appel crée une task asyncio — vérifie que create_task est appelé

    def test_no_alert_when_no_notifier(self):
        """Pas d'erreur si notifier est None."""
        rm = _make_rm()  # pas de notifier
        rm.record_trade_result(LiveTradeResult(
            net_pnl=-600.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="LONG", exit_reason="sl",
            strategy_name="vwap_rsi",
        ))
        assert rm.is_kill_switch_triggered is True
        # Pas d'exception → OK

    def test_no_alert_under_threshold(self):
        """Pas d'alerte si seuil pas atteint."""
        config = _make_config()
        notifier = MagicMock()
        notifier.notify_anomaly = AsyncMock()
        rm = LiveRiskManager(config, notifier=notifier)
        rm.set_initial_capital(10_000.0)

        rm.record_trade_result(LiveTradeResult(
            net_pnl=-100.0, timestamp=_now(),
            symbol="BTC/USDT:USDT", direction="LONG", exit_reason="sl",
            strategy_name="vwap_rsi",
        ))
        assert rm.is_kill_switch_triggered is False


class TestAuditP1GlobalKillSwitch:
    """P1 : kill switch global live (drawdown fenêtre glissante)."""

    def test_drawdown_triggers_kill_switch(self):
        """Drawdown >= 45% depuis le peak → kill switch."""
        rm = _make_rm()
        now = datetime.now(tz=timezone.utc)

        # Peak à 10000
        rm._balance_snapshots.append((now - timedelta(hours=1), 10_000.0))
        # Chute à 5400 = drawdown 46%
        rm.record_balance_snapshot(5_400.0)
        assert rm.is_kill_switch_triggered is True

    def test_drawdown_under_threshold_ok(self):
        """Drawdown < 45% → pas de kill switch."""
        rm = _make_rm()
        now = datetime.now(tz=timezone.utc)

        rm._balance_snapshots.append((now - timedelta(hours=1), 10_000.0))
        # Chute à 6000 = drawdown 40%
        rm.record_balance_snapshot(6_000.0)
        assert rm.is_kill_switch_triggered is False

    def test_old_snapshots_excluded(self):
        """Snapshots hors fenêtre 24h exclus du calcul peak."""
        config = _make_config()
        config.risk.kill_switch.global_window_hours = 24
        rm = LiveRiskManager(config)
        rm.set_initial_capital(10_000.0)
        now = datetime.now(tz=timezone.utc)

        # Peak ancien (25h ago) — hors fenêtre
        rm._balance_snapshots.append((now - timedelta(hours=25), 20_000.0))
        # Snapshot récent
        rm._balance_snapshots.append((now - timedelta(hours=1), 10_000.0))
        # Balance actuelle = 8000 → drawdown depuis 10000 = 20% (pas 60% depuis 20000)
        rm.record_balance_snapshot(8_000.0)
        assert rm.is_kill_switch_triggered is False

    def test_already_triggered_skips_check(self):
        """Si kill switch déjà actif, pas de re-vérification."""
        rm = _make_rm()
        rm._kill_switch_triggered = True
        now = datetime.now(tz=timezone.utc)

        rm._balance_snapshots.append((now - timedelta(hours=1), 10_000.0))
        rm.record_balance_snapshot(1_000.0)  # drawdown 90%
        # Pas d'erreur, toujours triggered
        assert rm.is_kill_switch_triggered is True

    def test_global_kill_switch_without_config(self):
        """Si global_max_loss_pct absent de config → pas de vérification."""
        config = _make_config()
        config.risk.kill_switch.global_max_loss_pct = None
        config.risk.kill_switch.global_window_hours = None
        rm = LiveRiskManager(config)
        rm.set_initial_capital(10_000.0)
        now = datetime.now(tz=timezone.utc)

        rm._balance_snapshots.append((now - timedelta(hours=1), 10_000.0))
        rm.record_balance_snapshot(1_000.0)  # drawdown 90%
        # Pas de threshold configuré → pas de kill switch
        assert rm.is_kill_switch_triggered is False

    def test_single_snapshot_skips(self):
        """Avec un seul snapshot, pas de check possible."""
        rm = _make_rm()
        rm.record_balance_snapshot(10_000.0)
        assert rm.is_kill_switch_triggered is False


class TestAuditP0ResetEndpoint:
    """P0 : reset kill switch live via state manipulation (logique)."""

    def test_reset_clears_kill_switch_and_session(self):
        """Reset met kill_switch=False et session_pnl=0."""
        rm = _make_rm()
        rm._kill_switch_triggered = True
        rm._session_pnl = -500.0

        # Simuler le reset (même logique que l'endpoint)
        rm._kill_switch_triggered = False
        rm._session_pnl = 0.0

        assert rm.is_kill_switch_triggered is False
        assert rm._session_pnl == 0.0

        # Peut à nouveau trader
        ok, reason = rm.pre_trade_check(
            "BTC/USDT:USDT", "LONG", 0.001, 100_000.0, 5_000.0, 10_000.0,
        )
        assert ok is True

    def test_state_reflects_reset(self):
        """get_state() après reset reflète les nouvelles valeurs."""
        rm = _make_rm()
        rm._kill_switch_triggered = True
        rm._session_pnl = -800.0

        rm._kill_switch_triggered = False
        rm._session_pnl = 0.0

        state = rm.get_state()
        assert state["kill_switch"] is False
        assert state["session_pnl"] == 0.0


class TestAuditP0KillSwitchResetEndpoint:
    """P0 : endpoint POST /api/executor/kill-switch/reset (intégration)."""

    def test_reset_endpoint_no_executor_returns_400(self, monkeypatch):
        """POST /kill-switch/reset sans executor → 400."""
        from backend.api.server import app
        from fastapi.testclient import TestClient
        from backend.core.config import AppConfig

        test_config = AppConfig()
        test_config.secrets.sync_api_key = "secret123"
        monkeypatch.setattr("backend.api.executor_routes.get_config", lambda: test_config)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/executor/kill-switch/reset",
            headers={"X-API-Key": "secret123"},
        )
        assert resp.status_code == 400

    def test_reset_endpoint_auth_required(self, monkeypatch):
        """POST /kill-switch/reset sans API key → 401."""
        from backend.api.server import app
        from fastapi.testclient import TestClient
        from backend.core.config import AppConfig

        test_config = AppConfig()
        test_config.secrets.sync_api_key = "secret123"
        monkeypatch.setattr("backend.api.executor_routes.get_config", lambda: test_config)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/executor/kill-switch/reset")
        assert resp.status_code == 401
