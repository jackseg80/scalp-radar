"""RiskManager : vérifications pré-trade, kill switch live, marge.

Séparé de l'Executor pour clarifier les responsabilités.
Double kill switch : le Simulator a le sien (virtuel),
le RiskManager a le sien (argent réel).

Audit 2026-02-19 — 3 P0 + 3 P1 fixes :
- P0: seuil grid_max_session_loss_percent pour stratégies grid
- P0: endpoint reset kill switch live (dans executor_routes.py)
- P0: guard kill switch DCA niveaux 2+ (dans executor.py)
- P1: alerte Telegram quand kill switch live déclenché
- P1: reset quotidien session_pnl à minuit UTC
- P1: kill switch global live (drawdown 45% fenêtre glissante)
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from backend.alerts.notifier import Notifier
    from backend.core.config import AppConfig


@dataclass
class LiveTradeResult:
    """Résultat d'un trade live (pour le kill switch live)."""

    net_pnl: float
    timestamp: datetime
    symbol: str
    direction: str
    exit_reason: str
    strategy_name: str = ""  # Audit P0 : nécessaire pour grid_max_session_loss_percent


class LiveRiskManager:
    """Vérifications pré-trade et kill switch live.

    Le RiskManager a un kill switch indépendant basé sur le capital réel.
    Il ne touche pas au kill switch du Simulator (virtuel, par runner).
    """

    def __init__(self, config: AppConfig, notifier: Notifier | None = None) -> None:
        self._config = config
        self._notifier = notifier  # P1 : alerte Telegram
        self._initial_capital: float = 0.0
        self._session_pnl: float = 0.0
        self._kill_switch_triggered: bool = False
        self._open_positions: list[dict[str, Any]] = []
        self._trade_history: list[LiveTradeResult] = []
        self._total_orders: int = 0
        # P1 : reset quotidien session_pnl
        self._session_start_date: date = datetime.now(tz=timezone.utc).date()
        # P1 : kill switch global (drawdown fenêtre glissante)
        self._balance_snapshots: deque[tuple[datetime, float]] = deque(maxlen=288)  # 24h @ 5min

    def set_initial_capital(self, capital: float) -> None:
        """Définit le capital initial (appelé par Executor après fetch_balance)."""
        self._initial_capital = capital
        logger.info("RiskManager: capital initial = {:.2f} USDT", capital)

    # ─── Pre-trade checks ──────────────────────────────────────────────

    def pre_trade_check(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        entry_price: float,
        free_margin: float,
        total_balance: float,
        leverage_override: int | None = None,
    ) -> tuple[bool, str]:
        """Vérifie toutes les conditions avant de passer un ordre.

        Retourne (ok, reason).
        leverage_override : si fourni, utilise ce leverage au lieu du défaut
        (ex: grid DCA leverage=6 vs défaut=15).
        """
        # 1. Kill switch live déclenché ?
        if self._kill_switch_triggered:
            return False, "kill_switch_live"

        # 2. Position déjà ouverte pour ce symbol ?
        for pos in self._open_positions:
            if pos["symbol"] == symbol:
                return False, "position_already_open"

        # 3. Max concurrent positions atteint ?
        max_pos = self._config.risk.position.max_concurrent_positions
        if len(self._open_positions) >= max_pos:
            return False, "max_concurrent_positions"

        # 4. Limite direction dans le groupe de corrélation (assets.yaml)
        group = self._get_correlation_group(symbol)
        if group:
            max_same_dir = self._get_max_same_direction(group)
            same_dir_count = sum(
                1 for pos in self._open_positions
                if pos["direction"] == direction
                and self._get_correlation_group(pos["symbol"]) == group
            )
            if same_dir_count >= max_same_dir:
                return False, f"correlation_group_limit ({group})"

        # 5. Marge disponible suffisante ?
        leverage = leverage_override or self._config.risk.position.default_leverage
        required_margin = quantity * entry_price / leverage
        min_free_pct = self._config.risk.margin.min_free_margin_percent / 100
        min_free = total_balance * min_free_pct

        if free_margin - required_margin < min_free:
            logger.warning(
                "RiskManager: marge insuffisante — libre={:.2f}, requis={:.2f}, "
                "min_libre={:.2f}",
                free_margin, required_margin, min_free,
            )
            return False, "insufficient_margin"

        return True, "ok"

    # ─── Position tracking ─────────────────────────────────────────────

    def register_position(self, position: dict[str, Any]) -> None:
        """Enregistre une position ouverte."""
        self._open_positions.append(position)
        self._total_orders += 1

    def unregister_position(self, symbol: str) -> dict[str, Any] | None:
        """Retire une position fermée. Retourne la position ou None."""
        for i, pos in enumerate(self._open_positions):
            if pos["symbol"] == symbol:
                return self._open_positions.pop(i)
        return None

    @property
    def open_positions_count(self) -> int:
        return len(self._open_positions)

    # ─── Kill switch live ──────────────────────────────────────────────

    def record_trade_result(self, result: LiveTradeResult) -> None:
        """Enregistre un résultat et vérifie le kill switch live.

        P0 fix : utilise grid_max_session_loss_percent pour les stratégies grid.
        P1 fix : reset quotidien à minuit UTC.
        P1 fix : alerte Telegram si kill switch déclenché.
        """
        # P1 : auto-reset quotidien (minuit UTC)
        today = datetime.now(tz=timezone.utc).date()
        if today != self._session_start_date:
            logger.info(
                "RiskManager: reset session_pnl quotidien ({:+.2f} → 0.0)",
                self._session_pnl,
            )
            self._session_pnl = 0.0
            self._session_start_date = today

        self._session_pnl += result.net_pnl
        self._trade_history.append(result)

        if self._initial_capital <= 0:
            return

        loss_pct = abs(min(0, self._session_pnl)) / self._initial_capital * 100

        # P0 : seuil adapté au type de stratégie (grid vs mono)
        ks_config = self._config.risk.kill_switch
        from backend.optimization import is_grid_strategy

        if is_grid_strategy(result.strategy_name):
            max_loss = getattr(ks_config, "grid_max_session_loss_percent", None) or 25.0
        else:
            max_loss = ks_config.max_session_loss_percent

        if loss_pct >= max_loss:
            self._kill_switch_triggered = True
            logger.warning(
                "KILL SWITCH LIVE: perte session {:.1f}% >= {:.1f}% (stratégie={})",
                loss_pct, max_loss, result.strategy_name or "unknown",
            )

            # P1 : alerte Telegram
            if self._notifier:
                try:
                    from backend.alerts.notifier import AnomalyType
                    asyncio.get_running_loop().create_task(
                        self._notifier.notify_anomaly(
                            AnomalyType.KILL_SWITCH_GLOBAL,
                            f"KILL SWITCH LIVE déclenché ! perte={loss_pct:.1f}% "
                            f"/ seuil={max_loss:.1f}% (stratégie={result.strategy_name})",
                        )
                    )
                except Exception as e:
                    logger.error("RiskManager: erreur envoi alerte Telegram: {}", e)

    @property
    def is_kill_switch_triggered(self) -> bool:
        return self._kill_switch_triggered

    # ─── Kill switch global live (P1 — drawdown fenêtre glissante) ────

    def record_balance_snapshot(self, balance: float) -> None:
        """Enregistre un snapshot de balance et vérifie le drawdown global.

        Appelé par Executor._balance_refresh_loop() après chaque fetch (5 min).
        """
        self._balance_snapshots.append((datetime.now(tz=timezone.utc), balance))
        self._check_global_kill_switch(balance)

    def _check_global_kill_switch(self, current_balance: float) -> None:
        """Vérifie le drawdown global sur la fenêtre glissante (parité paper)."""
        if self._kill_switch_triggered:
            return
        if len(self._balance_snapshots) < 2:
            return

        ks = self._config.risk.kill_switch
        threshold = getattr(ks, "global_max_loss_pct", None)
        window_hours = getattr(ks, "global_window_hours", None)
        if not isinstance(threshold, (int, float)) or not isinstance(window_hours, (int, float)):
            return

        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=window_hours)
        recent = [b for ts, b in self._balance_snapshots if ts >= cutoff]
        if not recent:
            return

        peak = max(recent)
        if peak <= 0:
            return

        drawdown_pct = (peak - current_balance) / peak * 100
        if drawdown_pct >= threshold:
            self._kill_switch_triggered = True
            logger.critical(
                "KILL SWITCH LIVE GLOBAL: drawdown {:.1f}% >= {:.1f}% "
                "(peak={:.2f}, now={:.2f})",
                drawdown_pct, threshold, peak, current_balance,
            )

            # Alerte Telegram
            if self._notifier:
                try:
                    from backend.alerts.notifier import AnomalyType
                    asyncio.get_running_loop().create_task(
                        self._notifier.notify_anomaly(
                            AnomalyType.KILL_SWITCH_GLOBAL,
                            f"KILL SWITCH LIVE GLOBAL ! drawdown={drawdown_pct:.1f}% "
                            f"/ seuil={threshold:.0f}% "
                            f"(peak={peak:.2f}$, now={current_balance:.2f}$)",
                        )
                    )
                except Exception as e:
                    logger.error("RiskManager: erreur envoi alerte Telegram global: {}", e)

    # ─── State persistence ─────────────────────────────────────────────

    def get_state(self) -> dict[str, Any]:
        """Sérialise l'état pour le StateManager."""
        return {
            "session_pnl": self._session_pnl,
            "kill_switch": self._kill_switch_triggered,
            "total_orders": self._total_orders,
            "initial_capital": self._initial_capital,
            "positions": list(self._open_positions),
            "session_start_date": self._session_start_date.isoformat(),
            "trade_history": [
                {
                    "net_pnl": t.net_pnl,
                    "timestamp": t.timestamp.isoformat(),
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "exit_reason": t.exit_reason,
                    "strategy_name": t.strategy_name,
                }
                for t in self._trade_history
            ],
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restaure l'état depuis un snapshot."""
        self._session_pnl = state.get("session_pnl", 0.0)
        self._kill_switch_triggered = state.get("kill_switch", False)
        self._total_orders = state.get("total_orders", 0)
        self._initial_capital = state.get("initial_capital", 0.0)
        # P1 : restaurer session_start_date (si absent, la date courante)
        ssd = state.get("session_start_date")
        if ssd:
            try:
                self._session_start_date = date.fromisoformat(ssd)
            except (ValueError, TypeError):
                self._session_start_date = datetime.now(tz=timezone.utc).date()
        # Les positions seront réconciliées par Executor._reconcile_on_boot()
        # On ne restaure pas _open_positions ici (source de vérité = exchange)

        logger.info(
            "RiskManager: état restauré — pnl={:+.2f}, kill_switch={}, orders={}",
            self._session_pnl,
            self._kill_switch_triggered,
            self._total_orders,
        )

    # ─── Correlation groups ─────────────────────────────────────────────

    def _get_correlation_group(self, symbol: str) -> str | None:
        """Retourne le groupe de corrélation d'un symbole, ou None."""
        # Le symbole dans _open_positions est au format futures (BTC/USDT:USDT)
        # mais dans config.assets c'est le format spot (BTC/USDT).
        # On compare en enlevant le suffixe :USDT si présent.
        spot_symbol = symbol.split(":")[0] if ":" in symbol else symbol
        for asset in self._config.assets:
            if asset.symbol == spot_symbol:
                return asset.correlation_group
        return None

    def _get_max_same_direction(self, group_name: str) -> int:
        """Retourne max_concurrent_same_direction pour un groupe."""
        group_cfg = self._config.correlation_groups.get(group_name)
        if group_cfg is None:
            return 999  # Pas de limite si groupe non configuré
        return group_cfg.max_concurrent_same_direction

    # ─── Status ────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        return {
            "session_pnl": self._session_pnl,
            "kill_switch": self._kill_switch_triggered,
            "open_positions_count": len(self._open_positions),
            "total_orders": self._total_orders,
            "initial_capital": self._initial_capital,
            "session_start_date": self._session_start_date.isoformat(),
        }
