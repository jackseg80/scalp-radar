"""RiskManager : vérifications pré-trade, kill switch live, marge.

Séparé de l'Executor pour clarifier les responsabilités.
Double kill switch : le Simulator a le sien (virtuel),
le RiskManager a le sien (argent réel).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from backend.core.config import AppConfig


@dataclass
class LiveTradeResult:
    """Résultat d'un trade live (pour le kill switch live)."""

    net_pnl: float
    timestamp: datetime
    symbol: str
    direction: str
    exit_reason: str


class LiveRiskManager:
    """Vérifications pré-trade et kill switch live.

    Le RiskManager a un kill switch indépendant basé sur le capital réel.
    Il ne touche pas au kill switch du Simulator (virtuel, par runner).
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._initial_capital: float = 0.0
        self._session_pnl: float = 0.0
        self._kill_switch_triggered: bool = False
        self._open_positions: list[dict[str, Any]] = []
        self._trade_history: list[LiveTradeResult] = []
        self._total_orders: int = 0

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
    ) -> tuple[bool, str]:
        """Vérifie toutes les conditions avant de passer un ordre.

        Retourne (ok, reason).
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
        leverage = self._config.risk.position.default_leverage
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
        """Enregistre un résultat et vérifie le kill switch live."""
        self._session_pnl += result.net_pnl
        self._trade_history.append(result)

        if self._initial_capital <= 0:
            return

        loss_pct = abs(min(0, self._session_pnl)) / self._initial_capital * 100
        max_loss = self._config.risk.kill_switch.max_session_loss_percent

        if loss_pct >= max_loss:
            self._kill_switch_triggered = True
            logger.warning(
                "KILL SWITCH LIVE: perte session {:.1f}% >= {:.1f}%",
                loss_pct, max_loss,
            )

    @property
    def is_kill_switch_triggered(self) -> bool:
        return self._kill_switch_triggered

    # ─── State persistence ─────────────────────────────────────────────

    def get_state(self) -> dict[str, Any]:
        """Sérialise l'état pour le StateManager."""
        return {
            "session_pnl": self._session_pnl,
            "kill_switch": self._kill_switch_triggered,
            "total_orders": self._total_orders,
            "initial_capital": self._initial_capital,
            "positions": list(self._open_positions),
            "trade_history": [
                {
                    "net_pnl": t.net_pnl,
                    "timestamp": t.timestamp.isoformat(),
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "exit_reason": t.exit_reason,
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
        # Les positions seront réconciliées par Executor._reconcile_on_boot()
        # On ne restaure pas _open_positions ici (source de vérité = exchange)

        logger.info(
            "RiskManager: état restauré — pnl={:+.2f}, kill_switch={}, orders={}",
            self._session_pnl,
            self._kill_switch_triggered,
            self._total_orders,
        )

    # ─── Status ────────────────────────────────────────────────────────

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
        }
