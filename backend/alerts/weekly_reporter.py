"""Rapport Telegram hebdomadaire — Sprint 49.

Génère et envoie un résumé de performance chaque lundi à 08:00 UTC.
Réutilisable en CLI via `generate_report()`.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from backend.alerts.telegram import TelegramClient
    from backend.core.config import AppConfig
    from backend.core.database import Database


# ─── HELPERS ──────────────────────────────────────────────────────────────────


def _classify_strategies(config: AppConfig) -> tuple[list[str], list[str]]:
    """Retourne (live_strategies, paper_strategies) depuis la config."""
    live: list[str] = []
    paper: list[str] = []
    for name in config.strategies.model_fields:
        if name == "custom_strategies":
            continue
        cfg = getattr(config.strategies, name, None)
        if cfg is None or not getattr(cfg, "enabled", False):
            continue
        if getattr(cfg, "live_eligible", False):
            live.append(name)
        else:
            paper.append(name)
    return live, paper


def _get_week_bounds() -> tuple[datetime, datetime]:
    """Retourne (lundi 00:00 UTC, dimanche 23:59:59 UTC) de la semaine précédente."""
    now = datetime.now(tz=timezone.utc)
    # Lundi de cette semaine
    monday_this_week = now - timedelta(days=now.weekday())
    monday_this_week = monday_this_week.replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    # Semaine précédente
    week_start = monday_this_week - timedelta(days=7)
    week_end = monday_this_week - timedelta(seconds=1)
    return week_start, week_end


def _fmt_pnl(value: float) -> str:
    """Formate un P&L avec signe."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}$"


def _strategy_display(name: str) -> str:
    """Nom d'affichage d'une stratégie (ex: GRID_ATR)."""
    return name.upper()


# ─── COLLECTE DE DONNÉES ─────────────────────────────────────────────────────


async def _get_live_strategy_section(
    db: Database,
    config: AppConfig,
    name: str,
    since_iso: str,
) -> dict[str, Any]:
    """Collecte les données d'une stratégie LIVE."""
    stats = await db.get_live_stats(period="7d", strategy=name)
    summary = await db.get_daily_pnl_summary(strategy=name)
    per_asset = await db.get_live_per_asset_stats(period="7d", strategy=name)
    max_dd = await db.get_max_drawdown_from_snapshots(strategy=name, period="7d")

    # Dernier snapshot de balance
    snapshots = await db.get_balance_snapshots(strategy=name, days=1)
    balance = snapshots[-1]["equity"] if snapshots else None

    # Leverage depuis la config
    strat_cfg = getattr(config.strategies, name, None)
    leverage = getattr(strat_cfg, "leverage", None)

    # Top / Worst assets
    top_assets = []
    worst_asset = None
    if per_asset:
        for a in per_asset[:2]:
            if a["total_pnl"] > 0:
                top_assets.append(a)
        negatives = [a for a in per_asset if a["total_pnl"] < 0]
        if negatives:
            worst_asset = negatives[-1]  # per_asset trié DESC par total_pnl

    return {
        "name": name,
        "is_live": True,
        "stats": stats,
        "total_pnl": summary.get("total_pnl", 0),
        "balance": balance,
        "leverage": leverage,
        "max_dd": max_dd,
        "top_assets": top_assets,
        "worst_asset": worst_asset,
    }


async def _get_paper_strategy_section(
    db: Database,
    name: str,
    since_iso: str,
) -> dict[str, Any]:
    """Collecte les données d'une stratégie PAPER via simulation_trades."""
    assert db._conn is not None

    # Stats agrégées sur la semaine
    cursor = await db._conn.execute(
        "SELECT COUNT(*) as total_trades, "
        "SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins, "
        "COALESCE(SUM(net_pnl), 0) as total_pnl_week "
        "FROM simulation_trades "
        "WHERE strategy_name = ? AND exit_time >= ?",
        (name, since_iso),
    )
    row = await cursor.fetchone()
    total_trades = row["total_trades"] if row else 0
    wins = row["wins"] if row and row["wins"] else 0
    total_pnl_week = round(row["total_pnl_week"], 2) if row else 0.0
    win_rate = round(wins / total_trades * 100, 1) if total_trades > 0 else 0.0

    # P&L total (all time)
    cursor = await db._conn.execute(
        "SELECT COALESCE(SUM(net_pnl), 0) FROM simulation_trades "
        "WHERE strategy_name = ?",
        (name,),
    )
    row = await cursor.fetchone()
    total_pnl_all = round(row[0], 2) if row else 0.0

    return {
        "name": name,
        "is_live": False,
        "stats": {
            "total_trades": total_trades,
            "wins": wins,
            "win_rate": win_rate,
            "total_pnl": total_pnl_week,
        },
        "total_pnl": total_pnl_all,
        "balance": None,
        "leverage": None,
        "max_dd": None,
        "top_assets": [],
        "worst_asset": None,
    }


async def _compute_uptime(db: Database) -> float | None:
    """Calcule l'uptime depuis les balance_snapshots (7 derniers jours)."""
    snapshots = await db.get_balance_snapshots(days=7)
    if not snapshots:
        return None
    # 1 snapshot par heure attendu, toutes stratégies confondues
    # Compter les snapshots uniques par heure (arrondi)
    unique_hours: set[str] = set()
    for s in snapshots:
        ts = s.get("timestamp", "")
        if len(ts) >= 13:
            unique_hours.add(ts[:13])  # "2026-02-18T08" → heure unique
    expected = 168  # 7 jours × 24 heures
    actual = len(unique_hours)
    if actual == 0:
        return None
    return round(min(actual / expected * 100, 100), 1)


# ─── GÉNÉRATION DU RAPPORT ───────────────────────────────────────────────────


async def generate_report(db: Database, config: AppConfig) -> str:
    """Génère le rapport hebdomadaire. Réutilisable par le script CLI."""
    week_start, week_end = _get_week_bounds()
    since_iso = week_start.isoformat()

    live_names, paper_names = _classify_strategies(config)

    # Collecte parallèle
    tasks = []
    for name in live_names:
        tasks.append(_get_live_strategy_section(db, config, name, since_iso))
    for name in paper_names:
        tasks.append(_get_paper_strategy_section(db, name, since_iso))

    sections = await asyncio.gather(*tasks, return_exceptions=True)

    # Filtrer les erreurs
    valid_sections: list[dict[str, Any]] = []
    for s in sections:
        if isinstance(s, Exception):
            logger.error("WeeklyReport: erreur collecte section: {}", s)
        else:
            valid_sections.append(s)

    # Agrégation globale
    global_pnl_week = 0.0
    global_pnl_total = 0.0
    global_trades = 0
    global_wins = 0
    total_balance = 0.0
    has_balance = False

    for s in valid_sections:
        stats = s["stats"]
        global_pnl_week += stats.get("total_pnl", 0)
        global_pnl_total += s.get("total_pnl", 0)
        global_trades += stats.get("total_trades", 0)
        global_wins += stats.get("wins", 0)
        if s.get("balance") is not None:
            total_balance += s["balance"]
            has_balance = True

    global_wr = (
        round(global_wins / global_trades * 100, 0)
        if global_trades > 0 else 0
    )
    pnl_week_pct = (
        round(global_pnl_week / total_balance * 100, 1)
        if has_balance and total_balance > 0 else 0.0
    )

    # Formatage dates
    date_fmt = "%d %b"
    date_start = week_start.strftime(date_fmt)
    date_end = week_end.strftime(date_fmt)

    lines: list[str] = []

    # Header
    lines.append(
        f"\U0001f4ca SCALP-RADAR \u2014 Rapport Hebdo ({date_start} - {date_end})"
    )
    lines.append("\u2501" * 35)

    # Global
    lines.append("\U0001f4e6 GLOBAL")
    if has_balance:
        lines.append(f"Solde total     : {total_balance:,.0f} USDT")
    lines.append(
        f"P&L Semaine     : {_fmt_pnl(global_pnl_week)} ({pnl_week_pct:+.1f}%)"
    )
    lines.append(f"P&L Total       : {_fmt_pnl(global_pnl_total)}")
    lines.append(
        f"Trades          : {global_trades} (WR {global_wr:.0f}%)"
    )

    # Sections par stratégie
    for s in valid_sections:
        lines.append("")
        stats = s["stats"]
        display = _strategy_display(s["name"])

        if s["is_live"]:
            # Header live
            parts = [f"\u26a1 {display}"]
            if s.get("balance") is not None:
                parts[0] += f" ({s['balance']:,.0f}$"
                if s.get("leverage"):
                    parts[0] += f", x{s['leverage']}"
                parts[0] += ")"
            lines.append(parts[0])

            pnl_week = stats.get("total_pnl", 0)
            trades = stats.get("total_trades", 0)
            wr = stats.get("win_rate", 0)
            lines.append(f"P&L Semaine     : {_fmt_pnl(pnl_week)}")
            lines.append(f"Trades          : {trades} (WR {wr:.0f}%)")

            if s.get("max_dd") is not None:
                lines.append(f"Max DD          : {s['max_dd']:.1f}%")

            # Top / Worst
            top_parts = []
            for a in s.get("top_assets", []):
                sym = a["symbol"].replace("/USDT", "")
                top_parts.append(f"{sym} {_fmt_pnl(a['total_pnl'])}")
            if s.get("worst_asset"):
                wa = s["worst_asset"]
                sym = wa["symbol"].replace("/USDT", "")
                worst_str = f"{sym} {_fmt_pnl(wa['total_pnl'])}"
            else:
                worst_str = None

            if top_parts or worst_str:
                line = "Top : " + " | ".join(top_parts) if top_parts else ""
                if worst_str:
                    if line:
                        line += f"\nWorst : {worst_str}"
                    else:
                        line = f"Worst : {worst_str}"
                lines.append(line)
        else:
            # Header paper
            lines.append(f"\U0001f441\ufe0f {display} (paper)")
            pnl_week = stats.get("total_pnl", 0)
            trades = stats.get("total_trades", 0)
            wr = stats.get("win_rate", 0)
            lines.append(f"P&L Semaine     : {_fmt_pnl(pnl_week)}")
            lines.append(f"Trades          : {trades}" + (f" (WR {wr:.0f}%)" if trades > 0 else ""))

    # Uptime
    uptime = await _compute_uptime(db)
    if uptime is not None:
        lines.append("")
        lines.append(f"\u2699\ufe0f Uptime : {uptime}%")

    return "\n".join(lines)


# ─── CLASSE SCHEDULER ────────────────────────────────────────────────────────


class WeeklyReporter:
    """Envoie un rapport hebdomadaire chaque lundi à 08:00 UTC."""

    def __init__(
        self,
        telegram: TelegramClient,
        db: Database,
        config: AppConfig,
    ) -> None:
        self._telegram = telegram
        self._db = db
        self._config = config
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("WeeklyReporter: activé (envoi lundi 08:00 UTC)")

    async def _loop(self) -> None:
        while self._running:
            try:
                wait = self._seconds_until_next_monday_8utc()
                logger.info(
                    "WeeklyReporter: prochain rapport dans {:.1f}h",
                    wait / 3600,
                )
                await asyncio.sleep(wait)
                if not self._running:
                    break
                report = await generate_report(self._db, self._config)
                await self._telegram.send_message(report)
                logger.info("WeeklyReporter: rapport envoyé")
                # Petit sleep pour éviter double envoi
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("WeeklyReporter: erreur: {}", e)
                # Retry dans 1h en cas d'erreur
                await asyncio.sleep(3600)

    @staticmethod
    def _seconds_until_next_monday_8utc() -> float:
        """Calcule le nombre de secondes jusqu'au prochain lundi 08:00 UTC."""
        now = datetime.now(tz=timezone.utc)
        days_until_monday = (7 - now.weekday()) % 7
        next_monday = (now + timedelta(days=days_until_monday)).replace(
            hour=8, minute=0, second=0, microsecond=0,
        )
        # Si on est lundi mais déjà passé 08:00 → lundi prochain
        if next_monday <= now:
            next_monday += timedelta(days=7)
        return (next_monday - now).total_seconds()

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("WeeklyReporter: arrêté")
