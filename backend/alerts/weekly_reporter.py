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


# ─── CONSTANTES ──────────────────────────────────────────────────────────────

_CLOSE_TYPES = "('tp_close', 'sl_close', 'force_close', 'close', 'cycle_close')"


# ─── HELPERS GÉNÉRAUX ────────────────────────────────────────────────────────


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


def _get_week_bounds(current: bool = False) -> tuple[datetime, datetime]:
    """Retourne (lundi 00:00 UTC, fin de période) selon le mode.

    current=False (défaut) : semaine précédente complète (lun → dim 23:59:59)
    current=True           : semaine en cours (lun 00:00 → maintenant)
    """
    now = datetime.now(tz=timezone.utc)
    monday_this_week = (now - timedelta(days=now.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    if current:
        return monday_this_week, now
    week_start = monday_this_week - timedelta(days=7)
    week_end = monday_this_week - timedelta(seconds=1)
    return week_start, week_end


def _fmt_pnl(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}$"


# ─── REQUÊTES SQL DIRECTES ────────────────────────────────────────────────────


async def _live_week_stats(
    conn: Any, name: str, since: str, until: str,
) -> dict[str, Any]:
    """Stats live sur la plage exacte [since, until]."""
    # Trades agrégés
    cursor = await conn.execute(
        f"SELECT pnl, symbol FROM live_trades "
        f"WHERE trade_type IN {_CLOSE_TYPES} AND strategy_name = ? "
        f"AND timestamp >= ? AND timestamp <= ? "
        f"ORDER BY timestamp ASC",
        (name, since, until),
    )
    rows = await cursor.fetchall()
    total = len(rows)
    pnls = [r["pnl"] or 0.0 for r in rows]
    wins = sum(1 for p in pnls if p > 0)
    win_rate = round(wins / total * 100, 1) if total > 0 else 0.0
    total_pnl = round(sum(pnls), 2)

    # Per-asset
    cursor = await conn.execute(
        f"SELECT symbol, COUNT(*) as n, "
        f"SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins, "
        f"ROUND(SUM(pnl), 2) as total_pnl "
        f"FROM live_trades "
        f"WHERE trade_type IN {_CLOSE_TYPES} AND strategy_name = ? "
        f"AND timestamp >= ? AND timestamp <= ? "
        f"GROUP BY symbol ORDER BY total_pnl DESC",
        (name, since, until),
    )
    asset_rows = await cursor.fetchall()

    return {
        "total_trades": total,
        "wins": wins,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "per_asset": [dict(r) for r in asset_rows],
    }


async def _live_total_pnl(conn: Any, name: str) -> float:
    """P&L total all-time depuis live_trades."""
    cursor = await conn.execute(
        f"SELECT COALESCE(SUM(pnl), 0) FROM live_trades "
        f"WHERE trade_type IN {_CLOSE_TYPES} AND strategy_name = ?",
        (name,),
    )
    row = await cursor.fetchone()
    return round(float(row[0] or 0), 2) if row else 0.0


async def _latest_balance(conn: Any, name: str) -> float | None:
    """Dernier snapshot de balance pour une stratégie (quel que soit l'âge)."""
    cursor = await conn.execute(
        "SELECT equity FROM balance_snapshots "
        "WHERE strategy_name = ? ORDER BY timestamp DESC LIMIT 1",
        (name,),
    )
    row = await cursor.fetchone()
    return float(row["equity"]) if row else None


async def _max_drawdown_week(conn: Any, name: str, since: str) -> float | None:
    """Max drawdown sur les 7 derniers jours depuis balance_snapshots."""
    cursor = await conn.execute(
        "SELECT equity FROM balance_snapshots "
        "WHERE strategy_name = ? AND timestamp >= ? ORDER BY timestamp ASC",
        (name, since),
    )
    rows = await cursor.fetchall()
    if len(rows) < 2:
        return None
    peak = float(rows[0]["equity"])
    max_dd = 0.0
    for row in rows:
        eq = float(row["equity"])
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (eq - peak) / peak
            if dd < max_dd:
                max_dd = dd
    return round(max_dd * 100, 2)


async def _paper_week_stats(
    conn: Any, name: str, since: str, until: str,
) -> dict[str, Any]:
    """Stats paper sur la plage exacte [since, until]."""
    cursor = await conn.execute(
        "SELECT COUNT(*) as n, "
        "SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins, "
        "COALESCE(SUM(net_pnl), 0) as total_pnl "
        "FROM simulation_trades "
        "WHERE strategy_name = ? AND exit_time >= ? AND exit_time <= ?",
        (name, since, until),
    )
    row = await cursor.fetchone()
    total = int(row["n"] or 0) if row else 0
    wins = int(row["wins"] or 0) if row else 0
    pnl_week = round(float(row["total_pnl"] or 0), 2) if row else 0.0
    win_rate = round(wins / total * 100, 1) if total > 0 else 0.0

    # P&L total all-time pour cette stratégie
    cursor = await conn.execute(
        "SELECT COALESCE(SUM(net_pnl), 0) FROM simulation_trades "
        "WHERE strategy_name = ?",
        (name,),
    )
    row = await cursor.fetchone()
    total_pnl_all = round(float(row[0] or 0), 2) if row else 0.0

    return {
        "total_trades": total,
        "wins": wins,
        "win_rate": win_rate,
        "total_pnl_week": pnl_week,
        "total_pnl_all": total_pnl_all,
    }


async def _compute_uptime(conn: Any, since: str) -> float | None:
    """Uptime depuis les balance_snapshots sur la période."""
    cursor = await conn.execute(
        "SELECT timestamp FROM balance_snapshots WHERE timestamp >= ? "
        "ORDER BY timestamp ASC",
        (since,),
    )
    rows = await cursor.fetchall()
    if not rows:
        return None
    # Compter les heures uniques (1 snapshot/heure attendu)
    unique_hours: set[str] = {row["timestamp"][:13] for row in rows}
    expected = 168  # 7j × 24h
    return round(min(len(unique_hours) / expected * 100, 100.0), 1)


# ─── GÉNÉRATION DU RAPPORT ───────────────────────────────────────────────────


async def generate_report(
    db: Database, config: AppConfig, current_week: bool = False,
) -> str:
    """Génère le rapport hebdomadaire.

    current_week=False (défaut) : semaine précédente complète (lun-dim)
    current_week=True           : semaine en cours (lun → maintenant)
    Réutilisable par le script CLI.
    """
    week_start, week_end = _get_week_bounds(current=current_week)
    since_iso = week_start.isoformat()
    until_iso = week_end.isoformat()
    conn = db._conn
    assert conn is not None, "Database non initialisée"

    live_names, paper_names = _classify_strategies(config)

    # ── Collecte données LIVE ─────────────────────────────────────────────
    live_sections: list[dict[str, Any]] = []
    for name in live_names:
        try:
            stats = await _live_week_stats(conn, name, since_iso, until_iso)
            total_pnl = await _live_total_pnl(conn, name)
            balance = await _latest_balance(conn, name)
            max_dd = await _max_drawdown_week(conn, name, since_iso)

            strat_cfg = getattr(config.strategies, name, None)
            leverage = getattr(strat_cfg, "leverage", None)

            per_asset = stats["per_asset"]
            top_assets = [a for a in per_asset if a["total_pnl"] > 0][:2]
            negatives = [a for a in per_asset if a["total_pnl"] < 0]
            # per_asset trié DESC → negatives[-1] = le plus négatif = worst
            worst_asset = negatives[-1] if negatives else None

            live_sections.append({
                "name": name,
                "stats": stats,
                "total_pnl": total_pnl,
                "balance": balance,
                "leverage": leverage,
                "max_dd": max_dd,
                "top_assets": top_assets,
                "worst_asset": worst_asset,
            })
        except Exception as e:
            logger.error("WeeklyReport: erreur données live {}: {}", name, e)

    # ── Collecte données PAPER ────────────────────────────────────────────
    paper_sections: list[dict[str, Any]] = []
    for name in paper_names:
        try:
            data = await _paper_week_stats(conn, name, since_iso, until_iso)
            paper_sections.append({"name": name, "data": data})
        except Exception as e:
            logger.error("WeeklyReport: erreur données paper {}: {}", name, e)

    # ── Agrégation GLOBAL (live uniquement) ───────────────────────────────
    global_pnl_week = sum(s["stats"]["total_pnl"] for s in live_sections)
    global_pnl_total = sum(s["total_pnl"] for s in live_sections)
    global_trades = sum(s["stats"]["total_trades"] for s in live_sections)
    global_wins = sum(s["stats"]["wins"] for s in live_sections)
    global_wr = round(global_wins / global_trades * 100) if global_trades > 0 else 0

    balances = [s["balance"] for s in live_sections if s["balance"] is not None]
    total_balance = sum(balances)
    has_balance = len(balances) > 0
    pnl_week_pct = (
        global_pnl_week / total_balance * 100
        if has_balance and total_balance > 0 else None
    )

    # ── Formatage ─────────────────────────────────────────────────────────
    date_fmt = "%d %b"
    label = "Semaine en cours" if current_week else "Rapport Hebdo"
    date_range = f"{week_start.strftime(date_fmt)} - {week_end.strftime(date_fmt)}"

    lines: list[str] = []
    lines.append(f"\U0001f4ca SCALP-RADAR \u2014 {label} ({date_range})")
    lines.append("\u2501" * 35)

    # GLOBAL
    lines.append("\U0001f4e6 GLOBAL")
    if has_balance:
        lines.append(f"Solde total     : {total_balance:,.0f} USDT")
    pnl_str = _fmt_pnl(global_pnl_week)
    if pnl_week_pct is not None:
        pnl_str += f" ({pnl_week_pct:+.1f}%)"
    lines.append(f"P&L Semaine     : {pnl_str}")
    lines.append(f"P&L Total       : {_fmt_pnl(global_pnl_total)}")
    lines.append(f"Trades          : {global_trades} (WR {global_wr:.0f}%)")

    # Sections LIVE
    for s in live_sections:
        lines.append("")
        stats = s["stats"]
        display = s["name"].upper()

        # Header : NOM (balance$, xN)
        header = f"\u26a1 {display}"
        meta: list[str] = []
        if s["balance"] is not None:
            meta.append(f"{s['balance']:,.0f}$")
        if s["leverage"]:
            meta.append(f"x{s['leverage']}")
        if meta:
            header += f" ({', '.join(meta)})"
        lines.append(header)

        lines.append(f"P&L Semaine     : {_fmt_pnl(stats['total_pnl'])}")
        lines.append(
            f"Trades          : {stats['total_trades']}"
            + (f" (WR {stats['win_rate']:.0f}%)" if stats["total_trades"] > 0 else "")
        )
        if s["max_dd"] is not None:
            lines.append(f"Max DD          : {s['max_dd']:.1f}%")

        # Top assets
        top_str = " | ".join(
            f"{a['symbol'].replace('/USDT', '')} {_fmt_pnl(a['total_pnl'])}"
            for a in s["top_assets"]
        )
        if top_str:
            lines.append(f"Top : {top_str}")
        if s["worst_asset"]:
            wa = s["worst_asset"]
            sym = wa["symbol"].replace("/USDT", "")
            lines.append(f"Worst : {sym} {_fmt_pnl(wa['total_pnl'])}")

    # Sections PAPER
    for p in paper_sections:
        lines.append("")
        d = p["data"]
        display = p["name"].upper()
        lines.append(f"\U0001f441\ufe0f {display} (paper)")
        lines.append(f"P&L Semaine     : {_fmt_pnl(d['total_pnl_week'])}")
        if d["total_trades"] > 0:
            lines.append(
                f"Trades          : {d['total_trades']} (WR {d['win_rate']:.0f}%)"
            )
        else:
            lines.append("Trades          : 0")

    # Uptime
    try:
        uptime = await _compute_uptime(conn, since_iso)
        if uptime is not None:
            lines.append("")
            lines.append(f"\u2699\ufe0f Uptime : {uptime}%")
    except Exception as e:
        logger.warning("WeeklyReport: erreur uptime: {}", e)

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
                # Petit sleep pour éviter double envoi si redémarrage rapide
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("WeeklyReporter: erreur: {}", e)
                await asyncio.sleep(3600)

    @staticmethod
    def _seconds_until_next_monday_8utc() -> float:
        """Calcule le nombre de secondes jusqu'au prochain lundi 08:00 UTC.

        Correct après un restart en milieu de semaine : attend toujours
        le prochain lundi (pas 7 jours depuis maintenant).
        """
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
