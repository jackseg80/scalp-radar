"""Analyse de régression de performance WFO — comparaison avant/après re-run.

Lancement :
    uv run python -m scripts.analyze_wfo_regression --strategy grid_atr
    uv run python -m scripts.analyze_wfo_regression --strategy grid_atr --db data/scalp_radar.db
    uv run python -m scripts.analyze_wfo_regression --strategy grid_atr --losers NEAR/USDT DOGE/USDT OP/USDT AVAX/USDT
    uv run python -m scripts.analyze_wfo_regression --strategy grid_atr --leverage 7 --kill-switch 45

Sortie :
    Console (rapport formaté)
    data/analysis/wfo_regression_report.json
    data/analysis/wfo_regression_report.txt
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


# ─── Helpers ────────────────────────────────────────────────────────────────


def _db_path_default() -> str:
    for c in ("data/scalp_radar.db", str(Path(__file__).parent.parent / "data" / "scalp_radar.db")):
        if Path(c).exists():
            return c
    return "data/scalp_radar.db"


def _parse_json(val: str | None, default: Any = None) -> Any:
    if not val:
        return default
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return default


def _fmt_pct(val: float | None, decimals: int = 1) -> str:
    if val is None:
        return "N/A"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.{decimals}f}%"


def _fmt_delta_pct(old: float | None, new: float | None) -> str:
    if old is None or new is None or old == 0:
        return "N/A"
    delta = (new - old) / abs(old) * 100
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}%"


def _fv(val: float | None, dec: int = 2) -> str:
    return f"{val:.{dec}f}" if val is not None else "N/A"


def _sep(w: int = 100) -> str:
    return "─" * w


def _hdr(title: str, w: int = 100) -> str:
    pad = (w - len(title) - 2) // 2
    return "═" * pad + f" {title} " + "═" * (w - pad - len(title) - 2)


# ─── Requêtes DB ────────────────────────────────────────────────────────────


def fetch_wfo_results(conn: sqlite3.Connection, strategy: str) -> dict[str, list[dict]]:
    """Retourne {asset: [latest, previous?, ...]} triés DESC par date."""
    cursor = conn.execute(
        """
        SELECT id, asset, created_at, grade, total_score, oos_sharpe,
               consistency, best_params, regime_analysis,
               validation_summary, n_windows, n_distinct_combos, is_latest
        FROM optimization_results
        WHERE strategy_name = ?
        ORDER BY asset, created_at DESC
        """,
        (strategy,),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    by_asset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_asset[row["asset"]].append(row)
    return {asset: runs[:2] for asset, runs in by_asset.items()}


def fetch_portfolio(conn: sqlite3.Connection, strategy: str, offset: int = 0) -> dict | None:
    cursor = conn.execute(
        """
        SELECT id, strategy_name, initial_capital, n_assets, period_days,
               assets, leverage, kill_switch_pct, total_return_pct,
               max_drawdown_pct, total_trades, win_rate, realized_pnl,
               per_asset_results, kill_switch_triggers, peak_margin_ratio,
               created_at, label
        FROM portfolio_backtests
        WHERE strategy_name LIKE ?
        ORDER BY created_at DESC
        LIMIT 1 OFFSET ?
        """,
        (f"%{strategy}%", offset),
    )
    row = cursor.fetchone()
    if not row:
        return None
    d = dict(row)
    d["per_asset_results"] = _parse_json(d.get("per_asset_results"), {})
    d["assets"] = _parse_json(d.get("assets"), [])
    return d


def _params(row: dict) -> dict:
    return _parse_json(row.get("best_params"), {})


# ─── Section 1 : Diff paramétrique ─────────────────────────────────────────

NUMERIC_PARAMS = [
    "ma_period",
    "atr_period",
    "atr_multiplier_start",
    "atr_multiplier_step",
    "num_levels",
    "sl_percent",
    "atr_spacing_mult",
    "bol_window",
    "bol_std",
    "long_ma_window",
    "min_bol_spread",
]


def section1_param_diff(
    by_asset: dict[str, list[dict]],
    per_asset_pnl: dict | None,
) -> tuple[list[dict], list[str]]:
    lines: list[str] = []
    records: list[dict] = []

    # ── 1a ──────────────────────────────────────────────────────────────────
    lines.append(_hdr("SECTION 1a — Diff Paramétrique par Asset"))
    lines.append(f"  {'Asset':<18} {'Paramètre':<26} {'Ancien':>10} {'Nouveau':>10} {'Delta':>10}")
    lines.append("  " + _sep(76))

    assets_changed = 0
    for asset in sorted(by_asset):
        runs = by_asset[asset]
        if len(runs) < 2:
            lines.append(f"  {asset:<18}  (un seul run — pas de comparaison possible)")
            continue

        p_new = _params(runs[0])
        p_old = _params(runs[1])
        changed = False

        for param in NUMERIC_PARAMS:
            v_old = p_old.get(param)
            v_new = p_new.get(param)
            if v_old is None and v_new is None:
                continue
            if v_old == v_new:
                continue
            delta = _fmt_delta_pct(v_old, v_new)
            lines.append(f"  {asset:<18} {param:<26} {_fv(v_old):>10} {_fv(v_new):>10} {delta:>10}")
            records.append({"asset": asset, "param": param, "old": v_old, "new": v_new,
                             "delta_pct": (v_new - v_old) / abs(v_old) * 100 if v_old and v_old != 0 else None})
            changed = True

        tf_old = p_old.get("timeframe")
        tf_new = p_new.get("timeframe")
        if tf_old != tf_new and (tf_old or tf_new):
            lines.append(f"  {asset:<18} {'timeframe':<26} {str(tf_old or '?'):>10} {str(tf_new or '?'):>10} {'(changed)':>10}")
            changed = True

        if not changed:
            lines.append(f"  {asset:<18}  (aucun changement de paramètres)")
        else:
            assets_changed += 1

    lines.append(f"\n  → {assets_changed} assets avec des changements de paramètres\n")

    # ── 1b ──────────────────────────────────────────────────────────────────
    lines.append(_hdr("SECTION 1b — Tendances Agrégées par Paramètre"))
    lines.append(f"  {'Paramètre':<26} {'↑ Up':>6} {'↓ Down':>6} {'= Same':>7}   {'Old avg':>9} → {'New avg':<9}")
    lines.append("  " + _sep(76))

    for param in NUMERIC_PARAMS:
        olds, news, up, dn, same = [], [], 0, 0, 0
        for asset in sorted(by_asset):
            runs = by_asset[asset]
            if len(runs) < 2:
                continue
            v_old = _params(runs[1]).get(param)
            v_new = _params(runs[0]).get(param)
            if v_old is None and v_new is None:
                continue
            if v_old is not None:
                olds.append(v_old)
            if v_new is not None:
                news.append(v_new)
            if v_old is None or v_new is None:
                continue
            if v_new > v_old:
                up += 1
            elif v_new < v_old:
                dn += 1
            else:
                same += 1
        if not (olds or news):
            continue
        old_avg = mean(olds) if olds else None
        new_avg = mean(news) if news else None
        old_s = f"{old_avg:.2f}" if old_avg is not None else "N/A"
        new_s = f"{new_avg:.2f}" if new_avg is not None else "N/A"
        lines.append(f"  {param:<26} {up:>6} {dn:>6} {same:>7}   {old_s:>9} → {new_s:<9}")

    lines.append("")

    # ── 1c ──────────────────────────────────────────────────────────────────
    lines.append(_hdr("SECTION 1c — Corrélation Params ↔ P&L Portfolio"))

    if not per_asset_pnl:
        lines.append("  ⚠️  Aucun portfolio backtest disponible — croisement impossible.")
        lines.append("     Lancer un portfolio backtest puis relancer cette analyse.\n")
        return records, lines

    lines.append(f"  {'Asset':<18} {'SL ancien':>10} {'SL nouveau':>10} {'ΔSL':>8}   {'P&L':>9} {'Trades':>7} {'WR%':>7}")
    lines.append("  " + _sep(76))

    sl_delta_pnl: list[dict] = []
    for asset in sorted(by_asset):
        runs = by_asset[asset]
        if len(runs) < 2:
            continue
        sl_old = _params(runs[1]).get("sl_percent")
        sl_new = _params(runs[0]).get("sl_percent")

        pnl_data: dict | None = None
        for key in [f"grid_atr:{asset}", f"grid_boltrend:{asset}", asset]:
            if key in per_asset_pnl:
                pnl_data = per_asset_pnl[key]
                break

        pnl_val = trades_val = wr_val = None
        if pnl_data:
            pnl_val = pnl_data.get("net_pnl", pnl_data.get("realized_pnl", pnl_data.get("total_pnl")))
            trades_val = pnl_data.get("trades", pnl_data.get("n_trades", pnl_data.get("total_trades")))
            wr_val = pnl_data.get("win_rate")
            if sl_old is not None and sl_new is not None and pnl_val is not None:
                sl_delta_pnl.append({"asset": asset, "sl_delta": sl_new - sl_old, "pnl": pnl_val})

        sl_old_s = f"{sl_old:.1f}%" if sl_old is not None else "N/A"
        sl_new_s = f"{sl_new:.1f}%" if sl_new is not None else "N/A"
        dsl_s = f"{sl_new - sl_old:+.1f}%" if sl_old is not None and sl_new is not None else "N/A"
        pnl_s = f"{pnl_val:+.1f}" if pnl_val is not None else "N/A"
        tr_s = str(trades_val) if trades_val is not None else "N/A"
        wr_s = f"{wr_val:.1f}%" if wr_val is not None else "N/A"
        lines.append(f"  {asset:<18} {sl_old_s:>10} {sl_new_s:>10} {dsl_s:>8}   {pnl_s:>9} {tr_s:>7} {wr_s:>7}")

    if sl_delta_pnl:
        sl_up = [x["pnl"] for x in sl_delta_pnl if x["sl_delta"] > 0]
        sl_same = [x["pnl"] for x in sl_delta_pnl if x["sl_delta"] == 0]
        sl_dn = [x["pnl"] for x in sl_delta_pnl if x["sl_delta"] < 0]
        lines.append(f"\n  Corrélation SL change ↔ P&L moyen :")
        if sl_up:
            lines.append(f"    SL augmenté  (n={len(sl_up):2d}) → P&L moyen : {mean(sl_up):+.1f}")
        if sl_same:
            lines.append(f"    SL inchangé  (n={len(sl_same):2d}) → P&L moyen : {mean(sl_same):+.1f}")
        if sl_dn:
            lines.append(f"    SL diminué   (n={len(sl_dn):2d}) → P&L moyen : {mean(sl_dn):+.1f}")

    lines.append("")
    return records, lines


# ─── Section 2 : Analyse par régime ─────────────────────────────────────────

REGIMES = ["bull", "bear", "range", "crash"]


def section2_regime_diff(by_asset: dict[str, list[dict]]) -> tuple[dict, list[str]]:
    lines: list[str] = []
    regime_deltas: dict[str, list[float]] = defaultdict(list)

    lines.append(_hdr("SECTION 2 — Analyse par Régime de Marché (Avant vs Après)"))
    lines.append(f"  {'Asset':<18} {'Régime':<8} {'Ancien (n, sharpe)':>20} {'Nouveau (n, sharpe)':>20} {'ΔSharpe':>9}")
    lines.append("  " + _sep(80))

    no_regime = 0
    for asset in sorted(by_asset):
        runs = by_asset[asset]
        if len(runs) < 2:
            continue
        r_old = _parse_json(runs[1].get("regime_analysis"), {})
        r_new = _parse_json(runs[0].get("regime_analysis"), {})
        if not r_old and not r_new:
            no_regime += 1
            continue

        printed = False
        for regime in REGIMES:
            d_old = r_old.get(regime) if r_old else None
            d_new = r_new.get(regime) if r_new else None
            if d_old is None and d_new is None:
                continue

            n_old = d_old.get("n_windows", "?") if d_old else "?"
            sh_old = d_old.get("avg_oos_sharpe") if d_old else None
            n_new = d_new.get("n_windows", "?") if d_new else "?"
            sh_new = d_new.get("avg_oos_sharpe") if d_new else None

            old_s = f"({n_old}, {sh_old:.2f})" if sh_old is not None else f"({n_old}, N/A)"
            new_s = f"({n_new}, {sh_new:.2f})" if sh_new is not None else f"({n_new}, N/A)"

            delta_s = "N/A"
            if sh_old is not None and sh_new is not None:
                delta = sh_new - sh_old
                delta_s = f"{delta:+.2f}"
                regime_deltas[regime].append(delta)

            lines.append(f"  {asset:<18} {regime:<8} {old_s:>20} {new_s:>20} {delta_s:>9}")
            printed = True

        if printed:
            lines.append("  " + "·" * 80)

    if no_regime:
        lines.append(f"\n  ⚠️  {no_regime} assets sans données regime_analysis")

    lines.append(f"\n  Résumé — variation moyenne du Sharpe OOS par régime :")
    lines.append(f"  {'Régime':<10} {'N':>5} {'ΔSharpe moy':>14}   Interprétation")
    lines.append("  " + _sep(60))
    for regime in REGIMES:
        deltas = regime_deltas.get(regime, [])
        if not deltas:
            lines.append(f"  {regime:<10} {'0':>5} {'N/A':>14}   pas de données")
            continue
        avg = mean(deltas)
        sign = "+" if avg > 0 else ""
        if avg > 0.2:
            interp = "✓ Amélioration significative"
        elif avg > 0:
            interp = "~ Légère amélioration"
        elif avg > -0.2:
            interp = "~ Légère dégradation"
        else:
            interp = "✗ Dégradation significative"
        lines.append(f"  {regime:<10} {len(deltas):>5} {f'{sign}{avg:.3f}':>14}   {interp}")

    lines.append("")
    return dict(regime_deltas), lines


# ─── Section 3 : Analyse SL et risque ──────────────────────────────────────


def section3_sl_risk(
    by_asset: dict[str, list[dict]],
    leverage: int,
    ks_pct: float,
) -> tuple[dict, list[str]]:
    lines: list[str] = []
    lines.append(_hdr("SECTION 3 — Analyse du SL et du Risque Portfolio"))

    sl_old: list[float] = []
    sl_new: list[float] = []

    for asset in sorted(by_asset):
        runs = by_asset[asset]
        v_new = _params(runs[0]).get("sl_percent") if runs else None
        v_old = _params(runs[1]).get("sl_percent") if len(runs) >= 2 else None
        if v_new is not None:
            sl_new.append(v_new)
        if v_old is not None:
            sl_old.append(v_old)

    avg_old = mean(sl_old) if sl_old else 0.0
    avg_new = mean(sl_new) if sl_new else 0.0
    delta_avg = avg_new - avg_old

    # Histogramme côte à côte
    BIN_DEFS = [(10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 41)]
    c_olds = [sum(1 for v in sl_old if lo <= v < hi) for lo, hi in BIN_DEFS]
    c_news = [sum(1 for v in sl_new if lo <= v < hi) for lo, hi in BIN_DEFS]
    mc = max(max(c_olds, default=1), max(c_news, default=1), 1)
    BAR = 20

    lines.append("\n  Distribution des sl_percent (ancien vs nouveau) :")
    lines.append(f"  {'Bin':>10}  {'Ancien':<{BAR + 6}}  {'Nouveau':<{BAR + 6}}")
    lines.append("  " + _sep(55))
    for (lo, hi), co, cn in zip(BIN_DEFS, c_olds, c_news):
        label = f"{lo}-{hi}%"
        bo = "█" * int(co / mc * BAR)
        bn = "█" * int(cn / mc * BAR)
        lines.append(f"  {label:>10}  {bo.ljust(BAR)} ({co:2d})  {bn.ljust(BAR)} ({cn:2d})")

    lines.append(f"\n  SL moyen  : {avg_old:.2f}%  →  {avg_new:.2f}%   (Δ {delta_avg:+.2f}%)")
    if sl_new:
        lines.append(f"  SL min/max: {min(sl_new):.1f}% / {max(sl_new):.1f}%  (nouveaux params)")

    # Worst-case théorique
    n = len(sl_new)
    worst_dd_avg = worst_dd_max = margin_avg = margin_max = 0.0

    if n > 0 and sl_new:
        sl_max_val = max(sl_new)
        # Sizing égal : capital / n par asset, 1 SL = sl/100 × leverage / n × capital
        # Tous SL simultanés = sl/100 × leverage × 100%
        worst_dd_avg = avg_new / 100 * leverage * 100
        worst_dd_max = sl_max_val / 100 * leverage * 100
        margin_avg = ks_pct - worst_dd_avg
        margin_max = ks_pct - worst_dd_max

        lines.append(f"\n  Worst-case théorique ({n} assets, leverage {leverage}x, sizing égal capital/{n}) :")
        lines.append(f"    Perte unitaire (1 SL)  = {avg_new:.1f}% × {leverage}x / {n} = {avg_new / 100 * leverage / n * 100:.1f}% du capital")
        lines.append(f"    Tous SL avg ({avg_new:.1f}%)     → DD max théorique = {worst_dd_avg:.0f}%")
        lines.append(f"    Tous SL max ({sl_max_val:.1f}%)     → DD max théorique = {worst_dd_max:.0f}%")

        lines.append(f"\n  Marge de sécurité vs kill switch {ks_pct:.0f}% :")
        ok_a = "✓ OK" if margin_avg > 0 else "✗ INSUFFISANT"
        ok_m = "✓ OK" if margin_max > 0 else "✗ INSUFFISANT"
        lines.append(f"    SL moyen : {margin_avg:+.1f}pp  {ok_a}")
        lines.append(f"    SL max   : {margin_max:+.1f}pp  {ok_m}")
        if 0 < margin_avg < 10:
            lines.append(f"    ⚠️  Marge < 10pp — envisager KS {ks_pct + 5:.0f}% ou réduire leverage")
        if margin_max < 0:
            lines.append(f"    🚨 SL max dépasse le KS — risque de déclenchement avec seulement quelques SL")

    # Assets dangereux sl > 25%
    dangerous: list[tuple[str, float | None, float]] = []
    for asset in sorted(by_asset):
        runs = by_asset[asset]
        v_new = _params(runs[0]).get("sl_percent") if runs else None
        v_old = _params(runs[1]).get("sl_percent") if len(runs) >= 2 else None
        if v_new is not None and v_new > 25:
            dangerous.append((asset, v_old, v_new))

    if dangerous:
        lines.append(f"\n  Assets avec sl_percent > 25% (risque élevé) :")
        lines.append(f"  {'Asset':<18} {'SL ancien':>10} {'SL nouveau':>10} {'sl×lev':>8}  Verdict")
        lines.append("  " + _sep(62))
        for asset, sl_o, sl_n in sorted(dangerous, key=lambda x: x[2], reverse=True):
            expo = sl_n / 100 * leverage * 100
            old_s = f"{sl_o:.1f}%" if sl_o is not None else "N/A"
            verdict = "🚨 liq. risk" if expo > 100 else ("⚠️  high risk" if expo > 80 else "ok")
            lines.append(f"  {asset:<18} {old_s:>10} {sl_n:.1f}%{'':<4} {expo:.0f}%{'':<3}  {verdict}")

    lines.append("")

    result = {
        "avg_sl_old": avg_old,
        "avg_sl_new": avg_new,
        "delta_avg_sl": delta_avg,
        "n_assets": n,
        "dangerous_assets": [(a, so, sn) for a, so, sn in dangerous],
        "worst_dd_avg_pct": worst_dd_avg,
        "worst_dd_max_pct": worst_dd_max,
        "margin_vs_ks_avg": margin_avg,
        "margin_vs_ks_max": margin_max,
    }
    return result, lines


# ─── Section 4 : Deep dive assets perdants ──────────────────────────────────


def section4_losers(
    by_asset: dict[str, list[dict]],
    loser_assets: list[str],
    per_asset_pnl: dict | None,
) -> tuple[list[dict], list[str]]:
    lines: list[str] = []
    records: list[dict] = []

    lines.append(_hdr("SECTION 4 — Deep Dive : Assets Perdants"))

    if not loser_assets:
        lines.append("  Aucun asset perdant spécifié (utiliser --losers)")
        lines.append("")
        return records, lines

    for asset in loser_assets:
        lines.append(f"\n  ┌── {asset} " + "─" * max(0, 78 - len(asset)))
        runs = by_asset.get(asset, [])
        if not runs:
            lines.append("  │  ⚠️  Asset absent de la DB WFO")
            lines.append("  └" + "─" * 80)
            continue

        r_new = runs[0]
        r_old = runs[1] if len(runs) >= 2 else None
        p_new = _params(r_new)
        p_old = _params(r_old) if r_old else {}

        grade_new = r_new.get("grade", "?")
        grade_old = r_old.get("grade", "?") if r_old else "?"
        sh_new = r_new.get("oos_sharpe")
        sh_old = r_old.get("oos_sharpe") if r_old else None

        lines.append(f"  │  Grade     : {grade_old} → {grade_new}")
        lines.append(f"  │  Score     : {r_old.get('total_score', '?') if r_old else '?'} → {r_new.get('total_score', '?')}")
        lines.append(f"  │  Sharpe OOS: {_fv(sh_old)} → {_fv(sh_new)}")
        lines.append("  │")

        # Params changés
        changed_params = []
        for param in NUMERIC_PARAMS:
            v_old = p_old.get(param)
            v_new = p_new.get(param)
            if v_old != v_new and (v_old is not None or v_new is not None):
                direction = ""
                if v_old is not None and v_new is not None:
                    direction = "↑" if v_new > v_old else "↓"
                changed_params.append(f"{param}: {_fv(v_old)} → {_fv(v_new)} {direction}")

        if changed_params:
            lines.append("  │  Paramètres modifiés :")
            for cp in changed_params:
                lines.append(f"  │    {cp}")
        else:
            lines.append("  │  Paramètres : aucun changement")

        # P&L portfolio
        pnl_data: dict | None = None
        if per_asset_pnl:
            for key in [f"grid_atr:{asset}", f"grid_boltrend:{asset}", asset]:
                if key in per_asset_pnl:
                    pnl_data = per_asset_pnl[key]
                    break

        if pnl_data:
            pnl_val = pnl_data.get("net_pnl", pnl_data.get("realized_pnl", pnl_data.get("total_pnl")))
            trades = pnl_data.get("trades", pnl_data.get("n_trades", pnl_data.get("total_trades")))
            wr = pnl_data.get("win_rate")
            max_dd = pnl_data.get("max_dd_pct", pnl_data.get("max_drawdown_pct"))

            lines.append("  │")
            lines.append("  │  Portfolio backtest (dernier run) :")
            lines.append(f"  │    P&L réalisé : {f'{pnl_val:+.2f}' if pnl_val is not None else 'N/A'}")
            lines.append(f"  │    Trades      : {trades if trades is not None else 'N/A'}")
            lines.append(f"  │    Win rate    : {f'{wr:.1f}%' if wr is not None else 'N/A'}")
            lines.append(f"  │    Max DD      : {f'{max_dd:.1f}%' if max_dd is not None else 'N/A'}")

            # Diagnostic
            diags: list[str] = []
            if trades is not None and trades < 20:
                diags.append(f"⚠️  Peu de trades ({trades}) — signal statistiquement faible")
            if wr is not None and wr < 45:
                diags.append(f"⚠️  Win rate faible ({wr:.1f}%) — SL déclenché trop souvent")
            if pnl_val is not None and pnl_val < 0:
                diags.append(f"✗  P&L négatif sur la période")
            sl_n = p_new.get("sl_percent")
            sl_o = p_old.get("sl_percent")
            if sl_n and sl_o and sl_n > sl_o:
                diags.append(f"→  SL élargi {sl_o:.1f}% → {sl_n:.1f}% : moins de sorties prématurées, mais chaque perte coûte plus")
            if not diags:
                diags.append("(pas de problème flagrant détecté)")

            lines.append("  │")
            lines.append("  │  Diagnostic :")
            for d in diags:
                lines.append(f"  │    {d}")

            # Recommandation
            if pnl_val is not None and pnl_val < -200 and (trades is None or trades < 15):
                rec = "RETIRER — P&L très négatif avec peu de trades"
            elif pnl_val is not None and pnl_val < 0 and sl_n is not None and sl_n > 30:
                rec = "AJUSTER — contraindre sl_percent ≤ 25% dans le WFO"
            elif pnl_val is not None and pnl_val < 0:
                rec = "SURVEILLER — garder en paper, relancer WFO dans 3 mois"
            else:
                rec = "GARDER — performance acceptable"

            lines.append("  │")
            lines.append(f"  │  Recommandation : {rec}")

            records.append({
                "asset": asset,
                "grade_old": grade_old,
                "grade_new": grade_new,
                "sl_old": sl_o,
                "sl_new": sl_n,
                "pnl": pnl_val,
                "trades": trades,
                "win_rate": wr,
                "recommendation": rec,
            })
        else:
            lines.append("  │  (pas de données portfolio pour cet asset)")

        lines.append("  └" + "─" * 80)

    lines.append("")
    return records, lines


# ─── Section 5 : Synthèse ────────────────────────────────────────────────────


def section5_synthesis(
    param_records: list[dict],
    regime_deltas: dict[str, list[float]],
    sl_data: dict,
    loser_records: list[dict],
    old_portfolio: dict | None,
    new_portfolio: dict | None,
    leverage: int,
    ks_pct: float,
) -> list[str]:
    lines: list[str] = []
    lines.append(_hdr("SECTION 5 — Synthèse et Recommandations"))

    if old_portfolio and new_portfolio:
        ret_old = old_portfolio.get("total_return_pct") or 0
        ret_new = new_portfolio.get("total_return_pct") or 0
        dd_old = abs(old_portfolio.get("max_drawdown_pct") or 0)
        dd_new = abs(new_portfolio.get("max_drawdown_pct") or 0)
        ratio_old = ret_old / dd_old if dd_old else 0
        ratio_new = ret_new / dd_new if dd_new else 0

        lines.append(f"\n  Régression portfolio (ancien → nouveau) :")
        lines.append(f"  {'Métrique':<28} {'Avant':>12} {'Après':>12} {'Delta':>12}")
        lines.append("  " + _sep(68))
        lines.append(f"  {'Retour total':<28} {_fmt_pct(ret_old):>12} {_fmt_pct(ret_new):>12} {_fmt_pct(ret_new - ret_old):>12}")
        lines.append(f"  {'Max Drawdown':<28} {_fmt_pct(-dd_old):>12} {_fmt_pct(-dd_new):>12} {_fmt_pct(-(dd_new - dd_old)):>12}")
        lines.append(f"  {'Ratio Ret/DD':<28} {ratio_old:>12.2f} {ratio_new:>12.2f} {ratio_new - ratio_old:>+12.2f}")

    sl_delta = sl_data.get("delta_avg_sl", 0)
    bull_avg = mean(regime_deltas["bull"]) if regime_deltas.get("bull") else None
    bear_avg = mean(regime_deltas["bear"]) if regime_deltas.get("bear") else None
    crash_avg = mean(regime_deltas["crash"]) if regime_deltas.get("crash") else None

    lines.append(f"\n  ── Q1 : Nouveaux params meilleurs ou surcompensation ?")
    if sl_delta > 3:
        lines.append(f"  → SL moyen augmenté +{sl_delta:.1f}pp : params plus conservatifs, moins de faux SL")
    elif sl_delta < -3:
        lines.append(f"  → SL moyen réduit {sl_delta:.1f}pp : params plus agressifs")
    else:
        lines.append(f"  → SL moyen quasi inchangé ({sl_delta:+.1f}pp)")

    if bear_avg is not None and bear_avg > 0 and bull_avg is not None and bull_avg < 0:
        lines.append(f"  → Amélioration bear (+{bear_avg:.3f}) au prix d'une dégradation bull ({bull_avg:.3f})")
        lines.append(f"     ⚠️  SURCOMPENSATION probable — optimisation bear/crash au détriment du bull")
    elif bear_avg is not None and bear_avg > 0 and (bull_avg is None or bull_avg >= -0.1):
        lines.append(f"  → Amélioration bear sans pénalité bull significative — nouveaux params globalement meilleurs")
    elif bull_avg is not None and bull_avg < -0.2 and (bear_avg is None or bear_avg <= 0):
        lines.append(f"  → Dégradation bull ({bull_avg:.3f}) sans amélioration bear — régression pure")
    else:
        lines.append(f"  → Impact mixte — analyse par asset recommandée")

    lines.append(f"\n  ── Q2 : Faut-il ajuster le kill switch (actuel {ks_pct:.0f}%) ?")
    margin_avg = sl_data.get("margin_vs_ks_avg", 0)
    margin_max = sl_data.get("margin_vs_ks_max", 0)
    if margin_max < 0:
        lines.append(f"  → 🚨 Marge négative avec SL max — envisager KS {ks_pct + 10:.0f}% ou réduire leverage à {leverage - 1}x")
    elif margin_avg < 10:
        lines.append(f"  → ⚠️  Marge faible ({margin_avg:.1f}pp) — envisager KS {ks_pct + 5:.0f}%")
    else:
        lines.append(f"  → KS {ks_pct:.0f}% reste approprié (marge {margin_avg:.1f}pp)")

    lines.append(f"\n  ── Q3 : Faut-il contraindre sl_percent dans param_grids.yaml ?")
    sl_new_avg = sl_data.get("avg_sl_new", 0)
    dangerous = sl_data.get("dangerous_assets", [])
    if sl_new_avg > 25:
        lines.append(f"  → OUI — SL moyen {sl_new_avg:.1f}% dépasse 25%")
        lines.append(f"     Action : retirer les valeurs 30%+ de sl_percent dans param_grids.yaml grid_atr")
    elif dangerous:
        lines.append(f"  → PARTIEL — {len(dangerous)} assets ont sl > 25%, moyenne OK ({sl_new_avg:.1f}%)")
        lines.append(f"     Option : exclure ces assets ou ajouter une contrainte per_asset sl_percent ≤ 25%")
    else:
        lines.append(f"  → Non nécessaire — SL moyen {sl_new_avg:.1f}% sous 25%")

    lines.append(f"\n  ── Q4 : Faut-il baisser le leverage ({leverage}x → {leverage - 1}x) ?")
    worst_dd = sl_data.get("worst_dd_avg_pct", 0)
    if worst_dd > ks_pct * 0.8:
        lines.append(f"  → OUI — worst-case {worst_dd:.0f}% = {worst_dd / ks_pct * 100:.0f}% du KS")
        lines.append(f"     {leverage - 1}x réduirait le worst-case à {worst_dd * (leverage - 1) / leverage:.0f}%")
    elif worst_dd > ks_pct * 0.6:
        lines.append(f"  → À CONSIDÉRER — worst-case {worst_dd:.0f}% représente {worst_dd / ks_pct * 100:.0f}% du KS")
    else:
        lines.append(f"  → Non nécessaire — worst-case {worst_dd:.0f}% bien sous KS {ks_pct:.0f}%")

    lines.append(f"\n  ── Q5 : Assets à retirer du portfolio ?")
    if loser_records:
        to_remove = [r for r in loser_records if "RETIRER" in r.get("recommendation", "")]
        to_adjust = [r for r in loser_records if "AJUSTER" in r.get("recommendation", "")]
        to_watch = [r for r in loser_records if "SURVEILLER" in r.get("recommendation", "")]
        if to_remove:
            lines.append(f"  → Retrait recommandé :")
            for r in to_remove:
                lines.append(f"     • {r['asset']} (P&L={r.get('pnl', 'N/A')}, trades={r.get('trades', 'N/A')})")
        if to_adjust:
            lines.append(f"  → Ajustement WFO recommandé (sl ≤ 25%) :")
            for r in to_adjust:
                lines.append(f"     • {r['asset']}")
        if to_watch:
            lines.append(f"  → Surveillance (paper) :")
            for r in to_watch:
                lines.append(f"     • {r['asset']}")
        if not to_remove and not to_adjust and not to_watch:
            lines.append("  → Aucun retrait urgent — monitoring habituel")
    else:
        lines.append("  → Aucune analyse disponible (utiliser --losers pour un deep dive)")

    lines.append("")
    return lines


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    # Forcer UTF-8 sur la console Windows (cp1252 ne supporte pas les box-drawing chars)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Analyse de régression WFO — compare les 2 derniers runs d'une stratégie"
    )
    parser.add_argument("--strategy", default="grid_atr")
    parser.add_argument("--db", default=None, help="Chemin DB SQLite (auto-détecté si omis)")
    parser.add_argument("--losers", nargs="+",
                        default=["NEAR/USDT", "DOGE/USDT", "OP/USDT", "AVAX/USDT"],
                        metavar="ASSET",
                        help="Assets perdants pour le deep dive")
    parser.add_argument("--leverage", type=int, default=7)
    parser.add_argument("--kill-switch", type=float, default=45.0, dest="ks_pct")
    args = parser.parse_args()

    db_path = args.db or _db_path_default()
    if not Path(db_path).exists():
        print(f"[ERREUR] DB introuvable : {db_path}")
        sys.exit(1)

    strategy = args.strategy
    print(f"\n  Analyse WFO — {strategy.upper()}")
    print(f"  DB           : {db_path}")
    print(f"  Leverage     : {args.leverage}x  |  Kill switch : {args.ks_pct}%")
    print(f"  Assets perdants : {', '.join(args.losers)}\n")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    by_asset = fetch_wfo_results(conn, strategy)
    if not by_asset:
        print(f"[ERREUR] Aucun résultat WFO pour '{strategy}'")
        sys.exit(1)

    n_total = len(by_asset)
    n_with_prev = sum(1 for runs in by_asset.values() if len(runs) >= 2)
    print(f"  {n_total} assets trouvés, {n_with_prev} avec un run précédent pour comparaison\n")

    new_port = fetch_portfolio(conn, strategy, offset=0)
    old_port = fetch_portfolio(conn, strategy, offset=1)

    for label, port in [("récent", new_port), ("précédent", old_port)]:
        if port:
            print(f"  Portfolio {label}: {port.get('created_at')} — {port.get('label', '(sans label)')}")
            print(f"    Return={_fmt_pct(port.get('total_return_pct'))}, DD={_fmt_pct(-(port.get('max_drawdown_pct') or 0))}\n")
        elif label == "récent":
            print("  ⚠️  Aucun portfolio backtest dans la DB\n")

    per_asset_pnl = new_port["per_asset_results"] if new_port else None
    conn.close()

    # ── Génération du rapport ────────────────────────────────────────────────
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    all_lines: list[str] = [
        "",
        _hdr(f"RAPPORT RÉGRESSION WFO — {strategy.upper()} — {now_str}", 100),
        "",
    ]

    param_records, s1 = section1_param_diff(by_asset, per_asset_pnl)
    regime_deltas, s2 = section2_regime_diff(by_asset)
    sl_data, s3 = section3_sl_risk(by_asset, args.leverage, args.ks_pct)
    loser_records, s4 = section4_losers(by_asset, args.losers, per_asset_pnl)
    s5 = section5_synthesis(param_records, regime_deltas, sl_data, loser_records,
                             old_port, new_port, args.leverage, args.ks_pct)

    all_lines.extend(s1 + s2 + s3 + s4 + s5)
    all_lines.append(_sep(100))
    all_lines.append("")

    report_text = "\n".join(all_lines)
    print(report_text)

    # ── Sauvegarde ───────────────────────────────────────────────────────────
    out_dir = Path("data/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / "wfo_regression_report.txt"
    txt_path.write_text(report_text, encoding="utf-8")

    json_data: dict = {
        "strategy": strategy,
        "generated_at": datetime.now().isoformat(),
        "leverage": args.leverage,
        "kill_switch_pct": args.ks_pct,
        "n_assets": n_total,
        "n_with_previous": n_with_prev,
        "param_diffs": param_records,
        "regime_deltas": {k: [round(float(v), 4) for v in vals] for k, vals in regime_deltas.items()},
        "sl_risk": {
            k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
            for k, v in sl_data.items()
            if k != "dangerous_assets"
        },
        "sl_dangerous_assets": [
            {"asset": a, "sl_old": so, "sl_new": sn}
            for a, so, sn in sl_data.get("dangerous_assets", [])
        ],
        "loser_analysis": loser_records,
        "portfolio_new": (
            {k: new_port.get(k) for k in ("created_at", "label", "total_return_pct", "max_drawdown_pct", "leverage")}
            if new_port else None
        ),
        "portfolio_old": (
            {k: old_port.get(k) for k in ("created_at", "label", "total_return_pct", "max_drawdown_pct")}
            if old_port else None
        ),
    }

    json_path = out_dir / "wfo_regression_report.json"
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[OK] Rapport texte : {txt_path}")
    print(f"[OK] Rapport JSON  : {json_path}")


if __name__ == "__main__":
    main()
