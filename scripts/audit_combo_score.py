#!/usr/bin/env python3
"""
Audit combo_score Phase 1 : Analyse statique des variantes de scoring WFO.

Compare 4 formules de scoring pour identifier le "best combo" par asset.
Lecture seule -- aucune modification de DB.

Usage:
    uv run python -m scripts.audit_combo_score --strategy grid_atr
    uv run python -m scripts.audit_combo_score --strategy grid_atr --db data/scalp_radar.db
    uv run python -m scripts.audit_combo_score --strategy grid_atr > data/analysis/combo_score_audit.txt
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

# Windows : forcer stdout UTF-8 pour eviter UnicodeEncodeError sur cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- Setup path -------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "data" / "analysis"
DEFAULT_DB = ROOT / "data" / "scalp_radar.db"

# ---- Formules de scoring ----------------------------------------------------


def score_v1(
    sharpe_mean: float,
    sharpe_median: float | None,
    sharpe_p25: float | None,
    consistency: float,
    total_trades: int,
) -> float:
    """V1 -- Status quo : sharpe_mean x (0.4 + 0.6xcons) x trade_factor"""
    s = max(sharpe_mean, 0.0)
    tf = min(1.0, total_trades / 100)
    return s * (0.4 + 0.6 * consistency) * tf


def score_v2(
    sharpe_mean: float,
    sharpe_median: float | None,
    sharpe_p25: float | None,
    consistency: float,
    total_trades: int,
) -> float | None:
    """V2 -- Sharpe median (resistant aux outliers crash)"""
    if sharpe_median is None:
        return None
    s = max(sharpe_median, 0.0)
    tf = min(1.0, total_trades / 100)
    return s * (0.4 + 0.6 * consistency) * tf


def score_v3(
    sharpe_mean: float,
    sharpe_median: float | None,
    sharpe_p25: float | None,
    consistency: float,
    total_trades: int,
) -> float:
    """V3 -- Consistency reduite (Sharpe domine plus) : sharpe_mean x (0.7 + 0.3xcons) x trade_factor"""
    s = max(sharpe_mean, 0.0)
    tf = min(1.0, total_trades / 100)
    return s * (0.7 + 0.3 * consistency) * tf


def score_v4(
    sharpe_mean: float,
    sharpe_median: float | None,
    sharpe_p25: float | None,
    consistency: float,
    total_trades: int,
) -> float | None:
    """V4 -- Blend mean + P25 : (0.7xmean + 0.3xp25) x (0.4 + 0.6xcons) x trade_factor"""
    if sharpe_p25 is None:
        return None
    blended = 0.7 * sharpe_mean + 0.3 * sharpe_p25
    s = max(blended, 0.0)
    tf = min(1.0, total_trades / 100)
    return s * (0.4 + 0.6 * consistency) * tf


SCORE_FUNCS = [score_v1, score_v2, score_v3, score_v4]
VARIANT_NAMES = ["V1 (base)", "V2 (median)", "V3 (lcons)", "V4 (blend)"]
VARIANT_SHORT = ["V1", "V2", "V3", "V4"]


# ---- Acces DB ---------------------------------------------------------------


def load_results(db_path: str, strategy: str) -> list[dict]:
    """Charge les optimization_results is_latest=1 pour la strategie."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, asset, best_params, oos_sharpe, consistency, wfo_windows, n_windows
            FROM optimization_results
            WHERE strategy_name = ? AND is_latest = 1
            ORDER BY asset
            """,
            (strategy,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def load_combo_results(db_path: str, result_id: int) -> list[dict]:
    """Charge tous les combo_results pour un optimization_result."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT params, oos_sharpe, oos_return_pct, oos_trades,
                   consistency, oos_is_ratio, is_best, n_windows_evaluated,
                   is_sharpe, is_return_pct, is_trades
            FROM wfo_combo_results
            WHERE optimization_result_id = ?
            ORDER BY oos_sharpe DESC
            """,
            (result_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---- Extraction sharpe median/P25 depuis wfo_windows ------------------------


def try_extract_per_window_sharpes(
    wfo_windows_json: str | None,
    params: dict,
) -> tuple[float | None, float | None]:
    """
    Tente d'extraire sharpe_median et sharpe_p25 pour un combo depuis wfo_windows.

    wfo_windows contient les resultats par fenetre avec le best_params de cette
    fenetre -- PAS les sharpes de chaque combo pour chaque fenetre.
    Cette fonction retourne donc (None, None) dans la quasi-totalite des cas.

    Returns: (sharpe_median, sharpe_p25)
    """
    if not wfo_windows_json:
        return None, None

    try:
        windows = json.loads(wfo_windows_json)
    except (json.JSONDecodeError, TypeError):
        return None, None

    if not isinstance(windows, list) or not windows:
        return None, None

    params_key = json.dumps(params, sort_keys=True)
    sharpes: list[float] = []

    for w in windows:
        if not isinstance(w, dict):
            continue
        # Chercher combo_results par fenetre (format etendu non standard)
        for field in ("combo_results", "combos", "per_combo"):
            combo_list = w.get(field) or []
            for cr in combo_list:
                if not isinstance(cr, dict):
                    continue
                cr_params = cr.get("params", {})
                cr_key = json.dumps(cr_params, sort_keys=True)
                if cr_key == params_key:
                    oos_s = cr.get("oos_sharpe")
                    if oos_s is not None:
                        sharpes.append(float(oos_s))
                    break

    if len(sharpes) < 2:
        return None, None

    sharpes.sort()
    n = len(sharpes)

    # Median
    if n % 2 == 0:
        median = (sharpes[n // 2 - 1] + sharpes[n // 2]) / 2.0
    else:
        median = sharpes[n // 2]

    # Percentile 25
    p25_idx = max(0, int(math.floor(0.25 * n)) - 1)
    p25 = sharpes[p25_idx]

    return median, p25


# ---- Selection best combo par variante --------------------------------------


def select_best_combo(combos: list[dict], score_fn) -> tuple[dict | None, float | None]:
    """Retourne (best_combo, best_score) pour la variante donnee."""
    best_combo = None
    best_score: float | None = None

    for c in combos:
        s = score_fn(
            sharpe_mean=c["oos_sharpe"] or 0.0,
            sharpe_median=c.get("sharpe_median"),
            sharpe_p25=c.get("sharpe_p25"),
            consistency=c["consistency"] or 0.0,
            total_trades=c["oos_trades"] or 0,
        )
        if s is None:
            continue
        if best_score is None or s > best_score:
            best_score = s
            best_combo = c

    return best_combo, best_score


# ---- Helpers affichage ------------------------------------------------------


def parse_params(combo: dict | None) -> dict:
    if combo is None:
        return {}
    p = combo.get("params", {})
    if isinstance(p, str):
        try:
            return json.loads(p)
        except Exception:
            return {}
    return p or {}


def extract_sl(combo: dict | None) -> str:
    p = parse_params(combo)
    sl = p.get("sl_percent", p.get("sl_pct", p.get("stop_loss_pct", "?")))
    return str(sl) if sl is not None else "?"


def params_equal(a: dict | None, b: dict | None) -> bool:
    if a is None or b is None:
        return False
    pa = parse_params(a)
    pb = parse_params(b)
    return json.dumps(pa, sort_keys=True) == json.dumps(pb, sort_keys=True)


# ---- Rapport ----------------------------------------------------------------


def run_audit(db_path: str, strategy: str) -> dict:
    """Lance l'audit complet pour la strategie."""

    SEP1 = "=" * 72
    SEP2 = "-" * 72

    print(f"\n{SEP1}")
    print(f"  AUDIT COMBO_SCORE -- {strategy.upper()}")
    print(f"  DB : {db_path}")
    print(f"{SEP1}\n")

    results = load_results(db_path, strategy)
    if not results:
        print(f"[ERREUR] Aucun resultat is_latest=1 pour '{strategy}' dans la DB.")
        print("  Verifie que le WFO a ete lance et que strategy_name correspond exactement.")
        return {}

    print(f"Assets trouves : {len(results)}\n")

    # ---- Collecte des donnees -----------------------------------------------
    asset_data: dict[str, dict] = {}
    v2_available = False
    v4_available = False
    n_with_combos = 0

    for result in results:
        asset = result["asset"]
        result_id = result["id"]
        wfo_windows_json = result.get("wfo_windows")

        combos = load_combo_results(db_path, result_id)
        if not combos:
            print(f"  [{asset}] Aucun combo_result en DB -- skip (relancer WFO ?)")
            continue
        n_with_combos += 1

        # Tenter extraction sharpes par fenetre (pour V2/V4)
        for combo in combos:
            params = parse_params(combo)
            median, p25 = try_extract_per_window_sharpes(wfo_windows_json, params)
            combo["sharpe_median"] = median
            combo["sharpe_p25"] = p25
            if median is not None:
                v2_available = True
            if p25 is not None:
                v4_available = True

        # Selection best combo par variante
        best_by_variant: list[tuple[dict | None, float | None]] = [
            select_best_combo(combos, fn) for fn in SCORE_FUNCS
        ]

        asset_data[asset] = {
            "result_id": result_id,
            "n_combos": len(combos),
            "n_windows": result.get("n_windows", 0),
            "best_by_variant": best_by_variant,  # list of (combo, score)
            "all_combos": combos,
            "current_best_params": result.get("best_params"),
        }

    if not asset_data:
        print("[ERREUR] wfo_combo_results est vide pour cette strategie.")
        print("         La table necessite un run WFO avec collect_combo_results=True.")
        print("         Relance le WFO pour peupler la table.")
        return {}

    # ---- Avertissement V2/V4 ------------------------------------------------
    if not v2_available:
        print("  [NOTE] V2 (sharpe median) et V4 (blend P25) : non calculables.")
        print("         wfo_combo_results stocke uniquement les agregats (mean, consistency).")
        print("         Les sharpes individuels par fenetre par combo ne sont pas persistes.")
        print("         -> Seules V1 et V3 sont comparees dans ce rapport.\n")

    # =========================================================================
    # SECTION 1 -- Best combo par variante par asset
    # =========================================================================
    print("SECTION 1 -- Best combo par variante par asset")
    print(SEP2)

    # Colonnes calculables
    calculable = [True, v2_available, True, v4_available]
    active_variants = [i for i, ok in enumerate(calculable) if ok]
    active_names = [VARIANT_SHORT[i] for i in active_variants]

    # En-tete
    col_asset = 18
    hdr = f"{'Asset':<{col_asset}}"
    for vname in active_names:
        hdr += f"  {vname:<26}"
    print(hdr)
    sub = " " * col_asset
    for _ in active_names:
        sub += f"  {'SL':>5} {'Sharpe':>6} {'Cons':>5} {'Trd':>4} "
    print(sub)
    print(SEP2)

    changes_by_variant: dict[int, list[str]] = {i: [] for i in active_variants}

    for asset in sorted(asset_data.keys()):
        data = asset_data[asset]
        bv = data["best_by_variant"]
        v1_combo = bv[0][0]

        line = f"{asset:<{col_asset}}"

        for vi in active_variants:
            combo, score = bv[vi]
            if combo is None:
                line += f"  {'-- N/A --':<26}"
                continue

            sl = extract_sl(combo)
            sharpe = combo.get("oos_sharpe") or 0.0
            cons = (combo.get("consistency") or 0.0) * 100
            trades = combo.get("oos_trades") or 0

            # Marqueur si different de V1
            changed = vi != 0 and not params_equal(combo, v1_combo)
            marker = "<" if changed else " "

            if changed:
                changes_by_variant[vi].append(asset)

            cell = f"  {sl:>5} {sharpe:>6.2f} {cons:>4.0f}% {trades:>4}{marker}"
            line += cell

        print(line)

    print()
    print("  Legende : SL=stop_loss_pct  Sharpe=oos_sharpe  Cons=consistency  Trd=oos_trades")
    print("  < = combo different du V1 (baseline)")
    print()

    if not v2_available:
        print("  V2 et V4 : non calculables (voir note ci-dessus).\n")

    # =========================================================================
    # SECTION 2 -- Synthese
    # =========================================================================
    print("SECTION 2 -- Synthese des changements")
    print(SEP2)
    print(f"{'Variante':<20} {'Assets changes':>15}  {'SL moy sel.':>12}  {'Sharpe moy sel.':>16}")
    print(SEP2)

    summary_data = []
    for vi in active_variants:
        bv_combos = [asset_data[a]["best_by_variant"][vi][0] for a in sorted(asset_data.keys())]
        bv_combos_valid = [c for c in bv_combos if c is not None]

        sl_vals = []
        sharpe_vals = []
        for c in bv_combos_valid:
            p = parse_params(c)
            sl_raw = p.get("sl_percent", p.get("sl_pct", p.get("stop_loss_pct")))
            try:
                sl_vals.append(float(sl_raw))
            except (TypeError, ValueError):
                pass
            sharpe_vals.append(c.get("oos_sharpe") or 0.0)

        avg_sl = sum(sl_vals) / len(sl_vals) if sl_vals else float("nan")
        avg_sh = sum(sharpe_vals) / len(sharpe_vals) if sharpe_vals else float("nan")

        n_total = len(asset_data)
        changed = len(changes_by_variant[vi]) if vi != 0 else 0
        changed_str = "-- (reference)" if vi == 0 else f"{changed} / {n_total}"
        na_note = " [N/A]" if not calculable[vi] else ""

        vname = VARIANT_NAMES[vi] + na_note
        sl_str = "N/A" if math.isnan(avg_sl) else f"{avg_sl:.1f}"
        sh_str = "N/A" if math.isnan(avg_sh) else f"{avg_sh:.2f}"
        print(f"{vname:<20} {changed_str:>15}  {sl_str:>12}  {sh_str:>16}")

        summary_data.append(
            {
                "variant": VARIANT_NAMES[vi],
                "calculable": calculable[vi],
                "assets_changed": changed,
                "total_assets": n_total,
                "avg_sl": None if math.isnan(avg_sl) else round(avg_sl, 2),
                "avg_oos_sharpe": None if math.isnan(avg_sh) else round(avg_sh, 4),
            }
        )

    print()

    # =========================================================================
    # SECTION 3 -- Analyse des combos alternatives
    # =========================================================================
    print("SECTION 3 -- Analyse des combos alternatives (V3 vs V1)")
    print(SEP2)

    v3_idx = 2
    v3_changed: list[str] = changes_by_variant.get(v3_idx, [])

    if not v3_changed:
        print("  Aucun asset ne change de combo entre V1 et V3.\n")
        print("  -> Les combos selectionnees sont STABLES quelle que soit la pondertion")
        print("     de la consistency.\n")
    else:
        print(f"  {len(v3_changed)} asset(s) changent de combo avec V3 :\n")
        for asset in sorted(v3_changed):
            data = asset_data[asset]
            v1_combo = data["best_by_variant"][0][0]
            v3_combo = data["best_by_variant"][v3_idx][0]
            v1_score = data["best_by_variant"][0][1]
            v3_score = data["best_by_variant"][v3_idx][1]

            print(f"  {asset} ({data['n_windows']} fenetres, {data['n_combos']} combos):")

            for label, combo, score in [("V1", v1_combo, v1_score), ("V3", v3_combo, v3_score)]:
                if combo is None:
                    print(f"    {label}: N/A")
                    continue
                p = parse_params(combo)
                sl = p.get("sl_percent", p.get("sl_pct", "?"))
                sh = combo.get("oos_sharpe") or 0.0
                cn = (combo.get("consistency") or 0.0) * 100
                tr = combo.get("oos_trades") or 0
                s_str = f"{score:.4f}" if score is not None else "?"
                param_summary = ", ".join(
                    f"{k}={v}"
                    for k, v in sorted(p.items())
                    if k not in ("sl_percent", "sl_pct", "stop_loss_pct")
                )
                print(
                    f"    {label}: sl={sl}  sharpe={sh:.2f}  cons={cn:.0f}%  "
                    f"trades={tr}  score={s_str}"
                )
                print(f"        params: {param_summary}")

            # Interpretation
            if v1_combo and v3_combo:
                v1_sl = parse_params(v1_combo).get("sl_percent", parse_params(v1_combo).get("sl_pct"))
                v3_sl = parse_params(v3_combo).get("sl_percent", parse_params(v3_combo).get("sl_pct"))
                v1_sh = v1_combo.get("oos_sharpe") or 0.0
                v3_sh = v3_combo.get("oos_sharpe") or 0.0
                v1_cn = (v1_combo.get("consistency") or 0.0) * 100
                v3_cn = (v3_combo.get("consistency") or 0.0) * 100
                try:
                    sl_change = float(v3_sl) - float(v1_sl)
                    direction = "plus serre" if sl_change < 0 else "plus large"
                    print(
                        f"    -> V3 selectione SL {direction} "
                        f"({v3_sl} vs {v1_sl}, delta={sl_change:+.1f})"
                    )
                except (TypeError, ValueError):
                    pass
                sh_diff = v3_sh - v1_sh
                cn_diff = v3_cn - v1_cn
                if abs(sh_diff) >= 0.05:
                    print(f"    -> Sharpe : {v1_sh:.2f} -> {v3_sh:.2f} ({sh_diff:+.2f})")
                if abs(cn_diff) >= 1:
                    print(f"    -> Consistency : {v1_cn:.0f}% -> {v3_cn:.0f}% ({cn_diff:+.0f}pp)")
            print()

    # =========================================================================
    # SECTION 4 -- Preparation params alternatifs
    # =========================================================================
    print("SECTION 4 -- Preparation params alternatifs (si >=5 assets changent avec V3)")
    print(SEP2)

    if len(v3_changed) >= 5:
        alt_params: dict[str, dict] = {}
        for asset in sorted(v3_changed):
            v3_combo = asset_data[asset]["best_by_variant"][v3_idx][0]
            if v3_combo:
                alt_params[asset] = parse_params(v3_combo)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        alt_path = OUTPUT_DIR / f"combo_alt_params_{strategy}_v3.json"
        with open(alt_path, "w", encoding="utf-8") as f:
            json.dump(alt_params, f, indent=2)

        print(f"  {len(v3_changed)} assets changent -> params alternatifs V3 sauvegardes :")
        print(f"  {alt_path}\n")
        print(f"  Pour valider l'impact portfolio, lancer separement :")
        print(f"  uv run python -m scripts.portfolio_backtest --strategy {strategy}")
        print(f"  (puis comparer avec les params override du fichier ci-dessus)\n")
    else:
        print(
            f"  Seulement {len(v3_changed)} asset(s) changent"
            f" -> simulation portfolio non necessaire.\n"
        )

    # =========================================================================
    # SECTION 5 -- Conclusion
    # =========================================================================
    print("SECTION 5 -- Conclusion")
    print(SEP2)

    n_total = len(asset_data)
    n_changed = len(v3_changed)
    frac = n_changed / n_total if n_total > 0 else 0

    # Statistiques SL
    v1_sls = []
    v3_sls = []
    for data in asset_data.values():
        c1 = data["best_by_variant"][0][0]
        c3 = data["best_by_variant"][v3_idx][0]
        for combo, lst in [(c1, v1_sls), (c3, v3_sls)]:
            p = parse_params(combo)
            sl_raw = p.get("sl_percent", p.get("sl_pct", p.get("stop_loss_pct")))
            try:
                lst.append(float(sl_raw))
            except (TypeError, ValueError):
                pass

    avg_sl_v1 = sum(v1_sls) / len(v1_sls) if v1_sls else float("nan")
    avg_sl_v3 = sum(v3_sls) / len(v3_sls) if v3_sls else float("nan")
    sl_diff = (
        avg_sl_v3 - avg_sl_v1
        if not (math.isnan(avg_sl_v1) or math.isnan(avg_sl_v3))
        else 0
    )

    print(f"  Assets analyses      : {n_total}")
    print(f"  Assets avec combos   : {n_with_combos}")
    print(f"  Assets changent V3   : {n_changed} / {n_total} ({frac*100:.0f}%)")
    if not math.isnan(avg_sl_v1):
        print(f"  SL moyen V1 (base)   : {avg_sl_v1:.1f}")
    if not math.isnan(avg_sl_v3):
        print(f"  SL moyen V3          : {avg_sl_v3:.1f} ({sl_diff:+.1f}pp vs V1)")
    print()

    if n_changed == 0:
        print("  +-- CONCLUSION : combo_score N'EST PAS responsable de la regression --+")
        print("  |                                                                       |")
        print("  |  Les combos selectionnees sont IDENTIQUES quelle que soit la         |")
        print("  |  ponderation de la consistency (V1 = V3 sur tous les assets).        |")
        print("  |                                                                       |")
        print("  |  Le probleme est en AMONT -- dans les donnees WFO elles-memes :      |")
        print("  |  -> Le fix regimes (Hotfix 37c) a modifie le contenu des fenetres    |")
        print("  |  -> Les combos conservatrices (SL large) dominent mecaniquement      |")
        print("  |     car plus resilientes aux crashes inclus dans les fenetres OOS     |")
        print("  |                                                                       |")
        print("  |  Leviers a explorer pour recuperer le rendement :                    |")
        print("  |  1. Exclure les fenetres crash du scoring (pas de la selection)      |")
        print("  |  2. Parametrer is_days/oos_days pour eviter les fenetres crash       |")
        print("  |  3. Relancer WFO avec une grille de SL plus fine (ex: 15-20%)        |")
        print("  +-----------------------------------------------------------------------+")
    elif frac <= 0.33:
        print(f"  +-- CONCLUSION : combo_score est un levier PARTIEL --+")
        print(f"  |")
        print(f"  |  {n_changed}/{n_total} assets changent de combo avec V3 (minorite).")
        print(f"  |  La plupart des combos sont stables -> les donnees WFO dominent.")
        print(f"  |  Mais {n_changed} asset(s) pourraient beneficier de la formule V3.")
        print(f"  |  -> Recommandation : tester V3 en portfolio backtest avant deploiement")
        print(f"  +----------------------------------------------------+")
    else:
        print(f"  +-- CONCLUSION : combo_score EST un levier significatif --+")
        print(f"  |")
        print(f"  |  {n_changed}/{n_total} assets changent de combo avec V3 ({frac*100:.0f}% des assets).")
        print(f"  |  La formule (0.7 + 0.3xcons) vs (0.4 + 0.6xcons) change le choix")
        print(f"  |  pour la majorite des assets.")
        print(f"  |  -> Valider V3 en portfolio backtest complet avant de modifier")
        print(f"  |     le code de combo_score dans walk_forward.py")
        print(f"  +----------------------------------------------------------+")

    print()

    # ---- Sauvegarde JSON ----------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_output: dict[str, Any] = {
        "strategy": strategy,
        "n_assets_found": len(results),
        "n_assets_with_combos": n_with_combos,
        "v2_available": v2_available,
        "v4_available": v4_available,
        "summary": summary_data,
        "assets": {},
    }

    for asset in sorted(asset_data.keys()):
        data = asset_data[asset]
        asset_entry: dict[str, Any] = {
            "n_combos": data["n_combos"],
            "n_windows": data["n_windows"],
            "best_by_variant": [],
        }
        v1_combo = data["best_by_variant"][0][0]
        for vi in range(len(SCORE_FUNCS)):
            combo, score = data["best_by_variant"][vi]
            if combo is None:
                asset_entry["best_by_variant"].append({"variant": VARIANT_NAMES[vi], "na": True})
                continue
            p = parse_params(combo)
            changed = vi != 0 and not params_equal(combo, v1_combo)
            asset_entry["best_by_variant"].append(
                {
                    "variant": VARIANT_NAMES[vi],
                    "calculable": calculable[vi],
                    "params": p,
                    "oos_sharpe": combo.get("oos_sharpe"),
                    "consistency": combo.get("consistency"),
                    "oos_trades": combo.get("oos_trades"),
                    "score": round(score, 6) if score is not None else None,
                    "changed_from_v1": changed,
                }
            )
        json_output["assets"][asset] = asset_entry

    json_path = OUTPUT_DIR / f"combo_score_audit_{strategy}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    txt_note = OUTPUT_DIR / "combo_score_audit.txt"
    print(f"Rapport JSON : {json_path}")
    print(f"Rapport texte : lancer avec redirection :")
    print(f"  uv run python -m scripts.audit_combo_score --strategy {strategy} > {txt_note}\n")

    return json_output


# ---- Entree principale ------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit combo_score : compare 4 variantes de scoring WFO (lecture seule).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  uv run python -m scripts.audit_combo_score --strategy grid_atr
  uv run python -m scripts.audit_combo_score --strategy grid_boltrend --db data/scalp_radar.db
  uv run python -m scripts.audit_combo_score --strategy grid_atr > data/analysis/combo_score_audit.txt
        """,
    )
    parser.add_argument(
        "--strategy",
        required=True,
        help="Nom de la strategie (ex: grid_atr, grid_boltrend)",
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help=f"Chemin vers la DB SQLite (defaut: {DEFAULT_DB})",
    )
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"[ERREUR] DB non trouvee : {args.db}")
        print("  Verifie le chemin ou utilise --db pour specifier l'emplacement.")
        sys.exit(1)

    run_audit(args.db, args.strategy)


if __name__ == "__main__":
    main()
