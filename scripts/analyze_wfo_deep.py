"""Post-WFO Deep Analysis — Vérifie la viabilité réelle des assets Grade A/B
après un WFO. Identifie les red flags (SL critique, régimes catastrophiques,
overfitting) et produit un verdict enrichi : VIABLE / BORDERLINE / ELIMINATED.

Usage:
    uv run python -m scripts.analyze_wfo_deep --strategy grid_boltrend
    uv run python -m scripts.analyze_wfo_deep --all
"""

import sys
import io
import json
import argparse
import sqlite3
from pathlib import Path

import yaml

# Force UTF-8 sur Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DB_PATH = Path(__file__).parent.parent / "data" / "scalp_radar.db"
STRATEGIES_YAML = Path(__file__).parent.parent / "config" / "strategies.yaml"

# ── Seuils ────────────────────────────────────────────────────────────────────
SL_CRITICAL = 1.00  # SL × leverage > 100% de la marge → CRITICAL
SL_WARNING  = 0.80  # SL × leverage > 80%  de la marge → WARNING
SHARPE_SEVERE   = -5.0  # Sharpe dans un régime < -5 → SEVERE WEAKNESS
TRADES_MIN      = 10    # Trades Bitget minimum pour CI fiable
OIS_SUSPECT     = 5.0   # OOS/IS > 5 → probablement période exceptionnelle

REGIME_LABELS = {"bull": "BULL", "bear": "BEAR", "range": "RANGE", "crash": "CRASH"}

VERDICT_ICONS = {"VIABLE": "[OK]", "BORDERLINE": "[~~]", "ELIMINATED": "[XX]"}
SEV_ICONS     = {"critical": "[!!]", "warning": "[!] ", "info": "[i] ", "ok": "[ ] "}


# ── DB / Config ───────────────────────────────────────────────────────────────

def load_strategies_yaml() -> dict:
    with open(STRATEGIES_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_leverage(strategies_cfg: dict, strategy_name: str) -> int:
    return int(strategies_cfg.get(strategy_name, {}).get("leverage", 1))


def get_all_strategies(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT strategy_name FROM optimization_results WHERE is_latest=1"
    ).fetchall()
    return [r[0] for r in rows]


def load_results(conn: sqlite3.Connection, strategy: str | None) -> list[dict]:
    cur = conn.cursor()
    if strategy:
        cur.execute(
            "SELECT * FROM optimization_results "
            "WHERE strategy_name=? AND is_latest=1 AND grade IN ('A','B') "
            "ORDER BY total_score DESC",
            (strategy,),
        )
    else:
        cur.execute(
            "SELECT * FROM optimization_results "
            "WHERE is_latest=1 AND grade IN ('A','B') "
            "ORDER BY strategy_name, total_score DESC",
        )
    rows = cur.fetchall()
    col_names = [d[0] for d in cur.description]
    return [dict(zip(col_names, row)) for row in rows]


# ── Analyse par asset ─────────────────────────────────────────────────────────

def analyze_asset(row: dict, leverage: int) -> dict:
    """
    Retourne un dict avec :
      flags        : list[(severity, message)]  severity = critical|warning|info|ok
      verdict      : VIABLE | BORDERLINE | ELIMINATED
      regime_detail: list[dict]
      ci_status    : CONFIRMED | UNCONFIRMED | NEGATIVE | UNKNOWN
      transfer_significant: bool
      sl_ratio     : float (sl_pct * leverage / 100)
    """
    flags = []

    # ── best_params ──────────────────────────────────────────────────────────
    best_params = {}
    if row.get("best_params"):
        try:
            best_params = json.loads(row["best_params"])
        except (json.JSONDecodeError, TypeError):
            pass

    sl_pct   = float(best_params.get("sl_percent", 0))
    sl_ratio = sl_pct * leverage / 100

    # 1. SL × leverage check
    if sl_ratio > SL_CRITICAL:
        flags.append(("critical", f"SL {sl_pct:.0f}%×{leverage}x={sl_ratio:.0%} — depasse 100% marge"))
    elif sl_ratio > SL_WARNING:
        flags.append(("warning",  f"SL {sl_pct:.0f}%×{leverage}x={sl_ratio:.0%} — proche limite 80%"))
    else:
        flags.append(("ok",       f"SL {sl_pct:.0f}%×{leverage}x={sl_ratio:.0%}"))

    # 2. DSR
    dsr = float(row.get("dsr") or 0.0)
    if dsr < 0.01:
        flags.append(("warning", f"DSR={dsr:.4f} — risque data mining"))

    # 3. OOS/IS ratio
    ois = float(row.get("oos_is_ratio") or 0.0)
    if ois > OIS_SUSPECT:
        flags.append(("warning", f"OOS/IS={ois:.2f} — periodes OOS exceptionnelles?"))

    # 4. Régimes
    regime_detail  = []
    range_sharpe   = None
    dominant_key   = None
    dominant_n     = 0
    regime_analysis = None

    if row.get("regime_analysis"):
        try:
            regime_analysis = json.loads(row["regime_analysis"])
        except (json.JSONDecodeError, TypeError):
            regime_analysis = None

    if regime_analysis:
        for rkey in ["bull", "bear", "range", "crash"]:
            rd = regime_analysis.get(rkey, {})
            if not rd:
                continue
            n         = int(rd.get("n_windows", 0))
            sharpe    = float(rd.get("avg_oos_sharpe", 0.0))
            consist   = float(rd.get("consistency", 0.0))
            ret       = float(rd.get("avg_return_pct", 0.0))

            regime_detail.append({
                "regime": REGIME_LABELS.get(rkey, rkey.upper()),
                "sharpe": sharpe, "consistency": consist,
                "return_pct": ret, "n_windows": n,
            })

            if n > dominant_n:
                dominant_n   = n
                dominant_key = rkey

            if rkey == "range":
                range_sharpe = sharpe

            # Sharpe < -5 dans n'importe quel régime
            if sharpe < SHARPE_SEVERE:
                flags.append((
                    "critical",
                    f"Sharpe {sharpe:.2f} en {rkey.upper()} — perte systemique",
                ))

        # Sharpe négatif en RANGE → perd la plupart du temps
        if range_sharpe is not None and range_sharpe < 0:
            flags.append((
                "critical",
                f"Sharpe RANGE={range_sharpe:.2f} — perd en marche lateral",
            ))

        # Sharpe négatif dans le régime dominant (autre que RANGE, déjà couvert)
        if dominant_key and dominant_key != "range":
            dom_sharpe = float(regime_analysis[dominant_key].get("avg_oos_sharpe", 0.0))
            if dom_sharpe < 0:
                flags.append((
                    "critical",
                    f"Sharpe {dom_sharpe:.2f} en regime dominant {dominant_key.upper()} ({dominant_n} fen.)",
                ))

    # 5. Validation Bitget
    ci_status          = "UNKNOWN"
    transfer_significant = False
    val                = None

    if row.get("validation_summary"):
        try:
            val = json.loads(row["validation_summary"])
        except (json.JSONDecodeError, TypeError):
            val = None

    if val:
        bitget_trades = int(val.get("bitget_trades", 0) or 0)
        ci_low        = val.get("bitget_sharpe_ci_low")
        ci_high       = val.get("bitget_sharpe_ci_high")
        transfer_significant = bool(val.get("transfer_significant", False))

        if bitget_trades < TRADES_MIN:
            flags.append(("warning", f"{bitget_trades} trades Bitget < {TRADES_MIN} — faible confiance"))

        if ci_low is not None and ci_high is not None:
            # CI non calculable si les deux sont exactement 0
            ci_computable = not (ci_low == 0.0 and ci_high == 0.0)
            if ci_computable:
                if ci_low > 0:
                    ci_status = "CONFIRMED"
                    flags.append(("ok", f"CI95=[+{ci_low:.3f}/+{ci_high:.3f}] confirme"))
                elif ci_high > 0:
                    ci_status = "UNCONFIRMED"
                    flags.append(("info", f"CI95=[{ci_low:.3f}/+{ci_high:.3f}] chevauche zero"))
                else:
                    ci_status = "NEGATIVE"
                    flags.append(("critical", f"CI95=[{ci_low:.3f}/{ci_high:.3f}] entierement negatif"))

    # ── Verdict ──────────────────────────────────────────────────────────────
    has_critical = any(s == "critical" for s, _ in flags)
    has_warning  = any(s == "warning"  for s, _ in flags)

    if has_critical:
        verdict = "ELIMINATED"
    elif has_warning:
        verdict = "BORDERLINE"
    else:
        verdict = "VIABLE"

    return {
        "flags":                flags,
        "verdict":              verdict,
        "regime_detail":        regime_detail,
        "ci_status":            ci_status,
        "transfer_significant": transfer_significant,
        "sl_ratio":             sl_ratio,
        "sl_pct":               sl_pct,
        "best_params":          best_params,
        "val":                  val,
    }


# ── Affichage ─────────────────────────────────────────────────────────────────

def _regime_icon(sharpe: float) -> str:
    if sharpe >= 1.0:  return "[**]"
    if sharpe >= 0.0:  return "[  ]"
    return "[!!]"


def print_summary_table(
    results:  list[dict],
    analyses: list[dict],
    strategy: str | None,
) -> tuple[list[str], list[str], list[str]]:

    SEP = "=" * 80
    sep = "-" * 80

    print(f"\n{SEP}")
    if strategy:
        print(f"  {strategy.upper()} — Deep Analysis Post-WFO")
    else:
        print(f"  Deep Analysis Post-WFO — Toutes strategies")
    print(f"{SEP}\n")

    print(f"  {'Asset':<16} {'Grd':>4} {'Scr':>4}  {'Verdict':<16}  Red Flags (resume)")
    print(f"  {sep}")

    viable_list:     list[str] = []
    borderline_list: list[str] = []
    eliminated_list: list[str] = []

    for row, ana in zip(results, analyses):
        asset   = row["asset"]
        grade   = row["grade"]
        score   = int(row.get("total_score") or 0)
        verdict = ana["verdict"]
        icon    = VERDICT_ICONS[verdict]

        # Flags résumé : critical + warning seulement, forme courte
        parts = []
        for sev, msg in ana["flags"]:
            if sev in ("critical", "warning"):
                short = msg.split(" — ")[0][:35]
                parts.append(short)
        flag_str = " | ".join(parts[:2])

        print(f"  {asset:<16} {grade:>4} {score:>4}  {icon} {verdict:<11}  {flag_str}")

        if verdict == "VIABLE":
            viable_list.append(asset)
        elif verdict == "BORDERLINE":
            borderline_list.append(asset)
        else:
            eliminated_list.append(asset)

    total    = len(results)
    n_viable = len(viable_list)
    n_border = len(borderline_list)
    n_elim   = len(eliminated_list)

    print(f"\n  Summary: {total} Grade A/B  ->  {n_viable} VIABLE, {n_border} BORDERLINE, {n_elim} ELIMINATED")

    effective = viable_list + borderline_list
    if effective:
        print(f"  Effective assets:   {', '.join(effective)}")
    else:
        print(f"  Effective assets:   aucun — strategie non viable (critere : 5 minimum)")

    if eliminated_list:
        print(f"  Eliminated:         {', '.join(eliminated_list)}")

    return viable_list, borderline_list, eliminated_list


def print_asset_detail(row: dict, ana: dict, leverage: int) -> None:
    sep  = "-" * 70
    asset   = row["asset"]
    grade   = row["grade"]
    score   = int(row.get("total_score") or 0)
    verdict = ana["verdict"]
    icon    = VERDICT_ICONS[verdict]

    print(f"\n{sep}")
    print(f"  {asset}   Grade {grade}   Score {score}   {icon} {verdict}")
    print(sep)

    # Paramètres
    bp = ana["best_params"]
    param_parts = [f"{k}={v}" for k, v in bp.items() if k != "timeframe"]
    print(f"  Params:    {('  '.join(param_parts))[:75]}")

    # Risk
    sl_pct   = ana["sl_pct"]
    sl_ratio = ana["sl_ratio"]
    dsr      = float(row.get("dsr") or 0.0)
    ois      = float(row.get("oos_is_ratio") or 0.0)
    consist  = float(row.get("consistency") or 0.0)
    n_win    = int(row.get("n_windows") or 0)

    sl_icon  = "[!!]" if sl_ratio > SL_CRITICAL else ("[!] " if sl_ratio > SL_WARNING else "[ ] ")
    dsr_icon = "[!] " if dsr < 0.01 else "[ ] "
    ois_icon = "[!] " if ois > OIS_SUSPECT else "[ ] "

    print(f"  Risk:      SL {sl_pct:.0f}% x {leverage}x = {sl_ratio:.0%} {sl_icon}  "
          f"DSR {dsr:.4f} {dsr_icon}  OOS/IS {ois:.2f} {ois_icon}  "
          f"Consist {consist:.0%}  {n_win} fen.")

    # Bitget
    val = ana["val"]
    if val:
        bs      = float(val.get("bitget_sharpe") or float("nan"))
        bret    = float(val.get("bitget_net_return_pct") or float("nan"))
        btrades = int(val.get("bitget_trades") or 0)
        ci_low  = val.get("bitget_sharpe_ci_low")
        ci_high = val.get("bitget_sharpe_ci_high")
        tr      = float(val.get("transfer_ratio") or float("nan"))

        ci_computable = ci_low is not None and ci_high is not None and not (ci_low == 0 and ci_high == 0)
        if ci_computable:
            ci_str = f"CI95=[{ci_low:+.3f}/{ci_high:+.3f}]"
        else:
            ci_str = "CI95=n/a"

        tsig = "[OK]" if ana["transfer_significant"] else "[~~]"
        ci_status = ana["ci_status"]

        print(f"  Bitget:    Sharpe {bs:.2f}  Return {bret:+.1f}%  {btrades} trades  "
              f"{ci_str} {ci_status}  Transfer {tr:.3f} {tsig}")
    else:
        print(f"  Bitget:    pas de donnees validation")

    # Régimes
    if ana["regime_detail"]:
        print(f"  Regimes:")
        for rd in sorted(ana["regime_detail"], key=lambda r: -r["n_windows"]):
            ric   = _regime_icon(rd["sharpe"])
            cpct  = rd["consistency"] * 100
            print(
                f"    {rd['regime']:<6}  Sharpe {rd['sharpe']:+6.2f}  "
                f"Consist {cpct:3.0f}%  Return {rd['return_pct']:+7.1f}%  "
                f"{rd['n_windows']:2d} fen.  {ric}"
            )
    else:
        print(f"  Regimes:   pas de donnees (regime_analysis NULL)")

    # Flags
    critical_flags = [(s, m) for s, m in ana["flags"] if s == "critical"]
    warning_flags  = [(s, m) for s, m in ana["flags"] if s == "warning"]

    if critical_flags:
        print(f"  RED FLAGS:")
        for _, msg in critical_flags:
            print(f"    [!!] {msg}")
    if warning_flags:
        print(f"  Warnings:")
        for _, msg in warning_flags:
            print(f"    [!]  {msg}")


def print_workflow_advice(strategy: str, viable: list[str], borderline: list[str], eliminated: list[str]) -> None:
    SEP = "=" * 80
    effective = viable + borderline
    print(f"\n  {SEP}")
    print(f"  WORKFLOW — ETAPE SUIVANTE (--apply)")
    print(f"  {SEP}")
    if not effective:
        print(f"  [XX] Strategie non viable — ne pas appliquer")
        print(f"       Critere requis : >= 5 assets VIABLE ou BORDERLINE")
    elif len(effective) < 5:
        print(f"  [!]  Seulement {len(effective)} asset(s) VIABLE/BORDERLINE (critere : >= 5)")
        print(f"       Recommande : ne pas deployer. Revoir parametres ou cible d'assets.")
    else:
        if eliminated:
            excl_str = ",".join(eliminated)
            print(f"  uv run python -m scripts.optimize --strategy {strategy} --apply --exclude {excl_str}")
        else:
            print(f"  uv run python -m scripts.optimize --strategy {strategy} --apply")
        print(f"\n  VIABLE  ({len(viable)}) : {', '.join(viable) if viable else 'aucun'}")
        print(f"  BORDER  ({len(borderline)}) : {', '.join(borderline) if borderline else 'aucun'}")
        if eliminated:
            print(f"  EXCLU   ({len(eliminated)}) : {', '.join(eliminated)}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep Analysis post-WFO — vérifie la viabilité des assets Grade A/B"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--strategy", metavar="NAME", help="Analyser une stratégie spécifique")
    grp.add_argument("--all", action="store_true",  help="Analyser toutes les stratégies avec résultats Grade A/B")
    args = parser.parse_args()

    strategies_cfg = load_strategies_yaml()
    conn = sqlite3.connect(str(DB_PATH))

    try:
        strategies_to_run = get_all_strategies(conn) if args.all else [args.strategy]

        for strategy in strategies_to_run:
            results = load_results(conn, strategy)
            if not results:
                if not args.all:
                    print(f"\n[INFO] Aucun résultat Grade A/B pour '{strategy}'")
                    print(f"       Vérifier : uv run python -m scripts.optimize --strategy {strategy} --all-symbols")
                continue

            leverage = get_leverage(strategies_cfg, strategy)
            analyses = [analyze_asset(row, leverage) for row in results]

            viable, borderline, eliminated = print_summary_table(results, analyses, strategy)

            for row, ana in zip(results, analyses):
                print_asset_detail(row, ana, leverage)

            print_workflow_advice(strategy, viable, borderline, eliminated)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
