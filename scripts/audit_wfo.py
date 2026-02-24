"""Audit WFO — Vérification anomalies grid_multi_tf + grid_range_atr."""
import sqlite3
import json

conn = sqlite3.connect("data/scalp_radar.db")
conn.row_factory = sqlite3.Row


def audit_strategy(strategy_name):
    print("=" * 95)
    print(f"  AUDIT {strategy_name}")
    print("=" * 95)

    rows = conn.execute(
        """SELECT asset, grade, total_score, oos_sharpe, consistency,
                  oos_is_ratio, dsr, param_stability, monte_carlo_pvalue, mc_underpowered,
                  n_windows, n_distinct_combos, best_params, wfo_windows
           FROM optimization_results
           WHERE is_latest = 1 AND strategy_name = ?
           ORDER BY total_score DESC""",
        (strategy_name,),
    ).fetchall()

    # 1. Tableau principal
    header = (
        f"{'Asset':<12} {'Gr':>2} {'Score':>5} {'Sharpe':>7} {'Cons':>5} "
        f"{'OOS/IS':>6} {'DSR':>5} {'Stab':>5} {'MC p':>6} {'Undpw':>5} "
        f"{'Win':>4} {'Combos':>6}"
    )
    print(header)
    print("-" * 95)
    for r in rows:
        undpw = "YES" if r["mc_underpowered"] else "no"
        mc_p = f"{r['monte_carlo_pvalue']:.3f}" if r["monte_carlo_pvalue"] is not None else "N/A"
        print(
            f"{r['asset']:<12} {r['grade']:>2} {r['total_score']:>5} "
            f"{r['oos_sharpe']:>7.2f} {r['consistency']:>5.2f} "
            f"{r['oos_is_ratio']:>6.2f} {r['dsr']:>5.2f} {r['param_stability']:>5.2f} "
            f"{mc_p:>6} {undpw:>5} {r['n_windows']:>4} {r['n_distinct_combos']:>6}"
        )

    # 2. Red flags
    print()
    print("  RED FLAGS:")
    flags = []
    for r in rows:
        asset = r["asset"]
        if r["oos_is_ratio"] and r["oos_is_ratio"] > 1.5:
            flags.append(f"  ⚠️  {asset}: OOS/IS ratio = {r['oos_is_ratio']:.2f} (suspect > 1.5)")
        if r["oos_sharpe"] and r["oos_sharpe"] > 20:
            flags.append(f"  ⚠️  {asset}: Sharpe OOS = {r['oos_sharpe']:.1f} (anormalement élevé)")
        if r["mc_underpowered"]:
            flags.append(f"  ⚠️  {asset}: Monte Carlo UNDERPOWERED (trop peu de trades)")
        if r["consistency"] and r["consistency"] < 0.5:
            flags.append(f"  ⚠️  {asset}: Consistance = {r['consistency']:.2f} (< 50%)")
        if r["param_stability"] and r["param_stability"] < 0.3:
            flags.append(f"  ⚠️  {asset}: Stabilité params = {r['param_stability']:.2f} (fragile)")
    if flags:
        for f in flags:
            print(f)
    else:
        print("  ✅ Aucun red flag détecté")

    # 3. Nombre de trades OOS
    print()
    print("  TRADES OOS PAR FENETRE:")
    for r in rows:
        raw = r["wfo_windows"]
        if not raw:
            print(f"  {r['asset']:<12} Grade {r['grade']} | PAS DE WINDOWS DATA")
            continue
        windows = json.loads(raw) if isinstance(raw, str) else raw
        # Double-encoded? Si c'est encore une string, re-parse
        if isinstance(windows, str):
            windows = json.loads(windows)
        if not isinstance(windows, list):
            print(f"  {r['asset']:<12} Grade {r['grade']} | FORMAT INATTENDU: {type(windows)}")
            continue
        if windows:
            oos_trades = [w.get("oos_trades", 0) if isinstance(w, dict) else 0 for w in windows]
            total_oos = sum(oos_trades)
            min_oos = min(oos_trades)
            max_oos = max(oos_trades)
            avg_oos = total_oos / len(oos_trades) if oos_trades else 0
            flag = " ⚠️ PEU" if total_oos < 50 else ""
            print(
                f"  {r['asset']:<12} Grade {r['grade']} | "
                f"total={total_oos:>4}, avg={avg_oos:>5.1f}, "
                f"min={min_oos:>3}, max={max_oos:>3}{flag}"
            )

    # 4. Convergence des params
    print()
    print("  CONVERGENCE PARAMS:")
    param_vals = {}
    for r in rows:
        params = json.loads(r["best_params"]) if isinstance(r["best_params"], str) else r["best_params"]
        if isinstance(params, str):
            params = json.loads(params)
        if params:
            for k, v in params.items():
                param_vals.setdefault(k, []).append(v)

    for k, vals in sorted(param_vals.items()):
        unique = sorted(set(str(v) for v in vals))
        n_unique = len(unique)
        total = len(vals)
        # Trouver la valeur la plus fréquente
        from collections import Counter
        counter = Counter(str(v) for v in vals)
        most_common_val, most_common_count = counter.most_common(1)[0]
        pct = most_common_count / total * 100
        status = "✅" if pct >= 60 else "⚠️" if pct >= 40 else "❌"
        print(f"  {status} {k:<25}: {n_unique:>2} valeurs, mode={most_common_val} ({pct:.0f}%) — {unique}")

    print()


# Run
audit_strategy("grid_multi_tf")
audit_strategy("grid_range_atr")

# Resume final
print("=" * 95)
print("  RESUME")
print("=" * 95)
for strat in ["grid_multi_tf", "grid_range_atr"]:
    rows = conn.execute(
        """SELECT grade, COUNT(*) as cnt
           FROM optimization_results
           WHERE is_latest = 1 AND strategy_name = ?
           GROUP BY grade ORDER BY grade""",
        (strat,),
    ).fetchall()
    grades = {r["grade"]: r["cnt"] for r in rows}
    total = sum(grades.values())
    print(f"  {strat:<16}: {total} assets — " + ", ".join(f"{g}={n}" for g, n in sorted(grades.items())))

conn.close()