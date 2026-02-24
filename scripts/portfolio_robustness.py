"""Portfolio Robustness Analysis — Sprint 44

Valide la robustesse d'un portfolio backtest via 4 méthodes :
  1. Block Bootstrap (CI sur return et max DD)
  2. Regime-Conditional Stress Test (scénarios marché)
  3. Historical Stress Injection (crashes réels)
  4. CVaR (Conditional Value at Risk)

Usage:
  uv run python -m scripts.portfolio_robustness --label "grid_atr_14assets_7x_post40a"
  uv run python -m scripts.portfolio_robustness --labels "label1,label2"
  uv run python -m scripts.portfolio_robustness --label "..." \\
      --n-simulations 10000 --block-size 7 --confidence 95 --seed 42 --save
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── Windows UTF-8 fix ────────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DB_PATH = Path(__file__).parent.parent / "data" / "scalp_radar.db"

# ── Constantes ───────────────────────────────────────────────────

HISTORICAL_CRASHES = {
    "COVID crash": ("2020-03-09", "2020-03-23"),
    "China ban": ("2021-05-19", "2021-05-26"),
    "LUNA collapse": ("2022-05-05", "2022-05-14"),
    "FTX collapse": ("2022-11-06", "2022-11-14"),
    "Aug 2024 crash": ("2024-08-02", "2024-08-08"),
}

STRESS_SCENARIOS = {
    "Bear prolongé 6m": {"RANGE": 0.30, "BULL": 0.05, "BEAR": 0.50, "CRASH": 0.15},
    "Double crash": {"RANGE": 0.50, "BULL": 0.10, "BEAR": 0.10, "CRASH": 0.30},
    "Range permanent": {"RANGE": 0.95, "BULL": 0.03, "BEAR": 0.01, "CRASH": 0.01},
    "Bull run": {"RANGE": 0.40, "BULL": 0.50, "BEAR": 0.05, "CRASH": 0.05},
    "Crypto winter": {"RANGE": 0.20, "BULL": 0.02, "BEAR": 0.60, "CRASH": 0.18},
}

RECOVERY_CAP_DAYS = 365


# ── Data Loading ─────────────────────────────────────────────────


def load_backtest(conn: sqlite3.Connection, label: str) -> dict | None:
    """Charge un portfolio backtest par label, parse les colonnes JSON."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM portfolio_backtests WHERE label = ? ORDER BY id DESC LIMIT 1",
        (label,),
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    for key in ("equity_curve", "per_asset_results", "kill_switch_events", "assets"):
        if d.get(key):
            d[key] = json.loads(d[key])
    for key in ("regime_analysis", "btc_equity_curve"):
        d[key] = json.loads(d[key]) if d.get(key) else None
    return d


def extract_daily_returns(
    equity_curve: list[dict],
) -> tuple[list[date], list[float]]:
    """Agrège l'equity curve par jour, retourne (dates, daily_returns)."""
    by_day: dict[date, float] = {}
    for pt in equity_curve:
        ts = pt["timestamp"]
        if isinstance(ts, str):
            # Parse ISO 8601 — accepte avec ou sans timezone
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            dt = ts
        d = dt.date() if hasattr(dt, "date") else dt
        by_day[d] = pt["equity"]

    sorted_days = sorted(by_day.keys())
    if len(sorted_days) < 2:
        return [], []

    dates: list[date] = []
    returns: list[float] = []
    for i in range(1, len(sorted_days)):
        prev_eq = by_day[sorted_days[i - 1]]
        curr_eq = by_day[sorted_days[i]]
        if prev_eq > 0:
            dates.append(sorted_days[i])
            returns.append((curr_eq - prev_eq) / prev_eq)

    if len(returns) < 30:
        print(f"  [WARNING] Seulement {len(returns)} jours de données (< 30)")

    return dates, returns


def extract_daily_prices(equity_curve: list[dict]) -> tuple[list[date], list[float]]:
    """Extrait les prix/equity journaliers depuis une equity curve."""
    by_day: dict[date, float] = {}
    for pt in equity_curve:
        ts = pt["timestamp"]
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            dt = ts
        d = dt.date() if hasattr(dt, "date") else dt
        by_day[d] = pt["equity"]
    sorted_days = sorted(by_day.keys())
    return sorted_days, [by_day[d] for d in sorted_days]


# ── Méthode 1 : Block Bootstrap ─────────────────────────────────


def block_bootstrap(
    daily_returns: list[float],
    n_sims: int,
    block_size: int,
    confidence: float,
    kill_switch_pct: float,
) -> dict:
    """Block bootstrap sur les returns journaliers."""
    n = len(daily_returns)
    arr = np.array(daily_returns)
    n_blocks_available = n - block_size + 1
    if n_blocks_available < 1:
        print(f"  [WARNING] Pas assez de jours ({n}) pour block_size={block_size}")
        return {}

    blocks = [arr[i : i + block_size] for i in range(n_blocks_available)]

    sim_returns = np.empty(n_sims)
    sim_dds = np.empty(n_sims)
    n_blocks_needed = math.ceil(n / block_size)

    for i in tqdm(range(n_sims), desc="  Block Bootstrap"):
        idx = np.random.randint(0, len(blocks), size=n_blocks_needed)
        sim = np.concatenate([blocks[j] for j in idx])[:n]
        equity = np.cumprod(1.0 + sim)
        peak = np.maximum.accumulate(equity)
        sim_returns[i] = equity[-1] - 1.0
        dd = (equity - peak) / peak
        sim_dds[i] = dd.min()

    alpha = (100 - confidence) / 2
    return {
        "n_sims": n_sims,
        "block_size": block_size,
        "n_days": n,
        "median_return": float(np.median(sim_returns)),
        "ci_return_low": float(np.percentile(sim_returns, alpha)),
        "ci_return_high": float(np.percentile(sim_returns, 100 - alpha)),
        "median_dd": float(np.median(sim_dds)),
        "ci_dd_low": float(np.percentile(sim_dds, 100 - alpha)),
        "ci_dd_high": float(np.percentile(sim_dds, alpha)),
        "prob_loss": float((sim_returns < 0).mean()),
        "prob_dd_30": float((sim_dds < -0.30).mean()),
        "prob_dd_ks": float((sim_dds < -kill_switch_pct / 100).mean()),
    }


# ── Méthode 4 : CVaR ────────────────────────────────────────────


def compute_cvar(
    daily_returns: list[float],
    kill_switch_pct: float,
    regime_pools: dict[str, list[float]] | None = None,
) -> dict:
    """CVaR 5% journalier + compound annualisé."""
    arr = np.sort(np.array(daily_returns))
    n = len(arr)
    if n < 20:
        print(f"  [WARNING] Seulement {n} jours — CVaR peu fiable")

    var_idx = max(1, int(n * 0.05))
    var_5 = float(arr[var_idx])
    cvar_5 = float(arr[:var_idx].mean())
    # Compound 30j (worst month estimate) — plus actionable que 365j
    cvar_30d = (1.0 + cvar_5) ** 30 - 1.0
    # Compound annualization (informatif — quasi toujours ~-100%)
    cvar_annualized = (1.0 + cvar_5) ** 365 - 1.0

    cvar_by_regime: dict[str, float] = {}
    if regime_pools:
        for regime, pool in regime_pools.items():
            if len(pool) >= 5:
                s = np.sort(np.array(pool))
                idx = max(1, int(len(s) * 0.05))
                cvar_by_regime[regime] = float(s[:idx].mean())

    return {
        "var_5_daily": var_5,
        "cvar_5_daily": cvar_5,
        "cvar_30d": cvar_30d,
        "cvar_5_annualized": cvar_annualized,
        "cvar_by_regime": cvar_by_regime,
    }


# ── Méthode 3 : Historical Stress ───────────────────────────────


def historical_stress(
    equity_curve: list[dict],
    btc_equity_curve: list[dict] | None,
) -> dict:
    """Performance réelle du portfolio pendant les crashes historiques."""
    port_dates, port_prices = extract_daily_prices(equity_curve)
    btc_dates, btc_prices = (
        extract_daily_prices(btc_equity_curve) if btc_equity_curve else ([], [])
    )

    if not port_dates:
        return {}

    port_start = port_dates[0]
    port_end = port_dates[-1]

    results: dict[str, dict] = {}
    for event_name, (start_str, end_str) in HISTORICAL_CRASHES.items():
        ev_start = date.fromisoformat(start_str)
        ev_end = date.fromisoformat(end_str)

        if ev_start > port_end or ev_end < port_start:
            results[event_name] = {"status": "N/A", "reason": "backtest hors période"}
            continue

        # Trouver les indices les plus proches
        i_start = _find_closest_idx(port_dates, ev_start)
        i_end = _find_closest_idx(port_dates, ev_end)

        # Peak avant le crash
        peak_before = max(port_prices[: i_start + 1]) if i_start > 0 else port_prices[0]
        # Min pendant le crash
        min_during = min(port_prices[i_start : i_end + 1])
        portfolio_dd = (min_during - peak_before) / peak_before

        # BTC DD
        btc_dd = None
        if btc_dates:
            bi_start = _find_closest_idx(btc_dates, ev_start)
            bi_end = _find_closest_idx(btc_dates, ev_end)
            btc_peak = max(btc_prices[: bi_start + 1]) if bi_start > 0 else btc_prices[0]
            btc_min = min(btc_prices[bi_start : bi_end + 1])
            btc_dd = (btc_min - btc_peak) / btc_peak

        # Recovery (cap 365j)
        recovery_days = _compute_recovery(port_dates, port_prices, i_end, peak_before)

        results[event_name] = {
            "status": "OK",
            "period": f"{start_str} → {end_str}",
            "portfolio_dd": portfolio_dd,
            "btc_dd": btc_dd,
            "recovery_days": recovery_days,
        }

    return results


def _find_closest_idx(sorted_dates: list[date], target: date) -> int:
    """Trouve l'index de la date la plus proche dans une liste triée."""
    if target <= sorted_dates[0]:
        return 0
    if target >= sorted_dates[-1]:
        return len(sorted_dates) - 1
    # Recherche binaire
    lo, hi = 0, len(sorted_dates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_dates[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    # Vérifier si lo ou lo-1 est plus proche
    if lo > 0 and (target - sorted_dates[lo - 1]) < (sorted_dates[lo] - target):
        return lo - 1
    return lo


def _compute_recovery(
    dates: list[date],
    prices: list[float],
    end_idx: int,
    peak_before: float,
) -> int | None:
    """Jours pour revenir au peak pré-crash après end_idx. None si >365j."""
    if end_idx >= len(dates) - 1:
        return None
    end_date = dates[end_idx]
    for i in range(end_idx + 1, len(dates)):
        if prices[i] >= peak_before:
            delta = (dates[i] - end_date).days
            return delta
        if (dates[i] - end_date).days > RECOVERY_CAP_DAYS:
            return None
    return None


# ── Méthode 2 : Regime Stress ───────────────────────────────────


def classify_btc_regimes(
    btc_daily_prices: list[float], dates: list[date]
) -> list[str]:
    """Classifie chaque jour par régime BTC (mêmes seuils que metrics.py).

    Priorité : CRASH > BULL > BEAR > RANGE
    - CRASH : DD > 30% en ≤ 14j
    - BULL  : return 30j > +20%
    - BEAR  : return 30j < -20%
    - RANGE : sinon
    """
    n = len(btc_daily_prices)
    regimes: list[str] = []
    for i in range(n):
        # Return 30j
        j = max(0, i - 30)
        if btc_daily_prices[j] > 0:
            ret_30d = (btc_daily_prices[i] - btc_daily_prices[j]) / btc_daily_prices[j] * 100
        else:
            ret_30d = 0.0

        # Crash detection : DD > 30% en ≤ 14j
        window_start = max(0, i - 14)
        window = btc_daily_prices[window_start : i + 1]
        dd_14d = 0.0
        if len(window) >= 2:
            peak_14d = max(window)
            if peak_14d > 0:
                dd_14d = (min(window) - peak_14d) / peak_14d * 100

        if dd_14d < -30:
            regimes.append("CRASH")
        elif ret_30d > 20:
            regimes.append("BULL")
        elif ret_30d < -20:
            regimes.append("BEAR")
        else:
            regimes.append("RANGE")
    return regimes


def regime_stress(
    daily_returns: list[float],
    daily_dates: list[date],
    btc_equity_curve: list[dict] | None,
    n_sims: int = 1000,
) -> dict:
    """Stress test par régime de marché."""
    if not btc_equity_curve:
        return {"skipped": True, "reason": "btc_equity_curve non disponible"}

    btc_dates, btc_prices = extract_daily_prices(btc_equity_curve)
    if len(btc_prices) < 31:
        return {"skipped": True, "reason": "btc_equity_curve trop courte (< 31j)"}

    # Classifier régimes BTC
    btc_regimes = classify_btc_regimes(btc_prices, btc_dates)

    # Aligner portfolio returns sur les dates BTC (nearest-date matching)
    # Les equity curves sont sous-échantillonnées (~500 pts / 2000j)
    # donc les dates exactes ne matchent quasiment jamais
    date_to_return = dict(zip(daily_dates, daily_returns))
    regime_pools: dict[str, list[float]] = {
        "RANGE": [], "BULL": [], "BEAR": [], "CRASH": [],
    }
    all_aligned: list[float] = []
    sorted_port_dates = sorted(daily_dates)

    for i, btc_d in enumerate(btc_dates):
        # Chercher la date portfolio la plus proche
        if btc_d in date_to_return:
            r = date_to_return[btc_d]
        elif sorted_port_dates:
            closest_idx = _find_closest_idx(sorted_port_dates, btc_d)
            closest_d = sorted_port_dates[closest_idx]
            # Accepter un écart max de 5 jours
            if abs((closest_d - btc_d).days) <= 5 and closest_d in date_to_return:
                r = date_to_return[closest_d]
            else:
                continue
        else:
            continue
        regime_pools[btc_regimes[i]].append(r)
        all_aligned.append(r)

    if not all_aligned:
        return {"skipped": True, "reason": "Aucun jour en commun portfolio/BTC"}

    # Distribution observée
    observed_dist = {
        regime: len(pool) / len(all_aligned) * 100 if all_aligned else 0
        for regime, pool in regime_pools.items()
    }

    # Simuler chaque scénario
    sim_days = min(365, len(all_aligned))
    regime_names = ["RANGE", "BULL", "BEAR", "CRASH"]
    all_arr = np.array(all_aligned)

    scenario_results: dict[str, dict] = {}
    for scenario_name, weights in STRESS_SCENARIOS.items():
        probs = np.array([weights[r] for r in regime_names])
        # Préparer les pools numpy
        pools_np = {}
        for r in regime_names:
            if regime_pools[r]:
                pools_np[r] = np.array(regime_pools[r])
            else:
                pools_np[r] = all_arr  # fallback

        sim_rets = np.empty(n_sims)
        sim_dds = np.empty(n_sims)

        for s in range(n_sims):
            # Tirer les régimes pour chaque jour
            chosen_regimes = np.random.choice(regime_names, size=sim_days, p=probs)
            sim = np.empty(sim_days)
            for day_idx in range(sim_days):
                pool = pools_np[chosen_regimes[day_idx]]
                sim[day_idx] = pool[np.random.randint(0, len(pool))]

            equity = np.cumprod(1.0 + sim)
            peak = np.maximum.accumulate(equity)
            sim_rets[s] = equity[-1] - 1.0
            sim_dds[s] = ((equity - peak) / peak).min()

        scenario_results[scenario_name] = {
            "median_return": float(np.median(sim_rets)),
            "median_dd": float(np.median(sim_dds)),
            "prob_loss": float((sim_rets < 0).mean()),
        }

    return {
        "skipped": False,
        "n_sims": n_sims,
        "sim_days": sim_days,
        "observed_distribution": observed_dist,
        "regime_pools": regime_pools,
        "scenarios": scenario_results,
    }


# ── Verdict ──────────────────────────────────────────────────────


def compute_verdict(
    bootstrap: dict,
    cvar: dict,
    hist_stress: dict,
    regime: dict,
    kill_switch_pct: float,
) -> dict:
    """Évalue les critères GO/NO-GO et retourne le verdict."""
    criteria: list[dict] = []

    # 1. CI95 return borne basse > 0%
    if bootstrap:
        ci_low = bootstrap["ci_return_low"]
        criteria.append({
            "name": "CI95 return borne basse > 0%",
            "value": f"{ci_low * 100:+.1f}%",
            "pass": ci_low > 0,
        })

    # 2. Probabilité de perte < 10%
    if bootstrap:
        prob = bootstrap["prob_loss"]
        criteria.append({
            "name": "Probabilité de perte < 10%",
            "value": f"{prob * 100:.1f}%",
            "pass": prob < 0.10,
        })

    # 3. CVaR 5% 30j (compound) < kill_switch
    if cvar:
        cvar_30d = abs(cvar["cvar_30d"]) * 100
        criteria.append({
            "name": f"CVaR 5% 30j < kill_switch ({kill_switch_pct}%)",
            "value": f"{cvar_30d:.1f}%",
            "pass": cvar_30d < kill_switch_pct,
        })

    # 4. Survit à ≥ 3/5 crashes historiques avec DD < -40%
    if hist_stress:
        survived = 0
        total_evaluated = 0
        for _ev, info in hist_stress.items():
            if info.get("status") == "OK":
                total_evaluated += 1
                if info["portfolio_dd"] > -0.40:
                    survived += 1
        if total_evaluated > 0:
            criteria.append({
                "name": f"Survit crashes historiques (DD < -40%)",
                "value": f"{survived}/{total_evaluated}",
                "pass": survived >= min(3, total_evaluated),
            })

    n_pass = sum(1 for c in criteria if c["pass"])
    n_fail = sum(1 for c in criteria if not c["pass"])

    if n_fail == 0:
        verdict = "VIABLE"
    elif n_fail <= 1:
        verdict = "CAUTION"
    else:
        verdict = "FAIL"

    return {"verdict": verdict, "criteria": criteria, "n_pass": n_pass, "n_fail": n_fail}


# ── Affichage ────────────────────────────────────────────────────

SEP = "=" * 70


def print_header(label: str, backtest: dict) -> None:
    print(f"\n{SEP}")
    print(f"  PORTFOLIO ROBUSTNESS ANALYSIS — {label}")
    print(SEP)
    print(f"  Capital initial : {backtest['initial_capital']:,.0f} USDT")
    print(f"  Return total    : {backtest['total_return_pct']:+.1f}%")
    print(f"  Max DD          : {backtest['max_drawdown_pct']:.1f}%")
    print(f"  Trades          : {backtest['total_trades']}")
    print(f"  Kill switch     : {backtest['kill_switch_pct']}%")
    print()


def print_bootstrap(bs: dict) -> None:
    if not bs:
        print("  [SKIP] Block Bootstrap — données insuffisantes\n")
        return
    sep = "-" * 70
    print(sep)
    print(f"  1. BLOCK BOOTSTRAP ({bs['n_sims']} simulations, blocs {bs['block_size']}j)")
    print(sep)
    print(f"  Return  : median {bs['median_return'] * 100:+.1f}%"
          f"  CI{int(100 - 2 * ((100 - 95) / 2))} [{bs['ci_return_low'] * 100:+.1f}%, {bs['ci_return_high'] * 100:+.1f}%]")
    print(f"  Max DD  : median {bs['median_dd'] * 100:.1f}%"
          f"  CI [{bs['ci_dd_low'] * 100:.1f}%, {bs['ci_dd_high'] * 100:.1f}%]")
    print(f"  Prob. perte (return < 0)          : {bs['prob_loss'] * 100:.1f}%")
    print(f"  Prob. DD > -30%                   : {bs['prob_dd_30'] * 100:.1f}%")
    print(f"  Prob. DD > kill_switch             : {bs['prob_dd_ks'] * 100:.1f}%")
    print()


def print_cvar(cv: dict, kill_switch_pct: float) -> None:
    if not cv:
        print("  [SKIP] CVaR — données insuffisantes\n")
        return
    sep = "-" * 70
    print(sep)
    print("  4. VALUE AT RISK / CONDITIONAL VALUE AT RISK")
    print(sep)
    print(f"  VaR 5% journalier  : {cv['var_5_daily'] * 100:.2f}%"
          f"  (1 jour sur 20, perte > {abs(cv['var_5_daily']) * 100:.2f}%)")
    print(f"  CVaR 5% journalier : {cv['cvar_5_daily'] * 100:.2f}%"
          f"  (quand ça va mal, perte moyenne {abs(cv['cvar_5_daily']) * 100:.2f}%)")
    print(f"  CVaR 5% 30j (compound) : {cv['cvar_30d'] * 100:.1f}%"
          f"  (pire mois estimé)")
    print(f"  CVaR 5% 365j (compound) : {cv['cvar_5_annualized'] * 100:.1f}%"
          f"  (informatif)")
    if cv["cvar_by_regime"]:
        print("  CVaR par régime :")
        for regime in ("RANGE", "BULL", "BEAR", "CRASH"):
            if regime in cv["cvar_by_regime"]:
                print(f"    {regime:6s} : {cv['cvar_by_regime'][regime] * 100:.2f}%")
    cvar_30d_abs = abs(cv["cvar_30d"]) * 100
    status = "OK" if cvar_30d_abs < kill_switch_pct else "DANGER"
    margin = kill_switch_pct - cvar_30d_abs
    print(f"  CVaR 30j vs kill_switch : {cvar_30d_abs:.1f}% vs {kill_switch_pct}%"
          f" {'✅' if status == 'OK' else '❌'} (marge {margin:+.1f} pts)")
    print()


def print_historical(hs: dict) -> None:
    if not hs:
        print("  [SKIP] Historical Stress — données insuffisantes\n")
        return
    sep = "-" * 70
    print(sep)
    print("  3. HISTORICAL STRESS — Performance pendant crashes réels")
    print(sep)
    header = f"  {'Événement':<20s} {'Période':<22s} {'Portfolio DD':>12s} {'BTC DD':>10s} {'Recovery':>10s}"
    print(header)
    print("  " + "-" * 76)
    for event_name, info in hs.items():
        if info.get("status") == "N/A":
            print(f"  {event_name:<20s} {'N/A — ' + info.get('reason', '')}")
            continue
        dd_str = f"{info['portfolio_dd'] * 100:.1f}%"
        btc_str = f"{info['btc_dd'] * 100:.1f}%" if info.get("btc_dd") is not None else "N/A"
        if info["recovery_days"] is not None:
            rec_str = f"{info['recovery_days']}j"
        else:
            rec_str = ">365j"
        print(f"  {event_name:<20s} {info['period']:<22s} {dd_str:>12s} {btc_str:>10s} {rec_str:>10s}")
    print()


def print_regime(rs: dict) -> None:
    if rs.get("skipped"):
        print(f"  [SKIP] Regime Stress — {rs.get('reason', 'non disponible')}\n")
        return
    sep = "-" * 70
    print(sep)
    print(f"  2. REGIME STRESS SCENARIOS ({rs['n_sims']} sims × {rs['sim_days']}j par scénario)")
    print(sep)
    # Distribution observée
    obs = rs.get("observed_distribution", {})
    print(f"  Distribution observée : "
          + " | ".join(f"{r}={obs.get(r, 0):.0f}%" for r in ("RANGE", "BULL", "BEAR", "CRASH")))
    print()
    header = f"  {'Scénario':<22s} {'Return (med)':>12s} {'Max DD (med)':>12s} {'Prob. perte':>12s}"
    print(header)
    print("  " + "-" * 60)
    for name, info in rs["scenarios"].items():
        print(
            f"  {name:<22s} {info['median_return'] * 100:>+11.1f}%"
            f" {info['median_dd'] * 100:>11.1f}%"
            f" {info['prob_loss'] * 100:>11.1f}%"
        )
    print()


def print_verdict(v: dict) -> None:
    print(SEP)
    print(f"  VERDICT ROBUSTESSE : {v['verdict']}")
    print(SEP)
    for c in v["criteria"]:
        icon = "✅" if c["pass"] else "❌"
        print(f"  {icon} {c['name']:<45s} : {c['value']}")
    print()


# ── Sauvegarde DB ────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS portfolio_robustness (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id INTEGER REFERENCES portfolio_backtests(id),
    label TEXT,
    created_at TEXT,
    bootstrap_n_sims INTEGER,
    bootstrap_block_size INTEGER,
    bootstrap_median_return REAL,
    bootstrap_ci95_return_low REAL,
    bootstrap_ci95_return_high REAL,
    bootstrap_median_dd REAL,
    bootstrap_ci95_dd_low REAL,
    bootstrap_ci95_dd_high REAL,
    bootstrap_prob_loss REAL,
    bootstrap_prob_dd_30 REAL,
    bootstrap_prob_dd_ks REAL,
    regime_stress_results TEXT,
    historical_stress_results TEXT,
    var_5_daily REAL,
    cvar_5_daily REAL,
    cvar_30d REAL,
    cvar_5_annualized REAL,
    cvar_by_regime TEXT,
    verdict TEXT,
    verdict_details TEXT
)
"""


def save_results(
    conn: sqlite3.Connection,
    backtest_id: int,
    label: str,
    bootstrap: dict,
    cvar: dict,
    hist_stress: dict,
    regime: dict,
    verdict: dict,
) -> int:
    """Sauvegarde les résultats dans portfolio_robustness. Retourne l'ID."""
    conn.execute(CREATE_TABLE_SQL)

    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """INSERT INTO portfolio_robustness (
            backtest_id, label, created_at,
            bootstrap_n_sims, bootstrap_block_size,
            bootstrap_median_return, bootstrap_ci95_return_low, bootstrap_ci95_return_high,
            bootstrap_median_dd, bootstrap_ci95_dd_low, bootstrap_ci95_dd_high,
            bootstrap_prob_loss, bootstrap_prob_dd_30, bootstrap_prob_dd_ks,
            regime_stress_results, historical_stress_results,
            var_5_daily, cvar_5_daily, cvar_30d, cvar_5_annualized, cvar_by_regime,
            verdict, verdict_details
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            backtest_id,
            label,
            now,
            bootstrap.get("n_sims"),
            bootstrap.get("block_size"),
            bootstrap.get("median_return"),
            bootstrap.get("ci_return_low"),
            bootstrap.get("ci_return_high"),
            bootstrap.get("median_dd"),
            bootstrap.get("ci_dd_low"),
            bootstrap.get("ci_dd_high"),
            bootstrap.get("prob_loss"),
            bootstrap.get("prob_dd_30"),
            bootstrap.get("prob_dd_ks"),
            json.dumps(regime.get("scenarios", {}), ensure_ascii=False) if regime else None,
            json.dumps(hist_stress, ensure_ascii=False) if hist_stress else None,
            cvar.get("var_5_daily"),
            cvar.get("cvar_5_daily"),
            cvar.get("cvar_30d"),
            cvar.get("cvar_5_annualized"),
            json.dumps(cvar.get("cvar_by_regime", {}), ensure_ascii=False) if cvar else None,
            verdict.get("verdict"),
            json.dumps(verdict, ensure_ascii=False),
        ),
    )
    conn.commit()
    return cursor.lastrowid


# ── Main ─────────────────────────────────────────────────────────


def analyze_label(
    conn: sqlite3.Connection,
    label: str,
    n_sims: int,
    block_size: int,
    confidence: float,
    save: bool,
) -> dict | None:
    """Analyse complète d'un label. Retourne le verdict."""
    backtest = load_backtest(conn, label)
    if backtest is None:
        print(f"\n[ERREUR] Label '{label}' introuvable en DB")
        return None

    kill_switch_pct = backtest.get("kill_switch_pct", 45.0)
    print_header(label, backtest)

    # Extract daily returns
    dates, daily_returns = extract_daily_returns(backtest["equity_curve"])
    if len(daily_returns) < 5:
        print("[ERREUR] Equity curve trop courte pour l'analyse")
        return None

    # Méthode 1 — Block Bootstrap
    bs = block_bootstrap(daily_returns, n_sims, block_size, confidence, kill_switch_pct)
    print_bootstrap(bs)

    # Méthode 4 — CVaR
    # Build regime pools si possible
    regime_pools = None
    rs = regime_stress(daily_returns, dates, backtest.get("btc_equity_curve"), n_sims=1000)
    if not rs.get("skipped"):
        regime_pools = rs.get("regime_pools")
    cv = compute_cvar(daily_returns, kill_switch_pct, regime_pools)
    print_cvar(cv, kill_switch_pct)

    # Méthode 3 — Historical Stress
    hs = historical_stress(backtest["equity_curve"], backtest.get("btc_equity_curve"))
    print_historical(hs)

    # Méthode 2 — Regime Stress
    print_regime(rs)

    # Verdict
    verdict = compute_verdict(bs, cv, hs, rs, kill_switch_pct)
    print_verdict(verdict)

    # Save
    if save:
        rid = save_results(conn, backtest["id"], label, bs, cv, hs, rs, verdict)
        print(f"  [SAVE] Résultats sauvegardés en DB (id={rid})")
        print()

    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portfolio Robustness Analysis — Bootstrap, Regime Stress, Historical Stress, CVaR"
    )
    parser.add_argument("--label", type=str, help="Label du backtest à analyser")
    parser.add_argument("--labels", type=str, help="Labels séparés par virgule")
    parser.add_argument("--n-simulations", type=int, default=5000, help="Nombre de simulations bootstrap (défaut: 5000)")
    parser.add_argument("--block-size", type=int, default=7, help="Taille des blocs en jours (défaut: 7)")
    parser.add_argument("--confidence", type=float, default=95, help="Niveau de confiance %% (défaut: 95)")
    parser.add_argument("--seed", type=int, default=42, help="Seed RNG pour reproductibilité (défaut: 42)")
    parser.add_argument("--save", action="store_true", help="Sauvegarder les résultats en DB")
    parser.add_argument("--db", type=str, default=None, help="Chemin DB custom")
    args = parser.parse_args()

    # Seed reproductibilité
    np.random.seed(args.seed)

    # Résoudre labels
    labels: list[str] = []
    if args.label:
        labels = [args.label]
    elif args.labels:
        labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    else:
        print("[ERREUR] Spécifier --label ou --labels")
        sys.exit(1)

    db_path = Path(args.db) if args.db else DB_PATH
    if not db_path.exists():
        print(f"[ERREUR] DB introuvable : {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    try:
        for label in labels:
            analyze_label(
                conn, label, args.n_simulations, args.block_size,
                args.confidence, args.save,
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
