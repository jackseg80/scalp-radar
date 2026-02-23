"""Stress test leverage multi-fenêtre.

Lance N portfolio backtests pour grid_atr et grid_boltrend avec différents
leverages et fenêtres temporelles, puis génère un rapport comparatif pour
choisir le leverage optimal de chaque stratégie.

Kill switch DÉSACTIVÉ (99%) pendant les runs pour voir le vrai max drawdown.
L'analyse KS est effectuée a posteriori aux seuils 30%/45%/60%.

Usage :
    uv run python -m scripts.stress_test_leverage
    uv run python -m scripts.stress_test_leverage --strategy grid_boltrend
    uv run python -m scripts.stress_test_leverage --leverages 5,7
    uv run python -m scripts.stress_test_leverage --days 180
    uv run python -m scripts.stress_test_leverage --strategy grid_boltrend --leverages 3,5 --days 90
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import faulthandler
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

# Activer faulthandler pour capter les segfaults (traceback Python même sur SIGSEGV)
faulthandler.enable()

from loguru import logger

from backend.core.config import get_config
from backend.core.database import Database
from backend.core.logging_setup import setup_logging


# ---------------------------------------------------------------------------
# Matrice de runs par défaut
# ---------------------------------------------------------------------------

DEFAULT_MATRIX: dict[str, dict[str, Any]] = {
    "grid_boltrend": {
        "leverages": [2, 4, 6, 8],
        "days": ["auto", 180, 90],
    },
    "grid_atr": {
        "leverages": [2, 4, 6, 8],
        "days": ["auto", 180],
    },
}

# Seuils kill switch analysés a posteriori
KS_THRESHOLDS = [30, 45, 60]
KS_WINDOW_HOURS = 24

# Critères de recommandation
MIN_LIQ_DIST_PCT = 50.0  # Distance liquidation minimale acceptable (%)
MAX_DD_PCT = 40.0         # Max drawdown acceptable (%)
# KS@45 = 0 requis sur la fenêtre longue (auto)


# ---------------------------------------------------------------------------
# Utilitaires métriques
# ---------------------------------------------------------------------------


def _count_ks_triggers(
    snapshots: list,
    threshold_pct: float,
    window_hours: int = 24,
) -> int:
    """Compte les déclenchements du kill switch à un seuil donné (a posteriori).

    Même logique que PortfolioBacktester._check_kill_switch() mais standalone.
    """
    if not snapshots:
        return 0

    window = timedelta(hours=window_hours)
    count = 0
    in_trigger = False
    window_start_idx = 0

    for i, snap in enumerate(snapshots):
        # Avancer le début de la fenêtre glissante
        while (
            window_start_idx < i
            and snap.timestamp - snapshots[window_start_idx].timestamp > window
        ):
            window_start_idx += 1

        start_equity = snapshots[window_start_idx].total_equity
        if start_equity <= 0:
            continue

        dd_pct = (1 - snap.total_equity / start_equity) * 100

        if dd_pct >= threshold_pct and not in_trigger:
            in_trigger = True
            count += 1
        elif dd_pct < threshold_pct * 0.5:
            # Sortie de zone critique -> reset pour detecter le prochain
            in_trigger = False

    return count


def _compute_sharpe(snapshots: list) -> float:
    """Sharpe annualisé depuis la courbe d'equity (snapshots horaires)."""
    if len(snapshots) < 10:
        return 0.0

    equities = [s.total_equity for s in snapshots]
    returns = []
    for i in range(1, len(equities)):
        prev = equities[i - 1]
        curr = equities[i]
        if prev > 0:
            returns.append((curr / prev) - 1)

    if len(returns) < 5:
        return 0.0

    n = len(returns)
    mean_r = sum(returns) / n
    variance = sum((r - mean_r) ** 2 for r in returns) / n
    std_r = math.sqrt(variance) if variance > 0 else 0.0

    if std_r == 0:
        return 0.0

    # Annualiser : données 1h, 8760h/an
    return round((mean_r / std_r) * math.sqrt(8760), 2)


def _compute_calmar(total_return_pct: float, max_drawdown_pct: float) -> float:
    """Calmar = Return% / |Max DD%|. Retourne float('inf') si DD=0 et return>0."""
    if max_drawdown_pct == 0:
        return float("inf") if total_return_pct > 0 else 0.0
    return round(total_return_pct / abs(max_drawdown_pct), 2)


# ---------------------------------------------------------------------------
# Détection historique disponible
# ---------------------------------------------------------------------------


async def _detect_auto_days(
    config,
    strategy_name: str,
    exchange: str,
    db_path: str,
) -> tuple[int, datetime | None]:
    """Détecte les jours max disponibles pour une stratégie depuis la DB.

    Scanne les per_asset de la stratégie et retourne la couverture commune
    (= asset avec le moins d'historique = goulot).

    Returns:
        (common_days, goulot_start_date)
    """
    strat_config = getattr(config.strategies, strategy_name, None)
    per_asset = getattr(strat_config, "per_asset", {}) if strat_config else {}
    all_assets = sorted(per_asset.keys())

    if not all_assets:
        return 90, None

    db = Database(db_path)
    await db.init()

    # latest_start = la date de début la plus récente parmi tous les assets
    # = le goulot d'étranglement (asset avec le moins d'historique)
    latest_start: datetime | None = None
    exchanges_to_try = (
        [exchange, "bitget"] if exchange == "binance" else [exchange, "binance"]
    )

    for symbol in all_assets:
        candles = None
        for ex in exchanges_to_try:
            candles = await db.get_candles(symbol, "1h", exchange=ex, limit=1)
            if candles:
                break

        if candles:
            first_ts = candles[0].timestamp
            if latest_start is None or first_ts > latest_start:
                latest_start = first_ts

    await db.close()

    if latest_start is None:
        return 90, None

    # Soustraire ~2j de warm-up
    common_days = max((datetime.now(timezone.utc) - latest_start).days - 3, 30)
    return common_days, latest_start


# ---------------------------------------------------------------------------
# Exécution d'un run unique
# ---------------------------------------------------------------------------


_SUBPROCESS_SCRIPT = r'''
"""Subprocess worker : lance un seul portfolio backtest et retourne les résultats en JSON."""
import asyncio, gc, json, os, sys, faulthandler
faulthandler.enable()
sys.path.insert(0, os.getcwd())

from backend.backtesting.portfolio_engine import PortfolioBacktester
from backend.core.config import get_config
from backend.core.logging_setup import setup_logging
from datetime import datetime, timezone

setup_logging(level="WARNING")
config = get_config()

strategy_name = sys.argv[1]
leverage = int(sys.argv[2])
start_iso = sys.argv[3]
end_iso = sys.argv[4]
capital = float(sys.argv[5])
db_path = sys.argv[6]
exchange = sys.argv[7]
ks_window_hours = int(sys.argv[8])

start_dt = datetime.fromisoformat(start_iso)
end_dt = datetime.fromisoformat(end_iso)

# Override leverage
strat_cfg = getattr(config.strategies, strategy_name, None)
if strat_cfg is not None and hasattr(strat_cfg, "leverage"):
    strat_cfg.leverage = leverage

# Override KS runner à 99% (voir le vrai DD sans interruption)
ks_cfg = getattr(config.risk, "kill_switch", None)
if ks_cfg is not None:
    if hasattr(ks_cfg, "grid_max_session_loss_percent"):
        ks_cfg.grid_max_session_loss_percent = 99.0
    if hasattr(ks_cfg, "max_session_loss_percent"):
        ks_cfg.max_session_loss_percent = 99.0

async def _run():
    backtester = PortfolioBacktester(
        config=config,
        initial_capital=capital,
        strategy_name=strategy_name,
        exchange=exchange,
        kill_switch_pct=99.0,
        kill_switch_window_hours=ks_window_hours,
    )
    result = await backtester.run(start=start_dt, end=end_dt, db_path=db_path)
    return result

result = asyncio.run(_run())

# Calculs métriques
from scripts.stress_test_leverage import _compute_calmar, _compute_sharpe, _count_ks_triggers, KS_THRESHOLDS

calmar = _compute_calmar(result.total_return_pct, result.max_drawdown_pct)
sharpe = _compute_sharpe(result.snapshots)
ks_counts = {thr: _count_ks_triggers(result.snapshots, thr, ks_window_hours) for thr in KS_THRESHOLDS}

row = {
    "total_return_pct": round(result.total_return_pct, 1),
    "max_drawdown_pct": round(result.max_drawdown_pct, 1),
    "calmar": calmar if calmar != float("inf") else 999.0,
    "ks_30": ks_counts[30], "ks_45": ks_counts[45], "ks_60": ks_counts[60],
    "worst_case_sl_pct": round(result.worst_case_sl_loss_pct, 1),
    "min_liq_dist_pct": round(result.min_liquidation_distance_pct, 1),
    "was_liquidated": result.was_liquidated,
    "funding_total": round(result.funding_paid_total, 2),
    "total_trades": result.total_trades,
    "win_rate": round(result.win_rate, 1),
    "sharpe": sharpe,
}
print("__RESULT_JSON__" + json.dumps(row))
'''

MAX_RETRIES = 2


def _run_single(
    strategy_name: str,
    leverage: int,
    resolved_days: int,
    start_dt: datetime,
    end_dt: datetime,
    capital: float,
    db_path: str,
    exchange: str,
) -> dict | None:
    """Lance un portfolio backtest dans un subprocess isolé.

    Chaque run tourne dans un process Python séparé pour survivre aux
    segfaults aléatoires (numpy + CPython 3.13 heap corruption).
    Retry automatique jusqu'à MAX_RETRIES tentatives.
    """
    args = [
        sys.executable, "-c", _SUBPROCESS_SCRIPT,
        strategy_name,
        str(leverage),
        start_dt.isoformat(),
        end_dt.isoformat(),
        str(capital),
        db_path,
        exchange,
        str(KS_WINDOW_HOURS),
    ]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=1200,  # 20 min max par run
                cwd=os.getcwd(),
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "Timeout {}@{}x / {}j (tentative {}/{})",
                strategy_name, leverage, resolved_days, attempt, MAX_RETRIES,
            )
            continue

        if proc.returncode == 0:
            # Chercher la ligne JSON dans stdout
            for line in proc.stdout.splitlines():
                if line.startswith("__RESULT_JSON__"):
                    raw = json.loads(line[len("__RESULT_JSON__"):])
                    raw["strategy"] = strategy_name
                    raw["leverage"] = leverage
                    raw["days"] = resolved_days
                    raw["window_start"] = start_dt.date().isoformat()
                    raw["window_end"] = end_dt.date().isoformat()
                    return raw

            # Pas de JSON trouvé
            logger.error(
                "Pas de résultat JSON pour {}@{}x / {}j",
                strategy_name, leverage, resolved_days,
            )
            return None

        # Crash (segfault, etc.)
        suffix = f" (tentative {attempt}/{MAX_RETRIES})"
        if proc.stderr:
            # Afficher les dernières lignes de stderr (traceback faulthandler)
            stderr_lines = proc.stderr.strip().splitlines()
            for line in stderr_lines[-10:]:
                print(f"         | {line}")
        if attempt < MAX_RETRIES:
            logger.warning(
                "Crash {}@{}x / {}j (exit {}){} -> retry dans 3s",
                strategy_name, leverage, resolved_days,
                proc.returncode, suffix,
            )
            print(f"         -> CRASH (exit {proc.returncode}), retry {attempt + 1}/{MAX_RETRIES}...")
            time.sleep(1)  # Cooldown avant retry
        else:
            logger.error(
                "Échec définitif {}@{}x / {}j après {} tentatives (exit {})",
                strategy_name, leverage, resolved_days,
                MAX_RETRIES, proc.returncode,
            )

    return None


# ---------------------------------------------------------------------------
# Formatage rapport console
# ---------------------------------------------------------------------------

# Largeurs de colonnes (header, largeur)
_COLS = [
    ("Lev",    4),
    ("Return", 8),
    ("Max DD", 8),
    ("Calmar", 7),
    ("KS@30",  6),
    ("KS@45",  6),
    ("KS@60",  6),
    ("W-SL",   7),
    ("Liq%",   7),
    ("Fund$",  8),
    ("Trades", 7),
    ("WR",     6),
    ("Sharpe", 7),
]


def _fmt_pct(val: float, plus: bool = True) -> str:
    sign = "+" if plus and val >= 0 else ""
    return f"{sign}{val:.1f}%"


def _fmt_dollar(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.0f}$"


def _print_table(rows: list[dict], title: str) -> None:
    """Affiche un tableau formaté pour une (stratégie, fenêtre)."""
    headers = [h for h, _ in _COLS]
    widths = {h: w for h, w in _COLS}

    sep = "-+-".join("-" * w for _, w in _COLS)

    print(f"\n  === {title} ===")
    print("  " + " | ".join(h.ljust(widths[h]) for h in headers))
    print("  " + sep)

    for r in rows:
        if r is None:
            print("  [ERREUR - run ignore]")
            continue

        liq_str = "LIQD!" if r["was_liquidated"] else f"{r['min_liq_dist_pct']:.0f}%"
        calmar_str = f"{r['calmar']:.2f}" if r["calmar"] < 900 else "inf"

        vals = [
            f"{r['leverage']}x",
            _fmt_pct(r["total_return_pct"]),
            _fmt_pct(r["max_drawdown_pct"], plus=False),
            calmar_str,
            str(r["ks_30"]),
            str(r["ks_45"]),
            str(r["ks_60"]),
            f"{r['worst_case_sl_pct']:.0f}%",
            liq_str,
            _fmt_dollar(r["funding_total"]),
            str(r["total_trades"]),
            f"{r['win_rate']:.0f}%",
            f"{r['sharpe']:.2f}",
        ]

        line = " | ".join(v.ljust(widths[h]) for h, v in zip(headers, vals))
        print("  " + line)


# ---------------------------------------------------------------------------
# Sauvegarde CSV
# ---------------------------------------------------------------------------


def _save_csv(rows: list[dict], path: str, append: bool = False) -> None:
    """Sauvegarde les résultats en CSV.

    En mode append (--append), écrit à la suite du fichier existant sans
    réécrire le header. En mode normal, écrase le fichier.
    """
    if not rows:
        return

    mode = "a" if append and os.path.exists(path) else "w"
    write_header = mode == "w"
    fieldnames = list(rows[0].keys())

    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

    action = "Ajouté au" if mode == "a" else "Sauvegardé dans"
    print(f"\n  CSV {action} : {path} ({len(rows)} lignes)")


def _load_csv(path: str) -> list[dict]:
    """Charge toutes les lignes d'un CSV de résultats existant.

    Utilisé en mode --append pour construire le rapport complet depuis
    plusieurs lots de runs.
    """
    if not os.path.exists(path):
        return []

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["leverage"] = int(row["leverage"])
                row["days"] = int(row["days"])
                row["total_return_pct"] = float(row["total_return_pct"])
                row["max_drawdown_pct"] = float(row["max_drawdown_pct"])
                row["calmar"] = float(row["calmar"])
                row["ks_30"] = int(row["ks_30"])
                row["ks_45"] = int(row["ks_45"])
                row["ks_60"] = int(row["ks_60"])
                row["worst_case_sl_pct"] = float(row["worst_case_sl_pct"])
                row["min_liq_dist_pct"] = float(row["min_liq_dist_pct"])
                row["was_liquidated"] = row["was_liquidated"].lower() in ("true", "1", "yes")
                row["funding_total"] = float(row["funding_total"])
                row["total_trades"] = int(row["total_trades"])
                row["win_rate"] = float(row["win_rate"])
                row["sharpe"] = float(row["sharpe"])
                rows.append(row)
            except (KeyError, ValueError):
                pass  # Ligne corrompue, on ignore

    return rows


# ---------------------------------------------------------------------------
# Recommandation automatique
# ---------------------------------------------------------------------------


def _recommend(
    all_rows: list[dict],
    strategies: list[str],
    auto_days: dict[str, int],
) -> None:
    """Affiche les recommandations par stratégie selon les critères de sélection."""
    print("\n" + "=" * 70)
    print("  RECOMMANDATION")
    print("=" * 70)
    print(f"  Critères : Liq% > {MIN_LIQ_DIST_PCT:.0f}%  |  Max DD > -{MAX_DD_PCT:.0f}%  |  KS@45=0 (fenêtre longue)")
    print()

    elimination_notes: list[str] = []

    for strat in strategies:
        strat_rows = [r for r in all_rows if r["strategy"] == strat]
        if not strat_rows:
            print(f"  {strat:20s} : pas de données")
            continue

        leverages = sorted(set(r["leverage"] for r in strat_rows))
        long_window = auto_days.get(strat, 0)

        def is_valid(lev: int) -> bool:
            lev_rows = [r for r in strat_rows if r["leverage"] == lev]
            if not lev_rows:
                return False

            for r in lev_rows:
                label = f"{r['days']}j"

                if r["was_liquidated"]:
                    elimination_notes.append(
                        f"  - {strat} {lev}x éliminé : LIQUIDÉ ({label})"
                    )
                    return False

                if r["min_liq_dist_pct"] <= MIN_LIQ_DIST_PCT:
                    elimination_notes.append(
                        f"  - {strat} {lev}x éliminé : Liq dist {r['min_liq_dist_pct']:.0f}%"
                        f" ≤ {MIN_LIQ_DIST_PCT:.0f}% ({label})"
                    )
                    return False

                if r["max_drawdown_pct"] < -MAX_DD_PCT:
                    elimination_notes.append(
                        f"  - {strat} {lev}x éliminé : Max DD {r['max_drawdown_pct']:.1f}%"
                        f" < -{MAX_DD_PCT:.0f}% ({label})"
                    )
                    return False

                # KS@45 = 0 requis sur la fenêtre longue (auto)
                if r["days"] == long_window and r["ks_45"] > 0:
                    elimination_notes.append(
                        f"  - {strat} {lev}x éliminé : {r['ks_45']} KS@45%"
                        f" sur fenêtre longue ({label})"
                    )
                    return False

            return True

        valid_leverages = [lev for lev in leverages if is_valid(lev)]

        if not valid_leverages:
            print(f"  {strat:20s} : AUCUN leverage valide (voir notes)")
            print(f"  {'':20s}   (Assouplir les critères ou tester des leverages plus bas)")
            continue

        # Parmi les valides, choisir le meilleur Calmar sur la fenêtre longue
        def calmar_for_lev(lev: int) -> float:
            row = next(
                (r for r in strat_rows if r["leverage"] == lev and r["days"] == long_window),
                None,
            )
            return row["calmar"] if row else -999.0

        best_lev = max(valid_leverages, key=calmar_for_lev)
        best = next(
            (r for r in strat_rows if r["leverage"] == best_lev and r["days"] == long_window),
            None,
        )

        if best:
            print(
                f"  {strat:20s} : {best_lev}x  "
                f"(Return {_fmt_pct(best['total_return_pct'])}, "
                f"Max DD {best['max_drawdown_pct']:.1f}%, "
                f"Calmar {best['calmar']:.2f}, "
                f"Sharpe {best['sharpe']:.2f})"
            )
        else:
            print(f"  {strat:20s} : {best_lev}x (données fenêtre longue manquantes)")

        # Suggestion test intermédiaire si deux candidats valides sont adjacents (gap=2)
        sorted_valid = sorted(valid_leverages)
        for i in range(len(sorted_valid) - 1):
            a, b = sorted_valid[i], sorted_valid[i + 1]
            if b - a == 2:
                elimination_notes.append(
                    f"  -> Affiner {strat} : tester {a + 1}x"
                    f" (entre {a}x et {b}x, tous deux valides)"
                )

    if elimination_notes:
        print()
        print("  Notes :")
        for note in elimination_notes:
            print(note)

    print("=" * 70)


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


async def main() -> None:
    try:
        await _main_inner()
    except Exception as e:
        print(f"\nERREUR FATALE: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


async def _main_inner() -> None:
    parser = argparse.ArgumentParser(
        description="Stress test leverage multi-fenêtre (portfolio backtest)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  uv run python -m scripts.stress_test_leverage
  uv run python -m scripts.stress_test_leverage --strategy grid_boltrend
  uv run python -m scripts.stress_test_leverage --leverages 5,7
  uv run python -m scripts.stress_test_leverage --days 180
  uv run python -m scripts.stress_test_leverage --strategy grid_boltrend --leverages 3,5 --days 90
        """,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Stratégie unique (défaut: toutes — grid_atr + grid_boltrend)",
    )
    parser.add_argument(
        "--leverages",
        type=str,
        default=None,
        help="Leverages séparés par virgule, ex: 2,4,6,8 (défaut: matrice standard)",
    )
    parser.add_argument(
        "--days",
        type=str,
        default=None,
        help="Fenêtre unique : nombre de jours ou 'auto' (défaut: toutes les fenêtres)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1_000.0,
        help="Capital initial en $ (défaut: 1000)",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="binance",
        help="Source des candles (défaut: binance)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/scalp_radar.db",
        help="Chemin de la DB (défaut: data/scalp_radar.db)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/stress_test_results.csv",
        help="Fichier CSV de sortie (défaut: data/stress_test_results.csv)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Ajouter au CSV existant au lieu de l'écraser. "
            "Le rapport final est généré depuis l'intégralité du CSV "
            "(lots précédents + lot actuel)."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Sauter les runs déjà présents dans le CSV (auto-resume après crash).",
    )

    args = parser.parse_args()

    # Silence les logs du moteur (on gère notre propre progression)
    setup_logging(level="WARNING")

    config = get_config()

    # --- Construire la matrice de runs ---
    matrix: dict[str, dict[str, Any]] = {k: dict(v) for k, v in DEFAULT_MATRIX.items()}

    if args.strategy:
        if args.strategy not in matrix:
            print(f"Stratégie inconnue : '{args.strategy}'")
            print(f"Stratégies disponibles : {', '.join(matrix.keys())}")
            sys.exit(1)
        matrix = {args.strategy: matrix[args.strategy]}

    if args.leverages:
        try:
            leverages_override = [int(x.strip()) for x in args.leverages.split(",") if x.strip()]
        except ValueError:
            print(f"Format leverages invalide : '{args.leverages}' (ex: 2,4,6,8)")
            sys.exit(1)
        for k in matrix:
            matrix[k] = {**matrix[k], "leverages": leverages_override}

    if args.days:
        if args.days == "auto":
            days_override: list[str | int] = ["auto"]
        else:
            try:
                days_override = [int(args.days)]
            except ValueError:
                print(f"Format days invalide : '{args.days}' (entier ou 'auto')")
                sys.exit(1)
        for k in matrix:
            matrix[k] = {**matrix[k], "days": days_override}

    # --- Détecter l'historique disponible pour les fenêtres "auto" ---
    auto_days: dict[str, int] = {}
    auto_start_dates: dict[str, datetime | None] = {}

    needs_auto = [s for s, cfg in matrix.items() if "auto" in cfg["days"]]
    if needs_auto:
        print("\n  Détection historique disponible...")
        for strat_name in needs_auto:
            d, start_dt = await _detect_auto_days(
                config, strat_name, args.exchange, args.db
            )
            auto_days[strat_name] = d
            auto_start_dates[strat_name] = start_dt
            start_str = start_dt.date().isoformat() if start_dt else "N/A"
            print(f"  {strat_name:20s} : {d}j (depuis {start_str})")

    # Remplir auto_days pour les stratégies sans fenêtre "auto"
    for strat_name in matrix:
        if strat_name not in auto_days:
            auto_days[strat_name] = 0

    # --- Construire la liste ordonnée des runs ---
    runs: list[tuple[str, int, int | str]] = []
    for strat_name, cfg in matrix.items():
        for lev in cfg["leverages"]:
            for days_val in cfg["days"]:
                runs.append((strat_name, lev, days_val))

    total_runs = len(runs)

    # Résoudre les jours pour chaque run
    end_dt = datetime.now(timezone.utc)
    run_specs: list[tuple[str, int, int, datetime, datetime]] = []
    for strat_name, lev, days_val in runs:
        if days_val == "auto":
            resolved = auto_days.get(strat_name, 90)
        else:
            resolved = int(days_val)
        start_dt = end_dt - timedelta(days=resolved)
        run_specs.append((strat_name, lev, resolved, start_dt, end_dt))

    # --- Charger les runs existants si --skip-existing ---
    existing_keys: set[tuple[str, int, int]] = set()
    if args.skip_existing:
        existing_rows = _load_csv(args.output)
        existing_keys = {(r["strategy"], r["leverage"], r["days"]) for r in existing_rows}
        if existing_keys:
            print(f"\n  --skip-existing : {len(existing_keys)} runs déjà en CSV")

    # --- Afficher le plan ---
    skippable = sum(1 for s, l, d, _, _ in run_specs if (s, l, d) in existing_keys)
    actual_runs = total_runs - skippable
    print(f"\n  {total_runs} runs planifies | {actual_runs} à lancer | capital={args.capital:,.0f}$ | KS=DESACTIVE(99%)")
    print(f"  Mode subprocess isolé (retry auto {MAX_RETRIES}x par run)")
    print()

    # --- Exécution ---
    all_rows: list[dict] = []
    t_global = time.monotonic()
    completed = 0

    for run_idx, (strat_name, lev, resolved_days, start_dt, end_dt_run) in enumerate(
        run_specs, 1
    ):
        # Skip si déjà dans le CSV
        if (strat_name, lev, resolved_days) in existing_keys:
            print(f"  [{run_idx:2d}/{total_runs}] {strat_name} @ {lev}x / {resolved_days}j  SKIP (déjà en CSV)")
            continue

        # Calcul ETA
        elapsed = time.monotonic() - t_global
        if completed > 0 and elapsed > 0:
            avg = elapsed / completed
            remaining = avg * (actual_runs - completed)
            eta_str = f" - ETA ~{remaining:.0f}s"
        else:
            eta_str = ""

        print(
            f"  [{run_idx:2d}/{total_runs}] {strat_name} @ {lev}x / "
            f"{resolved_days}j  ({start_dt.date()} -> {end_dt_run.date()})"
            f"{eta_str}",
            flush=True,
        )

        row = _run_single(
            strategy_name=strat_name,
            leverage=lev,
            resolved_days=resolved_days,
            start_dt=start_dt,
            end_dt=end_dt_run,
            capital=args.capital,
            db_path=args.db,
            exchange=args.exchange,
        )

        completed += 1

        if row is not None:
            all_rows.append(row)
            ret_str = _fmt_pct(row["total_return_pct"])
            dd_str = _fmt_pct(row["max_drawdown_pct"], plus=False)
            liq_str = "LIQD!" if row["was_liquidated"] else f"Liq={row['min_liq_dist_pct']:.0f}%"
            print(
                f"         -> Return={ret_str}  DD={dd_str}  "
                f"Calmar={row['calmar']:.2f}  {liq_str}  "
                f"Trades={row['total_trades']}"
            )
        else:
            print(f"         -> ERREUR (run ignore apres {MAX_RETRIES} tentatives)")

    total_elapsed = time.monotonic() - t_global
    print(f"\n  Terminé en {total_elapsed:.0f}s ({total_runs} runs)")

    if not all_rows:
        print("\n  Aucun résultat — vérifier que les données sont en DB.")
        print("  Commande : uv run python -m scripts.fetch_history")
        return

    # --- Sauvegarde CSV (avant le rapport, pour ne pas perdre les données) ---
    _save_csv(all_rows, args.output, append=args.append)

    # --- En mode --append : charger l'intégralité du CSV pour le rapport ---
    if args.append:
        report_rows = _load_csv(args.output)
        if not report_rows:
            report_rows = all_rows
        report_strategies = sorted(set(r["strategy"] for r in report_rows))
        # Compléter auto_days pour les stratégies des lots précédents
        for strat in report_strategies:
            if strat not in auto_days:
                # Utiliser la fenêtre la plus longue trouvée dans le CSV comme référence
                auto_days[strat] = max(
                    (r["days"] for r in report_rows if r["strategy"] == strat),
                    default=0,
                )
        batch_note = f" | Lot actuel : {len(all_rows)} runs  |  Total CSV : {len(report_rows)} runs"
    else:
        report_rows = all_rows
        report_strategies = list(matrix.keys())
        batch_note = ""

    # --- Rapport comparatif ---
    print("\n\n" + "=" * 70)
    print("  STRESS TEST LEVERAGE — RAPPORT COMPARATIF")
    print("=" * 70)
    print(f"  Capital : {args.capital:,.0f}$  |  Kill switch : DESACTIVE (analyse a posteriori){batch_note}")
    print(f"  Seuils KS analysés : {', '.join(str(t) + '%' for t in KS_THRESHOLDS)}")

    for strat_name in report_strategies:
        strat_rows = [r for r in report_rows if r["strategy"] == strat_name]
        if not strat_rows:
            continue

        # Trier les fenêtres : longue en premier, puis décroissant
        windows_in_data = sorted(set(r["days"] for r in strat_rows), reverse=True)

        for window_days in windows_in_data:
            win_rows = sorted(
                [r for r in strat_rows if r["days"] == window_days],
                key=lambda r: r["leverage"],
            )
            if not win_rows:
                continue

            sample = win_rows[0]
            is_auto = window_days == auto_days.get(strat_name, 0)
            label = f"Auto ({window_days}j)" if is_auto else f"{window_days}j"
            title = f"{strat_name} - {label}  [{sample['window_start']} -> {sample['window_end']}]"

            _print_table(win_rows, title)

    _recommend(report_rows, report_strategies, auto_days)


if __name__ == "__main__":
    asyncio.run(main())
