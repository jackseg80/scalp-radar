"""BTC Regime Detector Analysis — Sprint 50a.

Orchestre les 3 detecteurs, grid search, evaluation, et genere le rapport.

Usage:
    uv run python -m scripts.regime_analysis
    uv run python -m scripts.regime_analysis --detector sma_stress
    uv run python -m scripts.regime_analysis --skip-plots
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    sys.exit("pandas et matplotlib requis. Installer: uv sync --group analysis")

from tqdm import tqdm

from scripts.regime_detectors import (
    ALL_DETECTORS,
    LABELS,
    BaseDetector,
    DetectorResult,
    accuracy,
    avg_regime_duration,
    confusion_matrix_manual,
    crash_detection_delay,
    f1_per_class,
    n_transitions,
    regime_distribution,
    resample_4h_to_daily,
    stability_score,
)

REPORT_PATH = Path("docs/regime_detector_report.md")
IMAGES_DIR = Path("docs/images")
REGIME_COLORS = {
    "bull": "#2ecc71",
    "bear": "#e74c3c",
    "range": "#95a5a6",
    "crash": "#8e44ad",
}


# ─── Chargement des donnees ───────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge le CSV labele et calcule le daily."""
    df_4h = pd.read_csv("data/btc_4h_labeled.csv")
    df_daily = resample_4h_to_daily(df_4h)
    return df_4h, df_daily


# ─── Evaluation ───────────────────────────────────────────────────────────

def evaluate_single(
    detector: BaseDetector,
    df_4h: pd.DataFrame,
    df_daily: pd.DataFrame,
    y_true: list[str],
    params: dict,
) -> dict:
    """Run un detecteur avec params et calcule toutes les metriques."""
    result = detector.run(df_4h, df_daily, **params)

    # Aligner sur la meme longueur
    min_len = min(len(y_true), len(result.labels_4h))
    yt = y_true[:min_len]
    yp = result.labels_4h[:min_len]

    # Exclure warmup
    warmup = result.warmup_end_idx
    yt_eval = yt[warmup:]
    yp_eval = yp[warmup:]

    if not yt_eval:
        return {
            "params": result.params,
            "accuracy": 0.0,
            "f1": {r: 0.0 for r in LABELS},
            "macro_f1": 0.0,
            "crash_analysis": {"n_crashes_gt": 0, "n_detected": 0,
                               "avg_delay_hours": float("inf"),
                               "max_delay_hours": float("inf"),
                               "delays_hours": [], "false_positives": 0},
            "n_transitions": 0,
            "avg_duration": {r: 0.0 for r in LABELS},
            "stability": 0.0,
            "distribution": {r: 0.0 for r in LABELS},
            "warmup_end_idx": warmup,
            "n_candles_eval": 0,
        }

    f1 = f1_per_class(yt_eval, yp_eval, LABELS)
    macro_f1 = sum(f1.values()) / len(f1)

    return {
        "params": result.params,
        "accuracy": accuracy(yt_eval, yp_eval),
        "f1": f1,
        "macro_f1": macro_f1,
        "crash_analysis": crash_detection_delay(yt_eval, yp_eval),
        "n_transitions": n_transitions(yp_eval),
        "avg_duration": avg_regime_duration(yp_eval),
        "stability": stability_score(yp_eval),
        "distribution": regime_distribution(yp_eval),
        "warmup_end_idx": warmup,
        "n_candles_eval": len(yt_eval),
        "_result": result,  # garde pour les plots
    }


def grid_search(
    detector: BaseDetector,
    df_4h: pd.DataFrame,
    df_daily: pd.DataFrame,
    y_true: list[str],
) -> list[dict]:
    """Evalue toutes les combinaisons de parametres. Trie par macro F1."""
    grid = detector.param_grid()
    results = []
    for params in tqdm(grid, desc=f"Grid {detector.name}", leave=False):
        metrics = evaluate_single(detector, df_4h, df_daily, y_true, params)
        results.append(metrics)
    results.sort(key=lambda r: r["macro_f1"], reverse=True)
    return results


# ─── Robustesse ───────────────────────────────────────────────────────────

def robustness_check(
    detector: BaseDetector,
    df_4h: pd.DataFrame,
    df_daily: pd.DataFrame,
    y_true: list[str],
    best_result: dict,
) -> dict:
    """Verifie que les voisins (+-1 step) donnent un F1 similaire."""
    best_params = best_result["params"]
    best_f1 = best_result["macro_f1"]
    grid = detector.param_grid()

    # Trouver les voisins : configs qui different par exactement 1 parametre
    neighbors = []
    for params in grid:
        diff_count = sum(1 for k in best_params if k in params and params[k] != best_params[k])
        same_count = sum(1 for k in best_params if k in params and params[k] == best_params[k])
        if diff_count == 1 and same_count == len(best_params) - 1:
            neighbors.append(params)

    if not neighbors:
        return {
            "robust": False,
            "reason": "Aucun voisin trouve",
            "n_neighbors": 0,
            "delta_pct": 100.0,
            "best_f1": best_f1,
            "avg_neighbor_f1": 0.0,
            "warning": "NO NEIGHBORS",
        }

    neighbor_f1s = []
    for params in tqdm(neighbors, desc="Robustesse", leave=False):
        metrics = evaluate_single(detector, df_4h, df_daily, y_true, params)
        neighbor_f1s.append(metrics["macro_f1"])

    avg_f1 = sum(neighbor_f1s) / len(neighbor_f1s)
    delta_pct = abs(best_f1 - avg_f1) / best_f1 * 100 if best_f1 > 0 else 100

    return {
        "robust": delta_pct < 5.0,
        "delta_pct": delta_pct,
        "best_f1": best_f1,
        "avg_neighbor_f1": avg_f1,
        "min_neighbor_f1": min(neighbor_f1s),
        "max_neighbor_f1": max(neighbor_f1s),
        "n_neighbors": len(neighbors),
        "warning": (f"ISOLATED PEAK: F1 varie de {delta_pct:.1f}% chez les voisins"
                    if delta_pct >= 5.0 else ""),
    }


# ─── Plots ────────────────────────────────────────────────────────────────

def _plot_regime_bands(ax, dates, labels, alpha=0.2):
    """Dessine les bandes colorees de regime sur un axe."""
    i = 0
    while i < len(labels):
        regime = labels[i]
        j = i
        while j < len(labels) and labels[j] == regime:
            j += 1
        color = REGIME_COLORS.get(regime, "#cccccc")
        ax.axvspan(dates.iloc[i], dates.iloc[min(j, len(dates) - 1)],
                    alpha=alpha, color=color, linewidth=0)
        i = j


def generate_detector_plot(
    detector_name: str,
    best_result: dict,
    df_4h: pd.DataFrame,
    y_true: list[str],
) -> str:
    """Genere le plot timeline pour un detecteur. Retourne le chemin."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = IMAGES_DIR / f"regime_{detector_name}_best.png"

    result: DetectorResult = best_result.get("_result")
    if result is None:
        return str(filepath)

    dates = pd.to_datetime(df_4h["timestamp_utc"])
    close = df_4h["close"].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    # Top : Ground truth
    _plot_regime_bands(ax1, dates, y_true)
    ax1.plot(dates, close, color="black", linewidth=0.5, alpha=0.8)
    ax1.set_yscale("log")
    ax1.set_title("Ground Truth")
    ax1.set_ylabel("BTC Prix (log)")

    # Bottom : Detection
    y_pred = result.labels_4h[:len(dates)]
    _plot_regime_bands(ax2, dates, y_pred)
    ax2.plot(dates, close, color="black", linewidth=0.5, alpha=0.8)
    ax2.set_yscale("log")
    ax2.set_title(f"Detecteur: {detector_name} (F1={best_result['macro_f1']:.3f})")
    ax2.set_ylabel("BTC Prix (log)")

    # Legende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=REGIME_COLORS[r], alpha=0.3, label=r.capitalize())
        for r in ["bull", "bear", "range", "crash"]
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    return str(filepath)


# ─── Generation du rapport ────────────────────────────────────────────────

def _format_confusion_matrix(cm: dict[str, dict[str, int]]) -> list[str]:
    """Formate la confusion matrix en markdown."""
    lines = []
    header = "| True \\ Pred | " + " | ".join(r.capitalize() for r in LABELS) + " |"
    sep = "|" + "|".join(["---"] * (len(LABELS) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for true_label in LABELS:
        row_vals = " | ".join(str(cm[true_label].get(p, 0)) for p in LABELS)
        lines.append(f"| **{true_label.capitalize()}** | {row_vals} |")
    return lines


def generate_report(
    all_results: dict[str, list[dict]],
    robustness_results: dict[str, dict],
    df_4h: pd.DataFrame,
) -> None:
    """Genere le rapport markdown."""
    lines = [
        "# BTC Regime Detector Calibration Report — Sprint 50a",
        "",
        f"*Genere automatiquement le {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        f"**Donnees** : {len(df_4h)} candles 4h BTC/USDT",
        "",
    ]

    # --- Resume executif ---
    lines.append("## 1. Resume executif")
    lines.append("")

    # Trouver le meilleur
    best_overall = None
    best_det_name = ""
    for name, results in all_results.items():
        if results and (best_overall is None or results[0]["macro_f1"] > best_overall["macro_f1"]):
            best_overall = results[0]
            best_det_name = name

    if best_overall:
        rob = robustness_results.get(best_det_name, {})
        rob_status = "ROBUSTE" if rob.get("robust", False) else "ATTENTION (pic isole)"
        lines.append(
            f"**Detecteur recommande** : `{best_det_name}` "
            f"(Macro F1 = {best_overall['macro_f1']:.3f}, "
            f"Crash F1 = {best_overall['f1']['crash']:.3f}, "
            f"Robustesse = {rob_status})"
        )
        lines.append("")
        lines.append(f"Parametres optimaux : `{json.dumps(best_overall['params'], indent=None)}`")
        lines.append("")

    # --- Tableau comparatif ---
    lines.append("## 2. Tableau comparatif (top 3 par detecteur)")
    lines.append("")
    lines.append("| Detecteur | Rang | Macro F1 | Accuracy | Crash F1 | Delay moy | Transitions | Stable | Robuste |")
    lines.append("|-----------|------|----------|----------|----------|-----------|-------------|--------|---------|")

    for name, results in all_results.items():
        rob = robustness_results.get(name, {})
        for rank, r in enumerate(results[:3], 1):
            crash_delay = r["crash_analysis"]["avg_delay_hours"]
            delay_str = f"{crash_delay:.0f}h" if crash_delay != float("inf") else "N/A"
            rob_str = "OUI" if rob.get("robust", False) else "NON"
            lines.append(
                f"| {name} | {rank} | {r['macro_f1']:.3f} | "
                f"{r['accuracy']:.3f} | {r['f1']['crash']:.3f} | "
                f"{delay_str} | {r['n_transitions']} | "
                f"{r['stability']:.2f} | {rob_str if rank == 1 else '-'} |"
            )

    lines.append("")

    # --- Par detecteur ---
    for name, results in all_results.items():
        lines.append(f"## 3. Detecteur : {name}")
        lines.append("")

        if not results:
            lines.append("*Aucun resultat*")
            continue

        best = results[0]
        lines.append(f"### Meilleure configuration")
        lines.append("")
        lines.append(f"```json\n{json.dumps(best['params'], indent=2)}\n```")
        lines.append("")
        lines.append(f"- Macro F1 : {best['macro_f1']:.4f}")
        lines.append(f"- Accuracy : {best['accuracy']:.4f}")
        lines.append(f"- Candles evaluees : {best.get('n_candles_eval', '?')} (apres warmup)")
        lines.append(f"- Transitions : {best['n_transitions']}")
        lines.append(f"- Stabilite : {best['stability']:.3f}")
        lines.append("")

        # F1 par regime
        lines.append("**F1 par regime :**")
        lines.append("")
        for regime in LABELS:
            lines.append(f"- {regime.capitalize()} : {best['f1'][regime]:.3f}")
        lines.append("")

        # Distribution
        lines.append("**Distribution :**")
        lines.append("")
        for regime in LABELS:
            pct = best["distribution"].get(regime, 0) * 100
            lines.append(f"- {regime.capitalize()} : {pct:.1f}%")
        lines.append("")

        # Confusion matrix
        # Recalculer proprement
        if "_result" in best:
            result = best["_result"]
            y_true = df_4h["regime_label"].tolist()
            warmup = result.warmup_end_idx
            min_len = min(len(y_true), len(result.labels_4h))
            yt = y_true[warmup:min_len]
            yp = result.labels_4h[warmup:min_len]
            cm = confusion_matrix_manual(yt, yp, LABELS)
            lines.append("### Confusion Matrix")
            lines.append("")
            lines.extend(_format_confusion_matrix(cm))
            lines.append("")

        # Timeline plot
        lines.append(f"### Timeline")
        lines.append("")
        lines.append(f"![{name}](../docs/images/regime_{name}_best.png)")
        lines.append("")

    # --- Analyse des crashs ---
    lines.append("## 4. Analyse des crashs")
    lines.append("")
    lines.append("| Detecteur | Crashs GT | Detectes | Delay moy | Delay max | Faux positifs |")
    lines.append("|-----------|-----------|----------|-----------|-----------|---------------|")

    for name, results in all_results.items():
        if not results:
            continue
        ca = results[0]["crash_analysis"]
        avg_d = f"{ca['avg_delay_hours']:.0f}h" if ca["avg_delay_hours"] != float("inf") else "N/A"
        max_d = f"{ca['max_delay_hours']:.0f}h" if ca["max_delay_hours"] != float("inf") else "N/A"
        lines.append(
            f"| {name} | {ca['n_crashes_gt']} | {ca['n_detected']} | "
            f"{avg_d} | {max_d} | {ca['false_positives']} |"
        )

    lines.append("")

    # Detail par crash
    if best_overall and "_result" in best_overall:
        ca = best_overall["crash_analysis"]
        if ca["delays_hours"]:
            lines.append(f"**Detail pour {best_det_name} (meilleur detecteur) :**")
            lines.append("")
            for i, delay in enumerate(ca["delays_hours"]):
                delay_str = f"{delay:.0f}h" if delay != float("inf") else "NON DETECTE"
                lines.append(f"- Crash #{i + 1} : delai = {delay_str}")
            lines.append("")

    # --- Robustesse ---
    lines.append("## 5. Analyse de robustesse")
    lines.append("")

    for name, rob in robustness_results.items():
        status = "ROBUSTE" if rob.get("robust", False) else "WARNING: PIC ISOLE"
        lines.append(f"### {name} : {status}")
        lines.append("")
        lines.append(f"- Best F1 : {rob['best_f1']:.4f}")
        lines.append(f"- Avg voisins F1 : {rob['avg_neighbor_f1']:.4f} (delta: {rob['delta_pct']:.1f}%)")
        lines.append(f"- Voisins testes : {rob['n_neighbors']}")
        if rob.get("warning"):
            lines.append(f"- **{rob['warning']}**")
        lines.append("")

    # --- Recommandation ---
    lines.append("## 6. Recommandation")
    lines.append("")
    if best_overall:
        lines.append(
            f"Le detecteur `{best_det_name}` est recommande avec les parametres "
            f"ci-dessus. "
        )
        rob = robustness_results.get(best_det_name, {})
        if rob.get("robust"):
            lines.append("La configuration est robuste (voisins dans +-5% F1).")
        else:
            lines.append(
                "**ATTENTION** : la configuration optimale est un pic isole. "
                "Les seuils devront etre valides en forward test avant mise en production."
            )
        lines.append("")
        lines.append(
            "Limitation Detecteur 3 : sma_fast=50, sma_slow=200 fixes. "
            "Si retenu, explorer d'autres MAs en Sprint 50b."
        )
    lines.append("")

    # Ecriture
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapport genere : {REPORT_PATH}")


# ─── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Analyse des detecteurs de regime BTC (Sprint 50a)",
    )
    parser.add_argument(
        "--detector",
        choices=["sma_stress", "ema_atr", "multi_ma_vol"],
        help="Filtrer un seul detecteur",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Ne pas generer les graphiques",
    )
    args = parser.parse_args()

    print("=== BTC Regime Detector Analysis — Sprint 50a ===\n")

    # Charger les donnees
    df_4h, df_daily = load_data()
    y_true = df_4h["regime_label"].tolist()
    print(f"Donnees : {len(df_4h)} candles 4h, {len(df_daily)} jours daily")
    print(f"Ground truth : {len(set(y_true))} regimes\n")

    # Filtrer les detecteurs
    detectors = ALL_DETECTORS
    if args.detector:
        detectors = [d for d in ALL_DETECTORS if d.name == args.detector]

    all_results: dict[str, list[dict]] = {}
    robustness_results: dict[str, dict] = {}

    for detector in detectors:
        grid_size = len(detector.param_grid())
        print(f"{'=' * 60}")
        print(f"Detecteur : {detector.name} ({grid_size} combinaisons)")

        t0 = time.time()
        results = grid_search(detector, df_4h, df_daily, y_true)
        elapsed = time.time() - t0
        print(f"Grid search : {elapsed:.1f}s")

        all_results[detector.name] = results

        # Afficher top 3
        for i, r in enumerate(results[:3]):
            crash_delay = r["crash_analysis"]["avg_delay_hours"]
            delay_str = f"{crash_delay:.0f}h" if crash_delay != float("inf") else "N/A"
            print(
                f"  #{i + 1} F1={r['macro_f1']:.3f} | "
                f"Acc={r['accuracy']:.3f} | "
                f"Crash F1={r['f1']['crash']:.3f} | "
                f"Delay={delay_str} | "
                f"Trans={r['n_transitions']}"
            )

        # Robustesse
        if results:
            print(f"\nAnalyse de robustesse...")
            rob = robustness_check(detector, df_4h, df_daily, y_true, results[0])
            robustness_results[detector.name] = rob
            status = "ROBUSTE" if rob["robust"] else f"WARNING ({rob['delta_pct']:.1f}% delta)"
            print(f"  Robustesse : {status} ({rob['n_neighbors']} voisins)")

        # Plot
        if not args.skip_plots and results:
            print(f"Generation du plot...")
            path = generate_detector_plot(detector.name, results[0], df_4h, y_true)
            print(f"  -> {path}")

        print()

    # Rapport
    generate_report(all_results, robustness_results, df_4h)

    # Resume final
    print("\n" + "=" * 60)
    print("RESUME FINAL")
    for name, results in all_results.items():
        if results:
            r = results[0]
            rob = robustness_results.get(name, {})
            rob_str = "ROBUSTE" if rob.get("robust") else "PIC ISOLE"
            print(f"  {name:15} F1={r['macro_f1']:.3f}  Crash={r['f1']['crash']:.3f}  {rob_str}")

    print(f"\nRapport : {REPORT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
