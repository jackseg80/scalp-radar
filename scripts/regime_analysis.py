"""BTC Regime Detector Analysis — Sprint 50a / 50a-bis.

Orchestre les 3 detecteurs, grid search, evaluation, et genere le rapport.

Usage:
    # Sprint 50a (4 classes)
    uv run python -m scripts.regime_analysis
    uv run python -m scripts.regime_analysis --detector sma_stress
    uv run python -m scripts.regime_analysis --skip-plots

    # Sprint 50a-bis (binaire + stress)
    uv run python -m scripts.regime_analysis --binary
    uv run python -m scripts.regime_analysis --stress-only
    uv run python -m scripts.regime_analysis --binary --stress
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import time
from itertools import groupby
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
except ImportError:
    sys.exit("pandas et matplotlib requis. Installer: uv sync --group analysis")

from tqdm import tqdm

from scripts.regime_detectors import (
    ALL_DETECTORS,
    BINARY_LABELS,
    CHAR_LABEL_MAP,
    LABEL_CHAR_MAP,
    LABELS,
    BaseDetector,
    DetectorResult,
    StressIndicator4h,
    accuracy,
    avg_regime_duration,
    binary_metrics,
    confusion_matrix_manual,
    crash_detection_delay,
    f1_per_class,
    n_transitions,
    regime_distribution,
    resample_4h_to_daily,
    stability_score,
    to_binary_labels,
)

REPORT_PATH = Path("docs/regime_detector_report.md")
BINARY_REPORT_PATH = Path("docs/regime_binary_report.md")
IMAGES_DIR = Path("docs/images")
CACHE_PATH = Path("data/regime_grid_results.json")
CSV_PATH = "data/btc_4h_labeled.csv"

REGIME_COLORS = {
    "bull": "#2ecc71",
    "bear": "#e74c3c",
    "range": "#95a5a6",
    "crash": "#8e44ad",
}
BINARY_COLORS = {
    "normal": "#2ecc71",
    "defensive": "#e74c3c",
}


# ─── Chargement des donnees ───────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge le CSV labele et calcule le daily."""
    df_4h = pd.read_csv(CSV_PATH)
    df_daily = resample_4h_to_daily(df_4h)
    return df_4h, df_daily


# ─── Cache grid search (Sprint 50a-bis) ─────────────────────────────────

def _compute_data_hash(csv_path: str) -> str:
    """MD5 du fichier CSV pour invalider le cache."""
    h = hashlib.md5()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _encode_labels(labels: list[str]) -> str:
    """Encode les labels 4-classes en string compacte (b/e/r/c)."""
    return "".join(LABEL_CHAR_MAP.get(lbl, "r") for lbl in labels)


def _decode_labels(encoded: str) -> list[str]:
    """Decode une string compacte en labels 4-classes."""
    return [CHAR_LABEL_MAP.get(c, "range") for c in encoded]


def _json_default(obj):
    """Serialiseur JSON pour float inf/nan."""
    if isinstance(obj, float):
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if math.isnan(obj):
            return "NaN"
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        if math.isnan(v):
            return "NaN"
        return v
    raise TypeError(f"Not serializable: {type(obj)}")


def _walk_convert(obj):
    """Reconvertit Infinity/NaN depuis JSON."""
    if isinstance(obj, str):
        if obj == "Infinity":
            return float("inf")
        if obj == "-Infinity":
            return float("-inf")
        if obj == "NaN":
            return float("nan")
    if isinstance(obj, dict):
        return {k: _walk_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_convert(item) for item in obj]
    return obj


def save_grid_cache(
    all_results: dict[str, list[dict]],
    n_rows: int,
) -> None:
    """Persiste les resultats de grid search avec labels encodes."""
    cache = {
        "version": "50a-bis-v1",
        "generated_utc": pd.Timestamp.now("UTC").isoformat(),
        "data_file": CSV_PATH,
        "data_rows": n_rows,
        "data_hash_md5": _compute_data_hash(CSV_PATH),
        "detectors": {},
    }
    for name, results in all_results.items():
        serializable = []
        for r in results:
            entry = {k: v for k, v in r.items() if k != "_result"}
            # Encoder les labels 4h si le DetectorResult est disponible
            det_result = r.get("_result")
            if det_result is not None:
                entry["labels_4h_encoded"] = _encode_labels(det_result.labels_4h)
            serializable.append(entry)
        cache["detectors"][name] = {
            "grid_size": len(results),
            "results": serializable,
        }

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, default=_json_default, indent=1),
        encoding="utf-8",
    )
    size_mb = CACHE_PATH.stat().st_size / 1024 / 1024
    print(f"Cache sauvegarde : {CACHE_PATH} ({size_mb:.1f} MB)")


def load_grid_cache() -> dict[str, list[dict]] | None:
    """Charge le cache si valide (meme fichier, meme hash). Retourne None si invalide."""
    if not CACHE_PATH.exists():
        return None
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    current_hash = _compute_data_hash(CSV_PATH)
    if raw.get("data_hash_md5") != current_hash:
        print("Cache invalide (hash mismatch). Recalcul...")
        return None

    all_results = {}
    for name, det_data in raw.get("detectors", {}).items():
        all_results[name] = _walk_convert(det_data["results"])
    return all_results


# ─── Evaluation 4-classes (Sprint 50a) ──────────────────────────────────

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


# ─── Evaluation binaire (Sprint 50a-bis) ────────────────────────────────

def evaluate_binary(
    cached_results: dict[str, list[dict]],
    y_true: list[str],
) -> dict[str, list[dict]]:
    """Calcule les metriques binaires a partir du cache (sans re-run)."""
    y_true_bin = to_binary_labels(y_true)
    binary_results: dict[str, list[dict]] = {}

    for det_name, results in cached_results.items():
        entries = []
        for r in tqdm(results, desc=f"Binary {det_name}", leave=False):
            encoded = r.get("labels_4h_encoded")
            if not encoded:
                continue

            labels_4h = _decode_labels(encoded)
            warmup = r.get("warmup_end_idx", 0)
            min_len = min(len(y_true_bin), len(labels_4h))

            yt_eval = y_true_bin[warmup:min_len]
            yp_bin = to_binary_labels(labels_4h[:min_len])
            yp_eval = yp_bin[warmup:]

            if not yt_eval:
                continue

            bm = binary_metrics(yt_eval, yp_eval)
            entries.append({
                "params": r["params"],
                **bm,
                "original_macro_f1": r.get("macro_f1", 0.0),
                "_labels_4h_bin": yp_bin,
            })

        # Trier par F1 defensive decroissant
        entries.sort(
            key=lambda x: x["binary_f1"].get("defensive", 0),
            reverse=True,
        )
        binary_results[det_name] = entries
    return binary_results


# ─── Evaluation stress 4h (Sprint 50a-bis) ──────────────────────────────

def _load_crash_events() -> list[dict]:
    """Charge les 5 crashs depuis le YAML ground truth."""
    from scripts.regime_labeler import load_events
    events, _ = load_events("data/btc_regime_events.yaml")
    return [e for e in events if e["regime"] == "crash"]


def evaluate_stress_per_crash(
    stress_on: list[bool],
    df_4h: pd.DataFrame,
    crash_events: list[dict],
    op_drop_pct: float = -3.0,
    candle_hours: float = 4.0,
) -> list[dict]:
    """Metriques de stress par crash."""
    timestamps = pd.to_datetime(df_4h["timestamp_utc"], utc=True)
    close = df_4h["close"].values
    results = []

    for event in crash_events:
        crash_start = event["start"]
        crash_end = event["end"] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        # Trouver les indices GT
        gt_start_idx = None
        for i in range(len(timestamps)):
            if timestamps.iloc[i] >= crash_start:
                gt_start_idx = i
                break
        if gt_start_idx is None:
            results.append({"event": event["name"], "error": "not found"})
            continue

        gt_end_idx = gt_start_idx
        for i in range(gt_start_idx, len(timestamps)):
            if timestamps.iloc[i] > crash_end:
                break
            gt_end_idx = i

        # Bottom (min close dans le crash)
        crash_close = close[gt_start_idx:gt_end_idx + 1]
        bottom_idx = gt_start_idx + int(np.argmin(crash_close))

        # Detection delay : premier stress_on dans [gt_start - 12, gt_end]
        search_start = max(0, gt_start_idx - 12)
        first_stress_idx = None
        for i in range(search_start, min(gt_end_idx + 1, len(stress_on))):
            if stress_on[i]:
                first_stress_idx = i
                break

        if first_stress_idx is not None:
            detection_delay_hours = (first_stress_idx - gt_start_idx) * candle_hours
            detected_before_bottom = first_stress_idx < bottom_idx
        else:
            detection_delay_hours = float("inf")
            detected_before_bottom = False

        # Recovery delay : premier stress_off apres gt_end
        recovery_delay_hours = float("inf")
        for i in range(gt_end_idx, min(gt_end_idx + 150, len(stress_on))):
            if not stress_on[i]:
                recovery_delay_hours = (i - gt_end_idx) * candle_hours
                break

        # Delai operationnel : premier candle avec chute > op_drop_pct
        op_search_start = max(1, gt_start_idx - 6)
        operational_start_idx = None
        for i in range(op_search_start, min(gt_end_idx + 1, len(close))):
            pct_chg = (close[i] - close[i - 1]) / close[i - 1] * 100
            if pct_chg < op_drop_pct:
                operational_start_idx = i
                break

        operational_delay_hours = float("inf")
        if operational_start_idx is not None and first_stress_idx is not None:
            operational_delay_hours = (first_stress_idx - operational_start_idx) * candle_hours

        # Pre-crash false alarm (48h avant GT start)
        pre_window = max(0, gt_start_idx - 12)
        pre_crash_alarm = any(stress_on[i] for i in range(pre_window, gt_start_idx))

        results.append({
            "event": event["name"],
            "gt_start_idx": gt_start_idx,
            "gt_end_idx": gt_end_idx,
            "bottom_idx": bottom_idx,
            "detection_delay_hours": detection_delay_hours,
            "detected_before_bottom": detected_before_bottom,
            "recovery_delay_hours": recovery_delay_hours,
            "operational_start_idx": operational_start_idx,
            "operational_delay_hours": operational_delay_hours,
            "pre_crash_false_alarm_48h": pre_crash_alarm,
        })
    return results


def evaluate_stress_global(
    stress_on: list[bool],
    df_4h: pd.DataFrame,
    crash_events: list[dict],
    candle_hours: float = 4.0,
) -> dict:
    """Metriques globales du stress indicator."""
    timestamps = pd.to_datetime(df_4h["timestamp_utc"], utc=True)
    n = len(stress_on)

    # Blocs contigus de stress ON
    stress_events = []
    in_stress = False
    start_idx = 0
    for i in range(n):
        if stress_on[i] and not in_stress:
            in_stress = True
            start_idx = i
        elif not stress_on[i] and in_stress:
            in_stress = False
            stress_events.append((start_idx, i - 1))
    if in_stress:
        stress_events.append((start_idx, n - 1))

    # Identifier les faux positifs (stress hors des crashs GT)
    crash_ranges = []
    for event in crash_events:
        cs = event["start"]
        ce = event["end"] + pd.Timedelta(days=1)
        cr_start = cr_end = None
        for i in range(n):
            if cr_start is None and timestamps.iloc[i] >= cs:
                cr_start = i
            if timestamps.iloc[i] <= ce:
                cr_end = i
        if cr_start is not None and cr_end is not None:
            crash_ranges.append((cr_start, cr_end))

    false_events = []
    for start, end in stress_events:
        overlaps = any(
            not (end < cr_s or start > cr_e)
            for cr_s, cr_e in crash_ranges
        )
        if not overlaps:
            duration_h = (end - start + 1) * candle_hours
            date_str = str(timestamps.iloc[start].date())
            false_events.append({"start_idx": start, "date": date_str, "duration_h": duration_h})

    stress_count = sum(stress_on)
    total_hours = n * candle_hours
    total_years = total_hours / (365.25 * 24) if total_hours > 0 else 1.0
    durations = [(e - s + 1) * candle_hours for s, e in stress_events]

    return {
        "total_stress_hours": stress_count * candle_hours,
        "stress_pct": stress_count / n * 100 if n > 0 else 0.0,
        "n_stress_events": len(stress_events),
        "false_stress_events": len(false_events),
        "false_stress_per_year": len(false_events) / total_years,
        "avg_false_stress_duration_hours": (
            sum(fe["duration_h"] for fe in false_events) / len(false_events)
            if false_events else 0.0
        ),
        "false_events_detail": false_events,
        "avg_stress_duration_hours": sum(durations) / len(durations) if durations else 0.0,
    }


def evaluate_stress_4h(
    df_4h: pd.DataFrame,
    crash_events: list[dict],
    op_drop_pct: float = -3.0,
) -> list[dict]:
    """Grid search sur les 30 combos stress."""
    grid = StressIndicator4h.param_grid()
    results = []

    for params in tqdm(grid, desc="Stress 4h"):
        sr = StressIndicator4h.compute(df_4h, **params)
        per_crash = evaluate_stress_per_crash(
            sr.stress_on, df_4h, crash_events, op_drop_pct,
        )
        global_m = evaluate_stress_global(sr.stress_on, df_4h, crash_events)

        n_detected = sum(
            1 for c in per_crash
            if c.get("detection_delay_hours", float("inf")) != float("inf")
        )
        finite_delays = [
            c["detection_delay_hours"] for c in per_crash
            if c.get("detection_delay_hours", float("inf")) != float("inf")
        ]
        avg_delay = sum(finite_delays) / len(finite_delays) if finite_delays else float("inf")

        finite_op = [
            c["operational_delay_hours"] for c in per_crash
            if c.get("operational_delay_hours", float("inf")) != float("inf")
        ]
        avg_op_delay = sum(finite_op) / len(finite_op) if finite_op else float("inf")

        before_bottom = sum(1 for c in per_crash if c.get("detected_before_bottom", False))

        # Score composite : penalise les faux positifs
        false_events = global_m["false_stress_events"]
        score = n_detected * 100 - (avg_delay if avg_delay != float("inf") else 500) - false_events * 10

        results.append({
            "params": params,
            "global": global_m,
            "per_crash": per_crash,
            "n_crashes_detected": n_detected,
            "avg_detection_delay_hours": avg_delay,
            "avg_operational_delay_hours": avg_op_delay,
            "crashes_detected_before_bottom": before_bottom,
            "false_stress_events": false_events,
            "score": score,
            "_stress_on": sr.stress_on,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ─── Plots 4-classes (Sprint 50a) ───────────────────────────────────────

def _plot_regime_bands(ax, dates, labels, colors=None, alpha=0.2):
    """Dessine les bandes colorees de regime sur un axe."""
    if colors is None:
        colors = REGIME_COLORS
    i = 0
    while i < len(labels):
        regime = labels[i]
        j = i
        while j < len(labels) and labels[j] == regime:
            j += 1
        color = colors.get(regime, "#cccccc")
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

    _plot_regime_bands(ax1, dates, y_true)
    ax1.plot(dates, close, color="black", linewidth=0.5, alpha=0.8)
    ax1.set_yscale("log")
    ax1.set_title("Ground Truth")
    ax1.set_ylabel("BTC Prix (log)")

    y_pred = result.labels_4h[:len(dates)]
    _plot_regime_bands(ax2, dates, y_pred)
    ax2.plot(dates, close, color="black", linewidth=0.5, alpha=0.8)
    ax2.set_yscale("log")
    ax2.set_title(f"Detecteur: {detector_name} (F1={best_result['macro_f1']:.3f})")
    ax2.set_ylabel("BTC Prix (log)")

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


# ─── Plots binaire (Sprint 50a-bis) ─────────────────────────────────────

def generate_binary_plot(
    detector_name: str,
    best_binary: dict,
    df_4h: pd.DataFrame,
    y_true_bin: list[str],
) -> str:
    """Plot timeline binaire : GT (top) + prediction (bottom)."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = IMAGES_DIR / f"regime_binary_{detector_name}.png"

    labels_bin = best_binary.get("_labels_4h_bin")
    if labels_bin is None:
        return str(filepath)

    dates = pd.to_datetime(df_4h["timestamp_utc"])
    close = df_4h["close"].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    _plot_regime_bands(ax1, dates, y_true_bin[:len(dates)], colors=BINARY_COLORS)
    ax1.plot(dates, close, color="black", linewidth=0.5, alpha=0.8)
    ax1.set_yscale("log")
    ax1.set_title("Ground Truth (binaire)")
    ax1.set_ylabel("BTC Prix (log)")

    _plot_regime_bands(ax2, dates, labels_bin[:len(dates)], colors=BINARY_COLORS)
    ax2.plot(dates, close, color="black", linewidth=0.5, alpha=0.8)
    ax2.set_yscale("log")
    f1_def = best_binary["binary_f1"].get("defensive", 0)
    ax2.set_title(f"Detecteur: {detector_name} (F1 def={f1_def:.3f})")
    ax2.set_ylabel("BTC Prix (log)")

    legend_elements = [
        Patch(facecolor=BINARY_COLORS[r], alpha=0.3, label=r.capitalize())
        for r in ["normal", "defensive"]
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    return str(filepath)


# ─── Plots stress (Sprint 50a-bis) ──────────────────────────────────────

def generate_stress_heatmap(stress_results: list[dict]) -> str:
    """Heatmap lookback x threshold, couleur = score composite."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = IMAGES_DIR / "stress_4h_heatmap.png"

    lookbacks = [6, 12, 18, 24, 48]
    thresholds = [-5.0, -8.0, -10.0, -12.0, -15.0, -20.0]

    matrix = np.zeros((len(lookbacks), len(thresholds)))
    result_map = {}
    for r in stress_results:
        lb = r["params"]["lookback_candles"]
        th = r["params"]["threshold_pct"]
        result_map[(lb, th)] = r

    for i, lb in enumerate(lookbacks):
        for j, th in enumerate(thresholds):
            r = result_map.get((lb, th))
            matrix[i, j] = r["score"] if r else 0

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"{t}%" for t in thresholds])
    ax.set_yticks(range(len(lookbacks)))
    ax.set_yticklabels([f"{lb} ({lb * 4}h)" for lb in lookbacks])
    ax.set_xlabel("Threshold (%)")
    ax.set_ylabel("Lookback (candles / heures)")
    ax.set_title("Stress 4h : Score composite (detected*100 - delay - false*10)")

    for i, lb in enumerate(lookbacks):
        for j, th in enumerate(thresholds):
            r = result_map.get((lb, th))
            if r:
                nd = r["n_crashes_detected"]
                delay = r["avg_detection_delay_hours"]
                d_str = f"{delay:.0f}h" if delay != float("inf") else "N/A"
                ax.text(j, i, f"{nd}/5\n{d_str}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, label="Score composite")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    return str(filepath)


def generate_stress_crashes_plot(
    best_stress: dict,
    df_4h: pd.DataFrame,
    crash_events: list[dict],
) -> str:
    """5 subplots zoom sur chaque crash avec stress ON/OFF."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = IMAGES_DIR / "stress_4h_crashes.png"

    timestamps = pd.to_datetime(df_4h["timestamp_utc"], utc=True)
    close = df_4h["close"].values
    stress_on = best_stress["_stress_on"]

    n_crashes = len(crash_events)
    fig, axes = plt.subplots(n_crashes, 1, figsize=(16, 4 * n_crashes))
    if n_crashes == 1:
        axes = [axes]

    for idx, event in enumerate(crash_events):
        ax = axes[idx]
        crash_info = best_stress["per_crash"][idx]

        gt_start = crash_info.get("gt_start_idx", 0)
        gt_end = crash_info.get("gt_end_idx", gt_start + 10)
        margin = 42  # +-7 jours = +-42 candles 4h

        plot_start = max(0, gt_start - margin)
        plot_end = min(len(close) - 1, gt_end + margin)

        ax.plot(timestamps.iloc[plot_start:plot_end + 1],
                close[plot_start:plot_end + 1],
                color="black", linewidth=1)

        # Bande GT crash (grise)
        ax.axvspan(timestamps.iloc[gt_start], timestamps.iloc[gt_end],
                    alpha=0.2, color="#888888", label="GT crash")

        # Bandes stress ON (rouge)
        for i in range(plot_start, plot_end + 1):
            if stress_on[i]:
                ax.axvspan(timestamps.iloc[i],
                           timestamps.iloc[min(i + 1, len(timestamps) - 1)],
                           alpha=0.15, color="red", linewidth=0)

        # Marqueurs
        bottom_idx = crash_info.get("bottom_idx")
        if bottom_idx and plot_start <= bottom_idx <= plot_end:
            ax.plot(timestamps.iloc[bottom_idx], close[bottom_idx],
                    "v", color="red", markersize=10, label="Bottom")

        op_idx = crash_info.get("operational_start_idx")
        if op_idx and plot_start <= op_idx <= plot_end:
            ax.plot(timestamps.iloc[op_idx], close[op_idx],
                    "D", color="orange", markersize=8, label="Op. start")

        delay_h = crash_info.get("detection_delay_hours", float("inf"))
        d_str = f"{delay_h:.0f}h" if delay_h != float("inf") else "N/D"
        op_h = crash_info.get("operational_delay_hours", float("inf"))
        op_str = f"{op_h:.0f}h" if op_h != float("inf") else "N/D"
        before = "OUI" if crash_info.get("detected_before_bottom") else "NON"

        ax.set_title(
            f"{event['name']} — Delay GT: {d_str}, Delay Op: {op_str}, "
            f"Avant bottom: {before}",
            fontsize=10,
        )
        ax.set_ylabel("BTC")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    return str(filepath)


def generate_combined_plot(
    best_binary: dict,
    detector_name: str,
    best_stress: dict,
    df_4h: pd.DataFrame,
    y_true_bin: list[str],
) -> str:
    """Timeline combinee : GT binaire + Couche 1 + Couche 2."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = IMAGES_DIR / "regime_binary_combined.png"

    dates = pd.to_datetime(df_4h["timestamp_utc"])
    close = df_4h["close"].values
    labels_bin = best_binary.get("_labels_4h_bin", [])
    stress_on = best_stress["_stress_on"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 14), sharex=True)

    # GT binaire
    _plot_regime_bands(ax1, dates, y_true_bin[:len(dates)], colors=BINARY_COLORS)
    ax1.plot(dates, close, color="black", linewidth=0.5, alpha=0.8)
    ax1.set_yscale("log")
    ax1.set_title("Ground Truth (binaire)")
    ax1.set_ylabel("BTC Prix (log)")
    legend_elements = [
        Patch(facecolor=BINARY_COLORS[r], alpha=0.3, label=r.capitalize())
        for r in ["normal", "defensive"]
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=9)

    # Couche 1 : prediction binaire
    _plot_regime_bands(ax2, dates, labels_bin[:len(dates)], colors=BINARY_COLORS)
    ax2.plot(dates, close, color="black", linewidth=0.5, alpha=0.8)
    ax2.set_yscale("log")
    f1_def = best_binary["binary_f1"].get("defensive", 0)
    ax2.set_title(f"Couche 1 : {detector_name} (F1 def={f1_def:.3f})")
    ax2.set_ylabel("BTC Prix (log)")

    # Couche 2 : stress markers
    ax3.plot(dates, close, color="black", linewidth=0.5, alpha=0.8)
    ax3.set_yscale("log")
    for i in range(len(stress_on)):
        if stress_on[i]:
            ax3.axvspan(dates.iloc[i], dates.iloc[min(i + 1, len(dates) - 1)],
                        alpha=0.2, color="red", linewidth=0)
    lb = best_stress["params"]["lookback_candles"]
    th = best_stress["params"]["threshold_pct"]
    ax3.set_title(f"Couche 2 : Stress 4h (lookback={lb}, threshold={th}%)")
    ax3.set_ylabel("BTC Prix (log)")

    stress_legend = [
        Patch(facecolor="red", alpha=0.3, label="Stress ON"),
        Patch(facecolor="white", edgecolor="gray", label="Stress OFF"),
    ]
    ax3.legend(handles=stress_legend, loc="upper left", fontsize=9)

    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    return str(filepath)


# ─── Generation rapport 4-classes (Sprint 50a) ──────────────────────────

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
    """Genere le rapport markdown Sprint 50a."""
    lines = [
        "# BTC Regime Detector Calibration Report — Sprint 50a",
        "",
        f"*Genere automatiquement le {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        f"**Donnees** : {len(df_4h)} candles 4h BTC/USDT",
        "",
    ]

    lines.append("## 1. Resume executif")
    lines.append("")

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

    for name, results in all_results.items():
        lines.append(f"## 3. Detecteur : {name}")
        lines.append("")
        if not results:
            lines.append("*Aucun resultat*")
            continue
        best = results[0]
        lines.append("### Meilleure configuration")
        lines.append("")
        lines.append(f"```json\n{json.dumps(best['params'], indent=2)}\n```")
        lines.append("")
        lines.append(f"- Macro F1 : {best['macro_f1']:.4f}")
        lines.append(f"- Accuracy : {best['accuracy']:.4f}")
        lines.append(f"- Candles evaluees : {best.get('n_candles_eval', '?')} (apres warmup)")
        lines.append(f"- Transitions : {best['n_transitions']}")
        lines.append(f"- Stabilite : {best['stability']:.3f}")
        lines.append("")
        lines.append("**F1 par regime :**")
        lines.append("")
        for regime in LABELS:
            lines.append(f"- {regime.capitalize()} : {best['f1'][regime]:.3f}")
        lines.append("")
        lines.append("**Distribution :**")
        lines.append("")
        for regime in LABELS:
            pct = best["distribution"].get(regime, 0) * 100
            lines.append(f"- {regime.capitalize()} : {pct:.1f}%")
        lines.append("")
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
        lines.append("### Timeline")
        lines.append("")
        lines.append(f"![{name}](../docs/images/regime_{name}_best.png)")
        lines.append("")

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

    if best_overall and "_result" in best_overall:
        ca = best_overall["crash_analysis"]
        if ca["delays_hours"]:
            lines.append(f"**Detail pour {best_det_name} (meilleur detecteur) :**")
            lines.append("")
            for i, delay in enumerate(ca["delays_hours"]):
                delay_str = f"{delay:.0f}h" if delay != float("inf") else "NON DETECTE"
                lines.append(f"- Crash #{i + 1} : delai = {delay_str}")
            lines.append("")

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

    lines.append("## 6. Recommandation")
    lines.append("")
    if best_overall:
        lines.append(
            f"Le detecteur `{best_det_name}` est recommande avec les parametres ci-dessus. "
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

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapport genere : {REPORT_PATH}")


# ─── Generation rapport binaire (Sprint 50a-bis) ────────────────────────

def generate_binary_report(
    binary_results: dict[str, list[dict]],
    stress_results: list[dict] | None,
    df_4h: pd.DataFrame,
    crash_events: list[dict],
    best_binary_det: str,
    op_drop_pct: float,
) -> None:
    """Genere le rapport Sprint 50a-bis."""
    lines = [
        "# Regime Binary Analysis Report — Sprint 50a-bis",
        "",
        f"*Genere automatiquement le {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        f"**Donnees** : {len(df_4h)} candles 4h BTC/USDT",
        f"**Seuil operationnel** : {op_drop_pct}% (single-candle drop)",
        "",
    ]

    # --- 1. Resume executif ---
    lines.append("## 1. Resume executif")
    lines.append("")

    if binary_results and best_binary_det in binary_results:
        best_bin = binary_results[best_binary_det][0]
        f1_def = best_bin["binary_f1"].get("defensive", 0)
        f1_4c = best_bin.get("original_macro_f1", 0)
        lines.append(f"**Couche 1 (Tendance)** : `{best_binary_det}` — "
                      f"F1 defensive = {f1_def:.3f} (vs F1 4-classes = {f1_4c:.3f})")
        lines.append(f"- false_defensive = {best_bin['false_defensive_rate']:.1f}% "
                      f"(cible <15%), missed_defensive = {best_bin['missed_defensive_rate']:.1f}% "
                      f"(cible <20%)")
        lines.append("")

    if stress_results:
        best_stress = stress_results[0]
        lb = best_stress["params"]["lookback_candles"]
        th = best_stress["params"]["threshold_pct"]
        nd = best_stress["n_crashes_detected"]
        avg_d = best_stress["avg_detection_delay_hours"]
        d_str = f"{avg_d:.0f}h" if avg_d != float("inf") else "N/A"
        lines.append(f"**Couche 2 (Stress 4h)** : lookback={lb} ({lb * 4}h), threshold={th}%")
        lines.append(f"- Detecte {nd}/5 crashs, delai moyen {d_str}")
        lines.append(f"- Faux positifs : {best_stress['false_stress_events']}")
        lines.append("")

    go = "GO" if (binary_results and stress_results) else "A CONFIRMER"
    lines.append(f"**Verdict Sprint 50b** : {go}")
    lines.append("")

    # --- 2. Couche 1 ---
    lines.append("## 2. Couche 1 — Tendance binaire")
    lines.append("")
    lines.append("### 2.1 Comparaison des detecteurs")
    lines.append("")
    lines.append("| Detecteur | F1 def | F1 norm | Accuracy | time_def% | false_def% | missed_def% | trans/an | F1 4-classes |")
    lines.append("|-----------|--------|---------|----------|-----------|------------|-------------|----------|-------------|")

    for det_name, entries in binary_results.items():
        if not entries:
            continue
        b = entries[0]
        lines.append(
            f"| {det_name} | {b['binary_f1'].get('defensive', 0):.3f} | "
            f"{b['binary_f1'].get('normal', 0):.3f} | {b['binary_accuracy']:.3f} | "
            f"{b['time_in_defensive_pct']:.1f} | {b['false_defensive_rate']:.1f} | "
            f"{b['missed_defensive_rate']:.1f} | {b['transition_frequency']:.1f} | "
            f"{b.get('original_macro_f1', 0):.3f} |"
        )
    lines.append("")

    # Top 3 par detecteur
    for det_name, entries in binary_results.items():
        lines.append(f"### 2.2 Top 3 : {det_name}")
        lines.append("")
        lines.append("| Rang | F1 def | false_def% | missed_def% | trans/an | Params |")
        lines.append("|------|--------|------------|-------------|----------|--------|")
        for rank, b in enumerate(entries[:3], 1):
            p = json.dumps(b["params"], separators=(",", ":"))
            lines.append(
                f"| {rank} | {b['binary_f1'].get('defensive', 0):.3f} | "
                f"{b['false_defensive_rate']:.1f} | {b['missed_defensive_rate']:.1f} | "
                f"{b['transition_frequency']:.1f} | `{p}` |"
            )
        lines.append("")

    # --- 3. Couche 2 ---
    if stress_results:
        lines.append("## 3. Couche 2 — Stress 4h")
        lines.append("")
        lines.append("### 3.1 Heatmap")
        lines.append("")
        lines.append("![stress_heatmap](../docs/images/stress_4h_heatmap.png)")
        lines.append("")

        lines.append("### 3.2 Analyse par crash")
        lines.append("")
        best_stress = stress_results[0]
        lines.append(f"**Config** : lookback={best_stress['params']['lookback_candles']}, "
                      f"threshold={best_stress['params']['threshold_pct']}%")
        lines.append("")
        lines.append("| Crash | Delay GT (h) | Delay Op (h) | Avant bottom | Recovery (h) | Alarm 48h |")
        lines.append("|-------|-------------|-------------|-------------|-------------|-----------|")
        for pc in best_stress["per_crash"]:
            d_gt = pc.get("detection_delay_hours", float("inf"))
            d_op = pc.get("operational_delay_hours", float("inf"))
            d_gt_s = f"{d_gt:.0f}" if d_gt != float("inf") else "N/D"
            d_op_s = f"{d_op:.0f}" if d_op != float("inf") else "N/D"
            before = "OUI" if pc.get("detected_before_bottom") else "NON"
            rec = pc.get("recovery_delay_hours", float("inf"))
            rec_s = f"{rec:.0f}" if rec != float("inf") else "N/D"
            alarm = "OUI" if pc.get("pre_crash_false_alarm_48h") else "NON"
            lines.append(
                f"| {pc['event']} | {d_gt_s} | {d_op_s} | {before} | {rec_s} | {alarm} |"
            )
        lines.append("")

        # Metriques globales
        gm = best_stress["global"]
        lines.append("### 3.3 Metriques globales")
        lines.append("")
        lines.append(f"- Stress total : {gm['total_stress_hours']:.0f}h ({gm['stress_pct']:.1f}%)")
        lines.append(f"- Faux positifs : {gm['false_stress_events']} ({gm['false_stress_per_year']:.1f}/an)")
        lines.append(f"- Duree moyenne faux positifs : {gm['avg_false_stress_duration_hours']:.0f}h")
        lines.append("")

        # Top 5
        lines.append("### 3.4 Top 5 configurations")
        lines.append("")
        lines.append("| Rang | Lookback | Threshold | Score | Detected | Delay moy | False events |")
        lines.append("|------|----------|-----------|-------|----------|-----------|-------------|")
        for rank, sr in enumerate(stress_results[:5], 1):
            avg_d = sr["avg_detection_delay_hours"]
            d_str = f"{avg_d:.0f}h" if avg_d != float("inf") else "N/A"
            lines.append(
                f"| {rank} | {sr['params']['lookback_candles']} ({sr['params']['lookback_candles'] * 4}h) | "
                f"{sr['params']['threshold_pct']}% | {sr['score']:.0f} | "
                f"{sr['n_crashes_detected']}/5 | {d_str} | {sr['false_stress_events']} |"
            )
        lines.append("")

        # Tableau faux positifs
        if gm["false_events_detail"]:
            lines.append("### 3.5 Detail faux positifs")
            lines.append("")
            lines.append("| Date | Duree (h) |")
            lines.append("|------|-----------|")
            for fe in gm["false_events_detail"][:20]:
                lines.append(f"| {fe['date']} | {fe['duration_h']:.0f} |")
            lines.append("")

    # --- 4. Combinaison ---
    if binary_results and stress_results:
        lines.append("## 4. Combinaison des 2 couches")
        lines.append("")
        lines.append("![combined](../docs/images/regime_binary_combined.png)")
        lines.append("")
        lines.append("Verification visuelle : les 2 couches se completent-elles ?")
        lines.append("- Couche 1 (lente) couvre les regimes tendanciels (bear markets)")
        lines.append("- Couche 2 (rapide) reagit aux chocs ponctuels (crashs)")
        lines.append("")

    # --- 5. Recommandation ---
    lines.append("## 5. Recommandation finale")
    lines.append("")
    if binary_results and best_binary_det in binary_results:
        best_bin = binary_results[best_binary_det][0]
        lines.append(f"**Couche 1** : `{best_binary_det}` avec params "
                      f"`{json.dumps(best_bin['params'], separators=(',', ':'))}`")
        lines.append(f"- leverage normal=7x, defensive=4x")
        lines.append("")
    if stress_results:
        best_stress = stress_results[0]
        lines.append(f"**Couche 2** : lookback={best_stress['params']['lookback_candles']}, "
                      f"threshold={best_stress['params']['threshold_pct']}%")
        lines.append("- DCA throttle quand stress ON (reduire taille DCA de 50%)")
        lines.append("")
    lines.append("**Sprint 50b** : integrer ces 2 couches dans le backtester portfolio")
    lines.append("pour mesurer l'impact sur le PnL et le drawdown.")
    lines.append("")

    BINARY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BINARY_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapport binaire genere : {BINARY_REPORT_PATH}")


# ─── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Analyse des detecteurs de regime BTC (Sprint 50a / 50a-bis)",
    )
    parser.add_argument(
        "--detector",
        choices=["sma_stress", "ema_atr", "multi_ma_vol"],
        help="Filtrer un seul detecteur",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Ne pas generer les graphiques")
    # Sprint 50a-bis
    parser.add_argument("--binary", action="store_true", help="Mode binaire (normal/defensive)")
    parser.add_argument("--stress", action="store_true", help="Stress 4h")
    parser.add_argument("--stress-only", action="store_true", help="Stress 4h uniquement")
    parser.add_argument("--force-recalc", action="store_true", help="Ignorer le cache grid")
    parser.add_argument("--op-drop-pct", type=float, default=-3.0,
                        help="Seuil drop single-candle pour delai operationnel (defaut: -3.0%%)")
    args = parser.parse_args()

    # Determiner le mode
    mode_binary = args.binary
    mode_stress = args.stress or args.stress_only
    mode_50a_bis = mode_binary or mode_stress

    if not mode_50a_bis:
        # Sprint 50a classique (4-classes)
        _run_4class_analysis(args)
        return

    # === Sprint 50a-bis ===
    print("=== BTC Regime Analysis — Sprint 50a-bis ===\n")

    df_4h, df_daily = load_data()
    y_true = df_4h["regime_label"].tolist()
    print(f"Donnees : {len(df_4h)} candles 4h")

    # --- Charger ou calculer le cache 4-classes ---
    all_results = None
    if mode_binary and not args.force_recalc:
        all_results = load_grid_cache()
        if all_results:
            total = sum(len(v) for v in all_results.values())
            print(f"Cache charge : {total} resultats\n")

    if mode_binary and all_results is None:
        print("Calcul de la grille 4-classes (cache manquant)...\n")
        all_results = {}
        for detector in ALL_DETECTORS:
            if args.detector and detector.name != args.detector:
                continue
            results = grid_search(detector, df_4h, df_daily, y_true)
            all_results[detector.name] = results
        save_grid_cache(all_results, len(df_4h))

    # --- Evaluation binaire ---
    binary_results = None
    best_binary_det = ""
    if mode_binary and all_results:
        print("Evaluation binaire...")
        binary_results = evaluate_binary(all_results, y_true)
        # Trouver le meilleur detecteur binaire
        best_f1_def = -1.0
        for det_name, entries in binary_results.items():
            if entries:
                f1_d = entries[0]["binary_f1"].get("defensive", 0)
                if f1_d > best_f1_def:
                    best_f1_def = f1_d
                    best_binary_det = det_name
        print(f"Meilleur detecteur binaire : {best_binary_det} "
              f"(F1 def={best_f1_def:.3f})\n")

        # Afficher top 1 par detecteur
        for det_name, entries in binary_results.items():
            if entries:
                b = entries[0]
                print(f"  {det_name:15} F1_def={b['binary_f1'].get('defensive', 0):.3f} "
                      f"false_def={b['false_defensive_rate']:.1f}% "
                      f"missed_def={b['missed_defensive_rate']:.1f}% "
                      f"trans/an={b['transition_frequency']:.1f}")

        if not args.skip_plots:
            print("\nGeneration des plots binaires...")
            y_true_bin = to_binary_labels(y_true)
            for det_name, entries in binary_results.items():
                if entries:
                    path = generate_binary_plot(det_name, entries[0], df_4h, y_true_bin)
                    print(f"  -> {path}")

    # --- Evaluation stress ---
    stress_results = None
    crash_events = None
    if mode_stress:
        crash_events = _load_crash_events()
        print(f"\nEvaluation stress 4h ({len(crash_events)} crashs GT)...")
        stress_results = evaluate_stress_4h(df_4h, crash_events, args.op_drop_pct)

        best = stress_results[0]
        avg_d = best["avg_detection_delay_hours"]
        d_str = f"{avg_d:.0f}h" if avg_d != float("inf") else "N/A"
        print(f"\nMeilleure config stress : lookback={best['params']['lookback_candles']}, "
              f"threshold={best['params']['threshold_pct']}%")
        print(f"  Detecte {best['n_crashes_detected']}/5, delay moy={d_str}, "
              f"score={best['score']:.0f}")
        print(f"  Faux positifs : {best['false_stress_events']}")

        if not args.skip_plots:
            print("\nGeneration des plots stress...")
            path = generate_stress_heatmap(stress_results)
            print(f"  -> {path}")
            path = generate_stress_crashes_plot(best, df_4h, crash_events)
            print(f"  -> {path}")

    # --- Plot combine ---
    if binary_results and stress_results and not args.skip_plots:
        print("\nGeneration du plot combine...")
        y_true_bin = to_binary_labels(y_true)
        best_binary = binary_results[best_binary_det][0]
        best_stress = stress_results[0]
        path = generate_combined_plot(
            best_binary, best_binary_det, best_stress, df_4h, y_true_bin,
        )
        print(f"  -> {path}")

    # --- Rapport ---
    if crash_events is None:
        crash_events = _load_crash_events()
    generate_binary_report(
        binary_results or {},
        stress_results,
        df_4h,
        crash_events,
        best_binary_det,
        args.op_drop_pct,
    )

    print("\nDone.")


def _run_4class_analysis(args) -> None:
    """Sprint 50a : analyse 4-classes (code original)."""
    print("=== BTC Regime Detector Analysis — Sprint 50a ===\n")

    df_4h, df_daily = load_data()
    y_true = df_4h["regime_label"].tolist()
    print(f"Donnees : {len(df_4h)} candles 4h, {len(df_daily)} jours daily")
    print(f"Ground truth : {len(set(y_true))} regimes\n")

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

        if results:
            print("\nAnalyse de robustesse...")
            rob = robustness_check(detector, df_4h, df_daily, y_true, results[0])
            robustness_results[detector.name] = rob
            status = "ROBUSTE" if rob["robust"] else f"WARNING ({rob['delta_pct']:.1f}% delta)"
            print(f"  Robustesse : {status} ({rob['n_neighbors']} voisins)")

        if not args.skip_plots and results:
            print("Generation du plot...")
            path = generate_detector_plot(detector.name, results[0], df_4h, y_true)
            print(f"  -> {path}")

        print()

    # Sauvegarder le cache pour les runs futurs
    save_grid_cache(all_results, len(df_4h))

    generate_report(all_results, robustness_results, df_4h)

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
