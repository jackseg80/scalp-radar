"""Label BTC 4h candles avec les regimes de marche ground truth.

Charge le CSV de candles 4h et le YAML d'evenements annotes,
assigne un label par candle, et produit un CSV enrichi + plot.

Usage:
    uv run python -m scripts.regime_labeler
    uv run python -m scripts.regime_labeler --csv data/btc_4h_2017_2025.csv
"""

from __future__ import annotations

import argparse
import io
import sys
from itertools import groupby
from pathlib import Path

import yaml

try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    sys.exit("pandas et matplotlib requis. Installer: uv sync --group analysis")

# Priorite : crash(0) > bear(1) > range(2) > bull(3)
REGIME_PRIORITY = {"crash": 0, "bear": 1, "range": 2, "bull": 3}
REGIME_COLORS = {
    "bull": "#2ecc71",
    "bear": "#e74c3c",
    "range": "#95a5a6",
    "crash": "#8e44ad",
}


def load_events(yaml_path: str) -> tuple[list[dict], str]:
    """Charge les evenements depuis le YAML.

    Retourne (events, default_regime).
    Chaque event a start/end convertis en pd.Timestamp.
    """
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    default_regime = data.get("default_regime", "range")
    events = []
    for evt in data.get("events", []):
        events.append({
            "name": evt["name"],
            "regime": evt["type"],
            "start": pd.Timestamp(evt["start"], tz="UTC"),
            "end": pd.Timestamp(evt["end"], tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
            "severity": evt.get("severity", "minor"),
            "notes": evt.get("notes", ""),
        })

    return events, default_regime


def label_candles(
    df: pd.DataFrame,
    events: list[dict],
    default_regime: str = "range",
) -> pd.DataFrame:
    """Assigne regime_label a chaque candle 4h.

    Pour chaque candle, trouve tous les evenements dont [start, end] contient
    le timestamp. Si multiple : priorite crash > bear > range > bull.
    Si aucun : default_regime.
    """
    timestamps = pd.to_datetime(df["timestamp_utc"], utc=True)
    labels = []

    for ts in timestamps:
        matched = []
        for evt in events:
            if evt["start"] <= ts <= evt["end"]:
                matched.append(evt["regime"])
        if matched:
            labels.append(min(matched, key=lambda r: REGIME_PRIORITY.get(r, 99)))
        else:
            labels.append(default_regime)

    df = df.copy()
    df["regime_label"] = labels
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Affiche la distribution des regimes et le nombre de transitions."""
    counts = df["regime_label"].value_counts()
    total = len(df)

    print("\n=== Distribution des regimes (Ground Truth) ===")
    for regime in ["bull", "bear", "range", "crash"]:
        n = counts.get(regime, 0)
        pct = n / total * 100
        print(f"  {regime:6}: {n:6} candles ({pct:5.1f}%)")
    print(f"  Total : {total} candles")

    # Transitions
    labels = df["regime_label"].tolist()
    transitions = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
    print(f"\n  Transitions : {transitions}")

    # Segments
    segments = [(regime, sum(1 for _ in group)) for regime, group in groupby(labels)]
    print(f"  Segments    : {len(segments)}")
    for regime in ["bull", "bear", "range", "crash"]:
        regime_segs = [length for r, length in segments if r == regime]
        if regime_segs:
            avg_h = sum(regime_segs) / len(regime_segs) * 4
            print(f"  {regime:6} duree moyenne : {avg_h:.0f}h ({avg_h / 24:.1f}j)")


def plot_ground_truth(df: pd.DataFrame, output_path: str) -> None:
    """Genere BTC prix + bandes colorees par regime."""
    fig, ax = plt.subplots(figsize=(20, 8))
    dates = pd.to_datetime(df["timestamp_utc"])
    close = df["close"].values

    # Prix
    ax.plot(dates, close, color="black", linewidth=0.5, alpha=0.8)
    ax.set_yscale("log")

    # Bandes colorees par blocs contigus
    labels = df["regime_label"].tolist()
    i = 0
    while i < len(labels):
        regime = labels[i]
        j = i
        while j < len(labels) and labels[j] == regime:
            j += 1
        color = REGIME_COLORS.get(regime, "#cccccc")
        ax.axvspan(dates.iloc[i], dates.iloc[min(j, len(dates) - 1)],
                    alpha=0.2, color=color, linewidth=0)
        i = j

    ax.set_title("BTC/USDT 4h — Ground Truth Regimes (2017-2025)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix (USD, echelle log)")

    # Legende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=REGIME_COLORS[r], alpha=0.3, label=r.capitalize())
        for r in ["bull", "bear", "range", "crash"]
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"\nPlot sauvegarde: {output}")


def main() -> None:
    # Windows UTF-8
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Label BTC 4h candles avec les regimes ground truth",
    )
    parser.add_argument(
        "--csv", type=str, default="data/btc_4h_2017_2025.csv",
        help="CSV source (defaut: data/btc_4h_2017_2025.csv)",
    )
    parser.add_argument(
        "--yaml", type=str, default="data/btc_regime_events.yaml",
        help="YAML d'evenements (defaut: data/btc_regime_events.yaml)",
    )
    parser.add_argument(
        "--output", type=str, default="data/btc_4h_labeled.csv",
        help="CSV de sortie (defaut: data/btc_4h_labeled.csv)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Ne pas generer le graphique",
    )
    args = parser.parse_args()

    # Charger les donnees
    df = pd.read_csv(args.csv)
    print(f"Charge {len(df)} candles depuis {args.csv}")

    events, default_regime = load_events(args.yaml)
    print(f"Charge {len(events)} evenements depuis {args.yaml}")
    print(f"Regime par defaut: {default_regime}")

    # Labeling
    df = label_candles(df, events, default_regime)

    # Summary
    print_summary(df)

    # Export
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nCSV enrichi sauvegarde: {output_path}")

    # Plot
    if not args.no_plot:
        plot_ground_truth(df, "docs/images/btc_ground_truth_regimes.png")


if __name__ == "__main__":
    main()
