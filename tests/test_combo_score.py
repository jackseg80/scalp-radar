"""Tests combo_score — sélection du best combo WFO par score composite."""

import json

import pytest

from backend.optimization.walk_forward import combo_score


# ─── Test 1 : Haute consistance bat haut Sharpe ──────────────────────────

def test_combo_score_prefers_high_consistency():
    """Combo 93% consistance + 7.5 Sharpe + 300 trades bat combo 63% + 8.8 Sharpe + 32 trades."""
    # Combo A : haute consistance, bon volume, Sharpe légèrement plus bas
    score_a = combo_score(oos_sharpe=7.5, consistency=0.93, total_trades=300)
    # Combo B : haut Sharpe mais faible consistance ET peu de trades
    score_b = combo_score(oos_sharpe=8.8, consistency=0.63, total_trades=32)

    assert score_a > score_b, f"Score A ({score_a:.2f}) devrait battre Score B ({score_b:.2f})"

    # Vérification valeurs attendues (seuil = 100 trades)
    # A = 7.5 * (0.4 + 0.6*0.93) * min(1, 300/100) = 7.5 * 0.958 * 1.0 = 7.185
    # B = 8.8 * (0.4 + 0.6*0.63) * min(1, 32/100) = 8.8 * 0.778 * 0.32 = 2.191
    assert score_a == pytest.approx(7.185, abs=0.01)
    assert score_b == pytest.approx(2.191, abs=0.01)


# ─── Test 2 : Pénalité pour faible volume de trades ─────────────────────

def test_combo_score_penalizes_low_trades():
    """Combo avec 20 trades pénalisé vs combo avec 200 trades (même Sharpe/consistance)."""
    score_low = combo_score(oos_sharpe=2.0, consistency=0.8, total_trades=20)
    score_high = combo_score(oos_sharpe=2.0, consistency=0.8, total_trades=200)

    # Le trade_factor pénalise : 20/100=0.2 vs min(1,200/100)=1.0
    assert score_high > score_low
    assert score_high / score_low == pytest.approx(1.0 / 0.2, abs=0.01)


def test_combo_score_39_trades_loses_to_111_trades():
    """Cas ETH : combo 39 trades/Sharpe 7.92 perd face à combo 111 trades/Sharpe 5.81."""
    # Combo #1 : haut Sharpe mais 39 trades (< 100 → trade_factor = 0.39)
    score_1 = combo_score(oos_sharpe=7.92, consistency=0.80, total_trades=39)
    # Combo #2 : Sharpe plus bas mais 111 trades (> 100 → trade_factor = 1.0)
    score_2 = combo_score(oos_sharpe=5.81, consistency=0.80, total_trades=111)

    # #1 = 7.92 * (0.4 + 0.6*0.80) * 0.39 = 7.92 * 0.88 * 0.39 = 2.718
    # #2 = 5.81 * (0.4 + 0.6*0.80) * 1.00 = 5.81 * 0.88 * 1.00 = 5.113
    assert score_2 > score_1, f"111 trades ({score_2:.2f}) devrait battre 39 trades ({score_1:.2f})"
    assert score_1 == pytest.approx(2.718, abs=0.01)
    assert score_2 == pytest.approx(5.113, abs=0.01)


def test_combo_score_negative_sharpe_is_zero():
    """Combo avec Sharpe négatif → score = 0."""
    score = combo_score(oos_sharpe=-1.5, consistency=0.9, total_trades=200)
    assert score == 0.0


def test_combo_score_zero_trades():
    """Combo avec 0 trades → score = 0."""
    score = combo_score(oos_sharpe=3.0, consistency=0.8, total_trades=0)
    assert score == 0.0


# ─── Test 3 : Seuils OOS/IS ratio (documentation des 3 niveaux) ────────

def test_oos_is_ratio_thresholds():
    """Vérifier les 3 niveaux de diagnostic OOS/IS (bon/modéré/fort).

    Seuils frontend (diagnosticUtils.js) :
    - ratio > 0.7  → vert  (bon transfert)
    - 0.5 ≤ ratio < 0.7 → orange (dégradation modérée)
    - ratio < 0.5  → rouge (dégradation forte, overfitting probable)
    """

    def classify_oos_is_ratio(ratio: float) -> str:
        """Miroir Python de la logique diagnosticUtils.js (Règle 3)."""
        if ratio >= 0.7:
            return "green"
        elif ratio >= 0.5:
            return "orange"
        else:
            return "red"

    # Cas bon transfert
    assert classify_oos_is_ratio(0.85) == "green"
    assert classify_oos_is_ratio(0.70) == "green"  # Seuil exact

    # Cas dégradation modérée
    assert classify_oos_is_ratio(0.69) == "orange"
    assert classify_oos_is_ratio(0.50) == "orange"  # Seuil exact

    # Cas dégradation forte / overfitting
    assert classify_oos_is_ratio(0.49) == "red"
    assert classify_oos_is_ratio(0.20) == "red"
    assert classify_oos_is_ratio(0.0) == "red"


# ─── Test 4 : regime_analysis croise toutes les fenêtres OOS (Hotfix 37c) ──

def test_regime_analysis_uses_best_scored_combo():
    """Hotfix 37c : regime_analysis croise window_results × window_regimes directement.

    L'ancienne implémentation utilisait combo_accumulator.get(recommended_key)
    qui retournait quasi-vide (le combo médian n'est testé que dans 0-1 fenêtre).
    La nouvelle itère sur toutes les fenêtres OOS → tous les régimes sont couverts.
    """
    import math
    from types import SimpleNamespace

    import numpy as np

    # 3 fenêtres OOS avec 3 régimes distincts
    window_regimes = [
        {"regime": "bull", "return_pct": 15.0, "max_dd_pct": -5.0},
        {"regime": "range", "return_pct": 2.0, "max_dd_pct": -8.0},
        {"regime": "bear", "return_pct": -12.0, "max_dd_pct": -20.0},
    ]
    window_results = [
        SimpleNamespace(oos_sharpe=3.0, oos_net_return_pct=8.0),   # bull
        SimpleNamespace(oos_sharpe=2.8, oos_net_return_pct=7.0),   # range
        SimpleNamespace(oos_sharpe=3.2, oos_net_return_pct=9.0),   # bear
    ]

    # Reproduire la logique de walk_forward.py (Hotfix 37c)
    regime_groups: dict[str, list] = {}
    for i, w in enumerate(window_results):
        if i >= len(window_regimes):
            break
        regime = window_regimes[i]["regime"]
        oos_sharpe_val = w.oos_sharpe if not math.isnan(w.oos_sharpe) else None
        regime_groups.setdefault(regime, []).append({
            "oos_sharpe": oos_sharpe_val,
            "oos_return_pct": w.oos_net_return_pct,
        })

    regime_analysis = {}
    for regime, entries in regime_groups.items():
        oos_sharpes = [e["oos_sharpe"] for e in entries if e["oos_sharpe"] is not None]
        oos_returns = [e["oos_return_pct"] for e in entries if e["oos_return_pct"] is not None]
        n_positive = sum(1 for s in oos_sharpes if s > 0)
        regime_analysis[regime] = {
            "n_windows": len(entries),
            "avg_oos_sharpe": round(float(np.nanmean(oos_sharpes)), 4) if oos_sharpes else 0.0,
            "consistency": round(n_positive / len(oos_sharpes), 4) if oos_sharpes else 0.0,
            "avg_return_pct": round(float(np.mean(oos_returns)), 4) if oos_returns else 0.0,
        }

    # Les 3 régimes sont couverts (auparavant : 0 car combo_accumulator lookup vide)
    assert len(regime_analysis) == 3
    assert "bull" in regime_analysis
    assert "range" in regime_analysis
    assert "bear" in regime_analysis

    # Chaque régime a exactement 1 fenêtre avec des données OOS > 0
    for regime_name, data in regime_analysis.items():
        assert data["n_windows"] == 1, f"Régime {regime_name} : {data['n_windows']} fenêtres (attendu 1)"
        assert data["avg_oos_sharpe"] > 0, f"Régime {regime_name} : Sharpe <= 0"


def test_regime_analysis_uses_all_windows():
    """Hotfix 37c : 6 fenêtres → 4 régimes distincts, avec agrégation correcte.

    Vérifie que toutes les fenêtres sont utilisées (pas seulement 1 combo) :
    - bull×2 : moyenne Sharpe = 2.8, consistance 100%
    - bear×2 : moyenne Sharpe = 0.1 (0.5 + -0.3), consistance 50%
    - range×1 : 1 fenêtre
    - crash×1 : oos_sharpe=NaN → avg=0.0
    """
    import math
    from types import SimpleNamespace

    import numpy as np

    window_regimes = [
        {"regime": "bull"},
        {"regime": "range"},
        {"regime": "bear"},
        {"regime": "bull"},
        {"regime": "crash"},
        {"regime": "bear"},
    ]
    window_results = [
        SimpleNamespace(oos_sharpe=2.5, oos_net_return_pct=6.0),           # bull
        SimpleNamespace(oos_sharpe=1.8, oos_net_return_pct=3.0),           # range
        SimpleNamespace(oos_sharpe=0.5, oos_net_return_pct=1.0),           # bear
        SimpleNamespace(oos_sharpe=3.1, oos_net_return_pct=8.0),           # bull
        SimpleNamespace(oos_sharpe=float("nan"), oos_net_return_pct=-15.0),  # crash
        SimpleNamespace(oos_sharpe=-0.3, oos_net_return_pct=-2.0),         # bear
    ]

    # Reproduire la logique de walk_forward.py (Hotfix 37c)
    regime_groups: dict[str, list] = {}
    for i, w in enumerate(window_results):
        if i >= len(window_regimes):
            break
        regime = window_regimes[i]["regime"]
        oos_sharpe_val = w.oos_sharpe if not math.isnan(w.oos_sharpe) else None
        regime_groups.setdefault(regime, []).append({
            "oos_sharpe": oos_sharpe_val,
            "oos_return_pct": w.oos_net_return_pct,
        })

    regime_analysis = {}
    for regime, entries in regime_groups.items():
        oos_sharpes = [e["oos_sharpe"] for e in entries if e["oos_sharpe"] is not None]
        oos_returns = [e["oos_return_pct"] for e in entries if e["oos_return_pct"] is not None]
        n_positive = sum(1 for s in oos_sharpes if s > 0)
        regime_analysis[regime] = {
            "n_windows": len(entries),
            "avg_oos_sharpe": round(float(np.nanmean(oos_sharpes)), 4) if oos_sharpes else 0.0,
            "consistency": round(n_positive / len(oos_sharpes), 4) if oos_sharpes else 0.0,
            "avg_return_pct": round(float(np.mean(oos_returns)), 4) if oos_returns else 0.0,
        }

    # 4 régimes distincts détectés
    assert len(regime_analysis) == 4, (
        f"Attendu 4 régimes, obtenu {len(regime_analysis)}: {list(regime_analysis.keys())}"
    )

    # bull : 2 fenêtres, sharpe moyen = (2.5 + 3.1) / 2 = 2.8, consistance 100%
    assert regime_analysis["bull"]["n_windows"] == 2
    assert regime_analysis["bull"]["avg_oos_sharpe"] == pytest.approx(2.8, abs=0.01)
    assert regime_analysis["bull"]["consistency"] == 1.0

    # bear : 2 fenêtres (0.5 et -0.3), sharpe moyen = 0.1, consistance 50%
    assert regime_analysis["bear"]["n_windows"] == 2
    assert regime_analysis["bear"]["avg_oos_sharpe"] == pytest.approx(0.1, abs=0.01)
    assert regime_analysis["bear"]["consistency"] == 0.5

    # range : 1 fenêtre
    assert regime_analysis["range"]["n_windows"] == 1

    # crash : 1 fenêtre, oos_sharpe=NaN filtré → avg=0.0
    assert regime_analysis["crash"]["n_windows"] == 1
    assert regime_analysis["crash"]["avg_oos_sharpe"] == 0.0


# ─── Test 5 : Consistance impacte le grade ──────────────────────────────


def test_consistency_impacts_grade():
    """ETH à 68% consistance ne peut pas obtenir 100/100.

    Avec consistance=0.68 → 8/20 pts consistance (tranche ≥60%, au lieu de 20/20).
    Score = 20+20+15+8+10+15 = 88, pas 100.
    """
    from backend.optimization.report import compute_grade

    # Cas parfait SAUF consistance = 68%
    result = compute_grade(
        oos_is_ratio=0.65, mc_p_value=0.01, dsr=0.97,
        stability=0.85, bitget_transfer=0.60,
        consistency=0.68,
    )
    assert result.score < 100, f"Score {result.score} devrait être < 100 avec consistency=68%"
    # 68% → ≥60% bracket → 8/20 pts, score = 20+20+15+8+10+15 = 88
    assert result.score == 88
    assert result.grade == "A"  # 88 ≥ 85


def test_top5_sorted_by_combo_score():
    """Le premier du Top 5 (combo_results) correspond au best combo sélectionné.

    Simule des combo_results non triés et vérifie que le tri par combo_score
    place le best combo en #1.
    """
    combo_results = [
        # Combo A : haut Sharpe, faible consistance, peu de trades
        {"params": {"a": 1}, "oos_sharpe": 9.0, "consistency": 0.40, "oos_trades": 25, "is_best": False},
        # Combo B : best combo — haute consistance + volume
        {"params": {"a": 2}, "oos_sharpe": 5.5, "consistency": 0.95, "oos_trades": 300, "is_best": True},
        # Combo C : Sharpe élevé, consistance moyenne, trades moyens
        {"params": {"a": 3}, "oos_sharpe": 7.0, "consistency": 0.70, "oos_trades": 80, "is_best": False},
    ]

    # Vérifier les scores attendus
    # B: 5.5 * (0.4 + 0.6*0.95) * min(1, 300/100) = 5.5 * 0.97 * 1.0 = 5.335
    # C: 7.0 * (0.4 + 0.6*0.70) * min(1, 80/100) = 7.0 * 0.82 * 0.80 = 4.592
    # A: 9.0 * (0.4 + 0.6*0.40) * min(1, 25/100) = 9.0 * 0.64 * 0.25 = 1.44
    score_b = combo_score(5.5, 0.95, 300)
    score_c = combo_score(7.0, 0.70, 80)
    score_a = combo_score(9.0, 0.40, 25)
    assert score_b > score_c > score_a

    # Tri par combo_score (même logique que walk_forward.py)
    sorted_results = sorted(
        combo_results,
        key=lambda c: combo_score(c["oos_sharpe"], c["consistency"], c["oos_trades"]),
        reverse=True,
    )

    # Le #1 doit être le combo marqué is_best (Combo B)
    assert sorted_results[0]["is_best"] is True, (
        f"Le #1 du top 5 devrait être le best combo, "
        f"mais c'est {sorted_results[0]['params']}"
    )
    assert sorted_results[0]["params"] == {"a": 2}
