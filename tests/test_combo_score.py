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

    # Vérification valeurs attendues
    # A = 7.5 * (0.4 + 0.6*0.93) * min(1, 300/50) = 7.5 * 0.958 * 1.0 = 7.185
    # B = 8.8 * (0.4 + 0.6*0.63) * min(1, 32/50) = 8.8 * 0.778 * 0.64 = 4.382
    assert score_a == pytest.approx(7.185, abs=0.01)
    assert score_b == pytest.approx(4.382, abs=0.01)


# ─── Test 2 : Pénalité pour faible volume de trades ─────────────────────

def test_combo_score_penalizes_low_trades():
    """Combo avec 20 trades pénalisé vs combo avec 100 trades (même Sharpe/consistance)."""
    score_low = combo_score(oos_sharpe=2.0, consistency=0.8, total_trades=20)
    score_high = combo_score(oos_sharpe=2.0, consistency=0.8, total_trades=100)

    # Le trade_factor pénalise : 20/50=0.4 vs min(1,100/50)=1.0
    assert score_high > score_low
    assert score_high / score_low == pytest.approx(1.0 / 0.4, abs=0.01)


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


# ─── Test 4 : regime_analysis utilise le best scored combo ──────────────

def test_regime_analysis_uses_best_scored_combo():
    """Vérifier que le combo sélectionné pour regime_analysis est celui
    avec le meilleur combo_score (pas le Sharpe max)."""

    # Simuler 2 combos dans combo_accumulator avec window_regimes
    combo_a_params = {"ma_period": 7, "num_levels": 3}
    combo_b_params = {"ma_period": 14, "num_levels": 5}
    key_a = json.dumps(combo_a_params, sort_keys=True)
    key_b = json.dumps(combo_b_params, sort_keys=True)

    # Combo A : Sharpe max (8.0) mais faible consistance + peu de trades
    combo_accumulator = {
        key_a: [
            {"is_sharpe": 10.0, "is_return_pct": 20, "is_trades": 10,
             "oos_sharpe": 8.0, "oos_return_pct": 15, "oos_trades": 8, "window_idx": 0},
            {"is_sharpe": 9.0, "is_return_pct": 18, "is_trades": 12,
             "oos_sharpe": None, "oos_return_pct": None, "oos_trades": None, "window_idx": 1},
            {"is_sharpe": 11.0, "is_return_pct": 22, "is_trades": 10,
             "oos_sharpe": 8.5, "oos_return_pct": 16, "oos_trades": 8, "window_idx": 2},
        ],
        # Combo B : Sharpe plus bas (3.0) mais haute consistance + beaucoup de trades
        key_b: [
            {"is_sharpe": 4.0, "is_return_pct": 10, "is_trades": 80,
             "oos_sharpe": 3.0, "oos_return_pct": 8, "oos_trades": 70, "window_idx": 0},
            {"is_sharpe": 3.5, "is_return_pct": 9, "is_trades": 85,
             "oos_sharpe": 2.8, "oos_return_pct": 7, "oos_trades": 75, "window_idx": 1},
            {"is_sharpe": 4.2, "is_return_pct": 11, "is_trades": 90,
             "oos_sharpe": 3.2, "oos_return_pct": 9, "oos_trades": 80, "window_idx": 2},
        ],
    }

    window_regimes = [
        {"regime": "bull", "return_pct": 15.0, "max_dd_pct": -5.0},
        {"regime": "range", "return_pct": 2.0, "max_dd_pct": -8.0},
        {"regime": "bear", "return_pct": -12.0, "max_dd_pct": -20.0},
    ]

    # Construire combo_results (même logique que walk_forward.py)
    import numpy as np

    combo_results = []
    for params_key, window_data in combo_accumulator.items():
        params = json.loads(params_key)
        oos_sharpes = [d["oos_sharpe"] for d in window_data if d["oos_sharpe"] is not None]
        oos_trades_list = [d["oos_trades"] for d in window_data if d["oos_trades"] is not None]
        n_oos_positive = sum(1 for s in oos_sharpes if s > 0)
        consistency_c = n_oos_positive / len(oos_sharpes) if oos_sharpes else 0.0
        avg_oos = float(np.nanmean(oos_sharpes)) if oos_sharpes else 0.0
        total_trades = sum(oos_trades_list) if oos_trades_list else 0

        combo_results.append({
            "params": params,
            "params_key": params_key,
            "oos_sharpe": avg_oos,
            "consistency": consistency_c,
            "oos_trades": total_trades,
            "is_best": False,
        })

    # Sélectionner le best combo par combo_score
    best_combo = max(
        combo_results,
        key=lambda c: combo_score(c["oos_sharpe"], c["consistency"], c["oos_trades"]),
    )
    best_combo["is_best"] = True
    recommended = best_combo["params"]
    recommended_key = json.dumps(recommended, sort_keys=True)

    # Combo B devrait être sélectionné (haute consistance + volume)
    assert recommended == combo_b_params, (
        f"Le best combo devrait être B ({combo_b_params}), pas A ({combo_a_params})"
    )

    # Vérifier que regime_analysis utilise le combo B
    best_window_data = combo_accumulator.get(recommended_key, [])
    assert len(best_window_data) == 3  # Combo B a 3 fenêtres

    # Construire regime_analysis sur combo B
    regime_groups = {}
    for wd in best_window_data:
        w_idx = wd.get("window_idx", -1)
        if 0 <= w_idx < len(window_regimes):
            regime = window_regimes[w_idx]["regime"]
            regime_groups.setdefault(regime, []).append(wd)

    # Combo B a des données dans les 3 régimes (bull, range, bear)
    assert len(regime_groups) == 3
    assert "bull" in regime_groups
    assert "range" in regime_groups
    assert "bear" in regime_groups

    # Chaque régime a des trades > 0 (pas des 0% partout)
    for regime, entries in regime_groups.items():
        oos_sharpes = [e["oos_sharpe"] for e in entries if e["oos_sharpe"] is not None]
        assert len(oos_sharpes) > 0, f"Régime {regime} n'a pas de données OOS"
        assert all(s > 0 for s in oos_sharpes), f"Régime {regime} a des Sharpe <= 0"
