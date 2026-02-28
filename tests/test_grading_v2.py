"""Tests pour le grading V2 — win_rate_oos, tail_risk_ratio, scoring continu."""

import pytest

from backend.optimization.report import (
    GradeResult,
    compute_grade,
    compute_tail_ratio,
    compute_win_rate_oos,
)


# ─── Tests compute_win_rate_oos ────────────────────────────────────────────


class TestWinRateOOS:
    def test_basic_calculation(self):
        """[+10%, +20%, -5%, +15%, -30%] → 3/5 = 0.6."""
        windows = [
            {"oos_net_return_pct": 10},
            {"oos_net_return_pct": 20},
            {"oos_net_return_pct": -5},
            {"oos_net_return_pct": 15},
            {"oos_net_return_pct": -30},
        ]
        assert compute_win_rate_oos(windows) == pytest.approx(0.6)

    def test_empty_windows(self):
        """Aucune fenêtre → 0.0."""
        assert compute_win_rate_oos([]) == 0.0

    def test_all_positive(self):
        """Toutes positives → 1.0."""
        windows = [{"oos_net_return_pct": r} for r in [5, 10, 3, 1]]
        assert compute_win_rate_oos(windows) == 1.0

    def test_all_negative(self):
        """Toutes négatives → 0.0."""
        windows = [{"oos_net_return_pct": r} for r in [-5, -10, -3]]
        assert compute_win_rate_oos(windows) == 0.0

    def test_zero_return_not_counted(self):
        """Return = 0 n'est pas > 0, donc pas compté comme win."""
        windows = [{"oos_net_return_pct": 0}, {"oos_net_return_pct": 10}]
        assert compute_win_rate_oos(windows) == pytest.approx(0.5)


# ─── Tests compute_tail_ratio ──────────────────────────────────────────────


class TestTailRatio:
    def test_basic_calculation(self):
        """pos_sum=45, neg_bad=-30 → 30/45 = 0.667."""
        windows = [
            {"oos_net_return_pct": 10},
            {"oos_net_return_pct": 20},
            {"oos_net_return_pct": -5},   # pas < -20 → ignoré
            {"oos_net_return_pct": 15},
            {"oos_net_return_pct": -30},  # < -20 → compté
        ]
        result = compute_tail_ratio(windows)
        assert result == pytest.approx(30 / 45, abs=0.001)

    def test_no_gain_returns_one(self):
        """pos_sum=0 → tail_ratio=1.0."""
        windows = [
            {"oos_net_return_pct": -5},
            {"oos_net_return_pct": -25},
        ]
        assert compute_tail_ratio(windows) == 1.0

    def test_no_bad_window(self):
        """Aucune fenêtre < -20% → tail_ratio=0.0."""
        windows = [
            {"oos_net_return_pct": 10},
            {"oos_net_return_pct": -15},  # pas < -20
            {"oos_net_return_pct": 5},
        ]
        assert compute_tail_ratio(windows) == 0.0

    def test_empty_windows(self):
        """Aucune fenêtre → 1.0 (conservateur)."""
        assert compute_tail_ratio([]) == 1.0

    def test_all_catastrophic(self):
        """Toutes < -20%, aucun gain → 1.0."""
        windows = [{"oos_net_return_pct": -25}, {"oos_net_return_pct": -40}]
        assert compute_tail_ratio(windows) == 1.0


# ─── Tests scoring formula ─────────────────────────────────────────────────


class TestScoringFormula:
    def test_exact_calculation(self):
        """Sharpe=5, wr=0.8, tail=0.1, dsr=0.9, stab=0.85, cons=0.7, n=30 → 84.5 → B."""
        r = compute_grade(
            oos_sharpe=5.0,
            win_rate_oos=0.8,
            tail_ratio=0.1,
            dsr=0.9,
            param_stability=0.85,
            consistency=0.7,
            n_windows=30,
        )
        # sharpe = min(20, 5*3.5) = 17.5
        # win_rate = 0.8*20 = 16.0
        # tail = max(0, 15*(1-0.1*1.5)) = 15*0.85 = 12.75
        # dsr = 0.9*15 = 13.5
        # stability = 0.85*15 = 12.75
        # consistency = 0.7*10 = 7.0
        # mc = 5
        # total = 17.5+16+12.75+13.5+12.75+7+5 = 84.5
        assert r.score == pytest.approx(84.5)
        assert r.grade == "B"  # 70 ≤ 84.5 < 85

    def test_perfect_score(self):
        """Métriques parfaites → score 100, grade A."""
        r = compute_grade(
            oos_sharpe=6.0,   # min(20, 21) = 20
            win_rate_oos=1.0,  # 20
            tail_ratio=0.0,    # 15
            dsr=1.0,           # 15
            param_stability=1.0,  # 15
            consistency=1.0,   # 10
            n_windows=30,
        )
        assert r.score == pytest.approx(100)
        assert r.grade == "A"

    def test_zero_score(self):
        """Métriques nulles → score ~5 (forfait MC), grade F."""
        r = compute_grade(
            oos_sharpe=0.0,
            win_rate_oos=0.0,
            tail_ratio=1.0,   # tail = max(0, 15*(1-1.5)) = max(0, -7.5) = 0
            dsr=0.0,
            param_stability=0.0,
            consistency=0.0,
            n_windows=30,
        )
        assert r.score == pytest.approx(5.0)
        assert r.grade == "F"

    def test_sharpe_capped_at_20(self):
        """Sharpe très élevé → plafonné à 20."""
        r = compute_grade(
            oos_sharpe=10.0,  # min(20, 35) = 20
            win_rate_oos=1.0,
            tail_ratio=0.0,
            dsr=1.0,
            param_stability=1.0,
            consistency=1.0,
            n_windows=30,
        )
        assert r.score == pytest.approx(100)  # pas 135


# ─── Tests shallow penalty dégressive ──────────────────────────────────────


class TestShallowPenaltyDegressive:
    def test_penalty_n17(self):
        """n_windows=17 → penalty = (24-17)*0.8 = 5.6."""
        r = compute_grade(
            oos_sharpe=6.0, win_rate_oos=1.0, tail_ratio=0.0,
            dsr=1.0, param_stability=1.0, consistency=1.0,
            n_windows=17,
        )
        assert r.raw_score == pytest.approx(100)
        assert r.score == pytest.approx(100 - 5.6)
        assert r.is_shallow is True

    def test_penalty_n24(self):
        """n_windows=24 → penalty = 0."""
        r = compute_grade(
            oos_sharpe=6.0, win_rate_oos=1.0, tail_ratio=0.0,
            dsr=1.0, param_stability=1.0, consistency=1.0,
            n_windows=24,
        )
        assert r.score == pytest.approx(100)
        assert r.is_shallow is False

    def test_penalty_n10(self):
        """n_windows=10 → penalty = (24-10)*0.8 = 11.2."""
        r = compute_grade(
            oos_sharpe=6.0, win_rate_oos=1.0, tail_ratio=0.0,
            dsr=1.0, param_stability=1.0, consistency=1.0,
            n_windows=10,
        )
        assert r.score == pytest.approx(100 - 11.2)
        assert r.is_shallow is True

    def test_penalty_n0(self):
        """n_windows=0 → penalty = (24-0)*0.8 = 19.2."""
        r = compute_grade(
            oos_sharpe=6.0, win_rate_oos=1.0, tail_ratio=0.0,
            dsr=1.0, param_stability=1.0, consistency=1.0,
            n_windows=0,
        )
        assert r.score == pytest.approx(100 - 19.2)
        assert r.is_shallow is True

    def test_none_no_penalty(self):
        """n_windows=None → backward compat, pas de pénalité."""
        r = compute_grade(
            oos_sharpe=6.0, win_rate_oos=1.0, tail_ratio=0.0,
            dsr=1.0, param_stability=1.0, consistency=1.0,
            n_windows=None,
        )
        assert r.score == pytest.approx(100)
        assert r.is_shallow is False


# ─── Tests trade cap ───────────────────────────────────────────────────────


class TestTradeCap:
    def test_under_30_trades_cap_c(self):
        """< 30 trades → grade plafonné à C."""
        r = compute_grade(
            oos_sharpe=6.0, win_rate_oos=1.0, tail_ratio=0.0,
            dsr=1.0, param_stability=1.0, consistency=1.0,
            total_trades=10, n_windows=30,
        )
        assert r.score == pytest.approx(100)  # score inchangé
        assert r.grade == "C"

    def test_under_50_trades_cap_b(self):
        """< 50 trades → grade plafonné à B."""
        r = compute_grade(
            oos_sharpe=6.0, win_rate_oos=1.0, tail_ratio=0.0,
            dsr=1.0, param_stability=1.0, consistency=1.0,
            total_trades=40, n_windows=30,
        )
        assert r.score == pytest.approx(100)
        assert r.grade == "B"

    def test_above_50_trades_no_cap(self):
        """≥ 50 trades → pas de plafond."""
        r = compute_grade(
            oos_sharpe=6.0, win_rate_oos=1.0, tail_ratio=0.0,
            dsr=1.0, param_stability=1.0, consistency=1.0,
            total_trades=100, n_windows=30,
        )
        assert r.grade == "A"


# ─── Tests backward compat structure ───────────────────────────────────────


class TestBackwardCompat:
    def test_grade_result_fields(self):
        """GradeResult a les 4 champs attendus."""
        r = compute_grade(
            oos_sharpe=3.0, win_rate_oos=0.7, tail_ratio=0.2,
            dsr=0.8, param_stability=0.7, consistency=0.6,
            n_windows=24,
        )
        assert isinstance(r, GradeResult)
        assert hasattr(r, "grade")
        assert hasattr(r, "score")
        assert hasattr(r, "is_shallow")
        assert hasattr(r, "raw_score")
        assert isinstance(r.score, float)
        assert isinstance(r.raw_score, float)
