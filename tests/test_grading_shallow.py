"""Tests pour la pénalité shallow validation V2 (dégressive)."""

import pytest

from backend.optimization.report import GradeResult, compute_grade

# Helper : params parfaits donnant 100/100 sans pénalité (V2)
PERFECT = dict(
    oos_sharpe=5.72,       # min(20, 5.72*3.5) = min(20, 20.02) ≈ 20.0
    win_rate_oos=1.0,       # 20.0
    tail_ratio=0.0,         # 15.0
    dsr=1.0,                # 15.0
    param_stability=1.0,    # 15.0
    consistency=1.0,        # 10.0
    total_trades=100,
)
# mc = 5.0 → raw_score = 20+20+15+15+15+10+5 = 100


class TestShallowPenalty:
    """Pénalités dégressives par n_windows."""

    def test_shallow_under_12_penalty(self):
        """10 fenêtres → pénalité (24-10)*0.8 = 11.2, score 88.8, grade A."""
        r = compute_grade(**PERFECT, n_windows=10)
        assert r.raw_score == pytest.approx(100, abs=0.1)
        assert r.score == pytest.approx(88.8, abs=0.1)
        assert r.grade == "A"
        assert r.is_shallow is True

    def test_shallow_12_to_17_penalty(self):
        """15 fenêtres → pénalité (24-15)*0.8 = 7.2, score 92.8, grade A."""
        r = compute_grade(**PERFECT, n_windows=15)
        assert r.raw_score == pytest.approx(100, abs=0.1)
        assert r.score == pytest.approx(92.8, abs=0.1)
        assert r.grade == "A"
        assert r.is_shallow is True

    def test_shallow_18_to_23_penalty(self):
        """20 fenêtres → pénalité (24-20)*0.8 = 3.2, score 96.8, grade A."""
        r = compute_grade(**PERFECT, n_windows=20)
        assert r.raw_score == pytest.approx(100, abs=0.1)
        assert r.score == pytest.approx(96.8, abs=0.1)
        assert r.grade == "A"
        assert r.is_shallow is True

    def test_shallow_24_plus_no_penalty(self):
        """30 fenêtres → 0 pénalité, score 100, grade A."""
        r = compute_grade(**PERFECT, n_windows=30)
        assert r.score == pytest.approx(100, abs=0.1)
        assert r.raw_score == pytest.approx(100, abs=0.1)
        assert r.grade == "A"
        assert r.is_shallow is False


class TestShallowFlag:
    """Le flag is_shallow est True si n_windows < 24."""

    def test_shallow_flag_true_at_23(self):
        """23 fenêtres → is_shallow=True."""
        r = compute_grade(**PERFECT, n_windows=23)
        assert r.is_shallow is True
        assert r.score == pytest.approx(100 - 0.8, abs=0.1)

    def test_shallow_flag_false_at_24(self):
        """24 fenêtres → is_shallow=False, pas de pénalité."""
        r = compute_grade(**PERFECT, n_windows=24)
        assert r.is_shallow is False
        assert r.score == pytest.approx(100, abs=0.1)


class TestShallowAndTradesCap:
    """Interaction entre pénalité shallow et garde-fous trades."""

    def test_shallow_and_trades_both_apply(self):
        """n_windows=10 (-11.2) + total_trades=25 (cap C). Score 88.8 (A), cap → C."""
        r = compute_grade(**{**PERFECT, "total_trades": 25}, n_windows=10)
        assert r.score == pytest.approx(88.8, abs=0.1)
        assert r.grade == "C"  # A capped to C par trades < 30
        assert r.is_shallow is True

    def test_shallow_penalty_then_trades_cap(self):
        """n_windows=15 (-7.2) + total_trades=40 (cap B). Score 92.8 (A), cap → B."""
        r = compute_grade(**{**PERFECT, "total_trades": 40}, n_windows=15)
        assert r.score == pytest.approx(92.8, abs=0.1)
        assert r.grade == "B"  # A capped to B par trades < 50
        assert r.is_shallow is True


class TestBackwardCompat:
    """n_windows=None = backward compat, pas de pénalité."""

    def test_n_windows_none_no_penalty(self):
        """n_windows=None → pas de pénalité, is_shallow=False."""
        r = compute_grade(**PERFECT, n_windows=None)
        assert r.score == pytest.approx(100, abs=0.1)
        assert r.is_shallow is False
        assert r.raw_score == pytest.approx(100, abs=0.1)

    def test_n_windows_zero_max_penalty(self):
        """n_windows=0 → pénalité (24-0)*0.8 = 19.2."""
        r = compute_grade(**PERFECT, n_windows=0)
        assert r.score == pytest.approx(80.8, abs=0.1)
        assert r.is_shallow is True
        assert r.raw_score == pytest.approx(100, abs=0.1)


class TestRawScoreAndGradeResult:
    """Vérification du raw_score et du type GradeResult."""

    def test_raw_score_preserved(self):
        """raw_score = score brut avant pénalité."""
        r = compute_grade(**PERFECT, n_windows=15)
        assert r.raw_score == pytest.approx(100, abs=0.1)
        assert r.score == pytest.approx(92.8, abs=0.1)
        assert r.grade == "A"

    def test_grade_result_fields(self):
        """GradeResult a les 4 champs attendus."""
        r = compute_grade(**PERFECT, n_windows=24)
        assert isinstance(r, GradeResult)
        assert hasattr(r, "grade")
        assert hasattr(r, "score")
        assert hasattr(r, "is_shallow")
        assert hasattr(r, "raw_score")

    def test_shallow_does_not_upgrade(self):
        """n_windows=30 mais score brut bas → grade reste bas."""
        r = compute_grade(
            oos_sharpe=1.0,        # min(20, 3.5) = 3.5
            win_rate_oos=0.3,       # 6.0
            tail_ratio=0.8,         # max(0, 15*(1-1.2)) = 0
            dsr=0.5,                # 7.5
            param_stability=0.4,    # 6.0
            consistency=0.3,        # 3.0
            total_trades=100,
            n_windows=30,
        )
        # 3.5+6+0+7.5+6+3+5 = 31 → F
        assert r.is_shallow is False
        assert r.score == r.raw_score
        assert r.grade == "F"
