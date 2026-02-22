"""Tests pour la pénalité shallow validation (n_windows < 24) — Sprint 38."""

from backend.optimization.report import GradeResult, compute_grade

# Helper : params parfaits donnant 100/100 sans pénalité
PERFECT = dict(
    oos_is_ratio=0.65,
    mc_p_value=0.01,
    dsr=0.97,
    stability=0.85,
    bitget_transfer=0.60,
    consistency=0.95,
    total_trades=100,
    transfer_significant=True,
    bitget_trades=20,
)


class TestShallowPenalty:
    """Pénalités par palier de n_windows."""

    def test_shallow_under_12_penalty_25(self):
        """10 fenêtres → pénalité -25, score 75, grade B."""
        r = compute_grade(**PERFECT, n_windows=10)
        assert r.raw_score == 100
        assert r.score == 75
        assert r.grade == "B"
        assert r.is_shallow is True

    def test_shallow_12_to_17_penalty_20(self):
        """15 fenêtres → pénalité -20, score 80, grade B."""
        r = compute_grade(**PERFECT, n_windows=15)
        assert r.raw_score == 100
        assert r.score == 80
        assert r.grade == "B"
        assert r.is_shallow is True

    def test_shallow_18_to_23_penalty_10(self):
        """20 fenêtres → pénalité -10, score 90, grade A."""
        r = compute_grade(**PERFECT, n_windows=20)
        assert r.raw_score == 100
        assert r.score == 90
        assert r.grade == "A"
        assert r.is_shallow is True

    def test_shallow_24_plus_no_penalty(self):
        """30 fenêtres → 0 pénalité, score 100, grade A."""
        r = compute_grade(**PERFECT, n_windows=30)
        assert r.score == 100
        assert r.raw_score == 100
        assert r.grade == "A"
        assert r.is_shallow is False


class TestShallowFlag:
    """Le flag is_shallow est True si n_windows < 24."""

    def test_shallow_flag_true_at_23(self):
        """23 fenêtres → is_shallow=True (même si pénalité -10 seulement)."""
        r = compute_grade(**PERFECT, n_windows=23)
        assert r.is_shallow is True
        assert r.score == 90  # -10

    def test_shallow_flag_false_at_24(self):
        """24 fenêtres → is_shallow=False, pas de pénalité."""
        r = compute_grade(**PERFECT, n_windows=24)
        assert r.is_shallow is False
        assert r.score == 100


class TestShallowAndTradesCap:
    """Interaction entre pénalité shallow et garde-fous trades."""

    def test_shallow_and_trades_both_apply(self):
        """n_windows=10 (-25) + total_trades=25 (cap C). Score 75 (B), cap → C."""
        r = compute_grade(**{**PERFECT, "total_trades": 25}, n_windows=10)
        assert r.score == 75  # 100 - 25
        assert r.grade == "C"  # B capped to C par trades < 30
        assert r.is_shallow is True

    def test_shallow_penalty_then_trades_cap(self):
        """n_windows=15 (-20) + total_trades=40 (cap B). Score 80 (B), cap B → B."""
        r = compute_grade(**{**PERFECT, "total_trades": 40}, n_windows=15)
        assert r.score == 80  # 100 - 20
        assert r.grade == "B"  # B, cap B = inchangé
        assert r.is_shallow is True


class TestBackwardCompat:
    """n_windows=None = backward compat, pas de pénalité."""

    def test_n_windows_none_no_penalty(self):
        """n_windows=None → pas de pénalité, is_shallow=False."""
        r = compute_grade(**PERFECT, n_windows=None)
        assert r.score == 100
        assert r.is_shallow is False
        assert r.raw_score == 100

    def test_n_windows_zero_max_penalty(self):
        """n_windows=0 → pénalité max -25 (pas d'amnistie !)."""
        r = compute_grade(**PERFECT, n_windows=0)
        assert r.score == 75
        assert r.is_shallow is True
        assert r.raw_score == 100


class TestRawScoreAndGradeResult:
    """Vérification du raw_score et du type GradeResult."""

    def test_raw_score_preserved(self):
        """raw_score = score brut avant pénalité."""
        r = compute_grade(**PERFECT, n_windows=15)
        assert r.raw_score == 100
        assert r.score == 80  # 100 - 20
        assert r.grade == "B"

    def test_grade_result_fields(self):
        """GradeResult a les 4 champs attendus."""
        r = compute_grade(**PERFECT, n_windows=24)
        assert isinstance(r, GradeResult)
        assert hasattr(r, "grade")
        assert hasattr(r, "score")
        assert hasattr(r, "is_shallow")
        assert hasattr(r, "raw_score")

    def test_shallow_does_not_upgrade(self):
        """n_windows=30 mais score brut 50 → grade D, pas d'amélioration."""
        r = compute_grade(
            oos_is_ratio=0.45,   # 12 pts
            mc_p_value=0.08,     # 12 pts
            dsr=0.92,            # 12 pts
            stability=0.45,      # 4 pts
            bitget_transfer=0.35, # 5 pts
            consistency=0.55,    # 4 pts
            total_trades=100,
            n_windows=30,
        )
        # Score ~49, pas de pénalité shallow, grade reste bas
        assert r.is_shallow is False
        assert r.score == r.raw_score
        assert r.grade in ("D", "F")
