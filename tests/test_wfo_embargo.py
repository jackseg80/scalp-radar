"""Tests Sprint 40a #3 : Embargo IS→OOS dans _build_windows() + param_grids.yaml.

L'embargo ajoute un tampon de N jours entre la fin de la fenêtre IS et le
début de la fenêtre OOS, pour éviter que des positions ouvertes en fin d'IS
contaminent les premiers jours OOS.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
import yaml


# ─── Helpers ──────────────────────────────────────────────────────────────


def _make_optimizer() -> "WalkForwardOptimizer":
    from backend.optimization.walk_forward import WalkForwardOptimizer
    opt = WalkForwardOptimizer.__new__(WalkForwardOptimizer)
    # Config minimale
    opt.config = MagicMock()
    opt.config.assets = []
    return opt


# ─── Tests _build_windows() ───────────────────────────────────────────────


class TestBuildWindowsEmbargo:

    def test_embargo_0_retro_compat(self):
        """embargo_days=0 (défaut) → oos_start = is_end (comportement original)."""
        opt = _make_optimizer()

        data_start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        data_end = datetime(2024, 6, 1, tzinfo=timezone.utc)

        windows = opt._build_windows(
            data_start=data_start,
            data_end=data_end,
            is_days=180,
            oos_days=60,
            step_days=60,
            embargo_days=0,
        )

        assert len(windows) > 0
        is_start, is_end, oos_start, oos_end = windows[0]
        # Avec embargo=0, oos_start = is_end exactement
        assert oos_start == is_end, (
            f"embargo=0 doit donner oos_start=is_end. "
            f"oos_start={oos_start}, is_end={is_end}"
        )

    def test_embargo_7_decale_oos(self):
        """embargo_days=7 → oos_start = is_end + 7j."""
        opt = _make_optimizer()

        data_start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        data_end = datetime(2024, 12, 1, tzinfo=timezone.utc)

        windows = opt._build_windows(
            data_start=data_start,
            data_end=data_end,
            is_days=180,
            oos_days=60,
            step_days=60,
            embargo_days=7,
        )

        assert len(windows) > 0
        is_start, is_end, oos_start, oos_end = windows[0]
        expected_oos_start = is_end + timedelta(days=7)
        assert oos_start == expected_oos_start, (
            f"embargo=7 doit donner oos_start=is_end+7j. "
            f"oos_start={oos_start}, expected={expected_oos_start}"
        )

    def test_embargo_reduces_window_count(self):
        """Avec embargo, il y a moins de fenêtres qu'sans (données OOS raccourcies)."""
        opt = _make_optimizer()

        data_start = datetime(2022, 1, 1, tzinfo=timezone.utc)
        data_end = datetime(2024, 1, 1, tzinfo=timezone.utc)

        windows_no_embargo = opt._build_windows(
            data_start=data_start,
            data_end=data_end,
            is_days=180,
            oos_days=60,
            step_days=60,
            embargo_days=0,
        )

        windows_with_embargo = opt._build_windows(
            data_start=data_start,
            data_end=data_end,
            is_days=180,
            oos_days=60,
            step_days=60,
            embargo_days=7,
        )

        # L'embargo consomme 7j de données → potentiellement moins de fenêtres
        # En tout cas, pas plus de fenêtres
        assert len(windows_with_embargo) <= len(windows_no_embargo), (
            f"Avec embargo, au plus autant de fenêtres qu'sans. "
            f"no_embargo={len(windows_no_embargo)}, "
            f"with_embargo={len(windows_with_embargo)}"
        )

    def test_all_windows_embargo_consistent(self):
        """Toutes les fenêtres ont le même écart oos_start - is_end."""
        opt = _make_optimizer()

        data_start = datetime(2022, 1, 1, tzinfo=timezone.utc)
        data_end = datetime(2024, 6, 1, tzinfo=timezone.utc)

        embargo = 7
        windows = opt._build_windows(
            data_start=data_start,
            data_end=data_end,
            is_days=180,
            oos_days=60,
            step_days=60,
            embargo_days=embargo,
        )

        assert len(windows) > 0
        for is_start, is_end, oos_start, oos_end in windows:
            gap = (oos_start - is_end).days
            assert gap == embargo, (
                f"Chaque fenêtre doit avoir embargo={embargo}j. "
                f"Trouvé gap={gap}j pour is_end={is_end}, oos_start={oos_start}"
            )


# ─── Test param_grids.yaml a embargo_days ─────────────────────────────────


class TestParamGridsEmbargo:

    def test_grid_atr_has_embargo_7(self):
        """grid_atr.wfo.embargo_days = 7 dans param_grids.yaml."""
        with open("config/param_grids.yaml") as f:
            data = yaml.safe_load(f)

        assert "grid_atr" in data, "grid_atr absent de param_grids.yaml"
        assert "wfo" in data["grid_atr"], "Pas de section wfo dans grid_atr"
        assert data["grid_atr"]["wfo"].get("embargo_days") == 7, (
            f"grid_atr.wfo.embargo_days devrait être 7, "
            f"trouvé {data['grid_atr']['wfo'].get('embargo_days')}"
        )

    def test_grid_boltrend_has_embargo_7(self):
        """grid_boltrend.wfo.embargo_days = 7 dans param_grids.yaml."""
        with open("config/param_grids.yaml") as f:
            data = yaml.safe_load(f)

        assert "grid_boltrend" in data, "grid_boltrend absent de param_grids.yaml"
        wfo = data["grid_boltrend"].get("wfo", {})
        assert wfo.get("embargo_days") == 7, (
            f"grid_boltrend.wfo.embargo_days devrait être 7, "
            f"trouvé {wfo.get('embargo_days')}"
        )

    def test_strategies_without_embargo_still_work(self):
        """Stratégies sans embargo_days dans wfo: → défaut 0 (rétrocompat)."""
        # Vérifier que walk_forward.py utilise embargo_days=0 par défaut
        from backend.optimization.walk_forward import WalkForwardOptimizer
        import inspect
        source = inspect.getsource(WalkForwardOptimizer._build_windows)
        # Le paramètre a une valeur par défaut de 0
        assert "embargo_days: int = 0" in source or "embargo_days=0" in source, (
            "_build_windows doit accepter embargo_days=0 comme défaut"
        )
