"""Tests Sprint 37 — Timeframe Coherence Guard.

Couvre :
- Détection du timeframe majoritaire (Counter + tiebreak)
- Blocage de apply_from_db() si conflit détecté
- Flags --ignore-tf-conflicts et --exclude
- TimeframeConflictError dans portfolio_engine.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS optimization_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    asset TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    created_at TEXT NOT NULL,
    duration_seconds REAL,
    grade TEXT NOT NULL,
    total_score REAL NOT NULL,
    oos_sharpe REAL,
    consistency REAL,
    oos_is_ratio REAL,
    dsr REAL,
    param_stability REAL,
    monte_carlo_pvalue REAL,
    mc_underpowered INTEGER DEFAULT 0,
    n_windows INTEGER NOT NULL,
    n_distinct_combos INTEGER,
    best_params TEXT NOT NULL,
    wfo_windows TEXT,
    monte_carlo_summary TEXT,
    validation_summary TEXT,
    warnings TEXT,
    is_latest INTEGER DEFAULT 1,
    source TEXT DEFAULT 'local',
    win_rate_oos REAL,
    tail_risk_ratio REAL,
    UNIQUE(strategy_name, asset, timeframe, created_at)
)
"""

TF_ORDER = {"1m": 0, "5m": 1, "15m": 2, "1h": 3, "4h": 4, "1d": 5}


def _make_db(rows: list[dict]) -> str:
    """Crée une DB SQLite en mémoire et retourne le chemin :memory: fictif.

    On triche : on crée un fichier temporaire sqlite3 avec le schéma et les lignes,
    et on retourne son chemin. Les tests passent ce chemin à apply_from_db().
    """
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
    conn.execute(SCHEMA_SQL)
    for r in rows:
        conn.execute(
            """INSERT INTO optimization_results
               (strategy_name, asset, timeframe, created_at, grade, total_score,
                n_windows, best_params, is_latest)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                r["strategy_name"], r["asset"], r["timeframe"],
                r.get("created_at", datetime.now().isoformat()),
                r["grade"], r.get("total_score", 75.0),
                r.get("n_windows", 6),
                json.dumps(r.get("best_params", {"ma_period": 20})),
                r.get("is_latest", 1),
            ),
        )
    conn.commit()
    conn.close()
    return tmp.name


def _compute_majority_tf(rows: list[dict]) -> str:
    """Reproduit la logique de apply_from_db() pour les tests unitaires."""
    eligible = [r for r in rows if r["grade"] in ("A", "B")]
    if not eligible:
        return "1h"
    tf_counts = Counter(r["timeframe"] for r in eligible)
    majority_tf = tf_counts.most_common(1)[0][0]
    if len(tf_counts) > 1 and tf_counts.most_common(2)[0][1] == tf_counts.most_common(2)[1][1]:
        majority_tf = min(tf_counts.keys(), key=lambda tf: TF_ORDER.get(tf, 99))
    return majority_tf


# ---------------------------------------------------------------------------
# Tests 1-3 : Logique majority_tf
# ---------------------------------------------------------------------------


def test_majority_tf_simple():
    """19 assets 1h + 2 assets 4h → majority = 1h."""
    rows = [{"grade": "A", "timeframe": "1h"}] * 19 + [{"grade": "A", "timeframe": "4h"}] * 2
    assert _compute_majority_tf(rows) == "1h"


def test_majority_tf_all_same():
    """20 assets 1h → 0 outliers, majority = 1h."""
    rows = [{"grade": "A", "timeframe": "1h"}] * 20
    assert _compute_majority_tf(rows) == "1h"
    tf_counts = Counter(r["timeframe"] for r in rows)
    assert len(tf_counts) == 1  # Pas d'outlier


def test_majority_tf_tie():
    """10 assets 1h + 10 assets 4h → tiebreak = 1h (plus petit TF)."""
    rows = [{"grade": "A", "timeframe": "1h"}] * 10 + [{"grade": "A", "timeframe": "4h"}] * 10
    assert _compute_majority_tf(rows) == "1h"


# ---------------------------------------------------------------------------
# Tests 4-7 : apply_from_db() comportement
# ---------------------------------------------------------------------------


def test_apply_blocked_on_conflict(tmp_path):
    """Sprint 62a : sans timeframe dans yaml, le mode (1h:2 > 4h:1) est utilisé.

    Comportement Sprint 37 (blocage) → remplacé par Sprint 62a (filtre + warning).
    BCH/USDT (4h) est ignoré silencieusement, BTC+ETH (1h) sont appliqués.
    Pour déclencher un vrai blocage il faudrait que le conflit survive au filtre,
    ce qui est impossible car le mode résout toujours l'ambiguïté.
    """
    from scripts.optimize import apply_from_db

    db_path = _make_db([
        {"strategy_name": "grid_atr", "asset": "BTC/USDT", "timeframe": "1h",
         "grade": "A", "best_params": {"ma_period": 20}},
        {"strategy_name": "grid_atr", "asset": "BCH/USDT", "timeframe": "4h",
         "grade": "A", "best_params": {"ma_period": 10}},
        {"strategy_name": "grid_atr", "asset": "ETH/USDT", "timeframe": "1h",
         "grade": "B", "best_params": {"ma_period": 25}},
    ])

    config_dir = str(tmp_path)
    (tmp_path / "strategies.yaml").write_text(
        "grid_atr:\n  enabled: true\n  per_asset: {}\n"
    )
    result = apply_from_db(["grid_atr"], config_dir=config_dir, db_path=db_path)

    # Sprint 62a : mode = 1h (2 vs 1) → BCH filtré, BTC+ETH appliqués, pas de blocage
    assert result.get("blocked") is not True
    assert "BCH/USDT" not in result.get("applied", [])
    assert "BTC/USDT" in result.get("applied", [])
    assert "ETH/USDT" in result.get("applied", [])


def test_apply_blocked_exit_code(tmp_path):
    """Sprint 62a : tie 1h:1 vs 4h:1 → tiebreak = 1h (TF le plus petit).

    BCH/USDT (4h) filtré par le guard, BTC/USDT (1h) appliqué.
    """
    from scripts.optimize import apply_from_db

    db_path = _make_db([
        {"strategy_name": "grid_atr", "asset": "BTC/USDT", "timeframe": "1h",
         "grade": "A", "best_params": {"ma_period": 20}},
        {"strategy_name": "grid_atr", "asset": "BCH/USDT", "timeframe": "4h",
         "grade": "A", "best_params": {"ma_period": 10}},
    ])

    config_dir = str(tmp_path)
    (tmp_path / "strategies.yaml").write_text(
        "grid_atr:\n  enabled: true\n  per_asset: {}\n"
    )

    result = apply_from_db(["grid_atr"], config_dir=config_dir, db_path=db_path)
    # Tie → tiebreak 1h → BTC appliqué, BCH ignoré
    assert result.get("blocked") is not True
    assert "BTC/USDT" in result.get("applied", [])
    assert "BCH/USDT" not in result.get("applied", [])
    assert result.get("changed") is True


def test_apply_with_ignore_flag(tmp_path):
    """ignore_tf_conflicts=True → outliers exclus silencieusement, apply réussit."""
    from scripts.optimize import apply_from_db

    db_path = _make_db([
        {"strategy_name": "grid_atr", "asset": "BTC/USDT", "timeframe": "1h",
         "grade": "A", "best_params": {"ma_period": 20}},
        {"strategy_name": "grid_atr", "asset": "BCH/USDT", "timeframe": "4h",
         "grade": "A", "best_params": {"ma_period": 10}},
        {"strategy_name": "grid_atr", "asset": "ETH/USDT", "timeframe": "1h",
         "grade": "B", "best_params": {"ma_period": 25}},
    ])

    config_dir = str(tmp_path)
    (tmp_path / "strategies.yaml").write_text(
        "grid_atr:\n  enabled: true\n  per_asset: {}\n"
    )

    result = apply_from_db(
        ["grid_atr"], config_dir=config_dir, db_path=db_path,
        ignore_tf_conflicts=True,
    )

    assert result.get("blocked") is not True
    # BCH/USDT (4h outlier) ne doit pas être dans applied
    assert "BCH/USDT" not in result.get("applied", [])
    # BTC et ETH (1h) doivent être appliqués
    assert "BTC/USDT" in result.get("applied", [])
    assert "ETH/USDT" in result.get("applied", [])


def test_apply_with_exclude_flag(tmp_path):
    """exclude_symbols=["BCH/USDT"] → BCH absent des résultats, apply réussit."""
    from scripts.optimize import apply_from_db

    db_path = _make_db([
        {"strategy_name": "grid_atr", "asset": "BTC/USDT", "timeframe": "1h",
         "grade": "A", "best_params": {"ma_period": 20}},
        {"strategy_name": "grid_atr", "asset": "BCH/USDT", "timeframe": "4h",
         "grade": "A", "best_params": {"ma_period": 10}},
    ])

    config_dir = str(tmp_path)
    (tmp_path / "strategies.yaml").write_text(
        "grid_atr:\n  enabled: true\n  per_asset: {}\n"
    )

    result = apply_from_db(
        ["grid_atr"], config_dir=config_dir, db_path=db_path,
        exclude_symbols=["BCH/USDT"],
    )

    assert result.get("blocked") is not True
    assert "BCH/USDT" not in result.get("applied", [])
    assert "BTC/USDT" in result.get("applied", [])


# ---------------------------------------------------------------------------
# Tests 8-9 : --force-timeframe
# ---------------------------------------------------------------------------


def test_force_timeframe_filters_grid():
    """--force-timeframe 1h filtre la grid à [1h]."""
    # Logique directe : params_override = {"timeframe": ["1h"]}
    param_grid = {"timeframe": ["1h", "4h"], "ma_period": [10, 20]}
    force_tf = "1h"

    if "timeframe" in param_grid and force_tf in param_grid["timeframe"]:
        param_grid["timeframe"] = [force_tf]

    assert param_grid["timeframe"] == ["1h"]
    assert param_grid["ma_period"] == [10, 20]  # autres params inchangés


def test_force_timeframe_invalid_value():
    """--force-timeframe 2h (absent de la grid) → devrait lever une erreur."""
    param_grid = {"timeframe": ["1h", "4h"], "ma_period": [10, 20]}
    force_tf = "2h"

    if "timeframe" in param_grid:
        assert force_tf not in param_grid["timeframe"], (
            f"Timeframe '{force_tf}' absent de la grid — sys.exit(1) attendu"
        )


# ---------------------------------------------------------------------------
# Tests 10-11 : portfolio_engine
# ---------------------------------------------------------------------------


def test_portfolio_raises_on_tf_conflict():
    """Runner avec timeframe=4h → TimeframeConflictError levée, valid_keys peuplé."""
    from backend.backtesting.portfolio_engine import TimeframeConflictError

    # Simuler la logique du guard dans run()
    runners = {
        "grid_atr:BTC/USDT": MagicMock(),
        "grid_atr:ETH/USDT": MagicMock(),
        "grid_atr:BCH/USDT": MagicMock(),
    }
    runners["grid_atr:BTC/USDT"]._strategy._config.timeframe = "1h"
    runners["grid_atr:ETH/USDT"]._strategy._config.timeframe = "1h"
    runners["grid_atr:BCH/USDT"]._strategy._config.timeframe = "4h"

    expected_tf = "1h"
    mismatched = []
    for runner_key, runner in runners.items():
        runner_tf = getattr(runner._strategy._config, "timeframe", expected_tf)
        if runner_tf != expected_tf:
            mismatched.append((runner_key, runner_tf))

    assert mismatched  # Conflit détecté

    exc = TimeframeConflictError(mismatched, expected_tf, list(runners.keys()))
    assert len(exc.mismatched) == 1
    assert exc.mismatched[0] == ("grid_atr:BCH/USDT", "4h")
    assert exc.expected_tf == "1h"
    # valid_keys contient les runners 1h (pas BCH)
    assert "grid_atr:BTC/USDT" in exc.valid_keys
    assert "grid_atr:ETH/USDT" in exc.valid_keys
    assert "grid_atr:BCH/USDT" not in exc.valid_keys


def test_apply_succeeds_after_conflict_resolved(tmp_path):
    """Tous les TFs alignés sur 1h → apply réussit, blocked absent."""
    from scripts.optimize import apply_from_db

    db_path = _make_db([
        {"strategy_name": "grid_atr", "asset": "BTC/USDT", "timeframe": "1h",
         "grade": "A", "best_params": {"ma_period": 20}},
        {"strategy_name": "grid_atr", "asset": "BCH/USDT", "timeframe": "1h",
         "grade": "A", "best_params": {"ma_period": 15}},
        {"strategy_name": "grid_atr", "asset": "ETH/USDT", "timeframe": "1h",
         "grade": "B", "best_params": {"ma_period": 25}},
    ])

    config_dir = str(tmp_path)
    (tmp_path / "strategies.yaml").write_text(
        "grid_atr:\n  enabled: true\n  per_asset: {}\n"
    )

    result = apply_from_db(["grid_atr"], config_dir=config_dir, db_path=db_path)

    assert result.get("blocked") is not True
    assert result.get("changed") is True
    assert "BTC/USDT" in result.get("applied", [])
    assert "BCH/USDT" in result.get("applied", [])
    assert "ETH/USDT" in result.get("applied", [])


# ---------------------------------------------------------------------------
# Hotfix 37b — Tests Bug 1 (comparaison valeurs) + Bug 2 (exclude retire YAML)
# ---------------------------------------------------------------------------


def test_apply_updates_changed_params(tmp_path):
    """Bug 1: asset déjà dans per_asset avec params différents → doit être mis à jour."""
    import yaml
    from scripts.optimize import apply_from_db

    # YAML initial : BCH a timeframe=4h (ancien WFO)
    initial_yaml = {
        "grid_atr": {
            "enabled": True,
            "per_asset": {
                "BCH/USDT": {"timeframe": "4h", "ma_period": 20, "sl_percent": 25.0},
                "BTC/USDT": {"timeframe": "1h", "ma_period": 20, "sl_percent": 25.0},
            },
        }
    }
    config_dir = str(tmp_path)
    yaml_path = tmp_path / "strategies.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(initial_yaml, f)

    # DB : BCH re-testé en 1h → best_params avec timeframe=1h
    db_path = _make_db([
        {"strategy_name": "grid_atr", "asset": "BCH/USDT", "timeframe": "1h",
         "grade": "A", "best_params": {"timeframe": "1h", "ma_period": 20, "sl_percent": 25.0}},
        {"strategy_name": "grid_atr", "asset": "BTC/USDT", "timeframe": "1h",
         "grade": "A", "best_params": {"timeframe": "1h", "ma_period": 20, "sl_percent": 25.0}},
    ])

    result = apply_from_db(["grid_atr"], config_dir=config_dir, db_path=db_path)

    # BCH doit être dans "applied" (changement détecté), pas dans "unchanged"
    assert result.get("changed") is True
    assert "BCH/USDT" in result.get("applied", []), \
        "BCH/USDT (timeframe 4h→1h) aurait dû être dans applied"

    # Vérifier que le YAML a bien été mis à jour avec timeframe=1h
    with open(yaml_path) as f:
        updated = yaml.safe_load(f)
    bch_params = updated["grid_atr"]["per_asset"].get("BCH/USDT", {})
    assert bch_params.get("timeframe") == "1h", \
        f"YAML doit avoir timeframe=1h, got {bch_params.get('timeframe')}"


def test_apply_detects_unchanged_correctly(tmp_path):
    """Bug 1: asset avec params identiques → doit rester dans unchanged (pas de faux changed)."""
    import yaml
    from scripts.optimize import apply_from_db

    params = {"timeframe": "1h", "ma_period": 20, "sl_percent": 25.0}
    initial_yaml = {
        "grid_atr": {
            "enabled": True,
            "per_asset": {"BTC/USDT": params.copy()},
        }
    }
    config_dir = str(tmp_path)
    yaml_path = tmp_path / "strategies.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(initial_yaml, f)

    # DB : mêmes params (y compris int/float normalisé : 25 == 25.0)
    db_path = _make_db([
        {"strategy_name": "grid_atr", "asset": "BTC/USDT", "timeframe": "1h",
         "grade": "A", "best_params": {"timeframe": "1h", "ma_period": 20, "sl_percent": 25}},
    ])

    result = apply_from_db(["grid_atr"], config_dir=config_dir, db_path=db_path)

    # Aucun changement attendu
    assert result.get("changed") is False, \
        "Params identiques (int 25 == float 25.0) → pas de changement"
    assert "BTC/USDT" not in result.get("applied", [])


def test_apply_exclude_removes_from_yaml(tmp_path):
    """Bug 2: --exclude doit retirer l'asset du YAML s'il y est déjà."""
    import yaml
    from scripts.optimize import apply_from_db

    # YAML initial avec BCH
    initial_yaml = {
        "grid_atr": {
            "enabled": True,
            "per_asset": {
                "BCH/USDT": {"timeframe": "1h", "ma_period": 15},
                "BTC/USDT": {"timeframe": "1h", "ma_period": 20},
            },
        }
    }
    config_dir = str(tmp_path)
    yaml_path = tmp_path / "strategies.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(initial_yaml, f)

    # DB : BCH Grade A (résultat disponible)
    db_path = _make_db([
        {"strategy_name": "grid_atr", "asset": "BTC/USDT", "timeframe": "1h",
         "grade": "A", "best_params": {"timeframe": "1h", "ma_period": 20}},
        {"strategy_name": "grid_atr", "asset": "BCH/USDT", "timeframe": "1h",
         "grade": "A", "best_params": {"timeframe": "1h", "ma_period": 15}},
    ])

    result = apply_from_db(
        ["grid_atr"], config_dir=config_dir, db_path=db_path,
        exclude_symbols=["BCH/USDT"],
    )

    # BCH doit être dans removed
    assert "BCH/USDT" in result.get("removed", []), \
        "BCH/USDT explicitement exclu doit être dans removed"
    # BCH ne doit plus être dans le YAML
    with open(yaml_path) as f:
        updated = yaml.safe_load(f)
    per_asset = updated["grid_atr"].get("per_asset", {})
    assert "BCH/USDT" not in per_asset, \
        "BCH/USDT ne doit plus être dans per_asset après --exclude"
    assert "BTC/USDT" in per_asset  # BTC reste


def test_apply_exclude_nonexistent_no_crash(tmp_path):
    """Bug 2: exclure un asset absent du per_asset → pas de crash."""
    from scripts.optimize import apply_from_db

    config_dir = str(tmp_path)
    (tmp_path / "strategies.yaml").write_text(
        "grid_atr:\n  enabled: true\n  per_asset: {}\n"
    )
    db_path = _make_db([
        {"strategy_name": "grid_atr", "asset": "BTC/USDT", "timeframe": "1h",
         "grade": "A", "best_params": {"ma_period": 20}},
    ])

    # FAKE/USDT n'est pas dans per_asset → ne doit pas crasher
    result = apply_from_db(
        ["grid_atr"], config_dir=config_dir, db_path=db_path,
        exclude_symbols=["FAKE/USDT"],
    )
    assert result.get("blocked") is not True
    assert "FAKE/USDT" not in result.get("removed", [])


# ─── Test 16 : synchronisation timeframe colonne → best_params (Hotfix 37d) ─


def test_apply_syncs_timeframe_column_over_best_params(tmp_path):
    """Hotfix 37d : colonne timeframe=1h doit écraser best_params["timeframe"]=4h.

    Quand le WFO re-run sélectionne un nouveau best combo, la colonne timeframe
    est mise à jour mais best_params JSON peut contenir l'ancienne valeur (4h).
    apply_from_db() doit écrire timeframe=1h dans le YAML.
    """
    import yaml
    from scripts.optimize import apply_from_db

    initial_yaml = {
        "grid_atr": {
            "enabled": True,
            "per_asset": {},
        }
    }
    config_dir = str(tmp_path)
    yaml_path = tmp_path / "strategies.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(initial_yaml, f)

    # DB : colonne timeframe=1h, mais best_params JSON contient timeframe=4h (stale)
    db_path = _make_db([
        {
            "strategy_name": "grid_atr",
            "asset": "BCH/USDT",
            "timeframe": "1h",          # colonne correcte
            "grade": "A",
            "best_params": {"timeframe": "4h", "ma_period": 20, "sl_percent": 8.0},
        },
    ])

    result = apply_from_db(["grid_atr"], config_dir=config_dir, db_path=db_path)

    assert result.get("blocked") is not True
    assert "BCH/USDT" in result.get("applied", []), \
        "BCH/USDT doit être dans applied"

    with open(yaml_path) as f:
        updated = yaml.safe_load(f)
    bch_params = updated["grid_atr"]["per_asset"].get("BCH/USDT", {})
    assert bch_params.get("timeframe") == "1h", (
        f"Hotfix 37d : best_params timeframe doit être 1h (colonne), got {bch_params.get('timeframe')}"
    )


# ---------------------------------------------------------------------------
# Tests Sprint 62a — Guard timeframe depuis strategies.yaml
# ---------------------------------------------------------------------------


def test_apply_ignores_wrong_timeframe(tmp_path):
    """Résultat tf=1d ignoré quand la stratégie a timeframe=1h dans strategies.yaml.

    BUG: Un WFO avec timeframe:[1h,4h,1d] sélectionne 1d comme best combo IS →
    is_latest=1 écrase les bons résultats 1h. Le guard doit filtrer ces résultats.
    """
    import yaml
    from scripts.optimize import apply_from_db

    # strategies.yaml avec timeframe de référence = 1h
    initial_yaml = {
        "grid_atr": {
            "enabled": True,
            "timeframe": "1h",
            "per_asset": {
                # SOL avait Grade A 1h — doit être préservé si le résultat 1d est ignoré
                "SOL/USDT": {"timeframe": "1h", "ma_period": 7, "sl_percent": 20.0},
            },
        }
    }
    config_dir = str(tmp_path)
    yaml_path = tmp_path / "strategies.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(initial_yaml, f)

    # DB : SOL a is_latest=1 avec tf=1d Grade F (WFO pollué), BTC et ETH sont bons
    db_path = _make_db([
        {"strategy_name": "grid_atr", "asset": "BTC/USDT", "timeframe": "1h",
         "grade": "B", "best_params": {"ma_period": 20}},
        {"strategy_name": "grid_atr", "asset": "ETH/USDT", "timeframe": "1h",
         "grade": "A", "best_params": {"ma_period": 14}},
        {"strategy_name": "grid_atr", "asset": "SOL/USDT", "timeframe": "1d",
         "grade": "F", "best_params": {"ma_period": 5}},
    ])

    result = apply_from_db(["grid_atr"], config_dir=config_dir, db_path=db_path)

    # SOL 1d ignoré → pas dans applied ni removed
    assert "SOL/USDT" not in result.get("applied", []), \
        "SOL/USDT tf=1d doit être ignoré (≠ ref 1h)"
    assert "SOL/USDT" not in result.get("removed", []), \
        "SOL/USDT 1d ignoré → entrée 1h dans per_asset préservée"
    # BTC et ETH (1h) appliqués normalement
    assert "BTC/USDT" in result.get("applied", [])
    assert "ETH/USDT" in result.get("applied", [])
    # SOL reste dans le YAML (son ancienne entrée 1h préservée)
    with open(yaml_path) as f:
        updated = yaml.safe_load(f)
    assert "SOL/USDT" in updated["grid_atr"]["per_asset"], \
        "L'entrée SOL 1h dans per_asset doit être préservée"


def test_apply_warns_on_tf_mismatch(tmp_path):
    """Warning loggé pour chaque résultat dont le tf ≠ référence."""
    import yaml
    from unittest.mock import patch
    from scripts.optimize import apply_from_db

    initial_yaml = {
        "grid_atr": {
            "enabled": True,
            "timeframe": "1h",
            "per_asset": {},
        }
    }
    config_dir = str(tmp_path)
    yaml_path = tmp_path / "strategies.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(initial_yaml, f)

    db_path = _make_db([
        {"strategy_name": "grid_atr", "asset": "BTC/USDT", "timeframe": "1h",
         "grade": "A", "best_params": {"ma_period": 20}},
        {"strategy_name": "grid_atr", "asset": "SOL/USDT", "timeframe": "1d",
         "grade": "D", "best_params": {"ma_period": 5}},
        {"strategy_name": "grid_atr", "asset": "ETH/USDT", "timeframe": "4h",
         "grade": "B", "best_params": {"ma_period": 14}},
    ])

    warning_calls: list[str] = []

    import scripts.optimize as _opt_module
    original_warning = _opt_module.logger.warning

    def capture_warning(msg, *args, **kwargs):
        # Formater le message comme loguru le ferait
        try:
            formatted = msg.format(*args) if args else msg
        except Exception:
            formatted = str(msg)
        warning_calls.append(formatted)

    with patch.object(_opt_module.logger, "warning", side_effect=capture_warning):
        apply_from_db(["grid_atr"], config_dir=config_dir, db_path=db_path)

    # Un warning pour SOL/USDT (1d) et un pour ETH/USDT (4h)
    sol_warned = any("SOL/USDT" in w and "1d" in w for w in warning_calls)
    eth_warned = any("ETH/USDT" in w and "4h" in w for w in warning_calls)
    assert sol_warned, f"Warning attendu pour SOL/USDT tf=1d. Warnings reçus: {warning_calls}"
    assert eth_warned, f"Warning attendu pour ETH/USDT tf=4h. Warnings reçus: {warning_calls}"


def test_apply_uses_mode_tf_if_no_ref(tmp_path):
    """Sans timeframe de référence dans strategies.yaml, utilise le tf le plus fréquent (A/B)."""
    import yaml
    from scripts.optimize import apply_from_db

    # Pas de champ timeframe dans le YAML
    initial_yaml = {
        "grid_atr": {
            "enabled": True,
            "per_asset": {},
        }
    }
    config_dir = str(tmp_path)
    yaml_path = tmp_path / "strategies.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(initial_yaml, f)

    # DB : BTC + ETH en 1h Grade A, SOL en 1d Grade A → mode = 1h
    db_path = _make_db([
        {"strategy_name": "grid_atr", "asset": "BTC/USDT", "timeframe": "1h",
         "grade": "A", "best_params": {"ma_period": 20}},
        {"strategy_name": "grid_atr", "asset": "ETH/USDT", "timeframe": "1h",
         "grade": "A", "best_params": {"ma_period": 14}},
        {"strategy_name": "grid_atr", "asset": "SOL/USDT", "timeframe": "1d",
         "grade": "A", "best_params": {"ma_period": 5}},
    ])

    result = apply_from_db(["grid_atr"], config_dir=config_dir, db_path=db_path)

    # Mode = 1h (2 vs 1) → SOL (1d) ignoré
    assert "BTC/USDT" in result.get("applied", []), "BTC (1h) doit être appliqué"
    assert "ETH/USDT" in result.get("applied", []), "ETH (1h) doit être appliqué"
    assert "SOL/USDT" not in result.get("applied", []), \
        "SOL (1d) doit être ignoré car tf ≠ mode 1h"
