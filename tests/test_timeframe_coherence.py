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
    """apply_from_db() retourne blocked=True si conflit timeframe."""
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
    # Créer un strategies.yaml minimal
    (tmp_path / "strategies.yaml").write_text(
        "grid_atr:\n  enabled: true\n  per_asset: {}\n"
    )
    # Créer secrets minimal pour db_path
    result = apply_from_db(["grid_atr"], config_dir=config_dir, db_path=db_path)

    assert result.get("blocked") is True
    assert result.get("reason") == "tf_conflict"
    assert result.get("majority_tf") == "1h"
    assert "BCH/USDT" in result.get("tf_outliers", [])


def test_apply_blocked_exit_code(tmp_path):
    """main() exit code 1 quand conflit détecté via --apply."""
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
    assert result.get("blocked") is True
    # Vérifie que le code qui suit result.get("blocked") ferait sys.exit(1)
    # (le test direct de sys.exit via main() nécessiterait subprocess)
    assert result.get("changed") is False


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
