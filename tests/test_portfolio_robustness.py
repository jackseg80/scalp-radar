from __future__ import annotations

import math

import pytest

from scripts.portfolio_robustness import (
    _rolling_compounded_returns,
    compute_cvar,
    save_results,
)


def test_save_results_persists_certification_evidence():
    import sqlite3

    conn = sqlite3.connect(":memory:")
    row_id = save_results(
        conn,
        7,
        "external-oos",
        {
            "n_sims": 10,
            "block_size": 7,
            "median_return": 2.0,
            "ci_return_low": 1.0,
            "ci_return_high": 3.0,
            "median_dd": -2.0,
            "ci_dd_low": -1.0,
            "ci_dd_high": -3.0,
            "prob_loss": 0.0,
            "prob_dd_30": 0.0,
            "prob_dd_ks": 0.0,
        },
        {"var_5_daily": -0.1, "cvar_5_daily": -0.2, "cvar_30d": -4.0},
        {},
        {},
        {"verdict": "VIABLE"},
        {
            "external_oos_return_pct": 12.5,
            "adverse_max_drawdown_pct": -9.0,
            "degraded_cost_return_pct": 4.5,
            "adverse_backtest_id": 8,
            "manifest_hash": "manifest-1",
            "engine_parity_passed": True,
            "engine_parity_max_delta_pct": 0.25,
        },
    )
    row = conn.execute(
        "SELECT * FROM portfolio_robustness WHERE id=?", (row_id,),
    ).fetchone()
    columns = [item[0] for item in conn.execute(
        "SELECT * FROM portfolio_robustness LIMIT 0",
    ).description]
    saved = dict(zip(columns, row))
    assert saved["external_oos_return_pct"] == 12.5
    assert saved["adverse_backtest_id"] == 8
    assert saved["manifest_hash"] == "manifest-1"
    assert saved["engine_parity_passed"] == 1


def test_rolling_30d_returns_use_observed_sequences():
    returns = [0.01] * 29 + [-0.10] + [0.0] * 30
    rolling = _rolling_compounded_returns(returns, 30)
    assert len(rolling) == 31
    assert rolling[0] == pytest.approx((1.01 ** 29) * 0.90 - 1.0)
    assert rolling[-1] == pytest.approx(0.0)


def test_monthly_cvar_is_not_daily_cvar_compounded():
    # Alternating losses and recoveries have a much smaller observed monthly
    # loss than repeating the worst daily tail 30 times.
    returns = [-0.02, 0.02] * 60
    result = compute_cvar(returns, kill_switch_pct=45.0)
    synthetic = (1.0 + result["cvar_5_daily"]) ** 30 - 1.0
    assert result["rolling_30d_count"] == 91
    assert result["cvar_30d"] != pytest.approx(synthetic)
    assert result["cvar_30d"] == pytest.approx((0.98 * 1.02) ** 15 - 1.0)


def test_monthly_cvar_is_nan_when_history_is_too_short():
    result = compute_cvar([0.001] * 20, kill_switch_pct=45.0)
    assert math.isnan(result["cvar_30d"])
    assert result["rolling_30d_count"] == 0
