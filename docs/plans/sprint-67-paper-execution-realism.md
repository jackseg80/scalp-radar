# Sprint 67 — Paper Execution Realism

## Goal

Remove look-ahead bias from grid paper trading and make dashboard accounting
consistent with the economic state of the simulated account.

## Production diagnosis

The `grid_multi_tf` paper runner on `robot2` showed +1,826% and a 97% win rate.
Of 2,553 trades, 2,315 opened and closed on the same H1 candle and generated
+$37,700.95. The 236 positive-duration trades generated -$3,569.71. The
headline performance was invalid.

## Scope delivered

- Split realtime mark-to-market updates from closed-candle decisions.
- Activate grid orders only on candles after the signal candle.
- Use TP/SL thresholds known before the triggering candle.
- Reject duplicate and out-of-order candles.
- Persist virtual orders, exit plans, execution clock, and funding.
- Reject legacy intrabar snapshots in chronological production runners.
- Include funding in realized P&L.
- Correct equity, total P&L, realized P&L, and margin percentage formulas.
- Add focused regression coverage.
- Document the production audit and remaining OHLC limitations.

## Files

- `backend/backtesting/simulator.py`
- `backend/core/state_manager.py`
- `frontend/src/components/SessionStats.jsx`
- `tests/test_journal.py`
- `tests/test_paper_execution_realism.py`
- `docs/audit/audit-grid-multi-tf-paper-realism-20260720.md`
- `docs/ROADMAP.md`

## Validation

- Dedicated and related state tests: 105 passed.
- Full backend suite: 2,269 passed.
- Frontend production build: passed.
- Ruff: passed.
- `git diff --check`: passed.

## Rollout

Deploy only after review. The first startup automatically rejects the old
intrabar paper snapshot, so invalid paper equity does not survive the upgrade.
Pre-upgrade database trades remain audit-only and must not be included in
strategy evaluation.
