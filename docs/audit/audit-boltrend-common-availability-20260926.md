# grid_boltrend research common availability — 2026-09-26

## Scope and evidence

User approved the pre-OOS amendment in
`docs/plans/boltrend-common-availability-20260926.md`. Only snapshot construction,
revalidation, reuse identity and CLI changed. Existing WFO filtering already
requires continuous coverage from IS start through OOS end; `_snapshot_bounds`
passes the clipped, signed Binance range to it. External OOS starts after 180d
IS + 7d embargo, so its 500h warmup is later than the asset's common boundary.
No changes to strategy, engines, risk, order activation, fees or funding profile.

`common_hour_v1` is explicit opt-in, synthetic funding/grid_boltrend only,
Binance 1h + Bitget 1m only, with explicit start and zero tolerated missing bars.
The signed manifest records original observed starts and rounded common start.
Both consumed series must actually contain the boundary candle and reach cutoff.
All consumed rows are validated and hashed. Excluded prefixes remain in SQLite;
they are not consumed or rehashed by this experiment. The frozen observed starts
are provenance, not a claim of exchange inception. New earlier data requires a
new experiment, not silently changed eligibility for an existing snapshot.

The original strict path still rejects the prefix mismatch. Synthetic rates and
availability do not authorize PAPER_READY or deployment. Historical failures
remain closed. No historical snapshot, WFO, portfolio run, download or robot2
action was executed. Full real-data coverage is still unverified; the user runs
snapshot validation first and stops on errors. COMMANDS.md separates that step
from expensive research runs.

## Validation

- Focused research/availability suite: 38 passed (16 new availability tests).
- Full guarded suite: **2468 passed** (`uv run python -m pytest --tb=short -q`).
- Ruff F checks on changed Python files, CLI help and `git diff --check` pass.
- Protected CLAUDE.md, GEMINI.md and AGENTS.md hashes unchanged.
