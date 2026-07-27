# Sprint 70a — `grid_multi_tf` Certification

## Frozen policy

- Universe: all 28 `assets.yaml` symbols. A late listing is admitted only
  when a complete IS/embargo/OOS interval exists on the common calendar.
- Calendar: `2022-01-01T00:00:00Z`, 1h signals, IS 180 days, embargo 7 days,
  OOS 60 days, step 60 days, seed 0.
- Selection: exhaustive current 1,152-combination grid, existing positive IS
  eligibility criteria, Top 8 IS-only per external window.
- Fixed parameters outside the grid: LONG+SHORT, cooldown 3 bars, no time
  stop; YAML shared risks (`max_live_grids=4`, margin 70%, simultaneous SL
  loss 30%, kill switch 45%).
- Portfolio: 1,646 USDT, primary 3x, pre-declared sensitivities 2x and 4x.
- Data: Binance 1h is canonical. Supertrend 4h is derived from complete UTC
  buckets. Native 4h is diagnostic only.
- Capability ceiling: a historical pass is `RESEARCH_ONLY` until a real 1m
  canonical broker and Bitget calibration exist. A performance gate failure
  is `HISTORICAL_FAIL`.

## Implementation

1. Audit strategy, fast path, canonical portfolio, WFO, snapshots, risk,
   1h/4h data and regression tests.
2. Replace duplicated 4h resampling with one shared closed-bar helper.
3. Pair fast entry price and Supertrend direction at T→T+1 while retaining
   current direction for flip exits.
4. Add selected-candidate tracing to the existing fast loop and compare it
   with canonical replay without changing legacy parity JSON fields.
5. Freeze portfolio capital in new universal snapshots; reject manifest/CLI
   divergence.
6. Resolve statuses by performance failure first, then operational evidence.
7. Run focused tests, then the full test suite.
8. Record implementation and pending evidence in roadmap, workflow, commands
   and the dated audit.

## Evidence runs

The snapshot, WFO, external-OOS and certification commands are intentionally
not run by Codex. Their IDs and metrics remain pending until the user runs the
commands in `COMMANDS.md`. No post-hoc universe, Top-N, parameter, capital or
leverage change is permitted.

## Safety

Do not change `strategies.yaml`, `risk.yaml`, any `grid_atr` result, or any
robot2 state, deployment or configuration.
