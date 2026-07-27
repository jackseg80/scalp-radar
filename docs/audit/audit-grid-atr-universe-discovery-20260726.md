# grid_atr Universal Discovery WFO Audit — 2026-07-26

## Scope

The earlier nine-asset portfolio reports are scoped `candidate_replay` results
for an existing configuration.  They are useful evidence about that candidate,
but they cannot decide whether the `grid_atr` strategy family is viable across
the configured universe.  They remain preserved as `RESEARCH_ONLY`.

## Frozen universal policy

- Universe: all 28 symbols declared in `assets.yaml`; unconfigured MATIC is
  excluded.
- Signal: 1h.  Common WFO calendar starts 2022-01-01; IS 180d, embargo 7d,
  OOS 60d, step 60d.
- An asset participates only in a window for which the complete interval is
  present.  It cannot shift the shared OOS schedule.
- Selection is fixed before replay: positive IS Sharpe, positive IS return,
  at least 10 IS trades; sort by Sharpe descending, trades descending, then
  symbol; keep Top 8.
- `grid_atr` evaluates every valid grid combination on every IS window.  The
  selected params for every window must pass fast/canonical parity.
- Shared-account guards stay live-equivalent: correlation/exposure, margin,
  simultaneous SL loss and `max_live_grids=4`.
- Capital is 1,502.59 USDT.  The historical verdict is 4x; 2x and 6x are
  pre-declared sensitivity replays and never selection levers.

## Fail-closed conditions

The universal replay refuses partial WFO output, overlapping windows, data
outside the snapshot, missing funding, a non-primary leverage verdict or a
parity failure.  A 4x gate failure means `HISTORICAL_FAIL` for this universal
policy, with no post-hoc asset deletion or re-optimisation from that outcome.

One source-confirmed one-hour Binance outage may be recorded by the snapshot
as its explicit gap tolerance.  It is never interpolated: the affected
asset's IS/embargo/OOS window is removed while the global calendar remains
unchanged for other assets.

No historical result is deployable yet: the canonical broker still runs on 1h
bars and Bitget calibration needs real complete observations.  A historical
success therefore remains `RESEARCH_ONLY`, followed by 1m broker parity,
paper forward and canary.  Nothing in this audit authorizes or changes robot2.

## Invalidated scenario replay evidence

The first universal external-OOS run on `snapshot-035a6e48851ac4c1` produced
result ids 100 (requested 2x), 101 (requested 4x), and 102 (requested 6x).
All three displayed and persisted 6x.  The leverage override reached sizing,
but did not reach the canonical runner's risk state, so margin, liquidation
and displayed leverage were inconsistent.  They are retained as audit data,
but must not be used for a gate or a strategy conclusion.

The portfolio builder now applies an override atomically to the runner, its
grid position manager and its strategy configuration.  A regression test
asserts that all three use the requested leverage.  Fresh snapshot-bound
evidence is required before drawing a 4x conclusion.

When a canonical portfolio-only correction follows an already-completed WFO,
the external replay can explicitly reuse that WFO through a second snapshot
identifier.  It compares a dedicated WFO input fingerprint covering series,
YAML, calendar, grid, seed and IS-only universe policy; any mismatch fails
closed.  The target snapshot is still revalidated against the corrected
canonical code, avoiding an unnecessary exhaustive WFO rerun.

## Final historical result

The compatible exhaustive WFO evidence from
`snapshot-035a6e48851ac4c1` was replayed through the corrected canonical
portfolio under `snapshot-d97f794987803145`. The 4x primary external-OOS
result returned +92.9% but reached -47.5% maximum drawdown, exceeding the
30% nominal gate. `grid_atr` is therefore `HISTORICAL_FAIL` for the frozen
universal 4x policy. The 2x (+60.4%, -22.3%), 6x (+112.5%, -56.3%) and 3x
(+96.4%, -32.4%) scenarios remain sensitivity evidence only; none may replace
the pre-declared primary decision after observing the results.

There is no further historical robustness or certification computation to run
for this policy: the primary drawdown gate already fails closed. No robot2
configuration, deployment or order changed during this evaluation.

## Verification

- Targeted selection/snapshot/WFO/portfolio/certification tests: 113 passed.
- Full project suite: 2362 passed in 131.79 seconds.
