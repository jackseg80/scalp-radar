# Live inactivity audit — 2026-07-19

## Scope

Production host `robot2`, repository `~/scalp-radar`. The audit was read-only:
no container restart, deployment, exchange mutation, or server-side file edit was
performed.

## Observed production state

- The executor was connected, its kill switch was inactive, and it had no open
  position.
- The last confirmed live fill was on 2026-07-01.
- During a 10.5-hour sample, 200 entry limit orders were created
  (ADA 80, NEAR 69, XRP 51), none filled, and 26 were cancelled after price
  drift.
- 15 of 28 market feeds were stale. Live symbols ATOM, CRV, ETC, and ICP were
  several days stale.
- The logs repeatedly reported missing `DataEngine` watch tasks.
- SQLite reported `database is locked` approximately every 30 seconds.
- The production database was about 1.4 GB; the WAL was about 7.4 MB.
- Historical grid state contained levels 5 and 6 although the strategy was
  configured for three levels.
- `/health` still returned a healthy status while feeds and writes were stale.

## Root causes

1. Fallback recovery only replaced an existing watch task. Once the task had
   disappeared, recovery never recreated it.
2. Polling fallback tasks survived stop/full reconnect and could accumulate.
3. Intra-candle updates were appended as new observations. Indicators froze on
   the first tick while grid closes and live limit prices churned on every tick.
4. A cancelled order could fill after its replacement. Tracking by logical
   level allowed the late fill to delete the replacement and create excess grid
   levels.
5. Pending-order expiry was evaluated only after a successful exchange fetch.
6. The live executor applied only a subset of `per_asset` WFO parameters.
7. The first incomplete candle was kept by `INSERT OR IGNORE`; later updates
   were discarded. A locked periodic batch was also dropped instead of retried.
8. Market grid opening reacquired `_state_lock` while already holding it when
   placing the mandatory SL, causing a deadlock after an entry.
9. Health reporting did not expose per-symbol freshness or write failures.

## Implemented locally

- Recreate missing watch tasks and clean all fallback pollers during reconnect
  and shutdown.
- Add closed-candle callbacks; live entry grids now refresh only when their
  strategy candle closes.
- Replace same-timestamp values in the incremental indicator and grid close
  buffers.
- Make cancel/replace conservative, deduplicate fills by exchange order ID,
  preserve replacements, merge unavoidable late exposure, and enforce the
  configured logical level count.
- Check order expiry before exchange fetch.
- Apply complete isolated `per_asset` strategy configuration in live entry,
  exit, sizing, leverage, and SL paths.
- Remove the nested state-lock acquisition during market grid SL placement.
- Upsert incomplete candles, serialize/retry SQLite writes, and requeue failed
  DataEngine batches.
- Bound the live maintenance backfill to 30 days.
- Report stale symbols, fallback modes, abandoned tasks, pending write count,
  and last flush failure through `/health`.

## Validation

- Targeted DataEngine, SQLite, indicator, grid runner, autonomous executor, and
  order-race tests pass.
- The complete suite is run with the two assertions tied to the user's
  uncommitted WFO YAML experiment deselected. Those YAML files were not changed
  by this audit.

## Deployment status

Not deployed. Production still runs its existing server commit and hotfix.
Deployment must reconcile the server-only commit first, snapshot persistent
state/database, deploy the reviewed change set, then verify feed freshness,
SQLite lock rate, pending-order churn, fill accounting, and `/health`.
