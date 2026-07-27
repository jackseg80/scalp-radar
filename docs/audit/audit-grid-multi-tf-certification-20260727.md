# `grid_multi_tf` Certification Audit — 2026-07-27

## Status

Implementation is complete; long-run evidence is pending user execution.
There is no new historical verdict yet.

## Frozen research question

Can `grid_multi_tf` pass the universal 28-asset external-OOS performance
gates at 3x under representative closed-bar 1h execution?

The immutable policy is the Sprint 70a plan: common calendar from 2022-01-01
UTC, IS 180d, embargo 7d, OOS/step 60d, exhaustive 1,152-combination search,
positive IS eligibility, Top 8 IS-only, seed 0, 1,646 USDT, primary 3x and
pre-declared 2x/4x sensitivity.

## Initial audit

- Strategy batch and live paths derived Supertrend 4h from 1h, but duplicated
  resampling existed between strategy and fast cache.
- The former resamplers grouped any observed rows into a bucket. A
  non-aligned start, partial final bucket or internal hole could therefore
  create a false 4h candle and bridge state across unknown data.
- The fast engine already shifted grid prices from T to T+1 but used the
  current Supertrend direction. Around a flip, an old LONG price could be
  evaluated as SHORT or conversely.
- Historical parity evidence used the event-driven validation engine as its
  “fast” side; the actual WFO fast loop did not expose a selected-candidate
  trace.
- Universal snapshots froze universe/calendar/Top-N/leverage, but not
  portfolio capital. Replay CLIs still accepted capital or explicit leverage
  values that could diverge from the manifest.
- Status resolution classified operational capability failures as
  `HISTORICAL_FAIL` and could let missing evidence mask a real performance
  failure.

## Data inventory

Read-only inspection found Binance 1h history for all 28 configured assets
through 2026-07-26. Native Binance 4h data is stale around February 2026 and
absent for DOT, ATOM, LTC, FIL, ETC, TRX and XLM. Funding rows exist for all
28 assets.

Decision: Binance 1h is the sole canonical price input. Native 4h remains a
diagnostic and does not condition the snapshot.

## Implemented controls

- `backend/core/multi_timeframe.py` is the shared 1h→4h source for batch,
  cache and live. It accepts exactly four UTC hourly opens, exposes a bucket
  only from its close, rejects partial buckets and restarts indicator warmup
  after gaps.
- Fast entry direction is shifted with entry prices. Current direction stays
  independent for exits; pending orders retain their actual side and no
  retrospective replacement is created on the flip candle.
- The existing fast grid loop emits selected entry-candidate and exit traces.
  `engine_parity_json` retains its existing top-level fields and adds actual
  search-loop plus event-engine evidence.
- `UniverseSelectionSpec.portfolio_initial_capital` is optional for legacy
  reads and mandatory in newly created universal snapshots. External replay
  and certification resolve capital from the manifest and reject divergence.
  Explicit leverage must be one of the frozen scenarios.
- Verdict precedence is now:
  1. failed performance gate → `HISTORICAL_FAIL`;
  2. otherwise missing/failed calibration, 1m, parity, integrity or coverage
     evidence → `RESEARCH_ONLY`;
  3. all gates pass → `PAPER_READY`.
- No changes were made to `strategies.yaml`, `risk.yaml`, grid_atr evidence or
  robot2.

## Regression evidence

The focused suite passes **135 tests**, covering:

- non-aligned starts, exact 04:00 UTC visibility, partial final buckets,
  internal gaps and post-gap warmup;
- current 4h changes remaining invisible before close;
- batch/cache/live direction parity at every eligible timestamp;
- paired price+direction T→T+1, pending orders, flip exits and absence of
  retrospective same-candle entries;
- exhaustive 1,152-combination grid, complete 28-row WFO requirement,
  late-listing calendar behavior and IS-only Top 8;
- capital/leverage immutability and all three verdict-resolution cases.

The complete `uv run python -m pytest --tb=short -q` suite passes
**2,372 tests in 123.89s**, with 0 failures.

## Pending immutable evidence

| Evidence | ID / metric |
|---|---|
| Snapshot | PENDING |
| 28 WFO rows | PENDING |
| Primary 3x external OOS | PENDING |
| 2x sensitivity | PENDING |
| 4x sensitivity | PENDING |
| Canonical parity | PENDING |
| Adverse/fresh-capital/bootstrap | PENDING |
| Final status | PENDING |

Run only the commands recorded in `COMMANDS.md`. Analyze their outputs without
changing universe, Top-N, parameters, capital or leverage.
