# `grid_multi_tf` Certification Audit — 2026-07-27

## Status

Implementation and the frozen long-run evidence are complete. The primary
historical verdict is **HISTORICAL_FAIL**; `grid_multi_tf` is not viable under
this policy.

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

## Immutable evidence and verdict

| Evidence | ID / metric |
|---|---|
| Snapshot | `snapshot-2b7082795d127834` (VALID) |
| WFO | 28 required rows completed; per-window Top 8 IS-only |
| 2x sensitivity | portfolio id 107, return -27.33%, DD -45.70% |
| Primary 3x nominal external OOS | portfolio id 108, return -37.00%, DD -58.81% |
| 4x sensitivity | portfolio id 109, return -52.30%, DD -71.07% |
| Primary 3x adverse | portfolio id 110, return -55.10%, DD -56.98% |
| Fresh capital 180d / 365d | ids 111 / 112; return -37.46% / -48.96%; DD -49.63% / -54.36% |
| Robustness | id 19; bootstrap CI95 [-82.7%, +121.9%], loss probability 73.4% |
| Certification | `cert-e3d40ad40dd768a6` — `HISTORICAL_FAIL` |

### Gate analysis

The primary 3x run fails return, bootstrap lower bound, loss probability,
nominal DD, adverse DD, degraded-cost return and both fresh-capital return/DD
gates. The historical failure is therefore independent of the missing Bitget
calibration and 1h capability ceiling.

Shared-account risk controls themselves held: 0 kill switches, peak margin
64.04% (<70%), simultaneous SL loss 29.8% (<=30%), minimum liquidation
distance 99.16%, and no missing funding settlements. They cannot offset the
large negative performance and drawdown.

Actual fast/canonical parity also failed (cumulative delta 641.95954766%). It
is retained as a separate reliability finding, not post-hoc tuning input. The
performance gates already settle the historical verdict.

No universe, Top-N, parameter, capital or leverage change is permitted after
this observation. Do not reoptimize `grid_multi_tf`, modify `grid_atr`, or
change robot2 as a response to this result.
