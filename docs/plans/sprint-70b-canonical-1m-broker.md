# Sprint 70b — Canonical 1m Execution Broker

## Objective

Make the existing canonical grid portfolio consume a frozen Bitget execution
timeframe beneath closed Binance 1h signals, without adding a second strategy
or portfolio engine.

This lot changes infrastructure only. It does not reopen or rerun
`grid_atr` or `grid_multi_tf`, and it performs no robot2 action.

## Frozen execution contract

- Signal source: closed Binance 1h candles.
- Broker source: `ExecutionSpec.exchange` and
  `ExecutionSpec.execution_timeframe`, normally Bitget 1m.
- Timestamp convention: candle timestamps are opens. A 10:00 1h candle is
  visible at 11:00; its intents may first interact with the 11:00 1m candle.
- Same-timestamp priority: signal close, then broker minute.
- Entries: persistent limit intents, configured expiry, deterministic latency,
  missed-fill and partial-fill assumptions.
- Exits: existing server-side TP/SL thresholds have priority over a newly
  submitted signal market exit. Every grid cycle closes as a taker market
  order with configured slippage.
- A fill minute cannot apply an earlier extreme from that same minute to the
  protection installed after the fill. Protection becomes observable on the
  following broker candle.
- Funding: exact 00:00, 08:00 and 16:00 UTC broker boundaries, once per symbol
  and settlement.
- Data: no fallback. Missing series, incomplete start/end coverage,
  non-monotonic rows or a gap above one frozen bar abort the replay.

## Reused components

- `GridStrategyRunner`
- `PendingGridOrder` and `PlannedGridExit`
- `OrderIntent` and `FillEvent`
- `ExecutionSpec`
- `GridPositionManager`
- `PortfolioBacktester`
- `LiveRiskManager`
- immutable snapshots and external-OOS concatenation

## Implementation

1. Add an execution-only path to `GridStrategyRunner`. The signal path keeps
   indicator/context/grid planning; the broker path consumes only pre-existing
   intents and protection.
2. Add signal-generated reduce-only market exits and immediate post-fill
   TP/SL planning from the last closed signal.
3. Merge signal-close events and intrabar candles in the existing portfolio
   loop. Preserve account risk, snapshots, liquidation checks, leverage
   changes and trade collection.
4. Load Binance 1h warm-up/signal rows separately from frozen Bitget broker
   rows. Validate runtime boundaries and gaps.
5. Persist `execution_timeframe_used`, `execution_candles_processed` and
   `intrabar_max_gap_bars`; require positive consumption and the frozen gap
   bound in certification.
6. Refuse snapshot-bound CLI replays when the immutable manifest does not
   contain every required broker series.

## Validation

- Boundary and look-ahead tests.
- Immediate protection and same-minute ambiguity test.
- Server-stop versus signal-market priority.
- Portfolio common-clock and actual-consumption evidence.
- Missing series and multi-bar gap rejection.
- Immutable snapshot broker-series requirement.
- Certification broker-count and gap gates.
- Existing order persistence, partial-fill, portfolio risk, parity and
  database regressions.
- Complete pytest suite.

## Results

- Focused suite: 152 passed.
- Wider realism/parity/risk suite: 168 passed.
- Full suite: 2384 passed in 149.51 seconds.
- Long WFO/OOS/portfolio runs: not run.
- Deployment, server and robot2 actions: none.

## Next step

Start a separate discussion for the next strategy. Before its frozen WFO,
collect qualifying Bitget execution observations, backfill/freeze its required
Bitget 1m series and create a clean validated snapshot. A historical success
can reach `PAPER_READY` only if every performance, calibration, coverage and
parity gate also passes.
