# Grid BolTrend certification readiness audit — 2026-07-28

## Scope

Reviewed the strategy, fast WFO engine, canonical `GridStrategyRunner`, snapshot/WFO plumbing, execution calibration and targeted tests. No deployment, robot2 configuration, WFO, portfolio replay or long certification run was performed.

## Corrected gaps

- The fast engine now bases the breakout on closed 1h indicators, activates intent-derived levels from the following bar only, keeps the breakout ladder fixed, uses maker-limit entry economics and expires pending levels after the `ExecutionSpec` duration.
- Signal/time exits are deferred as T→T+1 market intents; persistent protective SL remains the conservative same-bar outcome.
- The runner preserves GridBolTrend levels until 1m fill/partial-fill/expiry and uses immutable per-asset strategy instances.
- Snapshot creation freezes the actual Binance 1h and Bitget 1m consumers. WFO cache funding source and expiry derive from the snapshot execution spec.
- Calibration can consume a read-only shared grid observation DB and enforces the qualifying sample thresholds.

## Remaining evidence and current status

Current status: `RESEARCH_ONLY` pending immutable data repair and user-run WFO/OOS/certification evidence. This is not a performance verdict. Calibration `cal-1b1bb1cce72e7cd8` is qualifying (30 filled, 3 partial/cancelled remainders), but the first snapshot attempt, `snapshot-5a23b693c8fb56af`, was `INVALID` before WFO: local Bitget 1m ended on 2026-02-21, seven configured series were empty and source rows included gaps. The explicit Bitget UTA v3 long-history fetch and cutoff-coverage guard now fail closed instead of allowing a stale end-of-series to appear usable.

Bitget's public UTA funding endpoint was also measured after pagination: it yields only 270 records per asset (about 90 days), not the required 2022–2026 funding history. The snapshot now fails closed on a missing/recent/gapped broker-funding series. Certification remains blocked pending an immutable full-period Bitget funding archive; Binance funding is not an admissible substitute.

### Read-only Bitget order evidence recovered

On 2026-07-28, a private read-only history query through the existing robot2
`grid_atr` credential returned 48 closed limit fills and five cancelled limit
orders. Three cancelled partial orders fell before the 2026-07-27 cutoff and
have a confirmed non-filled remainder. The local importer stores the raw order
identifier, intent/fill timestamps, prices, quantities and status without
writing to robot2. Calibration `cal-1b1bb1cce72e7cd8` therefore meets the
30-filled / one-confirmed-unfilled requirement. It is operational evidence,
not a performance verdict.

## Test evidence

`tests/test_grid_boltrend.py`, `tests/test_grid_boltrend_parity.py` and `tests/test_grid_runner.py`: 105 passed after the implementation changes. `tests/test_experiment_snapshot.py` and `tests/test_fetch_history.py`: 17 passed for the UTA v3 candle/funding paths, concurrent batching, interrupted-range resume and cutoff guard. The repository-wide suite remains required before commit.
