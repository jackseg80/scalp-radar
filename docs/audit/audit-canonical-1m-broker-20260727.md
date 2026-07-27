# Audit — Canonical 1m Broker — 2026-07-27

## Scope and safety

Audited the canonical grid certification path after Sprints 68–70a:
strategy runner, fast/canonical boundary, portfolio, external OOS, immutable
snapshots, risk, funding, Binance 1h data, Bitget intrabar data, persistence,
certification gates and tests.

No YAML strategy/risk value was changed. No WFO or long backtest was run.
`grid_atr` and `grid_multi_tf` were not reevaluated. No deployment, restart,
configuration or state action was performed on robot2.

User-owned `CLAUDE.md`, `GEMINI.md` and `AGENTS.md` changes were excluded from
the implementation scope.

## Initial findings

### Reusable foundations

The repository already had one chronological paper/canonical grid runner,
canonical `OrderIntent`/`FillEvent` models, persistent pending limits,
expiry, partial/missed-fill scenarios, taker/maker costs, `ExecutionSpec`,
external-OOS concatenation and shared `LiveRiskManager` account controls.

The live Executor also already used persistent exchange limits and
server-side SL. A new strategy simulator or a parallel portfolio engine was
neither necessary nor desirable.

### Blocking defect

`GridStrategyRunner.on_candle()` combined:

- closed-bar indicator and strategy evaluation;
- order planning;
- OHLC fill and exit execution.

`PortfolioBacktester` loaded and merged only Binance 1h rows. Therefore a
snapshot could contain 1m rows without the canonical broker consuming them.
The existing `intrabar_execution_used` gate correctly kept `PAPER_READY`
unreachable.

### Look-ahead boundary

Candle timestamps represent opens. Existing planning correctly stamped an
intent from the 10:00–11:00 candle at 11:00, but execution still waited for
the next 1h OHLC. The required event ordering is:

1. consume all broker minutes strictly before 11:00;
2. evaluate the closed 10:00 signal at 11:00;
3. allow its intent to interact first with the 11:00 broker candle.

### Provenance gap

Revalidation only hashes series listed in the immutable manifest. An old
Binance-1h-only snapshot could otherwise have been replayed against mutable
local Bitget 1m rows. Snapshot-bound CLIs therefore needed an explicit
required-series guard in addition to runtime data loading.

## Implemented result

### Shared runner broker path

`GridStrategyRunner` now has an execution-only candle path. It does not update
signal indicators or compute a new grid. It consumes the runner's existing
pending limits, exits and protection using the same position manager,
account callbacks and event audit models.

Closed 1h bars remain the only source of:

- new entry intents;
- price-drift/direction replacement decisions;
- dynamic TP/SL plans;
- signal-driven reduce-only market exits.

### Chronology and fills

The existing portfolio loop uses a common sorted event clock. Signal closes
have priority over broker candles at the same timestamp. Duplicate or
out-of-order broker candles raise an error.

Limit fills occur at the declared limit with maker fees. Expiry, latency,
missed fills and partial fills use the frozen `ExecutionSpec`. A partial
remainder preserves its original intent and expiry.

A new fill receives TP/SL derived from the last closed signal. Because OHLC
cannot reveal whether a minute's extreme occurred before or after the limit
fill, that same minute cannot retrospectively trigger the new protection.
The next minute can.

Existing server-side protection is evaluated before a newly submitted market
exit at the same boundary. SL gaps retain the configured trigger/open
interpolation and market slippage accounting.

### Data and funding

- Signals/warm-up: Binance 1h.
- Execution: frozen `ExecutionSpec.exchange` and timeframe, normally Bitget
  1m.
- Funding observations: execution exchange.
- Funding settlement: exact UTC broker minute, deduplicated.
- Runtime coverage: required for every selected asset at both boundaries.
- Gaps: maximum one missing broker bar, matching the declared snapshot
  allowance; larger gaps fail closed.
- No 1h execution fallback exists in snapshot-bound canonical runs.

### Evidence and certification

Portfolio rows now persist:

- actual `execution_timeframe_used`;
- `execution_candles_processed`;
- `intrabar_max_gap_bars`.

Certification requires a positive broker event count and a runtime gap no
larger than the immutable manifest bound. Snapshot-bound portfolio,
external-OOS and certification CLIs reject manifests that do not freeze every
required execution series.

The historical verdict precedence is unchanged. Performance failures still
produce `HISTORICAL_FAIL`; operational evidence failures produce
`RESEARCH_ONLY` when historical performance otherwise passes.

## Validation

- Focused broker/portfolio/snapshot/certification suite: 152 passed.
- Wider realism/parity/risk suite: 168 passed.
- Full command: `uv run python -m pytest --tb=short -q`
- Full result: 2384 passed in 149.51 seconds.

No long optimization, external-OOS, fresh-capital or certification run was
started.

## Remaining operational evidence

The broker implementation removes the infrastructure-wide 1h blocker, but
does not manufacture:

- complete historical Bitget 1m coverage for a future candidate;
- at least 30 qualifying filled execution observations;
- at least one confirmed unfilled observation and observation hash;
- a passing fast/canonical parity result;
- passing historical performance and robustness.

Those inputs must be frozen before observing the next strategy's OOS result.
They must be handled in that strategy's separate discussion.
