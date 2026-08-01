# Backtest and Live Certification Workflow

This is the authoritative path from historical research to live trading. Grades and ad-hoc portfolio reports are diagnostics only. They cannot authorize a live configuration.

## Core rules

- Research may use the existing fast engines; certification must replay the selected candidates through the canonical event-driven engine.
- A certification is bound to an immutable data snapshot, Git commit, complete config hashes, execution calibration and random seed.
- Candidate selection uses nested walk-forward. Parameters and the asset universe are selected from inner/IS data only; portfolio results are concatenated from untouched external OOS windows.
- `max_live_grids=4`, leverage and account risk constraints are fixed before WFO. Changing leverage, sizing, universe, cycle limits or risk limits invalidates the certification.
- Existing WFO/portfolio rows are `legacy`. They remain readable but can never be promoted.
- `optimize --apply` is blocked. Only a `LIVE_APPROVED` certification may create a local promotion artifact.
- No certification command deploys or modifies robot2.

## Current capability status

The canonical shared-account replay is available for the nine registered grid strategies. Unsupported mono-position strategies and `trend_follow_daily` fail closed as `RESEARCH_ONLY`; they are never routed through a grid engine.

The canonical grid portfolio now evaluates strategies only on closed 1h
Binance bars and executes their resulting intents on the frozen Bitget
intrabar series (normally 1m). The shared event clock processes an hourly
close before the new minute at the same timestamp. It rejects missing or
non-monotonic execution data and gaps above the snapshot allowance instead of
falling back to 1h.

This removes the former infrastructure-wide 1h blocker. It does not make a
strategy `PAPER_READY` by itself: the snapshot must freeze the consumed 1m
rows, the replay must persist a positive broker event count within the gap
bound, and Bitget calibration, parity, coverage and all performance gates must
still pass.

## 1. Calibrate execution from Bitget observations

Calibration uses persisted entry outcomes, including confirmed unfilled and partial orders. At least 30 filled observations and at least one confirmed unfilled observation are required by certification.

```powershell
uv run python -m scripts.calibrate_execution --strategy grid_atr --since <ISO_DATE> --until <ISO_DATE>
```

Keep the returned `calibration_id`.

## 2. Create and validate an immutable snapshot

The certification snapshot requires a clean worktree. It records exact closed candles, gaps, duplicates, OHLC validity, per-series hashes, funding/OI availability, Git/config hashes and the calibrated execution model. Binance may provide long history; Bitget data is included for transfer and execution calibration. There is no silent exchange fallback.

```powershell
uv run python -m scripts.create_data_snapshot `
  --cutoff <ISO_DATE> `
  --since <ISO_DATE> `
  --symbols SOL/USDT,ADA/USDT,XRP/USDT `
  --timeframes 1h,1m `
  --exchange binance `
  --execution-timeframe 1m `
  --calibration-id <CALIBRATION_ID> `
  --validate
```

Keep the returned `snapshot_id`. Any later change to code, configuration or frozen market rows makes revalidation fail.

The canonical replay consumes Binance signal rows from `--exchange` and
Bitget broker rows from the calibrated `ExecutionSpec`. A new certifiable
snapshot must therefore use `--validate`; an older snapshot containing only
Binance 1h rows is intentionally rejected rather than supplemented from the
mutable local database.

If only the canonical portfolio/execution path changes after a completed WFO,
create a fresh snapshot and pass the prior WFO snapshot explicitly through
`external_oos_portfolio --wfo-snapshot <ID>`. This is accepted only when the
two snapshots have the same WFO input fingerprint (data hashes, YAML hashes,
calendar, parameter grid, seed and IS-only selection); it never bypasses
revalidation of the fresh canonical replay.

### `grid_atr` universe discovery (authoritative historical evaluation)

This is distinct from a `candidate_replay`: it evaluates every configured
asset (currently 28), not the asset list currently running on an exchange.
The policy is frozen inside the snapshot: 1h signals, a common calendar from
2022-01-01, IS 180d, embargo 7d, OOS 60d/step 60d, exhaustive valid-grid
search, and an IS-only Top 8 selected independently for every OOS window.
Late-listed assets join only once their full IS/embargo/OOS interval exists.

```powershell
uv run python -m scripts.create_data_snapshot `
  --strategy grid_atr `
  --universe-discovery `
  --calendar-start "2022-01-01T00:00:00+00:00" `
  --since "2022-01-01T00:00:00+00:00" `
  --cutoff <ISO_DATE> `
  --timeframes 1h `
  --exchange binance `
  --max-gap-bars 1 `
  --seed 0
```

This historical research command intentionally does not use `--symbols`: the universe must match
`assets.yaml` exactly.  Funding must be backfilled before this snapshot.  The
closed `grid_atr` result remains `HISTORICAL_FAIL` and must not be rerun or
retrofitted with new intrabar evidence. Future candidate snapshots must add
`--calibration-id <CALIBRATION_ID> --execution-timeframe 1m --validate`.

`--max-gap-bars 1` is permitted only for documented, source-confirmed single
Binance outages.  Affected asset windows are excluded from IS and OOS rather
than interpolated; all other assets retain the same calendar.

### `grid_multi_tf` frozen certification policy (Sprint 70a)

### `grid_boltrend` certification policy (Sprint 70b follow-up)

`grid_boltrend` is the only active grid candidate in this workflow. `grid_atr`
and `grid_multi_tf` are closed `HISTORICAL_FAIL` cases and must not be rerun,
re-optimised or retrofitted. The frozen policy is: all 28 configured assets,
Binance closed 1h signals, Bitget 1m execution/funding, calendar
2022-01-01 UTC through the frozen cutoff, IS 180d / embargo 7d / OOS 60d /
step 60d, exhaustive 1,296 combinations, IS-only Top 8, 1,646 USDT, primary
5x and sensitivities 3x/5x/8x. The configured `cooldown_candles=3` is fixed.

The snapshot must use `--timeframes 1h`: validation adds exactly the consumed
Bitget 1m series, not unused Bitget 1h candles. Calibration must have at least
30 filled and one expired/cancelled/rejected Bitget entry observation. A shared
grid-capability calibration may be read from a separate immutable source DB.

For this full historical range, use `scripts.fetch_history --exchange bitget
--bitget-uta-history --timeframe 1m`: the explicit Bitget UTA v3 route retrieves
history older than the short classic-CCXT 1m retention. It retrieves bounded
16-page batches (1,600 candles) and commits each batch once, while remaining
below Bitget's 20-request/s public limit. On interruption, rerun the exact
same command: it queries SQLite and downloads only missing prefixes, internal
gaps and suffixes in the declared window. A certification snapshot
fails closed when any series does not reach the last closed candle before the
cutoff, or when Bitget execution starts after its Binance signal series. Run
snapshot creation in a separate clean Git worktree when the development
worktree contains user-owned edits; do not stash or alter those edits.

`grid_multi_tf` reuses the same universal snapshot, common-calendar WFO,
IS-only Top-N selection, external-OOS portfolio, shared `LiveRiskManager` and
parity evidence. Its 4h Supertrend is derived only from complete UTC Binance
1h buckets; native 4h rows are diagnostic and are not snapshot inputs.

The immutable policy is: 28 configured assets, calendar start 2022-01-01 UTC,
IS 180d, embargo 7d, OOS/step 60d, exhaustive 1,152-combination grid, Top 8,
seed 0, 1,646 USDT, primary 3x and sensitivity 2x/4x. New universal snapshots
persist the capital. Replay CLIs reject a different strategy, universe,
capital or leverage scenario.

```powershell
uv run python -m scripts.create_data_snapshot `
  --strategy grid_multi_tf `
  --universe-discovery `
  --calendar-start "2022-01-01T00:00:00+00:00" `
  --since "2022-01-01T00:00:00+00:00" `
  --cutoff <ISO_DATE> `
  --timeframes 1h `
  --exchange binance `
  --max-gap-bars 1 `
  --seed 0 `
  --top-n 8 `
  --primary-leverage 3 `
  --leverage-scenarios 2,3,4 `
  --portfolio-capital 1646
```

The snapshot is valid only from a clean implementation commit. No native 4h
backfill is required. Late-listed assets retain the global calendar and join
only once a complete IS/embargo/OOS interval exists.

## 3. Run or resume snapshot-bound WFO

```powershell
uv run python -m scripts.optimize `
  --strategy grid_atr `
  --all-symbols `
  --snapshot <SNAPSHOT_ID> `
  --resume `
  -v
```

`--resume` skips only assets already completed for the exact manifest hash. Results from another snapshot or legacy run never count. Snapshot mode remains incompatible with `--subprocess`; each asset failure is logged and the remaining assets continue.

For a universe-discovery snapshot this command is strict: it requires
the declared strategy with `--all-symbols`, forces the frozen 1h signal timeframe and evaluates
every valid parameter combination in every IS window.  The external replay
refuses to proceed if any declared asset lacks its snapshot-bound WFO result.

For Sprint 70a:

```powershell
uv run --isolated --python 3.12 --frozen python -m scripts.optimize `
  --strategy grid_multi_tf --all-symbols `
  --snapshot <SNAPSHOT_ID> --resume -v
```

## 4. Run or resume historical certification

To inspect the three pre-declared leverage scenarios before certification:

```powershell
uv run --isolated --python 3.12 --frozen python -m scripts.external_oos_portfolio `
  --strategy grid_atr `
  --snapshot <SNAPSHOT_ID> `
  --capital 1502.59 `
  --all-leverages
```

For an explicit compatible-WFO reuse after a portfolio-only correction:

```powershell
uv run --isolated --python 3.12 --frozen python -m scripts.external_oos_portfolio `
  --strategy grid_atr --snapshot <FRESH_PORTFOLIO_SNAPSHOT> `
  --wfo-snapshot <COMPATIBLE_WFO_SNAPSHOT> --capital 1502.59 --all-leverages
```

The dynamic asset choices and parameters are identical for 2x, 4x and 6x.
Only 4x is saved as `external_oos` and can become the primary historical
verdict.  2x and 6x are named sensitivity evidence; they can never replace
the declared 4x decision after seeing results.

For `grid_multi_tf`, 3x is the primary `external_oos` scope and 2x/4x are
pre-declared sensitivity scopes:

```powershell
uv run --isolated --python 3.12 --frozen python -m scripts.external_oos_portfolio `
  --strategy grid_multi_tf `
  --snapshot <SNAPSHOT_ID> `
  --capital 1646 `
  --all-leverages

uv run --isolated --python 3.12 --frozen python -m scripts.certify_strategy `
  --strategy grid_multi_tf `
  --snapshot <SNAPSHOT_ID> `
  --capital 1646
```

```powershell
uv run python -m scripts.certify_strategy `
  --strategy grid_atr `
  --snapshot <SNAPSHOT_ID> `
  --capital 1000
```

The orchestrator reuses matching evidence and creates only missing stages:

1. measured fast/canonical parity;
2. chronological external-OOS nominal portfolio;
3. adverse execution replay;
4. nominal fresh-capital replays over the last 180 and 365 days, using the
   same preselected external-OOS parameters/universe but restarting from the
   certification capital;
5. block bootstrap, empirical rolling 30-day CVaR and stress evidence;
6. strict historical gates.

Rerunning the same command is resumable and idempotent. To evaluate existing evidence without running missing stages:

```powershell
uv run python -m scripts.certify_strategy --strategy grid_atr --snapshot <SNAPSHOT_ID> --evaluate-only
```

Historical gates for `PAPER_READY` are: positive external-OOS return and bootstrap lower bound, loss probability below 10%, nominal DD at most 30%, adverse DD at most 40%, no 45% kill switch, simultaneous SL loss at most 30%, margin at most 70%, liquidation distance above 50%, positive degraded-cost return, complete funding/intrabar evidence and fast/canonical parity within 0.5%. Both fresh-capital windows must cover at least 95% of 180/365 days, remain profitable, keep DD at most 30% and trigger no kill switch.

Verdict precedence is strict: one failed performance gate is
`HISTORICAL_FAIL` even if 1m/calibration evidence is absent. If performance is
complete and passing but calibration, actual 1m consumption, parity or
coverage is insufficient, the status is `RESEARCH_ONLY`. `PAPER_READY`
requires all historical and operational gates. The canonical broker can now
satisfy the consumption gate, but only a qualifying frozen Bitget dataset and
calibration can provide the remaining operational evidence.

On Windows, any certification that runs missing portfolio evidence must use
the same isolated Python 3.12 runtime as `portfolio_backtest`:

```powershell
uv run --isolated --python 3.12 --frozen python -m scripts.certify_strategy `
  --strategy grid_atr `
  --snapshot <SNAPSHOT_ID> `
  --capital 1000
```

`--evaluate-only` is read-only and remains available from the normal runtime.

## 5. Record and review forward evidence

Forward evidence must contain raw per-intent observations, never manually entered aggregate metrics. JSON or JSONL rows are signed and idempotent; a conflicting rewrite of the same intent is rejected.

```powershell
uv run python -m scripts.record_forward_observations `
  --certification-id <CERT_ID> `
  --phase paper `
  --input <RAW_OBSERVATIONS.jsonl>

uv run python -m scripts.review_forward --certification-id <CERT_ID> --phase paper
```

Paper requires at least 60 days, 30 completed cycles and 100 entries, with 100% explained orders, sizes within 1%, at least 90% of fills inside the simulated interval, cumulative PnL deviation at most 20%, and no missing SL, orphan order or persistent state divergence.

Canary phases use `canary_1`, `canary_2`, `canary_3`, `canary_4` observations at 10%, 25%, 50% and 100% capital. Review each sequentially:

```powershell
uv run python -m scripts.review_forward --certification-id <CERT_ID> --phase canary
```

## 6. Create the local promotion artifact

```powershell
uv run python -m scripts.promote_strategy --certification-id <CERT_ID>
```

Promotion requires `LIVE_APPROVED`, unchanged clean code/config, completed paper and all four canary stages. A challenger must improve robust score by at least 15% without degrading adverse DD by more than three points. The command writes `data/promotions/<CERT_ID>.json`; it does not deploy or edit robot2.

## Statuses

- `RESEARCH_ONLY`: incomplete or unsupported canonical evidence.
- `HISTORICAL_FAIL`: complete historical evidence failed at least one gate.
- `PAPER_READY`: all historical gates passed.
- `LIVE_CANARY_READY`: forward paper passed.
- `LIVE_APPROVED`: all four canary stages passed.
- `REJECTED`: explicit operational rejection.

## Legacy commands

`portfolio_backtest`, `portfolio_robustness`, grades and `analyze_wfo_deep` remain useful research/diagnostic tools. Their standalone verdicts are not live authorization. Leave-one-out analysis may propose a future pre-registered challenger but may not alter the current external-OOS result after inspection.

`portfolio_backtest --save` is local-only. A legacy research row is sent to
the configured server only when `--push-server` is supplied explicitly;
snapshot-bound certification rows can never be pushed by that CLI.
