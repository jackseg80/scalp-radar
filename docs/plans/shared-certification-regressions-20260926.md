# Shared certification regressions — 2026-09-26

## Scope and diagnosis before edits

User authorized resolving the six remaining tests, then checking data read-only.
Four failures share one real defect: `_run_fast` refers to `funding_exchange`
without a parameter, and optimize's funding-source selection is not forwarded
through coarse/fine/OOS batch dispatch. Two order tests share an unconstrained
MagicMock whose `fixed_entry_levels` attribute evaluates truthy; they silently
exercise the fixed-ladder branch instead of the intended repricing contract.
No runner/broker behavior change is justified by those two tests.

## Frozen experiment

Unchanged: grid_boltrend only; 28 assets, IS-only Top 8; 2022-01-01 through
2026-07-27 exclusive; IS 180d, embargo 7d, OOS 60d, step 60d; 1,296 combinations;
capital 1,646 USDT; primary 5x, sensitivities 3x/5x/8x. Binance closed 1h signals,
Bitget 1m execution and Bitget funding/calibration. Existing historical failures
are closed; no reruns, parameter tuning, snapshots or long candidate runs here.

## Implementation and checks

1. Forward optional funding_exchange from optimize through all batch search
   paths into the fast-cache builder; retain exchange fallback for legacy callers.
2. Set the shared test double's fixed_entry_levels to the actual base contract
   (false). Keep true fixed-ladder coverage and anti-look-ahead tests passing.
3. Add routing regressions for explicit Bitget funding vs Binance signal source,
   default source, dispatcher and all three search stages; run targeted/full tests
   with the existing test-only I/O guards active.
4. Inspect snapshot/calibration records and bounded per-series timestamp endpoints
   with read-only SQLite. Do not run full 1m integrity scans or generate a snapshot.
   Distinguish endpoint presence from complete/gap-free coverage.
5. Update audit/workflow/commands/roadmap with actual results and next evidence gate.

No robot2 change or connection is needed. Preserve all existing reporting/isolation
work and the user's AGENTS.md, CLAUDE.md and GEMINI.md. No strategy-specific engine
duplication, execution-model change or alteration of historical verdicts.

## Completed checkpoint

Funding routing and the mock contract corrected; five new routing tests added.
Targeted suite: 109 passed. Full `uv run python -m pytest --tb=short -q`:
2428 passed. No strategy parameters or production runner order logic changed.
Read-only indexed checks establish price endpoints but not full continuity;
funding history and three price-series starts remain blockers. The old snapshot
is INVALID and no grid_boltrend certification record exists. Exact evidence and
next gate: `docs/audit/audit-grid-boltrend-readiness-20260926.md`.

## Bounded source investigation follow-up

On user continuation, inspect official documentation and only small public GET
samples; do not backfill, alter the frozen experiment or contact providers.
Completed: nine BTC funding-page probes and six first-candle probes, plus public
archive catalogue checks. No qualifying full-range funding archive identified.
Record evidence in `docs/audit/audit-bitget-history-recoverability-20260926.md`.
Next action requires an archive/support decision from the user, not a code bypass.
