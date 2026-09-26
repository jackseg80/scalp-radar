# Paper reporting audit — 2026-09-26

## Latest checkpoint: shared regressions resolved

The subsequent authorized correction passes **109 targeted and 2428 full-suite
tests**. Four previous failures were missing funding-source propagation; two
were a test double selecting the wrong ladder branch, not production order bugs.
Data readiness still blocks certification. See
`audit-grid-boltrend-readiness-20260926.md` and the corresponding archived plan.
The checkpoints below retain their original validation state for traceability.

## Earlier checkpoint: authorized cleanup and isolation complete

After explicit user authorization, result 2466 and its three combo rows were
exported with full row contents to `data/test-incident-20260926/backup.json`
before transactional deletion. Both reports were copied there, SHA-256 verified,
then removed from the active optimization folder. Prior result 2465 was
re-selected under the existing latest-run policy; this is a documented
reconstruction, not a claim that its pre-incident flag was captured.
The possible older intermediate file cannot be recovered from this archive;
it contains the incident-written version only. No unrelated history was deleted.

Post-test read-only verification: result 2466 absent, zero associated combo
rows, latest envelope_dca/BTC result 2465, no results created after the original
incident launch. Backup and reports remain recoverable locally. No robot2 changes.

Test-only safeguards now reject repository SQLite connections, mutations under
`data/` and `config/`, non-loopback network access, and real WFO-worker launches.
The real subprocess/protocol test injects a synthetic optimizer and installs
the same guard inside its child. Callback/cancellation/parameter-merge tests use
synthetic candles and mocked backtests. Lifespan tests use temporary directories.
There are no production-engine or strategy changes in this follow-up.

Validation:

- Changed-scope reporting/isolation/API/database/selector tests: **129 passed**.
- Full `uv run python -m pytest --tb=short -q`: **2417 passed, 6 failed**.
- The six failures reproduce alone. Their engine and test files have no local
  diff: four `_run_fast` failures (`funding_exchange` undefined), one missing
  expiry/replacement audit event, and one reused replacement-order identity.
  They remain blockers; they were not hidden, skipped or fixed by changing
  strategy parameters. Full-suite validation is still not green.
- `git diff --check` passes. CLAUDE.md, GEMINI.md and AGENTS.md SHA-256 hashes
  are unchanged from this follow-up's entry inventory. No commit/push/deploy.

Next: address the shared-engine failures under a separate explicit scope, then
recheck frozen data coverage/calibration before any user-run certification.
The earlier incident narrative below is retained as an audit trail.

## Read-only observation

SSH alias `robot2` only; no deployment, restart, configuration change or order.
The host checkout identified itself as `48ce9cd`; the active container's
simulator source hash matched the host file. The container started on
2026-09-09. This is not a claim that every running file matches that Git commit.

`grid_boltrend` is paper-only, 1h, leverage 5, watching BCH/BTC/DOGE/DYDX/ETH/LINK.
The current 65-trade state reconciles to the latest 65 database trades, first
entry 2026-07-20 16:00 UTC, last exit 2026-09-26 13:00 UTC:

- Initial capital: 1,646 USDT.
- Closed-trade PnL after trading fees/slippage: +132.888491 USDT.
- Cumulative funding charged: 12.515985 USDT.
- Realized PnL after funding: +120.372506 USDT (+7.313%).
- Unrealized PnL at observation: +82.47 USDT; equity 1,848.85 USDT.
- Wins: 21/65; closed-trade profit factor approximately 1.1183.
- Closed-trade peak-to-trough drawdown excluding funding: 31.4663%, from
  2,158.248928 to 1,479.128163 USDT (August 26 to September 17).

These are dated observations, not a stable-config forward certification.
Older database rows exist and must not be silently joined to the current
runner state. The actual full mark-to-market drawdown is not established here.

## Confirmed reporting defect

Arena combines persisted runner counters with `runner.get_trades()`, which
contains only trades since process restart. Its reported PF 0.785869 and DD
23.449642% described 11 trades (September 11–26), while its gain, win rate and
trade count described the 65-trade state. Two periods were presented together.

## Local correction

The HTTP ranking/performance/detail endpoints use one shared async reporting
path. It retrieves the latest N trades for the strategy, ordered by exit time
and database ID, and checks counts, wins/losses, in-memory trade identities and
sum(net PnL) minus cumulative funding against a frozen runner snapshot. A runner
change during the query, unavailable DB, missing or mismatched history returns
null PF/DD with a machine-readable reason; post-restart metrics are not used as
a silent fallback. Detail returns the same reconciled trades without duplication.

The existing synchronous Arena interface, automatic selection, runner state,
execution, strategy parameters and historical certification results are unchanged.
The WebSocket ranking has no PF/DD fields and remains on that unchanged path.

Legacy rows lack a session ID. The reconstructed latest-N period is explicitly
labelled as inferred/reconciled, not proven session membership or proof that
parameters remained unchanged. PF/DD exclude funding timing and unrealized
PnL; the realized return includes cumulative funding. No settlement chronology
is fabricated, and no funding correction is applied twice. An all-win sample
has a null PF with `profit_factor_unbounded=true` for JSON safety.

## Certification boundaries

The observed production paper used `closed_bar_v2`, 1h execution decisions and
current funding with a 0.01% fallback when unavailable. It is not the frozen
Binance 1h / Bitget 1m broker replay. Positive paper does not grant PAPER_READY
or LIVE_APPROVED. The July frozen policy and closed HISTORICAL_FAIL verdicts
remain untouched. No candidate certification run was launched. See the test
side-effect incident below; an existing integration test did launch a real
unrelated WFO despite the task's no-WFO constraint.

## Validation

Targeted suite: 99 passed (reporting, Arena, simulator API, database, automatic
selector), including identity-anchor regressions. `git diff --check` passes.
The required full command was attempted, then interrupted after discovering
that `tests/test_job_manager_wfo_integration.py` is not isolated: its temporary
DB stores only the job queue; `scripts.wfo_worker` calls `run_optimization`
without passing a data/output DB, which reaches the real local data directory.
The complete suite is therefore NOT validated. The second attempt, excluding
that file, also had to be interrupted: `tests/test_walk_forward_callback.py`
directly invokes the real optimizer for `envelope_dca/BTC` with the default
local database. Logs showed 33 windows, 2,160 combinations and a fallback from
the fast engine (`funding_exchange` undefined). No fix to that unrelated engine
was attempted. No broad-suite pass is claimed; validation is limited to the
99 targeted passing tests. No commit or push was performed.

## Test side-effect incident (local only)

The full suite launched `envelope_dca/BTC` WFO. The first worker completed
before the isolation defect was identified; the suite and subsequent worker
were stopped. No pytest-80 WFO worker remained in the process inventory.
This was outside the user's no-WFO constraint and was disclosed immediately.
No robot2 access or deployment occurred during implementation/testing.

The second suite (pytest-82, started 21:29:36 local) was stopped too, including
its main-process optimization. Its logs also reported a failed Telegram send
after two attempts; there is no successful delivery evidence in the inspected
log. This is another isolation defect requiring review before any broad rerun.
The final read-only check found only result 2466 after the first test launch,
the same two optimization report files, and no remaining identified test workers.

Confirmed local artifacts, retained without blind cleanup:

- `data/scalp_radar.db`: optimization result ID 2466,
  `envelope_dca`, `BTC/USDT`, created `2026-09-26T21:26:10.146052`,
  `is_latest=1`, plus three associated `wfo_combo_results` rows. It is the only
  optimization result found with a creation timestamp after this test launch.
  Prior result 2465 is now `is_latest=0`; its before-test flag
  was not captured, so no automatic rollback was attempted.
- `data/optimization/envelope_dca_BTC_USDT_20260926_212610.json`.
- `data/optimization/wfo_envelope_dca_BTC_USDT_intermediate.json` was written;
  whether it replaced an earlier intermediate file is not established.

These artifacts are not evidence for grid_boltrend certification and are not
included in the code change. Repairing isolation of the old integration tests
and a precise cleanup/restore decision require a separate follow-up. Do not
rerun the unrestricted suite on the populated research workspace meanwhile.
