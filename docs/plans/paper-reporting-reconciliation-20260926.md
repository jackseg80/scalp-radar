# Paper reporting reconciliation — 2026-09-26

## Scope and frozen decisions

Repair reporting only. Do not change strategy parameters, execution, risk,
automatic selection, stored trades, historical verdicts, or robot2.
The Sprint 70b experiment remains all 28 assets, IS-only Top 8, 2022-01-01
through 2026-07-27 exclusive, IS 180d / embargo 7d / OOS 60d / step 60d,
1,296 combinations, capital 1,646 USDT, primary 5x, sensitivities 3x/5x/8x.
No WFO, portfolio replay or certification run is authorized here.

## Implementation plan

1. Keep the synchronous Arena interface used by automatic selection unchanged.
2. Add a shared async reporting path for both ranking APIs and strategy detail.
   Read the latest N persisted trades for the strategy, where N is the frozen
   runner count. Order ties by database ID. Never merge pre-reset history or
   append the in-memory trades again.
3. Reconcile count, wins/losses and sum(net_pnl) minus cumulative funding with
   the runner snapshot. Without an explicit legacy session ID this is an
   inferred, reconciled window, not proof of unchanged historical parameters.
   Missing/mismatched/unavailable history yields null PF/DD and a reason.
4. Expose window dates, trade counts, funding and metric basis. PF and DD use
   closed trades net of trading costs but exclude settlement cash flows and
   unrealized PnL. Do not invent funding timestamps.
5. Test restart, pre-reset trades, tie ordering, reconciliation failures,
   no trades, funding paid/received, concurrent updates and all reporting APIs.
   Run targeted tests and `uv run python -m pytest --tb=short -q`.
6. Record audit, limitations and test results in project documentation.

## Acceptance boundaries

No deployment or restart. User files AGENTS.md, CLAUDE.md and GEMINI.md are
untouched. A positive paper result does not emit a certification verdict.

## Execution outcome

Local reporting implementation and 99 targeted tests are complete. Both broad
test attempts were stopped after discovering real WFO/external-I/O side effects
in legacy tests. The full suite is unvalidated; see the dated audit for exact
local artifacts. No commit, push or deployment at that checkpoint.

## Authorized follow-up: cleanup and test isolation

The user authorized cleanup and isolation on 2026-09-26. Archive result 2466,
its three combo rows and two reports before removal; re-select result 2465
according to the existing latest-run policy, without pretending its prior flag
was independently recorded. Keep the small archive in `data/test-incident-20260926`.

Block repository data/config mutations, repository SQLite connections, external
network access and real WFO workers from tests. Replace the old worker tests
with a real subprocess protocol test injecting a synthetic optimization function.
Exercise WFO callbacks/parameter merge/cancellation on synthetic candles with
mocked backtests, and give lifespan tests a temporary working directory.
Keep these safeguards in test code only; production behavior stays unchanged.
Verify targeted guards first, then run the complete suite. Record actual
failures without extending the work to unrelated strategy/engine changes.

Completed: guarded cleanup, synthetic worker/callback tests, temporary lifespan
directories and test-only I/O protections. 129 targeted tests pass. Full suite
terminates with 2417 passing and six reproducible failures in unchanged engine
paths. These remain the next blocker; no commit/push/deploy and no new historical
candidate runs. Cleanup is recoverable from the incident archive.

Subsequent authorized work is recorded in
`shared-certification-regressions-20260926.md`: the six failures are resolved,
109 targeted and 2428 full-suite tests pass. Data evidence, not these tests,
is now the next certification gate. No deployment occurred.
