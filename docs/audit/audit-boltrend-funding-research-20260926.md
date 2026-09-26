# grid_boltrend synthetic funding study — 2026-09-26

## Outcome and limits

Prepared a separate research path; **no historical strategy run was executed**.
The user approved synthetic funding sensitivity rather than abandoning research
because a full settled-rate archive could not be recovered. The original strict
certification remains unchanged and blocked. Synthetic results cannot authorize
PAPER_READY or LIVE_APPROVED and cannot become a historical-failure verdict.

Pre-registration: `docs/plans/boltrend-funding-research-20260926.md`, written
before implementation or any study OOS observation. Universe, calendar, WFO,
selection, capital and leverage policy are unchanged. The central +0.01% rate
is a hypothetical reference, **not a measured historical average**.

## Implementation

- `ResearchFundingSpec` freezes profile version, units, UTC schedule, constants
  and shock calendar. No empirical sample or future price is used to set rates.
- ExecutionSpec persists the research marker; the snapshot hash and WFO reuse
  fingerprint include it. Real and synthetic WFO evidence cannot be reused
  interchangeably. Market funding tables are not overwritten or filled.
- Snapshot validation exempts only observed funding coverage for this explicitly
  synthetic study. Candle OHLC/gaps/cutoff/start, clean code/config provenance and
  CLI calibration requirements remain. An INVALID old snapshot is never relabeled.
- WFO selects using central funding. The existing fast-cache array receives
  decimal rates; event-driven winner and stability diagnostics receive the same
  hypothesis through existing extra-data alignment. Research engine errors are
  fatal, not silently retried without funding or skipped as successful windows.
- Canonical portfolio runners obtain rates from a tiny provider with the same
  interface as historical funding. Existing settlement, capital, margin and risk
  logic are reused. No grid, fast engine, 1m broker or LiveRiskManager duplication.
- External OOS replays the same IS-only selections for all five scenarios and
  all declared leverages (15 runs for 3x/5x/8x), with fresh initial capital per
  scenario and normal capital carry across its windows. Nominal execution only.
- Persisted scopes are `funding_research_<scenario>_<leverage>x`, status
  RESEARCH_ONLY. Reports display a synthetic-funding banner and serialize the
  exact profile/scenario. The usual strict `external_oos` scope is not used.
- Strict certification CLI rejects the marker. The gate evaluator also returns
  RESEARCH_ONLY for synthetic evidence, even with otherwise passing or losing
  performance. Ordinary historical gate precedence is unchanged.

## Model limitations and open gate

All assets use the same invented signed rate per settlement; shocks occur on
UTC calendar days 1–7 each month. The scenarios are neither confidence intervals
nor worst-case bounds. They do not reconstruct actual market/position correlation.
The existing engines use entry-price notional for funding, not historical mark
price; their existing 8h UTC schedule is another approximation. These limitations
are disclosed rather than changing execution/accounting for this research feature.

No execution prices have been invented, forward-filled or substituted. The
FET/OP/SUI start mismatches remain a blocker to the unchanged 28-asset snapshot;
full candle continuity is still unverified. Decide separately how to establish
and pre-register exchange availability before a long snapshot/WFO run. Do not
rerun a multi-year download simply to try this option.

Auxiliary WFO grades/transfer diagnostics remain non-selection diagnostics;
neither those grades nor the best funding scenario constitutes a study verdict.
Inspect all scenario results, funding costs, drawdowns, liquidations and risk
events together. A partial 15-run matrix is incomplete evidence.

## Validation

- Initial focused suite: 111 passed.
- Additional central routing tests cover coarse, fine, OOS batch and winner
  evaluation; non-central WFO and other strategies are rejected before simulation.
- Final focused research/WFO suite: 28 passed.
- Complete guarded `uv run python -m pytest --tb=short -q`: **2452 passed**
  (2428 baseline plus 24 new tests). No skipped research failures.
- CLI `create_data_snapshot --help` exposes the explicit research option.
- Static F-rule check on changed Python files reports 13 existing findings in
  `scripts/optimize.py` only; the same findings reproduce on its HEAD version
  (unused import, annotation names, placeholder-free f-strings). No automatic
  cleanup was applied to unrelated lines. `git diff --check` passes.
- Temporary-database tests show absent funding is accepted only for synthetic
  research, missing execution prefixes still fail, revalidation preserves these
  rules, and no synthetic rates are inserted into the funding table.
- No robot2 connection, deployment, reconfiguration, data backfill or strategy
  parameter edits. User AGENTS.md, CLAUDE.md and GEMINI.md remain unchanged.
