# Sprint 68 — Reliable Backtests and Live Certification

## Objective

Build one reusable certification foundation for all registered strategies and validate the existing grid path without claiming live fidelity where evidence is missing.

## Delivered lots

1. Audit, clean baseline, result provenance, legacy status and idempotent migrations.
2. Immutable data snapshots, exact series/config/code hashes and calibrated `ExecutionSpec`.
3. Parameterized batch/incremental indicators and closed higher-timeframe visibility.
4. Common order/fill models, persistent grid limits, partial/late fills, fees, funding and restart state.
5. Shared-account portfolio risk with compounding, `max_live_grids=4`, correlation, margin and simultaneous-SL limits.
6. Snapshot-bound nested WFO, external-OOS chronological portfolio and measured fast/canonical parity.
7. Empirical robustness evidence and strict historical certification gates.
8. Raw signed forward observations, paper/canary state machine and local-only champion/challenger promotion.
9. Official workflow/commands and regression coverage.

## Important outcome

The infrastructure is fail-closed. Grid research can run end to end, but the current canonical portfolio consumes 1h execution bars. The explicit 1m-use gate prevents `PAPER_READY`; true intrabar execution is the next implementation lot. Mono-position and fast-only strategies are classified rather than routed through an incompatible engine.

## Operational safety

- Existing user config edits were preserved.
- No SSH, deploy, server config or robot2 mutation was performed.
- `optimize --apply` is blocked.
- Snapshot-bound WFO and certification are resumable and reuse exact matching evidence.
