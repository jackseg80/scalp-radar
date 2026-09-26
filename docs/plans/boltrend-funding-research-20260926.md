# grid_boltrend synthetic-funding research — pre-registration 2026-09-26

## Authority and scope

The user approved a separate approximate funding study. Preserve the original
strict historical certification and all closed HISTORICAL_FAIL cases. No robot2
changes, candidate runs, bulk downloads or post-hoc parameter choices by the agent.
This plan precedes implementation and any study OOS observation.

## Fixed experiment

Retain the 28 configured assets, Binance closed 1h signals, Bitget 1m execution,
2022-01-01 inclusive / 2026-07-27 exclusive, IS 180d / embargo 7d / OOS 60d /
step 60d, exhaustive 1296 combinations, IS-only Top 8, capital 1646 USDT,
primary leverage 5x and sensitivities 3x/5x/8x, seed 0. Keep existing fees,
execution calibration, order semantics and shared risk limits unchanged.

Profile `boltrend_funding_v1`, rates in percent per UTC 00:00/08:00/16:00
settlement, applied to positions by the existing engines (entry-notional basis;
not an exact reconstruction of exchange mark-price settlement):

| Scenario | Rate hypothesis |
| --- | --- |
| central | +0.01% at every settlement |
| positive_stress | +0.03% at every settlement |
| negative_stress | -0.03% at every settlement |
| positive_shocks | +0.10% on calendar days 1–7 each UTC month, +0.01% otherwise |
| negative_shocks | -0.10% on calendar days 1–7 each UTC month, +0.01% otherwise |

These are synthetic assumptions, not historical means, confidence bounds or
worst-case guarantees. Positive rates debit longs and credit shorts; negative
rates reverse that. Calendar shocks do not depend on prices, trades or outcomes.
No rate is estimated using later observed funding. No blending with real samples.

Run WFO once under central funding. Replay the same IS-only window selections
under all five hypotheses with nominal execution; do not optimize per hypothesis
or choose the best scenario/leverage after OOS. Preserve all results including
losses, liquidations and interrupted/failed scenarios. Each scenario carries its
own capital across windows. Interpret return, drawdown, funding and risk events
together, without a new performance-derived pass/fail threshold.

## Minimal implementation

1. Add a frozen, validated research funding specification to ExecutionSpec.
   Central snapshot signs the complete profile and the fixed scenario family.
2. Reuse cache funding arrays and the canonical funding provider; debit/credit
   inside the event loop before subsequent sizing/risk decisions, not at report end.
   Preserve all ordinary/historical funding behavior when the option is absent.
3. Extend common snapshot creation/revalidation only for explicitly synthetic
   funding coverage. Keep candle, calibration, provenance and availability gates.
   Never write generated rates into the market-data tables.
4. Propagate central assumptions through all WFO paths; no silent fallback to
   unfunded engines. Use the same external-OOS replay for the five scenarios.
5. Persist research assumptions and distinct evaluation scopes; refuse synthetic
   evidence in certification, including optimistic otherwise-passing results.
6. Add targeted routing, units/sign/UTC, snapshot and rejection tests; run the
   complete guarded pytest suite. Update audit, workflow, commands and roadmap.

## Known independent blocker and handoff

The old snapshot is INVALID. Missing/invalid candles and FET/OP/SUI start mismatches
are not waived by synthetic funding. Supply preparation commands first; do not
launch WFO if the new snapshot is invalid. A separate decision is required to
change exchange-availability rules or the calendar. No PAPER_READY/LIVE_APPROVED
claim can be derived from this study. Retain RESEARCH_ONLY provenance throughout.

## Implementation checkpoint

Implemented without strategy/risk parameter changes. Initial focused suite:
111 passed; final focused research/WFO suite: 28 passed; complete guarded suite:
2452 passed. Preparation/replay commands are archived in COMMANDS.md, explicitly
conditional on resolving price coverage first. No candidate run, research
snapshot, observed-rate rewrite or robot2 intervention was performed. Audit:
`docs/audit/audit-boltrend-funding-research-20260926.md`.
