# grid_atr Decision Record — 2026-07-26

## Scope

This record rejects the currently declared `grid_atr` candidate on `robot2`.
It does not yet close the historical evaluation of alternative parameters
selected by the snapshot-bound WFO, and it does not change any live
configuration or trading process.

The evaluated universe and configuration are the immutable robot2 YAML
snapshot at `data/config_snapshots/robot2_20260725`:

- Assets: ADA, ATOM, BTC, CRV, ETC, ICP, NEAR, SOL, XRP (all `/USDT`);
- Initial capital: 1,502.59 USDT;
- Leverage: 6x;
- Exchange history: Binance 1h;
- Snapshot: `snapshot-b6cd4b2ced5e31aa`, from 2023-03-25 through
  2026-07-21T08:00:00Z.

The start date excludes an irreparable source gap in the Binance 1h series.
All nine retained series contain 29,144 closed candles with no duplicate,
invalid OHLC, irregular timestamp, or missing bar.

## Evidence

### Exact declared-portfolio replay

The 365-day canonical portfolio replay at 6x returned -29.4%, reached a
-32.1% maximum drawdown, and triggered the grid session kill switch. The
longer replay also triggered the same session control.

Leverage research did not restore a robust economic edge:

| Leverage | 365-day return | Maximum drawdown |
| --- | ---: | ---: |
| 2x | +0.3% | -13.6% |
| 4x | -1.4% | -26.6% |
| 6x | -29.4% | -32.1% |

### Snapshot-bound WFO

The 9-asset WFO produced 17 external windows. Seven assets were Grade D
shallow. BTC and SOL were Grade B shallow diagnostics only; neither has
cross-exchange validation or sufficient independent windows. They are not
an asset-selection basis for a new portfolio.

The evidence remains `RESEARCH_ONLY`: Bitget execution observations and 1m
intrabar execution are unavailable, so no result can satisfy the
`PAPER_READY` or `LIVE_APPROVED` gates. The WFO did search alternative
parameters, but its selected IS candidates still require the pre-registered,
chronological external-OOS canonical portfolio replay.

## Decision

`grid_atr` with the declared robot2 universe and 6x risk profile is
**REJECTED / HISTORICAL_FAIL**. Do not promote it, add capital, expand its
universe, or perform post-hoc portfolio pruning based on the WFO grades.

This does not prove that every possible `grid_atr` parameter set is
unprofitable. It does close the declared universal 4x evaluation: its
WFO-selected parameters and IS-selected Top 8 universe have now failed the
external-OOS canonical portfolio replay.

## Universal-discovery external OOS result

The exhaustive 28-asset WFO from `snapshot-035a6e48851ac4c1` was explicitly
reused only after a matching WFO-input fingerprint check (series, YAML,
calendar, grid, seed and IS-only policy) by the fresh canonical replay
snapshot `snapshot-d97f794987803145`. The portfolio contains the union of 27
assets selected over time, never more than the declared Top 8 per window.

| Scenario | Return | Maximum drawdown | Decision |
| --- | ---: | ---: | --- |
| 2x sensitivity | +60.4% | -22.3% | informative only |
| **4x primary** | **+92.9%** | **-47.5%** | **HISTORICAL_FAIL** |
| 6x sensitivity | +112.5% | -56.3% | fail |
| 3x post-result sensitivity | +96.4% | -32.4% | fail |

The primary 4x scenario has complete funding, no account kill switch, peak
margin 51.9%, worst simultaneous SL loss 29.9% and liquidation distance
99.2%; it nevertheless fails the non-negotiable nominal drawdown gate of
30%. The 2x and 3x results cannot retroactively replace the pre-declared 4x
decision. A future 2x candidate would need a separate, pre-declared WFO and
external-OOS cycle.

No robot2 deployment, configuration or order was changed for this decision.

## Future promotion conditions

Promote a passing historical challenger only after all of the following are
available:

1. Persisted Bitget order lifecycle observations, including accepted orders,
   fills, partial/missed fills, cancellations, and attached TP/SL orders.
2. A calibrated Bitget execution specification and 1m intrabar broker path.
3. A pre-declared universe, leverage, SL-risk and correlation limits; no
   parameter or asset changes after observing the portfolio result.
4. A new snapshot and external-OOS, canonical portfolio, adverse-execution,
   and forward-paper evidence that pass the certification gates.
