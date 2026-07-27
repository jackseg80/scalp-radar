# Backtest and Live Certification Audit — 2026-07-22

## Scope

Audit of the existing WFO, fast engines, event-driven simulator, portfolio engine, paper/live execution, SQLite schemas and operational commands before extending them into a live-certification workflow.

Baseline before implementation: commit `48ce9cd6af314c59bcc0116310d7522c97fdca8f`, 2269 tests passed in 212.79 seconds. Existing user changes in `config/param_grids.yaml`, `config/strategies.yaml` and untracked `AGENTS.md` were preserved.

## Reuse classification

| Area | Classification | Decision |
|---|---|---|
| Closed-bar chronology (`closed_bar_v2`) | EXISTING_OK / EXTEND | Preserved and extended to persistent order events and external windows. |
| `GridStrategyRunner`, `GridPositionManager`, `PositionManager` | EXTEND | Reused for canonical grid replay, common execution spec and restart state. |
| Fast engines | EXISTING_OK / EXTEND | Kept for search; candidates now require measured canonical parity. |
| Incremental indicator cache | EXTEND | Parameterized per strategy/asset and checked against batch definitions. |
| Bitget Executor order state | EXTEND | Reused for calibration observations, partial/late fills and unfilled outcomes. |
| `LiveRiskManager` | EXTEND | Reused by portfolio pre-trade checks for grids, correlation, margin and SL loss. |
| WFO and portfolio tables | EXTEND | Idempotent migrations added; no parallel result database. |
| Portfolio engine | EXTEND | Shared account, compounding, pending orders, rejection ledger and external OOS. |
| Robustness script/table | EXTEND | Empirical 30-day CVaR, clustered regimes and signed evidence added. |
| Live certification orchestration | MISSING | Minimal orchestrator and state machine added around existing commands. |
| Mono-position canonical shared-account replay | MISSING | Explicitly fail-closed; no incompatible grid simulation. |
| True 1m intrabar execution in canonical portfolio | MISSING | Explicit gate added; certification cannot reach `PAPER_READY` until implemented. |

No second indicator cache, statistical engine, WFO table, portfolio table or exchange order abstraction was introduced.

## Reliability findings corrected

- Static portfolio backtests and grades could be mistaken for live authorization.
- Final WFO selection used OOS information; selection is now inner/IS-only with external-OOS concatenation.
- DSR used inconsistent Sharpe scaling/effective trial count.
- Portfolio sizing did not fully reproduce shared account limits, compounding, correlation and `max_live_grids=4`.
- Indicator periods could diverge between WFO and incremental live paths.
- Snapshot provenance did not exist and exchanges could be confused across long-history and transfer checks.
- Robustness used extrapolated CVaR rather than empirical rolling 30-day returns.
- A snapshot containing 1m rows could appear intrabar-complete even though the replay consumed only 1h OHLC.
- Late Bitget fills after a confirmed cancellation could conflict with the unique order observation.
- Re-running historical evaluation could erase paper/canary progress.
- Direct `optimize --apply` bypassed forward and canary evidence.
- Long Windows portfolio replays allocated full ADX working arrays on every candle while `grid_atr` flooded two queued Loguru file sinks at INFO level. The ADX path is now scalar/allocation-free and the repetitive floor diagnostic is DEBUG-only.
- The portfolio CLI now uses synchronous warning/error-only file logging. It deliberately avoids Loguru's Windows multiprocessing queue writers, which were the only concurrent native threads present in repeated access-violation traces.
- Windows Error Reporting ultimately confirmed faults in `python313.dll` (`0xc0000005` and `0xc0000409`) even after queued logging was removed. Long Windows portfolio replays now fail closed on Python 3.13 and use a frozen isolated Python 3.12 runtime. A real 365-day, eight-asset replay completed under Python 3.12 with code 0 and zero missing funding settlements.
- The 2025-10-10 flash-crash audit found that nominal SL fills were interpolated between the server trigger and the full 1h candle wick. This converted configured 20% stops into simulated 42-52% losses. Server-side SL fills now use the trigger for intrabar crossings and only degrade against an observable opening gap; normal market slippage remains charged separately.
- The live exit monitor could route a locally observed SL crossing through a close path that assumes Bitget had already filled the trigger, removing local state without confirmation. Local SL observations now wait for the existing `watchOrders`/position-poll confirmation and real exchange fill.
- Long-history compounding could hide poor outcomes for a newly funded account. Certification now replays the same preselected external-OOS decision stream from fresh capital over both 180 and 365 days and requires positive return, DD at most 30%, no kill switch and at least 95% period coverage.
- Portfolio saves no longer trigger implicit server synchronization. Remote push is explicit (`--push-server`) and remains forbidden for snapshot-bound certification evidence.

## Deliberate fail-closed limits

- The grid canonical replay reports `execution_timeframe_used=1h`; the required snapshot execution timeframe is normally 1m. `intrabar_execution_used` therefore fails.
- Mono-position strategies and `trend_follow_daily` remain `RESEARCH_ONLY` until their canonical account replay/live runner exists.
- Promotion creates a local signed candidate artifact only. It never deploys or edits robot2.
- Historical data through 2026-07-21 is development/nested-WFO data, not an intact holdout. The first independent validation is forward after candidate freeze.

## Acceptance evidence

Targeted suites cover snapshots and idempotent migrations, indicator parity, closed-bar execution, persistent/partial/late fills, funding, shared-account risk, external-OOS selection, engine parity, robustness persistence, raw forward observations, paper/canary gates and champion/challenger promotion. The simultaneous-SL cap is fixed against initial account capital even after compounding. See the final sprint entry in `docs/ROADMAP.md` for the authoritative complete-suite count.
