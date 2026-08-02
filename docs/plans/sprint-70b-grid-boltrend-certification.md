# Sprint 70b follow-up — Grid BolTrend certification plan

Date: 2026-07-28

## Frozen policy

- Strategy: `grid_boltrend`; `grid_atr` and `grid_multi_tf` are closed historical failures.
- Universe: all 28 configured assets; IS-only Top 8 for each OOS window.
- Calendar: 2022-01-01T00:00:00Z to the snapshot cutoff (exclusive).
- WFO: IS 180d, embargo 7d, OOS 60d, step 60d, exhaustive 1,296 combinations.
- Fixed: 1h signals, both sides, cooldown 3, no max-hold; capital 1,646 USDT; primary 5x; sensitivities 3x/5x/8x.
- Parameter grid: bol window 50/100, std 1.5/2/2.5, long MA 200/400, min spread 0/.01, ATR 10/14, levels 2/3/4, SL 10/15/20, spacing .5/1/1.5.

## Evidence sequence

1. Repair Binance 1h, Bitget 1m and Bitget funding history without modifying robot2.
2. Create a shared-grid Bitget calibration with >=30 filled and >=1 confirmed unfilled entry outcomes.
3. On a clean implementation commit, freeze a validated snapshot (Binance 1h + exactly Bitget 1m).
4. Run exhaustive snapshot-bound WFO, then `external_oos_portfolio`, then `certify_strategy`.
5. Accept only the emitted precedence verdict: `HISTORICAL_FAIL`, `RESEARCH_ONLY` or `PAPER_READY`; no parameter/universe/risk adjustment afterwards.
## Follow-up: invalid snapshot recovery (2026-07-28)

The first frozen snapshot was rejected before WFO because Bitget 1m coverage
was stale/incomplete. The approved remediation is deterministic data recovery,
not a policy change: repair Binance 1h gaps, backfill Bitget 1m through the
explicit public UTA v3 long-history endpoint, preserve the fixed calibration
and recreate the snapshot from a clean sibling worktree. Snapshot validation
now requires every series to reach the last closed candle before cutoff and
requires Bitget execution coverage to begin no later than its signal series.
UTA requests are bounded at 16 concurrent pages per second and persisted in
1,600-candle batches to make the 28-asset range operationally tractable.
Interrupted runs scan the requested range and resume only missing intervals;
they never re-download a complete symbol merely because `--since` is present.
Funding uses the UTA v3 cursor history too; a 15-row CCXT sample cannot qualify
as frozen 2022–2026 funding evidence.
Measured UTA retention is only about 90 days, so an immutable full-period
Bitget funding archive is a hard prerequisite before snapshot/WFO execution.
