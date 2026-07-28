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
