# grid_boltrend readiness audit — 2026-09-26

## Conclusion

Technical regressions resolved; historical certification **not completed**.
Treat grid_boltrend as research-only pending evidence, not HISTORICAL_FAIL or
PAPER_READY. No certification verdict was written. The positive paper observation
is separate from the frozen historical experiment; see the paper reporting audit.

No robot2 access or changes in this follow-up, no candidate WFO/optimization,
portfolio backtest, snapshot generation or certification run. Closed grid_atr
and grid_multi_tf verdicts remain untouched. No parameters were tuned.

## Minimal common-code correction

Four failing tests exposed `_run_fast` referencing undefined `funding_exchange`.
The shared optimizer now forwards the optional funding source through coarse,
fine and OOS batch dispatch into the cache. Legacy calls retain `exchange` as
fallback. This preserves Binance signals with Bitget funding, without a new engine.

Two order tests used a MagicMock with no explicit `fixed_entry_levels` boolean.
Its truthiness selected the immutable ladder branch. Set the test double to false,
matching BaseGridStrategy; do not change runner behavior to satisfy a faulty mock.
GridBolTrend's true immutable-ladder contract remains covered by parity tests.

Validation with test-only data/network/worker safeguards enabled:

- 109 targeted tests passed (intrabar broker, multi-timeframe, paper execution,
  funding routing, WFO callbacks, grid_boltrend strategy and parity).
- Full `uv run python -m pytest --tb=short -q`: **2428 passed**.
- Five additional tests cover explicit funding source, legacy defaults, dispatcher
  forwarding and coarse/fine/OOS propagation. Synthetic data only.

Earlier reporting corrections and incident cleanup are documented separately in
`audit-paper-reporting-20260926.md`; no research result from that incident is
certification evidence. The recoverable local archive is not committed.

## Local data evidence (read-only)

Database: `D:\Python\scalp-radar\data\scalp_radar.db`, opened with SQLite
`mode=ro`. Indexed first/last queries and small metadata tables only; bounded
queries completed in under one second. No migration, download or data rewrite.

All 28 Binance 1h series end at 2026-07-26 23:00 UTC before the cutoff; all 28
Bitget 1m series end at 2026-07-26 23:59 UTC. **Endpoints do not establish candle
continuity, OHLC validity or a certifiable snapshot.** Full validation is pending.

Three Bitget starts are later than the corresponding Binance signals:

| Symbol | Binance 1h first timestamp (UTC) | Bitget 1m first timestamp (UTC) |
| --- | --- | --- |
| FET/USDT | 2023-01-17 02:00 | 2023-03-02 08:30 |
| OP/USDT | 2022-06-01 14:00 | 2022-06-13 03:27 |
| SUI/USDT | 2023-05-03 16:00 | 2023-05-04 09:52 |

24 other Bitget series start at 2022-01-01. ARB starts at 2023-03-23 14:18,
before its Binance first signal at 15:00. Exchange listing dates and recoverability
of the three missing prefixes have not been established by these database checks.

Each of the 28 Bitget funding series has 270 rows, from 2026-05-04 16:00 to
2026-08-02 08:00 UTC. BTC has 250 rows strictly before the frozen 2026-07-27
cutoff. Thus the local archive cannot supply the frozen 2022–2026 experiment.
This is evidence about local contents, not a fresh universal claim about current
Bitget API retention. No external source was queried in this follow-up.

Calibration record `cal-1b1bb1cce72e7cd8` is present: Bitget, shared strategy scope,
30 filled, 3 confirmed unfilled, 3 partial observations, window July 23–26, 2026.
Its counts meet the configured thresholds; underlying evidence hashes were not
recomputed in this bounded check.

`snapshot-5a23b693c8fb56af` remains **INVALID** in the database. Other strategies'
persisted VALID snapshots do not substitute for grid_boltrend evidence and were
not revalidated. There is no grid_boltrend row in `strategy_certifications`.

## Frozen experiment and next gate

Unchanged: 28 assets, IS-only Top 8; 2022-01-01 to 2026-07-27 exclusive;
IS 180d / embargo 7d / OOS 60d / step 60d; 1,296 combinations; 1,646 USDT;
primary 5x, sensitivities 3x/5x/8x; closed Binance 1h signals, Bitget 1m
execution and Bitget funding/calibration. Existing order/risk abstractions remain.

1. Establish whether a complete, provenance-qualified Bitget funding archive is
   available and investigate FET/OP/SUI prefixes with source evidence. Do not
   restart the same full downloads blindly or substitute Binance funding.
2. If evidence is unavailable, document that blocker. Do not silently shorten
   the calendar, remove assets or infer zero funding. A different experiment
   requires an explicit new pre-registration, not a post-hoc certification claim.
3. After data repair, fully validate and create a new immutable snapshot from a
   clean checkout, preserving unrelated user files. Recheck calibration evidence.
4. Only then provide the user the frozen WFO, external OOS and certification
   commands. The user runs long jobs locally; no deployment is authorized.
