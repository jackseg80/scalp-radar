# Bitget historical-data recoverability — 2026-09-26

## Decision

Do not restart the full download or launch grid_boltrend WFO. The tested public
funding endpoints do not recover the missing 2022–early-May-2026 archive.
No complete qualifying replacement source was identified in this bounded review.
This does not prove that an archive cannot exist with Bitget or a data supplier.
Historical certification remains blocked; research-only handling continues, with
no formal verdict written and no inference of poor strategy performance.

Scope: public documentation, 15 small unauthenticated HTTP GET requests and local
code inspection. No database writes, credentials, bulk data download, robot2
connection, deployment, optimization or certification run. No provider contacted,
account created or purchase made. The frozen experiment remains unchanged.

## Funding: official documentation and live probes

The current [Bitget v3 documentation](https://www.bitget.com/docs/catalog/market/derivatives)
explicitly limits funding history to the last 90 days. The documented cursor is
a page number, not a date. The [v2 documentation](https://www.bitget.com/api-doc/classic/contract/market/Get-History-Funding-Rate)
offers pageNo/pageSize; it does not document a historical date override.

Public GET routes tested on BTCUSDT, 100 rows per page:

- `/api/v2/mix/market/history-fund-rate`, productType=usdt-futures,
  symbol=BTCUSDT, pageSize=100, pageNo=1/2/3/4/20.
- `/api/v3/market/history-fund-rate`, category=USDT-FUTURES,
  symbol=BTCUSDT, limit=100, cursor=1/2/3/4.

All responses returned code 00000. Both versions returned the following bounds:

| Page | Rows | Earliest event UTC | Latest event UTC |
| --- | ---: | --- | --- |
| 1 | 100 | 2026-08-24 16:00 | 2026-09-26 16:00 |
| 2 | 100 | 2026-07-22 08:00 | 2026-08-24 08:00 |
| 3 | 70 | 2026-06-29 00:00 | 2026-07-22 00:00 |
| 4 | 0 | — | — |

v2 page 20 was also empty. These are live response summaries, not a stored or
hash-qualified funding archive; equality of the full v2/v3 rate values was not
tested. Funding requestTime values ranged from 1790453995678 to 1790454056004.
The unchanged local archive still starts on May 4, 2026; today's rolling window
cannot backfill its earlier missing years. Increasing concurrency or pagination
does not remove this observed retention boundary. Other symbols were not probed.

## Three price-series starts: source boundary versus download defect

Six public GET probes used `/api/v3/market/history-candles`,
category=USDT-FUTURES, interval=1m, type=market, limit=100, and explicit
startTime/endTime in milliseconds. For each local first candle below, query
the ten minutes immediately before it, then the ten minutes starting at it.

| Symbol | Boundary UTC | Preceding window | Window starting at boundary |
| --- | --- | ---: | ---: |
| FETUSDT | 2023-03-02 08:30 | 0 rows | 10 rows, 08:30–08:39 |
| OPUSDT | 2022-06-13 03:27 | 0 rows | 10 rows, 03:27–03:36 |
| SUIUSDT | 2023-05-04 09:52 | 0 rows | 10 rows, 09:52–10:01 |

All returned code 00000; requestTime ranged from 1790454056321 to 1790454057847.
The local boundaries agree with these sampled API windows. This does not prove
the entire earlier prefix is absent, nor prove an exact tradable listing time.

Official announcements provide partial context, not a minute-level trading gate:

- [FET](https://www.bitget.com/asia/support/articles/12560603777583): announcement
  names March 3, 2023 (UTC+8), later than the sampled first candle.
- [OP](https://www.bitget.com/support/articles/7521598853273-OPUSDT-%26-RSRUSDT-are-Now-Available-on-Futures):
  body names June 17, 2022 (UTC+8), while the page header is June 15 and the
  sampled candle starts June 13. Do not resolve this inconsistency by assumption.
- [SUI](https://www.bitget.com/support/articles/12560603784505): names May 4, 2023
  (UTC+8), without an independently established first tradable UTC minute.

The evidence suggests exchange availability contributes to the Binance/Bitget
start mismatch, but it is insufficient to certify exact listing boundaries.
`backend/core/experiment.py` currently rejects execution coverage starting later
than signal coverage. No exception or listing-aware start rule was introduced.

## Alternative sources screened (not exhaustive)

- [Bitget public download catalogue](https://www.bitget.com/data-download) lists
  candle, trade and order-book files; the inspected page does not advertise a
  funding archive. This does not rule out a separate support export.
- [Tardis Bitget Futures coverage](https://docs.tardis.dev/historical-data-details/bitget-futures)
  starts November 8, 2024, so it cannot fill the complete frozen range.
  Its derivative ticker is not automatically certified settled funding; the
  [schema](https://docs.tardis.dev/downloadable-csv-files/data-types) describes a
  rate for the next settlement which can change until that event.
- [Coinalyze](https://api.coinalyze.net/v1/doc/) retains 1,500–2,000 intraday
  points. Daily aggregates are not substitutes for exact settlement events.
  No authenticated dataset was accessed or qualified.

## Next authorized decision

Preferred next step: the user requests an official archive from Bitget (or
provides an existing immutable export), including settlement timestamps/rates
and authoritative contract-availability evidence. Requirements:

1. All 28 frozen USDT perpetual symbols, 2022-01-01 inclusive to 2026-07-27
   exclusive, with contract inception boundaries explicitly documented.
2. Actual settled funding events, UTC timestamps and rate units; no daily
   averages, predicted rates, zero filling or Binance substitution.
3. Provenance, generation date, gaps, settlement-interval changes and sample data
   to validate before any bulk import. Preserve raw files and hash them.
4. Clarify FET/OP/SUI first tradable UTC timestamps and the announcement/API
   inconsistencies above. If pre-listing execution is impossible, separately
   approve and pre-register availability rules; do not relax a failed snapshot.

Sending a support request, buying data, changing the calendar or starting a new
forward experiment requires explicit user direction. None was done here.
If no qualified archive can be obtained, retain the historical block and propose
a separately registered forward study; do not relabel this run as PAPER_READY.

## Validation and handoff

Documentation-only follow-up; no executable behavior changed. The previously
verified 2428 passing tests at c7806d8 were not rerun unnecessarily. No new
snapshot or certification row was created. Continue from this audit and
`audit-grid-boltrend-readiness-20260926.md`; do not repeat bulk collection merely
because the old command accepts the desired date range.
