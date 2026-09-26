# Common availability research amendment — 2026-09-26

Approved by the user before any OOS run. Applies only to the synthetic-funding
grid_boltrend study, with explicit `--research-common-availability` opt-in.
Keep all 28 assets, 2022-01-01 / 2026-07-27, 180/7/60/60-day calendar,
Top 8 IS-only, 1296 combinations, capital 1646, 5x primary and 3/5/8x
sensitivities, seed 0, and the entire boltrend_funding_v1 profile unchanged.

For each asset, query the first actual Binance 1h and Bitget 1m timestamps
within the frozen interval. Take their maximum and round up to the next full
UTC hour (an exact hour stays unchanged). Freeze both observed starts and the
derived boundary in the signed snapshot. These are data-availability dates,
not independently confirmed exchange listing dates. Missing series still fail.

Hash and consume both price series only from that common boundary. No database
rows are removed or invented. Existing validators reject internal gaps, invalid
OHLC and incomplete tails. The existing WFO coverage filter admits only complete
global IS/embargo/OOS windows; do not shift the calendar or shorten training.
Expected first OOS dates, conditional on continuous data: OP 2023-01-03,
FET 2023-10-30, SUI 2023-12-29. The 500h OOS warmup fits after 180d IS + 7d embargo.

Implement solely in common snapshot construction/revalidation and CLI, with
policy provenance in WFO reuse identity. Do not duplicate engines or modify
strict certification. Add isolated tests for clipping, rounding, missing/gapped
series, policy isolation, revalidation and unchanged common-calendar admission.
Run the complete guarded pytest suite. Archive audit and update workflow/commands.
No long run, robot2 action or new certification verdict is authorized.
