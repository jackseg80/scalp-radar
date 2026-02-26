# Regime Impact Report — Sprint 50b

Date : 2026-02-26 12:02 UTC

## Résumé exécutif

- **Verdict** : **BORDERLINE**
  - return_ok : OK
  - dd_ok : NOK
  - sharpe_ok : NOK
- Transitions détectées : 3
- Leverage changes (Run C) : 39

## Tableau comparatif

| Run                  | Return   | Max DD  | Sharpe | Calmar | Trades | WinR   |
|----------------------|----------|---------|--------|--------|--------|--------|
| A: Fixed 7x          |    262.3% |   -6.57% |   4.94 |  39.92 |   8213 |  79.5% |
| B: Fixed 4x          |    149.9% |   -4.81% |   4.88 |  31.16 |   8213 |  79.5% |
| C: Dynamic 7x/4x     |    256.3% |   -6.57% |   4.89 |  39.02 |   8213 |  79.5% |

## Transitions de régime

| # | Timestamp | From | To |
|---|-----------|------|----|
| 1 | 2023-09-12T20:00:00+00:00 | normal | defensive |
| 2 | 2023-10-19T20:00:00+00:00 | defensive | normal |
| 3 | 2025-11-17T20:00:00+00:00 | normal | defensive |

## Changements de leverage (Run C)

| Timestamp | Runner | Old | New | Regime |
|-----------|--------|-----|-----|--------|
| 2023-09-12T20:00:00+00:00 | grid_atr:AAVE/USDT | 7x | 4x | defensive |
| 2023-09-12T20:00:00+00:00 | grid_atr:ADA/USDT | 7x | 4x | defensive |
| 2023-09-12T20:00:00+00:00 | grid_atr:AVAX/USDT | 7x | 4x | defensive |
| 2023-09-12T20:00:00+00:00 | grid_atr:BCH/USDT | 7x | 4x | defensive |
| 2023-09-12T20:00:00+00:00 | grid_atr:BNB/USDT | 7x | 4x | defensive |
| 2023-09-12T20:00:00+00:00 | grid_atr:DOGE/USDT | 7x | 4x | defensive |
| 2023-09-12T20:00:00+00:00 | grid_atr:DYDX/USDT | 7x | 4x | defensive |
| 2023-09-12T20:00:00+00:00 | grid_atr:LINK/USDT | 7x | 4x | defensive |
| 2023-09-12T20:00:00+00:00 | grid_atr:NEAR/USDT | 7x | 4x | defensive |
| 2023-09-12T20:00:00+00:00 | grid_atr:SOL/USDT | 7x | 4x | defensive |
| 2023-09-12T20:00:00+00:00 | grid_atr:UNI/USDT | 7x | 4x | defensive |
| 2023-09-12T20:00:00+00:00 | grid_atr:XRP/USDT | 7x | 4x | defensive |
| 2023-09-13T00:00:00+00:00 | grid_atr:FET/USDT | 7x | 4x | defensive |
| 2023-10-19T20:00:00+00:00 | grid_atr:AAVE/USDT | 4x | 7x | normal |
| 2023-10-19T20:00:00+00:00 | grid_atr:ADA/USDT | 4x | 7x | normal |
| 2023-10-19T20:00:00+00:00 | grid_atr:AVAX/USDT | 4x | 7x | normal |
| 2023-10-19T20:00:00+00:00 | grid_atr:BCH/USDT | 4x | 7x | normal |
| 2023-10-19T20:00:00+00:00 | grid_atr:BNB/USDT | 4x | 7x | normal |
| 2023-10-19T20:00:00+00:00 | grid_atr:DOGE/USDT | 4x | 7x | normal |
| 2023-10-19T20:00:00+00:00 | grid_atr:DYDX/USDT | 4x | 7x | normal |
| 2023-10-19T20:00:00+00:00 | grid_atr:FET/USDT | 4x | 7x | normal |
| 2023-10-19T20:00:00+00:00 | grid_atr:LINK/USDT | 4x | 7x | normal |
| 2023-10-19T20:00:00+00:00 | grid_atr:NEAR/USDT | 4x | 7x | normal |
| 2023-10-19T20:00:00+00:00 | grid_atr:SOL/USDT | 4x | 7x | normal |
| 2023-10-19T20:00:00+00:00 | grid_atr:UNI/USDT | 4x | 7x | normal |
| 2023-10-19T20:00:00+00:00 | grid_atr:XRP/USDT | 4x | 7x | normal |
| 2025-11-17T20:00:00+00:00 | grid_atr:AAVE/USDT | 7x | 4x | defensive |
| 2025-11-17T20:00:00+00:00 | grid_atr:ADA/USDT | 7x | 4x | defensive |
| 2025-11-17T20:00:00+00:00 | grid_atr:AVAX/USDT | 7x | 4x | defensive |
| 2025-11-17T20:00:00+00:00 | grid_atr:BCH/USDT | 7x | 4x | defensive |
| 2025-11-17T20:00:00+00:00 | grid_atr:BNB/USDT | 7x | 4x | defensive |
| 2025-11-17T20:00:00+00:00 | grid_atr:DOGE/USDT | 7x | 4x | defensive |
| 2025-11-17T20:00:00+00:00 | grid_atr:DYDX/USDT | 7x | 4x | defensive |
| 2025-11-17T20:00:00+00:00 | grid_atr:FET/USDT | 7x | 4x | defensive |
| 2025-11-17T20:00:00+00:00 | grid_atr:NEAR/USDT | 7x | 4x | defensive |
| 2025-11-17T20:00:00+00:00 | grid_atr:XRP/USDT | 7x | 4x | defensive |
| 2025-11-17T21:00:00+00:00 | grid_atr:UNI/USDT | 7x | 4x | defensive |
| 2025-11-17T22:00:00+00:00 | grid_atr:LINK/USDT | 7x | 4x | defensive |
| 2025-11-17T22:00:00+00:00 | grid_atr:SOL/USDT | 7x | 4x | defensive |

## Breakdown par régime (Run C)

| Régime | Heures | Return | Max DD |
|--------|--------|--------|--------|
| normal | 23838 | +250.70% | -6.57% |
| defensive | 3272 | +118.30% | -0.72% |
