# Sprint 23b — Grid Trend compute_live_indicators

## Problème

Le portfolio backtest et le paper trading de `grid_trend` produisent 0 trades car
`compute_live_indicators()` n'est pas implémenté. Le `IncrementalIndicatorEngine` ne calcule
pas EMA ni ADX — uniquement SMA + ATR.

## Solution

Override `compute_live_indicators()` dans `GridTrendStrategy` (pattern identique à
`GridMultiTFStrategy` pour le Supertrend 4h).

## Fichiers modifiés

- `backend/strategies/grid_trend.py` — ajout `compute_live_indicators()` (~30 lignes)
- `tests/test_grid_trend.py` — 3 tests (Section 9)
- `docs/ROADMAP.md` — mise à jour état actuel

## Implémentation

```python
def compute_live_indicators(self, candles: list[Candle]) -> dict[str, dict[str, Any]]:
    min_needed = max(self._config.ema_slow, self._config.adx_period * 2 + 1) + 20
    if len(candles) < min_needed:
        return {}
    closes = np.array([c.close for c in candles], dtype=float)
    highs = np.array([c.high for c in candles], dtype=float)
    lows = np.array([c.low for c in candles], dtype=float)
    ema_fast_arr = ema(closes, self._config.ema_fast)
    ema_slow_arr = ema(closes, self._config.ema_slow)
    adx_arr, _, _ = adx(highs, lows, closes, self._config.adx_period)
    return {
        self._config.timeframe: {
            "ema_fast": float(ema_fast_arr[-1]),
            "ema_slow": float(ema_slow_arr[-1]),
            "adx": float(adx_arr[-1]),
        }
    }
```

## Tests ajoutés

1. `test_returns_ema_adx_with_enough_candles` — vérifie EMA + ADX retournés comme float non-NaN
2. `test_returns_empty_with_too_few_candles` — vérifie le guard min candles → `{}`
3. `test_runner_merges_live_indicators` — pipeline IncrementalIndicatorEngine → buffer → merge

## Résultat

1007 tests (+3 nouveaux), 0 régression.
