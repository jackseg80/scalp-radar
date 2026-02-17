# Sprint 24b — Portfolio Backtest Multi-Stratégie

**Date** : 17 février 2026
**Résultat** : 1016 tests (+4 nouveaux), 0 régression

## Objectif

Supporter plusieurs stratégies simultanées dans le portfolio backtest pour mesurer la complémentarité grid_atr + grid_trend sur un pool de capital unique.

## Problème

`PortfolioBacktester` ne prenait qu'un seul `strategy_name`. Impossible de tester grid_atr (10 assets) + grid_trend (6 assets) ensemble avec capital partagé.

## Solution

### 1. `portfolio_engine.py` — paramètre `multi_strategies`

- Nouveau paramètre `multi_strategies: list[tuple[str, list[str]]] | None`
- Si fourni (ex: `[("grid_atr", ["BTC/USDT", ...]), ("grid_trend", ["SOL/USDT", ...])]`), prend le dessus sur `strategy_name` + `assets`
- Si absent, auto-construit `[(strategy_name, assets)]` → rétro-compatible

**Clés des runners** : format `strategy_name:symbol` (ex: `grid_atr:ICP/USDT`) pour supporter le même symbol dans 2 stratégies.

**`_symbol_from_key()`** : méthode statique, extrait le symbol depuis la clé. Gère les deux formats (avec et sans `:`).

**`_create_runners()`** : itère sur `multi_strategies`, crée un runner par (stratégie, symbol) avec params WFO per_asset.

**`_warmup_runners()`** : injecte les candles dans l'indicator engine une seule fois par symbol (même si 2 runners l'utilisent).

**`_simulate()`** : mapping `symbol → [runner_keys]` pour dispatcher une candle à TOUS les runners de ce symbol. Re-key les trades avec `runner_key`.

**`_build_result()`** : breakdown `per_asset_results` indexé par `runner_key` (pas symbol).

**`format_portfolio_report()`** : section "Par Runner" avec largeur dynamique des clés.

### 2. `scripts/portfolio_backtest.py` — CLI enrichi

- `--strategies strat1:sym1,sym2+strat2:sym3,sym4` — format explicite multi-stratégie
- `--preset combined` — lit automatiquement les per_asset de grid_atr + grid_trend depuis strategies.yaml

### 3. Rétro-compatibilité

Tous les tests existants (24 tests) continuent de passer sans modification (sauf renommage kwarg `assets` → `runner_keys` dans `_build_result` et "Par Asset" → "Par Runner" dans le rapport).

## Tests ajoutés

1. `test_multi_strategy_creates_runners` — 2 runners créés avec clés `grid_atr:AAA/USDT` et `grid_trend:AAA/USDT`
2. `test_same_symbol_dispatched_to_both` — même symbol dispatché aux 2 runners, close_buffer alimenté des deux côtés
3. `test_capital_split` — 2 stratégies × 2 assets = 4 runners, chacun reçoit 2500$ sur 10k
4. `test_backward_compatible` — `multi_strategies=None` + `strategy_name="grid_atr"` → fonctionne comme avant

## Usage

```bash
# Single strategy (inchangé)
uv run python -m scripts.portfolio_backtest --days 730 --capital 10000

# Multi-stratégie explicite
uv run python -m scripts.portfolio_backtest --days 730 --capital 10000 \
  --strategies "grid_atr:BTC/USDT,ICP/USDT+grid_trend:SOL/USDT,ICP/USDT"

# Preset combined (grid_atr + grid_trend per_asset depuis strategies.yaml)
uv run python -m scripts.portfolio_backtest --days 730 --capital 10000 \
  --preset combined --save --label "combined_atr10_trend6"
```
