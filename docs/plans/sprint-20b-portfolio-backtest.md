# Sprint 20b — Portfolio Backtest Multi-Asset

## Contexte

Le WFO backtest chaque asset isolément avec 10 000$ fictifs. En production, les 21 assets partagent le même pool. Le paper trading a montré -46% de drawdown non réalisé quand tout dip simultanément. On a besoin d'un backtest portfolio qui simule le capital partagé avec le **même code que la prod** (`GridStrategyRunner`).

**5 questions auxquelles le backtest doit répondre :**
1. Max drawdown historique sur le portfolio (capital partagé)
2. Corrélation des fills — combien de grilles se remplissent simultanément en crash
3. Margin peak — marge max mobilisée simultanément (% du capital)
4. Kill switch frequency — combien de fois le kill switch aurait déclenché
5. Sizing optimal — pour 1k/5k/10k$, combien d'assets max

---

## Architecture

### Approche : N runners (1 par asset) avec capital partagé

- Chaque `GridStrategyRunner` gère **1 seul asset**
- Chaque runner a sa propre `GridATRStrategy` avec les params WFO per_asset
- Capital split : `initial_capital / N` par runner
- `_nb_assets = 1` par runner → sizing = `(capital/N) / 1 / levels`
- Un `IncrementalIndicatorEngine` partagé (alimente tous les symbols)
- Pas de modification du code existant — 3 fichiers nouveaux uniquement

### Flux de simulation

```
1. Charger les params WFO per_asset depuis strategies.yaml
2. Créer N GridStrategyRunners (1/asset) via create_strategy_with_params()
3. Charger candles 1h depuis DB pour tous les assets
4. Warm-up : 50 premières candles → indicator engine + close_buffer
5. Simulation chronologique :
   Pour chaque candle (triée par timestamp) :
     a. indicator_engine.update(symbol, "1h", candle)
     b. runner[symbol].on_candle(symbol, "1h", candle)
     c. À chaque changement de timestamp : snapshot portfolio
6. Force-close positions restantes à la fin
7. Calculer métriques finales + rapport
```

---

## Fichiers à créer

| Fichier | Description | ~Lignes |
|---------|-------------|---------|
| `backend/backtesting/portfolio_engine.py` | Classe `PortfolioBacktester` | ~350 |
| `scripts/portfolio_backtest.py` | Script CLI | ~120 |
| `tests/test_portfolio_backtest.py` | ~10 tests | ~250 |

**Aucun fichier existant modifié.**

---

## Implémentation détaillée

### 1. `backend/backtesting/portfolio_engine.py`

#### Dataclasses

**`PortfolioSnapshot`** — enregistré à chaque timestamp unique :
- `timestamp`, `total_equity` (capital + unrealized), `total_capital` (cash)
- `total_realized_pnl`, `total_unrealized_pnl`, `total_margin_used`
- `margin_ratio` (margin/initial_capital), `n_open_positions`, `n_assets_with_positions`

**`PortfolioResult`** — résultat final :
- Config : `initial_capital`, `n_assets`, `period_days`, `assets`
- Aggregate : `final_equity`, `total_return_pct`, `total_trades`, `win_rate`
- Risk : `max_drawdown_pct`, `max_drawdown_duration_hours`, `peak_margin_ratio`
- Kill switch : `kill_switch_triggers`, `kill_switch_events[]`
- Détail : `per_asset_results`, `all_trades`, `equity_curve`, `snapshots`

#### Classe `PortfolioBacktester`

**Constructor :**
```python
def __init__(
    self,
    config: AppConfig,
    initial_capital: float = 10_000.0,
    strategy_name: str = "grid_atr",
    assets: list[str] | None = None,  # None = tous depuis per_asset
    exchange: str = "binance",
) -> None:
```

**Méthodes clés :**

1. **`async def run(start, end) -> PortfolioResult`** — Point d'entrée principal
   - Résout les assets (depuis `strategies.yaml` per_asset si non spécifiés)
   - Crée les strategies via `create_strategy_with_params("grid_atr", per_asset_params)` ([backend/optimization/__init__.py:65](backend/optimization/__init__.py#L65))
   - Crée l'`IncrementalIndicatorEngine` partagé ([backend/core/incremental_indicators.py:37](backend/core/incremental_indicators.py#L37))
   - Crée N `GridStrategyRunner` (pattern copié de [backend/backtesting/simulator.py:1317-1333](backend/backtesting/simulator.py#L1317-L1333))
   - Override runner : `_nb_assets=1`, `_capital=capital/N`, `_initial_capital=capital/N`
   - Charge candles DB, warm-up, simulation, compute result

2. **`_create_runner(symbol, params, config, indicator_engine) -> GridStrategyRunner`**
   - Crée `GridATRStrategy` via `create_strategy_with_params()`
   - Crée `PositionManagerConfig` avec leverage/fees du config
   - Crée `GridPositionManager(gpm_config)`
   - Crée `GridStrategyRunner(strategy, config, indicator_engine, gpm, data_engine=None, db_path=None)`
   - Override `_nb_assets=1`, `_capital`, `_initial_capital`

3. **`_warmup_runners(runners, candles_by_symbol, indicator_engine, count=50)`**
   - Pour chaque symbol, prend les `count` premières candles
   - Feed `indicator_engine.update(symbol, "1h", candle)`
   - Init `runner._close_buffer[symbol] = deque(maxlen=...)` et append closes
   - Après warm-up : `runner._is_warming_up = False`, reset `_capital`/`_realized_pnl`/`_stats`
   - Retourne l'index de début de simulation par symbol

4. **`_merge_candles(candles_by_symbol) -> list[Candle]`**
   - Merge toutes les candles, tri par `(timestamp, symbol)`
   - 21 assets × 90j = ~45k candles → rapide en mémoire

5. **`async _simulate(runners, engine, candles, start_indices) -> list[PortfolioSnapshot]`**
   - Boucle principale :
     ```
     Pour chaque candle (après warmup) :
       indicator_engine.update(symbol, "1h", candle)
       await runner[symbol].on_candle(symbol, "1h", candle)
       Si timestamp change → _take_snapshot()
     ```
   - Force-close positions restantes à la fin via `gpm.close_all_positions()`

6. **`_take_snapshot(runners, timestamp, last_closes) -> PortfolioSnapshot`**
   - Agrège : capital, realized_pnl, unrealized_pnl, margin_used, positions
   - Unrealized : via `gpm.unrealized_pnl(positions, last_close)` ([backend/core/grid_position_manager.py:237](backend/core/grid_position_manager.py#L237))

7. **`_compute_metrics(snapshots) -> dict`**
   - Max drawdown (peak-to-trough sur equity curve)
   - Kill switch check (fenêtre glissante 24h, seuil 30%)
   - Peak margin ratio, peak concurrent positions/assets

8. **`format_portfolio_report(result) -> str`** — Fonction standalone pour l'affichage CLI

---

### 2. `scripts/portfolio_backtest.py`

**Arguments CLI :**
- `--days` (défaut: 90) — Période de backtest
- `--capital` (défaut: 10000) — Capital initial
- `--assets` — Liste d'assets séparés par virgule (défaut: tous per_asset)
- `--exchange` (défaut: "binance") — Source des candles
- `--json` — Sortie JSON au lieu de tableau
- `--output` — Écrire dans un fichier
- `--kill-switch-pct` (défaut: 30) — Seuil kill switch %
- `--kill-switch-window` (défaut: 24) — Fenêtre kill switch (heures)

**Flow :**
```python
async def main(args):
    config = get_config()
    backtester = PortfolioBacktester(config, args.capital, ...)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    result = await backtester.run(start, end)
    print(format_portfolio_report(result))
```

---

### 3. `tests/test_portfolio_backtest.py`

~10 tests avec données synthétiques (candles `_make_candles()` + `np.random.seed(42)`) :

| Test | Vérifie |
|------|---------|
| `test_portfolio_snapshot_dataclass` | Tous les champs existent |
| `test_merge_candles_chronological` | Tri correct (timestamp, symbol) |
| `test_warmup_sets_capital_correctly` | Après warmup : `_is_warming_up=False`, capital=`total/N` |
| `test_flat_prices_no_trades` | Prix plat → equity = capital initial |
| `test_two_assets_capital_split` | 2 assets, 10k → chaque runner = 5k |
| `test_margin_tracking` | Positions ouvertes → margin_used > 0 |
| `test_drawdown_computation` | Equity synthétique → drawdown correct |
| `test_kill_switch_detection` | Drawdown 35% → événement enregistré |
| `test_per_asset_breakdown` | Trades et P&L par asset corrects |
| `test_force_close_at_end` | Positions ouvertes fermées à fin des données |

---

## Points techniques critiques

### A. Warm-up ne se termine jamais naturellement
En backtest, toutes les candles ont `age > 2h`. La détection auto ([simulator.py:684-687](backend/backtesting/simulator.py#L684-L687)) ne fire jamais.
**Fix :** Après avoir alimenté 50 candles warmup, setter manuellement `runner._is_warming_up = False`.

### B. `_nb_assets` doit être 1
Sinon le runner lit `len(strategy._config.per_asset)` = 21 ([simulator.py:550-552](backend/backtesting/simulator.py#L550-L552)), et le sizing devient `capital / 21 / 21 / levels` au lieu de `capital / 21 / 1 / levels`.
**Fix :** Override `runner._nb_assets = 1` après création.

### C. `_initial_capital` doit correspondre au capital par runner
Le runner utilise `_initial_capital` pour la marge cap 25% ([simulator.py:811](backend/backtesting/simulator.py#L811)).
**Fix :** Override `runner._initial_capital = capital / N`.

### D. `data_engine=None` est safe
`GridStrategyRunner` stocke `data_engine` ([simulator.py:518](backend/backtesting/simulator.py#L518)) mais ne l'utilise jamais dans `on_candle()`. Seul `_warmup_from_db()` en a besoin, et on bypass cette méthode.

### E. Force-close en fin de données
`GridStrategyRunner.on_candle()` ne ferme pas les positions automatiquement à la fin. Le `PortfolioBacktester` doit appeler `gpm.close_all_positions()` pour chaque runner avec des positions ouvertes, et ajuster `_capital` en conséquence.

### F. Limitation connue : ATR period
L'`IncrementalIndicatorEngine` calcule l'ATR avec `period=14` fixe ([incremental_indicators.py:116](backend/core/incremental_indicators.py#L116)). Le `atr_period` per_asset du WFO n'est pas utilisé dans le path live. Ceci est le comportement actuel de la prod — pas une régression. Le `ma_period` est correctement per_asset (le runner calcule sa propre SMA, [simulator.py:701](backend/backtesting/simulator.py#L701)).

### G. `_close_buffer` doit être pré-initialisé
Initialiser `runner._close_buffer[symbol] = deque(maxlen=max(ma_period+20, 50))` avant de feeder les candles warmup. Le runner le fait paresseusement ([simulator.py:690-693](backend/backtesting/simulator.py#L690-L693)) mais on a besoin du buffer prêt avant.

### H. `limit=1_000_000` pour charger les candles
Le défaut de `db.get_candles()` est `limit=500` — insuffisant pour un backtest de 90+ jours (21 assets × 90j × 24h = ~45k candles).

---

## Réponses aux 5 questions

| Question | Champ `PortfolioResult` | Calcul |
|----------|------------------------|--------|
| Max drawdown portfolio | `max_drawdown_pct` | Peak-to-trough sur `total_equity` (capital + unrealized) |
| Corrélation fills | `peak_concurrent_assets` + `snapshots[].n_assets_with_positions` | Compte d'assets avec ≥1 position ouverte à chaque snapshot |
| Margin peak | `peak_margin_ratio` | Max de `margin_used / initial_capital` sur tous les snapshots |
| Kill switch frequency | `kill_switch_triggers` + `kill_switch_events[]` | Fenêtre glissante 24h, seuil 30% (paramétrable) |
| Sizing optimal | Exécuter avec `--capital 1000/5000/10000` et comparer drawdown/return | Per-asset results montrent les assets qui contribuent le plus au risque |

---

## Vérification

```powershell
# Tests existants (doit rester 774+)
uv run python -m pytest --tb=short -q

# Tests portfolio
uv run python -m pytest tests/test_portfolio_backtest.py -v

# Sanity check 1 asset
uv run python -m scripts.portfolio_backtest --days 90 --assets BTC/USDT --capital 10000

# Full portfolio 21 assets
uv run python -m scripts.portfolio_backtest --days 90 --capital 10000

# Comparaisons capital
uv run python -m scripts.portfolio_backtest --days 90 --capital 1000 --assets CRV/USDT,ENJ/USDT,FET/USDT,DOGE/USDT,AVAX/USDT
uv run python -m scripts.portfolio_backtest --days 90 --capital 5000
uv run python -m scripts.portfolio_backtest --days 180 --capital 10000
```

---

## Fichiers critiques (référence)

- [backend/backtesting/simulator.py](backend/backtesting/simulator.py) — `GridStrategyRunner` (L505+), `Simulator.start()` (L1286+)
- [backend/core/incremental_indicators.py](backend/core/incremental_indicators.py) — `IncrementalIndicatorEngine` (L30+)
- [backend/optimization/__init__.py](backend/optimization/__init__.py) — `create_strategy_with_params()` (L65)
- [backend/core/grid_position_manager.py](backend/core/grid_position_manager.py) — `GridPositionManager` (L17+)
- [backend/strategies/grid_atr.py](backend/strategies/grid_atr.py) — `GridATRStrategy` (L21+)
- [backend/core/database.py](backend/core/database.py) — `get_candles()` (L421+)
- [config/strategies.yaml](config/strategies.yaml) — section `grid_atr.per_asset` (L242+)
