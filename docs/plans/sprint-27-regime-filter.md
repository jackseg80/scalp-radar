# Sprint 27 — Filtre Darwinien par Regime

## Contexte

Le WFO produit un `regime_analysis` par couple (strategie x asset) stocke dans `optimization_results.regime_analysis`. On veut utiliser cette donnee pour empecher l'ouverture de nouvelles grilles dans un regime defavorable (Sharpe < 0). Le filtre est binaire (ouvrir ou pas), ne touche jamais les positions existantes, et est entierement retrocompatible.

## Fichiers modifies

| Fichier | Action |
|---------|--------|
| `backend/core/config.py` | Ajouter `regime_filter_enabled` a `RiskConfig` |
| `config/risk.yaml` | Ajouter `regime_filter_enabled: true` |
| `backend/backtesting/simulator.py` | Constante mapping, `__init__` param, methode filtre, insertion `on_candle`, chargement dans `start()` |
| `backend/core/database.py` | Nouvelle methode `get_regime_profiles()` |
| `backend/backtesting/portfolio_engine.py` | Charger profiles dans `run()`, passer a `_create_runners()` |
| `tests/test_regime_filter.py` | 12 tests (nouveau fichier) |

## Plan d'implementation

### Etape 1 — Config (`config.py` + `risk.yaml`)

**`backend/core/config.py`** : Ajouter dans `RiskConfig` (ligne ~433, avant `@model_validator`) :
```python
regime_filter_enabled: bool = Field(default=True)
```

**`config/risk.yaml`** : Ajouter apres la section `adaptive_selector` (ligne 43) :
```yaml
regime_filter_enabled: true
```

### Etape 2 — Constante mapping (`simulator.py`)

Apres les imports (ligne ~45), ajouter :
```python
REGIME_LIVE_TO_WFO: dict[MarketRegime, str] = {
    MarketRegime.TRENDING_UP: "bull",
    MarketRegime.TRENDING_DOWN: "bear",
    MarketRegime.RANGING: "range",
    MarketRegime.HIGH_VOLATILITY: "crash",
    # LOW_VOLATILITY intentionnellement absent → autorise
}
```

### Etape 3 — `GridStrategyRunner.__init__()` (simulator.py:519)

Ajouter parametre `regime_profile: dict[str, dict] | None = None` a la signature.
Stocker apres `self._db_path = db_path` (ligne 533) :
```python
self._regime_profile = regime_profile
self._regime_filter_blocks: int = 0  # Compteur blocages filtre Darwinien
```
Format: `{"BTC/USDT": {"bull": {"avg_oos_sharpe": 6.88, ...}, ...}, ...}`

### Etape 4 — Methode `_should_allow_new_grid()` (simulator.py)

Apres `_get_sl_percent()` (~ligne 593). Utilise `self._current_regime` :
- **Verifie** : initialise ligne 540 (`MarketRegime.RANGING`), mis a jour lignes 790-796 via `detect_market_regime()`.
- Le filtre est insere ligne ~877, soit APRES la mise a jour du regime. Pas de risque de valeur stale.

```python
def _should_allow_new_grid(self, symbol: str) -> bool:
    """Sprint 27 : Filtre Darwinien — bloque si regime WFO defavorable."""
    if not getattr(self._config.risk, "regime_filter_enabled", True):
        return True
    if not self._regime_profile:
        return True
    symbol_profile = self._regime_profile.get(symbol)
    if not symbol_profile:
        return True
    wfo_key = REGIME_LIVE_TO_WFO.get(self._current_regime)
    if wfo_key is None:
        return True
    regime_data = symbol_profile.get(wfo_key)
    if regime_data is None:
        return True  # Non couvert → benefice du doute
    avg_sharpe = regime_data.get("avg_oos_sharpe", 0.0)
    if not isinstance(avg_sharpe, (int, float)):
        return True
    if avg_sharpe < 0:
        self._regime_filter_blocks += 1
        logger.info("[{}] REGIME FILTER : {} bloque (regime={}, sharpe={:.2f})",
                     self.name, symbol, wfo_key, avg_sharpe)
        return False
    return True
```

### Etape 4b — Exposer le compteur dans `get_status()` (simulator.py:1181-1207)

Ajouter dans le dict retourne par `GridStrategyRunner.get_status()` (apres `assets_with_positions`) :
```python
"regime_filter_blocks": self._regime_filter_blocks,
```

### Etape 5 — Insertion dans `on_candle()` (simulator.py:876-878)

Entre la fin du bloc TP/SL (return ligne 876) et le bloc "ouvrir nouveaux niveaux" (ligne 878) :
```python
        # Sprint 27 : Filtre Darwinien
        if not positions and not self._should_allow_new_grid(symbol):
            return
```
- `not positions` = aucune position existante (nouveau cycle)
- Si positions existent, le filtre est bypasse (DCA continue)

### Etape 6 — `Database.get_regime_profiles()` (database.py)

Nouvelle methode async apres les methodes de requete existantes :
```python
async def get_regime_profiles(self, strategy_name: str) -> dict[str, dict]:
    """Sprint 27 : {symbol: regime_analysis_dict} pour is_latest=1."""
    assert self._conn is not None
    cursor = await self._conn.execute(
        "SELECT asset, regime_analysis FROM optimization_results "
        "WHERE strategy_name = ? AND is_latest = 1 AND regime_analysis IS NOT NULL",
        (strategy_name,),
    )
    rows = await cursor.fetchall()
    profiles = {}
    for row in rows:
        try:
            ra = json.loads(row["regime_analysis"])
            if isinstance(ra, dict):
                profiles[row["asset"]] = ra
        except (json.JSONDecodeError, TypeError):
            continue
    return profiles
```
`json` est deja importe (ligne 10 de database.py).

### Etape 7 — Chargement dans `Simulator.start()` (simulator.py:1640)

Apres le warm-up (ligne 1640), avant la restauration du kill switch global (ligne 1642) :
```python
        # Sprint 27 : Charger les profils regime WFO
        if self._db is not None:
            for runner in self._runners:
                if isinstance(runner, GridStrategyRunner):
                    try:
                        profiles = await self._db.get_regime_profiles(runner.name)
                        if profiles:
                            runner._regime_profile = profiles
                            logger.info("Simulator: regime profiles charges pour '{}' ({} assets)",
                                        runner.name, len(profiles))
                    except Exception as e:
                        logger.warning("Simulator: erreur chargement regime profiles '{}': {}", runner.name, e)
```

### Etape 8 — Portfolio engine (`portfolio_engine.py`)

**Dans `run()`** (lignes 191-194) : Charger les profiles AVANT `await db.close()` :
```python
        regime_profiles_by_strategy = {}
        for strat_name, _ in self._multi_strategies:
            try:
                profiles = await db.get_regime_profiles(strat_name)
                if profiles:
                    regime_profiles_by_strategy[strat_name] = profiles
            except Exception:
                pass
```

Passer a `_create_runners()` :
```python
        runners, indicator_engine = self._create_runners(
            filtered_multi, per_runner_capital, regime_profiles_by_strategy
        )
```

**Dans `_create_runners()`** (ligne 286) : Ajouter parametre `regime_profiles: dict | None = None`.
A la creation du runner (ligne 328), passer :
```python
            strat_profiles = regime_profiles.get(strat_name) if regime_profiles else None
            runner = GridStrategyRunner(
                ...,
                regime_profile=strat_profiles,
            )
```

### Etape 9 — Tests (`tests/test_regime_filter.py`)

12 tests repartis en 3 classes, pattern recopie de `tests/test_hotfix_20e.py` :

**Helpers** : `_make_gpm_config()`, `_make_mock_strategy()`, `_make_mock_config()` (avec `config.risk.regime_filter_enabled = True`), `_make_grid_runner()`, `_fill_buffer()`, `_make_candle()`

**TestRegimeMapping** (2 tests) :
1. `test_mapping_four_regimes` — TRENDING_UP→bull, DOWN→bear, RANGING→range, HIGH_VOL→crash
2. `test_low_volatility_not_mapped` — LOW_VOLATILITY absent du dict

**TestShouldAllowNewGrid** (6 tests) :
3. `test_no_profile_allows` — profile=None → True
4. `test_negative_sharpe_blocks` — bear sharpe=-2.4, regime=TRENDING_DOWN → False
5. `test_positive_sharpe_allows` — bear sharpe=1.5, regime=TRENDING_DOWN → True
6. `test_uncovered_regime_allows` — profile sans "bear", regime=TRENDING_DOWN → True
7. `test_zero_sharpe_allows` — sharpe=0.0 → True (>= 0)
8. `test_filter_disabled_via_config` — regime_filter_enabled=False → True

**TestRegimeFilterIntegration** (4 tests) :
9. `test_blocks_new_grid_on_candle` — pas de positions, bear sharpe=-2.4, TRENDING_DOWN → compute_grid pas appele
10. `test_does_not_block_existing_positions` — positions existantes, meme profil negatif → filter bypasse
11. `test_get_regime_profiles_from_db` — insert en DB + appel get_regime_profiles → parsing correct
12. `test_portfolio_runners_receive_regime_profile` — `_create_runners()` avec regime_profiles → chaque runner a le bon `_regime_profile`

## Notes

- **runner.name** = `self._strategy.name` (propriete ligne 596) = meme string que `strategy_name` en DB. Verifie.
- **Scope du filtre** : paper trading (Simulator) + portfolio backtest (PortfolioBacktester) uniquement. Le fast engine numpy (backtest mono-coin) n'utilise pas GridStrategyRunner → pas de filtre. La validation du filtre se fait via portfolio backtest.
- **`_regime_filter_blocks`** : compteur expose dans `get_status()` → visible dans le dashboard pour evaluer l'impact du filtre.

## Verification

1. `uv run python -m pytest tests/test_regime_filter.py -v` — 12 tests passent
2. `uv run python -m pytest --tb=short -q` — 0 regression (1074+ tests)
3. Verifier le log `REGIME FILTER` dans les tests d'integration
