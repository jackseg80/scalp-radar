# Plan Sprint 26 : Funding Costs dans le Backtest

## Contexte

Actuellement, le backtest ignore les funding rate costs sur les positions ouvertes pour toutes les stratégies sauf `grid_funding`. En live sur Bitget futures, toutes les 8h (00:00, 08:00, 16:00 UTC), chaque position ouverte paie ou reçoit :

```
funding_payment = funding_rate × notional_value
```

- LONG + funding positif → on PAIE (coût)
- LONG + funding négatif → on REÇOIT (revenu)
- SHORT : inversé

Seule la stratégie `grid_funding` calcule actuellement ces coûts dans son fast engine (`_calc_grid_pnl_with_funding()`). Les autres stratégies grid (`grid_atr`, `envelope_dca`, `grid_multi_tf`, `grid_trend`) ignorent le funding.

**Objectif** : Appliquer les funding costs à TOUTES les stratégies grid dans les deux moteurs de backtest (event-driven + fast engine).

## Problème Critique Identifié

**Incohérence de convention /100** entre les deux chemins de données :

1. **extra_data_builder.py** (ligne 62) : stocke `funding_rate` SANS diviser par 100
   - Valeur DB brute en % (ex: 0.01 = 0.01%)
   - Utilisé par MultiPositionEngine via `extra_data_by_timestamp`

2. **indicator_cache.py** (ligne 134) : DIVISE par 100
   - `fr_values = ... / 100` → valeur en decimal (ex: 0.0001)
   - Utilisé par le fast engine dans `_simulate_grid_funding()`

**Résultat** : bug silencieux de facteur 100 si on ne corrige pas !

## Exploration Codebase

### Fichiers Critiques Identifiés

**Moteur event-driven :**
- [backend/backtesting/multi_engine.py](backend/backtesting/multi_engine.py) — boucle principale `run()`, ligne 100-234
- [backend/backtesting/engine.py](backend/backtesting/engine.py) — `BacktestResult` dataclass, ligne 46-57
- [backend/backtesting/extra_data_builder.py](backend/backtesting/extra_data_builder.py) — charge funding rates (BUG /100)

**Fast engine :**
- [backend/optimization/fast_multi_backtest.py](backend/optimization/fast_multi_backtest.py) — `_simulate_grid_common()` ligne 210+, `_calc_grid_pnl_with_funding()` ligne 438-482
- [backend/optimization/indicator_cache.py](backend/optimization/indicator_cache.py) — `_load_funding_rates_aligned()` ligne 103-142, `build_cache()` ligne 145+

**Plumbing WFO :**
- [backend/optimization/walk_forward.py](backend/optimization/walk_forward.py) — `_run_fast()` ligne 1039-1107, `optimize()` ligne 431+
- [backend/optimization/report.py](backend/optimization/report.py) — `validate_on_bitget()` ligne 240+
- [backend/optimization/overfitting.py](backend/optimization/overfitting.py) — à vérifier

**Tests :**
- Nouveau fichier : `tests/test_funding_costs.py` (~15-20 tests)

## Plan Détaillé

### Phase 0 — Fix Convention /100 (CRITIQUE + CASCADE)

**Fichier 1** : [backend/backtesting/extra_data_builder.py](backend/backtesting/extra_data_builder.py)

**Changement ligne 62** :
```python
# AVANT
extra[EXTRA_FUNDING_RATE] = current_funding_rate

# APRÈS
extra[EXTRA_FUNDING_RATE] = current_funding_rate / 100  # DB stocke en %, convertir en decimal
```

**Fichier 2** : [backend/strategies/grid_funding.py](backend/strategies/grid_funding.py)

**PIÈGE CASCADE IDENTIFIÉ** : grid_funding.py ligne 124-125 DIVISE AUSSI par 100 → double division !

**Changement ligne 125** :
```python
# AVANT
funding_rate_pct = ctx.extra_data.get("funding_rate", 0)
funding_rate = funding_rate_pct / 100  # DOUBLE DIVISION !

# APRÈS
funding_rate = ctx.extra_data.get("funding_rate", 0)  # Déjà en decimal depuis Phase 0
```

**Impacts** :
- MultiPositionEngine recevra des valeurs en decimal (cohérent avec le fast engine)
- grid_funding en event-driven utilisera la bonne échelle (pas de double division)
- **CRITIQUE** : Tester grid_funding AVANT/APRÈS pour valider la correction

**Tests à corriger** : `tests/test_funding_oi_data.py` doit valider que les deux chemins retournent la même échelle.

### Phase 1 — MultiPositionEngine (Event-Driven)

**Fichier** : [backend/backtesting/multi_engine.py](backend/backtesting/multi_engine.py)

**Emplacement** : Dans la boucle principale `run()`, après le traitement des signaux/trades (ligne ~163) mais AVANT le calcul equity curve (ligne 199).

**Implémentation** :
```python
from datetime import timezone

# c. Funding costs (si settlement 8h ET positions ouvertes)
GRID_STRATEGIES_WITH_FUNDING = {
    "grid_funding", "grid_atr", "envelope_dca", "envelope_dca_short",
    "grid_multi_tf", "grid_trend",
}

if (
    positions
    and extra
    and self._strategy.name in GRID_STRATEGIES_WITH_FUNDING
):
    # PIÈGE TIMEZONE : normaliser en UTC (candle.timestamp peut être naive ou autre TZ)
    if candle.timestamp.tzinfo is None:
        # Assumer UTC si naive
        utc_hour = candle.timestamp.hour
    else:
        utc_hour = candle.timestamp.astimezone(timezone.utc).hour

    if utc_hour in (0, 8, 16):
        fr = extra.get(EXTRA_FUNDING_RATE)
        if fr is not None:
            for pos in positions:
                # PIÈGE NOTIONAL : utiliser entry_price (cohérence fast engine)
                # PAS candle.close (divergerait avec fast engine ligne 454)
                notional = pos.entry_price * pos.quantity
                if pos.direction == Direction.LONG:
                    # LONG + funding positif → paie (signe négatif)
                    payment = -fr * notional
                else:
                    # SHORT + funding positif → reçoit (signe positif)
                    payment = fr * notional
                capital += payment  # Ajouter au capital (sera dans equity)
                funding_total += payment  # Tracker pour BacktestResult
```

**Variables à ajouter** :
- `funding_total = 0.0` initialisé en début de `run()`
- Import `EXTRA_FUNDING_RATE` depuis `backend.strategies.base`

**BacktestResult enrichi** :

Modifier [backend/backtesting/engine.py](backend/backtesting/engine.py) ligne 46-57 :

**PIÈGE DATACLASS** : Champs avec default DOIVENT être après les champs sans default !

```python
@dataclass
class BacktestResult:
    """Résultat complet d'un backtest."""

    config: BacktestConfig
    strategy_name: str
    strategy_params: dict[str, Any]
    trades: list[TradeResult]
    equity_curve: list[float]
    equity_timestamps: list[datetime]
    final_capital: float
    funding_paid_total: float = 0.0  # NOUVEAU champ optionnel (DEFAULT EN DERNIER)
```

**Impact tests** : Constructeurs positionnels dans `test_backtesting.py` sont backward-compat car dataclass accepte kwargs.

**Retour dans multi_engine.py** ligne 225 :
```python
return BacktestResult(
    config=self._config,
    strategy_name=self._strategy.name,
    strategy_params=strategy_params,
    trades=trades,
    equity_curve=equity_curve,
    equity_timestamps=equity_timestamps,
    final_capital=capital,
    funding_paid_total=funding_total,  # NOUVEAU
)
```

### Phase 2 — Fast Engine (_simulate_grid_common)

**Fichier** : [backend/optimization/fast_multi_backtest.py](backend/optimization/fast_multi_backtest.py)

**Partie A — _build_entry_prices** : Rien à changer (déjà OK)

**Partie B — _simulate_grid_common ligne 210+** :

**1. Signature existante** (ligne 131-141) — PAS de changement de signature.

Les nouveaux params `funding_rates` et `candle_timestamps` sont déjà dans `cache` :
- `cache.funding_rates_1h` — array ou None
- `cache.candle_timestamps` — array ou None

Pas besoin de les passer en argument, on les lit depuis le cache directement.

**2. Précalcul settlement mask** (avant la boucle, OPTIMISÉ) :
```python
# Settlement check : heures 0, 8, 16 UTC (vectorisé numpy)
settlement_mask = np.zeros(n, dtype=bool)
if candle_timestamps is not None:
    hours = ((candle_timestamps / 3600000) % 24).astype(int)
    settlement_mask = (hours % 8 == 0)
```

**3. Dans la boucle, APRÈS le check exit mais AVANT l'equity** :

**SIMPLIFICATION** : `direction` est toujours `1` (LONG) ou `-1` (SHORT), donc :
```python
payment = -fr * notional * direction
```
unifie les deux cas sans if/else.

**Note** : quand `directions` (array dynamique, grid_multi_tf/grid_trend) est fourni, `direction` est mis à jour à chaque candle (ligne 202). Les positions ouvertes sont toutes du MÊME côté (force-close au flip), donc `direction` courant au moment du settlement est correct.

```python
# Funding costs aux settlements
funding_rates = cache.funding_rates_1h
if positions and settlement_mask[i] and funding_rates is not None:
    fr = funding_rates[i]
    if not np.isnan(fr):
        for _lvl, entry_price, quantity, _fee in positions:
            notional = entry_price * quantity  # PAS closes[i]
            # direction=1 (LONG) : -fr × notional (positif=coût)
            # direction=-1 (SHORT) : +fr × notional (inversé)
            capital += -fr * notional * direction
```

**4. Appeler avec funding depuis les wrappers** :

Modifier `_simulate_grid_atr()`, `_simulate_envelope_dca()`, `_simulate_grid_multi_tf()`, `_simulate_grid_trend()` ligne ~630+ :

```python
return _simulate_grid_common(
    cache, params, bt_config, entry_prices, direction,
    funding_rates=cache.funding_rates_1h,      # NOUVEAU
    candle_timestamps=cache.candle_timestamps, # NOUVEAU
)
```

**Note** : `_simulate_grid_funding()` garde son code existant (ne pas toucher).

### Phase 3 — Indicator Cache (Charger Funding pour Toutes les Grids)

**Fichier** : [backend/optimization/indicator_cache.py](backend/optimization/indicator_cache.py)

**Ligne 305-323** — Condition grid_funding :

**AVANT** :
```python
if strategy_name == "grid_funding":
    # ... charge funding
```

**APRÈS** :
```python
GRID_STRATEGIES_WITH_FUNDING = {
    "grid_funding", "grid_atr", "envelope_dca", "envelope_dca_short",
    "grid_multi_tf", "grid_trend",
}

if strategy_name in GRID_STRATEGIES_WITH_FUNDING:
    # ... charge funding (code existant inchangé)
```

**Impacts** :
- Le cache chargera funding pour toutes les stratégies grid
- Les SMA nécessaires (ma_period) sont déjà chargées par les blocs précédents

### Phase 4 — Plumbing WFO et Event-Driven Fallback

**Fichier** : [backend/optimization/__init__.py](backend/optimization/__init__.py)

**CRITIQUE** : Étendre `STRATEGIES_NEED_EXTRA_DATA` ligne 60 pour charger funding dans le event-driven fallback.

**AVANT** :
```python
STRATEGIES_NEED_EXTRA_DATA: set[str] = {"funding", "liquidation", "grid_funding"}
```

**APRÈS** :
```python
STRATEGIES_NEED_EXTRA_DATA: set[str] = {
    "funding", "liquidation",
    "grid_funding", "grid_atr", "envelope_dca", "envelope_dca_short",
    "grid_multi_tf", "grid_trend",
}
```

**Impact** :
- walk_forward.py ligne 503 : `needs_extra = strategy_name in STRATEGIES_NEED_EXTRA_DATA`
- Charge funding/OI pour TOUTES les stratégies grid (event-driven fallback)
- Le fast engine (Phase 2) charge déjà via indicator_cache (Phase 3)

**Fichier** : [backend/optimization/walk_forward.py](backend/optimization/walk_forward.py)

**Vérifications** :
- `_run_fast()` ligne 1074 : passe déjà `db_path`, `symbol`, `exchange` ✓
- `_parallel_backtest()` ligne 990 : appelle `_run_fast` avec ces params ✓
- `optimize()` ligne 526 : `db_path = db.db_path` récupéré ✓

**Aucun changement nécessaire** dans walk_forward.py (plumbing déjà OK).

**Fichier** : [backend/optimization/report.py](backend/optimization/report.py)

**validate_on_bitget()** ligne 287-302 :

**Condition actuelle** :
```python
if strategy_name == "grid_funding":
    # Charge funding + OI
```

**APRÈS** :
```python
from backend.optimization import GRID_STRATEGIES

if strategy_name in GRID_STRATEGIES:
    # Charge funding + OI (code existant)
```

**Utiliser la constante existante** au lieu de redéfinir localement.

**Fichier** : [backend/optimization/overfitting.py](backend/optimization/overfitting.py)

**Vérification** : `_run_backtest_for_strategy()` ligne 31 accepte déjà `extra_data_by_timestamp` et le passe correctement → **OK, aucun changement nécessaire**.

**Fichier** : [backend/backtesting/portfolio_engine.py](backend/backtesting/portfolio_engine.py)

**NOUVEAU** : Agréger funding total dans les snapshots portfolio.

**Dans `_build_result()` ou `_take_snapshot()`** :
```python
# Calculer funding total de tous les runners
total_funding_paid = sum(
    r.result.funding_paid_total
    for r in self._runners.values()
    if r.result
)
# Sauvegarder dans le rapport ou les snapshots
```

### Phase 5 — Tests (ÉTENDU À 25 TESTS)

**Nouveau fichier** : `tests/test_funding_costs.py` (25 tests organisés en 7 sections)

**Section 1 : Settlement Detection (5 tests)**
1. `test_settlement_hour_detection_utc` — Candle 08:00 UTC détectée
2. `test_settlement_hour_detection_non_utc` — Candle 08:00 CET convertie en UTC
3. `test_no_settlement_between_hours` — 09:00 skip
4. `test_settlement_all_three_hours` — 0h, 8h, 16h OK
5. `test_settlement_mask_vectorized` — Précalcul numpy mask correct

**Section 2 : Funding Calculation (6 tests)**
6. `test_funding_long_negative_rate` — LONG + funding négatif → reçoit bonus
7. `test_funding_long_positive_rate` — LONG + funding positif → paie
8. `test_funding_short_negative_rate` — SHORT + funding négatif → paie
9. `test_funding_short_positive_rate` — SHORT + funding positif → reçoit bonus
10. `test_notional_entry_price_not_close` — Utilise entry_price PAS candle.close
11. `test_funding_accumulated_multi_settlements` — 3 settlements × funding

**Section 3 : Division /100 (4 tests)**
12. `test_extra_data_builder_divides_by_100` — Ligne 62 fix validé
13. `test_indicator_cache_receives_decimal` — Pas de double division
14. `test_grid_funding_no_double_division` — Ligne 125 division retirée
15. `test_end_to_end_funding_value` — DB (-0.05%) → backtest (-0.0005 decimal)

**Section 4 : Backward Compat (3 tests)**
16. `test_backtest_result_default_funding_zero` — Constructor sans funding_paid_total
17. `test_backtest_result_with_funding` — Constructor avec funding_paid_total
18. `test_old_tests_dont_break` — Constructeurs kwargs backward compat

**Section 5 : Parity Event-Driven vs Fast (4 tests)**
19. `test_parity_grid_atr_with_funding` — <0.1% delta
20. `test_parity_envelope_dca_with_funding`
21. `test_parity_grid_trend_with_funding`
22. `test_parity_no_funding_data_unchanged` — Stratégie sans funding inchangée

**Section 6 : Portfolio Aggregation (2 tests)**
23. `test_portfolio_funding_total_aggregated` — Somme runners.funding_paid_total
24. `test_portfolio_snapshot_includes_funding` — Snapshot equity post-funding

**Section 7 : Edge Cases (1 test)**
25. `test_funding_nan_skipped` — NaN funding rate → skip payment

**Régression complète** :
```bash
pytest tests/ --tb=short -q
```

Doit passer les 1037 tests existants sans échec.

## Points d'Attention (7 PIÈGES CRITIQUES IDENTIFIÉS)

1. **Convention /100 CASCADE** : Phase 0 doit fixer extra_data_builder.py ET grid_funding.py (double division !)
2. **NE PAS modifier** `_simulate_grid_funding()` ni `_calc_grid_pnl_with_funding()` (Sprint 22)
3. **Timestamp timezone** : normaliser en UTC explicitement (candle.timestamp peut être naive)
4. **Mark price proxy** : utiliser `entry_price` (cohérence fast engine ligne 454), PAS `candle.close`
5. **Backward compat dataclass** : champ avec default EN DERNIER (sinon TypeError)
6. **LONG vs SHORT signes** : LONG = `-fr × notional` (positif = coût), SHORT inversé
7. **Portfolio aggregation** : tracker funding_paid_total dans portfolio_engine.py

## Séquence d'Implémentation Recommandée

### Étape 1 : Fix /100 CASCADE (CRITIQUE)
- Modifier `extra_data_builder.py` ligne 62 : `/100`
- Modifier `grid_funding.py` ligne 125 : retirer division
- Lancer `pytest tests/test_grid_funding.py -v` → valider pas de régression

### Étape 2 : Tests Unitaires Funding Calculation
- Créer `test_funding_costs.py` Section 2 (6 tests)
- Valider formules LONG/SHORT avant intégration

### Étape 3 : MultiPositionEngine Integration
- Implémenter Phase 1 avec timezone handling UTC
- Ajouter BacktestResult.funding_paid_total (dataclass field avec default)
- Lancer tests Section 2 + Section 4

### Étape 4 : Fast Engine Integration
- Implémenter Phase 2 avec settlement mask vectorisé
- Corriger notional = entry_price (pas closes[i])
- Lancer tests Section 5 (parity <0.1%)

### Étape 5 : Plumbing + Portfolio + Tests Finaux
- Phase 3-4 (indicator_cache, __init__.py, report.py, portfolio_engine.py)
- Tests Section 1, 3, 6, 7
- Validation WFO end-to-end sur grid_atr BTC/USDT

## Vérification Finale

**Checklist avant commit** :
- [ ] Phase 0 : Division /100 CASCADE (extra_data_builder + grid_funding)
- [ ] Phase 1 : MultiPositionEngine + timezone UTC + BacktestResult.funding_paid_total
- [ ] Phase 2 : _simulate_grid_common + settlement mask vectorisé + entry_price
- [ ] Phase 3 : GRID_STRATEGIES_WITH_FUNDING dans indicator_cache
- [ ] Phase 4 : STRATEGIES_NEED_EXTRA_DATA + report.py + portfolio_engine.py
- [ ] Phase 5 : 25 tests test_funding_costs.py (7 sections)
- [ ] Régression : 1037 tests passent (pytest --tb=short -q)
- [ ] Parity test : grid_atr event vs fast < 0.1% diff
- [ ] Validation WFO : grid_atr BTC/USDT 730j avec funding

## Fichiers Critiques (11 fichiers)

**Phase 0 (FIX /100 CASCADE)** :
- [backend/backtesting/extra_data_builder.py](backend/backtesting/extra_data_builder.py) — ligne 62
- [backend/strategies/grid_funding.py](backend/strategies/grid_funding.py) — ligne 125

**Phase 1 (Event-Driven)** :
- [backend/backtesting/multi_engine.py](backend/backtesting/multi_engine.py) — ligne ~163
- [backend/backtesting/engine.py](backend/backtesting/engine.py) — ligne 46-57

**Phase 2 (Fast Engine)** :
- [backend/optimization/fast_multi_backtest.py](backend/optimization/fast_multi_backtest.py) — ligne 210+

**Phase 3 (Indicator Cache)** :
- [backend/optimization/indicator_cache.py](backend/optimization/indicator_cache.py) — ligne 305-323

**Phase 4 (Plumbing)** :
- [backend/optimization/__init__.py](backend/optimization/__init__.py) — ligne 60
- [backend/optimization/report.py](backend/optimization/report.py) — ligne 287-302
- [backend/backtesting/portfolio_engine.py](backend/backtesting/portfolio_engine.py) — aggregation funding

**Phase 5 (Tests)** :
- [tests/test_funding_costs.py](tests/test_funding_costs.py) — NOUVEAU (25 tests)
- [tests/test_funding_oi_data.py](tests/test_funding_oi_data.py) — corriger division /100

## Références

- [COMMANDS.md](COMMANDS.md) — commandes de test
- Sprint 22 : Grid Funding (ne pas toucher _calc_grid_pnl_with_funding)
- [backend/strategies/base.py](backend/strategies/base.py) ligne 16 : EXTRA_FUNDING_RATE
