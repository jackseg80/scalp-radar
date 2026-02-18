# Sprint 30 — Multi-Timeframe Support (timeframe comme paramètre WFO)

## Contexte

Les stratégies grid tournent uniquement en 1h. Certaines pourraient performer mieux en 4h ou 1d (moins de bruit, signaux plus forts). Ce sprint rend le timeframe optimisable dans le WFO : la DB stocke des candles 1h, et le pipeline resample 1h → 4h/1d à la volée. Aucune modification de la boucle chaude existante (`_simulate_grid_common`, `_build_entry_prices`).

## Contraintes

- ZÉRO modification de `_simulate_grid_common()` ni `_build_entry_prices()`
- ZÉRO modification du dataclass `IndicatorCache`
- ZÉRO modification des classes de stratégie
- Les 1169 tests existants passent sans modification
- `grid_multi_tf` exclu (filtre Supertrend 4h conçu pour main_tf=1h)
- `grid_funding` exclu (funding rates indexés 1h)
- `grid_range_atr` exclu (pas de timeframe pour l'instant)

---

## Fichiers à modifier

### 1. `backend/core/models.py` — Ajouter H4 et D1 au TimeFrame enum

Le modèle `Candle` utilise `timeframe: TimeFrame` (enum). Les candles resamplees ont besoin de `"4h"` et `"1d"`.

```python
class TimeFrame(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"   # NOUVEAU
    D1 = "1d"   # NOUVEAU
```

Mettre à jour `to_milliseconds()` avec `H4: 14_400_000` et `D1: 86_400_000`.

Backward compatible : `from_string()` itère déjà sur tous les membres.

### 2. `backend/optimization/indicator_cache.py` — Fonction `resample_candles()`

Nouvelle fonction **publique**, placée AVANT `build_cache()`.

Logique inspirée de `_resample_1h_to_4h()` (lignes 517-580) existante, mais :
- Retourne des objets `Candle` (pas des numpy arrays)
- Support `"1h"` (passthrough), `"4h"` et `"1d"`
- Ne garde que les buckets **COMPLETS** (4 candles pour 4h, 24 pour 1d)
- Log `WARNING` si un bucket incomplet est exclu **au milieu** des données (pas en début/fin)

Bucket alignment UTC :
- 4h : `epoch_seconds // 14400` (frontières 00/04/08/12/16/20h)
- 1d : `epoch_seconds // 86400` (frontière 00h)

Chaque candle resampleée :
- `timestamp` = timestamp de la PREMIÈRE candle du bucket
- `open` = open de la première, `close` = close de la dernière
- `high` = max des highs, `low` = min des lows, `volume` = somme
- `symbol`, `exchange` = copiés de la source
- `timeframe` = target_tf (nécessite H4/D1 dans TimeFrame)

### 3. `backend/optimization/walk_forward.py` — `_run_fast()` groupement par timeframe

**C'est le changement principal.** Modifier `_run_fast()` pour :

1. Grouper les combos par `params.get("timeframe", main_tf)`
2. Pour chaque groupe de timeframe :
   - Si tf ≠ `"1h"` : resampler via `resample_candles(candles_by_tf["1h"], tf)`
   - Construire `param_grid_values` SANS la clé `"timeframe"` (pas un indicateur)
   - Appeler `build_cache()` avec `main_tf=tf` et `db_path=None` si tf ≠ `"1h"` (skip funding)
   - Boucler les combos — `timeframe` reste dans les params mais est ignoré par le fast engine
   - Les résultats contiennent naturellement `timeframe` dans leur dict params
3. Si pas de clé `"timeframe"` dans les params → un seul groupe, comportement identique à l'actuel

**Note `total_days`** : `build_cache()` (indicator_cache.py:177-180) calcule déjà `total_days` depuis les timestamps réels (`(last_ts - first_ts) / 86400`). Le `n/24` mentionné est uniquement dans la fixture de test `make_indicator_cache` (conftest.py:167). Aucune modification de `build_cache()` nécessaire.

### 4. `backend/optimization/walk_forward.py` — OOS par timeframe

Dans `optimize()`, après sélection du meilleur IS :

```python
best_tf = best_params.get("timeframe", main_tf)

# Resampler les candles OOS si nécessaire
if best_tf != "1h":
    resampled = resample_candles(oos_candles_by_tf["1h"], best_tf)
    oos_candles_for_eval = {best_tf: resampled}
    oos_extra = None  # funding non applicable en 4h/1d
else:
    oos_candles_for_eval = oos_candles_by_tf
    oos_extra = oos_extra_data_map

# GARDER timeframe dans best_params — create_strategy_with_params()
# crée la config avec timeframe="4h", et compute_indicators() utilise
# self._config.timeframe pour chercher candles_by_tf["4h"]. Ça matche.
oos_result = run_multi_backtest_single(
    strategy_name, best_params, oos_candles_for_eval, bt_config,
    best_tf, extra_data_by_timestamp=oos_extra,
)
```

**Point critique** : Ne PAS retirer `timeframe` des params pour l'OOS event-driven. La stratégie en a besoin dans `compute_indicators()` pour trouver les candles (`candles_by_tf[self._config.timeframe]`). Toutes les configs Pydantic ont `timeframe: str` comme champ défini (pas extra), donc `create_strategy_with_params` le SET correctement.

### 5. `config/param_grids.yaml` — Ajouter timeframe

4 stratégies reçoivent `timeframe: ["1h", "4h", "1d"]` dans leur section `default:` :
- `grid_atr` (×3 → 9720 combos)
- `envelope_dca` (×3 → 972 combos)
- `envelope_dca_short` (×3 → 972 combos)
- `grid_trend` (×3 → 7776 combos)

3 stratégies EXCLUES :
- `grid_multi_tf` — filtre 4h spécifique à 1h
- `grid_funding` — funding rates indexés 1h
- `grid_range_atr` — pas de support multi-TF

**Note fenêtres 1d** : IS=180j = 180 candles 1d, SMA(50) warmup = 130 exploitables, OOS=60 candles. Résultats potentiellement instables en 1d — à surveiller dans les runs WFO. Pas de changement de code nécessaire, mais le praticien doit interpréter avec prudence.

### 6. `tests/test_multi_timeframe.py` — ~27 tests

**Section 1 — Resampling (8 tests)** :
- Passthrough `"1h"` → même liste
- 24 candles 1h → 6 candles 4h (OHLCV correct)
- Bucket incomplet exclu (23 candles → 5 candles 4h)
- 48 candles 1h → 2 candles 1d
- Bucket incomplet exclu pour 1d
- Timestamps alignées UTC
- Liste vide → liste vide
- symbol/exchange copiés de la source

**Section 2 — total_days via build_cache (3 tests)** :
- build_cache avec candles 4h resamplees : total_days ~ n_candles / 6 (vérifié via timestamps)
- build_cache avec candles 1d : total_days ~ n_candles / 1
- Cohérence : même période couverte → ≈ même total_days quelle que soit la TF

**Section 3 — Fast engine multi-TF (6 tests)** :
- grid_atr sur candles 4h : résultat 5-tuple valide
- grid_atr sur candles 1d : résultat valide
- envelope_dca sur candles 4h : résultat valide
- Nombre de trades réduit sur 4h vs 1h
- Sharpe annualisé cohérent (pas explosion à cause de total_days faux)
- Résultats déterministes (même input → même output)

**Section 4 — Parité (5 tests, LES PLUS IMPORTANTS)** :
- **Parité 1h ancien vs nouveau** : `_run_fast()` avec grid SANS timeframe → résultats identiques (net_return, n_trades) au comportement pré-refactoring. Test bit-à-bit.
- **Parité 1h explicit** : grid avec `timeframe="1h"` explicite → même résultat que sans timeframe
- SMA(14) sur candles 4h resamplees == SMA calculée manuellement
- grid_atr 4h ≠ grid_atr 1h (confirmer que le TF change les résultats)
- grid_atr existant parity (tests/test_grid_atr.py) toujours ok

**Section 5 — Intégration param_grids (3 tests)** :
- grid_atr contient `"timeframe"` dans param_grids
- grid_multi_tf ne contient PAS `"timeframe"`
- grid_funding ne contient PAS `"timeframe"`

**Section 6 — _run_fast groupement (2 tests)** :
- Grid mixte [1h, 4h] → résultats pour les deux TFs
- Grid sans timeframe → un seul groupe, même nombre de résultats

---

## Ce qu'on NE touche PAS

- `_simulate_grid_common()` — boucle chaude inchangée
- `_build_entry_prices()` — factory inchangée
- `IndicatorCache` dataclass — champs inchangés
- Classes de stratégie (grid_atr.py, etc.) — inchangées
- `MultiPositionEngine` — inchangé
- `build_cache()` — inchangé (total_days déjà correct via timestamps)
- `fast_multi_backtest.py` — inchangé
- `backend/optimization/__init__.py` — inchangé
- Les 1169 tests existants

---

## Vérification

```bash
# 1. Nouveau module
uv run python -m pytest tests/test_multi_timeframe.py -v

# 2. Régression complète
uv run python -m pytest --tb=short -q

# 3. Parité fast engine existante
uv run python -m pytest tests/test_grid_atr.py tests/test_fast_engine_refactor.py tests/test_grid_range_atr.py -v
```
