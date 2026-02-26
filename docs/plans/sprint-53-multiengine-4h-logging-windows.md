# Sprint 53 — Multi-Engine 4h/1d + Logging Windows Fix

**Date** : 26 fév 2026
**Durée** : 1 session

---

## Contexte

Deux bugs identifiés post-Sprint 52 :

### Bug 1 : WFO Phases 2+3 produisent 0 trades quand timeframe="4h" ou "1d"

Le WFO optimise `timeframe` parmi `["1h", "4h", "1d"]` pour la plupart des stratégies grid.
Quand le meilleur combo a `timeframe="4h"` :

- **Phase 1 (fast engine)** : OK — `_resample_1h_to_4h()` interne gère le cas
- **Phase 2 (multi_engine overfitting detection)** : **0 trades** — cherche `candles_by_tf["4h"]` qui n'existe pas
- **Phase 3 (multi_engine validation Bitget)** : **0 trades** — même cause

`optimize.py` et `report.py` chargent uniquement les bougies 1h (car `default_cfg.timeframe = "1h"`).
`resample_candles()` existait déjà dans `indicator_cache.py` mais n'était pas appelé par `multi_engine`.

**Impact** : Score Bitget toujours 0/15 pour les stratégies qui optimisent en 4h/1d → WFO sous-performant.

### Bug 2 : PermissionError [WinError 32] en milliers lors des backtests

Loguru tente `os.rename()` lors de la rotation à 50 MB.
Sur Windows, l'antivirus / Windows Search Indexer verrouille le fichier → `PermissionError`.
La rotation échoue, Loguru retente à chaque écriture → milliers d'erreurs sur stderr.

`enqueue=True` (ajouté en cours de session) sérialise les écritures mais ne suffit pas :
`os.rename()` échoue aussi depuis le thread enqueue.

---

## Corrections

### Fix 1 : Resampling 1h→4h/1d dans `MultiPositionEngine.run()`

**Fichier** : `backend/backtesting/multi_engine.py`

Ajout en début de `run()` :
1. Copie mutable de `candles_by_tf` pour ne pas modifier les données de l'appelant
2. Détection des TFs manquants : union de `strategy.min_candles.keys()` + `strategy._config.timeframe`
3. Pour chaque TF manquant dans `{"4h", "1d"}` : appel `resample_candles(candles_1h, tf)` depuis `indicator_cache`
4. Ajustement automatique de `main_tf` au TF natif de la stratégie

Aucune modification de `optimize.py`, `report.py`, `walk_forward.py` : le fix est localisé dans le moteur.
`grid_multi_tf` non affecté : son `config.timeframe` est toujours "1h", le 4h Supertrend est calculé en interne.

### Fix 2 : Logging Windows sans `os.rename()`

**Fichier** : `backend/core/logging_setup.py`

Sur Windows : utilisation de `{time}` dans le nom de fichier du sink.
Loguru crée alors un **nouveau fichier** à la rotation sans renommer l'ancien → zéro `os.rename()`.

```
scalp_radar_{time:YYYY-MM-DD_HH-mm-ss}.log  (Windows)
scalp_radar.log                              (Linux/prod, inchangé)
```

Compression désactivée sur Windows (même source de PermissionError post-rotation).
Sur Linux (prod Docker), comportement inchangé : rotation + compression gz.

---

## Tests

4 nouveaux tests dans `tests/test_multi_engine.py` (classe `TestMultiEngineResampling`) :

| Test | Description |
|------|-------------|
| `test_resample_utc_boundaries_4h` | 8 bougies 1h → 2 bougies 4h aux bonnes frontières UTC |
| `test_resample_utc_boundaries_1d` | 48 bougies 1h → 2 bougies 1d |
| `test_multi_engine_4h_resamples_and_produces_trades` | `grid_atr timeframe=4h` + seulement candles 1h → trades produits, equity 4h-granulaire |
| `test_multi_engine_1h_unchanged_no_resampling` | Régression : stratégie 1h + candles 1h → comportement inchangé |

Correction de 3 tests sur `config/assets.yaml` suite à suppression de ARB/USDT :
- `tests/test_config_assets.py` : `EXPECTED_COUNT` 21→20, `NEW_ASSETS` sans ARB, `REMOVED_ASSETS` + ARB, `TOP_SCALP_ASSETS` sans ARB
- `tests/test_dataengine_autoheal.py::TestConfigAssets` : count 21→20, assert ARB absent

---

## Résultat

- **2028 tests** (+4 resampling), **2024 passants** (1 flaky pré-existant `test_kill_switch_reliability` event-loop, 3 tests config ARB corrigés)
- 0 régression
- WFO Phases 2+3 supportent désormais les combos 4h/1d
- Plus de milliers d'erreurs PermissionError lors des backtests Windows
