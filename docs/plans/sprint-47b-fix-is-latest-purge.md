# Sprint 47b — Fix is_latest + Script Purge WFO

## Contexte

Après les WFO du Sprint 47, un bug structurel a été découvert dans la déduplication du flag `is_latest` dans `optimization_results`. Quand le meilleur timeframe change entre deux runs WFO (ex : best TF 1h → 4h pour grid_atr/BTC), la requête UPDATE filtrait aussi par `timeframe`, laissant plusieurs `is_latest=1` pour le même (strategy, asset). Résultat : `apply_from_db()` lisait plusieurs configs conflictuelles pour le même asset.

---

## Fixes implémentés

### Fix 1 — optimization_db.py : Supprimer le filtre timeframe du UPDATE

**Avant (bug) :**
```sql
UPDATE optimization_results SET is_latest=0
WHERE strategy_name=? AND asset=? AND timeframe=? AND is_latest=1
```

**Après (correct) :**
```sql
UPDATE optimization_results SET is_latest=0
WHERE strategy_name=? AND asset=? AND is_latest=1
```

Deux emplacements corrigés :
- Chemin local `save_result_sync()` (ligne ~115)
- Chemin push/remote `save_result_from_payload_sync()` (ligne ~243)

### Fix 2 — Script purge : scripts/purge_wfo_duplicates.py

Nouveau script pour corriger les doublons existants en base (créés par le bug avant correction) :
- Détecte les (strategy_name, asset) avec plusieurs `is_latest=1`
- Conserve uniquement l'entrée avec `MAX(total_score)` (plus récent en cas d'égalité)
- Supporte `--dry-run` et `--db-path`
- Usage : `uv run python -m scripts.purge_wfo_duplicates [--dry-run]`

### Fix 3 — param_grids.yaml (déjà fait en hotfix Sprint 47)

Déjà corrigé dans le commit hotfix `515b0f3` : restauration des valeurs `ma_period=7` et `atr_multiplier_start=1.0/3.0` qui étaient actives en production.

---

## Tests ajoutés (tests/test_optimization_db.py)

1. `test_is_latest_dedup_across_timeframes` : TF 1h → TF 4h pour le même (strategy, asset) → seul 1 `is_latest=1`
2. `test_is_latest_different_strategies_independent` : grid_atr/BTC et grid_multi_tf/BTC ont chacun leur `is_latest=1` (indépendants)
3. `test_purge_script_keeps_best` : 3 doublons manuels (scores 55/78/62) → purge → seul score=78 conservé

---

## Fichiers modifiés

- `backend/optimization/optimization_db.py` — 2 requêtes UPDATE corrigées
- `scripts/purge_wfo_duplicates.py` — nouveau script (créé)
- `tests/test_optimization_db.py` — 3 nouveaux tests
- `docs/ROADMAP.md` — sprint 47b documenté

## Résultats

- **1923 tests, 1923 passants**, 0 régression
- Bug reproductible avéré : chaque WFO multi-TF pouvait générer des doublons `is_latest=1`
