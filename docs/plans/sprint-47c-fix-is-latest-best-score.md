# Sprint 47c — Fix is_latest garde le meilleur score

## Contexte

Après le Sprint 47b (fix déduplication cross-timeframe), un second bug a été détecté dans la logique `is_latest`. Le WFO traite les timeframes séquentiellement (1h → 4h → 1d) ; l'implémentation précédente attribuait `is_latest=1` au dernier run inséré, de manière inconditionnelle — quel que soit son score.

**Exemple réel :**
```
BNB 1h → score 95, Grade A  → is_latest=1
BNB 4h → score 73, Grade B  → écrase le 1h → is_latest=1  (perte de 22 points)
BNB 1d → score 73, Grade B  → écrase le 4h → is_latest=1
```

Résultat : `apply_from_db()` lisait le pire résultat au lieu du meilleur.

---

## Fix

### optimization_db.py — save_result_sync() (chemin local)

Avant l'insertion, vérifier si un résultat existant a un meilleur score :

```python
existing = conn.execute(
    "SELECT id, total_score FROM optimization_results WHERE strategy_name=? AND asset=? AND is_latest=1",
    (strategy, asset),
).fetchone()

if existing is None or new_score >= existing[1]:
    # Nouveau meilleur ou égal → il prend is_latest
    if existing:
        conn.execute("UPDATE ... SET is_latest=0 WHERE ...")
    is_latest_val = 1
else:
    # Ancien meilleur → is_latest=0 pour le nouveau
    is_latest_val = 0
```

### optimization_db.py — save_result_from_payload_sync() (chemin push serveur)

Même logique, après l'INSERT :

```python
existing = conn.execute(
    "SELECT id, total_score FROM optimization_results WHERE ... AND id!=new_id",
    ...
).fetchone()

if existing is None or new_score >= existing[1]:
    # Garder is_latest=1 sur le nouveau, mettre 0 sur l'ancien
else:
    # Rétrograder le nouveau à is_latest=0
    conn.execute("UPDATE ... SET is_latest=0 WHERE id=new_id")
```

**Règle `>=` (pas `>`)** : à score égal, le run le plus récent gagne (données plus fraîches).

---

## Tests ajoutés (tests/test_optimization_db.py)

1. `test_is_latest_keeps_better_score` : 1h score=95 → 4h score=73 → is_latest reste sur le 1h
2. `test_is_latest_replaced_by_better_score` : 4h score=73 → 1h score=95 → is_latest passe au 1h
3. `test_is_latest_equal_score_newer_wins` : 1h score=80 → 4h score=80 → is_latest sur le 4h (plus récent)
4. `test_is_latest_payload_sync_keeps_better` : même que test 1 via `save_result_from_payload_sync()`

---

## Fichiers modifiés

- `backend/optimization/optimization_db.py` — 2 fonctions modifiées
- `tests/test_optimization_db.py` — 4 nouveaux tests

## Résultats

- **1927 tests, 1927 passants**, 0 régression
