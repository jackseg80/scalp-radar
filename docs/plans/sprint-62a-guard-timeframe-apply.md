# Sprint 62a — Guard timeframe dans `--apply`

**Date** : 1 mars 2026

## Problème

WFO `grid_atr` lancé avec `timeframe: [1h, 4h, 1d]` dans la grille de params.
Le best IS combo sélectionné avait `tf=1d` → `is_latest=1` en base → écrasement des bons résultats 1h :
- BTC : Grade B → D
- SOL : Grade A → F

## Solution

Dans `apply_from_db()` (`scripts/optimize.py`), ajout d'un **guard timeframe** (section 0) :

1. Lire le `timeframe` de référence depuis `strategies.yaml` (champ top-level de la stratégie)
2. Si pas de TF de référence dans yaml → mode = TF le plus fréquent parmi grades A/B (tiebreak `TF_ORDER` : préférer le TF le plus court)
3. Filtrer les résultats `is_latest=1` dont `timeframe ≠ ref_tf` :
   - Log `WARNING` pour chaque résultat ignoré
   - Seuls les résultats avec le bon TF passent à la suite

**Comportement** : Le mécanisme de blocage Sprint 37 (conflit A/B multi-TF) est maintenant inaccessible — le guard résout toujours l'ambiguïté en amont.

## Fichiers modifiés

- `scripts/optimize.py` — Guard section 0 dans `apply_from_db()`, import `Counter` ajouté
- `tests/test_timeframe_coherence.py` — 3 nouveaux tests + 2 tests Sprint 37 mis à jour
- `tests/test_optimization.py` — Fix pre-existant : glob backup `old/` au lieu de `tmp_path/`

## Tests nouveaux

- `test_apply_ignores_wrong_timeframe` : ref_tf=1h dans yaml, SOL 1d ignoré, BTC+ETH 1h appliqués
- `test_apply_warns_on_tf_mismatch` : WARNING capturé pour chaque TF ≠ ref_tf
- `test_apply_uses_mode_tf_if_no_ref` : pas de TF dans yaml → mode = 1h (BTC+ETH A), SOL 1d ignoré

## Tests mis à jour

- `test_apply_blocked_on_conflict` → comportement modifié (pas de blocage, filtrage + warning)
- `test_apply_blocked_exit_code` → idem

## Résultats

- **3 nouveaux tests** → **2165 tests, 2159 passants** (6 pré-existants), 0 régression
