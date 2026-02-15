# Sprint 15c — Fix Grading MC + combo_score

**Date** : 14-15 février 2026 (nuit)
**Tests** : 695 → 698 passants
**Objectif** : Corriger les bugs critiques dans le pipeline de grading qui faussaient les grades WFO.

---

## Problèmes identifiés

### 1. MC observed_sharpe IS au lieu d'OOS

Le Monte Carlo comparait les shuffles OOS au Sharpe **IS** (in-sample) au lieu du Sharpe **OOS** (out-of-sample). Conséquence : les stratégies avec un bon IS mais un OOS modéré obtenaient un p-value artificiellement bas.

**Impact** : DOGE passé de Grade C à Grade A après correction.

### 2. combo_score seuil trades trop bas

La formule `min(1, trades/50)` atteignait le plafond à 50 trades, permettant à des combos à faible volume d'être sélectionnées comme "best".

**Fix** : `min(1, trades/100)` — ETH sélectionne maintenant une combo avec 111 trades au lieu de 39.

### 3. Pas de garde-fou sur les faibles volumes

BTC avec seulement 6 trades pouvait obtenir Grade A — statistiquement non significatif.

**Fix** : Deux garde-fous dans `compute_grade()` :
- `total_trades < 30` → grade plafonné à C
- `total_trades < 50` → grade plafonné à B

### 4. Grille bornée à 0.05

`envelope_start` commençait à 0.05 — impossible de savoir si c'était l'optimum ou un artefact de borne.

**Fix** : Grille étendue à 0.05-0.15 (ajout 0.10 et 0.15). Résultat : 0.05 confirmé comme optimum réel.

### 5. DB polluée

Les résultats WFO en DB avaient été calculés avec le grading buggé.

**Fix** : Purge complète + recalcul avec le pipeline corrigé.

---

## Fichiers modifiés

- `backend/optimization/report.py` — `compute_grade()` : garde-fous trades, poids redistribués
- `backend/optimization/walk_forward.py` — `combo_score()` : seuil 50→100, MC observed_sharpe IS→OOS
- `backend/optimization/overfitting.py` — passage du bon Sharpe OOS au Monte Carlo
- `config/param_grids.yaml` — grille envelope_start étendue [0.05, 0.10, 0.15]
- `tests/test_combo_score.py` — tests combo_score avec seuil 100

---

## Résultats

Avant/après correction sur les 5 assets initiaux :

| Asset | Grade avant | Grade après |
| ----- | ----------- | ----------- |
| BTC   | B           | D (6 trades, plafonné) |
| ETH   | A           | A (111 trades) |
| SOL   | C           | A |
| DOGE  | C           | A |
| LINK  | B           | B |
