# Sprint 38 : Shallow Validation Penalty + --regrade

## Contexte

Le grading WFO (`compute_grade()` dans `report.py`) ne tient pas compte du nombre de fenêtres WFO (`n_windows`). Un asset avec 15 fenêtres peut obtenir Grade A malgré une confiance statistique insuffisante — un historique court ne contient souvent que des conditions favorables.

On ajoute une pénalité sur le score brut proportionnelle au manque de fenêtres, plus un flag `--regrade` pour recalculer les grades sans relancer le WFO.

---

## Barème de pénalité (révisé)

| n_windows | Pénalité | Score 100 → | Score 90 → | Grade (100) | Grade (90) |
|-----------|----------|-------------|------------|-------------|------------|
| ≥ 24      | 0        | 100         | 90         | A           | A          |
| 18–23     | -10      | 90          | 80         | A           | B          |
| 12–17     | -20      | 80          | 70         | B           | B          |
| < 12      | -25      | 75          | 65         | B           | C          |

Justification : avec -15/-5, un score parfait (100) avec 15 fenêtres survivait en A. Trop clément — un historique court est suspect. Les pénalités -20/-10 garantissent qu'un score réaliste (~90) est rétrogradé d'au moins un grade.

---

## Fichiers à modifier

| Fichier | Action |
|---------|--------|
| [report.py](backend/optimization/report.py) | `GradeResult` dataclass, modifier `compute_grade()`, `FinalReport`, `build_final_report()` |
| [optimization_db.py](backend/optimization/optimization_db.py) | Update import `GradeResult` (utilisé uniquement pour le type, pas d'appel direct à `compute_grade`) |
| [optimize.py](scripts/optimize.py) | Affichage CLI shallow + `--regrade` + `regrade_from_db()` |
| [migrate_optimization.py](scripts/migrate_optimization.py) | Update unpacking `compute_grade()` |
| [test_optimization.py](tests/test_optimization.py) | Update 11 call sites tuple → GradeResult |
| [test_combo_score.py](tests/test_combo_score.py) | Update 1 call site |
| `tests/test_grading_shallow.py` | **Nouveau** — ~13 tests |
| [ResearchPage.jsx](frontend/src/components/ResearchPage.jsx) | Badge ⚠ shallow à côté du grade |
| [ResearchPage.css](frontend/src/components/ResearchPage.css) | Style `.shallow-badge` |
| [STRATEGIES.md](docs/STRATEGIES.md) | Note shallow validation dans workflow |
| [ROADMAP.md](docs/ROADMAP.md) | Section Sprint 38 |

---

## Étapes d'implémentation

### 1. `backend/optimization/report.py` — Core logic

**1a. Ajouter `GradeResult` dataclass** (après `FinalReport`, ~ligne 90) :

```python
@dataclass
class GradeResult:
    grade: str        # Lettre finale (A-F) après pénalité + caps
    score: int        # Score final après pénalité shallow (cap trades ne modifie PAS le score)
    is_shallow: bool  # True si n_windows < 24
    raw_score: int    # Score brut avant pénalité
```

**1b. Modifier `compute_grade()`** :
- Ajouter paramètre `n_windows: int | None = None`
- Changer return type `tuple[str, int]` → `GradeResult`
- Après `score = sum(breakdown.values())` (ligne 194), sauvegarder `raw_score = score`
- Appliquer la pénalité :
  ```python
  shallow_penalty = 0
  if n_windows is not None:
      if n_windows >= 24:
          shallow_penalty = 0
      elif n_windows >= 18:
          shallow_penalty = 10
      elif n_windows >= 12:
          shallow_penalty = 20
      else:
          shallow_penalty = 25
      score = max(0, score - shallow_penalty)
  is_shallow = n_windows is not None and n_windows < 24
  ```
- Le grade lettre est calculé depuis `score` post-pénalité (lignes 196-206 inchangées)
- Les garde-fous cap trades s'appliquent après sur le grade lettre seulement (inchangés, ne modifient pas `score`)
- Update logger.info pour inclure `shallow_penalty` et `n_windows` si > 0
- `return GradeResult(grade=grade, score=score, is_shallow=is_shallow, raw_score=raw_score)`

**Note sur score vs grade** : le cap trades modifie uniquement la lettre, pas le score. Donc `GradeResult(score=85, grade="B")` est possible si trades < 50. C'est le comportement existant — on ne change pas.

**1c. Ajouter champs à `FinalReport`** (après `n_distinct_combos`, ligne 88) :
```python
shallow: bool = False
raw_score: int = 0
```

**1d. Modifier `build_final_report()`** (ligne 706) :
- `grade_result = compute_grade(..., n_windows=len(wfo.windows))`
- `grade = grade_result.grade`, `total_score = grade_result.score`
- Ajouter warning si shallow (avant la construction FinalReport) :
  ```python
  if grade_result.is_shallow:
      penalty = grade_result.raw_score - grade_result.score
      warnings.append(f"Shallow validation ({len(wfo.windows)} fenêtres < 24) — pénalité -{penalty} pts")
  ```
- Passer `shallow=grade_result.is_shallow, raw_score=grade_result.raw_score` au constructeur FinalReport

### 2. `backend/optimization/optimization_db.py` — Update import

Ligne 17 : ajouter `GradeResult` à l'import si nécessaire (pour le type). Pas d'appel direct à `compute_grade` dans ce fichier — il utilise `FinalReport` qui contient déjà le grade calculé. Vérifier que rien ne casse avec les nouveaux champs `shallow`/`raw_score` dans FinalReport (champs avec default → backward compat OK).

### 3. `scripts/migrate_optimization.py` — Update call site (ligne 102)

```python
# Avant : _, total_score = compute_grade(...)
# Après :
result = compute_grade(...)
total_score = result.score
```
Pas de `n_windows` disponible pour les vieux JSON → `None` par défaut → pas de pénalité.

### 4. `scripts/optimize.py` — CLI + --regrade

**4a. Display shallow** dans `_print_report()` (ligne 469) :
```python
# Avant : print(f"  GRADE : {report.grade}")
# Après :
if report.shallow:
    penalty = report.raw_score - report.total_score
    print(f"  GRADE : {report.grade}  ⚠ SHALLOW ({report.wfo_n_windows} fenêtres, raw: {report.raw_score}, pénalité: -{penalty})")
else:
    print(f"  GRADE : {report.grade} (score: {report.total_score})")
```

**4b. Recap** (ligne 1041) : ajouter `[SHALLOW]` si `r.shallow`

**4c. Argument `--regrade`** (après les autres argparse, ~ligne 877) :
- Requiert `--strategy` (erreur sinon)
- Incompatible avec `--apply`, `--symbol`, `--all-symbols`, `--resume`, `--all`

**4d. Fonction `regrade_from_db(strategy_name, config_dir)`** :
1. Résoudre `db_path` depuis config
2. Lit tous les résultats `is_latest=1` pour cette stratégie via sqlite3
3. Pour chaque row, extrait :
   - **Colonnes directes** : `oos_is_ratio`, `consistency`, `dsr`, `param_stability`, `monte_carlo_pvalue`, `mc_underpowered`, `n_windows`
   - **JSON `validation_summary`** : `bitget_trades`, `transfer_significant`, `transfer_ratio`
   - **`total_trades`** : cherche via `_get_best_combo_trades(conn, result_id)` (wfo_combo_results is_best=1), fallback somme `oos_trades` depuis JSON `wfo_windows`, **fallback ultime `total_trades=0`** (→ skip cap trades, pas de cap injuste)
4. Si une métrique clé est manquante (oos_is_ratio None, etc.) → **WARNING + skip asset** (ne pas crasher)
5. Appelle `compute_grade()` avec toutes les métriques + `n_windows`
6. Si grade ou score a changé → UPDATE `grade`, `total_score` en DB
7. Affiche tableau récapitulatif :
   ```
   --regrade grid_atr (21 assets)

   Asset          Old    New    OldS   NewS   RawS   Win  Shallow
   ─────────────────────────────────────────────────────────────
   BTC/USDT         A      A      92     92     92    28
   FET/USDT         A      B      88     68     88    15  -20pts *
   GALA/USDT        B      B      78     68     78    19  -10pts *
   ...
   3 grade(s) mis à jour sur 21 assets.
   ```

**4e. Helper `_get_best_combo_trades(conn, result_id)`** :
- Vérifier d'abord que la table `wfo_combo_results` existe (`.schema` ou try/except)
- Query `SELECT oos_trades FROM wfo_combo_results WHERE optimization_result_id=? AND is_best=1 LIMIT 1`
- Retourne `oos_trades` ou 0
- Si table inexistante ou erreur → retourne 0

### 5. Tests existants — Update unpacking

**`tests/test_optimization.py`** — 11 appels dans `TestGrading` (lignes 377-475) :
- `grade, score = compute_grade(...)` → `result = compute_grade(...)` puis `result.grade`, `result.score`
- Aucun ne passe `n_windows` → `None` par défaut → backward compat, assertions score/grade identiques
- Ajouter `assert result.is_shallow is False` et `assert result.raw_score == result.score` sur 1-2 tests pour vérifier le default

**`tests/test_combo_score.py`** — 1 appel (ligne 252) :
- `grade, score = compute_grade(...)` → `result = compute_grade(...)` puis `result.grade`, `result.score`

### 6. `tests/test_grading_shallow.py` — Nouveau fichier (~13 tests)

Helper `PERFECT` = params donnant 100/100 sans pénalité (avec `total_trades=100`, `bitget_trades=20`).

| Test | n_windows | raw | score | Grade | is_shallow |
|------|-----------|-----|-------|-------|------------|
| `test_shallow_under_12_penalty_25` | 10 | 100 | 75 | B | True |
| `test_shallow_12_to_17_penalty_20` | 15 | 100 | 80 | B | True |
| `test_shallow_18_to_23_penalty_10` | 20 | 100 | 90 | A | True |
| `test_shallow_24_plus_no_penalty` | 30 | 100 | 100 | A | False |
| `test_shallow_flag_true_at_23` | 23 | 100 | 90 | A | True |
| `test_shallow_flag_false_at_24` | 24 | 100 | 100 | A | False |
| `test_shallow_and_trades_both_apply` | 10 | 100 | 75→B, cap C (trades=25) | C | True |
| `test_shallow_penalty_then_trades_cap` | 15 | 100 | 80→B, cap B (trades=40) | B | True |
| `test_n_windows_none_no_penalty` | None | 100 | 100 | A | False |
| `test_n_windows_zero_max_penalty` | 0 | 100 | 75 | B | True |
| `test_raw_score_preserved` | 15 | 100 | 80 | B | True (raw=100) |
| `test_grade_result_fields` | 24 | — | — | — | GradeResult has 4 fields |
| `test_shallow_does_not_upgrade` | 30 | ~50 | 50 | D | False |

### 7. Frontend `ResearchPage.jsx`

**`n_windows` confirmé dans l'API** : ligne 496 de `optimization_db.py` (`r.n_windows` dans le SELECT), documenté dans la docstring route ligne 43 de `optimization_routes.py`.

**Ligne 704** (table row grade) — après le grade badge :
```jsx
<span className={`grade-badge grade-${r.grade}`}>{r.grade}</span>
{r.n_windows != null && r.n_windows < 24 && (
  <span className="shallow-badge" title={`Validation partielle (${r.n_windows} fenêtres < 24)`}>⚠</span>
)}
```

**Ligne 274** (detail view) — même pattern à côté du grade badge detail.

### 8. `ResearchPage.css` — Style shallow badge

```css
.shallow-badge {
  display: inline-block;
  margin-left: 4px;
  font-size: 0.75em;
  color: #f59e0b;
  cursor: help;
}
```

### 9. Documentation

**`docs/STRATEGIES.md`** : Note dans la section workflow WFO sur la pénalité shallow validation.
**`docs/ROADMAP.md`** : Section Sprint 38 avec résumé des changements.

---

## Vérification

1. `uv run pytest tests/test_grading_shallow.py -v` — 13 tests passent
2. `uv run pytest tests/test_optimization.py::TestGrading -v` — 11 tests existants passent (backward compat)
3. `uv run pytest tests/test_combo_score.py::test_consistency_impacts_grade -v` — passe
4. `uv run pytest` — les ~1648 tests passent sans régression
5. `uv run python scripts/optimize.py --regrade --strategy grid_atr` — affiche le tableau, met à jour les grades en DB
6. Vérifier le frontend ResearchPage : les badges ⚠ apparaissent à côté des grades avec n_windows < 24

---

## Contraintes respectées

- Zero régression sur ~1648 tests
- `n_windows=None` = backward compat (pas de pénalité)
- `n_windows=0` = pénalité max (-25)
- `total_trades` introuvable en `--regrade` → 0 → pas de cap trades (pas de cap C injuste)
- `score` dans GradeResult = post-pénalité shallow, le cap trades ne modifie que la lettre (documenté)
- Pas de nouvelle colonne DB — `is_shallow` calculé à la volée depuis `n_windows`
- Commit : `feat(grading): shallow validation penalty + --regrade (Sprint 38)`
- Plan copié dans `docs/plans/sprint-38-shallow-validation.md`
