# Plan : Refonte Grading WFO V2

## Contexte

Le grading actuel utilise des métriques par paliers (oos_is_ratio, monte_carlo, bitget_transfer) qui pénalisent injustement certains assets (ETH grade C malgré 83% win rate) et en surnotent d'autres (DYDX grade B malgré tail risk élevé). La V2 remplace par un scoring continu basé sur win_rate_oos et tail_risk_ratio.

## Tous les call sites de `compute_grade()` (7)

| # | Fichier | Ligne | Action |
|---|---------|-------|--------|
| 1 | `backend/optimization/report.py` | 107 | **Définition** — réécrire signature + corps |
| 2 | `backend/optimization/report.py` | 793 | `build_final_report()` — adapter l'appel |
| 3 | `scripts/optimize.py` | 1006 | `_regrade_from_db()` — adapter l'appel + update DB |
| 4 | `scripts/migrate_optimization.py` | 102 | Migration legacy — adapter l'appel |
| 5 | `tests/test_combo_score.py` | 252 | Test consistance — adapter params + expected |
| 6 | `tests/test_grading_shallow.py` | 24+ | 12 appels (PERFECT dict) — réécrire |
| 7 | `tests/test_optimization.py` | 377-474 | TestGrading (11 tests) — réécrire |

## Fichiers à modifier (8 + 1 nouveau)

| Fichier | Changement |
|---------|-----------|
| `backend/optimization/report.py` | Refonte `compute_grade()`, helpers, `build_final_report()`, warnings, types |
| `backend/core/database.py` | Migration ALTER TABLE (2 colonnes) |
| `backend/optimization/optimization_db.py` | Sauver `win_rate_oos` et `tail_risk_ratio` dans INSERT |
| `scripts/optimize.py` | Adapter `_regrade_from_db()` |
| `scripts/migrate_optimization.py` | Adapter l'appel `compute_grade()` |
| `tests/test_grading_v2.py` | **Nouveau** — 10 tests unitaires V2 |
| `tests/test_grading_shallow.py` | Adapter PERFECT + expected values |
| `tests/test_optimization.py` | Adapter TestGrading |
| `tests/test_combo_score.py` | Adapter 1 appel |

---

## Étape 1 : Types + Helpers dans `report.py`

### 1a. Changer les types

`GradeResult` : `score: int` → `score: float`, `raw_score: int` → `raw_score: float`
`FinalReport` : `total_score: int` → `total_score: float`, `raw_score: int` → `raw_score: float`

### 1b. Ajouter deux fonctions helper (avant `compute_grade`)

```python
def compute_win_rate_oos(windows: list) -> float:
    """% de fenêtres OOS avec return > 0%."""
    returns = [w.oos_net_return_pct if hasattr(w, 'oos_net_return_pct')
               else w.get('oos_net_return_pct', 0) for w in windows]
    if not returns:
        return 0.0
    return sum(1 for r in returns if r > 0) / len(returns)

def compute_tail_ratio(windows: list) -> float:
    """Ratio pertes sévères (<-20%) / gains. 0=parfait, 1=catastrophes mangent tout."""
    returns = [w.oos_net_return_pct if hasattr(w, 'oos_net_return_pct')
               else w.get('oos_net_return_pct', 0) for w in windows]
    pos_sum = sum(r for r in returns if r > 0)
    neg_bad = sum(r for r in returns if r < -20)  # valeur négative
    if pos_sum <= 0:
        return 1.0
    return abs(neg_bad) / pos_sum
```

## Étape 2 : Refonte `compute_grade()` dans `report.py`

**Anciens params supprimés** : `oos_is_ratio`, `mc_p_value`, `bitget_transfer`, `mc_underpowered`, `transfer_significant`, `bitget_trades`

**Nouvelle signature** :
```python
def compute_grade(
    oos_sharpe: float,
    win_rate_oos: float,
    tail_ratio: float,
    dsr: float,
    param_stability: float,
    consistency: float = 1.0,
    total_trades: int = 0,
    n_windows: int | None = None,
) -> GradeResult:
```

**Nouveau corps** (scoring continu, plus de paliers) :
```python
sharpe_score     = min(20, oos_sharpe * 3.5)              # /20
win_rate_score   = win_rate_oos * 20                       # /20
tail_score       = max(0, 15 * (1 - tail_ratio * 1.5))    # /15
dsr_score        = dsr * 15                                # /15
stability_score  = param_stability * 15                    # /15
consistency_score = consistency * 10                       # /10
mc_score         = 5                                       # /5 (forfait)

raw_score = sum des composantes ci-dessus

# Shallow penalty dégressive (remplace les paliers)
shallow_penalty = max(0, (24 - n_windows) * 0.8) if (n_windows is not None and n_windows < 24) else 0
score = raw_score - shallow_penalty
is_shallow = n_windows is not None and n_windows < 24

# Seuils grade (inchangés) : A≥85, B≥70, C≥55, D≥40, F<40
# Trade cap (inchangé) : <30→cap C, <50→cap B
```

**Logger** : adapter le message `logger.info(...)` aux nouvelles composantes (sharpe, win_rate, tail, dsr, stability, consistency).

## Étape 3 : Adapter `build_final_report()` dans `report.py` (call site #2)

Calculer les nouvelles métriques depuis `wfo.windows` :
```python
win_rate_oos = compute_win_rate_oos(wfo.windows)
tail_ratio_val = compute_tail_ratio(wfo.windows)
```

Nouvel appel :
```python
grade_result = compute_grade(
    oos_sharpe=wfo.avg_oos_sharpe,
    win_rate_oos=win_rate_oos,
    tail_ratio=tail_ratio_val,
    dsr=overfit.dsr.dsr,
    param_stability=overfit.stability.overall_stability,
    consistency=wfo.consistency_rate,
    total_trades=best_combo_trades,
    n_windows=len(wfo.windows),
)
```

Nouveaux warnings (après les existants) :
```python
if win_rate_oos < 0.6:
    warnings.append(f"Win rate OOS faible ({win_rate_oos:.0%})")
if tail_ratio_val > 0.5:
    warnings.append(f"Tail risk élevé ({tail_ratio_val:.2f})")
```

Warning shallow adapté au float :
```python
f"pénalité -{penalty:.1f} pts"
```

## Étape 4 : Migration DB dans `database.py`

Méthode `_migrate_grading_v2()` (même pattern que `_migrate_regime_analysis`) :
- ALTER TABLE ADD COLUMN `win_rate_oos REAL` (nullable)
- ALTER TABLE ADD COLUMN `tail_risk_ratio REAL` (nullable)
- Appelée dans `_ensure_optimization_table()` après `_migrate_regime_analysis`

## Étape 5 : Sauvegarder en DB dans `optimization_db.py`

`save_result_sync()` :
1. Ajouter params `win_rate_oos: float | None = None` et `tail_risk_ratio: float | None = None`
2. Ajouter 2 colonnes + 2 `?` dans l'INSERT SQL
3. Les caller dans `report.py` `save_report()` devra passer les valeurs

**Aussi** : vérifier que `save_report()` appelle `save_result_sync()` et lui passe les nouvelles valeurs.

## Étape 6 : Adapter `_regrade_from_db()` dans `scripts/optimize.py` (call site #3)

1. Importer `compute_win_rate_oos`, `compute_tail_ratio` depuis `report.py`
2. Lire `oos_sharpe` depuis la DB (déjà dans le SELECT)
3. Calculer depuis wfo_windows JSON :
   ```python
   win_rate_oos = compute_win_rate_oos(windows)
   tail_ratio = compute_tail_ratio(windows)
   ```
4. Appeler `compute_grade()` avec la nouvelle signature
5. UPDATE SQL : ajouter `win_rate_oos=?, tail_risk_ratio=?, warnings=?` (pas seulement grade+total_score)
6. **Ne PAS toucher** : wfo_windows, best_params, oos_sharpe, consistency, etc.
7. Affichage tableau : ajouter colonnes WinR et Tail

## Étape 7 : Adapter `migrate_optimization.py` (call site #4)

Ce script migre des résultats legacy. L'appel actuel utilise les anciens params.
Adapter : calculer `oos_sharpe`, `win_rate_oos=0.0`, `tail_ratio=1.0` (valeurs conservatrices pour legacy sans wfo_windows).

## Étape 8 : Tests

### 8a. Nouveau `tests/test_grading_v2.py` (10 tests)

1. `test_win_rate_oos_calculation` — [+10, +20, -5, +15, -30] → 0.6
2. `test_tail_risk_ratio_calculation` — pos=45, neg_bad=-30 → 0.667
3. `test_scoring_formula_exact` — Sharpe=5, wr=0.8, tail=0.1, dsr=0.9, stab=0.85, cons=0.7, n=30 → 84.5 → B
4. `test_shallow_penalty_degressive` — n=17→5.6, n=24→0, n=10→11.2
5. `test_tail_ratio_edge_no_gain` — pos_sum=0 → 1.0
6. `test_tail_ratio_edge_no_bad` — aucun < -20% → 0.0
7. `test_ada_regression_grade_a` — score ≥ 85 → A (avec métriques ADA)
8. `test_eth_regression_grade_b` — score ≥ 70 → B
9. `test_dydx_regression_grade_c` — score < 70 → C
10. `test_regrade_db_safety` — vérifier que regrade ne modifie que grade, total_score, win_rate_oos, tail_risk_ratio, warnings (pas wfo_windows, best_params, oos_sharpe)

### 8b. Adapter `tests/test_grading_shallow.py`

PERFECT V2 :
```python
PERFECT = dict(
    oos_sharpe=5.72,          # → min(20, 5.72*3.5) = 20.0
    win_rate_oos=1.0,          # → 20.0
    tail_ratio=0.0,            # → 15.0
    dsr=1.0,                   # → 15.0
    param_stability=1.0,       # → 15.0
    consistency=1.0,           # → 10.0
    total_trades=100,
)
# raw_score = 20+20+15+15+15+10+5 = 100
```

Expected values mise à jour :
- n=10 → penalty=11.2, score=88.8, grade=A (was B)
- n=15 → penalty=7.2, score=92.8, grade=A (was B)
- n=20 → penalty=3.2, score=96.8, grade=A (inchangé)
- n=24 → penalty=0, score=100, grade=A (inchangé)
- n=0  → penalty=19.2, score=80.8, grade=B (was B)

`test_shallow_does_not_upgrade` : adapter les params V2 pour donner un score brut ~49.

### 8c. Adapter `tests/test_optimization.py` TestGrading

Réécrire les 11 tests avec les nouveaux params V2.
- Tests spécifiques V1 supprimés (mc_underpowered, transfer_not_significant, few_bitget_trades)
- Tests conservés et adaptés : grade_a, grade_f, grade_c, trade_cap_30, trade_cap_50, trade_cap_above_50

### 8d. Adapter `tests/test_combo_score.py` (1 appel)

L'appel ligne 252 teste que consistency=0.68 réduit le score.
Adapter avec les nouveaux params V2 et recalculer expected values.

## Vérification

```bash
# 1. Tests ciblés
uv run pytest tests/test_grading_v2.py tests/test_grading_shallow.py tests/test_optimization.py tests/test_combo_score.py -x -q

# 2. Suite complète
uv run pytest tests/ -x -q

# 3. Re-grade prod
uv run python -m scripts.optimize --strategy grid_atr --regrade
# → Comparer avec tableau attendu (±1 point)
```
