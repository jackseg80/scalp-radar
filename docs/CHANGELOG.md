# CHANGELOG

Historique des changements notables par date de session.
Pour l'historique complet sprint par sprint, voir [ROADMAP.md](ROADMAP.md).

---

## 2026-02-22

### Sprint 38 — Shallow Validation + WFO Regression Analysis

- **feat(grading)** : `GradeResult` dataclass (grade, score, is_shallow, raw_score) ; pénalité shallow -10 (18-23 fenêtres), -20 (12-17), -25 (<12) ; seuil safe ≥24 fenêtres
- **feat(cli)** : `--regrade` dans `scripts/optimize.py` — recalcule les grades is_latest depuis la DB sans relancer le WFO
- **feat(frontend)** : badge ⚠ SHALLOW dans ResearchPage pour les assets sous le seuil
- **fix(regrade)** : `monte_carlo_pvalue = 0.0 or 1.0` → falsy en Python ; fix : `if x is not None else 1.0`
- **fix(regrade)** : `old_score` float passé à un format `%d` → `int(old_score)`
- **chore(param_grids)** : `sl_percent` max 25% pour grid_atr (retiré 30% de la grille WFO)
- **analysis** : régression rendement +74.8% → +43.6% après WFO re-run avec régimes corrigés (Hotfix 37c) — le fix expose correctement les fenêtres crash, WFO sélectionne des params plus conservateurs (SL moyen 22.1% → 25.5%)
- **analysis** : audit combo_score 4 variantes (V1/V3 = 20/21 assets identiques) — le scoring n'est pas la cause de la régression
- **analysis** : diagnostic margin starvation invalidé (0 trades skippés, margin moyenne 1.5%)
- **script** : `scripts/audit_combo_score.py` — analyse statique 4 variantes de scoring depuis la DB (lecture seule, rapport JSON dans `data/analysis/`)
- **décision** : leverage 7x maintenu (ratio rendement/DD 1.65 > 1.54 à 6x) ; ICP et OP exclus (Grade C shallow) ; 19 assets déployés (12 Grade A + 7 Grade B)
- **métriques** (portfolio backtest 365j, 7x, SL≤25%) : Return +43.6%, Max DD -26.4%, Worst SL 31.8%, Win rate 74.3%, 5949 trades, 0 kill switch, 15/19 assets profitables
