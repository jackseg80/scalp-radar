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

### Sprint 38b — Window Factor Fix

- **fix(wfo): CRITICAL — `window_factor` dans `combo_score()`** — pénalise les combos évaluées sur peu de fenêtres OOS
  - 20/21 assets grid_atr avaient leur best combo sélectionné sur 1-5 fenêtres au lieu de 30
  - Cause : 2-pass WFO (coarse LHS → fine autour du top 20) → combos fine spécifiques à chaque fenêtre → consistency triviale (1/1 = 100%)
  - Fix : `window_factor = min(1.0, n_windows / max_windows)` — combo à 1/30 fenêtres multipliée par 0.033
  - Impact grid_atr (19 assets, 7x, 365j) : Return +43.6% → +57.7%, Max DD -26.4% → -24.1%, perdants 4 → 1, ratio rendement/DD 1.65 → 2.39
- **feat(wfo)** : persistance `per_window_sharpes` dans `wfo_combo_results` (liste des Sharpe OOS par fenêtre par combo, JSON en DB) — précurseur du diagnostic, utilisable pour variantes V2/V4 futures
- **fix(frontend)** : `ExportDiagnostic.jsx` — `comboScore` JS aligné avec backend, trades/50 → trades/100
- **analysis** : audit combo_score 4 variantes (V1/V3 = 20/21 assets identiques → formule stable, biais vient du nombre de fenêtres, pas de la formule)
- **analysis** : diagnostic margin starvation invalidé (0 trades skippés lors du window_factor)
- **analysis** : grid_boltrend post-window-factor — 7 Grade B effondrés à Grade C (consistency réelle 62-77% vs 100% apparent) ; portfolio 2 assets (BTC+DYDX) non viable : +7.4%, DD -38.6% → stratégie mise en pause
- **tests** : 6 nouveaux (4 tests window_factor + 2 tests per_window_sharpes), 1687 total
