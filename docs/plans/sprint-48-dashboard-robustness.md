# Sprint 48 — Dashboard Robustness + Sidebar Portfolio (25 fév 2026)

## Context

Les résultats de robustness (Bootstrap, Regime Stress, Historical Stress, CVaR) sont calculés via CLI (`scripts/portfolio_robustness.py --save`) et stockés dans la table SQLite `portfolio_robustness`. Actuellement visibles uniquement dans le terminal. Ce sprint ajoute leur affichage dans le dashboard, et corrige également plusieurs bugs de la sidebar portfolio (scroll, assets-list, resize).

## Proposition alternative au spec original

**Spec original** : nouvel onglet/section + endpoint `GET /api/portfolio/robustness?label=...`

**Implémentation retenue** : **section contextuelle intégrée dans le panel de résultats**, pas un onglet séparé. Raisons :
- La robustness est **per-backtest** (liée par `backtest_id`) → l'afficher avec le run sélectionné est naturel
- Pas de changement de routing/tabs — le panel apparaît automatiquement sous BenchmarkBTC quand des données existent
- Le verdict (VIABLE/CAUTION/FAIL) est visible immédiatement
- Si aucune donnée robustness → le panel ne s'affiche pas du tout (zéro bruit)
- Endpoint REST propre : `GET /backtests/{id}/robustness` au lieu de filtrer par label

---

## 1. Backend

### 1.1 Nouvelle fonction DB async

**Fichier** : `backend/backtesting/portfolio_db.py`

Ajout de `_ROBUSTNESS_JSON_COLUMNS` tuple + `get_robustness_by_backtest_id_async(db_path, backtest_id) -> dict | None` :
- Vérifie que la table `portfolio_robustness` existe (via `sqlite_master`) — graceful si jamais créée
- `SELECT * WHERE backtest_id = ? ORDER BY id DESC LIMIT 1` — prend le plus récent si re-run
- Parse les 4 colonnes JSON : `regime_stress_results`, `historical_stress_results`, `cvar_by_regime`, `verdict_details`
- Retourne `None` si table absente ou pas de résultat

### 1.2 Nouvel endpoint API

**Fichier** : `backend/api/portfolio_routes.py`

```
GET /api/portfolio/backtests/{backtest_id}/robustness
→ {"robustness": {...}} ou {"robustness": null}
```

- Retourne `null` (pas 404) si pas de données — le frontend masque la section
- Zéro modification des endpoints existants

---

## 2. Frontend

### 2.1 Nouveau composant : `RobustnessPanel.jsx` + `.css`

**Props** : `backtestId: number | null`

**Comportement** :
- Fetch `GET /api/portfolio/backtests/{id}/robustness` quand `backtestId` change
- Retourne `null` si pas de données (section invisible)
- Gère loading + cleanup via `cancelled` flag (même pattern que PortfolioPage)

**Sous-composants internes** :
- `VerdictBadge` — badge VIABLE (vert) / CAUTION (jaune) / FAIL (rouge)
- `CriteriaList` — 4 critères GO/NO-GO avec icônes pass/fail
- `BootstrapSection` — 6 métriques (median return, CI95, DD, prob loss, etc.)
- `CvarSection` — VaR/CVaR journalier + 30j + par régime
- `RegimeStressSection` — tableau scénarios (return, DD, prob. perte)
- `HistoricalStressSection` — tableau crashes (période, portfolio DD, BTC DD, recovery)

**CSS** : suit les conventions existantes (#111, #1a1a1a, #333, monospace, .pnl-pos/.pnl-neg)

### 2.2 Intégration dans PortfolioPage

**Fichier** : `frontend/src/components/PortfolioPage.jsx`

- Import `RobustnessPanel`
- `<RobustnessPanel backtestId={detail.id} />` inséré entre BenchmarkBTC et AssetTable

### 2.3 Fix sidebar portfolio (3 bugs)

**Problèmes** :
1. Scroll vertical bloqué — les contraintes `max-height` sur `.assets-list` et `.portfolio-config` bloquaient le scroll
2. Assets-list minuscule en mode sélection manuelle — même cause
3. Pas de resize width — absente

**Fix** :
- Suppression de toutes les contraintes `max-height` / `overflow-y` sur `.assets-list` et `.portfolio-config`
- Scroll naturel via `.content { overflow-y: auto }` déjà existant dans `styles.css`
- Ajout d'un **resize handle** drag pour la colonne config (240–520px, persisté en localStorage `portfolio-config-width`, default 320px) — même pattern que le handle sidebar de `App.jsx`
- `PortfolioPage.jsx` : `configWidth` state + `mainRef` + `dragging` ref + `useEffect` mouseMove/mouseUp
- `PortfolioPage.css` : `.pf-resize-handle` + `.portfolio-main` columns via style prop inline
- Responsive : `grid-template-columns: 1fr !important` sur ≤1024px, handle masqué

---

## 3. Tests

**Fichier** : `tests/test_portfolio_robustness_routes.py` (3 tests)

| Test | Description |
|------|-------------|
| `test_get_robustness_with_data` | Mock retourne données complètes → vérifie 200 + verdict + métriques |
| `test_get_robustness_empty_db` | Mock retourne None → vérifie 200 + `robustness: null` |
| `test_get_robustness_correct_backtest_id` | Vérifie que le bon `backtest_id` est passé au mock |

Pattern : `@patch("backend.api.portfolio_routes.get_robustness_by_backtest_id_async")` + `TestClient(app)`

---

## 4. Fichiers modifiés/créés

| Fichier | Action |
|---------|--------|
| `backend/backtesting/portfolio_db.py` | Ajout `get_robustness_by_backtest_id_async()` |
| `backend/api/portfolio_routes.py` | Import + endpoint GET robustness |
| `frontend/src/components/RobustnessPanel.jsx` | Créé (~200 lignes) |
| `frontend/src/components/RobustnessPanel.css` | Créé (~160 lignes) |
| `frontend/src/components/PortfolioPage.jsx` | Import RobustnessPanel + resize handle |
| `frontend/src/components/PortfolioPage.css` | Fix scroll + resize handle + assets-list |
| `tests/test_portfolio_robustness_routes.py` | Créé (3 tests) |

## 5. Note sur les données DB

La table `portfolio_robustness` a :
- 11 colonnes REAL pour Bootstrap (median, CI, probas)
- 4 colonnes REAL pour CVaR
- `regime_stress_results` TEXT (JSON) — contient **seulement les scénarios**, pas la distribution observée
- `historical_stress_results` TEXT (JSON) — crashes avec status/period/DD/recovery
- `cvar_by_regime` TEXT (JSON) — CVaR par régime (RANGE/BULL/BEAR/CRASH)
- `verdict` TEXT — "VIABLE" | "CAUTION" | "FAIL"
- `verdict_details` TEXT (JSON) — critères détaillés avec pass/fail

## 6. Résultats

- **3 nouveaux tests** → **1933 tests, 1933 passants**, 0 régression
- `test_portfolio_robustness_routes.py` : 3/3 ✅
- `test_portfolio_routes.py` : 10/10 ✅ (non-régression)
