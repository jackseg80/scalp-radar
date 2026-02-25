# Sprint 46b — Sélecteur stratégie global + Panneau Executor contextuel

**Date** : 25 février 2026
**Commit** : `d5a03de feat(journal): Sprint 46b — sélecteur stratégie global + P&L contextuels`

## Contexte

Le `StrategyContext` + `StrategyBar` existaient déjà mais `StrategyBar` n'était visible que sur l'onglet
Scanner. Les métriques P&L de l'ExecutorPanel ignoraient la stratégie sélectionnée. JournalPage avait
son propre dropdown (Sprint 46) sans lien avec le contexte global.

## Changements

### Changement 1 — StrategyBar visible sur toutes les pages

`Header.jsx` ligne 17 : suppression de la condition `activeTab === 'scanner' &&`.

StrategyBar retourne `null` si aucune stratégie détectée → safe sur tous les onglets.

### Changement 2 — P&L Executor contextuels

**Backend** (`database.py`) : `get_daily_pnl_summary(strategy=None)` — filtre optionnel
`AND strategy_name = ?` dans les 3 requêtes SQL (daily_pnl, total_pnl, first_trade_date).

**Backend** (`journal_routes.py`) : `GET /api/journal/daily-pnl-summary?strategy=X`.

**Frontend** (`ExecutorPanel.jsx`) : import `useStrategyContext`, construction `?strategy=X` selon
`strategyFilter`, hook `useApi` mis à jour dynamiquement.

### Changement 3 — JournalPage synchro avec contexte global

`JournalPage.jsx::LiveJournal` :
- Import `useStrategyContext`
- `effectiveStrategy = strategyFilter || strategy` — le contexte global prime
- Dropdown local masqué quand `strategyFilter` est actif (redondant)

### Hotfix simultané — strategies.yaml leverage vide

Le commit `445b755 grid_atr to 4x` avait laissé `leverage:` vide (None) dans strategies.yaml.
Pydantic `StrategiesConfig.grid_atr.leverage` attend un int → 34 tests cassés.
Fix : `leverage: 4`.

## Fichiers modifiés (7)

- `frontend/src/components/Header.jsx` — supprimer condition scanner
- `frontend/src/components/ExecutorPanel.jsx` — useStrategyContext + URL dynamique
- `frontend/src/components/JournalPage.jsx` — effectiveStrategy + masquage dropdown
- `backend/core/database.py` — strategy param dans get_daily_pnl_summary()
- `backend/api/journal_routes.py` — Query param strategy
- `tests/test_sprint46_journal.py` — 2 nouveaux tests filtre + backward compat
- `config/strategies.yaml` — leverage grid_atr 4 (fix vide → None)

## Tests (2 nouveaux)

- `test_daily_pnl_filter_by_strategy` : grid_atr isolé, grid_multi_tf isolé
- `test_daily_pnl_no_strategy_returns_all` : backward compat sans filtre

**Total** : 1906 tests, 1906 passants, 0 régression.
