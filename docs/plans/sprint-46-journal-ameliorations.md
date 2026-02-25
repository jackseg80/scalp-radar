# Sprint 46 — Journal Améliorations

**Date** : 25 février 2026
**Commit** : `4f7e56f feat(journal): Sprint 46 — P&L Jour, filtre stratégie Live, balance snapshots`

## Objectifs

3 améliorations du journal de trading :

### Fix 1 — P&L Jour (remplace P&L Session)

**Problème** : "P&L Session" dans ExecutorPanel se réinitialise à chaque deploy/restart.

**Solution** :
- Endpoint `GET /api/journal/daily-pnl-summary` retourne `{daily_pnl, total_pnl, first_trade_date}`
- Méthode `get_daily_pnl_summary()` dans `database.py` (requête SQL agrégée sur live_trades)
- Frontend : 2 lignes dans ExecutorPanel ("P&L Jour" + "P&L Total") avec tooltip "depuis XX/02/2026"
- Polling 60s via `useApi`

### Fix 2 — Filtre stratégie dans l'onglet Live

**Problème** : l'onglet Live affichait toutes les stratégies mélangées.

**Solution** :
- Dropdown auto-peuplé depuis les stratégies détectées dans `live-stats`
- Propagation `?strategy=X` aux composants enfants (LiveStatsOverview, LiveDailyPnl, LivePerAssetSummary)
- Backend déjà prêt (Sprint 45 avait ajouté `?strategy=` sur tous les endpoints live)

### Fix 3 — Balance Snapshots (Equity Curve + Max DD)

**Problème** : Max Drawdown calculé uniquement depuis les trades (imprécis), pas d'equity curve live.

**Solution** :
- Table `balance_snapshots` (timestamp, strategy_name, balance, unrealized_pnl, margin_used, equity)
- Persistence best-effort toutes les ~1h, piggybacked sur `_balance_refresh_loop` existant
- `get_max_drawdown_from_snapshots()` : calcul peak-to-trough depuis les snapshots
- Endpoint `GET /api/journal/live-equity` pour l'equity curve
- Composant `LiveEquityCurve` (SVG, couleur verte/rouge selon tendance)
- `/api/journal/live-stats` enrichit `max_drawdown_pct` si null (fallback snapshots)

## Bugfix inclus

- `cycle_close` manquant dans `get_live_daily_pnl` (trade_type IN clause) — les cycles de grille n'étaient pas comptabilisés dans le P&L journalier

## Fichiers modifiés (7)

- `backend/core/database.py` (+143 lignes) — table, méthodes CRUD, max DD, daily PnL summary
- `backend/execution/executor.py` (+24 lignes) — persistence balance snapshots
- `backend/api/journal_routes.py` (+33 lignes) — 2 nouveaux endpoints
- `frontend/src/components/ExecutorPanel.jsx` (+30 lignes) — P&L Jour/Total
- `frontend/src/components/JournalPage.jsx` (+106 lignes) — filtre stratégie, equity curve
- `frontend/src/components/JournalPage.css` (+31 lignes) — styles filtre + equity SVG
- `tests/test_sprint46_journal.py` (+258 lignes) — 13 tests

## Tests (13 nouveaux)

- `TestDailyPnlSummary` (3) : P&L jour, sans trades, séparation jour/historique
- `TestLiveStatsStrategyFilter` (1) : filtre par stratégie
- `TestBalanceSnapshots` (2) : insert/read roundtrip, filtre par stratégie
- `TestMaxDrawdownFromSnapshots` (4) : calcul DD, sans snapshots, 1 snapshot, monotone up
- `TestBalanceSnapshotPersist` (2) : best-effort DB error, sans DB
- `TestDailyPnlCycleClose` (1) : cycle_close inclus dans daily PnL

**Total** : 1904 tests collectés, 1904 passants après Hotfix Tests Pré-existants (b3022a6).

## Hotfix post-Sprint 46

**React error #310** (commit `1ae9d14`) : `useApi` appelé après les early returns conditionnels dans
`ExecutorPanel` → page noire en prod. Fix : hook déplacé avant les `if (!executor) return` et
`if (executor.mode === 'paper') return`.
