# Sprint 24a — Portfolio Backtest Realistic Mode

## Contexte

Le portfolio backtest grid_atr 21 assets affichait peak margin 284%, soit une liquidation certaine en live. Trois corrections pour que le backtest reflète la réalité.

## Problèmes identifiés

1. **Compounding abusif** : les runners réinvestissaient les profits → sizing exponentiel
2. **Pas de global margin guard** : chaque runner vérifie sa marge locale, pas la marge totale du portfolio
3. **Kill switch passif** : détecté a posteriori via `_check_kill_switch()`, mais les runners continuaient à trader pendant la simulation

## Corrections

### Correction 1 — Sizing fixe (anti-compounding)

**Fichiers** : `backend/backtesting/simulator.py`, `backend/backtesting/portfolio_engine.py`

- Flag `_portfolio_mode = True` sur chaque runner portfolio
- En portfolio mode, sizing basé sur `_initial_capital` (pas `_capital` courant)
- Transparent pour live/paper : `getattr(self, "_portfolio_mode", False)` = False si absent

### Correction 2 — Global Margin Guard

**Fichiers** : `backend/backtesting/simulator.py`, `backend/backtesting/portfolio_engine.py`

- Chaque runner reçoit `_portfolio_runners` (dict tous les runners) et `_portfolio_initial_capital`
- Après le margin guard local existant (Sprint 20a), calcule la marge globale (tous runners) et skip si `> capital × max_margin_ratio` (70%)

### Correction 3 — Kill switch temps réel

**Fichier** : `backend/backtesting/portfolio_engine.py`

- Fenêtre glissante 24h dans `_simulate()` après chaque snapshot
- Si DD% ≥ seuil (30%), gèle tous les runners (`_kill_switch_triggered = True`)
- Cooldown 24h : après expiration de `_kill_freeze_until`, dégèle les runners
- Le kill switch se re-déclenche tant que les snapshots haute-equity sont dans la fenêtre

## Design : zéro impact live/paper

Tous les ajouts sont derrière `getattr(..., False/None)` :

- `_portfolio_mode` : absent en live → `getattr(self, "_portfolio_mode", False)` = False
- `_portfolio_runners` : absent en live → `getattr(self, "_portfolio_runners", None)` = None
- `_portfolio_initial_capital` : idem
- `_kill_freeze_until` : initialisé dans `__init__()` du PortfolioBacktester uniquement

## Tests ajoutés (5 nouveaux)

1. `test_portfolio_mode_fixed_sizing` — sizing basé sur initial_capital, pas le capital courant
2. `test_normal_mode_uses_current_capital` — vérifie que le mode compound normal fonctionne toujours
3. `test_global_margin_guard_blocks` — marge globale à 65% bloque les nouvelles ouvertures
4. `test_global_margin_under_threshold` — marge globale à 20% laisse passer
5. `test_kill_switch_freezes_all_runners` — trigger → gel → cooldown 24h → reset

## Ce qu'on NE touche PAS

- Code live/paper : tous les ajouts sont transparents si les attributs n'existent pas
- `fast_multi_backtest.py` : single-asset WFO, pas concerné
- Les 1007 tests existants passent tous
- L'equity curve et le drawdown calculation restent identiques

## Résultat

- **1012 tests** (+5 nouveaux), 0 régression
