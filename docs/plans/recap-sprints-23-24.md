# Scalp-Radar — Récapitulatif Sprints 23-24

## Contexte

Système de trading automatisé crypto sur Bitget futures. 13 stratégies implémentées, 1016 tests. Objectif : valider et déployer les stratégies grid les plus robustes via backtesting rigoureux et walk-forward optimization.

---

## Sprint 23 — Grid Trend Strategy

Nouvelle stratégie **grid_trend** (13e stratégie) : trend-following DCA avec trailing stop ATR, filtre directionnel EMA cross + ADX, force-close au flip. Complément à grid_atr (mean-reversion).

**WFO Results :**
- grid_atr : 28/28 assets Grade A/B (Sharpe OOS 6-17, suspicieusement élevé)
- grid_trend : 6/22 Grade B (SOL, ICP, XTZ, AR, CRV, JUP), convergence params exceptionnelle (4 params identiques sur 6 assets)

---

## Validation Bitget — Reality Check

Test de transferabilité sur données Bitget récentes (90 derniers jours = bear market soutenu nov 2025 → fév 2026).

**Résultat brutal : 7/21 assets grid_atr ont un Sharpe Bitget négatif** (BTC, DOGE, XTZ, SOL, ETH perdent de l'argent en réalité).

**Explication :** La fenêtre Bitget couvre un bear market soutenu SANS recovery (-44% depuis ATH BTC $126k). Mean-reversion DCA est piégée quand les prix ne remontent pas à la SMA. Les grades WFO ne sont pas faux (fenêtres OOS 2022-2024 contiennent crashes AVEC recovery), mais la validation Bitget montre honnêtement les limites en régime adverse.

**Enseignement :** Valide la thèse grid_trend comme complément — grid_atr piégé en bear, grid_trend SHORT profite de la baisse.

---

## Sprint 23b — Fix compute_live_indicators

`compute_live_indicators()` pas implémenté pour grid_trend → 0 trades en portfolio backtest. Fix : ajout calcul EMA + ADX dans grid_trend.py. 1007 tests passent.

---

## Sprint 24a — Portfolio Backtest Réaliste

### Problèmes identifiés

Le portfolio backtest original avait 5 écarts critiques vs la réalité :

| Problème | Impact |
|----------|--------|
| Runners isolés, pas de margin guard global | Peak margin 284-453% = liquidation |
| Compounding non-contrôlé (sizing grossit avec le capital) | ICP 67% du P&L total |
| Kill switch post-hoc (logge mais ne stoppe rien) | 7 triggers ignorés |
| P&L revient dans son runner (pas pool commun) | Biais concentration |
| Pas de corrélation inter-assets | DD sous-estimé |

### 3 Corrections implémentées

1. **Sizing fixe** — `_portfolio_mode = True` → sizing sur `_initial_capital` toujours, pas `_capital` (élimine compounding)
2. **Margin guard global** — Marge totale de tous runners comparée à 70% du capital initial (pas chaque runner isolé)
3. **Kill switch actif** — Freeze tous runners pendant 24h quand DD dépasse 30% en fenêtre glissante

### Résultat : avant vs après

| Métrique | Avant (artefact) | Après (réaliste) |
|----------|-------------------|------------------|
| Return grid_atr 21 | +1 247% | +181% |
| Max DD | -53.8% | -31.6% |
| Peak Margin | 284% | 23.5% |
| Kill Switch | 7 (ignorés) | 0 |

---

## Sprint 24b — Portfolio Backtest Multi-Stratégie

Support de N stratégies simultanées dans le portfolio backtest. Clé runner = `strategy:symbol` (ex: `grid_atr:ICP/USDT`). Dispatch candles à tous les runners du même symbol. Rétro-compatible. 1016 tests, 0 échec.

---

## Résultats Portfolio — 730 jours (complet)

| Config | Return | Max DD | Durée DD | Peak Margin | Runners prof. |
|--------|--------|--------|----------|-------------|---------------|
| grid_atr 21 assets | +181% | -31.6% | 77h | 23.5% | 20/21 |
| grid_atr Top 10 | +221% | -29.8% | 50h | 25.0% | 10/10 |
| grid_trend 6B | +77% | -20.2% | 4195h | 23.6% | 5/6 |
| Combiné 16 runners | +167% | -20.0% | 48h | 18.5% | 15/16 |
| **Combiné sans SOL (15)** | **+180%** | **-21.1%** | **48h** | **19.3%** | **15/15** |

**Enseignements :**
- Top 10 > 21 assets (moins de dilution, meilleur rendement)
- Combiné a le meilleur drawdown (-20%) et la meilleure récupération (48h)
- SOL est le seul perdant constant → exclu

---

## Forward Test — 365 derniers jours (validation overfitting)

Test sur la dernière année uniquement (fév 2025 → fév 2026), incluant bull terminal + crash -44% + bear actuel.

| Config | Return | Max DD | Runners prof. |
|--------|--------|--------|---------------|
| **grid_atr Top 10** | **+82.4%** | **-25.7%** | **9/10** |
| Combiné (atr+trend) | +49.3% | -23.7% | 9/16 |

**Découverte critique : grid_trend échoue en forward.** Seulement 1/5 runners profitables (XTZ). La dernière année est trop chaotique pour le trend-following (pump → dump → range → dump, pas de trends prolongés).

**grid_atr confirme sa robustesse** — 9/10 profitables en forward, mêmes leaders (ICP, CRV, DYDX, FET). La sélection d'assets n'est pas massivement overfittée.

---

## Fees & Funding

**Fees Bitget intégrées :** maker 0.02%, taker 0.06%, slippage 0.05%. Levier ×6.

**Funding rates NON intégrées** dans grid_atr/grid_trend (seulement dans grid_funding). Impact estimé ~5-10% du P&L. Données en DB (124k rows). TODO pour un futur sprint.

---

## Décisions & Config Actuelle

### Paper trading actif (depuis ce sprint)
- **grid_atr Top 10 assets** : BTC, CRV, DOGE, DYDX, ENJ, FET, GALA, ICP, NEAR, AVAX
- grid_trend : **pas déployé** (échoue en forward test)
- Backup 21 assets sauvegardé : `strategies.yaml.bak.21assets`

### Rationale
Le ratio rendement/risque de grid_atr Top 10 est le meilleur :
- 730j : +221%, DD -29.8%, ratio 7.4
- Forward 365j : +82.4%, DD -25.7%
- 0 kill switch, peak margin ~25% (sous le guard 70%)
- 9/10 assets profitables en forward

---

## TODO / Prochains sprints

| Priorité | Item | Détail |
|----------|------|--------|
| Observation | Paper trading Top 10 | Laisser tourner quelques semaines |
| Moyen | Funding cost integration | Pattern existe dans grid_funding, données en DB |
| Moyen | Fix grading : Bitget transfer < 0 → Grade plafonné D | Évite de recommander des assets qui perdent en live |
| Futur | Grid_trend re-évaluation | Quand les trends seront plus propres |
| Futur | Portfolio multi-strat live | Quand grid_trend sera validé en forward |
| Futur | Journal activité live trading | Positions ouvertes/fermées, P&L non réalisé historique |

---

## Fichiers clés modifiés

| Fichier | Sprint | Changement |
|---------|--------|-----------|
| `backend/backtesting/portfolio_engine.py` | 24a, 24b | Margin guard global, sizing fixe, kill switch actif, multi-stratégie |
| `backend/backtesting/simulator.py` | 24a | `_portfolio_mode`, global margin check |
| `backend/strategies/grid_trend.py` | 23b | `compute_live_indicators()` |
| `scripts/portfolio_backtest.py` | 24b | `--strategies`, `--preset combined` |
| `config/strategies.yaml` | — | Réduit à 10 assets grid_atr |
| `config/strategies.yaml.bak.21assets` | — | Backup 21 assets |
| `tests/test_portfolio_backtest.py` | 24a, 24b | 8 nouveaux tests |
| `tests/test_grid_trend.py` | 23b | 3 nouveaux tests |

**Total tests : 1016, 0 échec.**
