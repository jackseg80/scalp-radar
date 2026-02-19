# Sprint 35 — Stress Test Leverage Multi-Fenêtre

**Date** : 19 février 2026
**Durée** : 1 session
**Tests** : 0 nouveaux (script de benchmark, pas de logique métier à tester)

---

## Contexte

Le portfolio backtest a été enrichi (Sprint Backtest Réalisme) avec :
- Liquidation cross-margin (equity < 0.4% notional)
- Funding costs 8h
- Validation leverage×SL
- Kill switch depuis risk.yaml

Il faut maintenant revalider les deux stratégies avec ce moteur réaliste et
déterminer le leverage optimal pour `grid_boltrend` (actuellement 6x dans
strategies.yaml, fixé arbitrairement) et reconfirmer `grid_atr` (live à 6x).

---

## Objectif

Lancer 20 portfolio backtests (2 stratégies × 4 leverages × 2-3 fenêtres) et
produire un rapport comparatif pour choisir le leverage optimal.

---

## Implémentation

### Fichier créé

**`scripts/stress_test_leverage.py`** (nouveau) :

#### Matrice de runs par défaut
- `grid_boltrend` × [2, 4, 6, 8] × [auto, 180j, 90j] = **12 runs**
- `grid_atr` × [2, 4, 6, 8] × [auto, 180j] = **8 runs**
- **Total : 20 runs**

#### Design technique
1. **Kill switch désactivé** (99%) — voir le vrai max drawdown sans interruption
2. **Analyse KS a posteriori** sur les snapshots aux seuils 30%/45%/60%
   - Même logique que `PortfolioBacktester._check_kill_switch()` mais standalone
   - Fenêtre glissante 24h, reset après sortie à 50% du seuil
3. **Override leverage** via `strat_cfg.leverage = leverage` (même pattern que `portfolio_backtest.py`)
   - Restauré dans `finally` pour le run suivant
4. **Détection auto-days** : scanne la DB pour trouver l'asset avec le moins d'historique (goulot)
5. **Sharpe annualisé** : returns horaires depuis equity curve, × sqrt(8760)
6. **Calmar** : total_return_pct / |max_drawdown_pct|

#### Métriques collectées par run
| Colonne | Source |
|---------|--------|
| Return | `result.total_return_pct` |
| Max DD | `result.max_drawdown_pct` |
| Calmar | Return / \|Max DD\| |
| KS@30/45/60 | `_count_ks_triggers()` sur snapshots |
| W-SL | `result.worst_case_sl_loss_pct` |
| Liq% | `result.min_liquidation_distance_pct` |
| Fund$ | `result.funding_paid_total` |
| Trades/WR/Sharpe | `result.total_trades`, `result.win_rate`, `_compute_sharpe()` |

#### Critères de recommandation (ordre de priorité)
1. Distance liquidation > 50% sur TOUTES les fenêtres
2. Max DD > -40% sur TOUTES les fenêtres
3. KS@45 = 0 sur la fenêtre longue (auto)
4. Meilleur Calmar ratio parmi les candidats valides

---

## Usage

```bash
# 20 runs complets (~30 min)
uv run python -m scripts.stress_test_leverage

# Juste boltrend (dry-run validé : 5s)
uv run python -m scripts.stress_test_leverage --strategy grid_boltrend --leverages 2 --days 90

# Affiner entre deux valeurs
uv run python -m scripts.stress_test_leverage --leverages 5,7

# Une seule fenêtre
uv run python -m scripts.stress_test_leverage --days 180

# Combiné
uv run python -m scripts.stress_test_leverage --strategy grid_boltrend --leverages 3,5 --days 90
```

**Options** :
- `--strategy` : filtre une stratégie
- `--leverages` : override la liste (ex: `3,5,7`)
- `--days` : override les fenêtres (entier ou `auto`)
- `--capital` : capital initial (défaut 1000$)
- `--exchange` : source candles (défaut binance)
- `--output` : chemin CSV (défaut `data/stress_test_results.csv`)

---

## Sortie

### Console
```
  [ 1/20] grid_boltrend @ 2x / 750j  (2024-01-XX -> 2026-02-19)
           -> Return=+XX%  DD=-XX%  Calmar=X.XX  Liq=XX%  Trades=XXX

  === grid_boltrend - Auto (750j)  [2024-01-XX -> 2026-02-19] ===
  Lev  | Return   | Max DD   | Calmar  | KS@30  | KS@45  | KS@60  | ...
  -----+----------+----------+---------+--------+--------+--------+----
   2x  | +XX%     | -XX%     | X.XX    | 0      | 0      | 0      | ...
   ...

  RECOMMANDATION
  grid_atr      : 6x (Return +XX%, Max DD -XX%, Calmar X.XX)
  grid_boltrend : 4x (Return +XX%, Max DD -XX%, Calmar X.XX)
```

### CSV
`data/stress_test_results.csv` avec colonnes : strategy, leverage, days,
window_start, window_end, total_return_pct, max_drawdown_pct, calmar,
ks_30, ks_45, ks_60, worst_case_sl_pct, min_liq_dist_pct, was_liquidated,
funding_total, total_trades, win_rate, sharpe.

---

## Résultat dry-run validé

```
grid_boltrend @ 2x / 90j (2025-11-21 -> 2026-02-19) — 5s
Return=+7.9%  DD=-24.0%  Calmar=0.33  Liq=99.8%  Trades=157  WR=38.9%  Sharpe=0.66
```

---

## Décisions techniques

- **Pas de tests unitaires** : script de benchmark, logique triviale (`_count_ks_triggers`,
  `_compute_sharpe`, `_compute_calmar` sont de simples formules mathématiques)
- **Restore leverage dans `finally`** : évite la contamination entre runs
- **KS 99%** : voir le DD brut sans que le kill switch masque les pertes
- **Calmar préféré à Sharpe** pour la recommandation : capture mieux le rapport
  rendement/risque dans un contexte de DCA où les returns sont asymétriques
- **Suggestion intermédiaire** : si deux leverages valides ont un gap de 2x (ex: 4x et 6x),
  suggère explicitement de tester 5x

---

## Prochaine étape

1. Lancer `uv run python -m scripts.stress_test_leverage` (20 runs)
2. Analyser le rapport — choisir leverage optimal grid_boltrend
3. Mettre à jour `config/strategies.yaml` : `grid_boltrend.leverage = X`
4. Déployer grid_boltrend en paper trading (6 assets : BTC, ETH, DOGE, DYDX, LINK, SAND)
