# Backtest & WFO — Référence rapide

> Workflow complet, résultats robustness, et fonctionnement du pipeline.
> Détails : [WORKFLOW_WFO.md](WORKFLOW_WFO.md) | Commandes : [COMMANDS.md](../COMMANDS.md)

## Workflow validation (étapes 0-8)

> **CHAQUE STRATÉGIE EST INDÉPENDANTE** : chaque stratégie a son propre sous-compte Bitget
> avec son propre capital. Backtester chaque stratégie séparément (étape 3).

```text
Étape 0  — Calcul leverage (AVANT tout WFO)
             └─ Plancher = KS / (SL_max × margin_guard)
             └─ Fourchette typique : [3x, 6x]

Étape 1  — WFO mono-asset (21 assets)
             └─ uv run python -m scripts.optimize --strategy <n> --all-symbols --subprocess -v
             └─ Critère : ≥ 5 assets Grade A ou B

Étape 2  — Apply (TOUS les Grade A/B, sans filtre)
             └─ uv run python -m scripts.optimize --strategy <n> --apply

Étape 3  — Portfolio backtest (TOUS les Grade A/B ensemble, une seule stratégie)
             └─ uv run python -m scripts.portfolio_backtest --strategy <n> --days auto --save --label "<nom>"
             └─ Critères : Return > 0, Max DD < -35%, KS = 0, W-SL < KS - 5pts

Étape 3b — Deep Analysis (DIAGNOSTIC — pas un filtre)
             └─ uv run python -m scripts.analyze_wfo_deep --strategy <n>
             └─ Verdicts : VIABLE / BORDERLINE / AT RISK

Étape 4  — Stress test leverage
             └─ uv run python -m scripts.stress_test_leverage --strategy <n>
             └─ Critères : Liq > 50%, KS@45 = 0, W-SL < KS - 5pts

Étape 5  — Portfolio Robustness
             └─ uv run python -m scripts.portfolio_robustness --label "<nom>" --save
             └─ 4 méthodes : Block Bootstrap, Regime Stress, Historical Stress, CVaR
             └─ Critères : CI95 > 0%, prob. perte < 10%, CVaR 30j < KS, survit crashes
             └─ Verdicts : VIABLE → paper | CAUTION → paper 1 mois | FAIL → stop

Étape 6  — Corrélation inter-stratégies (si multi-stratégie)
             └─ uv run python -m scripts.analyze_correlation --labels "<l1>,<l2>"
             └─ Cible : corrélation DD r < 0.3

Étape 7  — Paper trading (≥ 2 semaines, 1 mois si CAUTION)

Étape 8  — Live trading (capital progressif, sous-compte dédié)
```

## Fonctionnement portfolio_robustness

Le script **ne relance pas de backtest**. Il lit les résultats déjà sauvés en DB :

- Table `portfolio_backtests` → colonnes `equity_curve` (JSON), `btc_equity_curve`, `regime_analysis`
- Extrait les returns journaliers depuis l'equity curve sous-échantillonnée (~500 points)
- Exécute 4 analyses statistiques sur ces returns
- Sauvegarde optionnelle dans la table `portfolio_robustness` (si `--save`)

## Résultats robustness — référence

| Stratégie | Leverage | Verdict | CI95 low | Prob. perte | CVaR 30j | Crashes |
| --------- | -------- | ------- | -------- | ----------- | -------- | ------- |
| grid_atr (14 assets) | 7x | **VIABLE** | +157% | 0.0% | 26.9% < 45% | 1/1 OK |
| grid_multi_tf (14 assets) | 6x | **VIABLE** | +121% | 0.0% | 35.6% < 45% | 1/1 OK |
| grid_boltrend (6 assets) | 6x | **CAUTION** | +177% | 0.0% | 57.2% > 45% | 4/4 OK |

**Note grid_boltrend** : seul critère en échec = CVaR 30j (57.2% vs kill_switch 45%).
Tous les autres critères sont verts. Le risque est concentré dans les 5% pires jours
en régime CRASH (CVaR journalier CRASH = -4.4%). Paper 1 mois avec surveillance.

## Points techniques

- Les equity curves sont sous-échantillonnées (~500 pts pour ~2000 jours)
  → les séries portfolio et BTC n'ont pas les mêmes dates exactes
  → le regime stress utilise un nearest-date matching (écart max 5j)
- CVaR annualisé (compound 365j) donne quasi-toujours ~-100% → non utilisé comme critère
  → le critère GO/NO-GO utilise le CVaR 30j (compound) = "pire mois estimé"
- `--seed 42` par défaut pour reproductibilité (deux runs identiques = même résultat)
