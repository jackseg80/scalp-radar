# Audit grid_multi_tf — Paper Trading Report

**Date:** 2026-03-27
**Période:** 06/02 → 26/03/2026 (49 jours)
**Mode:** Paper trading (Simulator)
**Stratégie:** grid_multi_tf (Supertrend 4h + Grid ATR 1h)
**Config actuelle (27/03):** `live_eligible: false`, leverage 5x, sides: long+short, 9 assets per_asset

> **Caveat important :** La période couverte (49 jours) a pu voir des re-optimisations WFO,
> ajouts/retraits d'assets, et changements de paramètres. Les chiffres agrégés ci-dessous
> mélangent potentiellement **plusieurs configs successives et soldes différents**.
> Les conclusions portent sur les tendances observées, pas sur une config fixe.

---

## 1. Vue d'ensemble

| Métrique           | Valeur                          |
|--------------------|---------------------------------|
| Trades             | 731                             |
| Win rate           | 89.5% (654/731)                 |
| PnL total          | +$1,363.71                      |
| Avg / trade        | +$1.87                          |
| Meilleur trade     | +$11.52                         |
| Pire trade         | -$38.87 (CRV LONG, 14/03)      |
| Positions ouvertes | 0                               |
| Kill switch        | OFF                             |
| Capital estimé     | ~$1,652 → ~$3,246 (+96%)       |

## 2. Évolution hebdomadaire — Deux régimes distincts

| Semaine      | Trades | PnL sem.  | Cumulé      | Régime        |
|--------------|--------|-----------|-------------|---------------|
| W05 (02/02)  | 58     | -$195.76  | -$195.76    | Trending      |
| W06 (09/02)  | 114    | -$35.33   | -$231.09    | Trending      |
| W07 (16/02)  | 94     | +$67.18   | -$163.91    | Transition    |
| W08 (23/02)  | 41     | -$86.46   | -$250.37    | Trending      |
| W09 (02/03)  | 23     | +$76.69   | -$173.68    | Transition    |
| W10 (09/03)  | 184    | +$625.37  | +$451.69    | Range         |
| W11 (16/03)  | 141    | +$605.66  | +$1,057.35  | Range         |
| W12 (23/03)  | 76     | +$306.36  | +$1,363.71  | Range         |

**Conclusion régime :** Février (bull trend) = -$250. Mars (range/consolidation) = +$1,614.
La stratégie est **fondamentalement une stratégie de range**.

## 3. Analyse des sorties

| Raison    | Trades | Win%  | PnL     |
|-----------|--------|-------|---------|
| tp_global | 725    | 90.2% | +$1,534 |
| sl_global | 6      | 0%    | -$170   |

Les 6 SL ont tous frappé sur **OP/USDT SHORT le 06/02** en une seule journée.
Quand le SL global déclenche, les pertes sont lourdes (~$25-34/trade).

### Anomalie CRV -$38 tp_global LONG (14/03)

Un LONG perdant $38.87 avec `exit_reason = tp_global` est **incohérent**.
Hypothèses :
- SL déguisé en tp_global (bug labeling exit_reason)
- Le TP global a fermé toutes les positions d'un coup, dont un LONG deep underwater
- Possible bug dans `should_close_all()` exit reason attribution

**Status : À investiguer en dev (priorité haute)**

## 4. Performance par direction

| Direction | Trades | Win%  | PnL   | % PnL total |
|-----------|--------|-------|-------|-------------|
| LONG      | 232    | 96.6% | +$869 | 64%         |
| SHORT     | 499    | 86.2% | +$493 | 36%         |

Les SHORT = 68% des trades mais seulement 36% du PnL.
En février, tous les gros trades perdants étaient des **SHORT pris dans un bull run**.

**Question ouverte :** Le filtre Supertrend 4h est-il suffisant pour protéger les SHORT en tendance haussière ?

## 5. Performance par asset

### Février — Assets perdants

| Asset     | Trades | Win%  | PnL    | Pire trade | Commentaire           |
|-----------|--------|-------|--------|------------|-----------------------|
| OP/USDT   | 33     | 69.7% | -$139  | -$34 (SL)  | 5 SL en 1 jour        |
| DYDX/USDT | 24     | 66.7% | -$26   | -$22       | Volatile en bull       |
| NEAR/USDT | 14     | 71.4% | -$21   | -$25       | Pas dans per_asset     |
| GALA/USDT | 12     | 75.0% | -$19   | -$21       |                       |
| CRV/USDT  | 23     | 73.9% | -$17   | -$16       |                       |

### Mars — Assets gagnants

| Asset | Trades | Win%  | PnL   | Commentaire                |
|-------|--------|-------|-------|----------------------------|
| DOGE  | 71     | 100%  | +$381 | Win rate irréaliste        |
| ETH   | 71     | 100%  | +$253 | Win rate irréaliste        |
| AAVE  | 69     | 94.2% | +$237 | Solide                     |
| XRP   | 68     | 100%  | +$205 | Win rate irréaliste        |
| ICP   | 39     | 97.4% | +$159 | Très bon                   |
| BCH   | 53     | 96.2% | +$128 | Solide                     |
| CRV   | 29     | 93.1% | +$127 | Retournement vs février    |

> 100% de win rate sur DOGE, ETH, XRP = marché range parfait pour la stratégie.
> Ces stats **ne se reproduiront pas** dans tous les régimes.

## 6. Config actuelle vs observations

**9 assets dans per_asset :** AAVE, BCH, CRV, DOGE, DYDX, ETH, GALA, ICP, XRP.

**OP/USDT et NEAR/USDT** sont mentionnés dans le rapport paper mais **ne sont plus dans per_asset** — soit déjà retirés, soit tournaient sur les paramètres par défaut.

**Paramètres WFO :** IS=180j, OOS=60j, step=60j, embargo=7j.
Grille : 2×2×3×2×2×2×2×3×2 = 1,152 combos par asset.

## 7. Forces et risques

### Forces
- PnL régulier et prévisible en marché range (situation mars 2026)
- +96% en 49 jours (dont ~14 jours sans trades en février)
- 9 assets diversifiés, bonne couverture
- TP fonctionne correctement (725/731 trades sortis par tp_global)
- 0 positions ouvertes = pas de risque latent

### Risques
1. **Marchés trending** : SHORT massacrés quand BTC fait +15% en quelques jours (février)
2. **OP/USDT** : 5 SL en 1 jour = $139. Asset à retirer
3. **CRV anomalie** : -$38 sur un "tp_global" LONG — bug probable
4. **Win rate mars** : 100% sur 3 assets = suroptimisme, non reproductible
5. **Ratio SHORT/LONG déséquilibré** : 2.15:1 en volume mais ratio PnL inversé

## 8. Recommandations avant passage en live

| Priorité | Action                                                                  | Status |
|----------|-------------------------------------------------------------------------|--------|
| **P0**   | Retirer OP/USDT de la config grid_multi_tf                              | À faire |
| **P0**   | Investiguer anomalie CRV -$38 tp_global LONG (bug exit_reason ?)        | À faire |
| **P1**   | Évaluer si SHORT vaut le risque — filtre Supertrend 4h suffisant ?      | À faire |
| **P1**   | Vérifier que NEAR/USDT n'est plus actif (absent de per_asset)           | Vérifié — absent |
| **P2**   | Continuer paper 2-4 semaines sur cycle différent (sortie du range)      | En cours |
| **P2**   | Comparer perf paper vs WFO OOS pour détecter surfit éventuel            | À faire |

## 9. Décision go/no-go

**Verdict : NO-GO pour le live immédiat.**

Raisons :
1. L'anomalie CRV doit être comprise avant de risquer du capital réel
2. La stratégie n'a été profitable que dans un seul régime (range)
3. Il faut observer au moins un cycle de transition range→trend pour valider le comportement du kill switch
4. Le paper devrait couvrir au minimum un event de volatilité (FOMC, CPI) en condition range→trend

**Prochaine revue :** mi-avril 2026, après au moins 2 semaines supplémentaires de paper.
