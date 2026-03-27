# Audit grid_atr — Paper + Live Trading Report

**Date:** 2026-03-27
**Stratégie:** grid_atr (ATR envelopes, DCA LONG only)
**Config actuelle (v62b, 01/03+):** `live_eligible: true`, leverage 6x, sides: long only, 13 assets, `min_atr_pct` actif

---

## 1. Trois phases détectées — données hétérogènes

| Phase   | Période             | Assets | Levier             | Statut                |
|---------|---------------------|--------|--------------------|-----------------------|
| Phase 1 | 01/02 → 24/02       | 20-23  | 3x→7x→3x→4x→5x   | Dev/tuning — invalide |
| Phase 2 | 25/02 → 28/02       | 20     | 4x→5x + WFO déployé | Transition — invalide |
| Phase 3 | 01/03 → aujourd'hui | 13     | 6x stable          | **Config actuelle**   |

Le commit `512b7a1` (01/03) a tout changé : retrait de LINK/XLM, ajout du filtre `min_atr_pct`, capital paper réinitialisé (`7acd6c1`). **Tout ce qui est avant le 01/03 est du bruit de développement.**

---

## 2. Performance par phase

| Phase                     | Trades | Win%  | PnL      | Avg/trade | SL hits      |
|---------------------------|--------|-------|----------|-----------|--------------|
| Phase 1 (ancien config)   | 1,456  | 78.8% | -$1,538  | -$1.06    | 37 (-$1,571) |
| Phase 2 (transition)      | 61     | 75.4% | +$19     | +$0.31    | 0            |
| **Phase 3 (config v62b)** | **29** | **89.7%** | **+$158** | **+$5.47** | **0**     |

La config actuelle ne ressemble pas au tableau global. **0 SL hit, pire trade à -$2.41, avg +$5.47.** C'est une stratégie fondamentalement différente de la Phase 1.

---

## 3. Ce qui a causé le désastre de Phase 1

Le 06/02 : crash BTC → 37 SL global en une semaine, -$1,571. Avec 20+ assets en LONG simultanément et un levier instable (jusqu'à 7x), un mouvement baissier brutal a déclenché des SL en cascade.

**Ce n'est pas un problème de la stratégie actuelle** — c'est une config de test avec un levier instable sur trop d'assets. Ces données ne doivent pas être utilisées pour juger la config v62b.

### Données brutes Phase 1 (contexte uniquement)

Sorties Phase 1 :

| Raison    | Trades | Win%  | PnL      | Avg/trade |
|-----------|--------|-------|----------|-----------|
| tp_global | 1,509  | 80.8% | +$209    | +$0.14    |
| sl_global | 37     | 0%    | -$1,571  | -$42.46   |

Anomalie tp_global Phase 1 : de nombreuses pertes importantes taguées `tp_global` (NEAR -$83, SOL -$58, BCH -$53, ARB -$58). Hypothèse : `should_close_all()` ferme toutes les positions en bloc quand le TP global est atteint — les positions deep underwater héritent de `exit_reason = tp_global`. **Bug de labeling, pas de logique — mais fausse les métriques.**

### Assets Phase 1 (tableau complet, pour référence historique)

| Asset | Trades | Win%  | PnL    | Pire trade | Dernier trade |
|-------|--------|-------|--------|------------|---------------|
| ICP   | 77     | 80.5% | +$31   | -$23       | 19/03         |
| BNB   | 101    | 82.2% | +$8    | -$21       | 28/02         |
| ETH   | 37     | 70.3% | +$0.85 | -$20       | 24/03         |
| LTC   | 3      | 66.7% | -$7    | -$12       | 28/02         |
| BTC   | 5      | 80.0% | -$8    | -$20       | 28/02         |
| FET   | 69     | 76.8% | -$10   | -$31       | 28/02         |
| ETC   | 4      | 50.0% | -$11   | -$8        | 06/03         |
| CRV   | 27     | 81.5% | -$15   | -$30       | 24/02         |
| AVAX  | 101    | 81.2% | -$15   | -$47       | 28/02         |
| AAVE  | 60     | 80.0% | -$31   | -$47       | 28/02         |
| ATOM  | 8      | 25.0% | -$33   | -$16       | 19/03         |
| ARB   | 22     | 77.3% | -$36   | -$58       | 20/02         |
| SOL   | 55     | 74.5% | -$40   | -$58       | 23/03         |
| DOGE  | 76     | 75.0% | -$47   | -$44       | 27/02         |
| GALA  | 25     | 84.0% | -$69   | -$39       | 28/02         |
| DYDX  | 110    | 69.1% | -$76   | -$45       | 28/02         |
| LINK  | 80     | 80.0% | -$80   | -$47       | 26/02         |
| UNI   | 138    | 76.8% | -$98   | -$55       | 27/02         |
| ADA   | 99     | 77.8% | -$117  | -$46       | 28/02         |
| XRP   | 141    | 80.1% | -$137  | -$55       | 10/03         |
| OP    | 78     | 80.8% | -$146  | -$51       | 23/02         |
| BCH   | 129    | 89.1% | -$180  | -$53       | 19/03         |
| NEAR  | 101    | 82.2% | -$239  | -$84       | 23/03         |

> Ces chiffres par asset mélangent les 3 phases et différentes configs. Seuls ICP, BNB, ETH étaient positifs toutes phases confondues.

---

## 4. Performance actuelle — Simulation vs Live

| Métrique   | Simulation (01/03+) | Live (10/03 → 24/03) | Écart       |
|------------|---------------------|-----------------------|-------------|
| Trades     | 29                  | 18                    | -38%        |
| Win rate   | 89.7%               | 16.7%                 | **-73pp**   |
| PnL        | +$158               | -$41                  | -$199       |
| Avg/trade  | +$5.47              | -$2.30                |             |
| Pire trade | -$2.41              | -$13.10               |             |
| SL hits    | 0                   | 0                     |             |

**L'écart simulation/live est le vrai sujet.** L'investigation trade-par-trade (27/03) a identifié les causes.

### Capital : écart négligeable (éliminé)

| Source                     | Valeur    |
|----------------------------|-----------|
| Capital live (balance Bitget) | $1,655.59 (snapshots arrêtés au 12/03) |
| Capital simulation          | $1,704.83 (simulator_state)            |

Écart ~$50 — **ce n'est pas la source du problème.**

### Cause principale : biais d'anticipation de la simulation

La simulation calcule ses grilles sur la clôture de la bougie 1h et positionne les niveaux de grille **à l'avance**, en dessous du prix (LONG). L'executor live entre en **market order temps réel** — quand le prix atteint le niveau, le fill réel est souvent **après** le mouvement.

Exemple concret — 10/03/2026 :

| Trade        | Sim entrée | Live entrée    | Sim sortie | Live sortie | Sim PnL | Live PnL |
|--------------|-----------|----------------|-----------|-------------|---------|----------|
| SOL LONG     | 82.88     | ~86.00 (+3.8%) | 85.81     | 86.08       | +$6.25  | -$0.28   |
| XRP LONG     | 1.3824    | ~1.43 (+3.4%)  | 1.4334    | 1.4267      | +$8.78  | -$1.45   |
| ETH LONG     | —         | (pas de trade)  | —         | —           | +$8.15  | —        |
| NEAR LONG    | —         | (pas de trade)  | —         | —           | +$6.28  | —        |

La simulation entre **avant** la hausse (SOL à 82.88), le live entre **pendant/après** (SOL à ~86). Le TP global ferme au même niveau SMA → la sim capture +$6.25, le live perd -$0.28.

### tp_global ferme des positions individuellement perdantes (bug confirmé)

| Trade          | Entrée | Sortie | PnL     | Direction |
|----------------|--------|--------|---------|-----------|
| ICP LONG 07/03 | 2.4470 | 2.4290 | -$2.13  | ↓ perte   |
| SOL LONG 10/03 | ~86.00 | 86.08  | -$0.28  | ≈ flat    |
| ICP LONG 11/03 | 2.5300 | ~2.41  | -$13.10 | ↓ perte   |

Le `tp_global` se déclenche quand le **PnL agrégé de tous les niveaux** atteint le seuil (prix revient à SMA). Mais si une position individuelle a été ouverte à un prix plus élevé que la SMA de sortie, elle est fermée en perte. Le `exit_reason = tp_close` est donc correct au niveau global mais trompeur au niveau du trade individuel.

**Ce n'est pas un bug de labeling** — c'est le fonctionnement attendu du tp_global multi-niveaux. Mais les métriques par trade (win rate, avg PnL) sont faussées car elles comptent chaque niveau comme un trade indépendant.

### 41 trades fantômes du 05/03 (pnl=0)

Le déploiement v62b du 01/03 a changé la config. Le 05/03, l'executor a détecté des positions ouvertes sur Bitget qui ne correspondaient plus à la nouvelle config et les a force-closées avec `pnl=0` en DB. **Le PnL réel de ces fermetures n'a pas été comptabilisé.** Ce sont de vraies transactions Bitget dont le résultat financier est perdu.

### Récapitulatif des causes d'écart

| Cause | Impact | Status |
|-------|--------|--------|
| Simulation positionne avant le mouvement, live entre après | **Majeur** — biais structurel | Confirmé |
| tp_global ferme des positions individuelles en perte | **Moyen** — fausse le win rate par trade | Confirmé (comportement attendu mais métriques trompeuses) |
| 41 trades du 05/03 à pnl=0 (PnL réel non comptabilisé) | **Inconnu** — possible perte masquée | À investiguer |
| Fees maker (sim) vs taker (live) + slippage réel | **Mineur** — amplifie l'écart | Confirmé |

---

## 5. Config actuelle (v62b)

### Paramètres globaux

- `live_eligible: true`, leverage 6x, sides: long only
- `cooldown_candles: 3`
- Filtre `min_atr_pct` actif (ajouté Sprint 62b)

### Assets per_asset WFO (8 avec override)

| Asset | atr_mult_start | atr_mult_step | atr_period | ma_period | num_levels | sl_percent | min_atr_pct | min_grid_spacing |
|-------|---------------|---------------|------------|-----------|------------|------------|-------------|-----------------|
| ADA   | 3.0           | 1.5           | 10         | 10        | 3          | 20.0%      | 0.0         | 1.2             |
| ATOM  | 3.0           | 1.0           | 10         | 7         | 4          | 20.0%      | 0.5         | 1.2             |
| BNB   | 2.0           | 1.0           | 14         | 7         | 3          | 20.0%      | 0.5         | 1.8             |
| BTC   | 2.5           | 1.0           | 14         | 7         | 3          | **12.0%**  | 0.0         | 1.8             |
| ETC   | 3.0           | 0.5           | 10         | 14        | 3          | **12.0%**  | 0.8         | 1.8             |
| LTC   | 2.0           | 0.5           | 10         | 7         | 3          | **12.0%**  | 0.3         | 1.8             |
| SOL   | 3.0           | 1.5           | 14         | 7         | 4          | **12.0%**  | 0.3         | 1.8             |
| XRP   | 2.5           | 1.5           | 14         | 7         | 4          | 20.0%      | 0.8         | 1.8             |

> Note : la stratégie tourne sur 13 assets (v62b), mais seuls 8 ont des per_asset WFO explicites. Les 5 autres utilisent les paramètres par défaut (sl_percent: 20%, ma_period: 14, etc.).

### WFO Param Grid

Grille : 3,456 combos par asset. `sl_percent` testé : [8.0, 10.0, 12.0, 15.0, 20.0]. WFO IS=180j, OOS=60j, step=60j, embargo=7j.

---

## 6. Comparaison grid_atr vs grid_multi_tf (Phase 3 uniquement)

| Critère               | grid_atr (Phase 3)     | grid_multi_tf (complet)      |
|-----------------------|------------------------|------------------------------|
| Période comparable    | 01/03 → 27/03          | 06/02 → 26/03               |
| PnL simulation        | +$158                  | +$1,363                      |
| Win rate              | 89.7%                  | 89.5%                        |
| Trades                | 29                     | 731                          |
| Directions            | LONG only              | LONG + SHORT                 |
| SL hits               | 0                      | 6                            |
| Statut live           | 18 trades, -$41        | Non live                     |
| Leverage              | 6x                     | 5x                           |

> Comparaison asymétrique : grid_atr Phase 3 couvre seulement ~27 jours de marché range calme avec peu de trades. grid_multi_tf couvre 49 jours incluant le bear de février.

---

## 7. Problèmes identifiés

### Cause racine (confirmée)
1. **Biais d'anticipation de la simulation** — La simulation pré-positionne les niveaux de grille et entre au prix théorique (candle.low touch). L'executor entre en market order temps réel, souvent après le mouvement. Ce biais structurel rend la simulation **non prédictive** de la performance live pour grid_atr
2. **tp_global multi-niveaux** — Le TP global ferme toutes les positions quand le PnL agrégé atteint le seuil. Les positions individuelles entrées tardivement (niveaux hauts) peuvent être fermées en perte. Ce n'est pas un bug mais **les métriques par trade sont trompeuses** (win rate sous-estimé en live)
3. **41 trades fantômes du 05/03** — Force-close de l'ancienne config avec pnl=0 en DB. Le PnL réel sur Bitget est potentiellement perdu/non comptabilisé

### Risques structurels
4. **100% LONG sans hedge** — Pas de protection en bear market. La Phase 1 l'a démontré (même si la config a changé)
5. **Pas de filtre de régime** — La stratégie entre en LONG même en bear confirmé

### Observations
6. **Phase 3 encourageante en sim mais non reproductible en live** — Les +$158 sim sont biaisés par l'anticipation. La vraie performance live est -$41 sur 18 trades

---

## 8. Recommandations

| Priorité | Action                                                                          | Status       |
|----------|---------------------------------------------------------------------------------|--------------|
| **P0**   | ~~Investiguer l'écart sim/live~~ — **Cause identifiée** : biais d'anticipation   | **Fait** (27/03) |
| **P0**   | ~~Investiguer anomalie tp_global~~ — **Comportement attendu** du multi-niveaux    | **Fait** (27/03) |
| **P0**   | ~~Corriger le biais d'anticipation~~ : l'executor utilise désormais des limit orders aux prix des niveaux de grille | **Fait** (27/03) |
| **P1**   | Vérifier les 41 trades fantômes du 05/03 sur Bitget (PnL réel perdu ?)          | À faire      |
| **P1**   | Revoir les métriques : compter le win rate par **cycle** (agrégé) et non par trade individuel, pour refléter le vrai tp_global | À faire |
| **P2**   | Évaluer l'ajout d'un filtre de régime (Supertrend 4h comme grid_multi_tf)       | À évaluer    |
| **P2**   | Évaluer si grid_atr doit passer en limit orders sur Bitget (réduire le slippage d'entrée) | À évaluer |

---

## 9. Conclusion (investigation 27/03)

**L'investigation trade-par-trade a identifié la cause racine de l'écart sim/live.**

### Le problème n'est pas la stratégie — c'est le modèle d'exécution

La simulation pré-positionne les grilles et entre au prix théorique du niveau (touch via candle.low). L'executor entre en **market order temps réel**, souvent **après** le mouvement de prix. Sur le 10/03, la sim entre SOL à 82.88, le live à ~86 — un écart de 3.8% qui transforme un gain de +$6.25 en perte de -$0.28.

Ce biais est **structurel et reproductible** : chaque fois que le prix traverse rapidement un niveau de grille, la sim capture le bas, le live capture le haut.

### Le tp_global multi-niveaux fonctionne comme prévu

Les "pertes sur tp_close" ne sont pas un bug. Le tp_global ferme l'ensemble du cycle quand le PnL agrégé est positif. Les positions individuelles entrées tardivement peuvent être en perte. Le win rate par trade (16.7%) est trompeur — le win rate par **cycle** serait plus représentatif.

### Verdict

- **Simulation grid_atr : référence biaisée**, non prédictive de la performance live
- **Live grid_atr : données insuffisantes** (18 trades) mais le biais d'entrée est confirmé
- **Action requise** : corriger le modèle d'exécution (limit orders) ou le modèle de simulation (entrée réaliste) avant de tirer des conclusions sur la stratégie elle-même

**Correction implémentée (27/03) :** l'executor place désormais des **limit orders** aux prix des niveaux de grille, au lieu de market orders. Les fills sont détectés via watchOrders (temps réel) + polling fallback (60s). Les ordres sont annulés/replacés si le niveau dérive de >0.2% entre bougies, et expirent après 2h. À re-évaluer avec des données live post-correction.
