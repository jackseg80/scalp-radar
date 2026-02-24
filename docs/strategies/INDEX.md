# Stratégies Scalp Radar — Index

> Documentation complète des 15 stratégies de trading documentées dans Scalp Radar.
> Chaque fichier détaille la logique d'entrée/sortie, les paramètres, les résultats WFO et les leçons apprises.

## Table des matières

- [Tableau récapitulatif](#tableau-récapitulatif)
- [Stratégies Grid/DCA 1h](#stratégies-griddca-1h)
- [Stratégies Scalp 5m](#stratégies-scalp-5m)
- [Stratégies Swing 1h](#stratégies-swing-1h)
- [Matrice de complémentarité par régime](#matrice-de-complémentarité-par-régime)
- [Stratégies en discussion](#stratégies-en-discussion)

---

## Tableau récapitulatif

| # | Stratégie | Catégorie | Timeframe | Grade | Statut | Fichier |
|---|-----------|-----------|-----------|-------|--------|---------|
| 1 | **grid_atr** | Grid/DCA | 1h | A/B (21/21) | Paper trading (Top 10) | [grid_atr.md](grid_atr.md) |
| 2 | **grid_multi_tf** | Grid/DCA | 1h + 4h | — | Désactivé (WFO terminé) | [grid_multi_tf.md](grid_multi_tf.md) |
| 3 | **grid_funding** | Grid/DCA | 1h | — | Désactivé (WFO terminé) | [grid_funding.md](grid_funding.md) |
| 4 | **grid_trend** | Grid/DCA | 1h | F (grille réduite) | Désactivé (échec forward) | [grid_trend.md](grid_trend.md) |
| 15 | **grid_momentum** | Grid/DCA | 1h | — | Désactivé (WFO à lancer) | [grid_momentum.md](grid_momentum.md) |
| 5 | **envelope_dca** | Grid/DCA | 1h | A/B/D (23 assets) | Désactivé (remplacé par grid_atr) | [envelope_dca.md](envelope_dca.md) |
| 6 | **envelope_dca_short** | Grid/DCA | 1h | — | Désactivé (WFO en attente) | [envelope_dca_short.md](envelope_dca_short.md) |
| 7 | **vwap_rsi** | Scalp mono | 5m + 15m | F | Désactivé | [vwap_rsi.md](vwap_rsi.md) |
| 8 | **momentum** | Scalp mono | 5m + 15m | F | Désactivé | [momentum.md](momentum.md) |
| 9 | **funding** | Scalp mono | 15m | — | Désactivé (paper only) | [funding.md](funding.md) |
| 10 | **liquidation** | Scalp mono | 5m | — | Désactivé (paper only) | [liquidation.md](liquidation.md) |
| 11 | **orderflow** | — | 1m | — | Non implémenté (config only) | [orderflow.md](orderflow.md) |
| 12 | **bollinger_mr** | Swing mono | 1h | F | Désactivé | [bollinger_mr.md](bollinger_mr.md) |
| 13 | **donchian_breakout** | Swing mono | 1h | F | Désactivé | [donchian_breakout.md](donchian_breakout.md) |
| 14 | **supertrend** | Swing mono | 1h | F | Désactivé | [supertrend.md](supertrend.md) |

**Légende grades** : A (85-100), B (71-84), C (55-70), D (40-54), F (<40). Grade basé sur 6 critères WFO (OOS/IS ratio, Monte Carlo, consistance, DSR, stabilité, validation Bitget).

---

## Stratégies Grid/DCA 1h

Les stratégies Grid/DCA ouvrent **plusieurs positions** (niveaux) simultanément, avec un DCA progressif. Elles héritent toutes de `BaseGridStrategy` et utilisent le `MultiPositionEngine` / `GridStrategyRunner`.

**Conclusion clé** : L'edge en crypto vient de la structure DCA multi-niveaux, pas des indicateurs mono-position. Toutes les stratégies mono = Grade F ; les stratégies DCA = Grade A/B.

| Stratégie | Base grille | TP | Direction(s) | Spécificité |
|-----------|-------------|-----|--------------|-------------|
| [grid_atr](grid_atr.md) | `SMA ± ATR × mult` | Retour SMA | LONG (défaut) | Enveloppes adaptatives à la volatilité |
| [grid_multi_tf](grid_multi_tf.md) | `SMA ± ATR × mult` | Retour SMA | LONG/SHORT (piloté par ST 4h) | Filtre Supertrend 4h anti-lookahead |
| [grid_funding](grid_funding.md) | Prix courant si FR < seuil | FR > 0 ou SMA cross | LONG-only | Entrée sur funding rate négatif |
| [grid_trend](grid_trend.md) | `EMA_fast ± ATR × pull` | Trailing stop ATR | LONG/SHORT (piloté par EMA cross) | Force close au flip de direction |
| [envelope_dca](envelope_dca.md) | `SMA × (1 - offset%)` | Retour SMA | LONG (défaut) | Enveloppes % fixes asymétriques |
| [envelope_dca_short](envelope_dca_short.md) | `SMA × (1 + offset%)` | Retour SMA | SHORT-only | Miroir SHORT d'envelope_dca |

---

## Stratégies Scalp 5m

Stratégies mono-position sur le timeframe 5 minutes. Héritent de `BaseStrategy`, utilisent `evaluate()` / `check_exit()`.

| Stratégie | Type | Filtre multi-TF | Spécificité |
|-----------|------|-----------------|-------------|
| [vwap_rsi](vwap_rsi.md) | Mean reversion | 15m anti-trend | RSI + VWAP + volume, régime RANGING uniquement |
| [momentum](momentum.md) | Breakout | 15m pro-trend | Rolling high/low cassure + ADX tendance |
| [funding](funding.md) | Arbitrage | — | Funding rate extrême, non backtestable |
| [liquidation](liquidation.md) | Cascade | — | Zones de liquidation OI, non backtestable |
| [orderflow](orderflow.md) | — | — | Non implémenté (config placeholder) |

---

## Stratégies Swing 1h

Stratégies mono-position sur le timeframe 1 heure. Héritent de `BaseStrategy`.

| Stratégie | Type | TP | SL | Spécificité |
|-----------|------|----|----|-------------|
| [bollinger_mr](bollinger_mr.md) | Mean reversion | Dynamique (SMA crossing) | % fixe (5%) | TP géré par `check_exit()`, pas `tp_price` |
| [donchian_breakout](donchian_breakout.md) | Breakout | ATR × 3.0 | ATR × 1.5 | Canal Donchian N périodes |
| [supertrend](supertrend.md) | Trend flip | % fixe (4%) | % fixe (2%) | Entrée sur flip de direction SuperTrend |

---

## Matrice de complémentarité par régime

Chaque stratégie cible un régime de marché spécifique. La combinaison idéale couvre tous les régimes.

| Régime | grid_atr | grid_multi_tf | grid_funding | grid_trend | grid_momentum | envelope_dca | Scalp 5m | Swing 1h |
|--------|----------|---------------|--------------|------------|---------------|--------------|----------|----------|
| **Range / Sideways** | Excellent | Bon | — | Bloqué (ADX < seuil) | Drawdown (faux breakouts) | Bon | vwap_rsi (F) | bollinger_mr (F) |
| **Trend haussier** | Bon (LONG) | Bon (LONG via ST) | — | Excellent (LONG) | Excellent (LONG breakout) | Bon (LONG) | momentum (F) | supertrend (F), donchian (F) |
| **Trend baissier** | Risqué (LONG) | Bon (SHORT via ST) | — | Excellent (SHORT) | Excellent (SHORT breakout) | Risqué (LONG) | momentum (F) | supertrend (F), donchian (F) |
| **Crash / Forte vol.** | SL touché | Force close (ST flip) | — | Force close (EMA flip) | Trail stop + direction flip | SL touché | — | — |
| **Funding négatif** | — | — | Excellent | — | — | — | funding (—) | — |

### Analyse

- **grid_atr** excelle en range et en trend modéré, mais souffre en bear market soutenu (7/21 assets Sharpe négatif sur 90j récents)
- **grid_multi_tf** corrige le défaut de grid_atr en bear grâce au filtre Supertrend 4h (SHORT autorisé)
- **grid_momentum** est décorrélé de grid_atr (profil convexe vs concave) — profite des breakouts que grid_atr subit comme des SL. WFO non encore lancé.
- **grid_trend** est complémentaire en théorie (trend following vs mean reversion) mais échoue en forward test sans trends prolongés
- **grid_funding** a un edge indépendant du prix (shorts paient les longs), mais événements rares
- Les **stratégies mono-position** (scalp 5m + swing 1h) n'ont aucun edge démontré en crypto — l'edge vient du DCA multi-niveaux
- Le **filtre Darwinien** (Sprint 27) bloque les nouvelles grilles si le WFO Sharpe < 0 dans le régime actuel, ajoutant une couche de protection dynamique

---

## Stratégies en discussion

### grid_bollinger (non implémentée)

Idée : combiner les bandes de Bollinger (entrée aux extrêmes) avec la structure DCA multi-niveaux de grid_atr. Pourrait remplacer bollinger_mr (Grade F en mono-position) en exploitant l'edge DCA démontré.

**Différences potentielles avec grid_atr** :
- Base grille : bandes de Bollinger (`SMA ± σ × mult`) au lieu d'enveloppes ATR
- TP : retour à la SMA (identique)
- Avantage théorique : les bandes de Bollinger s'adaptent à la distribution des prix, pas seulement à la volatilité (ATR)

**Statut** : Concept uniquement, pas de plan de sprint.
