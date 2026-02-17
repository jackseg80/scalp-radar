# bollinger_mr — Bollinger Band Mean Reversion

[Retour à l'index](INDEX.md)

## Identité

| Champ | Valeur |
|-------|--------|
| Nom technique | `bollinger_mr` |
| Catégorie | Swing mono-position |
| Timeframe | 1h |
| Sprint d'origine | Sprint 9 |
| Grade actuel | F |
| Statut | **Désactivé** |
| Fichier source | `backend/strategies/bollinger_mr.py` |
| Config class | `BollingerMRConfig` (`backend/core/config.py:132`) |

## Description

Mean reversion sur les bandes de Bollinger. Entre en position quand le prix sort des bandes (extrême statistique). Le TP est **dynamique** : géré par `check_exit()` quand le prix revient à la SMA (crossing), pas par un prix fixe.

**Régime ciblé** : Détecté mais pas utilisé comme filtre. Fonctionne en théorie en range/mean-reversion.

## Logique d'entrée

Méthode : `evaluate(ctx) -> StrategySignal | None`

Conditions très simples :
- **LONG** : `close < bb_lower` (prix sous la bande basse)
- **SHORT** : `close > bb_upper` (prix au-dessus de la bande haute)

Pas de filtre de régime, pas de filtre de volume, pas de multi-TF.

### Score

Basé sur la distance aux bandes :
`distance_score = min(1.0, (bb_lower - close) / band_width × 2)` pour LONG.

## Logique de sortie

**TP dynamique** (astuce d'implémentation) :
- Le `tp_price` est placé très loin : `close × 2.0` (LONG) ou `close × 0.5` (SHORT) — jamais touché par le prix
- Le **vrai TP** est dans `check_exit()` :
  - LONG : `close >= bb_sma` (prix revient à la SMA)
  - SHORT : `close <= bb_sma`

**SL** : `entry × (1 ∓ sl_percent / 100)` — défaut **5.0%** (plus large que les scalps 5m)

**Sortie anticipée** (`check_exit()`) — c'est le **vrai TP** :
- LONG : `close >= bb_sma` → `"signal_exit"`
- SHORT : `close <= bb_sma` → `"signal_exit"`

## Paramètres

| Paramètre | Type | Défaut | Contraintes | WFO Range | Description |
|-----------|------|--------|-------------|-----------|-------------|
| `bb_period` | int | 20 | >= 2 | [15, 20, 25, 30] | Période Bollinger (SMA) |
| `bb_std` | float | 2.0 | > 0 | [1.5, 2.0, 2.5] | Nombre de déviations standard |
| `sl_percent` | float | 5.0 | > 0 | [3.0, 5.0, 7.0, 10.0] | Stop loss (%) |
| `timeframe` | str | "1h" | — | — | Timeframe |

Pas de `tp_percent` explicite — le TP est dynamique via SMA crossing.

**Config WFO** : IS = 180j, OOS = 60j, step = 60j, **48 combinaisons**.

## Indicateurs utilisés

| Indicateur | Timeframe | Rôle |
|------------|-----------|------|
| Bollinger Bands (SMA, upper, lower) | 1h | Signal d'entrée + TP dynamique |
| ATR + ATR SMA(20) | 1h | Calcul auxiliaire |
| ADX + DI+/DI- | 1h | Information (pas de filtre) |

## Résultats

**Grade F** sur tous les assets testés. Aucun edge démontré en WFO.

### Points forts

- TP dynamique SMA crossing — concept supérieur au % fixe
- Logique simple et élégante (3 paramètres)
- Petit espace de recherche WFO (48 combos) — rapide à optimiser

### Points faibles

- Mono-position — pas de DCA pour moyenner les entrées
- Le prix peut rester sous les bandes longtemps en bear (pas de recovery → SL touché)
- Grille WFO très petite — peut manquer le bon espace de paramètres

## Remarques

- **Parité fast engine / BacktestEngine** : le TP dynamique (SMA crossing) doit être identique dans les deux moteurs. Le `tp_price` éloigné (×2.0 / ×0.5) est un hack pour que le `BacktestEngine` ne touche jamais le TP fixe
- **IndicatorCache** : nécessite `bb_sma`, `bb_upper`, `bb_lower` — champs ajoutés lors du Sprint 9
- **Candidat pour grid_bollinger** : la logique Bollinger pourrait fonctionner en DCA multi-niveaux (comme grid_atr a remplacé envelope_dca). Non implémenté
