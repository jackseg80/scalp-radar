# supertrend — SuperTrend Flip

[Retour à l'index](INDEX.md)

## Identité

| Champ | Valeur |
|-------|--------|
| Nom technique | `supertrend` |
| Catégorie | Swing mono-position |
| Timeframe | 1h |
| Sprint d'origine | Sprint 9 |
| Grade actuel | F |
| Statut | **Désactivé** |
| Fichier source | `backend/strategies/supertrend.py` |
| Config class | `SuperTrendConfig` (`backend/core/config.py:167`) |

## Description

Trade les retournements de tendance détectés par l'indicateur SuperTrend. Entre en position sur un flip de direction (passage de haussier à baissier ou inversement). TP et SL en pourcentage fixe. Stratégie trend-following sur 1h.

**Régime ciblé** : Détecté mais pas utilisé comme filtre. Fonctionne en théorie sur les retournements de tendance.

## Logique d'entrée

Méthode : `evaluate(ctx) -> StrategySignal | None`

Détection de flip uniquement :
- **LONG** : `prev_direction == -1 (DOWN)` ET `current_direction == 1 (UP)` — flip baissier → haussier
- **SHORT** : `prev_direction == 1 (UP)` ET `current_direction == -1 (DOWN)` — flip haussier → baissier

Pas de filtre supplémentaire (volume, ADX, régime).

### Score

Basé sur la distance au SuperTrend :
`distance_score = min(1.0, |close - st_value| / close × 100 / 3.0)`

## Logique de sortie

**TP** : `entry × (1 ± tp_percent / 100)` — défaut **4.0%** (large, swing)

**SL** : `entry × (1 ∓ sl_percent / 100)` — défaut **2.0%**

Ratio risque/récompense = 2:1.

**Sortie anticipée** (`check_exit()`) : `return None` — pas de sortie anticipée.

## Paramètres

| Paramètre | Type | Défaut | Contraintes | WFO Range | Description |
|-----------|------|--------|-------------|-----------|-------------|
| `atr_period` | int | 10 | >= 2 | [7, 10, 14, 20] | Période ATR (calcul SuperTrend) |
| `atr_multiplier` | float | 3.0 | > 0 | [2.0, 2.5, 3.0, 4.0] | Multiplicateur ATR du SuperTrend |
| `tp_percent` | float | 4.0 | > 0 | [3.0, 4.0, 6.0, 8.0] | Take profit (%) |
| `sl_percent` | float | 2.0 | > 0 | [1.5, 2.0, 3.0, 4.0] | Stop loss (%) |
| `timeframe` | str | "1h" | — | — | Timeframe |

**Config WFO** : IS = 180j, OOS = 60j, step = 60j, **256 combinaisons**.

## Indicateurs utilisés

| Indicateur | Timeframe | Rôle |
|------------|-----------|------|
| SuperTrend (valeur + direction) | 1h | Signal de flip |
| ATR | 1h | Calcul du SuperTrend |
| ATR SMA(20) | 1h | Calcul auxiliaire |
| ADX + DI+/DI- | 1h | Information (pas de filtre) |

Le SuperTrend est pré-calculé via une boucle itérative (~5ms pour 48k points).

## Résultats

**Grade F** sur tous les assets testés. Aucun edge démontré en WFO.

### Points forts

- Concept clair : trader les retournements de tendance
- TP/SL larges (4%/2%) adaptés au swing 1h
- SuperTrend pré-calculé efficacement

### Points faibles

- Les flips SuperTrend sont fréquents en range → beaucoup de faux signaux
- Mono-position — pas de DCA
- Pas de filtre de régime ou de tendance pour éliminer les faux flips

## Remarques

- Le SuperTrend est réutilisé dans grid_multi_tf comme filtre directionnel 4h (et non comme signal d'entrée)
- **IndicatorCache** : nécessite `supertrend_direction` — champ ajouté lors du Sprint 9
- Le pré-calcul SuperTrend utilise une boucle itérative (pas vectorisable) avec la direction précédente comme état
- **Différence avec grid_multi_tf** : ici le SuperTrend est un signal d'entrée mono-position 1h, dans grid_multi_tf il est un filtre directionnel 4h pour le DCA
