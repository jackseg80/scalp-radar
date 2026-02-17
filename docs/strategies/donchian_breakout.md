# donchian_breakout — Donchian Channel Breakout

[Retour à l'index](INDEX.md)

## Identité

| Champ | Valeur |
|-------|--------|
| Nom technique | `donchian_breakout` |
| Catégorie | Swing mono-position |
| Timeframe | 1h |
| Sprint d'origine | Sprint 9 |
| Grade actuel | F |
| Statut | **Désactivé** |
| Fichier source | `backend/strategies/donchian_breakout.py` |
| Config class | `DonchianBreakoutConfig` (`backend/core/config.py:149`) |

## Description

Trade les cassures du canal Donchian (plus haut/plus bas des N dernières bougies). TP et SL basés sur des multiples d'ATR. Stratégie de breakout classique adaptée au swing 1h.

**Régime ciblé** : Détecté mais pas utilisé comme filtre. Fonctionne en théorie en tendance.

## Logique d'entrée

Méthode : `evaluate(ctx) -> StrategySignal | None`

Conditions simples :
- **LONG** : `close > rolling_high(entry_lookback)` — cassure du plus haut des N bougies précédentes
- **SHORT** : `close < rolling_low(entry_lookback)` — cassure du plus bas

Le rolling exclut la bougie courante (lookback de i-N à i exclus).

### Score

Basé sur la force du breakout :
`breakout_pct = (close - rolling_high) / channel_width`, puis `breakout_score = min(1.0, breakout_pct × 5)`

## Logique de sortie

**TP** : `ATR × atr_tp_multiple` — défaut 3.0× ATR

**SL** : `ATR × atr_sl_multiple` — défaut 1.5× ATR

Ratio risque/récompense implicite = 2:1.

**Sortie anticipée** (`check_exit()`) : `return None` — pas de sortie anticipée, TP/SL ATR gèrent tout.

## Paramètres

| Paramètre | Type | Défaut | Contraintes | WFO Range | Description |
|-----------|------|--------|-------------|-----------|-------------|
| `entry_lookback` | int | 20 | >= 2 | [20, 30, 40, 55] | Bougies pour le canal Donchian |
| `atr_period` | int | 14 | >= 2 | [10, 14, 20] | Période ATR |
| `atr_tp_multiple` | float | 3.0 | > 0 | [2.0, 3.0, 4.0] | Multiplicateur ATR pour TP |
| `atr_sl_multiple` | float | 1.5 | > 0 | [1.0, 1.5, 2.0] | Multiplicateur ATR pour SL |
| `timeframe` | str | "1h" | — | — | Timeframe |

**Config WFO** : IS = 180j, OOS = 60j, step = 60j, **108 combinaisons**.

## Indicateurs utilisés

| Indicateur | Timeframe | Rôle |
|------------|-----------|------|
| Rolling High/Low (canal Donchian) | 1h | Signal breakout |
| ATR | 1h | TP/SL adaptatif |
| ATR SMA(20) | 1h | Calcul auxiliaire |
| ADX + DI+/DI- | 1h | Information (pas de filtre) |

## Résultats

**Grade F** sur tous les assets testés. Aucun edge démontré en WFO.

### Points forts

- TP/SL basé sur l'ATR — s'adapte à la volatilité
- Ratio R:R 2:1 par défaut
- Logique simple et classique

### Points faibles

- Les breakouts crypto sont souvent des faux breakouts suivis de mèches
- Mono-position — pas de DCA
- Pas de filtre de tendance (contrairement à momentum qui a l'ADX 15m)

## Remarques

- Similaire en concept à la stratégie momentum (breakout), mais sur 1h au lieu de 5m et sans filtre multi-TF
- Le `entry_lookback` affecte `compute_indicators()` (pré-calcul)
- Pas de sortie anticipée — confiance totale dans le TP/SL ATR
