# vwap_rsi — VWAP + RSI Mean Reversion

[Retour à l'index](INDEX.md)

## Identité

| Champ | Valeur |
|-------|--------|
| Nom technique | `vwap_rsi` |
| Catégorie | Scalp mono-position |
| Timeframe | 5m (principal) + 15m (filtre) |
| Sprint d'origine | Sprint 2 |
| Grade actuel | F |
| Statut | **Désactivé** |
| Fichier source | `backend/strategies/vwap_rsi.py` |
| Config class | `VwapRsiConfig` (`backend/core/config.py:40`) |

## Description

Mean reversion sur le timeframe 5m. Entre en position quand le prix s'éloigne du VWAP avec un RSI extrême et un spike de volume. Filtre par la tendance 15m pour éviter de trader contre un mouvement directionnel fort. Trade CONTRE la tendance (stratégie contrarian).

**Régime ciblé** : RANGING ou LOW_VOLATILITY uniquement. Rejette explicitement les marchés en tendance.

## Logique d'entrée

Méthode : `evaluate(ctx) -> StrategySignal | None`

### Filtres préalables (rejet)

1. ADX 15m > `trend_adx_threshold` (25.0) → rejeté (marché en tendance)
2. Régime 5m doit être RANGING ou LOW_VOLATILITY

### Conditions LONG

- `RSI < rsi_long_threshold` (défaut 30)
- `vwap_deviation < -vwap_deviation_entry` (close à plus de 0.3% sous le VWAP)
- Volume spike : `volume > volume_sma × volume_spike_multiplier` (défaut 2.0)
- 15m pas bearish (ADX > 20 et DI- > DI+ = bearish → bloque le LONG)

### Conditions SHORT

- `RSI > rsi_short_threshold` (défaut 70)
- `vwap_deviation > +vwap_deviation_entry`
- Volume spike identique
- 15m pas bullish (ADX > 20 et DI+ > DI- = bullish → bloque le SHORT)

### Score composé

- RSI score : 35%
- VWAP score : 25%
- Volume score : 20%
- Trend alignment score : 20% (1.0 si 15m aligné avec le trade, 0.5 sinon)

## Logique de sortie

**TP** : `entry × (1 ± tp_percent / 100)` — défaut 0.8%

**SL** : `entry × (1 ∓ sl_percent / 100)` — défaut 0.3%

**Sortie anticipée** (`check_exit()`) :
- RSI revient > 50 (LONG) ou < 50 (SHORT) **ET** le trade est en profit → `"signal_exit"`

## Paramètres

| Paramètre | Type | Défaut | Contraintes | WFO Range | Description |
|-----------|------|--------|-------------|-----------|-------------|
| `rsi_period` | int | 14 | >= 2 | [10, 14, 20] | Période RSI |
| `rsi_long_threshold` | float | 30 | 0-100 | [25, 30, 35] | Seuil RSI pour LONG |
| `rsi_short_threshold` | float | 70 | 0-100 | [65, 70, 75] | Seuil RSI pour SHORT |
| `volume_spike_multiplier` | float | 2.0 | > 0 | [1.5, 2.0, 2.5] | Multiplicateur volume spike |
| `vwap_deviation_entry` | float | 0.3 | > 0 | [0.1, 0.2, 0.3, 0.5] | Déviation VWAP % pour entrée |
| `trend_adx_threshold` | float | 25.0 | >= 0 | [20, 25, 30] | Seuil ADX 15m (rejet si >) |
| `tp_percent` | float | 0.8 | > 0 | [0.4, 0.6, 0.8, 1.0] | Take profit (%) |
| `sl_percent` | float | 0.3 | > 0 | [0.2, 0.3, 0.4, 0.5] | Stop loss (%) |
| `timeframe` | str | "5m" | — | — | Timeframe principal |
| `trend_filter_timeframe` | str | "15m" | — | — | Timeframe filtre |

**Config WFO** : IS = 120j, OOS = 30j, step = 30j, **~15 552 combinaisons**.

## Indicateurs utilisés

| Indicateur | Timeframe | Rôle |
|------------|-----------|------|
| RSI | 5m | Signal d'entrée (extrême) + sortie anticipée |
| VWAP rolling 24h | 5m | Distance au fair value |
| Volume SMA(20) | 5m | Détection spike de volume |
| ADX + DI+/DI- | 5m, 15m | Filtre tendance |
| ATR + ATR SMA(20) | 5m | Calcul auxiliaire |
| Régime de marché | 5m | Filtre (RANGING/LOW_VOL uniquement) |

## Résultats

**Grade F** sur tous les assets testés. Aucun edge démontré en WFO.

### Points forts

- Logique de filtre multi-TF sophistiquée (15m anti-trend)
- Score composé multi-critères bien calibré

### Points faibles

- Les stratégies mono-position à indicateurs techniques n'ont pas d'edge en crypto
- TP/SL trop serrés pour le bruit du marché 5m
- Le filtre RANGING rejette trop de signaux

## Remarques

- Première stratégie implémentée dans le projet (Sprint 2)
- Complémentaire avec momentum (vwap_rsi = mean reversion en range, momentum = breakout en tendance)
- La leçon clé : **l'edge vient du DCA multi-niveaux**, pas des indicateurs mono-position
